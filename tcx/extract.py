"""Orchestrates preset extraction from one or more aligned before/after pairs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

from . import colorspace as cs
from . import curves as C
from . import metrics as M
from . import basic
from .align import PairData, sample_pixels
from .grading import fit_grading
from .hsl import fit_hsl, fit_basic_saturation
from .model import PresetModel, Calibration, BAND_NAMES, GradeZone
from .render import render, invert_post


@dataclass
class ExtractOptions:
    color_mode: str = "curves"        # curves | grading | both
    saturation_mode: str = "hsl"      # hsl | basic
    iterations: int = 3
    max_samples: int = 1_500_000
    smooth: float = 2.0
    n_bins: int = 256
    max_points: int = 16
    point_tol: float = 0.5 / 255.0
    fit_hsl: bool = True
    fit_hsl_hue: bool = True
    fit_hsl_lum: bool = True
    grading_global: bool = False
    quantize: bool = True             # emit exactly the curve Lightroom will rebuild
    name: str = "Extracted Preset"
    group: str = "tcx"
    calibration: Calibration = field(default_factory=Calibration)


LUMA_MIX = np.array([0.2126, 0.7152, 0.0722])


def set_channel_transfers(model: PresetModel, f: list[np.ndarray]) -> None:
    """Store three absolute per-channel transfer LUTs as master + R/G/B curves."""
    master = sum(w * lut for w, lut in zip(LUMA_MIX, f))
    master = np.clip(np.maximum.accumulate(master), 0.0, 1.0)
    model.master = master
    inv = C.invert_lut(master)
    model.red, model.green, model.blue = (C.compose_lut(x, inv) for x in f)


def get_channel_transfers(model: PresetModel) -> list[np.ndarray]:
    return [C.compose_lut(ch, model.master) for ch in (model.red, model.green, model.blue)]


def extract(pairs: list[PairData], opts: ExtractOptions) -> tuple[PresetModel, dict]:
    t0 = time.time()
    B, A, W = sample_pixels(pairs, opts.max_samples)
    diag: dict = {"n_pairs": len(pairs), "n_samples": int(B.shape[0]),
                  "options": {k: v for k, v in asdict(opts).items() if k != "calibration"},
                  "pairs": [p.info for p in pairs]}

    model = PresetModel(calibration=opts.calibration, name=opts.name, group=opts.group)

    # ---- stage 1: global transfer functions -----------------------------
    if opts.color_mode == "grading":
        yb, ya = cs.luma(B), cs.luma(A)
        fit = C.estimate_transfer(yb, ya, W, n_bins=opts.n_bins, lam=opts.smooth)
        model.master = fit.lut
        diag["curve_coverage"] = {"luma": round(fit.coverage, 3)}
        diag["input_support"] = {"luma": [round(v * 255, 1) for v in fit.support]}
    else:
        fits = [C.estimate_transfer(B[:, c], A[:, c], W,
                                    n_bins=opts.n_bins, lam=opts.smooth) for c in range(3)]
        set_channel_transfers(model, [f.lut for f in fits])
        diag["curve_coverage"] = {n: round(f.coverage, 3)
                                  for n, f in zip("RGB", fits)}
        diag["curve_spread"] = {
            n: round(float(np.nanmedian(f.spread) * 255), 2) for n, f in zip("RGB", fits)}
        diag["input_support"] = {n: [round(v * 255, 1) for v in f.support]
                                 for n, f in zip("RGB", fits)}
        lo = max(f.support[0] for f in fits) * 255
        hi = min(f.support[1] for f in fits) * 255
        if lo > 12 or hi < 243:
            diag["warnings"] = diag.get("warnings", []) + [
                f"the images only exercise input levels {lo:.0f}-{hi:.0f}; the curve "
                f"outside that range is extrapolated, not measured"]

    # ---- stage 2: alternate residual stages and curve refinement --------
    # Each stage is fitted in its *own* domain: the later stages are peeled
    # off the target by inverting them, so the tone curve is not asked to
    # explain colour-mixer work and vice versa.
    history = []
    for it in range(max(1, opts.iterations)):
        if it > 0:
            pre = invert_post(A, model)
            if opts.color_mode == "grading":
                h = C.estimate_transfer(cs.luma(B), cs.luma(pre), W,
                                        n_bins=opts.n_bins, lam=opts.smooth)
                model.master = h.lut
            else:
                set_channel_transfers(model, [
                    C.estimate_transfer(B[:, c], pre[:, c], W,
                                        n_bins=opts.n_bins, lam=opts.smooth).lut
                    for c in range(3)])

        mid = render(B, model, upto="curves")

        if opts.fit_hsl:
            bands, hdiag = fit_hsl(mid, A, W, opts.calibration,
                                   fit_hue=opts.fit_hsl_hue,
                                   fit_sat=True,
                                   fit_lum=opts.fit_hsl_lum)
            model.hsl = bands
            diag["hsl_detail"] = hdiag

        if opts.color_mode in ("grading", "both"):
            mid2 = render(B, model, upto="hsl")
            zones, blending, balance, gdiag = fit_grading(
                mid2, A, W, opts.calibration, use_global=opts.grading_global)
            model.grade = zones
            model.grade_blending = round(blending, 1)
            model.grade_balance = round(balance, 1)
            diag["grading_detail"] = gdiag

        if opts.saturation_mode == "basic":
            mid3 = render(B, model, upto="grading")
            sat, vib = fit_basic_saturation(mid3, A, W, opts.calibration)
            model.saturation, model.vibrance = round(sat, 1), round(vib, 1)

        out = render(B, model)
        history.append(M.compare(out, A, W))

    diag["iteration_metrics"] = history

    # ---- stage 3: quantise to the control points Lightroom will store ----
    points = {}
    for key in ("master", "red", "green", "blue"):
        lut = getattr(model, key)
        pts = C.fit_control_points(lut, opts.max_points, opts.point_tol)
        points[key] = pts
        if opts.quantize:
            setattr(model, key, C.control_points_to_lut(pts))
    model.meta["control_points"] = points
    diag["control_points"] = {k: len(v) for k, v in points.items()}

    # ---- verification on the real (unblurred, full) images ---------------
    ver = []
    for p in pairs:
        pred = render(p.before, model)
        base = M.compare(p.before, p.after, p.weight)
        got = M.compare(pred, p.after, p.weight)
        ver.append({"size": p.info.get("working_size"),
                    "baseline": base, "preset": got,
                    "improvement": M.improvement(base, got)})
    diag["verification"] = ver
    diag["verification_mean"] = {
        k: round(float(np.mean([v["preset"][k] for v in ver])), 3)
        for k in ("rmse255", "psnr_db", "dE_mean", "dE_p95")}
    diag["baseline_mean"] = {
        k: round(float(np.mean([v["baseline"][k] for v in ver])), 3)
        for k in ("rmse255", "psnr_db", "dE_mean", "dE_p95")}

    diag["diagnostics"] = basic.diagnose(model, B, A, W)
    diag["elapsed_sec"] = round(time.time() - t0, 2)
    model.meta["diagnostics"] = diag["diagnostics"]
    model.meta["verification"] = diag["verification_mean"]
    return model, diag
