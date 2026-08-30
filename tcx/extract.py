"""Orchestrates preset extraction from one or more aligned before/after pairs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict, replace

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
    working_space: str = "auto"       # "auto" | "melissa" | "srgb"
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
    """Fit a preset in one concrete working space.  Use ``extract_auto`` to let
    the data choose the space."""
    if opts.working_space not in cs.WORKING_SPACES:
        raise ValueError(f"extract() needs a concrete working space "
                         f"{cs.WORKING_SPACES}, got {opts.working_space!r}; "
                         f"use extract_auto() for 'auto'")
    t0 = time.time()
    B, A, W = sample_pixels(pairs, opts.max_samples)
    diag: dict = {"n_pairs": len(pairs), "n_samples": int(B.shape[0]),
                  "options": {k: v for k, v in asdict(opts).items() if k != "calibration"},
                  "pairs": [p.info for p in pairs]}

    model = PresetModel(calibration=opts.calibration, name=opts.name, group=opts.group,
                        working_space=opts.working_space)

    # The whole edit is fitted in Lightroom's working space, so the curve we
    # write into the XMP is the curve Lightroom will actually apply.  The two
    # spaces coincide on the neutral axis and differ on saturated colour.
    Bw = cs.to_working(B, model.working_space)
    Aw = cs.to_working(A, model.working_space)
    diag["working_space"] = model.working_space

    # ---- stage 1: global transfer functions -----------------------------
    if opts.color_mode == "grading":
        yb, ya = cs.luma(Bw), cs.luma(Aw)
        fit = C.estimate_transfer(yb, ya, W, n_bins=opts.n_bins, lam=opts.smooth)
        model.master = fit.lut
        diag["curve_coverage"] = {"luma": round(fit.coverage, 3)}
        diag["input_support"] = {"luma": [round(v * 255, 1) for v in fit.support]}
    else:
        fits = [C.estimate_transfer(Bw[:, c], Aw[:, c], W,
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
        if lo > 24 or hi < 231:
            diag["warnings"] = diag.get("warnings", []) + [
                f"only input levels {lo:.0f}-{hi:.0f} carry enough pixels to measure "
                f"({model.working_space} space); outside that range the curve is "
                f"extrapolated along its local slope. Add more image pairs, or ones "
                f"with deeper blacks and brighter highlights, to widen it."]

    # ---- stage 2: alternate residual stages and curve refinement --------
    # Each stage is fitted in its *own* domain: the later stages are peeled
    # off the target by inverting them, so the tone curve is not asked to
    # explain colour-mixer work and vice versa.
    history = []
    for it in range(max(1, opts.iterations)):
        if it > 0:
            pre = invert_post(Aw, model, in_working_space=True)
            if opts.color_mode == "grading":
                h = C.estimate_transfer(cs.luma(Bw), cs.luma(pre), W,
                                        n_bins=opts.n_bins, lam=opts.smooth)
                model.master = h.lut
            else:
                set_channel_transfers(model, [
                    C.estimate_transfer(Bw[:, c], pre[:, c], W,
                                        n_bins=opts.n_bins, lam=opts.smooth).lut
                    for c in range(3)])

        mid = render(Bw, model, upto="curves", in_working_space=True)

        if opts.fit_hsl:
            bands, hdiag = fit_hsl(mid, Aw, W, opts.calibration,
                                   fit_hue=opts.fit_hsl_hue,
                                   fit_sat=True,
                                   fit_lum=opts.fit_hsl_lum)
            model.hsl = bands
            diag["hsl_detail"] = hdiag

        if opts.color_mode in ("grading", "both"):
            mid2 = render(Bw, model, upto="hsl", in_working_space=True)
            zones, blending, balance, gdiag = fit_grading(
                mid2, Aw, W, opts.calibration, use_global=opts.grading_global)
            model.grade = zones
            model.grade_blending = round(blending, 1)
            model.grade_balance = round(balance, 1)
            diag["grading_detail"] = gdiag

        if opts.saturation_mode == "basic":
            mid3 = render(Bw, model, upto="grading", in_working_space=True)
            sat, vib = fit_basic_saturation(mid3, Aw, W, opts.calibration)
            model.saturation, model.vibrance = round(sat, 1), round(vib, 1)

        out = render(B, model)
        history.append(M.compare(out, A, W))   # scored in sRGB, as delivered

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

    # how much does the working-space choice matter for *these* images?
    other = "srgb" if model.working_space == "melissa" else "melissa"
    alt = PresetModel.from_dict(model.to_dict())
    alt.working_space = other
    diag["working_space_sensitivity"] = {
        "alternative": other,
        "dE_mean_if_applied_in_alternative_space":
            M.compare(render(B, alt), render(B, model), W)["dE_mean"],
        "note": ("how far the same curves land if the host applies them in the "
                 "other space; large values mean the working space matters more "
                 "than the rest of the fit"),
    }

    diag["diagnostics"] = basic.diagnose(model, B, A, W)
    diag["elapsed_sec"] = round(time.time() - t0, 2)
    model.meta["diagnostics"] = diag["diagnostics"]
    model.meta["verification"] = diag["verification_mean"]
    return model, diag


def extract_auto(pairs: list[PairData], opts: ExtractOptions) -> tuple[PresetModel, dict]:
    """Fit in every candidate working space and keep the one that explains the
    pair best.

    Lightroom edits in ProPhoto primaries, but a "before/after" downloaded
    from the web may equally have been produced by a tool working in sRGB, or
    re-encoded in ways that blur the distinction.  A per-channel curve in one
    space is genuinely not expressible as a per-channel curve in the other, so
    the residual tells us which one the editor used -- no guessing required.
    """
    if opts.working_space != "auto":
        return extract(pairs, opts)

    results = []
    for space in cs.WORKING_SPACES:
        o = replace(opts, working_space=space)
        model, diag = extract(pairs, o)
        results.append((diag["verification_mean"]["dE_mean"], space, model, diag))

    results.sort(key=lambda r: r[0])
    best_de, best_space, model, diag = results[0]
    diag["working_space_selection"] = {
        "chosen": best_space,
        "dE_mean_by_space": {sp: de for de, sp, _, _ in results},
        "margin": round(results[1][0] - best_de, 3) if len(results) > 1 else None,
        "note": ("chosen by which space explains the pair best; a small margin "
                 "means the images cannot distinguish the two"),
    }
    return model, diag
