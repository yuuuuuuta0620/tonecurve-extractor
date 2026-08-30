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
from .robust import weighted_median
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
    # Rejection of pixels the model cannot explain (burned-in watermarks and
    # captions, local adjustments, residual misalignment).  The cut is the
    # widest of: a robust multiple of the residual spread, a perceptual floor,
    # and the quantile that caps how much data may be discarded.
    reject_sigma: float = 6.0         # robust sigmas; 0 disables rejection
    reject_floor_de: float = 3.0      # never cut below this CIEDE2000
    reject_max_fraction: float = 0.25 # never discard more than this much weight

    # Burned-in opaque marks (watermarks, captions, logos) are *not* findable
    # from the fit residual: they sit at high-leverage tonal values, so the
    # curve simply bends to fit them and their residual goes small.  They are
    # findable before fitting anything, from the pair alone: the whole frame
    # moved, and these pixels did not move at all.
    detect_frozen: bool = True
    frozen_abs_de: float = 0.6        # "did not move" floor, above JPEG noise
    frozen_rel: float = 0.06          # ...or this fraction of the frame's median move
    frozen_max_fraction: float = 0.30 # refuse to act if this much looks frozen
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
    W0 = W.copy()          # geometry-derived weights, never discarded

    # The whole edit is fitted in Lightroom's working space, so the curve we
    # write into the XMP is the curve Lightroom will actually apply.  The two
    # spaces coincide on the neutral axis and differ on saturated colour.
    Bw = cs.to_working(B, model.working_space)
    Aw = cs.to_working(A, model.working_space)
    diag["working_space"] = model.working_space

    # ---- stage 0: pixels that did not move at all ------------------------
    # A burned-in watermark or caption is identical in both images, so it
    # asserts f(x) = x.  Left in, that assertion is applied with enormous
    # leverage wherever the mark's tone is rare -- a white caption pins the
    # highlight end of the curve.  It has to be removed *before* the first
    # fit: afterwards the curve has already bent to accommodate it and the
    # residual no longer reveals it.
    if opts.detect_frozen:
        observed = cs.delta_e2000(cs.rgb_to_lab(A), cs.rgb_to_lab(B))
        med_obs = weighted_median(observed, W0)
        thr = max(opts.frozen_abs_de, opts.frozen_rel * med_obs)
        frozen = observed < thr
        frac = float(np.average(frozen, weights=W0))
        info = {"frame_median_move_dE": round(float(med_obs), 3),
                "threshold_dE": round(float(thr), 3),
                "fraction": round(frac, 4)}
        if med_obs < 2.0:
            info["action"] = "skipped: the pair barely differs, nothing to compare against"
        elif frac > opts.frozen_max_fraction:
            info["action"] = (f"skipped: {frac:.0%} of the frame looks unmoved, which is "
                              f"more likely a weak preset than a watermark")
        else:
            W = W0 * ~frozen
            info["action"] = "excluded from the fit"
            if frac > 0.002:
                diag["warnings"] = diag.get("warnings", []) + [
                    f"{frac:.1%} of pixels are identical in both images while the rest of "
                    f"the frame moved by ΔE {med_obs:.1f} — looks like a burned-in "
                    f"watermark or caption; excluded from the fit"]
        diag["frozen_pixels"] = info

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
        history.append(M.compare(out, A, W0))  # scored in sRGB, as delivered

        # Pixels the fitted preset cannot explain are not evidence about the
        # preset: burned-in watermarks and captions, local adjustments, brush
        # work, residual misalignment.  Down-weight them and refit.  We reject
        # rather than inpaint, because inpainted pixels are invented colour and
        # this pipeline exists to *measure* colour.
        keep_frozen = np.where(frozen, 0.0, 1.0) if (
            opts.detect_frozen and diag.get("frozen_pixels", {}).get(
                "action") == "excluded from the fit") else None

        if opts.reject_sigma > 0:
            resid = cs.delta_e2000(cs.rgb_to_lab(out), cs.rgb_to_lab(A))
            med = weighted_median(resid, W0)
            sigma = 1.4826 * weighted_median(np.abs(resid - med), W0)
            cut = max(med + opts.reject_sigma * sigma, opts.reject_floor_de)
            # hard cap on how much evidence may be thrown away
            cut = max(cut, weighted_median(resid, W0, 1.0 - opts.reject_max_fraction))
            if not (np.isfinite(cut) and cut > 0):
                if keep_frozen is not None:
                    W = W0 * keep_frozen
            if np.isfinite(cut) and cut > 0:
                # smooth shoulder over the last 25 % below the cut
                lo = cut * 0.75
                keep = np.clip((cut - resid) / max(cut - lo, 1e-6), 0.0, 1.0) ** 2
                keep = np.where(resid <= lo, 1.0, keep)
                if keep_frozen is not None:
                    keep = keep * keep_frozen
                W = W0 * keep
                diag["outlier_rejection"] = {
                    "residual_median_dE": round(float(med), 3),
                    "residual_sigma_dE": round(float(sigma), 4),
                    "cut_dE": round(float(cut), 3),
                    "weight_removed_fraction":
                        round(float(1.0 - W.sum() / max(W0.sum(), 1e-12)), 4),
                    "fully_rejected_pixel_fraction": round(float(np.mean(keep <= 0)), 4),
                }

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
    # Reported over every pixel, so a watermark or a local edit still counts
    # against the score; the "explained" figure additionally shows the error
    # over just the pixels the global model claims to describe.
    ver = []
    for p in pairs:
        pred = render(p.before, model)
        base = M.compare(p.before, p.after, p.weight)
        got = M.compare(pred, p.after, p.weight)
        entry = {"size": p.info.get("working_size"),
                 "baseline": base, "preset": got,
                 "improvement": M.improvement(base, got)}
        if opts.reject_sigma > 0 or opts.detect_frozen:
            lab_b, lab_a, lab_p = (cs.rgb_to_lab(p.before), cs.rgb_to_lab(p.after),
                                   cs.rgb_to_lab(pred))
            resid = cs.delta_e2000(lab_p, lab_a)
            cut = (diag.get("outlier_rejection") or {}).get("cut_dE") or np.inf
            bad = resid > cut
            if diag.get("frozen_pixels", {}).get("action") == "excluded from the fit":
                thr = diag["frozen_pixels"]["threshold_dE"]
                bad = bad | (cs.delta_e2000(lab_a, lab_b) < thr)
            if True:
                inlier = p.weight * ~bad
                entry["preset_explained_pixels"] = M.compare(pred, p.after, inlier)
                entry["explained_pixel_fraction"] = round(float((~bad).mean()), 4)
                p.outlier_mask = bad
        ver.append(entry)
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
