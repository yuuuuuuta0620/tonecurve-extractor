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
from .render import render, invert_post, apply_exposure


@dataclass
class ExtractOptions:
    color_mode: str = "curves"        # tone | curves | grading | both
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

    # With two or more pairs, each published sample carries the exposure the
    # photographer chose for *that* frame on top of the preset.  The common
    # part is the preset and cannot be separated from one pair alone, but the
    # part that differs between pairs is provably not the preset -- so it is
    # levelled out before the pairs are pooled, instead of letting them fight.
    normalize_pair_exposure: bool = True
    normalize_pair_white_balance: bool = True
    #: set when the caller has already prepared PairData.fit_before and does
    #: not want extract() to recompute it from the raw before/after ratios
    pair_inputs_prepared: bool = False
    #: measure the colour work in human terms, for reproducing by hand
    colour_guide: bool = True
    n_patches: int = 6                # crops of characteristic colour changes
    colour_chart: bool = True         # before/after swatch chart to match by eye
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


def pair_channel_gains(p: PairData) -> np.ndarray:
    """Per-channel linear-light gain of one pair: exposure and white balance
    rolled into three numbers."""
    lb = cs.srgb_to_linear(p.before_s)
    la = cs.srgb_to_linear(p.after_s)
    g = np.ones(3)
    for c in range(3):
        m = (lb[..., c] > 0.002) & (la[..., c] > 0.002) & (p.weight > 0)
        if m.sum() > 100:
            g[c] = 2.0 ** weighted_median(
                np.log2(la[..., c][m] / lb[..., c][m]), p.weight[m])
    return g


def pair_exposure_ev(p: PairData) -> float:
    """Robust overall brightness difference of one pair, in stops."""
    return float(np.log2(np.exp(np.mean(np.log(pair_channel_gains(p))))))


def level_pair_gains(pairs: list[PairData], white_balance: bool = True) -> dict:
    """Remove the between-pair exposure and white-balance spread.

    Sellers set exposure *and* white balance per frame.  Whatever is common to
    every sample might be the preset and is left alone; whatever differs
    between them provably is not, so it is levelled before the pairs are
    pooled -- otherwise they pull the shared curve in different directions and
    the fit is worse than any single pair would have been.
    """
    G = np.array([pair_channel_gains(p) for p in pairs])
    ref = np.median(G, axis=0)
    evs = np.log2(np.exp(np.log(G).mean(axis=1)))

    if not white_balance:                      # exposure only: keep the tint
        scale = np.exp(np.log(G).mean(axis=1))[:, None] / np.exp(np.log(ref).mean())
        adj = np.repeat(scale, 3, axis=1)
    else:
        adj = G / ref

    for p, a in zip(pairs, adj):
        if np.abs(np.log2(a)).max() > 0.02:
            lin = cs.srgb_to_linear(p.before_s) * a
            p.fit_before = np.clip(cs.linear_to_srgb(lin), 0.0, 1.0)
        else:
            p.fit_before = None

    tint = G / np.exp(np.log(G).mean(axis=1))[:, None]     # gains with exposure divided out
    return {"per_pair_ev": [round(float(v), 3) for v in evs],
            "reference_ev": round(float(np.log2(np.exp(np.log(ref).mean()))), 3),
            "spread_ev": round(float(evs.max() - evs.min()), 3),
            "per_pair_tint_rb": [round(float(t[0] / t[2]), 4) for t in tint],
            "spread_tint_pct": round(float(
                100 * np.ptp(tint[:, 0] / tint[:, 2])
                / np.median(tint[:, 0] / tint[:, 2])), 2),
            "white_balance_levelled": bool(white_balance)}


def extract(pairs: list[PairData], opts: ExtractOptions) -> tuple[PresetModel, dict]:
    """Fit a preset in one concrete working space.  Use ``extract_auto`` to let
    the data choose the space."""
    if opts.working_space not in cs.WORKING_SPACES:
        raise ValueError(f"extract() needs a concrete working space "
                         f"{cs.WORKING_SPACES}, got {opts.working_space!r}; "
                         f"use extract_auto() for 'auto'")
    t0 = time.time()

    if opts.pair_inputs_prepared:
        lev = {}
    elif opts.normalize_pair_exposure and len(pairs) > 1:
        lev = level_pair_gains(pairs, white_balance=opts.normalize_pair_white_balance)
    else:
        for p in pairs:
            p.fit_before = None
        lev = ({"per_pair_ev": [round(pair_exposure_ev(pairs[0]), 3)], "spread_ev": 0.0}
               if pairs else {})

    B, A, W = sample_pixels(pairs, opts.max_samples)
    diag: dict = {"n_pairs": len(pairs), "n_samples": int(B.shape[0]),
                  "options": {k: v for k, v in asdict(opts).items() if k != "calibration"},
                  "pairs": [p.info for p in pairs]}

    model = PresetModel(calibration=opts.calibration, name=opts.name, group=opts.group,
                        working_space=opts.working_space)
    W0 = W.copy()          # geometry-derived weights, never discarded

    # How much of the difference is plain brightness?  Always measured, and
    # moved into an Exposure slider only when asked.
    yb, ya = cs.relative_luminance(B), cs.relative_luminance(A)
    lit = (yb > 0.002) & (ya > 0.002)
    implied_ev = float(weighted_median(np.log2(ya[lit] / yb[lit]), W[lit])) \
        if lit.sum() > 100 else 0.0
    diag["implied_exposure_ev"] = round(implied_ev, 3)
    if lev:
        diag["pair_exposure"] = lev
    if lev.get("spread_tint_pct", 0) > 4:
        diag["warnings"] = diag.get("warnings", []) + [
            f"the sample pairs also disagree on white balance by "
            f"{lev['spread_tint_pct']:.1f}% (red/blue ratio {lev['per_pair_tint_rb']}). "
            f"Like the exposure spread, that is per-photo work; it has been levelled "
            f"out before fitting."]
    if lev.get("spread_ev", 0) > 0.35:
        diag["warnings"] = diag.get("warnings", []) + [
            f"the sample pairs disagree on exposure by {lev['spread_ev']:.2f} EV "
            f"({lev['per_pair_ev']}). That difference cannot be part of a preset — it is "
            f"per-photo work the seller did on each frame. It has been levelled out "
            f"before fitting, but whatever exposure they applied to *all* the samples "
            f"is still inside the curve and will follow the preset onto your photos."]
    n = len(pairs)
    if opts.color_mode == "tone":
        # A master curve has too few degrees of freedom to absorb per-frame
        # work, so it does not overfit and one pair is close to the ceiling.
        # Measured on unseen photographs (per-frame exposure sd 0.4 EV, WB sd
        # 2%, q85 4:2:0 JPEG): 1 pair ΔE 6.42 spanning 5.97-7.55, 6 pairs ΔE
        # 6.06 spanning 5.93-6.15.  Only the exposure caveat still applies.
        if abs(implied_ev) > 0.5:
            diag["warnings"] = diag.get("warnings", []) + [
                f"the curve carries {implied_ev:+.2f} EV of overall brightness, part of "
                f"it the preset and part the exposure chosen for these frames. Set "
                f"Exposure to taste on your own photographs; the shape of the curve is "
                f"what transfers."]
    elif n <= 4:
        # measured on ground truth with realistic per-frame variation in the
        # samples (exposure sd 0.4 EV, white balance sd 2 %, q85 4:2:0 JPEG):
        # median / worst-case ΔE when the preset is applied to an unseen photo
        table = {1: ("5.7", "6.2"), 2: ("2.7", "3.4"), 3: ("1.4", "5.6"), 4: ("1.7", "2.8")}
        med, worst = table[n]
        detail = ("a single pair cannot separate the preset from the exposure, white "
                  "balance and local work the seller did on that one frame"
                  if n == 1 else
                  f"{n} pairs begin to separate the preset from per-frame work, but not "
                  f"reliably")
        diag["warnings"] = diag.get("warnings", []) + [
            f"fitted from {n} pair{'s' if n > 1 else ''}: {detail}. On ground truth this "
            f"transfers to an unseen photograph at ΔE {med} typical, {worst} worst case "
            f"— against ΔE 1.4 typical / 2.3 worst at 8 pairs. Add more pairs from "
            f"different scenes if the preset is meant for your own photographs."]
    if abs(implied_ev) > 0.5 and opts.color_mode != "tone":
        diag["warnings"] = diag.get("warnings", []) + [
            f"the pair differs by {implied_ev:+.2f} EV of overall brightness, all of it "
            f"carried by the tone curve. Part of that is the preset and part is the "
            f"exposure the seller chose for this frame, and one pair cannot tell them "
            f"apart — so applied to your own raw files this curve will push them toward "
            f"this sample's brightness. Several pairs from different scenes let the "
            f"per-frame part be identified and removed."]

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
    if opts.color_mode in ("grading", "tone"):
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
            if opts.color_mode in ("grading", "tone"):
                h = C.estimate_transfer(cs.luma(Bw), cs.luma(pre), W,
                                        n_bins=opts.n_bins, lam=opts.smooth)
                model.master = h.lut
            else:
                set_channel_transfers(model, [
                    C.estimate_transfer(Bw[:, c], pre[:, c], W,
                                        n_bins=opts.n_bins, lam=opts.smooth).lut
                    for c in range(3)])

        mid = render(Bw, model, upto="curves", in_working_space=True)

        if opts.fit_hsl and opts.color_mode != "tone":
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

        if opts.saturation_mode == "basic" and opts.color_mode != "tone":
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

    detail = diag.get("hsl_detail") or {}
    thin = [n for n, v in detail.items()
            if v.get("used") and v.get("data_share", 1) < 0.01
            and any(abs(getattr(model.hsl[n], f)) > 15 for f in ("hue", "sat", "lum"))]
    railed = [f"{n}.{c}" for n, v in detail.items() for c in v.get("clipped", [])]
    if thin:
        diag["warnings"] = diag.get("warnings", []) + [
            f"colour-mixer bands {', '.join(thin)} were fitted from under 1% of the "
            f"pixels — those sliders are barely measured here and may not suit other "
            f"photographs. More image pairs, covering more colours, would settle them."]
    if railed:
        diag["warnings"] = diag.get("warnings", []) + [
            f"these sliders hit the ±100 limit and were clipped: {', '.join(railed)}. "
            f"A slider on the rail usually means the model is being asked to explain "
            f"something it cannot express, not that the preset really is that extreme."]

    if opts.colour_chart:
        from .chart import build_chart
        diag["colour_chart"] = build_chart(B, A, W0, model)

    if opts.colour_guide:
        from .guide import build_guide, characteristic_patches, describe_look
        g = build_guide(B, A, W0, model, diag.get("hsl_detail"))
        diag["colour_guide"] = g
        diag["look_description"] = describe_look(
            model, g, {"exposure_ev": diag.get("implied_exposure_ev")})
        model.meta["look"] = diag["look_description"]["summary_en"]
        try:
            diag["patches"] = characteristic_patches(pairs, model, opts.n_patches)
        except Exception as e:                     # never lose a fit over a nicety
            diag["patches_error"] = f"{type(e).__name__}: {e}"

    diag["diagnostics"] = basic.diagnose(model, B, A, W)
    diag["elapsed_sec"] = round(time.time() - t0, 2)
    model.meta["diagnostics"] = diag["diagnostics"]
    model.meta["verification"] = diag["verification_mean"]
    return model, diag


def residual_pair_gains(pairs: list[PairData], model: PresetModel) -> np.ndarray:
    """Per-pair, per-channel gain still left over after the fitted preset.

    Measuring the raw before/after ratio confounds the photographer's exposure
    with the preset's own brightening, and because the preset is a curve
    rather than a gain, that confound varies with each frame's histogram --
    so levelling on it injects differences of its own.  Measured against the
    rendered prediction instead, the preset cancels and what remains is the
    per-frame offset we actually want to remove.
    """
    G = np.ones((len(pairs), 3))
    for i, p in enumerate(pairs):
        src = p.fit_before if p.fit_before is not None else p.before_s
        pred = cs.srgb_to_linear(render(src, model))
        tgt = cs.srgb_to_linear(p.after_s)
        for c in range(3):
            m = (pred[..., c] > 0.004) & (tgt[..., c] > 0.004) & (p.weight > 0)
            if m.sum() > 100:
                G[i, c] = 2.0 ** weighted_median(
                    np.log2(tgt[..., c][m] / pred[..., c][m]), p.weight[m])
    return G


def relevel_pairs(pairs: list[PairData], model: PresetModel) -> dict:
    """Fold the leftover per-pair offsets into the sampling inputs."""
    G = residual_pair_gains(pairs, model)
    adj = G / np.median(G, axis=0)
    for p, a in zip(pairs, adj):
        base = p.fit_before if p.fit_before is not None else p.before_s
        if np.abs(np.log2(a)).max() > 0.01:
            p.fit_before = np.clip(
                cs.linear_to_srgb(cs.srgb_to_linear(base) * a), 0.0, 1.0)
    evs = np.log2(np.exp(np.log(G).mean(axis=1)))
    return {"residual_ev_per_pair": [round(float(v), 3) for v in evs],
            "residual_spread_ev": round(float(np.ptp(evs)), 3)}


def extract_levelled(pairs: list[PairData], opts: ExtractOptions,
                     space: str) -> tuple[PresetModel, dict]:
    """Fit, measure what each pair still disagrees about, level it, fit again."""
    o = replace(opts, working_space=space)
    model, diag = extract(pairs, o)
    if len(pairs) < 2 or not opts.normalize_pair_exposure:
        return model, diag
    prepared = [p.fit_before for p in pairs]
    info = relevel_pairs(pairs, model)
    model2, diag2 = extract(pairs, replace(o, pair_inputs_prepared=True))
    if diag2["verification_mean"]["dE_mean"] <= diag["verification_mean"]["dE_mean"]:
        diag2["releveling"] = info | {"kept": True}
        return model2, diag2
    for p, prev in zip(pairs, prepared):   # levelling made it worse: undo
        p.fit_before = prev
    diag["releveling"] = info | {"kept": False}
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
        return extract_levelled(pairs, opts, opts.working_space)

    results = []
    for space in cs.WORKING_SPACES:
        model, diag = extract_levelled(pairs, opts, space)
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


def explain_residual(pairs: list[PairData], opts: ExtractOptions,
                     joint_diag: dict, space: str) -> dict:
    """Say whether a disappointing fit is a limit or a fixable problem.

    Fit each pair on its own and compare with the joint fit.  A pair a global
    preset *can* explain fits well alone; if it also fits well jointly, it
    agrees with the others.  The two failure modes look completely different:

      solo good, joint bad -> the samples disagree.  Different per-frame work,
        a different preset variant, or one odd sample.  Droppable.
      solo bad too         -> no global preset explains that frame at all:
        brushes, masks, subject/sky selections.  That is the real limit.
    """
    solo = []
    o = replace(opts, iterations=min(2, opts.iterations), working_space=space,
                normalize_pair_exposure=False)
    for p in pairs:
        keep = p.fit_before
        p.fit_before = None
        try:
            _, d = extract([p], o)
            solo.append(d["verification"][0]["preset"]["dE_mean"])
        finally:
            p.fit_before = keep

    joint = [v["preset"]["dE_mean"] for v in joint_diag["verification"]]
    solo_med = float(np.median(solo))
    joint_med = float(np.median(joint))
    gap = joint_med - solo_med

    local = solo_med > 3.0        # no global preset explains even one frame alone
    disagree = gap > 1.5          # each is explainable, but not by the *same* preset
    if local and disagree:
        verdict = ("both problems: the frames carry local work no preset can reproduce, "
                   "and they also disagree with each other. Fix the disagreement first "
                   "— drop the worst pairs — then judge what is left.")
    elif local:
        verdict = ("the samples carry local work (brushes, subject or sky masks, "
                   "retouching) that no global preset can reproduce. Each frame resists "
                   "a preset even on its own, so more pairs will not help. This is the "
                   "real limit for these samples.")
    elif disagree:
        verdict = ("each frame fits well alone but they disagree with each other — they "
                   "were not all produced by the same global edit. Drop the worst pairs "
                   "and re-run; the preset is recoverable from the ones that agree.")
    else:
        verdict = ("the pairs agree and each is well explained; this fit is about as "
                   "good as these samples allow.")

    return {"local_work_suspected": bool(local),
            "pairs_disagree": bool(disagree),
            "solo_dE_per_pair": [round(v, 3) for v in solo],
            "joint_dE_per_pair": [round(v, 3) for v in joint],
            "solo_median": round(solo_med, 3),
            "joint_median": round(joint_med, 3),
            "disagreement": round(gap, 3),
            "verdict": verdict}
