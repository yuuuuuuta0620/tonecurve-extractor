"""Check an extracted preset against what Lightroom actually did with it.

Everything else in this program reasons about Lightroom from the outside.
This closes the loop: export the colour chart from Lightroom with the preset
applied, hand it back, and the ambiguity that cannot be settled from the
sample images alone -- which colour space Lightroom applied the curve in --
is settled by measurement, on your machine, with your version.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import colorspace as cs
from .model import PresetModel
from .render import render


def flat_interior_mask(chart: np.ndarray, tol: float = 0.004) -> np.ndarray:
    """Pixels well inside a swatch, away from the borders between them.

    Found from the image rather than from stored geometry, so a chart that was
    resized, re-exported or cropped on the way through Lightroom still works.
    """
    g = chart.astype(np.float32)
    k = np.ones((7, 7), np.float32) / 49.0
    mean = cv2.filter2D(g, -1, k)
    sq = cv2.filter2D(g * g, -1, k)
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0)).max(axis=2)
    flat = std < tol
    # pull back from the edges of each flat run so filtering cannot bleed in
    return cv2.erode(flat.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) \
        & (chart.max(axis=2) < 0.995) & (chart.min(axis=2) > 0.005)


def compare_export(chart_before: np.ndarray,
                   lightroom_export: np.ndarray,
                   model: PresetModel,
                   spaces=("melissa", "srgb")) -> dict:
    """Score each working-space hypothesis against Lightroom's own output."""
    if lightroom_export.shape[:2] != chart_before.shape[:2]:
        h, w = chart_before.shape[:2]
        lightroom_export = cv2.resize(lightroom_export, (w, h),
                                      interpolation=cv2.INTER_AREA)

    mask = flat_interior_mask(chart_before)
    if mask.sum() < 500:
        raise ValueError("could not find flat swatch areas — is this the chart image?")

    src = chart_before[mask]
    got = np.clip(lightroom_export[mask], 0, 1)
    lab_got = cs.rgb_to_lab(got)

    scores = {}
    for space in spaces:
        trial = PresetModel.from_dict(model.to_dict())
        trial.working_space = space
        d = cs.delta_e2000(cs.rgb_to_lab(render(src, trial)), lab_got)
        scores[space] = {"dE_mean": round(float(d.mean()), 3),
                         "dE_p95": round(float(np.percentile(d, 95)), 3),
                         "dE_max": round(float(d.max()), 3)}

    best = min(scores, key=lambda s: scores[s]["dE_mean"])
    others = [s for s in scores if s != best]
    margin = round(min(scores[s]["dE_mean"] for s in others) - scores[best]["dE_mean"], 3) \
        if others else None

    return {"pixels_compared": int(mask.sum()),
            "by_space": scores,
            "lightroom_uses": best,
            "margin": margin,
            "preset_uses": model.working_space,
            "agrees": best == model.working_space,
            "residual_dE": scores[best]["dE_mean"]}


def format_report(r: dict) -> str:
    out = ["", "what Lightroom actually did with this preset",
           "=" * 44,
           f"compared over {r['pixels_compared']:,} swatch pixels", ""]
    for space, s in r["by_space"].items():
        mark = "  <-- matches" if space == r["lightroom_uses"] else ""
        out.append(f"  {space:<9} ΔE mean {s['dE_mean']:>6}   p95 {s['dE_p95']:>6}"
                   f"   max {s['dE_max']:>6}{mark}")
    out.append("")
    if r["margin"] is not None and r["margin"] < 0.15:
        out.append("  the two hypotheses fit equally well — this chart cannot tell "
                   "them apart. A chart with more saturated colour would.")
    elif r["agrees"]:
        out.append(f"  Lightroom applies the curve in '{r['lightroom_uses']}', which is "
                   f"what this preset already assumes. Nothing to change.")
    else:
        out.append(f"  Lightroom applies the curve in '{r['lightroom_uses']}', but this "
                   f"preset was fitted for '{r['preset_uses']}'.")
        out.append(f"  Re-run the extraction with --working-space {r['lightroom_uses']} "
                   f"to remove that error.")
    out += ["",
            f"  residual after the better hypothesis: ΔE {r['residual_dE']}",
            "  what is left is 8-bit quantisation (~0.2), Lightroom's own curve",
            "  spline against ours, and its handling of non-raw files."]
    return "\n".join(out)
