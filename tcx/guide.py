"""A written guide to the colour work, for dialling in by hand.

The tone curve is measured directly and transfers reliably.  The colour
sliders are a different matter: Adobe does not publish how Hue/Saturation/
Luminance or the grading wheels respond, the input is lossy, and a hue band
the samples barely cover is fitted from almost nothing.  Rather than only
emitting numbers whose meaning we had to assume, this module reports *what
the transformation actually does*, ranked by how much it matters, in terms a
person can reproduce in Lightroom and judge by eye.

Everything here is measured from the pixels, not read back off the fitted
sliders, except where a band has too little data and it says so.
"""
from __future__ import annotations

import numpy as np

from . import colorspace as cs
from . import curves as C
from .model import BAND_NAMES, PresetModel
from .render import band_weights, grade_masks
from .robust import weighted_median

#: colours worth knowing about, as sRGB
MEMORY_COLOURS = {
    "skin (light)": (0.87, 0.71, 0.62),
    "skin (mid)": (0.68, 0.50, 0.42),
    "sky": (0.42, 0.62, 0.85),
    "foliage": (0.34, 0.48, 0.25),
    "neutral shadow": (0.20, 0.20, 0.20),
    "neutral mid": (0.46, 0.46, 0.46),
    "neutral highlight": (0.84, 0.84, 0.84),
}

ZONES = ("shadows", "midtones", "highlights")


def _tone_only(rgb: np.ndarray, m: PresetModel) -> np.ndarray:
    """The preset's luminance curve alone, applied to all three channels."""
    w = cs.to_working(rgb, m.working_space)
    out = np.stack([C.apply_lut(w[..., c], m.master) for c in range(3)], axis=-1)
    return cs.from_working(out, m.working_space)


def _median_lab(rgb: np.ndarray, w: np.ndarray) -> np.ndarray:
    lab = cs.rgb_to_lab(rgb)
    return np.array([weighted_median(lab[..., i], w) for i in range(3)])


def build_guide(before: np.ndarray, after: np.ndarray, weight: np.ndarray,
                model: PresetModel, hsl_detail: dict | None = None) -> dict:
    """Describe the colour work left over once the tone curve is accounted for."""
    tone = _tone_only(before, model)
    resid = cs.delta_e2000(cs.rgb_to_lab(tone), cs.rgb_to_lab(after))
    total = float(np.average(resid, weights=weight))

    h1, s1, l1 = cs.rgb_to_hsl(tone)
    h2, s2, l2 = cs.rgb_to_hsl(after)
    W = band_weights(h1, model.calibration.band_falloff)
    mass = (W * weight[:, None]).sum(axis=0)
    share = mass / max(mass.sum(), 1e-12)

    # ---- what happens inside each hue band -----------------------------
    bands = []
    for i, name in enumerate(BAND_NAMES):
        bw = W[:, i] * weight
        usable = bw > 0
        colour_ok = usable & (s1 > 0.10) & (l1 > 0.03) & (l1 < 0.97)
        entry = {"band": name, "data_share": round(float(share[i]), 4)}
        if colour_ok.sum() > 200 and bw[colour_ok].sum() > 50:
            ww = bw[colour_ok] * s1[colour_ok]
            entry["hue_shift_deg"] = round(
                float(weighted_median(cs.wrap_deg(h2 - h1)[colour_ok], ww)), 1)
            entry["saturation_pct"] = round(100.0 * (float(weighted_median(
                (s2 / np.maximum(s1, 1e-6))[colour_ok], ww)) - 1.0), 1)
            entry["lightness_pct"] = round(100.0 * (float(weighted_median(
                (l2 / np.maximum(l1, 1e-6))[colour_ok], ww)) - 1.0), 1)
            entry["explains_dE"] = round(
                float((resid * bw).sum() / max(weight.sum(), 1e-12)), 3)
            entry["measured"] = True
        else:
            entry["measured"] = False
            entry["note"] = "too few pixels of this hue to measure"
        if hsl_detail and name in hsl_detail:
            entry["fitted_slider_trustworthy"] = bool(hsl_detail[name].get("used"))
        bands.append(entry)

    ranked = sorted([b for b in bands if b.get("measured")],
                    key=lambda b: -b.get("explains_dE", 0))

    # ---- the cast on near-neutral pixels, by tonal zone ------------------
    # This is what the Colour Grading wheels do, so it maps straight onto them.
    y = cs.luma(tone)
    masks = grade_masks(y, 50.0, 0.0)
    neutralish = (s1 < 0.22) & (weight > 0)
    zones = []
    for zi, zname in enumerate(ZONES):
        zw = masks[:, zi] * weight * neutralish
        if zw.sum() < 200:
            zones.append({"zone": zname, "measured": False,
                          "note": "too few near-neutral pixels in this zone"})
            continue
        lab_b = _median_lab(tone, zw)
        lab_a = _median_lab(after, zw)
        da, db = lab_a[1] - lab_b[1], lab_a[2] - lab_b[2]
        chroma = float(np.hypot(da, db))
        hue = float(np.degrees(np.arctan2(db, da)) % 360.0)
        zones.append({
            "zone": zname, "measured": True,
            "tint_hue_deg": round(hue, 1),
            "tint_strength_dE": round(chroma, 2),
            "tint_name": _hue_name(hue),
            "lightness_dL": round(float(lab_a[0] - lab_b[0]), 2),
        })

    # ---- what it does to colours people care about ----------------------
    # Measured from the *tone-corrected* image, so the numbers describe the
    # colour work left to do rather than repeating what the curve already did.
    memory = []
    lab_all_b = cs.rgb_to_lab(tone)
    for name, ref in MEMORY_COLOURS.items():
        d = cs.delta_e2000(lab_all_b, cs.rgb_to_lab(np.array(ref)))
        near = (d < 12) & (weight > 0)
        item = {"colour": name, "reference": [round(v, 3) for v in ref]}
        if near.sum() > 300:
            item["source"] = "measured from the photographs"
            item["n_pixels"] = int(near.sum())
            item["before"] = [round(float(v), 3)
                              for v in np.median(tone[near], axis=0)]
            item["after"] = [round(float(v), 3)
                             for v in np.median(after[near], axis=0)]
        else:
            item["source"] = "predicted by the fitted preset (too few such pixels)"
            item["before"] = [round(v, 3) for v in ref]
            from .render import render
            item["before"] = [round(float(v), 3)
                              for v in _tone_only(np.array(ref)[None, :], model)[0]]
            item["after"] = [round(float(v), 3)
                             for v in render(np.array(ref)[None, :], model)[0]]
        lb = cs.rgb_to_lab(np.array(item["before"]))
        la = cs.rgb_to_lab(np.array(item["after"]))
        item["dE"] = round(float(cs.delta_e2000(lb, la)), 2)
        item["dL"] = round(float(la[0] - lb[0]), 1)
        cb, ca = float(np.hypot(lb[1], lb[2])), float(np.hypot(la[1], la[2]))
        item["chroma_before"] = round(cb, 1)
        item["dC"] = round(ca - cb, 1)
        # a percentage is meaningless off a near-neutral base
        item["chroma_pct"] = round(100.0 * (ca / cb - 1.0), 1) if cb > 5 else None
        item["hue_shift_deg"] = (
            round(float(cs.wrap_deg(np.degrees(np.arctan2(la[2], la[1]))
                                    - np.degrees(np.arctan2(lb[2], lb[1])))), 1)
            if cb > 5 and ca > 5 else None)
        memory.append(item)

    return {"total_colour_dE_after_tone": round(total, 3),
            "bands": bands, "ranked_bands": ranked[:4],
            "zones": zones, "memory_colours": memory}


def _hue_name(deg: float) -> str:
    names = [(15, "red"), (45, "orange"), (75, "yellow"), (150, "green"),
             (200, "teal"), (260, "blue"), (300, "purple"), (345, "magenta"),
             (361, "red")]
    for lim, n in names:
        if deg < lim:
            return n
    return "red"


def format_guide(g: dict) -> str:
    """Terminal rendering."""
    out = ["", "colour guide — what to dial in by hand",
           "=" * 44,
           f"once the tone curve is applied, ΔE {g['total_colour_dE_after_tone']} of colour "
           f"difference is left to reproduce", ""]

    out.append("tonal zones (these map onto the Colour Grading wheels)")
    for z in g["zones"]:
        if not z.get("measured"):
            out.append(f"  {z['zone']:<11} {z['note']}")
            continue
        out.append(f"  {z['zone']:<11} push toward {z['tint_name']} "
                   f"(hue {z['tint_hue_deg']:.0f}°), strength {z['tint_strength_dE']:.1f}"
                   f"   lightness {z['lightness_dL']:+.1f} L*")

    out += ["", "hue bands, strongest first (Colour Mixer)"]
    if not g["ranked_bands"]:
        out.append("  nothing measurable")
    for b in g["ranked_bands"]:
        trust = "" if b.get("fitted_slider_trustworthy", True) else "   [thin data]"
        out.append(f"  {b['band']:<9} hue {b['hue_shift_deg']:+6.1f}°   "
                   f"saturation {b['saturation_pct']:+6.1f}%   "
                   f"lightness {b['lightness_pct']:+6.1f}%   "
                   f"({b['data_share']:.0%} of pixels){trust}")

    out += ["", "colours you care about — after the tone curve, what colour is left to add"]
    for m in g["memory_colours"]:
        b = "".join(f"{int(v * 255):4d}" for v in m["before"])
        a = "".join(f"{int(v * 255):4d}" for v in m["after"])
        src = "" if m["source"].startswith("measured") else "  [predicted]"
        hue = f"  hue {m['hue_shift_deg']:+.0f}°" if m.get("hue_shift_deg") is not None else ""
        pct = f" ({m['chroma_pct']:+.0f}%)" if m.get("chroma_pct") is not None else ""
        out.append(f"  {m['colour']:<19}{b} ->{a}   ΔE {m['dE']:>5.2f}  "
                   f"L {m['dL']:+5.1f}  chroma {m['dC']:+5.1f}{pct}{hue}{src}")
    return "\n".join(out)
