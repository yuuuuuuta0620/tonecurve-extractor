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
        if m.get("measurable") is False:
            out.append(f"  {m['colour']:<19}— {m['source']}")
            continue
        src = "" if m["source"].startswith("measured") else "  [predicted]"
        hue = f"  hue {m['hue_shift_deg']:+.0f}°" if m.get("hue_shift_deg") is not None else ""
        pct = f" ({m['chroma_pct']:+.0f}%)" if m.get("chroma_pct") is not None else ""
        out.append(f"  {m['colour']:<19}{b} ->{a}   ΔE {m['dE']:>5.2f}  "
                   f"L {m['dL']:+5.1f}  chroma {m['dC']:+5.1f}{pct}{hue}{src}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Characteristic colour patches
#
# Numbers describe the colour work; crops let you see it.  These are the
# regions where the colour change is both strong and legible -- large, flat
# enough to read as a colour rather than an edge, and spread across different
# hues so you are not handed six crops of the same shirt.  Each one is a
# candidate for Lightroom's Point Colour: pick the "before" colour, then move
# hue / saturation / luminance until it matches the "after".
# --------------------------------------------------------------------------

def characteristic_patches(pairs, model: PresetModel, n_patches: int = 6,
                           grid: int = 22, context: float = 2.2) -> list[dict]:
    import cv2

    cands: list[dict] = []
    for pi, p in enumerate(pairs):
        before, after = p.before, p.after
        tone = _tone_only(before, model)
        h, w = before.shape[:2]
        ps = max(12, min(h, w) // grid)

        lab_t, lab_a = cs.rgb_to_lab(tone), cs.rgb_to_lab(after)
        d_ab = np.hypot(lab_a[..., 1] - lab_t[..., 1], lab_a[..., 2] - lab_t[..., 2])
        ok = p.weight > 0.15
        if getattr(p, "outlier_mask", None) is not None:
            ok &= ~p.outlier_mask

        def blocks(x):
            hh, ww = (h // ps) * ps, (w // ps) * ps
            return x[:hh, :ww].reshape(hh // ps, ps, ww // ps, ps).transpose(0, 2, 1, 3)

        change = blocks(d_ab).mean(axis=(2, 3))
        valid = blocks(ok.astype(np.float64)).mean(axis=(2, 3))
        # a patch is legible when its own colour is consistent inside
        spread = (blocks(lab_t[..., 1]).std(axis=(2, 3))
                  + blocks(lab_t[..., 2]).std(axis=(2, 3)))
        legible = 1.0 / (1.0 + (spread / 8.0) ** 2)
        score = change * legible * (valid > 0.9)

        for by, bx in zip(*np.unravel_index(np.argsort(score, axis=None)[::-1][:120],
                                            score.shape)):
            y0, x0 = by * ps, bx * ps
            sl = (slice(y0, y0 + ps), slice(x0, x0 + ps))
            src = np.median(tone[sl].reshape(-1, 3), axis=0)
            dst = np.median(after[sl].reshape(-1, 3), axis=0)
            cands.append({"pair": int(pi), "y": int(y0), "x": int(x0), "size": int(ps),
                          "score": float(score[by, bx]),
                          "before": src, "after": dst,
                          "raw_before": np.median(before[sl].reshape(-1, 3), axis=0)})

    cands.sort(key=lambda c: -c["score"])

    # keep them varied: at most two per hue family, and never overlapping
    chosen: list[dict] = []
    per_band: dict[int, int] = {}
    for c in cands:
        if len(chosen) >= n_patches:
            break
        hue, sat, _ = cs.rgb_to_hsl(c["before"])
        band = int(np.argmax(band_weights(np.array(float(hue)))))
        if per_band.get(band, 0) >= 2:
            continue
        if any(o["pair"] == c["pair"]
               and abs(o["y"] - c["y"]) < 3 * c["size"]
               and abs(o["x"] - c["x"]) < 3 * c["size"] for o in chosen):
            continue
        per_band[band] = per_band.get(band, 0) + 1

        lb, la = cs.rgb_to_lab(c["before"]), cs.rgb_to_lab(c["after"])
        cb, ca = float(np.hypot(lb[1], lb[2])), float(np.hypot(la[1], la[2]))
        h1, s1, l1 = cs.rgb_to_hsl(c["before"])
        h2, s2, l2 = cs.rgb_to_hsl(c["after"])
        c.update({
            "band": BAND_NAMES[band],
            "dE": round(float(cs.delta_e2000(lb, la)), 2),
            "dL": round(float(la[0] - lb[0]), 1),
            "dC": round(ca - cb, 1),
            "hue_shift_deg": round(float(cs.wrap_deg(h2 - h1)), 1) if min(s1, s2) > 0.04 else None,
            "saturation_pct": round(100.0 * (float(s2) / max(float(s1), 1e-6) - 1.0), 1)
                              if s1 > 0.04 else None,
            "lightness_pct": round(100.0 * (float(l2) / max(float(l1), 1e-6) - 1.0), 1),
        })
        chosen.append(c)

    # crop the surrounding context so the region is recognisable
    for c in chosen:
        p = pairs[c["pair"]]
        r = int(c["size"] * context)
        cy, cx = c["y"] + c["size"] // 2, c["x"] + c["size"] // 2
        h, w = p.before.shape[:2]
        y0, y1 = max(0, cy - r), min(h, cy + r)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        c["crop"] = [int(x0), int(y0), int(x1), int(y1)]
        c["crop_before"] = np.clip(p.before[y0:y1, x0:x1], 0, 1)
        c["crop_after"] = np.clip(p.after[y0:y1, x0:x1], 0, 1)
        c["crop_tone"] = np.clip(_tone_only(p.before[y0:y1, x0:x1], model), 0, 1)
        for k in ("before", "after", "raw_before"):
            c[k] = [round(float(v), 4) for v in c[k]]
        c.pop("score", None)
    return chosen


def format_patches(patches: list[dict]) -> str:
    if not patches:
        return ""
    out = ["", "characteristic colour changes — targets for Point Colour",
           "=" * 56,
           "pick the 'from' colour in Lightroom, then move it to 'to'.",
           "these are measured after the tone curve, so apply the curve first.", ""]
    for i, c in enumerate(patches, 1):
        f = "".join(f"{int(v * 255):4d}" for v in c["before"])
        t = "".join(f"{int(v * 255):4d}" for v in c["after"])
        hue = f"hue {c['hue_shift_deg']:+.0f}°" if c["hue_shift_deg"] is not None else "hue —"
        sat = (f"sat {c['saturation_pct']:+.0f}%" if c["saturation_pct"] is not None
               else "sat —")
        out.append(f"  {i}. {c['band']:<8} from{f}  to{t}   {hue}  {sat}  "
                   f"lum {c['lightness_pct']:+.0f}%   ΔE {c['dE']}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Putting the look into words
#
# Every statement below is a threshold on a measured quantity, listed with the
# number it came from, so it can be checked rather than taken on faith.  It is
# a reading of the measurements, not an interpretation of the photograph.
# --------------------------------------------------------------------------

def _warm_cool(hue: float) -> str:
    return "warm" if (hue < 100 or hue > 300) else "cool"


def describe_look(model: PresetModel, guide: dict, basic: dict | None = None) -> dict:
    """Traits of the look, each tied to the measurement that produced it."""
    traits: list[dict] = []
    m = C.apply_lut(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), model.master)
    black, white = float(m[0]), float(m[-1])
    slope = float((m[3] - m[1]) / 0.5)

    if black > 0.055:
        traits.append({"id": "lifted_blacks", "en": "lifted, matte blacks",
                       "ja": "黒が持ち上がったマット調",
                       "evidence": f"black point {black * 255:.0f}/255"})
    elif black < 0.012:
        traits.append({"id": "crushed_blacks", "en": "blacks held down, deep",
                       "ja": "黒を締めた深いコントラスト",
                       "evidence": f"black point {black * 255:.0f}/255"})
    if white < 0.95:
        traits.append({"id": "rolled_highlights", "en": "highlights pulled back",
                       "ja": "ハイライトを抑えた",
                       "evidence": f"white point {white * 255:.0f}/255"})
    if slope > 1.12:
        traits.append({"id": "punchy", "en": "punchy midtone contrast",
                       "ja": "中間調のコントラストが強い",
                       "evidence": f"mid slope {slope:.2f}"})
    elif slope < 0.9:
        traits.append({"id": "soft", "en": "soft, flattened midtones",
                       "ja": "中間調が寝ていて軟調",
                       "evidence": f"mid slope {slope:.2f}"})

    zones = {z["zone"]: z for z in guide["zones"] if z.get("measured")}
    sh, hi = zones.get("shadows"), zones.get("highlights")
    if sh and hi and sh["tint_strength_dE"] > 1.2 and hi["tint_strength_dE"] > 1.2:
        ws, wh = _warm_cool(sh["tint_hue_deg"]), _warm_cool(hi["tint_hue_deg"])
        if ws != wh:
            traits.append({
                "id": "split_tone",
                "en": f"split tone: {ws} {sh['tint_name']} shadows against "
                      f"{wh} {hi['tint_name']} highlights",
                "ja": f"シャドウ{sh['tint_name']}／ハイライト{hi['tint_name']}のスプリットトーン",
                "evidence": f"shadows {sh['tint_hue_deg']:.0f}° at {sh['tint_strength_dE']:.1f}, "
                            f"highlights {hi['tint_hue_deg']:.0f}° at {hi['tint_strength_dE']:.1f}"})
        else:
            traits.append({
                "id": "overall_cast",
                "en": f"an overall {ws} {hi['tint_name']} cast through the whole range",
                "ja": f"全域に{hi['tint_name']}系の色かぶり",
                "evidence": f"shadows {sh['tint_strength_dE']:.1f}, "
                            f"highlights {hi['tint_strength_dE']:.1f} ΔE"})
    elif sh and sh["tint_strength_dE"] > 1.5:
        traits.append({"id": "shadow_cast",
                       "en": f"{sh['tint_name']} shadows, highlights left alone",
                       "ja": f"シャドウだけ{sh['tint_name']}に振っている",
                       "evidence": f"shadow tint {sh['tint_strength_dE']:.1f} ΔE"})

    measured = [b for b in guide["bands"] if b.get("measured")]
    if measured:
        sats = {b["band"]: b["saturation_pct"] for b in measured}
        avg = float(np.mean(list(sats.values())))
        if avg < -8:
            traits.append({"id": "desaturated", "en": "desaturated overall",
                           "ja": "全体に彩度を落としている",
                           "evidence": f"mean saturation {avg:+.0f}%"})
        elif avg > 8:
            traits.append({"id": "saturated", "en": "saturation pushed up",
                           "ja": "全体に彩度を上げている",
                           "evidence": f"mean saturation {avg:+.0f}%"})
        drops = sorted((v, k) for k, v in sats.items() if v < -12)
        lifts = sorted(((v, k) for k, v in sats.items() if v > 12), reverse=True)
        if drops:
            names = ", ".join(k for _, k in drops[:2])
            traits.append({"id": "selective_desat",
                           "en": f"{names} pulled down while the rest is left",
                           "ja": f"{names} だけ彩度を大きく落としている",
                           "evidence": ", ".join(f"{k} {v:+.0f}%" for v, k in drops[:2])})
        if lifts:
            names = ", ".join(k for _, k in lifts[:2])
            traits.append({"id": "selective_sat", "en": f"{names} pushed up",
                           "ja": f"{names} を強調している",
                           "evidence": ", ".join(f"{k} {v:+.0f}%" for v, k in lifts[:2])})
        rot = sorted(measured, key=lambda b: -abs(b.get("hue_shift_deg") or 0))[0]
        if abs(rot.get("hue_shift_deg") or 0) > 8:
            d = rot["hue_shift_deg"]
            traits.append({"id": "hue_rotation",
                           "en": f"{rot['band']} rotated {d:+.0f}°",
                           "ja": f"{rot['band']} の色相を {d:+.0f}° 回している",
                           "evidence": f"{rot['band']} hue {d:+.1f}°"})

    ev = (basic or {}).get("exposure_ev")
    if ev is not None and abs(ev) > 0.4:
        traits.append({"id": "exposure",
                       "en": f"{'brighter' if ev > 0 else 'darker'} by {abs(ev):.1f} EV",
                       "ja": f"全体を {abs(ev):.1f} EV {'明るく' if ev > 0 else '暗く'}している",
                       "evidence": f"implied exposure {ev:+.2f} EV"})

    return {"traits": traits,
            "summary_en": "; ".join(t["en"] for t in traits) or "no strong character measured",
            "summary_ja": "、".join(t["ja"] for t in traits) or "際立った特徴は検出されませんでした"}


#: keys holding image data, which belong in the report and not in the JSON
PATCH_IMAGE_KEYS = ("crop_before", "crop_after", "crop_tone")


def patches_json(patches: list[dict]) -> list[dict]:
    """The numeric part of the patch list, safe to serialise."""
    return [{k: v for k, v in c.items() if k not in PATCH_IMAGE_KEYS} for c in patches]
