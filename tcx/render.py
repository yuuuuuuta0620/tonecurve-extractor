"""Forward renderer -- the explicit definition of what an extracted preset means.

Every parameter we fit is fitted *through* this model, and verification
re-renders the "before" image with it, so the reported error is an honest
measure of how well the model explains the pair.

Pipeline order mirrors Lightroom Classic's develop order for the parts we
model:  point tone curves (master, then R/G/B)  ->  Colour Mixer / HSL  ->
Colour Grading  ->  (optional) Basic vibrance & saturation.
"""
from __future__ import annotations

import numpy as np

from . import colorspace as cs
from . import curves as C
from .model import PresetModel, Calibration, BAND_CENTERS, BAND_NAMES, ZONE_NAMES


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def band_weights(hue_deg: np.ndarray, falloff: str = "smooth") -> np.ndarray:
    """Partition-of-unity weights over the eight colour-mixer hue bands.

    Returns an array shaped (..., 8) whose entries sum to 1.
    """
    h = np.asarray(hue_deg, dtype=np.float64) % 360.0
    centers = BAND_CENTERS
    k = len(centers)
    w = np.zeros(h.shape + (k,), dtype=np.float64)

    ext = np.concatenate([centers, [centers[0] + 360.0]])
    for i in range(k):
        lo, hi = ext[i], ext[i + 1]
        span = hi - lo
        inside = (h >= lo) & (h < hi)
        t = np.zeros_like(h)
        np.divide(h - lo, span, out=t, where=inside)
        t = np.where(inside, t, 0.0)
        blend = _smoothstep(t) if falloff == "smooth" else t
        j = (i + 1) % k
        w[..., i] += np.where(inside, 1.0 - blend, 0.0)
        w[..., j] += np.where(inside, blend, 0.0)
    return w


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def apply_curves(rgb: np.ndarray, m: PresetModel) -> np.ndarray:
    out = np.empty_like(rgb)
    for i, ch in enumerate((m.red, m.green, m.blue)):
        out[..., i] = C.apply_lut(C.apply_lut(rgb[..., i], m.master), ch)
    return out


def apply_hsl(rgb: np.ndarray, m: PresetModel) -> np.ndarray:
    cal = m.calibration
    hues = np.array([m.hsl[n].hue for n in BAND_NAMES])
    sats = np.array([m.hsl[n].sat for n in BAND_NAMES])
    lums = np.array([m.hsl[n].lum for n in BAND_NAMES])
    if not (np.any(hues) or np.any(sats) or np.any(lums)):
        return rgb

    h, s, l = cs.rgb_to_hsl(rgb)
    W = band_weights(h, cal.band_falloff)

    if np.any(hues):
        h = (h + (W @ hues) / 100.0 * cal.hue_degrees_per_100) % 360.0
    if np.any(sats):
        s = np.clip(s * np.maximum(1.0 + (W @ sats) / 100.0 * cal.sat_gain, 0.0), 0.0, 1.0)
    if np.any(lums):
        gamma = 2.0 ** (-(W @ lums) / 100.0 * cal.lum_gamma_stops)
        l = np.clip(l, 0.0, 1.0) ** gamma
    return np.clip(cs.hsl_to_rgb(h, s, l), 0.0, 1.0)


def grade_masks(y: np.ndarray, blending: float, balance: float) -> np.ndarray:
    """Shadow / midtone / highlight weights from luma.  Sums to 1."""
    pivot = 0.5 + 0.25 * np.clip(balance, -100.0, 100.0) / 100.0
    width = 0.18 + 0.42 * np.clip(blending, 0.0, 100.0) / 100.0
    up = _smoothstep((y - (pivot - width)) / (2.0 * width))
    mid = np.exp(-(((y - pivot) / (width * 0.85)) ** 2))
    sh = (1.0 - up) * (1.0 - mid)
    hi = up * (1.0 - mid)
    return np.stack([sh, mid, hi], axis=-1)


def apply_grading(rgb: np.ndarray, m: PresetModel) -> np.ndarray:
    if not m.has_grading():
        return rgb
    cal = m.calibration
    out = rgb.astype(np.float64, copy=True)
    y = cs.luma(out)
    W = grade_masks(y, m.grade_blending, m.grade_balance)

    zones = [(m.grade["Shadow"], W[..., 0]),
             (m.grade["Midtone"], W[..., 1]),
             (m.grade["Highlight"], W[..., 2]),
             (m.grade["Global"], np.ones_like(y))]

    offset = np.zeros_like(out)
    gamma_exp = np.zeros_like(y)
    for z, w in zones:
        if abs(z.sat) > 1e-9:
            c = cs.hsl_to_rgb(np.array(z.hue), np.array(1.0), np.array(0.5))
            d = c - c.mean()
            offset += (w[..., None] * (z.sat / 100.0) * cal.grade_sat_gain) * d
        if abs(z.lum) > 1e-9:
            gamma_exp += w * (z.lum / 100.0) * cal.grade_lum_stops

    out = np.clip(out + offset, 0.0, 1.0)
    if np.any(gamma_exp):
        out = np.clip(out, 0.0, 1.0) ** (2.0 ** (-gamma_exp))[..., None]
    return np.clip(out, 0.0, 1.0)


def apply_basic_saturation(rgb: np.ndarray, m: PresetModel) -> np.ndarray:
    if abs(m.saturation) < 1e-9 and abs(m.vibrance) < 1e-9:
        return rgb
    h, s, l = cs.rgb_to_hsl(rgb)
    f = np.maximum(1.0 + m.saturation / 100.0, 0.0)
    s = s * f
    if abs(m.vibrance) > 1e-9:
        protect = (1.0 - np.clip(s, 0.0, 1.0)) ** m.calibration.vibrance_exponent
        s = s * np.maximum(1.0 + (m.vibrance / 100.0) * protect, 0.0)
    return np.clip(cs.hsl_to_rgb(h, np.clip(s, 0.0, 1.0), l), 0.0, 1.0)


def render(rgb: np.ndarray, m: PresetModel, upto: str = "all",
           in_working_space: bool = False) -> np.ndarray:
    """Apply the preset.  ``upto`` in {curves, hsl, grading, all}.

    The whole edit happens in ``m.working_space``; pass
    ``in_working_space=True`` when the input is already converted (the
    extractor works there to avoid converting on every iteration).
    """
    out = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    if not in_working_space:
        out = cs.to_working(out, m.working_space)

    out = apply_curves(out, m)
    if upto != "curves":
        out = apply_hsl(out, m)
        if upto != "hsl":
            out = apply_grading(out, m)
            if upto != "grading":
                out = apply_basic_saturation(out, m)
    return out if in_working_space else cs.from_working(out, m.working_space)


# --------------------------------------------------------------------------
# Inverses -- used to peel later stages off the target so each stage can be
# fitted in its own domain instead of through the others.
# --------------------------------------------------------------------------

def invert_hsl(rgb: np.ndarray, m: PresetModel, iters: int = 4) -> np.ndarray:
    cal = m.calibration
    hues = np.array([m.hsl[n].hue for n in BAND_NAMES])
    sats = np.array([m.hsl[n].sat for n in BAND_NAMES])
    lums = np.array([m.hsl[n].lum for n in BAND_NAMES])
    if not (np.any(hues) or np.any(sats) or np.any(lums)):
        return rgb

    h2, s2, l2 = cs.rgb_to_hsl(rgb)
    h = h2.copy()
    if np.any(hues):
        # band weights key off the *source* hue, so recover it by fixed point
        for _ in range(iters):
            W = band_weights(h, cal.band_falloff)
            h = (h2 - (W @ hues) / 100.0 * cal.hue_degrees_per_100) % 360.0
    W = band_weights(h, cal.band_falloff)

    s = s2
    if np.any(sats):
        f = np.maximum(1.0 + (W @ sats) / 100.0 * cal.sat_gain, 1e-3)
        s = np.clip(s2 / f, 0.0, 1.0)
    l = l2
    if np.any(lums):
        gamma = 2.0 ** (-(W @ lums) / 100.0 * cal.lum_gamma_stops)
        l = np.clip(l2, 0.0, 1.0) ** (1.0 / gamma)
    return np.clip(cs.hsl_to_rgb(h, s, l), 0.0, 1.0)


def invert_grading(rgb: np.ndarray, m: PresetModel, iters: int = 4) -> np.ndarray:
    if not m.has_grading():
        return rgb
    cal = m.calibration
    target = np.clip(rgb, 0.0, 1.0)
    src = target.copy()
    for _ in range(iters):
        y = cs.luma(src)
        W = grade_masks(y, m.grade_blending, m.grade_balance)
        zones = [(m.grade["Shadow"], W[..., 0]), (m.grade["Midtone"], W[..., 1]),
                 (m.grade["Highlight"], W[..., 2]), (m.grade["Global"], np.ones_like(y))]
        offset = np.zeros_like(target)
        gexp = np.zeros_like(y)
        for z, w in zones:
            if abs(z.sat) > 1e-9:
                c = cs.hsl_to_rgb(np.array(z.hue), np.array(1.0), np.array(0.5))
                offset += (w[..., None] * (z.sat / 100.0) * cal.grade_sat_gain) * (c - c.mean())
            if abs(z.lum) > 1e-9:
                gexp += w * (z.lum / 100.0) * cal.grade_lum_stops
        src = np.clip(target ** (2.0 ** gexp)[..., None] - offset, 0.0, 1.0)
    return src


def invert_basic_saturation(rgb: np.ndarray, m: PresetModel, iters: int = 6) -> np.ndarray:
    if abs(m.saturation) < 1e-9 and abs(m.vibrance) < 1e-9:
        return rgb
    h, s_out, l = cs.rgb_to_hsl(rgb)
    f = max(1.0 + m.saturation / 100.0, 1e-3)
    s = s_out / f
    if abs(m.vibrance) > 1e-9:
        for _ in range(iters):
            s_mid = np.clip(s * f, 0.0, 1.0)
            protect = (1.0 - s_mid) ** m.calibration.vibrance_exponent
            g = np.maximum(1.0 + (m.vibrance / 100.0) * protect, 1e-3)
            s = s_out / (f * g)
    return np.clip(cs.hsl_to_rgb(h, np.clip(s, 0.0, 1.0), l), 0.0, 1.0)


def invert_post(rgb: np.ndarray, m: PresetModel,
                in_working_space: bool = False) -> np.ndarray:
    """Undo everything after the tone curves, so curves can be re-fitted
    directly in their own domain."""
    out = rgb if in_working_space else cs.to_working(rgb, m.working_space)
    out = invert_basic_saturation(out, m)
    out = invert_grading(out, m)
    out = invert_hsl(out, m)
    return out if in_working_space else cs.from_working(out, m.working_space)
