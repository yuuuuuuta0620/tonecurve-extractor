"""Fit Lightroom's Colour Grading wheels (shadows / midtones / highlights)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import colorspace as cs
from .model import GradeZone, Calibration
from .render import grade_masks

ZONES = ["Shadow", "Midtone", "Highlight"]


def _tint_dir(hue_deg: float) -> np.ndarray:
    c = cs.hsl_to_rgb(np.array(hue_deg), np.array(1.0), np.array(0.5))
    return c - c.mean()


def _apply(rgb: np.ndarray, p: np.ndarray, cal: Calibration, use_global: bool) -> np.ndarray:
    n_zone = 4 if use_global else 3
    uv = p[:n_zone * 3].reshape(n_zone, 3)     # u, v, lum
    blending, balance = p[n_zone * 3], p[n_zone * 3 + 1]

    y = cs.luma(rgb)
    W = grade_masks(y, blending, balance)
    weights = [W[..., 0], W[..., 1], W[..., 2]]
    if use_global:
        weights.append(np.ones_like(y))

    offset = np.zeros_like(rgb)
    gexp = np.zeros_like(y)
    for i, w in enumerate(weights):
        u, v, lum = uv[i]
        sat = min(float(np.hypot(u, v)), 1.0)
        if sat > 1e-6:
            hue = float(np.degrees(np.arctan2(v, u)) % 360.0)
            offset += (w[..., None] * sat * cal.grade_sat_gain) * _tint_dir(hue)
        if abs(lum) > 1e-9:
            gexp += w * lum * cal.grade_lum_stops
    out = np.clip(rgb + offset, 0.0, 1.0)
    if np.any(gexp):
        out = out ** (2.0 ** (-gexp))[..., None]
    return np.clip(out, 0.0, 1.0)


def fit_grading(mid: np.ndarray,
                target: np.ndarray,
                weight: np.ndarray,
                cal: Calibration,
                use_global: bool = False,
                max_samples: int = 20000,
                seed: int = 0) -> tuple[dict[str, GradeZone], float, float, dict]:
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(weight > 1e-3)
    if idx.size == 0:
        return {z: GradeZone() for z in ZONES + ["Global"]}, 50.0, 0.0, {"status": "no data"}
    if idx.size > max_samples:
        idx = rng.choice(idx, max_samples, replace=False)
    m, t, w = mid[idx], target[idx], np.sqrt(weight[idx])[:, None]
    lab_t = cs.rgb_to_lab(t)

    n_zone = 4 if use_global else 3
    p0 = np.zeros(n_zone * 3 + 2)
    p0[-2] = 50.0    # blending
    p0[-1] = 0.0     # balance
    lo = np.concatenate([np.tile([-1.0, -1.0, -1.0], n_zone), [0.0, -100.0]])
    hi = np.concatenate([np.tile([1.0, 1.0, 1.0], n_zone), [100.0, 100.0]])

    scale = np.array([1.0, 0.5, 0.5])   # weight L less than chroma

    def resid(p):
        out = _apply(m, p, cal, use_global)
        return ((cs.rgb_to_lab(out) - lab_t) * scale * w).ravel()

    base = float(np.sqrt((resid(p0) ** 2).mean()))
    r = least_squares(resid, p0, bounds=(lo, hi), xtol=1e-8, ftol=1e-8,
                      x_scale="jac", max_nfev=300)
    final = float(np.sqrt((r.fun ** 2).mean()))

    uv = r.x[:n_zone * 3].reshape(n_zone, 3)
    zones: dict[str, GradeZone] = {}
    names = ZONES + (["Global"] if use_global else [])
    for i, name in enumerate(names):
        u, v, lum = uv[i]
        sat = float(min(np.hypot(u, v), 1.0)) * 100.0
        hue = float(np.degrees(np.arctan2(v, u)) % 360.0)
        zones[name] = GradeZone(hue=round(hue, 1) if sat > 0.5 else 0.0,
                                sat=round(sat, 1),
                                lum=round(float(lum) * 100.0, 1))
    for name in ZONES + ["Global"]:
        zones.setdefault(name, GradeZone())

    diag = {"resid_before": round(base, 4), "resid_after": round(final, 4),
            "improved": round(base - final, 4), "nfev": int(r.nfev)}
    return zones, float(r.x[-2]), float(r.x[-1]), diag
