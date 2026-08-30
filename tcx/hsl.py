"""Fit the eight-band Colour Mixer (HSL) panel.

Each pixel is influenced by *two* adjacent hue bands, so estimating a band
from a median of the pixels nearest its centre systematically under-shoots.
All three sliders are linear in the band weights, so we solve the whole
8-band system at once by robust (Huber IRLS) weighted least squares.

    hue     :  Δhue      = W · (hue/100 · hue_degrees_per_100)
    sat     :  s'/s − 1  = W · (sat/100 · sat_gain)
    lum     :  −log2 γ   = W · (lum/100 · lum_gamma_stops)
"""
from __future__ import annotations

import numpy as np

from . import colorspace as cs
from .model import BAND_NAMES, HSLBand, Calibration
from .render import band_weights

#: a hue band needs this share of the total sample weight before its sliders
#: are trusted at all.  An absolute floor is useless: with 1.5 M samples a
#: band holding 0.01 % of the data would still qualify, and the fit for it is
#: then pure extrapolation -- which is exactly how a night scene ends up with
#: "Purple luminance -100".
MIN_BAND_FRACTION = 0.004
MIN_BAND_WEIGHT = 150.0

#: Mild Tikhonov shrinkage toward "no adjustment", scaled to the typical
#: band's data.  Kept gentle on purpose: measured against known ground truth,
#: heavier shrinkage costs real accuracy on well-sampled bands (slider error
#: 1.44 -> 2.30 at 0.06).  Thin bands are handled by reporting their data
#: share and warning, not by silently flattening them.
BAND_SHRINK = 0.02


def _irls(W: np.ndarray, y: np.ndarray, w: np.ndarray,
          ridge: np.ndarray | float = 1e-3, rounds: int = 4, huber: float = 1.5):
    """Robust weighted least squares for  y ≈ W · beta, with per-band ridge."""
    n_out = W.shape[1]
    rw = w.copy()
    beta = np.zeros(n_out)
    reg = np.broadcast_to(np.asarray(ridge, dtype=np.float64), (n_out,))
    for r in range(rounds):
        sw = np.sqrt(rw)
        A = W * sw[:, None]
        b = y * sw
        A = np.vstack([A, np.diag(reg)])
        b = np.concatenate([b, np.zeros(n_out)])
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        if r == rounds - 1:
            break
        res = y - W @ beta
        scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        u = np.abs(res) / (huber * scale)
        rw = w * np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
    return beta


def fit_hsl(mid: np.ndarray,
            target: np.ndarray,
            weight: np.ndarray,
            cal: Calibration,
            fit_hue: bool = True,
            fit_sat: bool = True,
            fit_lum: bool = True) -> tuple[dict[str, HSLBand], dict]:
    h1, s1, l1 = cs.rgb_to_hsl(mid)
    h2, s2, l2 = cs.rgb_to_hsl(target)
    W = band_weights(h1, cal.band_falloff)

    unclipped = (l1 > 0.02) & (l1 < 0.98) & (l2 > 0.02) & (l2 < 0.98)
    hue_ok = unclipped & (s1 > 0.10) & (s2 > 0.05) & (s1 < 0.999) & (s2 < 0.999)
    sat_ok = unclipped & (s1 > 0.05) & (s1 < 0.999) & (s2 < 0.999)
    lum_ok = (l1 > 0.02) & (l1 < 0.98) & (l2 > 0.02) & (l2 < 0.98)

    dh = cs.wrap_deg(h2 - h1)
    ratio = np.divide(s2, s1, out=np.ones_like(s1), where=s1 > 1e-6) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.log(np.clip(l2, 1e-6, 1)) / np.log(np.clip(l1, 1e-6, 1))
    k = -np.log2(np.clip(gamma, 1e-3, 1e3))

    band_mass = (W * weight[:, None]).sum(axis=0)
    total = max(band_mass.sum(), 1e-12)
    band_frac = band_mass / total
    have = (band_mass > MIN_BAND_WEIGHT) & (band_frac > MIN_BAND_FRACTION)

    def _ridge(mask, sample_w):
        """Shrinkage strength for this sub-problem.

        Each band's own diagonal in the normal equations is roughly its share
        of ``sample_w``, so a single ridge value scaled to the *typical* band
        leaves well-populated bands alone while pulling thin ones toward zero:
        the shrink factor is m_i / (m_i + r^2).
        """
        m = (W[mask] ** 2 * sample_w[:, None]).sum(axis=0)
        ref = float(np.median(m[m > 0])) if np.any(m > 0) else 1.0
        return np.full(8, np.sqrt(BAND_SHRINK * ref))

    hue_b = np.zeros(8); sat_b = np.zeros(8); lum_b = np.zeros(8)
    if fit_hue:
        m = hue_ok
        sw = weight[m] * s1[m]
        hue_b = _irls(W[m], dh[m], sw, ridge=_ridge(m, sw)) \
            / cal.hue_degrees_per_100 * 100.0
    if fit_sat:
        m = sat_ok
        sw = weight[m] * s1[m]
        sat_b = _irls(W[m], ratio[m], sw, ridge=_ridge(m, sw)) / cal.sat_gain * 100.0
    if fit_lum:
        m = lum_ok
        sw = weight[m] * np.clip(s1[m], 0.05, 1.0)
        lum_b = _irls(W[m], k[m], sw, ridge=_ridge(m, sw)) \
            / cal.lum_gamma_stops * 100.0

    bands, diag = {}, {}
    for i, name in enumerate(BAND_NAMES):
        clipped = [n for n, v in (("hue", hue_b[i]), ("sat", sat_b[i]), ("lum", lum_b[i]))
                   if abs(v) > 100]
        if have[i]:
            bands[name] = HSLBand(hue=float(np.clip(hue_b[i], -100, 100)),
                                  sat=float(np.clip(sat_b[i], -100, 100)),
                                  lum=float(np.clip(lum_b[i], -100, 100)))
        else:
            bands[name] = HSLBand()
        diag[name] = {"data_share": round(float(band_frac[i]), 4),
                      "used": bool(have[i]),
                      "clipped": clipped}
    return bands, diag


def fit_basic_saturation(mid: np.ndarray,
                         target: np.ndarray,
                         weight: np.ndarray,
                         cal: Calibration) -> tuple[float, float]:
    """Least-squares fit of the Basic panel's Saturation + Vibrance."""
    from scipy.optimize import least_squares

    _, s1, _ = cs.rgb_to_hsl(mid)
    _, s2, _ = cs.rgb_to_hsl(target)
    m = (s1 > 0.02) & (s1 < 0.999) & (s2 < 0.999) & (weight > 0)
    if m.sum() < 500:
        return 0.0, 0.0
    x, y, w = s1[m], s2[m], np.sqrt(weight[m])

    def resid(p):
        sat, vib = p
        s = x * max(1.0 + sat / 100.0, 0.0)
        protect = (1.0 - np.clip(s, 0, 1)) ** cal.vibrance_exponent
        s = s * np.maximum(1.0 + (vib / 100.0) * protect, 0.0)
        return (np.clip(s, 0, 1) - y) * w

    r = least_squares(resid, [0.0, 0.0], bounds=([-100, -100], [100, 100]),
                      xtol=1e-8, ftol=1e-8, max_nfev=200)
    return float(r.x[0]), float(r.x[1])
