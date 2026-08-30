"""Colour-space helpers.

Everything works on float arrays in [0, 1] with the last axis of size 3
holding sRGB-encoded (i.e. gamma-encoded, "display") R, G, B values --
the same domain Lightroom's tone curve and colour mixer operate in.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# sRGB transfer function
# --------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0.0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


# Rec.709 luma weights, applied to gamma-encoded values (what LR's tone
# curve / colour-grading masks effectively use).
LUMA_W = np.array([0.2126, 0.7152, 0.0722])


def luma(rgb: np.ndarray) -> np.ndarray:
    """Luma of gamma-encoded sRGB values."""
    return np.tensordot(rgb, LUMA_W, axes=([-1], [0]))


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """Photometric luminance (linear-light)."""
    return np.tensordot(srgb_to_linear(rgb), LUMA_W, axes=([-1], [0]))


# --------------------------------------------------------------------------
# HSL  (hue in degrees [0, 360), s and l in [0, 1])
# --------------------------------------------------------------------------

def rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    d = mx - mn
    l = 0.5 * (mx + mn)

    denom = 1.0 - np.abs(2.0 * l - 1.0)
    s = np.where(d > 1e-12, d / np.maximum(denom, 1e-12), 0.0)
    s = np.clip(s, 0.0, 1.0)

    h = np.zeros_like(mx)
    safe_d = np.where(d > 1e-12, d, 1.0)
    hr = ((g - b) / safe_d) % 6.0
    hg = (b - r) / safe_d + 2.0
    hb = (r - g) / safe_d + 4.0
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb))
    h = np.where(d > 1e-12, h * 60.0, 0.0) % 360.0
    return h, s, l


def hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64) % 360.0
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    l = np.clip(np.asarray(l, dtype=np.float64), 0.0, 1.0)

    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = h / 60.0
    x = c * (1.0 - np.abs(hp % 2.0 - 1.0))
    z = np.zeros_like(c)

    seg = np.floor(hp).astype(int) % 6
    r = np.select([seg == 0, seg == 1, seg == 2, seg == 3, seg == 4, seg == 5],
                  [c, x, z, z, x, c])
    g = np.select([seg == 0, seg == 1, seg == 2, seg == 3, seg == 4, seg == 5],
                  [x, c, c, x, z, z])
    b = np.select([seg == 0, seg == 1, seg == 2, seg == 3, seg == 4, seg == 5],
                  [z, z, x, c, c, x])
    m = l - 0.5 * c
    return np.stack([r + m, g + m, b + m], axis=-1)


def wrap_deg(d: np.ndarray) -> np.ndarray:
    """Wrap an angle difference into (-180, 180]."""
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# CIE Lab + deltaE 2000
# --------------------------------------------------------------------------

_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_D65 = np.array([0.95047, 1.00000, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(rgb)
    xyz = lin @ _M_RGB2XYZ.T
    t = xyz / _D65
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(t > eps, np.cbrt(np.maximum(t, 1e-12)), (kappa * t + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (kL = kC = kH = 1)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7 + 1e-30)))

    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = np.where(C1p * C2p == 0, 0.0, wrap_deg(h2p - h1p))
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hbp = np.where(C1p * C2p == 0, hsum,
                   np.where(hdiff <= 180.0, 0.5 * hsum,
                            np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0))))

    T = (1.0
         - 0.17 * np.cos(np.radians(hbp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbp))
         + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0)))
    dtheta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Rc = 2.0 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7 + 1e-30))
    Sl = 1.0 + (0.015 * (Lbp - 50.0) ** 2) / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbp
    Sh = 1.0 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc

    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))
