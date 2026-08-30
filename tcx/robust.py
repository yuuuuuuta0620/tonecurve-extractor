"""Small weighted robust-statistics helpers."""
from __future__ import annotations

import numpy as np


def weighted_median(x: np.ndarray, w: np.ndarray, q: float = 0.5) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    if x.size == 0 or w.sum() <= 0:
        return float("nan")
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, q * c[-1])])


def weighted_mad(x: np.ndarray, w: np.ndarray) -> float:
    m = weighted_median(x, w)
    if not np.isfinite(m):
        return float("nan")
    return 1.4826 * weighted_median(np.abs(x - m), w)


def circular_weighted_median(deg: np.ndarray, w: np.ndarray) -> float:
    """Median of angle differences already wrapped into (-180, 180]."""
    return weighted_median(deg, w)
