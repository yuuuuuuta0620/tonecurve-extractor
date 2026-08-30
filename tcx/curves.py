"""Robust estimation of a 1-D transfer function from scattered (src, dst) pairs.

The core assumption is that a global Lightroom edit maps an input tonal
value to an output value.  Local adjustments (masks, brushes), noise and
residual misalignment violate that, so we use a *conditional weighted
median* per input bin -- which tolerates up to 50 % contaminated samples --
followed by weighted isotonic regression (monotonicity) and a Whittaker
penalised-difference smoother.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import PchipInterpolator

#: number of nodes in the stored dense LUTs
LUT_N = 1024


# --------------------------------------------------------------------------
# LUT helpers
# --------------------------------------------------------------------------

def identity_lut(n: int = LUT_N) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def apply_lut(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Evaluate a LUT (uniform nodes on [0,1]) at arbitrary x with linear interp."""
    nodes = np.linspace(0.0, 1.0, len(lut))
    return np.interp(np.clip(x, 0.0, 1.0), nodes, lut)


def invert_lut(lut: np.ndarray, n: int = LUT_N) -> np.ndarray:
    """Invert a monotonically non-decreasing LUT."""
    nodes = np.linspace(0.0, 1.0, len(lut))
    # make strictly increasing so np.interp is well defined
    y = np.maximum.accumulate(lut)
    y = y + np.linspace(0.0, 1e-9, len(y))
    grid = np.linspace(0.0, 1.0, n)
    return np.clip(np.interp(grid, y, nodes), 0.0, 1.0)


def compose_lut(outer: np.ndarray, inner: np.ndarray, n: int = LUT_N) -> np.ndarray:
    """Return LUT for x -> outer(inner(x))."""
    grid = np.linspace(0.0, 1.0, n)
    return apply_lut(apply_lut(grid, inner), outer)


def resample_lut(lut: np.ndarray, n: int = LUT_N) -> np.ndarray:
    if len(lut) == n:
        return lut.astype(np.float64)
    src = np.linspace(0.0, 1.0, len(lut))
    interp = PchipInterpolator(src, np.asarray(lut, dtype=np.float64))
    return np.clip(interp(np.linspace(0.0, 1.0, n)), 0.0, 1.0)


# --------------------------------------------------------------------------
# Weighted isotonic regression (pool-adjacent-violators)
# --------------------------------------------------------------------------

def isotonic(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    w = np.ones(n) if w is None else np.maximum(np.asarray(w, dtype=np.float64), 1e-12)

    vals = np.empty(n)
    wts = np.empty(n)
    lens = np.empty(n, dtype=int)
    k = 0
    for i in range(n):
        vals[k], wts[k], lens[k] = y[i], w[i], 1
        k += 1
        while k > 1 and vals[k - 2] > vals[k - 1]:
            tw = wts[k - 2] + wts[k - 1]
            vals[k - 2] = (vals[k - 2] * wts[k - 2] + vals[k - 1] * wts[k - 1]) / tw
            wts[k - 2] = tw
            lens[k - 2] += lens[k - 1]
            k -= 1
    out = np.empty(n)
    pos = 0
    for j in range(k):
        out[pos:pos + lens[j]] = vals[j]
        pos += lens[j]
    return out


# --------------------------------------------------------------------------
# Whittaker smoother:  min  sum w_i (z_i - y_i)^2 + lam * ||D2 z||^2
# --------------------------------------------------------------------------

def whittaker(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    n = len(y)
    w = np.maximum(np.asarray(w, dtype=np.float64), 0.0)
    if w.max() <= 0:
        return np.asarray(y, dtype=np.float64)
    w = w / w.max()
    W = sp.diags(w)
    e = np.ones(n)
    D2 = sp.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n))
    A = (W + lam * (D2.T @ D2)).tocsc()
    return spla.spsolve(A, w * np.asarray(y, dtype=np.float64))


# --------------------------------------------------------------------------
# Conditional-median transfer estimation
# --------------------------------------------------------------------------

class TransferFit:
    """Result of estimating one channel's transfer function."""

    def __init__(self, lut, nodes, raw, counts, spread, coverage, support=(0.0, 1.0)):
        self.lut = lut            # dense LUT, LUT_N nodes on [0,1]
        self.nodes = nodes        # estimation grid (n_bins,)
        self.raw = raw            # raw conditional medians on that grid (NaN where empty)
        self.counts = counts      # sample weight per bin
        self.spread = spread      # robust IQR of dst per bin (NaN where empty)
        self.coverage = coverage  # fraction of bins with usable data
        self.support = support    # input range the images actually exercise


def _extrapolate(nodes, y, good, span=16):
    """Linearly extend the curve past the range the photograph covers.

    Outside the observed input range the mapping is genuinely unknown; a
    flat extension (what plain interpolation gives) is a much worse prior
    for a tone curve than continuing the local slope.
    """
    idx = np.flatnonzero(good)
    i0, i1 = idx[0], idx[-1]
    y = y.copy()

    def slope(a, b):
        xs, ys = nodes[a:b + 1], y[a:b + 1]
        if len(xs) < 2:
            return 1.0
        s = np.polyfit(xs, ys, 1)[0]
        return float(np.clip(s, 0.15, 4.0))

    if i0 > 0:
        s = slope(i0, min(i1, i0 + span))
        y[:i0] = y[i0] + s * (nodes[:i0] - nodes[i0])
    if i1 < len(nodes) - 1:
        s = slope(max(i0, i1 - span), i1)
        y[i1 + 1:] = y[i1] + s * (nodes[i1 + 1:] - nodes[i1])
    return np.clip(y, 0.0, 1.0), (float(nodes[i0]), float(nodes[i1]))


def estimate_transfer(src: np.ndarray,
                      dst: np.ndarray,
                      weight: np.ndarray | None = None,
                      n_bins: int = 256,
                      n_out: int = 1024,
                      min_weight: float | None = None,
                      lam: float = 2.0,
                      quantile: float = 0.5) -> TransferFit:
    """Estimate y = f(x) from scattered pairs.

    Parameters
    ----------
    src, dst : 1-D float arrays in [0, 1]
    weight   : per-sample weight
    n_bins   : input bins (256 matches 8-bit source material)
    n_out    : output-value histogram resolution
    min_weight : bins with less total weight than this are treated as missing
    lam      : Whittaker smoothing strength
    """
    src = np.clip(np.asarray(src, dtype=np.float64).ravel(), 0.0, 1.0)
    dst = np.clip(np.asarray(dst, dtype=np.float64).ravel(), 0.0, 1.0)
    w = np.ones_like(src) if weight is None else np.asarray(weight, dtype=np.float64).ravel()

    si = np.rint(src * (n_bins - 1)).astype(np.int32)
    di = np.rint(dst * (n_out - 1)).astype(np.int32)

    H = np.zeros((n_bins, n_out), dtype=np.float64)
    np.add.at(H, (si, di), w)

    counts = H.sum(axis=1)
    cum = np.cumsum(H, axis=1)

    def _quantile(q):
        target = q * counts
        idx = (cum < target[:, None]).sum(axis=1)
        idx = np.clip(idx, 0, n_out - 1)
        rows = np.arange(n_bins)
        lo = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
        binw = H[rows, idx]
        frac = np.where(binw > 0, (target - lo) / np.maximum(binw, 1e-12), 0.5)
        frac = np.clip(frac, 0.0, 1.0)
        return (idx + frac - 0.5) / (n_out - 1)

    med = _quantile(quantile)
    q25, q75 = _quantile(0.25), _quantile(0.75)
    spread = np.where(counts > 0, q75 - q25, np.nan)

    total = counts.sum()
    if min_weight is None:
        # a bin needs enough samples to make a median meaningful, scaled to
        # the amount of data we actually have
        min_weight = max(8.0, 2e-5 * total)
    good = counts >= min_weight
    coverage = float(good.mean())
    if good.sum() < 4:
        raise ValueError("not enough data to estimate a transfer function")

    nodes = np.linspace(0.0, 1.0, n_bins)
    raw = np.where(counts > 0, med, np.nan)

    # interpolate interior gaps, then extend the ends along the local slope
    y = np.interp(nodes, nodes[good], med[good])
    y, support = _extrapolate(nodes, y, good)

    # weight by sample count but damp the dynamic range so a huge midtone
    # bin does not completely dominate the smoother
    wt = np.where(good, np.sqrt(counts), 0.0)
    if wt.max() > 0:
        wt = wt / wt.max()
    wt = np.maximum(wt, 1e-3)          # keeps empty bins tied to the interpolation

    y = isotonic(y, wt)
    y = whittaker(y, wt, lam)
    y = isotonic(y, wt)
    y = np.clip(y, 0.0, 1.0)

    lut = resample_lut(y, LUT_N)
    return TransferFit(lut, nodes, raw, counts, spread, coverage, support)


# --------------------------------------------------------------------------
# Fitting a small set of Lightroom curve control points to a dense LUT
# --------------------------------------------------------------------------

def fit_control_points(lut: np.ndarray,
                       max_points: int = 16,
                       tol: float = 0.5 / 255.0) -> list[tuple[int, int]]:
    """Greedily place control points so a monotone cubic through them
    reproduces ``lut`` within ``tol``.

    Returns integer (x, y) pairs on Lightroom's 0-255 grid.
    """
    n = len(lut)
    x = np.linspace(0.0, 1.0, n)
    y = np.clip(np.asarray(lut, dtype=np.float64), 0.0, 1.0)

    idx = [0, n - 1]
    for _ in range(max_points - 2):
        xs = x[idx]
        ys = y[idx]
        if len(idx) == 2:
            approx = np.interp(x, xs, ys)
        else:
            approx = PchipInterpolator(xs, ys)(x)
        err = np.abs(approx - y)
        if err.max() <= tol:
            break
        k = int(np.argmax(err))
        if k in idx:
            break
        idx = sorted(idx + [k])

    pts = []
    for i in idx:
        px = int(round(x[i] * 255.0))
        py = int(round(y[i] * 255.0))
        if pts and px <= pts[-1][0]:
            px = pts[-1][0] + 1
            if px > 255:
                continue
        pts.append((px, min(255, max(0, py))))
    # Lightroom needs monotone, non-decreasing y for a sane curve
    for i in range(1, len(pts)):
        if pts[i][1] < pts[i - 1][1]:
            pts[i] = (pts[i][0], pts[i - 1][1])
    return pts


def control_points_to_lut(points: list[tuple[int, int]], n: int = LUT_N) -> np.ndarray:
    """Render Lightroom-style control points back into a dense LUT."""
    if len(points) < 2:
        return identity_lut(n)
    xs = np.array([p[0] for p in points], dtype=np.float64) / 255.0
    ys = np.array([p[1] for p in points], dtype=np.float64) / 255.0
    grid = np.linspace(0.0, 1.0, n)
    interp = PchipInterpolator(xs, ys, extrapolate=True)
    return np.clip(interp(grid), 0.0, 1.0)
