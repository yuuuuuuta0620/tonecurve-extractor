"""Data-driven 3D LUT extraction and .cube export.

The parametric preset is what Lightroom can import, but a 3D LUT is the
highest-fidelity representation of "whatever happened between these two
images".  We solve a regularised least-squares problem on a lattice:

    minimise  || S z - y ||_W^2  +  lam * ||L z||^2  +  mu * ||z - prior||^2

where S is trilinear interpolation from the lattice to the sample points,
L a 3-D Laplacian (smoothness) and ``prior`` the parametric model's own
prediction, which fills lattice cells the photograph never visits.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import colorspace as cs
from .render import render


def _trilinear_matrix(x: np.ndarray, n: int) -> sp.csr_matrix:
    """Sparse (samples x n^3) trilinear interpolation matrix."""
    p = np.clip(x, 0.0, 1.0) * (n - 1)
    i0 = np.floor(p).astype(np.int64)
    i0 = np.minimum(i0, n - 2)
    f = p - i0
    m = x.shape[0]

    rows = np.repeat(np.arange(m), 8)
    cols = np.empty(m * 8, dtype=np.int64)
    vals = np.empty(m * 8, dtype=np.float64)
    k = 0
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                wr = f[:, 0] if dr else 1 - f[:, 0]
                wg = f[:, 1] if dg else 1 - f[:, 1]
                wb = f[:, 2] if db else 1 - f[:, 2]
                idx = ((i0[:, 0] + dr) * n + (i0[:, 1] + dg)) * n + (i0[:, 2] + db)
                cols[k::8] = idx
                vals[k::8] = wr * wg * wb
                k += 1
    return sp.csr_matrix((vals, (rows, cols)), shape=(m, n ** 3))


def _laplacian(n: int) -> sp.csr_matrix:
    e = np.ones(n)
    D = sp.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n))
    I = sp.identity(n, format="csr")
    return sp.vstack([sp.kron(sp.kron(D, I), I),
                      sp.kron(sp.kron(I, D), I),
                      sp.kron(sp.kron(I, I), D)]).tocsr()


def fit_lut3d(before: np.ndarray,
              after: np.ndarray,
              weight: np.ndarray,
              model=None,
              size: int = 33,
              lam: float = 4.0,
              mu: float = 0.02,
              max_samples: int = 400_000,
              seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(weight > 1e-3)
    if idx.size > max_samples:
        idx = rng.choice(idx, max_samples, replace=False)
    x, y, w = before[idx], after[idx], weight[idx]

    n = size
    g = np.linspace(0.0, 1.0, n)
    R, G, Bc = np.meshgrid(g, g, g, indexing="ij")
    lattice = np.stack([R.ravel(), G.ravel(), Bc.ravel()], axis=1)
    prior = render(lattice, model) if model is not None else lattice.copy()

    S = _trilinear_matrix(x, n)
    Wd = sp.diags(w)
    L = _laplacian(n)
    A = (S.T @ Wd @ S + lam * (L.T @ L) + mu * sp.identity(n ** 3)).tocsc()
    solve = spla.factorized(A)

    out = np.empty((n ** 3, 3))
    for c in range(3):
        rhs = S.T @ (w * y[:, c]) + mu * prior[:, c]
        out[:, c] = solve(rhs)
    return np.clip(out.reshape(n, n, n, 3), 0.0, 1.0)


def bake_lut3d(model, size: int = 33) -> np.ndarray:
    """Evaluate the fitted preset on the lattice, with no reference to the data.

    Use this when the preset itself is the thing you want in the LUT -- a
    tone-only extraction, say, where a data-driven fit would quietly put the
    colour work back in.  Note that a 3D LUT samples the curve at ``size``
    points and interpolates linearly between them, while Lightroom rebuilds it
    as a monotone cubic through its control points, so the two are not
    identical on smooth gradients.
    """
    g = np.linspace(0.0, 1.0, size)
    R, G, Bc = np.meshgrid(g, g, g, indexing="ij")
    lattice = np.stack([R.ravel(), G.ravel(), Bc.ravel()], axis=1)
    return np.clip(render(lattice, model).reshape(size, size, size, 3), 0.0, 1.0)


def apply_lut3d(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    n = lut.shape[0]
    shape = rgb.shape
    x = np.clip(rgb.reshape(-1, 3), 0, 1)
    S = _trilinear_matrix(x, n)
    flat = lut.reshape(n ** 3, 3)
    return np.clip((S @ flat).reshape(shape), 0, 1)


def write_cube(path: str, lut: np.ndarray, title: str = "tcx") -> None:
    """Write an Adobe/IRIDAS .cube file (blue varies fastest)."""
    n = lut.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {n}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n\n")
        # .cube order: red index varies fastest
        for b in range(n):
            for gi in range(n):
                for r in range(n):
                    v = lut[r, gi, b]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def separability_error(model, n: int = 96) -> float:
    """How far the preset is from being three independent 1-D curves.

    A master-only tone curve *is* separable in sRGB, but not once it is applied
    in ProPhoto primaries: the space conversion mixes the channels, so
    out = M⁻¹ f(M x) depends on all three inputs.  Measured rather than
    assumed, because it decides whether a 1-D LUT can be used at all.
    """
    g = np.linspace(0.0, 1.0, n)
    # the per-channel curves implied by sweeping one channel with the others neutral
    axes = []
    for c in range(3):
        probe = np.tile(g[:, None], (1, 3))
        axes.append(render(probe, model)[:, c])

    rng = np.random.default_rng(0)
    x = rng.random((4000, 3))
    exact = render(x, model)
    sep = np.stack([np.interp(x[:, c], g, axes[c]) for c in range(3)], axis=1)
    return float(cs.delta_e2000(cs.rgb_to_lab(sep), cs.rgb_to_lab(exact)).mean())


def write_cube_1d(path: str, model, size: int = 1024, title: str = "tcx") -> None:
    """Write a 1-D .cube.  Only valid for a separable preset -- check with
    ``separability_error`` first."""
    g = np.linspace(0.0, 1.0, size)
    probe = np.tile(g[:, None], (1, 3))
    out = np.clip(render(probe, model), 0.0, 1.0)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_1D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n\n")
        for v in out:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
