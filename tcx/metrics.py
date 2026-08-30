"""Quality metrics for verifying an extracted preset."""
from __future__ import annotations

import numpy as np

from . import colorspace as cs


def compare(pred: np.ndarray, ref: np.ndarray, weight: np.ndarray | None = None) -> dict:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0, 1)
    ref = np.clip(np.asarray(ref, dtype=np.float64), 0, 1)
    diff = pred - ref
    if weight is None:
        w = np.ones(ref.shape[:-1])
    else:
        w = np.asarray(weight, dtype=np.float64)
    wsum = max(w.sum(), 1e-12)

    mse = float((w[..., None] * diff ** 2).sum() / (wsum * 3))
    rmse255 = float(np.sqrt(mse) * 255.0)
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

    de = cs.delta_e2000(cs.rgb_to_lab(pred), cs.rgb_to_lab(ref))
    flat_de = de.ravel()
    flat_w = w.ravel()
    order = np.argsort(flat_de)
    cw = np.cumsum(flat_w[order])
    p95 = float(flat_de[order][np.searchsorted(cw, 0.95 * cw[-1])])
    med = float(flat_de[order][np.searchsorted(cw, 0.50 * cw[-1])])
    mean_de = float((flat_de * flat_w).sum() / wsum)

    return {"rmse255": round(rmse255, 3),
            "psnr_db": round(psnr, 2),
            "dE_mean": round(mean_de, 3),
            "dE_median": round(med, 3),
            "dE_p95": round(p95, 3),
            "dE_max": round(float(de.max()), 3)}


def improvement(baseline: dict, result: dict) -> dict:
    return {"psnr_gain_db": round(result["psnr_db"] - baseline["psnr_db"], 2),
            "dE_mean_reduction_pct": round(
                100.0 * (1.0 - result["dE_mean"] / max(baseline["dE_mean"], 1e-9)), 1)}
