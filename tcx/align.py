"""Geometric alignment and per-pixel sample weighting for a before/after pair.

Preset extraction assumes pixel i in "before" corresponds to pixel i in
"after".  Sample images from the web are often resized, re-cropped or
slightly shifted, so we align first and then down-weight the pixels whose
value we cannot trust (edges, where sub-pixel error is amplified).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PairData:
    before: np.ndarray            # HxWx3 float, aligned
    after: np.ndarray             # HxWx3 float, aligned
    weight: np.ndarray            # HxW float in [0,1]
    before_s: np.ndarray          # slightly blurred copies used for sampling
    after_s: np.ndarray
    info: dict = field(default_factory=dict)
    outlier_mask: np.ndarray | None = None   # filled in by extract(), for reporting


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    interp = cv2.INTER_AREA if (img.shape[1] > w or img.shape[0] > h) else cv2.INTER_LANCZOS4
    return cv2.resize(img, (w, h), interpolation=interp).astype(np.float64)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def align_pair(before: np.ndarray,
               after: np.ndarray,
               max_dim: int = 1600,
               motion: str = "translation",
               blur_sigma: float = 0.8,
               edge_percentile: float = 60.0,
               do_align: bool = True,
               mask: np.ndarray | None = None,
               exclude: list[tuple[float, float, float, float]] | None = None) -> PairData:
    info: dict = {}

    # 1. common geometry -- never upsample, so we never invent detail
    hb, wb = before.shape[:2]
    ha, wa = after.shape[:2]
    info["input_size_before"] = (wb, hb)
    info["input_size_after"] = (wa, ha)
    if (hb, wb) != (ha, wa):
        w = min(wb, wa)
        h = min(hb, ha)
        # preserve aspect of the smaller one
        before = _resize(before, (w, h))
        after = _resize(after, (w, h))
        info["resized"] = True

    h, w = before.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        before = _resize(before, (nw, nh))
        after = _resize(after, (nw, nh))
        info["working_size"] = (nw, nh)
    else:
        info["working_size"] = (w, h)

    gb, ga = _gray(before), _gray(after)
    info["ncc_before_align"] = round(_ncc(gb, ga), 5)

    # 2. ECC alignment (intensity-change invariant, so an edit does not fool it)
    warp = np.eye(2, 3, dtype=np.float32)
    if do_align:
        mode = {"translation": cv2.MOTION_TRANSLATION,
                "euclidean": cv2.MOTION_EUCLIDEAN,
                "affine": cv2.MOTION_AFFINE}.get(motion, cv2.MOTION_TRANSLATION)
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            sm_b = cv2.GaussianBlur(gb, (0, 0), 1.5)
            sm_a = cv2.GaussianBlur(ga, (0, 0), 1.5)
            _, warp = cv2.findTransformECC(sm_b, sm_a, warp, mode, crit, None, 5)
            after = cv2.warpAffine(after, warp, (after.shape[1], after.shape[0]),
                                   flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
            ga = _gray(after)
            info["align_warp"] = np.round(warp, 4).tolist()
            info["align_shift_px"] = [round(float(warp[0, 2]), 3), round(float(warp[1, 2]), 3)]
        except cv2.error as e:
            info["align_error"] = str(e).strip().splitlines()[-1][:160]
    info["ncc_after_align"] = round(_ncc(gb, ga), 5)

    # 3. sample weights: trust flat areas, distrust edges
    gx = cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.0)
    g0 = float(np.percentile(grad, edge_percentile)) + 1e-6
    weight = (1.0 / (1.0 + (grad / g0) ** 2)).astype(np.float64)

    # drop a border where warping replicated pixels
    bw = max(2, int(0.01 * max(before.shape[:2])))
    weight[:bw] = weight[-bw:] = 0.0
    weight[:, :bw] = weight[:, -bw:] = 0.0

    # user-supplied exclusions: a mask image (white = use) and/or fractional
    # rectangles, for watermarks and logos you would rather not have measured
    if mask is not None:
        m = mask if mask.ndim == 2 else mask.mean(axis=2)
        m = cv2.resize(m.astype(np.float32), (weight.shape[1], weight.shape[0]),
                       interpolation=cv2.INTER_AREA)
        weight = weight * np.clip(m, 0.0, 1.0)
        info["user_mask_keeps"] = round(float(np.mean(m > 0.5)), 4)
    if exclude:
        h2, w2 = weight.shape
        for (l, t, ww, hh) in exclude:
            x0, y0 = int(round(l * w2)), int(round(t * h2))
            x1, y1 = int(round((l + ww) * w2)), int(round((t + hh) * h2))
            weight[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0.0
        info["excluded_rects"] = len(exclude)

    info["edge_weight_mean"] = round(float(weight.mean()), 4)

    if blur_sigma > 0:
        before_s = cv2.GaussianBlur(before, (0, 0), blur_sigma)
        after_s = cv2.GaussianBlur(after, (0, 0), blur_sigma)
    else:
        before_s, after_s = before, after

    return PairData(before=np.clip(before, 0, 1), after=np.clip(after, 0, 1),
                    weight=weight,
                    before_s=np.clip(before_s, 0, 1), after_s=np.clip(after_s, 0, 1),
                    info=info)


def sample_pixels(pairs: list[PairData], max_samples: int = 1_500_000, seed: int = 0):
    """Flatten a list of aligned pairs into weighted sample arrays."""
    rng = np.random.default_rng(seed)
    B, A, W = [], [], []
    per = max(1, max_samples // max(1, len(pairs)))
    for p in pairs:
        b = p.before_s.reshape(-1, 3)
        a = p.after_s.reshape(-1, 3)
        w = p.weight.reshape(-1)
        keep = w > 1e-3
        idx = np.flatnonzero(keep)
        if idx.size > per:
            idx = rng.choice(idx, per, replace=False)
        B.append(b[idx]); A.append(a[idx]); W.append(w[idx])
    return np.concatenate(B), np.concatenate(A), np.concatenate(W)
