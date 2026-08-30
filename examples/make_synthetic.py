"""Generate a synthetic before/after pair from a *known* preset.

This is the ground-truth harness: we invent a preset, render it, then ask
the extractor to recover it, so accuracy can be measured objectively.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcx import curves as C
from tcx.imageio_utils import save_image
from tcx.model import PresetModel, HSLBand, GradeZone
from tcx.render import render


def synthetic_photo(h=900, w=1350, seed=0) -> np.ndarray:
    """Multi-scale coloured noise: natural-ish statistics, wide gamut coverage."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3))
    for scale, amp in ((160, 1.0), (60, 0.5), (22, 0.25), (7, 0.12), (2.5, 0.05)):
        n = rng.normal(0, 1, (h, w, 3))
        img += amp * cv2.GaussianBlur(n, (0, 0), scale)
    img -= img.min()
    img /= img.max()
    # a vertical luminance ramp guarantees the full tonal range is exercised
    ramp = np.linspace(0.02, 0.98, w)[None, :, None]
    img = np.clip(0.55 * img + 0.45 * ramp, 0, 1)
    # a saturated colour patch strip so every hue band has data
    bh = max(1, h // 9)
    hues = np.linspace(0, 360, w, endpoint=False)
    from tcx.colorspace import hsl_to_rgb
    band = np.repeat(hsl_to_rgb(hues, np.full(w, 0.8), np.full(w, 0.5))[None, :, :], bh, axis=0)
    # vary lightness across the strip so hue bands cover the tonal range too
    lscale = np.linspace(0.45, 0.62, bh)[:, None, None]
    band = np.clip(hsl_to_rgb(np.repeat(hues[None, :], bh, axis=0),
                              np.full((bh, w), 0.8),
                              np.repeat(lscale, w, axis=1)[..., 0]), 0, 1)
    img[h - bh:] = band
    return np.clip(img, 0, 1)


def known_preset() -> PresetModel:
    m = PresetModel(name="Synthetic Truth")
    x = np.linspace(0, 1, C.LUT_N)
    # faded-matte S curve
    m.master = np.clip(0.045 + 0.93 * (x ** 0.92 + 0.10 * np.sin(np.pi * x)), 0, 1)
    m.master = np.maximum.accumulate(m.master)
    # teal shadows / warm highlights via per-channel curves
    m.red = np.clip(x + 0.030 * np.sin(np.pi * x) + 0.020 * x ** 2, 0, 1)
    m.green = np.clip(x + 0.010 * np.sin(np.pi * x), 0, 1)
    m.blue = np.clip(x + 0.045 * (1 - x) ** 2 - 0.030 * x ** 2, 0, 1)
    for k in ("red", "green", "blue"):
        setattr(m, k, np.maximum.accumulate(getattr(m, k)))
    m.hsl["Orange"] = HSLBand(hue=-14, sat=-22, lum=12)
    m.hsl["Blue"] = HSLBand(hue=10, sat=18, lum=-16)
    m.hsl["Green"] = HSLBand(hue=-25, sat=-30, lum=5)
    m.hsl["Aqua"] = HSLBand(hue=6, sat=12, lum=0)
    return m


def main() -> None:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic")
    os.makedirs(out, exist_ok=True)
    before = synthetic_photo()
    m = known_preset()
    after = render(before, m)
    save_image(os.path.join(out, "scene_before.jpg"), before)
    save_image(os.path.join(out, "scene_after.jpg"), after)
    m.to_json(os.path.join(out, "truth.json"))
    print("wrote", out)


if __name__ == "__main__":
    main()
