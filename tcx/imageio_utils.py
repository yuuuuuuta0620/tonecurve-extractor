"""Image loading / saving and before-after pair discovery."""
from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")


def load_image(path: str) -> np.ndarray:
    """Load an image as float64 HxWx3 in [0, 1] (sRGB-encoded)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is not None:
            return _from_cv(raw)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I", "F"):
            a = np.asarray(im).astype(np.float64)
            a = a / (65535.0 if a.max() > 255 else 255.0)
            return np.repeat(a[..., None], 3, axis=2)
        im = im.convert("RGB")
        return np.asarray(im).astype(np.float64) / 255.0


def _from_cv(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        raw = raw[:, :, :3]
    rgb = raw[:, :, ::-1].astype(np.float64)
    scale = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(raw.dtype, 1.0)
    return np.clip(rgb / scale, 0.0, 1.0)


def save_image(path: str, rgb: np.ndarray) -> None:
    arr = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path, quality=95)


# --------------------------------------------------------------------------
# Side-by-side sample images
# --------------------------------------------------------------------------

def split_pair(img: np.ndarray, mode: str = "lr", gap: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split one composite sample image into (before, after)."""
    h, w = img.shape[:2]
    if mode in ("lr", "rl"):
        half = w // 2
        a, b = img[:, :half - gap // 2], img[:, half + (gap + 1) // 2:]
    elif mode in ("tb", "bt"):
        half = h // 2
        a, b = img[:half - gap // 2], img[half + (gap + 1) // 2:]
    else:
        raise ValueError(f"unknown split mode: {mode}")
    n = min(a.shape[0], b.shape[0])
    m = min(a.shape[1], b.shape[1])
    a, b = a[:n, :m], b[:n, :m]
    return (b, a) if mode in ("rl", "bt") else (a, b)


_BEFORE_PAT = re.compile(r"(.*?)[ _\-.]*(before|orig|original|raw|b)$", re.I)
_AFTER_KEYS = ["after", "edit", "edited", "preset", "a"]


def discover_pairs(directory: str) -> list[tuple[str, str]]:
    """Find *_before.* / *_after.* style pairs in a directory."""
    files = [p for p in sorted(glob.glob(os.path.join(directory, "*")))
             if os.path.splitext(p)[1].lower() in EXTS]
    by_stem = {}
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        by_stem[stem.lower()] = p

    pairs = []
    used = set()
    for stem, path in by_stem.items():
        m = _BEFORE_PAT.match(stem)
        if not m:
            continue
        base, kw = m.group(1), m.group(2)
        for akw in _AFTER_KEYS:
            for sep in ("_", "-", " ", "."):
                cand = f"{base}{sep}{akw}" if base else akw
                if cand in by_stem and by_stem[cand] != path:
                    pairs.append((path, by_stem[cand]))
                    used.add(path)
                    used.add(by_stem[cand])
                    break
            if path in used:
                break
    return pairs


def fetch_url(url: str, dest_dir: str, timeout: int = 30) -> str:
    """Download one user-supplied image URL.  You are responsible for
    respecting the source site's terms of service and copyright."""
    import requests
    os.makedirs(dest_dir, exist_ok=True)
    name = os.path.basename(url.split("?")[0]) or "download"
    if os.path.splitext(name)[1].lower() not in EXTS:
        name += ".jpg"
    out = os.path.join(dest_dir, name)
    r = requests.get(url, timeout=timeout, stream=True,
                     headers={"User-Agent": "tonecurve-extractor/0.1"})
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    return out
