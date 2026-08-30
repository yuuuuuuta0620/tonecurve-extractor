"""Ground-truth evaluation: invent a preset, render it, recover it, score it."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcx import curves as C
from tcx.align import align_pair
from tcx.extract import ExtractOptions, extract_auto, get_channel_transfers
from tcx.imageio_utils import load_image, save_image
from tcx.model import BAND_NAMES
from examples.make_synthetic import synthetic_photo, known_preset
from tcx.render import render


def score(truth, got, before, after, tag, iterations=4, **kw):
    pair = align_pair(before, after, max_dim=1600, do_align=False, blur_sigma=0.0)
    opts = ExtractOptions(iterations=iterations, name=tag, **kw)
    model, diag = extract_auto([pair], opts)

    ft = get_channel_transfers(truth)
    fg = get_channel_transfers(model)
    curve_err = np.array([np.abs(a - b).mean() * 255 for a, b in zip(ft, fg)])

    rows = []
    for n in BAND_NAMES:
        t, g = truth.hsl[n], model.hsl[n]
        rows.append((n, t.hue, g.hue, t.sat, g.sat, t.lum, g.lum))
    hsl_err = np.mean([abs(r[1] - r[2]) + abs(r[3] - r[4]) + abs(r[5] - r[6]) for r in rows]) / 3

    v = diag["verification_mean"]
    b = diag["baseline_mean"]
    print(f"\n=== {tag} ===")
    print(f"  dE mean {v['dE_mean']:>6} (unedited {b['dE_mean']})   "
          f"p95 {v['dE_p95']:>6}   PSNR {v['psnr_db']} dB")
    print(f"  transfer-curve MAE (0-255): R {curve_err[0]:.2f}  G {curve_err[1]:.2f}  B {curve_err[2]:.2f}")
    print(f"  mean |slider error|: {hsl_err:.2f}")
    sel = diag.get("working_space_selection")
    if sel:
        print(f"  working space chosen: {sel['chosen']}  (ΔE {sel['dE_mean_by_space']})")
    print("  band      hue(t/g)     sat(t/g)     lum(t/g)")
    for n, th, gh, ts, gs, tl, gl in rows:
        if abs(th) + abs(ts) + abs(tl) + abs(gh) + abs(gs) + abs(gl) > 3:
            print(f"   {n:<9} {th:>5.0f}/{gh:>5.0f}  {ts:>5.0f}/{gs:>5.0f}  {tl:>5.0f}/{gl:>5.0f}")
    return model, diag


def main():
    out = "examples/synthetic"
    os.makedirs(out, exist_ok=True)
    before = synthetic_photo()
    truth = known_preset()
    after = render(before, truth)

    # 1. lossless -- pure algorithm error
    score(truth, None, before, after, "lossless (in-memory)")

    # 2. PNG round-trip (8-bit quantisation only)
    from PIL import Image
    for name, arr in (("b.png", before), ("a.png", after)):
        Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            os.path.join(out, name))
    score(truth, None, load_image(f"{out}/b.png"), load_image(f"{out}/a.png"), "PNG 8-bit")

    # 3. JPEG q92 4:4:4 -- realistic download quality
    for name, arr in (("b.jpg", before), ("a.jpg", after)):
        Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            os.path.join(out, name), quality=92, subsampling=0)
    score(truth, None, load_image(f"{out}/b.jpg"), load_image(f"{out}/a.jpg"), "JPEG q92 4:4:4")

    # 4. JPEG q85 4:2:0 -- typical web sample image
    for name, arr in (("b2.jpg", before), ("a2.jpg", after)):
        Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            os.path.join(out, name), quality=85, subsampling=2)
    score(truth, None, load_image(f"{out}/b2.jpg"), load_image(f"{out}/a2.jpg"),
          "JPEG q85 4:2:0 (chroma subsampled)")


if __name__ == "__main__":
    main()
