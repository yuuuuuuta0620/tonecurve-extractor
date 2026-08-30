"""Test suite.  Run with:  .venv/bin/python -m pytest tests -q"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcx import colorspace as cs
from tcx import curves as C
from tcx.align import align_pair, sample_pixels
from tcx.extract import ExtractOptions, extract, get_channel_transfers, set_channel_transfers
from tcx.imageio_utils import discover_pairs, load_image, save_image, split_pair
from tcx.lut3d import apply_lut3d, fit_lut3d, write_cube
from tcx.model import BAND_NAMES, GradeZone, HSLBand, PresetModel
from tcx.render import band_weights, grade_masks, invert_post, render
from tcx.xmp import build_xmp

from examples.make_synthetic import known_preset, synthetic_photo


# --------------------------------------------------------------------- colour
def test_hsl_roundtrip():
    x = np.random.default_rng(0).random((5000, 3))
    h, s, l = cs.rgb_to_hsl(x)
    assert np.abs(cs.hsl_to_rgb(h, s, l) - x).max() < 1e-12


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 1000)
    assert np.abs(cs.linear_to_srgb(cs.srgb_to_linear(x)) - x).max() < 1e-12


@pytest.mark.parametrize("lab1,lab2,expected", [
    ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
    ([50.0, 2.5, 0.0], [50.0, 0.0, -2.5], 4.3065),
    ([50.0, 2.5, 0.0], [73.0, 25.0, -18.0], 27.1492),
])
def test_delta_e2000_reference(lab1, lab2, expected):
    """Values from Sharma et al.'s CIEDE2000 test set."""
    got = cs.delta_e2000(np.array(lab1), np.array(lab2))
    assert abs(float(got) - expected) < 1e-3


# --------------------------------------------------------------------- curves
def test_isotonic_is_monotone():
    y = np.random.default_rng(1).random(200)
    z = C.isotonic(y)
    assert np.all(np.diff(z) >= -1e-12)


def test_transfer_recovers_curve_under_contamination():
    rng = np.random.default_rng(2)
    truth = lambda x: np.clip(x ** 0.85 + 0.12 * np.sin(np.pi * x), 0, 1)
    x = rng.random(800_000)
    y = np.clip(truth(x) + rng.normal(0, 0.01, x.size), 0, 1)
    bad = rng.random(x.size) < 0.25          # 25 % "local adjustment"
    y[bad] = np.clip(y[bad] + 0.25, 0, 1)
    lut = C.estimate_transfer(x, y).lut
    err = np.abs(lut - truth(np.linspace(0, 1, C.LUT_N)))
    assert err.mean() < 0.006 and err.max() < 0.02


def test_lut_inverse_and_composition():
    g = np.linspace(0, 1, C.LUT_N)
    lut = np.clip(g ** 0.7, 0, 1)
    assert np.abs(C.compose_lut(C.invert_lut(lut), lut) - g).max() < 5e-3


def test_control_points_reproduce_lut():
    g = np.linspace(0, 1, C.LUT_N)
    lut = np.clip(0.03 + 0.94 * (g ** 0.9 + 0.08 * np.sin(np.pi * g)), 0, 1)
    pts = C.fit_control_points(lut, 16)
    assert 2 <= len(pts) <= 16
    assert np.abs(C.control_points_to_lut(pts) - lut).max() < 4 / 255


# --------------------------------------------------------------------- render
def test_band_weights_partition_of_unity():
    W = band_weights(np.linspace(0, 359.9, 4000))
    assert np.abs(W.sum(axis=1) - 1).max() < 1e-12
    assert W.min() >= 0


def test_grade_masks_partition_of_unity():
    M = grade_masks(np.linspace(0, 1, 500), 50, 0)
    assert np.abs(M.sum(axis=1) - 1).max() < 1e-12


def test_stage_inverses():
    m = PresetModel()
    m.hsl["Orange"] = HSLBand(-14, -22, 12)
    m.hsl["Blue"] = HSLBand(10, 18, -16)
    m.grade["Shadow"] = GradeZone(210, 18, -5)
    m.grade["Highlight"] = GradeZone(40, 14, 3)
    m.saturation, m.vibrance = 8, 15
    x = np.random.default_rng(3).random((20000, 3))
    y = render(x, m)
    assert np.abs(render(invert_post(y, m), m) - y).max() < 2e-3


def test_channel_transfer_decomposition_is_lossless():
    g = np.linspace(0, 1, C.LUT_N)
    f = [np.clip(np.maximum.accumulate(g ** p + o), 0, 1)
         for p, o in ((0.8, 0.02), (0.95, 0.0), (1.1, -0.01))]
    m = PresetModel()
    set_channel_transfers(m, f)
    for a, b in zip(get_channel_transfers(m), f):
        assert np.abs(a - b).max() < 6e-3


# ------------------------------------------------------------------ end-to-end
@pytest.fixture(scope="module")
def synthetic():
    before = synthetic_photo(500, 750, seed=7)
    truth = known_preset()
    after = render(before, truth)
    return before, after, truth


def test_extract_recovers_known_preset(synthetic):
    before, after, truth = synthetic
    pair = align_pair(before, after, do_align=False, blur_sigma=0.0)
    model, diag = extract([pair], ExtractOptions(iterations=4))

    v, b = diag["verification_mean"], diag["baseline_mean"]
    assert v["dE_mean"] < 1.0, v
    assert v["dE_mean"] < b["dE_mean"] / 8
    assert v["psnr_db"] > 40

    ft, fg = get_channel_transfers(truth), get_channel_transfers(model)
    assert np.mean([np.abs(a - b_) for a, b_ in zip(ft, fg)]) * 255 < 3.0

    err = np.mean([abs(getattr(truth.hsl[k], f) - getattr(model.hsl[k], f))
                   for k in BAND_NAMES for f in ("hue", "sat", "lum")])
    assert err < 5.0, err


def test_extract_survives_jpeg_resize_and_shift(synthetic):
    import cv2
    from PIL import Image
    before, after, _ = synthetic

    def jpeg(a, q=85):
        buf = io.BytesIO()
        Image.fromarray((np.clip(a, 0, 1) * 255 + .5).astype(np.uint8)).save(
            buf, "JPEG", quality=q, subsampling=2)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB")).astype(np.float64) / 255

    big = cv2.resize(after, (1200, 800), interpolation=cv2.INTER_LANCZOS4)
    big = cv2.warpAffine(big, np.float32([[1, 0, 3.4], [0, 1, -2.1]]), (1200, 800),
                         borderMode=cv2.BORDER_REPLICATE)
    pair = align_pair(jpeg(before), jpeg(np.clip(big, 0, 1)))
    assert pair.info["ncc_after_align"] >= pair.info["ncc_before_align"]
    model, diag = extract([pair], ExtractOptions(iterations=3))
    assert diag["verification_mean"]["dE_mean"] < 2.5


def test_lut3d_beats_or_matches_preset(synthetic):
    from tcx import metrics as M
    before, after, _ = synthetic
    pair = align_pair(before, after, do_align=False, blur_sigma=0.0)
    model, _ = extract([pair], ExtractOptions(iterations=3))
    B, A, W = sample_pixels([pair], 200_000)
    lut = fit_lut3d(B, A, W, model=model, size=17)
    got = M.compare(apply_lut3d(pair.before, lut), pair.after, pair.weight)
    assert got["dE_mean"] < 1.5


def test_cube_file_is_parseable(tmp_path, synthetic):
    lut = np.clip(np.random.default_rng(0).random((9, 9, 9, 3)), 0, 1)
    p = tmp_path / "x.cube"
    write_cube(str(p), lut, "t")
    lines = [l for l in p.read_text().splitlines() if l and not l[0].isalpha()]
    assert len(lines) == 9 ** 3
    assert all(len(l.split()) == 3 for l in lines)


# ---------------------------------------------------------------------- I/O
def test_split_pair():
    img = np.zeros((10, 20, 3))
    img[:, 10:] = 1.0
    a, b = split_pair(img, "lr")
    assert a.mean() == 0 and b.mean() == 1
    a, b = split_pair(img, "rl")
    assert a.mean() == 1 and b.mean() == 0


def test_discover_pairs(tmp_path):
    for n in ("shot1_before.jpg", "shot1_after.jpg", "shot2-before.png", "shot2-after.png"):
        save_image(str(tmp_path / n), np.zeros((4, 4, 3)))
    pairs = discover_pairs(str(tmp_path))
    assert len(pairs) == 2
    assert all("before" in os.path.basename(b) for b, _ in pairs)


# ---------------------------------------------------------------------- XMP
def test_xmp_is_wellformed_and_complete():
    import xml.dom.minidom as md
    m = PresetModel(name="T")
    m.hsl["Red"] = HSLBand(sat=25)
    m.grade["Shadow"] = GradeZone(210, 20, -5)
    m.meta["control_points"] = {"master": [(0, 8), (128, 140), (255, 250)]}
    m.red = np.clip(np.linspace(0, 1, C.LUT_N) ** 0.9, 0, 1)
    s = build_xmp(m)
    md.parseString(s)
    for token in ("ToneCurvePV2012", "ToneCurvePV2012Red", "SaturationAdjustmentRed",
                  "SplitToningShadowHue", "ColorGradeBlending", 'ToneCurveName2012="Custom"'):
        assert token in s


# ---------------------------------------------------------------------- CLI
def test_cli_end_to_end(tmp_path):
    before = synthetic_photo(300, 450, seed=11)
    after = render(before, known_preset())
    save_image(str(tmp_path / "s_before.jpg"), before)
    save_image(str(tmp_path / "s_after.jpg"), after)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-m", "tcx", "extract", "--dir", str(tmp_path),
         "-o", str(tmp_path / "out"), "--stem", "p", "--iterations", "2", "--cube",
         "--cube-size", "9", "--preview"],
        cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ("p.xmp", "p.json", "p.png", "p.html", "p.cube", "p_rendered.jpg"):
        assert (tmp_path / "out" / f).exists(), f
    model = PresetModel.from_json(str(tmp_path / "out" / "p.json"))
    assert model.master.shape == (C.LUT_N,)


def test_webapp_smoke(tmp_path):
    from tcx.webapp import create_app
    before = synthetic_photo(220, 330, seed=13)
    after = render(before, known_preset())
    save_image(str(tmp_path / "b.jpg"), before)
    save_image(str(tmp_path / "a.jpg"), after)
    client = create_app(str(tmp_path / "work")).test_client()
    assert client.get("/").status_code == 200
    resp = client.post("/extract", content_type="multipart/form-data", data={
        "before": (open(tmp_path / "b.jpg", "rb"), "b.jpg"),
        "after": (open(tmp_path / "a.jpg", "rb"), "a.jpg"),
        "name": "W", "color_mode": "curves", "iterations": "2", "smooth": "2.0"})
    assert resp.status_code == 200
    assert "エラー:" not in resp.data.decode()
