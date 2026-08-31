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
from tcx.extract import (ExtractOptions, explain_residual, extract, extract_auto,
                         extract_levelled, get_channel_transfers, set_channel_transfers)
from tcx.imageio_utils import (discover_pairs, load_image, match_names, save_image,
                               split_pair)
from tcx.lut3d import (apply_lut3d, bake_lut3d, fit_lut3d, separability_error,
                       write_cube, write_cube_1d)
from tcx.model import BAND_NAMES, GradeZone, HSLBand, PresetModel
from tcx.render import (apply_exposure, band_weights, grade_masks, invert_post, render)
from tcx.xmp import build_xmp

from examples.make_synthetic import (known_preset, stamp_watermark, synthetic_photo,
                                     watermark_layers)


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
    # checked inside the working space: converting back to sRGB can clip
    # colours the edit pushed out of gamut, and clipping is not invertible
    xw = cs.to_working(x, m.working_space)
    yw = render(xw, m, in_working_space=True)
    back = render(invert_post(yw, m, in_working_space=True), m, in_working_space=True)
    err = np.abs(back - yw)
    ok = (yw > 0.01) & (yw < 0.99)
    assert err[ok].max() < 2e-3, err[ok].max()
    assert err.max() < 1e-2, err.max()


@pytest.mark.parametrize("space", ["srgb", "melissa"])
def test_working_space_roundtrip_and_neutral_axis(space):
    m = PresetModel(working_space=space)
    x = np.random.default_rng(5).random((5000, 3))
    assert np.abs(cs.from_working(cs.to_working(x, space), space) - x).max() < 1e-9
    # the two spaces must agree exactly on neutrals -- that is what makes the
    # master curve transfer faithfully regardless of the choice
    g = np.repeat(np.linspace(0, 1, 256)[:, None], 3, axis=1)
    assert np.abs(cs.to_working(g, space) - g).max() < 1e-6


def _pair_of_spaces(luts):
    from tcx.extract import set_channel_transfers
    a, b = PresetModel(working_space="srgb"), PresetModel(working_space="melissa")
    for m in (a, b):
        set_channel_transfers(m, luts)
    return a, b


def test_working_space_agrees_on_neutral_pixels():
    """A master (R=G=B) curve maps a grey pixel identically in both spaces --
    that is why overall tonality transfers to Lightroom regardless."""
    g = np.linspace(0, 1, C.LUT_N)
    mono = np.clip(np.maximum.accumulate(g ** 0.85 + 0.08 * np.sin(np.pi * g)), 0, 1)
    a, b = _pair_of_spaces([mono, mono, mono])
    grey = np.repeat(np.linspace(0.02, 0.98, 200)[:, None], 3, axis=1)
    assert np.abs(render(grey, a) - render(grey, b)).max() < 1e-6


def test_working_space_matters_most_for_per_channel_colour():
    """On photographic content the working space barely moves a neutral
    curve, but it moves a per-channel colour curve a lot: ProPhoto's primaries
    are far more saturated, so the same channel imbalance reads as a much
    stronger tint there than it does in sRGB."""
    g = np.linspace(0, 1, C.LUT_N)
    mono = np.clip(np.maximum.accumulate(g ** 0.85 + 0.08 * np.sin(np.pi * g)), 0, 1)
    colour = [np.clip(np.maximum.accumulate(g + o * np.sin(np.pi * g)), 0, 1)
              for o in (0.06, 0.0, -0.05)]
    x = synthetic_photo(300, 450, seed=9).reshape(-1, 3)

    def gap(luts):
        a, b = _pair_of_spaces(luts)
        return float(cs.delta_e2000(cs.rgb_to_lab(render(x, a)),
                                    cs.rgb_to_lab(render(x, b))).mean())

    g_mono, g_colour = gap([mono] * 3), gap(colour)
    assert g_mono < 1.5, g_mono
    assert g_colour > 4 * g_mono, (g_mono, g_colour)


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


def test_extract_rejects_abstract_working_space():
    with pytest.raises(ValueError, match="concrete working space"):
        extract([], ExtractOptions(working_space="auto"))


@pytest.mark.parametrize("truth_space", ["srgb", "melissa"])
def test_auto_identifies_the_space_the_edit_was_made_in(truth_space):
    """A per-channel curve in one space is not expressible as a per-channel
    curve in the other, so the residual reveals which one the editor used."""
    before = synthetic_photo(400, 600, seed=21)
    truth = known_preset()
    truth.working_space = truth_space
    pair = align_pair(before, render(before, truth), do_align=False, blur_sigma=0.0)
    model, diag = extract_auto([pair], ExtractOptions(iterations=3))
    sel = diag["working_space_selection"]
    assert sel["chosen"] == truth_space, sel
    assert model.working_space == truth_space
    assert sel["margin"] > 0.3, sel


def test_extract_recovers_known_preset(synthetic):
    before, after, truth = synthetic
    pair = align_pair(before, after, do_align=False, blur_sigma=0.0)
    model, diag = extract([pair], ExtractOptions(iterations=4,
                                                working_space=truth.working_space))

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
    model, diag = extract_auto([pair], ExtractOptions(iterations=3))
    assert diag["verification_mean"]["dE_mean"] < 2.5


def test_burned_in_caption_is_detected_and_excluded():
    """An opaque caption asserts f(x) = x with huge leverage at the white end.
    It cannot be found from the fit residual -- the curve bends to fit it --
    so it must be excluded before the first fit."""
    before = synthetic_photo(600, 900, seed=31)
    truth = known_preset()
    after = render(before, truth)
    soft, hard = watermark_layers(*before.shape[:2])
    bw = stamp_watermark(before, soft * 0, hard)
    aw = stamp_watermark(after, soft * 0, hard)

    opaque_fraction = float((hard > 0).mean())
    pair = align_pair(bw, aw, do_align=False, blur_sigma=0.0)
    model, diag = extract([pair], ExtractOptions(
        iterations=4, working_space=truth.working_space))

    fz = diag["frozen_pixels"]
    assert fz["action"] == "excluded from the fit"
    assert abs(fz["fraction"] - opaque_fraction) < 0.01, (fz, opaque_fraction)

    def highlight_err(m):
        ft, fg = get_channel_transfers(truth), get_channel_transfers(m)
        return float(np.mean([np.abs(a - b)[900:] for a, b in zip(ft, fg)]) * 255)

    off, _ = extract([align_pair(bw, aw, do_align=False, blur_sigma=0.0)],
                     ExtractOptions(iterations=4, working_space=truth.working_space,
                                    detect_frozen=False, reject_sigma=0))
    assert highlight_err(model) < 0.5 * highlight_err(off), (
        highlight_err(model), highlight_err(off))


def test_frozen_detection_does_not_fire_on_clean_pairs():
    before = synthetic_photo(400, 600, seed=33)
    truth = known_preset()
    pair = align_pair(before, render(before, truth), do_align=False, blur_sigma=0.0)
    _, diag = extract([pair], ExtractOptions(iterations=2,
                                             working_space=truth.working_space))
    assert diag["frozen_pixels"]["fraction"] < 0.005, diag["frozen_pixels"]


def test_manual_mask_and_exclude_zero_the_weights():
    img = synthetic_photo(200, 300, seed=1)
    mask = np.ones(img.shape[:2])
    mask[:, :90] = 0.0
    p1 = align_pair(img, img, do_align=False, mask=mask)
    assert p1.weight[:, :80].max() == 0
    assert p1.weight[:, 150:].max() > 0
    p2 = align_pair(img, img, do_align=False, exclude=[(0.0, 0.8, 1.0, 0.2)])
    assert p2.weight[int(0.9 * 200):].max() == 0
    assert p2.info["excluded_rects"] == 1


def test_clean_samples_transfer_to_an_unseen_photograph():
    """The whole point of a preset: fitted on the seller's frames, it has to
    work on a photograph it has never seen.  With samples that carry only the
    preset, it does."""
    look = known_preset()
    pairs = []
    for i in range(2):
        shot = synthetic_photo(400, 600, seed=200 + i)
        pairs.append(align_pair(shot, render(shot, look), do_align=False, blur_sigma=0.0))
    model, _ = extract(pairs, ExtractOptions(iterations=4,
                                             working_space=look.working_space))

    mine = synthetic_photo(400, 600, seed=999)
    want = render(mine, look)
    from tcx import metrics as Metrics
    assert Metrics.compare(mine, want)["dE_mean"] > 8          # the look is substantial
    assert Metrics.compare(render(mine, model), want)["dE_mean"] < 1.2


def test_per_frame_exposure_is_levelled_and_reported():
    """Samples that disagree on exposure prove per-photo work is present."""
    look = known_preset()
    pairs = []
    for i, ev in enumerate((-0.7, 0.0, 0.7)):
        shot = synthetic_photo(360, 540, seed=300 + i)
        pairs.append(align_pair(shot, render(apply_exposure(shot, ev), look),
                                do_align=False, blur_sigma=0.0))
    _, diag = extract(pairs, ExtractOptions(iterations=2,
                                            working_space=look.working_space))
    pe = diag["pair_exposure"]
    # the measured spread is compressed by the look's own tone curve, so it
    # under-reads the 1.4 EV that was applied -- it only has to be detected
    assert pe["spread_ev"] > 0.5, pe
    assert any("disagree on exposure" in w for w in diag.get("warnings", []))
    assert sum(p.fit_before is not None for p in pairs) >= 2


def test_tone_mode_does_not_warn_about_a_single_pair():
    """A master curve has too little freedom to overfit per-frame work, so the
    single-pair alarm that applies to a colour fit does not apply here."""
    shot = synthetic_photo(300, 450, seed=5)
    look = known_preset()
    pair = align_pair(shot, render(shot, look), do_align=False, blur_sigma=0.0)
    _, diag = extract([pair], ExtractOptions(iterations=2, color_mode="tone",
                                             colour_guide=False,
                                             working_space=look.working_space))
    warnings = diag.get("warnings", [])
    assert not any("single pair" in w for w in warnings), warnings
    assert not any("cannot tell them apart" in w for w in warnings), warnings


def test_single_pair_is_flagged_as_unseparable():
    shot = synthetic_photo(300, 450, seed=5)
    look = known_preset()
    pair = align_pair(shot, render(shot, look), do_align=False, blur_sigma=0.0)
    _, diag = extract([pair], ExtractOptions(iterations=2, colour_guide=False,
                                             working_space=look.working_space))
    assert any("single pair" in w for w in diag.get("warnings", []))


def _brushwork(img, seed):
    """A local adjustment no global preset can express."""
    import cv2
    h, w = img.shape[:2]
    r = np.random.default_rng(seed)
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(r.uniform(.3, .7) * w), int(r.uniform(.3, .7) * h)),
                (int(.28 * w), int(.34 * h)), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), 0.06 * w)[..., None]
    return np.clip(img * (1 - m) + np.clip(img * np.array([1.22, 1.12, 1.02]), 0, 1) * m, 0, 1)


@pytest.mark.parametrize("per_frame,local,expect_disagree,expect_local", [
    (True, False, True, False),     # only per-frame exposure/WB: droppable
    (False, True, False, True),     # only brushwork: the real limit
    (False, False, False, False),   # clean samples
])
def test_diagnostic_separates_disagreement_from_local_work(
        per_frame, local, expect_disagree, expect_local):
    look = known_preset()
    rng = np.random.default_rng(3)
    pairs = []
    for i in range(4):
        shot = synthetic_photo(240, 360, seed=700 + i)
        x = shot
        if per_frame:
            x = apply_exposure(shot, float(rng.normal(0, 0.5)))
            x = np.clip(x * (1 + rng.normal(0, 0.03, 3)), 0, 1)
        out = render(x, look)
        if local:
            out = _brushwork(out, 900 + i)
        pairs.append(align_pair(shot, out, do_align=False, blur_sigma=0.0))

    opts = ExtractOptions(iterations=2, working_space=look.working_space)
    _, diag = extract_levelled(pairs, opts, look.working_space)
    ex = explain_residual(pairs, opts, diag, look.working_space)
    assert ex["pairs_disagree"] is expect_disagree, ex
    assert ex["local_work_suspected"] is expect_local, ex


def test_releveling_never_makes_the_fit_worse():
    look = known_preset()
    rng = np.random.default_rng(11)
    pairs = []
    for i in range(3):
        shot = synthetic_photo(240, 360, seed=800 + i)
        pairs.append(align_pair(shot, render(apply_exposure(shot, float(rng.normal(0, .5))), look),
                                do_align=False, blur_sigma=0.0))
    opts = ExtractOptions(iterations=2, working_space=look.working_space)
    plain, d0 = extract(pairs, opts)
    for p in pairs:
        p.fit_before = None
    _, d1 = extract_levelled(pairs, opts, look.working_space)
    assert d1["verification_mean"]["dE_mean"] <= d0["verification_mean"]["dE_mean"] + 1e-6
    assert "releveling" in d1


def test_tone_only_mode_emits_a_neutral_curve_and_nothing_else():
    before = synthetic_photo(300, 450, seed=17)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    model, diag = extract([pair], ExtractOptions(
        iterations=3, color_mode="tone", working_space=look.working_space))

    ident = C.identity_lut(len(model.red))
    for ch in (model.red, model.green, model.blue):
        assert np.abs(ch - ident).max() < 1.5 / 255       # colour curves untouched
    assert np.abs(model.master - ident).max() > 10 / 255  # but the tone curve works
    assert not model.has_color_curves() and not model.has_grading()
    assert all(v.hue == 0 and v.sat == 0 and v.lum == 0 for v in model.hsl.values())

    # a preset containing only the master curve is the same in either working
    # space on neutrals, which is what makes it the safe part to transfer
    other = PresetModel.from_dict(model.to_dict())
    other.working_space = "srgb" if model.working_space == "melissa" else "melissa"
    grey = np.repeat(np.linspace(0.02, 0.98, 128)[:, None], 3, axis=1)
    assert np.abs(render(grey, model) - render(grey, other)).max() < 1e-6


def test_colour_guide_measures_the_work_left_after_the_tone_curve():
    from tcx.guide import build_guide, format_guide
    before = synthetic_photo(300, 450, seed=19)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    model, diag = extract([pair], ExtractOptions(
        iterations=3, color_mode="tone", working_space=look.working_space))

    g = diag["colour_guide"]
    assert g["total_colour_dE_after_tone"] > 1.0
    assert any(z["measured"] for z in g["zones"])
    assert g["ranked_bands"] and "hue_shift_deg" in g["ranked_bands"][0]
    # ranked strongest first
    ex = [b["explains_dE"] for b in g["ranked_bands"]]
    assert ex == sorted(ex, reverse=True)
    # a near-neutral reference must not report a meaningless chroma percentage
    neutral = next(m for m in g["memory_colours"] if m["colour"] == "neutral mid")
    assert neutral["chroma_pct"] is None or abs(neutral["chroma_pct"]) < 200
    assert "colour guide" in format_guide(g)


def test_characteristic_patches_are_varied_and_serialisable():
    from tcx.guide import characteristic_patches, format_patches, patches_json
    before = synthetic_photo(400, 600, seed=23)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    model, _ = extract([pair], ExtractOptions(iterations=2, color_mode="tone",
                                              working_space=look.working_space))
    patches = characteristic_patches([pair], model, n_patches=6)
    assert 3 <= len(patches) <= 6
    # never more than two crops of the same hue family, and none overlapping
    from collections import Counter
    assert max(Counter(c["band"] for c in patches).values()) <= 2
    for i, a in enumerate(patches):
        for b in patches[i + 1:]:
            if a["pair"] == b["pair"]:
                assert (abs(a["y"] - b["y"]) >= 3 * a["size"]
                        or abs(a["x"] - b["x"]) >= 3 * a["size"])
    for c in patches:
        assert c["crop_before"].ndim == 3 and c["crop_after"].shape == c["crop_before"].shape
        assert c["dE"] > 0
    json.dumps(patches_json(patches))          # must survive serialisation
    assert "Point Colour" in format_patches(patches)


def test_look_description_cites_its_evidence():
    from tcx.guide import build_guide, describe_look
    before = synthetic_photo(360, 540, seed=29)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    model, diag = extract([pair], ExtractOptions(iterations=3,
                                                 working_space=look.working_space))
    d = diag["look_description"]
    assert d["traits"], d
    for t in d["traits"]:
        assert t["en"] and t["ja"] and t["evidence"]
    ids = {t["id"] for t in d["traits"]}
    # the known preset lifts blacks and flattens the midtones
    assert "lifted_blacks" in ids and "soft" in ids, ids


def test_tone_only_guide_does_not_invent_unmeasurable_colours():
    """A tone-only preset has no colour to predict with; saying a colour is
    unchanged would be a different claim from saying it was not measurable."""
    from tcx.guide import format_guide
    before = synthetic_photo(200, 300, seed=31)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    _, diag = extract([pair], ExtractOptions(iterations=2, color_mode="tone",
                                             working_space=look.working_space))
    mem = diag["colour_guide"]["memory_colours"]
    for m in mem:
        if m.get("measurable") is False:
            assert "not measurable" in m["source"]
            assert "dE" not in m or m["dE"] is None or m["source"].startswith("not")
    format_guide(diag["colour_guide"])


def test_lut3d_beats_or_matches_preset(synthetic):
    from tcx import metrics as M
    before, after, _ = synthetic
    pair = align_pair(before, after, do_align=False, blur_sigma=0.0)
    model, _ = extract_auto([pair], ExtractOptions(iterations=3))
    B, A, W = sample_pixels([pair], 200_000)
    lut = fit_lut3d(B, A, W, model=model, size=17)
    got = M.compare(apply_lut3d(pair.before, lut), pair.after, pair.weight)
    assert got["dE_mean"] < 1.5


def test_baked_cube_reproduces_the_preset_it_came_from():
    """A 3D LUT samples the curve and interpolates linearly between nodes,
    so lattice size decides whether that is invisible or not."""
    before = synthetic_photo(300, 450, seed=61)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    model, _ = extract([pair], ExtractOptions(iterations=2, color_mode="tone",
                                              colour_guide=False,
                                              working_space=look.working_space))
    test = synthetic_photo(240, 360, seed=63)
    ref = render(test, model)
    errs = {}
    for n in (9, 33):
        got = apply_lut3d(test, bake_lut3d(model, n))
        errs[n] = float(cs.delta_e2000(cs.rgb_to_lab(got), cs.rgb_to_lab(ref)).mean())
    assert errs[33] < 0.15                 # well under a just-noticeable difference
    assert errs[9] > errs[33] * 3          # a coarse lattice really does cost


@pytest.mark.parametrize("space,separable", [("srgb", True), ("melissa", False)])
def test_separability_depends_on_the_working_space(space, separable):
    """A master curve is three identical 1-D curves in sRGB, but applying it in
    ProPhoto primaries mixes the channels, so a 1-D LUT is no longer valid."""
    before = synthetic_photo(260, 390, seed=65)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    model, _ = extract([pair], ExtractOptions(iterations=2, color_mode="tone",
                                              colour_guide=False, working_space=space))
    err = separability_error(model)
    assert (err < 0.05) is separable, (space, err)


def test_1d_cube_is_exact_when_separable(tmp_path):
    before = synthetic_photo(260, 390, seed=67)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    model, _ = extract([pair], ExtractOptions(iterations=2, color_mode="tone",
                                              colour_guide=False, working_space="srgb"))
    path = tmp_path / "t.cube"
    write_cube_1d(str(path), model)
    lines = [l for l in path.read_text().splitlines() if l[:1].isdigit()]
    assert "LUT_1D_SIZE 1024" in path.read_text()
    vals = np.array([[float(x) for x in l.split()] for l in lines])
    assert len(vals) == 1024

    g = np.linspace(0, 1, 1024)
    test = synthetic_photo(200, 300, seed=69)
    got = np.stack([np.interp(test[..., c], g, vals[:, c]) for c in range(3)], axis=-1)
    err = float(cs.delta_e2000(cs.rgb_to_lab(got), cs.rgb_to_lab(render(test, model))).max())
    assert err < 0.01, err


def test_colour_chart_covers_only_colours_the_photographs_contain(tmp_path):
    from tcx.chart import build_chart, chart_json, render_chart, write_charts
    before = synthetic_photo(400, 600, seed=71)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    model, diag = extract([pair], ExtractOptions(iterations=2, colour_guide=False,
                                                 working_space=look.working_space))
    chart = diag["colour_chart"]
    assert chart["cells"], "no swatches found"
    # empty rows and columns are trimmed, so every one that survives must
    # carry at least one swatch
    for ri in range(len(chart["rows"])):
        assert any(r == ri for r, _ in chart["cells"]), ri
    for ci in range(len(chart["columns"])):
        assert any(c == ci for _, c in chart["cells"]), ci
    # hue columns are labelled with the Colour Mixer band they fall in
    assert all(col == "neutral" or col.split()[0] in {b[:3] for b in BAND_NAMES}
               for col in chart["columns"]), chart["columns"]

    for c in chart["cells"].values():
        assert c["before"].shape == (3,) and c["after"].shape == (3,)
        assert np.all((c["before"] >= 0) & (c["before"] <= 1))
        assert "tone" in c

    json.dumps(chart_json(chart))

    # the three plain charts must share geometry, or they cannot be compared
    sizes = {k: render_chart(chart, k, labels=False).size
             for k in ("before", "tone", "after")}
    assert len(set(sizes.values())) == 1, sizes
    paths = write_charts(str(tmp_path / "x"), chart)
    assert len(paths) == 4 and all(os.path.exists(p) for p in paths)


@pytest.mark.parametrize("hues,rows", [(8, 4), (16, 6), (24, 8)])
def test_chart_density_is_parametrised(hues, rows):
    from tcx.chart import build_chart
    before = synthetic_photo(400, 600, seed=75)
    look = known_preset()
    pair = align_pair(before, render(before, look), do_align=False, blur_sigma=0.0)
    model, _ = extract([pair], ExtractOptions(iterations=2, colour_guide=False,
                                              colour_chart=False,
                                              working_space=look.working_space))
    B, A, W = sample_pixels([pair], 300_000)
    chart = build_chart(B, A, W, model, n_hues=hues, n_rows=rows)
    assert 1 <= len(chart["columns"]) <= hues + 1
    assert 1 <= len(chart["rows"]) <= rows
    # indices must stay inside the trimmed grid
    for (ri, ci) in chart["cells"]:
        assert 0 <= ri < len(chart["rows"]) and 0 <= ci < len(chart["columns"])


def test_webapp_offers_the_charts_for_download(tmp_path):
    """The charts are the deliverable for matching by eye in Lightroom, so the
    web UI has to hand over the files, not just render them inline."""
    import re
    from tcx.webapp import create_app
    look = known_preset()
    shot = synthetic_photo(240, 360, seed=77)
    save_image(str(tmp_path / "01_before.jpg"), shot)
    save_image(str(tmp_path / "01_after.jpg"), render(shot, look))
    client = create_app(str(tmp_path / "work")).test_client()
    resp = client.post("/extract", content_type="multipart/form-data", data={
        "before": (open(tmp_path / "01_before.jpg", "rb"), "01_before.jpg"),
        "after": (open(tmp_path / "01_after.jpg", "rb"), "01_after.jpg"),
        "name": "My Look", "color_mode": "tone", "iterations": "2", "smooth": "2.0"})
    body = resp.data.decode()
    token = re.search(r"/download/([0-9a-f]+)/preset\.xmp", body).group(1)
    charts = re.findall(r"/download/[0-9a-f]+/([\w.-]+_chart_[a-z]+\.png)", body)
    assert {c.split("_chart_")[1] for c in charts} == {
        "before.png", "tone.png", "after.png", "compare.png"}
    for name in charts:
        assert not name.startswith("chart_chart")      # no doubled stem
        got = client.get(f"/download/{token}/{name}")
        assert got.status_code == 200 and len(got.data) > 1000


def test_chart_swatches_match_what_the_preset_does():
    """The 'after' swatch has to be the target, and the 'tone' swatch has to be
    what the curve alone produces -- otherwise matching by eye is meaningless."""
    from tcx.chart import build_chart
    from tcx.guide import _tone_only
    before = synthetic_photo(300, 450, seed=73)
    look = known_preset()
    after = render(before, look)
    pair = align_pair(before, after, do_align=False, blur_sigma=0.0)
    model, _ = extract([pair], ExtractOptions(iterations=3, colour_guide=False,
                                              working_space=look.working_space))
    B, A, W = sample_pixels([pair], 300_000)
    chart = build_chart(B, A, W, model)
    for c in chart["cells"].values():
        got = _tone_only(np.asarray(c["before"])[None, :], model)[0]
        assert float(cs.delta_e2000(cs.rgb_to_lab(got),
                                    cs.rgb_to_lab(c["tone"]))) < 6.0


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
@pytest.mark.parametrize("befores,afters,expect", [
    (["s3_before.jpg", "s1_before.jpg", "s2_before.jpg"],
     ["s2_after.jpg", "s3_after.jpg", "s1_after.jpg"],
     [("s3_before.jpg", "s3_after.jpg"), ("s1_before.jpg", "s1_after.jpg"),
      ("s2_before.jpg", "s2_after.jpg")]),
    (["01-Nostargia_before.jpg"], ["01-Nostargia_after.jpg"],
     [("01-Nostargia_before.jpg", "01-Nostargia_after.jpg")]),
    (["DSC001.jpg", "DSC003.jpg", "DSC002.jpg"], ["e3.jpg", "e1.jpg", "e2.jpg"],
     [("DSC001.jpg", "e1.jpg"), ("DSC002.jpg", "e2.jpg"), ("DSC003.jpg", "e3.jpg")]),
])
def test_match_names_pairs_uploads_regardless_of_order(befores, afters, expect):
    got = [(befores[i], afters[j]) for i, j in match_names(befores, afters)]
    assert got == expect


def test_match_names_keeps_trailing_letters_of_real_words():
    """A single-letter role suffix needs a separator, or 'Nostargia' becomes
    'Nostargi' and unrelated files start matching."""
    from tcx.imageio_utils import _pair_key
    assert _pair_key("01-Nostargia_before.jpg") == "01-nostargia"
    assert _pair_key("01-Nostargia_after.jpg") == "01-nostargia"
    assert _pair_key("shot_a.jpg") == "shot"


def test_match_names_rejects_unequal_counts():
    with pytest.raises(ValueError, match="counts must match"):
        match_names(["a_before.jpg", "b_before.jpg"], ["a_after.jpg"])


def test_webapp_pairs_six_uploads_and_flags_a_bad_one(tmp_path):
    from tcx.webapp import create_app
    look = known_preset()
    names = []
    for i in range(4):
        shot = synthetic_photo(200, 300, seed=500 + i)
        save_image(str(tmp_path / f"0{i}-X_before.jpg"), shot)
        save_image(str(tmp_path / f"0{i}-X_after.jpg"), render(shot, look))
        names.append(f"0{i}-X")
    # a pair whose "after" is a different photograph entirely
    save_image(str(tmp_path / "zz_before.jpg"), synthetic_photo(200, 300, seed=42))
    save_image(str(tmp_path / "zz_after.jpg"), synthetic_photo(200, 300, seed=7))
    names.append("zz")

    client = create_app(str(tmp_path / "work")).test_client()
    resp = client.post("/extract", content_type="multipart/form-data", data={
        "before": [(open(tmp_path / f"{n}_before.jpg", "rb"), f"{n}_before.jpg")
                   for n in names],
        # handed over in the opposite order on purpose
        "after": [(open(tmp_path / f"{n}_after.jpg", "rb"), f"{n}_after.jpg")
                  for n in reversed(names)],
        "name": "T", "color_mode": "curves", "iterations": "2", "smooth": "2.0"})
    body = resp.data.decode()
    assert resp.status_code == 200 and "エラー:" not in body
    table = body[body.index("</form>"):]
    for n in names:
        row = f"<td>{n}_before.jpg</td><td>{n}_after.jpg</td>"
        assert row in table, f"{n} was mis-paired"
    assert "誤差が突出しているペア" in table


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
