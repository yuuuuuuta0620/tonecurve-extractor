"""Command line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import __version__
from . import curves as C
from .align import align_pair
from .extract import ExtractOptions, extract_auto
from .imageio_utils import (load_image, save_image, split_pair, discover_pairs, fetch_url)
from .lut3d import fit_lut3d, write_cube, apply_lut3d
from .model import PresetModel, Calibration
from .render import render
from .report import make_figure, write_html
from .xmp import write_xmp


def _load_pair_spec(spec: str, split: str | None, split_gap: int):
    """'before.jpg,after.jpg' or a single side-by-side image."""
    parts = [p for p in spec.split(",") if p]
    if len(parts) == 2:
        return load_image(parts[0]), load_image(parts[1])
    if len(parts) == 1:
        if not split:
            raise SystemExit(f"'{spec}' is a single image; pass --split lr|rl|tb|bt")
        return split_pair(load_image(parts[0]), split, split_gap)
    raise SystemExit(f"cannot parse pair spec: {spec}")


def cmd_extract(a) -> int:
    specs: list[str] = list(a.pair or [])
    if a.before and a.after:
        specs.append(f"{a.before},{a.after}")
    if a.dir:
        found = discover_pairs(a.dir)
        if not found:
            raise SystemExit(f"no *_before/*_after pairs found in {a.dir}")
        specs += [f"{b},{af}" for b, af in found]
    if not specs:
        raise SystemExit("give --before/--after, --pair, or --dir")

    cal = Calibration()
    if a.calibration:
        with open(a.calibration, encoding="utf-8") as f:
            cal = Calibration(**json.load(f))

    pairs = []
    for spec in specs:
        b, af = _load_pair_spec(spec, a.split, a.split_gap)
        pd = align_pair(b, af, max_dim=a.max_dim, motion=a.motion,
                        blur_sigma=a.blur, edge_percentile=a.edge_percentile,
                        do_align=not a.no_align)
        pd.info["source"] = spec
        pairs.append(pd)
        if a.verbose:
            print(f"  pair {spec}: {pd.info}", file=sys.stderr)

    opts = ExtractOptions(
        color_mode=a.color_mode, saturation_mode=a.saturation_mode,
        iterations=a.iterations, max_samples=a.max_samples, smooth=a.smooth,
        max_points=a.max_points, fit_hsl=not a.no_hsl, fit_hsl_hue=not a.no_hue,
        fit_hsl_lum=not a.no_hsl_lum, grading_global=a.grading_global,
        quantize=not a.no_quantize, working_space=a.working_space,
        name=a.name, group=a.group, calibration=cal)

    model, diag = extract_auto(pairs, opts)

    os.makedirs(a.outdir, exist_ok=True)
    stem = a.stem or "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in a.name)

    xmp_path = os.path.join(a.outdir, stem + ".xmp")
    xmp = write_xmp(xmp_path, model)
    model.to_json(os.path.join(a.outdir, stem + ".json"))
    with open(os.path.join(a.outdir, stem + ".diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=1, ensure_ascii=False)

    outputs = [xmp_path]
    if not a.no_report:
        png = make_figure(model, diag, pairs)
        png_path = os.path.join(a.outdir, stem + ".png")
        with open(png_path, "wb") as f:
            f.write(png)
        html_path = os.path.join(a.outdir, stem + ".html")
        write_html(html_path, model, diag, png, xmp)
        outputs += [png_path, html_path]

    if a.preview:
        save_image(os.path.join(a.outdir, stem + "_rendered.jpg"),
                   render(pairs[0].before, model))
        outputs.append(os.path.join(a.outdir, stem + "_rendered.jpg"))

    if a.cube:
        from .align import sample_pixels
        B, A, W = sample_pixels(pairs, min(a.max_samples, 600_000))
        lut = fit_lut3d(B, A, W, model=model, size=a.cube_size)
        cube_path = os.path.join(a.outdir, stem + ".cube")
        write_cube(cube_path, lut, title=a.name)
        outputs.append(cube_path)
        if a.verbose:
            from . import metrics as M
            print("  3D LUT fit:", M.compare(apply_lut3d(pairs[0].before, lut),
                                             pairs[0].after, pairs[0].weight), file=sys.stderr)

    _print_summary(model, diag, outputs)
    return 0


def _print_summary(model: PresetModel, diag: dict, outputs: list[str]) -> None:
    vm, bm = diag["verification_mean"], diag["baseline_mean"]
    d = diag["diagnostics"]
    print(f"\n{model.name}")
    print("=" * max(40, len(model.name)))
    print(f"pairs {diag['n_pairs']} · samples {diag['n_samples']:,} · "
          f"working space {diag.get('working_space')} · {diag['elapsed_sec']}s")
    sel = diag.get("working_space_selection")
    if sel:
        by = "  ".join(f"{k}={v}" for k, v in sel["dE_mean_by_space"].items())
        print(f"  working space chosen by fit: {sel['chosen']}  (ΔE by space: {by})")
        if sel["margin"] is not None and sel["margin"] < 0.15:
            print("  ! the two spaces fit almost equally well; this pair cannot "
                  "distinguish them")
    ws = diag.get("working_space_sensitivity", {})
    if ws:
        print(f"  applying these curves in '{ws['alternative']}' space instead "
              f"would shift the result by ΔE "
              f"{ws['dE_mean_if_applied_in_alternative_space']}")
    print("\nfit quality (extracted preset vs. the real 'after')")
    print(f"  ΔE2000 mean   {vm['dE_mean']:>7}   unedited {bm['dE_mean']:>7}")
    print(f"  ΔE2000 p95    {vm['dE_p95']:>7}   unedited {bm['dE_p95']:>7}")
    print(f"  RMSE /255     {vm['rmse255']:>7}   unedited {bm['rmse255']:>7}")
    print(f"  PSNR dB       {vm['psnr_db']:>7}   unedited {bm['psnr_db']:>7}")
    print("\ndiagnostics")
    print(f"  exposure      {d['exposure_ev']} EV")
    print(f"  curve         {d['curve_shape']} (mid slope {d['contrast_midslope']})")
    print(f"  black/white   {d['black_point_out']} / {d['white_point_out']}")
    print(f"  casts         shadows {d['cast_shadows']}  mid {d['cast_midtones']}"
          f"  highs {d['cast_highlights']}")
    nz = [(k, v) for k, v in model.hsl.items()
          if abs(v.hue) > 1 or abs(v.sat) > 1 or abs(v.lum) > 1]
    if nz:
        print("\ncolour mixer (H / S / L)")
        for k, v in nz:
            print(f"  {k:<9} {v.hue:>6.0f} {v.sat:>6.0f} {v.lum:>6.0f}")
    if model.has_grading():
        print("\ncolour grading")
        for k, z in model.grade.items():
            if abs(z.sat) > 0.5 or abs(z.lum) > 0.5:
                print(f"  {k:<10} hue {z.hue:>5.0f}°  sat {z.sat:>5.0f}  lum {z.lum:>5.0f}")
        print(f"  blending {model.grade_blending:.0f}  balance {model.grade_balance:.0f}")
    if diag.get("warnings"):
        print("\nwarnings")
        for w in diag["warnings"]:
            print("  ! " + w)
    print("\nwrote:")
    for o in outputs:
        print("  " + o)


def cmd_apply(a) -> int:
    model = PresetModel.from_json(a.preset)
    img = load_image(a.image)
    save_image(a.out, render(img, model))
    print(f"wrote {a.out}")
    return 0


def cmd_fetch(a) -> int:
    print("Note: only download images you are permitted to use, and respect the "
          "source site's terms of service.", file=sys.stderr)
    for u in a.url:
        print(fetch_url(u, a.outdir))
    return 0


def cmd_serve(a) -> int:
    from .webapp import create_app
    app = create_app()
    print(f"tcx web UI on http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, debug=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tcx",
        description="Reverse-engineer a Lightroom preset from before/after images.")
    p.add_argument("--version", action="version", version=f"tcx {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract a preset from image pair(s)")
    e.add_argument("--before"); e.add_argument("--after")
    e.add_argument("--pair", action="append",
                   help="'before,after', or one side-by-side image with --split")
    e.add_argument("--dir", help="directory of *_before.jpg / *_after.jpg pairs")
    e.add_argument("--split", choices=["lr", "rl", "tb", "bt"],
                   help="split a single composite sample image")
    e.add_argument("--split-gap", type=int, default=0)
    e.add_argument("-o", "--outdir", default="out")
    e.add_argument("--stem", help="output filename stem")
    e.add_argument("--name", default="Extracted Preset")
    e.add_argument("--group", default="tcx")
    e.add_argument("--color-mode", choices=["curves", "grading", "both"], default="curves",
                   help="curves: colour lives in the R/G/B curves (most faithful). "
                        "grading: express colour with the colour-grading wheels. "
                        "both: curves plus a residual grading fit.")
    e.add_argument("--saturation-mode", choices=["hsl", "basic"], default="hsl")
    e.add_argument("--working-space", choices=["auto", "melissa", "srgb"], default="auto",
                   help="space the edit is fitted and applied in. 'melissa' = "
                        "ProPhoto primaries + sRGB tone response, which is what "
                        "Lightroom's Develop module uses; 'srgb' takes the JPEGs at "
                        "face value; 'auto' (default) fits both and keeps whichever "
                        "explains the pair better.")
    e.add_argument("--iterations", type=int, default=3)
    e.add_argument("--max-samples", type=int, default=1_500_000)
    e.add_argument("--max-dim", type=int, default=1600)
    e.add_argument("--smooth", type=float, default=2.0, help="tone-curve smoothing strength")
    e.add_argument("--max-points", type=int, default=16, help="curve control points (LR max 16)")
    e.add_argument("--motion", choices=["translation", "euclidean", "affine"],
                   default="translation")
    e.add_argument("--blur", type=float, default=0.8, help="pre-sampling blur sigma")
    e.add_argument("--edge-percentile", type=float, default=60.0)
    e.add_argument("--no-align", action="store_true")
    e.add_argument("--no-hsl", action="store_true")
    e.add_argument("--no-hue", action="store_true")
    e.add_argument("--no-hsl-lum", action="store_true")
    e.add_argument("--grading-global", action="store_true")
    e.add_argument("--no-quantize", action="store_true",
                   help="keep the dense curve instead of the 16-point version")
    e.add_argument("--calibration", help="JSON file overriding slider-response constants")
    e.add_argument("--cube", action="store_true", help="also fit and export a 3D LUT")
    e.add_argument("--cube-size", type=int, default=33)
    e.add_argument("--preview", action="store_true", help="save the re-rendered before image")
    e.add_argument("--no-report", action="store_true")
    e.add_argument("-v", "--verbose", action="store_true")
    e.set_defaults(func=cmd_extract)

    ap = sub.add_parser("apply", help="apply an extracted preset (.json) to an image")
    ap.add_argument("preset"); ap.add_argument("image"); ap.add_argument("-o", "--out", required=True)
    ap.set_defaults(func=cmd_apply)

    f = sub.add_parser("fetch", help="download sample images you are permitted to use")
    f.add_argument("url", nargs="+"); f.add_argument("-o", "--outdir", default="samples")
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("serve", help="run the local web UI")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=7860)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
