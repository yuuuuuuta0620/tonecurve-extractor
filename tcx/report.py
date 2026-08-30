"""Visual report: curve plots, slider charts, and a rendered verification strip."""
from __future__ import annotations

import base64
import html
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import colorspace as cs
from . import curves as C
from .model import PresetModel, BAND_NAMES, BAND_CENTERS
from .render import render

BAND_COLORS = [tuple(cs.hsl_to_rgb(np.array(h), np.array(0.85), np.array(0.5)))
               for h in BAND_CENTERS]


def _thumb(img: np.ndarray, max_dim: int = 460) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 1)


def make_figure(model: PresetModel, diag: dict, pairs) -> bytes:
    n_img_rows = 1 if pairs else 0
    if pairs:
        h, w = pairs[0].before.shape[:2]
        img_row_h = 1.05 * (h / w) * 1.35
    else:
        img_row_h = 0.0
    fig = plt.figure(figsize=(15, 5.0 + 3.6 * n_img_rows), dpi=110)
    gs = fig.add_gridspec(1 + n_img_rows, 3,
                          height_ratios=[1] + ([max(0.5, img_row_h)] if n_img_rows else []),
                          hspace=0.22, wspace=0.22)

    # ---- tone curves ----
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 255, len(model.master))
    f = [C.compose_lut(ch, model.master) for ch in (model.red, model.green, model.blue)]
    ax.plot([0, 255], [0, 255], color="0.8", lw=1, ls="--")
    ax.plot(x, model.master * 255, color="0.15", lw=2.2, label="RGB (master)")
    for lut, col, name in zip(f, ["#d33", "#2a2", "#36c"], "RGB"):
        ax.plot(x, lut * 255, color=col, lw=1.3, alpha=0.9, label=f"{name} (absolute)")
    pts = (model.meta.get("control_points") or {}).get("master")
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=18, color="0.15", zorder=5)
    ax.set_xlim(0, 255); ax.set_ylim(0, 255)
    ax.set_title("Tone curves", fontsize=11)
    ax.set_xlabel("input"); ax.set_ylabel("output")
    ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.25)

    # ---- colour mixer ----
    ax = fig.add_subplot(gs[0, 1])
    idx = np.arange(len(BAND_NAMES))
    hue = [model.hsl[n].hue for n in BAND_NAMES]
    sat = [model.hsl[n].sat for n in BAND_NAMES]
    lum = [model.hsl[n].lum for n in BAND_NAMES]
    w = 0.26
    ax.bar(idx - w, hue, w, color=BAND_COLORS, edgecolor="0.25", alpha=0.42)
    ax.bar(idx, sat, w, color=BAND_COLORS, edgecolor="0.25")
    ax.bar(idx + w, lum, w, color=BAND_COLORS, edgecolor="0.25", alpha=0.85, hatch="///")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(idx); ax.set_xticklabels(BAND_NAMES, rotation=45, ha="right", fontsize=8)
    lim = max(20, 1.15 * max([abs(v) for v in hue + sat + lum] + [1]))
    ax.set_ylim(-lim, lim)
    ax.set_title("Colour mixer (HSL) — bar colour = hue band", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.55", edgecolor="0.25", alpha=0.42, label="Hue"),
                       Patch(facecolor="0.55", edgecolor="0.25", label="Saturation"),
                       Patch(facecolor="0.55", edgecolor="0.25", alpha=0.85, hatch="///",
                             label="Luminance")], fontsize=7.5, ncol=3, loc="upper right")
    ax.grid(alpha=0.25, axis="y")

    # ---- metrics / grading ----
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    d = diag.get("diagnostics", {})
    vm = diag.get("verification_mean", {})
    bm = diag.get("baseline_mean", {})
    lines = [
        f"pairs used        : {diag.get('n_pairs')}   samples: {diag.get('n_samples'):,}",
        f"curve coverage    : {diag.get('curve_coverage')}",
        "",
        "── fit quality (rendered vs. real after) ──",
        f"  ΔE2000 mean     : {vm.get('dE_mean')}   (unedited: {bm.get('dE_mean')})",
        f"  ΔE2000 p95      : {vm.get('dE_p95')}   (unedited: {bm.get('dE_p95')})",
        f"  RMSE (0-255)    : {vm.get('rmse255')}   (unedited: {bm.get('rmse255')})",
        f"  PSNR            : {vm.get('psnr_db')} dB (unedited: {bm.get('psnr_db')} dB)",
        "",
        "── diagnostics (not written to the preset) ──",
        f"  exposure        : {d.get('exposure_ev')} EV",
        f"  curve shape     : {d.get('curve_shape')}  (mid slope {d.get('contrast_midslope')})",
        f"  black / white   : {d.get('black_point_out')} / {d.get('white_point_out')}",
        f"  cast shadows    : {d.get('cast_shadows')}",
        f"  cast midtones   : {d.get('cast_midtones')}",
        f"  cast highlights : {d.get('cast_highlights')}",
    ]
    if model.has_grading():
        lines += ["", "── colour grading ──"]
        for k, z in model.grade.items():
            if abs(z.sat) > 0.5 or abs(z.lum) > 0.5:
                lines.append(f"  {k:<10}: hue {z.hue:.0f}°  sat {z.sat:.0f}  lum {z.lum:.0f}")
        lines.append(f"  blending {model.grade_blending:.0f}  balance {model.grade_balance:.0f}")
    if abs(model.saturation) > 0.5 or abs(model.vibrance) > 0.5:
        lines += ["", f"  Saturation {model.saturation:.0f}   Vibrance {model.vibrance:.0f}"]
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.6)

    # ---- verification strip ----
    if pairs:
        p = pairs[0]
        pred = render(p.before, model)
        err = cs.delta_e2000(cs.rgb_to_lab(pred), cs.rgb_to_lab(p.after))
        panels = [(_thumb(p.before), "before"),
                  (_thumb(p.after), "after (reference)"),
                  (_thumb(pred), "before + extracted preset")]
        sub = gs[1, :].subgridspec(1, 4, wspace=0.06)
        for i, (im, title) in enumerate(panels):
            a = fig.add_subplot(sub[0, i]); a.imshow(im); a.set_title(title, fontsize=9); a.axis("off")
        a = fig.add_subplot(sub[0, 3])
        im = a.imshow(_thumb(np.repeat(err[..., None], 3, axis=2))[..., 0],
                      cmap="magma", vmin=0, vmax=6)
        a.set_title("ΔE2000 error", fontsize=9); a.axis("off")
        cb = fig.colorbar(im, ax=a, fraction=0.046)
        cb.set_label("ΔE2000  (≈1 = just noticeable)", fontsize=7.5)
        cb.ax.tick_params(labelsize=7.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def write_html(path: str, model: PresetModel, diag: dict, png: bytes, xmp: str) -> None:
    b64 = base64.b64encode(png).decode()
    body = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(model.name)} — tcx report</title>
<style>
 body{{font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:32px;
      background:#fbfbfc;color:#1c1c1e;max-width:1180px}}
 h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:15px;margin:28px 0 8px;color:#444}}
 img{{max-width:100%;border:1px solid #e3e3e6;border-radius:8px;background:#fff}}
 pre{{background:#f4f4f6;border:1px solid #e3e3e6;border-radius:8px;padding:12px;
     overflow:auto;font-size:11.5px;max-height:420px}}
 .sub{{color:#666;margin:0 0 20px}}
</style>
<h1>{html.escape(model.name)}</h1>
<p class="sub">extracted by tonecurve-extractor · {diag.get('n_pairs')} pair(s) ·
{diag.get('n_samples', 0):,} samples · {diag.get('elapsed_sec')}s</p>
<img src="data:image/png;base64,{b64}">
<h2>Lightroom preset (.xmp)</h2>
<pre>{html.escape(xmp)}</pre>
<h2>Full diagnostics</h2>
<pre>{html.escape(json.dumps(diag, indent=1, ensure_ascii=False))}</pre>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
