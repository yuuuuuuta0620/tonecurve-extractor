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
    sel = diag.get("working_space_selection") or {}
    ws_line = f"working space     : {diag.get('working_space')}"
    if sel:
        by = "  ".join(f"{k} {v}" for k, v in sel["dE_mean_by_space"].items())
        ws_line += f"   (chosen by fit — ΔE {by})"
    lines = [
        f"pairs used        : {diag.get('n_pairs')}   samples: {diag.get('n_samples'):,}",
        ws_line,
        f"measured range    : {diag.get('input_support')}",
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
        before_vis = _thumb(p.before)
        title_b = "before"
        if getattr(p, "outlier_mask", None) is not None and p.outlier_mask.any():
            import cv2
            m = cv2.resize(p.outlier_mask.astype(np.float32),
                           (before_vis.shape[1], before_vis.shape[0]),
                           interpolation=cv2.INTER_AREA)[..., None]
            before_vis = np.clip(before_vis * (1 - 0.75 * m)
                                 + np.array([1.0, 0.15, 0.25]) * 0.75 * m, 0, 1)
            title_b = f"before — red = excluded ({p.outlier_mask.mean():.1%})"
        panels = [(before_vis, title_b),
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


def _chip(rgb) -> str:
    c = "#%02x%02x%02x" % tuple(int(max(0, min(1, v)) * 255 + 0.5) for v in rgb)
    return (f'<span style="display:inline-block;width:30px;height:16px;border-radius:3px;'
            f'border:1px solid #0003;background:{c};vertical-align:-3px"></span>')


def _img_tag(arr, height=132) -> str:
    import io as _io
    from PIL import Image
    a = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    buf = _io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return (f'<img style="height:{height}px;width:auto;border-radius:4px;margin:0" '
            f'src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">')


def patches_html(patches: list[dict]) -> str:
    if not patches:
        return ""
    cards = []
    for i, c in enumerate(patches, 1):
        hue = (f"色相 {c['hue_shift_deg']:+.0f}°" if c["hue_shift_deg"] is not None else "色相 —")
        sat = (f"彩度 {c['saturation_pct']:+.0f}%" if c["saturation_pct"] is not None
               else "彩度 —")
        cards.append(f"""
<div style="display:inline-block;vertical-align:top;margin:0 18px 20px 0;max-width:330px">
 <div style="font-size:12.5px;color:#555;margin-bottom:4px">
  {i}. {c['band']} — ΔE {c['dE']}</div>
 <div>{_img_tag(c['crop_tone'])} {_img_tag(c['crop_after'])}</div>
 <div style="font-size:11.5px;color:#777;margin:2px 0 4px">
  左：トーンカーブのみ　右：目標</div>
 <div style="font-size:12.5px">
  {_chip(c['before'])} → {_chip(c['after'])}
  &nbsp; {hue} / {sat} / 明度 {c['lightness_pct']:+.0f}%</div>
</div>""")
    return f"""
<h2>特徴的な色の変化（ポイントカラーの狙い目）</h2>
<p class="sub">左の色を Lightroom のポイントカラーで拾い、右の色になるまで動かしてください。
トーンカーブ適用後の状態を基準にしているので、先にカーブを当ててから作業します。</p>
{''.join(cards)}
"""


def look_html(d: dict) -> str:
    if not d or not d.get("traits"):
        return ""
    rows = "".join(f"<tr><td>{html.escape(t['ja'])}</td>"
                   f"<td style='color:#777;font-size:12px'>{html.escape(t['evidence'])}</td></tr>"
                   for t in d["traits"])
    return f"""
<h2>このプリセットの性格</h2>
<p class="sub" style="font-size:15px;color:#1c1c1e">{html.escape(d['summary_ja'])}</p>
<table><tr><th>特徴</th><th>根拠となった実測値</th></tr>{rows}</table>
"""


def guide_html(g: dict) -> str:
    if not g:
        return ""
    rows = []
    for z in g["zones"]:
        if not z.get("measured"):
            rows.append(f"<tr><td>{html.escape(z['zone'])}</td>"
                        f"<td colspan=3>{html.escape(z['note'])}</td></tr>")
            continue
        sw = cs.hsl_to_rgb(np.array(float(z["tint_hue_deg"])), np.array(0.75), np.array(0.5))
        rows.append(
            f"<tr><td>{z['zone']}</td><td>{_chip(sw)} {z['tint_name']} "
            f"({z['tint_hue_deg']:.0f}°)</td><td>{z['tint_strength_dE']:.1f}</td>"
            f"<td>{z['lightness_dL']:+.1f}</td></tr>")

    brows = []
    for b in g["ranked_bands"]:
        trust = "" if b.get("fitted_slider_trustworthy", True) else " ⚠"
        brows.append(
            f"<tr><td>{b['band']}{trust}</td><td>{b['hue_shift_deg']:+.1f}°</td>"
            f"<td>{b['saturation_pct']:+.1f}%</td><td>{b['lightness_pct']:+.1f}%</td>"
            f"<td>{b['data_share']:.0%}</td></tr>")

    mrows = []
    for m in g["memory_colours"]:
        if m.get("measurable") is False:
            mrows.append(f"<tr><td>{html.escape(m['colour'])}</td>"
                         f"<td colspan=5 style='text-align:left;color:#888'>"
                         f"この写真には十分に含まれていないため測定不能</td></tr>")
            continue
        pct = f" ({m['chroma_pct']:+.0f}%)" if m.get("chroma_pct") is not None else ""
        hue = f"{m['hue_shift_deg']:+.0f}°" if m.get("hue_shift_deg") is not None else "—"
        note = "" if m["source"].startswith("measured") else " <i>(予測)</i>"
        mrows.append(
            f"<tr><td>{html.escape(m['colour'])}{note}</td>"
            f"<td>{_chip(m['before'])} → {_chip(m['after'])}</td>"
            f"<td>{m['dE']:.2f}</td><td>{m['dL']:+.1f}</td>"
            f"<td>{m['dC']:+.1f}{pct}</td><td>{hue}</td></tr>")

    return f"""
<h2>手で再現するためのカラーガイド</h2>
<p class="sub">トーンカーブを当てた時点で、色の差が ΔE
{g['total_colour_dE_after_tone']} 残っています。以下はすべて画素からの実測値で、
スライダーの当てはめ結果を読み返したものではありません。</p>
<h3>階調ゾーン（カラーグレーディングのホイールに対応）</h3>
<table><tr><th>ゾーン</th><th>色の向き</th><th>強さ ΔE</th><th>明度 ΔL*</th></tr>
{''.join(rows)}</table>
<h3>色相バンド（カラーミキサー・影響の大きい順）</h3>
<table><tr><th>バンド</th><th>色相</th><th>彩度</th><th>明度</th><th>画素比</th></tr>
{''.join(brows)}</table>
<h3>気になる色がどう動くか（トーンカーブ適用後 → 目標）</h3>
<table><tr><th>色</th><th>変化</th><th>ΔE</th><th>ΔL*</th><th>Δ彩度</th><th>色相</th></tr>
{''.join(mrows)}</table>
"""


def _dumpable(diag: dict) -> dict:
    """The diagnostics minus the patch crops, which are pictures not data."""
    from .guide import patches_json
    out = dict(diag)
    if out.get("patches"):
        out["patches"] = patches_json(out["patches"])
    return out


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
 table{{border-collapse:collapse;font-size:13px;margin:6px 0 18px}}
 td,th{{border:1px solid #e3e3e6;padding:5px 12px;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 h3{{font-size:13.5px;margin:18px 0 6px;color:#555}}
 pre{{background:#f4f4f6;border:1px solid #e3e3e6;border-radius:8px;padding:12px;
     overflow:auto;font-size:11.5px;max-height:420px}}
 .sub{{color:#666;margin:0 0 20px}}
</style>
<h1>{html.escape(model.name)}</h1>
<p class="sub">extracted by tonecurve-extractor · {diag.get('n_pairs')} pair(s) ·
{diag.get('n_samples', 0):,} samples · {diag.get('elapsed_sec')}s</p>
<img src="data:image/png;base64,{b64}">
{look_html(diag.get("look_description"))}
{patches_html(diag.get("patches") or [])}
{guide_html(diag.get("colour_guide"))}
<h2>Lightroom preset (.xmp)</h2>
<pre>{html.escape(xmp)}</pre>
<h2>Full diagnostics</h2>
<pre>{html.escape(json.dumps(_dumpable(diag), indent=1, ensure_ascii=False, default=float))}</pre>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
