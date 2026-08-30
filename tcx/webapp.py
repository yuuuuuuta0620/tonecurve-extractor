"""Minimal local web UI: drop in before/after images, get a preset back."""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import uuid

import numpy as np
from flask import Flask, Response, abort, render_template_string, request, send_file

from .align import align_pair, sample_pixels
from .extract import ExtractOptions, extract
from .imageio_utils import load_image, split_pair
from .model import Calibration
from .report import make_figure
from .xmp import build_xmp

PAGE = """<!doctype html><meta charset="utf-8"><title>tonecurve-extractor</title>
<style>
 :root{--bg:#fbfbfc;--fg:#1c1c1e;--mut:#6b6b70;--line:#e2e2e6;--acc:#3057d6}
 @media(prefers-color-scheme:dark){:root{--bg:#151517;--fg:#ececee;--mut:#9a9aa0;--line:#2c2c30;--acc:#7fa0ff}}
 *{box-sizing:border-box}
 body{font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Hiragino Sans",sans-serif;
      margin:0;padding:36px 24px;background:var(--bg);color:var(--fg)}
 .wrap{max-width:1120px;margin:0 auto}
 h1{font-size:24px;margin:0 0 4px} p.lead{color:var(--mut);margin:0 0 28px}
 form{border:1px solid var(--line);border-radius:12px;padding:22px;background:rgba(127,127,127,.04)}
 fieldset{border:0;padding:0;margin:0 0 18px}
 legend{font-weight:600;font-size:13px;margin-bottom:8px}
 .row{display:flex;gap:18px;flex-wrap:wrap}
 label{display:block;font-size:12.5px;color:var(--mut);margin-bottom:4px}
 input[type=file]{font-size:13px}
 select,input[type=number],input[type=text]{padding:6px 8px;border:1px solid var(--line);
   border-radius:7px;background:var(--bg);color:var(--fg);font-size:13px}
 button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:10px 22px;
   font-size:14px;font-weight:600;cursor:pointer}
 .hint{color:var(--mut);font-size:12px;margin-top:6px}
 img{max-width:100%;border:1px solid var(--line);border-radius:10px;margin-top:20px;background:#fff}
 pre{background:rgba(127,127,127,.09);border:1px solid var(--line);border-radius:10px;
     padding:12px;overflow:auto;font-size:11.5px;max-height:340px}
 .dl a{display:inline-block;margin-right:10px;margin-top:12px;padding:8px 16px;border:1px solid var(--line);
    border-radius:8px;text-decoration:none;color:var(--fg);font-size:13px}
 table{border-collapse:collapse;font-size:13px;margin-top:14px}
 td,th{border:1px solid var(--line);padding:5px 12px;text-align:right}
 th:first-child,td:first-child{text-align:left}
 .err{border:1px solid #d33;background:#d3333312;border-radius:10px;padding:14px;margin-top:20px}
</style>
<div class="wrap">
<h1>tonecurve-extractor</h1>
<p class="lead">Before / After の画像ペアから Lightroom プリセット（トーンカーブ・カラーミキサー・カラーグレーディング）を復元します。</p>
<form method="post" enctype="multipart/form-data" action="/extract">
 <fieldset><legend>画像</legend>
  <div class="row">
   <div><label>Before（複数可）</label><input type="file" name="before" accept="image/*" multiple required></div>
   <div><label>After（複数可・Before と同じ順）</label><input type="file" name="after" accept="image/*" multiple></div>
  </div>
  <div class="hint">After を空にして Before に 1 枚だけ入れると、左右／上下に並んだ合成サンプル画像として分割します。</div>
 </fieldset>
 <fieldset><legend>設定</legend>
  <div class="row">
   <div><label>プリセット名</label><input type="text" name="name" value="Extracted Preset"></div>
   <div><label>色の表現</label><select name="color_mode">
     <option value="curves">curves — RGB カーブに色を載せる（最も忠実）</option>
     <option value="grading">grading — カラーグレーディングホイールで表現</option>
     <option value="both">both — カーブ＋残差をグレーディングに</option></select></div>
   <div><label>合成画像の分割</label><select name="split">
     <option value="">しない</option><option value="lr">左=Before 右=After</option>
     <option value="rl">左=After 右=Before</option><option value="tb">上=Before 下=After</option>
     <option value="bt">上=After 下=Before</option></select></div>
   <div><label>反復回数</label><input type="number" name="iterations" value="4" min="1" max="8" style="width:72px"></div>
   <div><label>カーブ平滑化</label><input type="number" name="smooth" value="2.0" step="0.1" min="0" style="width:82px"></div>
   <div><label>3D LUT</label><select name="cube"><option value="">出力しない</option>
     <option value="1">.cube も出力</option></select></div>
  </div>
 </fieldset>
 <button type="submit">プリセットを抽出</button>
</form>
{% if error %}<div class="err"><b>エラー:</b> {{ error }}</div>{% endif %}
{% if result %}
 <img src="data:image/png;base64,{{ result.png }}">
 <table>
  <tr><th>指標</th><th>抽出プリセット</th><th>未編集（差分の大きさ）</th></tr>
  <tr><td>ΔE2000 平均</td><td>{{ result.v.dE_mean }}</td><td>{{ result.b.dE_mean }}</td></tr>
  <tr><td>ΔE2000 p95</td><td>{{ result.v.dE_p95 }}</td><td>{{ result.b.dE_p95 }}</td></tr>
  <tr><td>RMSE /255</td><td>{{ result.v.rmse255 }}</td><td>{{ result.b.rmse255 }}</td></tr>
  <tr><td>PSNR dB</td><td>{{ result.v.psnr_db }}</td><td>{{ result.b.psnr_db }}</td></tr>
 </table>
 {% if result.warnings %}<div class="err">{% for w in result.warnings %}<div>⚠ {{ w }}</div>{% endfor %}</div>{% endif %}
 <div class="dl">
  <a href="/download/{{ result.token }}/preset.xmp">⬇ Lightroom プリセット (.xmp)</a>
  <a href="/download/{{ result.token }}/preset.json">⬇ モデル (.json)</a>
  {% if result.has_cube %}<a href="/download/{{ result.token }}/preset.cube">⬇ 3D LUT (.cube)</a>{% endif %}
  <a href="/download/{{ result.token }}/report.png">⬇ レポート画像</a>
 </div>
 <h3 style="font-size:15px;margin:26px 0 8px">.xmp の中身</h3>
 <pre>{{ result.xmp }}</pre>
{% endif %}
<p class="hint" style="margin-top:34px">Lightroom Classic への導入: 現像モジュール → プリセット → ＋ → プリセットを読み込み、で .xmp を選択。
（手動で置く場合 macOS: <code>~/Library/Application Support/Adobe/Lightroom/Develop Presets/</code>）</p>
</div>
"""

_STORE: dict[str, str] = {}


def create_app(workdir: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
    root = workdir or os.path.join(tempfile.gettempdir(), "tcx-web")
    os.makedirs(root, exist_ok=True)

    @app.get("/")
    def index():
        return render_template_string(PAGE, result=None, error=None)

    @app.post("/extract")
    def do_extract():
        try:
            token = uuid.uuid4().hex[:16]
            d = os.path.join(root, token)
            os.makedirs(d, exist_ok=True)

            befores = [f for f in request.files.getlist("before") if f.filename]
            afters = [f for f in request.files.getlist("after") if f.filename]
            split = request.form.get("split") or None
            if not befores:
                raise ValueError("Before 画像を選んでください")

            def _read(fs):
                p = os.path.join(d, "in_" + uuid.uuid4().hex[:8] + os.path.splitext(fs.filename)[1])
                fs.save(p)
                return load_image(p)

            pairs = []
            if afters:
                if len(afters) != len(befores):
                    raise ValueError("Before と After の枚数を揃えてください")
                for fb, fa in zip(befores, afters):
                    pairs.append((_read(fb), _read(fa)))
            else:
                if not split:
                    raise ValueError("After が無い場合は「合成画像の分割」を選んでください")
                for fb in befores:
                    pairs.append(split_pair(_read(fb), split))

            aligned = [align_pair(b, a) for b, a in pairs]
            opts = ExtractOptions(
                color_mode=request.form.get("color_mode", "curves"),
                iterations=int(request.form.get("iterations", 4)),
                smooth=float(request.form.get("smooth", 2.0)),
                name=request.form.get("name") or "Extracted Preset")
            model, diag = extract(aligned, opts)

            xmp = build_xmp(model)
            open(os.path.join(d, "preset.xmp"), "w", encoding="utf-8").write(xmp)
            model.to_json(os.path.join(d, "preset.json"))
            png = make_figure(model, diag, aligned)
            open(os.path.join(d, "report.png"), "wb").write(png)

            has_cube = False
            if request.form.get("cube"):
                from .lut3d import fit_lut3d, write_cube
                B, A, W = sample_pixels(aligned, 400_000)
                write_cube(os.path.join(d, "preset.cube"),
                           fit_lut3d(B, A, W, model=model, size=33), title=model.name)
                has_cube = True

            _STORE[token] = d
            return render_template_string(
                PAGE, error=None,
                result={"png": base64.b64encode(png).decode(), "token": token,
                        "v": diag["verification_mean"], "b": diag["baseline_mean"],
                        "xmp": xmp, "has_cube": has_cube,
                        "warnings": diag.get("warnings", [])})
        except Exception as e:  # surfaced to the user rather than a 500 page
            return render_template_string(PAGE, result=None, error=f"{type(e).__name__}: {e}")

    @app.get("/download/<token>/<name>")
    def download(token, name):
        d = _STORE.get(token)
        if not d or "/" in name or "\\" in name or name.startswith("."):
            abort(404)
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            abort(404)
        return send_file(p, as_attachment=True, download_name=name)

    return app
