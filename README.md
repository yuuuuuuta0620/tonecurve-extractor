# tonecurve-extractor (tcx)

**Before / After の画像ペアから Lightroom のプリセットを逆算するツール。**

プリセット販売ページのサンプル画像のように「同じ写真の編集前と編集後」が手に入るとき、
その 2 枚の差分からトーンカーブ・カラーミキサー (HSL)・カラーグレーディングを推定し、
Lightroom / Camera Raw にそのまま読み込める `.xmp` プリセットと 3D LUT (`.cube`) を書き出します。

推定したプリセットで before を**実際に再レンダリングして after と比較**するので、
「どれくらい忠実に復元できたか」が ΔE2000 / PSNR という数値で必ず付いてきます。

---

## できること

| 出力 | 内容 |
|---|---|
| `preset.xmp` | Lightroom Classic / Camera Raw の現像プリセット（トーンカーブ 4 本 + HSL 24 スライダー + カラーグレーディング） |
| `preset.cube` | 33³ の 3D LUT。パラメトリックなプリセットより忠実（`--cube`） |
| `preset.json` | 抽出したモデル本体。`tcx apply` で他の画像に適用できる |
| `report.html` / `.png` | カーブ・スライダー・検証画像・誤差マップを 1 枚にまとめたレポート |
| `diagnostics.json` | 収束履歴、バンドごとのデータ量、警告など全診断 |

---

## インストール

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

---

## 使い方

### 1 ペアから抽出

```bash
.venv/bin/python -m tcx extract --before before.jpg --after after.jpg --name "Moody Film" -o out
```

### 複数ペアをまとめて使う（精度が上がります）

サンプルが複数あるなら全部渡してください。1 枚の写真には写っていない色域・階調を
他の写真が埋めるので、カーブの外挿区間が減りスライダーの推定も安定します。

```bash
# 明示指定
.venv/bin/python -m tcx extract --pair b1.jpg,a1.jpg --pair b2.jpg,a2.jpg --pair b3.jpg,a3.jpg -o out

# ディレクトリから *_before.jpg / *_after.jpg を自動検出
.venv/bin/python -m tcx extract --dir samples/ -o out
```

### 左右に並んだ 1 枚の比較画像から

```bash
.venv/bin/python -m tcx extract --pair sample.jpg --split lr -o out    # lr / rl / tb / bt
```

### 3D LUT も出す・結果画像も保存する

```bash
.venv/bin/python -m tcx extract --before b.jpg --after a.jpg --cube --preview -o out
```

### 抽出したモデルを別の写真に適用して確かめる

```bash
.venv/bin/python -m tcx apply out/preset.json myphoto.jpg -o myphoto_graded.jpg
```

### ブラウザ UI

```bash
.venv/bin/python -m tcx serve
```

`http://127.0.0.1:7860` で画像をドロップ → レポート表示 → `.xmp` をダウンロード。

---

## Lightroom への読み込み

- **Lightroom Classic**: 現像モジュール → プリセットパネルの `＋` → 「プリセットを読み込み」→ `.xmp` を選択
- **手動で置く場合 (macOS)**: `~/Library/Application Support/Adobe/Lightroom/Develop Presets/`
- **Camera Raw**: `~/Library/Application Support/Adobe/CameraRaw/Settings/`
- **`.cube`**: Camera Raw の「プロファイル」→ LUT からプロファイル化、または動画編集ソフトで直接利用

---

## 仕組み

```
before ──▶ 位置合わせ ──▶ サンプリング ──┐
after  ──▶ (ECC)         (エッジ重み)    │
                                          ▼
     ┌────────── 反復座標降下 (既定 3〜4 回) ──────────┐
     │  ① 後段を逆変換して after からはがす            │
     │  ② R/G/B ごとの伝達関数を条件付き中央値で推定    │
     │     → 単調化 (PAVA) → 平滑化 (Whittaker)        │
     │  ③ カーブ適用後の残差から HSL 8 バンドを         │
     │     ロバスト連立最小二乗 (Huber IRLS) で推定     │
     │  ④ 必要ならカラーグレーディングを非線形最小二乗   │
     └──────────────────────────────────────────────┘
                                          ▼
        16 点の制御点に量子化 ──▶ .xmp / .cube / レポート
                                          ▼
                          before を再レンダリング → ΔE2000 で検証
```

要点となる設計判断：

- **条件付き中央値**：入力値ごとの出力値の重み付き中央値を取ります。平均ではなく中央値なので、
  部分補正（ブラシ・マスク・段階フィルター）が画素の 50% 未満なら推定は引きずられません。
- **単調性の強制**：重み付き PAVA（isotonic 回帰）でトーンカーブが必ず単調増加になります。
- **端の線形外挿**：写真に存在しない入力レベルは平坦に延ばすのではなく局所の傾きで外挿し、
  「実測できた範囲」を警告として報告します。
- **段ごとに固有の領域でフィット**：HSL やグレーディングを**逆変換して after からはがしてから**
  カーブを再推定します。これをやらないとカーブが HSL の仕事を吸収してしまい、
  スライダーが系統的に過小評価されます（開発中の実測では、真値 +18 の Blue 彩度が
  この処理なしでは +7、ありでは +14 と推定されました）。
- **バンド混合を考慮**：ある画素は隣り合う 2 つの色相バンドの影響を同時に受けます。
  3 つのスライダーはいずれもバンド重みについて線形なので、8 バンドを一括で解きます。
  バンド中心付近の画素だけで中央値を取ると必ず過小評価になります。
- **エッジの除外**：わずかな位置ずれは輪郭で最も誤差を生むため、勾配の大きい画素を滑らかに減衰させます。

---

## 精度（合成データによる実測）

既知のプリセットを合成写真に適用して after を作り、それを復元できるかを測ったものです
（`tests/eval_synthetic.py`）。

| 入力の品質 | ΔE2000 平均 | ΔE2000 p95 | 伝達関数の誤差 (0–255) | スライダー誤差 |
|---|---|---|---|---|
| 可逆（アルゴリズムの限界） | **0.36** | 0.72 | 1.05 | 1.9 |
| PNG 8bit | 0.46 | 0.97 | 1.13 | 2.3 |
| JPEG q92 4:4:4 | 1.03 | 2.26 | 1.93 | 3.2 |
| JPEG q85 4:2:0（Web のサンプル画像相当） | 1.54 | 3.58 | 1.92 | 4.0 |

未編集の状態（before と after の差そのもの）は ΔE2000 平均 11.3。
**ΔE2000 は 1 前後が「並べてやっと分かる」水準**なので、Web からダウンロードした
JPEG サンプルでも実用的な精度で復元できます。誤差の主因は圧縮、特に JPEG の
クロマサブサンプリングです（可逆入力なら ΔE 0.36 まで下がります）。

3D LUT (`--cube`) はさらに一段忠実です（同条件・1 ペアで ΔE 1.38 対 1.56）。

この数値はあくまで**合成データでの上限性能**です。実際の販売用サンプル画像は、
部分補正やレンズ補正が混ざっていたり、before/after が別現像だったりするので、
自分のペアでの実測値は必ずレポートで確認してください。

---

## 何が復元でき、何ができないか

**復元できる**（画像全体に一様にかかる編集）

- トーンカーブ（RGB マスター + R/G/B 個別）、露出・コントラスト・白黒レベルの効果
- ホワイトバランスの色かぶり（チャンネル別カーブとして）
- カラーミキサー（HSL）8 バンド × 色相／彩度／輝度
- カラーグレーディング（シャドウ／中間調／ハイライト／全体）
- 自然な彩度・彩度（`--saturation-mode basic`）

**復元できない**（原理的に無理・ペアの差分に一様な形で現れない）

- 部分補正：ブラシ、円形／段階フィルター、被写体・空マスク
- ディテール系：シャープ、ノイズ除去、明瞭度、かすみの除去、テクスチャ
- レンズ補正、変形、切り抜き、粒子、周辺光量
- カメラプロファイル（Adobe Color / カメラマッチング等）の違い
- raw 現像特有の処理（ハイライト復元など）。サンプル画像は書き出し済み JPEG なので、
  Basic パネルの絶対値（露出 +0.75 のような数値）は raw に対して意味が変わります。
  そのため本ツールは**階調を全部トーンカーブに載せ**、露出やかぶりは
  「診断値」として表示するだけにしています。

**近似であること**：Adobe はスライダーの応答関数を公開していないため、
HSL とカラーグレーディングの解釈は本ツールが定義した近似モデルです
（`tcx/model.py` の `Calibration`）。フィットと検証は同じモデルで一貫しているので
レポートの数値は正しいですが、Lightroom 側で見たときの一致度はこの定数の近さに依存します。
`--calibration cal.json` で調整できます。**トーンカーブは近似ではなく実測値**なので、
最も忠実に転写されるのはカーブ部分です。

---

## 主なオプション

| オプション | 既定 | 説明 |
|---|---|---|
| `--color-mode {curves,grading,both}` | `curves` | 色をどこで表現するか。`curves` が最も忠実。`grading` は編集しやすいホイールに落とすが誤差は増える |
| `--saturation-mode {hsl,basic}` | `hsl` | 全体彩度を HSL バンドに畳むか、Basic パネルの彩度／自然な彩度として出すか |
| `--iterations N` | 3 | 座標降下の反復回数。4〜5 で概ね収束 |
| `--smooth F` | 2.0 | トーンカーブの平滑化強度。ノイズの多い素材では上げる |
| `--max-points N` | 16 | カーブの制御点数（Lightroom の上限が 16） |
| `--max-dim N` | 1600 | 内部処理の最大辺。大きくすると精度がわずかに上がり遅くなる |
| `--motion {translation,euclidean,affine}` | `translation` | 位置合わせのモデル |
| `--no-align` | off | 完全に同一構図と分かっている場合 |
| `--edge-percentile P` | 60 | エッジ画素の減衰しきい値 |
| `--cube` / `--cube-size N` | off / 33 | 3D LUT の書き出し |
| `--preview` | off | 再レンダリング結果を JPEG で保存 |
| `--no-quantize` | off | 16 点に量子化せず密なカーブのまま検証する（アルゴリズム評価用） |

---

## 開発

```bash
.venv/bin/python -m pytest tests -q          # 22 テスト、約 10 秒
.venv/bin/python examples/make_synthetic.py  # 既知プリセットで before/after を生成
.venv/bin/python tests/eval_synthetic.py     # 圧縮条件別の精度ベンチマーク
```

```
tcx/
  colorspace.py  sRGB / HSL / Lab / CIEDE2000
  curves.py      条件付き中央値・PAVA・Whittaker・制御点フィット
  align.py       ECC 位置合わせとサンプル重み
  render.py      順方向レンダラと各段の逆変換（＝プリセットの定義）
  hsl.py         8 バンド HSL のロバスト連立フィット
  grading.py     カラーグレーディングの非線形フィット
  extract.py     反復座標降下の統括
  lut3d.py       正則化付き 3D LUT 推定と .cube 出力
  xmp.py         Lightroom プリセット書き出し
  report.py      図とレポート
  webapp.py      Flask の Web UI
```

---

## 注意

サンプル画像のダウンロードや、市販プリセットの復元結果の扱いは、
**取得元サイトの利用規約と著作権に従ってください。** 個人的な学習・解析の範囲を超えて
復元したプリセットを再配布・販売することは想定していません。
`tcx fetch` はユーザーが明示した URL のみを取得します。
