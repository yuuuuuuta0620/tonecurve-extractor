"""A reference chart of the colours that actually change.

Not a HALD identity CLUT: those cover the whole cube uniformly, most of which
never occurs in the photographs.  This samples the colours the pictures really
contain, keeps the ones the edit moves, and lays them out the way the Colour
Mixer is organised -- hue bands across, lightness down -- so a swatch on the
chart corresponds to a slider you can reach for.

Three files come out with identical geometry:

  *_chart_before.png   the source colours, with nothing applied.  Import this
                       into Lightroom, apply your preset-in-progress, and it
                       should turn into...
  *_chart_after.png    ...this, the same swatches as they appear in the edited
                       samples.  The target to match by eye.
  *_chart_compare.png  both halves per swatch with the measured deltas, for
                       reading rather than for Lightroom.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import colorspace as cs
from .model import BAND_NAMES
from .render import band_weights
from .robust import weighted_median

#: lightness rows, from the bottom of the range to the top
ROWS = [("very dark", 0.06, 0.20), ("dark", 0.20, 0.36), ("mid", 0.36, 0.55),
        ("light", 0.55, 0.74), ("very light", 0.74, 0.92)]

MIN_TILE_WEIGHT = 60.0
NEUTRAL_SAT = 0.09


def _font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_chart(before: np.ndarray, after: np.ndarray, weight: np.ndarray,
                model=None) -> dict:
    """Median before / tone-corrected / after colour for each cell.

    The deltas are measured from the tone-corrected colour, because that is
    where you actually start: the curve is already decided, and what is left
    is the colour work.
    """
    tone = None
    if model is not None:
        from .guide import _tone_only
        tone = _tone_only(before, model)
    h1, s1, l1 = cs.rgb_to_hsl(before)
    W = band_weights(h1)
    coloured = s1 >= NEUTRAL_SAT

    columns = ["neutral"] + BAND_NAMES
    cells: dict[tuple[int, int], dict] = {}
    for ri, (_, lo, hi) in enumerate(ROWS):
        band_row = (l1 >= lo) & (l1 < hi) & (weight > 0)
        for ci, name in enumerate(columns):
            if name == "neutral":
                w = weight * band_row * ~coloured
            else:
                w = weight * band_row * coloured * W[:, ci - 1]
            if w.sum() < MIN_TILE_WEIGHT:
                continue
            src = np.array([weighted_median(before[:, c], w) for c in range(3)])
            dst = np.array([weighted_median(after[:, c], w) for c in range(3)])
            mid = (np.array([weighted_median(tone[:, c], w) for c in range(3)])
                   if tone is not None else src)
            hb, sb, lb = cs.rgb_to_hsl(mid)
            ha, sa, la = cs.rgb_to_hsl(dst)
            cells[(ri, ci)] = {
                "before": src, "tone": mid, "after": dst, "weight": float(w.sum()),
                "dE": float(cs.delta_e2000(cs.rgb_to_lab(mid), cs.rgb_to_lab(dst))),
                "hue": float(cs.wrap_deg(ha - hb)) if min(sb, sa) > 0.04 else None,
                "sat": (100.0 * (float(sa) / float(sb) - 1.0)) if sb > 0.04 else None,
                "lum": 100.0 * (float(la) / max(float(lb), 1e-6) - 1.0),
            }

    used_rows = sorted({ri for ri, _ in cells})       # drop rows with no data
    remap = {old: new for new, old in enumerate(used_rows)}
    cells = {(remap[ri], ci): c for (ri, ci), c in cells.items()}
    return {"columns": columns, "rows": [ROWS[i][0] for i in used_rows], "cells": cells}


def _paint(draw, box, rgb):
    draw.rectangle(box, fill=tuple(int(np.clip(v, 0, 1) * 255 + 0.5) for v in rgb))


def render_chart(chart: dict, which: str = "before", tile: int = 84,
                 gap: int = 5, labels: bool = True) -> Image.Image:
    """``which`` in {before, tone, after, compare}."""
    cols, rows = chart["columns"], chart["rows"]
    annotated = which == "compare"
    if annotated:
        tile = 132                       # room for the numbers underneath
    cell_h = tile + (30 if annotated else 0)

    caption_h = 20 if annotated else 0
    header_h = 22 if labels else 0
    pad_l = 96 if labels else gap
    pad_t = caption_h + header_h + (gap if labels else gap)

    w = pad_l + len(cols) * (tile + gap) + gap
    h = pad_t + len(rows) * (cell_h + gap) + gap

    img = Image.new("RGB", (w, h), (250, 250, 251))
    d = ImageDraw.Draw(img)
    f_small, f_tiny = _font(12), _font(10)

    if annotated:
        d.text((gap + 2, 4), "each swatch: before | tone curve only | target   —   "
                             "numbers are the colour work left after the curve",
               fill=(110, 110, 115), font=f_small)
    if labels:
        for ci, name in enumerate(cols):
            d.text((pad_l + ci * (tile + gap) + 2, caption_h + 4), name,
                   fill=(70, 70, 74), font=f_small)
        for ri, name in enumerate(rows):
            d.text((8, pad_t + ri * (cell_h + gap) + tile // 2 - 7), name,
                   fill=(70, 70, 74), font=f_small)

    for (ri, ci), c in chart["cells"].items():
        x = pad_l + ci * (tile + gap)
        y = pad_t + ri * (cell_h + gap)
        if not annotated:
            _paint(d, [x, y, x + tile, y + tile], c.get(which, c["before"]))
            continue

        third = tile // 3
        for k, key in enumerate(("before", "tone", "after")):
            x0 = x + k * third
            x1 = x + tile if k == 2 else x0 + third
            _paint(d, [x0, y, x1, y + tile], c.get(key, c["before"]))
            if k:
                d.line([x0, y, x0, y + tile], fill=(255, 255, 255), width=1)

        bits = []
        if c["hue"] is not None:
            bits.append(f"H{c['hue']:+.0f}\u00b0")
        # a ratio off a pale or near-neutral swatch runs away (+234 % of almost
        # nothing), so below that point report saturation in absolute points
        sat_base = float(cs.rgb_to_hsl(c.get("tone", c["before"]))[1])
        if c["sat"] is not None and sat_base > 0.12:
            bits.append(f"S{c['sat']:+.0f}%")
        elif c["sat"] is not None:
            sat_after = float(cs.rgb_to_hsl(c["after"])[1])
            bits.append(f"S{(sat_after - sat_base) * 100:+.0f}pt")
        bits.append(f"L{c['lum']:+.0f}%")
        d.text((x + 1, y + tile + 4), "  ".join(bits), fill=(90, 90, 94), font=f_tiny)
        d.text((x + 1, y + tile + 16), f"\u0394E {c['dE']:.1f}",
               fill=(150, 150, 155), font=f_tiny)
    return img


def write_charts(stem: str, chart: dict) -> list[str]:
    out = []
    for which, suffix in (("before", "_chart_before.png"),
                          ("tone", "_chart_tone.png"),
                          ("after", "_chart_after.png"),
                          ("compare", "_chart_compare.png")):
        # the two plain charts carry no labels: they are meant to be pushed
        # through Lightroom, and text would be graded along with the swatches
        img = render_chart(chart, which, labels=(which == "compare"))
        path = stem + suffix
        img.save(path)
        out.append(path)
    return out


def chart_json(chart: dict) -> dict:
    """JSON-safe view: tuple keys become "row,col" and colours become lists."""
    cells = {}
    for (ri, ci), c in chart["cells"].items():
        cells[f"{ri},{ci}"] = {
            "row": chart["rows"][ri], "column": chart["columns"][ci],
            "before": [round(float(v), 4) for v in c["before"]],
            "tone": [round(float(v), 4) for v in c.get("tone", c["before"])],
            "after": [round(float(v), 4) for v in c["after"]],
            "dE": round(c["dE"], 2),
            "hue_shift_deg": None if c["hue"] is None else round(c["hue"], 1),
            "saturation_pct": None if c["sat"] is None else round(c["sat"], 1),
            "lightness_pct": round(c["lum"], 1),
        }
    return {"columns": chart["columns"], "rows": chart["rows"], "cells": cells}
