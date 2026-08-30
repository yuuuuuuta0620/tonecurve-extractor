"""Human-readable diagnostics derived from the fitted curves.

These are *reported*, not written into the preset: Lightroom's Basic-panel
sliders act on scene-referred raw data, so an absolute Exposure/Temperature
value inferred from two rendered JPEGs would not transfer faithfully.  All
global tonality is carried by the tone curve instead.
"""
from __future__ import annotations

import numpy as np

from . import colorspace as cs
from . import curves as C


def _at(lut: np.ndarray, x: float) -> float:
    return float(C.apply_lut(np.array([x]), lut)[0])


def diagnose(model, before: np.ndarray, after: np.ndarray, weight: np.ndarray) -> dict:
    f = {c: C.compose_lut(getattr(model, c), model.master) for c in ("red", "green", "blue")}

    # exposure: robust median of the linear-light luminance ratio
    yb = cs.relative_luminance(before)
    ya = cs.relative_luminance(after)
    m = (yb > 0.002) & (ya > 0.002) & (weight > 0)
    ev = float(np.median(np.log2(ya[m] / yb[m]))) if m.sum() > 100 else float("nan")

    mid_slope = (_at(model.master, 0.60) - _at(model.master, 0.40)) / 0.20
    shadow_slope = (_at(model.master, 0.25) - _at(model.master, 0.10)) / 0.15
    highlight_slope = (_at(model.master, 0.90) - _at(model.master, 0.75)) / 0.15

    grey = {k: _at(v, 0.5) for k, v in f.items()}
    shadows = {k: _at(v, 0.20) for k, v in f.items()}
    highs = {k: _at(v, 0.80) for k, v in f.items()}

    def cast(d):
        rb = d["red"] - d["blue"]
        gm = d["green"] - 0.5 * (d["red"] + d["blue"])
        return {"warm_cool": round(rb * 255, 1), "green_magenta": round(-gm * 255, 1)}

    return {
        "exposure_ev": round(ev, 3),
        "black_point_out": round(_at(model.master, 0.0) * 255, 1),
        "white_point_out": round(_at(model.master, 1.0) * 255, 1),
        "contrast_midslope": round(mid_slope, 3),
        "shadow_slope": round(shadow_slope, 3),
        "highlight_slope": round(highlight_slope, 3),
        "curve_shape": ("S-curve (contrast up)" if mid_slope > 1.05 else
                        "flattened / matte" if mid_slope < 0.95 else "near-linear"),
        "cast_shadows": cast(shadows),
        "cast_midtones": cast(grey),
        "cast_highlights": cast(highs),
        "note": ("Positive warm_cool = warmer (R above B). These are diagnostics in "
                 "8-bit output units, not Lightroom slider values."),
    }
