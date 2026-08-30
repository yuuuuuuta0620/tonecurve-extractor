"""Write a Lightroom / Camera Raw develop preset (.xmp)."""
from __future__ import annotations

import uuid
from xml.sax.saxutils import escape

import numpy as np

from . import curves as C
from .model import PresetModel, BAND_NAMES

CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"

# how the eight colour-mixer bands are named in the XMP
XMP_BAND = {"Red": "Red", "Orange": "Orange", "Yellow": "Yellow", "Green": "Green",
            "Aqua": "Aqua", "Blue": "Blue", "Purple": "Purple", "Magenta": "Magenta"}


def _i(v: float, lo: int = -100, hi: int = 100) -> int:
    return int(np.clip(round(float(v)), lo, hi))


def _seq(name: str, points: list[tuple[int, int]]) -> str:
    lis = "\n".join(f"     <rdf:li>{x}, {y}</rdf:li>" for x, y in points)
    return (f"   <crs:{name}>\n    <rdf:Seq>\n{lis}\n    </rdf:Seq>\n   </crs:{name}>")


def _alt(name: str, value: str) -> str:
    return (f"   <crs:{name}>\n    <rdf:Alt>\n"
            f"     <rdf:li xml:lang=\"x-default\">{escape(value)}</rdf:li>\n"
            f"    </rdf:Alt>\n   </crs:{name}>")


def build_xmp(model: PresetModel,
              include_curves: bool = True,
              include_hsl: bool = True,
              include_grading: bool = True,
              include_basic: bool = True) -> str:
    pts = model.meta.get("control_points") or {}

    def curve_points(key: str) -> list[tuple[int, int]]:
        if key in pts:
            return [tuple(p) for p in pts[key]]
        return C.fit_control_points(getattr(model, key))

    attrs = [
        'crs:PresetType="Normal"',
        'crs:Cluster=""',
        f'crs:UUID="{uuid.uuid4().hex.upper()}"',
        'crs:SupportsAmount="False"',
        'crs:SupportsAmount2="True"',
        'crs:SupportsColor="True"',
        'crs:SupportsMonochrome="True"',
        'crs:SupportsHighDynamicRange="True"',
        'crs:SupportsNormalDynamicRange="True"',
        'crs:SupportsSceneReferred="True"',
        'crs:SupportsOutputReferred="True"',
        'crs:CameraModelRestriction=""',
        'crs:Copyright=""',
        'crs:ContactInfo=""',
        'crs:Version="15.0"',
        'crs:ProcessVersion="11.0"',
        'crs:HasSettings="True"',
    ]

    if include_curves:
        attrs += [
            'crs:ToneCurveName2012="Custom"',
            'crs:ParametricShadows="0"',
            'crs:ParametricDarks="0"',
            'crs:ParametricLights="0"',
            'crs:ParametricHighlights="0"',
            'crs:ParametricShadowSplit="25"',
            'crs:ParametricMidtoneSplit="50"',
            'crs:ParametricHighlightSplit="75"',
        ]

    if include_hsl:
        for name in BAND_NAMES:
            b = model.hsl[name]
            k = XMP_BAND[name]
            attrs += [f'crs:HueAdjustment{k}="{_i(b.hue)}"',
                      f'crs:SaturationAdjustment{k}="{_i(b.sat)}"',
                      f'crs:LuminanceAdjustment{k}="{_i(b.lum)}"']

    if include_grading and model.has_grading():
        sh, mt, hi, gl = (model.grade["Shadow"], model.grade["Midtone"],
                          model.grade["Highlight"], model.grade["Global"])
        attrs += [
            f'crs:SplitToningShadowHue="{_i(sh.hue, 0, 359)}"',
            f'crs:SplitToningShadowSaturation="{_i(sh.sat, 0, 100)}"',
            f'crs:SplitToningHighlightHue="{_i(hi.hue, 0, 359)}"',
            f'crs:SplitToningHighlightSaturation="{_i(hi.sat, 0, 100)}"',
            f'crs:SplitToningBalance="{_i(model.grade_balance)}"',
            f'crs:ColorGradeMidtoneHue="{_i(mt.hue, 0, 359)}"',
            f'crs:ColorGradeMidtoneSat="{_i(mt.sat, 0, 100)}"',
            f'crs:ColorGradeMidtoneLum="{_i(mt.lum)}"',
            f'crs:ColorGradeShadowLum="{_i(sh.lum)}"',
            f'crs:ColorGradeHighlightLum="{_i(hi.lum)}"',
            f'crs:ColorGradeGlobalHue="{_i(gl.hue, 0, 359)}"',
            f'crs:ColorGradeGlobalSat="{_i(gl.sat, 0, 100)}"',
            f'crs:ColorGradeGlobalLum="{_i(gl.lum)}"',
            f'crs:ColorGradeBlending="{_i(model.grade_blending, 0, 100)}"',
        ]

    if include_basic and (abs(model.saturation) >= 0.5 or abs(model.vibrance) >= 0.5):
        attrs += [f'crs:Saturation="{_i(model.saturation)}"',
                  f'crs:Vibrance="{_i(model.vibrance)}"']

    children = [_alt("Name", model.name),
                f"   <crs:ShortName>{escape(model.name[:31])}</crs:ShortName>",
                f"   <crs:SortName>{escape(model.name)}</crs:SortName>",
                _alt("Group", model.group)]

    if include_curves:
        children.append(_seq("ToneCurvePV2012", curve_points("master")))
        ident = C.identity_lut(len(model.red))
        for key, tag in (("red", "Red"), ("green", "Green"), ("blue", "Blue")):
            lut = getattr(model, key)
            if np.abs(lut - ident).max() > 0.5 / 255.0:
                children.append(_seq(f"ToneCurvePV2012{tag}", curve_points(key)))

    attr_block = "\n    ".join(attrs)
    child_block = "\n".join(children)
    return f"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="tonecurve-extractor">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="{CRS}"
    {attr_block}>
{child_block}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def write_xmp(path: str, model: PresetModel, **kw) -> str:
    s = build_xmp(model, **kw)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    return s
