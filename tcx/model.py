"""Data model for an extracted preset."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

from . import curves as C

BAND_NAMES = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
BAND_CENTERS = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0])

ZONE_NAMES = ["Shadow", "Midtone", "Highlight", "Global"]


@dataclass
class Calibration:
    """Constants describing how our renderer interprets Lightroom sliders.

    Adobe does not publish the exact slider response curves, so these are
    documented approximations.  They are used consistently for *fitting* and
    for *verification*, which makes the round-trip self-consistent; the
    accuracy of the numbers actually written into the XMP depends on how
    close these constants are to Lightroom's internals.  Tune them with
    ``--calibration cal.json`` if you have reference material.
    """
    hue_degrees_per_100: float = 30.0   # colour-mixer Hue slider, deg at +-100
    sat_gain: float = 1.0               # colour-mixer Saturation, relative at +-100
    lum_gamma_stops: float = 1.0        # colour-mixer Luminance, gamma stops at +-100
    band_falloff: str = "smooth"        # "smooth" | "linear"
    grade_sat_gain: float = 0.35        # colour-grading tint strength at sat=100
    grade_lum_stops: float = 1.0        # colour-grading Luminance, gamma stops at +-100
    vibrance_exponent: float = 2.0      # how strongly Vibrance protects saturated pixels


@dataclass
class HSLBand:
    hue: float = 0.0
    sat: float = 0.0
    lum: float = 0.0


@dataclass
class GradeZone:
    hue: float = 0.0
    sat: float = 0.0
    lum: float = 0.0


@dataclass
class PresetModel:
    """A Lightroom-representable edit."""
    master: np.ndarray = field(default_factory=C.identity_lut)
    red: np.ndarray = field(default_factory=C.identity_lut)
    green: np.ndarray = field(default_factory=C.identity_lut)
    blue: np.ndarray = field(default_factory=C.identity_lut)

    hsl: dict[str, HSLBand] = field(
        default_factory=lambda: {n: HSLBand() for n in BAND_NAMES})
    grade: dict[str, GradeZone] = field(
        default_factory=lambda: {n: GradeZone() for n in ZONE_NAMES})
    grade_blending: float = 50.0
    grade_balance: float = 0.0

    saturation: float = 0.0   # Basic panel Saturation (only in --saturation-mode basic)
    vibrance: float = 0.0     # Basic panel Vibrance


    #: colour space the whole edit is applied in.  "melissa" = ProPhoto
    #: primaries with the sRGB tone response, which is what Lightroom's
    #: Develop module is understood to use; "srgb" applies everything in
    #: plain sRGB.  Only affects saturated colours -- the two spaces are
    #: identical on the neutral axis.
    working_space: str = "melissa"

    calibration: Calibration = field(default_factory=Calibration)
    name: str = "Extracted Preset"
    group: str = "tcx"
    meta: dict = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def channel_luts(self) -> list[np.ndarray]:
        return [self.red, self.green, self.blue]

    def has_color_curves(self) -> bool:
        ident = C.identity_lut(len(self.red))
        return any(np.abs(l - ident).max() > 1.5 / 255.0
                   for l in (self.red, self.green, self.blue))

    def has_grading(self) -> bool:
        return any(abs(z.sat) > 0.5 or abs(z.lum) > 0.5 for z in self.grade.values())

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "group": self.group,
            "master": [round(float(v), 6) for v in self.master],
            "red": [round(float(v), 6) for v in self.red],
            "green": [round(float(v), 6) for v in self.green],
            "blue": [round(float(v), 6) for v in self.blue],
            "hsl": {k: asdict(v) for k, v in self.hsl.items()},
            "grade": {k: asdict(v) for k, v in self.grade.items()},
            "grade_blending": self.grade_blending,
            "grade_balance": self.grade_balance,
            "saturation": self.saturation,
            "vibrance": self.vibrance,
            "working_space": self.working_space,
            "calibration": asdict(self.calibration),
            "meta": self.meta,
        }

    def to_json(self, path=None, indent=1) -> str:
        s = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict) -> "PresetModel":
        m = cls()
        m.name = d.get("name", m.name)
        m.group = d.get("group", m.group)
        for k in ("master", "red", "green", "blue"):
            if k in d:
                setattr(m, k, np.asarray(d[k], dtype=np.float64))
        m.hsl = {k: HSLBand(**v) for k, v in d.get("hsl", {}).items()} or m.hsl
        m.grade = {k: GradeZone(**v) for k, v in d.get("grade", {}).items()} or m.grade
        m.grade_blending = d.get("grade_blending", m.grade_blending)
        m.grade_balance = d.get("grade_balance", m.grade_balance)
        m.saturation = d.get("saturation", 0.0)
        m.vibrance = d.get("vibrance", 0.0)
        m.working_space = d.get("working_space", m.working_space)
        if "calibration" in d:
            m.calibration = Calibration(**d["calibration"])
        m.meta = d.get("meta", {})
        return m

    @classmethod
    def from_json(cls, path) -> "PresetModel":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
