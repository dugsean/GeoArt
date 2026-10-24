"""
Centralised configuration for GeoArt.

All CLI arguments and module-level constants are consolidated here.
Load from a JSON file with `load_config()`, or override individual
values programmatically.  CLI arguments in `main.py` take precedence
over the JSON file, which takes precedence over the defaults.

Usage:
    from config import cfg
    cfg.load("my_config.json")          # optional: load from file
    cfg.set("height_step", 50)          # optional: programmatic override
    cfg.apply_to_modules()              # push values into svg_exporter / nesting
"""

import json
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Optional

_DEFAULTS = {
    "data_dir":          "data20",
    "output_dir":        "laser_output",
    "height_step":       100,
    "step_xy":           15,
    "blur_sigma":        5.0,
    "threshold":         0.5,
    "z_scale":           2.0,
    "visual":            False,
    "canton_name":       "Neuch",
    "canton_shapefile":  "./swissboundaries3d_2026-01_2056_5728.shp/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp",

    "cut_stroke_mm":     0.1,
    "guide_width_mm":    0.1,
    "split_paths":       True,
    "group_holes":       True,
    "group_with_guides": True,
    "contain_min_frac":  0.5,

    "text_labels":            True,
    "text_id_fmt":            "{layer:02d}-{index:03d}",
    "text_font_family":       "sans-serif",
    "text_min_size":          1.0,
    "text_max_size":          40.0,
    "text_fit_pad":           0.85,
    "mirror_when_no_guide":   True,
    "label_in_largest_guide": True,
    "guide_label_inset":      None,   # None = use guide_width/2

    "trace_turdsize":     2,
    "trace_alphamax":     1.0,
    "trace_opticurve":    True,
    "trace_opttolerance": 0.2,

    "fab_enabled":         True,
    "fab_dirname":         "fabrication",
    "work_zone_w_mm":      730.0,
    "work_zone_h_mm":      430.0,
    "min_gap_mm":          5.0,
    "fab_allow_rot":       True,
    "fab_include_guides":  True,
    "fab_text_labels":     True,
    "fab_label_upright":   False,
    "fab_bin_algo":        "BNF",
    "fab_sort_algo":       "NONE",

    "fab_nest_backend":    "raster",        # "raster" | "rectpack"
    "fab_raster_mm":       0.2,
    "fab_nest_angle_step": 15,             # degrees between candidate rotations
    "fab_arc_tol_frac":    0.05,
    "fab_debug_placement": True,

    "nest_allow_inner":      True,
    "nest_inner_margin_mm":  5.0,
    "nest_inner_debug":      True,

    "fab_label_avoid_overlap": True,
    "label_min_gap_mm":        1.0,
    "label_debug":             True,
    "engrave_rgb":             "rgb(0,0,0)",
    "engrave_class":           "engrave",
}

class Config:
    """Central configuration store.  Values can be loaded from JSON,
    overridden programmatically, and pushed into the module-level
    constants of svg_exporter and nesting."""

    def __init__(self):
        self._values = dict(_DEFAULTS)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(f"Unknown config key: {name}")

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._values[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def set(self, name: str, value: Any):
        if name not in _DEFAULTS:
            raise KeyError(f"Unknown config key: {name}")
        self._values[name] = value

    def update(self, d: dict):
        for k, v in d.items():
            self.set(k, v)

    def to_dict(self) -> dict:
        return dict(self._values)

    def load(self, path: str):
        """Load values from a JSON file.  Unknown keys are ignored with a warning."""
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            if k in _DEFAULTS:
                self._values[k] = v
            else:
                print(f"  ! config: ignoring unknown key '{k}'")
        print(f"  config loaded from {path}")

    def save(self, path: str):
        """Save current values to a JSON file."""
        with open(path, "w") as f:
            json.dump(self._values, f, indent=2)
        print(f"  config saved to {path}")

    def apply_to_modules(self):
        """Push config values into svg_exporter and nesting module constants."""
        import svg_exporter as se
        import nesting as ne

        v = self._values

        se.CUT_STROKE_MM   = v["cut_stroke_mm"]
        se.GUIDE_WIDTH_MM  = v["guide_width_mm"]
        se.SPLIT_PATHS     = v["split_paths"]
        se.GROUP_HOLES     = v["group_holes"]
        se.GROUP_WITH_GUIDES = v["group_with_guides"]
        se.CONTAIN_MIN_FRAC = v["contain_min_frac"]
        se.TEXT_LABELS     = v["text_labels"]
        se.TEXT_ID_FMT     = v["text_id_fmt"]
        se.TEXT_FONT_FAMILY = v["text_font_family"]
        se.TEXT_MIN_SIZE   = v["text_min_size"]
        se.TEXT_MAX_SIZE   = v["text_max_size"]
        se.TEXT_FIT_PAD    = v["text_fit_pad"]
        se.MIRROR_WHEN_NO_GUIDE = v["mirror_when_no_guide"]
        se.LABEL_IN_LARGEST_GUIDE = v["label_in_largest_guide"]
        se.GUIDE_LABEL_INSET = v["guide_label_inset"]
        se.TRACE_KW = dict(turdsize=v["trace_turdsize"],
                           alphamax=v["trace_alphamax"],
                           opticurve=v["trace_opticurve"],
                           opttolerance=v["trace_opttolerance"])
        se.GUIDE_KW = dict(se.TRACE_KW)

        ne.FAB_ENABLED        = v["fab_enabled"]
        ne.FAB_DIRNAME        = v["fab_dirname"]
        ne.WORK_ZONE_W_MM     = v["work_zone_w_mm"]
        ne.WORK_ZONE_H_MM     = v["work_zone_h_mm"]
        ne.MIN_GAP_MM         = v["min_gap_mm"]
        ne.FAB_ALLOW_ROT      = v["fab_allow_rot"]
        ne.FAB_INCLUDE_GUIDES = v["fab_include_guides"]
        ne.FAB_TEXT_LABELS    = v["fab_text_labels"]
        ne.FAB_LABEL_UPRIGHT  = v["fab_label_upright"]
        ne.FAB_NEST_BACKEND   = v["fab_nest_backend"]
        ne.FAB_RASTER_MM      = v["fab_raster_mm"]
        ne.FAB_NEST_ANGLES_DEG = tuple(range(0, 360, v["fab_nest_angle_step"]))
        ne.FAB_ARC_TOL_FRAC   = v["fab_arc_tol_frac"]
        ne.FAB_DEBUG_PLACEMENT = v["fab_debug_placement"]
        ne.NEST_ALLOW_INNER     = v["nest_allow_inner"]
        ne.NEST_INNER_MARGIN_MM = v["nest_inner_margin_mm"]
        ne.NEST_INNER_DEBUG     = v["nest_inner_debug"]
        ne.FAB_LABEL_AVOID_OVERLAP = v["fab_label_avoid_overlap"]
        ne.LABEL_MIN_GAP_MM    = v["label_min_gap_mm"]
        ne.LABEL_DEBUG         = v["label_debug"]
        ne.ENGRAVE_RGB         = v["engrave_rgb"]
        ne.ENGRAVE_CLASS       = v["engrave_class"]

        ne.TEXT_FONT_FAMILY = v["text_font_family"]
        ne.TEXT_MIN_SIZE    = v["text_min_size"]
        ne.TEXT_MAX_SIZE    = v["text_max_size"]
        ne.TEXT_FIT_PAD     = v["text_fit_pad"]
        ne.TRACE_KW = dict(se.TRACE_KW)

# Global singleton
cfg = Config()

def load_config(path: Optional[str] = None):
    """Load config from a JSON file (if given) and apply to all modules."""
    if path:
        cfg.load(path)
    cfg.apply_to_modules()
    return cfg
