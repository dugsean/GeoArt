# GeoArt – Layered Topographic Laser-Cut Generator

Turns a set of elevation GeoTIFFs into a **stack of laser-cut layers**. The area is clipped to a canton, sliced into elevation bands, traced to smooth vector outlines and exported as SVG. The parts are then **nested onto fabrication sheets** with space optimisation (inner nesting + overlap-aware labels).

Each layer carries a red **guide outline** showing where the next layer up must be glued. It also carries an ID label, so the stack can be assembled in order.

## Pipeline

```
GeoTIFF tiles ──► data_loader ──► layer_generator ──► svg_exporter ──► layer_XX.svg
 + canton .shp    merge + clip    slice + blur        trace + export   fabrication/sheet_NN.svg
                                        │                              ▲
                                        └──── nesting.py ──────────────┘
                                        └──────────► visualization (optional 3D preview)
```

| File | Role |
|---|---|
| `main.py` | CLI entry point; runs the whole pipeline |
| `config.py` | Centralised configuration store (defaults, JSON I/O, module sync) |
| `config.json` | Default configuration file (auto-loaded by `main.py`) |
| `data_loader.py` | Merges `*.tif` tiles, clips them to the canton polygon (outside = `NaN`) |
| `layer_generator.py` | Decimates the grid, crops it to the canton, builds one boolean mask per elevation band |
| `svg_exporter.py` | Potrace tracing, guide bands, label zones, per-layer SVG export |
| `nesting.py` | **Space optimisation**: raster/FFT nester, inner nesting, post-placement labels, fabrication sheet SVG writing |
| `visualization.py` | Optional PyVista 3D stack preview |

## Installation

Python 3.9+ is recommended.

```bash
pip install numpy scipy rasterio geopandas scikit-image \
            potracer pyclipper rectpack pillow \
            pyvista matplotlib
```

- `potracer` provides the `potrace` module (pure Python). `pypotrace` also works if you have it built.
- `pillow` is required for the raster nester. Without it, the exporter falls back to rectpack (bounding-box packing).
- `pyvista` and `matplotlib` are only needed for `--visual`.

## Input data

- A folder of **elevation GeoTIFFs** (`*.tif`) sharing one CRS, at 2 m resolution.
- The **swissBOUNDARIES3D** cantons shapefile (`...KANTONSGEBIET.shp`).

## Usage

```bash
# Uses config.json automatically (or built-in defaults if absent)
python main.py

# Override specific settings via CLI
python main.py --height-step 50 --blur-sigma 8.0

# Use a different config file
python main.py --config my_project.json

# Save the effective config (defaults + JSON + CLI overrides) to a file
python main.py --save-config snapshot.json

# Add --visual to open the 3D preview
python main.py --visual
```

### CLI options

All options have corresponding keys in `config.json`. CLI values override the JSON file.

| Option | Config key | Default | Description |
|---|---|---|---|
| `--config` | — | `config.json` | Path to JSON config file |
| `--save-config` | — | — | Save effective config to JSON and exit |
| `--data-dir` | `data_dir` | `data20` | Folder with input `*.tif` files |
| `--output-dir` | `output_dir` | `laser_output` | Output folder |
| `--height-step` | `height_step` | `100` | Elevation per layer (m) |
| `--step-xy` | `step_xy` | `15` | Decimation factor; one grid cell = `2 m × step_xy` |
| `--blur-sigma` | `blur_sigma` | `5.0` | Gaussian blur σ (in grid cells) |
| `--threshold` | `threshold` | `0.5` | Threshold applied to the blurred mask |
| `--z-scale` | `z_scale` | `2.0` | Vertical exaggeration (3D preview only) |
| `--visual` | `visual` | `false` | Show the 3D model |
| `--canton-name` | `canton_name` | `Neuch` | Case-insensitive substring matched against the shapefile `NAME` |
| `--canton-shapefile` | `canton_shapefile` | *(see config.json)* | Path to the cantons shapefile |
| `--nest-inner` / `--no-nest-inner` | `nest_allow_inner` | `true` | Toggle inner nesting |
| `--raster-mm` | `fab_raster_mm` | `0.2` | Fabrication raster cell size in mm |

### All config keys

**Pipeline:** `data_dir`, `output_dir`, `height_step`, `step_xy`, `blur_sigma`, `threshold`, `z_scale`, `visual`, `canton_name`, `canton_shapefile`

**SVG output:** `cut_stroke_mm`, `guide_width_mm`, `split_paths`, `group_holes`, `group_with_guides`, `contain_min_frac`

**Text labels:** `text_labels`, `text_id_fmt`, `text_font_family`, `text_min_size`, `text_max_size`, `text_fit_pad`, `mirror_when_no_guide`, `label_in_largest_guide`, `guide_label_inset`

**Tracing:** `trace_turdsize`, `trace_alphamax`, `trace_opticurve`, `trace_opttolerance`

**Fabrication sheet:** `fab_enabled`, `fab_dirname`, `work_zone_w_mm`, `work_zone_h_mm`, `min_gap_mm`, `fab_allow_rot`, `fab_include_guides`, `fab_text_labels`, `fab_label_upright`, `fab_bin_algo`, `fab_sort_algo`

**Raster nester:** `fab_nest_backend`, `fab_raster_mm`, `fab_nest_angle_step`, `fab_arc_tol_frac`, `fab_debug_placement`

**Inner nesting:** `nest_allow_inner`, `nest_inner_margin_mm`, `nest_inner_debug`

**Post-placement labels:** `fab_label_avoid_overlap`, `label_min_gap_mm`, `label_debug`, `engrave_rgb`, `engrave_class`

## Output

```
laser_output/
├── layer_00_z<elev>m.svg      # one editable SVG per layer
├── layer_01_z<elev>m.svg
├── …
└── fabrication/
    ├── sheet_00.svg           # nested sheets, ready to cut
    ├── sheet_01.svg
    └── …
```

### SVG conventions

| Element | Colour | Meaning |
|---|---|---|
| Cut path | blue stroke, 0.1 mm | Laser cut line |
| Guide band | red fill, 0.5 mm wide | Outline of the next layer up (glue position) |
| Label (mirrored part) | red text, class `label` | ID on bottom face — visible when picking up, hidden in stack |
| Label (non-mirrored part) | black text, class `engrave` | Engraved ID on top face — hidden by next layer up |

- **Scale:** 1 grid cell = `2 m × step_xy`, and 1 unit = 0.01 mm. With the default `step_xy=15`, that is 30 m → 0.3 mm, or **1:100 000**.
- **Mirroring:** a part with **no** guide outline on it is **flipped** on the fabrication sheet.
- Each part is a single `<g>` containing its guides and cut path, so it can be moved as one object. Labels are written outside the `<g>` in absolute sheet coordinates.

## Fabrication nesting

Sheet size is `work_zone_w_mm × work_zone_h_mm` (730 × 430 mm by default). Parts that don't fit go on additional sheets.

Nesting uses a **raster / FFT collision search** (in `nesting.py`):

1. Each (mirrored) outline is offset outward by `min_gap_mm/2 + 0.71 cell` with Clipper. It is then rotated and rasterised, once per angle (every `fab_nest_angle_step` degrees).
2. For each candidate, an FFT cross-correlation against the sheet occupancy grid gives every collision-free integer position.
3. The candidate with the smallest right edge `x + w` wins, with the lowest `y` as tie-break (bottom-left-style rule). Parts are processed largest first.
4. The mask is added to the occupancy grid. If a part fits no existing sheet, a new sheet opens.

Properties:

- **No overlap:** parts cannot overlap, by construction.
- **Concave parts:** handled exactly, to cell resolution, so parts can interleave.
- **Gap:** the real gap is `min_gap_mm` plus up to about 1.4 cells.
- **Holes:** treated as solid, so no part is placed inside another part's hole.
- **Inner nesting:** when `nest_allow_inner` is `true`, the interior of a part's red guide outline (the dead space that the next layer will cover in the stack) is freed after placement. Smaller parts can then be nested inside that interior, saving sheet area. The interior is eroded by `nest_inner_margin_mm + guide_width_mm/2` so nested parts keep clear of the guide band and the parent's cut edge. The parent's label area is also excluded from the inner-free zone, so nested parts don't cover it.
- **Post-placement labels:** when `fab_label_avoid_overlap` is `true`, labels on fabrication sheets are placed *after* all parts are nested. Labels are for **assembly** — they must not be visible in the finished model:
  - **Mirrored parts** (no guide outline, flipped): the label is on the **bottom face** — visible when picking up the part, hidden once placed in the stack. Drawn in **red** (`class="label"`). Can sit anywhere on the part's material.
  - **Non-mirrored parts** (has guide outline, top face): the label is on the **top face** and must be hidden by the next layer up. It must sit **entirely inside the guide zone** (the area covered by the next layer's outline), and must **not overlap with any nested part**'s ring material (a nested part would physically cover it on the sheet). Drawn as an **engraved guide** (black, `class="engrave"`).
  - Each label's bounding box must be entirely inside its label zone and must not overlap with other parts' ring material or already-placed labels. If the original position is invalid, a new position is found via distance-transform search with progressively relaxed constraints, plus a multi-candidate fallback. Labels are always **horizontal** (no rotation).
- **Optimality:** the search is greedy, not globally optimal.

`libnest2d` is deliberately **not** used. Its no-fit polygon is convex-only, which is invalid for these concave outlines.

## Tuning tips

- **Slow nesting:** raise `fab_raster_mm` to `0.5`–`1.0`, or increase `fab_nest_angle_step` (e.g. 30). Cost is roughly parts × angles × two FFTs per sheet tried.
- **Tighter packing:** lower `fab_raster_mm`, decrease `fab_nest_angle_step`, or change the `score` expression in `nesting._nest_regions`.
- **More material savings:** keep `nest_allow_inner=true` (default) so upper-layer parts nest inside lower-layer interiors. Reduce `nest_inner_margin_mm` for tighter inner nesting (at the cost of less clearance from the red guide band).
- **Disable inner nesting:** set `nest_allow_inner=false` in `config.json` or use `--no-nest-inner`.
- **Label overlaps on sheets:** keep `fab_label_avoid_overlap=true` (default). Set `label_min_gap_mm` higher for more clearance around labels.
- **Smoother or simpler outlines:** change `blur_sigma`, `step_xy`, and the `trace_*` config keys.
- **Memory:** the loader frees intermediates with `gc.collect()`. Use a larger `step_xy` for big areas.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Expected 1 … polygon, got N` | `canton_name` matched 0 or several cantons. Use a more specific substring. |
| `region … does not fit an empty sheet` | The part is larger than the work zone. Increase `work_zone_w_mm`/`work_zone_h_mm` or `step_xy`. |
| `Pillow not installed … using the rectpack fallback` | `pip install pillow` |
| Nesting is very slow or memory-heavy | `fab_raster_mm` is too small for the sheet size, or `fab_nest_angle_step` is too small. |
| Empty or missing layers | Check `height_step` against the elevation range printed at startup. |
| `config: ignoring unknown key '…'` | The JSON file has a key not in `config.py` defaults. Check spelling. |
