# GeoArt – Layered Topographic Laser-Cut Generator

Turns a set of elevation GeoTIFFs into a **stack of laser-cut layers**. The area is clipped to a canton, sliced into elevation bands, traced to smooth vector outlines and exported as SVG. The parts are then **nested onto fabrication sheets**.

Each layer carries a red **guide outline** showing where the next layer up must be glued. It also carries an ID label, so the stack can be assembled in order.

## Pipeline

```
GeoTIFF tiles ──► data_loader ──► layer_generator ──► svg_exporter ──► layer_XX.svg
 + canton .shp    merge + clip    slice + blur        trace + nest     fabrication/sheet_NN.svg
                                        └──────────► visualization (optional 3D preview)
```

| File | Role |
|---|---|
| `main.py` | CLI entry point; runs the whole pipeline |
| `data_loader.py` | Merges `*.tif` tiles, clips them to the canton polygon (outside = `NaN`) |
| `layer_generator.py` | Decimates the grid, crops it to the canton, builds one boolean mask per elevation band |
| `svg_exporter.py` | Potrace tracing, guide bands, labels, **raster/FFT nesting**, SVG writing |
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
python main.py \
  --data-dir data20 \
  --canton-name Neuch \
  --canton-shapefile ./swissboundaries3d_.../swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp \
  --height-step 100 --step-xy 15 \
  --output-dir laser_output
```

Add `--visual` to open the 3D preview.

### CLI options

| Option | Default | Description |
|---|---|---|
| `--data-dir` | `data20` | Folder with input `*.tif` files |
| `--output-dir` | `laser_output` | Output folder |
| `--height-step` | `100` | Elevation per layer (m) |
| `--step-xy` | `15` | Decimation factor; one grid cell = `2 m × step_xy` |
| `--blur-sigma` | `5.0` | Gaussian blur σ (in grid cells) used to smooth outlines |
| `--threshold` | `0.5` | Threshold applied to the blurred mask |
| `--z-scale` | `2.0` | Vertical exaggeration (3D preview only) |
| `--visual` | off | Show the 3D model |
| `--canton-name` | `Neuch` | Case-insensitive substring matched against the shapefile `NAME`. It must match exactly one canton. |
| `--canton-shapefile` | *(see `main.py`)* | Path to the cantons shapefile |

## Output

```
laser_output/
├── layer_00_z<elev>m.svg      # one editable SVG per layer
├── layer_01_z<elev>m.svg
├── ...
└── fabrication/
    ├── sheet_00.svg           # nested sheets, ready to cut
    └── sheet_01.svg
```

### SVG conventions

| Element | Colour | Meaning |
|---|---|---|
| Cut path | blue stroke, 0.1 mm | Laser cut line |
| Guide band | red fill, 0.5 mm wide | Outline of the next layer up (glue position) |
| Label | red text | `LL-III` ID (layer-index), placed in the largest guide interior |

- **Scale:** 1 grid cell = `2 m × step_xy`, and 1 unit = 0.01 mm. With the default `--step-xy 15`, that is 30 m → 0.3 mm, or **1:100 000**.
- **Mirroring:** a part with **no** guide outline on it is **flipped** on the fabrication sheet. Its label is mirrored with it.
- Each part is a single `<g>` containing its guides, cut path and label, so it can be moved as one object.

## Fabrication nesting

Sheet size is `WORK_ZONE_W_MM × WORK_ZONE_H_MM` (730 × 430 mm by default). Parts that don't fit go on additional sheets.

Nesting uses a **raster / FFT collision search**:

1. Each (mirrored) outline is offset outward by `MIN_GAP/2 + 0.71 cell` with Clipper. It is then rotated and rasterised, once per angle in `FAB_NEST_ANGLES_DEG`.
2. For each candidate, an FFT cross-correlation against the sheet occupancy grid gives every collision-free integer position.
3. The candidate with the smallest right edge `x + w` wins, with the lowest `y` as tie-break (a bottom-left-style rule). Parts are processed largest first.
4. The mask is added to the occupancy grid. If a part fits no existing sheet, a new sheet opens.

Properties:

- **No overlap:** parts cannot overlap, by construction.
- **Concave parts:** they are handled exactly, to cell resolution, so parts can interleave.
- **Gap:** the real gap is `MIN_GAP_MM` plus up to about 1.4 cells.
- **Holes:** they are treated as solid, so no part is placed inside another part's hole.
- **Optimality:** the search is greedy, not globally optimal.

`libnest2d` is deliberately **not** used. Its no-fit polygon is convex-only, which is invalid for these concave outlines.

## Configuration

Constants at the top of `svg_exporter.py`:

| Constant | Default | Effect |
|---|---|---|
| `WORK_ZONE_W_MM`, `WORK_ZONE_H_MM` | `730`, `430` | Sheet size (mm) |
| `MIN_GAP_MM` | `5.0` | Guaranteed clearance between parts |
| `FAB_NEST_BACKEND` | `"raster"` | `"raster"` or `"rectpack"` |
| `FAB_RASTER_MM` | `0.2` | Cell size. Smaller gives a tighter gap but is slower and uses more memory. |
| `FAB_NEST_ANGLES_DEG` | every 15° | Candidate rotations. Fewer is faster. |
| `FAB_INCLUDE_GUIDES` | `True` | Draw guide bands on sheets |
| `FAB_TEXT_LABELS` | `True` | Draw ID labels on sheets |
| `FAB_LABEL_UPRIGHT` | `False` | `True` makes labels read horizontally on the sheet |
| `FAB_DEBUG_PLACEMENT` | `True` | Log each placement |
| `CUT_STROKE_MM`, `GUIDE_WIDTH_MM` | `0.1`, `0.5` | Line widths |
| `TEXT_ID_FMT` / `TEXT_ID_FUNC` | `"{layer:02d}-{index:03d}"` / `None` | Label format, or a custom callable `(layer, idx) -> str` |
| `TRACE_KW` | `turdsize=2, alphamax=1.0, …` | Potrace smoothing and speckle removal |
| `SPLIT_PATHS`, `GROUP_HOLES`, `GROUP_WITH_GUIDES` | `True` | Output granularity |
| `MIRROR_WHEN_NO_GUIDE` | `True` | Flip parts that carry no guide |

## Tuning tips

- **Slow nesting:** raise `FAB_RASTER_MM` to `0.5`–`1.0`, or use fewer angles such as `range(0, 360, 30)`. Cost is roughly parts × angles × two FFTs per sheet tried.
- **Tighter packing:** lower `FAB_RASTER_MM`, add more angles, or change the `score` expression in `_nest_regions.try_sheet`.
- **Smoother or simpler outlines:** change `--blur-sigma`, `--step-xy` and `TRACE_KW`.
- **Memory:** the loader frees intermediates with `gc.collect()`. Use a larger `--step-xy` for big areas.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Expected 1 … polygon, got N` | `--canton-name` matched 0 or several cantons. Use a more specific substring. |
| `region … does not fit an empty sheet` | The part is larger than the work zone. Increase `WORK_ZONE_*` or `--step-xy`. |
| `Pillow not installed … using the rectpack fallback` | `pip install pillow` |
| Nesting is very slow or memory-heavy | `FAB_RASTER_MM` is too small for the sheet size, or there are too many angles. |
| Empty or missing layers | Check `--height-step` against the elevation range printed at startup. |

