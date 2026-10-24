"""
SVG export module for GeoArt.

Cut lines : stroked, traced contour.
Guide lines: FILLED band built by tracing the contour ONCE, then offsetting the
             smooth curve with round joins (Clipper).

Fabrication sheets (delegated to nesting.py):
  <output_dir>/fabrication/sheet_NN.svg
  * space-optimised nesting with inner nesting and overlap-aware labels.
  * see nesting.py for details.
"""

import os
import html

import numpy as np
from scipy import ndimage
import potrace
import pyclipper

from nesting import (
    FAB_ENABLED,
    FAB_INCLUDE_GUIDES,
    FAB_TEXT_LABELS,
    NEST_ALLOW_INNER,
    NEST_INNER_MARGIN_MM,
    FAB_LABEL_AVOID_OVERLAP,
    LABEL_DEBUG,
    ENGRAVE_RGB,
    ENGRAVE_CLASS,
    _inner_free_polys,
    _text_bbox,
    _place_labels_post,
    _write_fabrication,
)

PAD = 1
CUT_STROKE_RGB  = "rgb(0,0,255)"
CUT_STROKE_MM   = 0.1
GUIDE_FILL_RGB  = "rgb(255,0,0)"
GUIDE_WIDTH_MM  = 0.1
MM_PER_UNIT     = 0.01

SPLIT_PATHS       = True
GROUP_HOLES       = True
GROUP_WITH_GUIDES = True
CONTAIN_MIN_FRAC  = 0.5

TEXT_LABELS            = True
TEXT_ID_FMT            = "{layer:02d}-{index:03d}"
TEXT_ID_FUNC           = None
TEXT_FONT_FAMILY       = "sans-serif"
TEXT_MIN_SIZE          = 1.0
TEXT_MAX_SIZE          = 40.0
TEXT_FIT_PAD           = 0.85
MIRROR_WHEN_NO_GUIDE   = True
MIRROR_TRANSFORM       = "scale(-1,1)"

LABEL_IN_LARGEST_GUIDE = True
GUIDE_LABEL_INSET      = None

TRACE_KW = dict(turdsize=2, alphamax=1.0, opticurve=True, opttolerance=0.2)
GUIDE_KW = dict(TRACE_KW)

_SC      = 1000.0
_MAX_SEG = 1.0
_MIN_N, _MAX_N = 2, 32

def _to_shape(layer):
    return ~np.asarray(layer).astype(bool)

def _poly_area(poly):
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def _poly_bbox(poly):
    p = np.asarray(poly, float)
    return (float(p[:, 0].min()), float(p[:, 1].min()),
            float(p[:, 0].max()), float(p[:, 1].max()))

def _point_in_polygon(px, py, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > py) != (yj > py):
            x_cross = (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
            if px < x_cross:
                inside = not inside
        j = i
    return inside

def _flatten_curve(curve, max_seg=_MAX_SEG):
    pts = [(curve.start_point.x, curve.start_point.y)]
    p0 = curve.start_point
    for seg in curve.segments:
        if seg.is_corner:
            pts.append((seg.c.x, seg.c.y))
            pts.append((seg.end_point.x, seg.end_point.y))
        else:
            c1, c2, e = seg.c1, seg.c2, seg.end_point
            L = (abs(c1.x - p0.x) + abs(c1.y - p0.y)
                 + abs(c2.x - c1.x) + abs(c2.y - c1.y)
                 + abs(e.x - c2.x) + abs(e.y - c2.y))
            n = int(np.clip(round(L / max_seg), _MIN_N, _MAX_N))
            for t in np.linspace(1.0 / n, 1.0, n):
                mt = 1.0 - t
                pts.append((
                    mt**3 * p0.x + 3*mt**2*t * c1.x + 3*mt*t**2 * c2.x + t**3 * e.x,
                    mt**3 * p0.y + 3*mt**2*t * c1.y + 3*mt*t**2 * c2.y + t**3 * e.y,
                ))
        p0 = seg.end_point
    return pts

def _curve_to_subpath(curve, cv=lambda v: v):
    fs = curve.start_point
    s = [f"M{cv(fs.x):.3f},{cv(fs.y):.3f}"]
    for seg in curve.segments:
        if seg.is_corner:
            a, b = seg.c, seg.end_point
            s.append(f"L{cv(a.x):.3f},{cv(a.y):.3f}L{cv(b.x):.3f},{cv(b.y):.3f}")
        else:
            a, b, c = seg.c1, seg.c2, seg.end_point
            s.append(f"C{cv(a.x):.3f},{cv(a.y):.3f} "
                     f"{cv(b.x):.3f},{cv(b.y):.3f} {cv(c.x):.3f},{cv(c.y):.3f}")
    s.append("z")
    return " ".join(s)

def _dedupe_consecutive(pts, eps=1e-9):
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return pts
    keep = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - keep[-1][0]) > eps or abs(q[1] - keep[-1][1]) > eps:
            keep.append(q)
    if len(keep) > 1 and abs(keep[0][0] - keep[-1][0]) < eps \
            and abs(keep[0][1] - keep[-1][1]) < eps:
        keep.pop()
    return np.asarray(keep, float)

def _poly_ccw(pts):
    p = np.asarray(pts, float)
    return p[::-1] if _poly_area(p) < 0 else p

def _mirror_pts(pts, bbox):
    minx, _, maxx, _ = bbox
    cx = 0.5 * (minx + maxx)
    p = np.asarray(pts, float).copy()
    p[:, 0] = 2.0 * cx - p[:, 0]
    return p

def _offset_poly(pts, delta, arc_px=None):
    """Offset a polygon by `delta` px (>0 outward). -> (N,2) or None."""
    p = _poly_ccw(np.asarray(pts, float))
    if len(p) < 3:
        return None
    path = [(int(round(x * _SC)), int(round(y * _SC))) for x, y in p]
    a0 = abs(_poly_area(p))

    def run(d):
        pco = pyclipper.PyclipperOffset()
        if arc_px is not None:
            pco.ArcTolerance = max(arc_px * _SC, 0.25)
        pco.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        res = pco.Execute(d * _SC)
        if not res:
            return None
        res.sort(key=lambda q: -abs(_poly_area(np.asarray(q, float))))
        return np.asarray(res[0], float) / _SC

    out = run(delta)
    if out is None:
        return None
    if (abs(_poly_area(out)) > a0) != (delta > 0):
        out2 = run(-delta)
        if out2 is not None:
            out = out2
    return out

def _mat_mul(A, B):
    """A . B  (apply B first, then A -- matches SVG transform order)."""
    a1, b1, c1, d1, e1, f1 = A
    a2, b2, c2, d2, e2, f2 = B
    return (a1 * a2 + c1 * b2,
            b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2,
            b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1,
            b1 * e2 + d1 * f2 + f1)

def _mat_apply(M, x, y):
    a, b, c, d, e, f = M
    return (a * x + c * y + e, b * x + d * y + f)

def _mat_svg(M):
    a, b, c, d, e, f = M
    return f"matrix({a:.8f} {b:.8f} {c:.8f} {d:.8f} {e:.4f} {f:.4f})"

def _trace_curves(shape, **trace_kwargs):
    arr = np.pad(~np.asarray(shape, dtype=bool), PAD,
                 mode="constant", constant_values=True)
    return potrace.Bitmap(arr).trace(**trace_kwargs)

def _region_subpaths(shape, scale=1.0, **trace_kwargs):
    plist = _trace_curves(shape, **trace_kwargs)
    cv = lambda v: (v - PAD) / scale
    return [_curve_to_subpath(c, cv) for c in plist]

def _group_curves_by_containment(plist):
    curves = [c for c in plist if len(c.segments) > 0]
    polys = [list(_flatten_curve(c)) for c in curves]
    if not polys:
        return []
    areas  = [abs(_poly_area(p)) for p in polys]
    bboxes = [_poly_bbox(p) for p in polys]
    n = len(polys)

    parent = [-1] * n
    for i in range(n):
        best, best_area = -1, float("inf")
        bi = bboxes[i]
        for j in range(n):
            if i == j or areas[j] <= areas[i]:
                continue
            bj = bboxes[j]
            if not (bj[0] <= bi[0] and bj[1] <= bi[1]
                    and bj[2] >= bi[2] and bj[3] >= bi[3]):
                continue
            if _point_in_polygon(polys[i][0][0], polys[i][0][1], polys[j]):
                if areas[j] < best_area:
                    best_area, best = areas[j], j
        parent[i] = best

    children = {i: [] for i in range(n)}
    roots = []
    for i in range(n):
        (roots if parent[i] == -1 else children[parent[i]]).append(i)

    groups = []
    for r in roots:
        stack, order = [r], []
        while stack:
            k = stack.pop()
            order.append(k)
            stack.extend(children[k])
        groups.append([curves[k] for k in order])
    return groups

def _outline_band_subpaths(shape, thickness_px, **trace_kwargs):
    plist = _trace_curves(shape, **trace_kwargs)

    paths = []
    for curve in plist:
        poly = [(int(round((x - PAD) * _SC)), int(round((y - PAD) * _SC)))
                for (x, y) in _flatten_curve(curve)]
        if len(poly) >= 3:
            paths.append(poly)
    if not paths:
        return []

    r = max(thickness_px, 0.0) * _SC / 2.0

    def offset(delta):
        pco = pyclipper.PyclipperOffset()
        pco.AddPaths(paths, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return pco.Execute(delta)

    def to_subpaths(polys):
        subs = []
        for poly in polys:
            pts = [(x / _SC, y / _SC) for (x, y) in poly]
            s = [f"M{pts[0][0]:.3f},{pts[0][1]:.3f}"]
            s += [f"L{x:.3f},{y:.3f}" for (x, y) in pts[1:]]
            s.append("z")
            subs.append(" ".join(s))
        return subs

    return to_subpaths(offset(+r)) + to_subpaths(offset(-r))

def _cut_records(shape, **trace_kwargs):
    plist = _trace_curves(shape, **trace_kwargs)
    cv = lambda v: v - PAD

    if not SPLIT_PATHS:
        d = " ".join(_curve_to_subpath(c, cv) for c in plist)
        outer = [(x - PAD, y - PAD) for (x, y) in _flatten_curve(plist[0])] \
                if len(plist) else []
        return [{"d": d, "outer": outer, "holes": []}]

    if GROUP_HOLES:
        groups = _group_curves_by_containment(plist)
    else:
        groups = [[c] for c in plist]

    out = []
    for g in groups:
        d = " ".join(_curve_to_subpath(c, cv) for c in g)
        outer = [(x - PAD, y - PAD) for (x, y) in _flatten_curve(g[0])]
        holes = [[(x - PAD, y - PAD) for (x, y) in _flatten_curve(hc)]
                 for hc in g[1:]]
        out.append({"d": d, "outer": outer, "holes": holes})
    return out

def _guide_records(shape, thickness_px, **trace_kwargs):
    shape = np.asarray(shape, bool)

    if not SPLIT_PATHS:
        subs = _outline_band_subpaths(shape, thickness_px, **trace_kwargs)
        return [{"d": " ".join(subs), "comp": shape}] if subs else []

    labels, n = ndimage.label(shape)
    out = []
    for k in range(1, n + 1):
        comp = (labels == k)
        subs = _outline_band_subpaths(comp, thickness_px, **trace_kwargs)
        if subs:
            out.append({"d": " ".join(subs), "comp": comp})
    return out

def _enclosing_component(comp_mask, cut_labels, min_frac):
    vals = cut_labels[comp_mask]
    if vals.size == 0:
        return 0
    counts = np.bincount(vals.ravel())
    counts[0] = 0
    lab = int(counts.argmax())
    if lab == 0 or counts[lab] / vals.size < min_frac:
        return 0
    return lab

def _record_labels(labels, records):
    n = int(labels.max())
    if n == 0:
        return [0] * len(records)
    idxs = np.arange(1, n + 1)
    coms = np.asarray(ndimage.center_of_mass(labels > 0, labels, idxs), float)
    out = []
    for rec in records:
        pts = np.asarray(rec["outer"], float)
        if pts.size == 0:
            out.append(0)
            continue
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        d = (coms[:, 1] - cx) ** 2 + (coms[:, 0] - cy) ** 2
        out.append(int(idxs[int(np.argmin(d))]))
    return out

def _erode(mask, inset):
    if inset <= 0.0:
        return np.asarray(mask, bool)
    d = ndimage.distance_transform_edt(np.asarray(mask, bool))
    return d > inset

def _inscribed_circle(zone):
    zone = np.asarray(zone, bool)
    if not zone.any():
        return None
    dist = ndimage.distance_transform_edt(zone)
    r = float(dist.max())
    if r <= 0.0:
        return None
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(x), float(y), r

def _fit_font_size(radius, text):
    n = max(len(text), 1)
    denom = float(np.hypot(0.375, 0.30 * n))
    fit = (radius * TEXT_FIT_PAD) / denom
    return max(min(fit, TEXT_MAX_SIZE), TEXT_MIN_SIZE)

def _place_label(region, guide_records, text, guide_w):
    """-> (x, y, font_size, mirror) in user units, or None if no room."""
    region = np.asarray(region, bool)

    if guide_records and LABEL_IN_LARGEST_GUIDE:
        inset = (GUIDE_LABEL_INSET if GUIDE_LABEL_INSET is not None
                 else guide_w / 2.0)
        best = None
        for grec in guide_records:
            comp = np.asarray(grec["comp"], bool) & region
            if not comp.any():
                continue
            area = int(comp.sum())
            if best is None or area > best[0]:
                best = (area, comp)
        if best is not None:
            ins = _inscribed_circle(_erode(best[1], inset))
            if ins is not None:
                x, y, r = ins
                return (x, y, _fit_font_size(r, text), False)

    free = region.copy()
    for grec in guide_records:
        free &= ~np.asarray(grec["comp"], bool)
    ins = _inscribed_circle(free) if free.any() else None
    if ins is None:
        ins = _inscribed_circle(region)
    if ins is None:
        return None
    x, y, r = ins
    return (x, y, _fit_font_size(r, text), bool(MIRROR_WHEN_NO_GUIDE))

def _text_id(layer_i, idx):
    if TEXT_ID_FUNC is not None:
        return TEXT_ID_FUNC(layer_i, idx)
    return TEXT_ID_FMT.format(layer=layer_i, index=idx)

def export_svgs(layer_masks, z_list, coarse_shape, output_dir, step_xy,
                height_step, z_scale, do_svg=True, do_fabrication=FAB_ENABLED):
    """Export per-layer SVGs and (optionally) nested fabrication sheets."""
    CELL = step_xy * 2
    ny_grid, nx_grid = coarse_shape

    svg_w_mm = nx_grid * CELL * MM_PER_UNIT
    svg_h_mm = ny_grid * CELL * MM_PER_UNIT
    mm_per_pixel = svg_w_mm / nx_grid

    stroke_w = CUT_STROKE_MM / mm_per_pixel
    guide_w  = GUIDE_WIDTH_MM / mm_per_pixel

    def _write_guide(f, d, gid):
        f.write(f'<path id="{gid}" class="guide" fill="{GUIDE_FILL_RGB}" '
                f'fill-rule="evenodd" stroke="none" d="{d}"/>')

    def _write_cut(f, d, cid):
        f.write(f'<path id="{cid}" class="cut" fill="none" '
                f'stroke="{CUT_STROKE_RGB}" stroke-width="{stroke_w}" '
                f'stroke-linejoin="round" stroke-linecap="round" d="{d}"/>')

    def _write_text(f, x, y, size, mirror, text, tid):
        tf = f"translate({x:.3f},{y:.3f})"
        if mirror:
            tf += " " + MIRROR_TRANSFORM
        f.write(f'<text id="{tid}" class="label" x="0" y="0" '
                f'transform="{tf}" font-family="{TEXT_FONT_FAMILY}" '
                f'font-size="{size:.3f}" fill="{GUIDE_FILL_RGB}" '
                f'text-anchor="middle" dominant-baseline="central">'
                f'{html.escape(text)}</text>')

    want_places = TEXT_LABELS or (do_fabrication and FAB_TEXT_LABELS)
    fab_regions = []

    for i, (z, layer) in enumerate(zip(z_list, layer_masks)):
        if layer is None:
            continue
        print(f"Processing layer {i+1}/{len(z_list)}: z={z}")
        svg_path = os.path.join(output_dir, f"layer_{i:02d}_z{z}m.svg")

        cut_mask = _to_shape(layer)
        cut_records = _cut_records(cut_mask, **TRACE_KW)

        guide_records = []
        if i + 1 < len(layer_masks) and layer_masks[i + 1] is not None:
            guide_records = _guide_records(_to_shape(layer_masks[i + 1]),
                                           guide_w, **GUIDE_KW)

        cut_labels, _ = ndimage.label(cut_mask)
        rec_labels = _record_labels(cut_labels, cut_records)

        assigned = {idx: [] for idx in range(len(cut_records))}
        loose = list(guide_records)
        if guide_records:
            label_to_idx = {}
            for idx, lab in enumerate(rec_labels):
                if lab != 0:
                    label_to_idx.setdefault(lab, idx)
            loose = []
            for grec in guide_records:
                lab = _enclosing_component(grec["comp"], cut_labels,
                                           CONTAIN_MIN_FRAC)
                idx = label_to_idx.get(lab)
                (assigned[idx] if idx is not None else loose).append(grec)

        rec_places = [None] * len(cut_records)
        if want_places:
            for idx in range(len(cut_records)):
                lab = rec_labels[idx] if idx < len(rec_labels) else 0
                region = (cut_labels == lab) if lab != 0 else cut_mask
                g_for_label = assigned[idx] if GROUP_WITH_GUIDES else []
                rec_places[idx] = _place_label(region, g_for_label,
                                               _text_id(i, idx), guide_w)

        if do_svg:
            with open(svg_path, "w") as f:
                f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                        f'width="{svg_w_mm}mm" height="{svg_h_mm}mm" '
                        f'viewBox="0 0 {nx_grid} {ny_grid}">')

                if GROUP_WITH_GUIDES:
                    for idx, rec in enumerate(cut_records):
                        f.write(f'<g id="part_{i:02d}_{idx:03d}" class="part">')
                        for j, grec in enumerate(assigned[idx]):
                            _write_guide(f, grec["d"],
                                         f"guide_{i:02d}_{idx:03d}_{j:03d}")
                        _write_cut(f, rec["d"], f"cut_{i:02d}_{idx:03d}")

                        if TEXT_LABELS and rec_places[idx] is not None:
                            _write_text(f, *rec_places[idx], _text_id(i, idx),
                                        f"label_{i:02d}_{idx:03d}")
                        f.write('</g>')

                    for j, grec in enumerate(loose):
                        _write_guide(f, grec["d"], f"guide_{i:02d}_loose_{j:03d}")
                else:
                    for j, grec in enumerate(guide_records):
                        _write_guide(f, grec["d"], f"guide_{i:02d}_{j:03d}")
                    for idx, rec in enumerate(cut_records):
                        _write_cut(f, rec["d"], f"cut_{i:02d}_{idx:03d}")
                        if TEXT_LABELS and rec_places[idx] is not None:
                            _write_text(f, *rec_places[idx], _text_id(i, idx),
                                        f"label_{i:02d}_{idx:03d}")

                f.write('</svg>')

        if do_fabrication:
            for idx, rec in enumerate(cut_records):
                lab = rec_labels[idx] if idx < len(rec_labels) else 0
                region = (cut_labels == lab) if lab != 0 else cut_mask
                has_guides = len(assigned[idx]) > 0
                lbl_bbox = None
                if NEST_ALLOW_INNER and rec_places[idx] is not None:
                    lx_r, ly_r, sz_r, _ = rec_places[idx]
                    lbl_bbox = _text_bbox(lx_r, ly_r, sz_r, _text_id(i, idx))
                inner_free = _inner_free_polys(
                    assigned[idx], region, guide_w,
                    NEST_INNER_MARGIN_MM / mm_per_pixel,
                    label_bbox=lbl_bbox) if NEST_ALLOW_INNER else []
                if has_guides:
                    guide_zone = np.zeros_like(region)
                    for grec in assigned[idx]:
                        guide_zone |= np.asarray(grec["comp"], bool)
                    label_zone = guide_zone & region
                    inset = guide_w / 2.0
                    if inset > 0:
                        label_zone = _erode(label_zone, inset)
                else:
                    label_zone = region.copy()
                fab_regions.append({
                    "layer": i,
                    "z": z,
                    "d": rec["d"],
                    "outer": rec["outer"],
                    "holes": rec.get("holes", []),
                    "bbox": _poly_bbox(rec["outer"]),
                    "area": abs(_poly_area(rec["outer"])),
                    "mirrored": not has_guides,
                    "guides": ([g["d"] for g in assigned[idx]]
                               if FAB_INCLUDE_GUIDES else []),
                    "label": (rec_places[idx] if (FAB_TEXT_LABELS and
                                                  rec_places[idx] is not None)
                              else None),
                    "text": _text_id(i, idx),
                    "inner_free": inner_free,
                    "label_zone": label_zone,
                })

    if do_fabrication:
        _write_fabrication(fab_regions, output_dir, mm_per_pixel, stroke_w)

def export_fabrication_svgs(layer_masks, z_list, coarse_shape, output_dir,
                            step_xy, height_step, z_scale):
    """Fabrication sheets only (skips the editable SVGs)."""
    return export_svgs(layer_masks, z_list, coarse_shape, output_dir, step_xy,
                       height_step, z_scale, do_svg=False, do_fabrication=True)
