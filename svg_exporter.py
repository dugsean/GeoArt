"""
SVG export module for GeoArt.

Cut lines : stroked, traced contour.
Guide lines: FILLED band built by tracing the contour ONCE, then offsetting the
             smooth curve with round joins (Clipper).

Fabrication sheets (same pass, shared computation):
  <output_dir>/fabrication/sheet_NN.svg
  * a region is FLIPPED when it does NOT contain a red guiding outline;
  * parts are nested with a RASTER / FFT nester (true concave nesting):
      - each mirrored part is grown by MIN_GAP/2 + 0.71 cell (Clipper, round),
        rotated, and rasterised;
      - FFT cross-correlation with the sheet occupancy grid gives EVERY
        collision-free integer placement; the leftmost is taken;
      - no overlap is possible by construction, and the gap is >= MIN_GAP;
  * libnest2d is NOT used: its NFP is convex-only (MaxNfpLevel = CONVEX_ONLY),
    which is invalid for concave outlines and was the root cause of overlap;
  * every part is ONE <g> carrying the placement matrix; cut path, guides and
    label live inside it.
"""

import os
import math
import html
from collections import Counter

import numpy as np
from scipy import ndimage
from scipy import fft as sfft
import potrace
import pyclipper

from rectpack import newPacker, PackingMode, PackingBin, SORT_NONE

try:
    from PIL import Image, ImageDraw
    _HAVE_PIL = True
except Exception:
    Image = ImageDraw = None
    _HAVE_PIL = False


PAD = 1
CUT_STROKE_RGB  = "rgb(0,0,255)"
CUT_STROKE_MM   = 0.1
GUIDE_FILL_RGB  = "rgb(255,0,0)"
GUIDE_WIDTH_MM  = 0.5
MM_PER_UNIT     = 0.01

# --- output granularity ----------------------------------------------------
SPLIT_PATHS       = True
GROUP_HOLES       = True
GROUP_WITH_GUIDES = True
CONTAIN_MIN_FRAC  = 0.5

# --- text labels -----------------------------------------------------------
TEXT_LABELS            = True
TEXT_ID_FMT            = "{layer:02d}-{index:03d}"
TEXT_ID_FUNC           = None
TEXT_FONT_FAMILY       = "sans-serif"
TEXT_MIN_SIZE          = 1.0
TEXT_MAX_SIZE          = 40.0
TEXT_FIT_PAD           = 0.85
MIRROR_WHEN_NO_GUIDE   = True
MIRROR_TRANSFORM       = "scale(-1,1)"

# --- guide-interior labelling ---------------------------------------------
LABEL_IN_LARGEST_GUIDE = True
GUIDE_LABEL_INSET      = None

TRACE_KW = dict(turdsize=2, alphamax=1.0, opticurve=True, opttolerance=0.2)
GUIDE_KW = dict(TRACE_KW)

# --- Clipper / flattening constants ---------------------------------------
_SC      = 1000.0
_MAX_SEG = 1.0
_MIN_N, _MAX_N = 2, 32

# --- fabrication outputs ---------------------------------------------------
FAB_ENABLED        = True
FAB_DIRNAME        = "fabrication"
WORK_ZONE_W_MM     = 730.0
WORK_ZONE_H_MM     = 430.0
MIN_GAP_MM         = 5.0           # guaranteed minimum clearance (parts AND edge/2)
FAB_ALLOW_ROT      = True          # (rectpack fallback only)
FAB_INCLUDE_GUIDES = True
FAB_TEXT_LABELS    = TEXT_LABELS
FAB_LABEL_UPRIGHT  = False         # True -> ids read horizontally on the sheet
FAB_BIN_ALGO       = PackingBin.BNF
FAB_SORT_ALGO      = SORT_NONE

# --- fabrication: raster nester -------------------------------------------
FAB_NEST_BACKEND    = "raster"     # "raster" | "rectpack"
FAB_RASTER_MM       = 0.2          # cell size: smaller = tighter but slower.
                                   # true gap = MIN_GAP_MM + up to ~1.4 * this
FAB_NEST_ANGLES_DEG = tuple(range(0, 360, 15))   # fewer angles = faster
FAB_ARC_TOL_FRAC    = 0.05         # Clipper arc tolerance as a fraction of a cell
FAB_DEBUG_PLACEMENT = True


# ===========================================================================
#  geometry helpers
# ===========================================================================
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


# --- affine 2x3 matrices  (a, b, c, d, e, f) : x'=ax+cy+e, y'=bx+dy+f -------
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


# ===========================================================================
#  tracing
# ===========================================================================
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


# ===========================================================================
#  structured records
# ===========================================================================
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


# ===========================================================================
#  containment + labelling helpers
# ===========================================================================
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


# ===========================================================================
#  fabrication: raster / FFT nester (true concave nesting)
# ===========================================================================
_RPAD = 1   # cells of padding around every rasterised mask


def _rasterise(F, o, cpx):
    """Polygon F (px) -> bool mask. Pixel i has its centre at o + (i - 0.5)*cpx
    (one cell of padding), so a mask placed at integer cell ix is translated by
    T = (ix + 1) * cpx - o."""
    u = (F - o) / cpx - 0.5 + _RPAD
    w = int(math.ceil(u[:, 0].max())) + 2
    h = int(math.ceil(u[:, 1].max())) + 2
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in u.tolist()], fill=1, outline=1)
    return np.asarray(img) > 0


class _Sheet:
    __slots__ = ("occ", "FO", "free")

    def __init__(self, H, W):
        self.occ = np.zeros((H, W), np.float64)
        self.FO = None
        self.free = H * W


def _nest_regions(regions, mm_per_pixel):
    """Raster nesting. Returns (sheets, sheet_w, sheet_h); sheets maps a bin
    index to a list of {"r", "G" (2x3 matrix), "theta"}."""
    sheet_w = WORK_ZONE_W_MM / mm_per_pixel
    sheet_h = WORK_ZONE_H_MM / mm_per_pixel
    cpx = FAB_RASTER_MM / mm_per_pixel                 # cell size in px
    W, H = int(sheet_w // cpx), int(sheet_h // cpx)    # grid in cells
    g = (MIN_GAP_MM / 2.0) / mm_per_pixel              # half gap in px
    rg = g + 0.7072 * cpx                              # grow so cells cover it
    Hf, Wf = sfft.next_fast_len(H, real=True), sfft.next_fast_len(W, real=True)

    print(f"  zone {WORK_ZONE_W_MM:.0f}x{WORK_ZONE_H_MM:.0f} mm = "
          f"{sheet_w:.0f}x{sheet_h:.0f} px | grid {W}x{H} cells of "
          f"{FAB_RASTER_MM} mm | gap >= {MIN_GAP_MM} mm "
          f"({len(FAB_NEST_ANGLES_DEG)} angles)")

    sheets, state = {}, []
    order = sorted(range(len(regions)), key=lambda k: -regions[k]["area"])

    def try_sheet(sh, cands):
        if sh.FO is None:
            sh.FO = sfft.rfft2(sh.occ, s=(Hf, Wf), workers=-1)
        best = None
        for th, mask, o in cands:
            h, w = mask.shape
            if h > H or w > W or mask.sum() > sh.free:
                continue
            FM = sfft.rfft2(mask.astype(np.float64), s=(Hf, Wf), workers=-1)
            corr = sfft.irfft2(sh.FO * np.conj(FM), s=(Hf, Wf), workers=-1)
            feas = corr[:H - h + 1, :W - w + 1] < 0.5   # integer overlap counts
            idx = np.flatnonzero(feas)
            if idx.size == 0:
                continue
            y, x = np.divmod(idx, W - w + 1)
            score = (x + w).astype(np.int64) * (H + 1) + y   # leftmost, then lowest
            k = int(score.argmin())
            if best is None or score[k] < best[0]:
                best = (int(score[k]), int(x[k]), int(y[k]), th, mask, o)
        return best

    for k in order:
        r = regions[k]
        A = _dedupe_consecutive(np.asarray(r["outer"], float))
        if r["mirrored"]:
            A = _mirror_pts(A, r["bbox"])
        F0 = _offset_poly(A, rg, arc_px=FAB_ARC_TOL_FRAC * cpx)
        if F0 is None or len(F0) < 3:
            print(f"  ! layer {r['layer']}: could not offset outline, skipped")
            continue

        cands = []
        for deg in FAB_NEST_ANGLES_DEG:
            th = math.radians(deg)
            ca, sa = math.cos(th), math.sin(th)
            Fr = np.column_stack([ca * F0[:, 0] - sa * F0[:, 1],
                                  sa * F0[:, 0] + ca * F0[:, 1]])
            o = Fr.min(axis=0)
            cands.append((th, _rasterise(Fr, o, cpx), o))

        placed = False
        for b in range(len(state) + 1):
            fresh = (b == len(state))
            if fresh:
                state.append(_Sheet(H, W))
            best = try_sheet(state[b], cands)
            if best is None:
                if fresh:
                    bw = r["bbox"][2] - r["bbox"][0]
                    bh = r["bbox"][3] - r["bbox"][1]
                    print(f"  ! layer {r['layer']} region {bw:.0f}x{bh:.0f} px "
                          f"does not fit an empty sheet")
                    state.pop()
                continue

            _, ix, iy, th, mask, o = best
            h, w = mask.shape
            sh = state[b]
            sh.occ[iy:iy + h, ix:ix + w] = np.maximum(sh.occ[iy:iy + h, ix:ix + w],
                                                      mask)
            sh.FO = None
            sh.free -= int(mask.sum())

            ca, sa = math.cos(th), math.sin(th)
            Tx = (ix + _RPAD) * cpx - o[0]
            Ty = (iy + _RPAD) * cpx - o[1]
            M = (ca, sa, -sa, ca, Tx, Ty)              # world(A) = R.A + T
            if r["mirrored"]:
                minx, _, maxx, _ = r["bbox"]
                G = _mat_mul(M, (-1.0, 0.0, 0.0, 1.0, minx + maxx, 0.0))
            else:
                G = M
            sheets.setdefault(b, []).append({"r": r, "G": G, "theta": th})
            if FAB_DEBUG_PLACEMENT:
                print(f"    layer {r['layer']:02d}: sheet {b}  "
                      f"rot {math.degrees(th):5.1f}  cell ({ix},{iy})")
            placed = True
            break

    for b, sh in enumerate(state):
        print(f"  sheet {b}: {100.0 * (1 - sh.free / (H * W)):.1f}% of the grid "
              f"covered (incl. gap margins)")
    return sheets, sheet_w, sheet_h


def _write_fabrication(regions, output_dir, mm_per_pixel, stroke_w):
    fab_dir = os.path.join(output_dir, FAB_DIRNAME)
    os.makedirs(fab_dir, exist_ok=True)
    if not regions:
        print("Fabrication: no regions to place")
        return fab_dir

    if FAB_NEST_BACKEND == "rectpack" or not _HAVE_PIL:
        if not _HAVE_PIL and FAB_NEST_BACKEND != "rectpack":
            print("  ! Pillow not installed (pip install pillow) -> "
                  "using the rectpack fallback")
        return _write_fabrication_rectpack(regions, output_dir,
                                           mm_per_pixel, stroke_w)

    sheets, sheet_w, sheet_h = _nest_regions(regions, mm_per_pixel)
    n_mirror = sum(1 for r in regions if r["mirrored"])
    print(f"Fabrication (raster): {len(regions)} regions "
          f"({n_mirror} mirrored), {len(sheets)} sheet(s)")

    for b, entries in sorted(sheets.items()):
        out = os.path.join(fab_dir, f"sheet_{b:02d}.svg")
        with open(out, "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{WORK_ZONE_W_MM}mm" height="{WORK_ZONE_H_MM}mm" '
                    f'viewBox="0 0 {sheet_w:.4f} {sheet_h:.4f}">')
            for pi, e in enumerate(entries):
                r, G = e["r"], e["G"]
                f.write(f'<g id="part_{b:02d}_{pi:04d}" '
                        f'data-layer="{r["layer"]}" data-z="{r["z"]}" '
                        f'data-mirrored="{int(r["mirrored"])}" '
                        f'transform="{_mat_svg(G)}">')

                for gd in r["guides"]:
                    f.write(f'<path class="guide" fill="{GUIDE_FILL_RGB}" '
                            f'fill-rule="evenodd" stroke="none" d="{gd}"/>')
                f.write(f'<path class="cut" fill="none" stroke="{CUT_STROKE_RGB}" '
                        f'stroke-width="{stroke_w}" stroke-linejoin="round" '
                        f'stroke-linecap="round" d="{r["d"]}"/>')

                upright_text = None
                if r["label"] is not None:
                    lx, ly, size, lbl_mirror = r["label"]
                    if FAB_LABEL_UPRIGHT:
                        px, py = _mat_apply(G, lx, ly)
                        upright_text = (px, py, size, r["text"])
                    else:
                        ltf = f"translate({lx:.3f},{ly:.3f})"
                        if lbl_mirror:
                            ltf += " " + MIRROR_TRANSFORM
                        f.write(f'<text x="0" y="0" transform="{ltf}" '
                                f'font-family="{TEXT_FONT_FAMILY}" '
                                f'font-size="{size:.3f}" fill="{GUIDE_FILL_RGB}" '
                                f'text-anchor="middle" '
                                f'dominant-baseline="central">'
                                f'{html.escape(r["text"])}</text>')
                f.write('</g>')

                if upright_text is not None:     # outside the group: no inherited flip
                    px, py, size, txt = upright_text
                    f.write(f'<text x="0" y="0" '
                            f'transform="translate({px:.3f},{py:.3f})" '
                            f'font-family="{TEXT_FONT_FAMILY}" '
                            f'font-size="{size:.3f}" fill="{GUIDE_FILL_RGB}" '
                            f'text-anchor="middle" dominant-baseline="central">'
                            f'{html.escape(txt)}</text>')
            f.write('</svg>')
        print(f"  wrote {out} ({len(entries)} regions)")
    return fab_dir


# ===========================================================================
#  fabrication: rectpack fallback (bounding-box packing)
# ===========================================================================
def _region_transform(bbox, X, Y, mirrored, rotated):
    minx, miny, maxx, maxy = bbox
    cx, cy = 0.5 * (minx + maxx), 0.5 * (miny + maxy)
    rw, rh = maxx - minx, maxy - miny
    ew, eh = (rh, rw) if rotated else (rw, rh)

    seq = []
    if rotated:
        seq.append(f"rotate(90 {cx:.4f} {cy:.4f})")
    if mirrored:
        seq.append(f"translate({cx:.4f} {cy:.4f}) scale(-1 1) "
                   f"translate({-cx:.4f} {-cy:.4f})")

    tx = X - (cx - ew / 2.0)
    ty = Y - (cy - eh / 2.0)
    return (f"translate({tx:.4f} {ty:.4f}) " + " ".join(seq)).strip()


def _write_fabrication_rectpack(regions, output_dir, mm_per_pixel, stroke_w):
    fab_dir = os.path.join(output_dir, FAB_DIRNAME)
    os.makedirs(fab_dir, exist_ok=True)

    packer = newPacker(mode=PackingMode.Offline, bin_algo=FAB_BIN_ALGO,
                       sort_algo=FAB_SORT_ALGO, rotation=FAB_ALLOW_ROT)
    order = sorted(range(len(regions)), key=lambda k: -regions[k]["area"])

    meta = {}
    for rid, k in enumerate(order):
        minx, miny, maxx, maxy = regions[k]["bbox"]
        w = (maxx - minx) * mm_per_pixel
        h = (maxy - miny) * mm_per_pixel
        packer.add_rect(w + MIN_GAP_MM, h + MIN_GAP_MM, rid)
        meta[rid] = {"k": k, "W": w + MIN_GAP_MM, "H": h + MIN_GAP_MM}

    packer.add_bin(WORK_ZONE_W_MM, WORK_ZONE_H_MM, count=float("inf"))
    packer.pack()

    sheets, placed = {}, set()
    for (b, x, y, w, h, rid) in packer.rect_list():
        placed.add(rid)
        sheets.setdefault(b, []).append((rid, x, y, w, h))

    for rid in set(meta) - placed:
        r = regions[meta[rid]["k"]]
        print(f"  ! region (layer {r['layer']}, bbox {r['bbox']}) does not fit "
              f"a {WORK_ZONE_W_MM}x{WORK_ZONE_H_MM} mm zone")

    inv = 1.0 / mm_per_pixel
    sheet_w = WORK_ZONE_W_MM * inv
    sheet_h = WORK_ZONE_H_MM * inv
    n_mirror = sum(1 for r in regions if r["mirrored"])
    print(f"Fabrication (rectpack): {len(regions)} regions "
          f"({n_mirror} mirrored), {len(sheets)} sheet(s)")

    for b, items in sorted(sheets.items()):
        out = os.path.join(fab_dir, f"sheet_{b:02d}.svg")
        with open(out, "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{WORK_ZONE_W_MM}mm" height="{WORK_ZONE_H_MM}mm" '
                    f'viewBox="0 0 {sheet_w:.4f} {sheet_h:.4f}">')

            layers_here = sorted({regions[meta[rid]["k"]]["layer"]
                                  for rid, *_ in items})
            for li in layers_here:
                f.write(f'<g id="layer-{li:02d}">')
                for rid, x, y, w, h in items:
                    m, r = meta[rid], regions[meta[rid]["k"]]
                    if r["layer"] != li:
                        continue
                    rotated = (FAB_ALLOW_ROT and
                               abs(w - m["H"]) < abs(w - m["W"]))
                    X, Y = x * inv, y * inv
                    tf = _region_transform(r["bbox"], X, Y, r["mirrored"], rotated)

                    f.write(f'<g id="part_{rid:04d}" data-layer="{r["layer"]}" '
                            f'data-z="{r["z"]}" '
                            f'data-mirrored="{int(r["mirrored"])}" '
                            f'transform="{tf}">')

                    for gd in r["guides"]:
                        f.write(f'<path class="guide" fill="{GUIDE_FILL_RGB}" '
                                f'fill-rule="evenodd" stroke="none" d="{gd}"/>')
                    f.write(f'<path class="cut" fill="none" stroke="{CUT_STROKE_RGB}" '
                            f'stroke-width="{stroke_w}" stroke-linejoin="round" '
                            f'stroke-linecap="round" d="{r["d"]}"/>')

                    if r["label"] is not None:
                        lx, ly, size, lbl_mirror = r["label"]
                        if FAB_LABEL_UPRIGHT:
                            want_mirror = bool(r["mirrored"])
                        else:
                            want_mirror = bool(lbl_mirror) != bool(r["mirrored"])
                        ltf = f"translate({lx:.3f},{ly:.3f})"
                        if want_mirror:
                            ltf += " " + MIRROR_TRANSFORM
                        f.write(f'<text x="0" y="0" transform="{ltf}" '
                                f'font-family="{TEXT_FONT_FAMILY}" '
                                f'font-size="{size:.3f}" fill="{GUIDE_FILL_RGB}" '
                                f'text-anchor="middle" '
                                f'dominant-baseline="central">'
                                f'{html.escape(r["text"])}</text>')

                    f.write('</g>')
                f.write('</g>')
            f.write('</svg>')
        print(f"  wrote {out} ({len(items)} regions)")
    return fab_dir


# ===========================================================================
#  export
# ===========================================================================
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
                fab_regions.append({
                    "layer": i,
                    "z": z,
                    "d": rec["d"],
                    "outer": rec["outer"],
                    "holes": rec.get("holes", []),
                    "bbox": _poly_bbox(rec["outer"]),
                    "area": abs(_poly_area(rec["outer"])),
                    "mirrored": (len(assigned[idx]) == 0),
                    "guides": ([g["d"] for g in assigned[idx]]
                               if FAB_INCLUDE_GUIDES else []),
                    "label": (rec_places[idx] if (FAB_TEXT_LABELS and
                                                  rec_places[idx] is not None)
                              else None),
                    "text": _text_id(i, idx),
                })

    if do_fabrication:
        _write_fabrication(fab_regions, output_dir, mm_per_pixel, stroke_w)


def export_fabrication_svgs(layer_masks, z_list, coarse_shape, output_dir,
                            step_xy, height_step, z_scale):
    """Fabrication sheets only (skips the editable SVGs)."""
    return export_svgs(layer_masks, z_list, coarse_shape, output_dir, step_xy,
                       height_step, z_scale, do_svg=False, do_fabrication=True)
