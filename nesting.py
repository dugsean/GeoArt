"""
Space-optimisation module: fabrication nesting and label placement.

Handles raster/FFT nesting, inner nesting (parts inside guide-covered
interiors), post-placement overlap-aware labels, and fabrication sheet SVG output.
"""

import os
import math
import html

import numpy as np
from scipy import ndimage
from scipy import fft as sfft
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
GUIDE_FILL_RGB  = "rgb(255,0,0)"
TEXT_FONT_FAMILY = "sans-serif"
TEXT_MIN_SIZE    = 1.0
TEXT_MAX_SIZE    = 40.0
TEXT_FIT_PAD     = 0.85
MIRROR_TRANSFORM = "scale(-1,1)"
TRACE_KW = dict(turdsize=2, alphamax=1.0, opticurve=True, opttolerance=0.2)

_svg = None

def _S():
    """Lazily import svg_exporter for shared geometry helpers."""
    global _svg
    if _svg is None:
        import svg_exporter
        _svg = svg_exporter
    return _svg

def _dedupe_consecutive(pts, eps=1e-9):
    return _S()._dedupe_consecutive(pts, eps)

def _mirror_pts(pts, bbox):
    return _S()._mirror_pts(pts, bbox)

def _poly_area(poly):
    return _S()._poly_area(poly)

def _poly_bbox(poly):
    return _S()._poly_bbox(poly)

def _mat_mul(A, B):
    return _S()._mat_mul(A, B)

def _mat_apply(M, x, y):
    return _S()._mat_apply(M, x, y)

def _mat_svg(M):
    return _S()._mat_svg(M)

def _erode(mask, inset):
    return _S()._erode(mask, inset)

def _fit_font_size(radius, text):
    return _S()._fit_font_size(radius, text)

def _poly_ccw(pts):
    return _S()._poly_ccw(pts)

def _flatten_curve(curve, max_seg=1.0):
    return _S()._flatten_curve(curve, max_seg)

def _trace_curves(shape, **kw):
    return _S()._trace_curves(shape, **kw)

def _offset_poly(pts, delta, arc_px=None):
    return _S()._offset_poly(pts, delta, arc_px=arc_px)

FAB_ENABLED        = True
FAB_DIRNAME        = "fabrication"
WORK_ZONE_W_MM     = 730.0
WORK_ZONE_H_MM     = 430.0
MIN_GAP_MM         = 5.0           # guaranteed minimum clearance (parts AND edge/2)
FAB_ALLOW_ROT      = True          # (rectpack fallback only)
FAB_INCLUDE_GUIDES = True
FAB_TEXT_LABELS    = True
FAB_LABEL_UPRIGHT  = False         # True -> ids read horizontally on the sheet
FAB_BIN_ALGO       = PackingBin.BNF
FAB_SORT_ALGO      = SORT_NONE

FAB_NEST_BACKEND    = "raster"     # "raster" | "rectpack"
FAB_RASTER_MM       = 0.2          # cell size: smaller = tighter but slower.
                                   # true gap = MIN_GAP_MM + up to ~1.4 * this
FAB_NEST_ANGLES_DEG = tuple(range(0, 360, 15))   # fewer angles = faster
FAB_ARC_TOL_FRAC    = 0.05         # Clipper arc tolerance as a fraction of a cell
FAB_DEBUG_PLACEMENT = True

NEST_ALLOW_INNER      = True       # master switch for inner nesting
NEST_INNER_MARGIN_MM  = 5.0        # min clearance between nested part and guide band
NEST_INNER_DEBUG      = True       # log freed interior cells per placement

FAB_LABEL_AVOID_OVERLAP = True     # re-place labels after nesting
LABEL_MIN_GAP_MM        = 1.0      # min gap between label and other parts/labels
LABEL_DEBUG             = True     # log label placement decisions
ENGRAVE_RGB             = "rgb(0,0,0)"   # engraved label colour (black = engrave)
ENGRAVE_CLASS           = "engrave"       # CSS class for engraved labels

def _inner_free_polys(guide_records, region_mask, guide_w_px, margin_px,
                       label_bbox=None):
    """Return a list of (N,2) float polygons representing the interior of a
    part that is covered by guide outlines, eroded by (guide_w/2 + margin).
    These are the zones where smaller parts may be nested.

    If `label_bbox` is given as (minx, miny, maxx, maxy) in region-local coords,
    that area is subtracted from the free zone so nested parts are not placed
    on top of the parent's label (which would make it unreadable during assembly).
    The label area is grown by a few cells before subtraction.
    """
    if not guide_records or not NEST_ALLOW_INNER:
        return []
    region = np.asarray(region_mask, bool)
    inset = guide_w_px / 2.0 + margin_px
    label_exclude = np.zeros_like(region)
    if label_bbox is not None:
        lminx, lminy, lmaxx, lmaxy = label_bbox
        grow = max(2, int(label_bbox[2] - label_bbox[0]) // 4)
        ex0 = max(0, int(lminx) - grow)
        ey0 = max(0, int(lminy) - grow)
        ex1 = min(region.shape[1], int(lmaxx) + grow + 1)
        ey1 = min(region.shape[0], int(lmaxy) + grow + 1)
        label_exclude[ey0:ey1, ex0:ex1] = True
    polys = []
    for grec in guide_records:
        comp = np.asarray(grec["comp"], bool) & region
        if not comp.any():
            continue
        free = _erode(comp, inset)
        if not free.any():
            continue
        if label_exclude.any():
            free = free & ~label_exclude
            if not free.any():
                continue
        lbl, n = ndimage.label(free)
        if n == 0:
            continue
        sizes = ndimage.sum(free, lbl, range(1, n + 1))
        for k in range(1, n + 1):
            if sizes[k - 1] < 100:  # skip tiny fragments
                continue
            comp_free = (lbl == k)
            plist = _trace_curves(comp_free, **TRACE_KW)
            if not plist:
                continue
            best_curve = max(plist, key=lambda c: abs(_poly_area(_flatten_curve(c))))
            pts = _flatten_curve(best_curve)
            pts = np.asarray([(x - PAD, y - PAD) for (x, y) in pts], float)
            pts = _dedupe_consecutive(pts)
            if len(pts) >= 3:
                polys.append(_poly_ccw(pts))
    return polys

def _text_bbox(x, y, size, text, char_w_frac=0.6, char_h_frac=1.0):
    """Approximate bounding box of a centered text string at (x, y).
    Returns (minx, miny, maxx, maxy) in the same coordinate space."""
    n = max(len(text), 1)
    half_w = size * char_w_frac * n / 2.0
    half_h = size * char_h_frac / 2.0
    return (x - half_w, y - half_h, x + half_w, y + half_h)

def _label_mask(occ_H, occ_W, cx, cy, size, text, cpx):
    """Rasterise a text bounding-box onto a boolean grid.
    Returns (y0, x0, mask) or None if it falls outside the grid."""
    minx, miny, maxx, maxy = _text_bbox(cx, cy, size, text)
    gx0 = int(np.floor(minx / cpx))
    gx1 = int(np.ceil(maxx / cpx))
    gy0 = int(np.floor(miny / cpx))
    gy1 = int(np.ceil(maxy / cpy)) if (cpy := cpx) else 0
    gx0c, gy0c = max(0, gx0), max(0, gy0)
    gx1c, gy1c = min(occ_W, gx1), min(occ_H, gy1)
    if gx1c <= gx0c or gy1c <= gy0c:
        return None
    return gy0c, gx0c, (gy1c - gy0c, gx1c - gx0c)

def _place_labels_post(sheets, regions, mm_per_pixel, cpx, W, H):
    """Post-placement label pass: place labels in sheet coordinates after all
    regions are nested.  For each region, tries the original label position first
    (transformed to sheet coords), then searches for an alternative inside the
    region's *label zone* (the part's covered area, optionally excluding guide
    bands) that doesn't overlap with other parts or labels.

    The label bounding box must be **entirely inside** the label zone — it must
    sit on the piece's material, not over empty space or the cut edge.

    Mutates entries in `sheets`: sets e["label_sheet"] = (px, py, size, mirror)
    in sheet pixel coordinates, or None if no valid position is found.

    sheets: dict {bin_idx: [{"r": region, "G": matrix, "theta": th}, ...]}
    regions: list of all region dicts (for looking up region-local data)
    """
    if not FAB_LABEL_AVOID_OVERLAP:
        return

    label_gap_px = LABEL_MIN_GAP_MM / mm_per_pixel
    label_grow = int(np.ceil(label_gap_px / cpx)) + 1  # cells of growth around labels

    for b, entries in sorted(sheets.items()):
        part_masks = []
        ring_masks = []
        label_zones = []
        for e in entries:
            r = e["r"]
            G = e["G"]
            A = _dedupe_consecutive(np.asarray(r["outer"], float))
            if r["mirrored"]:
                A = _mirror_pts(A, r["bbox"])
            sheet_pts = np.column_stack([
                G[0] * A[:, 0] + G[2] * A[:, 1] + G[4],
                G[1] * A[:, 0] + G[3] * A[:, 1] + G[5],
            ])
            y0, x0, mask = _rasterise_sheet(sheet_pts, cpx, W, H)
            part_masks.append((y0, x0, mask))

            ring = np.zeros((H, W), bool)
            ph, pw = mask.shape
            sy0c = max(0, y0)
            sx0c = max(0, x0)
            sy1c = min(H, y0 + ph)
            sx1c = min(W, x0 + pw)
            if sy1c > sy0c and sx1c > sx0c:
                ring[sy0c:sy1c, sx0c:sx1c] = mask[sy0c - y0:sy1c - y0,
                                                    sx0c - x0:sx1c - x0]
            inner_polys = r.get("inner_free", [])
            for ipoly in inner_polys:
                if r["mirrored"]:
                    minx, _, maxx, _ = r["bbox"]
                    ipoly = np.asarray(ipoly, float).copy()
                    ipoly[:, 0] = minx + maxx - ipoly[:, 0]
                transformed = np.column_stack([
                    G[0] * ipoly[:, 0] + G[2] * ipoly[:, 1] + G[4],
                    G[1] * ipoly[:, 0] + G[3] * ipoly[:, 1] + G[5],
                ])
                iy0, ix0, imask = _rasterise_sheet(transformed, cpx, W, H)
                ih, iw = imask.shape
                ry0 = max(0, iy0)
                rx0 = max(0, ix0)
                ry1 = min(H, iy0 + ih)
                rx1 = min(W, ix0 + iw)
                if ry1 > ry0 and rx1 > rx0:
                    ring[ry0:ry1, rx0:rx1] &= ~imask[ry0 - iy0:ry1 - iy0,
                                                       rx0 - ix0:rx1 - ix0]
            ring_masks.append((0, 0, ring))

            lz = r.get("label_zone")
            if lz is not None:
                lz = np.asarray(lz, bool)
                try:
                    a, bb, c, d, ee, f = G
                    det = a * d - bb * c
                    if abs(det) < 1e-12:
                        label_zones.append((y0, x0, mask))
                        continue
                    ia, ib = d / det, -bb / det
                    ic, id_ = -c / det, a / det
                    mat = np.array([[id_ * cpx, ib * cpx],
                                    [ic * cpx,  ia * cpx]])
                    off_y = ib * (0.5 * cpx - ee) + id_ * (0.5 * cpx - f)
                    off_x = ia * (0.5 * cpx - ee) + ic * (0.5 * cpx - f)
                    offset = np.array([off_y, off_x])
                    warped = ndimage.affine_transform(
                        lz.astype(np.float64), mat, offset=offset,
                        output_shape=(H, W), order=0, mode='constant', cval=0.0)
                    zmask = warped > 0.5
                    label_zones.append((0, 0, zmask))
                except Exception:
                    label_zones.append((y0, x0, mask))
            else:
                label_zones.append((y0, x0, mask))

        label_occ = np.zeros((H, W), bool)

        for pi, e in enumerate(entries):
            r = e["r"]
            G = e["G"]
            text = r["text"]

            if r.get("label") is None and not FAB_TEXT_LABELS:
                e["label_sheet"] = None
                continue
            if r.get("label") is None:
                e["label_sheet"] = None
                continue

            lx, ly, size, lbl_mirror = r["label"]

            sx, sy = _mat_apply(G, lx, ly)

            def _check_label(sx, sy, size):
                """Check if a label at (sx, sy) with given size is valid.
                The label bbox must be:
                1. Entirely inside the label zone (the part's covered area)
                2. Not overlapping with any OTHER part's mask
                3. Not overlapping with already-placed labels
                Returns True if valid, False otherwise."""
                lminx, lminy, lmaxx, lmaxy = _text_bbox(sx, sy, size, text)
                gx0 = int(np.floor(lminx / cpx))
                gx1 = int(np.ceil(lmaxx / cpx))
                gy0 = int(np.floor(lminy / cpx))
                gy1 = int(np.ceil(lmaxy / cpy)) if (cpy := cpx) else 0
                gx0c, gy0c = max(0, gx0), max(0, gy0)
                gx1c, gy1c = min(W, gx1), min(H, gy1)
                if gx1c <= gx0c or gy1c <= gy0c:
                    return False  # outside sheet
                lb = np.ones((gy1c - gy0c, gx1c - gx0c), bool)

                zy0, zx0, zmask = label_zones[pi]
                zh, zw = zmask.shape
                ly0 = gy0c - zy0
                lx0 = gx0c - zx0
                ly1 = gy1c - zy0
                lx1 = gx1c - zx0
                if ly0 < 0 or lx0 < 0 or ly1 > zh or lx1 > zw:
                    return False  # label bbox extends outside label zone mask
                if not zmask[ly0:ly1, lx0:lx1].all():
                    return False  # label bbox not entirely inside label zone

                for pj, (py0, px0, pmask) in enumerate(ring_masks):
                    if pj == pi:
                        continue  # skip self
                    other_grown = ndimage.binary_dilation(pmask, iterations=label_grow) if label_grow > 0 else pmask
                    gh, gw = other_grown.shape
                    dpy0 = py0 - label_grow
                    dpx0 = px0 - label_grow
                    oy0 = max(gy0c, dpy0)
                    oy1 = min(gy0c + lb.shape[0], dpy0 + gh)
                    ox0 = max(gx0c, dpx0)
                    ox1 = min(gx0c + lb.shape[1], dpx0 + gw)
                    if oy1 <= oy0 or ox1 <= ox0:
                        continue
                    sub_lb = lb[oy0 - gy0c:oy1 - gy0c, ox0 - gx0c:ox1 - gx0c]
                    sub_pm = other_grown[oy0 - dpy0:oy1 - dpy0, ox0 - dpx0:ox1 - dpx0]
                    if (sub_lb & sub_pm).any():
                        return False

                if label_grow > 0:
                    lb_grown = ndimage.binary_dilation(lb, iterations=label_grow)
                else:
                    lb_grown = lb
                oy1l = min(H, gy0c + lb_grown.shape[0])
                ox1l = min(W, gx0c + lb_grown.shape[1])
                if (label_occ[gy0c:oy1l, gx0c:ox1l] & lb_grown[:oy1l - gy0c, :ox1l - gx0c]).any():
                    return False

                return True

            def _check_label_no_labels(sx, sy, size):
                """Like _check_label but without the label-label overlap check.
                Used for relaxed attempts where label-label overlap is tolerated."""
                lminx, lminy, lmaxx, lmaxy = _text_bbox(sx, sy, size, text)
                gx0 = int(np.floor(lminx / cpx))
                gx1 = int(np.ceil(lmaxx / cpx))
                gy0 = int(np.floor(lminy / cpx))
                gy1 = int(np.ceil(lmaxy / cpy)) if (cpy := cpx) else 0
                gx0c, gy0c = max(0, gx0), max(0, gy0)
                gx1c, gy1c = min(W, gx1), min(H, gy1)
                if gx1c <= gx0c or gy1c <= gy0c:
                    return False
                lb = np.ones((gy1c - gy0c, gx1c - gx0c), bool)

                zy0, zx0, zmask = label_zones[pi]
                zh, zw = zmask.shape
                ly0 = gy0c - zy0
                lx0 = gx0c - zx0
                ly1 = gy1c - zy0
                lx1 = gx1c - zx0
                if ly0 < 0 or lx0 < 0 or ly1 > zh or lx1 > zw:
                    return False
                if not zmask[ly0:ly1, lx0:lx1].all():
                    return False

                for pj, (py0, px0, pmask) in enumerate(ring_masks):
                    if pj == pi:
                        continue
                    other_grown = ndimage.binary_dilation(pmask, iterations=label_grow) if label_grow > 0 else pmask
                    gh, gw = other_grown.shape
                    dpy0 = py0 - label_grow
                    dpx0 = px0 - label_grow
                    oy0 = max(gy0c, dpy0)
                    oy1 = min(gy0c + lb.shape[0], dpy0 + gh)
                    ox0 = max(gx0c, dpx0)
                    ox1 = min(gx0c + lb.shape[1], dpx0 + gw)
                    if oy1 <= oy0 or ox1 <= ox0:
                        continue
                    sub_lb = lb[oy0 - gy0c:oy1 - gy0c, ox0 - gx0c:ox1 - gx0c]
                    sub_pm = other_grown[oy0 - dpy0:oy1 - dpy0, ox0 - dpx0:ox1 - dpx0]
                    if (sub_lb & sub_pm).any():
                        return False

                return True

            def _mark_label(sx, sy, size):
                """Mark a label's cells in label_occ."""
                lminx, lminy, lmaxx, lmaxy = _text_bbox(sx, sy, size, text)
                gx0 = int(np.floor(lminx / cpx))
                gx1 = int(np.ceil(lmaxx / cpx))
                gy0 = int(np.floor(lminy / cpx))
                gy1 = int(np.ceil(lmaxy / cpy)) if (cpy := cpx) else 0
                gx0c, gy0c = max(0, gx0), max(0, gy0)
                gx1c, gy1c = min(W, gx1), min(H, gy1)
                if gx1c <= gx0c or gy1c <= gy0c:
                    return
                lb = np.ones((gy1c - gy0c, gx1c - gx0c), bool)
                if label_grow > 0:
                    lb = ndimage.binary_dilation(lb, iterations=label_grow)
                oy1l = min(H, gy0c + lb.shape[0])
                ox1l = min(W, gx0c + lb.shape[1])
                label_occ[gy0c:oy1l, gx0c:ox1l] |= lb[:oy1l - gy0c, :ox1l - gx0c]

            if _check_label(sx, sy, size):
                e["label_sheet"] = (sx, sy, size, lbl_mirror)
                _mark_label(sx, sy, size)
                continue

            zy0, zx0, zmask = label_zones[pi]
            zh, zw = zmask.shape

            zone_full = np.zeros((H, W), bool)
            zsy0 = max(0, zy0)
            zsx0 = max(0, zx0)
            zsy1 = min(H, zy0 + zh)
            zsx1 = min(W, zx0 + zw)
            if zsy1 > zsy0 and zsx1 > zsx0:
                zone_full[zsy0:zsy1, zsx0:zsx1] = zmask[zsy0 - zy0:zsy1 - zy0,
                                                          zsx0 - zx0:zsx1 - zx0]

            for pj, (opy0, opx0, opmask) in enumerate(ring_masks):
                if pj == pi:
                    continue
                other_grown = ndimage.binary_dilation(opmask, iterations=label_grow) if label_grow > 0 else opmask
                gh, gw = other_grown.shape
                dpy0 = opy0 - label_grow
                dpx0 = opx0 - label_grow
                gy0c2 = max(0, dpy0)
                gx0c2 = max(0, dpx0)
                gy1c2 = min(H, dpy0 + gh)
                gx1c2 = min(W, dpx0 + gw)
                if gy1c2 > gy0c2 and gx1c2 > gx0c2:
                    zone_full[gy0c2:gy1c2, gx0c2:gx1c2] &= ~other_grown[
                        gy0c2 - dpy0:gy1c2 - dpy0,
                        gx0c2 - dpx0:gx1c2 - dpx0]

            zone_full &= ~label_occ

            if label_grow > 0:
                zone_eroded = ndimage.binary_erosion(zone_full, iterations=label_grow)
            else:
                zone_eroded = zone_full

            is_mirrored = r["mirrored"]
            placed_label = False
            for attempt in range(3):
                if attempt == 0:
                    search_zone = zone_eroded
                    verify = True
                elif attempt == 1:
                    search_zone = zone_full.copy()
                    for pj, (opy0, opx0, opmask) in enumerate(ring_masks):
                        if pj == pi:
                            continue
                        other_grown = ndimage.binary_dilation(opmask, iterations=label_grow) if label_grow > 0 else opmask
                        gh, gw = other_grown.shape
                        dpy0 = opy0 - label_grow
                        dpx0 = opx0 - label_grow
                        gy0c2 = max(0, dpy0)
                        gx0c2 = max(0, dpx0)
                        gy1c2 = min(H, dpy0 + gh)
                        gx1c2 = min(W, dpx0 + gw)
                        if gy1c2 > gy0c2 and gx1c2 > gx0c2:
                            search_zone[gy0c2:gy1c2, gx0c2:gx1c2] &= ~other_grown[
                                gy0c2 - dpy0:gy1c2 - dpy0,
                                gx0c2 - dpx0:gx1c2 - dpx0]
                    if label_grow > 0:
                        search_zone = ndimage.binary_erosion(
                            search_zone, iterations=max(1, label_grow // 2))
                    verify = True
                else:
                    search_zone = np.zeros((H, W), bool)
                    if zsy1 > zsy0 and zsx1 > zsx0:
                        search_zone[zsy0:zsy1, zsx0:zsx1] = zmask[zsy0 - zy0:zsy1 - zy0,
                                                                     zsx0 - zx0:zsx1 - zx0]
                    for pj, (opy0, opx0, opmask) in enumerate(ring_masks):
                        if pj == pi:
                            continue
                        other_grown = ndimage.binary_dilation(opmask, iterations=label_grow) if label_grow > 0 else opmask
                        gh, gw = other_grown.shape
                        dpy0 = opy0 - label_grow
                        dpx0 = opx0 - label_grow
                        gy0c2 = max(0, dpy0)
                        gx0c2 = max(0, dpx0)
                        gy1c2 = min(H, dpy0 + gh)
                        gx1c2 = min(W, dpx0 + gw)
                        if gy1c2 > gy0c2 and gx1c2 > gx0c2:
                            search_zone[gy0c2:gy1c2, gx0c2:gx1c2] &= ~other_grown[
                                gy0c2 - dpy0:gy1c2 - dpy0,
                                gx0c2 - dpx0:gx1c2 - dpx0]
                    if label_grow > 0:
                        search_zone = ndimage.binary_erosion(
                            search_zone, iterations=label_grow)
                    if is_mirrored:
                        py0_m, px0_m, pmask_m = part_masks[pi]
                        ph_m, pw_m = pmask_m.shape
                        part_full = np.zeros((H, W), bool)
                        psy0_m = max(0, py0_m)
                        psx0_m = max(0, px0_m)
                        psy1_m = min(H, py0_m + ph_m)
                        psx1_m = min(W, px0_m + pw_m)
                        if psy1_m > psy0_m and psx1_m > psx0_m:
                            part_full[psy0_m:psy1_m, psx0_m:psx1_m] = \
                                pmask_m[psy0_m - py0_m:psy1_m - py0_m,
                                         psx0_m - px0_m:psx1_m - px0_m]
                        search_zone &= part_full
                        if label_grow > 0:
                            search_zone = ndimage.binary_erosion(
                                search_zone, iterations=label_grow)
                    verify = True

                if not search_zone.any():
                    continue

                dist_s = ndimage.distance_transform_edt(search_zone)
                max_r_s = float(dist_s.max())
                if max_r_s <= 0:
                    continue

                iy_s, ix_s = np.unravel_index(int(np.argmax(dist_s)), dist_s.shape)
                cand_sx = (ix_s + 0.5) * cpx
                cand_sy = (iy_s + 0.5) * cpx
                cand_r = float(dist_s[iy_s, ix_s]) * cpx
                cand_size = _fit_font_size(cand_r, text)

                if verify and not _check_label_no_labels(cand_sx, cand_sy, cand_size):
                    continue

                e["label_sheet"] = (cand_sx, cand_sy, cand_size, lbl_mirror)
                _mark_label(cand_sx, cand_sy, cand_size)
                if cand_sx != sx or cand_sy != sy or cand_size != size:
                    tag = ["", " (relaxed)", " (zone-only)"][attempt]
                    if LABEL_DEBUG:
                        print(f"    label moved{tag}: layer {r['layer']:02d} "
                              f"part {pi} ({sx:.0f},{sy:.0f}) -> "
                              f"({cand_sx:.0f},{cand_sy:.0f}) "
                              f"size {size:.1f} -> {cand_size:.1f}")
                placed_label = True
                break

            if not placed_label:
                raw_zone = np.zeros((H, W), bool)
                if zsy1 > zsy0 and zsx1 > zsx0:
                    raw_zone[zsy0:zsy1, zsx0:zsx1] = zmask[zsy0 - zy0:zsy1 - zy0,
                                                             zsx0 - zx0:zsx1 - zx0]
                for pj, (opy0, opx0, opmask) in enumerate(ring_masks):
                    if pj == pi:
                        continue
                    other_grown = ndimage.binary_dilation(opmask, iterations=label_grow) if label_grow > 0 else opmask
                    gh, gw = other_grown.shape
                    dpy0 = opy0 - label_grow
                    dpx0 = opx0 - label_grow
                    gy0c2 = max(0, dpy0)
                    gx0c2 = max(0, dpx0)
                    gy1c2 = min(H, dpy0 + gh)
                    gx1c2 = min(W, dpx0 + gw)
                    if gy1c2 > gy0c2 and gx1c2 > gx0c2:
                        raw_zone[gy0c2:gy1c2, gx0c2:gx1c2] &= ~other_grown[
                            gy0c2 - dpy0:gy1c2 - dpy0,
                            gx0c2 - dpx0:gx1c2 - dpx0]
                if is_mirrored:
                    py0_m, px0_m, pmask_m = part_masks[pi]
                    ph_m, pw_m = pmask_m.shape
                    part_full = np.zeros((H, W), bool)
                    psy0_m = max(0, py0_m)
                    psx0_m = max(0, px0_m)
                    psy1_m = min(H, py0_m + ph_m)
                    psx1_m = min(W, px0_m + pw_m)
                    if psy1_m > psy0_m and psx1_m > psx0_m:
                        part_full[psy0_m:psy1_m, psx0_m:psx1_m] = \
                            pmask_m[psy0_m - py0_m:psy1_m - py0_m,
                                     psx0_m - px0_m:psx1_m - px0_m]
                    raw_zone &= part_full

                if raw_zone.any():
                    dist_r = ndimage.distance_transform_edt(raw_zone)
                    flat = dist_r.ravel()
                    n_candidates = min(50, flat.size)
                    top_idx = np.argpartition(flat, -n_candidates)[-n_candidates:]
                    top_idx = top_idx[np.argsort(-flat[top_idx])]
                    for ci, fi in enumerate(top_idx):
                        iy_r, ix_r = np.unravel_index(fi, dist_r.shape)
                        mr = float(dist_r[iy_r, ix_r])
                        if mr <= 0:
                            continue
                        cand_sx2 = (ix_r + 0.5) * cpx
                        cand_sy2 = (iy_r + 0.5) * cpx
                        max_sz = _fit_font_size(mr * cpx, text)
                        for shrink_frac in (1.0, 0.7, 0.5, 0.3, 0.15, 0.08, 0.04):
                            cand_size2 = max(min(max_sz, size * shrink_frac), 0.5)
                            if _check_label_no_labels(cand_sx2, cand_sy2, cand_size2):
                                e["label_sheet"] = (cand_sx2, cand_sy2, cand_size2, lbl_mirror)
                                _mark_label(cand_sx2, cand_sy2, cand_size2)
                                if LABEL_DEBUG:
                                    print(f"    label moved (multi-cand #{ci}): "
                                          f"layer {r['layer']:02d} part {pi} "
                                          f"({sx:.0f},{sy:.0f}) -> "
                                          f"({cand_sx2:.0f},{cand_sy2:.0f}) "
                                          f"size {size:.1f} -> {cand_size2:.1f}")
                                placed_label = True
                                break
                        if placed_label:
                            break

                if not placed_label:
                    if LABEL_DEBUG:
                        print(f"    ! label for layer {r['layer']:02d} part {pi}: "
                              f"no conflict-free position found, using original")
                    e["label_sheet"] = (sx, sy, size, lbl_mirror)
                    _mark_label(sx, sy, size)

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

def _rasterise_sheet(F, cpx, W, H):
    """Rasterise a polygon F (in sheet pixel coordinates) directly onto a
    (H, W) occupancy grid.  Used to free pre-computed inner zones.
    Returns (y0, x0, mask) so the caller can do occ[y0:y0+h, x0:x0+w] -= mask."""
    o = F.min(axis=0)
    u = (F - o) / cpx - 0.5 + _RPAD
    w = int(math.ceil(u[:, 0].max())) + 2
    h = int(math.ceil(u[:, 1].max())) + 2
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in u.tolist()], fill=1, outline=1)
    mask = np.asarray(img) > 0
    #                 = o_x/cpx + j + _RPAD - 1
    x0 = int(round(o[0] / cpx)) - _RPAD + 1
    y0 = int(round(o[1] / cpx)) - _RPAD + 1
    return y0, x0, mask

class _Sheet:
    __slots__ = ("occ", "FO", "free")

    def __init__(self, H, W):
        self.occ = np.zeros((H, W), np.float64)
        self.FO = None
        self.free = H * W

def _nest_regions(regions, mm_per_pixel):
    """Raster nesting with inner-zone nesting. Returns (sheets, sheet_w, sheet_h);
    sheets maps a bin index to a list of {"r", "G" (2x3 matrix), "theta"}.

    Inner nesting: when a region has inner-free polygons (zones covered by red
    guide outlines, eroded by margin), those zones are subtracted from the sheet
    occupancy grid after placement, so smaller regions can be nested inside them.
    This exploits the dead space that the next layer will cover in the stack."""
    sheet_w = WORK_ZONE_W_MM / mm_per_pixel
    sheet_h = WORK_ZONE_H_MM / mm_per_pixel
    cpx = FAB_RASTER_MM / mm_per_pixel                 # cell size in px
    W, H = int(sheet_w // cpx), int(sheet_h // cpx)    # grid in cells
    g = (MIN_GAP_MM / 2.0) / mm_per_pixel              # half gap in px
    rg = g + 0.7072 * cpx                              # grow so cells cover it
    Hf, Wf = sfft.next_fast_len(H, real=True), sfft.next_fast_len(W, real=True)

    n_inner_zones = sum(len(r.get("inner_free", [])) for r in regions if r.get("inner_free"))
    print(f"  zone {WORK_ZONE_W_MM:.0f}x{WORK_ZONE_H_MM:.0f} mm = "
          f"{sheet_w:.0f}x{sheet_h:.0f} px | grid {W}x{H} cells of "
          f"{FAB_RASTER_MM} mm | gap >= {MIN_GAP_MM} mm "
          f"({len(FAB_NEST_ANGLES_DEG)} angles)")
    if NEST_ALLOW_INNER and n_inner_zones:
        print(f"  inner nesting: ON  (margin {NEST_INNER_MARGIN_MM} mm, "
              f"{n_inner_zones} inner-free zones across regions)")
    else:
        print(f"  inner nesting: OFF")

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

    def _free_inner(sh, r, G):
        """Subtract a placed region's inner-free polygons from the sheet occupancy.
        Returns the number of grid cells freed."""
        inner_polys = r.get("inner_free", [])
        if not inner_polys:
            return 0
        total_freed = 0
        for ipoly in inner_polys:
            transformed = np.column_stack([
                G[0] * ipoly[:, 0] + G[2] * ipoly[:, 1] + G[4],
                G[1] * ipoly[:, 0] + G[3] * ipoly[:, 1] + G[5],
            ])
            y0, x0, imask = _rasterise_sheet(transformed, cpx, W, H)
            h, w = imask.shape
            sy0, sx0 = max(0, y0), max(0, x0)
            sy1, sx1 = min(H, y0 + h), min(W, x0 + w)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            sub = imask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
            region_slice = sh.occ[sy0:sy1, sx0:sx1]
            freed = int((region_slice * sub).sum())
            sh.occ[sy0:sy1, sx0:sx1] = np.where(
                sub > 0, 0.0, region_slice)
            total_freed += freed
        if total_freed > 0:
            sh.FO = None
            sh.free += total_freed
        return total_freed

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

            n_freed = 0
            if NEST_ALLOW_INNER and r.get("inner_free"):
                if r["mirrored"]:
                    minx, _, maxx, _ = r["bbox"]
                    mirrored_inner = []
                    for ipoly in r["inner_free"]:
                        ipoly = np.asarray(ipoly, float).copy()
                        ipoly[:, 0] = minx + maxx - ipoly[:, 0]
                        mirrored_inner.append(ipoly)
                    r_mirrored = dict(r)
                    r_mirrored["inner_free"] = mirrored_inner
                    n_freed = _free_inner(sh, r_mirrored, G)
                else:
                    n_freed = _free_inner(sh, r, G)

            sheets.setdefault(b, []).append({"r": r, "G": G, "theta": th})
            if FAB_DEBUG_PLACEMENT:
                line = (f"    layer {r['layer']:02d}: sheet {b}  "
                        f"rot {math.degrees(th):5.1f}  cell ({ix},{iy})")
                if n_freed and NEST_INNER_DEBUG:
                    line += f"  +inner freed {n_freed} cells"
                print(line)
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

    if FAB_LABEL_AVOID_OVERLAP and FAB_TEXT_LABELS:
        cpx_fab = FAB_RASTER_MM / mm_per_pixel
        W_fab = int(sheet_w // cpx_fab)
        H_fab = int(sheet_h // cpx_fab)
        print("  placing labels (post-nesting, overlap-aware)...")
        _place_labels_post(sheets, regions, mm_per_pixel, cpx_fab, W_fab, H_fab)

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
                lbl_data = e.get("label_sheet") if FAB_LABEL_AVOID_OVERLAP else None
                if lbl_data is not None:
                    sx, sy, size, lbl_mirror = lbl_data
                    is_engraved = not r["mirrored"]
                    upright_text = (sx, sy, size, r["text"], is_engraved)
                elif not FAB_LABEL_AVOID_OVERLAP and r["label"] is not None:
                    lx, ly, size, lbl_mirror = r["label"]
                    if FAB_LABEL_UPRIGHT:
                        px, py = _mat_apply(G, lx, ly)
                        upright_text = (px, py, size, r["text"], False)
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
                    px, py, size, txt, is_engraved = upright_text
                    if is_engraved:
                        fill = ENGRAVE_RGB
                        cls = ENGRAVE_CLASS
                    else:
                        fill = GUIDE_FILL_RGB
                        cls = "label"
                    f.write(f'<text x="0" y="0" '
                            f'transform="translate({px:.3f},{py:.3f})" '
                            f'class="{cls}" '
                            f'font-family="{TEXT_FONT_FAMILY}" '
                            f'font-size="{size:.3f}" fill="{fill}" '
                            f'text-anchor="middle" dominant-baseline="central">'
                            f'{html.escape(txt)}</text>')
            f.write('</svg>')
        print(f"  wrote {out} ({len(entries)} regions)")
    return fab_dir

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
