"""Surface classes and change (detachment) detection.

Every grid point of an analysis gets a class:

    VALID       rock recognised: displacement measured
    APPEARANCE  rock recognised only after local normalisation: the colour
                changed (wet rock, shadow, light) but not the shape;
                displacement measured on the normalised images
    CHANGE      not recognisable in either representation although the
                texture is good: the surface itself changed -> possible
                detachment / rockfall / new debris
    VEGETATION  inside the automatic vegetation mask
    WATER       inside the automatic water/foam mask
    NODATA      outside the analysis mask, no-data (black) areas or borders
    LOWTEXTURE  subset too uniform to say anything
    UNRELIABLE  intermediate correlation: neither measured nor a change

Change points are NEVER filled or smoothed into displacements; they are
grouped into zones with an area in m2.
"""

from typing import List, Optional

import numpy as np
import cv2

from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)

VALID, APPEARANCE, CHANGE, VEGETATION, WATER, NODATA, LOWTEXTURE, UNRELIABLE = range(8)

CLASS_NAMES = {
    VALID: "Roccia misurata",
    APPEARANCE: "Cambio d'aspetto (bagnato/ombra)",
    CHANGE: "Cambiamento (possibile distacco)",
    VEGETATION: "Vegetazione",
    WATER: "Acqua / schiuma",
    NODATA: "Escluso / senza dati",
    LOWTEXTURE: "Texture insufficiente",
    UNRELIABLE: "Incerto",
}

# RGB colours for the class map
CLASS_COLORS = {
    VALID: (60, 160, 60),
    APPEARANCE: (90, 170, 230),
    CHANGE: (230, 30, 30),
    VEGETATION: (40, 110, 40),
    WATER: (40, 70, 200),
    NODATA: (80, 80, 80),
    LOWTEXTURE: (200, 200, 200),
    UNRELIABLE: (240, 190, 40),
}


# ----------------------------------------------------------------------
# Surface masks from colour
# ----------------------------------------------------------------------

def _balanced_chroma(rgb: np.ndarray):
    """Gray-world white balance, then chromatic coordinates r, g, b.

    Balancing first matters: an orthophoto with a cyan/green cast would
    otherwise turn bare rock and sea into "vegetation".
    """
    f = rgb.astype(np.float32)
    valid = f.sum(2) > 15
    if not np.any(valid):
        return None, None, None, valid, f
    means = f[valid].mean(0)
    f = f / np.maximum(means, 1.0) * means.mean()
    s = f.sum(2) + 1.0
    return f[..., 0] / s, f[..., 1] / s, f[..., 2] / s, valid, f


def vegetation_mask(rgb: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """Vegetation from excess-green (2g - r - b) with an adaptive threshold.

    rgb: (H, W, 3) uint8 in RGB order.  sensitivity > 1 marks more pixels.
    """
    r, g, b, valid, _ = _balanced_chroma(rgb)
    if r is None:
        return np.zeros(rgb.shape[:2], bool)
    exg = 2 * g - r - b
    v = exg[valid]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    thr = max(0.04, med + (3.0 / max(sensitivity, 0.1)) * mad)
    m = (exg > thr) & valid & (g > r) & (g > b)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return m > 0


def _balanced_hsv(rgb: np.ndarray):
    _, _, _, valid, f = _balanced_chroma(rgb)
    hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0          # degrees
    return hue, hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32), valid


def water_mask(rgb: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """Water and foam: cyan-blue hue after white balance.

    Rock is yellow/grey (hue ~20-60 deg after balance); sea water is
    teal/blue and foam is bright, slightly bluish and low-saturated.
    """
    k = max(sensitivity, 0.1)
    hue, sat, val, valid = _balanced_hsv(rgb)
    blue_hue = (hue >= 165) & (hue <= 255)
    water = blue_hue & (sat > 40 / k)
    foam = blue_hue & (val > 175 - 15 * (k - 1)) & (sat < 70 * k)
    m = (water | foam) & valid
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m > 0


def water_compatible(rgb: np.ndarray) -> np.ndarray:
    """Loose test: the pixel has a water-like (cyan-blue) hue at all."""
    hue, sat, val, valid = _balanced_hsv(rgb)
    m = valid & (hue >= 150) & (hue <= 270) & ((sat > 12) | (val > 150))
    return cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0


SURF_NONE, SURF_VEGETATION, SURF_WATER, SURF_NODATA = 0, 1, 2, 3


def surface_labels(ref_rgb: Optional[np.ndarray], def_rgb: Optional[np.ndarray],
                   shape, vegetation: bool = True, water: bool = False,
                   veg_sensitivity: float = 1.0,
                   water_sensitivity: float = 1.0) -> np.ndarray:
    """Pixel label image (uint8) from both epochs: union of the masks.

    A pixel is vegetation/water if it is so in EITHER image (vegetation
    that grew or disappeared is not a rock change).  Black no-data areas
    of either orthophoto are labelled NODATA.
    """
    lab = np.zeros(shape[:2], np.uint8)
    imgs = [x[..., :3] for x in (ref_rgb, def_rgb)
            if x is not None and x.shape[:2] == tuple(shape[:2])]
    if not imgs:
        return lab
    if water:
        # Water if clearly water in one epoch AND water-like in the other.
        # The sea may be foam in one survey and blue water in the next, but
        # it never turns rock-coloured; shaded rock with a bluish cast in
        # one orthophoto is yellow/grey in the other and is rejected.
        strict = [water_mask(x, water_sensitivity) for x in imgs]
        like = [water_compatible(x) for x in imgs]
        if len(imgs) == 1:
            wm = strict[0]
        else:
            wm = (strict[0] & like[1]) | (strict[1] & like[0])
        # A water body is one large connected area; bluish shadows on rock
        # (sky-lit) form smaller separate patches: keep only big bodies.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            wm.astype(np.uint8), connectivity=8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep_min = max(0.004 * wm.size, 0.25 * areas.max())
            keep = np.zeros(n, bool)
            keep[1:] = areas >= keep_min
            wm = keep[labels]
        lab[wm] = SURF_WATER
    if vegetation:
        # Vegetation in EITHER epoch: plants that grew or were removed are
        # not a change of the rock.
        for rgb3 in imgs:
            lab[vegetation_mask(rgb3, veg_sensitivity)] = SURF_VEGETATION
    for rgb3 in imgs:
        lab[rgb3.sum(2) < 8] = SURF_NODATA
    return lab


# ----------------------------------------------------------------------
# Per-point classification
# ----------------------------------------------------------------------

def subset_fraction(mask: np.ndarray, cx, cy, half: int) -> np.ndarray:
    """Fraction of each subset covered by *mask* (box filter, O(1) per point)."""
    side = 2 * half + 1
    frac = cv2.boxFilter(mask.astype(np.float32), -1, (side, side),
                         normalize=True, borderType=cv2.BORDER_CONSTANT)
    return frac[cy, cx]


def classify_points(zncc_raw, zncc_norm, texture_ok, veg_frac, water_frac,
                    nodata_frac, outside_mask, valid_threshold: float = 0.6,
                    change_threshold: float = 0.4, mask_fraction: float = 0.3):
    """Assign a class to every point (all inputs are (N,) arrays)."""
    n = len(zncc_raw)
    cls = np.full(n, UNRELIABLE, np.uint8)
    zr = np.nan_to_num(zncc_raw, nan=-1.0)
    zn = np.nan_to_num(zncc_norm, nan=-1.0)
    best = np.maximum(zr, zn)
    cls[zr >= valid_threshold] = VALID
    cls[(zr < valid_threshold) & (zn >= valid_threshold)] = APPEARANCE
    cls[(best < change_threshold) & texture_ok] = CHANGE
    cls[(best < valid_threshold) & ~texture_ok] = LOWTEXTURE
    # Masks override: vegetation/water changes are not rock changes
    cls[water_frac > mask_fraction] = WATER
    cls[veg_frac > mask_fraction] = VEGETATION
    cls[(nodata_frac > 0.05) | outside_mask] = NODATA
    return cls


# ----------------------------------------------------------------------
# Change zones
# ----------------------------------------------------------------------

def find_change_zones(point_class: np.ndarray, zncc: np.ndarray,
                      grid_x: np.ndarray, grid_y: np.ndarray, step: int,
                      gsd: Optional[float] = None, min_points: int = 4,
                      min_area_m2: float = 0.0, half: int = 0,
                      label_out: Optional[dict] = None) -> List[dict]:
    """Group CHANGE points into connected zones (8-connectivity).

    Zone dicts use the same fields as the active displacement zones
    (bbox in grid indices: row0, col0, row1, col1) so the existing
    inspection/validation tools and the report can show them.
    """
    seed = (point_class == CHANGE).astype(np.uint8)
    if label_out is not None:
        label_out['labels'] = np.zeros(point_class.shape, np.int16)
    if not seed.any():
        return []
    # Hysteresis: grow each changed area into adjacent UNRELIABLE points.
    # Near the edge of a scar the subset is half old rock / half new
    # surface, so its correlation is intermediate; without growing, the
    # area would be underestimated by a band ~ half a subset wide.
    allowed = ((point_class == CHANGE) | (point_class == UNRELIABLE)).astype(np.uint8)
    m = seed.copy()
    k3 = np.ones((3, 3), np.uint8)
    # The mixed band is at most half a subset wide: never grow further,
    # otherwise separate changes get chained together through uncertain
    # (e.g. vegetated or shadowed) areas into one huge zone.
    max_grow = int(np.ceil(half / max(step, 1))) + 1 if half else 2
    for _ in range(max_grow):
        grown = cv2.dilate(m, k3) & allowed
        if np.array_equal(grown, m):
            break
        m = grown
    # bridge single-point gaps inside a scar
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3) & \
        (point_class != NODATA).astype(np.uint8) & \
        (point_class != VEGETATION).astype(np.uint8) & \
        (point_class != WATER).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    cell_area_px = float(step * step)
    zones = []
    for i in range(1, n):
        npts = int(stats[i, cv2.CC_STAT_AREA])
        area_px = npts * cell_area_px
        area_m2 = area_px * gsd * gsd if gsd else None
        reg = labels == i
        # A zone needs a core of genuinely changed points; growing into
        # uncertain neighbours only refines the outline of a real change.
        n_core = int(np.sum(seed[reg]))
        if n_core < min_points or (area_m2 is not None and area_m2 < min_area_m2):
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        zones.append({
            'id': len(zones) + 1,
            'kind': 'change',
            'centroid_row': float(cent[i][1]),
            'centroid_col': float(cent[i][0]),
            'area_points': npts,
            'core_points': n_core,
            'area_px': area_px,
            'area_m2': area_m2,
            'mean_quality': float(np.nanmean(zncc[reg])) if np.any(np.isfinite(zncc[reg])) else float('nan'),
            # displacement fields kept for compatibility with zone tools
            'max_displacement': float('nan'),
            'mean_displacement': float('nan'),
            'bbox': (int(y), int(x), int(y + h), int(x + w)),
            'centroid_x_px': float(np.interp(cent[i][0], np.arange(grid_x.shape[1]), grid_x[0])),
            'centroid_y_px': float(np.interp(cent[i][1], np.arange(grid_y.shape[0]), grid_y[:, 0])),
            'status': 'pending',
            '_cc': i,
        })
    zones.sort(key=lambda z: z['area_points'], reverse=True)
    zone_grid = np.zeros(point_class.shape, np.int16)
    for k, z in enumerate(zones, start=1):
        z['id'] = k
        zone_grid[labels == z.pop('_cc')] = k
    if label_out is not None:
        label_out['labels'] = zone_grid
    logger.info(f"Change detection: {len(zones)} zones")
    return zones


def class_counts(point_class: np.ndarray) -> dict:
    total = max(int(point_class.size), 1)
    return {CLASS_NAMES[c]: 100.0 * float(np.sum(point_class == c)) / total
            for c in CLASS_NAMES}
