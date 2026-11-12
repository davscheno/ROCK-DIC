"""Real detachment perimeters drawn by the operator.

A zone (active displacement zone, automatic change zone "D#", or a cell
of the grid inspection that became a zone) can carry the perimeters of
the REAL detachment(s) seen in the images:

    zone['detachments'] = [
        {'id': 1, 'points': [[x, y], ...],          # analysed-reference px
         'area_px': float, 'perimeter_px': float, 'centroid': [x, y],
         'area_m2': float|None, 'perimeter_m': float|None,
         'n_grid_points': int, 'mean_disp_px': float, 'max_disp_px': float},
        ...]

These perimeters are what statistics, maps, PDF report and GeoJSON use
for detachment areas (instead of the grid-point estimate).  Totals are
computed on the UNION of the perimeters, so the same scar drawn in two
zones (e.g. an active zone and a change zone) is not counted twice.
"""

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _pts(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def polygon_area(points) -> float:
    p = _pts(points)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_perimeter(points) -> float:
    p = _pts(points)
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.hypot(*(np.roll(p, -1, axis=0) - p).T)))


def polygon_centroid(points) -> Tuple[float, float]:
    p = _pts(points)
    a = polygon_area(p)
    if a <= 0:
        return float(p[:, 0].mean()), float(p[:, 1].mean())
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    s = np.sum(cross) / 2.0
    return float(np.sum((x + x1) * cross) / (6 * s)), float(np.sum((y + y1) * cross) / (6 * s))


def points_in_polygon(points, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Boolean array: which (xs, ys) lie inside the polygon."""
    poly = _pts(points).astype(np.float32).reshape(-1, 1, 2)
    flat_x, flat_y = np.ravel(xs), np.ravel(ys)
    inside = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
                       for x, y in zip(flat_x, flat_y)], bool)
    return inside.reshape(np.shape(xs))


def make_detachment(points, gsd: Optional[float] = None, result=None,
                    u=None, v=None, det_id: int = 1) -> Dict:
    """Build the stored record of one drawn perimeter."""
    pts = [[int(round(x)), int(round(y))] for x, y in _pts(points)]
    area = polygon_area(pts)
    per = polygon_perimeter(pts)
    cx, cy = polygon_centroid(pts)
    d = {'id': int(det_id), 'points': pts,
         'area_px': area, 'perimeter_px': per, 'centroid': [cx, cy],
         'area_m2': area * gsd * gsd if gsd else None,
         'perimeter_m': per * gsd if gsd else None,
         'n_grid_points': 0, 'mean_disp_px': float('nan'),
         'max_disp_px': float('nan')}
    if result is not None:
        gx, gy = result.grid_x, result.grid_y
        x0, y0 = np.min(_pts(pts), axis=0)
        x1, y1 = np.max(_pts(pts), axis=0)
        near = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
        inside = np.zeros_like(near)
        if near.any():
            inside[near] = points_in_polygon(pts, gx[near], gy[near])
        d['n_grid_points'] = int(inside.sum())
        uu = result.u if u is None else u
        vv = result.v if v is None else v
        mag = np.hypot(uu, vv)[inside]
        mag = mag[np.isfinite(mag)]
        if mag.size:
            d['mean_disp_px'] = float(np.mean(mag))
            d['max_disp_px'] = float(np.max(mag))
    return d


def apply_gsd(gsd: Optional[float], *zone_lists) -> None:
    """Recompute every metric value of the zones for a new GSD.

    Pixel quantities are the measurement; m / m2 are derived from them,
    so a GSD corrected after the analysis updates perimeters (area_m2,
    perimeter_m) and automatic change zones (area_m2) in place.
    """
    g = gsd if gsd and gsd > 0 else None
    for zl in zone_lists:
        for z in (zl or []):
            if 'area_px' in z:                      # automatic change zone
                z['area_m2'] = z['area_px'] * g * g if g else None
            for d in zone_detachments(z):
                d['area_m2'] = d['area_px'] * g * g if g else None
                d['perimeter_m'] = d['perimeter_px'] * g if g else None


def zone_detachments(zone: Dict) -> List[Dict]:
    return list(zone.get('detachments') or [])


def zone_detachment_area(zone: Dict) -> Tuple[float, Optional[float]]:
    """(area px, area m2) of the zone's perimeters (sum; a zone's own
    perimeters are drawn by the operator as separate blocks)."""
    dets = zone_detachments(zone)
    px = sum(d.get('area_px', 0.0) for d in dets)
    m2 = [d.get('area_m2') for d in dets]
    return px, (sum(m2) if dets and all(a is not None for a in m2) else None)


def live_zones(*zone_lists: Iterable[Dict]) -> List[Dict]:
    """Zones that count: everything not rejected as a false positive."""
    out = []
    for zl in zone_lists:
        out.extend(z for z in (zl or []) if z.get('status') != 'rejected')
    return out


def union_mask(shape, zones: Iterable[Dict]) -> np.ndarray:
    """uint8 mask (255 inside any perimeter) in analysed-reference px."""
    m = np.zeros(tuple(shape[:2]), np.uint8)
    for z in zones:
        for d in zone_detachments(z):
            pts = np.asarray(d['points'], np.int32).reshape(-1, 1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(m, [pts], 255)
    return m


def summarize(active_zones=None, change_zones=None, gsd=None, shape=None) -> Dict:
    """Totals for statistics and report (rejected zones excluded)."""
    zones = live_zones(active_zones, change_zones)
    dets = [(z, d) for z in zones for d in zone_detachments(z)]
    out = {'n_perimeters': len(dets),
           'n_zones': sum(1 for z in zones if zone_detachments(z)),
           'sum_area_px': float(sum(d['area_px'] for _, d in dets)),
           'union_area_px': None, 'union_area_m2': None,
           'largest_area_px': max((d['area_px'] for _, d in dets), default=0.0),
           'largest_area_m2': None, 'sum_area_m2': None}
    if shape is not None and dets:
        # Rasterising adds the boundary pixels: use the raster only for the
        # overlap ratio, so without overlaps union == exact sum of areas.
        each = 0
        for _, d in dets:
            m = np.zeros(tuple(shape[:2]), np.uint8)
            cv2.fillPoly(m, [np.asarray(d['points'], np.int32).reshape(-1, 1, 2)], 255)
            each += np.count_nonzero(m)
        union = np.count_nonzero(union_mask(shape, zones))
        out['union_area_px'] = (out['sum_area_px'] * union / each) if each else 0.0
    elif dets:
        out['union_area_px'] = out['sum_area_px']
    if gsd and dets:
        g2 = gsd * gsd
        out['sum_area_m2'] = out['sum_area_px'] * g2
        out['union_area_m2'] = out['union_area_px'] * g2
        out['largest_area_m2'] = out['largest_area_px'] * g2
    return out


def zone_label(zone: Dict) -> str:
    return (f"D{zone['id']}" if zone.get('kind') == 'change'
            else f"Z{zone['id']}")


def rows(active_zones=None, change_zones=None, gsd=None) -> List[Dict]:
    """One row per perimeter (for tables / CSV)."""
    out = []
    for z in live_zones(active_zones, change_zones):
        for d in zone_detachments(z):
            out.append({'zone': zone_label(z), 'perimeter': f"{zone_label(z)}-R{d['id']}",
                        'area_px': d['area_px'], 'area_m2': d.get('area_m2'),
                        'perimeter_m': d.get('perimeter_m'),
                        'centroid_x': d['centroid'][0], 'centroid_y': d['centroid'][1],
                        'mean_disp_px': d.get('mean_disp_px'),
                        'max_disp_px': d.get('max_disp_px'),
                        'mean_disp_mm': (d['mean_disp_px'] * gsd * 1000
                                         if gsd and d.get('mean_disp_px') == d.get('mean_disp_px')
                                         and d.get('mean_disp_px') is not None else None),
                        'max_disp_mm': (d['max_disp_px'] * gsd * 1000
                                        if gsd and d.get('max_disp_px') == d.get('max_disp_px')
                                        and d.get('max_disp_px') is not None else None),
                        'status': z.get('status', 'pending')})
    return out


def export_geojson(path: str, active_zones=None, change_zones=None, gsd=None,
                   gps_info: Optional[Dict] = None, geo_frame: Optional[Dict] = None,
                   image_shape=None) -> bool:
    """Write the perimeters as a GeoJSON FeatureCollection.

    With GPS + GSD the vertices are converted to lon/lat (WGS84) with the
    same model used by the GeoTIFF export (north-up photo centred on the
    GPS position, alignment crop offset applied); otherwise they are left
    in image pixels (``"crs": "pixel"`` in the properties).
    Returns True if a file was written.
    """
    zones = live_zones(active_zones, change_zones)
    feats = []
    to_geo = None
    if gps_info and gsd and gps_info.get('latitude') is not None \
            and gps_info.get('longitude') is not None:
        from dic_app.core.geo_utils import GeoReferencer
        from dic_app.io.image_loader import GPSData
        gref = GeoReferencer()
        gps = GPSData(latitude=gps_info['latitude'], longitude=gps_info['longitude'],
                      altitude=gps_info.get('altitude'))
        frame = geo_frame or {}
        orig = tuple(frame.get('original_shape') or image_shape or (0, 0))
        ox, oy = frame.get('pixel_offset') or (0, 0)
        if orig[0] and orig[1]:
            def to_geo(x, y):
                return list(gref.pixel_to_geo(x + ox, y + oy, gps, gsd, orig[:2]))
    for z in zones:
        for d in zone_detachments(z):
            ring = [list(map(float, p)) for p in d['points']]
            if to_geo is not None:
                ring = [to_geo(x, y) for x, y in ring]
            ring.append(ring[0])
            feats.append({
                'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': [ring]},
                'properties': {
                    'zona': zone_label(z), 'perimetro': f"{zone_label(z)}-R{d['id']}",
                    'stato': z.get('status', 'pending'),
                    'area_m2': d.get('area_m2'), 'area_px': d['area_px'],
                    'perimetro_m': d.get('perimeter_m'),
                    'spost_medio_px': _finite(d.get('mean_disp_px')),
                    'spost_max_px': _finite(d.get('max_disp_px')),
                    'crs': 'EPSG:4326' if to_geo is not None else 'pixel'}})
    if not feats:
        return False
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'type': 'FeatureCollection', 'features': feats}, f, indent=1)
    return True


def _finite(v):
    return v if (v is not None and isinstance(v, (int, float)) and math.isfinite(v)) else None
