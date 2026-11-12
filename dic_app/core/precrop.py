"""Preliminary crop of very large photos.

Workflow
--------
1. A *guide* image (full resolution) is chosen and a rectangle is drawn on it.
2. Every other selected image is aligned to the guide *coarsely*: features
   are matched on downscaled copies (fast even for 100+ MP photos) and the
   resulting transform is scaled back to full resolution.
3. The guide is cropped to the rectangle; each other image is warped into
   the guide frame **only inside the rectangle** (no full-size warp) and
   cropped, so all crops have identical size and approximately the same
   geometry.  Fine alignment can then be done in the Alignment tab.
4. Crops are written as lossless PNG together with a JSON sidecar that
   keeps the source file, the crop box, the transform and the original
   EXIF metadata (GPS, camera, capture time) which PNG crops would lose.

The "2 a 2" use case is simply a guide plus one image.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import cv2

from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)

SIDECAR_SUFFIX = '.precrop.json'


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class CoarseAlignment:
    """Transform mapping a full-resolution image onto the guide frame."""
    matrix: np.ndarray                 # 3x3, image (full res) -> guide (full res)
    n_matches: int = 0
    n_inliers: int = 0
    rmse_px: float = 0.0               # RANSAC inlier error, full-res guide px
    scale: float = 1.0                 # downscale factor used for matching
    # Set when this is the result of the local refinement step
    global_alignment: Optional['CoarseAlignment'] = None
    refine_correction_px: Optional[float] = None


@dataclass
class CropOutcome:
    """Result for one image of a preliminary-crop run."""
    source_path: str
    output_path: Optional[str] = None
    is_guide: bool = False
    alignment: Optional[CoarseAlignment] = None
    coverage_pct: float = 100.0        # % of the rectangle covered by the image
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.output_path is not None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def read_color(path: str) -> np.ndarray:
    """Read an image as 8-bit BGR, honouring EXIF orientation like ImageLoader.

    Uses imdecode on the raw bytes so paths with non-ASCII characters work.
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Impossibile leggere l'immagine: {path}")
    return img


def downscale_factor(shape: Tuple[int, int], max_dim: int) -> float:
    """Factor s <= 1 so that max(h, w) * s <= max_dim."""
    longest = max(shape[:2])
    return 1.0 if longest <= max_dim else max_dim / float(longest)


def _scale_matrix(full_shape, small_shape) -> np.ndarray:
    """Full-res pixel -> downscaled pixel (pixel-centre convention of resize).

    Uses the ACTUAL per-axis factors after rounding the downscaled size:
    e.g. 5800 px * 1/3 -> 1933 px is a factor 0.33328, not 0.33333, which
    would otherwise bias the transform by ~1 px at the image edges.
    """
    sx = small_shape[1] / float(full_shape[1])
    sy = small_shape[0] / float(full_shape[0])
    return np.array([[sx, 0.0, 0.5 * sx - 0.5],
                     [0.0, sy, 0.5 * sy - 0.5],
                     [0.0, 0.0, 1.0]])


def _to_small_gray(img: np.ndarray, s: float) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if s >= 1.0:
        return gray
    h, w = gray.shape[:2]
    return cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _create_detector(max_features: int):
    try:
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    except (AttributeError, cv2.error):
        return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING


# ----------------------------------------------------------------------
# Coarse alignment
# ----------------------------------------------------------------------

def _match_and_estimate(g_small: np.ndarray, i_small: np.ndarray, model: str,
                        max_features: int, match_ratio: float, ransac_px: float,
                        g_mask: Optional[np.ndarray] = None,
                        i_mask: Optional[np.ndarray] = None):
    """Feature matching + RANSAC between two gray images.

    Returns (M, n_matches, n_inliers, rmse) with M mapping i_small -> g_small
    and rmse in g_small pixels.
    """
    detector, norm = _create_detector(max_features)
    kp_g, des_g = detector.detectAndCompute(g_small, g_mask)
    kp_i, des_i = detector.detectAndCompute(i_small, i_mask)
    if des_g is None or des_i is None or len(kp_g) < 4 or len(kp_i) < 4:
        raise RuntimeError("Troppe poche feature per l'allineamento "
                           "(immagini troppo uniformi o troppo diverse).")

    matches = cv2.BFMatcher(norm).knnMatch(des_i, des_g, k=2)
    good = [p[0] for p in matches
            if len(p) == 2 and p[0].distance < match_ratio * p[1].distance]
    min_needed = 4 if model == 'homography' else 3
    if len(good) < max(min_needed, 8):
        raise RuntimeError(f"Solo {len(good)} corrispondenze affidabili: "
                           "le immagini non sembrano inquadrare la stessa zona.")

    src = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_g[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if model == 'homography':
        M, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    else:
        M2, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_px)
        M = None if M2 is None else np.vstack([M2, [0.0, 0.0, 1.0]])
    if M is None:
        raise RuntimeError("Stima della trasformazione fallita (RANSAC).")

    inl = inl.ravel().astype(bool) if inl is not None else np.ones(len(good), bool)
    n_inl = int(inl.sum())
    proj = cv2.perspectiveTransform(src[inl], M)
    rmse = float(np.sqrt(np.mean(np.sum((proj - dst[inl]) ** 2, axis=2)))) \
        if n_inl else float('nan')
    return M, len(good), n_inl, rmse


def coarse_align(guide: np.ndarray, image: np.ndarray,
                 max_dim: int = 2000, model: str = 'homography',
                 max_features: int = 8000, match_ratio: float = 0.75,
                 ransac_px: float = 3.0,
                 guide_mask: Optional[np.ndarray] = None) -> CoarseAlignment:
    """Estimate the transform image -> guide on downscaled copies.

    Parameters
    ----------
    guide, image : full-resolution images (gray or BGR)
    max_dim : longest side used for matching.  Each image is reduced to
        this size independently (SIFT is scale invariant), so a photo with
        a much finer pixel size does not force the other one to be
        shrunk too far.
    model : 'homography' (different viewpoints) or 'affine'
        (similarity: shift + rotation + scale, for fixed cameras)
    ransac_px : RANSAC threshold in *downscaled* pixels
    guide_mask : optional full-res uint8 mask on the guide (255 = use)

    Returns
    -------
    CoarseAlignment with a 3x3 matrix valid at full resolution.
    """
    s_g = downscale_factor(guide.shape, max_dim)
    s_i = downscale_factor(image.shape, max_dim)
    g_small = _to_small_gray(guide, s_g)
    i_small = _to_small_gray(image, s_i)
    m_small = None
    if guide_mask is not None:
        m_small = cv2.resize(guide_mask, (g_small.shape[1], g_small.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    M, n_good, n_inl, rmse_small = _match_and_estimate(
        g_small, i_small, model, max_features, match_ratio, ransac_px, m_small)

    S_img = _scale_matrix(image.shape, i_small.shape)
    S_guide = _scale_matrix(guide.shape, g_small.shape)
    M_full = np.linalg.inv(S_guide) @ M @ S_img
    M_full /= M_full[2, 2]

    logger.info(f"Coarse alignment (guide scale {s_g:.3f}, image scale {s_i:.3f}): "
                f"{n_inl}/{n_good} inliers, RMSE {rmse_small / s_g:.2f} px full-res")
    return CoarseAlignment(matrix=M_full, n_matches=n_good, n_inliers=n_inl,
                           rmse_px=rmse_small / s_g, scale=s_g)


# ----------------------------------------------------------------------
# Local refinement around the crop rectangle
# ----------------------------------------------------------------------

def _warp_to_grid(img: np.ndarray, M: np.ndarray, out_w: int, out_h: int,
                  gray: bool = True) -> np.ndarray:
    """Resample *img* onto an (out_h, out_w) grid; M maps img px -> grid px.

    Only the source area that lands on the grid is read, and it is first
    reduced with INTER_AREA when the grid is much coarser than the source
    (plain warping would alias fine texture, e.g. a 2.9 cm orthophoto
    resampled to a 7.5 cm grid at reduced scale).
    """
    Minv = np.linalg.inv(M)
    corners = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]]
                         ).reshape(-1, 1, 2)
    src_quad = cv2.perspectiveTransform(corners, Minv).reshape(-1, 2)
    ih, iw = img.shape[:2]
    pad = 8
    x0 = int(max(0, np.floor(src_quad[:, 0].min()) - pad))
    y0 = int(max(0, np.floor(src_quad[:, 1].min()) - pad))
    x1 = int(min(iw, np.ceil(src_quad[:, 0].max()) + pad))
    y1 = int(min(ih, np.ceil(src_quad[:, 1].max()) + pad))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((out_h, out_w), np.uint8) if gray else \
            np.zeros((out_h, out_w) + img.shape[2:], np.uint8)
    roi = img[y0:y1, x0:x1]
    if gray and roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    T = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])
    M_roi = M @ T  # roi px -> grid px

    # Local scale of the mapping (grid px per source px) at the grid centre
    c = cv2.perspectiveTransform(np.float32([[[out_w / 2, out_h / 2]]]), Minv)[0, 0]
    pts = np.float32([[c], [c + [1, 0]], [c + [0, 1]]]) - [x0, y0]
    q = cv2.perspectiveTransform(pts.astype(np.float32), M_roi).reshape(-1, 2)
    ex, ey = q[1] - q[0], q[2] - q[0]
    # area scale of the 2x2 Jacobian (np.cross rejects 2D vectors in NumPy 2)
    local_scale = float(np.sqrt(abs(ex[0] * ey[1] - ex[1] * ey[0])))
    if local_scale < 0.7:
        rh, rw = roi.shape[:2]
        nw, nh = max(1, round(rw * local_scale)), max(1, round(rh * local_scale))
        small = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
        M_roi = M_roi @ np.linalg.inv(_scale_matrix(roi.shape, small.shape))
        roi = small
    return cv2.warpPerspective(roi, M_roi, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def refine_local(guide: np.ndarray, image: np.ndarray, M: np.ndarray, rect,
                 max_dim: int = 2000, model: str = 'homography',
                 margin: float = 0.5, iterations: int = 2,
                 max_features: int = 8000, match_ratio: float = 0.8,
                 ransac_px: float = 2.0) -> Tuple[np.ndarray, CoarseAlignment, float]:
    """Refine M (image -> guide) using only the area around *rect*.

    The guide area (rect + margin) and the image resampled onto the same
    grid through the current M are matched at up to *max_dim* px, i.e. at
    a much higher resolution than the whole-image coarse step.  The
    residual transform is composed into M.

    Returns (M_refined, stats, correction_px) where correction_px is the
    largest change at the rectangle corners (full-res guide px).
    """
    gh, gw = guide.shape[:2]
    x0, y0, x1, y1 = rect
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    rx0, ry0 = max(0, x0 - mx), max(0, y0 - my)
    rx1, ry1 = min(gw, x1 + mx), min(gh, y1 + my)
    s = downscale_factor((ry1 - ry0, rx1 - rx0), max_dim)
    out_w = max(16, int(round((rx1 - rx0) * s)))
    out_h = max(16, int(round((ry1 - ry0) * s)))
    # guide px -> ROI grid px (pixel-centre consistent for both images)
    A = np.array([[s, 0.0, -rx0 * s + 0.5 * s - 0.5],
                  [0.0, s, -ry0 * s + 0.5 * s - 0.5],
                  [0.0, 0.0, 1.0]])
    A_inv = np.linalg.inv(A)

    g_roi = _warp_to_grid(guide, A, out_w, out_h)
    rect_pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)

    M_cur = M.copy()
    stats = None
    for _ in range(max(1, iterations)):
        i_roi = _warp_to_grid(image, A @ M_cur, out_w, out_h)
        valid = cv2.erode((i_roi > 0).astype(np.uint8) * 255,
                          np.ones((7, 7), np.uint8))
        H, n_good, n_inl, rmse = _match_and_estimate(
            g_roi, i_roi, model, max_features, match_ratio, ransac_px,
            g_mask=None, i_mask=valid)
        M_new = A_inv @ H @ A @ M_cur
        M_new /= M_new[2, 2]
        step = float(np.abs(cv2.perspectiveTransform(rect_pts, M_new @ np.linalg.inv(M_cur))
                            - rect_pts).max())
        M_cur = M_new
        stats = CoarseAlignment(matrix=M_cur, n_matches=n_good, n_inliers=n_inl,
                                rmse_px=rmse / s, scale=s)
        if step < 0.3:
            break

    correction = float(np.abs(cv2.perspectiveTransform(rect_pts, M_cur @ np.linalg.inv(M))
                              - rect_pts).max())
    logger.info(f"Local refinement (scale {s:.3f}): {stats.n_inliers}/{stats.n_matches} "
                f"inliers, RMSE {stats.rmse_px:.2f} px, correction {correction:.1f} px")
    return M_cur, stats, correction


# ----------------------------------------------------------------------
# Cropping
# ----------------------------------------------------------------------

def clamp_rect(rect, shape) -> Tuple[int, int, int, int]:
    """Clamp (x0, y0, x1, y1) to the image and validate it."""
    h, w = shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError("Il rettangolo di ritaglio e' troppo piccolo.")
    return x0, y0, x1, y1


def warp_rect_from_image(image: np.ndarray, M: np.ndarray,
                         rect) -> Tuple[np.ndarray, float]:
    """Resample *image* into the guide rectangle using M (image -> guide).

    Only the rectangle is computed, so this is cheap even for huge images.
    Returns (crop, coverage_pct) where coverage is the share of the
    rectangle that falls inside *image*.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    M_rect = T @ M
    crop = cv2.warpPerspective(image, M_rect, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Coverage: footprint of the image corners inside the rectangle
    ih, iw = image.shape[:2]
    corners = np.float32([[0, 0], [iw, 0], [iw, ih], [0, ih]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, M_rect).reshape(-1, 2)
    cover = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(cover, [np.round(quad).astype(np.int32)], 255)
    coverage = 100.0 * float(np.count_nonzero(cover)) / cover.size
    return crop, coverage


def _write_png(path: str, img: np.ndarray):
    ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"Impossibile codificare {path}")
    buf.tofile(path)  # works with non-ASCII paths


def sidecar_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + SIDECAR_SUFFIX


def read_sidecar(image_path: str) -> Optional[dict]:
    p = sidecar_path(image_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Sidecar non leggibile {p}: {e}")
        return None


def _source_metadata(path: str) -> dict:
    """EXIF metadata of the source photo (kept because PNG crops lose it)."""
    from dic_app.io.image_loader import ImageLoader
    meta = {}
    try:
        gps = ImageLoader._extract_exif_gps(path)
        if gps is not None:
            meta['gps'] = {'latitude': gps.latitude, 'longitude': gps.longitude,
                           'altitude': gps.altitude, 'timestamp': gps.timestamp}
        meta.update(ImageLoader._extract_camera_info(path))
    except Exception as e:
        logger.debug(f"No metadata for {path}: {e}")
    return meta


def output_name(source_path: str, out_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(out_dir, f"{stem}_crop.png")


def run_precrop(guide_path: str, image_paths: List[str], rect,
                out_dir: str, max_dim: int = 2000, model: str = 'homography',
                refine: bool = True,
                progress: Optional[Callable[[int, str], None]] = None,
                is_cancelled: Optional[Callable[[], bool]] = None
                ) -> List[CropOutcome]:
    """Crop the guide to *rect* and align+crop every image in *image_paths*.

    rect is (x0, y0, x1, y1) in full-resolution guide pixels.
    Returns one CropOutcome per image (guide first).  Images that fail are
    reported with .error and skipped; the run continues with the others.
    """
    def report(pct, msg):
        logger.info(msg)
        if progress:
            progress(int(pct), msg)

    os.makedirs(out_dir, exist_ok=True)
    n_steps = 1 + len(image_paths)

    report(0, f"Lettura guida {os.path.basename(guide_path)}...")
    guide = read_color(guide_path)
    rect = clamp_rect(rect, guide.shape)
    x0, y0, x1, y1 = rect
    gh, gw = guide.shape[:2]

    outcomes = []
    guide_out = output_name(guide_path, out_dir)
    _write_png(guide_out, guide[y0:y1, x0:x1])
    _write_sidecar(guide_out, guide_path, guide_path, rect, (gh, gw),
                   np.eye(3), None, 100.0, is_guide=True)
    outcomes.append(CropOutcome(source_path=guide_path, output_path=guide_out,
                                is_guide=True))
    report(100 / n_steps, f"Guida ritagliata: {x1 - x0}x{y1 - y0} px")

    for k, path in enumerate(image_paths, start=1):
        if is_cancelled and is_cancelled():
            report(100, "Ritaglio annullato")
            break
        name = os.path.basename(path)
        outcome = CropOutcome(source_path=path)
        try:
            report(100 * (k - 0.9) / n_steps, f"[{k}/{len(image_paths)}] Lettura {name}...")
            img = read_color(path)
            report(100 * (k - 0.6) / n_steps, f"[{k}/{len(image_paths)}] Allineamento grossolano {name}...")
            al = coarse_align(guide, img, max_dim=max_dim, model=model)
            if refine:
                report(100 * (k - 0.45) / n_steps,
                       f"[{k}/{len(image_paths)}] Affinamento locale {name}...")
                try:
                    # The local area is small, so it can be matched at a much
                    # higher resolution: on real orthophotos 4000 px brought
                    # the residual from ~4 px to <1 px for ~3 s extra.
                    M_ref, al_ref, corr = refine_local(
                        guide, img, al.matrix, rect,
                        max_dim=max(4000, max_dim), model=model)
                    if al_ref.n_inliers >= 12:
                        al_ref.global_alignment = al
                        al_ref.refine_correction_px = corr
                        al = al_ref
                    else:
                        outcome.warnings.append(
                            f"affinamento locale scartato (solo {al_ref.n_inliers} "
                            f"punti): resta l'allineamento grossolano")
                except Exception as e:
                    outcome.warnings.append(
                        f"affinamento locale non riuscito ({e}): resta "
                        f"l'allineamento grossolano")
            outcome.alignment = al
            report(100 * (k - 0.3) / n_steps, f"[{k}/{len(image_paths)}] Ritaglio {name}...")
            crop, coverage = warp_rect_from_image(img, al.matrix, rect)
            outcome.coverage_pct = coverage
            if coverage < 99.0:
                outcome.warnings.append(
                    f"il rettangolo esce dalla foto per il {100 - coverage:.1f}% "
                    f"(zone nere nel ritaglio)")
            if al.n_inliers < 30:
                outcome.warnings.append(
                    f"solo {al.n_inliers} punti di controllo: verificare l'allineamento")
            # RMSE above ~1.5 px at matching resolution = poor fit
            if al.rmse_px * al.scale > 1.5:
                outcome.warnings.append(
                    f"errore di allineamento elevato ({al.rmse_px:.1f} px)")
            out = output_name(path, out_dir)
            _write_png(out, crop)
            _write_sidecar(out, path, guide_path, rect, img.shape[:2],
                           al.matrix, al, coverage, is_guide=False)
            outcome.output_path = out
            del img, crop
        except Exception as e:
            outcome.error = str(e)
            logger.error(f"Precrop failed for {path}: {e}")
        outcomes.append(outcome)
        report(100 * k / n_steps,
               f"[{k}/{len(image_paths)}] {name}: "
               + ("OK" if outcome.ok else f"ERRORE - {outcome.error}"))

    return outcomes


def _write_sidecar(out_path, source_path, guide_path, rect, source_shape,
                   matrix, alignment: Optional[CoarseAlignment], coverage,
                   is_guide: bool):
    x0, y0, x1, y1 = rect
    data = {
        'tool': 'ROCK-DIC preliminary crop',
        'source_path': os.path.abspath(source_path),
        'guide_path': os.path.abspath(guide_path),
        'is_guide': is_guide,
        # Crop box in GUIDE full-resolution pixels; for the guide itself this
        # is also the position of the crop inside its own original photo.
        'crop_rect_guide': [x0, y0, x1, y1],
        'guide_original_shape': None,
        'source_original_shape': list(source_shape),
        # 3x3 matrix mapping source pixels -> guide pixels
        'source_to_guide': np.asarray(matrix, dtype=float).tolist(),
        'coverage_pct': round(float(coverage), 3),
        'source_metadata': _source_metadata(source_path),
    }
    if is_guide:
        data['guide_original_shape'] = list(source_shape)
    try:
        from dic_app.io.image_loader import ImageLoader
        g = ImageLoader._extract_geotiff_gsd(guide_path)
        if g:
            data['guide_gsd'] = g
    except Exception:
        pass
    if alignment is not None:
        data['alignment'] = {
            'n_matches': alignment.n_matches,
            'n_inliers': alignment.n_inliers,
            'rmse_px': round(alignment.rmse_px, 3),
            'downscale': round(alignment.scale, 5),
            'local_refinement': alignment.global_alignment is not None,
        }
        if alignment.global_alignment is not None:
            ga = alignment.global_alignment
            data['alignment']['refine_correction_px'] = round(
                alignment.refine_correction_px, 2)
            data['alignment']['global'] = {
                'n_matches': ga.n_matches, 'n_inliers': ga.n_inliers,
                'rmse_px': round(ga.rmse_px, 3), 'downscale': round(ga.scale, 5)}
    with open(sidecar_path(out_path), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
