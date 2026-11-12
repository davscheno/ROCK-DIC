"""Subset-based DIC building blocks.

* ``ncc_search``      - integer ZNCC search + sub-pixel peak fit, per point,
                        parallel on all CPU cores (OpenCV releases the GIL).
* ``pyramid_guess``   - coarse-to-fine initial guess, so large displacements
                        do not need a huge search radius.
* ``icgn_refine``     - Inverse-Compositional Gauss-Newton with a first-order
                        (affine) subset shape function and the ZNSSD
                        criterion, computed for thousands of subsets at once.
                        This is the standard of scientific DIC: the subset
                        may stretch, shear and rotate, which is what rock
                        does inside a moving landslide body, and it is
                        insensitive to brightness/contrast changes.
* ``local_normalize`` - per-pixel local mean/contrast normalisation used to
                        see through appearance changes (wet rock, shadows).
* noise / SSSIG helpers used to choose the subset size from the image.

References: Pan et al., Exp. Mech. 2013 (IC-GN), Pan et al., Opt. Lasers
Eng. 2008 (SSSIG), Immerkaer, CVIU 1996 (noise estimation).
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import cv2

from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)


def n_threads(requested: int = 0) -> int:
    if requested and requested > 0:
        return requested
    return max(1, min(16, (os.cpu_count() or 2)))


# ----------------------------------------------------------------------
# Image representations
# ----------------------------------------------------------------------

def to_float(img: np.ndarray, prefilter_sigma: float = 0.0) -> np.ndarray:
    f = img.astype(np.float32)
    if prefilter_sigma and prefilter_sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), prefilter_sigma)
    return f


def local_normalize(img: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """(I - local mean) / local std  ->  float32, ~zero mean, unit contrast.

    Removes smooth changes of brightness and contrast (wet rock, shadows,
    different exposure) while keeping the shape of the texture.
    """
    f = img.astype(np.float32)
    mean = cv2.GaussianBlur(f, (0, 0), sigma)
    d = f - mean
    std = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), sigma))
    return d / (std + 2.0)


def gradients(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4th-order central differences (more accurate than Sobel for IC-GN)."""
    k = np.array([[1.0, -8.0, 0.0, 8.0, -1.0]], dtype=np.float32) / 12.0
    gx = cv2.filter2D(f, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(f, cv2.CV_32F, k.T, borderType=cv2.BORDER_REFLECT)
    return gx, gy


def estimate_noise(img: np.ndarray) -> float:
    """Gaussian noise sigma (grey levels), Immerkaer's fast method."""
    f = img.astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REFLECT)
    # The kernel has L2 norm 6, so for pure noise conv ~ N(0, 6 sigma).
    # The median of |conv| is robust to texture edges, unlike the mean.
    sigma = np.median(np.abs(conv[1:-1, 1:-1])) / (6.0 * 0.6745)
    return float(max(sigma, 0.3))


def sssig(img: np.ndarray, cx: np.ndarray, cy: np.ndarray, half: int) -> np.ndarray:
    """Sum of squared subset intensity gradients, min of x and y (per point)."""
    gx, gy = gradients(img.astype(np.float32))
    side = 2 * half + 1
    sx = cv2.boxFilter(gx * gx, -1, (side, side), normalize=False,
                       borderType=cv2.BORDER_REFLECT)
    sy = cv2.boxFilter(gy * gy, -1, (side, side), normalize=False,
                       borderType=cv2.BORDER_REFLECT)
    return np.minimum(sx[cy, cx], sy[cy, cx])


def predicted_std_px(noise_sigma: float, sssig_vals: np.ndarray) -> np.ndarray:
    """Expected displacement std (px) of a subset: sqrt(2)*sigma/sqrt(SSSIG)."""
    return np.sqrt(2.0) * noise_sigma / np.sqrt(np.maximum(sssig_vals, 1e-6))


# ----------------------------------------------------------------------
# Integer NCC search with sub-pixel peak fit
# ----------------------------------------------------------------------

def _peak_fit(c_m, c_0, c_p, gaussian=True):
    if gaussian and c_m > 0 and c_0 > 0 and c_p > 0:
        lm, l0, lp = np.log(c_m), np.log(c_0), np.log(c_p)
        den = 2.0 * (lm - 2 * l0 + lp)
        if abs(den) > 1e-12:
            return float(np.clip((lm - lp) / den, -0.5, 0.5))
    den = 2.0 * (c_m - 2 * c_0 + c_p)
    if abs(den) > 1e-12:
        return float(np.clip((c_m - c_p) / den, -0.5, 0.5))
    return 0.0


def _ncc_chunk(ref, deff, cx, cy, half, gu, gv, rx, ry, subpixel, cancel):
    n = len(cx)
    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    h, w = ref.shape
    for k in range(n):
        if cancel is not None and cancel():
            break
        x, y = int(cx[k]), int(cy[k])
        t = ref[y - half:y + half + 1, x - half:x + half + 1]
        if t.shape != (2 * half + 1, 2 * half + 1) or t.std() < 1e-3:
            continue
        ox = int(round(gu[k])) if np.isfinite(gu[k]) else 0
        oy = int(round(gv[k])) if np.isfinite(gv[k]) else 0
        sx0, sx1 = x + ox - half - rx, x + ox + half + rx + 1
        sy0, sy1 = y + oy - half - ry, y + oy + half + ry + 1
        cx0, cy0 = max(0, sx0), max(0, sy0)
        cx1, cy1 = min(w, sx1), min(h, sy1)
        win = deff[cy0:cy1, cx0:cx1]
        if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
            continue
        res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        px, py = ml
        dx = dy = 0.0
        if subpixel and 0 < px < res.shape[1] - 1:
            dx = _peak_fit(res[py, px - 1], res[py, px], res[py, px + 1])
        if subpixel and 0 < py < res.shape[0] - 1:
            dy = _peak_fit(res[py - 1, px], res[py, px], res[py + 1, px])
        # top-left of the match -> displacement of the subset centre
        u[k] = (cx0 + px + dx + half) - x
        v[k] = (cy0 + py + dy + half) - y
        peak[k] = mv
    return u, v, peak


def ncc_search(ref: np.ndarray, deff: np.ndarray, cx: np.ndarray, cy: np.ndarray,
               half: int, guess_u=None, guess_v=None, radius_x: int = 10,
               radius_y: int = 10, subpixel: bool = True, threads: int = 0,
               cancel=None):
    """ZNCC template search around (cx+gu, cy+gv) for every point.

    ref, deff must be the same dtype (uint8 or float32).
    Returns (u, v, peak_zncc), NaN where the subset/window is unusable.
    """
    n = len(cx)
    gu = np.zeros(n) if guess_u is None else np.asarray(guess_u, float)
    gv = np.zeros(n) if guess_v is None else np.asarray(guess_v, float)
    nt = n_threads(threads)
    if n < 64 or nt == 1:
        return _ncc_chunk(ref, deff, cx, cy, half, gu, gv, radius_x, radius_y,
                          subpixel, cancel)
    bounds = np.linspace(0, n, nt * 4 + 1).astype(int)
    parts = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] > bounds[i]]
    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    pk = np.full(n, np.nan)
    with ThreadPoolExecutor(max_workers=nt) as ex:
        futs = {ex.submit(_ncc_chunk, ref, deff, cx[a:b], cy[a:b], half,
                          gu[a:b], gv[a:b], radius_x, radius_y, subpixel,
                          cancel): (a, b) for a, b in parts}
        for f, (a, b) in futs.items():
            u[a:b], v[a:b], pk[a:b] = f.result()
    return u, v, pk


# ----------------------------------------------------------------------
# Robust field cleaning (for guesses)
# ----------------------------------------------------------------------

def median_test_fill(u: np.ndarray, v: np.ndarray, threshold: float = 2.0,
                     eps: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised median test (Westerweel & Scarano) on a 2D grid.

    Outliers and NaN are replaced by the median of their 3x3 neighbours
    (only for use as *initial guesses*, never as results).
    Returns (u, v, outlier_mask).
    """
    ny, nx = u.shape
    pad_u = np.pad(u, 1, constant_values=np.nan)
    pad_v = np.pad(v, 1, constant_values=np.nan)
    neigh_u = np.stack([pad_u[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if not (dx == 0 and dy == 0)])
    neigh_v = np.stack([pad_v[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if not (dx == 0 and dy == 0)])
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu = np.nanmedian(neigh_u, axis=0)
            mv = np.nanmedian(neigh_v, axis=0)
            ru = np.nanmedian(np.abs(neigh_u - mu), axis=0)
            rv = np.nanmedian(np.abs(neigh_v - mv), axis=0)
    r = np.sqrt(((u - mu) / (ru + eps)) ** 2 + ((v - mv) / (rv + eps)) ** 2)
    out = ~(r <= threshold)  # includes NaN
    uo, vo = u.copy(), v.copy()
    uo[out] = mu[out]
    vo[out] = mv[out]
    return uo, vo, out


def median_outliers(u: np.ndarray, v: np.ndarray, threshold: float = 3.0,
                    eps: float = 0.2, radius: int = 2) -> np.ndarray:
    """Normalised median test on a 2D grid (NaN-aware), returns outlier mask.

    Uses a (2r+1)^2 neighbourhood without the centre.  Points with fewer
    than 4 valid neighbours are not judged.
    """
    import warnings
    ny, nx = u.shape
    r = radius
    pu = np.pad(u, r, constant_values=np.nan)
    pv = np.pad(v, r, constant_values=np.nan)
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if not (dx == 0 and dy == 0)]
    nu = np.stack([pu[r + dy:r + dy + ny, r + dx:r + dx + nx] for dy, dx in offs])
    nv = np.stack([pv[r + dy:r + dy + ny, r + dx:r + dx + nx] for dy, dx in offs])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu, mv = np.nanmedian(nu, 0), np.nanmedian(nv, 0)
        ru = np.nanmedian(np.abs(nu - mu), 0)
        rv = np.nanmedian(np.abs(nv - mv), 0)
        cnt = np.sum(np.isfinite(nu), 0)
        res = np.sqrt(((u - mu) / (ru + eps)) ** 2 + ((v - mv) / (rv + eps)) ** 2)
    return np.isfinite(u) & (cnt >= 4) & (res > threshold)


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    m = np.isnan(a)
    if not m.any() or m.all():
        return np.where(m, 0.0, a)
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]


# ----------------------------------------------------------------------
# Coarse-to-fine initial guess
# ----------------------------------------------------------------------

def pyramid_guess(ref: np.ndarray, deff: np.ndarray, grid_x: np.ndarray,
                  grid_y: np.ndarray, half: int, max_disp: float,
                  radius: int = 6, threads: int = 0, cancel=None,
                  progress=None):
    """Displacement guess for every grid node, robust to large motions.

    Works on an image pyramid: at the coarsest level a displacement of
    *max_disp* px becomes a few pixels, so a small search radius is enough;
    each finer level only corrects the previous estimate.
    Returns (gu, gv) on the grid (full-resolution px), NaN-free.
    """
    ny, nx = grid_x.shape
    levels = 0
    while max_disp / (2 ** levels) > radius and levels < 6:
        levels += 1
    gu = np.zeros((ny, nx))
    gv = np.zeros((ny, nx))
    if levels == 0:
        return gu, gv, 0

    pyr_r, pyr_d = [ref], [deff]
    for _ in range(levels):
        pyr_r.append(cv2.pyrDown(pyr_r[-1]))
        pyr_d.append(cv2.pyrDown(pyr_d[-1]))

    for lev in range(levels, 0, -1):
        if cancel is not None and cancel():
            break
        s = 2.0 ** -lev
        r_img, d_img = pyr_r[lev], pyr_d[lev]
        h, w = r_img.shape
        hl = int(max(7, round(half * s * 1.5)))
        x = np.clip(np.round((grid_x + 0.5) * s - 0.5).astype(int), hl, w - hl - 1)
        y = np.clip(np.round((grid_y + 0.5) * s - 0.5).astype(int), hl, h - hl - 1)
        rr = radius + (int(np.ceil(max_disp * s)) if lev == levels else 2)
        u, v, pk = ncc_search(r_img, d_img, x.ravel(), y.ravel(), hl,
                              (gu * s).ravel(), (gv * s).ravel(), rr, rr,
                              subpixel=False, threads=threads, cancel=cancel)
        u = u.reshape(ny, nx) / s
        v = v.reshape(ny, nx) / s
        bad = ~(pk.reshape(ny, nx) > 0.3)
        u[bad] = np.nan
        v[bad] = np.nan
        u, v, _ = median_test_fill(u, v, threshold=2.5, eps=0.5 / s)
        gu, gv = _fill_nan_nearest(u), _fill_nan_nearest(v)
        if progress:
            progress(lev)
    return gu, gv, levels


# ----------------------------------------------------------------------
# IC-GN refinement (batched)
# ----------------------------------------------------------------------

def _warp_from_p(p):
    """p = (u, ux, uy, v, vx, vy) -> 3x3 affine warp, batched (N,6)->(N,3,3)."""
    n = p.shape[0]
    W = np.zeros((n, 3, 3))
    W[:, 0, 0] = 1 + p[:, 1]
    W[:, 0, 1] = p[:, 2]
    W[:, 0, 2] = p[:, 0]
    W[:, 1, 0] = p[:, 4]
    W[:, 1, 1] = 1 + p[:, 5]
    W[:, 1, 2] = p[:, 3]
    W[:, 2, 2] = 1.0
    return W


def _sample(img: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Bicubic sampling at (N,S) coordinates in one OpenCV call per chunk."""
    return cv2.remap(img, X.astype(np.float32), Y.astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0.0)


def _zncc_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 1e-9, (a * b).sum(1) / np.maximum(den, 1e-9), np.nan)


def _icgn_chunk(ref_f, def_f, gx, gy, cx, cy, u0, v0, dx, dy, half,
                max_iter, tol, out, sel_all, cancel):
    """IC-GN for one block of points; writes into the shared *out* dict.

    Per-pixel arrays are float32 (memory bandwidth is the bottleneck);
    6x6 systems and warps stay in float64.
    """
    if cancel is not None and cancel():
        return
    h, w = def_f.shape
    side = 2 * half + 1
    rad = float(half)
    S = dx.size
    ok0 = np.isfinite(u0[sel_all]) & np.isfinite(v0[sel_all])
    ok0 &= (cx[sel_all] - half >= 0) & (cx[sel_all] + half < ref_f.shape[1])
    ok0 &= (cy[sel_all] - half >= 0) & (cy[sel_all] + half < ref_f.shape[0])
    sel = sel_all[ok0]
    if sel.size == 0:
        return
    pcx = cx[sel].astype(np.float32)
    pcy = cy[sel].astype(np.float32)
    Xi = (cx[sel][:, None] + dx.astype(int)[None, :])
    Yi = (cy[sel][:, None] + dy.astype(int)[None, :])
    f = ref_f[Yi, Xi]
    fx = gx[Yi, Xi]
    fy = gy[Yi, Xi]
    fc = f - f.mean(1, keepdims=True)
    df = np.sqrt((fc * fc).sum(1, dtype=np.float64))
    good = df > 1e-3 * np.sqrt(S)
    SD = np.stack([fx, fx * dx, fx * dy, fy, fy * dx, fy * dy], axis=2)  # (n,S,6) f32
    SDt = np.ascontiguousarray(SD.transpose(0, 2, 1))                      # (n,6,S)
    H = (SDt.astype(np.float64) @ SD.astype(np.float64))
    tr = np.trace(H, axis1=1, axis2=2)[:, None, None]
    H = H + np.eye(6)[None] * (1e-9 * tr + 1e-12)   # keeps flat subsets invertible
    Hinv = np.linalg.inv(H)

    p = np.zeros((sel.size, 6))
    p[:, 0] = u0[sel]
    p[:, 3] = v0[sel]
    W = _warp_from_p(p)
    active = good.copy()
    iters = np.zeros(sel.size, int)
    conv = np.zeros(sel.size, bool)
    u_init, v_init = u0[sel], v0[sel]
    for _ in range(max_iter):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        Wa = W[idx].astype(np.float32)
        Xw = Wa[:, 0, 0:1] * dx + Wa[:, 0, 1:2] * dy + (Wa[:, 0, 2] + pcx[idx])[:, None]
        Yw = Wa[:, 1, 0:1] * dx + Wa[:, 1, 1:2] * dy + (Wa[:, 1, 2] + pcy[idx])[:, None]
        inside = ((Xw.min(1) >= 1) & (Xw.max(1) <= w - 2) &
                  (Yw.min(1) >= 1) & (Yw.max(1) <= h - 2))
        ga = _sample(def_f, Xw, Yw)
        gc = ga - ga.mean(1, keepdims=True)
        dg = np.sqrt((gc * gc).sum(1, dtype=np.float64))
        ratio = np.where(dg > 1e-9, df[idx] / np.maximum(dg, 1e-9), 0.0).astype(np.float32)
        r = fc[idx] - ratio[:, None] * gc
        bvec = (SDt[idx] @ r[:, :, None])[:, :, 0].astype(np.float64)
        dp = -(Hinv[idx] @ bvec[:, :, None])[:, :, 0]
        W[idx] = W[idx] @ np.linalg.inv(_warp_from_p(dp))
        iters[idx] += 1
        step = np.sqrt(dp[:, 0] ** 2 + dp[:, 3] ** 2 +
                       (rad * dp[:, 1]) ** 2 + (rad * dp[:, 2]) ** 2 +
                       (rad * dp[:, 4]) ** 2 + (rad * dp[:, 5]) ** 2)
        done = step < tol
        lost = (~inside | ~np.isfinite(step)
                | (np.abs(W[idx, 0, 2] - u_init[idx]) > side)
                | (np.abs(W[idx, 1, 2] - v_init[idx]) > side))
        conv[idx[done & ~lost]] = True
        active[idx[done | lost]] = False
        good[idx[lost]] = False          # left the image or diverged

    Wf = W.astype(np.float32)
    Xw = Wf[:, 0, 0:1] * dx + Wf[:, 0, 1:2] * dy + (Wf[:, 0, 2] + pcx)[:, None]
    Yw = Wf[:, 1, 0:1] * dx + Wf[:, 1, 1:2] * dy + (Wf[:, 1, 2] + pcy)[:, None]
    zn = _zncc_rows(f, _sample(def_f, Xw, Yw))
    valid = good & np.isfinite(zn)
    tgt = sel[valid]
    Wv = W[valid]
    out['u'][tgt] = Wv[:, 0, 2]
    out['v'][tgt] = Wv[:, 1, 2]
    out['ux'][tgt] = Wv[:, 0, 0] - 1
    out['uy'][tgt] = Wv[:, 0, 1]
    out['vx'][tgt] = Wv[:, 1, 0]
    out['vy'][tgt] = Wv[:, 1, 1] - 1
    out['zncc'][tgt] = zn[valid]
    out['iters'][sel] = iters
    out['converged'][tgt] = conv[valid]
    out['warp'][tgt] = Wv


def icgn_refine(ref_f: np.ndarray, def_f: np.ndarray, cx: np.ndarray,
                cy: np.ndarray, half: int, u0: np.ndarray, v0: np.ndarray,
                grads: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                max_iter: int = 30, tol: float = 1e-3, chunk: int = 256,
                threads: int = 0, cancel=None) -> dict:
    """IC-GN with first-order shape function, ZNSSD criterion.

    ref_f, def_f : float32 images (same representation)
    cx, cy       : integer subset centres (N,)
    u0, v0       : initial displacement (N,), NaN = skip
    Returns dict of (N,) arrays: u, v, ux, uy, vx, vy, zncc, iters,
    converged, and 'warp' (N,3,3) for re-evaluation on other images.
    Blocks of points run in parallel threads (NumPy/OpenCV release the GIL).
    """
    n = len(cx)
    out = {k: np.full(n, np.nan) for k in
           ('u', 'v', 'ux', 'uy', 'vx', 'vy', 'zncc')}
    out['iters'] = np.zeros(n, int)
    out['converged'] = np.zeros(n, bool)
    out['warp'] = np.full((n, 3, 3), np.nan)
    if n == 0:
        return out
    ref_f = np.ascontiguousarray(ref_f, dtype=np.float32)
    def_f = np.ascontiguousarray(def_f, dtype=np.float32)
    gx, gy = grads if grads is not None else gradients(ref_f)
    dyg, dxg = np.mgrid[-half:half + 1, -half:half + 1]
    dx = dxg.ravel().astype(np.float32)
    dy = dyg.ravel().astype(np.float32)
    cx = np.asarray(cx, int)
    cy = np.asarray(cy, int)
    u0 = np.asarray(u0, float)
    v0 = np.asarray(v0, float)
    blocks = [np.arange(a, min(n, a + chunk)) for a in range(0, n, chunk)]
    nt = n_threads(threads)
    if nt == 1 or len(blocks) == 1:
        for sel in blocks:
            _icgn_chunk(ref_f, def_f, gx, gy, cx, cy, u0, v0, dx, dy, half,
                        max_iter, tol, out, sel, cancel)
    else:
        with ThreadPoolExecutor(max_workers=nt) as ex:
            list(ex.map(lambda sel: _icgn_chunk(
                ref_f, def_f, gx, gy, cx, cy, u0, v0, dx, dy, half,
                max_iter, tol, out, sel, cancel), blocks))
    return out


def zncc_at_warp(ref_img: np.ndarray, def_img: np.ndarray, cx: np.ndarray,
                 cy: np.ndarray, half: int, warp: np.ndarray,
                 chunk: int = 2000) -> np.ndarray:
    """ZNCC between ref subsets and def sampled through given warps."""
    n = len(cx)
    out = np.full(n, np.nan)
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
    dx = dx.ravel().astype(np.float64)
    dy = dy.ravel().astype(np.float64)
    ok = np.all(np.isfinite(warp.reshape(n, -1)), axis=1)
    idx_all = np.nonzero(ok)[0]
    for a in range(0, idx_all.size, chunk):
        idx = idx_all[a:a + chunk]
        W = warp[idx]
        X0 = (cx[idx][:, None] + dx).astype(int)
        Y0 = (cy[idx][:, None] + dy).astype(int)
        f = ref_img[Y0, X0].astype(np.float64)
        Xw = W[:, 0, 0:1] * dx + W[:, 0, 1:2] * dy + W[:, 0, 2:3] + cx[idx][:, None]
        Yw = W[:, 1, 0:1] * dx + W[:, 1, 1:2] * dy + W[:, 1, 2:3] + cy[idx][:, None]
        g = _sample(def_img, Xw, Yw).astype(np.float64)
        out[idx] = _zncc_rows(f, g)
    return out
