"""Guided choice of DIC parameters and a-priori precision estimate.

* ``suggest_parameters``  chooses subset, step and maximum displacement from
  measurable image properties instead of fixed thresholds:
    - image noise (Immerkaer) and SSSIG (sum of squared subset intensity
      gradients) give the expected displacement error of every subset size
      (Pan et al. 2008):  sigma_u ~ sqrt(2) * noise / sqrt(SSSIG);
    - the smallest subset reaching the target precision on most of the
      rock is chosen, capped by the size of the smallest phenomenon;
    - the maximum displacement comes from the user (metres, via GSD) or
      from a quick coarse-to-fine estimate.
* ``estimate_precision`` runs the chosen settings on a synthetic copy of
  the user's own image with a known sub-pixel motion and the measured noise
  level, and reports the real error and the time for the full analysis.
"""

import time
from typing import Optional

import numpy as np
import cv2

from dic_app.core import subset_dic as sd
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)

CANDIDATE_SUBSETS = (21, 25, 31, 41, 51, 61, 71, 81, 101, 121)


def _sample_points(img: np.ndarray, half_max: int, n: int = 2500):
    h, w = img.shape
    m = half_max + 2
    if h <= 2 * m or w <= 2 * m:
        m = 2
    step = max(4, int(np.sqrt((h - 2 * m) * (w - 2 * m) / n)))
    ys, xs = np.mgrid[m:h - m:step, m:w - m:step]
    xs, ys = xs.ravel(), ys.ravel()
    side = 31
    nod = cv2.boxFilter((img == 0).astype(np.float32), -1, (side, side))[ys, xs]
    keep = nod < 0.01
    return xs[keep], ys[keep]


def estimate_max_displacement(ref: np.ndarray, deff: np.ndarray) -> float:
    """Quick coarse-to-fine estimate of the largest displacement (px)."""
    h, w = ref.shape
    step = max(40, min(h, w) // 25)
    half = 20
    gx, gy = np.meshgrid(np.arange(half + 1, w - half - 1, step),
                         np.arange(half + 1, h - half - 1, step))
    gu, gv, _ = sd.pyramid_guess(ref, deff, gx, gy, half,
                                 max_disp=min(h, w) / 8.0, radius=6)
    # Verify every guess at full resolution and keep only points that
    # really correlate: water, vegetation and changed areas give random
    # guesses that would inflate the maximum.
    u, v, pk = sd.ncc_search(ref, deff, gx.ravel(), gy.ravel(), half,
                             gu.ravel(), gv.ravel(), 3, 3)
    good = pk > 0.7
    if good.sum() < 10:
        return 0.0
    mag = np.hypot(u[good], v[good])
    return float(np.percentile(mag, 98))


def _pair_texture(ref, deff, xs, ys, half):
    """Median SSSIG of both images at the same points (None for missing deff)."""
    m_r = float(np.median(sd.sssig(ref, xs, ys, half)))
    if deff is None:
        return m_r, None
    return m_r, float(np.median(sd.sssig(deff, xs, ys, half)))


def weaker_image(ref: np.ndarray, deff: Optional[np.ndarray]):
    """Return (image, 'ref'|'def') with less usable texture.

    When the two surveys have a different resolution or sharpness (e.g. a
    blurry flight resampled onto a sharp orthophoto) the precision is set
    by the WORSE image, so tests must be run on that one.
    """
    if deff is None or deff.shape != ref.shape:
        return ref, 'ref'
    xs, ys = _sample_points(ref, 16, n=1500)
    if len(xs) == 0:
        return ref, 'ref'
    m_r, m_d = _pair_texture(ref, deff, xs, ys, 15)
    return (deff, 'def') if m_d < m_r else (ref, 'ref')


def suggest_parameters(ref: np.ndarray, deff: Optional[np.ndarray] = None,
                       gsd: Optional[float] = None,
                       max_disp_m: Optional[float] = None,
                       min_feature_m: Optional[float] = None,
                       target_px: float = 0.02,
                       coverage: float = 0.8) -> dict:
    """Return suggested settings plus the reasons, as a dict."""
    ref = np.ascontiguousarray(ref)
    h, w = ref.shape
    pair = deff is not None and deff.shape == ref.shape
    if pair:
        deff = np.ascontiguousarray(deff)
    noise_r = sd.estimate_noise(ref)
    noise_d = sd.estimate_noise(deff) if pair else noise_r
    noise = max(noise_r, noise_d)
    # sqrt(2)*sigma in the single-image formula stands for the noise of both
    # images; blur also lowers the measured noise, so use the larger one
    noise_eq = noise
    half_max = CANDIDATE_SUBSETS[-1] // 2
    xs, ys = _sample_points(ref, half_max)
    if pair and len(xs):
        side = 31
        nod_d = cv2.boxFilter((deff == 0).astype(np.float32), -1, (side, side))[ys, xs]
        xs, ys = xs[nod_d < 0.01], ys[nod_d < 0.01]
    if pair:
        reasons = [f"Rumore stimato: riferimento {noise_r:.1f}, deformata "
                   f"{noise_d:.1f} livelli di grigio."]
        m_r, m_d = _pair_texture(ref, deff, xs, ys, 15) if len(xs) else (1.0, 1.0)
        ratio = m_d / max(m_r, 1e-6)
        if ratio < 0.7 or ratio > 1.0 / 0.7:
            worse = "deformata" if ratio < 1 else "di riferimento"
            reasons.append(
                f"Le due immagini hanno nitidezza diversa (texture {worse} "
                f"{min(ratio, 1 / ratio) * 100:.0f}% dell'altra): il calcolo usa, "
                f"punto per punto, l'immagine meno nitida.")
    else:
        reasons = [f"Rumore dell'immagine stimato: {noise:.1f} livelli di grigio."]

    # Upper bound from the smallest phenomenon to resolve (>= 2 subsets)
    max_subset = CANDIDATE_SUBSETS[-1]
    if gsd and min_feature_m:
        max_subset = int(min_feature_m / gsd / 2.0)
        reasons.append(f"Fenomeno minimo {min_feature_m:g} m = "
                       f"{min_feature_m / gsd:.0f} px: subset al massimo {max_subset} px.")
    max_subset = max(CANDIDATE_SUBSETS[0], min(max_subset, min(h, w) // 3))

    chosen, table = None, []
    for s in CANDIDATE_SUBSETS:
        if s > max_subset:
            break
        ss = sd.sssig(ref, xs, ys, s // 2)
        if pair:
            ss = np.minimum(ss, sd.sssig(deff, xs, ys, s // 2))
        sig = sd.predicted_std_px(noise_eq, ss)
        frac = float(np.mean(sig <= target_px))
        table.append((s, frac, float(np.median(sig))))
        if chosen is None and frac >= coverage:
            chosen = s
    if chosen is None:
        # best compromise: largest allowed subset
        chosen = table[-1][0] if table else 41
        reasons.append(f"Nessun subset raggiunge {target_px} px sull'{coverage * 100:.0f}% "
                       f"della roccia entro il limite: uso il piu' grande consentito.")
    row = next(t for t in table if t[0] == chosen) if table else (chosen, 0, 0)
    reasons.append(f"Subset {chosen} px: errore previsto {row[2]:.3f} px (mediana), "
                   f"{row[1] * 100:.0f}% dei punti entro {target_px} px.")

    step = max(3, chosen // 3)
    n_pts = ((h - chosen) // step) * ((w - chosen) // step)
    while n_pts > 250_000:
        step += 1
        n_pts = ((h - chosen) // step) * ((w - chosen) // step)
    reasons.append(f"Passo griglia {step} px (~1/3 del subset): circa {n_pts:,} punti."
                   .replace(",", "."))

    if max_disp_m and gsd:
        max_disp = max_disp_m / gsd
        reasons.append(f"Spostamento massimo indicato {max_disp_m:g} m = {max_disp:.0f} px.")
    elif deff is not None and deff.shape == ref.shape:
        est = estimate_max_displacement(ref, deff)
        max_disp = max(5.0, est * 1.5 + 3)
        reasons.append(f"Spostamento massimo stimato dalle immagini: {est:.1f} px "
                       f"(uso {max_disp:.0f} px con margine).")
    else:
        max_disp = 20.0
        reasons.append("Spostamento massimo non noto: uso 20 px (modificabile).")

    out = {
        'subset_size': int(chosen if chosen % 2 else chosen + 1),
        'step_size': int(step),
        'max_displacement_px': float(round(max_disp, 1)),
        'border_margin': int(chosen),
        'noise': noise,
        'n_points': int(n_pts),
        'table': table,
        'reasons': reasons,
    }
    if gsd:
        out['precision_mm'] = row[2] * gsd * 1000
        reasons.append(f"Con GSD {gsd * 100:.1f} cm/px l'errore previsto e' "
                       f"circa {row[2] * gsd * 1000:.1f} mm.")
    return out


def _best_region(ref: np.ndarray, size: int):
    """Most textured, data-covered square window of ~size px."""
    h, w = ref.shape
    size = min(size, h, w)
    f = ref.astype(np.float32)
    g = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))
    g[ref == 0] = 0
    k = max(8, size // 4)
    score = cv2.boxFilter(g, -1, (k, k))
    small = cv2.resize(score, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    yy, xx = np.unravel_index(np.argmax(small), small.shape)
    cy, cx = (yy + 0.5) * h / small.shape[0], (xx + 0.5) * w / small.shape[1]
    x0 = int(np.clip(cx - size / 2, 0, w - size))
    y0 = int(np.clip(cy - size / 2, 0, h - size))
    return x0, y0, size


def estimate_precision(ref: np.ndarray, params, gsd: Optional[float] = None,
                       region_size: int = 900, progress=None,
                       deff: Optional[np.ndarray] = None) -> dict:
    """Run *params* on a synthetic copy of *ref* with known motion.

    The copy is moved by a sub-pixel translation plus a small strain
    (0.1 %) and receives fresh noise at the measured level, so it behaves
    like a second survey with no real movement besides the known one.

    With *deff* the test is run on the LESS SHARP of the two images, with
    the noise of the noisier one, so a blurry survey is not hidden by a
    sharp reference.
    """
    from dic_app.core.dic_engine import DICEngine
    import dataclasses

    source = 'ref'
    noise = sd.estimate_noise(ref)
    if deff is not None and deff.shape == ref.shape:
        noise = max(noise, sd.estimate_noise(deff))
        ref, source = weaker_image(ref, deff)

    x0, y0, size = _best_region(ref, region_size)
    crop = np.ascontiguousarray(ref[y0:y0 + size, x0:x0 + size])
    hh, ww = crop.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cxm, cym = ww / 2.0, hh / 2.0
    tu, tv, eps = 0.37, -0.23, 1e-3
    u_t = tu + eps * (xx - cxm)
    v_t = tv - 0.5 * eps * (yy - cym)
    # def(x + d(x)) = ref(x): invert the mapping by fixed point iteration
    mx, my = xx.copy(), yy.copy()
    for _ in range(6):
        mx = xx - (tu + eps * (mx - cxm))
        my = yy - (tv - 0.5 * eps * (my - cym))
    rng = np.random.default_rng(0)
    d = cv2.remap(crop.astype(np.float32), mx, my, cv2.INTER_CUBIC,
                  borderMode=cv2.BORDER_REFLECT) + rng.normal(0, noise, crop.shape)
    d = np.clip(d, 0, 255).astype(np.uint8)
    d[crop == 0] = 0

    p = dataclasses.replace(params, border_margin=max(params.border_margin,
                                                      params.subset_size // 2 + 2),
                            max_displacement_px=max(3.0, params.max_displacement_px or 3.0))
    eng = DICEngine(p)
    if progress:
        eng.set_progress_callback(progress)
    t = time.time()
    r = eng.run(crop, d, gsd=gsd)
    dt = time.time() - t
    gx, gy = r.grid_x.astype(int), r.grid_y.astype(int)
    err = np.hypot(r.u - u_t[gy, gx], r.v - v_t[gy, gx])
    ok = np.isfinite(err)
    n_grid = r.u.size
    res = {
        'region': (x0, y0, size),
        'median_px': float(np.median(err[ok])) if ok.any() else float('nan'),
        'p95_px': float(np.percentile(err[ok], 95)) if ok.any() else float('nan'),
        'gross_pct': float(100 * np.mean(err[ok] > 1.0)) if ok.any() else float('nan'),
        'valid_pct': float(100 * ok.sum() / max(n_grid, 1)),
        'seconds': dt,
        'sec_per_point': dt / max(n_grid, 1),
        'noise': noise,
        'source': source,
    }
    if gsd:
        res['median_mm'] = res['median_px'] * gsd * 1000
        res['p95_mm'] = res['p95_px'] * gsd * 1000
    return res


def estimate_full_time(sec_per_point: float, shape, params) -> float:
    h, w = shape[:2]
    s, st = params.subset_size, max(1, params.step_size)
    n = max(0, (h - s) // st) * max(0, (w - s) // st)
    return sec_per_point * n
