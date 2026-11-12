"""Guided choice of the preprocessing filters, judged by DIC itself.

Why a new advisor
-----------------
The old optimiser (``auto_filter``) scored a filtered image by how it
*looks* (gradient, contrast, sharpness, entropy).  Those metrics reward
contrast stretching and sharpening, which amplify noise and change the
grey levels non-linearly: on test pairs with a known motion the pipeline
it picked increased the DIC error (up to 2x the 95th percentile) and
histogram equalisation dropped the valid points to ~30 %.

This advisor judges every candidate the only way that matters, by
running the project's DIC on the user's images:

* **Recognised area** - the real image pair (deformed registered onto the
  reference) is correlated on a few tiles spread over the scene; the
  share of points DIC can match is what illumination / shadow / haze
  filters must improve.
* **Precision** - a synthetic copy of the weaker image is moved by a
  known sub-pixel motion, gets the measured noise and the real
  illumination change of the pair, and both images are filtered; the
  error against the known motion is what denoising must improve and what
  aggressive filters destroy.

Candidates are few and DIC-safe (Wallis, CLAHE, light Gaussian, NLM),
with parameters scaled to the subset size and the measured noise; the
answers to the questionnaire decide which ones are tried.  "No filter"
is always a candidate and wins unless a filter is clearly better.
"""

import copy
import dataclasses
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from dic_app.core import subset_dic as sd
from dic_app.core.preprocessing import FilterPipeline
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# ----------------------------------------------------------------------
# Questionnaire
# ----------------------------------------------------------------------

UNKNOWN = 'unknown'

# key -> (question, [(value, label, help)])
QUESTIONS: Dict[str, Tuple[str, List[Tuple[str, str, str]]]] = {
    'lighting': (
        "Tra le foto da confrontare cambia l'illuminazione?",
        [('same', "No, luce simile",
          "Stessa ora del giorno e stesso meteo."),
         ('some', "Un po'",
          "Ora diversa o cielo diverso (sole/nuvoloso), senza ombre nette."),
         ('strong', "Molto, con ombre nette",
          "Ombre che si spostano, controluce, parete bagnata in una sola foto."),
         (UNKNOWN, "Non so", "Lo misura il programma.")]),
    'noise': (
        "Le foto sono rumorose (granulose)?",
        [('low', "No, pulite", "Giorno pieno, ISO bassi."),
         ('medium', "Un po'", "Poca luce, ISO alti, compressione JPEG forte."),
         ('high', "Molto", "Foto notturne, zoom digitale, sensore piccolo."),
         (UNKNOWN, "Non so", "Lo misura il programma.")]),
    'contrast': (
        "Com'è il contrasto delle immagini?",
        [('good', "Buono", "Dettagli ben visibili su tutta la parete."),
         ('haze', "Basso / foschia", "Immagine 'slavata', foschia, distanza elevata."),
         ('dark', "Zone molto scure o bruciate",
          "Parti in ombra profonda o cielo/roccia chiara sovraesposti."),
         (UNKNOWN, "Non so", "Lo misura il programma.")]),
    'priority': (
        "Cosa conta di più?",
        [('precision', "Precisione massima",
          "Prova anche i filtri lenti (Non-Local Means)."),
         ('balanced', "Equilibrio", "Scelta consigliata."),
         ('speed', "Velocità", "Solo filtri rapidi.")]),
}

DEFAULT_ANSWERS = {'lighting': UNKNOWN, 'noise': UNKNOWN,
                   'contrast': UNKNOWN, 'priority': 'balanced'}


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------

@dataclass
class Diagnostics:
    """What the images say, independently of the answers."""
    noise: float = 0.0                 # grey levels (Immerkaer), worst image
    lighting_change: float = 0.0       # 0 = same light, ~0.3+ = strong
    shadow_fraction: float = 0.0       # share of the scene with a local light jump
    low_contrast: bool = False
    dark_or_saturated_pct: float = 0.0
    contrast_std: float = 0.0          # median local grey std at subset scale
    registered: bool = False           # deformed registered onto reference
    registration_note: str = ""

    def lighting_level(self) -> str:
        if self.lighting_change > 0.30 or self.shadow_fraction > 0.08:
            return 'strong'
        if self.lighting_change > 0.08:
            return 'some'
        return 'same'

    def noise_level(self) -> str:
        if self.noise > 6.0:
            return 'high'
        if self.noise > 3.0:
            return 'medium'
        return 'low'

    def contrast_level(self) -> str:
        if self.dark_or_saturated_pct > 10.0:
            return 'dark'
        if self.low_contrast:
            return 'haze'
        return 'good'


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(img, dtype=np.uint8)


def register_pair(ref: np.ndarray, deff: np.ndarray
                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Warp *deff* onto *ref* (coarse homography on downscaled copies).

    Returns (deff_registered, valid_mask, note); (None, None, note) if the
    registration fails.  Filters are chosen before the alignment step, so
    the advisor registers the pair itself only for its tests.
    """
    from dic_app.core.precrop import coarse_align
    al, err = None, ""
    # Low-contrast (hazy) photos have too few features: retry on
    # contrast-enhanced copies (used only to find the transform).
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    for prep in (lambda x: x, clahe.apply):
        try:
            al = coarse_align(prep(ref), prep(deff), max_dim=1600, model='homography')
            if al is not None and al.matrix is not None and al.n_inliers >= 15:
                break
        except Exception as e:                   # data dependent
            err = str(e)
        al = None
    if al is None:
        return None, None, ("registrazione non riuscita"
                            + (f" ({err})" if err else " (pochi punti comuni)"))
    h, w = ref.shape[:2]
    M = np.asarray(al.matrix, np.float64)
    warped = cv2.warpPerspective(deff, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpPerspective(np.full(deff.shape[:2], 255, np.uint8), M, (w, h),
                                flags=cv2.INTER_NEAREST, borderValue=0)
    valid = cv2.erode(valid, np.ones((15, 15), np.uint8)) > 0
    return warped, valid, (f"registrazione ok ({al.n_inliers} punti, "
                           f"errore {al.rmse_px:.1f} px)")


def illumination_ratio_maps(ref: np.ndarray, deff: np.ndarray, ok: np.ndarray,
                            sigma: float):
    """Low-pass log-intensity of both images, computed only on *ok* pixels.

    Normalised convolution: blur(log I * ok) / blur(ok), so the black
    borders of the registration and saturated pixels do not leak into
    the estimate.  ``ld - lr`` is the log of the illumination ratio
    (texture cancels out at this scale).
    """
    w = ok.astype(np.float32)
    den = cv2.GaussianBlur(w, (0, 0), sigma) + 1e-6
    lr = cv2.GaussianBlur(np.log(ref.astype(np.float32) + 1) * w, (0, 0), sigma) / den
    ld = cv2.GaussianBlur(np.log(deff.astype(np.float32) + 1) * w, (0, 0), sigma) / den
    return lr, ld


def _work_scale(shape, max_dim: int = 2400) -> float:
    return min(1.0, max_dim / float(max(shape[:2])))


def measure(ref: np.ndarray, deff: Optional[np.ndarray], subset_size: int = 41
            ) -> Tuple[Diagnostics, Optional[np.ndarray], Optional[np.ndarray]]:
    """Measure noise, contrast and illumination change of the pair.

    Returns (diagnostics, deff_registered, valid_mask).
    """
    ref = _gray(ref)
    d = Diagnostics()
    d.noise = sd.estimate_noise(ref)
    reg = valid = None
    if deff is not None:
        deff = _gray(deff)
        d.noise = max(d.noise, sd.estimate_noise(deff))
        reg, valid, d.registration_note = register_pair(ref, deff)
        d.registered = reg is not None

    s = _work_scale(ref.shape)
    small = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else ref
    win = max(5, int(subset_size * s) | 1)
    f = small.astype(np.float32)
    mu = cv2.boxFilter(f, -1, (win, win))
    sdv = np.sqrt(np.maximum(cv2.boxFilter(f * f, -1, (win, win)) - mu * mu, 0))
    data = small > 0
    d.contrast_std = float(np.median(sdv[data])) if data.any() else 0.0
    p1, p99 = np.percentile(small[data], [1, 99]) if data.any() else (0, 255)
    d.low_contrast = d.contrast_std < 6.0 or (p99 - p1) < 60
    d.dark_or_saturated_pct = float(100 * np.mean((small[data] < 12) | (small[data] > 245))) \
        if data.any() else 0.0

    if reg is not None:
        r_s = small
        d_s = cv2.resize(reg, (r_s.shape[1], r_s.shape[0]), interpolation=cv2.INTER_AREA) \
            if s < 1 else reg
        v_s = cv2.resize(valid.astype(np.uint8), (r_s.shape[1], r_s.shape[0]),
                         interpolation=cv2.INTER_NEAREST) > 0 if s < 1 else valid
        ok = v_s & (r_s > 8) & (d_s > 8) & (r_s < 248) & (d_s < 248)
        if ok.sum() > 1000:
            lr, ld = illumination_ratio_maps(r_s, d_s, ok, max(4.0, 1.5 * win))
            inner = ok & (cv2.erode(ok.astype(np.uint8),
                                    np.ones((int(2 * win) | 1,) * 2, np.uint8)) > 0)
            if inner.sum() > 1000:
                ok = inner
            ratio = (ld - lr)[ok]
            ratio = ratio - np.median(ratio)          # a global gain is harmless
            d.lighting_change = float(np.percentile(np.abs(ratio), 90))
            # Shadows: share of the scene whose light differs by > ~35 %
            d.shadow_fraction = float(np.mean(np.abs(ratio) > 0.30))
    return d, reg, valid


# ----------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    pipeline: FilterPipeline
    why: str                         # when / why it is proposed
    slow: bool = False


def _odd(v: int) -> int:
    v = int(round(v))
    return v if v % 2 == 1 else v + 1


def resolve_answers(answers: Dict[str, str], diag: Diagnostics) -> Dict[str, str]:
    """Replace 'Non so' with the measured level."""
    a = dict(DEFAULT_ANSWERS)
    a.update({k: v for k, v in (answers or {}).items() if v})
    if a['lighting'] == UNKNOWN:
        a['lighting'] = diag.lighting_level() if diag.registered else 'some'
    if a['noise'] == UNKNOWN:
        a['noise'] = diag.noise_level()
    if a['contrast'] == UNKNOWN:
        a['contrast'] = diag.contrast_level()
    return a


def build_candidates(answers: Dict[str, str], diag: Diagnostics,
                     subset_size: int = 41) -> List[Candidate]:
    """DIC-safe candidate pipelines with parameters scaled to the data."""
    a = resolve_answers(answers, diag)
    S = int(subset_size)
    wallis_win = min(101, _odd(max(15, 1.5 * S)))
    tiles = int(np.clip(round(8 * 41 / max(S, 11)), 4, 16))
    g_sigma = float(np.clip(0.25 * diag.noise, 0.6, 1.4))
    nlm_h = float(np.clip(1.1 * diag.noise, 3.0, 20.0))

    def wallis(p, cf=0.8):
        p.add_step('Wallis Filter', target_mean=127, target_std=50,
                   brightness_factor=0.8, contrast_factor=cf, window_size=wallis_win)

    def clahe(p, clip=2.0):
        p.add_step('CLAHE', clip_limit=clip, tile_grid_size=(tiles, tiles))

    def gauss(p):
        p.add_step('Gaussian Blur', kernel_size=5, sigma=round(g_sigma, 2))

    out = [Candidate("Nessun filtro", FilterPipeline(),
                     "Il motore DIC (IC-GN, ZNSSD) compensa gia' le variazioni "
                     "lineari di luminosita' e contrasto.")]
    light = a['lighting'] in ('some', 'strong')
    noisy = a['noise'] in ('medium', 'high')
    weak = a['contrast'] in ('haze', 'dark')

    if light or weak:
        p = FilterPipeline(); wallis(p)
        out.append(Candidate(f"Wallis (finestra {wallis_win} px)", p,
                             "Uniforma luce e contrasto locali: utile con "
                             "ombre e illuminazione diversa tra le foto."))
    if a['lighting'] == 'strong':
        p = FilterPipeline(); wallis(p, cf=0.95)
        out.append(Candidate(f"Wallis forte (finestra {wallis_win} px)", p,
                             "Normalizzazione piu' decisa per ombre nette."))
    if weak:
        p = FilterPipeline(); clahe(p)
        out.append(Candidate(f"CLAHE (griglia {tiles}x{tiles})", p,
                             "Aumenta il contrasto locale in foschia e nelle "
                             "zone scure, limitando l'amplificazione del rumore."))
    if noisy:
        p = FilterPipeline(); gauss(p)
        out.append(Candidate(f"Gaussiano leggero (sigma {g_sigma:.1f})", p,
                             "Attenua il rumore del sensore: un filtro "
                             "passa-basso leggero riduce anche l'errore "
                             "sistematico sub-pixel."))
        if light or weak:
            p = FilterPipeline(); gauss(p); wallis(p)
            out.append(Candidate("Gaussiano + Wallis", p,
                                 "Rumore e luce diversa insieme: prima si "
                                 "attenua il rumore, poi si uniforma la luce."))
        if a['noise'] == 'high' and a['priority'] != 'speed':
            p = FilterPipeline()
            p.add_step('Non-Local Means', h=round(nlm_h, 1),
                       template_window=7, search_window=21)
            out.append(Candidate(f"Non-Local Means (h {nlm_h:.0f})", p,
                                 "Riduzione del rumore forte che preserva la "
                                 "tessitura; lento sulle foto grandi.",
                                 slow=True))
    if weak and light:
        p = FilterPipeline(); wallis(p); clahe(p)
        out.append(Candidate("Wallis + CLAHE", p,
                             "Luce diversa e contrasto basso insieme."))
    if a['priority'] == 'precision' and len(out) == 1:
        # Nothing flagged: still check the two most common filters
        for name, fn, why in (
                (f"Wallis (finestra {wallis_win} px)", wallis,
                 "Verifica di controllo."),
                (f"Gaussiano leggero (sigma {g_sigma:.1f})", gauss,
                 "Verifica di controllo.")):
            p = FilterPipeline(); fn(p)
            out.append(Candidate(name, p, why))
    if a['priority'] == 'speed':
        out = [c for c in out if not c.slow]
    return out


# ----------------------------------------------------------------------
# Evaluation with the DIC engine
# ----------------------------------------------------------------------

@dataclass
class CandidateScore:
    name: str
    pipeline: FilterPipeline
    why: str
    recognised_pct: float = float('nan')   # real pair
    median_px: float = float('nan')        # synthetic, known motion
    p95_px: float = float('nan')
    seconds: float = 0.0
    utility: float = float('nan')
    verdict: str = ""


@dataclass
class Recommendation:
    best: CandidateScore
    scores: List[CandidateScore]
    diagnostics: Diagnostics
    answers: Dict[str, str]                # after resolving "Non so"
    reasons: List[str] = field(default_factory=list)
    tiles: List[Tuple[int, int, int]] = field(default_factory=list)  # (x0, y0, size)
    seconds: float = 0.0


def _tiles(valid: np.ndarray, size: int, n: int = 4) -> List[Tuple[int, int, int]]:
    """Up to *n* square tiles spread over the valid area (2x2 layout)."""
    h, w = valid.shape
    size = int(min(size, h, w))
    out = []
    for fy in (0.27, 0.73):
        for fx in (0.27, 0.73):
            x0 = int(np.clip(fx * w - size / 2, 0, w - size))
            y0 = int(np.clip(fy * h - size / 2, 0, h - size))
            if valid[y0:y0 + size, x0:x0 + size].mean() > 0.9:
                out.append((x0, y0, size))
    if not out:                                   # fall back to the centre
        out.append(((w - size) // 2, (h - size) // 2, size))
    # drop duplicates (small images)
    return list(dict.fromkeys(out))[:n]


def _synthetic_pair(ref_c: np.ndarray, light_ratio: Optional[np.ndarray],
                    noise: float, seed: int = 0):
    """Known sub-pixel motion + fresh noise + real illumination change."""
    hh, ww = ref_c.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cxm, cym = ww / 2.0, hh / 2.0
    tu, tv, eps = 0.37, -0.23, 1e-3
    u_t = tu + eps * (xx - cxm)
    v_t = tv - 0.5 * eps * (yy - cym)
    mx, my = xx.copy(), yy.copy()
    for _ in range(6):
        mx = xx - (tu + eps * (mx - cxm))
        my = yy - (tv - 0.5 * eps * (my - cym))
    rng = np.random.default_rng(seed)
    base = ref_c.astype(np.float32)
    r = base + rng.normal(0, noise * 0.7, base.shape)
    d = cv2.remap(base, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if light_ratio is not None:
        d = d * light_ratio
    d = d + rng.normal(0, noise * 0.7, base.shape)
    return (np.clip(r, 0, 255).astype(np.uint8), np.clip(d, 0, 255).astype(np.uint8),
            u_t, v_t)


def _run(params, a: np.ndarray, b: np.ndarray):
    from dic_app.core.dic_engine import DICEngine
    return DICEngine(params).run(a, b)


def _eval_params(params, size: int):
    """DIC settings for the tests: the project's, on a coarser grid."""
    from dic_app.core.dic_engine import DICParameters
    p = params if params is not None else DICParameters()
    step = max(p.step_size, size // 28)
    return dataclasses.replace(
        p, step_size=int(step),
        border_margin=max(p.border_margin, p.subset_size // 2 + 2),
        max_displacement_px=max(3.0, p.max_displacement_px or 3.0),
        exclude_vegetation=False, exclude_water=False, threads=0)


def utility(recognised_pct: float, p95_px: float, base_p95: float) -> float:
    """One number to rank candidates.

    Recognised area in % minus 20 points for every doubling of the 95th
    percentile error (plus 20 for every halving) with respect to no filter.
    """
    if not np.isfinite(recognised_pct):
        recognised_pct = 0.0
    if not (np.isfinite(p95_px) and np.isfinite(base_p95)) or p95_px <= 0 or base_p95 <= 0:
        return recognised_pct
    return recognised_pct - 20.0 * float(np.log2(p95_px / base_p95))


MIN_GAIN = 2.0      # a filter must beat "no filter" by this utility margin


def evaluate(ref: np.ndarray, deff: Optional[np.ndarray],
             answers: Optional[Dict[str, str]] = None, dic_params=None,
             tile_size: int = 448,
             progress: Optional[Callable[[int, str], None]] = None,
             cancel: Optional[Callable[[], bool]] = None) -> Recommendation:
    """Measure, build candidates, test them with DIC, recommend one."""
    t0 = time.time()

    def report(pct, msg):
        if progress:
            progress(int(pct), msg)
        if cancel and cancel():
            raise InterruptedError("Analisi annullata")

    from dic_app.core.dic_engine import DICParameters
    base_params = dic_params if dic_params is not None else DICParameters()
    S = base_params.subset_size

    report(2, "Misura di rumore, contrasto e illuminazione...")
    ref = _gray(ref)
    diag, reg, valid = measure(ref, deff, S)
    resolved = resolve_answers(answers or {}, diag)
    cands = build_candidates(answers or {}, diag, S)

    if valid is None:
        valid = ref > 0
    tile_size = int(max(tile_size, 6 * S))
    tiles = _tiles(valid & (ref > 0), tile_size)
    params = _eval_params(base_params, tiles[0][2])

    # Synthetic test region: the least textured of the tiles (worst case
    # for precision), with the real illumination change of the pair.
    def tex(t):
        x0, y0, s = t
        return float(np.std(ref[y0:y0 + s, x0:x0 + s]))
    sx0, sy0, ss = min(tiles, key=tex)
    ref_c = ref[sy0:sy0 + ss, sx0:sx0 + ss]
    light = None
    if reg is not None:
        def_c = reg[sy0:sy0 + ss, sx0:sx0 + ss]
        okc = (valid[sy0:sy0 + ss, sx0:sx0 + ss] & (ref_c > 8) & (def_c > 8)
               & (ref_c < 248) & (def_c < 248))
        if okc.mean() > 0.5:
            lr, ld = illumination_ratio_maps(ref_c, def_c, okc, max(4.0, 1.5 * S))
            light = np.exp(np.clip(ld - lr, -1.6, 1.6)).astype(np.float32)
            light /= float(np.median(light))
    syn_r, syn_d, u_t, v_t = _synthetic_pair(ref_c, light, diag.noise)

    scores: List[CandidateScore] = []
    n = len(cands)
    for i, c in enumerate(cands):
        report(8 + 88 * i / max(n, 1), f"Prova {i + 1}/{n}: {c.name}")
        sc = CandidateScore(c.name, c.pipeline, c.why)
        tc = time.time()
        try:
            # Precision on the synthetic pair
            fr, fd = c.pipeline.apply(syn_r), c.pipeline.apply(syn_d)
            r = _run(params, fr, fd)
            gx, gy = r.grid_x.astype(int), r.grid_y.astype(int)
            err = np.hypot(r.u - u_t[gy, gx], r.v - v_t[gy, gx])
            ok = np.isfinite(err)
            if ok.any():
                sc.median_px = float(np.median(err[ok]))
                sc.p95_px = float(np.percentile(err[ok], 95))
            # Recognised area on the real pair
            if reg is not None:
                got = tot = 0
                for (x0, y0, s) in tiles:
                    a_ = c.pipeline.apply(ref[y0:y0 + s, x0:x0 + s])
                    b_ = c.pipeline.apply(reg[y0:y0 + s, x0:x0 + s])
                    rr = _run(params, a_, b_)
                    q = rr.correlation_quality
                    good = np.isfinite(rr.u)
                    if q is not None:
                        good &= np.nan_to_num(q, nan=0) >= params.correlation_threshold
                    got += int(good.sum()); tot += good.size
                sc.recognised_pct = 100.0 * got / max(tot, 1)
        except InterruptedError:
            raise
        except Exception as e:
            logger.warning(f"Filter candidate '{c.name}' failed: {e}")
            sc.verdict = f"errore: {e}"
        sc.seconds = time.time() - tc
        scores.append(sc)

    base = scores[0]
    for sc in scores:
        sc.utility = utility(sc.recognised_pct if reg is not None else 100.0,
                             sc.p95_px, base.p95_px)
    best = base
    for sc in scores[1:]:
        if np.isfinite(sc.utility) and sc.utility > best.utility + (MIN_GAIN if best is base else 0):
            best = sc

    for sc in scores:
        if sc.verdict:
            continue
        if sc is best:
            sc.verdict = "CONSIGLIATO"
        elif not np.isfinite(sc.utility):
            sc.verdict = "non valutabile"
        elif sc.utility < base.utility - MIN_GAIN:
            sc.verdict = "peggiora"
        elif sc.utility > base.utility + MIN_GAIN:
            sc.verdict = "migliora"
        else:
            sc.verdict = "nessun vantaggio"

    reasons = _explain(diag, resolved, answers or {}, best, base, reg is not None)
    report(100, "Analisi completata")
    return Recommendation(best=best, scores=scores, diagnostics=diag,
                          answers=resolved, reasons=reasons, tiles=tiles,
                          seconds=time.time() - t0)


_LEVEL_TXT = {
    'lighting': {'same': "luce simile", 'some': "luce un po' diversa",
                 'strong': "luce molto diversa / ombre nette"},
    'noise': {'low': "rumore basso", 'medium': "rumore medio", 'high': "rumore alto"},
    'contrast': {'good': "contrasto buono", 'haze': "contrasto basso (foschia)",
                 'dark': "zone scure o bruciate"},
}


def _explain(diag, resolved, given, best, base, has_pair) -> List[str]:
    out = []
    src = {k: ("risposta" if given.get(k, UNKNOWN) not in (UNKNOWN, None, '') else "misurato")
           for k in ('lighting', 'noise', 'contrast')}
    out.append(f"Illuminazione: {_LEVEL_TXT['lighting'][resolved['lighting']]} "
               f"({src['lighting']}"
               + (f"; variazione misurata {100 * diag.lighting_change:.0f}%, "
                  f"zone d'ombra {100 * diag.shadow_fraction:.0f}%" if diag.registered else "")
               + ").")
    out.append(f"Rumore: {_LEVEL_TXT['noise'][resolved['noise']]} "
               f"({src['noise']}; sigma misurato {diag.noise:.1f} livelli di grigio).")
    out.append(f"Contrasto: {_LEVEL_TXT['contrast'][resolved['contrast']]} "
               f"({src['contrast']}; zone scure/bruciate {diag.dark_or_saturated_pct:.0f}%).")
    if not has_pair:
        out.append("Attenzione: " + (diag.registration_note or "manca la seconda immagine")
                   + ". L'area riconosciuta sulla coppia reale non e' stata misurata; "
                     "il confronto usa solo il test di precisione.")
    if best is base:
        out.append("Nessun filtro migliora in modo apprezzabile la misura: conviene "
                   "lavorare sulle immagini originali (meno alterazioni = misura piu' "
                   "affidabile).")
    else:
        parts = []
        if np.isfinite(best.recognised_pct) and np.isfinite(base.recognised_pct):
            parts.append(f"area riconosciuta {base.recognised_pct:.0f}% -> "
                         f"{best.recognised_pct:.0f}%")
        if np.isfinite(best.p95_px) and np.isfinite(base.p95_px):
            parts.append(f"errore p95 {base.p95_px:.3f} -> {best.p95_px:.3f} px")
        out.append(f"Consigliato: {best.name} ({', '.join(parts)}). {best.why}")
    out.append("Filtri come equalizzazione dell'istogramma, sharpening, soglia "
               "adattiva e operatori morfologici non sono proposti: alterano i "
               "livelli di grigio e nei test peggiorano la precisione del DIC.")
    return out


def copy_pipeline(p: FilterPipeline) -> FilterPipeline:
    """Independent copy (the panel and the project must not share steps)."""
    return FilterPipeline.from_dict(copy.deepcopy(p.to_dict()))
