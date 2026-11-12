"""Multi-epoch DIC: displacement history and collapse history of a slope.

With N surveys of the same slope (all registered onto the reference
survey by the project) this module runs:

* **cumulative** analyses  reference -> every other survey, on the SAME
  grid of points, which gives the displacement history of every point;
* **interval** analyses between consecutive surveys (optional, default
  on), which say WHEN the surface changed: a detachment appears in the
  first interval whose two images no longer match there.  Comparing
  consecutive surveys also keeps the images similar (less vegetation /
  lighting decorrelation than comparing with a survey years older).

From those it derives per point: mean velocity (least squares over all
valid epochs), velocity in the last interval, an acceleration index and
the date of the first detected collapse, plus the collapsed area per
interval.
"""

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from dic_app.core.change_detection import VALID, APPEARANCE, CHANGE
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)


class TimeSeriesCancelled(Exception):
    pass


# ----------------------------------------------------------------------
# Dates
# ----------------------------------------------------------------------

def parse_date(capture_time: Optional[str], filename: str) -> Optional[_dt.date]:
    """Survey date from the EXIF capture time, else from the file name.

    File names understood: 2024-11-15, 20241115, 2024_11_15,
    15_11_24, 15-11-2024, 15.11.2024 (day first, as in Italy).
    """
    if capture_time:
        s = str(capture_time).strip()
        for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S', '%Y:%m:%d', '%Y-%m-%d'):
            try:
                return _dt.datetime.strptime(s[:len(_dt.datetime(2000, 1, 1)
                                                    .strftime(fmt))], fmt).date()
            except ValueError:
                continue
    name = os.path.basename(filename or '')
    stem, ext = os.path.splitext(name)
    if ext.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
        name = stem
    m = re.search(r'(?<!\d)(19\d{2}|20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)', name)
    if m:
        try:
            return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = re.search(r'(?<!\d)(\d{1,2})[-_.](\d{1,2})[-_.](\d{4}|\d{2})(?!\d)', name)
    if m:
        y = int(m.group(3))
        y = y + 2000 if y < 100 else y
        try:
            return _dt.date(y, int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return None


@dataclass
class Epoch:
    index: int                      # image index in the project
    label: str                      # file name
    date: Optional[_dt.date] = None
    is_reference: bool = False
    path: str = ''

    def date_str(self):
        return self.date.strftime('%d/%m/%Y') if self.date else f"rilievo {self.label}"

    def to_dict(self):
        return {'index': self.index, 'label': self.label, 'path': self.path,
                'date': self.date.isoformat() if self.date else None,
                'is_reference': self.is_reference}

    @classmethod
    def from_dict(cls, d):
        date = _dt.date.fromisoformat(d['date']) if d.get('date') else None
        return cls(index=int(d['index']), label=d.get('label', ''), date=date,
                   is_reference=bool(d.get('is_reference')), path=d.get('path', ''))


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------

def _valid_points(res):
    """Points whose displacement is a real measurement."""
    pc = res.extra.get('point_class') if res.extra else None
    if pc is not None:
        ok = (pc == VALID) | (pc == APPEARANCE)
    else:
        ok = res.mask_valid.astype(bool)
    return ok & np.isfinite(res.u) & np.isfinite(res.v)


@dataclass
class TimeSeriesResult:
    """Epochs in time order (reference included) and their analyses.

    ``cumulative[k]`` is reference -> epoch k (None for the reference);
    ``intervals[k]`` is epoch k-1 -> epoch k (None for k = 0 or when the
    interval analyses were not run).
    """
    epochs: List[Epoch]
    cumulative: list
    intervals: list
    gsd: Optional[float] = None
    params: object = None
    _cache: dict = field(default_factory=dict, repr=False)

    # -- basic geometry ------------------------------------------------
    @property
    def ref_pos(self) -> int:
        return next(i for i, e in enumerate(self.epochs) if e.is_reference)

    def _any(self):
        for r in self.cumulative + self.intervals:
            if r is not None:
                return r
        raise ValueError("Serie temporale vuota")

    @property
    def grid_x(self):
        return self._any().grid_x

    @property
    def grid_y(self):
        return self._any().grid_y

    @property
    def n(self):
        return len(self.epochs)

    @property
    def has_dates(self) -> bool:
        return all(e.date is not None for e in self.epochs)

    @property
    def has_intervals(self) -> bool:
        return any(r is not None for r in self.intervals)

    def times(self) -> np.ndarray:
        """Time of every epoch relative to the reference: days, or epoch
        number when some dates are unknown."""
        if self.has_dates:
            t0 = self.epochs[self.ref_pos].date
            return np.array([(e.date - t0).days for e in self.epochs], float)
        return np.arange(self.n, dtype=float) - self.ref_pos

    @property
    def time_unit(self) -> str:
        return 'anno' if self.has_dates else 'rilievo'

    def _per_unit(self):
        """Factor from (displacement / time unit of times()) to per year."""
        return 365.25 if self.has_dates else 1.0

    # -- stacks ----------------------------------------------------------
    def cumulative_stack(self):
        """U, V of shape (n_epochs, ny, nx) in px; NaN where not measured;
        zero at the reference epoch."""
        if 'stack' in self._cache:
            return self._cache['stack']
        ny, nx = self.grid_x.shape
        U = np.full((self.n, ny, nx), np.nan, np.float32)
        V = np.full((self.n, ny, nx), np.nan, np.float32)
        U[self.ref_pos] = 0.0
        V[self.ref_pos] = 0.0
        for k, r in enumerate(self.cumulative):
            if r is None:
                continue
            ok = _valid_points(r)
            U[k][ok] = r.u[ok]
            V[k][ok] = r.v[ok]
        # From the survey in which a point collapses on, it (and its direct
        # neighbours, whose subsets straddle the scar) is a different
        # surface: keep only its history BEFORE the collapse (precursors).
        onset = self.collapse_onset()
        if (onset >= 0).any():
            import cv2
            big = np.where(onset >= 0, onset, self.n + 1).astype(np.float32)
            near = cv2.erode(big, np.ones((3, 3), np.uint8))   # min over 3x3
            for k in range(self.n):
                if k != self.ref_pos:
                    gone = near <= k
                    U[k][gone] = np.nan
                    V[k][gone] = np.nan
        self._cache['stack'] = (U, V)
        return U, V

    def cumulative_class(self, k):
        r = self.cumulative[k]
        if r is None:
            return None
        return r.extra.get('point_class') if r.extra else None

    # -- velocity ----------------------------------------------------------
    def velocity(self):
        """Mean velocity per point (least squares on all valid epochs).

        Returns dict with vu, vv, vmag (px per year, or per survey without
        dates) and n (number of epochs used).  At least 3 epochs (reference
        included) are required for a point, so that a single bad match
        cannot make up a velocity.
        """
        if 'vel' in self._cache:
            return self._cache['vel']
        U, V = self.cumulative_stack()
        t = self.times()[:, None, None]
        m = np.isfinite(U) & np.isfinite(V)
        n = m.sum(0)
        tm = np.where(m, t, 0).sum(0) / np.maximum(n, 1)
        dt = np.where(m, t - tm, 0)
        den = (dt ** 2).sum(0)
        out = {}
        for key, D in (('vu', U), ('vv', V)):
            dm = np.where(m, D, 0).sum(0) / np.maximum(n, 1)
            num = (dt * np.where(m, D - dm, 0)).sum(0)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[key] = (num / den * self._per_unit()).astype(np.float32)
        bad = (n < 3) | (den <= 0)
        out['vu'][bad] = np.nan
        out['vv'][bad] = np.nan
        out['vmag'] = np.hypot(out['vu'], out['vv'])
        out['n'] = n
        self._cache['vel'] = out
        return out

    def last_interval_velocity(self):
        """Velocity (px/year or px/survey) between the last two epochs where
        each point was measured, plus the acceleration index
        v_last / v_mean along the mean direction of motion."""
        if 'vlast' in self._cache:
            return self._cache['vlast']
        U, V = self.cumulative_stack()
        t = self.times()
        ny, nx = U.shape[1:]
        last = np.full((ny, nx), -1, int)
        prev = np.full((ny, nx), -1, int)
        for k in range(self.n):
            ok = np.isfinite(U[k])
            prev = np.where(ok, last, prev)
            last = np.where(ok, k, last)
        vu = np.full((ny, nx), np.nan, np.float32)
        vv = np.full((ny, nx), np.nan, np.float32)
        good = (prev >= 0) & (last >= 0)
        if good.any():
            r, c = np.nonzero(good)
            a, b = prev[good], last[good]
            dtt = (t[b] - t[a])
            vu[r, c] = (U[b, r, c] - U[a, r, c]) / dtt * self._per_unit()
            vv[r, c] = (V[b, r, c] - V[a, r, c]) / dtt * self._per_unit()
        mean = self.velocity()
        with np.errstate(invalid='ignore', divide='ignore'):
            vm = mean['vmag']
            along = (vu * mean['vu'] + vv * mean['vv']) / vm
            ratio = along / vm
        # the ratio is meaningless for points that do not move
        ratio[~(vm > 2.0 * self.velocity_noise_floor())] = np.nan
        out = {'vu': vu, 'vv': vv, 'vmag': np.hypot(vu, vv), 'ratio': ratio}
        self._cache['vlast'] = out
        return out

    def velocity_noise_floor(self):
        """Velocity below which a point is considered stable: 3 times the
        robust spread of the mean velocity over the whole slope."""
        v = self.velocity()['vmag']
        v = v[np.isfinite(v)]
        if v.size < 10:
            return 0.0
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826
        return float(med + 3 * mad)

    # -- collapses ---------------------------------------------------------
    def change_source(self) -> str:
        return 'intervals' if self.has_intervals else 'cumulative'

    def _zone_grid(self, k):
        res = (self.intervals if self.has_intervals else self.cumulative)[k]
        if res is None or not res.extra:
            return None, res
        z = res.extra.get('change_zone_id')
        if z is None:
            return None, res
        rejected = {zz['id'] for zz in res.change_zones
                    if zz.get('status') == 'rejected'}
        m = z > 0
        if rejected:
            m &= ~np.isin(z, list(rejected))
        return m, res

    def collapse_onset(self) -> np.ndarray:
        """Epoch position at which each point was first found collapsed
        (inside a change zone), -1 if never."""
        if 'onset' in self._cache:
            return self._cache['onset']
        onset = np.full(self.grid_x.shape, -1, np.int16)
        for k in range(self.n):
            if k == self.ref_pos and not self.has_intervals:
                continue
            m, _ = self._zone_grid(k)
            if m is None:
                continue
            onset[(onset < 0) & m] = k
        self._cache['onset'] = onset
        return onset

    def cell_area_m2(self):
        gx, gy = self.grid_x, self.grid_y
        sx = float(gx[0, 1] - gx[0, 0]) if gx.shape[1] > 1 else 1.0
        sy = float(gy[1, 0] - gy[0, 0]) if gy.shape[0] > 1 else 1.0
        return sx * sy * (self.gsd ** 2) if self.gsd else None

    def collapse_table(self):
        """One row per epoch after the first: new collapsed area, number
        of zones, cumulative area."""
        onset = self.collapse_onset()
        cell = self.cell_area_m2()
        rows, cum = [], 0.0
        for k in range(self.n):
            if k == 0:
                continue
            m, res = self._zone_grid(k)
            if res is None:
                continue
            new_pts = int(np.sum(onset == k))
            zones = [z for z in res.change_zones if z.get('status') != 'rejected']
            area = new_pts * cell if cell else None
            cum += area or 0.0
            rows.append({
                'pos': k,
                'from': self.epochs[k - 1] if self.has_intervals else self.epochs[self.ref_pos],
                'to': self.epochs[k],
                'n_zones': len(zones),
                'new_points': new_pts,
                'new_area_m2': area,
                'cum_area_m2': cum if cell else None,
                'zones': zones,
            })
        return rows

    # -- single point ----------------------------------------------------
    def nearest_node(self, x, y):
        gx, gy = self.grid_x, self.grid_y
        c = int(np.argmin(np.abs(gx[0] - x)))
        r = int(np.argmin(np.abs(gy[:, 0] - y)))
        return r, c

    def point_series(self, r, c):
        U, V = self.cumulative_stack()
        classes = []
        for k in range(self.n):
            pc = self.cumulative_class(k)
            classes.append(None if pc is None else int(pc[r, c]))
        return {'t': self.times(), 'u': U[:, r, c], 'v': V[:, r, c],
                'mag': np.hypot(U[:, r, c], V[:, r, c]), 'classes': classes,
                'onset': int(self.collapse_onset()[r, c])}

    # -- persistence -----------------------------------------------------
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        for f in os.listdir(directory):
            if f.endswith('.npz') or f == 'series.json':
                os.remove(os.path.join(directory, f))
        meta = {'epochs': [e.to_dict() for e in self.epochs], 'gsd': self.gsd,
                'cumulative': [], 'intervals': []}
        for kind, lst in (('cumulative', self.cumulative), ('intervals', self.intervals)):
            for k, r in enumerate(lst):
                if r is None:
                    meta[kind].append(None)
                    continue
                name = f"{kind[:3]}_{k}.npz"
                r.to_npz(os.path.join(directory, name))
                meta[kind].append(name)
        with open(os.path.join(directory, 'series.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory, params=None):
        from dic_app.core.dic_engine import DICResult
        with open(os.path.join(directory, 'series.json')) as f:
            meta = json.load(f)
        lists = {}
        for kind in ('cumulative', 'intervals'):
            lists[kind] = [DICResult.from_npz(os.path.join(directory, n), params)
                           if n else None for n in meta.get(kind, [])]
        return cls(epochs=[Epoch.from_dict(d) for d in meta['epochs']],
                   cumulative=lists['cumulative'], intervals=lists['intervals'],
                   gsd=meta.get('gsd'), params=params)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

class TimeSeriesRunner:
    """Runs all analyses of a series.  Image access goes through
    callables so the module does not depend on the project/GUI:

    get_gray(image_index) -> 2-D uint8 in the reference geometry
    get_rgb(image_index)  -> RGB image in the same geometry, or None
    """

    def __init__(self, params, epochs: List[Epoch], get_gray: Callable,
                 get_rgb: Optional[Callable] = None, mask=None, gsd=None,
                 intervals: bool = True):
        self.params = params
        self.epochs = sorted(epochs, key=lambda e: (e.date is None, e.date or _dt.date.min))
        if sum(e.is_reference for e in self.epochs) != 1:
            raise ValueError("La serie deve contenere esattamente un riferimento")
        self.get_gray = get_gray
        self.get_rgb = get_rgb
        self.mask = mask
        self.gsd = gsd
        self.do_intervals = intervals
        self._engine = None
        self._cancelled = False
        self._progress = None

    def set_progress_callback(self, cb):
        self._progress = cb

    def cancel(self):
        self._cancelled = True
        if self._engine is not None:
            self._engine.cancel()

    def _emit(self, pct, msg):
        if self._progress:
            self._progress(int(pct), msg)

    def _check(self):
        if self._cancelled:
            raise TimeSeriesCancelled()

    def _surface(self, ia, ib, shape):
        p = self.params
        from dic_app.core.dic_engine import DICMethod
        if self.get_rgb is None or p.method != DICMethod.ICGN or not (
                p.exclude_vegetation or p.exclude_water):
            return None
        try:
            from dic_app.core.change_detection import surface_labels
            h, w = shape
            ra, rb = self.get_rgb(ia), self.get_rgb(ib)
            ra = ra[:h, :w] if ra is not None else None
            rb = rb[:h, :w] if rb is not None else None
            return surface_labels(ra, rb, shape, vegetation=p.exclude_vegetation,
                                  water=p.exclude_water,
                                  veg_sensitivity=p.veg_sensitivity,
                                  water_sensitivity=p.water_sensitivity)
        except Exception as e:
            logger.warning(f"Surface masks not computed for {ia}->{ib}: {e}")
            return None

    def run(self) -> TimeSeriesResult:
        from dic_app.core.dic_engine import DICEngine, DICCancelledError
        ep = self.epochs
        n = len(ep)
        if n < 3:
            raise ValueError("Per una serie temporale servono almeno 3 rilievi "
                             "(per 2 usa la normale analisi)")
        rp = next(i for i, e in enumerate(ep) if e.is_reference)

        self._emit(0, "Lettura e registrazione delle immagini...")
        grays = {}
        for k, e in enumerate(ep):
            self._check()
            g = self.get_gray(e.index)
            if g is None:
                raise ValueError(f"Impossibile allineare '{e.label}' al riferimento "
                                 f"(vedi log)")
            grays[e.index] = g
            self._emit(2 + 8 * (k + 1) / n, f"Immagine {k + 1}/{n} pronta: {e.label}")
        h = min(g.shape[0] for g in grays.values())
        w = min(g.shape[1] for g in grays.values())
        if any(g.shape[:2] != (h, w) for g in grays.values()):
            logger.warning(f"Images of different size: all cropped to {w}x{h}")
            grays = {k: np.ascontiguousarray(g[:h, :w]) for k, g in grays.items()}
        mask = self.mask[:h, :w] if self.mask is not None else None

        jobs = [('cum', k, rp, k) for k in range(n) if k != rp]
        if self.do_intervals:
            # interval (k-1 -> k); when k-1 is the reference it is the same
            # pair as the cumulative job and is reused below
            jobs += [('int', k, k - 1, k) for k in range(1, n) if k - 1 != rp]
        cumulative = [None] * n
        intervals = [None] * n
        n_jobs = len(jobs)
        for j, (kind, k, a, b) in enumerate(jobs):
            self._check()
            ia, ib = ep[a].index, ep[b].index
            label = (f"Cumulato {ep[a].date_str()} -> {ep[b].date_str()}" if kind == 'cum'
                     else f"Intervallo {ep[a].date_str()} -> {ep[b].date_str()}")
            base = 10 + 88 * j / n_jobs
            span = 88 / n_jobs
            eng = DICEngine(self.params)
            eng.set_progress_callback(
                lambda p, m, base=base, span=span, label=label, j=j:
                self._emit(base + span * p / 100.0, f"[{j + 1}/{n_jobs}] {label}: {m}"))
            self._engine = eng
            surface = self._surface(ia, ib, (h, w))
            try:
                res = eng.run(grays[ia], grays[ib], mask, surface=surface, gsd=self.gsd)
            except DICCancelledError:
                raise TimeSeriesCancelled()
            finally:
                self._engine = None
            if kind == 'cum':
                cumulative[k] = res
            else:
                intervals[k] = res
            logger.info(f"Time series: {label} done "
                        f"({res.computation_time_s:.1f}s)")
        if self.do_intervals and rp + 1 < n:
            intervals[rp + 1] = cumulative[rp + 1]

        shapes = {r.grid_x.shape for r in cumulative + intervals if r is not None}
        if len(shapes) != 1:
            raise ValueError(f"Le analisi hanno griglie diverse {shapes}: "
                             f"controlla la maschera e l'allineamento")
        self._emit(100, "Serie temporale completata")
        return TimeSeriesResult(epochs=ep, cumulative=cumulative,
                                intervals=intervals, gsd=self.gsd,
                                params=self.params)
