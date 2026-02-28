"""Core DIC computation engine with 4 correlation algorithms.

Algorithms:
1. Template Matching (NCC) - subset-based normalized cross-correlation
2. Optical Flow (Farneback) - dense optical flow
3. Phase Correlation (FFT) - frequency-domain cross-correlation
4. Feature Matching (ORB/SIFT) - sparse keypoint matching + interpolation
"""

import time
import threading
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
from enum import Enum
from dic_app.utils.helpers import compute_magnitude, setup_logger

logger = setup_logger(__name__)


class DICMethod(Enum):
    TEMPLATE_NCC = "template_ncc"
    OPTICAL_FLOW_FARNEBACK = "optical_flow"
    PHASE_CORRELATION = "phase_correlation"
    FEATURE_MATCHING = "feature_matching"


class SubPixelMethod(Enum):
    NONE = "none"
    PARABOLIC = "parabolic"
    GAUSSIAN = "gaussian"
    BICUBIC = "bicubic"


@dataclass
class DICParameters:
    """Configuration parameters for DIC analysis."""
    method: DICMethod = DICMethod.TEMPLATE_NCC
    subset_size: int = 31
    step_size: int = 5
    search_radius_x: int = 50
    search_radius_y: int = 50
    subpixel_method: SubPixelMethod = SubPixelMethod.GAUSSIAN
    correlation_threshold: float = 0.6
    # Optical flow specific
    of_pyr_scale: float = 0.5
    of_levels: int = 5
    of_winsize: int = 15
    of_iterations: int = 3
    of_poly_n: int = 5
    of_poly_sigma: float = 1.2
    # Phase correlation specific
    upsample_factor: int = 20
    # Feature matching specific
    max_features: int = 10000
    match_ratio: float = 0.75
    # Border margin to exclude noisy edges (pixels).
    # Recommended ≥ subset_size to avoid edge artifacts from alignment.
    border_margin: int = 10

    def to_dict(self):
        return {
            'method': self.method.value,
            'subset_size': self.subset_size,
            'step_size': self.step_size,
            'search_radius_x': self.search_radius_x,
            'search_radius_y': self.search_radius_y,
            'subpixel_method': self.subpixel_method.value,
            'correlation_threshold': self.correlation_threshold,
            'of_pyr_scale': self.of_pyr_scale,
            'of_levels': self.of_levels,
            'of_winsize': self.of_winsize,
            'of_iterations': self.of_iterations,
            'of_poly_n': self.of_poly_n,
            'of_poly_sigma': self.of_poly_sigma,
            'upsample_factor': self.upsample_factor,
            'max_features': self.max_features,
            'match_ratio': self.match_ratio,
            'border_margin': self.border_margin,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            method=DICMethod(d.get('method', 'template_ncc')),
            subset_size=d.get('subset_size', 31),
            step_size=d.get('step_size', 5),
            search_radius_x=d.get('search_radius_x', 50),
            search_radius_y=d.get('search_radius_y', 50),
            subpixel_method=SubPixelMethod(d.get('subpixel_method', 'gaussian')),
            correlation_threshold=d.get('correlation_threshold', 0.6),
            of_pyr_scale=d.get('of_pyr_scale', 0.5),
            of_levels=d.get('of_levels', 5),
            of_winsize=d.get('of_winsize', 15),
            of_iterations=d.get('of_iterations', 3),
            of_poly_n=d.get('of_poly_n', 5),
            of_poly_sigma=d.get('of_poly_sigma', 1.2),
            upsample_factor=d.get('upsample_factor', 20),
            max_features=d.get('max_features', 10000),
            match_ratio=d.get('match_ratio', 0.75),
            border_margin=d.get('border_margin', 0),
        )


@dataclass
class DICResult:
    """Holds all output from a single DIC analysis."""
    u: np.ndarray = field(repr=False)
    v: np.ndarray = field(repr=False)
    magnitude: np.ndarray = field(repr=False)
    correlation_quality: np.ndarray = field(repr=False)
    grid_x: np.ndarray = field(repr=False)
    grid_y: np.ndarray = field(repr=False)
    mask_valid: np.ndarray = field(repr=False)
    method_used: DICMethod = DICMethod.TEMPLATE_NCC
    parameters: DICParameters = field(default_factory=DICParameters)
    computation_time_s: float = 0.0
    ref_shape: Tuple[int, int] = (0, 0)

    def to_npz(self, filepath):
        """Save result arrays to a compressed numpy archive."""
        np.savez_compressed(
            filepath,
            u=self.u, v=self.v, magnitude=self.magnitude,
            correlation_quality=self.correlation_quality,
            grid_x=self.grid_x, grid_y=self.grid_y,
            mask_valid=self.mask_valid,
            ref_shape=np.array(self.ref_shape),
        )

    @classmethod
    def from_npz(cls, filepath, params=None):
        """Load result from npz archive."""
        data = np.load(filepath)
        return cls(
            u=data['u'], v=data['v'], magnitude=data['magnitude'],
            correlation_quality=data['correlation_quality'],
            grid_x=data['grid_x'], grid_y=data['grid_y'],
            mask_valid=data['mask_valid'],
            ref_shape=tuple(data['ref_shape']),
            parameters=params or DICParameters(),
        )


class DICEngine:
    """Core DIC computation engine supporting 4 algorithms."""

    def __init__(self, params: DICParameters):
        self.params = params
        self._progress_callback: Optional[Callable] = None
        self._cancelled = False
        self._lock = threading.Lock()

    def set_progress_callback(self, callback: Callable):
        """Set callback: callback(percent: int, message: str)"""
        self._progress_callback = callback

    def cancel(self):
        """Request cancellation of running analysis."""
        with self._lock:
            self._cancelled = True

    def _report_progress(self, percent, message=""):
        with self._lock:
            cb = self._progress_callback
        if cb:
            cb(int(percent), message)

    @staticmethod
    def validate_texture(image: np.ndarray, subset_size: int,
                         mask: np.ndarray = None,
                         min_variance: float = 50.0) -> dict:
        """Validate image texture quality against subset size.

        Computes local variance in windows of subset_size and checks
        if sufficient texture exists for reliable correlation.

        Parameters
        ----------
        image : np.ndarray (H, W) uint8 grayscale
        subset_size : int, DIC subset size
        mask : optional mask (255=analyze)
        min_variance : float, minimum acceptable local variance

        Returns
        -------
        dict with:
            'mean_variance': float, average local variance
            'pct_low_texture': float, % of subsets with low variance
            'warning': str or None, human-readable warning if texture is poor
            'ok': bool, True if texture is adequate
        """
        from scipy.ndimage import uniform_filter

        img = image.astype(np.float64)
        # Local mean and local mean of squares
        local_mean = uniform_filter(img, size=subset_size)
        local_sq_mean = uniform_filter(img ** 2, size=subset_size)
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.maximum(local_var, 0)

        if mask is not None:
            analyze = mask > 0
        else:
            # Exclude borders (half subset from each edge)
            half = subset_size // 2
            analyze = np.zeros_like(image, dtype=bool)
            analyze[half:-half, half:-half] = True

        if not np.any(analyze):
            return {
                'mean_variance': 0.0,
                'pct_low_texture': 100.0,
                'warning': "Nessuna area analizzabile.",
                'ok': False,
            }

        var_values = local_var[analyze]
        mean_var = float(np.mean(var_values))
        pct_low = float(np.sum(var_values < min_variance) / len(var_values) * 100)

        warning = None
        ok = True

        if pct_low > 50:
            warning = (
                f"ATTENZIONE: {pct_low:.0f}% dell'immagine ha varianza locale "
                f"molto bassa (< {min_variance:.0f}) con subset {subset_size}px.\n"
                f"La correlazione DIC potrebbe essere inaffidabile.\n"
                f"Suggerimento: aumentare la dimensione del subset o "
                f"applicare filtri di miglioramento texture (es. CLAHE).")
            ok = False
        elif pct_low > 25:
            warning = (
                f"Nota: {pct_low:.0f}% dell'immagine ha texture insufficiente "
                f"per subset {subset_size}px (varianza media: {mean_var:.0f}).\n"
                f"Considerare un subset piu grande per migliore affidabilita.")
            ok = True  # proceed but warn

        return {
            'mean_variance': mean_var,
            'pct_low_texture': pct_low,
            'warning': warning,
            'ok': ok,
        }

    def run(self, ref_image: np.ndarray, def_image: np.ndarray,
            mask: Optional[np.ndarray] = None) -> DICResult:
        """Run DIC analysis using the selected method.

        Parameters
        ----------
        ref_image : np.ndarray (H, W) uint8 grayscale reference
        def_image : np.ndarray (H, W) uint8 grayscale deformed
        mask : optional np.ndarray (H, W) uint8, 255=analyze, 0=skip

        Returns
        -------
        DICResult
        """
        with self._lock:
            self._cancelled = False
        t0 = time.time()

        if ref_image.size == 0 or def_image.size == 0:
            raise ValueError("Input images must not be empty")

        if ref_image.shape != def_image.shape:
            raise ValueError("Reference and deformed images must have the same shape")

        if ref_image.std() < 5:
            logger.warning("Reference image has very low contrast (std=%.1f). DIC results may be unreliable.", ref_image.std())
        if def_image.std() < 5:
            logger.warning("Deformed image has very low contrast (std=%.1f).", def_image.std())

        method_map = {
            DICMethod.TEMPLATE_NCC: self._run_template_ncc,
            DICMethod.OPTICAL_FLOW_FARNEBACK: self._run_optical_flow,
            DICMethod.PHASE_CORRELATION: self._run_phase_correlation,
            DICMethod.FEATURE_MATCHING: self._run_feature_matching,
        }

        runner = method_map.get(self.params.method)
        if runner is None:
            raise ValueError(f"Unknown method: {self.params.method}")

        result = runner(ref_image, def_image, mask)
        result.computation_time_s = time.time() - t0
        result.method_used = self.params.method
        result.parameters = self.params
        result.ref_shape = ref_image.shape

        # Apply border margin: invalidate grid points within margin of image edges
        bm = self.params.border_margin
        if bm > 0 and result.grid_x is not None:
            h, w = ref_image.shape[:2]
            border_mask = (
                (result.grid_x < bm) | (result.grid_x >= w - bm) |
                (result.grid_y < bm) | (result.grid_y >= h - bm)
            )
            result.u[border_mask] = np.nan
            result.v[border_mask] = np.nan
            result.magnitude[border_mask] = np.nan
            result.correlation_quality[border_mask] = np.nan
            result.mask_valid[border_mask] = False
            n_invalidated = int(np.sum(border_mask))
            if n_invalidated > 0:
                logger.info(f"Border margin {bm}px: invalidated {n_invalidated} "
                             f"edge points")

        logger.info(f"DIC completed in {result.computation_time_s:.1f}s "
                     f"using {self.params.method.value}")
        return result

    # ------------------------------------------------------------------
    # Algorithm 1: Template Matching (NCC)
    # ------------------------------------------------------------------

    def _run_template_ncc(self, ref, deformed, mask) -> DICResult:
        """Subset-based Normalized Cross-Correlation template matching."""
        h, w = ref.shape
        half = self.params.subset_size // 2
        step = self.params.step_size
        sr_x = self.params.search_radius_x
        sr_y = self.params.search_radius_y

        # Generate grid of analysis points
        gy = np.arange(half, h - half, step)
        gx = np.arange(half, w - half, step)
        ny, nx = len(gy), len(gx)

        # Output arrays
        u = np.full((ny, nx), np.nan, dtype=np.float64)
        v = np.full((ny, nx), np.nan, dtype=np.float64)
        quality = np.full((ny, nx), np.nan, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(gx, gy)

        total_points = ny * nx
        processed = 0
        skipped_low_corr = 0

        self._report_progress(0, "Template matching NCC...")

        for iy in range(ny):
            with self._lock:
                cancelled = self._cancelled
            if cancelled:
                break
            for ix in range(nx):
                cy, cx = gy[iy], gx[ix]

                # Check mask
                if mask is not None and mask[cy, cx] == 0:
                    processed += 1
                    continue

                # Extract template from reference
                t_y0, t_y1 = cy - half, cy + half + 1
                t_x0, t_x1 = cx - half, cx + half + 1
                template = ref[t_y0:t_y1, t_x0:t_x1]

                if template.size == 0:
                    processed += 1
                    continue

                # Define search window in deformed image
                s_y0 = max(0, cy - half - sr_y)
                s_y1 = min(h, cy + half + sr_y + 1)
                s_x0 = max(0, cx - half - sr_x)
                s_x1 = min(w, cx + half + sr_x + 1)
                search_window = deformed[s_y0:s_y1, s_x0:s_x1]

                if search_window.size == 0:
                    processed += 1
                    continue

                # Check search window is large enough
                if (search_window.shape[0] < template.shape[0] or
                        search_window.shape[1] < template.shape[1]):
                    processed += 1
                    continue

                # Template matching (ZNCC – zero-mean normalised cross-correlation)
                result_map = cv2.matchTemplate(
                    search_window, template, cv2.TM_CCOEFF_NORMED)

                _, max_val, _, max_loc = cv2.minMaxLoc(result_map)

                if max_val < self.params.correlation_threshold:
                    skipped_low_corr += 1
                    processed += 1
                    continue

                # Integer pixel displacement
                match_x = max_loc[0]
                match_y = max_loc[1]

                # Sub-pixel refinement
                if self.params.subpixel_method != SubPixelMethod.NONE:
                    sub_x, sub_y = self._subpixel_refinement(
                        result_map, match_x, match_y)
                else:
                    sub_x, sub_y = float(match_x), float(match_y)

                # Expected position of template center in search window
                expected_x = float(t_x0 - s_x0)
                expected_y = float(t_y0 - s_y0)

                u[iy, ix] = sub_x - expected_x
                v[iy, ix] = sub_y - expected_y
                quality[iy, ix] = max_val

                processed += 1

            # Progress per row
            self._report_progress(
                (processed / total_points) * 100,
                f"NCC: row {iy + 1}/{ny}"
            )

        if total_points > 0 and skipped_low_corr > total_points * 0.5:
            logger.warning("%.0f%% of points skipped due to low correlation (threshold=%.2f). Consider lowering the threshold.",
                           skipped_low_corr / total_points * 100, self.params.correlation_threshold)

        magnitude = compute_magnitude(u, v)
        mask_valid = ~np.isnan(u)

        return DICResult(
            u=u, v=v, magnitude=magnitude,
            correlation_quality=quality,
            grid_x=grid_x, grid_y=grid_y,
            mask_valid=mask_valid,
        )

    def _subpixel_refinement(self, corr_surface, peak_x, peak_y):
        """Refine integer peak location to sub-pixel accuracy.

        Parameters
        ----------
        corr_surface : np.ndarray, correlation result map
        peak_x, peak_y : int, integer peak location

        Returns
        -------
        (sub_x, sub_y) : float, refined peak location
        """
        h, w = corr_surface.shape
        sub_x, sub_y = float(peak_x), float(peak_y)

        if self.params.subpixel_method == SubPixelMethod.PARABOLIC:
            # 3-point parabolic fit
            if 1 <= peak_x <= w - 2:
                c_left = float(corr_surface[peak_y, peak_x - 1])
                c_center = float(corr_surface[peak_y, peak_x])
                c_right = float(corr_surface[peak_y, peak_x + 1])
                denom = 2.0 * (c_left - 2 * c_center + c_right)
                if abs(denom) > 1e-6 * (abs(c_left) + abs(c_center) + abs(c_right) + 1e-10):
                    sub_x = peak_x + (c_left - c_right) / denom

            if 1 <= peak_y <= h - 2:
                c_top = float(corr_surface[peak_y - 1, peak_x])
                c_center = float(corr_surface[peak_y, peak_x])
                c_bottom = float(corr_surface[peak_y + 1, peak_x])
                denom = 2.0 * (c_top - 2 * c_center + c_bottom)
                if abs(denom) > 1e-6 * (abs(c_top) + abs(c_center) + abs(c_bottom) + 1e-10):
                    sub_y = peak_y + (c_top - c_bottom) / denom

            # Clamp sub-pixel offset to at most +/-0.5 pixels
            sub_x = np.clip(sub_x, peak_x - 0.5, peak_x + 0.5)
            sub_y = np.clip(sub_y, peak_y - 0.5, peak_y + 0.5)

        elif self.params.subpixel_method == SubPixelMethod.GAUSSIAN:
            # 3-point Gaussian fit – requires all 3 values > 0.
            # With TM_CCOEFF_NORMED correlation can be negative, so we
            # fall back to parabolic fitting when Gaussian is not applicable.
            _gauss_ok_x, _gauss_ok_y = False, False

            if 1 <= peak_x <= w - 2:
                c_left = float(corr_surface[peak_y, peak_x - 1])
                c_center = float(corr_surface[peak_y, peak_x])
                c_right = float(corr_surface[peak_y, peak_x + 1])
                if c_left > 0 and c_center > 0 and c_right > 0:
                    ln_left = np.log(c_left)
                    ln_center = np.log(c_center)
                    ln_right = np.log(c_right)
                    denom = 2.0 * (ln_left - 2 * ln_center + ln_right)
                    if abs(denom) > 1e-6 * (abs(ln_left) + abs(ln_center) + abs(ln_right) + 1e-10):
                        sub_x = peak_x + (ln_left - ln_right) / denom
                        _gauss_ok_x = True
                if not _gauss_ok_x:
                    # Fallback: parabolic fit for X
                    denom = 2.0 * (c_left - 2 * c_center + c_right)
                    if abs(denom) > 1e-6 * (abs(c_left) + abs(c_center) + abs(c_right) + 1e-10):
                        sub_x = peak_x + (c_left - c_right) / denom

            if 1 <= peak_y <= h - 2:
                c_top = float(corr_surface[peak_y - 1, peak_x])
                c_center = float(corr_surface[peak_y, peak_x])
                c_bottom = float(corr_surface[peak_y + 1, peak_x])
                if c_top > 0 and c_center > 0 and c_bottom > 0:
                    ln_top = np.log(c_top)
                    ln_center = np.log(c_center)
                    ln_bottom = np.log(c_bottom)
                    denom = 2.0 * (ln_top - 2 * ln_center + ln_bottom)
                    if abs(denom) > 1e-6 * (abs(ln_top) + abs(ln_center) + abs(ln_bottom) + 1e-10):
                        sub_y = peak_y + (ln_top - ln_bottom) / denom
                        _gauss_ok_y = True
                if not _gauss_ok_y:
                    # Fallback: parabolic fit for Y
                    denom = 2.0 * (c_top - 2 * c_center + c_bottom)
                    if abs(denom) > 1e-6 * (abs(c_top) + abs(c_center) + abs(c_bottom) + 1e-10):
                        sub_y = peak_y + (c_top - c_bottom) / denom

            # Clamp sub-pixel offset to at most +/-0.5 pixels
            sub_x = np.clip(sub_x, peak_x - 0.5, peak_x + 0.5)
            sub_y = np.clip(sub_y, peak_y - 0.5, peak_y + 0.5)

        elif self.params.subpixel_method == SubPixelMethod.BICUBIC:
            # Bicubic interpolation on 5x5 neighborhood
            try:
                from scipy.interpolate import RectBivariateSpline
                from scipy.optimize import minimize

                # Extract 5x5 neighborhood
                y0 = max(0, peak_y - 2)
                y1 = min(h, peak_y + 3)
                x0 = max(0, peak_x - 2)
                x1 = min(w, peak_x + 3)

                patch = corr_surface[y0:y1, x0:x1].astype(np.float64)
                py_vals = np.arange(y0, y1).astype(np.float64)
                px_vals = np.arange(x0, x1).astype(np.float64)

                if patch.shape[0] >= 3 and patch.shape[1] >= 3:
                    spline = RectBivariateSpline(
                        py_vals, px_vals, patch, kx=3, ky=3)

                    def neg_corr(xy):
                        return -float(spline(xy[1], xy[0]))

                    res = minimize(neg_corr, [float(peak_x), float(peak_y)],
                                   method='Nelder-Mead',
                                   options={'xatol': 0.001, 'fatol': 1e-8})
                    sub_x, sub_y = res.x[0], res.x[1]
            except ImportError:
                pass

        return sub_x, sub_y

    # ------------------------------------------------------------------
    # Algorithm 2: Dense Optical Flow (Farneback)
    # ------------------------------------------------------------------

    def _run_optical_flow(self, ref, deformed, mask) -> DICResult:
        """Dense optical flow using Farneback's algorithm."""
        self._report_progress(10, "Computing optical flow...")

        flow = cv2.calcOpticalFlowFarneback(
            ref, deformed, None,
            pyr_scale=self.params.of_pyr_scale,
            levels=self.params.of_levels,
            winsize=self.params.of_winsize,
            iterations=self.params.of_iterations,
            poly_n=self.params.of_poly_n,
            poly_sigma=self.params.of_poly_sigma,
            flags=0
        )

        self._report_progress(60, "Subsampling to grid...")

        # Full-pixel flow field
        u_full = flow[:, :, 0].astype(np.float64)
        v_full = flow[:, :, 1].astype(np.float64)

        # Subsample to grid
        h, w = ref.shape
        step = self.params.step_size
        half = self.params.subset_size // 2

        gy = np.arange(half, h - half, step)
        gx = np.arange(half, w - half, step)
        grid_x, grid_y = np.meshgrid(gx, gy)

        u = u_full[grid_y, grid_x]
        v = v_full[grid_y, grid_x]

        # Free full-resolution arrays (can be large for high-res images)
        del flow, u_full, v_full

        # Apply mask
        if mask is not None:
            mask_grid = mask[grid_y, grid_x]
            u[mask_grid == 0] = np.nan
            v[mask_grid == 0] = np.nan

        self._report_progress(80, "Computing correlation quality...")

        # Post-hoc correlation quality estimation
        quality = self._compute_posthoc_ncc(ref, deformed, u, v, grid_x, grid_y)

        # Apply correlation threshold
        low_quality = quality < self.params.correlation_threshold
        u[low_quality] = np.nan
        v[low_quality] = np.nan

        magnitude = compute_magnitude(u, v)
        mask_valid = ~np.isnan(u)

        self._report_progress(100, "Optical flow complete")

        return DICResult(
            u=u, v=v, magnitude=magnitude,
            correlation_quality=quality,
            grid_x=grid_x, grid_y=grid_y,
            mask_valid=mask_valid,
        )

    def _compute_posthoc_ncc(self, ref, deformed, u, v, grid_x, grid_y):
        """Compute NCC quality metric at each grid point after flow estimation."""
        half = self.params.subset_size // 2
        h, w = ref.shape
        ny, nx = u.shape
        quality = np.full((ny, nx), np.nan, dtype=np.float64)

        for iy in range(ny):
            with self._lock:
                cancelled = self._cancelled
            if cancelled:
                break
            for ix in range(nx):
                if np.isnan(u[iy, ix]):
                    continue
                cy = int(grid_y[iy, ix])
                cx = int(grid_x[iy, ix])

                # Reference subset
                r_y0, r_y1 = cy - half, cy + half + 1
                r_x0, r_x1 = cx - half, cx + half + 1

                # Shifted position in deformed
                dy = int(round(v[iy, ix]))
                dx = int(round(u[iy, ix]))
                d_y0 = r_y0 + dy
                d_y1 = r_y1 + dy
                d_x0 = r_x0 + dx
                d_x1 = r_x1 + dx

                # Bounds check
                if d_y0 < 0 or d_y1 > h or d_x0 < 0 or d_x1 > w:
                    continue

                ref_patch = ref[r_y0:r_y1, r_x0:r_x1].astype(np.float64)
                def_patch = deformed[d_y0:d_y1, d_x0:d_x1].astype(np.float64)

                # Normalized cross-correlation
                ref_mean = np.mean(ref_patch)
                def_mean = np.mean(def_patch)
                ref_norm = ref_patch - ref_mean
                def_norm = def_patch - def_mean

                num = np.sum(ref_norm * def_norm)
                den = np.sqrt(np.sum(ref_norm ** 2) * np.sum(def_norm ** 2))
                if den > 1e-10:
                    quality[iy, ix] = num / den

        return quality

    # ------------------------------------------------------------------
    # Algorithm 3: Phase Correlation (FFT)
    # ------------------------------------------------------------------

    def _run_phase_correlation(self, ref, deformed, mask) -> DICResult:
        """Windowed FFT phase correlation for displacement estimation."""
        h, w = ref.shape
        half = self.params.subset_size // 2
        step = self.params.step_size
        upsample = self.params.upsample_factor

        gy = np.arange(half, h - half, step)
        gx = np.arange(half, w - half, step)
        ny, nx = len(gy), len(gx)

        u = np.full((ny, nx), np.nan, dtype=np.float64)
        v = np.full((ny, nx), np.nan, dtype=np.float64)
        quality = np.full((ny, nx), np.nan, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(gx, gy)

        # Hanning window for spectral leakage reduction
        hanning = np.outer(
            np.hanning(self.params.subset_size),
            np.hanning(self.params.subset_size)
        )

        total = ny * nx
        processed = 0

        self._report_progress(0, "Phase correlation FFT...")

        try:
            from skimage.registration import phase_cross_correlation
        except ImportError:
            logger.warning("scikit-image not available, using basic FFT")
            phase_cross_correlation = None

        for iy in range(ny):
            with self._lock:
                cancelled = self._cancelled
            if cancelled:
                break
            for ix in range(nx):
                cy, cx = gy[iy], gx[ix]

                if mask is not None and mask[cy, cx] == 0:
                    processed += 1
                    continue

                y0, y1 = cy - half, cy + half + 1
                x0, x1 = cx - half, cx + half + 1

                # Ensure we get exactly subset_size
                ref_win = ref[y0:y0 + self.params.subset_size,
                              x0:x0 + self.params.subset_size].astype(np.float64)
                def_win = deformed[y0:y0 + self.params.subset_size,
                                   x0:x0 + self.params.subset_size].astype(np.float64)

                if ref_win.shape != hanning.shape or def_win.shape != hanning.shape:
                    processed += 1
                    continue

                ref_win *= hanning
                def_win *= hanning

                if phase_cross_correlation is not None:
                    shift, error, diffphase = phase_cross_correlation(
                        ref_win, def_win,
                        upsample_factor=upsample,
                        return_error=True
                    )
                    v[iy, ix] = shift[0]
                    u[iy, ix] = shift[1]
                    quality[iy, ix] = 1.0 - error if error <= 1.0 else 0.0
                else:
                    # Basic FFT phase correlation
                    shift, corr_peak = self._basic_phase_correlation(ref_win, def_win)
                    v[iy, ix] = shift[0]
                    u[iy, ix] = shift[1]
                    quality[iy, ix] = corr_peak

                processed += 1

            self._report_progress(
                (processed / total) * 100,
                f"Phase correlation: row {iy + 1}/{ny}"
            )

        # Apply threshold
        low_q = quality < self.params.correlation_threshold
        u[low_q] = np.nan
        v[low_q] = np.nan

        magnitude = compute_magnitude(u, v)
        mask_valid = ~np.isnan(u)

        return DICResult(
            u=u, v=v, magnitude=magnitude,
            correlation_quality=quality,
            grid_x=grid_x, grid_y=grid_y,
            mask_valid=mask_valid,
        )

    @staticmethod
    def _basic_phase_correlation(ref_win, def_win):
        """Basic FFT phase correlation fallback."""
        f_ref = np.fft.fft2(ref_win)
        f_def = np.fft.fft2(def_win)
        cross_power = f_ref * np.conj(f_def)
        denom = np.abs(cross_power)
        denom[denom < 1e-10] = 1e-10
        cross_power_norm = cross_power / denom
        correlation = np.fft.ifft2(cross_power_norm).real

        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        peak_val = correlation[peak_idx]

        # Convert to signed shift
        shift_y = peak_idx[0]
        shift_x = peak_idx[1]
        if shift_y > ref_win.shape[0] // 2:
            shift_y -= ref_win.shape[0]
        if shift_x > ref_win.shape[1] // 2:
            shift_x -= ref_win.shape[1]

        return (float(shift_y), float(shift_x)), float(peak_val)

    # ------------------------------------------------------------------
    # Algorithm 4: Feature Matching (ORB/SIFT)
    # ------------------------------------------------------------------

    def _run_feature_matching(self, ref, deformed, mask) -> DICResult:
        """Sparse feature matching with grid interpolation."""
        self._report_progress(10, "Detecting features...")

        # Try SIFT first (more accurate), fall back to ORB
        try:
            detector = cv2.SIFT_create(nfeatures=self.params.max_features)
            norm_type = cv2.NORM_L2
        except cv2.error:
            detector = cv2.ORB_create(nfeatures=self.params.max_features)
            norm_type = cv2.NORM_HAMMING

        mask_input = mask if mask is not None else None
        kp_ref, desc_ref = detector.detectAndCompute(ref, mask_input)
        kp_def, desc_def = detector.detectAndCompute(deformed, None)

        if desc_ref is None or desc_def is None or len(kp_ref) < 4 or len(kp_def) < 4:
            logger.warning("Not enough features detected for matching")
            return self._empty_result(ref.shape)

        self._report_progress(30, f"Matching {len(kp_ref)} features...")

        # Match using BFMatcher with ratio test
        bf = cv2.BFMatcher(norm_type)
        try:
            matches = bf.knnMatch(desc_ref, desc_def, k=2)
        except cv2.error:
            return self._empty_result(ref.shape)

        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.params.match_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            logger.warning(f"Only {len(good_matches)} good matches found")
            return self._empty_result(ref.shape)

        self._report_progress(50, f"{len(good_matches)} good matches, interpolating...")

        # Extract displacement vectors from matches
        pts_ref = np.array([kp_ref[m.queryIdx].pt for m in good_matches])
        pts_def = np.array([kp_def[m.trainIdx].pt for m in good_matches])
        disp_u = pts_def[:, 0] - pts_ref[:, 0]
        disp_v = pts_def[:, 1] - pts_ref[:, 1]
        max_dist = max(m.distance for m in good_matches) if good_matches else 1.0
        max_dist = max(max_dist, 1.0)  # avoid division by zero
        match_quality = np.array([1.0 - m.distance / max_dist for m in good_matches])
        match_quality = np.clip(match_quality, 0, 1)

        self._report_progress(60, "Interpolating to grid...")

        # Interpolate sparse matches to regular grid
        from scipy.interpolate import griddata

        h, w = ref.shape
        step = self.params.step_size
        half = self.params.subset_size // 2

        gy = np.arange(half, h - half, step)
        gx = np.arange(half, w - half, step)
        grid_x, grid_y = np.meshgrid(gx, gy)

        # Interpolate displacement fields
        interp_method = 'cubic' if len(good_matches) >= 20 else 'linear'
        try:
            u = griddata(pts_ref, disp_u, (grid_x, grid_y), method=interp_method)
            v = griddata(pts_ref, disp_v, (grid_x, grid_y), method=interp_method)
            q = griddata(pts_ref, match_quality, (grid_x, grid_y), method=interp_method)
        except Exception:
            u = griddata(pts_ref, disp_u, (grid_x, grid_y), method='linear')
            v = griddata(pts_ref, disp_v, (grid_x, grid_y), method='linear')
            q = griddata(pts_ref, match_quality, (grid_x, grid_y), method='linear')

        # Fill NaN only where a match point is close enough (within
        # max_fill_distance pixels).  Points far from any match are
        # unreliable and must stay NaN rather than being hallucinated by
        # nearest-neighbour extrapolation.
        from scipy.spatial import cKDTree

        nan_mask = np.isnan(u)
        if np.any(nan_mask) and np.any(~nan_mask):
            max_fill_dist = max(step * 3, self.params.subset_size * 2)
            nan_coords = np.column_stack((
                grid_x[nan_mask].ravel(), grid_y[nan_mask].ravel()))
            tree = cKDTree(pts_ref)
            dists, _ = tree.query(nan_coords, k=1)
            close_enough = dists <= max_fill_dist

            if np.any(close_enough):
                u_nn = griddata(pts_ref, disp_u, (grid_x, grid_y),
                                method='nearest')
                v_nn = griddata(pts_ref, disp_v, (grid_x, grid_y),
                                method='nearest')
                # Build full-size boolean mask from the subset of NaN positions
                fill_mask = np.zeros_like(u, dtype=bool)
                nan_indices = np.argwhere(nan_mask)
                for idx, ok in zip(nan_indices, close_enough):
                    if ok:
                        fill_mask[idx[0], idx[1]] = True
                u[fill_mask] = u_nn[fill_mask]
                v[fill_mask] = v_nn[fill_mask]

        if q is None:
            q = np.ones_like(u) * 0.5
        q = np.nan_to_num(q, nan=0.0)

        # Apply mask
        if mask is not None:
            mask_grid = mask[grid_y, grid_x]
            u[mask_grid == 0] = np.nan
            v[mask_grid == 0] = np.nan

        magnitude = compute_magnitude(u, v)
        mask_valid = ~np.isnan(u)

        self._report_progress(100, "Feature matching complete")

        return DICResult(
            u=u, v=v, magnitude=magnitude,
            correlation_quality=q,
            grid_x=grid_x, grid_y=grid_y,
            mask_valid=mask_valid,
        )

    def _empty_result(self, shape):
        """Create an empty result when analysis fails."""
        step = self.params.step_size
        half = self.params.subset_size // 2
        h, w = shape
        gy = np.arange(half, h - half, step)
        gx = np.arange(half, w - half, step)
        grid_x, grid_y = np.meshgrid(gx, gy)
        ny, nx = grid_x.shape
        nans = np.full((ny, nx), np.nan)
        return DICResult(
            u=nans.copy(), v=nans.copy(), magnitude=nans.copy(),
            correlation_quality=nans.copy(),
            grid_x=grid_x, grid_y=grid_y,
            mask_valid=np.zeros((ny, nx), dtype=bool),
        )


class DICSequenceRunner:
    """Run DIC on image sequences for multi-temporal monitoring."""

    def __init__(self, engine: DICEngine):
        self.engine = engine
        self.results: List[DICResult] = []

    def run_sequence(self, reference: np.ndarray,
                     deformed_images: List[np.ndarray],
                     mask: Optional[np.ndarray] = None,
                     incremental: bool = False,
                     progress_callback: Optional[Callable] = None) -> List[DICResult]:
        """Run DIC for each deformed image.

        Parameters
        ----------
        reference : np.ndarray, reference image
        deformed_images : list of np.ndarray, deformed images
        mask : optional mask
        incremental : if True, compare each image to the previous one
                      if False, compare all to the reference
        progress_callback : optional overall progress callback
        """
        self.results = []
        n_images = len(deformed_images)

        current_ref = reference

        for i, def_img in enumerate(deformed_images):
            if progress_callback:
                progress_callback(
                    int((i / n_images) * 100),
                    f"Processing image {i + 1}/{n_images}"
                )

            result = self.engine.run(current_ref, def_img, mask)
            self.results.append(result)

            if incremental:
                current_ref = def_img

        if progress_callback:
            progress_callback(100, "Sequence analysis complete")

        return self.results

    def get_timeseries_at_point(self, grid_row: int, grid_col: int) -> dict:
        """Extract displacement time series at a specific grid point.

        Returns
        -------
        dict with keys 'u', 'v', 'magnitude', 'quality' (each a list of float)
        """
        ts = {'u': [], 'v': [], 'magnitude': [], 'quality': []}
        for result in self.results:
            if (0 <= grid_row < result.u.shape[0] and
                    0 <= grid_col < result.u.shape[1]):
                ts['u'].append(float(result.u[grid_row, grid_col]))
                ts['v'].append(float(result.v[grid_row, grid_col]))
                ts['magnitude'].append(float(result.magnitude[grid_row, grid_col]))
                ts['quality'].append(float(result.correlation_quality[grid_row, grid_col]))
        return ts

    def get_cumulative_displacement(self) -> Optional[DICResult]:
        """Sum incremental displacements to get cumulative field.

        Only meaningful when incremental=True was used in run_sequence.
        """
        if not self.results:
            return None

        cum_u = np.zeros_like(self.results[0].u)
        cum_v = np.zeros_like(self.results[0].v)
        cum_weights = np.zeros_like(self.results[0].u)
        min_quality = np.ones_like(self.results[0].correlation_quality)

        for result in self.results:
            valid = ~np.isnan(result.u)
            q = np.nan_to_num(result.correlation_quality, nan=0)
            mean_q = float(np.nanmean(q[valid])) if np.any(valid) else 0.5
            weight = max(mean_q, 0.1)  # floor to avoid zero weight
            cum_u[valid] += result.u[valid] * weight
            cum_v[valid] += result.v[valid] * weight
            cum_weights[valid] += weight
            min_quality = np.minimum(min_quality, q)

        # Normalize by cumulative weights
        nonzero = cum_weights > 0
        cum_u[nonzero] /= cum_weights[nonzero]
        cum_v[nonzero] /= cum_weights[nonzero]

        cum_magnitude = compute_magnitude(cum_u, cum_v)
        mask_valid = ~np.isnan(cum_u)

        return DICResult(
            u=cum_u, v=cum_v, magnitude=cum_magnitude,
            correlation_quality=min_quality,
            grid_x=self.results[0].grid_x,
            grid_y=self.results[0].grid_y,
            mask_valid=mask_valid,
            ref_shape=self.results[0].ref_shape,
        )
