"""Post-processing: strain computation, smoothing, outlier removal, statistics."""

import numpy as np
import cv2
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.ndimage import uniform_filter
from scipy.interpolate import RectBivariateSpline
from dic_app.utils.helpers import nan_safe_gaussian_filter, setup_logger

logger = setup_logger(__name__)


# -------------------------------------------------------------------
# Helper: NaN-safe local median via scipy (C-optimized, no Python callback)
# -------------------------------------------------------------------

def _nan_safe_median_filter(field, size):
    """Compute local median that approximates NaN-safe behaviour.

    Strategy: replace NaN with the global median (neutral value),
    run the fast C-based scipy median_filter, then restore NaN
    positions.  This is ~100× faster than generic_filter with a
    Python callback and uses negligible extra memory.
    """
    nan_mask = np.isnan(field)
    if not np.any(nan_mask):
        return scipy_median_filter(field, size=size)

    valid_vals = field[~nan_mask]
    if len(valid_vals) == 0:
        return field.copy()

    fill_val = np.median(valid_vals)
    filled = field.copy()
    filled[nan_mask] = fill_val
    result = scipy_median_filter(filled, size=size)
    result[nan_mask] = np.nan
    return result


def _nan_safe_local_mad(field, local_median, size):
    """Compute local MAD (Median Absolute Deviation) using uniform_filter.

    MAD = median(|x - median(x)|).  We approximate the outer median
    with a mean (uniform_filter) which is very fast in C.  For the
    purpose of outlier detection this approximation is sufficient —
    the threshold multiplier absorbs the factor difference (~1.48).
    """
    nan_mask = np.isnan(field)
    abs_dev = np.abs(field - local_median)
    abs_dev[nan_mask] = 0.0
    valid = (~nan_mask).astype(np.float64)

    # Weighted mean of |deviation| ≈ MAD (up to constant factor)
    dev_sum = uniform_filter(abs_dev, size=size)
    count = uniform_filter(valid, size=size)
    count = np.maximum(count, 1e-10)

    local_mad = dev_sum / count
    local_mad[nan_mask] = np.nan
    return local_mad


class DisplacementSmoother:
    """Smoothing methods for noisy displacement fields."""

    @staticmethod
    def gaussian_smooth(u, v, sigma=2.0):
        """Apply NaN-safe Gaussian filter to displacement fields.

        Parameters
        ----------
        u, v : np.ndarray (NY, NX) displacement components
        sigma : float, Gaussian kernel sigma

        Returns
        -------
        (u_smooth, v_smooth)
        """
        u_s = nan_safe_gaussian_filter(u, sigma)
        v_s = nan_safe_gaussian_filter(v, sigma)
        return u_s, v_s

    @staticmethod
    def spline_smooth(u, v, grid_x, grid_y, smoothing_factor=None):
        """Fit cubic spline to displacement fields.

        Parameters
        ----------
        u, v : np.ndarray (NY, NX)
        grid_x, grid_y : np.ndarray coordinate arrays
        smoothing_factor : float or None, controls fit quality vs smoothness

        Returns
        -------
        (u_smooth, v_smooth)
        """
        ny, nx = u.shape

        rows = np.arange(ny).astype(np.float64)
        cols = np.arange(nx).astype(np.float64)

        valid_mask = ~np.isnan(u)

        u_filled = np.where(valid_mask, u, 0.0)
        v_filled = np.where(valid_mask, v, 0.0)
        weights = valid_mask.astype(np.float64)

        if smoothing_factor is None:
            smoothing_factor = ny * nx * 0.1

        try:
            spline_u = RectBivariateSpline(rows, cols, u_filled,
                                            w=weights, s=smoothing_factor)
            spline_v = RectBivariateSpline(rows, cols, v_filled,
                                            w=weights, s=smoothing_factor)
            u_smooth = spline_u(rows, cols)
            v_smooth = spline_v(rows, cols)
        except Exception:
            logger.warning("Spline smoothing failed, returning original data")
            return u.copy(), v.copy()

        u_smooth[~valid_mask] = np.nan
        v_smooth[~valid_mask] = np.nan

        return u_smooth, v_smooth

    @staticmethod
    def median_smooth(u, v, kernel_size=5):
        """Apply median filter to remove outlier displacement vectors.

        Parameters
        ----------
        kernel_size : int, filter window size
        """
        nan_mask = np.isnan(u)
        u_filled = np.nan_to_num(u, nan=0.0)
        v_filled = np.nan_to_num(v, nan=0.0)

        u_med = scipy_median_filter(u_filled, size=kernel_size)
        v_med = scipy_median_filter(v_filled, size=kernel_size)

        u_med[nan_mask] = np.nan
        v_med[nan_mask] = np.nan

        return u_med, v_med

    @staticmethod
    def outlier_removal(u, v, threshold_std=3.0, window_size=5):
        """Remove outliers using Normalized Median Test (NMT).

        For each point the local median and local MAD of the
        displacement components are computed.  A point is flagged as
        outlier when its residual exceeds *threshold_std × local_MAD*.

        Uses C-optimized scipy.ndimage filters instead of Python
        callbacks for ~100× speedup and minimal memory overhead.

        Parameters
        ----------
        threshold_std : float, NMT multiplier (typically 3–5)
        window_size : int, neighbourhood size (must be odd, ≥ 3)

        Returns
        -------
        (u_clean, v_clean) with outliers set to NaN
        """
        u_clean = u.copy()
        v_clean = v.copy()

        nan_mask = np.isnan(u) | np.isnan(v)

        if np.all(nan_mask):
            return u_clean, v_clean

        window_size = max(3, window_size | 1)  # ensure odd and ≥ 3

        # Local medians (C-optimized)
        u_local_med = _nan_safe_median_filter(u, size=window_size)
        v_local_med = _nan_safe_median_filter(v, size=window_size)

        # Local MAD (fast approximation via uniform_filter)
        u_local_mad = _nan_safe_local_mad(u, u_local_med, size=window_size)
        v_local_mad = _nan_safe_local_mad(v, v_local_med, size=window_size)

        # Floor to avoid division by zero in uniform fields
        mad_floor = 0.1
        u_local_mad = np.maximum(u_local_mad, mad_floor)
        v_local_mad = np.maximum(v_local_mad, mad_floor)

        # Normalized residual
        u_residual = np.abs(u - u_local_med) / u_local_mad
        v_residual = np.abs(v - v_local_med) / v_local_mad

        # A point is outlier if EITHER component exceeds the threshold
        outlier_mask = ((u_residual > threshold_std) |
                        (v_residual > threshold_std))
        outlier_mask = outlier_mask & ~nan_mask

        n_outliers = int(np.sum(outlier_mask))
        if n_outliers > 0:
            logger.info(f"NMT outlier removal: {n_outliers} points removed "
                        f"(threshold={threshold_std}, window={window_size})")
            u_clean[outlier_mask] = np.nan
            v_clean[outlier_mask] = np.nan

        return u_clean, v_clean

    @staticmethod
    def spatial_coherence_filter(u, v, window_size=3, threshold=3.0):
        """Remove spatially incoherent displacement vectors.

        Each vector is compared to the median of its neighbours.
        If the Euclidean distance between the vector and the local
        median vector exceeds *threshold × local_MAD*, the point is
        replaced with NaN.

        Uses C-optimized filters for performance.

        Parameters
        ----------
        window_size : int, neighbourhood size (must be odd, ≥ 3)
        threshold : float, multiplier for local MAD

        Returns
        -------
        (u_filtered, v_filtered)
        """
        u_out = u.copy()
        v_out = v.copy()
        nan_mask = np.isnan(u) | np.isnan(v)

        if np.all(nan_mask):
            return u_out, v_out

        window_size = max(3, window_size | 1)

        # Local medians (C-optimized)
        u_med = _nan_safe_median_filter(u, size=window_size)
        v_med = _nan_safe_median_filter(v, size=window_size)

        # Distance of each vector from its local median vector
        dist = np.sqrt((u - u_med) ** 2 + (v - v_med) ** 2)

        # Local MAD of the distance field
        dist_med = _nan_safe_median_filter(dist, size=window_size)
        dist_mad = _nan_safe_local_mad(dist, dist_med, size=window_size)
        dist_mad = np.maximum(dist_mad, 0.1)  # floor

        incoherent = (dist / dist_mad > threshold) & ~nan_mask

        n_removed = int(np.sum(incoherent))
        if n_removed > 0:
            logger.info(f"Spatial coherence filter: {n_removed} vectors removed "
                        f"(threshold={threshold}, window={window_size})")
            u_out[incoherent] = np.nan
            v_out[incoherent] = np.nan

        return u_out, v_out


class StrainCalculator:
    """Compute strain fields from displacement data."""

    @staticmethod
    def compute_strain(u, v, grid_spacing=1.0):
        """Compute 2D strain tensor components from displacement fields.

        Computes both Green-Lagrangian (finite) and engineering (small) strains.

        Parameters
        ----------
        u, v : np.ndarray (NY, NX) displacement in x and y
        grid_spacing : float, physical spacing between grid points

        Returns
        -------
        dict with keys:
            'E_xx', 'E_yy', 'E_xy' : Green-Lagrangian strain components
            'eps_xx', 'eps_yy', 'gamma_xy' : Engineering strain components
            'principal_1', 'principal_2' : Principal strains
            'max_shear' : Maximum shear strain
            'principal_angle' : Principal direction (radians)
            'von_mises' : Von Mises equivalent strain
        """
        du_dy, du_dx = np.gradient(u, grid_spacing)
        dv_dy, dv_dx = np.gradient(v, grid_spacing)

        # ---- Engineering (small) strain ----
        eps_xx = du_dx
        eps_yy = dv_dy
        gamma_xy = du_dy + dv_dx

        # ---- Green-Lagrangian strain ----
        F11 = 1.0 + du_dx
        F12 = du_dy
        F21 = dv_dx
        F22 = 1.0 + dv_dy

        E_xx = 0.5 * (F11 * F11 + F21 * F21 - 1.0)
        E_yy = 0.5 * (F12 * F12 + F22 * F22 - 1.0)
        E_xy = 0.5 * (F11 * F12 + F21 * F22)

        # ---- Principal strains ----
        principal_1, principal_2, max_shear, principal_angle = \
            StrainCalculator.compute_principal_strains(E_xx, E_yy, E_xy)

        # ---- Von Mises equivalent strain ----
        von_mises = np.sqrt(
            E_xx ** 2 + E_yy ** 2 - E_xx * E_yy + 3.0 * E_xy ** 2
        )

        return {
            'E_xx': E_xx,
            'E_yy': E_yy,
            'E_xy': E_xy,
            'eps_xx': eps_xx,
            'eps_yy': eps_yy,
            'gamma_xy': gamma_xy,
            'principal_1': principal_1,
            'principal_2': principal_2,
            'max_shear': max_shear,
            'principal_angle': principal_angle,
            'von_mises': von_mises,
        }

    @staticmethod
    def compute_principal_strains(E_xx, E_yy, E_xy):
        """Eigenvalue decomposition of 2D strain tensor at each point.

        Returns
        -------
        (E1, E2, max_shear, angle)
        """
        mean_strain = 0.5 * (E_xx + E_yy)
        diff_sq = 0.25 * (E_xx - E_yy) ** 2 + E_xy ** 2
        diff_sq = np.maximum(diff_sq, 0)
        radius = np.sqrt(diff_sq)

        E1 = mean_strain + radius
        E2 = mean_strain - radius
        max_shear = radius

        angle = 0.5 * np.arctan2(2.0 * E_xy, E_xx - E_yy)

        return E1, E2, max_shear, angle


class DisplacementStatistics:
    """Statistical analysis of displacement fields."""

    @staticmethod
    def compute_statistics(u, v, magnitude, quality, gsd=None):
        """Compute comprehensive statistics of displacement field.

        Parameters
        ----------
        u, v, magnitude, quality : np.ndarray
        gsd : float or None, meters/pixel for metric conversion

        Returns
        -------
        dict of statistics
        """
        valid = ~np.isnan(magnitude)
        valid_mag = magnitude[valid]

        if len(valid_mag) == 0:
            return {'n_valid_points': 0}

        stats = {
            'n_valid_points': int(np.sum(valid)),
            'n_total_points': int(magnitude.size),
            'coverage_percent': float(np.sum(valid) / magnitude.size * 100),
            'mean_displacement_px': float(np.mean(valid_mag)),
            'max_displacement_px': float(np.max(valid_mag)),
            'min_displacement_px': float(np.min(valid_mag)),
            'std_displacement_px': float(np.std(valid_mag)),
            'median_displacement_px': float(np.median(valid_mag)),
            'p25_displacement_px': float(np.percentile(valid_mag, 25)),
            'p75_displacement_px': float(np.percentile(valid_mag, 75)),
            'p95_displacement_px': float(np.percentile(valid_mag, 95)),
            'p99_displacement_px': float(np.percentile(valid_mag, 99)),
            'mean_quality': float(np.nanmean(quality[valid])),
            'min_quality': float(np.nanmin(quality[valid])),
        }

        if gsd and gsd > 0:
            for key in ['mean', 'max', 'min', 'std', 'median',
                        'p25', 'p75', 'p95', 'p99']:
                px_key = f'{key}_displacement_px'
                if px_key in stats:
                    stats[f'{key}_displacement_m'] = stats[px_key] * gsd
                    stats[f'{key}_displacement_mm'] = stats[px_key] * gsd * 1000

        valid_u = u[valid]
        valid_v = v[valid]
        directions = np.degrees(np.arctan2(-valid_v, valid_u)) % 360
        stats['mean_direction_deg'] = float(np.mean(directions))
        stats['direction_histogram'] = np.histogram(
            directions, bins=36, range=(0, 360))[0].tolist()

        return stats

    @staticmethod
    def estimate_noise_level(magnitude, quality=None, quality_threshold=0.8):
        """Estimate the DIC noise floor from the displacement field.

        Uses only high-quality points (quality > threshold) to estimate
        the baseline noise.  The noise level is the MAD (Median Absolute
        Deviation) of the magnitude, scaled to sigma equivalent.

        If quality is not available, uses all valid points and estimates
        noise from the lower 50th percentile (assumes most of the image
        is static).

        Parameters
        ----------
        magnitude : np.ndarray
        quality : np.ndarray or None, correlation quality map
        quality_threshold : float, minimum quality to consider reliable

        Returns
        -------
        float, estimated noise level in pixels
        """
        valid = ~np.isnan(magnitude)

        if quality is not None:
            # Use high-quality points only
            q_valid = ~np.isnan(quality)
            high_q = valid & q_valid & (quality >= quality_threshold)
            if np.sum(high_q) >= 10:
                vals = magnitude[high_q]
            else:
                vals = magnitude[valid]
        else:
            vals = magnitude[valid]

        if len(vals) < 5:
            return 1.0  # fallback

        # Use lower half of magnitude distribution to estimate noise
        # (upper half may contain real signal)
        median_val = np.median(vals)
        lower_half = vals[vals <= median_val]

        if len(lower_half) < 3:
            lower_half = vals

        # MAD scaled to sigma equivalent (factor 1.4826)
        mad = np.median(np.abs(lower_half - np.median(lower_half)))
        noise_sigma = mad * 1.4826

        return max(noise_sigma, 0.01)  # floor

    @staticmethod
    def auto_threshold(magnitude, quality=None, snr_factor=5.0):
        """Compute an automatic threshold for active zone detection.

        threshold = noise_level × snr_factor

        This ensures only displacements significantly above the noise
        floor are considered active.  Default snr_factor=5 means 5×
        the estimated noise.

        Also returns the median + 3*MAD of the magnitude as an
        alternative robust threshold.

        Parameters
        ----------
        magnitude : np.ndarray
        quality : np.ndarray or None
        snr_factor : float, signal-to-noise multiplier

        Returns
        -------
        dict with:
            'noise_level': float, estimated noise in px
            'snr_threshold': float, noise × snr_factor
            'mad_threshold': float, median + 3 × MAD
            'recommended': float, max of the two methods
        """
        noise = DisplacementStatistics.estimate_noise_level(
            magnitude, quality)
        snr_thresh = noise * snr_factor

        valid = magnitude[~np.isnan(magnitude)]
        if len(valid) >= 5:
            med = np.median(valid)
            mad = np.median(np.abs(valid - med))
            mad_thresh = med + 3.0 * mad * 1.4826
        else:
            mad_thresh = snr_thresh

        recommended = max(snr_thresh, mad_thresh)

        return {
            'noise_level': float(noise),
            'snr_threshold': float(snr_thresh),
            'mad_threshold': float(mad_thresh),
            'recommended': float(recommended),
        }

    @staticmethod
    def detect_active_zones(magnitude, threshold, min_area_px=100,
                            quality=None, min_quality=0.0):
        """Detect connected regions where displacement exceeds threshold.

        Applies morphological opening before connected-component analysis
        to remove isolated single-pixel noise.

        Optionally filters by correlation quality: points with quality
        below *min_quality* are excluded even if above the displacement
        threshold.

        Parameters
        ----------
        magnitude : np.ndarray (NY, NX) displacement magnitude
        threshold : float, minimum displacement to consider "active"
        min_area_px : int, minimum connected region area in grid points
        quality : np.ndarray or None, correlation quality map
        min_quality : float, minimum quality to include a point (0 = no filter)

        Returns
        -------
        list of dicts, each with:
            'id', 'centroid_row', 'centroid_col',
            'area_points', 'max_displacement', 'mean_displacement',
            'mean_quality', 'bbox' (row0, col0, row1, col1)
        """
        # Binary mask of active pixels
        active = np.zeros_like(magnitude, dtype=np.uint8)
        valid = ~np.isnan(magnitude)
        above_thresh = valid & (magnitude > threshold)

        # Quality filter
        if quality is not None and min_quality > 0:
            q_valid = ~np.isnan(quality)
            above_thresh = above_thresh & q_valid & (quality >= min_quality)

        active[above_thresh] = 255

        # Morphological opening to remove isolated pixels / thin noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        active = cv2.morphologyEx(active, cv2.MORPH_OPEN, kernel)

        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            active, connectivity=8)

        zones = []
        for i in range(1, n_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area_px:
                continue

            region_mask = labels == i
            region_mag = magnitude[region_mask]
            valid_region = ~np.isnan(region_mag)

            if not np.any(valid_region):
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            zone_info = {
                'id': i,
                'centroid_row': float(centroids[i][1]),
                'centroid_col': float(centroids[i][0]),
                'area_points': int(area),
                'max_displacement': float(np.nanmax(region_mag)),
                'mean_displacement': float(np.nanmean(region_mag)),
                'bbox': (int(y), int(x), int(y + h), int(x + w)),
            }

            # Include mean quality if available
            if quality is not None:
                region_q = quality[region_mask]
                zone_info['mean_quality'] = float(np.nanmean(region_q))
            else:
                zone_info['mean_quality'] = float('nan')

            zones.append(zone_info)

        zones.sort(key=lambda z: z['max_displacement'], reverse=True)
        logger.info(f"Detected {len(zones)} active zones above threshold {threshold}")
        return zones

    @staticmethod
    def generate_rose_diagram_data(u, v, n_bins=36):
        """Generate data for a displacement direction rose diagram.

        Returns
        -------
        dict with 'angles' (bin centers in degrees) and 'counts'
        """
        valid = ~(np.isnan(u) | np.isnan(v))
        if not np.any(valid):
            return {'angles': [], 'counts': []}

        directions = np.degrees(np.arctan2(-v[valid], u[valid])) % 360
        counts, bin_edges = np.histogram(directions, bins=n_bins, range=(0, 360))
        angles = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return {
            'angles': angles.tolist(),
            'counts': counts.tolist(),
        }
