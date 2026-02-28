"""Common utility functions for DIC Landslide Monitor."""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import uniform_filter, uniform_filter1d


def setup_logger(name, log_file=None, level=logging.INFO):
    """Configure a named logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Colormap and overlay utilities
# ---------------------------------------------------------------------------

def displacement_colormap(magnitude, vmin=None, vmax=None, cmap='jet'):
    """Convert scalar displacement field to RGBA uint8 image.

    Parameters
    ----------
    magnitude : np.ndarray (H, W), float
    vmin, vmax : optional explicit range
    cmap : str, matplotlib colormap name

    Returns
    -------
    np.ndarray (H, W, 4) uint8 RGBA image
    """
    if vmin is None:
        vmin = np.nanmin(magnitude)
    if vmax is None:
        vmax = np.nanmax(magnitude)
    # Handle all-NaN input
    if np.isnan(vmin):
        vmin = 0.0
    if np.isnan(vmax):
        vmax = 1.0
    if vmax == vmin:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    rgba = colormap(norm(magnitude))
    # Set NaN pixels to fully transparent
    mask_nan = np.isnan(magnitude)
    rgba[mask_nan] = [0, 0, 0, 0]
    return (rgba * 255).astype(np.uint8)


def overlay_heatmap(base_image, heatmap_rgba, alpha=0.5):
    """Alpha-blend RGBA heatmap onto a base image.

    Parameters
    ----------
    base_image : np.ndarray (H, W) or (H, W, 3) uint8
    heatmap_rgba : np.ndarray (H, W, 4) uint8
    alpha : float, overlay opacity

    Returns
    -------
    np.ndarray (H, W, 3) uint8 RGB blended image
    """
    if base_image.ndim == 2:
        base_rgb = np.stack([base_image] * 3, axis=-1)
    else:
        base_rgb = base_image.copy()

    h, w = base_rgb.shape[:2]
    hh, hw = heatmap_rgba.shape[:2]
    # Crop to common area
    ch, cw = min(h, hh), min(w, hw)
    base_crop = base_rgb[:ch, :cw].astype(np.float32)
    heat_crop = heatmap_rgba[:ch, :cw].astype(np.float32)

    # Use heatmap alpha channel combined with global alpha
    heat_alpha = (heat_crop[:, :, 3:4] / 255.0) * alpha
    blended = base_crop * (1 - heat_alpha) + heat_crop[:, :, :3] * heat_alpha
    result = base_rgb.copy()
    result[:ch, :cw] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def pixels_to_meters(displacement_px, gsd):
    """Convert pixel displacement to meters using Ground Sampling Distance.

    Parameters
    ----------
    displacement_px : float or np.ndarray
    gsd : float, meters per pixel

    Returns
    -------
    float or np.ndarray in meters
    """
    return displacement_px * gsd


def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert EXIF GPS DMS (degrees, minutes, seconds) to decimal degrees.

    Parameters
    ----------
    degrees, minutes, seconds : float
    direction : str, one of 'N', 'S', 'E', 'W'

    Returns
    -------
    float, decimal degrees (negative for S and W)
    """
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if direction in ('S', 'W'):
        decimal = -decimal
    return decimal


# ---------------------------------------------------------------------------
# Vector field drawing
# ---------------------------------------------------------------------------

def draw_vector_field(ax, u, v, grid_x=None, grid_y=None, step=10,
                      scale=1.0, color='white', width=0.002):
    """Draw quiver plot of displacement vectors on matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    u, v : np.ndarray (H, W) displacement components
    grid_x, grid_y : optional coordinate arrays
    step : int, subsample step for display
    scale : float, arrow scale factor
    color : str
    width : float, arrow width
    """
    h, w = u.shape
    if grid_x is None or grid_y is None:
        grid_y_arr, grid_x_arr = np.mgrid[0:h:step, 0:w:step]
    else:
        grid_x_arr = grid_x[::step, ::step] if grid_x.ndim == 2 else grid_x[::step]
        grid_y_arr = grid_y[::step, ::step] if grid_y.ndim == 2 else grid_y[::step]

    u_sub = u[::step, ::step]
    v_sub = v[::step, ::step]

    ax.quiver(grid_x_arr, grid_y_arr, u_sub * scale, v_sub * scale,
              color=color, width=width, headwidth=4, headlength=5)


# ---------------------------------------------------------------------------
# Array utilities
# ---------------------------------------------------------------------------

def nan_safe_gaussian_filter(array, sigma):
    """Apply Gaussian filter to an array with NaN values.

    Replaces NaN with 0 during filtering but preserves the NaN mask,
    then corrects the filtered result by the weight of valid pixels.

    Parameters
    ----------
    array : np.ndarray (2D)
    sigma : float, Gaussian sigma

    Returns
    -------
    np.ndarray, filtered array with original NaN positions preserved
    """
    from scipy.ndimage import gaussian_filter
    nan_mask = np.isnan(array)
    if not np.any(nan_mask):
        return gaussian_filter(array, sigma)

    arr_filled = np.where(nan_mask, 0.0, array)
    weight = np.where(nan_mask, 0.0, 1.0)

    filtered_arr = gaussian_filter(arr_filled, sigma)
    filtered_weight = gaussian_filter(weight, sigma)

    # Avoid division by zero
    filtered_weight = np.where(filtered_weight < 1e-10, 1e-10, filtered_weight)
    result = filtered_arr / filtered_weight
    result[nan_mask] = np.nan
    return result


def crop_to_valid(array):
    """Trim NaN borders from a 2D array.

    Returns
    -------
    (cropped_array, row_offset, col_offset)
    """
    valid = ~np.isnan(array)
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if not np.any(rows) or not np.any(cols):
        return array, 0, 0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return array[rmin:rmax + 1, cmin:cmax + 1], rmin, cmin


def compute_magnitude(u, v):
    """Compute displacement magnitude from u, v components."""
    return np.sqrt(u ** 2 + v ** 2)
