"""Image enhancement filters for DIC preprocessing.

Provides 12+ filters optimized for DIC analysis on drone/camera imagery,
plus a FilterPipeline class for chaining filters in sequence.
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter


class ImageFilters:
    """Static methods for each image enhancement filter.

    All filters take a grayscale uint8 np.ndarray and return uint8.
    """

    @staticmethod
    def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Contrast Limited Adaptive Histogram Equalization.

        Enhances local contrast while preventing noise amplification.
        Best general-purpose enhancement for DIC on natural terrain.

        Parameters
        ----------
        image : np.ndarray (H, W) uint8
        clip_limit : float, contrast limiting threshold (default 2.0)
        tile_grid_size : tuple, number of tiles (rows, cols)
        """
        clahe_obj = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        return clahe_obj.apply(image)

    @staticmethod
    def histogram_equalization(image):
        """Global histogram equalization.

        Spreads intensity distribution across the full range.
        Good for uniformly dark or bright images.
        """
        return cv2.equalizeHist(image)

    @staticmethod
    def gaussian_blur(image, kernel_size=5, sigma=0):
        """Gaussian smoothing for noise reduction.

        Parameters
        ----------
        kernel_size : int, must be odd (default 5)
        sigma : float, Gaussian sigma (0 = auto from kernel_size)
        """
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(image, (k, k), sigma)

    @staticmethod
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        """Edge-preserving denoising via bilateral filtering.

        Smooths flat regions while preserving edges. Useful for reducing
        sensor noise without blurring texture boundaries.

        Parameters
        ----------
        d : int, diameter of pixel neighborhood
        sigma_color : float, filter sigma in the color space
        sigma_space : float, filter sigma in the coordinate space
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def non_local_means(image, h=10, template_window=7, search_window=21):
        """Non-Local Means denoising.

        Averages similar patches across the image for noise reduction.
        Slower but better quality than Gaussian/bilateral.

        Parameters
        ----------
        h : float, filter strength (higher = more denoising)
        template_window : int, size of template patch (odd)
        search_window : int, size of area to search for similar patches (odd)
        """
        return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)

    @staticmethod
    def unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
        """Sharpening via unsharp masking.

        sharpened = image + amount * (image - blurred)
        Only applies sharpening where the difference exceeds threshold,
        to avoid amplifying noise.

        Parameters
        ----------
        kernel_size : int, Gaussian kernel size for blur (odd)
        sigma : float, Gaussian sigma
        amount : float, sharpening strength (1.0 = mild, 2.0 = strong)
        threshold : int, minimum difference to apply sharpening (0-255)
        """
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred = cv2.GaussianBlur(image, (k, k), sigma)
        diff = image.astype(np.float32) - blurred.astype(np.float32)

        if threshold > 0:
            mask = np.abs(diff) > threshold
            diff = diff * mask

        sharpened = image.astype(np.float32) + amount * diff
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def wallis_filter(image, target_mean=127, target_std=50,
                      brightness_factor=0.9, contrast_factor=0.9,
                      window_size=21):
        """Wallis local contrast normalization.

        CRITICAL for DIC on outdoor/drone imagery: normalizes local contrast
        to compensate for illumination changes between acquisition times.

        Formula:
            output = (img - local_mean) * gain + target_brightness
        where:
            gain = contrast_factor * target_std /
                   (contrast_factor * local_std + (1 - contrast_factor) * target_std)
            target_brightness = brightness_factor * target_mean +
                                (1 - brightness_factor) * local_mean

        Parameters
        ----------
        image : np.ndarray (H, W) uint8
        target_mean : float, desired local mean (default 127)
        target_std : float, desired local standard deviation (default 50)
        brightness_factor : float, 0-1, controls brightness normalization
        contrast_factor : float, 0-1, controls contrast normalization
        window_size : int, size of local window (odd number)
        """
        img = image.astype(np.float64)

        # Compute local mean using uniform (box) filter
        local_mean = uniform_filter(img, size=window_size)

        # Compute local standard deviation
        local_mean_sq = uniform_filter(img ** 2, size=window_size)
        local_var = local_mean_sq - local_mean ** 2
        local_var = np.maximum(local_var, 0)  # numerical safety
        local_std = np.sqrt(local_var)

        # Wallis transformation
        denominator = contrast_factor * local_std + (1 - contrast_factor) * target_std
        denominator = np.maximum(denominator, 1e-6)  # avoid division by zero

        gain = contrast_factor * target_std / denominator
        target_brightness = (brightness_factor * target_mean +
                             (1 - brightness_factor) * local_mean)

        output = (img - local_mean) * gain + target_brightness
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def median_filter(image, kernel_size=5):
        """Median filter for impulse (salt-and-pepper) noise removal.

        Parameters
        ----------
        kernel_size : int, must be odd
        """
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.medianBlur(image, k)

    @staticmethod
    def morphological_opening(image, kernel_size=5):
        """Morphological opening (erosion then dilation).

        Removes small bright noise while preserving larger bright regions.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def morphological_closing(image, kernel_size=5):
        """Morphological closing (dilation then erosion).

        Fills small dark gaps while preserving larger dark regions.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def gamma_correction(image, gamma=1.0):
        """Gamma correction for brightness adjustment.

        gamma < 1.0: brightens image (enhances dark areas)
        gamma > 1.0: darkens image (enhances bright areas)

        Parameters
        ----------
        gamma : float, gamma value (default 1.0 = no change)
        """
        if gamma <= 0:
            gamma = 0.01
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        return cv2.LUT(image, table)

    @staticmethod
    def adaptive_threshold_enhancement(image, block_size=11, c=2):
        """Enhance texture visibility using adaptive thresholding.

        Not a binarization step: blends the adaptive threshold result
        with the original to enhance local texture contrast.

        Parameters
        ----------
        block_size : int, neighborhood size (odd, >= 3)
        c : float, constant subtracted from mean
        """
        bs = block_size if block_size % 2 == 1 else block_size + 1
        bs = max(bs, 3)
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, bs, c)
        # Blend with original: enhance texture while keeping grayscale info
        blended = cv2.addWeighted(image, 0.7, thresh, 0.3, 0)
        return blended

    @staticmethod
    def laplacian_sharpening(image, kernel_size=3, strength=0.5):
        """Sharpen using Laplacian edge enhancement.

        Parameters
        ----------
        kernel_size : int, Laplacian kernel size (1, 3, 5, or 7)
        strength : float, blending weight for edge enhancement
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
        sharpened = image.astype(np.float64) - strength * laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def normalize_brightness(image, target_mean=128):
        """Shift overall brightness to target mean value.

        Simple global brightness normalization useful as a first step
        before other local filters.
        """
        current_mean = np.mean(image).astype(np.float64)
        shift = target_mean - current_mean
        adjusted = image.astype(np.float64) + shift
        return np.clip(adjusted, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Filter registry: maps filter names to (function, default_params) pairs
# ---------------------------------------------------------------------------

FILTER_REGISTRY = {
    'CLAHE': {
        'func': ImageFilters.clahe,
        'params': {'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
        'param_types': {'clip_limit': ('float', 0.5, 40.0),
                        'tile_grid_size': ('tuple_int', 2, 64)},
        'description': 'Adaptive histogram equalization (local contrast enhancement)',
    },
    'Histogram Equalization': {
        'func': ImageFilters.histogram_equalization,
        'params': {},
        'param_types': {},
        'description': 'Global histogram equalization',
    },
    'Gaussian Blur': {
        'func': ImageFilters.gaussian_blur,
        'params': {'kernel_size': 5, 'sigma': 0},
        'param_types': {'kernel_size': ('odd_int', 3, 31),
                        'sigma': ('float', 0, 10.0)},
        'description': 'Gaussian smoothing (noise reduction)',
    },
    'Bilateral Filter': {
        'func': ImageFilters.bilateral_filter,
        'params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
        'param_types': {'d': ('int', 1, 25),
                        'sigma_color': ('float', 10, 200),
                        'sigma_space': ('float', 10, 200)},
        'description': 'Edge-preserving denoising',
    },
    'Non-Local Means': {
        'func': ImageFilters.non_local_means,
        'params': {'h': 10, 'template_window': 7, 'search_window': 21},
        'param_types': {'h': ('float', 1, 30),
                        'template_window': ('odd_int', 3, 21),
                        'search_window': ('odd_int', 7, 51)},
        'description': 'Advanced denoising (slow but high quality)',
    },
    'Unsharp Mask': {
        'func': ImageFilters.unsharp_mask,
        'params': {'kernel_size': 5, 'sigma': 1.0, 'amount': 1.5, 'threshold': 0},
        'param_types': {'kernel_size': ('odd_int', 3, 31),
                        'sigma': ('float', 0.1, 10.0),
                        'amount': ('float', 0.5, 5.0),
                        'threshold': ('int', 0, 50)},
        'description': 'Sharpening via unsharp masking',
    },
    'Wallis Filter': {
        'func': ImageFilters.wallis_filter,
        'params': {'target_mean': 127, 'target_std': 50,
                   'brightness_factor': 0.9, 'contrast_factor': 0.9,
                   'window_size': 21},
        'param_types': {'target_mean': ('int', 0, 255),
                        'target_std': ('int', 10, 127),
                        'brightness_factor': ('float', 0.0, 1.0),
                        'contrast_factor': ('float', 0.0, 1.0),
                        'window_size': ('odd_int', 5, 101)},
        'description': 'Local contrast normalization (critical for drone imagery)',
    },
    'Median Filter': {
        'func': ImageFilters.median_filter,
        'params': {'kernel_size': 5},
        'param_types': {'kernel_size': ('odd_int', 3, 31)},
        'description': 'Salt-and-pepper noise removal',
    },
    'Morphological Opening': {
        'func': ImageFilters.morphological_opening,
        'params': {'kernel_size': 5},
        'param_types': {'kernel_size': ('odd_int', 3, 31)},
        'description': 'Remove small bright noise',
    },
    'Morphological Closing': {
        'func': ImageFilters.morphological_closing,
        'params': {'kernel_size': 5},
        'param_types': {'kernel_size': ('odd_int', 3, 31)},
        'description': 'Fill small dark gaps',
    },
    'Gamma Correction': {
        'func': ImageFilters.gamma_correction,
        'params': {'gamma': 1.0},
        'param_types': {'gamma': ('float', 0.1, 5.0)},
        'description': 'Brightness adjustment (< 1 brightens, > 1 darkens)',
    },
    'Adaptive Threshold Enhancement': {
        'func': ImageFilters.adaptive_threshold_enhancement,
        'params': {'block_size': 11, 'c': 2},
        'param_types': {'block_size': ('odd_int', 3, 51),
                        'c': ('float', -10, 20)},
        'description': 'Enhance local texture visibility',
    },
    'Laplacian Sharpening': {
        'func': ImageFilters.laplacian_sharpening,
        'params': {'kernel_size': 3, 'strength': 0.5},
        'param_types': {'kernel_size': ('odd_int', 1, 7),
                        'strength': ('float', 0.1, 2.0)},
        'description': 'Edge-based sharpening',
    },
    'Normalize Brightness': {
        'func': ImageFilters.normalize_brightness,
        'params': {'target_mean': 128},
        'param_types': {'target_mean': ('int', 0, 255)},
        'description': 'Shift overall brightness to target mean',
    },
}


class FilterPipeline:
    """Chainable filter pipeline with named steps."""

    def __init__(self):
        self.steps = []  # list of (name, filter_name, kwargs)

    def add_step(self, filter_name, **kwargs):
        """Add a filter step to the pipeline.

        Parameters
        ----------
        filter_name : str, key in FILTER_REGISTRY
        **kwargs : override default parameters
        """
        if filter_name not in FILTER_REGISTRY:
            raise ValueError(f"Unknown filter: {filter_name}. "
                             f"Available: {list(FILTER_REGISTRY.keys())}")

        entry = FILTER_REGISTRY[filter_name]
        params = dict(entry['params'])
        params.update(kwargs)
        self.steps.append({
            'filter_name': filter_name,
            'params': params,
        })

    def remove_step(self, index):
        """Remove a step by index."""
        if 0 <= index < len(self.steps):
            self.steps.pop(index)

    def move_step(self, from_index, to_index):
        """Move a step from one position to another."""
        if 0 <= from_index < len(self.steps) and 0 <= to_index < len(self.steps):
            step = self.steps.pop(from_index)
            self.steps.insert(to_index, step)

    def update_step_params(self, index, **kwargs):
        """Update parameters of an existing step."""
        if 0 <= index < len(self.steps):
            self.steps[index]['params'].update(kwargs)

    def apply(self, image):
        """Apply all steps sequentially to the image.

        Parameters
        ----------
        image : np.ndarray (H, W) uint8 grayscale

        Returns
        -------
        np.ndarray (H, W) uint8 processed image
        """
        result = image.copy()
        for step in self.steps:
            entry = FILTER_REGISTRY[step['filter_name']]
            func = entry['func']
            params = step['params']
            result = func(result, **params)
        return result

    def clear(self):
        """Remove all steps."""
        self.steps = []

    def to_dict(self):
        """Serialize pipeline for project saving."""
        serialized = []
        for step in self.steps:
            params = {}
            for k, v in step['params'].items():
                if isinstance(v, tuple):
                    params[k] = list(v)
                else:
                    params[k] = v
            serialized.append({
                'filter_name': step['filter_name'],
                'params': params,
            })
        return serialized

    @classmethod
    def from_dict(cls, data):
        """Deserialize pipeline from project data."""
        pipeline = cls()
        for step_data in data:
            name = step_data['filter_name']
            params = step_data.get('params', {})
            # Convert lists back to tuples where needed
            if name in FILTER_REGISTRY:
                for k, v in params.items():
                    if isinstance(v, list):
                        params[k] = tuple(v)
            pipeline.steps.append({
                'filter_name': name,
                'params': params,
            })
        return pipeline

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        names = [s['filter_name'] for s in self.steps]
        return f"FilterPipeline({' -> '.join(names) if names else 'empty'})"
