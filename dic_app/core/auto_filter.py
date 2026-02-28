"""Automatic filter optimization for DIC image preprocessing.

Evaluates image quality metrics relevant to DIC correlation and tests
candidate filter pipelines to find the combination that maximizes
texture quality, local contrast, and signal-to-noise ratio.

Metrics used:
    - Mean Gradient Magnitude: measures edge/texture richness
    - Local Contrast (std): how much intensity variation in local patches
    - Shannon Entropy: information content of the image
    - Laplacian Variance: sharpness/focus measure
    - SNR estimate: signal-to-noise ratio (mean / noise_std)
    - NCC self-score: average NCC of random patches (DIC-specific)

The optimizer tests predefined "strategy" pipelines suited for
common scenarios (drone imagery, low contrast, noisy, etc.) and
returns the best one ranked by a composite quality score.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from dic_app.core.preprocessing import FilterPipeline, FILTER_REGISTRY

logger = logging.getLogger(__name__)


# ======================================================================
# Image quality metrics for DIC
# ======================================================================

@dataclass
class ImageQualityMetrics:
    """Quality metrics relevant to DIC correlation."""
    mean_gradient: float = 0.0       # Edge/texture richness
    local_contrast: float = 0.0      # Average local std deviation
    entropy: float = 0.0             # Shannon entropy (bits)
    laplacian_var: float = 0.0       # Sharpness (Laplacian variance)
    snr_estimate: float = 0.0        # Signal-to-noise ratio
    ncc_self_score: float = 0.0      # DIC-specific: avg NCC of shifted patches
    composite_score: float = 0.0     # Weighted combination

    def to_dict(self) -> dict:
        return {
            'mean_gradient': round(self.mean_gradient, 4),
            'local_contrast': round(self.local_contrast, 4),
            'entropy': round(self.entropy, 4),
            'laplacian_var': round(self.laplacian_var, 4),
            'snr_estimate': round(self.snr_estimate, 4),
            'ncc_self_score': round(self.ncc_self_score, 4),
            'composite_score': round(self.composite_score, 4),
        }


def compute_quality_metrics(image: np.ndarray,
                            subset_size: int = 31) -> ImageQualityMetrics:
    """Compute DIC-relevant image quality metrics.

    Parameters
    ----------
    image : (H, W) uint8 grayscale
    subset_size : int, DIC subset size for NCC self-test

    Returns
    -------
    ImageQualityMetrics
    """
    img = image.astype(np.float64)
    h, w = img.shape

    # 1. Mean Gradient Magnitude (Sobel)
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    mean_gradient = float(np.mean(grad_mag))

    # 2. Local contrast: average local standard deviation
    win = min(21, min(h, w) // 4)
    if win < 3:
        win = 3
    local_mean = uniform_filter(img, size=win)
    local_sq_mean = uniform_filter(img ** 2, size=win)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)
    local_contrast = float(np.mean(local_std))

    # 3. Shannon entropy
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    # 4. Laplacian variance (sharpness)
    lap = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_var = float(np.var(lap))

    # 5. SNR estimate: mean intensity / noise std
    # Estimate noise from high-frequency component
    blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
    noise = img - blurred
    noise_std = float(np.std(noise))
    mean_val = float(np.mean(img))
    snr_estimate = mean_val / max(noise_std, 1e-6)

    # 6. NCC self-score: sample random patches and check NCC
    # with a 1-pixel shifted version (good texture = high NCC)
    ncc_score = _compute_ncc_self_score(image, subset_size)

    # Compute composite score
    # Normalize each metric relative to typical ranges and weight
    # Weights based on importance for DIC:
    #   texture/gradient is most important, then contrast, then sharpness
    s_grad = min(mean_gradient / 30.0, 1.0)          # typical range 5-50
    s_contrast = min(local_contrast / 40.0, 1.0)      # typical range 10-60
    s_entropy = min(entropy / 7.5, 1.0)                # typical range 5-8
    s_sharp = min(laplacian_var / 500.0, 1.0)          # typical range 50-1000
    s_snr = min(snr_estimate / 20.0, 1.0)              # typical range 5-30
    s_ncc = ncc_score                                   # already 0-1

    composite = (
        0.25 * s_grad +
        0.20 * s_contrast +
        0.10 * s_entropy +
        0.15 * s_sharp +
        0.10 * s_snr +
        0.20 * s_ncc
    )

    return ImageQualityMetrics(
        mean_gradient=mean_gradient,
        local_contrast=local_contrast,
        entropy=entropy,
        laplacian_var=laplacian_var,
        snr_estimate=snr_estimate,
        ncc_self_score=ncc_score,
        composite_score=composite,
    )


def _compute_ncc_self_score(image: np.ndarray, subset_size: int = 31,
                             n_samples: int = 50) -> float:
    """Compute average NCC of random patches vs 1px-shifted version.

    High NCC self-score means the image has good texture for
    template matching (NCC) — the core of DIC.

    Returns float in [0, 1].
    """
    h, w = image.shape
    half = subset_size // 2
    if h < subset_size + 4 or w < subset_size + 4:
        return 0.5

    img = image.astype(np.float64)
    scores = []

    np.random.seed(0)
    for _ in range(n_samples):
        y = np.random.randint(half + 2, h - half - 2)
        x = np.random.randint(half + 2, w - half - 2)

        patch = img[y - half:y + half + 1, x - half:x + half + 1]
        # Shift 1px right
        patch_shifted = img[y - half:y + half + 1, x - half + 1:x + half + 2]

        p_mean = patch.mean()
        s_mean = patch_shifted.mean()
        p_centered = patch - p_mean
        s_centered = patch_shifted - s_mean

        denom = np.sqrt(np.sum(p_centered ** 2) * np.sum(s_centered ** 2))
        if denom < 1e-10:
            continue

        ncc = np.sum(p_centered * s_centered) / denom
        scores.append(max(0.0, ncc))

    return float(np.mean(scores)) if scores else 0.5


# ======================================================================
# Candidate pipeline strategies
# ======================================================================

def _build_candidate_pipelines() -> List[Tuple[str, FilterPipeline]]:
    """Build a set of candidate pipelines for different scenarios."""
    candidates = []

    # 0. No filters (baseline)
    p0 = FilterPipeline()
    candidates.append(("Nessun filtro (originale)", p0))

    # 1. CLAHE only (general purpose)
    p1 = FilterPipeline()
    p1.add_step('CLAHE', clip_limit=2.0, tile_grid_size=(8, 8))
    candidates.append(("CLAHE (contrasto adattivo)", p1))

    # 2. CLAHE aggressive
    p2 = FilterPipeline()
    p2.add_step('CLAHE', clip_limit=4.0, tile_grid_size=(8, 8))
    candidates.append(("CLAHE forte", p2))

    # 3. Wallis (illumination normalization for outdoor)
    p3 = FilterPipeline()
    p3.add_step('Wallis Filter', target_mean=127, target_std=50,
                brightness_factor=0.9, contrast_factor=0.9, window_size=21)
    candidates.append(("Wallis (normalizzazione illuminazione)", p3))

    # 4. Wallis + CLAHE
    p4 = FilterPipeline()
    p4.add_step('Wallis Filter', target_mean=127, target_std=50,
                brightness_factor=0.9, contrast_factor=0.9, window_size=21)
    p4.add_step('CLAHE', clip_limit=2.0, tile_grid_size=(8, 8))
    candidates.append(("Wallis + CLAHE", p4))

    # 5. Denoise + CLAHE (noisy images)
    p5 = FilterPipeline()
    p5.add_step('Bilateral Filter', d=9, sigma_color=75, sigma_space=75)
    p5.add_step('CLAHE', clip_limit=2.0, tile_grid_size=(8, 8))
    candidates.append(("Denoise Bilateral + CLAHE", p5))

    # 6. NLM Denoise + CLAHE (very noisy)
    p6 = FilterPipeline()
    p6.add_step('Non-Local Means', h=10, template_window=7, search_window=21)
    p6.add_step('CLAHE', clip_limit=2.0, tile_grid_size=(8, 8))
    candidates.append(("Denoise NLM + CLAHE", p6))

    # 7. Normalize + CLAHE (dark/bright images)
    p7 = FilterPipeline()
    p7.add_step('Normalize Brightness', target_mean=128)
    p7.add_step('CLAHE', clip_limit=3.0, tile_grid_size=(8, 8))
    candidates.append(("Normalizza luminosità + CLAHE", p7))

    # 8. CLAHE + Unsharp (enhance texture)
    p8 = FilterPipeline()
    p8.add_step('CLAHE', clip_limit=2.0, tile_grid_size=(8, 8))
    p8.add_step('Unsharp Mask', kernel_size=5, sigma=1.0,
                amount=1.0, threshold=0)
    candidates.append(("CLAHE + Sharpening leggero", p8))

    # 9. Wallis + Unsharp (outdoor + sharpen)
    p9 = FilterPipeline()
    p9.add_step('Wallis Filter', target_mean=127, target_std=50,
                brightness_factor=0.9, contrast_factor=0.9, window_size=21)
    p9.add_step('Unsharp Mask', kernel_size=5, sigma=1.0,
                amount=1.0, threshold=0)
    candidates.append(("Wallis + Sharpening", p9))

    # 10. Full: Normalize + Wallis + CLAHE (comprehensive)
    p10 = FilterPipeline()
    p10.add_step('Normalize Brightness', target_mean=128)
    p10.add_step('Wallis Filter', target_mean=127, target_std=50,
                 brightness_factor=0.85, contrast_factor=0.85, window_size=31)
    p10.add_step('CLAHE', clip_limit=2.0, tile_grid_size=(8, 8))
    candidates.append(("Normalizza + Wallis + CLAHE", p10))

    # 11. Denoise + Wallis + CLAHE (noisy outdoor)
    p11 = FilterPipeline()
    p11.add_step('Bilateral Filter', d=7, sigma_color=50, sigma_space=50)
    p11.add_step('Wallis Filter', target_mean=127, target_std=50,
                 brightness_factor=0.9, contrast_factor=0.9, window_size=21)
    p11.add_step('CLAHE', clip_limit=2.0, tile_grid_size=(8, 8))
    candidates.append(("Denoise + Wallis + CLAHE", p11))

    # 12. Gamma + CLAHE (very dark or very bright)
    p12 = FilterPipeline()
    p12.add_step('Gamma Correction', gamma=0.7)
    p12.add_step('CLAHE', clip_limit=3.0, tile_grid_size=(8, 8))
    candidates.append(("Gamma 0.7 + CLAHE", p12))

    # 13. Histogram Eq + Unsharp
    p13 = FilterPipeline()
    p13.add_step('Histogram Equalization')
    p13.add_step('Unsharp Mask', kernel_size=5, sigma=1.0,
                 amount=0.8, threshold=0)
    candidates.append(("Hist. Equalization + Sharpening", p13))

    return candidates


# ======================================================================
# Optimizer
# ======================================================================

@dataclass
class FilterOptimizationResult:
    """Result of automatic filter optimization."""
    best_pipeline: FilterPipeline
    best_name: str
    best_score: float
    best_metrics: ImageQualityMetrics
    baseline_metrics: ImageQualityMetrics
    all_results: List[dict] = field(default_factory=list)
    computation_time_s: float = 0.0
    improvement_percent: float = 0.0


class AutoFilterOptimizer:
    """Automatic filter pipeline optimizer for DIC.

    Tests multiple candidate filter pipelines on the input image and
    selects the one that produces the best composite quality score.

    Usage::

        optimizer = AutoFilterOptimizer()
        result = optimizer.optimize(image_gray)
        best_pipeline = result.best_pipeline
        # Apply: processed = best_pipeline.apply(image_gray)
    """

    def __init__(self, subset_size: int = 31):
        """
        Parameters
        ----------
        subset_size : int, DIC subset size for NCC quality metric
        """
        self.subset_size = subset_size
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """Set callback(percent, message)."""
        self._progress_callback = callback

    def _report(self, pct: int, msg: str):
        if self._progress_callback:
            self._progress_callback(pct, msg)

    def optimize(self, image: np.ndarray,
                 extra_pipelines: List[Tuple[str, FilterPipeline]] = None,
                 images: List[np.ndarray] = None,
                 ) -> FilterOptimizationResult:
        """Find the best filter pipeline for the given image(s).

        When *images* is provided (list of grayscale arrays), the optimizer
        evaluates every candidate pipeline on **all** images and picks the
        pipeline with the best *average* composite score across the set.
        This guarantees that the chosen filters work well for the whole
        image sequence, not just a single frame.

        Parameters
        ----------
        image : (H, W) uint8 grayscale – single reference image
            Used as fallback when *images* is None.
        extra_pipelines : optional additional pipelines to test
        images : list of (H, W) uint8 grayscale images
            When provided, all images are used for scoring.

        Returns
        -------
        FilterOptimizationResult
        """
        t0 = time.time()

        # Build the list of test images
        if images and len(images) > 0:
            raw_images = list(images)
        else:
            raw_images = [image]

        n_images = len(raw_images)
        logger.info(f"Auto-filter: evaluating on {n_images} image(s)")

        # Downscale large images for speed
        test_images = []
        for img in raw_images:
            h, w = img.shape
            if max(h, w) > 1500:
                scale = 1500.0 / max(h, w)
                resized = cv2.resize(img, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
                test_images.append(resized)
            else:
                test_images.append(img)

        if n_images > 1:
            self._report(0, f"Analisi di {n_images} immagini originali...")
        else:
            self._report(0, "Analisi immagine originale...")

        # Build candidates
        candidates = _build_candidate_pipelines()
        if extra_pipelines:
            candidates.extend(extra_pipelines)

        n_total = len(candidates)
        all_results = []

        # Compute baseline (no filter) metrics – average across all images
        baseline_scores = []
        baseline_metrics_list = []
        for test_img in test_images:
            m = compute_quality_metrics(test_img, self.subset_size)
            baseline_scores.append(m.composite_score)
            baseline_metrics_list.append(m)
        avg_baseline_score = float(np.mean(baseline_scores))

        # Use first image's metrics as the representative baseline metrics
        # (for display purposes; the average score is used for comparison)
        baseline_metrics = baseline_metrics_list[0]

        best_score = -1

        for i, (name, pipeline) in enumerate(candidates):
            pct = int((i + 1) / n_total * 95)
            if n_images > 1:
                self._report(pct,
                             f"Test pipeline {i + 1}/{n_total} su "
                             f"{n_images} immagini: {name}")
            else:
                self._report(pct, f"Test pipeline {i + 1}/{n_total}: {name}")

            try:
                img_scores = []
                img_metrics_list = []
                for test_img in test_images:
                    if len(pipeline) == 0:
                        processed = test_img
                    else:
                        processed = pipeline.apply(test_img)

                    metrics = compute_quality_metrics(
                        processed, self.subset_size)
                    img_scores.append(metrics.composite_score)
                    img_metrics_list.append(metrics)

                avg_score = float(np.mean(img_scores))

                # Use first image's metrics as representative
                representative_metrics = img_metrics_list[0]

                all_results.append({
                    'name': name,
                    'pipeline': pipeline,
                    'metrics': representative_metrics,
                    'score': avg_score,
                })

                if avg_score > best_score:
                    best_score = avg_score

            except Exception as e:
                logger.warning(f"Pipeline '{name}' failed: {e}")
                all_results.append({
                    'name': name,
                    'pipeline': pipeline,
                    'metrics': None,
                    'score': -1,
                    'error': str(e),
                })

        # Sort by score descending
        all_results.sort(key=lambda x: x['score'], reverse=True)

        best = all_results[0]
        improvement = 0.0
        if avg_baseline_score > 0:
            improvement = (
                (best['score'] - avg_baseline_score)
                / avg_baseline_score * 100
            )

        elapsed = time.time() - t0
        self._report(100,
                     f"Ottimizzazione completata in {elapsed:.1f}s "
                     f"({n_images} immagini)")

        logger.info(
            f"Auto-filter optimization: best='{best['name']}' "
            f"score={best['score']:.4f} "
            f"improvement={improvement:+.1f}% "
            f"n_images={n_images} "
            f"time={elapsed:.1f}s")

        return FilterOptimizationResult(
            best_pipeline=best['pipeline'],
            best_name=best['name'],
            best_score=best['score'],
            best_metrics=best['metrics'],
            baseline_metrics=baseline_metrics,
            all_results=[{
                'name': r['name'],
                'score': r['score'],
                'metrics': r['metrics'].to_dict() if r['metrics'] else None,
            } for r in all_results],
            computation_time_s=elapsed,
            improvement_percent=improvement,
        )
