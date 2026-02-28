"""Automatic image registration (alignment) for DIC pre-processing.

Aligns a deformed image to the reference image using feature-based or
intensity-based methods, then crops the common valid area.

Methods:
    1. Feature Homography  - SIFT/ORB + RANSAC homography (perspective)
    2. Feature Affine      - SIFT/ORB + RANSAC affine (6 DOF)
    3. ECC Affine          - Enhanced Correlation Coefficient (sub-pixel)
    4. Phase Shift          - FFT phase correlation (pure translation)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Enums & Data Classes
# ======================================================================

class AlignmentMethod(Enum):
    FEATURE_HOMOGRAPHY = "feature_homography"
    FEATURE_AFFINE = "feature_affine"
    ECC_AFFINE = "ecc_affine"
    PHASE_SHIFT = "phase_shift"


@dataclass
class AlignmentParameters:
    """Configuration for image alignment."""
    method: AlignmentMethod = AlignmentMethod.FEATURE_HOMOGRAPHY
    detector_type: str = "SIFT"          # "SIFT" or "ORB"
    max_features: int = 10000
    match_ratio: float = 0.75            # Lowe's ratio test threshold
    ransac_threshold: float = 5.0        # RANSAC reprojection threshold (px)
    auto_crop: bool = True               # Crop to valid region after warp
    ecc_iterations: int = 200            # ECC max iterations
    ecc_epsilon: float = 1e-6            # ECC convergence threshold

    def to_dict(self) -> dict:
        return {
            'method': self.method.value,
            'detector_type': self.detector_type,
            'max_features': self.max_features,
            'match_ratio': self.match_ratio,
            'ransac_threshold': self.ransac_threshold,
            'auto_crop': self.auto_crop,
            'ecc_iterations': self.ecc_iterations,
            'ecc_epsilon': self.ecc_epsilon,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AlignmentParameters':
        return cls(
            method=AlignmentMethod(d.get('method', 'feature_homography')),
            detector_type=d.get('detector_type', 'SIFT'),
            max_features=d.get('max_features', 10000),
            match_ratio=d.get('match_ratio', 0.75),
            ransac_threshold=d.get('ransac_threshold', 5.0),
            auto_crop=d.get('auto_crop', True),
            ecc_iterations=d.get('ecc_iterations', 200),
            ecc_epsilon=d.get('ecc_epsilon', 1e-6),
        )


@dataclass
class AlignmentResult:
    """Result of image alignment."""
    transform_matrix: np.ndarray = field(repr=False)   # 3x3 or 2x3
    n_inliers: int = 0
    n_matches: int = 0
    reprojection_error: float = 0.0     # RMSE in pixels
    crop_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, y0, x1, y1
    aligned_shape: Tuple[int, int] = (0, 0)  # (h, w) after crop
    method_used: AlignmentMethod = AlignmentMethod.FEATURE_HOMOGRAPHY
    computation_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            'transform_matrix': self.transform_matrix.tolist(),
            'n_inliers': self.n_inliers,
            'n_matches': self.n_matches,
            'reprojection_error': self.reprojection_error,
            'crop_bbox': list(self.crop_bbox),
            'aligned_shape': list(self.aligned_shape),
            'method_used': self.method_used.value,
            'computation_time_s': self.computation_time_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AlignmentResult':
        return cls(
            transform_matrix=np.array(d['transform_matrix']),
            n_inliers=d.get('n_inliers', 0),
            n_matches=d.get('n_matches', 0),
            reprojection_error=d.get('reprojection_error', 0.0),
            crop_bbox=tuple(d.get('crop_bbox', (0, 0, 0, 0))),
            aligned_shape=tuple(d.get('aligned_shape', (0, 0))),
            method_used=AlignmentMethod(d.get('method_used', 'feature_homography')),
            computation_time_s=d.get('computation_time_s', 0.0),
        )


# ======================================================================
# Image Registration Engine
# ======================================================================

class ImageRegistration:
    """Automatic image alignment engine.

    Usage::

        reg = ImageRegistration(AlignmentParameters())
        aligned_def, result = reg.align(ref_gray, def_gray)
        # or apply crop to both:
        ref_crop, def_crop = reg.apply_to_pair(ref_gray, aligned_def, result)
    """

    def __init__(self, params: AlignmentParameters = None):
        self.params = params or AlignmentParameters()
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """Set callback(percent, message) for progress updates."""
        self._progress_callback = callback

    def _report(self, pct: int, msg: str):
        if self._progress_callback:
            self._progress_callback(pct, msg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self, ref_gray: np.ndarray, def_gray: np.ndarray,
              mask: np.ndarray = None) -> Tuple[np.ndarray, AlignmentResult]:
        """Align *def_gray* to *ref_gray*.

        Parameters
        ----------
        ref_gray : (H, W) uint8 – reference image
        def_gray : (H, W) uint8 – deformed image to be warped
        mask     : (H, W) uint8 optional – 255 = use, 0 = ignore

        Returns
        -------
        aligned_image : (H', W') uint8 – aligned (and optionally cropped)
        result        : AlignmentResult
        """
        t0 = time.time()

        dispatch = {
            AlignmentMethod.FEATURE_HOMOGRAPHY: self._feature_based_align,
            AlignmentMethod.FEATURE_AFFINE: self._feature_based_align,
            AlignmentMethod.ECC_AFFINE: self._ecc_align,
            AlignmentMethod.PHASE_SHIFT: self._phase_shift_align,
        }

        func = dispatch[self.params.method]
        aligned, result = func(ref_gray, def_gray, mask)
        result.computation_time_s = time.time() - t0
        result.method_used = self.params.method

        logger.info(
            f"Alignment completed in {result.computation_time_s:.2f}s "
            f"({result.method_used.value}): "
            f"{result.n_inliers}/{result.n_matches} inliers, "
            f"RMSE={result.reprojection_error:.3f}px, "
            f"crop={result.crop_bbox}"
        )
        return aligned, result

    def apply_to_pair(self, ref_gray: np.ndarray,
                      def_gray: np.ndarray,
                      result: AlignmentResult
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp deformed and crop both images to valid overlap region.

        Returns (ref_cropped, def_aligned_cropped).
        """
        h, w = ref_gray.shape[:2]
        M = result.transform_matrix

        # Warp deformed image
        if M.shape == (3, 3):
            warped = cv2.warpPerspective(def_gray, M, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
        else:  # 2x3 affine
            warped = cv2.warpAffine(def_gray, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)

        # Validity mask (small memory: uint8)
        ones = np.full((h, w), 255, dtype=np.uint8)
        if M.shape == (3, 3):
            vmask = cv2.warpPerspective(ones, M, (w, h),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
        else:
            vmask = cv2.warpAffine(ones, M, (w, h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        del ones

        bbox = self._compute_valid_crop(warped, ref_gray.shape, vmask)
        del vmask
        x0, y0, x1, y1 = bbox

        if x1 > x0 and y1 > y0:
            ref_crop = ref_gray[y0:y1, x0:x1].copy()
            def_crop = warped[y0:y1, x0:x1].copy()
        else:
            ref_crop = ref_gray.copy()
            def_crop = warped.copy()
        del warped

        return ref_crop, def_crop

    def apply_to_rgb(self, ref_rgb: np.ndarray,
                     def_rgb: np.ndarray,
                     result: AlignmentResult
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Same as apply_to_pair but for RGB (H,W,3) images."""
        h, w = ref_rgb.shape[:2]
        M = result.transform_matrix

        if M.shape == (3, 3):
            warped = cv2.warpPerspective(def_rgb, M, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
        else:
            warped = cv2.warpAffine(def_rgb, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)

        ones = np.full((h, w), 255, dtype=np.uint8)
        if M.shape == (3, 3):
            vmask = cv2.warpPerspective(ones, M, (w, h),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
        else:
            vmask = cv2.warpAffine(ones, M, (w, h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        del ones

        bbox = self._compute_valid_crop(warped, ref_rgb.shape, vmask)
        del vmask
        x0, y0, x1, y1 = bbox

        if x1 > x0 and y1 > y0:
            ref_crop = ref_rgb[y0:y1, x0:x1].copy()
            def_crop = warped[y0:y1, x0:x1].copy()
        else:
            ref_crop = ref_rgb.copy()
            def_crop = warped.copy()
        del warped

        return ref_crop, def_crop

    # ------------------------------------------------------------------
    # Feature-based alignment (Homography or Affine)
    # ------------------------------------------------------------------

    # Maximum dimension (pixels) for ECC / Phase-shift downscaling.
    # Feature-based methods run at full resolution (SIFT/ORB memory
    # footprint is small).  ECC and Phase are iterative / FFT-based
    # and need float64 copies — downscaling keeps RAM in check.
    _MAX_ECC_DIM = 3000

    # Maximum dimension for ECC fine-refinement (pyramid level 2).
    _MAX_ECC_REFINE_DIM = 4000

    def _feature_based_align(self, ref, deformed, mask) -> Tuple[np.ndarray, AlignmentResult]:
        """SIFT/ORB feature matching + RANSAC geometric estimation.

        Feature detection and matching run at full resolution to preserve
        sub-pixel accuracy.  SIFT/ORB memory footprint is modest (only
        keypoints + descriptors, no full float64 copies of the image).
        """
        self._report(0, "Rilevamento feature...")

        h_full, w_full = ref.shape[:2]

        # 1. Feature detection at full resolution
        detector, norm_type = self._create_detector()
        kp_ref, desc_ref = detector.detectAndCompute(ref, mask)
        kp_def, desc_def = detector.detectAndCompute(deformed, None)

        if desc_ref is None or desc_def is None:
            raise RuntimeError("Impossibile rilevare feature nelle immagini")

        n_kp = (len(kp_ref), len(kp_def))
        logger.info(f"Features detected: ref={n_kp[0]}, def={n_kp[1]}")

        if n_kp[0] < 4 or n_kp[1] < 4:
            raise RuntimeError(
                f"Troppe poche feature ({n_kp[0]}/{n_kp[1]}). "
                "Prova ad aumentare max_features o cambiare detector."
            )

        self._report(30, f"Matching {n_kp[0]} / {n_kp[1]} feature...")

        # 2. Feature matching with Lowe's ratio test
        bf = cv2.BFMatcher(norm_type)
        matches_knn = bf.knnMatch(desc_ref, desc_def, k=2)

        # Free descriptors after matching
        del desc_ref, desc_def

        good = []
        for pair in matches_knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.params.match_ratio * n.distance:
                    good.append(m)
        del matches_knn

        n_good = len(good)
        logger.info(f"Good matches after ratio test: {n_good}")

        min_matches = 4 if self.params.method == AlignmentMethod.FEATURE_HOMOGRAPHY else 3
        if n_good < min_matches:
            raise RuntimeError(
                f"Solo {n_good} corrispondenze trovate (minimo {min_matches}). "
                "Prova ad abbassare il match_ratio."
            )

        self._report(50, "Stima trasformazione RANSAC...")

        # 3. Extract matched point coordinates (full resolution)
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_def = np.float32([kp_def[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        del kp_ref, kp_def  # free keypoints

        # 4. Estimate transformation with RANSAC
        if self.params.method == AlignmentMethod.FEATURE_HOMOGRAPHY:
            M, inlier_mask = cv2.findHomography(
                pts_def, pts_ref,
                cv2.RANSAC, self.params.ransac_threshold
            )
            if M is None:
                raise RuntimeError("findHomography fallito: nessuna trasformazione trovata")
        else:  # FEATURE_AFFINE
            M, inlier_mask = cv2.estimateAffinePartial2D(
                pts_def, pts_ref,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.params.ransac_threshold
            )
            if M is None:
                raise RuntimeError("estimateAffine fallito: nessuna trasformazione trovata")

        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else n_good

        # 5. Compute reprojection error on inliers
        rmse = self._compute_reprojection_error(pts_def, pts_ref, M, inlier_mask)

        self._report(70, "Applicazione warp...")

        # 7. Warp deformed image at full resolution
        h, w = ref.shape[:2]

        if M.shape == (3, 3):
            warped = cv2.warpPerspective(deformed, M, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
        else:
            warped = cv2.warpAffine(deformed, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)

        # Compute validity mask (uint8 is tiny compared to full image)
        ones_mask = np.full((h, w), 255, dtype=np.uint8)
        if M.shape == (3, 3):
            validity_mask = cv2.warpPerspective(ones_mask, M, (w, h),
                                                flags=cv2.INTER_NEAREST,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0)
        else:
            validity_mask = cv2.warpAffine(ones_mask, M, (w, h),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
        del ones_mask

        # 8. Compute valid crop
        if self.params.auto_crop:
            self._report(85, "Calcolo area valida...")
            bbox = self._compute_valid_crop(warped, ref.shape, validity_mask)
        else:
            bbox = (0, 0, w, h)

        del validity_mask  # no longer needed

        crop_h = bbox[3] - bbox[1]
        crop_w = bbox[2] - bbox[0]

        # Ensure 3x3 matrix for consistent storage
        if M.shape == (2, 3):
            M_full = np.eye(3, dtype=np.float64)
            M_full[:2, :] = M
        else:
            M_full = M.astype(np.float64)

        result = AlignmentResult(
            transform_matrix=M_full,
            n_inliers=n_inliers,
            n_matches=n_good,
            reprojection_error=rmse,
            crop_bbox=bbox,
            aligned_shape=(crop_h, crop_w),
        )

        self._report(100, "Allineamento completato")

        # Return cropped or full warped
        if self.params.auto_crop and crop_w > 0 and crop_h > 0:
            cropped = warped[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
            del warped  # free full warped image
            return cropped, result
        return warped, result

    # ------------------------------------------------------------------
    # ECC (Enhanced Correlation Coefficient) alignment
    # ------------------------------------------------------------------

    def _ecc_align(self, ref, deformed, mask) -> Tuple[np.ndarray, AlignmentResult]:
        """Intensity-based ECC alignment (sub-pixel accurate).

        Uses a coarse-to-fine pyramid approach:
          1. Coarse ECC at _MAX_ECC_DIM  (fast convergence)
          2. Refine ECC at _MAX_ECC_REFINE_DIM  (sub-pixel accuracy)
          3. Warp at full resolution  (no quality loss)
        """
        self._report(0, "Allineamento ECC in corso...")

        h, w = ref.shape[:2]
        max_dim = max(h, w)

        # --- Level 1: Coarse ECC at small resolution ---
        if max_dim > self._MAX_ECC_DIM:
            scale1 = self._MAX_ECC_DIM / max_dim
            w1 = int(w * scale1)
            h1 = int(h * scale1)
            ref_l1 = cv2.resize(ref, (w1, h1), interpolation=cv2.INTER_AREA)
            def_l1 = cv2.resize(deformed, (w1, h1), interpolation=cv2.INTER_AREA)
            mask_l1 = (cv2.resize(mask, (w1, h1), interpolation=cv2.INTER_NEAREST)
                       if mask is not None else None)
            logger.info(f"ECC L1 (coarse): {w}x{h} -> {w1}x{h1}")
        else:
            scale1 = 1.0
            ref_l1, def_l1, mask_l1 = ref, deformed, mask

        # Initialize with phase correlation
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            shift, _ = cv2.phaseCorrelate(
                ref_l1.astype(np.float64), def_l1.astype(np.float64))
            warp_matrix[0, 2] = shift[0]
            warp_matrix[1, 2] = shift[1]
            logger.info(f"ECC initialized with phase shift: ({shift[0]:.2f}, {shift[1]:.2f})")
        except Exception:
            pass

        criteria_coarse = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           self.params.ecc_iterations, self.params.ecc_epsilon)

        self._report(20, "ECC coarse optimization...")

        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_l1, def_l1, warp_matrix,
                cv2.MOTION_AFFINE, criteria_coarse,
                inputMask=mask_l1
            )
        except cv2.error as e:
            raise RuntimeError(f"ECC non convergente (coarse): {e}")

        if scale1 < 1.0:
            del ref_l1, def_l1, mask_l1

        # --- Level 2: Fine ECC at higher resolution ---
        if max_dim > self._MAX_ECC_REFINE_DIM:
            scale2 = self._MAX_ECC_REFINE_DIM / max_dim
        else:
            scale2 = 1.0

        # Only do L2 refinement if it's at a meaningfully higher
        # resolution than L1 (at least 20% more pixels on the long side)
        if scale2 > scale1 * 1.2:
            w2 = int(w * scale2)
            h2 = int(h * scale2)
            ref_l2 = cv2.resize(ref, (w2, h2), interpolation=cv2.INTER_AREA)
            def_l2 = cv2.resize(deformed, (w2, h2), interpolation=cv2.INTER_AREA)
            mask_l2 = (cv2.resize(mask, (w2, h2), interpolation=cv2.INTER_NEAREST)
                       if mask is not None else None)

            # Rescale L1 warp matrix to L2 coordinates
            ratio = scale2 / scale1
            warp_l2 = warp_matrix.astype(np.float64).copy()
            warp_l2[0, 2] *= ratio
            warp_l2[1, 2] *= ratio
            warp_l2 = warp_l2.astype(np.float32)

            criteria_fine = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                             100, 1e-6)

            self._report(45, "ECC fine refinement...")
            logger.info(f"ECC L2 (fine): {w}x{h} -> {w2}x{h2}")

            try:
                cc, warp_l2 = cv2.findTransformECC(
                    ref_l2, def_l2, warp_l2,
                    cv2.MOTION_AFFINE, criteria_fine,
                    inputMask=mask_l2
                )
                # Use refined matrix; rescale to full-res
                warp_matrix = warp_l2.astype(np.float64)
                warp_matrix[0, 2] /= scale2
                warp_matrix[1, 2] /= scale2
                warp_matrix = warp_matrix.astype(np.float32)
                logger.info("ECC L2 refinement succeeded")
            except cv2.error:
                logger.warning("ECC L2 non convergente, uso risultato L1")
                # Fall back to L1 result rescaled to full-res
                warp_matrix = warp_matrix.astype(np.float64)
                warp_matrix[0, 2] /= scale1
                warp_matrix[1, 2] /= scale1
                warp_matrix = warp_matrix.astype(np.float32)
            finally:
                del ref_l2, def_l2
                if mask_l2 is not None:
                    del mask_l2
        else:
            # Only one level was used; rescale to full-res
            if scale1 < 1.0:
                warp_matrix = warp_matrix.astype(np.float64)
                warp_matrix[0, 2] /= scale1
                warp_matrix[1, 2] /= scale1
                warp_matrix = warp_matrix.astype(np.float32)

        self._report(70, "Applicazione warp...")

        warped = cv2.warpAffine(deformed, warp_matrix, (w, h),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)

        # Validity mask
        ones = np.full((h, w), 255, dtype=np.uint8)
        vmask = cv2.warpAffine(ones, warp_matrix, (w, h),
                                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        del ones

        if self.params.auto_crop:
            bbox = self._compute_valid_crop(warped, ref.shape, vmask)
        else:
            bbox = (0, 0, w, h)
        del vmask

        # ECC produces an inverse-warp matrix (ref → def mapping).
        # Convert to forward matrix (def → ref) for consistency with
        # feature-based methods and apply_to_pair().
        M_inv_3x3 = np.eye(3, dtype=np.float64)
        M_inv_3x3[:2, :] = warp_matrix.astype(np.float64)
        try:
            M_full = np.linalg.inv(M_inv_3x3)
        except np.linalg.LinAlgError:
            logger.warning("Cannot invert ECC matrix; using inverse-warp as-is")
            M_full = M_inv_3x3

        result = AlignmentResult(
            transform_matrix=M_full,
            n_inliers=0,
            n_matches=0,
            reprojection_error=float(1.0 - cc),
            crop_bbox=bbox,
            aligned_shape=(bbox[3] - bbox[1], bbox[2] - bbox[0]),
        )

        self._report(100, "Allineamento ECC completato")

        if self.params.auto_crop:
            cropped = warped[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
            del warped
            return cropped, result
        return warped, result

    # ------------------------------------------------------------------
    # Phase-only translation alignment
    # ------------------------------------------------------------------

    def _phase_shift_align(self, ref, deformed, mask) -> Tuple[np.ndarray, AlignmentResult]:
        """FFT phase correlation for pure translation.

        Phase correlation runs on downscaled images for memory efficiency;
        since it only estimates a global translation, the sub-pixel shift
        is simply rescaled to full resolution.
        """
        self._report(0, "Phase correlation...")

        h, w = ref.shape[:2]
        max_dim = max(h, w)

        # Downscale for FFT (large float64 arrays)
        if max_dim > self._MAX_ECC_DIM:
            scale = self._MAX_ECC_DIM / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            ref_small = cv2.resize(ref, (new_w, new_h), interpolation=cv2.INTER_AREA)
            def_small = cv2.resize(deformed, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            ref_small, def_small = ref, deformed

        ref_f = ref_small.astype(np.float64)
        def_f = def_small.astype(np.float64)
        if scale < 1.0:
            del ref_small, def_small

        hann_y = np.hanning(ref_f.shape[0])
        hann_x = np.hanning(ref_f.shape[1])
        window = np.outer(hann_y, hann_x)

        self._report(30, "Calcolo FFT...")

        shift, response = cv2.phaseCorrelate(ref_f * window, def_f * window)
        del ref_f, def_f, window

        # Rescale shift to full resolution
        dx, dy = shift[0] / scale, shift[1] / scale

        self._report(60, "Applicazione traslazione...")

        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        warped = cv2.warpAffine(deformed, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)

        ones = np.full((h, w), 255, dtype=np.uint8)
        vmask = cv2.warpAffine(ones, M, (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        del ones

        if self.params.auto_crop:
            bbox = self._compute_valid_crop(warped, ref.shape, vmask)
        else:
            bbox = (0, 0, w, h)
        del vmask

        M_full = np.eye(3, dtype=np.float64)
        M_full[0, 2] = -dx
        M_full[1, 2] = -dy

        result = AlignmentResult(
            transform_matrix=M_full,
            n_inliers=0,
            n_matches=0,
            reprojection_error=float(1.0 - response),
            crop_bbox=bbox,
            aligned_shape=(bbox[3] - bbox[1], bbox[2] - bbox[0]),
        )

        self._report(100, "Allineamento completato")
        logger.info(f"Phase shift: dx={dx:.3f}, dy={dy:.3f}, response={response:.4f}")

        if self.params.auto_crop:
            cropped = warped[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
            del warped
            return cropped, result
        return warped, result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _create_detector(self):
        """Create feature detector (SIFT or ORB)."""
        if self.params.detector_type.upper() == "SIFT":
            try:
                det = cv2.SIFT_create(nfeatures=self.params.max_features)
                return det, cv2.NORM_L2
            except (cv2.error, AttributeError):
                logger.warning("SIFT non disponibile, uso ORB")

        det = cv2.ORB_create(nfeatures=self.params.max_features)
        return det, cv2.NORM_HAMMING

    @staticmethod
    def _compute_reprojection_error(pts_src, pts_dst, M, inlier_mask):
        """Compute RMSE reprojection error for inlier matches.

        pts_src, pts_dst : (N, 1, 2) arrays
        M : 3x3 or 2x3 transform matrix
        """
        if inlier_mask is None:
            return 0.0

        inliers = inlier_mask.ravel().astype(bool)
        if not np.any(inliers):
            return 0.0

        src = pts_src[inliers].reshape(-1, 2)
        dst = pts_dst[inliers].reshape(-1, 2)

        if M.shape == (3, 3):
            # Perspective transform
            src_h = np.hstack([src, np.ones((len(src), 1))])
            projected = (M @ src_h.T).T
            projected = projected[:, :2] / projected[:, 2:3]
        else:
            # Affine transform
            src_h = np.hstack([src, np.ones((len(src), 1))])
            projected = (M @ src_h.T).T

        errors = np.sqrt(np.sum((projected - dst) ** 2, axis=1))
        return float(np.mean(errors))

    @staticmethod
    def _compute_valid_crop(warped: np.ndarray,
                            original_shape: tuple,
                            validity_mask: np.ndarray = None
                            ) -> Tuple[int, int, int, int]:
        """Find the largest-area inscribed rectangle in the valid region.

        Uses a proper validity mask (from warping a white image) rather
        than pixel values, so dark image regions are not mistaken for
        invalid warp borders.

        The algorithm uses the "maximal rectangle in a histogram"
        approach applied row-by-row, which correctly handles
        non-rectangular valid regions from perspective warps.

        Parameters
        ----------
        warped : warped image (used only for shape if validity_mask given)
        original_shape : original image shape (h, w)
        validity_mask : (H, W) uint8, 255 = valid pixel, 0 = outside warp.
                        If None, falls back to pixel-value heuristic.

        Returns (x0, y0, x1, y1).
        """
        h, w = warped.shape[:2]

        if validity_mask is not None:
            valid = (validity_mask > 127).astype(np.uint8)
        else:
            # Fallback: use pixel values (less reliable)
            if warped.ndim == 3:
                valid = np.all(warped > 0, axis=2).astype(np.uint8)
            else:
                valid = (warped > 0).astype(np.uint8)

        if not np.any(valid):
            margin = max(int(min(h, w) * 0.02), 5)
            return (margin, margin, w - margin, h - margin)

        # --- Maximal rectangle using histogram approach ---
        # Build height histogram: for each cell, count consecutive
        # valid pixels above (including self).
        heights = np.zeros((h, w), dtype=int)
        heights[0] = valid[0]
        for y in range(1, h):
            heights[y] = np.where(valid[y], heights[y - 1] + 1, 0)

        # For each row, find the largest rectangle in that histogram
        best_area = 0
        best_rect = (0, 0, w, h)  # x0, y0, x1, y1

        for y in range(h):
            row_h = heights[y]
            # Largest rectangle in histogram using stack method
            stack = []  # stack of (start_x, height)
            for x in range(w + 1):
                cur_h = row_h[x] if x < w else 0
                start = x
                while stack and stack[-1][1] > cur_h:
                    sx, sh = stack.pop()
                    area = sh * (x - sx)
                    if area > best_area:
                        best_area = area
                        # Rectangle bottom row is y, height is sh,
                        # so top row is y - sh + 1
                        best_rect = (sx, y - sh + 1, x, y + 1)
                    start = sx
                if not stack or cur_h > stack[-1][1]:
                    stack.append((start, cur_h))

        x0, y0, x1, y1 = best_rect

        if x1 <= x0 or y1 <= y0:
            margin = max(int(min(h, w) * 0.02), 5)
            return (margin, margin, w - margin, h - margin)

        # Add safety margin (3px) to avoid interpolation edge artifacts
        safety = 3
        x0 = min(x0 + safety, x1 - 1)
        y0 = min(y0 + safety, y1 - 1)
        x1 = max(x1 - safety, x0 + 1)
        y1 = max(y1 - safety, y0 + 1)

        logger.debug(
            f"Valid crop: ({x0},{y0})-({x1},{y1}) = "
            f"{x1 - x0}x{y1 - y0} px "
            f"({(x1-x0)*(y1-y0) / (w*h) * 100:.1f}% of original)")

        return (x0, y0, x1, y1)

    @staticmethod
    def draw_match_overlay(ref: np.ndarray, deformed: np.ndarray,
                           kp_ref_pts: np.ndarray, kp_def_pts: np.ndarray,
                           inlier_mask: np.ndarray = None,
                           max_draw: int = 200) -> np.ndarray:
        """Draw matching keypoints on a side-by-side image.

        Parameters
        ----------
        ref, deformed : (H, W) grayscale
        kp_ref_pts, kp_def_pts : (N, 2) matched point arrays
        inlier_mask : (N,) bool mask of inliers
        max_draw : max lines to draw

        Returns
        -------
        (H, 2*W, 3) uint8 BGR visualization
        """
        h, w = ref.shape[:2]
        vis = np.zeros((h, 2 * w, 3), dtype=np.uint8)
        vis[:, :w] = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        vis[:, w:] = cv2.cvtColor(deformed, cv2.COLOR_GRAY2BGR)

        n = min(len(kp_ref_pts), max_draw)
        indices = np.random.choice(len(kp_ref_pts), n, replace=False) if len(kp_ref_pts) > n else range(n)

        for i in indices:
            pt1 = (int(kp_ref_pts[i, 0]), int(kp_ref_pts[i, 1]))
            pt2 = (int(kp_def_pts[i, 0]) + w, int(kp_def_pts[i, 1]))

            is_inlier = inlier_mask is not None and inlier_mask[i]
            color = (0, 255, 0) if is_inlier else (0, 0, 255)

            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)
            cv2.line(vis, pt1, pt2, color, 1)

        return vis
