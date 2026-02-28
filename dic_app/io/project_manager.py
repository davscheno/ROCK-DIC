"""Project save/load manager for .dicproj directories."""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from dic_app.core.dic_engine import DICParameters, DICResult
from dic_app.core.preprocessing import FilterPipeline
from dic_app.core.mask_manager import MaskManager
from dic_app.core.image_registration import AlignmentParameters, AlignmentResult
from dic_app.io.image_loader import ImageData, ImageLoader
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)


@dataclass
class ProjectState:
    """Central data model shared across all GUI components."""
    project_name: str = "Senza Titolo"
    project_filepath: Optional[str] = None

    # Images
    image_paths: List[str] = field(default_factory=list)
    reference_index: int = 0
    images_data: List[ImageData] = field(default_factory=list, repr=False)

    # Preprocessing
    filter_pipeline: Optional[FilterPipeline] = None
    preprocessed_images: List[np.ndarray] = field(default_factory=list, repr=False)

    # Alignment
    alignment_params: Optional[AlignmentParameters] = None
    alignment_result: Optional[AlignmentResult] = None
    aligned_ref: Optional[np.ndarray] = field(default=None, repr=False)
    aligned_def: Optional[np.ndarray] = field(default=None, repr=False)
    aligned_def_index: Optional[int] = None  # Index of the deformed image that was aligned

    # Masking
    mask_manager: Optional[MaskManager] = None

    # DIC Parameters
    dic_params: Optional[DICParameters] = None

    # Results
    results: List[DICResult] = field(default_factory=list, repr=False)
    strain_data: Optional[dict] = None
    active_zones: Optional[list] = None

    # Metadata
    gsd: Optional[float] = None
    site_name: str = ""
    analyst_name: str = ""
    notes: str = ""

    def get_reference_image(self):
        """Get current reference image (aligned > preprocessed > raw)."""
        # Aligned images have highest priority
        if self.aligned_ref is not None:
            return self.aligned_ref
        if self.preprocessed_images and self.reference_index < len(self.preprocessed_images):
            return self.preprocessed_images[self.reference_index]
        if self.images_data and self.reference_index < len(self.images_data):
            return self.images_data[self.reference_index].image_gray
        return None

    def get_deformed_image(self, index=None):
        """Get deformed image (aligned > preprocessed > raw).

        If alignment exists and the requested index matches the aligned
        deformed image, returns the aligned+cropped version.  If the
        requested index is *different*, warps+crops that image on the
        fly using the stored transform so both ref and def have the
        same geometry.
        """
        if index is None:
            # Return first non-reference image
            for i in range(len(self.images_data)):
                if i != self.reference_index:
                    index = i
                    break
        if index is None:
            return None

        # If alignment exists, return aligned & cropped version
        if self.alignment_result is not None and self.aligned_ref is not None:
            # Check if this is the same deformed image that was aligned
            def_index = self.aligned_def_index
            if def_index is None:
                # Legacy: aligned_def_index not set, guess first non-ref
                for i in range(len(self.images_data)):
                    if i != self.reference_index:
                        def_index = i
                        break

            if index == def_index and self.aligned_def is not None:
                return self.aligned_def

            # Different index: warp+crop this image using stored transform
            raw_img = None
            if self.preprocessed_images and index < len(self.preprocessed_images):
                raw_img = self.preprocessed_images[index]
            elif self.images_data and index < len(self.images_data):
                raw_img = self.images_data[index].image_gray

            if raw_img is not None:
                try:
                    from dic_app.core.image_registration import ImageRegistration
                    reg = ImageRegistration(self.alignment_params)
                    _, def_crop = reg.apply_to_pair(
                        raw_img, raw_img, self.alignment_result)
                    # apply_to_pair warps the 2nd argument; but here we
                    # need to warp raw_img with the same transform.
                    # Since apply_to_pair crops the ref too, just return
                    # the cropped version at the correct bbox.
                    import cv2
                    h, w = raw_img.shape[:2]
                    M = self.alignment_result.transform_matrix
                    ones_mask = np.ones((h, w), dtype=np.uint8) * 255
                    if M.shape == (3, 3):
                        warped = cv2.warpPerspective(
                            raw_img, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        validity_mask = cv2.warpPerspective(
                            ones_mask, M, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    else:
                        M2x3 = M[:2] if M.shape[0] == 3 else M
                        warped = cv2.warpAffine(
                            raw_img, M2x3, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        validity_mask = cv2.warpAffine(
                            ones_mask, M2x3, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                    bbox = ImageRegistration._compute_valid_crop(
                        warped, raw_img.shape, validity_mask)
                    x0, y0, x1, y1 = bbox
                    if x1 > x0 and y1 > y0:
                        return warped[y0:y1, x0:x1].copy()
                    return warped
                except Exception as e:
                    logger.warning(
                        f"Failed to warp+crop deformed image {index}: {e}")

        if self.preprocessed_images and index < len(self.preprocessed_images):
            return self.preprocessed_images[index]
        if self.images_data and index < len(self.images_data):
            return self.images_data[index].image_gray
        return None

    def get_pre_alignment_ref(self):
        """Get reference image before alignment (preprocessed or raw)."""
        if self.preprocessed_images and self.reference_index < len(self.preprocessed_images):
            return self.preprocessed_images[self.reference_index]
        if self.images_data and self.reference_index < len(self.images_data):
            return self.images_data[self.reference_index].image_gray
        return None

    def get_pre_alignment_def(self, index=None):
        """Get deformed image before alignment (preprocessed or raw)."""
        if index is None:
            for i in range(len(self.images_data)):
                if i != self.reference_index:
                    index = i
                    break
        if index is None:
            return None
        if self.preprocessed_images and index < len(self.preprocessed_images):
            return self.preprocessed_images[index]
        if self.images_data and index < len(self.images_data):
            return self.images_data[index].image_gray
        return None

    def get_reference_rgb(self):
        """Get reference RGB image for display.

        If alignment has been applied, crops the RGB image to the same
        valid overlap region so heatmaps overlay correctly.
        """
        if not self.images_data or self.reference_index >= len(self.images_data):
            return None

        rgb = self.images_data[self.reference_index].image_rgb
        if rgb is None:
            return None

        # If alignment was performed, crop RGB to match aligned gray
        if (self.alignment_result is not None
                and self.aligned_ref is not None):
            try:
                from dic_app.core.image_registration import ImageRegistration
                import cv2
                h, w = rgb.shape[:2]
                M = self.alignment_result.transform_matrix

                # Create validity mask by warping a white image
                ones_mask = np.ones((h, w), dtype=np.uint8) * 255
                if M.shape == (3, 3):
                    validity_mask = cv2.warpPerspective(
                        ones_mask, M, (w, h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                else:
                    validity_mask = cv2.warpAffine(
                        ones_mask, M[:2], (w, h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                bbox = ImageRegistration._compute_valid_crop(
                    validity_mask, (h, w), validity_mask)
                x0, y0, x1, y1 = bbox
                if x1 > x0 and y1 > y0:
                    return rgb[y0:y1, x0:x1].copy()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to crop RGB for alignment: {e}")

        return rgb

    def get_deformed_rgb(self, index=None):
        """Get deformed RGB image for display.

        If alignment has been applied, warps+crops the RGB image to the
        same valid overlap region.
        """
        if index is None:
            for i in range(len(self.images_data)):
                if i != self.reference_index:
                    index = i
                    break
        if index is None:
            return None
        if not self.images_data or index >= len(self.images_data):
            return None

        rgb = self.images_data[index].image_rgb
        if rgb is None:
            return None

        # If alignment was performed, warp+crop RGB to match aligned geometry
        if (self.alignment_result is not None
                and self.aligned_ref is not None):
            try:
                from dic_app.core.image_registration import ImageRegistration
                import cv2
                h, w = rgb.shape[:2]
                M = self.alignment_result.transform_matrix

                ones_mask = np.ones((h, w), dtype=np.uint8) * 255
                if M.shape == (3, 3):
                    warped = cv2.warpPerspective(
                        rgb, M, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    validity_mask = cv2.warpPerspective(
                        ones_mask, M, (w, h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                else:
                    M2x3 = M[:2] if M.shape[0] == 3 else M
                    warped = cv2.warpAffine(
                        rgb, M2x3, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    validity_mask = cv2.warpAffine(
                        ones_mask, M2x3, (w, h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                bbox = ImageRegistration._compute_valid_crop(
                    warped[:, :, 0] if warped.ndim == 3 else warped,
                    (h, w), validity_mask)
                x0, y0, x1, y1 = bbox
                if x1 > x0 and y1 > y0:
                    return warped[y0:y1, x0:x1].copy()
            except Exception as e:
                logger.warning(f"Failed to crop deformed RGB for alignment: {e}")

        return rgb

    def get_gps_info(self):
        """Get GPS info from reference image as dict."""
        if self.images_data and self.reference_index < len(self.images_data):
            img = self.images_data[self.reference_index]
            if img.has_gps:
                return {
                    'latitude': img.gps.latitude,
                    'longitude': img.gps.longitude,
                    'altitude': img.gps.altitude,
                    'timestamp': img.gps.timestamp,
                    'camera': img.camera_model,
                    'focal_length_mm': img.focal_length_mm,
                    'capture_time': img.capture_time,
                }
        return {}


class ProjectManager:
    """Save and load project state to/from .dicproj directories."""

    PROJECT_EXT = '.dicproj'
    META_FILE = 'project.json'
    MASK_FILE = 'mask.png'
    RESULTS_DIR = 'results'

    @staticmethod
    def save(state: ProjectState, filepath: str):
        """Save project state to a .dicproj directory.

        Directory structure:
        project_name.dicproj/
        ├── project.json          # metadata, params, filter pipeline, ROIs
        ├── mask.png              # rasterized mask
        └── results/
            ├── result_0.npz      # DIC result arrays
            └── result_1.npz
        """
        proj_dir = filepath
        if not proj_dir.endswith(ProjectManager.PROJECT_EXT):
            proj_dir += ProjectManager.PROJECT_EXT

        os.makedirs(proj_dir, exist_ok=True)
        results_dir = os.path.join(proj_dir, ProjectManager.RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)

        # Build metadata dict
        meta = {
            'project_name': state.project_name,
            'image_paths': state.image_paths,
            'reference_index': state.reference_index,
            'gsd': state.gsd,
            'site_name': state.site_name,
            'analyst_name': state.analyst_name,
            'notes': state.notes,
        }

        # Filter pipeline
        if state.filter_pipeline:
            meta['filter_pipeline'] = state.filter_pipeline.to_dict()

        # Mask manager
        if state.mask_manager:
            meta['mask_manager'] = state.mask_manager.to_dict()
            # Also save rasterized mask
            mask_path = os.path.join(proj_dir, ProjectManager.MASK_FILE)
            state.mask_manager.save_mask(mask_path)

        # Alignment
        if state.alignment_params:
            meta['alignment_params'] = state.alignment_params.to_dict()
        if state.alignment_result:
            meta['alignment_result'] = state.alignment_result.to_dict()
        if state.aligned_def_index is not None:
            meta['aligned_def_index'] = state.aligned_def_index

        # DIC parameters
        if state.dic_params:
            meta['dic_params'] = state.dic_params.to_dict()

        # Active zones
        if state.active_zones:
            meta['active_zones'] = state.active_zones

        # Number of results
        meta['n_results'] = len(state.results)

        # Save metadata JSON
        meta_path = os.path.join(proj_dir, ProjectManager.META_FILE)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        # Save results as npz
        for i, result in enumerate(state.results):
            npz_path = os.path.join(results_dir, f"result_{i}.npz")
            result.to_npz(npz_path)

        state.project_filepath = proj_dir
        logger.info(f"Project saved to: {proj_dir}")

    @staticmethod
    def load(filepath: str) -> ProjectState:
        """Load project state from a .dicproj directory."""
        proj_dir = filepath
        if not os.path.isdir(proj_dir):
            raise FileNotFoundError(f"Project directory not found: {proj_dir}")

        # Load metadata
        meta_path = os.path.join(proj_dir, ProjectManager.META_FILE)
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        state = ProjectState()
        state.project_filepath = proj_dir
        state.project_name = meta.get('project_name', 'Senza Titolo')
        state.image_paths = meta.get('image_paths', [])
        state.reference_index = meta.get('reference_index', 0)
        state.gsd = meta.get('gsd')
        state.site_name = meta.get('site_name', '')
        state.analyst_name = meta.get('analyst_name', '')
        state.notes = meta.get('notes', '')
        state.active_zones = meta.get('active_zones')

        # Reload images
        for path in state.image_paths:
            try:
                img_data = ImageLoader.load(path)
                state.images_data.append(img_data)
            except Exception as e:
                logger.warning(f"Failed to reload image {path}: {e}")

        # Filter pipeline
        if 'filter_pipeline' in meta:
            state.filter_pipeline = FilterPipeline.from_dict(meta['filter_pipeline'])
            # Re-apply pipeline to images
            if state.filter_pipeline and state.images_data:
                state.preprocessed_images = []
                for img_data in state.images_data:
                    processed = state.filter_pipeline.apply(img_data.image_gray)
                    state.preprocessed_images.append(processed)

        # Mask manager
        if 'mask_manager' in meta:
            image_shape = None
            if state.images_data:
                image_shape = state.images_data[0].image_gray.shape
            state.mask_manager = MaskManager.from_dict(
                meta['mask_manager'], image_shape)

        # Alignment
        if 'alignment_params' in meta:
            state.alignment_params = AlignmentParameters.from_dict(meta['alignment_params'])
        state.aligned_def_index = meta.get('aligned_def_index', None)
        if 'alignment_result' in meta:
            state.alignment_result = AlignmentResult.from_dict(meta['alignment_result'])
            # Re-apply alignment if we have images and result
            if state.alignment_result and state.images_data:
                try:
                    from dic_app.core.image_registration import ImageRegistration
                    reg = ImageRegistration(state.alignment_params)
                    ref_img = state.get_pre_alignment_ref()
                    def_img = state.get_pre_alignment_def()
                    if ref_img is not None and def_img is not None:
                        state.aligned_ref, state.aligned_def = reg.apply_to_pair(
                            ref_img, def_img, state.alignment_result)
                except Exception as e:
                    logger.warning(f"Failed to re-apply alignment: {e}")

        # DIC parameters
        if 'dic_params' in meta:
            state.dic_params = DICParameters.from_dict(meta['dic_params'])

        # Load results
        n_results = meta.get('n_results', 0)
        results_dir = os.path.join(proj_dir, ProjectManager.RESULTS_DIR)
        for i in range(n_results):
            npz_path = os.path.join(results_dir, f"result_{i}.npz")
            if os.path.exists(npz_path):
                try:
                    result = DICResult.from_npz(npz_path, state.dic_params)
                    state.results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to load result {i}: {e}")

        logger.info(f"Project loaded from: {proj_dir} "
                     f"({len(state.images_data)} images, {len(state.results)} results)")

        return state
