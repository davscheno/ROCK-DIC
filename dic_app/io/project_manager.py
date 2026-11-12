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
    # (reference index, deformed index) of the images that produced
    # ``results``; None = legacy project (reference + first deformed).
    results_pair: Optional[tuple] = None
    strain_data: Optional[dict] = None
    active_zones: Optional[list] = None

    # Multi-epoch analysis (dic_app.core.time_series.TimeSeriesResult)
    time_series: Optional[object] = field(default=None, repr=False)

    # Metadata
    gsd: Optional[float] = None
    site_name: str = ""
    analyst_name: str = ""
    notes: str = ""

    # Cache of per-image transforms for deformed images other than the
    # one aligned in the alignment panel: {index: (raw, ref, result, M)}
    _extra_alignments: dict = field(default_factory=dict, repr=False)

    def get_reference_image(self):
        """Get current reference image (aligned > preprocessed > raw)."""
        # Aligned images have highest priority
        if self.aligned_ref is not None:
            return self.aligned_ref
        return self._gray_for_analysis(self.reference_index)

    # ------------------------------------------------------------------
    # Images and their filtered versions (always kept in lock-step)
    # ------------------------------------------------------------------

    def _gray_for_analysis(self, index):
        """Filtered image *index* if a pipeline was applied, else raw.

        preprocessed_images[i] always belongs to images_data[i]; an image
        added after the pipeline was applied is filtered on demand, so DIC
        never compares a filtered image with an unfiltered one.
        """
        if not self.images_data or not (0 <= index < len(self.images_data)):
            return None
        if self.filter_pipeline is not None and self.preprocessed_images:
            while len(self.preprocessed_images) < len(self.images_data):
                i = len(self.preprocessed_images)
                self.preprocessed_images.append(
                    self.filter_pipeline.apply(self.images_data[i].image_gray))
        if index < len(self.preprocessed_images):
            return self.preprocessed_images[index]
        return self.images_data[index].image_gray

    def add_image(self, img_data, path):
        """Append an image (filtered at once if a pipeline is applied)."""
        self.images_data.append(img_data)
        self.image_paths.append(path)
        if self.filter_pipeline is not None and self.preprocessed_images:
            self._gray_for_analysis(len(self.images_data) - 1)

    def remove_image(self, row):
        """Remove image *row* and remap every index that refers to images.

        Returns a list of messages for the user (what had to be reset).
        """
        notes = []
        if not (0 <= row < len(self.images_data)):
            return notes
        self.images_data.pop(row)
        if row < len(self.image_paths):
            self.image_paths.pop(row)
        if row < len(self.preprocessed_images):
            self.preprocessed_images.pop(row)
        self.time_series = None              # its image indices shift
        self.clear_extra_alignments()        # cached by image index

        shift = (lambda i: i - 1 if i > row else i)
        old_ref = self.reference_index
        aligned_def = (self._effective_aligned_def_index()
                       if self.alignment_result is not None else None)

        if self.alignment_result is not None and row in (old_ref, aligned_def):
            self.alignment_result = None
            self.aligned_ref = self.aligned_def = None
            self.aligned_def_index = None
            notes.append("Allineamento annullato: usava l'immagine rimossa.")
        elif self.aligned_def_index is not None:
            self.aligned_def_index = shift(self.aligned_def_index)
        elif aligned_def is not None:        # legacy: make it explicit
            self.aligned_def_index = shift(aligned_def)

        if row == old_ref:
            self.reference_index = 0
            if self.images_data:
                notes.append("Il riferimento era l'immagine rimossa: ora il "
                             "riferimento e' la prima immagine.")
        else:
            self.reference_index = shift(old_ref)

        pair = self.results_pair
        if self.results and pair is None:
            pair = (old_ref, self._default_deformed_index_before(row, old_ref))
        if self.results and pair is not None:
            if row in pair or row == old_ref:
                self.results = []
                self.results_pair = None
                self.active_zones = None
                notes.append("Risultati DIC eliminati: riguardavano "
                             "l'immagine rimossa.")
            else:
                self.results_pair = tuple(shift(i) for i in pair)
        return notes

    def _default_deformed_index_before(self, removed, ref):
        """Default deformed index as it was before *removed* was popped."""
        n = len(self.images_data) + 1
        return next((i for i in range(n) if i != ref), None)

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------

    def clear_extra_alignments(self):
        """Drop cached per-image transforms (call when alignment changes)."""
        self._extra_alignments.clear()

    def _default_deformed_index(self):
        for i in range(len(self.images_data)):
            if i != self.reference_index:
                return i
        return None

    def _effective_aligned_def_index(self):
        if self.aligned_def_index is not None:
            return self.aligned_def_index
        # Legacy: aligned_def_index not set, alignment was on first non-ref
        return self._default_deformed_index()

    def _alignment_active(self):
        return self.alignment_result is not None and self.aligned_ref is not None

    def _reference_crop_bbox(self, ref_shape):
        from dic_app.core.image_registration import ImageRegistration
        return ImageRegistration.reference_crop_bbox(
            self.alignment_result, ref_shape)

    def _transform_for_index(self, index):
        """Transform that maps image *index* onto the reference frame.

        The stored alignment_result belongs to ONE specific deformed image
        (aligned_def_index).  Every other image has its own camera motion,
        so it is registered against the reference on demand (and cached).
        Returns None if registration fails.
        """
        if index == self._effective_aligned_def_index():
            return self.alignment_result.transform_matrix

        ref_pre = self.get_pre_alignment_ref()
        raw = self.get_pre_alignment_def(index)
        if ref_pre is None or raw is None:
            return None

        cached = self._extra_alignments.get(index)
        if (cached is not None and cached[0] is raw and cached[1] is ref_pre
                and cached[2] is self.alignment_result):
            return cached[3]

        from dic_app.core.image_registration import ImageRegistration
        try:
            reg = ImageRegistration(self.alignment_params)
            _, res = reg.align(ref_pre, raw)
        except Exception as e:
            logger.error(f"Registration of image {index} onto the reference "
                         f"failed: {e}")
            return None

        M = res.transform_matrix
        logger.info(f"Image {index} registered onto reference "
                    f"(RMSE={res.reprojection_error:.3f}px)")
        self._extra_alignments[index] = (raw, ref_pre, self.alignment_result, M)
        return M

    def _warp_and_crop_to_reference(self, image, index):
        """Warp *image* (gray or RGB, of image *index*) into the aligned
        reference geometry: same transform family, same crop as aligned_ref."""
        from dic_app.core.image_registration import ImageRegistration
        M = self._transform_for_index(index)
        if M is None:
            return None
        ref_pre = self.get_pre_alignment_ref()
        ref_shape = ref_pre.shape[:2]
        warped, vmask = ImageRegistration.warp_to_reference(image, M, ref_shape)
        x0, y0, x1, y1 = self._reference_crop_bbox(ref_shape)
        crop_valid = vmask[y0:y1, x0:x1]
        invalid_pct = 100.0 * float(np.mean(crop_valid == 0)) if crop_valid.size else 0.0
        if invalid_pct > 0.5:
            logger.warning(
                f"Image {index}: {invalid_pct:.1f}% of the aligned reference "
                f"area is outside this image (black borders) - those zones "
                f"will give unreliable DIC values.")
        return warped[y0:y1, x0:x1].copy()

    def get_deformed_image(self, index=None):
        """Get deformed image (aligned > preprocessed > raw).

        With an active alignment the returned image always has the same
        geometry as aligned_ref.  The image that was aligned in the
        alignment panel is returned directly; any other image is
        registered against the reference with its OWN transform and
        cropped with the reference crop box.  Returns None if that
        registration fails (never an unregistered image).
        """
        if index is None:
            index = self._default_deformed_index()
        if index is None:
            return None

        if self._alignment_active():
            if (index == self._effective_aligned_def_index()
                    and self.aligned_def is not None):
                return self.aligned_def

            raw_img = self.get_pre_alignment_def(index)
            if raw_img is None:
                return None
            try:
                return self._warp_and_crop_to_reference(raw_img, index)
            except Exception as e:
                logger.error(f"Failed to align deformed image {index}: {e}")
                return None

        return self.get_pre_alignment_def(index)

    def get_pre_alignment_ref(self):
        """Get reference image before alignment (preprocessed or raw)."""
        return self._gray_for_analysis(self.reference_index)

    def get_pre_alignment_def(self, index=None):
        """Get deformed image before alignment (preprocessed or raw)."""
        if index is None:
            for i in range(len(self.images_data)):
                if i != self.reference_index:
                    index = i
                    break
        if index is None:
            return None
        return self._gray_for_analysis(index)

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

        if self._alignment_active():
            try:
                x0, y0, x1, y1 = self._reference_crop_bbox(rgb.shape[:2])
                return rgb[y0:y1, x0:x1].copy()
            except Exception as e:
                logger.warning(f"Failed to crop RGB for alignment: {e}")

        return rgb

    def get_deformed_rgb(self, index=None):
        """Get deformed RGB image for display.

        If alignment has been applied, warps+crops the RGB image with the
        transform of THIS image into the aligned reference geometry.
        """
        if index is None:
            index = self._default_deformed_index()
        if index is None:
            return None
        if not self.images_data or index >= len(self.images_data):
            return None

        rgb = self.images_data[index].image_rgb
        if rgb is None:
            return None

        if self._alignment_active():
            try:
                warped = self._warp_and_crop_to_reference(rgb, index)
                if warped is not None:
                    return warped
            except Exception as e:
                logger.warning(f"Failed to crop deformed RGB for alignment: {e}")

        return rgb

    def reference_frame_offset(self):
        """(x, y) of the analysed reference inside the original reference
        photo: the alignment crop origin, (0, 0) without alignment."""
        if self._alignment_active():
            pre = self.get_pre_alignment_ref()
            if pre is not None:
                x0, y0, _, _ = self._reference_crop_bbox(pre.shape[:2])
                return int(x0), int(y0)
        return 0, 0

    def mask_manager_in_reference_frame(self, shape=None):
        """The ROIs expressed on the current reference image.

        ROIs may have been drawn before the alignment (on the full photo)
        or after it (on the crop): both are translated onto the current
        reference.  Returns None when the frame of an old project cannot be
        determined (the caller then uses the legacy crop logic).
        """
        mm = self.mask_manager
        if mm is None:
            return None
        ref = self.get_reference_image()
        if shape is None:
            if ref is None:
                return None
            shape = ref.shape[:2]
        offset = self.reference_frame_offset()
        src_offset = mm.frame_offset
        if src_offset is None:                       # legacy project
            pre = self.get_pre_alignment_ref()
            if pre is not None and tuple(mm.image_shape) == pre.shape[:2]:
                src_offset = (0, 0)
            elif ref is not None and tuple(mm.image_shape) == ref.shape[:2]:
                src_offset = offset
            else:
                return None
            mm.frame_offset = src_offset
        return mm.reframed(shape, offset)

    def get_analysis_frame(self):
        """Where the analysed reference sits inside the original photo.

        Returns {'original_shape': (h, w), 'pixel_offset': (x0, y0)};
        the offset is the alignment crop origin (0, 0 without alignment).
        """
        if not self.images_data or self.reference_index >= len(self.images_data):
            return {}
        ref_img = self.images_data[self.reference_index]
        loaded_shape = tuple(ref_img.image_gray.shape[:2])
        original_shape = loaded_shape
        offset = (0, 0)
        if self._alignment_active():
            try:
                x0, y0, _, _ = self._reference_crop_bbox(loaded_shape)
                offset = (x0, y0)
            except Exception as e:
                logger.warning(f"Could not determine alignment crop offset: {e}")

        # Reference is a preliminary crop of a guide photo: the GPS refers
        # to the centre of that original photo, so add the crop origin.
        pc = ref_img.precrop
        if pc and pc.get('is_guide') and pc.get('guide_original_shape'):
            px0, py0 = pc['crop_rect_guide'][:2]
            original_shape = tuple(pc['guide_original_shape'][:2])
            offset = (offset[0] + px0, offset[1] + py0)
        return {'original_shape': original_shape, 'pixel_offset': offset}

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


def _jsonable(obj):
    """NumPy scalars/arrays and tuples -> plain JSON types (json's
    default=str would turn numbers into strings)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


class ProjectManager:
    """Save and load project state to/from .dicproj directories."""

    PROJECT_EXT = '.dicproj'
    META_FILE = 'project.json'
    MASK_FILE = 'mask.png'
    RESULTS_DIR = 'results'
    SERIES_DIR = 'timeseries'

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
        # Image paths are stored both absolute and relative to the project
        # folder, so the project still opens after moving the whole folder
        # (e.g. a synced Drive on another computer).
        rel_paths = []
        for p in state.image_paths:
            try:
                rel_paths.append(os.path.relpath(os.path.abspath(p), proj_dir))
            except ValueError:  # different drive on Windows
                rel_paths.append(None)

        meta = {
            'project_name': state.project_name,
            'image_paths': [os.path.abspath(p) for p in state.image_paths],
            'image_paths_rel': rel_paths,
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

        # Active zones (status, manual flag, detachment perimeters)
        if state.active_zones:
            meta['active_zones'] = _jsonable(state.active_zones)

        # Number of results and the image pair they were computed on
        meta['n_results'] = len(state.results)
        if state.results and state.results_pair is not None:
            meta['results_pair'] = [int(i) for i in state.results_pair]

        # Save metadata JSON
        meta_path = os.path.join(proj_dir, ProjectManager.META_FILE)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        # Remove result files from a previous save that had more results
        for fname in os.listdir(results_dir):
            if fname.startswith('result_') and fname.endswith('.npz'):
                os.remove(os.path.join(results_dir, fname))

        # Save results as npz
        for i, result in enumerate(state.results):
            npz_path = os.path.join(results_dir, f"result_{i}.npz")
            result.to_npz(npz_path)

        # Time series (multi-epoch analysis)
        series_dir = os.path.join(proj_dir, ProjectManager.SERIES_DIR)
        if state.time_series is not None:
            for e in state.time_series.epochs:
                if 0 <= e.index < len(state.image_paths):
                    e.path = os.path.abspath(state.image_paths[e.index])
            state.time_series.save(series_dir)
        elif os.path.isdir(series_dir):
            import shutil
            shutil.rmtree(series_dir, ignore_errors=True)

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

        # Reload images.  image_paths, images_data and all stored indices
        # must stay in lock-step, so images that fail to load are removed
        # from image_paths too and the indices are remapped.
        saved_paths = list(state.image_paths)
        rel_paths = meta.get('image_paths_rel') or []
        old_to_new = {}
        loaded_paths = []
        for old_idx, path in enumerate(saved_paths):
            if not os.path.exists(path) and old_idx < len(rel_paths) \
                    and rel_paths[old_idx]:
                candidate = os.path.normpath(
                    os.path.join(proj_dir, rel_paths[old_idx]))
                if os.path.exists(candidate):
                    logger.info(f"Image moved: using {candidate}")
                    path = candidate
            try:
                img_data = ImageLoader.load(path)
            except Exception as e:
                logger.warning(f"Failed to reload image {path}: {e}")
                continue
            old_to_new[old_idx] = len(state.images_data)
            state.images_data.append(img_data)
            loaded_paths.append(path)
        state.image_paths = loaded_paths

        saved_ref_index = state.reference_index
        saved_aligned_def_index = meta.get('aligned_def_index', None)
        if saved_aligned_def_index is None and 'alignment_result' in meta:
            # Legacy projects: alignment was always on the first non-ref image
            saved_aligned_def_index = next(
                (i for i in range(len(saved_paths)) if i != saved_ref_index),
                None)

        alignment_lost = False
        if saved_ref_index in old_to_new:
            state.reference_index = old_to_new[saved_ref_index]
        else:
            if saved_paths:
                logger.warning(
                    f"Reference image {saved_ref_index} could not be loaded; "
                    f"falling back to the first available image. "
                    f"Alignment and results are discarded.")
            state.reference_index = 0
            alignment_lost = True

        if (saved_aligned_def_index is not None
                and saved_aligned_def_index not in old_to_new):
            logger.warning("Aligned deformed image could not be loaded; "
                           "alignment is discarded.")
            alignment_lost = True
        mapped_aligned_def_index = (
            old_to_new.get(saved_aligned_def_index)
            if saved_aligned_def_index is not None else None)

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
        if 'alignment_result' in meta and not alignment_lost:
            state.aligned_def_index = mapped_aligned_def_index
            state.alignment_result = AlignmentResult.from_dict(meta['alignment_result'])
            # Re-apply alignment to the SAME pair that was originally aligned
            if state.alignment_result and state.images_data:
                try:
                    from dic_app.core.image_registration import ImageRegistration
                    reg = ImageRegistration(state.alignment_params)
                    ref_img = state.get_pre_alignment_ref()
                    def_img = state.get_pre_alignment_def(state.aligned_def_index)
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
        if alignment_lost and n_results:
            # Results refer to image geometry that can no longer be rebuilt
            logger.warning("Saved DIC results discarded: their reference/"
                           "aligned images are missing.")
            n_results = 0
            state.active_zones = None
        saved_pair = meta.get('results_pair')
        if n_results and saved_pair is not None:
            if all(i in old_to_new for i in saved_pair):
                state.results_pair = tuple(old_to_new[i] for i in saved_pair)
            else:
                # Showing them on other images would mislead the user
                logger.warning("Saved DIC results discarded: the images they "
                               "were computed on are missing.")
                n_results = 0
                state.active_zones = None
        results_dir = os.path.join(proj_dir, ProjectManager.RESULTS_DIR)
        for i in range(n_results):
            npz_path = os.path.join(results_dir, f"result_{i}.npz")
            if os.path.exists(npz_path):
                try:
                    result = DICResult.from_npz(npz_path, state.dic_params)
                    state.results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to load result {i}: {e}")

        # Time series: image indices must still point to the same files
        series_dir = os.path.join(proj_dir, ProjectManager.SERIES_DIR)
        if os.path.isfile(os.path.join(series_dir, 'series.json')) and not alignment_lost:
            try:
                from dic_app.core.time_series import TimeSeriesResult
                ts = TimeSeriesResult.load(series_dir, state.dic_params)
                names = [os.path.basename(p) for p in state.image_paths]
                ok = True
                for e in ts.epochs:
                    if e.index not in old_to_new:
                        ok = False
                        break
                    e.index = old_to_new[e.index]
                    if names[e.index] != e.label:
                        ok = False
                        break
                if ok and ts.epochs[ts.ref_pos].index == state.reference_index:
                    state.time_series = ts
                else:
                    logger.warning("Saved time series discarded: its images "
                                   "are no longer in the project")
            except Exception as e:
                logger.warning(f"Failed to load time series: {e}")

        logger.info(f"Project loaded from: {proj_dir} "
                     f"({len(state.images_data)} images, {len(state.results)} results)")

        return state
