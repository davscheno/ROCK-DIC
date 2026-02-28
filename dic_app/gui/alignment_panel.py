"""GUI panel for automatic image alignment (registration).

Provides before/after viewers, alignment parameter controls,
and match visualization.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QSplitter, QProgressBar,
    QApplication, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from dic_app.gui.image_viewer import ImageViewer
from dic_app.core.image_registration import (
    ImageRegistration, AlignmentParameters, AlignmentResult, AlignmentMethod
)
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)


class AlignmentPanelWidget(QWidget):
    """Panel for aligning deformed image to reference.

    Signals
    -------
    alignment_applied : object
        Emitted with (AlignmentResult) when alignment is computed.
    alignment_reset : None
        Emitted when alignment is cleared.
    """

    alignment_applied = pyqtSignal(object)
    alignment_reset = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ref_image = None        # (H, W) uint8
        self._def_image = None        # (H, W) uint8
        self._result = None           # AlignmentResult
        self._aligned_def = None      # (H', W') uint8
        self._ref_cropped = None      # (H', W') uint8
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # --- Left: viewers ---
        viewer_splitter = QSplitter(Qt.Vertical)

        # Before: overlay (flicker/checker)
        before_group = QGroupBox("Prima (Sovrapposizione)")
        bg_layout = QVBoxLayout(before_group)
        self.before_viewer = ImageViewer()
        bg_layout.addWidget(self.before_viewer)
        viewer_splitter.addWidget(before_group)

        # After: aligned result
        after_group = QGroupBox("Dopo Allineamento")
        ag_layout = QVBoxLayout(after_group)
        self.after_viewer = ImageViewer()
        ag_layout.addWidget(self.after_viewer)
        viewer_splitter.addWidget(after_group)

        layout.addWidget(viewer_splitter, stretch=3)

        # --- Right: controls ---
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)

        # Method
        method_group = QGroupBox("Metodo di Allineamento")
        mg_layout = QVBoxLayout(method_group)

        mg_layout.addWidget(QLabel("Metodo:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("Omografia (Prospettica)", AlignmentMethod.FEATURE_HOMOGRAPHY)
        self.method_combo.addItem("Affine (6 DOF)", AlignmentMethod.FEATURE_AFFINE)
        self.method_combo.addItem("ECC Sub-pixel", AlignmentMethod.ECC_AFFINE)
        self.method_combo.addItem("Traslazione (FFT)", AlignmentMethod.PHASE_SHIFT)
        mg_layout.addWidget(self.method_combo)

        mg_layout.addWidget(QLabel("Detector:"))
        self.detector_combo = QComboBox()
        self.detector_combo.addItem("SIFT (preciso)", "SIFT")
        self.detector_combo.addItem("ORB (veloce)", "ORB")
        mg_layout.addWidget(self.detector_combo)

        ctrl_layout.addWidget(method_group)

        # Parameters
        param_group = QGroupBox("Parametri")
        pg_layout = QVBoxLayout(param_group)

        pg_layout.addWidget(QLabel("Max Feature:"))
        self.max_features_spin = QSpinBox()
        self.max_features_spin.setRange(500, 50000)
        self.max_features_spin.setValue(10000)
        self.max_features_spin.setSingleStep(1000)
        pg_layout.addWidget(self.max_features_spin)

        pg_layout.addWidget(QLabel("Lowe's Ratio:"))
        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.3, 0.95)
        self.ratio_spin.setValue(0.75)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setDecimals(2)
        pg_layout.addWidget(self.ratio_spin)

        pg_layout.addWidget(QLabel("RANSAC Soglia (px):"))
        self.ransac_spin = QDoubleSpinBox()
        self.ransac_spin.setRange(1.0, 30.0)
        self.ransac_spin.setValue(5.0)
        self.ransac_spin.setSingleStep(0.5)
        self.ransac_spin.setDecimals(1)
        pg_layout.addWidget(self.ransac_spin)

        self.crop_check = QCheckBox("Auto-crop area valida")
        self.crop_check.setChecked(True)
        pg_layout.addWidget(self.crop_check)

        ctrl_layout.addWidget(param_group)

        # Actions
        action_group = QGroupBox("Azioni")
        act_layout = QVBoxLayout(action_group)

        self.btn_align = QPushButton("▶  Allinea")
        self.btn_align.setMinimumHeight(36)
        self.btn_align.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.btn_align.clicked.connect(self._run_alignment)
        act_layout.addWidget(self.btn_align)

        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.clicked.connect(self._reset_alignment)
        act_layout.addWidget(self.btn_reset)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        act_layout.addWidget(self.progress)

        ctrl_layout.addWidget(action_group)

        # Results info
        info_group = QGroupBox("Risultato")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(180)
        self.info_text.setPlaceholderText("Eseguire l'allineamento per vedere i risultati...")
        info_layout.addWidget(self.info_text)
        ctrl_layout.addWidget(info_group)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl_widget, stretch=1)

        # Connect method change to enable/disable feature params
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._on_method_changed()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_images(self, ref_gray: np.ndarray, def_gray: np.ndarray):
        """Set reference and deformed images for alignment."""
        self._ref_image = ref_gray
        self._def_image = def_gray
        self._result = None
        self._aligned_def = None
        self._ref_cropped = None

        # Show overlay (average blend) as "before"
        self._show_before_overlay()
        self.after_viewer.set_image(None)
        self.info_text.clear()

    def get_alignment_result(self) -> AlignmentResult:
        return self._result

    def get_aligned_images(self):
        """Return (ref_cropped, def_aligned_cropped) or None if not aligned."""
        if self._ref_cropped is not None and self._aligned_def is not None:
            return self._ref_cropped, self._aligned_def
        return None

    def get_parameters(self) -> AlignmentParameters:
        """Collect current UI parameters."""
        return AlignmentParameters(
            method=self.method_combo.currentData(),
            detector_type=self.detector_combo.currentData(),
            max_features=self.max_features_spin.value(),
            match_ratio=self.ratio_spin.value(),
            ransac_threshold=self.ransac_spin.value(),
            auto_crop=self.crop_check.isChecked(),
        )

    def set_parameters(self, params: AlignmentParameters):
        """Restore UI from AlignmentParameters."""
        # Method
        for i in range(self.method_combo.count()):
            if self.method_combo.itemData(i) == params.method:
                self.method_combo.setCurrentIndex(i)
                break
        # Detector
        for i in range(self.detector_combo.count()):
            if self.detector_combo.itemData(i) == params.detector_type:
                self.detector_combo.setCurrentIndex(i)
                break
        self.max_features_spin.setValue(params.max_features)
        self.ratio_spin.setValue(params.match_ratio)
        self.ransac_spin.setValue(params.ransac_threshold)
        self.crop_check.setChecked(params.auto_crop)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_method_changed(self):
        """Enable/disable feature-specific parameters based on method."""
        method = self.method_combo.currentData()
        is_feature = method in (AlignmentMethod.FEATURE_HOMOGRAPHY,
                                AlignmentMethod.FEATURE_AFFINE)
        self.detector_combo.setEnabled(is_feature)
        self.max_features_spin.setEnabled(is_feature)
        self.ratio_spin.setEnabled(is_feature)
        self.ransac_spin.setEnabled(is_feature)

    # Maximum pixel dimension for overlay previews in the viewer.
    # Overlays are only for visual feedback; full-resolution is wasteful.
    _MAX_PREVIEW_DIM = 1500

    @staticmethod
    def _downscale_for_preview(img, max_dim):
        """Downscale image for viewer preview if needed."""
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _show_before_overlay(self):
        """Show blended overlay of ref and def as 'before' visualization.

        Uses downscaled images to avoid allocating a full-resolution
        RGB array (which can be hundreds of MB for drone images).
        """
        if self._ref_image is None or self._def_image is None:
            return

        ref = self._downscale_for_preview(self._ref_image, self._MAX_PREVIEW_DIM)
        deformed = self._def_image

        # Resize deformed to match ref (may differ in dimensions)
        if ref.shape != deformed.shape:
            ref_h, ref_w = ref.shape[:2]
            deformed = cv2.resize(deformed, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

        # Create color overlay: ref=green channel, def=magenta channel
        overlay = np.zeros((*ref.shape, 3), dtype=np.uint8)
        overlay[:, :, 1] = ref       # green = reference
        overlay[:, :, 0] = deformed  # red = deformed
        overlay[:, :, 2] = deformed  # blue = deformed

        self.before_viewer.set_image(overlay)
        self.before_viewer.fit_in_view()

    def _show_after_overlay(self):
        """Show aligned overlay as 'after' visualization."""
        if self._ref_cropped is None or self._aligned_def is None:
            return

        ref = self._downscale_for_preview(self._ref_cropped, self._MAX_PREVIEW_DIM)
        aligned = self._aligned_def
        if ref.shape != aligned.shape:
            ref_h, ref_w = ref.shape[:2]
            aligned = cv2.resize(aligned, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

        overlay = np.zeros((*ref.shape, 3), dtype=np.uint8)
        overlay[:, :, 1] = ref       # green
        overlay[:, :, 0] = aligned   # red
        overlay[:, :, 2] = aligned   # blue

        self.after_viewer.set_image(overlay)
        self.after_viewer.fit_in_view()

    def _run_alignment(self):
        """Execute alignment with current parameters."""
        if self._ref_image is None or self._def_image is None:
            QMessageBox.warning(self, "Errore",
                                "Caricare prima le immagini (riferimento + deformata)")
            return

        params = self.get_parameters()
        reg = ImageRegistration(params)

        self.btn_align.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.info_text.clear()
        QApplication.processEvents()

        def on_progress(pct, msg):
            self.progress.setValue(pct)
            self.info_text.append(msg)
            QApplication.processEvents()

        reg.set_progress_callback(on_progress)

        try:
            aligned_def, result = reg.align(
                self._ref_image, self._def_image)

            self._result = result
            self._aligned_def = aligned_def

            # Crop reference to same region
            x0, y0, x1, y1 = result.crop_bbox
            if params.auto_crop and x1 > x0 and y1 > y0:
                self._ref_cropped = self._ref_image[y0:y1, x0:x1].copy()
            else:
                self._ref_cropped = self._ref_image.copy()

            # Show result
            self._show_after_overlay()
            self._show_result_info(result)

            # Emit signal
            self.alignment_applied.emit(result)

        except Exception as e:
            QMessageBox.critical(self, "Errore Allineamento", str(e))
            self.info_text.append(f"\nERRORE: {e}")

        finally:
            self.btn_align.setEnabled(True)
            self.progress.setVisible(False)

    def _reset_alignment(self):
        """Clear alignment results."""
        self._result = None
        self._aligned_def = None
        self._ref_cropped = None
        self.after_viewer.set_image(None)
        self.info_text.clear()
        self.info_text.setPlaceholderText("Allineamento resettato.")
        self._show_before_overlay()
        self.alignment_reset.emit()

    def _show_result_info(self, result: AlignmentResult):
        """Display alignment result info."""
        self.info_text.clear()

        text = "═══ RISULTATO ALLINEAMENTO ═══\n\n"
        text += f"Metodo: {result.method_used.value}\n"
        text += f"Corrispondenze: {result.n_matches}\n"
        text += f"Inlier (RANSAC): {result.n_inliers}\n"

        if result.n_matches > 0:
            pct = result.n_inliers / result.n_matches * 100
            text += f"Percentuale inlier: {pct:.1f}%\n"

        text += f"Errore RMSE: {result.reprojection_error:.4f} px\n"
        text += f"Tempo: {result.computation_time_s:.2f} s\n"

        x0, y0, x1, y1 = result.crop_bbox
        crop_w, crop_h = x1 - x0, y1 - y0
        text += f"\nArea valida: {crop_w} × {crop_h} px\n"

        if self._ref_image is not None:
            orig_area = self._ref_image.shape[0] * self._ref_image.shape[1]
            crop_area = crop_w * crop_h
            text += f"Area mantenuta: {crop_area / orig_area * 100:.1f}%\n"

        # Decompose transformation
        M = result.transform_matrix
        text += f"\nMatrice trasformazione:\n"
        for row in M:
            text += "  [" + ", ".join(f"{v:10.6f}" for v in row) + "]\n"

        # Extract translation
        tx, ty = M[0, 2], M[1, 2]
        text += f"\nTraslazione: dx={tx:.2f}, dy={ty:.2f} px\n"

        # Extract rotation (approximate from upper-left 2x2)
        angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
        text += f"Rotazione: {angle:.3f}°\n"
        text += f"Scala: {scale:.6f}\n"

        self.info_text.setText(text)
