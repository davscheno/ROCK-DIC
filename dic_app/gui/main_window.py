"""Main application window with 8-tab workflow."""

import os
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QDoubleSpinBox, QSpinBox, QComboBox, QFileDialog, QProgressBar,
    QStatusBar, QMenuBar, QToolBar, QMessageBox, QSplitter,
    QFormLayout, QTextEdit, QApplication, QAction
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap

from dic_app.gui.image_viewer import ImageViewer
from dic_app.gui.filter_panel import FilterPanelWidget
from dic_app.gui.alignment_panel import AlignmentPanelWidget
from dic_app.gui.roi_editor import ROIEditorWidget
from dic_app.gui.params_panel import ParamsPanelWidget
from dic_app.gui.results_viewer import ResultsViewerWidget
from dic_app.gui.report_dialog import ReportDialog

from dic_app.core.dic_engine import DICEngine, DICParameters, DICResult
from dic_app.core.mask_manager import MaskManager
from dic_app.io.image_loader import ImageLoader, ImageData
from dic_app.io.project_manager import ProjectState, ProjectManager
from dic_app.io.report_generator import ReportGenerator
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)


class DICWorker(QObject):
    """Worker for running DIC analysis in a background thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, engine, ref_image, def_image, mask=None):
        super().__init__()
        self.engine = engine
        self.ref_image = ref_image
        self.def_image = def_image
        self.mask = mask

    def run(self):
        try:
            self.engine.set_progress_callback(
                lambda p, m: self.progress.emit(p, m))
            result = self.engine.run(self.ref_image, self.def_image, self.mask)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ReportWorker(QObject):
    """Worker for generating reports in background."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def run(self):
        try:
            self.generator.set_progress_callback(
                lambda p, m: self.progress.emit(p, m))
            self.generator.generate_all()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class DICMainWindow(QMainWindow):
    """Main application window with 7-tab workflow.

    Tabs:
    0: Progetto - Image loading, GPS, GSD
    1: Preprocessing - Filter pipeline with preview
    2: Maschere - ROI/mask drawing
    3: Parametri DIC - Algorithm selection and parameters
    4: Analisi - Run DIC with progress
    5: Risultati - Post-processing and visualization
    6: Report - Export reports
    """

    def __init__(self):
        super().__init__()
        self.project = ProjectState()
        self._worker_thread: QThread = None
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

    def _setup_ui(self):
        """Build the main UI with tabbed workflow."""
        self.setWindowTitle("DIC Landslide Monitor v1.0")
        self.resize(1400, 900)

        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # --- Tab 0: Project Setup ---
        self.tab_project = self._create_project_tab()
        self.tabs.addTab(self.tab_project, "1. Progetto")

        # --- Tab 1: Preprocessing ---
        self.filter_panel = FilterPanelWidget()
        self.tabs.addTab(self.filter_panel, "2. Preprocessing")

        # --- Tab 2: Alignment ---
        self.alignment_panel = AlignmentPanelWidget()
        self.tabs.addTab(self.alignment_panel, "3. Allineamento")

        # --- Tab 3: ROI / Masks ---
        self.roi_editor = ROIEditorWidget()
        self.tabs.addTab(self.roi_editor, "4. Maschere")

        # --- Tab 4: DIC Parameters ---
        self.params_panel = ParamsPanelWidget()
        self.tabs.addTab(self.params_panel, "5. Parametri DIC")

        # --- Tab 5: Analysis ---
        self.tab_analysis = self._create_analysis_tab()
        self.tabs.addTab(self.tab_analysis, "6. Analisi")

        # --- Tab 6: Results ---
        self.results_viewer = ResultsViewerWidget()
        self.tabs.addTab(self.results_viewer, "7. Risultati")

        # --- Tab 7: Report ---
        self.tab_report = self._create_report_tab()
        self.tabs.addTab(self.tab_report, "8. Report")

        main_layout.addWidget(self.tabs)

    # ------------------------------------------------------------------
    # Tab 0: Project Setup
    # ------------------------------------------------------------------

    def _create_project_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left: Image list
        left = QVBoxLayout()

        img_group = QGroupBox("Immagini")
        img_layout = QVBoxLayout(img_group)

        btn_row = QHBoxLayout()
        self.btn_add_images = QPushButton("Aggiungi Immagini")
        self.btn_add_images.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 8px; }")
        self.btn_add_images.clicked.connect(self._add_images)
        btn_row.addWidget(self.btn_add_images)

        self.btn_remove_image = QPushButton("Rimuovi")
        self.btn_remove_image.clicked.connect(self._remove_image)
        btn_row.addWidget(self.btn_remove_image)
        img_layout.addLayout(btn_row)

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        img_layout.addWidget(self.image_list)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Immagine di Riferimento:"))
        self.ref_combo = QComboBox()
        self.ref_combo.currentIndexChanged.connect(self._on_reference_changed)
        ref_row.addWidget(self.ref_combo)
        img_layout.addLayout(ref_row)

        left.addWidget(img_group)

        # GSD settings
        gsd_group = QGroupBox("Ground Sampling Distance (GSD)")
        gsd_layout = QFormLayout(gsd_group)

        self.gsd_spin = QDoubleSpinBox()
        self.gsd_spin.setRange(0.0001, 10.0)
        self.gsd_spin.setDecimals(4)
        self.gsd_spin.setSingleStep(0.001)
        self.gsd_spin.setValue(0.01)
        self.gsd_spin.setSuffix(" m/px")
        self.gsd_spin.valueChanged.connect(self._on_gsd_changed)
        gsd_layout.addRow("GSD manuale:", self.gsd_spin)

        calc_row = QHBoxLayout()
        self.altitude_spin = QDoubleSpinBox()
        self.altitude_spin.setRange(1, 10000)
        self.altitude_spin.setValue(100)
        self.altitude_spin.setSuffix(" m")
        calc_row.addWidget(QLabel("Altitudine:"))
        calc_row.addWidget(self.altitude_spin)

        self.sensor_spin = QDoubleSpinBox()
        self.sensor_spin.setRange(1, 100)
        self.sensor_spin.setValue(13.2)
        self.sensor_spin.setSuffix(" mm")
        calc_row.addWidget(QLabel("Sensore:"))
        calc_row.addWidget(self.sensor_spin)

        self.btn_calc_gsd = QPushButton("Calcola GSD")
        self.btn_calc_gsd.clicked.connect(self._calculate_gsd)
        calc_row.addWidget(self.btn_calc_gsd)
        gsd_layout.addRow(calc_row)

        left.addWidget(gsd_group)
        layout.addLayout(left, stretch=1)

        # Right: Image preview and info
        right = QVBoxLayout()

        preview_group = QGroupBox("Anteprima")
        preview_layout = QVBoxLayout(preview_group)
        self.project_viewer = ImageViewer()
        preview_layout.addWidget(self.project_viewer)
        right.addWidget(preview_group, stretch=3)

        info_group = QGroupBox("Informazioni Immagine")
        info_layout = QVBoxLayout(info_group)
        self.image_info_text = QTextEdit()
        self.image_info_text.setReadOnly(True)
        self.image_info_text.setMaximumHeight(150)
        self.image_info_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        info_layout.addWidget(self.image_info_text)
        right.addWidget(info_group)

        layout.addLayout(right, stretch=2)

        return widget

    # ------------------------------------------------------------------
    # Tab 4: Analysis
    # ------------------------------------------------------------------

    def _create_analysis_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Summary before run
        summary_group = QGroupBox("Riepilogo Pre-Analisi")
        summary_layout = QVBoxLayout(summary_group)
        self.analysis_summary = QTextEdit()
        self.analysis_summary.setReadOnly(True)
        self.analysis_summary.setMaximumHeight(200)
        self.analysis_summary.setStyleSheet("font-family: monospace;")
        summary_layout.addWidget(self.analysis_summary)
        layout.addWidget(summary_group)

        # Run controls
        run_group = QGroupBox("Esecuzione Analisi")
        run_layout = QVBoxLayout(run_group)

        self.deformed_combo = QComboBox()
        run_combo_row = QHBoxLayout()
        run_combo_row.addWidget(QLabel("Immagine deformata:"))
        run_combo_row.addWidget(self.deformed_combo, stretch=1)
        run_layout.addLayout(run_combo_row)

        btn_run_row = QHBoxLayout()
        self.btn_run = QPushButton("AVVIA ANALISI DIC")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 15px 30px; font-size: 16px; font-weight: bold; }")
        self.btn_run.clicked.connect(self._run_analysis)
        btn_run_row.addWidget(self.btn_run)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "padding: 15px 20px; font-size: 16px; }")
        self.btn_stop.clicked.connect(self._stop_analysis)
        self.btn_stop.setEnabled(False)
        btn_run_row.addWidget(self.btn_stop)
        run_layout.addLayout(btn_run_row)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(30)
        run_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Pronto")
        self.progress_label.setStyleSheet("font-size: 12px;")
        run_layout.addWidget(self.progress_label)

        layout.addWidget(run_group)

        # Analysis log
        log_group = QGroupBox("Log Analisi")
        log_layout = QVBoxLayout(log_group)
        self.analysis_log = QTextEdit()
        self.analysis_log.setReadOnly(True)
        self.analysis_log.setStyleSheet(
            "font-family: monospace; font-size: 11px; background: #1e1e1e; color: #e0e0e0;")
        log_layout.addWidget(self.analysis_log)
        layout.addWidget(log_group, stretch=1)

        return widget

    # ------------------------------------------------------------------
    # Tab 6: Report
    # ------------------------------------------------------------------

    def _create_report_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_label = QLabel(
            "Genera un report completo dell'analisi DIC con mappe, "
            "statistiche e dati esportati.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 13px; padding: 10px;")
        layout.addWidget(info_label)

        self.btn_open_report_dialog = QPushButton("Configura e Genera Report")
        self.btn_open_report_dialog.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 15px; font-size: 16px; font-weight: bold; }")
        self.btn_open_report_dialog.clicked.connect(self._open_report_dialog)
        layout.addWidget(self.btn_open_report_dialog)

        self.report_progress = QProgressBar()
        self.report_progress.setVisible(False)
        layout.addWidget(self.report_progress)

        self.report_status = QLabel()
        layout.addWidget(self.report_status)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # Menu, toolbar, statusbar
    # ------------------------------------------------------------------

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        self.act_new = QAction("Nuovo Progetto", self)
        self.act_new.triggered.connect(self._new_project)
        file_menu.addAction(self.act_new)

        self.act_open = QAction("Apri Progetto...", self)
        self.act_open.triggered.connect(self._open_project)
        file_menu.addAction(self.act_open)

        self.act_save = QAction("Salva Progetto", self)
        self.act_save.triggered.connect(self._save_project)
        file_menu.addAction(self.act_save)

        self.act_save_as = QAction("Salva Progetto Come...", self)
        self.act_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(self.act_save_as)

        file_menu.addSeparator()

        self.act_exit = QAction("Esci", self)
        self.act_exit.triggered.connect(self.close)
        file_menu.addAction(self.act_exit)

        # Help menu
        help_menu = menubar.addMenu("Aiuto")
        self.act_about = QAction("Informazioni", self)
        self.act_about.triggered.connect(self._show_about)
        help_menu.addAction(self.act_about)

    def _setup_toolbar(self):
        toolbar = self.addToolBar("Principale")
        toolbar.setMovable(False)

        btn_add = QPushButton("Apri Immagini")
        btn_add.clicked.connect(self._add_images)
        toolbar.addWidget(btn_add)

        toolbar.addSeparator()

        btn_run = QPushButton("Avvia Analisi")
        btn_run.clicked.connect(self._run_analysis)
        toolbar.addWidget(btn_run)

        toolbar.addSeparator()

        btn_report = QPushButton("Genera Report")
        btn_report.clicked.connect(self._open_report_dialog)
        toolbar.addWidget(btn_report)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.status_coords = QLabel("X: - Y: - Val: -")
        self.statusbar.addPermanentWidget(self.status_coords)

    def _connect_signals(self):
        """Connect inter-panel signals."""
        self.filter_panel.filter_applied.connect(self._on_filter_applied)
        self.alignment_panel.alignment_applied.connect(self._on_alignment_applied)
        self.alignment_panel.alignment_reset.connect(self._on_alignment_reset)
        self.roi_editor.mask_updated.connect(self._on_mask_updated)
        self.params_panel.params_changed.connect(self._on_params_changed)
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _add_images(self):
        """Open file dialog and load images."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Seleziona Immagini", "",
            "Immagini (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;Tutti (*)")

        if not filepaths:
            return

        for fp in filepaths:
            try:
                img_data = ImageLoader.load(fp)
                self.project.images_data.append(img_data)
                self.project.image_paths.append(fp)

                # Add to list widget
                name = img_data.filename
                if img_data.has_gps:
                    name += f"  ({img_data.gps.latitude:.5f}, {img_data.gps.longitude:.5f})"
                self.image_list.addItem(name)

                # Add to reference combo and deformed combo
                self.ref_combo.addItem(img_data.filename)
                self.deformed_combo.addItem(img_data.filename)

            except Exception as e:
                QMessageBox.warning(self, "Errore",
                                    f"Impossibile caricare {fp}:\n{e}")

        if self.project.images_data:
            self.image_list.setCurrentRow(0)
            self._update_status(f"{len(self.project.images_data)} immagini caricate")

    def _remove_image(self):
        """Remove selected image."""
        row = self.image_list.currentRow()
        if row < 0:
            return
        self.image_list.takeItem(row)
        self.project.images_data.pop(row)
        self.project.image_paths.pop(row)
        self.ref_combo.removeItem(row)
        self.deformed_combo.removeItem(row)

    def _on_image_selected(self, row):
        """Preview selected image and show info."""
        if row < 0 or row >= len(self.project.images_data):
            return

        img = self.project.images_data[row]
        self.project_viewer.set_image(img.image_rgb)
        self.project_viewer.fit_in_view()

        # Info text
        info = f"File: {img.filename}\n"
        info += f"Dimensioni: {img.width} x {img.height} pixel\n"
        if img.camera_model:
            info += f"Camera: {img.camera_model}\n"
        if img.focal_length_mm:
            info += f"Lunghezza focale: {img.focal_length_mm} mm\n"
        if img.capture_time:
            info += f"Data scatto: {img.capture_time}\n"
        if img.has_gps:
            info += f"\nGPS:\n"
            info += f"  Latitudine:  {img.gps.latitude:.7f}\n"
            info += f"  Longitudine: {img.gps.longitude:.7f}\n"
            if img.gps.altitude:
                info += f"  Altitudine:  {img.gps.altitude:.1f} m\n"
            if img.gps.timestamp:
                info += f"  Timestamp:   {img.gps.timestamp}\n"
        else:
            info += "\nGPS: non disponibile\n"

        self.image_info_text.setText(info)

    def _on_reference_changed(self, index):
        """Update reference image index."""
        if 0 <= index < len(self.project.images_data):
            self.project.reference_index = index

    def _calculate_gsd(self):
        """Calculate GSD from flight parameters."""
        ref_idx = self.project.reference_index
        if ref_idx < len(self.project.images_data):
            img = self.project.images_data[ref_idx]
            fl = img.focal_length_mm or 24.0
            altitude = self.altitude_spin.value()
            sensor_w = self.sensor_spin.value()

            gsd = ImageLoader.compute_gsd(
                altitude, fl, sensor_w, img.width)
            self.gsd_spin.setValue(gsd)
            self.project.gsd = gsd
            self._update_status(f"GSD calcolato: {gsd:.4f} m/px ({gsd * 100:.2f} cm/px)")

    def _on_gsd_changed(self, value):
        self.project.gsd = value

    # ------------------------------------------------------------------
    # Filter pipeline
    # ------------------------------------------------------------------

    def _on_filter_applied(self, pipeline):
        """Apply filter pipeline to all images.

        If alignment exists, re-applies the warp + crop to the newly
        preprocessed images so that aligned_ref and aligned_def stay
        in sync with the updated preprocessing.
        """
        self.project.filter_pipeline = pipeline
        self.project.preprocessed_images = []

        for img_data in self.project.images_data:
            processed = pipeline.apply(img_data.image_gray)
            self.project.preprocessed_images.append(processed)

        n = len(self.project.preprocessed_images)

        # Re-apply alignment if it exists, using newly preprocessed images
        if (self.project.alignment_result is not None
                and len(self.project.preprocessed_images) >= 2):
            try:
                self._reapply_alignment()
                self._update_status(
                    f"Pipeline applicata a {n} immagini + allineamento riapplicato")
            except Exception as e:
                logger.warning(f"Failed to re-apply alignment after preprocessing: {e}")
                self._update_status(f"Pipeline applicata a {n} immagini "
                                    f"(ATTENZIONE: riallineamento fallito)")
        else:
            self._update_status(f"Pipeline applicata a {n} immagini")

        QMessageBox.information(self, "Preprocessing",
                                f"Pipeline applicata con successo a {n} immagini")

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _on_alignment_applied(self, result):
        """Handle alignment result from alignment panel."""
        from dic_app.core.image_registration import ImageRegistration
        self.project.alignment_result = result
        self.project.alignment_params = self.alignment_panel.get_parameters()

        aligned = self.alignment_panel.get_aligned_images()
        if aligned:
            self.project.aligned_ref, self.project.aligned_def = aligned

            # Track which deformed image was aligned
            for i in range(len(self.project.images_data)):
                if i != self.project.reference_index:
                    self.project.aligned_def_index = i
                    break

            ref_shape = self.project.aligned_ref.shape
            self._update_status(
                f"Allineamento completato – area valida {ref_shape[1]}×{ref_shape[0]} px, "
                f"RMSE={result.reprojection_error:.3f} px")
        else:
            self._update_status("Allineamento completato")

    def _on_alignment_reset(self):
        """Clear alignment data."""
        self.project.alignment_result = None
        self.project.alignment_params = None
        self.project.aligned_ref = None
        self.project.aligned_def = None
        self.project.aligned_def_index = None
        self._update_status("Allineamento resettato")

    def _reapply_alignment(self):
        """Re-apply existing alignment transform to current preprocessed images.

        Called when preprocessing changes after alignment has been computed.
        Uses the stored transform matrix and crop bbox to warp+crop the
        freshly preprocessed images, keeping aligned_ref and aligned_def
        in sync with the current preprocessing state.
        """
        from dic_app.core.image_registration import ImageRegistration

        result = self.project.alignment_result
        params = self.project.alignment_params
        if result is None:
            return

        ref_img = self.project.get_pre_alignment_ref()
        def_img = self.project.get_pre_alignment_def()
        if ref_img is None or def_img is None:
            return

        reg = ImageRegistration(params)
        ref_crop, def_crop = reg.apply_to_pair(ref_img, def_img, result)

        self.project.aligned_ref = ref_crop
        self.project.aligned_def = def_crop

        logger.info(
            f"Alignment re-applied after preprocessing: "
            f"ref={ref_crop.shape}, def={def_crop.shape}")

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------

    def _on_mask_updated(self, mask_manager):
        self.project.mask_manager = mask_manager

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _on_params_changed(self, params):
        self.project.dic_params = params

    # ------------------------------------------------------------------
    # Tab change
    # ------------------------------------------------------------------

    def _on_tab_changed(self, index):
        """Update tab contents when switching."""
        if index == 1:  # Preprocessing
            ref_raw = self.project.get_pre_alignment_ref()
            if ref_raw is None and self.project.images_data:
                ref_raw = self.project.images_data[
                    self.project.reference_index].image_gray
            if ref_raw is not None:
                self.filter_panel.set_reference_image(ref_raw)

            # Pass ALL grayscale images for multi-image auto-optimization
            if self.project.images_data:
                all_grays = [img.image_gray for img in self.project.images_data
                             if img.image_gray is not None]
                self.filter_panel.set_all_images(all_grays)

        elif index == 2:  # Alignment
            ref = self.project.get_pre_alignment_ref()
            def_img = self.project.get_pre_alignment_def()
            if ref is not None and def_img is not None:
                self.alignment_panel.set_images(ref, def_img)

        elif index == 3:  # Masks
            ref = self.project.get_reference_image()
            if ref is not None:
                self.roi_editor.set_image(ref)
                if self.project.mask_manager:
                    self.roi_editor.set_mask_manager(self.project.mask_manager)

        elif index == 4:  # Parameters
            ref = self.project.get_reference_image()
            if ref is not None:
                self.params_panel.update_estimation(ref.shape)
            elif self.project.images_data:
                shape = self.project.images_data[0].image_gray.shape
                self.params_panel.update_estimation(shape)

        elif index == 5:  # Analysis
            self._update_analysis_summary()

    # ------------------------------------------------------------------
    # DIC Analysis
    # ------------------------------------------------------------------

    def _update_analysis_summary(self):
        """Update pre-analysis summary."""
        summary = ""
        n_images = len(self.project.images_data)
        summary += f"Immagini caricate: {n_images}\n"

        if n_images > 0:
            ref = self.project.images_data[self.project.reference_index]
            summary += f"Riferimento: {ref.filename} ({ref.width}x{ref.height})\n"

        n_pre = len(self.project.preprocessed_images)
        summary += f"Immagini preprocessate: {n_pre}\n"

        if self.project.mask_manager:
            summary += f"ROI definite: {len(self.project.mask_manager)}\n"
        else:
            summary += "Maschere: nessuna (analisi su tutta l'immagine)\n"

        params = self.project.dic_params or self.params_panel.get_parameters()
        summary += f"\nMetodo: {params.method.value}\n"
        summary += f"Subset: {params.subset_size} px\n"
        summary += f"Passo griglia: {params.step_size} px\n"
        summary += f"Soglia correlazione: {params.correlation_threshold}\n"

        if self.project.gsd:
            summary += f"\nGSD: {self.project.gsd:.4f} m/px\n"

        self.analysis_summary.setText(summary)

        # Update deformed combo
        self.deformed_combo.clear()
        for i, img in enumerate(self.project.images_data):
            if i != self.project.reference_index:
                self.deformed_combo.addItem(img.filename, i)

    def _run_analysis(self):
        """Start DIC analysis in background thread."""
        if len(self.project.images_data) < 2:
            QMessageBox.warning(self, "Errore",
                                "Servono almeno 2 immagini (riferimento + deformata)")
            return

        # Get images
        ref_image = self.project.get_reference_image()
        def_idx = self.deformed_combo.currentData()
        if def_idx is None:
            # Use first non-reference image
            for i in range(len(self.project.images_data)):
                if i != self.project.reference_index:
                    def_idx = i
                    break

        if def_idx is None:
            QMessageBox.warning(self, "Errore", "Nessuna immagine deformata selezionata")
            return

        def_image = self.project.get_deformed_image(def_idx)

        if ref_image is None or def_image is None:
            QMessageBox.warning(self, "Errore", "Impossibile caricare le immagini")
            return

        # Ensure ref and def images have the same dimensions.
        # After alignment+crop they should match, but mismatches can
        # occur with different source resolutions or rounding in crop.
        import cv2 as _cv2
        if ref_image.shape != def_image.shape:
            logger.warning(
                f"Dimensioni diverse: ref={ref_image.shape}, "
                f"def={def_image.shape}. Crop al minimo comune.")
            h = min(ref_image.shape[0], def_image.shape[0])
            w = min(ref_image.shape[1], def_image.shape[1])
            ref_image = ref_image[:h, :w].copy()
            def_image = def_image[:h, :w].copy()

        # Get mask
        mask = None
        if self.project.mask_manager:
            mask = self.project.mask_manager.generate_mask()
            # Resize mask to match image if needed
            if mask.shape != ref_image.shape:
                mask = _cv2.resize(mask, (ref_image.shape[1], ref_image.shape[0]))

        # Get parameters
        params = self.project.dic_params or self.params_panel.get_parameters()

        # Auto-enforce sensible border_margin when alignment was used.
        # Aligned images have black borders from the warp; any subset
        # that overlaps them will produce fictitious large displacement.
        if self.project.aligned_ref is not None:
            min_margin = max(params.border_margin, params.subset_size)
            if params.border_margin < min_margin:
                logger.info(
                    f"Allineamento attivo: border_margin alzato da "
                    f"{params.border_margin} a {min_margin} px")
                from dataclasses import replace as _dc_replace
                params = _dc_replace(params, border_margin=min_margin)

        self.project.dic_params = params

        # Validate texture vs subset size (C5)
        tex_check = DICEngine.validate_texture(
            ref_image, params.subset_size, mask)
        if tex_check['warning']:
            if not tex_check['ok']:
                # Severe warning — ask user before proceeding
                reply = QMessageBox.warning(
                    self, "Texture insufficiente",
                    tex_check['warning'] + "\n\nProcedere comunque?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return
            else:
                # Mild warning — inform but proceed
                QMessageBox.information(
                    self, "Nota sulla texture", tex_check['warning'])

        # Setup engine and worker
        engine = DICEngine(params)

        self._worker_thread = QThread()
        self._worker = DICWorker(engine, ref_image, def_image, mask)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_analysis_progress)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)

        # UI state
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Analisi in corso...")
        self.analysis_log.clear()
        self.analysis_log.append("Avvio analisi DIC...\n")

        self._worker_thread.start()

    def _stop_analysis(self):
        """Request cancellation."""
        if hasattr(self, '_worker') and self._worker:
            self._worker.engine.cancel()
            self.progress_label.setText("Annullamento in corso...")

    def _on_analysis_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)
        self.analysis_log.append(f"[{percent:3d}%] {message}")
        QApplication.processEvents()

    def _on_analysis_finished(self, result: DICResult):
        """Handle analysis completion."""
        self.project.results = [result]
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText(
            f"Analisi completata in {result.computation_time_s:.1f} secondi")
        self.analysis_log.append(
            f"\nAnalisi completata in {result.computation_time_s:.1f}s")
        self.analysis_log.append(
            f"Punti validi: {int(np.sum(result.mask_valid))} / {result.mask_valid.size}")

        # Load results into viewer
        base_img = self.project.get_reference_rgb()
        if base_img is None:
            base_img = self.project.get_reference_image()
        self.results_viewer.set_result(result, base_img, self.project.gsd)

        # Pass ref/def images to results viewer for area inspection (gray + RGB)
        ref_image = self.project.get_reference_image()
        def_idx = self.deformed_combo.currentData()
        def_image = self.project.get_deformed_image(def_idx)
        ref_rgb = self.project.get_reference_rgb()
        def_rgb = self.project.get_deformed_rgb(def_idx)
        if ref_image is not None and def_image is not None:
            self.results_viewer.set_dic_images(
                ref_image, def_image, ref_rgb, def_rgb)

        # Auto-switch to results tab (index 6 with alignment tab)
        self.tabs.setCurrentIndex(6)
        self._update_status("Analisi completata - visualizzazione risultati")

    def _on_analysis_error(self, error_msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText(f"Errore: {error_msg}")
        self.analysis_log.append(f"\nERRORE: {error_msg}")
        QMessageBox.critical(self, "Errore Analisi", error_msg)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _open_report_dialog(self):
        """Open report configuration dialog."""
        if not self.project.results:
            QMessageBox.warning(self, "Errore",
                                "Nessun risultato disponibile. Esegui prima l'analisi.")
            return

        dialog = ReportDialog(self)
        dialog.analyst_edit.setText(self.project.analyst_name)
        dialog.site_edit.setText(self.project.site_name)
        dialog.report_requested.connect(self._generate_report)
        dialog.exec()

    def _generate_report(self, config):
        """Generate report with given configuration."""
        result = self.project.results[0]

        # Gather pipeline info
        pipeline_info = ""
        if self.project.filter_pipeline:
            pipeline_info = str(self.project.filter_pipeline)

        # Gather strain data from results viewer
        strain_data = self.results_viewer.get_strain_data()
        active_zones = self.results_viewer.get_active_zones()

        base_img = self.project.get_reference_rgb()
        if base_img is None:
            base_img = self.project.get_reference_image()

        # Retrieve deformed images for zone detail pages
        def_idx = self.deformed_combo.currentData()
        deformed_gray = self.project.get_deformed_image(def_idx)
        deformed_rgb = self.project.get_deformed_rgb(def_idx)

        generator = ReportGenerator(
            result=result,
            config=config,
            base_image=base_img,
            deformed_image=deformed_gray,
            deformed_image_rgb=deformed_rgb,
            strain_data=strain_data,
            active_zones=active_zones,
            pipeline_info=pipeline_info,
            gsd=self.project.gsd,
            gps_info=self.project.get_gps_info()
        )

        # Run in main thread (matplotlib is NOT thread-safe)
        self.report_progress.setVisible(True)
        self.report_progress.setValue(0)
        self.report_status.setText("Generazione report in corso...")
        QApplication.processEvents()

        def on_progress(pct, msg):
            self.report_progress.setValue(pct)
            self.report_status.setText(msg)
            QApplication.processEvents()

        generator.set_progress_callback(on_progress)

        try:
            generator.generate_all()
            self._on_report_finished()
        except Exception as e:
            self._on_report_error(str(e))

    def _on_report_progress(self, percent, msg):
        self.report_progress.setValue(percent)
        self.report_status.setText(msg)

    def _on_report_finished(self):
        self.report_progress.setVisible(False)
        self.report_status.setText("Report generato con successo!")
        QMessageBox.information(self, "Report",
                                "Report generato con successo!")

    def _on_report_error(self, error_msg):
        self.report_progress.setVisible(False)
        self.report_status.setText(f"Errore: {error_msg}")
        QMessageBox.critical(self, "Errore Report", error_msg)

    # ------------------------------------------------------------------
    # Project save/load
    # ------------------------------------------------------------------

    def _new_project(self):
        self.project = ProjectState()
        self.image_list.clear()
        self.ref_combo.clear()
        self.deformed_combo.clear()
        self.image_info_text.clear()
        self.analysis_log.clear()
        self.setWindowTitle("DIC Landslide Monitor v1.0 - Nuovo Progetto")

    def _open_project(self):
        dirpath = QFileDialog.getExistingDirectory(
            self, "Apri Progetto .dicproj")
        if not dirpath:
            return
        try:
            self.project = ProjectManager.load(dirpath)
            self._restore_project_ui()
            self.setWindowTitle(
                f"DIC Landslide Monitor v1.0 - {self.project.project_name}")
            self._update_status(f"Progetto caricato: {self.project.project_name}")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile aprire il progetto:\n{e}")

    def _save_project(self):
        if self.project.project_filepath:
            self._do_save(self.project.project_filepath)
        else:
            self._save_project_as()

    def _save_project_as(self):
        dirpath = QFileDialog.getExistingDirectory(
            self, "Salva Progetto Come...")
        if dirpath:
            project_dir = os.path.join(dirpath, self.project.project_name)
            self._do_save(project_dir)

    def _do_save(self, filepath):
        try:
            # Sync current params
            self.project.dic_params = self.params_panel.get_parameters()
            ProjectManager.save(self.project, filepath)
            self._update_status(f"Progetto salvato: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile salvare:\n{e}")

    def _restore_project_ui(self):
        """Restore UI state from loaded project."""
        self.image_list.clear()
        self.ref_combo.clear()
        self.deformed_combo.clear()

        for img in self.project.images_data:
            name = img.filename
            if img.has_gps:
                name += f"  ({img.gps.latitude:.5f}, {img.gps.longitude:.5f})"
            self.image_list.addItem(name)
            self.ref_combo.addItem(img.filename)
            self.deformed_combo.addItem(img.filename)

        if self.project.reference_index < self.ref_combo.count():
            self.ref_combo.setCurrentIndex(self.project.reference_index)

        if self.project.gsd:
            self.gsd_spin.setValue(self.project.gsd)

        if self.project.filter_pipeline:
            self.filter_panel.set_pipeline(self.project.filter_pipeline)

        if self.project.mask_manager:
            self.roi_editor.set_mask_manager(self.project.mask_manager)

        if self.project.dic_params:
            self.params_panel.set_parameters(self.project.dic_params)

        if self.project.results:
            base_img = self.project.get_reference_rgb()
            self.results_viewer.set_result(
                self.project.results[0], base_img, self.project.gsd)
            # Pass ref/def images for area inspection tool (gray + RGB)
            ref_image = self.project.get_reference_image()
            def_image = self.project.get_deformed_image()
            ref_rgb = self.project.get_reference_rgb()
            def_rgb = self.project.get_deformed_rgb()
            if ref_image is not None and def_image is not None:
                self.results_viewer.set_dic_images(
                    ref_image, def_image, ref_rgb, def_rgb)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_status(self, message):
        self.statusbar.showMessage(message, 5000)

    def _show_about(self):
        QMessageBox.about(self, "Informazioni",
                          "DIC Landslide Monitor v1.0\n\n"
                          "Digital Image Correlation per il monitoraggio\n"
                          "di frane e crolli da immagini aeree e terrestri.\n\n"
                          "Algoritmi: Template NCC, Optical Flow,\n"
                          "Phase Correlation, Feature Matching\n\n"
                          "Filtri: CLAHE, Wallis, Bilateral, NLM,\n"
                          "Unsharp Mask, e altri 10+")
