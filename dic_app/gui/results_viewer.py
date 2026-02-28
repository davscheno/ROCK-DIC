"""Results visualization panel with interactive heatmaps and vector fields."""

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QSlider, QSplitter, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QFormLayout, QDialog,
    QDialogButtonBox, QFileDialog, QMessageBox, QTabWidget,
    QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from dic_app.gui.image_viewer import ImageViewer
from dic_app.core.dic_engine import DICResult
from dic_app.core.postprocessing import (
    DisplacementSmoother, StrainCalculator, DisplacementStatistics
)
from dic_app.utils.helpers import (
    displacement_colormap, overlay_heatmap, compute_magnitude
)


class ResultsViewerWidget(QWidget):
    """Post-processing and visualization panel.

    Features:
    - Display displacement/strain heatmaps overlaid on image
    - Vector field overlay
    - Interactive point inspection
    - Smoothing controls
    - Strain computation
    - Active zone detection
    """

    zones_detected = pyqtSignal(list)  # emits list of active zone dicts

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: DICResult = None
        self._strain_data: dict = None
        self._base_image: np.ndarray = None
        self._gsd: float = None
        self._smoothed_u = None
        self._smoothed_v = None
        self._active_zones = []

        # Rectangle selection state
        self._selecting = False        # True when selection mode is active
        self._sel_start = None         # (x, y) of drag start
        self._sel_rect = None          # (x0, y0, x1, y1) current/final selection
        self._heatmap_overlay = None   # Cached heatmap overlay (before selection rect)

        # DIC source images (set externally by main_window)
        self._ref_image = None         # Reference image used for DIC (grayscale)
        self._def_image = None         # Deformed image used for DIC (grayscale)
        self._ref_image_rgb = None     # Reference image RGB for color extraction
        self._def_image_rgb = None     # Deformed image RGB for color extraction

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Left: Image viewer ---
        left_panel = QVBoxLayout()
        self.viewer = ImageViewer()
        self.viewer.mouse_clicked.connect(self._on_point_clicked)
        self.viewer.mouse_moved.connect(self._on_mouse_moved)
        left_panel.addWidget(self.viewer, stretch=4)

        # Point info panel
        self.info_label = QLabel("Clicca su un punto per visualizzare i dati")
        self.info_label.setStyleSheet(
            "background-color: #e8eaf6; color: #212121; padding: 8px; "
            "font-family: monospace; font-size: 11px;")
        self.info_label.setMinimumHeight(80)
        self.info_label.setWordWrap(True)
        left_panel.addWidget(self.info_label)

        # Connect drag signals for rectangle selection
        self.viewer.drag_started.connect(self._on_drag_started)
        self.viewer.drag_moved.connect(self._on_drag_moved)
        self.viewer.drag_finished.connect(self._on_drag_finished)

        main_layout.addLayout(left_panel, stretch=3)

        # --- Right: Controls ---
        right_panel = QVBoxLayout()

        # Display options
        display_group = QGroupBox("Visualizzazione")
        display_layout = QFormLayout(display_group)

        self.field_combo = QComboBox()
        self.field_combo.addItem("Magnitudine Spostamento", "magnitude")
        self.field_combo.addItem("Spostamento U (orizzontale)", "u")
        self.field_combo.addItem("Spostamento V (verticale)", "v")
        self.field_combo.addItem("Qualita Correlazione", "quality")
        self.field_combo.currentIndexChanged.connect(self._update_display)
        display_layout.addRow("Campo:", self.field_combo)

        self.cmap_combo = QComboBox()
        for cm in ['jet', 'viridis', 'hot', 'coolwarm', 'RdYlGn', 'plasma',
                    'inferno', 'turbo', 'seismic']:
            self.cmap_combo.addItem(cm)
        self.cmap_combo.currentIndexChanged.connect(self._update_display)
        display_layout.addRow("Colormap:", self.cmap_combo)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self._update_display)
        display_layout.addRow("Opacita Overlay:", self.alpha_slider)

        range_row = QHBoxLayout()
        self.auto_range = QCheckBox("Auto")
        self.auto_range.setChecked(True)
        self.auto_range.toggled.connect(self._update_display)
        range_row.addWidget(self.auto_range)
        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1000, 1000)
        self.vmin_spin.setValue(0)
        self.vmin_spin.setDecimals(2)
        self.vmin_spin.setEnabled(False)
        range_row.addWidget(QLabel("Min:"))
        range_row.addWidget(self.vmin_spin)
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1000, 1000)
        self.vmax_spin.setValue(10)
        self.vmax_spin.setDecimals(2)
        self.vmax_spin.setEnabled(False)
        range_row.addWidget(QLabel("Max:"))
        range_row.addWidget(self.vmax_spin)
        display_layout.addRow("Range:", range_row)
        self.auto_range.toggled.connect(
            lambda c: (self.vmin_spin.setEnabled(not c),
                       self.vmax_spin.setEnabled(not c)))

        self.show_vectors = QCheckBox("Mostra Vettori")
        self.show_vectors.setChecked(False)
        self.show_vectors.toggled.connect(self._update_display)
        display_layout.addRow(self.show_vectors)

        self.vector_step = QSpinBox()
        self.vector_step.setRange(1, 20)
        self.vector_step.setValue(3)
        self.vector_step.valueChanged.connect(self._update_display)
        display_layout.addRow("Passo Vettori:", self.vector_step)

        self.vector_scale = QDoubleSpinBox()
        self.vector_scale.setRange(0.1, 50.0)
        self.vector_scale.setValue(3.0)
        self.vector_scale.setSingleStep(0.5)
        self.vector_scale.valueChanged.connect(self._update_display)
        display_layout.addRow("Scala Vettori:", self.vector_scale)

        right_panel.addWidget(display_group)

        # Area selection / inspection tool
        sel_group = QGroupBox("Ispezione Area")
        sel_layout = QVBoxLayout(sel_group)

        self.btn_select_area = QPushButton("✦  Seleziona Area")
        self.btn_select_area.setCheckable(True)
        self.btn_select_area.setMinimumHeight(34)
        self.btn_select_area.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 12px; }"
            "QPushButton:checked { background-color: #FF5722; color: white; }")
        self.btn_select_area.toggled.connect(self._toggle_selection_mode)
        sel_layout.addWidget(self.btn_select_area)

        self.sel_info_label = QLabel("Traccia un rettangolo sulla mappa per estrarre "
                                     "le porzioni di immagine riferimento e deformata.")
        self.sel_info_label.setWordWrap(True)
        self.sel_info_label.setStyleSheet("font-size: 10px; color: #616161;")
        sel_layout.addWidget(self.sel_info_label)

        right_panel.addWidget(sel_group)

        # Smoothing controls
        smooth_group = QGroupBox("Smoothing Spostamenti")
        smooth_layout = QFormLayout(smooth_group)

        self.smooth_combo = QComboBox()
        self.smooth_combo.addItem("Nessuno", "none")
        self.smooth_combo.addItem("Gaussiano", "gaussian")
        self.smooth_combo.addItem("Spline", "spline")
        self.smooth_combo.addItem("Mediano", "median")
        smooth_layout.addRow("Metodo:", self.smooth_combo)

        self.smooth_sigma = QDoubleSpinBox()
        self.smooth_sigma.setRange(0.5, 20.0)
        self.smooth_sigma.setValue(2.0)
        self.smooth_sigma.setSingleStep(0.5)
        smooth_layout.addRow("Sigma/Fattore:", self.smooth_sigma)

        self.outlier_threshold = QDoubleSpinBox()
        self.outlier_threshold.setRange(1.0, 10.0)
        self.outlier_threshold.setValue(3.0)
        self.outlier_threshold.setSingleStep(0.5)
        self.outlier_threshold.setToolTip(
            "Moltiplicatore NMT: valori più bassi = più aggressivo. "
            "Consigliato: 3.0 (standard) o 2.5 (conservativo).")
        smooth_layout.addRow("Soglia Outlier (NMT):", self.outlier_threshold)

        smooth_btn_row1 = QHBoxLayout()
        self.btn_smooth = QPushButton("Smoothing")
        self.btn_smooth.clicked.connect(self._apply_smoothing)
        smooth_btn_row1.addWidget(self.btn_smooth)

        self.btn_remove_outliers = QPushButton("Rimuovi Outlier")
        self.btn_remove_outliers.clicked.connect(self._remove_outliers)
        self.btn_remove_outliers.setToolTip(
            "Rimuovi outlier con Normalized Median Test (NMT) locale")
        smooth_btn_row1.addWidget(self.btn_remove_outliers)
        smooth_layout.addRow(smooth_btn_row1)

        smooth_btn_row2 = QHBoxLayout()
        self.btn_coherence = QPushButton("Coerenza Spaziale")
        self.btn_coherence.clicked.connect(self._apply_coherence_filter)
        self.btn_coherence.setToolTip(
            "Filtro coerenza spaziale: rimuovi vettori incoerenti "
            "rispetto ai vicini (median vector test)")
        smooth_btn_row2.addWidget(self.btn_coherence)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_smoothing)
        smooth_btn_row2.addWidget(self.btn_reset)
        smooth_layout.addRow(smooth_btn_row2)

        right_panel.addWidget(smooth_group)

        # Strain computation
        strain_group = QGroupBox("Calcolo Strain")
        strain_layout = QFormLayout(strain_group)

        self.btn_compute_strain = QPushButton("Calcola Strain")
        self.btn_compute_strain.clicked.connect(self._compute_strain)
        self.btn_compute_strain.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 6px; }")
        strain_layout.addRow(self.btn_compute_strain)

        self.strain_combo = QComboBox()
        self.strain_combo.addItem("E_xx (Green-Lagrangian)", "E_xx")
        self.strain_combo.addItem("E_yy (Green-Lagrangian)", "E_yy")
        self.strain_combo.addItem("E_xy (Green-Lagrangian)", "E_xy")
        self.strain_combo.addItem("Strain Principale 1", "principal_1")
        self.strain_combo.addItem("Strain Principale 2", "principal_2")
        self.strain_combo.addItem("Taglio Massimo", "max_shear")
        self.strain_combo.addItem("Von Mises", "von_mises")
        self.strain_combo.addItem("eps_xx (Ingegneristico)", "eps_xx")
        self.strain_combo.addItem("eps_yy (Ingegneristico)", "eps_yy")
        self.strain_combo.addItem("gamma_xy (Ingegneristico)", "gamma_xy")
        self.strain_combo.setEnabled(False)
        self.strain_combo.currentIndexChanged.connect(self._update_strain_display)
        strain_layout.addRow("Componente:", self.strain_combo)

        right_panel.addWidget(strain_group)

        # Active zone detection
        zone_group = QGroupBox("Rilevamento Zone Attive")
        zone_layout = QFormLayout(zone_group)

        thresh_row = QHBoxLayout()
        self.zone_threshold = QDoubleSpinBox()
        self.zone_threshold.setRange(0.1, 500.0)
        self.zone_threshold.setValue(3.0)
        self.zone_threshold.setSingleStep(0.5)
        self.zone_threshold.setToolTip(
            "Spostamento minimo in pixel per considerare un punto come "
            "\"attivo\". Tipicamente 3-5x il rumore DIC atteso.")
        thresh_row.addWidget(self.zone_threshold)

        self.btn_auto_threshold = QPushButton("Auto")
        self.btn_auto_threshold.setToolTip(
            "Calcola automaticamente la soglia in base al rumore "
            "stimato del campo di spostamento (SNR + MAD)")
        self.btn_auto_threshold.setFixedWidth(50)
        self.btn_auto_threshold.clicked.connect(self._auto_threshold)
        thresh_row.addWidget(self.btn_auto_threshold)
        zone_layout.addRow("Soglia (px):", thresh_row)

        self.zone_min_area = QSpinBox()
        self.zone_min_area.setRange(1, 10000)
        self.zone_min_area.setValue(50)
        self.zone_min_area.setToolTip(
            "Numero minimo di punti griglia connessi per "
            "considerare una zona come attiva. Valori più alti "
            "riducono i falsi positivi.")
        zone_layout.addRow("Area Min (punti):", self.zone_min_area)

        zone_btn_row = QHBoxLayout()
        self.btn_detect_zones = QPushButton("Rileva Zone")
        self.btn_detect_zones.clicked.connect(self._detect_zones)
        self.btn_detect_zones.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; padding: 6px; }")
        zone_btn_row.addWidget(self.btn_detect_zones)

        self.btn_validate_zones = QPushButton("Valida Zone")
        self.btn_validate_zones.clicked.connect(self._open_zone_validation)
        self.btn_validate_zones.setEnabled(False)
        self.btn_validate_zones.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 6px; }")
        self.btn_validate_zones.setToolTip(
            "Ispeziona e valida/elimina le zone attive una per una")
        zone_btn_row.addWidget(self.btn_validate_zones)
        zone_layout.addRow(zone_btn_row)

        grid_row = QHBoxLayout()
        self.grid_cell_size = QSpinBox()
        self.grid_cell_size.setRange(100, 2000)
        self.grid_cell_size.setValue(350)
        self.grid_cell_size.setSingleStep(50)
        self.grid_cell_size.setSuffix(" px")
        self.grid_cell_size.setToolTip(
            "Dimensione lato della cella di ispezione (pixel)")
        grid_row.addWidget(QLabel("Cella:"))
        grid_row.addWidget(self.grid_cell_size)

        self.btn_grid_inspection = QPushButton("Ispezione Griglia")
        self.btn_grid_inspection.clicked.connect(self._open_grid_inspection)
        self.btn_grid_inspection.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 6px; }")
        self.btn_grid_inspection.setToolTip(
            "Divide l'area in celle e permette di ispezionare e marcare "
            "manualmente ogni cella come deformata o normale")
        grid_row.addWidget(self.btn_grid_inspection)
        zone_layout.addRow(grid_row)

        self.zone_table = QTableWidget(0, 5)
        self.zone_table.setHorizontalHeaderLabels(
            ["ID", "Area", "Max", "Media", "Stato"])
        self.zone_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.zone_table.setMaximumHeight(140)
        self.zone_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.zone_table.setSelectionMode(QTableWidget.SingleSelection)
        self.zone_table.cellDoubleClicked.connect(self._on_zone_double_clicked)
        zone_layout.addRow(self.zone_table)

        right_panel.addWidget(zone_group)

        # Statistics
        stats_group = QGroupBox("Statistiche")
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setStyleSheet(
            "font-family: monospace; font-size: 11px;")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.addWidget(self.stats_text)
        right_panel.addWidget(stats_group)

        # Magnitude histogram with noise/threshold indicators
        hist_group = QGroupBox("Istogramma Magnitudine")
        hist_layout = QVBoxLayout(hist_group)
        hist_layout.setContentsMargins(2, 2, 2, 2)
        self._hist_figure = Figure(figsize=(3.5, 2.0), dpi=80)
        self._hist_figure.patch.set_facecolor('#ffffff')
        self._hist_canvas = FigureCanvas(self._hist_figure)
        self._hist_canvas.setMinimumHeight(140)
        self._hist_canvas.setMaximumHeight(180)
        hist_layout.addWidget(self._hist_canvas)
        right_panel.addWidget(hist_group)

        # Quality indicator
        self.quality_bar = QLabel()
        self.quality_bar.setStyleSheet(
            "font-size: 11px; padding: 4px; font-weight: bold;")
        self.quality_bar.setAlignment(Qt.AlignCenter)
        self.quality_bar.setMaximumHeight(28)
        self.quality_bar.hide()
        right_panel.addWidget(self.quality_bar)

        right_panel.addStretch()

        # Wrap right panel in scroll area for small screens
        right_container = QWidget()
        right_container.setLayout(right_panel)
        right_scroll = QScrollArea()
        right_scroll.setWidget(right_container)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(310)
        right_scroll.setMaximumWidth(420)
        main_layout.addWidget(right_scroll, stretch=1)

    def set_result(self, result: DICResult, base_image: np.ndarray = None,
                   gsd: float = None):
        """Load a DIC result and display default visualization."""
        self._result = result
        self._base_image = base_image
        self._gsd = gsd
        self._smoothed_u = result.u.copy()
        self._smoothed_v = result.v.copy()
        self._strain_data = None
        self.strain_combo.setEnabled(False)

        if base_image is not None:
            self.viewer.set_image(base_image)
            self.viewer.fit_in_view()

        self._update_display()
        self._update_statistics()

    def _get_current_field(self):
        """Get the currently selected display field."""
        if self._result is None:
            return None

        field_key = self.field_combo.currentData()

        if field_key == 'magnitude':
            return compute_magnitude(self._smoothed_u, self._smoothed_v)
        elif field_key == 'u':
            return self._smoothed_u
        elif field_key == 'v':
            return self._smoothed_v
        elif field_key == 'quality':
            return self._result.correlation_quality
        return None

    def _update_display(self):
        """Re-render the overlay based on current display options."""
        field = self._get_current_field()
        if field is None or self._base_image is None:
            return

        cmap = self.cmap_combo.currentText()
        alpha = self.alpha_slider.value() / 100.0

        if self.auto_range.isChecked():
            vmin, vmax = None, None
        else:
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()

        # Create heatmap at result grid resolution
        heatmap_rgba = displacement_colormap(field, vmin=vmin, vmax=vmax, cmap=cmap)

        # Upscale heatmap to image size
        if self._base_image is not None:
            h_img, w_img = self._base_image.shape[:2]
            heatmap_upscaled = cv2.resize(
                heatmap_rgba, (w_img, h_img),
                interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_upscaled = heatmap_rgba

        # Apply alpha
        heatmap_upscaled[:, :, 3] = (
            heatmap_upscaled[:, :, 3].astype(np.float32) * alpha
        ).astype(np.uint8)

        # Draw vectors if enabled
        if self.show_vectors.isChecked() and self._result is not None:
            self._draw_vectors_on_overlay(heatmap_upscaled)

        self.viewer.set_overlay(heatmap_upscaled)

    def _draw_vectors_on_overlay(self, overlay):
        """Draw displacement vectors on the overlay image."""
        step = self.vector_step.value()
        scale = self.vector_scale.value()

        u = self._smoothed_u
        v = self._smoothed_v
        grid_x = self._result.grid_x
        grid_y = self._result.grid_y

        ny, nx = u.shape
        for iy in range(0, ny, step):
            for ix in range(0, nx, step):
                if np.isnan(u[iy, ix]):
                    continue
                x0 = int(grid_x[iy, ix])
                y0 = int(grid_y[iy, ix])
                x1 = int(x0 + u[iy, ix] * scale)
                y1 = int(y0 + v[iy, ix] * scale)

                cv2.arrowedLine(overlay, (x0, y0), (x1, y1),
                                (255, 255, 255, 255), 1, tipLength=0.3)

    def _on_point_clicked(self, x, y):
        """Show displacement/strain values at clicked point."""
        if self._result is None:
            return

        grid_x = self._result.grid_x
        grid_y = self._result.grid_y

        # Find nearest grid point
        dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
        idx = np.unravel_index(np.nanargmin(dist), dist.shape)
        iy, ix = idx

        u_val = self._smoothed_u[iy, ix]
        v_val = self._smoothed_v[iy, ix]
        mag = np.sqrt(u_val ** 2 + v_val ** 2) if not np.isnan(u_val) else float('nan')
        q_val = self._result.correlation_quality[iy, ix]

        text = (f"Posizione pixel: ({x}, {y})\n"
                f"Punto griglia: [{iy}, {ix}]\n"
                f"U (px): {u_val:.3f}   V (px): {v_val:.3f}\n"
                f"Magnitudine: {mag:.3f} px\n"
                f"Qualita correlazione: {q_val:.4f}")

        if self._gsd and self._gsd > 0 and not np.isnan(mag):
            u_m = u_val * self._gsd
            v_m = v_val * self._gsd
            mag_m = mag * self._gsd
            text += (f"\nU: {u_m * 1000:.1f} mm   V: {v_m * 1000:.1f} mm\n"
                     f"Magnitudine: {mag_m * 1000:.1f} mm")

        if self._strain_data:
            exx = self._strain_data['E_xx'][iy, ix]
            eyy = self._strain_data['E_yy'][iy, ix]
            exy = self._strain_data['E_xy'][iy, ix]
            text += f"\nStrain: E_xx={exx:.6f}  E_yy={eyy:.6f}  E_xy={exy:.6f}"

        self.info_label.setText(text)

    def _on_mouse_moved(self, x, y, val):
        """Update status with mouse position."""
        pass  # Handled by main window status bar

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _apply_smoothing(self):
        """Apply selected smoothing to displacement fields."""
        if self._result is None:
            return

        method = self.smooth_combo.currentData()
        sigma = self.smooth_sigma.value()

        u = self._result.u.copy()
        v = self._result.v.copy()

        if method == 'gaussian':
            self._smoothed_u, self._smoothed_v = \
                DisplacementSmoother.gaussian_smooth(u, v, sigma)
        elif method == 'spline':
            self._smoothed_u, self._smoothed_v = \
                DisplacementSmoother.spline_smooth(
                    u, v, self._result.grid_x, self._result.grid_y,
                    smoothing_factor=sigma * 100)
        elif method == 'median':
            ks = max(3, int(sigma) * 2 + 1)
            self._smoothed_u, self._smoothed_v = \
                DisplacementSmoother.median_smooth(u, v, kernel_size=ks)
        else:
            self._smoothed_u = u
            self._smoothed_v = v

        self._update_display()
        self._update_statistics()

    def _remove_outliers(self):
        """Remove outlier displacements using Normalized Median Test."""
        if self._result is None:
            return
        threshold = self.outlier_threshold.value()
        self._smoothed_u, self._smoothed_v = \
            DisplacementSmoother.outlier_removal(
                self._smoothed_u, self._smoothed_v,
                threshold_std=threshold)
        self._update_display()
        self._update_statistics()

    def _apply_coherence_filter(self):
        """Remove spatially incoherent displacement vectors."""
        if self._result is None:
            return
        threshold = self.outlier_threshold.value()
        self._smoothed_u, self._smoothed_v = \
            DisplacementSmoother.spatial_coherence_filter(
                self._smoothed_u, self._smoothed_v,
                window_size=3, threshold=threshold)
        self._update_display()
        self._update_statistics()

    def _reset_smoothing(self):
        """Reset to original unsmoothed data."""
        if self._result is None:
            return
        self._smoothed_u = self._result.u.copy()
        self._smoothed_v = self._result.v.copy()
        self._update_display()
        self._update_statistics()

    # ------------------------------------------------------------------
    # Strain
    # ------------------------------------------------------------------

    def _compute_strain(self):
        """Compute strain from current (possibly smoothed) displacement."""
        if self._result is None:
            return

        grid_spacing = float(self._result.parameters.step_size)
        self._strain_data = StrainCalculator.compute_strain(
            self._smoothed_u, self._smoothed_v, grid_spacing)

        self.strain_combo.setEnabled(True)

        # Add strain fields to display options
        has_strain = False
        for i in range(self.field_combo.count()):
            if self.field_combo.itemData(i) == "strain":
                has_strain = True
                break
        if not has_strain:
            self.field_combo.addItem("--- Strain (vedi sotto) ---", "strain_separator")

    def _update_strain_display(self):
        """Display selected strain component."""
        if self._strain_data is None or self._base_image is None:
            return

        key = self.strain_combo.currentData()
        if key not in self._strain_data:
            return

        field = self._strain_data[key]
        cmap = self.cmap_combo.currentText()
        alpha = self.alpha_slider.value() / 100.0

        heatmap_rgba = displacement_colormap(field, cmap=cmap)

        h_img, w_img = self._base_image.shape[:2]
        heatmap_upscaled = cv2.resize(
            heatmap_rgba, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        heatmap_upscaled[:, :, 3] = (
            heatmap_upscaled[:, :, 3].astype(np.float32) * alpha
        ).astype(np.uint8)

        self.viewer.set_overlay(heatmap_upscaled)

    # ------------------------------------------------------------------
    # Active zones
    # ------------------------------------------------------------------

    def _auto_threshold(self):
        """Compute and apply automatic threshold based on noise estimation."""
        if self._result is None:
            return

        magnitude = compute_magnitude(self._smoothed_u, self._smoothed_v)
        quality = self._result.correlation_quality

        result = DisplacementStatistics.auto_threshold(magnitude, quality)

        self.zone_threshold.setValue(result['recommended'])

        QMessageBox.information(
            self, "Auto Soglia",
            f"Rumore stimato: {result['noise_level']:.3f} px\n"
            f"Soglia SNR (5x rumore): {result['snr_threshold']:.3f} px\n"
            f"Soglia MAD (mediana+3MAD): {result['mad_threshold']:.3f} px\n"
            f"\nSoglia raccomandata: {result['recommended']:.3f} px")

    def _detect_zones(self):
        """Detect and display active displacement zones."""
        if self._result is None:
            return

        threshold = self.zone_threshold.value()
        min_area = self.zone_min_area.value()
        magnitude = compute_magnitude(self._smoothed_u, self._smoothed_v)

        self._active_zones = DisplacementStatistics.detect_active_zones(
            magnitude, threshold, min_area)

        # Mark all zones as pending (not yet validated)
        for zone in self._active_zones:
            zone['status'] = 'pending'  # pending | validated | rejected

        self._refresh_zone_table()
        self.btn_validate_zones.setEnabled(len(self._active_zones) > 0)

    def _refresh_zone_table(self):
        """Update zone table from current _active_zones list."""
        self.zone_table.setRowCount(len(self._active_zones))
        for i, zone in enumerate(self._active_zones):
            self.zone_table.setItem(i, 0, QTableWidgetItem(str(zone['id'])))
            self.zone_table.setItem(i, 1, QTableWidgetItem(str(zone['area_points'])))
            self.zone_table.setItem(i, 2, QTableWidgetItem(f"{zone['max_displacement']:.2f}"))
            self.zone_table.setItem(i, 3, QTableWidgetItem(f"{zone['mean_displacement']:.2f}"))
            # Status column with colour
            status = zone.get('status', 'pending')
            status_item = QTableWidgetItem(
                {'pending': '⏳', 'validated': '✅', 'rejected': '❌'}.get(status, '?'))
            status_item.setTextAlignment(Qt.AlignCenter)
            self.zone_table.setItem(i, 4, status_item)

        self.zones_detected.emit(self._active_zones)

    def _zone_bbox_to_image_coords(self, zone):
        """Convert a zone's grid-index bbox to image pixel coordinates.

        Zone bbox is (row0, col0, row1, col1) in displacement-grid indices.
        Returns (x0, y0, x1, y1) in image pixel coordinates with a margin.
        """
        row0, col0, row1, col1 = zone['bbox']
        grid_x = self._result.grid_x
        grid_y = self._result.grid_y
        ny, nx = grid_x.shape

        # Clamp indices to valid range
        r0 = max(0, min(row0, ny - 1))
        r1 = max(0, min(row1 - 1, ny - 1))
        c0 = max(0, min(col0, nx - 1))
        c1 = max(0, min(col1 - 1, nx - 1))

        x0 = int(grid_x[r0, c0])
        y0 = int(grid_y[r0, c0])
        x1 = int(grid_x[r1, c1])
        y1 = int(grid_y[r1, c1])

        # Add margin (half subset size) for context
        margin = self._result.parameters.subset_size if self._result.parameters else 30
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        if self._ref_image is not None:
            h_img, w_img = self._ref_image.shape[:2]
            x1 = min(w_img, x1 + margin)
            y1 = min(h_img, y1 + margin)
        else:
            x1 += margin
            y1 += margin

        return x0, y0, x1, y1

    def _on_zone_double_clicked(self, row, col):
        """Open inspection dialog for the double-clicked zone."""
        if row < 0 or row >= len(self._active_zones):
            return
        self._inspect_zone(row)

    def _inspect_zone(self, zone_index):
        """Open the inspection dialog for a single zone."""
        if self._ref_image is None or self._def_image is None:
            QMessageBox.warning(
                self, "Errore",
                "Immagini riferimento/deformata non disponibili.\n"
                "Eseguire prima l'analisi DIC.")
            return

        zone = self._active_zones[zone_index]
        x0, y0, x1, y1 = self._zone_bbox_to_image_coords(zone)

        if x1 <= x0 or y1 <= y0:
            return

        # Extract RGB crops (preferred) or fallback to grayscale
        if self._ref_image_rgb is not None:
            rh, rw = self._ref_image_rgb.shape[:2]
            ref_crop = self._ref_image_rgb[
                max(0, y0):min(rh, y1), max(0, x0):min(rw, x1)].copy()
        else:
            ref_crop = self._ref_image[y0:y1, x0:x1].copy()

        if self._def_image_rgb is not None:
            dh, dw = self._def_image_rgb.shape[:2]
            def_crop = self._def_image_rgb[
                max(0, y0):min(dh, y1), max(0, x0):min(dw, x1)].copy()
        else:
            def_crop = self._def_image[y0:y1, x0:x1].copy()

        dialog = AreaInspectionDialog(
            ref_crop, def_crop,
            (x0, y0, x1, y1),
            gsd=self._gsd,
            parent=self)
        dialog.exec()

    def _open_zone_validation(self):
        """Open the zone validation dialog to review all zones."""
        if not self._active_zones:
            QMessageBox.information(
                self, "Nessuna zona",
                "Nessuna zona attiva rilevata. Eseguire prima "
                "'Rileva Zone'.")
            return
        if self._ref_image is None or self._def_image is None:
            QMessageBox.warning(
                self, "Errore",
                "Immagini riferimento/deformata non disponibili.")
            return

        dialog = ZoneValidationDialog(
            zones=self._active_zones,
            result=self._result,
            ref_image=self._ref_image,
            def_image=self._def_image,
            ref_image_rgb=self._ref_image_rgb,
            def_image_rgb=self._def_image_rgb,
            gsd=self._gsd,
            parent=self)

        if dialog.exec() == QDialog.Accepted:
            # Apply validated zones: remove rejected, keep validated + pending
            validated_zones = dialog.get_validated_zones()
            rejected_indices = dialog.get_rejected_zone_indices()

            # Set NaN in displacement field for rejected zones
            if rejected_indices and self._result is not None:
                magnitude = compute_magnitude(
                    self._smoothed_u, self._smoothed_v)
                threshold = self.zone_threshold.value()
                # Re-compute labels to identify pixels of rejected zones
                active_mask = np.zeros_like(magnitude, dtype=np.uint8)
                valid = ~np.isnan(magnitude)
                active_mask[valid & (magnitude > threshold)] = 255
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (3, 3))
                active_mask = cv2.morphologyEx(
                    active_mask, cv2.MORPH_OPEN, kernel)
                n_labels, labels, _, _ = \
                    cv2.connectedComponentsWithStats(
                        active_mask, connectivity=8)

                # Set rejected zone pixels to NaN
                for zone in self._active_zones:
                    if zone.get('status') == 'rejected':
                        zone_label = zone['id']
                        if zone_label < n_labels:
                            reject_mask = labels == zone_label
                            self._smoothed_u[reject_mask] = np.nan
                            self._smoothed_v[reject_mask] = np.nan

            # Update zone list to only validated + pending
            self._active_zones = validated_zones
            self._refresh_zone_table()
            self._update_display()
            self._update_statistics()
            self.btn_validate_zones.setEnabled(
                len(self._active_zones) > 0)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _update_statistics(self):
        """Update statistics display."""
        if self._result is None:
            return

        magnitude = compute_magnitude(self._smoothed_u, self._smoothed_v)
        stats = DisplacementStatistics.compute_statistics(
            self._smoothed_u, self._smoothed_v, magnitude,
            self._result.correlation_quality, self._gsd)

        text = f"Punti validi: {stats.get('n_valid_points', 0)} / {stats.get('n_total_points', 0)}\n"
        text += f"Copertura: {stats.get('coverage_percent', 0):.1f}%\n"
        text += f"---\n"
        text += f"Spostamento medio: {stats.get('mean_displacement_px', 0):.3f} px\n"
        text += f"Spostamento max:   {stats.get('max_displacement_px', 0):.3f} px\n"
        text += f"Deviazione std:    {stats.get('std_displacement_px', 0):.3f} px\n"
        text += f"Percentile 95%:    {stats.get('p95_displacement_px', 0):.3f} px\n"
        text += f"Qualita media:     {stats.get('mean_quality', 0):.4f}\n"

        if self._gsd and 'mean_displacement_mm' in stats:
            text += f"---\n"
            text += f"Spostamento medio: {stats.get('mean_displacement_mm', 0):.1f} mm\n"
            text += f"Spostamento max:   {stats.get('max_displacement_mm', 0):.1f} mm\n"

        self.stats_text.setText(text)

        self._update_histogram(magnitude)
        self._update_quality_indicator()

    def _update_histogram(self, magnitude):
        """Draw magnitude histogram with noise level and threshold lines."""
        self._hist_figure.clear()
        ax = self._hist_figure.add_subplot(111)
        ax.set_facecolor('#fafafa')

        valid = magnitude[~np.isnan(magnitude)]
        if len(valid) < 5:
            ax.text(0.5, 0.5, "Dati insufficienti", ha='center', va='center',
                    color='#757575', transform=ax.transAxes)
            self._hist_canvas.draw_idle()
            return

        # Clip extreme outliers for display (99.5th percentile)
        clip_max = np.percentile(valid, 99.5)
        clipped = valid[valid <= clip_max]
        n_bins = min(80, max(20, len(clipped) // 50))

        ax.hist(clipped, bins=n_bins, color='#42A5F5', alpha=0.8,
                edgecolor='none', density=True)

        # Estimate noise level
        quality = self._result.correlation_quality if self._result else None
        auto = DisplacementStatistics.auto_threshold(magnitude, quality)
        noise = auto['noise_level']
        threshold = self.zone_threshold.value()

        # Draw noise level line
        ax.axvline(noise, color='#FFA726', linewidth=1.5, linestyle='--',
                   label=f'Rumore: {noise:.2f} px')
        # Draw current zone threshold line
        ax.axvline(threshold, color='#EF5350', linewidth=1.5, linestyle='-',
                   label=f'Soglia: {threshold:.2f} px')

        ax.legend(fontsize=7, loc='upper right',
                  facecolor='#ffffff', edgecolor='#bdbdbd',
                  labelcolor='#212121')
        ax.set_xlabel('Magnitudine (px)', fontsize=8, color='#424242')
        ax.set_ylabel('Densita', fontsize=8, color='#424242')
        ax.tick_params(labelsize=7, colors='#616161')
        for spine in ax.spines.values():
            spine.set_color('#bdbdbd')

        self._hist_figure.tight_layout(pad=0.5)
        self._hist_canvas.draw_idle()

    def _update_quality_indicator(self):
        """Show quality indicator: % of low-quality points + warning."""
        if self._result is None:
            self.quality_bar.hide()
            return

        quality = self._result.correlation_quality
        if quality is None:
            self.quality_bar.hide()
            return

        valid = ~np.isnan(quality)
        n_valid = np.sum(valid)
        if n_valid == 0:
            self.quality_bar.hide()
            return

        # Low quality = below correlation threshold (default 0.6)
        threshold = 0.6
        if self._result.parameters:
            threshold = self._result.parameters.correlation_threshold

        n_low = int(np.sum(valid & (quality < threshold)))
        pct_low = n_low / n_valid * 100

        mean_q = float(np.nanmean(quality[valid]))

        if pct_low > 20:
            self.quality_bar.setStyleSheet(
                "font-size: 11px; padding: 4px; font-weight: bold; "
                "background-color: #D32F2F; color: white;")
            self.quality_bar.setText(
                f"ATTENZIONE: {pct_low:.0f}% punti bassa qualita "
                f"(media NCC: {mean_q:.3f})")
        elif pct_low > 10:
            self.quality_bar.setStyleSheet(
                "font-size: 11px; padding: 4px; font-weight: bold; "
                "background-color: #FF9800; color: white;")
            self.quality_bar.setText(
                f"Qualita discreta: {pct_low:.0f}% punti sotto soglia "
                f"(media NCC: {mean_q:.3f})")
        else:
            self.quality_bar.setStyleSheet(
                "font-size: 11px; padding: 4px; font-weight: bold; "
                "background-color: #388E3C; color: white;")
            self.quality_bar.setText(
                f"Buona qualita: {100 - pct_low:.0f}% punti validi "
                f"(media NCC: {mean_q:.3f})")

        self.quality_bar.show()

    # ------------------------------------------------------------------
    # Area selection / inspection
    # ------------------------------------------------------------------

    def set_dic_images(self, ref_image: np.ndarray, def_image: np.ndarray,
                       ref_rgb: np.ndarray = None, def_rgb: np.ndarray = None):
        """Store reference and deformed images used for DIC analysis.

        These are used by the rectangle selection tool to extract
        corresponding image portions.  All images are cropped to the
        minimum common size so that pixel coordinates are consistent.

        Parameters
        ----------
        ref_image, def_image : grayscale images used for DIC
        ref_rgb, def_rgb : optional RGB versions for color extraction
        """
        # Ensure gray images match
        if ref_image is not None and def_image is not None:
            h = min(ref_image.shape[0], def_image.shape[0])
            w = min(ref_image.shape[1], def_image.shape[1])
            ref_image = ref_image[:h, :w]
            def_image = def_image[:h, :w]

        self._ref_image = ref_image
        self._def_image = def_image

        # Ensure RGB images match the gray images dimensions
        if ref_rgb is not None and ref_image is not None:
            rh, rw = ref_image.shape[:2]
            self._ref_image_rgb = ref_rgb[:rh, :rw].copy()
        else:
            self._ref_image_rgb = ref_rgb

        if def_rgb is not None and def_image is not None:
            dh, dw = def_image.shape[:2]
            self._def_image_rgb = def_rgb[:dh, :dw].copy()
        else:
            self._def_image_rgb = def_rgb

    def _toggle_selection_mode(self, enabled):
        """Toggle rectangle selection mode on/off."""
        self._selecting = enabled
        self.viewer.set_drag_mode(enabled)
        self._sel_rect = None
        self._sel_start = None

        if enabled:
            self.sel_info_label.setText(
                "Modalita selezione ATTIVA – trascina un rettangolo sulla mappa.")
            self.sel_info_label.setStyleSheet("font-size: 10px; color: #d84315; font-weight: bold;")
        else:
            self.sel_info_label.setText(
                "Traccia un rettangolo sulla mappa per estrarre "
                "le porzioni di immagine riferimento e deformata.")
            self.sel_info_label.setStyleSheet("font-size: 10px; color: #616161;")
            # Restore heatmap overlay without selection rectangle
            self._update_display()

    def _on_drag_started(self, x, y):
        """Handle start of rectangle drag."""
        if not self._selecting:
            return
        self._sel_start = (x, y)
        # Cache the current heatmap overlay for drawing the rectangle on top
        self._cache_current_overlay()

    def _on_drag_moved(self, x, y):
        """Handle drag movement – draw live rectangle preview."""
        if not self._selecting or self._sel_start is None:
            return
        x0, y0 = self._sel_start
        # Normalize
        rx0, rx1 = min(x0, x), max(x0, x)
        ry0, ry1 = min(y0, y), max(y0, y)
        self._sel_rect = (rx0, ry0, rx1, ry1)
        self._draw_selection_rectangle(rx0, ry0, rx1, ry1)

    def _on_drag_finished(self, x0, y0, x1, y1):
        """Handle end of rectangle drag – extract and show images."""
        if not self._selecting:
            return
        self._sel_rect = (x0, y0, x1, y1)
        # Draw the final rectangle
        self._draw_selection_rectangle(x0, y0, x1, y1)
        # Extract and display
        self._extract_and_show(x0, y0, x1, y1)

    def _cache_current_overlay(self):
        """Cache the current heatmap overlay (without selection rect)."""
        field = self._get_current_field()
        if field is None or self._base_image is None:
            self._heatmap_overlay = None
            return

        cmap = self.cmap_combo.currentText()
        alpha = self.alpha_slider.value() / 100.0

        if self.auto_range.isChecked():
            vmin, vmax = None, None
        else:
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()

        heatmap_rgba = displacement_colormap(field, vmin=vmin, vmax=vmax, cmap=cmap)

        h_img, w_img = self._base_image.shape[:2]
        heatmap_upscaled = cv2.resize(
            heatmap_rgba, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        heatmap_upscaled[:, :, 3] = (
            heatmap_upscaled[:, :, 3].astype(np.float32) * alpha
        ).astype(np.uint8)

        if self.show_vectors.isChecked() and self._result is not None:
            self._draw_vectors_on_overlay(heatmap_upscaled)

        # No copy needed: heatmap_upscaled is a fresh array, not shared
        self._heatmap_overlay = heatmap_upscaled

    def _draw_selection_rectangle(self, x0, y0, x1, y1):
        """Draw the selection rectangle on top of the cached heatmap overlay."""
        if self._heatmap_overlay is None:
            self._cache_current_overlay()
        if self._heatmap_overlay is None:
            return

        overlay = self._heatmap_overlay.copy()
        h, w = overlay.shape[:2]

        # Clamp coordinates
        x0c = max(0, min(x0, w - 1))
        y0c = max(0, min(y0, h - 1))
        x1c = max(0, min(x1, w - 1))
        y1c = max(0, min(y1, h - 1))

        # Semi-transparent yellow fill
        fill_color = np.array([255, 255, 0, 40], dtype=np.uint8)
        roi = overlay[y0c:y1c, x0c:x1c]
        if roi.size > 0:
            # Blend fill
            fill_layer = np.full_like(roi, fill_color)
            alpha_f = fill_color[3] / 255.0
            for c in range(3):
                roi[:, :, c] = np.clip(
                    roi[:, :, c].astype(np.float32) * (1 - alpha_f) +
                    fill_layer[:, :, c].astype(np.float32) * alpha_f,
                    0, 255).astype(np.uint8)
            # Ensure alpha channel is visible
            roi[:, :, 3] = np.maximum(roi[:, :, 3], 80)

        # Draw rectangle border (bright yellow, thick)
        thickness = max(2, int(min(h, w) / 400))
        cv2.rectangle(overlay, (x0c, y0c), (x1c, y1c),
                      (255, 255, 0, 255), thickness)

        # Draw corner markers
        marker_size = max(6, thickness * 3)
        for cx, cy in [(x0c, y0c), (x1c, y0c), (x0c, y1c), (x1c, y1c)]:
            cv2.circle(overlay, (cx, cy), marker_size, (255, 255, 0, 255), -1)
            cv2.circle(overlay, (cx, cy), marker_size, (255, 255, 255, 255), 1)

        # Dimensions text
        rect_w, rect_h = x1c - x0c, y1c - y0c
        label = f"{rect_w} x {rect_h} px"
        if self._gsd and self._gsd > 0:
            w_m = rect_w * self._gsd * 1000  # mm
            h_m = rect_h * self._gsd * 1000
            label += f"  ({w_m:.0f} x {h_m:.0f} mm)"
        font_scale = max(0.4, min(h, w) / 2000.0)
        text_thickness = max(1, int(font_scale * 2))
        text_x = x0c + 5
        text_y = y0c - 8 if y0c > 25 else y1c + 20
        cv2.putText(overlay, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255, 255), text_thickness + 1, cv2.LINE_AA)
        cv2.putText(overlay, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 0, 255), text_thickness, cv2.LINE_AA)

        self.viewer.set_overlay(overlay)

    def _extract_and_show(self, x0, y0, x1, y1):
        """Extract ref/def image crops and show in dialog."""
        if self._ref_image is None or self._def_image is None:
            QMessageBox.warning(
                self, "Errore",
                "Immagini riferimento/deformata non disponibili.\n"
                "Eseguire prima l'analisi DIC.")
            return

        # Clamp to image bounds
        h, w = self._ref_image.shape[:2]
        x0c = max(0, min(x0, w - 1))
        y0c = max(0, min(y0, h - 1))
        x1c = max(0, min(x1, w))
        y1c = max(0, min(y1, h))

        if x1c <= x0c or y1c <= y0c:
            QMessageBox.warning(self, "Errore", "Area selezionata troppo piccola.")
            return

        # Extract RGB crops (preferred) or fallback to grayscale
        if self._ref_image_rgb is not None:
            rh, rw = self._ref_image_rgb.shape[:2]
            # Clamp to RGB image bounds too
            rx0 = max(0, min(x0c, rw - 1))
            ry0 = max(0, min(y0c, rh - 1))
            rx1 = max(0, min(x1c, rw))
            ry1 = max(0, min(y1c, rh))
            ref_crop = self._ref_image_rgb[ry0:ry1, rx0:rx1].copy()
        else:
            ref_crop = self._ref_image[y0c:y1c, x0c:x1c].copy()

        if self._def_image_rgb is not None:
            dh, dw = self._def_image_rgb.shape[:2]
            dx0 = max(0, min(x0c, dw - 1))
            dy0 = max(0, min(y0c, dh - 1))
            dx1 = max(0, min(x1c, dw))
            dy1 = max(0, min(y1c, dh))
            def_crop = self._def_image_rgb[dy0:dy1, dx0:dx1].copy()
        else:
            def_crop = self._def_image[y0c:y1c, x0c:x1c].copy()

        # Show in dialog – enable zone addition if analysis data exists
        has_result = self._result is not None
        dialog = AreaInspectionDialog(
            ref_crop, def_crop,
            (x0c, y0c, x1c, y1c),
            gsd=self._gsd,
            allow_add_zone=has_result,
            parent=self)
        if has_result:
            dialog.zone_added.connect(self._add_manual_zone_from_bbox)
        dialog.exec()

    # ------------------------------------------------------------------
    # Manual zone addition
    # ------------------------------------------------------------------

    def _add_manual_zone_from_bbox(self, bbox):
        """Create a manual active zone from an image-pixel bbox.

        Parameters
        ----------
        bbox : tuple (x0, y0, x1, y1) in image pixel coordinates
        """
        if self._result is None or self._smoothed_u is None:
            return

        x0, y0, x1, y1 = bbox
        grid_x = self._result.grid_x
        grid_y = self._result.grid_y
        ny, nx = grid_x.shape

        magnitude = compute_magnitude(self._smoothed_u, self._smoothed_v)

        # Find grid points inside the bbox
        inside = ((grid_x >= x0) & (grid_x <= x1) &
                  (grid_y >= y0) & (grid_y <= y1))

        # Get displacement values in the selected region
        region_mag = magnitude.copy()
        region_mag[~inside] = np.nan
        valid_mask = inside & ~np.isnan(magnitude)

        n_points = int(valid_mask.sum())
        if n_points == 0:
            return

        valid_magnitudes = magnitude[valid_mask]

        # Find bounding box in grid indices
        rows, cols = np.where(valid_mask)
        grid_row0 = int(rows.min())
        grid_row1 = int(rows.max()) + 1
        grid_col0 = int(cols.min())
        grid_col1 = int(cols.max()) + 1

        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())

        # Assign a new unique ID
        existing_ids = [z['id'] for z in self._active_zones] if self._active_zones else []
        new_id = max(existing_ids, default=0) + 1

        zone = {
            'id': new_id,
            'centroid_row': centroid_row,
            'centroid_col': centroid_col,
            'area_points': n_points,
            'max_displacement': float(np.nanmax(valid_magnitudes)),
            'mean_displacement': float(np.nanmean(valid_magnitudes)),
            'mean_quality': float('nan'),
            'bbox': (grid_row0, grid_col0, grid_row1, grid_col1),
            'status': 'validated',   # manual zones are pre-validated
            'manual': True,          # flag to distinguish from auto-detected
        }

        # Add quality if available
        if self._result.correlation_quality is not None:
            q = self._result.correlation_quality[valid_mask]
            zone['mean_quality'] = float(np.nanmean(q))

        if self._active_zones is None:
            self._active_zones = []

        self._active_zones.append(zone)
        self._refresh_zone_table()
        self.btn_validate_zones.setEnabled(True)

    # ------------------------------------------------------------------
    # Grid inspection
    # ------------------------------------------------------------------

    def _open_grid_inspection(self):
        """Open the systematic grid inspection dialog."""
        if self._ref_image is None or self._def_image is None:
            QMessageBox.warning(
                self, "Errore",
                "Immagini riferimento/deformata non disponibili.\n"
                "Eseguire prima l'analisi DIC.")
            return
        if self._result is None:
            QMessageBox.warning(
                self, "Errore",
                "Nessun risultato DIC disponibile.")
            return

        dialog = GridInspectionDialog(
            ref_image=self._ref_image,
            def_image=self._def_image,
            ref_image_rgb=self._ref_image_rgb,
            def_image_rgb=self._def_image_rgb,
            cell_size=self.grid_cell_size.value(),
            gsd=self._gsd,
            parent=self)

        if dialog.exec() == QDialog.Accepted:
            marked_bboxes = dialog.get_marked_cells()
            if marked_bboxes:
                for bbox in marked_bboxes:
                    self._add_manual_zone_from_bbox(bbox)
                QMessageBox.information(
                    self, "Zone Aggiunte",
                    f"{len(marked_bboxes)} celle marcate come deformate "
                    f"aggiunte alle zone attive.")

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_result(self):
        return self._result

    def get_strain_data(self):
        return self._strain_data

    def get_active_zones(self):
        return self._active_zones

    def get_smoothed_displacement(self):
        return self._smoothed_u, self._smoothed_v


class AreaInspectionDialog(QDialog):
    """Compact dialog showing extracted ref/def image portions in tabs.

    Features:
    - Tabbed views: Riferimento | Deformata | Differenza
    - Info header with area dimensions
    - Save individual or all crops to disk
    - "Aggiungi a Zone Attive" button to mark the area as an active zone
    - Fits within small screens (starts at 700x500)
    """

    # Emitted when the user clicks "Aggiungi a Zone Attive".
    # Carries the image-pixel bbox (x0, y0, x1, y1).
    zone_added = pyqtSignal(tuple)

    def __init__(self, ref_crop: np.ndarray, def_crop: np.ndarray,
                 bbox: tuple, gsd: float = None,
                 allow_add_zone: bool = False, parent=None):
        super().__init__(parent)
        self._ref_crop = ref_crop
        self._def_crop = def_crop
        self._bbox = bbox  # (x0, y0, x1, y1)
        self._gsd = gsd
        self._allow_add_zone = allow_add_zone
        self.setWindowTitle("Ispezione Area Selezionata")
        self.setMinimumSize(400, 300)
        self.resize(700, 500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Info header (compact)
        x0, y0, x1, y1 = self._bbox
        w, h = x1 - x0, y1 - y0
        info = f"Area: ({x0}, {y0})-({x1}, {y1})  |  {w}x{h} px"
        if self._gsd and self._gsd > 0:
            w_mm = w * self._gsd * 1000
            h_mm = h * self._gsd * 1000
            info += f"  |  {w_mm:.1f}x{h_mm:.1f} mm"

        info_label = QLabel(info)
        info_label.setStyleSheet(
            "font-weight: bold; font-size: 11px; padding: 4px; "
            "background-color: #e3f2fd; color: #1565c0;")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setMaximumHeight(28)
        layout.addWidget(info_label)

        # Tabbed image viewer – one tab per image type
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabPosition(QTabWidget.North)

        # Tab 1: Reference
        self.ref_viewer = ImageViewer()
        self._tab_widget.addTab(self.ref_viewer, "Riferimento")

        # Tab 2: Deformed
        self.def_viewer = ImageViewer()
        self._tab_widget.addTab(self.def_viewer, "Deformata")

        # Tab 3: Difference
        self.diff_viewer = ImageViewer()
        self._tab_widget.addTab(self.diff_viewer, "Differenza")

        layout.addWidget(self._tab_widget, stretch=1)

        # Buttons (compact row)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.btn_save_ref = QPushButton("Salva Rif.")
        self.btn_save_ref.clicked.connect(
            lambda: self._save_image(self._ref_crop, "riferimento"))
        btn_layout.addWidget(self.btn_save_ref)

        self.btn_save_def = QPushButton("Salva Def.")
        self.btn_save_def.clicked.connect(
            lambda: self._save_image(self._def_crop, "deformata"))
        btn_layout.addWidget(self.btn_save_def)

        self.btn_save_diff = QPushButton("Salva Diff.")
        self.btn_save_diff.clicked.connect(self._save_diff)
        btn_layout.addWidget(self.btn_save_diff)

        self.btn_save_both = QPushButton("Salva Tutto")
        self.btn_save_both.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 4px 10px; }")
        self.btn_save_both.clicked.connect(self._save_all)
        btn_layout.addWidget(self.btn_save_both)

        btn_layout.addStretch()

        # "Add to active zones" button – only visible when analysis
        # data is available (i.e. called from results viewer, not standalone)
        self.btn_add_zone = QPushButton("➕ Aggiungi a Zone Attive")
        self.btn_add_zone.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "font-weight: bold; padding: 4px 10px; }")
        self.btn_add_zone.setToolTip(
            "Aggiungi quest'area come zona attiva manuale")
        self.btn_add_zone.clicked.connect(self._add_as_zone)
        self.btn_add_zone.setVisible(self._allow_add_zone)
        btn_layout.addWidget(self.btn_add_zone)

        self.btn_close = QPushButton("Chiudi")
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

        # Display images
        self._display_crops()

    def _add_as_zone(self):
        """Emit the bbox as a new manual active zone."""
        self.zone_added.emit(self._bbox)
        self.btn_add_zone.setEnabled(False)
        self.btn_add_zone.setText("✅ Aggiunta")
        QMessageBox.information(
            self, "Zona Aggiunta",
            f"Area ({self._bbox[0]}, {self._bbox[1]})-"
            f"({self._bbox[2]}, {self._bbox[3]}) aggiunta "
            f"alle zone attive.")

    def _display_crops(self):
        """Display the crops in the viewers."""
        self.ref_viewer.set_image(self._ref_crop)
        self.ref_viewer.fit_in_view()

        self.def_viewer.set_image(self._def_crop)
        self.def_viewer.fit_in_view()

        # Compute and display difference image
        self._display_difference()

    def _display_difference(self):
        """Compute and display absolute difference between ref and def."""
        ref = self._ref_crop
        def_img = self._def_crop

        if ref.shape != def_img.shape:
            h = min(ref.shape[0], def_img.shape[0])
            w = min(ref.shape[1], def_img.shape[1])
            ref = ref[:h, :w]
            def_img = def_img[:h, :w]

        # For RGB images: convert to grayscale for diff, then colormap
        if ref.ndim == 3:
            ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
            def_gray = cv2.cvtColor(def_img, cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(ref_gray, def_gray)
        else:
            diff = cv2.absdiff(ref, def_img)

        # Enhance contrast for visibility
        diff_float = diff.astype(np.float32)
        dmax = diff_float.max()
        if dmax > 0:
            diff_enhanced = (diff_float / dmax * 255).astype(np.uint8)
        else:
            diff_enhanced = diff

        # Apply colormap for better visualization
        diff_color = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
        diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

        self.diff_viewer.set_image(diff_color)
        self.diff_viewer.fit_in_view()

    def _get_diff_image(self):
        """Compute the difference image for saving."""
        ref = self._ref_crop
        def_img = self._def_crop

        if ref.shape != def_img.shape:
            h = min(ref.shape[0], def_img.shape[0])
            w = min(ref.shape[1], def_img.shape[1])
            ref = ref[:h, :w]
            def_img = def_img[:h, :w]

        return cv2.absdiff(ref, def_img)

    def _save_image(self, image: np.ndarray, name: str):
        """Save a single image crop."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Salva {name}",
            f"area_{self._bbox[0]}_{self._bbox[1]}_{name}.png",
            "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;Tutti (*)")
        if not filepath:
            return

        try:
            if image.ndim == 3 and image.shape[2] == 3:
                save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                save_img = image
            cv2.imwrite(filepath, save_img)
            QMessageBox.information(self, "Salvato",
                                    f"Immagine salvata: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore nel salvataggio:\n{e}")

    def _save_diff(self):
        """Save the difference image."""
        diff = self._get_diff_image()
        self._save_image(diff, "differenza")

    def _save_all(self):
        """Save all three images (ref, def, diff) to a directory."""
        dirpath = QFileDialog.getExistingDirectory(
            self, "Seleziona cartella di destinazione")
        if not dirpath:
            return

        import os
        x0, y0 = self._bbox[0], self._bbox[1]
        prefix = f"area_{x0}_{y0}"

        try:
            # Reference
            ref_path = os.path.join(dirpath, f"{prefix}_riferimento.png")
            if self._ref_crop.ndim == 3 and self._ref_crop.shape[2] == 3:
                cv2.imwrite(ref_path, cv2.cvtColor(self._ref_crop, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(ref_path, self._ref_crop)

            # Deformed
            def_path = os.path.join(dirpath, f"{prefix}_deformata.png")
            if self._def_crop.ndim == 3 and self._def_crop.shape[2] == 3:
                cv2.imwrite(def_path, cv2.cvtColor(self._def_crop, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(def_path, self._def_crop)

            # Difference
            diff = self._get_diff_image()
            diff_path = os.path.join(dirpath, f"{prefix}_differenza.png")
            if diff.ndim == 3 and diff.shape[2] == 3:
                cv2.imwrite(diff_path, cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(diff_path, diff)

            QMessageBox.information(
                self, "Salvato",
                f"3 immagini salvate in:\n{dirpath}\n\n"
                f"- {prefix}_riferimento.png\n"
                f"- {prefix}_deformata.png\n"
                f"- {prefix}_differenza.png")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore nel salvataggio:\n{e}")


class ZoneValidationDialog(QDialog):
    """Dialog for reviewing and validating/rejecting active zones one by one.

    Features:
    - Navigate zones with Prev/Next buttons or keyboard arrows
    - Shows zone detail: ref, def, diff images in tabs
    - Zone info: coordinates, area, displacement stats
    - Validate (confirm real movement) or Reject (mark as false positive)
    - Summary bar with progress and count of validated/rejected/pending
    - Accept applies validated changes, Cancel discards
    """

    def __init__(self, zones, result, ref_image, def_image,
                 ref_image_rgb=None, def_image_rgb=None,
                 gsd=None, parent=None):
        super().__init__(parent)
        self._zones = zones  # list of zone dicts (mutable — we modify status)
        self._result = result
        self._ref_image = ref_image
        self._def_image = def_image
        self._ref_image_rgb = ref_image_rgb
        self._def_image_rgb = def_image_rgb
        self._gsd = gsd
        self._current_idx = 0
        # Snapshot original statuses so Cancel can restore them
        self._original_statuses = [z.get('status', 'pending')
                                   for z in self._zones]

        self.setWindowTitle("Validazione Zone Attive")
        self.setMinimumSize(550, 420)
        self.resize(780, 560)
        self._setup_ui()
        if self._zones:
            self._show_zone(0)

    # ----------------------------------------------------------
    # UI setup
    # ----------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # --- Top bar: navigation + zone info ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        self.btn_prev = QPushButton("◀ Precedente")
        self.btn_prev.clicked.connect(self._prev_zone)
        top_bar.addWidget(self.btn_prev)

        self.zone_label = QLabel()
        self.zone_label.setAlignment(Qt.AlignCenter)
        self.zone_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 2px 8px;")
        top_bar.addWidget(self.zone_label, stretch=1)

        self.btn_next = QPushButton("Successiva ▶")
        self.btn_next.clicked.connect(self._next_zone)
        top_bar.addWidget(self.btn_next)

        layout.addLayout(top_bar)

        # --- Zone info panel ---
        self.info_label = QLabel()
        self.info_label.setStyleSheet(
            "background-color: #e8eaf6; color: #212121; padding: 5px; "
            "font-size: 11px; font-family: monospace;")
        self.info_label.setWordWrap(True)
        self.info_label.setMaximumHeight(55)
        layout.addWidget(self.info_label)

        # --- Tabbed viewers ---
        self._tab_widget = QTabWidget()
        self.ref_viewer = ImageViewer()
        self._tab_widget.addTab(self.ref_viewer, "Riferimento")
        self.def_viewer = ImageViewer()
        self._tab_widget.addTab(self.def_viewer, "Deformata")
        self.diff_viewer = ImageViewer()
        self._tab_widget.addTab(self.diff_viewer, "Differenza")
        layout.addWidget(self._tab_widget, stretch=1)

        # --- Action buttons ---
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_validate = QPushButton("Valida Zona")
        self.btn_validate.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px 16px; font-size: 13px; }")
        self.btn_validate.clicked.connect(self._validate_current)
        action_row.addWidget(self.btn_validate)

        self.btn_reject = QPushButton("Falso Positivo")
        self.btn_reject.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 8px 16px; font-size: 13px; }")
        self.btn_reject.clicked.connect(self._reject_current)
        action_row.addWidget(self.btn_reject)

        self.btn_skip = QPushButton("Salta")
        self.btn_skip.setStyleSheet(
            "QPushButton { padding: 8px 12px; font-size: 12px; }")
        self.btn_skip.clicked.connect(self._next_zone)
        action_row.addWidget(self.btn_skip)

        layout.addLayout(action_row)

        # --- Summary bar ---
        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setStyleSheet(
            "font-size: 12px; padding: 4px; "
            "background-color: #e8eaf6; color: #37474f;")
        layout.addWidget(self.summary_label)

        # --- Bottom buttons ---
        bottom_row = QHBoxLayout()

        help_label = QLabel(
            "Scorciatoie: V=Valida  R=Rifiuta  "
            "Frecce=Naviga  Spazio=Salta")
        help_label.setStyleSheet("font-size: 10px; color: #616161;")
        bottom_row.addWidget(help_label)

        bottom_row.addStretch()

        self.btn_accept = QPushButton("Applica Validazione")
        self.btn_accept.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 6px 20px; }")
        self.btn_accept.clicked.connect(self.accept)
        bottom_row.addWidget(self.btn_accept)

        self.btn_cancel = QPushButton("Annulla")
        self.btn_cancel.clicked.connect(self._cancel)
        bottom_row.addWidget(self.btn_cancel)

        layout.addLayout(bottom_row)

    # ----------------------------------------------------------
    # Zone display
    # ----------------------------------------------------------

    def _show_zone(self, index):
        """Display zone at given index."""
        if not self._zones:
            return
        self._current_idx = max(0, min(index, len(self._zones) - 1))
        zone = self._zones[self._current_idx]

        # Navigation label
        status_icon = {'pending': '[ ? ]', 'validated': '[ OK ]',
                       'rejected': '[ X ]'}.get(
            zone.get('status', 'pending'), '?')
        self.zone_label.setText(
            f"Zona {self._current_idx + 1} / {len(self._zones)}  "
            f"(ID {zone['id']})  {status_icon}")

        # Info
        bbox = zone['bbox']
        info_parts = [
            f"Area: {zone['area_points']} punti",
            f"Max: {zone['max_displacement']:.2f} px",
            f"Media: {zone['mean_displacement']:.2f} px",
        ]
        if self._gsd and self._gsd > 0:
            max_mm = zone['max_displacement'] * self._gsd * 1000
            mean_mm = zone['mean_displacement'] * self._gsd * 1000
            info_parts.append(f"Max: {max_mm:.1f} mm  Media: {mean_mm:.1f} mm")

        self.info_label.setText("  |  ".join(info_parts))

        # Enable/disable nav buttons
        self.btn_prev.setEnabled(self._current_idx > 0)
        self.btn_next.setEnabled(
            self._current_idx < len(self._zones) - 1)

        # Highlight action buttons based on current status
        status = zone.get('status', 'pending')
        self.btn_validate.setEnabled(status != 'validated')
        self.btn_reject.setEnabled(status != 'rejected')

        # Extract and display crops
        self._display_zone_crops(zone)

        # Update summary
        self._update_summary()

    def _display_zone_crops(self, zone):
        """Extract ref/def crops for the zone and display in tabs."""
        x0, y0, x1, y1 = self._zone_to_image_coords(zone)

        if x1 <= x0 or y1 <= y0:
            return

        # Reference crop
        if self._ref_image_rgb is not None:
            rh, rw = self._ref_image_rgb.shape[:2]
            ref_crop = self._ref_image_rgb[
                max(0, y0):min(rh, y1),
                max(0, x0):min(rw, x1)].copy()
        else:
            ref_crop = self._ref_image[y0:y1, x0:x1].copy()

        # Deformed crop
        if self._def_image_rgb is not None:
            dh, dw = self._def_image_rgb.shape[:2]
            def_crop = self._def_image_rgb[
                max(0, y0):min(dh, y1),
                max(0, x0):min(dw, x1)].copy()
        else:
            def_crop = self._def_image[y0:y1, x0:x1].copy()

        self.ref_viewer.set_image(ref_crop)
        self.ref_viewer.fit_in_view()

        self.def_viewer.set_image(def_crop)
        self.def_viewer.fit_in_view()

        # Difference
        self._display_diff(ref_crop, def_crop)

    def _display_diff(self, ref_crop, def_crop):
        """Compute and show diff of two crops."""
        r, d = ref_crop, def_crop
        if r.shape != d.shape:
            h = min(r.shape[0], d.shape[0])
            w = min(r.shape[1], d.shape[1])
            r, d = r[:h, :w], d[:h, :w]

        if r.ndim == 3:
            r = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
            d = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(r, d)
        diff_f = diff.astype(np.float32)
        dmax = diff_f.max()
        if dmax > 0:
            diff_f = (diff_f / dmax * 255).astype(np.uint8)
        else:
            diff_f = diff
        diff_c = cv2.applyColorMap(diff_f.astype(np.uint8),
                                   cv2.COLORMAP_JET)
        diff_c = cv2.cvtColor(diff_c, cv2.COLOR_BGR2RGB)
        self.diff_viewer.set_image(diff_c)
        self.diff_viewer.fit_in_view()

    def _zone_to_image_coords(self, zone):
        """Convert zone grid bbox to image pixel coords with margin."""
        row0, col0, row1, col1 = zone['bbox']
        grid_x = self._result.grid_x
        grid_y = self._result.grid_y
        ny, nx = grid_x.shape

        r0 = max(0, min(row0, ny - 1))
        r1 = max(0, min(row1 - 1, ny - 1))
        c0 = max(0, min(col0, nx - 1))
        c1 = max(0, min(col1 - 1, nx - 1))

        x0 = int(grid_x[r0, c0])
        y0 = int(grid_y[r0, c0])
        x1 = int(grid_x[r1, c1])
        y1 = int(grid_y[r1, c1])

        margin = (self._result.parameters.subset_size
                  if self._result.parameters else 30)
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        if self._ref_image is not None:
            h_img, w_img = self._ref_image.shape[:2]
            x1 = min(w_img, x1 + margin)
            y1 = min(h_img, y1 + margin)
        else:
            x1 += margin
            y1 += margin

        return x0, y0, x1, y1

    # ----------------------------------------------------------
    # Actions
    # ----------------------------------------------------------

    def _validate_current(self):
        """Mark current zone as validated and advance."""
        self._zones[self._current_idx]['status'] = 'validated'
        self._advance_after_action()

    def _reject_current(self):
        """Mark current zone as rejected (false positive) and advance."""
        self._zones[self._current_idx]['status'] = 'rejected'
        self._advance_after_action()

    def _advance_after_action(self):
        """After an action, move to next unreviewed zone or stay."""
        # Find next pending zone
        for i in range(self._current_idx + 1, len(self._zones)):
            if self._zones[i].get('status') == 'pending':
                self._show_zone(i)
                return
        # Wrap around
        for i in range(0, self._current_idx):
            if self._zones[i].get('status') == 'pending':
                self._show_zone(i)
                return
        # All reviewed — refresh current to update buttons
        self._show_zone(self._current_idx)

    def _prev_zone(self):
        self._show_zone(self._current_idx - 1)

    def _next_zone(self):
        self._show_zone(self._current_idx + 1)

    def _cancel(self):
        """Restore original statuses and close."""
        for z, orig in zip(self._zones, self._original_statuses):
            z['status'] = orig
        self.reject()

    def _update_summary(self):
        """Update the summary bar with counts."""
        n_total = len(self._zones)
        n_val = sum(1 for z in self._zones
                    if z.get('status') == 'validated')
        n_rej = sum(1 for z in self._zones
                    if z.get('status') == 'rejected')
        n_pend = n_total - n_val - n_rej
        self.summary_label.setText(
            f"Totale: {n_total}  |  "
            f"Validate: {n_val}  |  "
            f"Rifiutate: {n_rej}  |  "
            f"In attesa: {n_pend}")

    # ----------------------------------------------------------
    # Results
    # ----------------------------------------------------------

    def get_validated_zones(self):
        """Return zones that were NOT rejected."""
        return [z for z in self._zones if z.get('status') != 'rejected']

    def get_rejected_zone_indices(self):
        """Return original indices of rejected zones."""
        return [i for i, z in enumerate(self._zones)
                if z.get('status') == 'rejected']

    # ----------------------------------------------------------
    # Keyboard navigation
    # ----------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self._prev_zone()
        elif key == Qt.Key_Right:
            self._next_zone()
        elif key == Qt.Key_V:
            self._validate_current()
        elif key == Qt.Key_R:
            self._reject_current()
        elif key == Qt.Key_Space:
            self._next_zone()
        else:
            super().keyPressEvent(event)


# ======================================================================
# Grid Inspection Dialog
# ======================================================================

class GridInspectionDialog(QDialog):
    """Systematic grid inspection of the analysis area.

    Divides the image into a grid of cells (configurable size, default
    350×350 px) and lets the user navigate each cell viewing Reference,
    Deformed, and Difference in tabs (same layout as AreaInspectionDialog).

    Each cell can be marked as "deformed" or left as "normal".
    Marked cells are returned as bounding boxes to be added as manual
    active zones.

    Controls:
        D / Enter  = mark cell as Deformed
        N          = mark cell as Normal (undo mark)
        Arrows     = navigate cells
        Space      = skip to next cell
    """

    def __init__(self, ref_image, def_image,
                 ref_image_rgb=None, def_image_rgb=None,
                 cell_size=350, gsd=None, parent=None):
        super().__init__(parent)
        self._ref_gray = ref_image
        self._def_gray = def_image
        self._ref_rgb = ref_image_rgb
        self._def_rgb = def_image_rgb
        self._gsd = gsd
        self._cell_size = max(50, cell_size)

        # Build the grid of cell bounding boxes
        h, w = ref_image.shape[:2]
        self._img_h, self._img_w = h, w
        self._cells = []          # list of (x0, y0, x1, y1)
        self._cell_status = []    # 'normal' | 'deformed'
        cs = self._cell_size
        for row_start in range(0, h, cs):
            for col_start in range(0, w, cs):
                x0 = col_start
                y0 = row_start
                x1 = min(col_start + cs, w)
                y1 = min(row_start + cs, h)
                # Skip very small edge cells (< 25% of full cell)
                if (x1 - x0) * (y1 - y0) < cs * cs * 0.25:
                    continue
                self._cells.append((x0, y0, x1, y1))
                self._cell_status.append('normal')

        self._n_cols = max(1, (w + cs - 1) // cs)
        self._current_idx = 0
        self._n_cells = len(self._cells)

        self.setWindowTitle(
            f"Ispezione Griglia ({cs}×{cs} px — "
            f"{self._n_cells} celle)")
        self.setMinimumSize(500, 400)
        self.resize(750, 550)
        self._setup_ui()
        if self._cells:
            self._show_cell(0)

    # ----------------------------------------------------------
    # UI
    # ----------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # --- Top: navigation + cell info ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        self.btn_prev = QPushButton("◀ Precedente")
        self.btn_prev.clicked.connect(self._prev_cell)
        top_bar.addWidget(self.btn_prev)

        self.cell_label = QLabel()
        self.cell_label.setAlignment(Qt.AlignCenter)
        self.cell_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 2px 8px;")
        top_bar.addWidget(self.cell_label, stretch=1)

        self.btn_next = QPushButton("Successiva ▶")
        self.btn_next.clicked.connect(self._next_cell)
        top_bar.addWidget(self.btn_next)

        layout.addLayout(top_bar)

        # --- Info bar ---
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(
            "background-color: #e3f2fd; color: #1565c0; padding: 5px; "
            "font-size: 11px;")
        self.info_label.setMaximumHeight(28)
        layout.addWidget(self.info_label)

        # --- Tabbed viewer (same style as AreaInspectionDialog) ---
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabPosition(QTabWidget.North)

        self.ref_viewer = ImageViewer()
        self._tab_widget.addTab(self.ref_viewer, "Riferimento")

        self.def_viewer = ImageViewer()
        self._tab_widget.addTab(self.def_viewer, "Deformata")

        self.diff_viewer = ImageViewer()
        self._tab_widget.addTab(self.diff_viewer, "Differenza")

        layout.addWidget(self._tab_widget, stretch=1)

        # --- Action buttons ---
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_mark_deformed = QPushButton("Deformata (D)")
        self.btn_mark_deformed.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 8px 18px; font-size: 13px; }")
        self.btn_mark_deformed.clicked.connect(self._mark_deformed)
        action_row.addWidget(self.btn_mark_deformed)

        self.btn_mark_normal = QPushButton("Normale (N)")
        self.btn_mark_normal.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px 18px; font-size: 13px; }")
        self.btn_mark_normal.clicked.connect(self._mark_normal)
        action_row.addWidget(self.btn_mark_normal)

        self.btn_skip = QPushButton("Salta (Spazio)")
        self.btn_skip.setStyleSheet(
            "QPushButton { padding: 8px 14px; font-size: 12px; }")
        self.btn_skip.clicked.connect(self._next_cell)
        action_row.addWidget(self.btn_skip)

        layout.addLayout(action_row)

        # --- Summary bar ---
        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setStyleSheet(
            "font-size: 12px; padding: 4px; "
            "background-color: #e8eaf6; color: #37474f;")
        layout.addWidget(self.summary_label)

        # --- Bottom: shortcuts + Accept/Cancel ---
        bottom_row = QHBoxLayout()

        help_label = QLabel(
            "D/Invio=Deformata  N=Normale  "
            "Frecce=Naviga  Spazio=Salta")
        help_label.setStyleSheet("font-size: 10px; color: #616161;")
        bottom_row.addWidget(help_label)

        bottom_row.addStretch()

        self.btn_accept = QPushButton("Applica")
        self.btn_accept.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 6px 20px; }")
        self.btn_accept.clicked.connect(self.accept)
        bottom_row.addWidget(self.btn_accept)

        self.btn_cancel = QPushButton("Annulla")
        self.btn_cancel.clicked.connect(self.reject)
        bottom_row.addWidget(self.btn_cancel)

        layout.addLayout(bottom_row)

    # ----------------------------------------------------------
    # Cell display
    # ----------------------------------------------------------

    def _show_cell(self, index):
        """Display the cell at *index*."""
        if not self._cells:
            return
        self._current_idx = max(0, min(index, self._n_cells - 1))
        x0, y0, x1, y1 = self._cells[self._current_idx]
        status = self._cell_status[self._current_idx]

        grid_row = self._current_idx // self._n_cols
        grid_col = self._current_idx % self._n_cols

        status_txt = "DEFORMATA" if status == 'deformed' else "Normale"
        self.cell_label.setText(
            f"Cella {self._current_idx + 1} / {self._n_cells}  "
            f"[riga {grid_row + 1}, col {grid_col + 1}]  "
            f"{'🔴' if status == 'deformed' else '⚪'} {status_txt}")

        # Info
        cw, ch = x1 - x0, y1 - y0
        info = f"Posizione: ({x0}, {y0})-({x1}, {y1})  |  {cw}x{ch} px"
        if self._gsd and self._gsd > 0:
            info += (f"  |  {cw * self._gsd * 1000:.0f}x"
                     f"{ch * self._gsd * 1000:.0f} mm")
        self.info_label.setText(info)

        # Navigation
        self.btn_prev.setEnabled(self._current_idx > 0)
        self.btn_next.setEnabled(self._current_idx < self._n_cells - 1)

        # Button state
        self.btn_mark_deformed.setEnabled(status != 'deformed')
        self.btn_mark_normal.setEnabled(True)

        # Display crops
        self._display_cell_crops(x0, y0, x1, y1)
        self._update_summary()

    def _display_cell_crops(self, x0, y0, x1, y1):
        """Extract ref/def/diff crops and display in tabs."""
        # Reference
        if self._ref_rgb is not None:
            rh, rw = self._ref_rgb.shape[:2]
            ref_crop = self._ref_rgb[
                max(0, y0):min(rh, y1),
                max(0, x0):min(rw, x1)].copy()
        else:
            ref_crop = self._ref_gray[y0:y1, x0:x1].copy()

        # Deformed
        if self._def_rgb is not None:
            dh, dw = self._def_rgb.shape[:2]
            def_crop = self._def_rgb[
                max(0, y0):min(dh, y1),
                max(0, x0):min(dw, x1)].copy()
        else:
            def_crop = self._def_gray[y0:y1, x0:x1].copy()

        self.ref_viewer.set_image(ref_crop)
        self.ref_viewer.fit_in_view()

        self.def_viewer.set_image(def_crop)
        self.def_viewer.fit_in_view()

        # Difference with colormap
        r, d = ref_crop, def_crop
        if r.shape != d.shape:
            mh = min(r.shape[0], d.shape[0])
            mw = min(r.shape[1], d.shape[1])
            r, d = r[:mh, :mw], d[:mh, :mw]

        if r.ndim == 3:
            r = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
            d = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)

        diff = cv2.absdiff(r, d)
        diff_f = diff.astype(np.float32)
        dmax = diff_f.max()
        if dmax > 0:
            diff_norm = (diff_f / dmax * 255).astype(np.uint8)
        else:
            diff_norm = diff
        diff_c = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        diff_c = cv2.cvtColor(diff_c, cv2.COLOR_BGR2RGB)

        self.diff_viewer.set_image(diff_c)
        self.diff_viewer.fit_in_view()

    # ----------------------------------------------------------
    # Actions
    # ----------------------------------------------------------

    def _mark_deformed(self):
        """Mark current cell as deformed and advance."""
        self._cell_status[self._current_idx] = 'deformed'
        self._advance()

    def _mark_normal(self):
        """Mark current cell as normal and advance."""
        self._cell_status[self._current_idx] = 'normal'
        self._advance()

    def _advance(self):
        """Move to next cell after marking."""
        if self._current_idx < self._n_cells - 1:
            self._show_cell(self._current_idx + 1)
        else:
            self._show_cell(self._current_idx)

    def _prev_cell(self):
        self._show_cell(self._current_idx - 1)

    def _next_cell(self):
        self._show_cell(self._current_idx + 1)

    def _update_summary(self):
        """Update the summary bar."""
        n_def = sum(1 for s in self._cell_status if s == 'deformed')
        self.summary_label.setText(
            f"Celle totali: {self._n_cells}  |  "
            f"Deformate: {n_def}  |  "
            f"Normali: {self._n_cells - n_def}")

    # ----------------------------------------------------------
    # Results
    # ----------------------------------------------------------

    def get_marked_cells(self):
        """Return bboxes of cells marked as deformed.

        Returns list of (x0, y0, x1, y1) in image pixel coords.
        """
        return [self._cells[i] for i in range(self._n_cells)
                if self._cell_status[i] == 'deformed']

    # ----------------------------------------------------------
    # Keyboard
    # ----------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self._prev_cell()
        elif key == Qt.Key_Right:
            self._next_cell()
        elif key in (Qt.Key_D, Qt.Key_Return, Qt.Key_Enter):
            self._mark_deformed()
        elif key == Qt.Key_N:
            self._mark_normal()
        elif key == Qt.Key_Space:
            self._next_cell()
        else:
            super().keyPressEvent(event)
