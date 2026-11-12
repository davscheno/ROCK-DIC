"""DIC algorithm parameter configuration panel.

Besides the raw parameters it offers:
* settings in real units (metres) through the GSD,
* "Suggerisci parametri": subset/step/max displacement chosen from the
  measured image noise and texture (SSSIG), see core.param_advisor,
* "Stima precisione": runs the chosen settings on a synthetic copy of the
  user's image with a known motion and reports the real error and time,
* detachment / noisy-surface options with a preview of the automatic
  vegetation and water masks.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QStackedWidget, QFormLayout, QMessageBox, QScrollArea, QDialog,
    QTextBrowser,
)
from PyQt5.QtCore import pyqtSignal, Qt, QObject, QThread
from dic_app.core.dic_engine import DICParameters, DICMethod, SubPixelMethod
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# combo order -> stacked page index
METHODS = [
    (DICMethod.ICGN, "DIC a subset IC-GN (consigliato)", 0),
    (DICMethod.TEMPLATE_NCC, "Template Matching NCC (veloce, meno preciso)", 1),
    (DICMethod.OPTICAL_FLOW_FARNEBACK, "Optical Flow Farneback (anteprima rapida)", 2),
    (DICMethod.PHASE_CORRELATION, "Phase Correlation FFT (sperimentale)", 3),
    (DICMethod.FEATURE_MATCHING, "Feature Matching SIFT/ORB (sperimentale)", 4),
]

DESCRIPTIONS = {
    DICMethod.ICGN: (
        "Metodo di riferimento della DIC scientifica. Ogni subset puo' "
        "allungarsi, ruotare e deformarsi come la roccia in frana; insensibile "
        "a luminosita' e contrasto. Stima iniziale a piramide: gli spostamenti "
        "grandi non richiedono un raggio di ricerca. Classifica anche i "
        "distacchi, la vegetazione e i cambi di colore."),
    DICMethod.TEMPLATE_NCC: (
        "Confronto di subset rigidi. Veloce ma meno preciso dove il terreno "
        "si deforma; serve impostare un raggio di ricerca >= spostamento."),
    DICMethod.OPTICAL_FLOW_FARNEBACK: (
        "Flusso ottico denso. Molto veloce, utile come anteprima; soffre i "
        "bordi netti e i cambi di luce, qualita' stimata solo a posteriori."),
    DICMethod.PHASE_CORRELATION: (
        "Correlazione in frequenza su finestre fisse. Solo traslazioni rigide "
        "piccole; poco preciso negli spostamenti grandi. Sperimentale."),
    DICMethod.FEATURE_MATCHING: (
        "Punti caratteristici SIFT/ORB interpolati sulla griglia. Risultato "
        "sparso e meno affidabile; utile solo con immagini molto diverse."),
}


class _PrecisionWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, ref, params, gsd, deff=None):
        super().__init__()
        self.ref, self.params, self.gsd = ref, params, gsd
        self.deff = deff

    def run(self):
        try:
            from dic_app.core.param_advisor import estimate_precision
            res = estimate_precision(self.ref, self.params, self.gsd,
                                     progress=lambda p, m: self.progress.emit(p, m),
                                     deff=self.deff)
            self.finished.emit(res)
        except Exception as e:
            logger.exception("Precision estimate failed")
            self.error.emit(str(e))


class ParamsPanelWidget(QWidget):
    """DIC parameter configuration panel."""

    params_changed = pyqtSignal(object)  # emits DICParameters

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ref_image = None
        self._def_image = None
        self._ref_rgb = None
        self._def_rgb = None
        self._gsd = None
        self._image_shape = None
        self._syncing = False
        self._prec_thread = None
        self._prec_worker = None
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        main_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # --- Algorithm ---
        algo_group = QGroupBox("Algoritmo DIC")
        algo_layout = QFormLayout(algo_group)
        self.method_combo = QComboBox()
        for m, label, _ in METHODS:
            self.method_combo.addItem(label, m)
        algo_layout.addRow("Metodo:", self.method_combo)
        self.method_desc = QLabel()
        self.method_desc.setWordWrap(True)
        self.method_desc.setStyleSheet("color: #546e7a; font-style: italic;")
        algo_layout.addRow(self.method_desc)
        main_layout.addWidget(algo_group)

        # --- Buttons ---
        row = QHBoxLayout()
        self.btn_guide = QPushButton("Guida")
        self.btn_guide.clicked.connect(self._show_algorithm_guide)
        row.addWidget(self.btn_guide)
        self.btn_auto_guess = QPushButton("Suggerisci parametri")
        self.btn_auto_guess.setStyleSheet(
            "QPushButton { background-color: #ff8f00; color: white; "
            "padding: 6px 10px; font-weight: bold; }")
        self.btn_auto_guess.setToolTip(
            "Sceglie subset, passo e spostamento massimo dal rumore e dalla "
            "texture misurati sulle tue immagini (criterio SSSIG).")
        self.btn_auto_guess.clicked.connect(self._auto_guess_parameters)
        row.addWidget(self.btn_auto_guess)
        self.btn_precision = QPushButton("Stima precisione")
        self.btn_precision.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "padding: 6px 10px; font-weight: bold; }")
        self.btn_precision.setToolTip(
            "Prova i parametri su una copia della tua immagine con uno "
            "spostamento noto: mostra l'errore reale in px/mm e il tempo "
            "previsto per l'analisi completa.")
        self.btn_precision.clicked.connect(self._estimate_precision)
        row.addWidget(self.btn_precision)
        main_layout.addLayout(row)

        # --- Real units ---
        real_group = QGroupBox("Fenomeno da misurare (unita' reali)")
        real_layout = QFormLayout(real_group)
        self.gsd_label = QLabel("GSD non impostato (scheda Progetto)")
        self.gsd_label.setStyleSheet("color: #555;")
        real_layout.addRow(self.gsd_label)
        self.max_disp_m = QDoubleSpinBox()
        self.max_disp_m.setRange(0.0, 1000.0)
        self.max_disp_m.setDecimals(2)
        self.max_disp_m.setSingleStep(0.1)
        self.max_disp_m.setSuffix(" m")
        self.max_disp_m.setSpecialValueText("automatico")
        self.max_disp_m.setToolTip(
            "Spostamento piu' grande atteso tra i due rilievi. 0 = stimato "
            "dalle immagini con 'Suggerisci parametri'.")
        real_layout.addRow("Spostamento max atteso:", self.max_disp_m)
        self.min_feature_m = QDoubleSpinBox()
        self.min_feature_m.setRange(0.0, 1000.0)
        self.min_feature_m.setDecimals(1)
        self.min_feature_m.setSuffix(" m")
        self.min_feature_m.setSpecialValueText("nessun limite")
        self.min_feature_m.setValue(5.0)
        self.min_feature_m.setToolTip(
            "Dimensione del piu' piccolo distacco o corpo in movimento da "
            "vedere: limita la dimensione del subset (almeno 2 subset).")
        real_layout.addRow("Fenomeno piu' piccolo:", self.min_feature_m)
        main_layout.addWidget(real_group)

        # --- Common ---
        common_group = QGroupBox("Parametri comuni")
        common_layout = QFormLayout(common_group)
        self.subset_spin = QSpinBox()
        self.subset_spin.setRange(5, 201)
        self.subset_spin.setSingleStep(2)
        self.subset_spin.setValue(41)
        self.subset_spin.setToolTip("Lato del subset in pixel (dispari).")
        common_layout.addRow("Dim. subset (px):", self.subset_spin)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 200)
        self.step_spin.setValue(10)
        self.step_spin.setToolTip("Distanza tra i punti di analisi.")
        common_layout.addRow("Passo griglia (px):", self.step_spin)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.6)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setToolTip("Correlazione minima per accettare una misura.")
        common_layout.addRow("Soglia correlazione:", self.threshold_spin)
        self.border_margin_spin = QSpinBox()
        self.border_margin_spin.setRange(0, 500)
        self.border_margin_spin.setValue(10)
        self.border_margin_spin.setSingleStep(5)
        self.border_margin_spin.setSuffix(" px")
        self.border_margin_spin.setToolTip(
            "Fascia dai bordi esclusa. Consigliato >= subset con allineamento.")
        common_layout.addRow("Margine bordi:", self.border_margin_spin)
        self.grid_info = QLabel()
        self.grid_info.setWordWrap(True)
        common_layout.addRow(self.grid_info)
        main_layout.addWidget(common_group)

        # --- Method specific ---
        self.method_stack = QStackedWidget()

        icgn_w = QWidget()
        icgn_l = QFormLayout(icgn_w)
        self.max_disp_px = QDoubleSpinBox()
        self.max_disp_px.setRange(1.0, 5000.0)
        self.max_disp_px.setDecimals(1)
        self.max_disp_px.setValue(20.0)
        self.max_disp_px.setSuffix(" px")
        self.max_disp_px.setToolTip(
            "Spostamento massimo: definisce i livelli della piramide. Non serve "
            "precisione, basta un ordine di grandezza (meglio in eccesso).")
        icgn_l.addRow("Spostamento max:", self.max_disp_px)
        self.icgn_iter = QSpinBox()
        self.icgn_iter.setRange(5, 100)
        self.icgn_iter.setValue(30)
        icgn_l.addRow("Iterazioni max IC-GN:", self.icgn_iter)
        self.method_stack.addWidget(icgn_w)          # page 0

        ncc_widget = QWidget()
        ncc_layout = QFormLayout(ncc_widget)
        self.search_rx = QSpinBox()
        self.search_rx.setRange(5, 500)
        self.search_rx.setValue(50)
        ncc_layout.addRow("Raggio ricerca X (px):", self.search_rx)
        self.search_ry = QSpinBox()
        self.search_ry.setRange(5, 500)
        self.search_ry.setValue(50)
        ncc_layout.addRow("Raggio ricerca Y (px):", self.search_ry)
        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItem("Gaussiano (consigliato)", SubPixelMethod.GAUSSIAN)
        self.subpixel_combo.addItem("Parabolico", SubPixelMethod.PARABOLIC)
        self.subpixel_combo.addItem("Bicubico (lento)", SubPixelMethod.BICUBIC)
        self.subpixel_combo.addItem("Nessuno (solo pixel interi)", SubPixelMethod.NONE)
        ncc_layout.addRow("Metodo sub-pixel:", self.subpixel_combo)
        self.method_stack.addWidget(ncc_widget)      # page 1

        of_widget = QWidget()
        of_layout = QFormLayout(of_widget)
        self.of_pyr_scale = QDoubleSpinBox()
        self.of_pyr_scale.setRange(0.1, 0.9)
        self.of_pyr_scale.setValue(0.5)
        self.of_pyr_scale.setSingleStep(0.1)
        of_layout.addRow("Scala piramide:", self.of_pyr_scale)
        self.of_levels = QSpinBox()
        self.of_levels.setRange(1, 10)
        self.of_levels.setValue(5)
        of_layout.addRow("Livelli piramide:", self.of_levels)
        self.of_winsize = QSpinBox()
        self.of_winsize.setRange(3, 51)
        self.of_winsize.setSingleStep(2)
        self.of_winsize.setValue(15)
        of_layout.addRow("Dim. finestra:", self.of_winsize)
        self.of_iterations = QSpinBox()
        self.of_iterations.setRange(1, 20)
        self.of_iterations.setValue(3)
        of_layout.addRow("Iterazioni:", self.of_iterations)
        self.of_poly_n = QSpinBox()
        self.of_poly_n.setRange(3, 9)
        self.of_poly_n.setSingleStep(2)
        self.of_poly_n.setValue(5)
        of_layout.addRow("Poly N:", self.of_poly_n)
        self.of_poly_sigma = QDoubleSpinBox()
        self.of_poly_sigma.setRange(0.1, 5.0)
        self.of_poly_sigma.setValue(1.2)
        of_layout.addRow("Poly sigma:", self.of_poly_sigma)
        self.method_stack.addWidget(of_widget)       # page 2

        phase_widget = QWidget()
        phase_layout = QFormLayout(phase_widget)
        self.upsample_spin = QSpinBox()
        self.upsample_spin.setRange(1, 100)
        self.upsample_spin.setValue(20)
        phase_layout.addRow("Fattore upsampling:", self.upsample_spin)
        self.method_stack.addWidget(phase_widget)    # page 3

        feat_widget = QWidget()
        feat_layout = QFormLayout(feat_widget)
        self.max_features = QSpinBox()
        self.max_features.setRange(100, 100000)
        self.max_features.setValue(10000)
        self.max_features.setSingleStep(1000)
        feat_layout.addRow("Max features:", self.max_features)
        self.match_ratio = QDoubleSpinBox()
        self.match_ratio.setRange(0.1, 1.0)
        self.match_ratio.setValue(0.75)
        self.match_ratio.setSingleStep(0.05)
        feat_layout.addRow("Ratio test:", self.match_ratio)
        self.method_stack.addWidget(feat_widget)     # page 4

        main_layout.addWidget(self.method_stack)

        # --- Detachments / noisy surfaces (IC-GN) ---
        self.change_group = QGroupBox("Distacchi e zone rumorose")
        cg = QFormLayout(self.change_group)
        self.dual_check = QCheckBox("Doppia correlazione (ignora bagnato/ombre)")
        self.dual_check.setChecked(True)
        self.dual_check.setToolTip(
            "Ripete il confronto su immagini a contrasto normalizzato: se la "
            "roccia e' riconoscibile li', e' cambiato solo il colore "
            "(bagnato, ombra) e lo spostamento viene misurato comunque.")
        cg.addRow(self.dual_check)
        self.change_thr = QDoubleSpinBox()
        self.change_thr.setRange(0.0, 0.9)
        self.change_thr.setSingleStep(0.05)
        self.change_thr.setValue(0.4)
        self.change_thr.setToolTip(
            "Sotto questa correlazione (in entrambi i confronti) la superficie "
            "e' considerata cambiata: possibile distacco.")
        cg.addRow("Soglia cambiamento:", self.change_thr)
        self.min_change_area = QDoubleSpinBox()
        self.min_change_area.setRange(0.0, 10000.0)
        self.min_change_area.setDecimals(1)
        self.min_change_area.setValue(0.5)
        self.min_change_area.setSuffix(" m²")
        cg.addRow("Area minima distacco:", self.min_change_area)
        self.veg_check = QCheckBox("Escludi vegetazione (automatica)")
        self.veg_check.setChecked(True)
        self.veg_sens = QDoubleSpinBox()
        self.veg_sens.setRange(0.3, 3.0)
        self.veg_sens.setSingleStep(0.1)
        self.veg_sens.setValue(1.0)
        self.veg_sens.setToolTip(">1 marca piu' vegetazione, <1 meno.")
        vrow = QHBoxLayout()
        vrow.addWidget(self.veg_check)
        vrow.addWidget(QLabel("sens."))
        vrow.addWidget(self.veg_sens)
        cg.addRow(vrow)
        self.water_check = QCheckBox("Escludi acqua/schiuma (automatica)")
        self.water_check.setChecked(True)
        self.water_sens = QDoubleSpinBox()
        self.water_sens.setRange(0.3, 3.0)
        self.water_sens.setSingleStep(0.1)
        self.water_sens.setValue(1.0)
        wrow = QHBoxLayout()
        wrow.addWidget(self.water_check)
        wrow.addWidget(QLabel("sens."))
        wrow.addWidget(self.water_sens)
        cg.addRow(wrow)
        self.btn_mask_preview = QPushButton("Anteprima maschere")
        self.btn_mask_preview.clicked.connect(self._preview_masks)
        cg.addRow(self.btn_mask_preview)
        tip = QLabel("Mare, spiaggia o altre zone da ignorare si possono anche "
                     "escludere a mano nella scheda Maschere.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #555;")
        cg.addRow(tip)
        main_layout.addWidget(self.change_group)

        # --- Info ---
        info_group = QGroupBox("Informazioni e stima")
        info_layout = QVBoxLayout(info_group)
        self.estimation_label = QLabel()
        self.estimation_label.setWordWrap(True)
        self.estimation_label.setTextFormat(Qt.RichText)
        info_layout.addWidget(self.estimation_label)
        self.precision_label = QLabel()
        self.precision_label.setWordWrap(True)
        self.precision_label.setTextFormat(Qt.RichText)
        info_layout.addWidget(self.precision_label)
        main_layout.addWidget(info_group)
        main_layout.addStretch()

        self._update_method_ui(0)

    def _connect_signals(self):
        self.method_combo.currentIndexChanged.connect(self._update_method_ui)
        for w in (self.subset_spin, self.step_spin, self.threshold_spin,
                  self.border_margin_spin, self.search_rx, self.search_ry,
                  self.icgn_iter, self.change_thr, self.min_change_area,
                  self.veg_sens, self.water_sens):
            w.valueChanged.connect(self._on_params_changed)
        for c in (self.dual_check, self.veg_check, self.water_check):
            c.toggled.connect(self._on_params_changed)
        self.max_disp_px.valueChanged.connect(self._on_max_disp_px_changed)
        self.max_disp_m.valueChanged.connect(self._on_max_disp_m_changed)

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _current_method(self):
        return self.method_combo.currentData()

    def _update_method_ui(self, index):
        page = METHODS[index][2] if 0 <= index < len(METHODS) else 0
        self.method_stack.setCurrentIndex(page)
        m = self._current_method()
        self.method_desc.setText(DESCRIPTIONS.get(m, ""))
        self.change_group.setEnabled(m == DICMethod.ICGN)
        self._on_params_changed()

    def _on_max_disp_px_changed(self, val):
        if self._syncing:
            return
        if self._gsd:
            self._syncing = True
            self.max_disp_m.setValue(val * self._gsd)
            self._syncing = False
        self._on_params_changed()

    def _on_max_disp_m_changed(self, val):
        if self._syncing or not self._gsd or val <= 0:
            return
        self._syncing = True
        self.max_disp_px.setValue(val / self._gsd)
        self._syncing = False
        self._on_params_changed()

    def _on_params_changed(self, *_):
        subset = self.subset_spin.value()
        if subset % 2 == 0:
            self.subset_spin.blockSignals(True)
            self.subset_spin.setValue(subset + 1)
            self.subset_spin.blockSignals(False)
        step = self.step_spin.value()
        txt = f"Distanza tra i punti: {step} px"
        if self._gsd:
            txt += (f" = {step * self._gsd:.2f} m; subset "
                    f"{self.subset_spin.value() * self._gsd:.2f} m di lato")
        self.grid_info.setText(txt)
        if self._image_shape is not None:
            self.update_estimation(self._image_shape)
        self.params_changed.emit(self.get_parameters())

    # ------------------------------------------------------------------
    # Parameters in/out
    # ------------------------------------------------------------------

    def get_parameters(self) -> DICParameters:
        subset = self.subset_spin.value()
        if subset % 2 == 0:
            subset += 1
        return DICParameters(
            method=self._current_method(),
            subset_size=subset,
            step_size=self.step_spin.value(),
            search_radius_x=self.search_rx.value(),
            search_radius_y=self.search_ry.value(),
            subpixel_method=self.subpixel_combo.currentData(),
            correlation_threshold=self.threshold_spin.value(),
            border_margin=self.border_margin_spin.value(),
            of_pyr_scale=self.of_pyr_scale.value(),
            of_levels=self.of_levels.value(),
            of_winsize=self.of_winsize.value(),
            of_iterations=self.of_iterations.value(),
            of_poly_n=self.of_poly_n.value(),
            of_poly_sigma=self.of_poly_sigma.value(),
            upsample_factor=self.upsample_spin.value(),
            max_features=self.max_features.value(),
            match_ratio=self.match_ratio.value(),
            max_displacement_px=self.max_disp_px.value(),
            icgn_max_iter=self.icgn_iter.value(),
            dual_correlation=self.dual_check.isChecked(),
            change_threshold=self.change_thr.value(),
            min_change_area_m2=self.min_change_area.value(),
            exclude_vegetation=self.veg_check.isChecked(),
            exclude_water=self.water_check.isChecked(),
            veg_sensitivity=self.veg_sens.value(),
            water_sensitivity=self.water_sens.value(),
        )

    def set_parameters(self, params: DICParameters):
        for i in range(self.method_combo.count()):
            if self.method_combo.itemData(i) == params.method:
                self.method_combo.setCurrentIndex(i)
                break
        self.subset_spin.setValue(params.subset_size)
        self.step_spin.setValue(params.step_size)
        self.threshold_spin.setValue(params.correlation_threshold)
        self.border_margin_spin.setValue(params.border_margin)
        self.search_rx.setValue(params.search_radius_x)
        self.search_ry.setValue(params.search_radius_y)
        for i in range(self.subpixel_combo.count()):
            if self.subpixel_combo.itemData(i) == params.subpixel_method:
                self.subpixel_combo.setCurrentIndex(i)
                break
        self.of_pyr_scale.setValue(params.of_pyr_scale)
        self.of_levels.setValue(params.of_levels)
        self.of_winsize.setValue(params.of_winsize)
        self.of_iterations.setValue(params.of_iterations)
        self.of_poly_n.setValue(params.of_poly_n)
        self.of_poly_sigma.setValue(params.of_poly_sigma)
        self.upsample_spin.setValue(params.upsample_factor)
        self.max_features.setValue(params.max_features)
        self.match_ratio.setValue(params.match_ratio)
        if params.max_displacement_px:
            self.max_disp_px.setValue(params.max_displacement_px)
        self.icgn_iter.setValue(params.icgn_max_iter)
        self.dual_check.setChecked(params.dual_correlation)
        self.change_thr.setValue(params.change_threshold)
        self.min_change_area.setValue(params.min_change_area_m2)
        self.veg_check.setChecked(params.exclude_vegetation)
        self.water_check.setChecked(params.exclude_water)
        self.veg_sens.setValue(params.veg_sensitivity)
        self.water_sens.setValue(params.water_sensitivity)

    def update_estimation(self, image_shape):
        if image_shape is None:
            return
        self._image_shape = image_shape
        h, w = image_shape[:2]
        params = self.get_parameters()
        half = params.subset_size // 2
        ny = len(range(half, h - half, max(1, params.step_size)))
        nx = len(range(half, w - half, max(1, params.step_size)))
        total = ny * nx
        txt = (f"Immagine: {w} x {h} px<br>"
               f"Punti griglia: {nx} x {ny} = {total:,}".replace(",", ".") + "<br>"
               f"Subset: {params.subset_size} x {params.subset_size} px")
        if params.method == DICMethod.ICGN:
            txt += f"<br>Tempo indicativo IC-GN: ~{max(1, total * 0.0007):.0f} s"
        self.estimation_label.setText(txt)

    def set_images_for_guess(self, ref_gray, def_gray):
        self._ref_image = ref_gray
        self._def_image = def_gray
        if ref_gray is not None:
            self._image_shape = ref_gray.shape

    def set_context(self, gsd=None, ref_rgb=None, def_rgb=None):
        """GSD and colour images (for real units and surface masks)."""
        self._gsd = gsd if gsd and gsd > 0 else None
        self._ref_rgb = ref_rgb
        self._def_rgb = def_rgb
        if self._gsd:
            self.gsd_label.setText(
                f"GSD {self._gsd * 100:.2f} cm/px: 1 px = {self._gsd * 1000:.1f} mm")
            self._syncing = True
            self.max_disp_m.setValue(self.max_disp_px.value() * self._gsd)
            self._syncing = False
        else:
            self.gsd_label.setText("GSD non impostato (scheda Progetto): "
                                   "valori solo in pixel")
        self.max_disp_m.setEnabled(bool(self._gsd))
        self.min_feature_m.setEnabled(bool(self._gsd))
        self._on_params_changed()

    # ------------------------------------------------------------------
    # Guided parameters
    # ------------------------------------------------------------------

    def _auto_guess_parameters(self):
        if self._ref_image is None:
            QMessageBox.warning(self, "Immagini non disponibili",
                                "Carica le immagini prima di usare il suggerimento.")
            return
        from PyQt5.QtWidgets import QApplication
        from dic_app.core.param_advisor import suggest_parameters
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            s = suggest_parameters(
                self._ref_image, self._def_image, gsd=self._gsd,
                max_disp_m=self.max_disp_m.value() or None,
                min_feature_m=self.min_feature_m.value() or None)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Suggerimento parametri", str(e))
            return
        QApplication.restoreOverrideCursor()

        self.method_combo.setCurrentIndex(0)   # IC-GN
        self.subset_spin.setValue(s['subset_size'])
        self.step_spin.setValue(s['step_size'])
        self.border_margin_spin.setValue(max(self.border_margin_spin.value(),
                                             s['border_margin']))
        self.max_disp_px.setValue(s['max_displacement_px'])
        html = "<h3>Parametri suggeriti</h3><ul>" + "".join(
            f"<li>{r}</li>" for r in s['reasons']) + "</ul>"
        html += ("<p>Premi <b>Stima precisione</b> per verificarli sulla tua "
                 "immagine prima dell'analisi.</p>")
        msg = QMessageBox(self)
        msg.setWindowTitle("Suggerisci parametri")
        msg.setTextFormat(Qt.RichText)
        msg.setText(html)
        msg.exec()

    def _estimate_precision(self):
        if self._ref_image is None:
            QMessageBox.warning(self, "Stima precisione",
                                "Carica le immagini prima della stima.")
            return
        if self._prec_thread is not None:
            return
        params = self.get_parameters()
        self.btn_precision.setEnabled(False)
        self.precision_label.setText("<i>Stima della precisione in corso...</i>")
        self._prec_thread = QThread()
        self._prec_worker = _PrecisionWorker(self._ref_image, params, self._gsd,
                                             self._def_image)
        self._prec_worker.moveToThread(self._prec_thread)
        self._prec_thread.started.connect(self._prec_worker.run)
        self._prec_worker.finished.connect(self._on_precision_done)
        self._prec_worker.error.connect(self._on_precision_error)
        self._prec_worker.finished.connect(self._prec_thread.quit, Qt.DirectConnection)
        self._prec_worker.error.connect(self._prec_thread.quit, Qt.DirectConnection)
        self._prec_thread.start()

    def _cleanup_prec(self):
        if self._prec_thread is not None:
            self._prec_thread.wait(3000)
            self._prec_thread.deleteLater()
        if self._prec_worker is not None:
            self._prec_worker.deleteLater()
        self._prec_thread = None
        self._prec_worker = None
        self.btn_precision.setEnabled(True)

    def _on_precision_error(self, msg):
        self._cleanup_prec()
        self.precision_label.setText(f"<span style='color:#c62828'>Errore: {msg}</span>")

    def _on_precision_done(self, res):
        self._cleanup_prec()
        from dic_app.core.param_advisor import estimate_full_time
        params = self.get_parameters()
        t_full = (estimate_full_time(res['sec_per_point'], self._image_shape, params)
                  if self._image_shape is not None else float('nan'))
        mm = ""
        if 'median_mm' in res:
            mm = (f" = <b>{res['median_mm']:.1f} mm</b> (95%: "
                  f"{res['p95_mm']:.1f} mm)")
        ok = res['median_px'] < 0.05 and res['gross_pct'] < 1
        color = "#2e7d32" if ok else "#ef6c00"
        self.precision_label.setText(
            f"<b style='color:{color}'>Precisione stimata</b> (copia "
            f"dell'immagine {'deformata' if res.get('source') == 'def' else 'di riferimento'}"
            f", la meno nitida, con spostamento noto, rumore {res['noise']:.1f}):<br>"
            f"errore mediano <b>{res['median_px']:.3f} px</b>{mm}<br>"
            f"95% dei punti entro {res['p95_px']:.3f} px; errori grossolani "
            f"{res['gross_pct']:.1f}%; punti validi {res['valid_pct']:.0f}%<br>"
            f"Tempo previsto per l'analisi completa: ~{t_full:.0f} s")

    # ------------------------------------------------------------------
    # Mask preview
    # ------------------------------------------------------------------

    def _preview_masks(self):
        if self._ref_rgb is None:
            QMessageBox.information(
                self, "Anteprima maschere",
                "Servono le immagini a colori: carica le immagini e apri "
                "questa scheda dopo aver selezionato riferimento e deformata.")
            return
        import cv2
        from dic_app.core import change_detection as cd
        from dic_app.gui.image_viewer import ImageViewer
        p = self.get_parameters()
        rgb = self._ref_rgb[..., :3]
        def_rgb = self._def_rgb[..., :3] if (self._def_rgb is not None and
                                             self._def_rgb.shape[:2] == rgb.shape[:2]) else None
        lab = cd.surface_labels(rgb, def_rgb, rgb.shape,
                                vegetation=p.exclude_vegetation, water=p.exclude_water,
                                veg_sensitivity=p.veg_sensitivity,
                                water_sensitivity=p.water_sensitivity)
        ov = rgb.astype(np.float32).copy()
        colors = {cd.SURF_VEGETATION: (255, 0, 255), cd.SURF_WATER: (0, 90, 255),
                  cd.SURF_NODATA: (40, 40, 40)}
        for k, c in colors.items():
            m = lab == k
            ov[m] = 0.4 * ov[m] + 0.6 * np.array(c, np.float32)
        ov = ov.astype(np.uint8)
        s = min(1.0, 2500.0 / max(ov.shape[:2]))
        if s < 1.0:
            ov = cv2.resize(ov, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        dlg = QDialog(self)
        dlg.setWindowTitle("Anteprima maschere automatiche")
        dlg.resize(1100, 700)
        lay = QVBoxLayout(dlg)
        viewer = ImageViewer()
        viewer.set_image(ov)
        lay.addWidget(viewer)
        veg_pct = 100 * np.mean(lab == cd.SURF_VEGETATION)
        wat_pct = 100 * np.mean(lab == cd.SURF_WATER)
        info = QLabel(f"<span style='color:#c000c0'>&#9632;</span> Vegetazione "
                      f"{veg_pct:.1f}% &nbsp; <span style='color:#0050ff'>&#9632;"
                      f"</span> Acqua/schiuma {wat_pct:.1f}% &nbsp; (unione dei "
                      f"due rilievi). Regola la sensibilita' se la maschera "
                      f"copre troppo o troppo poco.")
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        lay.addWidget(info)
        dlg.show()
        viewer.fit_in_view()
        dlg.exec_()

    # ------------------------------------------------------------------
    # Guide
    # ------------------------------------------------------------------

    def _show_algorithm_guide(self):
        guide_html = """
        <h2 style="color: #1565c0;">Come scegliere metodo e parametri</h2>
        <h3>Metodo consigliato: DIC a subset IC-GN</h3>
        <p>Per il monitoraggio di falesie e frane usa sempre IC-GN, salvo
        casi particolari. Sulla tua falesia, con uno spostamento simulato
        noto, l'errore e' stato ~0,006 px sulla roccia stabile e ~0,04 px in
        una frana di 40 px (NCC: 0,065 e 0,38 px), senza errori grossolani.</p>
        <ul>
        <li><b>Subset</b>: usa <i>Suggerisci parametri</i>; piu' grande = piu'
        preciso ma meno dettagliato. Deve stare almeno 2 volte nel fenomeno
        piu' piccolo da vedere.</li>
        <li><b>Passo</b>: circa 1/3 del subset.</li>
        <li><b>Spostamento max</b>: un ordine di grandezza in eccesso basta
        (la piramide fa il resto).</li>
        </ul>
        <h3>Distacchi e zone rumorose</h3>
        <ul>
        <li>Dove la roccia si e' staccata non c'e' uno spostamento: i punti
        sono classificati <b>Cambiamento</b> e raggruppati in zone con area
        in m&sup2;, mai riempiti con valori inventati.</li>
        <li>La <b>doppia correlazione</b> distingue un cambio di colore
        (bagnato, ombra: spostamento misurato lo stesso) da un cambio di
        forma (distacco).</li>
        <li><b>Vegetazione</b> e <b>acqua</b> vengono escluse con maschere
        automatiche dal colore: controllale con <i>Anteprima maschere</i>.
        Mare e spiaggia si possono escludere anche a mano.</li>
        </ul>
        <h3>Altri metodi</h3>
        <ul>
        <li><b>NCC</b>: piu' veloce, meno preciso dove il terreno si deforma.</li>
        <li><b>Optical Flow</b>: anteprima in pochi secondi.</li>
        <li><b>Phase Correlation</b> e <b>Feature Matching</b>: sperimentali,
        solo per casi particolari.</li>
        </ul>
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Guida scelta metodo e parametri")
        dlg.resize(700, 620)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(guide_html)
        lay.addWidget(tb)
        btn = QPushButton("Chiudi")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec_()
