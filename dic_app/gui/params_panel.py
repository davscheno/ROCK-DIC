"""DIC algorithm parameter configuration panel."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QStackedWidget, QFormLayout
)
from PyQt5.QtCore import pyqtSignal
from dic_app.core.dic_engine import DICParameters, DICMethod, SubPixelMethod


class ParamsPanelWidget(QWidget):
    """DIC parameter configuration panel.

    Allows selecting the DIC algorithm and configuring method-specific
    parameters. Shows a visual subset size preview hint.
    """

    params_changed = pyqtSignal(object)  # emits DICParameters

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Algorithm selection ---
        algo_group = QGroupBox("Algoritmo DIC")
        algo_layout = QFormLayout(algo_group)

        self.method_combo = QComboBox()
        self.method_combo.addItem("Template Matching (NCC)", DICMethod.TEMPLATE_NCC)
        self.method_combo.addItem("Optical Flow (Farneback)", DICMethod.OPTICAL_FLOW_FARNEBACK)
        self.method_combo.addItem("Phase Correlation (FFT)", DICMethod.PHASE_CORRELATION)
        self.method_combo.addItem("Feature Matching (ORB/SIFT)", DICMethod.FEATURE_MATCHING)
        algo_layout.addRow("Metodo:", self.method_combo)

        self.method_desc = QLabel()
        self.method_desc.setWordWrap(True)
        self.method_desc.setStyleSheet("color: #888; font-style: italic;")
        algo_layout.addRow(self.method_desc)

        main_layout.addWidget(algo_group)

        # --- Common parameters ---
        common_group = QGroupBox("Parametri Comuni")
        common_layout = QFormLayout(common_group)

        self.subset_spin = QSpinBox()
        self.subset_spin.setRange(5, 201)
        self.subset_spin.setSingleStep(2)
        self.subset_spin.setValue(31)
        self.subset_spin.setToolTip("Dimensione del subset (template) in pixel. Deve essere dispari.")
        common_layout.addRow("Dim. Subset (px):", self.subset_spin)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100)
        self.step_spin.setValue(5)
        self.step_spin.setToolTip("Passo della griglia di analisi in pixel.")
        common_layout.addRow("Passo Griglia (px):", self.step_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.6)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setToolTip("Soglia minima di correlazione per accettare un match.")
        common_layout.addRow("Soglia Correlazione:", self.threshold_spin)

        self.border_margin_spin = QSpinBox()
        self.border_margin_spin.setRange(0, 500)
        self.border_margin_spin.setValue(10)
        self.border_margin_spin.setSingleStep(5)
        self.border_margin_spin.setSuffix(" px")
        self.border_margin_spin.setToolTip(
            "Margine in pixel dal bordo dell'immagine entro cui i risultati "
            "vengono scartati. Consigliato >= subset_size quando si usa "
            "l'allineamento, per eliminare artefatti ai bordi.")
        common_layout.addRow("Margine Bordi (px):", self.border_margin_spin)

        # Info labels
        self.grid_info = QLabel()
        common_layout.addRow(self.grid_info)

        main_layout.addWidget(common_group)

        # --- Method-specific parameters (stacked) ---
        self.method_stack = QStackedWidget()

        # NCC params
        ncc_widget = QWidget()
        ncc_layout = QFormLayout(ncc_widget)

        self.search_rx = QSpinBox()
        self.search_rx.setRange(5, 500)
        self.search_rx.setValue(50)
        ncc_layout.addRow("Raggio Ricerca X (px):", self.search_rx)

        self.search_ry = QSpinBox()
        self.search_ry.setRange(5, 500)
        self.search_ry.setValue(50)
        ncc_layout.addRow("Raggio Ricerca Y (px):", self.search_ry)

        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItem("Gaussiano (consigliato)", SubPixelMethod.GAUSSIAN)
        self.subpixel_combo.addItem("Parabolico", SubPixelMethod.PARABOLIC)
        self.subpixel_combo.addItem("Bicubico (lento)", SubPixelMethod.BICUBIC)
        self.subpixel_combo.addItem("Nessuno (solo pixel interi)", SubPixelMethod.NONE)
        ncc_layout.addRow("Metodo Sub-Pixel:", self.subpixel_combo)

        self.method_stack.addWidget(ncc_widget)  # index 0

        # Optical Flow params
        of_widget = QWidget()
        of_layout = QFormLayout(of_widget)

        self.of_pyr_scale = QDoubleSpinBox()
        self.of_pyr_scale.setRange(0.1, 0.9)
        self.of_pyr_scale.setValue(0.5)
        self.of_pyr_scale.setSingleStep(0.1)
        of_layout.addRow("Scala Piramide:", self.of_pyr_scale)

        self.of_levels = QSpinBox()
        self.of_levels.setRange(1, 10)
        self.of_levels.setValue(5)
        of_layout.addRow("Livelli Piramide:", self.of_levels)

        self.of_winsize = QSpinBox()
        self.of_winsize.setRange(3, 51)
        self.of_winsize.setSingleStep(2)
        self.of_winsize.setValue(15)
        of_layout.addRow("Dim. Finestra:", self.of_winsize)

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
        of_layout.addRow("Poly Sigma:", self.of_poly_sigma)

        self.method_stack.addWidget(of_widget)  # index 1

        # Phase Correlation params
        phase_widget = QWidget()
        phase_layout = QFormLayout(phase_widget)

        self.upsample_spin = QSpinBox()
        self.upsample_spin.setRange(1, 100)
        self.upsample_spin.setValue(20)
        self.upsample_spin.setToolTip("Fattore di upsampling per precisione sub-pixel (1/N pixel)")
        phase_layout.addRow("Fattore Upsampling:", self.upsample_spin)

        self.method_stack.addWidget(phase_widget)  # index 2

        # Feature Matching params
        feat_widget = QWidget()
        feat_layout = QFormLayout(feat_widget)

        self.max_features = QSpinBox()
        self.max_features.setRange(100, 100000)
        self.max_features.setValue(10000)
        self.max_features.setSingleStep(1000)
        feat_layout.addRow("Max Features:", self.max_features)

        self.match_ratio = QDoubleSpinBox()
        self.match_ratio.setRange(0.1, 1.0)
        self.match_ratio.setValue(0.75)
        self.match_ratio.setSingleStep(0.05)
        self.match_ratio.setToolTip("Soglia ratio test di Lowe (piu basso = piu stringente)")
        feat_layout.addRow("Ratio Test:", self.match_ratio)

        self.method_stack.addWidget(feat_widget)  # index 3

        main_layout.addWidget(self.method_stack)

        # --- Estimation info ---
        info_group = QGroupBox("Informazioni Stima")
        info_layout = QVBoxLayout(info_group)
        self.estimation_label = QLabel()
        self.estimation_label.setWordWrap(True)
        info_layout.addWidget(self.estimation_label)
        main_layout.addWidget(info_group)

        main_layout.addStretch()

        self._update_method_ui(0)

    def _connect_signals(self):
        self.method_combo.currentIndexChanged.connect(self._update_method_ui)
        self.subset_spin.valueChanged.connect(self._on_params_changed)
        self.step_spin.valueChanged.connect(self._on_params_changed)
        self.threshold_spin.valueChanged.connect(self._on_params_changed)
        self.border_margin_spin.valueChanged.connect(self._on_params_changed)
        self.search_rx.valueChanged.connect(self._on_params_changed)
        self.search_ry.valueChanged.connect(self._on_params_changed)

    def _update_method_ui(self, index):
        """Switch stacked widget and update description."""
        self.method_stack.setCurrentIndex(index)

        descriptions = {
            0: ("Template Matching NCC: confronta subset dell'immagine di riferimento "
                "con finestre di ricerca nell'immagine deformata. Ideale per piccoli "
                "spostamenti con buona texture superficiale."),
            1: ("Optical Flow Farneback: calcola il flusso ottico denso pixel-per-pixel. "
                "Veloce e produce mappe dense, adatto per spostamenti moderati."),
            2: ("Phase Correlation FFT: correlazione nel dominio della frequenza. "
                "Molto veloce, robusto per grandi traslazioni uniformi."),
            3: ("Feature Matching: rileva keypoint ORB/SIFT e li confronta. "
                "Funziona con grandi spostamenti e texture scarsa."),
        }
        self.method_desc.setText(descriptions.get(index, ""))
        self._on_params_changed()

    def _on_params_changed(self):
        """Update info labels and emit signal."""
        subset = self.subset_spin.value()
        if subset % 2 == 0:
            subset += 1
            self.subset_spin.setValue(subset)

        step = self.step_spin.value()
        self.grid_info.setText(
            f"Risoluzione spaziale: {step} pixel tra i punti di analisi")

        self.params_changed.emit(self.get_parameters())

    def get_parameters(self) -> DICParameters:
        """Collect all widget values into DICParameters."""
        subset = self.subset_spin.value()
        if subset % 2 == 0:
            subset += 1

        return DICParameters(
            method=self.method_combo.currentData(),
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
        )

    def set_parameters(self, params: DICParameters):
        """Set widget values from DICParameters."""
        # Method
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

    def update_estimation(self, image_shape):
        """Update computation estimation based on image size and parameters."""
        if image_shape is None:
            return
        h, w = image_shape[:2]
        params = self.get_parameters()
        half = params.subset_size // 2
        ny = len(range(half, h - half, params.step_size))
        nx = len(range(half, w - half, params.step_size))
        total = ny * nx

        self.estimation_label.setText(
            f"Immagine: {w} x {h} pixel\n"
            f"Punti griglia: {nx} x {ny} = {total:,} punti\n"
            f"Subset: {params.subset_size} x {params.subset_size} pixel"
        )
