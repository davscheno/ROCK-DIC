"""DIC algorithm parameter configuration panel."""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QStackedWidget, QFormLayout, QMessageBox, QTextBrowser
)
from PyQt5.QtCore import pyqtSignal, Qt
from dic_app.core.dic_engine import DICParameters, DICMethod, SubPixelMethod
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)


class ParamsPanelWidget(QWidget):
    """DIC parameter configuration panel.

    Allows selecting the DIC algorithm and configuring method-specific
    parameters. Shows a visual subset size preview hint.
    """

    params_changed = pyqtSignal(object)  # emits DICParameters

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ref_image = None       # stored for auto-guess
        self._def_image = None       # stored for auto-guess
        self._image_shape = None
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
        self.method_desc.setStyleSheet("color: #546e7a; font-style: italic;")
        algo_layout.addRow(self.method_desc)

        main_layout.addWidget(algo_group)

        # --- Guide & Auto-guess buttons ---
        guide_row = QHBoxLayout()

        self.btn_guide = QPushButton("Guida Scelta Algoritmo")
        self.btn_guide.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; "
            "padding: 6px 14px; font-weight: bold; }")
        self.btn_guide.setToolTip(
            "Mostra una guida interattiva per scegliere l'algoritmo DIC "
            "e i parametri ottimali per il tuo caso d'uso.")
        self.btn_guide.clicked.connect(self._show_algorithm_guide)
        guide_row.addWidget(self.btn_guide)

        self.btn_auto_guess = QPushButton("Auto-Stima Parametri")
        self.btn_auto_guess.setStyleSheet(
            "QPushButton { background-color: #ff8f00; color: white; "
            "padding: 6px 14px; font-weight: bold; }")
        self.btn_auto_guess.setToolTip(
            "Analizza le immagini caricate e propone automaticamente "
            "algoritmo e parametri ottimali.")
        self.btn_auto_guess.clicked.connect(self._auto_guess_parameters)
        guide_row.addWidget(self.btn_auto_guess)

        main_layout.addLayout(guide_row)

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
        self._image_shape = image_shape
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

    def set_images_for_guess(self, ref_gray, def_gray):
        """Store reference/deformed images for auto-parameter estimation."""
        self._ref_image = ref_gray
        self._def_image = def_gray
        if ref_gray is not None:
            self._image_shape = ref_gray.shape

    # ------------------------------------------------------------------
    # Algorithm Guide
    # ------------------------------------------------------------------

    def _show_algorithm_guide(self):
        """Show interactive algorithm selection guide dialog."""
        guide_html = """
        <h2 style="color: #1565c0;">Guida alla Scelta dell'Algoritmo DIC</h2>

        <h3>1. Template Matching NCC</h3>
        <p><b>Quando usarlo:</b> Piccoli spostamenti (1-50 pixel), buona texture
        superficiale, roccia con pattern visibili, vegetazione, terreno con
        granulometria variabile.</p>
        <p><b>Vantaggi:</b> Massima precisione sub-pixel (fino a 0.01 px),
        misura affidabile della qualita tramite NCC.</p>
        <p><b>Svantaggi:</b> Lento per grandi immagini, limitato dal raggio di
        ricerca impostato.</p>
        <p><b>Parametri chiave:</b></p>
        <ul>
        <li><b>Subset</b>: 21-41 px per buona texture, 51-81 px per texture scarsa</li>
        <li><b>Raggio ricerca</b>: impostare >= spostamento atteso massimo</li>
        <li><b>Sub-pixel</b>: Gaussiano (veloce e preciso)</li>
        </ul>
        <hr>

        <h3>2. Optical Flow (Farneback)</h3>
        <p><b>Quando usarlo:</b> Spostamenti moderati (5-200 pixel), serve una
        mappa densa e veloce, analisi preliminare rapida.</p>
        <p><b>Vantaggi:</b> Molto veloce, produce campo denso pixel-per-pixel,
        gestisce spostamenti grandi tramite piramidi.</p>
        <p><b>Svantaggi:</b> Meno preciso in sub-pixel rispetto a NCC,
        sensibile a variazioni di illuminazione.</p>
        <p><b>Parametri chiave:</b></p>
        <ul>
        <li><b>Livelli piramide</b>: 3-5 per spostamenti piccoli, 6-8 per grandi</li>
        <li><b>Dim. finestra</b>: 15-21 px (piu grande = piu robusto)</li>
        <li><b>Iterazioni</b>: 3-5 (piu = piu preciso ma lento)</li>
        </ul>
        <hr>

        <h3>3. Phase Correlation (FFT)</h3>
        <p><b>Quando usarlo:</b> Traslazioni uniformi (frane con movimento rigido),
        analisi rapida, spostamenti molto grandi.</p>
        <p><b>Vantaggi:</b> Estremamente veloce, robusto al rumore,
        insensibile a variazioni di illuminazione.</p>
        <p><b>Svantaggi:</b> Funziona bene solo per traslazioni rigide
        (non deformazioni complesse), precisione limitata per rotazioni.</p>
        <p><b>Parametri chiave:</b></p>
        <ul>
        <li><b>Upsampling</b>: 20 per 1/20 pixel di precisione, 100 per 0.01 px</li>
        <li><b>Subset</b>: 41-81 px (subset piu grandi = piu affidabile)</li>
        </ul>
        <hr>

        <h3>4. Feature Matching (SIFT/ORB)</h3>
        <p><b>Quando usarlo:</b> Spostamenti molto grandi (> 100 pixel),
        texture scarsa, immagini con pochi punti di riferimento.</p>
        <p><b>Vantaggi:</b> Nessun limite di spostamento, robusto a rotazioni
        e scala, funziona con pochi punti di texture.</p>
        <p><b>Svantaggi:</b> Risultato sparso (interpolato), meno preciso,
        non adatto per piccoli spostamenti sub-pixel.</p>
        <p><b>Parametri chiave:</b></p>
        <ul>
        <li><b>Max features</b>: 5000-20000 (piu = migliore copertura)</li>
        <li><b>Ratio test</b>: 0.7-0.8 (piu basso = meno falsi positivi)</li>
        </ul>
        <hr>

        <h3 style="color: #2e7d32;">Raccomandazione per Monitoraggio Frane/Falesie</h3>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #e8f5e9;">
            <th>Scenario</th><th>Algoritmo</th><th>Motivazione</th>
        </tr>
        <tr>
            <td>Distacchi massi da falesia</td>
            <td><b>Template NCC</b></td>
            <td>Massima precisione, texture rocciosa favorevole</td>
        </tr>
        <tr>
            <td>Frana lenta (mm/anno)</td>
            <td><b>Template NCC</b></td>
            <td>Precisione sub-pixel critica per piccoli spostamenti</td>
        </tr>
        <tr>
            <td>Frana rapida (m/evento)</td>
            <td><b>Optical Flow</b></td>
            <td>Veloce, gestisce grandi spostamenti</td>
        </tr>
        <tr>
            <td>Analisi preliminare rapida</td>
            <td><b>Optical Flow</b></td>
            <td>Risultato in secondi per valutazione iniziale</td>
        </tr>
        <tr>
            <td>Movimento rigido uniforme</td>
            <td><b>Phase Correlation</b></td>
            <td>Velocissimo per traslazioni pure</td>
        </tr>
        <tr>
            <td>Immagini molto diverse</td>
            <td><b>Feature Matching</b></td>
            <td>Robusto a grandi differenze tra le immagini</td>
        </tr>
        </table>
        """

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Guida Scelta Algoritmo DIC")
        dialog.setTextFormat(Qt.RichText)
        dialog.setText(guide_html)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.setMinimumWidth(700)
        dialog.exec()

    # ------------------------------------------------------------------
    # Auto-Guess Parameters
    # ------------------------------------------------------------------

    def _auto_guess_parameters(self):
        """Analyze loaded images and suggest optimal DIC parameters."""
        if self._ref_image is None or self._def_image is None:
            QMessageBox.warning(
                self, "Immagini non disponibili",
                "Carica le immagini e seleziona riferimento/deformata "
                "prima di usare l'auto-stima.")
            return

        try:
            import cv2
            ref = self._ref_image
            deformed = self._def_image
            h, w = ref.shape[:2]

            # --- 1. Analyze texture quality ---
            # Local variance as texture measure
            blur = cv2.GaussianBlur(ref, (31, 31), 0).astype(np.float64)
            sq_blur = cv2.GaussianBlur(
                ref.astype(np.float64) ** 2, (31, 31), 0)
            local_var = np.mean(sq_blur - blur ** 2)

            # Gradient magnitude (Sobel)
            gx = cv2.Sobel(ref, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(ref, cv2.CV_64F, 0, 1, ksize=3)
            mean_gradient = np.mean(np.sqrt(gx**2 + gy**2))

            # Texture quality score (0-1)
            texture_score = min(1.0, mean_gradient / 30.0)

            # --- 2. Estimate displacement magnitude ---
            # Quick optical flow for displacement estimation
            flow = cv2.calcOpticalFlowFarneback(
                ref, deformed, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

            # Robust displacement estimate (95th percentile)
            p95_disp = float(np.percentile(flow_mag, 95))
            p99_disp = float(np.percentile(flow_mag, 99))
            median_disp = float(np.median(flow_mag))

            # --- 3. Choose algorithm ---
            if p95_disp < 2.0 and texture_score > 0.3:
                # Very small displacements, decent texture → NCC
                method = DICMethod.TEMPLATE_NCC
                method_reason = (
                    "Spostamenti molto piccoli ({:.1f} px al 95%), "
                    "texture sufficiente → Template NCC per massima "
                    "precisione sub-pixel.".format(p95_disp))
            elif p95_disp < 50 and texture_score > 0.2:
                # Small-moderate displacements → NCC
                method = DICMethod.TEMPLATE_NCC
                method_reason = (
                    "Spostamenti moderati ({:.1f} px al 95%), "
                    "buona texture → Template NCC.".format(p95_disp))
            elif p95_disp < 200:
                # Large displacements → Optical Flow
                method = DICMethod.OPTICAL_FLOW_FARNEBACK
                method_reason = (
                    "Spostamenti grandi ({:.1f} px al 95%) → "
                    "Optical Flow per velocita e copertura densa."
                    .format(p95_disp))
            elif texture_score < 0.15:
                # Very poor texture → Feature Matching
                method = DICMethod.FEATURE_MATCHING
                method_reason = (
                    "Texture molto scarsa (score {:.2f}), spostamenti "
                    "grandi → Feature Matching.".format(texture_score))
            else:
                # Very large displacements → Optical Flow
                method = DICMethod.OPTICAL_FLOW_FARNEBACK
                method_reason = (
                    "Spostamenti molto grandi ({:.1f} px al 95%) → "
                    "Optical Flow.".format(p95_disp))

            # --- 4. Calculate optimal parameters ---
            # Subset size: based on texture quality
            if texture_score > 0.5:
                subset = 21  # good texture
            elif texture_score > 0.3:
                subset = 31  # moderate texture
            elif texture_score > 0.15:
                subset = 51  # poor texture
            else:
                subset = 71  # very poor texture

            # Ensure odd
            if subset % 2 == 0:
                subset += 1

            # Ensure subset < image dimensions
            subset = min(subset, min(h, w) // 4)
            if subset % 2 == 0:
                subset += 1

            # Step size: ~1/3 to 1/2 of subset for good overlap
            step = max(3, subset // 4)

            # Search radius: based on displacement magnitude + margin
            search_r = max(10, int(p99_disp * 1.5) + 10)
            search_r = min(search_r, min(h, w) // 4)

            # Correlation threshold
            if texture_score > 0.4:
                corr_thresh = 0.7  # high texture → strict
            elif texture_score > 0.2:
                corr_thresh = 0.6  # moderate → standard
            else:
                corr_thresh = 0.5  # low texture → relaxed

            # Border margin
            border = max(10, subset)

            # OF levels based on displacement
            of_levels = max(3, min(8, int(np.log2(max(1, p95_disp))) + 2))

            # --- 5. Apply parameters ---
            # Set method
            for i in range(self.method_combo.count()):
                if self.method_combo.itemData(i) == method:
                    self.method_combo.setCurrentIndex(i)
                    break

            self.subset_spin.setValue(subset)
            self.step_spin.setValue(step)
            self.threshold_spin.setValue(corr_thresh)
            self.border_margin_spin.setValue(border)
            self.search_rx.setValue(search_r)
            self.search_ry.setValue(search_r)
            self.of_levels.setValue(of_levels)

            # --- 6. Show summary ---
            summary = (
                f"<h3>Risultato Auto-Stima Parametri</h3>"
                f"<p><b>Analisi immagine:</b></p>"
                f"<ul>"
                f"<li>Dimensione: {w} x {h} pixel</li>"
                f"<li>Qualita texture: {texture_score:.2f} "
                f"({'buona' if texture_score > 0.4 else 'moderata' if texture_score > 0.2 else 'scarsa'})</li>"
                f"<li>Gradiente medio: {mean_gradient:.1f}</li>"
                f"<li>Varianza locale: {local_var:.1f}</li>"
                f"</ul>"
                f"<p><b>Stima spostamento:</b></p>"
                f"<ul>"
                f"<li>Mediana: {median_disp:.1f} px</li>"
                f"<li>95° percentile: {p95_disp:.1f} px</li>"
                f"<li>99° percentile: {p99_disp:.1f} px</li>"
                f"</ul>"
                f"<p><b>Algoritmo scelto:</b> {method.value}</p>"
                f"<p><i>{method_reason}</i></p>"
                f"<p><b>Parametri impostati:</b></p>"
                f"<ul>"
                f"<li>Subset: {subset} px</li>"
                f"<li>Passo griglia: {step} px</li>"
                f"<li>Raggio ricerca: {search_r} px</li>"
                f"<li>Soglia correlazione: {corr_thresh}</li>"
                f"<li>Margine bordi: {border} px</li>"
                f"</ul>"
                f"<p style='color: #1565c0;'><b>Nota:</b> Questi parametri "
                f"sono un punto di partenza. Regola manualmente in base "
                f"ai risultati ottenuti.</p>"
            )

            msg = QMessageBox(self)
            msg.setWindowTitle("Auto-Stima Parametri DIC")
            msg.setTextFormat(Qt.RichText)
            msg.setText(summary)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()

        except Exception as e:
            QMessageBox.critical(
                self, "Errore Auto-Stima",
                f"Errore durante l'analisi automatica:\n{e}")
