"""Image enhancement filter panel with live before/after preview."""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QListWidget, QListWidgetItem, QSplitter,
    QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from dic_app.gui.image_viewer import ImageViewer
from dic_app.core.preprocessing import FilterPipeline, FILTER_REGISTRY
from dic_app.core.filter_advisor import copy_pipeline
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)

# Italian label, when to use it, and whether it is safe for DIC
# (tested with known motions: the "not recommended" ones change grey
# levels non-linearly and increase the DIC error).
FILTER_INFO_IT = {
    'Wallis Filter': ("Wallis - uniforma luce e contrasto locali",
                      "Luce diversa tra le foto, ombre, parete bagnata. "
                      "Finestra ~1.5 volte il subset.", True),
    'CLAHE': ("CLAHE - contrasto locale adattivo",
              "Foschia, contrasto basso, zone in ombra profonda.", True),
    'Gaussian Blur': ("Gaussiano - attenua il rumore",
                      "Foto rumorose. Usare sigma basso (0.6-1.4): troppo "
                      "sfocato riduce la tessitura.", True),
    'Non-Local Means': ("Non-Local Means - riduzione rumore forte (lento)",
                        "Foto molto rumorose (notturne, ISO alti).", True),
    'Bilateral Filter': ("Bilaterale - riduce il rumore preservando i bordi",
                         "Rumore con bordi netti; nei test peggiora spesso "
                         "la precisione.", True),
    'Median Filter': ("Mediano - rimuove pixel isolati",
                      "Solo per rumore impulsivo (pixel bianchi/neri).", True),
    'Normalize Brightness': ("Normalizza luminosita' media",
                             "Inutile per il DIC (gia' invariante alla "
                             "luminosita'), utile solo per la visualizzazione.", True),
    'Gamma Correction': ("Correzione gamma",
                         "Immagini molto scure o chiare; preferire CLAHE.", True),
    'Histogram Equalization': ("Equalizzazione istogramma globale",
                               "SCONSIGLIATO per il DIC: altera i livelli di "
                               "grigio, nei test perde fino al 70% dei punti.", False),
    'Unsharp Mask': ("Nitidezza (unsharp mask)",
                     "SCONSIGLIATO per il DIC: amplifica il rumore.", False),
    'Laplacian Sharpening': ("Nitidezza laplaciana",
                             "SCONSIGLIATO per il DIC: amplifica il rumore.", False),
    'Adaptive Threshold Enhancement': ("Soglia adattiva",
                                       "SCONSIGLIATO per il DIC: introduce "
                                       "salti artificiali di grigio.", False),
    'Morphological Opening': ("Apertura morfologica",
                              "SCONSIGLIATO per il DIC: cancella dettagli "
                              "della tessitura.", False),
    'Morphological Closing': ("Chiusura morfologica",
                              "SCONSIGLIATO per il DIC: cancella dettagli "
                              "della tessitura.", False),
}


def filter_label(name):
    return FILTER_INFO_IT.get(name, (name, "", True))[0]


class FilterParamWidget(QFrame):
    """Widget for a single filter's parameters."""

    params_changed = pyqtSignal()

    def __init__(self, filter_name, params, param_types, parent=None):
        super().__init__(parent)
        self.filter_name = filter_name
        self.param_widgets = {}
        self.setFrameShape(QFrame.StyledPanel)

        layout = QGridLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        row = 0
        for param_name, value in params.items():
            ptype_info = param_types.get(param_name)
            if ptype_info is None:
                continue

            label = QLabel(param_name.replace('_', ' ').title() + ':')
            layout.addWidget(label, row, 0)

            ptype = ptype_info[0]

            if ptype == 'float':
                _, pmin, pmax = ptype_info
                widget = QDoubleSpinBox()
                widget.setRange(pmin, pmax)
                widget.setSingleStep(max(0.01, (pmax - pmin) / 100))
                widget.setDecimals(2)
                widget.setValue(float(value))
                widget.valueChanged.connect(self.params_changed.emit)

            elif ptype in ('int', 'odd_int'):
                _, pmin, pmax = ptype_info
                widget = QSpinBox()
                widget.setRange(int(pmin), int(pmax))
                if ptype == 'odd_int':
                    widget.setSingleStep(2)
                    val = int(value)
                    if val % 2 == 0:
                        val += 1
                    widget.setValue(val)
                else:
                    widget.setValue(int(value))
                widget.valueChanged.connect(self.params_changed.emit)

            elif ptype == 'tuple_int':
                _, pmin, pmax = ptype_info
                widget = QSpinBox()
                widget.setRange(int(pmin), int(pmax))
                if isinstance(value, (tuple, list)):
                    widget.setValue(int(value[0]))
                else:
                    widget.setValue(int(value))
                widget.valueChanged.connect(self.params_changed.emit)
            else:
                continue

            layout.addWidget(widget, row, 1)
            self.param_widgets[param_name] = (ptype, widget)
            row += 1

    def get_params(self):
        """Get current parameter values as dict."""
        params = {}
        for param_name, (ptype, widget) in self.param_widgets.items():
            if ptype == 'tuple_int':
                val = widget.value()
                params[param_name] = (val, val)
            elif ptype == 'odd_int':
                val = widget.value()
                if val % 2 == 0:
                    val += 1
                params[param_name] = val
            elif ptype == 'float':
                params[param_name] = widget.value()
            elif ptype == 'int':
                params[param_name] = widget.value()
        return params


class FilterPanelWidget(QWidget):
    """Filter configuration panel with before/after preview.

    Features:
    - Side-by-side before/after image preview
    - Add/remove/reorder filters from registry
    - Per-filter parameter editing
    - Live preview with debouncing
    - Apply to all images
    """

    filter_applied = pyqtSignal(object)  # emits FilterPipeline

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reference_image = None
        self._all_images = []  # All loaded grayscale images for multi-image optimization
        self._deformed_image = None   # raw deformed image for the guided choice
        self._dic_params = None
        self._gsd = None
        self._pipeline = FilterPipeline()
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._do_update_preview)
        self._filter_param_widgets = []
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Image viewers (before / after) ---
        viewer_splitter = QSplitter(Qt.Horizontal)
        before_group = QGroupBox("Originale")
        before_layout = QVBoxLayout(before_group)
        self.before_viewer = ImageViewer()
        before_layout.addWidget(self.before_viewer)

        after_group = QGroupBox("Processata")
        after_layout = QVBoxLayout(after_group)
        self.after_viewer = ImageViewer()
        after_layout.addWidget(self.after_viewer)

        viewer_splitter.addWidget(before_group)
        viewer_splitter.addWidget(after_group)
        main_layout.addWidget(viewer_splitter, stretch=3)

        # --- Guided choice (recommended path) ---
        guide_group = QGroupBox("1. Scelta guidata dei filtri (consigliata)")
        gl = QVBoxLayout(guide_group)
        guide_txt = QLabel(
            "I filtri servono solo se migliorano la misura: la procedura li "
            "<b>prova con il DIC sulle tue immagini</b> e propone quello "
            "migliore (spesso nessuno), con i parametri gia' impostati.")
        guide_txt.setWordWrap(True)
        guide_txt.setStyleSheet("font-size: 11px; color: #37474f;")
        gl.addWidget(guide_txt)
        grow = QHBoxLayout()
        self.btn_wizard = QPushButton("🧭  Procedura guidata (con domande)")
        self.btn_wizard.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 8px; font-weight: bold; }")
        self.btn_wizard.setToolTip(
            "Alcune domande sulle foto (luce, rumore, contrasto), poi prova "
            "dei filtri adatti e proposta con pulsante Applica.")
        self.btn_wizard.clicked.connect(lambda: self._open_wizard(auto=False))
        grow.addWidget(self.btn_wizard)
        self.btn_auto = QPushButton("⚡  Scelta automatica")
        self.btn_auto.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "padding: 8px; font-weight: bold; }")
        self.btn_auto.setToolTip(
            "Nessuna domanda: il programma misura luce, rumore e contrasto, "
            "prova i filtri e propone il migliore.")
        self.btn_auto.clicked.connect(lambda: self._open_wizard(auto=True))
        grow.addWidget(self.btn_auto)
        gl.addLayout(grow)
        self.applied_label = QLabel("Filtri applicati alle immagini: nessuno")
        self.applied_label.setStyleSheet(
            "font-size: 11px; padding: 3px; background-color: #e8eaf6;")
        self.applied_label.setWordWrap(True)
        gl.addWidget(self.applied_label)
        self.auto_progress = self.applied_label      # legacy name
        main_layout.addWidget(guide_group)

        # --- Filter pipeline controls (manual) ---
        controls_group = QGroupBox("2. Regolazione manuale (facoltativa)")
        controls_layout = QVBoxLayout(controls_group)

        # Add filter row
        add_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        for name in FILTER_REGISTRY:
            label, when, ok = FILTER_INFO_IT.get(name, (name, "", True))
            self.filter_combo.addItem(label if ok else f"⚠ {label}", name)
            self.filter_combo.setItemData(self.filter_combo.count() - 1,
                                          when, Qt.ToolTipRole)
        self.filter_combo.currentIndexChanged.connect(self._on_combo_changed)
        add_row.addWidget(self.filter_combo, stretch=1)

        self.btn_add = QPushButton("Aggiungi Filtro")
        self.btn_add.clicked.connect(self._add_filter)
        add_row.addWidget(self.btn_add)
        controls_layout.addLayout(add_row)
        self.when_label = QLabel()
        self.when_label.setWordWrap(True)
        self.when_label.setStyleSheet("font-size: 11px; color: #616161;")
        controls_layout.addWidget(self.when_label)
        self._on_combo_changed()

        # Filter list
        self.filter_list = QListWidget()
        self.filter_list.setMaximumHeight(120)
        self.filter_list.currentRowChanged.connect(self._on_filter_selected)
        controls_layout.addWidget(self.filter_list)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_remove = QPushButton("Rimuovi")
        self.btn_remove.clicked.connect(self._remove_filter)
        btn_row.addWidget(self.btn_remove)

        self.btn_move_up = QPushButton("Su")
        self.btn_move_up.clicked.connect(self._move_up)
        btn_row.addWidget(self.btn_move_up)

        self.btn_move_down = QPushButton("Giu")
        self.btn_move_down.clicked.connect(self._move_down)
        btn_row.addWidget(self.btn_move_down)

        self.btn_clear = QPushButton("Pulisci Tutto")
        self.btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(self.btn_clear)

        controls_layout.addLayout(btn_row)

        # Parameters area
        self.params_container = QVBoxLayout()
        self.params_label = QLabel("Seleziona un filtro per modificare i parametri")
        self.params_container.addWidget(self.params_label)
        controls_layout.addLayout(self.params_container)

        main_layout.addWidget(controls_group, stretch=2)

        # Apply button (manual pipeline)
        self.btn_apply = QPushButton("Applica questi filtri a tutte le immagini")
        self.btn_apply.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 8px; font-weight: bold; }")
        self.btn_apply.clicked.connect(self._apply_pipeline)
        main_layout.addWidget(self.btn_apply)

    def set_reference_image(self, image_gray):
        """Set the reference image for preview."""
        self._reference_image = image_gray
        if image_gray is not None:
            self.before_viewer.set_image(image_gray)
            self.before_viewer.fit_in_view()
            self._schedule_preview()

    def set_all_images(self, images_gray):
        """Set all loaded grayscale images for multi-image auto-optimization.

        Parameters
        ----------
        images_gray : list of np.ndarray (H, W) uint8
            All grayscale images to evaluate during auto-optimization.
        """
        self._all_images = list(images_gray) if images_gray else []

    def set_analysis_context(self, deformed_gray=None, dic_params=None, gsd=None):
        """Raw deformed image + DIC settings used by the guided choice."""
        self._deformed_image = deformed_gray
        self._dic_params = dic_params
        self._gsd = gsd

    def _on_combo_changed(self, *_):
        name = self.filter_combo.currentData()
        label, when, ok = FILTER_INFO_IT.get(name, (name, "", True))
        self.when_label.setText(("" if ok else "⚠ ") + "Quando usarlo: " + when
                                if when else "")
        self.when_label.setStyleSheet(
            "font-size: 11px; color: %s;" % ("#616161" if ok else "#c62828"))

    def _add_filter(self):
        """Add selected filter to pipeline."""
        filter_name = self.filter_combo.currentData()
        if filter_name is None:
            return

        entry = FILTER_REGISTRY[filter_name]
        params = dict(entry['params'])
        param_types = entry['param_types']

        self._pipeline.add_step(filter_name, **params)

        param_widget = FilterParamWidget(filter_name, params, param_types)
        param_widget.params_changed.connect(self._on_params_changed)
        self._filter_param_widgets.append(param_widget)

        item = QListWidgetItem(f"{len(self._pipeline)}. {filter_label(filter_name)}")
        self.filter_list.addItem(item)
        self.filter_list.setCurrentItem(item)

        self._schedule_preview()

    def _remove_filter(self):
        """Remove selected filter from pipeline."""
        row = self.filter_list.currentRow()
        if row < 0:
            return

        self._pipeline.remove_step(row)
        self.filter_list.takeItem(row)

        widget = self._filter_param_widgets.pop(row)
        widget.deleteLater()

        self._refresh_list_labels()
        self._schedule_preview()

    def _move_up(self):
        row = self.filter_list.currentRow()
        if row <= 0:
            return
        self._pipeline.move_step(row, row - 1)
        self._filter_param_widgets[row], self._filter_param_widgets[row - 1] = \
            self._filter_param_widgets[row - 1], self._filter_param_widgets[row]
        self._refresh_list_labels()
        self.filter_list.setCurrentRow(row - 1)
        self._schedule_preview()

    def _move_down(self):
        row = self.filter_list.currentRow()
        if row < 0 or row >= len(self._pipeline) - 1:
            return
        self._pipeline.move_step(row, row + 1)
        self._filter_param_widgets[row], self._filter_param_widgets[row + 1] = \
            self._filter_param_widgets[row + 1], self._filter_param_widgets[row]
        self._refresh_list_labels()
        self.filter_list.setCurrentRow(row + 1)
        self._schedule_preview()

    def _clear_all(self):
        self._pipeline.clear()
        self.filter_list.clear()
        for w in self._filter_param_widgets:
            w.deleteLater()
        self._filter_param_widgets.clear()
        self._schedule_preview()

    def _refresh_list_labels(self):
        for i, step in enumerate(self._pipeline.steps):
            item = self.filter_list.item(i)
            if item:
                item.setText(f"{i + 1}. {filter_label(step['filter_name'])}")

    def _on_filter_selected(self, row):
        """Show parameters for the selected filter."""
        for i in reversed(range(self.params_container.count())):
            w = self.params_container.itemAt(i).widget()
            if w:
                w.setParent(None)

        if 0 <= row < len(self._filter_param_widgets):
            widget = self._filter_param_widgets[row]
            self.params_container.addWidget(widget)
        else:
            label = QLabel("Seleziona un filtro per modificare i parametri")
            self.params_container.addWidget(label)

    def _on_params_changed(self):
        """Update pipeline parameters from widget and refresh preview."""
        row = self.filter_list.currentRow()
        if 0 <= row < len(self._filter_param_widgets):
            params = self._filter_param_widgets[row].get_params()
            self._pipeline.update_step_params(row, **params)
        self._schedule_preview()

    def _schedule_preview(self):
        """Debounced preview update."""
        self._preview_timer.start()

    def _do_update_preview(self):
        """Apply pipeline and update after viewer."""
        if self._reference_image is None:
            return
        try:
            result = self._pipeline.apply(self._reference_image)
            self.after_viewer.set_image(result)
            self.after_viewer.fit_in_view()
        except Exception as e:
            print(f"Filter preview error: {e}")

    def _apply_pipeline(self):
        """Emit signal to apply pipeline to all images.

        A copy is emitted: later edits in the panel must not silently
        change the pipeline stored in the project (whose images were
        filtered with the applied version).
        """
        applied = copy_pipeline(self._pipeline)
        self._show_applied(applied)
        self.filter_applied.emit(applied)

    def _show_applied(self, pipeline):
        names = [filter_label(s['filter_name']) for s in pipeline.steps]
        self.applied_label.setText(
            "Filtri applicati alle immagini: "
            + (" → ".join(names) if names else "nessuno (immagini originali)"))

    def get_pipeline(self):
        """Return current filter pipeline."""
        return self._pipeline

    def set_pipeline(self, pipeline):
        """Restore a saved pipeline (the panel works on its own copy)."""
        pipeline = copy_pipeline(pipeline)
        self._clear_all()
        self._pipeline = pipeline
        self._show_applied(pipeline)
        for step in pipeline.steps:
            filter_name = step['filter_name']
            params = step['params']
            entry = FILTER_REGISTRY.get(filter_name)
            if entry is None:
                continue

            param_widget = FilterParamWidget(
                filter_name, params, entry['param_types'])
            param_widget.params_changed.connect(self._on_params_changed)
            self._filter_param_widgets.append(param_widget)

            item = QListWidgetItem(
                f"{self.filter_list.count() + 1}. {filter_label(filter_name)}")
            self.filter_list.addItem(item)

        self._schedule_preview()

    # ------------------------------------------------------------------
    # Guided choice (questions + DIC tests + Apply button)
    # ------------------------------------------------------------------

    def _open_wizard(self, auto=False):
        """Open the guided choice; on Apply, set and apply the pipeline."""
        from PyQt5.QtWidgets import QMessageBox, QDialog
        from dic_app.gui.filter_wizard import FilterWizardDialog

        if self._reference_image is None:
            QMessageBox.warning(self, "Filtri",
                                "Caricare prima le immagini nella scheda Progetto.")
            return
        dlg = FilterWizardDialog(self._reference_image, self._deformed_image,
                                 self._dic_params, self._gsd,
                                 auto_start=auto, parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.chosen_pipeline is not None:
            self.set_pipeline(dlg.chosen_pipeline)
            self._apply_pipeline()
