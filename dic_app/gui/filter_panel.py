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
from dic_app.core.auto_filter import AutoFilterOptimizer, compute_quality_metrics


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

        # --- Filter pipeline controls ---
        controls_group = QGroupBox("Pipeline Filtri")
        controls_layout = QVBoxLayout(controls_group)

        # Add filter row
        add_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        for name, info in FILTER_REGISTRY.items():
            self.filter_combo.addItem(f"{name} - {info['description']}", name)
        add_row.addWidget(self.filter_combo, stretch=1)

        self.btn_add = QPushButton("Aggiungi Filtro")
        self.btn_add.clicked.connect(self._add_filter)
        add_row.addWidget(self.btn_add)
        controls_layout.addLayout(add_row)

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

        # Auto-optimize button
        auto_group = QGroupBox("Ottimizzazione Automatica")
        auto_layout = QVBoxLayout(auto_group)

        self.btn_auto = QPushButton("🔍  Auto-Ottimizza Filtri")
        self.btn_auto.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "padding: 8px; font-weight: bold; font-size: 13px; }")
        self.btn_auto.setToolTip(
            "Testa automaticamente diverse combinazioni di filtri "
            "e seleziona quella che produce la migliore qualità "
            "per l'analisi DIC (texture, contrasto, nitidezza).")
        self.btn_auto.clicked.connect(self._run_auto_optimize)
        auto_layout.addWidget(self.btn_auto)

        self.auto_progress = QLabel("")
        self.auto_progress.setStyleSheet("font-size: 11px; color: #666;")
        self.auto_progress.setWordWrap(True)
        auto_layout.addWidget(self.auto_progress)

        main_layout.addWidget(auto_group)

        # Apply button
        self.btn_apply = QPushButton("Applica Pipeline a Tutte le Immagini")
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

        item = QListWidgetItem(f"{len(self._pipeline)}. {filter_name}")
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
                item.setText(f"{i + 1}. {step['filter_name']}")

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
        """Emit signal to apply pipeline to all images."""
        self.filter_applied.emit(self._pipeline)

    def get_pipeline(self):
        """Return current filter pipeline."""
        return self._pipeline

    def set_pipeline(self, pipeline):
        """Restore a saved pipeline."""
        self._clear_all()
        self._pipeline = pipeline
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

            item = QListWidgetItem(f"{self.filter_list.count() + 1}. {filter_name}")
            self.filter_list.addItem(item)

        self._schedule_preview()

    # ------------------------------------------------------------------
    # Auto-optimization
    # ------------------------------------------------------------------

    def _run_auto_optimize(self):
        """Run automatic filter optimization on all loaded images."""
        from PyQt5.QtWidgets import QApplication, QMessageBox

        if self._reference_image is None:
            QMessageBox.warning(self, "Errore",
                                "Caricare prima un'immagine di riferimento.")
            return

        # Use all images if available, otherwise just the reference
        if self._all_images and len(self._all_images) > 0:
            images_to_test = self._all_images
        else:
            images_to_test = [self._reference_image]

        n_imgs = len(images_to_test)
        self.btn_auto.setEnabled(False)
        self.auto_progress.setText(
            f"Analisi in corso su {n_imgs} immagini...")
        QApplication.processEvents()

        optimizer = AutoFilterOptimizer(subset_size=31)

        def on_progress(pct, msg):
            self.auto_progress.setText(f"[{pct}%] {msg}")
            QApplication.processEvents()

        optimizer.set_progress_callback(on_progress)

        try:
            result = optimizer.optimize(
                self._reference_image,
                images=images_to_test)

            # Build result summary
            summary = f"Migliore: {result.best_name}\n"
            summary += f"Punteggio medio: {result.best_score:.4f} "
            summary += f"({result.improvement_percent:+.1f}% vs originale)\n"
            summary += f"Valutato su: {n_imgs} immagini\n"
            summary += f"Tempo: {result.computation_time_s:.1f}s\n\n"
            summary += "Classifica:\n"
            for i, r in enumerate(result.all_results[:5]):
                marker = " ** " if i == 0 else "    "
                summary += f"{marker}{i + 1}. {r['name']} ({r['score']:.4f})\n"

            # Metrics comparison (reference image)
            bm = result.baseline_metrics
            am = result.best_metrics
            summary += f"\nMetriche rif. (originale -> ottimizzata):\n"
            summary += f"  Gradiente:  {bm.mean_gradient:.1f} -> {am.mean_gradient:.1f}\n"
            summary += f"  Contrasto:  {bm.local_contrast:.1f} -> {am.local_contrast:.1f}\n"
            summary += f"  Entropia:   {bm.entropy:.2f} -> {am.entropy:.2f}\n"
            summary += f"  Nitidezza:  {bm.laplacian_var:.0f} -> {am.laplacian_var:.0f}\n"
            summary += f"  SNR:        {bm.snr_estimate:.1f} -> {am.snr_estimate:.1f}\n"
            summary += f"  NCC score:  {bm.ncc_self_score:.4f} -> {am.ncc_self_score:.4f}\n"

            self.auto_progress.setText(summary)

            # Ask user if they want to apply
            reply = QMessageBox.question(
                self, "Ottimizzazione Completata",
                f"Pipeline migliore: {result.best_name}\n"
                f"Miglioramento: {result.improvement_percent:+.1f}%\n"
                f"(valutata su {n_imgs} immagini)\n\n"
                f"Applicare questa pipeline?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)

            if reply == QMessageBox.Yes:
                self.set_pipeline(result.best_pipeline)
                self._schedule_preview()

        except Exception as e:
            self.auto_progress.setText(f"Errore: {e}")
            QMessageBox.critical(self, "Errore", f"Ottimizzazione fallita:\n{e}")

        finally:
            self.btn_auto.setEnabled(True)
