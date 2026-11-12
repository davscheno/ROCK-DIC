"""Dialog for the preliminary crop of very large photos.

The user picks a guide image, draws a rectangle on it, selects the images
to crop (one image = "2 a 2" pair crop) and the tool coarsely aligns each
image to the guide and saves crops of identical size.
"""

import os
import threading

import cv2
from PyQt5.QtCore import Qt, QObject, QThread, QRectF, pyqtSignal
from PyQt5.QtGui import QPen, QColor, QBrush
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout, QComboBox,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QSpinBox, QLineEdit,
    QFileDialog, QCheckBox, QProgressBar, QTextEdit, QMessageBox, QWidget,
)

from dic_app.core.precrop import run_precrop, output_name, downscale_factor
from dic_app.gui.image_viewer import ImageViewer
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)

PREVIEW_MAX_DIM = 2500


class _PrecropWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            outcomes = run_precrop(
                progress=lambda p, m: self.progress.emit(p, m),
                is_cancelled=self._cancel.is_set,
                **self.kwargs)
            self.finished.emit(outcomes)
        except Exception as e:
            logger.exception("Preliminary crop failed")
            self.error.emit(str(e))


class PrecropDialog(QDialog):
    """Guide image + rectangle -> coarse alignment -> crops on disk.

    After exec() == Accepted, ``output_paths`` holds the crops (guide
    first) and ``replace_in_project`` tells whether to load them into
    the project instead of the originals.
    """

    def __init__(self, images_data, reference_index=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ritaglio preliminare - allineamento grossolano")
        self.resize(1300, 820)
        self.images_data = images_data
        self.output_paths = []
        self.replace_in_project = True
        self._preview_scale = 1.0
        self._rect_item = None
        self._drag_origin = None
        self._out_dir_user_set = False
        self._thread = None
        self._worker = None
        self._syncing = False

        self._build_ui()
        idx = reference_index if 0 <= reference_index < len(images_data) else 0
        self.guide_combo.setCurrentIndex(idx)
        self._on_guide_changed(idx)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)

        left_widget = QWidget()
        left_widget.setFixedWidth(380)
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)

        # 1. Guide
        g1 = QGroupBox("1. Immagine guida")
        l1 = QVBoxLayout(g1)
        self.guide_combo = QComboBox()
        for img in self.images_data:
            self.guide_combo.addItem(img.filename)
        self.guide_combo.currentIndexChanged.connect(self._on_guide_changed)
        l1.addWidget(self.guide_combo)
        self.guide_info = QLabel()
        self.guide_info.setStyleSheet("color: #555;")
        l1.addWidget(self.guide_info)
        left.addWidget(g1)

        # 2. Images to crop
        g2 = QGroupBox("2. Immagini da allineare e ritagliare")
        l2 = QVBoxLayout(g2)
        hint = QLabel("Selezionane una sola per il ritaglio a coppie (2 a 2), "
                      "piu' di una per usare la guida su tutte.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        l2.addWidget(hint)
        self.images_list = QListWidget()
        self.images_list.setMinimumHeight(120)
        l2.addWidget(self.images_list)
        row = QHBoxLayout()
        b_all = QPushButton("Tutte")
        b_all.clicked.connect(lambda: self._set_all_checked(True))
        b_none = QPushButton("Nessuna")
        b_none.clicked.connect(lambda: self._set_all_checked(False))
        row.addWidget(b_all)
        row.addWidget(b_none)
        l2.addLayout(row)
        left.addWidget(g2)

        # 3. Rectangle
        g3 = QGroupBox("3. Rettangolo di ritaglio (pixel originali)")
        l3 = QFormLayout(g3)
        self.spin_x = QSpinBox()
        self.spin_y = QSpinBox()
        self.spin_w = QSpinBox()
        self.spin_h = QSpinBox()
        for sp in (self.spin_x, self.spin_y, self.spin_w, self.spin_h):
            sp.setRange(0, 200000)
            sp.setSuffix(" px")
            sp.valueChanged.connect(self._on_spin_changed)
        l3.addRow("X:", self.spin_x)
        l3.addRow("Y:", self.spin_y)
        l3.addRow("Larghezza:", self.spin_w)
        l3.addRow("Altezza:", self.spin_h)
        b_full = QPushButton("Tutta l'immagine")
        b_full.clicked.connect(self._select_full_image)
        l3.addRow(b_full)
        tip = QLabel("Trascina con il mouse sull'anteprima per disegnarlo. "
                     "Rotella = zoom, Ctrl+trascina = sposta.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #555;")
        l3.addRow(tip)
        left.addWidget(g3)

        # 4. Alignment
        g4 = QGroupBox("4. Allineamento grossolano")
        l4 = QFormLayout(g4)
        self.model_combo = QComboBox()
        self.model_combo.addItem("Omografia (punti di presa diversi)", 'homography')
        self.model_combo.addItem("Similitudine (camera fissa)", 'affine')
        l4.addRow("Modello:", self.model_combo)
        self.res_spin = QSpinBox()
        self.res_spin.setRange(500, 8000)
        self.res_spin.setSingleStep(250)
        self.res_spin.setValue(2000)
        self.res_spin.setSuffix(" px")
        self.res_spin.setToolTip(
            "Lato maggiore usato per cercare i punti comuni. Piu' alto = piu' "
            "preciso ma piu' lento. L'allineamento fine si fa poi nella "
            "scheda Allineamento.")
        l4.addRow("Risoluzione:", self.res_spin)
        self.refine_check = QCheckBox("Affinamento locale attorno al rettangolo")
        self.refine_check.setChecked(True)
        self.refine_check.setToolTip(
            "Dopo l'allineamento sull'intera foto, ripete la ricerca dei punti "
            "comuni solo nella zona del rettangolo (+50% di margine) a "
            "risoluzione molto piu' alta: i ritagli escono quasi sovrapposti.")
        l4.addRow(self.refine_check)
        left.addWidget(g4)

        # 5. Output
        g5 = QGroupBox("5. Salvataggio")
        l5 = QVBoxLayout(g5)
        orow = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.textEdited.connect(self._on_out_dir_edited)
        orow.addWidget(self.out_edit)
        b_browse = QPushButton("Sfoglia...")
        b_browse.clicked.connect(self._browse_out_dir)
        orow.addWidget(b_browse)
        l5.addLayout(orow)
        self.replace_check = QCheckBox(
            "Usa i ritagli nel progetto al posto degli originali")
        self.replace_check.setChecked(True)
        l5.addWidget(self.replace_check)
        left.addWidget(g5)

        # Run
        self.btn_run = QPushButton("Allinea e ritaglia")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #1976D2; color: white; "
            "padding: 8px; font-weight: bold; }")
        self.btn_run.clicked.connect(self._run)
        left.addWidget(self.btn_run)
        brow = QHBoxLayout()
        self.btn_stop = QPushButton("Interrompi")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        brow.addWidget(self.btn_stop)
        self.btn_close = QPushButton("Chiudi")
        self.btn_close.clicked.connect(self.reject)
        brow.addWidget(self.btn_close)
        left.addLayout(brow)
        self.progress = QProgressBar()
        left.addWidget(self.progress)
        left.addStretch()

        root.addWidget(left_widget)

        # Right: preview + log
        right = QVBoxLayout()
        self.viewer = ImageViewer()
        self.viewer.set_drag_mode(True)
        self.viewer.drag_started.connect(self._on_drag_started)
        self.viewer.drag_moved.connect(self._on_drag_moved)
        self.viewer.drag_finished.connect(self._on_drag_finished)
        right.addWidget(self.viewer, stretch=5)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(170)
        right.addWidget(self.log, stretch=1)
        root.addLayout(right, stretch=1)

    # ------------------------------------------------------------------
    # Guide / list
    # ------------------------------------------------------------------

    def _on_guide_changed(self, index):
        if not (0 <= index < len(self.images_data)):
            return
        guide = self.images_data[index]
        h, w = guide.image_gray.shape[:2]
        self.guide_info.setText(f"{w} x {h} px ({w * h / 1e6:.1f} MP)")

        # Preview at reduced resolution (the full image is only read at run time)
        s = downscale_factor((h, w), PREVIEW_MAX_DIM)
        self._preview_scale = s
        prev = guide.image_gray if s >= 1.0 else cv2.resize(
            guide.image_gray, (max(1, round(w * s)), max(1, round(h * s))),
            interpolation=cv2.INTER_AREA)
        self.viewer.set_image(prev)
        self._rect_item = None
        self.viewer.fit_in_view()

        # Other images, checked by default
        self.images_list.clear()
        for i, img in enumerate(self.images_data):
            if i == index:
                continue
            item = QListWidgetItem(f"{img.filename}  ({img.width}x{img.height})")
            item.setData(Qt.UserRole, i)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.images_list.addItem(item)

        for sp in (self.spin_x, self.spin_w):
            sp.setMaximum(w)
        for sp in (self.spin_y, self.spin_h):
            sp.setMaximum(h)

        if not self._out_dir_user_set:
            self.out_edit.setText(os.path.join(
                os.path.dirname(os.path.abspath(guide.filepath)), "ritagli"))

        # Default rectangle: central 50%
        self._set_rect_full_res(w // 4, h // 4, w // 2, h // 2)

    def _set_all_checked(self, checked):
        for i in range(self.images_list.count()):
            self.images_list.item(i).setCheckState(
                Qt.Checked if checked else Qt.Unchecked)

    def _selected_indices(self):
        out = []
        for i in range(self.images_list.count()):
            item = self.images_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        return out

    # ------------------------------------------------------------------
    # Rectangle handling (preview coords <-> full-resolution coords)
    # ------------------------------------------------------------------

    def _draw_rect_preview(self, x0, y0, x1, y1):
        scene = self.viewer.scene()
        if self._rect_item is None or self._rect_item.scene() is None:
            pen = QPen(QColor(255, 60, 0))
            pen.setWidth(3)
            pen.setCosmetic(True)
            self._rect_item = scene.addRect(
                QRectF(), pen, QBrush(QColor(255, 120, 0, 45)))
            self._rect_item.setZValue(10)
        self._rect_item.setRect(QRectF(min(x0, x1), min(y0, y1),
                                       abs(x1 - x0), abs(y1 - y0)))

    def _set_rect_full_res(self, x, y, w, h):
        self._syncing = True
        try:
            self.spin_x.setValue(int(x))
            self.spin_y.setValue(int(y))
            self.spin_w.setValue(int(w))
            self.spin_h.setValue(int(h))
        finally:
            self._syncing = False
        self._on_spin_changed()

    def _on_spin_changed(self, *_):
        if self._syncing:
            return
        s = self._preview_scale
        x, y = self.spin_x.value(), self.spin_y.value()
        w, h = self.spin_w.value(), self.spin_h.value()
        self._draw_rect_preview(x * s, y * s, (x + w) * s, (y + h) * s)

    def _select_full_image(self):
        idx = self.guide_combo.currentIndex()
        h, w = self.images_data[idx].image_gray.shape[:2]
        self._set_rect_full_res(0, 0, w, h)

    def _on_drag_started(self, x, y):
        self._drag_origin = (x, y)

    def _on_drag_moved(self, x, y):
        if self._drag_origin is not None:
            self._draw_rect_preview(self._drag_origin[0], self._drag_origin[1], x, y)

    def _on_drag_finished(self, x0, y0, x1, y1):
        self._drag_origin = None
        s = self._preview_scale
        idx = self.guide_combo.currentIndex()
        H, W = self.images_data[idx].image_gray.shape[:2]
        fx0 = max(0, min(W, round(x0 / s)))
        fy0 = max(0, min(H, round(y0 / s)))
        fx1 = max(0, min(W, round(x1 / s)))
        fy1 = max(0, min(H, round(y1 / s)))
        self._set_rect_full_res(fx0, fy0, fx1 - fx0, fy1 - fy0)

    # ------------------------------------------------------------------
    # Output folder
    # ------------------------------------------------------------------

    def _on_out_dir_edited(self, _text):
        self._out_dir_user_set = True

    def _browse_out_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Cartella per i ritagli", self.out_edit.text())
        if d:
            self.out_edit.setText(d)
            self._out_dir_user_set = True

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        guide_idx = self.guide_combo.currentIndex()
        targets = self._selected_indices()
        if not targets:
            QMessageBox.warning(self, "Ritaglio",
                                "Seleziona almeno un'immagine da ritagliare.")
            return
        x, y = self.spin_x.value(), self.spin_y.value()
        w, h = self.spin_w.value(), self.spin_h.value()
        if w < 16 or h < 16:
            QMessageBox.warning(self, "Ritaglio",
                                "Disegna un rettangolo di ritaglio piu' grande.")
            return
        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Ritaglio", "Indica la cartella di salvataggio.")
            return

        guide_path = self.images_data[guide_idx].filepath
        paths = [self.images_data[i].filepath for i in targets]

        # Never overwrite a source photo, ask before overwriting old crops
        all_src = [guide_path] + paths
        outs = [os.path.abspath(output_name(p, out_dir)) for p in all_src]
        if len(set(outs)) != len(outs):
            QMessageBox.warning(self, "Ritaglio",
                                "Due immagini hanno lo stesso nome file: "
                                "i ritagli si sovrascriverebbero.")
            return
        if set(outs) & {os.path.abspath(p) for p in all_src}:
            QMessageBox.warning(self, "Ritaglio",
                                "La cartella scelta sovrascriverebbe una foto "
                                "originale. Scegli un'altra cartella.")
            return
        existing = [o for o in outs if os.path.exists(o)]
        if existing:
            reply = QMessageBox.question(
                self, "Ritagli esistenti",
                f"{len(existing)} ritagli esistono gia' in\n{out_dir}\n\n"
                f"Sovrascriverli?", QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        self.log.clear()
        self.log.append(f"Guida: {os.path.basename(guide_path)} - rettangolo "
                        f"{w}x{h} px in ({x}, {y})")
        self._set_running(True)

        self._thread = QThread()
        self._worker = _PrecropWorker(dict(
            guide_path=guide_path, image_paths=paths,
            rect=(x, y, x + w, y + h), out_dir=out_dir,
            max_dim=self.res_spin.value(),
            model=self.model_combo.currentData(),
            refine=self.refine_check.isChecked()))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        # Direct connections: quit() is thread-safe and must not wait for the
        # (possibly blocked) GUI event loop, so _cleanup_thread can wait().
        self._worker.finished.connect(self._thread.quit, Qt.DirectConnection)
        self._worker.error.connect(self._thread.quit, Qt.DirectConnection)
        self._thread.start()

    def _set_running(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_close.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.viewer.set_drag_mode(not running)
        if running:
            self.progress.setValue(0)

    def _stop(self):
        if self._worker:
            self._worker.cancel()
            self.log.append("Interruzione richiesta (dopo l'immagine in corso)...")

    def _on_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.log.append(msg)

    def _cleanup_thread(self):
        if self._thread is not None:
            self._thread.wait(2000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _on_error(self, msg):
        self._set_running(False)
        self._cleanup_thread()
        self.log.append(f"\nERRORE: {msg}")
        QMessageBox.critical(self, "Ritaglio", msg)

    def _on_finished(self, outcomes):
        self._set_running(False)
        self._cleanup_thread()
        self.progress.setValue(100)

        ok = [o for o in outcomes if o.ok]
        failed = [o for o in outcomes if not o.ok]
        lines = []
        for o in outcomes:
            name = os.path.basename(o.source_path)
            if o.is_guide:
                lines.append(f"- {name}: guida ritagliata")
            elif o.ok:
                a = o.alignment
                extra = ""
                if a.global_alignment is not None:
                    extra = (f", affinamento locale: correzione "
                             f"{a.refine_correction_px:.1f} px")
                lines.append(f"- {name}: OK ({a.n_inliers} punti, errore "
                             f"{a.rmse_px:.2f} px, copertura {o.coverage_pct:.1f}%"
                             f"{extra})")
                for wmsg in o.warnings:
                    lines.append(f"    ATTENZIONE: {wmsg}")
            else:
                lines.append(f"- {name}: NON ritagliata - {o.error}")
        summary = "\n".join(lines)
        self.log.append("\nRiepilogo:\n" + summary)

        n_targets = len(outcomes) - 1
        if not any(o.ok and not o.is_guide for o in outcomes):
            QMessageBox.warning(self, "Ritaglio",
                                "Nessuna immagine e' stata ritagliata.\n\n" + summary)
            return

        self.output_paths = [o.output_path for o in ok]
        self.replace_in_project = self.replace_check.isChecked()
        out_dir = os.path.dirname(self.output_paths[0])
        msg = (f"Ritagliate {len(ok) - 1} immagini su {n_targets} + la guida.\n"
               f"File salvati in:\n{out_dir}\n\n{summary}")
        if failed:
            msg += "\n\nLe immagini non ritagliate non verranno usate nel progetto."
        if self.replace_in_project:
            msg += ("\n\nI ritagli sostituiranno le immagini del progetto. "
                    "Per un allineamento preciso usa poi la scheda Allineamento.")
        QMessageBox.information(self, "Ritaglio completato", msg)
        self.accept()

    def reject(self):
        if self._thread is not None:
            QMessageBox.information(self, "Ritaglio",
                                    "Attendi la fine o premi Interrompi.")
            return
        super().reject()
