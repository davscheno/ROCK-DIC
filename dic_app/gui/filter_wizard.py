"""Guided choice of the preprocessing filters (questions -> DIC tests ->
recommendation with an Apply button).  Logic in core.filter_advisor."""

import numpy as np
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QStackedWidget, QWidget, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSplitter, QScrollArea, QMessageBox
)
from PyQt5.QtGui import QColor

from dic_app.core import filter_advisor as fa
from dic_app.gui.image_viewer import ImageViewer
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)


class _AdvisorWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, ref, deff, answers, params):
        super().__init__()
        self.ref, self.deff, self.answers, self.params = ref, deff, answers, params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            rec = fa.evaluate(self.ref, self.deff, self.answers, self.params,
                              progress=lambda p, m: self.progress.emit(p, m),
                              cancel=lambda: self._cancel)
            self.finished.emit(rec)
        except InterruptedError:
            self.failed.emit("")
        except Exception as e:
            logger.exception("Filter advisor failed")
            self.failed.emit(str(e))


class FilterWizardDialog(QDialog):
    """Three pages: questions, analysis, result.

    After ``exec() == Accepted`` the chosen pipeline is in
    ``chosen_pipeline`` (an independent copy).
    """

    PAGE_QUESTIONS, PAGE_RUN, PAGE_RESULT = range(3)

    def __init__(self, ref_gray, def_gray=None, dic_params=None, gsd=None,
                 auto_start=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scelta guidata dei filtri")
        self.resize(900, 680)
        self._ref = ref_gray
        self._def = def_gray
        self._params = dic_params
        self._gsd = gsd
        self._rec = None
        self._thread = None
        self._worker = None
        self.chosen_pipeline = None
        self.chosen_name = ""
        self._groups = {}

        lay = QVBoxLayout(self)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack)
        self.stack.addWidget(self._build_questions())
        self.stack.addWidget(self._build_run())
        self.stack.addWidget(self._build_result())

        if auto_start:
            self._start()

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _build_questions(self):
        page = QWidget()
        v = QVBoxLayout(page)
        intro = QLabel(
            "<b>Quale filtro usare?</b><br>"
            "Rispondi a qualche domanda sulle foto (oppure lascia <i>Non so</i>: "
            "lo misura il programma). Poi i filtri adatti vengono <b>provati "
            "con il DIC sulle tue immagini</b> e ti viene proposto quello che "
            "misura meglio, con i parametri gia' impostati."
            + ("" if self._def is not None else
               "<br><span style='color:#c62828'>Manca una seconda immagine: "
               "sara' eseguito solo il test di precisione.</span>"))
        intro.setWordWrap(True)
        v.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        iv = QVBoxLayout(inner)
        for key, (question, options) in fa.QUESTIONS.items():
            box = QGroupBox(question)
            bl = QVBoxLayout(box)
            grp = QButtonGroup(box)
            default = fa.DEFAULT_ANSWERS.get(key)
            for value, label, help_txt in options:
                rb = QRadioButton(f"{label}  —  {help_txt}")
                rb.setProperty("answer", value)
                rb.setToolTip(help_txt)
                if value == default:
                    rb.setChecked(True)
                grp.addButton(rb)
                bl.addWidget(rb)
            self._groups[key] = grp
            iv.addWidget(box)
        iv.addStretch()
        scroll.setWidget(inner)
        v.addWidget(scroll, stretch=1)

        row = QHBoxLayout()
        row.addStretch()
        b_cancel = QPushButton("Annulla")
        b_cancel.clicked.connect(self.reject)
        row.addWidget(b_cancel)
        self.btn_analyse = QPushButton("Analizza le immagini  ▶")
        self.btn_analyse.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px 18px; }")
        self.btn_analyse.clicked.connect(self._start)
        row.addWidget(self.btn_analyse)
        v.addLayout(row)
        return page

    def _build_run(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.addStretch()
        lab = QLabel("<b>Prova dei filtri in corso</b><br>"
                     "Ogni filtro candidato viene applicato alle immagini e "
                     "valutato con il DIC (area riconosciuta sulla coppia reale "
                     "e precisione su uno spostamento noto).")
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignCenter)
        v.addWidget(lab)
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 100)
        v.addWidget(self.run_progress)
        self.run_label = QLabel("")
        self.run_label.setAlignment(Qt.AlignCenter)
        v.addWidget(self.run_label)
        v.addStretch()
        row = QHBoxLayout()
        row.addStretch()
        self.btn_stop = QPushButton("Annulla")
        self.btn_stop.clicked.connect(self._cancel_run)
        row.addWidget(self.btn_stop)
        v.addLayout(row)
        return page

    def _build_result(self):
        page = QWidget()
        v = QVBoxLayout(page)
        self.res_title = QLabel()
        self.res_title.setWordWrap(True)
        self.res_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; padding: 8px; "
            "background-color: #e8f5e9; color: #1b5e20; border-radius: 4px;")
        v.addWidget(self.res_title)

        self.res_reasons = QLabel()
        self.res_reasons.setWordWrap(True)
        self.res_reasons.setTextFormat(Qt.RichText)
        v.addWidget(self.res_reasons)

        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Filtro", "Area riconosciuta", "Errore (95%)", "Esito"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setToolTip(
            "Area riconosciuta: quota di punti che il DIC riesce ad agganciare "
            "tra le due foto reali (piu' alta = meglio).\n"
            "Errore (95%): errore sul 95% dei punti in un test con spostamento "
            "noto (piu' basso = meglio).")
        self.table.currentCellChanged.connect(lambda r, *_: self._preview_row(r))
        split.addWidget(self.table)

        prev = QWidget()
        ph = QHBoxLayout(prev)
        ph.setContentsMargins(0, 0, 0, 0)
        g1 = QGroupBox("Originale (dettaglio)")
        QVBoxLayout(g1).addWidget(self._viewer("before"))
        g2 = QGroupBox("Con il filtro selezionato")
        QVBoxLayout(g2).addWidget(self._viewer("after"))
        ph.addWidget(g1)
        ph.addWidget(g2)
        split.addWidget(prev)
        split.setSizes([180, 320])
        v.addWidget(split, stretch=1)

        row = QHBoxLayout()
        b_back = QPushButton("◀ Cambia le risposte")
        b_back.clicked.connect(lambda: self.stack.setCurrentIndex(self.PAGE_QUESTIONS))
        row.addWidget(b_back)
        row.addStretch()
        b_close = QPushButton("Chiudi senza applicare")
        b_close.clicked.connect(self.reject)
        row.addWidget(b_close)
        self.btn_apply_sel = QPushButton("Applica il filtro selezionato")
        self.btn_apply_sel.clicked.connect(self._apply_selected)
        row.addWidget(self.btn_apply_sel)
        self.btn_apply_best = QPushButton("✔  Applica il filtro consigliato")
        self.btn_apply_best.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px 18px; }")
        self.btn_apply_best.clicked.connect(self._apply_best)
        row.addWidget(self.btn_apply_best)
        v.addLayout(row)
        return page

    def _viewer(self, which):
        w = ImageViewer()
        setattr(self, f"view_{which}", w)
        return w

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def answers(self):
        out = {}
        for key, grp in self._groups.items():
            b = grp.checkedButton()
            out[key] = b.property("answer") if b is not None else fa.UNKNOWN
        return out

    def _start(self):
        if self._ref is None:
            QMessageBox.warning(self, "Filtri", "Nessuna immagine di riferimento.")
            return
        self.stack.setCurrentIndex(self.PAGE_RUN)
        self.run_progress.setValue(0)
        self.run_label.setText("Avvio...")
        self.btn_stop.setEnabled(True)
        self._thread = QThread(self)
        self._worker = _AdvisorWorker(self._ref, self._def, self.answers(), self._params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, pct, msg):
        self.run_progress.setValue(pct)
        self.run_label.setText(msg)

    def _cancel_run(self):
        if self._worker:
            self._worker.cancel()
            self.btn_stop.setEnabled(False)
            self.run_label.setText("Annullamento...")

    def _on_failed(self, msg):
        self.stack.setCurrentIndex(self.PAGE_QUESTIONS)
        if msg:
            QMessageBox.critical(self, "Filtri", f"Analisi non riuscita:\n{msg}")

    def _on_finished(self, rec):
        self._rec = rec
        self.show_recommendation(rec)

    def show_recommendation(self, rec):
        """Fill the result page (also used by tests)."""
        self._rec = rec
        best = rec.best
        if best.pipeline is not None and len(best.pipeline) == 0:
            self.res_title.setText("Consiglio: nessun filtro — usa le immagini originali")
            self.btn_apply_best.setText("✔  Applica: nessun filtro")
        else:
            self.res_title.setText(f"Consiglio: {best.name}")
            self.btn_apply_best.setText("✔  Applica il filtro consigliato")
        self.res_reasons.setText("<ul>" + "".join(
            f"<li>{r}</li>" for r in rec.reasons) + "</ul>"
            f"<span style='color:#616161;font-size:10px'>Analisi in "
            f"{rec.seconds:.0f} s su {len(rec.tiles)} zone dell'immagine.</span>")

        self.table.setRowCount(len(rec.scores))
        best_row = 0
        for r, sc in enumerate(rec.scores):
            area = "—" if not np.isfinite(sc.recognised_pct) else f"{sc.recognised_pct:.1f} %"
            if np.isfinite(sc.p95_px):
                err = f"{sc.p95_px:.3f} px"
                if self._gsd:
                    err += f"  ({sc.p95_px * self._gsd * 1000:.1f} mm)"
            else:
                err = "—"
            for c, txt in enumerate((sc.name, area, err, sc.verdict)):
                it = QTableWidgetItem(txt)
                it.setToolTip(sc.why)
                if sc is best:
                    it.setBackground(QColor("#c8e6c9"))
                    best_row = r
                elif sc.verdict == "peggiora":
                    it.setForeground(QColor("#c62828"))
                self.table.setItem(r, c, it)
        self.stack.setCurrentIndex(self.PAGE_RESULT)
        self.table.selectRow(best_row)
        self._preview_row(best_row)

    def _preview_row(self, row):
        if self._rec is None or not (0 <= row < len(self._rec.scores)):
            return
        x0, y0, s = self._rec.tiles[0]
        crop = np.ascontiguousarray(self._ref[y0:y0 + s, x0:x0 + s])
        self.view_before.set_image(crop)
        self.view_before.fit_in_view()
        try:
            out = self._rec.scores[row].pipeline.apply(crop)
        except Exception as e:
            logger.warning(f"Preview failed: {e}")
            out = crop
        self.view_after.set_image(out)
        self.view_after.fit_in_view()

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _choose(self, sc):
        self.chosen_pipeline = fa.copy_pipeline(sc.pipeline)
        self.chosen_name = sc.name
        self.accept()

    def _apply_best(self):
        if self._rec is not None:
            self._choose(self._rec.best)

    def _apply_selected(self):
        if self._rec is None:
            return
        row = self.table.currentRow()
        if 0 <= row < len(self._rec.scores):
            self._choose(self._rec.scores[row])

    def reject(self):
        self._cancel_run()
        super().reject()

    def done(self, r):
        if self._thread is not None and self._thread.isRunning():
            if self._worker:
                self._worker.cancel()
            self._thread.quit()
            self._thread.wait(5000)
        super().done(r)
