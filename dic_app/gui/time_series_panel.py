"""Tab 'Serie temporale': analysis of more than two surveys."""

import datetime as _dt
import os

import numpy as np
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QComboBox,
    QDoubleSpinBox, QProgressBar, QSplitter, QMessageBox, QFileDialog,
    QScrollArea, QAbstractItemView
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavToolbar)
from matplotlib.figure import Figure

from dic_app.core.time_series import (
    Epoch, TimeSeriesRunner, TimeSeriesCancelled, parse_date)
from dic_app.io import time_series_report as tsr
from dic_app.utils.helpers import setup_logger

logger = setup_logger(__name__)


class _SeriesWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, runner):
        super().__init__()
        self.runner = runner

    def run(self):
        try:
            self.runner.set_progress_callback(lambda p, m: self.progress.emit(p, m))
            self.finished.emit(self.runner.run())
        except TimeSeriesCancelled:
            self.cancelled.emit()
        except Exception as e:
            logger.exception("Time series failed")
            self.error.emit(str(e))


def _fmt_date(d):
    return d.strftime('%d/%m/%Y') if d else ''


def _read_date(text):
    text = (text or '').strip()
    for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d', '%d.%m.%Y', '%d-%m-%Y'):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


class TimeSeriesPanel(QWidget):
    """Choose the surveys, run the series, explore maps and point histories."""

    open_in_results = pyqtSignal(object, int, int)  # DICResult, ref index, def index
    series_computed = pyqtSignal(object)

    COL_USE, COL_NAME, COL_DATE, COL_ROLE = range(4)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project = None
        self._params_getter = None
        self._mask_getter = None
        self._ts = None
        self._base = None
        self._point = None
        self._thread = None
        self._worker = None
        self._runner = None
        self._limits = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self):
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        # ---- left column
        left = QWidget()
        lv = QVBoxLayout(left)

        g = QGroupBox("Rilievi della serie")
        gl = QVBoxLayout(g)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Usa", "Immagine", "Data", "Ruolo"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(self.COL_USE, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        hh.setSectionResizeMode(self.COL_DATE, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(self.COL_ROLE, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(170)
        self.table.itemChanged.connect(lambda *_: self._update_info())
        gl.addWidget(self.table)
        hint = QLabel("Le date si leggono dai dati EXIF o dal nome del file "
                      "(es. ortofoto_15_11_24); correggile con doppio clic "
                      "(gg/mm/aaaa). Il riferimento è quello della scheda Progetto.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px;")
        gl.addWidget(hint)
        b = QPushButton("Rileggi date dai file")
        b.clicked.connect(self._reload_dates)
        gl.addWidget(b)
        lv.addWidget(g)

        g = QGroupBox("Analisi")
        gl = QVBoxLayout(g)
        self.chk_intervals = QCheckBox("Confronta anche i rilievi consecutivi "
                                       "(consigliato: data dei crolli)")
        self.chk_intervals.setChecked(True)
        self.chk_intervals.toggled.connect(lambda *_: self._update_info())
        gl.addWidget(self.chk_intervals)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size:11px;")
        gl.addWidget(self.info_label)
        row = QHBoxLayout()
        self.btn_run = QPushButton("AVVIA SERIE TEMPORALE")
        self.btn_run.setStyleSheet("QPushButton { background-color:#4CAF50; color:white;"
                                   " padding:8px; font-weight:bold; }")
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run, 1)
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_stop)
        gl.addLayout(row)
        self.progress = QProgressBar()
        gl.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size:11px;")
        gl.addWidget(self.status)
        lv.addWidget(g)

        g = QGroupBox("Crolli per intervallo")
        gl = QVBoxLayout(g)
        self.collapse_table = QTableWidget(0, 4)
        self.collapse_table.setHorizontalHeaderLabels(
            ["Intervallo", "Zone", "Nuova area m²", "Cumulata m²"])
        self.collapse_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.collapse_table.verticalHeader().setVisible(False)
        self.collapse_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.collapse_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.collapse_table.setToolTip("Doppio clic: apri l'intervallo nella scheda "
                                       "Risultati (validazione dei distacchi)")
        self.collapse_table.cellDoubleClicked.connect(self._open_interval)
        self.collapse_table.setMinimumHeight(120)
        gl.addWidget(self.collapse_table)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size:11px;")
        gl.addWidget(self.summary)
        lv.addWidget(g)

        g = QGroupBox("Esporta")
        gl = QVBoxLayout(g)
        self.btn_open = QPushButton("Apri il rilievo mostrato nella scheda Risultati")
        self.btn_open.clicked.connect(self._open_epoch)
        gl.addWidget(self.btn_open)
        row = QHBoxLayout()
        self.btn_csv = QPushButton("CSV")
        self.btn_csv.clicked.connect(self._export_csv)
        row.addWidget(self.btn_csv)
        self.btn_pdf = QPushButton("Report PDF serie")
        self.btn_pdf.clicked.connect(self._export_pdf)
        row.addWidget(self.btn_pdf)
        gl.addLayout(row)
        lv.addWidget(g)
        lv.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(380)
        split.addWidget(scroll)

        # ---- right column: map + point history
        right = QWidget()
        rv = QVBoxLayout(right)
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Mappa:"))
        self.mode_combo = QComboBox()
        for key, name in tsr.MAP_MODES:
            self.mode_combo.addItem(name, key)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        ctl.addWidget(self.mode_combo)
        ctl.addWidget(QLabel("Rilievo:"))
        self.epoch_combo = QComboBox()
        self.epoch_combo.currentIndexChanged.connect(lambda *_: self._draw())
        ctl.addWidget(self.epoch_combo)
        ctl.addWidget(QLabel("Scala max:"))
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(0, 1e6)
        self.vmax_spin.setDecimals(2)
        self.vmax_spin.setSpecialValueText("auto")
        self.vmax_spin.valueChanged.connect(lambda *_: self._draw())
        ctl.addWidget(self.vmax_spin)
        self.chk_outline = QCheckBox("Contorni crolli")
        self.chk_outline.setChecked(True)
        self.chk_outline.toggled.connect(lambda *_: self._draw())
        ctl.addWidget(self.chk_outline)
        ctl.addStretch()
        rv.addLayout(ctl)

        vsplit = QSplitter(Qt.Vertical)
        mapw = QWidget()
        ml = QVBoxLayout(mapw)
        ml.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.toolbar = NavToolbar(self.canvas, self)
        ml.addWidget(self.toolbar)
        ml.addWidget(self.canvas)
        vsplit.addWidget(mapw)
        self.pfig = Figure(figsize=(8, 2.4))
        self.pcanvas = FigureCanvas(self.pfig)
        vsplit.addWidget(self.pcanvas)
        vsplit.setSizes([620, 260])
        rv.addWidget(vsplit)
        split.addWidget(right)
        split.setSizes([400, 1000])
        self._set_result_enabled(False)

    def _set_result_enabled(self, on):
        for w in (self.btn_open, self.btn_csv, self.btn_pdf, self.mode_combo,
                  self.epoch_combo, self.vmax_spin, self.chk_outline):
            w.setEnabled(on)

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    def set_context(self, project, params_getter, mask_getter):
        """Called when the tab is shown: refresh the list of surveys."""
        changed = project is not self._project
        self._project = project
        self._params_getter = params_getter
        self._mask_getter = mask_getter
        self._fill_table(keep=not changed)
        if changed or (self._ts is not project.time_series):
            self.set_result(project.time_series)

    def _fill_table(self, keep=True):
        prev = {}
        if keep:
            for r in range(self.table.rowCount()):
                name = self.table.item(r, self.COL_NAME).text()
                prev[name] = (self.table.item(r, self.COL_USE).checkState(),
                              self.table.item(r, self.COL_DATE).text())
        ts_dates = {}
        if self._project.time_series is not None:
            ts_dates = {e.label: e.date for e in self._project.time_series.epochs}
        self.table.blockSignals(True)
        imgs = self._project.images_data
        self.table.setRowCount(len(imgs))
        for i, img in enumerate(imgs):
            is_ref = i == self._project.reference_index
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            name = img.filename
            if name in prev:
                state, dtext = prev[name]
            else:
                state = Qt.Checked
                d = ts_dates.get(name) or parse_date(img.capture_time, img.filepath)
                dtext = _fmt_date(d)
            use.setCheckState(Qt.Checked if is_ref else state)
            if is_ref:
                use.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, self.COL_USE, use)
            it = QTableWidgetItem(name)
            it.setFlags(Qt.ItemIsEnabled)
            it.setToolTip(img.filepath)
            self.table.setItem(i, self.COL_NAME, it)
            self.table.setItem(i, self.COL_DATE, QTableWidgetItem(dtext))
            role = QTableWidgetItem("riferimento" if is_ref else "")
            role.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, self.COL_ROLE, role)
        self.table.blockSignals(False)
        self._update_info()

    def _reload_dates(self):
        if self._project is None:
            return
        self.table.blockSignals(True)
        for i, img in enumerate(self._project.images_data):
            d = parse_date(img.capture_time, img.filepath)
            self.table.item(i, self.COL_DATE).setText(_fmt_date(d))
        self.table.blockSignals(False)
        self._update_info()

    def _selected_epochs(self):
        out = []
        for i in range(self.table.rowCount()):
            if self.table.item(i, self.COL_USE).checkState() != Qt.Checked:
                continue
            img = self._project.images_data[i]
            out.append(Epoch(index=i, label=img.filename,
                             date=_read_date(self.table.item(i, self.COL_DATE).text()),
                             is_reference=(i == self._project.reference_index),
                             path=img.filepath))
        return out

    def _update_info(self):
        if self._project is None:
            return
        ep = self._selected_epochs()
        n = len(ep)
        if n < 3:
            self.info_label.setText(f"{n} rilievi selezionati: ne servono almeno 3 "
                                    "(per 2 usa la scheda Analisi).")
            self.btn_run.setEnabled(False)
            return
        n_jobs = (n - 1) + ((n - 2) if self.chk_intervals.isChecked() else 0)
        msg = f"{n} rilievi: {n_jobs} analisi DIC."
        missing = [e.label for e in ep if e.date is None]
        if missing:
            msg += (f"<br><span style='color:#ef6c00'>Senza data: {', '.join(missing)}. "
                    "Ordine della lista e velocità per rilievo (non per anno).</span>")
        else:
            dates = sorted(e.date for e in ep)
            ref = next(e for e in ep if e.is_reference)
            if ref.date != dates[0]:
                msg += ("<br><span style='color:#ef6c00'>Il riferimento non è il "
                        "rilievo più vecchio: gli spostamenti prima del riferimento "
                        "risultano con segno opposto.</span>")
            if len(set(dates)) != len(dates):
                msg += "<br><span style='color:#c62828'>Due rilievi con la stessa data.</span>"
        msg += ("<br>Imposta nei Parametri lo spostamento massimo atteso "
                "<b>tra il primo e l'ultimo</b> rilievo.")
        self.info_label.setText(msg)
        self.btn_run.setEnabled(self._thread is None)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _run(self):
        if self._project is None or self._thread is not None:
            return
        ep = self._selected_epochs()
        if len(ep) < 3:
            return
        dated = [e.date for e in ep if e.date]
        if len(set(dated)) != len(dated):
            QMessageBox.warning(self, "Serie temporale",
                                "Due rilievi hanno la stessa data: correggi le date.")
            return
        params = self._params_getter()
        if self._project.aligned_ref is not None and params.border_margin < params.subset_size:
            from dataclasses import replace
            params = replace(params, border_margin=params.subset_size)
        mask = None
        if self._mask_getter is not None:
            ok, mask = self._mask_getter()
            if not ok:
                return
        pr = self._project
        ref_idx = pr.reference_index

        def get_gray(i):
            return pr.get_reference_image() if i == ref_idx else pr.get_deformed_image(i)

        def get_rgb(i):
            return pr.get_reference_rgb() if i == ref_idx else pr.get_deformed_rgb(i)

        try:
            self._runner = TimeSeriesRunner(params, ep, get_gray, get_rgb, mask=mask,
                                            gsd=pr.gsd,
                                            intervals=self.chk_intervals.isChecked())
        except ValueError as e:
            QMessageBox.warning(self, "Serie temporale", str(e))
            return
        self._thread = QThread()
        self._worker = _SeriesWorker(self._runner)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.error.connect(self._on_error)
        for sig in (self._worker.finished, self._worker.cancelled, self._worker.error):
            sig.connect(self._thread.quit, Qt.DirectConnection)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText("Serie temporale in corso...")
        self._thread.start()

    def _stop(self):
        if self._runner is not None:
            self._runner.cancel()
            self.status.setText("Annullamento in corso...")

    def _cleanup(self):
        if self._thread is not None:
            self._thread.wait(5000)
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = self._worker = self._runner = None
        self.btn_stop.setEnabled(False)
        self._update_info()

    def _on_progress(self, p, msg):
        self.progress.setValue(p)
        self.status.setText(msg)

    def _on_finished(self, ts):
        self._cleanup()
        self.progress.setValue(100)
        t = sum(r.computation_time_s for r in ts.cumulative + ts.intervals if r is not None)
        self.status.setText(f"Serie completata ({t:.0f} s di calcolo). Clicca sulla "
                            "mappa per vedere la storia di un punto.")
        self._project.time_series = ts
        self.set_result(ts)
        self.series_computed.emit(ts)

    def _on_cancelled(self):
        self._cleanup()
        self.progress.setValue(0)
        self.status.setText("Serie annullata.")

    def _on_error(self, msg):
        self._cleanup()
        self.status.setText(f"<span style='color:#c62828'>Errore: {msg}</span>")
        QMessageBox.critical(self, "Serie temporale", msg)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def set_result(self, ts):
        self._ts = ts
        self._point = None
        self._limits = None
        self.fig.clear()
        self.pfig.clear()
        if ts is None:
            self.canvas.draw_idle()
            self.pcanvas.draw_idle()
            self.collapse_table.setRowCount(0)
            self.summary.setText("")
            self._set_result_enabled(False)
            return
        base = None
        try:
            base = self._project.get_reference_rgb() if self._project else None
            if base is None and self._project is not None:
                base = self._project.get_reference_image()
        except Exception as e:
            logger.warning(f"No base image for the series: {e}")
        self._base = base
        self.epoch_combo.blockSignals(True)
        self.epoch_combo.clear()
        for k, e in enumerate(ts.epochs):
            if k != ts.ref_pos:
                self.epoch_combo.addItem(e.date_str(), k)
        self.epoch_combo.setCurrentIndex(self.epoch_combo.count() - 1)
        self.epoch_combo.blockSignals(False)
        self._set_result_enabled(True)
        self._fill_collapses()
        self._mode_changed()

    def _fill_collapses(self):
        rows = self._ts.collapse_table()
        self.collapse_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [f"{r['from'].date_str()} → {r['to'].date_str()}", str(r['n_zones']),
                    '' if r['new_area_m2'] is None else f"{r['new_area_m2']:.1f}",
                    '' if r['cum_area_m2'] is None else f"{r['cum_area_m2']:.1f}"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setData(Qt.UserRole, r['pos'])
                self.collapse_table.setItem(i, j, it)
        self.summary.setText("<br>".join(tsr.summary_lines(self._ts)))

    def _mode_changed(self, *_):
        mode = self.mode_combo.currentData()
        self.epoch_combo.setEnabled(self._ts is not None and mode in ('cum', 'class'))
        self.vmax_spin.setEnabled(self._ts is not None and mode in ('cum', 'vel', 'vlast'))
        self._draw()

    def _draw(self):
        if self._ts is None:
            return
        if self.fig.axes:
            ax0 = self.fig.axes[0]
            self._limits = (ax0.get_xlim(), ax0.get_ylim())
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        mode = self.mode_combo.currentData()
        k = self.epoch_combo.currentData()
        try:
            tsr.draw_map(self.fig, ax, self._ts, self._base, mode, k,
                         vmax=self.vmax_spin.value() or None,
                         outlines=self.chk_outline.isChecked())
        except Exception as e:
            logger.exception("Map drawing failed")
            ax.set_title(f"Errore nel disegno: {e}")
        if self._limits is not None:
            ax.set_xlim(*self._limits[0])
            ax.set_ylim(*self._limits[1])
        if self._point is not None:
            r, c = self._point
            ax.plot(self._ts.grid_x[r, c], self._ts.grid_y[r, c], 'o', ms=9,
                    mfc='none', mec='white', mew=2)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_click(self, event):
        if self._ts is None or not self.fig.axes or event.inaxes is not self.fig.axes[0]:
            return
        if self.toolbar.mode:          # zoom / pan active
            return
        if event.xdata is None:
            return
        self._point = self._ts.nearest_node(event.xdata, event.ydata)
        self._draw()
        self.pfig.clear()
        ax = self.pfig.add_subplot(111)
        tsr.draw_point_series(ax, self._ts, *self._point)
        self.pfig.autofmt_xdate() if self._ts.has_dates else None
        self.pfig.tight_layout()
        self.pcanvas.draw_idle()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _open_epoch(self):
        if self._ts is None:
            return
        k = self.epoch_combo.currentData()
        r = self._ts.cumulative[k] if k is not None else None
        if r is not None:
            self.open_in_results.emit(r, self._ts.epochs[self._ts.ref_pos].index,
                                      self._ts.epochs[k].index)

    def _open_interval(self, row, _col):
        it = self.collapse_table.item(row, 0)
        if it is None or self._ts is None:
            return
        k = it.data(Qt.UserRole)
        ts = self._ts
        if ts.has_intervals and ts.intervals[k] is not None:
            self.open_in_results.emit(ts.intervals[k], ts.epochs[k - 1].index,
                                      ts.epochs[k].index)
        elif ts.cumulative[k] is not None:
            self.open_in_results.emit(ts.cumulative[k], ts.epochs[ts.ref_pos].index,
                                      ts.epochs[k].index)

    def _default_dir(self):
        pf = getattr(self._project, 'project_filepath', None)
        return os.path.dirname(pf) if pf else os.path.expanduser('~')

    def _export_csv(self):
        if self._ts is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta serie (CSV)", os.path.join(self._default_dir(), "serie_punti.csv"),
            "CSV (*.csv)")
        if not path:
            return
        base, _ = os.path.splitext(path)
        try:
            tsr.export_csv(self._ts, path, base + "_crolli.csv")
            QMessageBox.information(self, "Esporta CSV",
                                    f"Salvati:\n{path}\n{base}_crolli.csv")
        except Exception as e:
            QMessageBox.critical(self, "Esporta CSV", str(e))

    def _export_pdf(self):
        if self._ts is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Report serie temporale", os.path.join(self._default_dir(),
                                                         "serie_temporale.pdf"),
            "PDF (*.pdf)")
        if not path:
            return
        from PyQt5.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pr = self._project
            tsr.export_pdf(self._ts, self._base, path,
                           {'site': getattr(pr, 'site_name', ''),
                            'analyst': getattr(pr, 'analyst_name', '')})
        except Exception as e:
            QApplication.restoreOverrideCursor()
            logger.exception("Series PDF failed")
            QMessageBox.critical(self, "Report serie", str(e))
            return
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Report serie", f"Report salvato:\n{path}")
