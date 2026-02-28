"""Report configuration and export dialog."""

import logging
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QTextEdit, QCheckBox,
    QFileDialog, QFormLayout, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)


class ReportDialog(QDialog):
    """Dialog for configuring and generating analysis reports.

    Allows the user to:
    - Set report metadata (title, analyst, site, notes)
    - Select which sections to include
    - Choose export formats (PDF, CSV, GeoTIFF)
    - Select output directory
    """

    report_requested = pyqtSignal(dict)  # emits configuration dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Genera Report Analisi")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        self._output_dir = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Metadata ---
        meta_group = QGroupBox("Informazioni Report")
        meta_layout = QFormLayout(meta_group)

        self.title_edit = QLineEdit("Report Analisi DIC - Monitoraggio Frana")
        meta_layout.addRow("Titolo:", self.title_edit)

        self.analyst_edit = QLineEdit()
        meta_layout.addRow("Analista:", self.analyst_edit)

        self.site_edit = QLineEdit()
        meta_layout.addRow("Sito:", self.site_edit)

        self.date_edit = QLineEdit()
        from datetime import datetime
        self.date_edit.setText(datetime.now().strftime("%Y-%m-%d %H:%M"))
        meta_layout.addRow("Data:", self.date_edit)

        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        self.notes_edit.setPlaceholderText("Note aggiuntive...")
        meta_layout.addRow("Note:", self.notes_edit)

        layout.addWidget(meta_group)

        # --- Sections ---
        sections_group = QGroupBox("Sezioni da Includere")
        sections_layout = QVBoxLayout(sections_group)

        self.cb_overview = QCheckBox("Panoramica e informazioni GPS")
        self.cb_overview.setChecked(True)
        sections_layout.addWidget(self.cb_overview)

        self.cb_preprocessing = QCheckBox("Dettagli preprocessing (filtri applicati)")
        self.cb_preprocessing.setChecked(True)
        sections_layout.addWidget(self.cb_preprocessing)

        self.cb_parameters = QCheckBox("Parametri analisi DIC")
        self.cb_parameters.setChecked(True)
        sections_layout.addWidget(self.cb_parameters)

        self.cb_displacement = QCheckBox("Mappe spostamento (magnitudine, U, V)")
        self.cb_displacement.setChecked(True)
        sections_layout.addWidget(self.cb_displacement)

        self.cb_vectors = QCheckBox("Campo vettoriale spostamenti")
        self.cb_vectors.setChecked(True)
        sections_layout.addWidget(self.cb_vectors)

        self.cb_correlation = QCheckBox("Mappa qualita correlazione")
        self.cb_correlation.setChecked(True)
        sections_layout.addWidget(self.cb_correlation)

        self.cb_strain = QCheckBox("Mappe strain (se calcolato)")
        self.cb_strain.setChecked(True)
        sections_layout.addWidget(self.cb_strain)

        self.cb_zones = QCheckBox("Zone attive rilevate (tavola sinottica)")
        self.cb_zones.setChecked(True)
        sections_layout.addWidget(self.cb_zones)

        self.cb_zone_details = QCheckBox("Dettaglio zone (immagini rif./def./diff. per zona)")
        self.cb_zone_details.setChecked(True)
        sections_layout.addWidget(self.cb_zone_details)

        self.cb_statistics = QCheckBox("Riepilogo statistico")
        self.cb_statistics.setChecked(True)
        sections_layout.addWidget(self.cb_statistics)

        self.cb_timeseries = QCheckBox("Serie temporale (se analisi sequenza)")
        self.cb_timeseries.setChecked(True)
        sections_layout.addWidget(self.cb_timeseries)

        layout.addWidget(sections_group)

        # --- Export formats ---
        format_group = QGroupBox("Formati Esportazione")
        format_layout = QVBoxLayout(format_group)

        self.cb_pdf = QCheckBox("Report PDF")
        self.cb_pdf.setChecked(True)
        format_layout.addWidget(self.cb_pdf)

        self.cb_csv = QCheckBox("Dati spostamento CSV")
        self.cb_csv.setChecked(True)
        format_layout.addWidget(self.cb_csv)

        self.cb_geotiff = QCheckBox("Raster GeoTIFF georeferenziati")
        self.cb_geotiff.setChecked(False)
        format_layout.addWidget(self.cb_geotiff)

        self.cb_images = QCheckBox("Immagini singole PNG delle mappe")
        self.cb_images.setChecked(True)
        format_layout.addWidget(self.cb_images)

        layout.addWidget(format_group)

        # --- Output directory ---
        dir_group = QGroupBox("Cartella Output")
        dir_layout = QHBoxLayout(dir_group)
        self.dir_label = QLineEdit()
        self.dir_label.setReadOnly(True)
        self.dir_label.setPlaceholderText("Seleziona cartella...")
        dir_layout.addWidget(self.dir_label)
        self.btn_browse = QPushButton("Sfoglia...")
        self.btn_browse.clicked.connect(self._browse_directory)
        dir_layout.addWidget(self.btn_browse)
        layout.addWidget(dir_group)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.btn_generate = QPushButton("Genera Report")
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 10px 20px; font-weight: bold; font-size: 14px; }")
        self.btn_generate.clicked.connect(self._generate)
        btn_layout.addWidget(self.btn_generate)

        self.btn_cancel = QPushButton("Annulla")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleziona Cartella Output")
        if dir_path:
            self._output_dir = dir_path
            self.dir_label.setText(dir_path)

    def _generate(self):
        """Collect settings and emit report request signal."""
        if not self._output_dir:
            QMessageBox.warning(self, "Errore",
                                "Seleziona una cartella di output")
            return

        config = {
            'title': self.title_edit.text(),
            'analyst': self.analyst_edit.text(),
            'site': self.site_edit.text(),
            'date': self.date_edit.text(),
            'notes': self.notes_edit.toPlainText(),
            'output_dir': self._output_dir,
            'sections': {
                'overview': self.cb_overview.isChecked(),
                'preprocessing': self.cb_preprocessing.isChecked(),
                'parameters': self.cb_parameters.isChecked(),
                'displacement': self.cb_displacement.isChecked(),
                'vectors': self.cb_vectors.isChecked(),
                'correlation': self.cb_correlation.isChecked(),
                'strain': self.cb_strain.isChecked(),
                'zones': self.cb_zones.isChecked(),
                'zone_details': self.cb_zone_details.isChecked(),
                'statistics': self.cb_statistics.isChecked(),
                'timeseries': self.cb_timeseries.isChecked(),
            },
            'formats': {
                'pdf': self.cb_pdf.isChecked(),
                'csv': self.cb_csv.isChecked(),
                'geotiff': self.cb_geotiff.isChecked(),
                'images': self.cb_images.isChecked(),
            },
        }

        self.report_requested.emit(config)

    def set_progress(self, percent):
        """Update progress bar."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(percent)
        if percent >= 100:
            self.progress_bar.setVisible(False)

    def get_config(self):
        """Return current configuration without emitting signal."""
        return {
            'title': self.title_edit.text(),
            'analyst': self.analyst_edit.text(),
            'site': self.site_edit.text(),
            'date': self.date_edit.text(),
            'notes': self.notes_edit.toPlainText(),
            'output_dir': self._output_dir,
        }
