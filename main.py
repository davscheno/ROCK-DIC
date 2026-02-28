#!/usr/bin/env python3
"""DIC Landslide Monitor - Entry Point.

Digital Image Correlation application for landslide and slope
monitoring from drone and camera imagery.

Usage:
    python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from dic_app.gui.main_window import DICMainWindow


def main():
    # High-DPI support
    os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'

    app = QApplication(sys.argv)
    app.setApplicationName("DIC Landslide Monitor")
    app.setOrganizationName("DIC_frane")
    app.setApplicationVersion("1.0.0")

    # Application-wide light stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
            color: #212121;
        }
        QWidget {
            font-size: 12px;
            color: #212121;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #bdbdbd;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 16px;
            background-color: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #37474f;
        }
        QTabWidget::pane {
            border: 1px solid #bdbdbd;
            border-radius: 4px;
            background-color: #ffffff;
        }
        QTabBar::tab {
            padding: 8px 16px;
            margin-right: 2px;
            border: 1px solid #bdbdbd;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            background-color: #e0e0e0;
            color: #424242;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            font-weight: bold;
            color: #1565c0;
        }
        QProgressBar {
            border: 1px solid #bdbdbd;
            border-radius: 4px;
            text-align: center;
            background-color: #e0e0e0;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        QToolTip {
            background-color: #fffde7;
            color: #212121;
            border: 1px solid #bdbdbd;
            padding: 4px;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
            background-color: #ffffff;
            border: 1px solid #bdbdbd;
            border-radius: 3px;
            padding: 3px;
            color: #212121;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #1565c0;
        }
        QPushButton {
            background-color: #e0e0e0;
            border: 1px solid #bdbdbd;
            border-radius: 4px;
            padding: 5px 12px;
            color: #212121;
        }
        QPushButton:hover {
            background-color: #d0d0d0;
        }
        QPushButton:pressed {
            background-color: #bdbdbd;
        }
        QTableWidget {
            background-color: #ffffff;
            gridline-color: #e0e0e0;
            color: #212121;
        }
        QHeaderView::section {
            background-color: #e8eaf6;
            color: #212121;
            border: 1px solid #bdbdbd;
            padding: 4px;
            font-weight: bold;
        }
        QScrollBar:vertical {
            background-color: #f5f5f5;
            width: 12px;
        }
        QScrollBar::handle:vertical {
            background-color: #bdbdbd;
            border-radius: 4px;
            min-height: 20px;
        }
    """)

    window = DICMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
