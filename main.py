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

    # Application-wide dark stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
        }
        QWidget {
            font-size: 12px;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QTabWidget::pane {
            border: 1px solid #555;
            border-radius: 4px;
        }
        QTabBar::tab {
            padding: 8px 16px;
            margin-right: 2px;
            border: 1px solid #555;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #3d3d3d;
            font-weight: bold;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        QToolTip {
            background-color: #3d3d3d;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 4px;
        }
    """)

    window = DICMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
