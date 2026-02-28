"""Interactive ROI drawing tools for mask definition."""

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QFileDialog, QHeaderView, QButtonGroup, QRadioButton
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor
from dic_app.gui.image_viewer import ImageViewer
from dic_app.core.mask_manager import (
    MaskManager, ROIDefinition, ROIType, ROIMode
)


class ROIEditorWidget(QWidget):
    """ROI editing panel with interactive drawing on the image viewer.

    Features:
    - Draw polygons, rectangles, ellipses, freehand shapes
    - Toggle include/exclude mode
    - ROI list with name, type, mode
    - Import/export masks as PNG or ROI definitions as JSON
    - Live mask overlay preview
    """

    mask_updated = pyqtSignal(object)  # emits MaskManager

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mask_manager: MaskManager = None
        self._drawing = False
        self._current_points = []
        self._draw_mode = ROIType.POLYGON
        self._roi_mode = ROIMode.INCLUDE
        self._roi_counter = 0
        self._mouse_pos = None  # Current mouse position for live preview
        self._cached_roi_overlay = None  # Cache for finalized ROI overlay
        # Debounce timer for live preview (avoid too many redraws on mouse move)
        self._live_preview_timer = QTimer()
        self._live_preview_timer.setSingleShot(True)
        self._live_preview_timer.setInterval(30)  # ~33 fps max
        self._live_preview_timer.timeout.connect(self._update_live_preview)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Image viewer ---
        viewer_group = QGroupBox("Vista Immagine con Maschere")
        viewer_layout = QVBoxLayout(viewer_group)
        self.viewer = ImageViewer()
        self.viewer.mouse_clicked.connect(self._on_click)
        self.viewer.mouse_double_clicked.connect(self._on_double_click)
        self.viewer.mouse_moved.connect(self._on_mouse_moved)
        viewer_layout.addWidget(self.viewer)

        # Status label
        self.status_label = QLabel("Seleziona uno strumento e clicca sull'immagine")
        viewer_layout.addWidget(self.status_label)
        main_layout.addWidget(viewer_group, stretch=3)

        # --- Drawing tools ---
        tools_group = QGroupBox("Strumenti Disegno")
        tools_layout = QVBoxLayout(tools_group)

        # Shape selection
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Forma:"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("Poligono", ROIType.POLYGON)
        self.shape_combo.addItem("Rettangolo", ROIType.RECTANGLE)
        self.shape_combo.addItem("Ellisse", ROIType.ELLIPSE)
        self.shape_combo.addItem("Mano Libera", ROIType.FREEHAND)
        self.shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        shape_row.addWidget(self.shape_combo)
        tools_layout.addLayout(shape_row)

        # Mode selection
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Modalita:"))
        self.mode_group = QButtonGroup()
        self.rb_include = QRadioButton("Includi (verde)")
        self.rb_include.setChecked(True)
        self.rb_exclude = QRadioButton("Escludi (rosso)")
        self.mode_group.addButton(self.rb_include, 0)
        self.mode_group.addButton(self.rb_exclude, 1)
        self.rb_include.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.rb_include)
        mode_row.addWidget(self.rb_exclude)
        tools_layout.addLayout(mode_row)

        # Draw buttons
        draw_row = QHBoxLayout()
        self.btn_start = QPushButton("Inizia Disegno")
        self.btn_start.clicked.connect(self._start_drawing)
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 6px; }")
        draw_row.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("Annulla")
        self.btn_cancel.clicked.connect(self._cancel_drawing)
        self.btn_cancel.setEnabled(False)
        draw_row.addWidget(self.btn_cancel)

        self.btn_finish = QPushButton("Completa ROI")
        self.btn_finish.clicked.connect(self._finish_drawing)
        self.btn_finish.setEnabled(False)
        draw_row.addWidget(self.btn_finish)
        tools_layout.addLayout(draw_row)

        main_layout.addWidget(tools_group)

        # --- ROI List ---
        list_group = QGroupBox("Lista ROI")
        list_layout = QVBoxLayout(list_group)

        self.roi_table = QTableWidget(0, 4)
        self.roi_table.setHorizontalHeaderLabels(
            ["Nome", "Tipo", "Modalita", "Punti"])
        self.roi_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.roi_table.setSelectionBehavior(
            QTableWidget.SelectRows)
        self.roi_table.setMaximumHeight(150)
        list_layout.addWidget(self.roi_table)

        list_btn_row = QHBoxLayout()
        self.btn_delete_roi = QPushButton("Elimina Selezionata")
        self.btn_delete_roi.clicked.connect(self._delete_selected_roi)
        list_btn_row.addWidget(self.btn_delete_roi)

        self.btn_clear_all = QPushButton("Elimina Tutte")
        self.btn_clear_all.clicked.connect(self._clear_all_rois)
        list_btn_row.addWidget(self.btn_clear_all)
        list_layout.addLayout(list_btn_row)

        # Import/Export
        io_row = QHBoxLayout()
        self.btn_import_mask = QPushButton("Importa Maschera PNG")
        self.btn_import_mask.clicked.connect(self._import_mask)
        io_row.addWidget(self.btn_import_mask)

        self.btn_export_mask = QPushButton("Esporta Maschera PNG")
        self.btn_export_mask.clicked.connect(self._export_mask)
        io_row.addWidget(self.btn_export_mask)

        self.btn_import_json = QPushButton("Importa ROI JSON")
        self.btn_import_json.clicked.connect(self._import_rois_json)
        io_row.addWidget(self.btn_import_json)

        self.btn_export_json = QPushButton("Esporta ROI JSON")
        self.btn_export_json.clicked.connect(self._export_rois_json)
        io_row.addWidget(self.btn_export_json)
        list_layout.addLayout(io_row)

        main_layout.addWidget(list_group)

    def set_image(self, image_gray, image_shape=None):
        """Set the image and initialize mask manager."""
        if image_gray is not None:
            self.viewer.set_image(image_gray)
            self.viewer.fit_in_view()
            shape = image_gray.shape[:2]
            if self._mask_manager is None or self._mask_manager.image_shape != shape:
                self._mask_manager = MaskManager(shape)
            self._update_overlay()

    def set_mask_manager(self, manager):
        """Set an existing mask manager (e.g., from project load)."""
        self._mask_manager = manager
        self._refresh_roi_table()
        self._update_overlay()

    def get_mask_manager(self):
        return self._mask_manager

    def get_mask(self):
        """Get the current binary mask."""
        if self._mask_manager:
            return self._mask_manager.generate_mask()
        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _on_shape_changed(self, index):
        self._draw_mode = self.shape_combo.currentData()

    def _on_mode_changed(self):
        self._roi_mode = ROIMode.INCLUDE if self.rb_include.isChecked() else ROIMode.EXCLUDE

    def _start_drawing(self):
        self._drawing = True
        self._current_points = []
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_finish.setEnabled(True)

        mode = self._draw_mode
        if mode == ROIType.POLYGON:
            self.status_label.setText(
                "Clicca per aggiungere vertici. Doppio-clic o 'Completa' per chiudere.")
        elif mode == ROIType.RECTANGLE:
            self.status_label.setText(
                "Clicca per l'angolo 1, poi clicca per l'angolo 2.")
        elif mode == ROIType.ELLIPSE:
            self.status_label.setText(
                "Clicca per il centro, poi clicca per definire il raggio.")
        elif mode == ROIType.FREEHAND:
            self.status_label.setText(
                "Clicca per aggiungere punti. 'Completa' per chiudere.")

    def _cancel_drawing(self):
        self._drawing = False
        self._current_points = []
        self._mouse_pos = None
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_finish.setEnabled(False)
        self.status_label.setText("Disegno annullato")
        self._update_overlay()  # Restore clean overlay (ROI only, no live preview)

    def _on_click(self, x, y):
        if not self._drawing:
            return

        self._current_points.append((x, y))

        n = len(self._current_points)
        mode = self._draw_mode

        if mode == ROIType.RECTANGLE and n >= 2:
            self._finish_drawing()
        elif mode == ROIType.ELLIPSE and n >= 2:
            self._finish_drawing()
        else:
            self.status_label.setText(
                f"Punto {n} aggiunto ({x}, {y}). "
                f"{'Doppio-clic o Completa per chiudere.' if mode == ROIType.POLYGON else 'Continua...'}")
            self._update_live_preview()

    def _on_double_click(self, x, y):
        if self._drawing and self._draw_mode in (ROIType.POLYGON, ROIType.FREEHAND):
            self._finish_drawing()

    def _on_mouse_moved(self, x, y, value):
        """Track mouse position for live drawing preview (debounced)."""
        if self._drawing and self._current_points:
            self._mouse_pos = (x, y)
            # Use debounced timer to avoid too many redraws
            if not self._live_preview_timer.isActive():
                self._live_preview_timer.start()

    def _finish_drawing(self):
        if not self._current_points:
            self._cancel_drawing()
            return

        mode = self._draw_mode
        min_points = {
            ROIType.POLYGON: 3,
            ROIType.RECTANGLE: 2,
            ROIType.ELLIPSE: 2,
            ROIType.FREEHAND: 3,
        }

        if len(self._current_points) < min_points.get(mode, 2):
            self.status_label.setText(
                f"Servono almeno {min_points[mode]} punti per {mode.value}")
            return

        self._roi_counter += 1
        roi = ROIDefinition(
            roi_type=mode,
            mode=self._roi_mode,
            points=self._current_points.copy(),
            name=f"ROI_{self._roi_counter}",
        )

        if self._mask_manager is None:
            self.status_label.setText("Errore: nessuna immagine caricata")
            return

        self._mask_manager.add_roi(roi)
        self._drawing = False
        self._current_points = []
        self._mouse_pos = None

        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_finish.setEnabled(False)

        self._refresh_roi_table()
        self._update_overlay()
        self.mask_updated.emit(self._mask_manager)
        self.status_label.setText(
            f"ROI '{roi.name}' ({mode.value}, {self._roi_mode.value}) aggiunta")

    # ------------------------------------------------------------------
    # Live drawing preview
    # ------------------------------------------------------------------

    def _update_live_preview(self):
        """Draw live preview overlay showing the polygon being constructed.

        Renders:
        - Filled vertices (circles) at each clicked point
        - Solid lines connecting consecutive points
        - Dashed line from the last point to the current mouse position
        - For rectangle/ellipse: preview of the shape with 1 point + mouse
        - Color: green for INCLUDE, red for EXCLUDE
        """
        if self._mask_manager is None:
            return

        h, w = self._mask_manager.image_shape
        # Start from cached ROI overlay (so finalized ROIs remain visible)
        if self._cached_roi_overlay is not None:
            overlay = self._cached_roi_overlay.copy()
        elif len(self._mask_manager) > 0:
            self._cached_roi_overlay = self._mask_manager.get_roi_overlay(alpha=0.3)
            overlay = self._cached_roi_overlay.copy()
        else:
            overlay = np.zeros((h, w, 4), dtype=np.uint8)

        points = self._current_points
        mouse = self._mouse_pos
        mode = self._draw_mode

        if not points:
            self.viewer.set_overlay(overlay)
            return

        # Colors based on include/exclude mode
        if self._roi_mode == ROIMode.INCLUDE:
            color_line = (0, 220, 0, 220)       # Green solid
            color_fill = (0, 200, 0, 60)         # Green transparent fill
            color_vertex = (0, 255, 0, 255)       # Green bright vertex
            color_dash = (0, 180, 0, 150)         # Green dashed
        else:
            color_line = (220, 0, 0, 220)         # Red solid
            color_fill = (200, 0, 0, 60)          # Red transparent fill
            color_vertex = (255, 0, 0, 255)       # Red bright vertex
            color_dash = (180, 0, 0, 150)         # Red dashed

        # Convert colors for cv2 (BGRA)
        line_bgra = (color_line[2], color_line[1], color_line[0], color_line[3])
        fill_bgra = (color_fill[2], color_fill[1], color_fill[0], color_fill[3])
        vert_bgra = (color_vertex[2], color_vertex[1], color_vertex[0], color_vertex[3])
        dash_bgra = (color_dash[2], color_dash[1], color_dash[0], color_dash[3])

        if mode in (ROIType.POLYGON, ROIType.FREEHAND):
            # Draw filled preview polygon (points + mouse as closing vertex)
            preview_pts = list(points)
            if mouse:
                preview_pts_with_mouse = preview_pts + [mouse]
            else:
                preview_pts_with_mouse = preview_pts

            if len(preview_pts_with_mouse) >= 3:
                pts_arr = np.array(preview_pts_with_mouse, dtype=np.int32)
                cv2.fillPoly(overlay, [pts_arr],
                             color=fill_bgra)

            # Draw solid lines between placed points
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
                cv2.line(overlay, pt1, pt2, line_bgra, 2, cv2.LINE_AA)

            # Draw dashed line from last point to mouse and from mouse to first point
            if mouse and points:
                last_pt = (int(points[-1][0]), int(points[-1][1]))
                mouse_pt = (int(mouse[0]), int(mouse[1]))
                first_pt = (int(points[0][0]), int(points[0][1]))
                self._draw_dashed_line(overlay, last_pt, mouse_pt, dash_bgra, 2)
                if len(points) >= 2:
                    self._draw_dashed_line(overlay, mouse_pt, first_pt, dash_bgra, 1)

        elif mode == ROIType.RECTANGLE:
            if len(points) == 1 and mouse:
                # Preview rectangle from point 1 to mouse
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(mouse[0]), int(mouse[1])
                top_left = (min(x1, x2), min(y1, y2))
                bottom_right = (max(x1, x2), max(y1, y2))
                cv2.rectangle(overlay, top_left, bottom_right, fill_bgra, -1)
                cv2.rectangle(overlay, top_left, bottom_right, line_bgra, 2)

        elif mode == ROIType.ELLIPSE:
            if len(points) == 1 and mouse:
                # Preview ellipse: center = point 1, radii from mouse
                cx, cy = int(points[0][0]), int(points[0][1])
                mx, my = int(mouse[0]), int(mouse[1])
                rx = abs(mx - cx)
                ry = abs(my - cy)
                if rx > 0 and ry > 0:
                    cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360,
                                fill_bgra, -1)
                    cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360,
                                line_bgra, 2, cv2.LINE_AA)

        # Draw vertices as filled circles
        vertex_radius = max(4, min(h, w) // 200)
        for pt in points:
            center = (int(pt[0]), int(pt[1]))
            cv2.circle(overlay, center, vertex_radius, vert_bgra, -1, cv2.LINE_AA)
            # White outline for better visibility
            cv2.circle(overlay, center, vertex_radius, (255, 255, 255, 200),
                        1, cv2.LINE_AA)

        # Show mouse position as a smaller hollow circle
        if mouse:
            mouse_center = (int(mouse[0]), int(mouse[1]))
            cv2.circle(overlay, mouse_center, max(3, vertex_radius - 1),
                        vert_bgra, 1, cv2.LINE_AA)

        self.viewer.set_overlay(overlay)

    @staticmethod
    def _draw_dashed_line(overlay, pt1, pt2, color, thickness,
                           dash_length=12, gap_length=8):
        """Draw a dashed line on the overlay (RGBA image)."""
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1:
            return
        dx /= length
        dy /= length

        dist = 0.0
        drawing = True
        while dist < length:
            seg_len = dash_length if drawing else gap_length
            end_dist = min(dist + seg_len, length)
            if drawing:
                sx = int(x1 + dx * dist)
                sy = int(y1 + dy * dist)
                ex = int(x1 + dx * end_dist)
                ey = int(y1 + dy * end_dist)
                cv2.line(overlay, (sx, sy), (ex, ey), color, thickness,
                         cv2.LINE_AA)
            dist = end_dist
            drawing = not drawing

    # ------------------------------------------------------------------
    # ROI management
    # ------------------------------------------------------------------

    def _refresh_roi_table(self):
        if self._mask_manager is None:
            return

        self.roi_table.setRowCount(len(self._mask_manager.rois))
        for i, roi in enumerate(self._mask_manager.rois):
            self.roi_table.setItem(i, 0, QTableWidgetItem(roi.name))
            self.roi_table.setItem(i, 1, QTableWidgetItem(roi.roi_type.value))
            mode_item = QTableWidgetItem(roi.mode.value)
            if roi.mode == ROIMode.INCLUDE:
                mode_item.setForeground(QColor(0, 180, 0))
            else:
                mode_item.setForeground(QColor(200, 0, 0))
            self.roi_table.setItem(i, 2, mode_item)
            self.roi_table.setItem(i, 3, QTableWidgetItem(str(len(roi.points))))

    def _delete_selected_roi(self):
        row = self.roi_table.currentRow()
        if row >= 0 and self._mask_manager:
            self._mask_manager.remove_roi(row)
            self._refresh_roi_table()
            self._update_overlay()
            self.mask_updated.emit(self._mask_manager)

    def _clear_all_rois(self):
        if self._mask_manager:
            self._mask_manager.clear_all()
            self._refresh_roi_table()
            self._update_overlay()
            self.mask_updated.emit(self._mask_manager)

    def _update_overlay(self):
        """Update ROI overlay on the viewer and refresh cache."""
        if self._mask_manager and len(self._mask_manager) > 0:
            self._cached_roi_overlay = self._mask_manager.get_roi_overlay(alpha=0.3)
            self.viewer.set_overlay(self._cached_roi_overlay)
        else:
            self._cached_roi_overlay = None
            self.viewer.clear_overlay()

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def _import_mask(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Importa Maschera", "",
            "Immagini (*.png *.jpg *.bmp *.tif)")
        if filepath and self._mask_manager:
            self._mask_manager.load_mask(filepath)
            self._refresh_roi_table()
            self._update_overlay()
            self.mask_updated.emit(self._mask_manager)

    def _export_mask(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Esporta Maschera", "mask.png",
            "PNG (*.png)")
        if filepath and self._mask_manager:
            self._mask_manager.save_mask(filepath)

    def _import_rois_json(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Importa ROI", "",
            "JSON (*.json)")
        if filepath and self._mask_manager:
            self._mask_manager.import_rois(filepath)
            self._refresh_roi_table()
            self._update_overlay()
            self.mask_updated.emit(self._mask_manager)

    def _export_rois_json(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Esporta ROI", "rois.json",
            "JSON (*.json)")
        if filepath and self._mask_manager:
            self._mask_manager.export_rois(filepath)
