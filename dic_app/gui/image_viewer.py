"""Zoomable, pannable image viewer with overlay support."""

import logging
import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent

logger = logging.getLogger(__name__)


class ImageViewer(QGraphicsView):
    """Custom image viewer based on QGraphicsView.

    Features:
    - Smooth zoom with mouse wheel
    - Pan with middle mouse button or Ctrl+left click
    - Mouse position tracking with pixel coordinates and value
    - Overlay support for heatmaps and vector fields
    - Fit to window
    """

    mouse_moved = pyqtSignal(int, int, int)  # x, y, pixel_value
    mouse_clicked = pyqtSignal(int, int)       # x, y on left click
    mouse_double_clicked = pyqtSignal(int, int)
    drag_started = pyqtSignal(int, int)         # x, y scene coords
    drag_moved = pyqtSignal(int, int)           # x, y scene coords
    drag_finished = pyqtSignal(int, int, int, int)  # x0, y0, x1, y1 scene rect

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._base_item = None
        self._overlay_item = None
        self._image_gray = None
        self._image_rgb = None
        self._zoom_factor = 1.0
        self._zoom_step = 1.15
        self._panning = False
        self._pan_start = None
        self._drag_mode = False       # True when external code enables drag selection
        self._dragging = False        # True while a drag is in progress
        self._drag_start = None       # (x, y) scene coords of drag origin

        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #e0e0e0;")

    def set_image(self, image):
        """Display a numpy array image (H,W) grayscale or (H,W,3) RGB uint8."""
        if image is None:
            return

        self._scene.clear()
        self._base_item = None
        self._overlay_item = None

        if image.ndim == 2:
            self._image_gray = np.ascontiguousarray(image, dtype=np.uint8)
            h, w = self._image_gray.shape
            # Keep reference to prevent garbage collection before QPixmap copy
            self._qimg_buffer = self._image_gray
            qimg = QImage(self._qimg_buffer.data, w, h, w, QImage.Format_Grayscale8)
            self._image_rgb = np.stack([self._image_gray] * 3, axis=-1)
        elif image.ndim == 3:
            self._image_rgb = np.ascontiguousarray(image, dtype=np.uint8)
            self._image_gray = np.mean(self._image_rgb, axis=2).astype(np.uint8)
            h, w, ch = self._image_rgb.shape
            if ch == 3:
                self._qimg_buffer = self._image_rgb
                qimg = QImage(self._qimg_buffer.data, w, h, 3 * w, QImage.Format_RGB888)
            elif ch == 4:
                self._qimg_buffer = self._image_rgb
                qimg = QImage(self._qimg_buffer.data, w, h, 4 * w, QImage.Format_RGBA8888)
            else:
                return
        else:
            return

        pixmap = QPixmap.fromImage(qimg.copy())
        self._base_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

    def set_overlay(self, rgba_array):
        """Set an RGBA overlay on top of the base image."""
        if rgba_array is None:
            if self._overlay_item:
                self._scene.removeItem(self._overlay_item)
                self._overlay_item = None
            return

        h, w = rgba_array.shape[:2]
        self._overlay_buffer = np.ascontiguousarray(rgba_array, dtype=np.uint8)
        qimg = QImage(self._overlay_buffer.data, w, h, 4 * w, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg.copy())

        if self._overlay_item:
            self._scene.removeItem(self._overlay_item)
        self._overlay_item = self._scene.addPixmap(pixmap)
        self._overlay_item.setZValue(1)

    def clear_overlay(self):
        self.set_overlay(None)

    def fit_in_view(self):
        if self._base_item:
            self.fitInView(self._base_item, Qt.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()

    def get_image_size(self):
        if self._image_gray is not None:
            h, w = self._image_gray.shape
            return w, h
        return None

    def get_pixel_value(self, x, y):
        if self._image_gray is not None:
            h, w = self._image_gray.shape
            if 0 <= x < w and 0 <= y < h:
                return int(self._image_gray[y, x])
        return -1

    def set_drag_mode(self, enabled: bool):
        """Enable/disable drag-rectangle selection mode.

        When enabled, left-click-drag emits drag_started/drag_moved/drag_finished
        instead of mouse_clicked.
        """
        self._drag_mode = enabled
        self._dragging = False
        self._drag_start = None
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def save_view_as_image(self, filepath):
        pixmap = self.grab()
        pixmap.save(filepath)

    # --- Events ---

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = self._zoom_step
        else:
            factor = 1.0 / self._zoom_step
        self._zoom_factor *= factor
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.LeftButton and self._drag_mode:
            scene_pos = self.mapToScene(event.pos())
            sx, sy = int(scene_pos.x()), int(scene_pos.y())
            self._dragging = True
            self._drag_start = (sx, sy)
            self.drag_started.emit(sx, sy)
            event.accept()
        elif event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.mouse_clicked.emit(int(scene_pos.x()), int(scene_pos.y()))
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.mouse_double_clicked.emit(int(scene_pos.x()), int(scene_pos.y()))
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
        elif self._dragging and self._drag_start is not None:
            scene_pos = self.mapToScene(event.pos())
            self.drag_moved.emit(int(scene_pos.x()), int(scene_pos.y()))
            event.accept()
        else:
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            val = self.get_pixel_value(x, y)
            if val >= 0:
                self.mouse_moved.emit(x, y, val)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        elif self._dragging and self._drag_start is not None:
            scene_pos = self.mapToScene(event.pos())
            x0, y0 = self._drag_start
            x1, y1 = int(scene_pos.x()), int(scene_pos.y())
            self._dragging = False
            self._drag_start = None
            # Normalize to top-left / bottom-right
            rx0, rx1 = min(x0, x1), max(x0, x1)
            ry0, ry1 = min(y0, y1), max(y0, y1)
            if rx1 > rx0 and ry1 > ry0:
                self.drag_finished.emit(rx0, ry0, rx1, ry1)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
