"""Tool bar + overlay to draw the REAL detachment perimeter(s) on the
image crops of the inspection dialogs (Riferimento / Deformata /
Differenza tabs).  Perimeters are kept in analysed-reference pixels."""

import copy

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout)

from dic_app.core import detachment as dt

EDGE = (255, 235, 0, 255)       # yellow outline
DRAW = (0, 229, 255, 255)       # cyan while drawing


class DetachmentDrawer(QGroupBox):
    """Draw / edit detachment perimeters on one or more ImageViewers.

    ``changed`` is emitted with the new list of perimeter records (same
    format as ``core.detachment.make_detachment``) after every edit.
    """

    changed = pyqtSignal(list)

    def __init__(self, viewers, tab_widget=None, parent=None):
        super().__init__("Perimetro reale del distacco", parent)
        self._viewers = list(viewers)
        self._tabs = tab_widget
        self._offset = (0, 0)
        self._shape = None
        self._gsd = None
        self._result = None
        self._u = self._v = None
        self._dets = []               # records, image coordinates
        self._points = []             # current polygon, CROP coordinates
        self._mouse = None
        self._drawing = False
        self._enabled = True

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        row = QHBoxLayout()
        self.btn_draw = QPushButton("✏  Disegna perimetro")
        self.btn_draw.setStyleSheet(
            "QPushButton { background-color: #8e24aa; color: white; "
            "font-weight: bold; padding: 5px 10px; }")
        self.btn_draw.setToolTip(
            "Clicca i vertici del contorno della nicchia di distacco sulla "
            "foto (meglio nella scheda Deformata). Doppio clic o 'Chiudi' per "
            "terminare. Ctrl+trascina per spostare la vista, rotella per lo zoom.")
        self.btn_draw.clicked.connect(self.start)
        row.addWidget(self.btn_draw)
        self.btn_close_poly = QPushButton("Chiudi perimetro")
        self.btn_close_poly.clicked.connect(self.finish)
        row.addWidget(self.btn_close_poly)
        self.btn_undo_pt = QPushButton("↶ Punto")
        self.btn_undo_pt.setToolTip("Annulla l'ultimo vertice")
        self.btn_undo_pt.clicked.connect(self.undo_point)
        row.addWidget(self.btn_undo_pt)
        self.btn_del_last = QPushButton("Elimina ultimo")
        self.btn_del_last.clicked.connect(self.delete_last)
        row.addWidget(self.btn_del_last)
        self.btn_del_all = QPushButton("Elimina tutti")
        self.btn_del_all.clicked.connect(self.delete_all)
        row.addWidget(self.btn_del_all)
        row.addStretch()
        lay.addLayout(row)
        self.info = QLabel()
        self.info.setStyleSheet("font-size: 11px; color: #4a148c;")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

        for v in self._viewers:
            v.mouse_clicked.connect(self._on_click)
            v.mouse_double_clicked.connect(self._on_double)
            v.mouse_moved.connect(self._on_move)
        self._update_buttons()

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def set_context(self, offset, crop_shape, detachments=None, gsd=None,
                    result=None, u=None, v=None):
        """New crop: (x0, y0) of the crop in the image, its (h, w) and the
        perimeters already stored (image coordinates)."""
        self._offset = (int(offset[0]), int(offset[1]))
        self._shape = tuple(crop_shape[:2])
        self._dets = copy.deepcopy(list(detachments or []))
        self._gsd, self._result, self._u, self._v = gsd, result, u, v
        self._points, self._mouse, self._drawing = [], None, False
        self._update_buttons()
        self.redraw()

    def detachments(self):
        return copy.deepcopy(self._dets)

    def set_enabled_drawing(self, ok: bool):
        self._enabled = ok
        if not ok:
            self._drawing = False
            self._points = []
        self._update_buttons()

    def is_drawing(self):
        return self._drawing

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def start(self):
        if self._shape is None or not self._enabled:
            return
        self._drawing, self._points, self._mouse = True, [], None
        if self._tabs is not None and self._tabs.count() > 1:
            self._tabs.setCurrentIndex(1)            # "Deformata": the scar
        self._update_buttons()
        self.redraw()

    def add_point(self, x, y):
        """Add a vertex in CROP coordinates (also used by tests)."""
        h, w = self._shape
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        if self._points and abs(self._points[-1][0] - x) <= 1 \
                and abs(self._points[-1][1] - y) <= 1:
            return                                   # double-click repeat
        self._points.append((x, y))
        self._update_buttons()
        self.redraw()

    def finish(self):
        if not self._drawing:
            return
        if len(self._points) < 3:
            self.info.setText("Servono almeno 3 vertici per un perimetro.")
            return
        ox, oy = self._offset
        pts = [(x + ox, y + oy) for x, y in self._points]
        if dt.polygon_area(pts) < 1:
            self.info.setText("Perimetro senza area: ridisegnalo.")
            return
        new_id = max((d['id'] for d in self._dets), default=0) + 1
        self._dets.append(dt.make_detachment(pts, self._gsd, self._result,
                                             self._u, self._v, new_id))
        self._drawing, self._points, self._mouse = False, [], None
        self._update_buttons()
        self.redraw()
        self.changed.emit(self.detachments())

    def cancel(self):
        self._drawing, self._points, self._mouse = False, [], None
        self._update_buttons()
        self.redraw()

    def undo_point(self):
        if self._points:
            self._points.pop()
            self._update_buttons()
            self.redraw()

    def delete_last(self):
        if self._drawing:
            self.cancel()
            return
        if self._dets:
            self._dets.pop()
            self._update_buttons()
            self.redraw()
            self.changed.emit(self.detachments())

    def delete_all(self):
        if self._drawing:
            self.cancel()
        if self._dets:
            self._dets = []
            self._update_buttons()
            self.redraw()
            self.changed.emit(self.detachments())

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _on_click(self, x, y):
        if self._drawing:
            self.add_point(x, y)

    def _on_double(self, x, y):
        if self._drawing:
            self.finish()

    def _on_move(self, x, y, _val):
        if self._drawing and self._points:
            self._mouse = (x, y)
            self.redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _update_buttons(self):
        has = bool(self._dets)
        self.btn_draw.setEnabled(self._enabled and not self._drawing
                                 and self._shape is not None)
        self.btn_close_poly.setEnabled(self._drawing and len(self._points) >= 3)
        self.btn_undo_pt.setEnabled(self._drawing and bool(self._points))
        self.btn_del_last.setEnabled(self._drawing or has)
        self.btn_del_all.setEnabled(has)
        if self._drawing:
            self.info.setText(
                f"Disegno in corso: {len(self._points)} vertici. Clic = vertice, "
                f"doppio clic = chiudi.")
            return
        if not has:
            self.info.setText(
                "Nessun perimetro. Se vedi un distacco reale, disegnane il "
                "contorno: la sua area entra nelle statistiche e nella carta.")
            return
        tot = sum(d['area_px'] for d in self._dets)
        txt = f"{len(self._dets)} perimetr{'o' if len(self._dets) == 1 else 'i'}: "
        parts = []
        for d in self._dets:
            parts.append(f"R{d['id']} " + (f"{d['area_m2']:.2f} m²" if d.get('area_m2')
                                           is not None else f"{d['area_px']:.0f} px²"))
        txt += ", ".join(parts)
        if self._gsd:
            txt += f"  —  totale {tot * self._gsd ** 2:.2f} m²"
        self.info.setText(txt)

    def overlay(self):
        """RGBA overlay of the crop (also used by tests)."""
        if self._shape is None:
            return None
        h, w = self._shape
        ov = np.zeros((h, w, 4), np.uint8)
        th = max(1, min(h, w) // 250 + 1)
        ox, oy = self._offset
        for d in self._dets:
            pts = (np.asarray(d['points'], np.int32) - [ox, oy]).reshape(-1, 1, 2)
            # outline only: the inside must stay visible (rock, displacement)
            cv2.polylines(ov, [pts], True, (0, 0, 0, 200), th + 3, cv2.LINE_AA)
            cv2.polylines(ov, [pts], True, EDGE, th + 1, cv2.LINE_AA)
            cx, cy = d['centroid'][0] - ox, d['centroid'][1] - oy
            fs = max(0.4, min(h, w) / 600)
            cv2.putText(ov, f"R{d['id']}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (0, 0, 0, 255), th + 2, cv2.LINE_AA)
            cv2.putText(ov, f"R{d['id']}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, EDGE, th, cv2.LINE_AA)
        if self._drawing and self._points:
            pts = list(self._points)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(ov, a, b, DRAW, th + 1, cv2.LINE_AA)
            if self._mouse is not None:
                cv2.line(ov, pts[-1], self._mouse, DRAW, th, cv2.LINE_AA)
                if len(pts) >= 2:
                    cv2.line(ov, self._mouse, pts[0], (0, 229, 255, 140), 1, cv2.LINE_AA)
            r = max(3, th + 2)
            for p in pts:
                cv2.circle(ov, p, r, DRAW, -1, cv2.LINE_AA)
                cv2.circle(ov, p, r, (255, 255, 255, 255), 1, cv2.LINE_AA)
        return ov

    def redraw(self):
        ov = self.overlay()
        for v in self._viewers:
            img = v._image_gray
            if ov is None or img is None:
                continue
            if img.shape[:2] == ov.shape[:2]:
                v.set_overlay(ov)
            else:                                   # diff tab may be trimmed
                hh, ww = img.shape[:2]
                v.set_overlay(np.ascontiguousarray(ov[:hh, :ww]))
