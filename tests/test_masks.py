"""Masks tab: ROIs stay on the same rock across alignment, the preview
matches the analysed mask, PNG import keeps holes, JSON keeps the frame."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dic_app.core.mask_manager import (  # noqa: E402
    MaskManager, ROIDefinition, ROIMode, ROIType)


def R(t, m, pts):
    return ROIDefinition(t, m, pts)


def test_preview_matches_mask_when_exclude_drawn_first():
    mm = MaskManager((200, 200))
    mm.add_roi(R(ROIType.RECTANGLE, ROIMode.EXCLUDE, [(50, 50), (100, 100)]))
    mm.add_roi(R(ROIType.RECTANGLE, ROIMode.INCLUDE, [(0, 0), (199, 199)]))
    assert mm.generate_mask()[75, 75] == 0
    ov = mm.get_roi_overlay()
    assert ov[75, 75, 0] > ov[75, 75, 1]          # red, not green


@pytest.mark.parametrize("t,pts", [
    (ROIType.RECTANGLE, [(10, 10), (60, 40)]),
    (ROIType.ELLIPSE, [(100, 100), (130, 120)]),
    (ROIType.POLYGON, [(10, 150), (60, 150), (40, 190)]),
    (ROIType.FREEHAND, [(150, 10), (190, 20), (170, 60), (140, 40)]),
])
def test_every_shape(t, pts):
    mm = MaskManager((200, 200))
    mm.add_roi(R(t, ROIMode.INCLUDE, pts))
    assert (mm.generate_mask() > 0).sum() > 100
    mm.get_roi_overlay()


def test_png_import_keeps_holes(tmp_path):
    src = np.zeros((300, 300), np.uint8)
    cv2.rectangle(src, (20, 20), (280, 280), 255, -1)
    cv2.circle(src, (150, 150), 60, 0, -1)
    p = str(tmp_path / "m.png")
    cv2.imwrite(p, src)
    mm = MaskManager((300, 300))
    mm.load_mask(p)
    got = mm.generate_mask()
    assert got[150, 150] == 0
    assert (got != src).mean() < 0.01


def test_reframe_is_exact_translation():
    mm = MaskManager((500, 600), (0, 0))
    mm.add_roi(R(ROIType.RECTANGLE, ROIMode.EXCLUDE, [(200, 150), (260, 210)]))
    crop = mm.reframed((400, 450), (40, 30))
    full = mm.generate_mask()
    assert np.array_equal(crop.generate_mask(), full[30:430, 40:490])
    back = crop.reframed((500, 600), (0, 0))
    assert np.array_equal(back.generate_mask(), full)


def test_json_roundtrip_across_frames(tmp_path):
    a = MaskManager((500, 600), (0, 0))
    a.add_roi(R(ROIType.RECTANGLE, ROIMode.EXCLUDE, [(200, 150), (260, 210)]))
    p = str(tmp_path / "r.json")
    a.export_rois(p)
    b = MaskManager((400, 450), (40, 30))
    b.import_rois(p)
    assert np.array_equal(b.generate_mask(), a.generate_mask()[30:430, 40:490])


def test_rois_before_and_after_alignment_stay_on_the_rock():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from test_filter_advisor import shot
    from dic_app.core.image_registration import (
        AlignmentParameters, AlignmentResult, ImageRegistration)
    from dic_app.gui.main_window import DICMainWindow
    from dic_app.io.image_loader import ImageData

    win = DICMainWindow()
    pr = win.project
    a, b = shot(), shot(4, 3, seed=1)
    for im in (a, b):
        pr.images_data.append(ImageData(filepath="", image_gray=im,
                                        width=im.shape[1], height=im.shape[0]))
    ed = win.roi_editor
    win._on_tab_changed(3)                         # full photo
    ed._mask_manager.add_roi(R(ROIType.RECTANGLE, ROIMode.EXCLUDE, [(380, 380), (420, 420)]))
    ed.mask_updated.emit(ed._mask_manager)

    # Alignment with a large crop (60, 45) -> (840, 870)
    x0, y0, x1, y1 = 60, 45, 840, 870
    res = AlignmentResult(transform_matrix=np.eye(3), crop_bbox=(x0, y0, x1, y1))
    pr.alignment_result, pr.alignment_params = res, AlignmentParameters()
    pr.aligned_ref = a[y0:y1, x0:x1].copy()
    pr.aligned_def = b[y0:y1, x0:x1].copy()
    pr.aligned_def_index = 1

    win._on_tab_changed(3)                         # aligned (cropped) image
    # ROI drawn before alignment is shown at its new place
    assert ed._mask_manager.generate_mask()[400 - y0, 400 - x0] == 0
    ed._mask_manager.add_roi(R(ROIType.RECTANGLE, ROIMode.EXCLUDE, [(180, 180), (220, 220)]))
    ed.mask_updated.emit(ed._mask_manager)

    ref = pr.get_reference_image()
    mask = win._reference_mask(ref.shape)
    assert mask.shape == ref.shape
    assert mask[400 - y0, 400 - x0] == 0           # ROI drawn on the full photo
    assert mask[200, 200] == 0                     # ROI drawn on the crop
    assert mask[200 - y0 - 25, 200 - x0 - 25] == 255
    ys, xs = np.where(mask == 0)
    assert xs.min() == 180 and ys.min() == 180     # not shifted by the crop

    # Alignment reset: both ROIs go back onto the full photo
    pr.alignment_result = None
    pr.aligned_ref = pr.aligned_def = None
    full = win._reference_mask(a.shape)
    assert full[400, 400] == 0 and full[200 + y0, 200 + x0] == 0
    win.close()


def test_project_roundtrip_keeps_frame(tmp_path):
    from dic_app.io.project_manager import ProjectManager, ProjectState
    st = ProjectState()
    st.mask_manager = MaskManager((100, 120), (7, 9))
    st.mask_manager.add_roi(R(ROIType.RECTANGLE, ROIMode.EXCLUDE, [(1, 1), (20, 20)]))
    ProjectManager.save(st, str(tmp_path / "p.dicproj"))
    st2 = ProjectManager.load(str(tmp_path / "p.dicproj"))
    assert st2.mask_manager.frame_offset == (7, 9)


def test_editor_preview_exclude_is_red():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from dic_app.gui.roi_editor import ROIEditorWidget
    ed = ROIEditorWidget()
    ed.set_image(np.zeros((200, 200), np.uint8))
    ed.rb_exclude.setChecked(True)
    ed.shape_combo.setCurrentIndex(1)              # rectangle
    ed._start_drawing()
    ed._on_click(20, 20)
    ed._mouse_pos = (120, 120)
    ed._update_live_preview()
    ov = ed.viewer._overlay_buffer
    px = ov[70, 70]
    assert px[0] > px[2]                           # red, not blue
    ed._on_click(120, 120)
    assert "Area analizzata" in ed.coverage_label.text()
