"""The images handed to the DIC engine are the FILTERED ones, for the
right photos, also after adding/removing images and with alignment."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_filter_advisor import shot  # noqa: E402


@pytest.fixture
def setup(monkeypatch):
    from PyQt5.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])  # noqa: F841
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    import dic_app.gui.main_window as mw
    from dic_app.core.preprocessing import FilterPipeline
    from dic_app.io.image_loader import ImageData

    seen = {}

    class FakeWorker:
        def __init__(self, engine, ref, deff, mask, **kw):
            seen['ref'], seen['def'] = ref, deff
            raise RuntimeError("captured")
    monkeypatch.setattr(mw, "DICWorker", FakeWorker)

    win = mw.DICMainWindow()
    raw = [shot(seed=0), shot(3, 2, seed=1), shot(6, 4, seed=2)]
    for i, im in enumerate(raw):
        win.project.add_image(ImageData(filepath=f"img{i}", image_gray=im,
                                        width=im.shape[1], height=im.shape[0]), f"img{i}")
        win.ref_combo.addItem(f"img{i}")
        win.image_list.addItem(f"img{i}")
    p = FilterPipeline()
    p.add_step('Wallis Filter')
    win._on_filter_applied(p)

    def run(def_idx):
        win._update_analysis_summary()
        win.deformed_combo.setCurrentIndex(win.deformed_combo.findData(def_idx))
        seen.clear()
        with pytest.raises(RuntimeError):
            win._run_analysis()
        return seen['ref'], seen['def']

    yield win, raw, p, run, ImageData
    win.close()


def test_filtered_images_reach_dic(setup):
    win, raw, p, run, _ = setup
    for k in (1, 2):
        r, d = run(k)
        assert np.array_equal(r, p.apply(raw[0]))
        assert np.array_equal(d, p.apply(raw[k]))
    assert "Wallis" in win.analysis_summary.toPlainText()


def test_image_added_after_filtering_is_filtered(setup):
    win, raw, p, run, ImageData = setup
    extra = shot(5, 5, seed=3)
    win.project.add_image(ImageData(filepath="img3", image_gray=extra,
                                    width=900, height=900), "img3")
    r, d = run(3)
    assert np.array_equal(d, p.apply(extra))


def test_remove_image_keeps_filtered_images_in_step(setup):
    win, raw, p, run, _ = setup
    win.image_list.setCurrentRow(0)
    win._remove_image()                       # remove the reference
    assert win.project.reference_index == 0
    r, d = run(1)
    assert np.array_equal(r, p.apply(raw[1]))  # was the removed image before
    assert np.array_equal(d, p.apply(raw[2]))


def test_remove_middle_image_keeps_reference(setup):
    win, raw, p, run, _ = setup
    win.ref_combo.setCurrentIndex(2)           # reference = img2
    assert win.project.reference_index == 2
    win.image_list.setCurrentRow(1)
    win._remove_image()
    assert win.project.reference_index == 1    # still img2
    r, d = run(0)
    assert np.array_equal(r, p.apply(raw[2]))
    assert np.array_equal(d, p.apply(raw[0]))


def test_aligned_images_are_filtered(setup):
    win, raw, p, run, _ = setup
    from dic_app.core.image_registration import AlignmentParameters, ImageRegistration
    pr = win.project
    reg = ImageRegistration(AlignmentParameters())
    al, res = reg.align(pr.get_pre_alignment_ref(), pr.get_pre_alignment_def(1))
    x0, y0, x1, y1 = res.crop_bbox
    pr.alignment_result, pr.alignment_params = res, AlignmentParameters()
    pr.aligned_ref = pr.get_pre_alignment_ref()[y0:y1, x0:x1].copy()
    pr.aligned_def, pr.aligned_def_index = al, 1
    r, d = run(1)
    assert np.array_equal(r, p.apply(raw[0])[y0:y1, x0:x1])
    assert d is al


def test_unapplied_panel_edits_are_flagged(setup):
    win, raw, p, run, _ = setup
    win.filter_panel._add_filter()             # edited, not applied
    win._update_analysis_summary()
    assert "non applicate" in win.analysis_summary.toPlainText()
