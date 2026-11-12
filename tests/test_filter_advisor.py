"""Filters tab: every filter works, the guided choice picks what DIC
measures best, the wizard applies it, the panel never shares its pipeline
with the project."""
import os
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dic_app.core import filter_advisor as fa  # noqa: E402
from dic_app.core.preprocessing import FILTER_REGISTRY, FilterPipeline  # noqa: E402

N = 900
_rng = np.random.default_rng(3)
_t = np.zeros((N + 80, N + 80), np.float32)
for s, a in [(1.5, .5), (4, 1), (12, 1.2), (40, 1.5)]:
    _t += a * cv2.GaussianBlur(_rng.standard_normal(_t.shape).astype(np.float32), (0, 0), s)
TEX = cv2.normalize(_t, None, 40, 200, cv2.NORM_MINMAX)
YY, XX = np.mgrid[0:N, 0:N].astype(np.float32)


def shot(dx=0.0, dy=0.0, gain=1.0, grad=0.0, shadow=False, noise=1.5, seed=0):
    rng = np.random.default_rng(seed)
    m = cv2.remap(TEX, XX + 40 - dx, YY + 40 - dy, cv2.INTER_CUBIC)
    light = gain * (1 + grad * (XX / N - .5))
    if shadow:
        light = light * np.where(XX + 0.6 * YY > 0.8 * N, 0.4, 1.0)
    return np.clip(m * light + rng.normal(0, noise, m.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", list(FILTER_REGISTRY))
def test_every_filter_runs(name):
    img = shot()
    p = FilterPipeline()
    p.add_step(name)
    out = p.apply(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_same_light_measures_no_change():
    d, reg, _ = fa.measure(shot(), shot(6.3, -3.1, seed=1))
    assert d.registered
    assert d.lighting_level() == 'same'
    assert d.noise_level() == 'low'


def test_clean_pair_recommends_no_filter():
    rec = fa.evaluate(shot(), shot(6.3, -3.1, seed=1))
    assert len(rec.best.pipeline) == 0
    assert rec.scores[0].verdict == "CONSIGLIATO"


def test_shadow_pair_recommends_wallis():
    ref = shot(noise=3)
    deff = shot(6.3, -3.1, gain=.7, grad=.6, shadow=True, noise=3, seed=1)
    rec = fa.evaluate(ref, deff)
    assert rec.answers['lighting'] == 'strong'
    assert rec.best.pipeline.steps[0]['filter_name'] == 'Wallis Filter'
    assert rec.best.utility > rec.scores[0].utility


def test_answers_override_measurement():
    d = fa.Diagnostics(noise=1.0)
    a = fa.resolve_answers({'noise': 'high'}, d)
    assert a['noise'] == 'high'
    names = [c.name for c in fa.build_candidates({'noise': 'high'}, d)]
    assert any("Gaussiano" in n for n in names)
    assert any("Non-Local" in n for n in names)
    fast = [c.name for c in fa.build_candidates({'noise': 'high', 'priority': 'speed'}, d)]
    assert not any("Non-Local" in n for n in fast)


def test_unsafe_filters_never_proposed():
    d = fa.Diagnostics(noise=9, lighting_change=.5, shadow_fraction=.3,
                       low_contrast=True, dark_or_saturated_pct=20, registered=True)
    used = {s['filter_name'] for c in fa.build_candidates({'priority': 'precision'}, d)
            for s in c.pipeline.steps}
    assert used <= {'Wallis Filter', 'CLAHE', 'Gaussian Blur', 'Non-Local Means'}


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_wizard_auto_flow_and_apply(qapp):
    from dic_app.gui.filter_wizard import FilterWizardDialog
    ref = shot(noise=3)
    deff = shot(6.3, -3.1, gain=.7, grad=.6, shadow=True, noise=3, seed=1)
    dlg = FilterWizardDialog(ref, deff, auto_start=True)
    t = time.time()
    while dlg.stack.currentIndex() != dlg.PAGE_RESULT and time.time() - t < 120:
        qapp.processEvents()
        time.sleep(0.02)
    assert dlg.stack.currentIndex() == dlg.PAGE_RESULT
    assert dlg.table.rowCount() >= 2
    dlg.btn_apply_best.click()
    assert dlg.result() == dlg.Accepted
    assert dlg.chosen_pipeline is not dlg._rec.best.pipeline
    assert dlg.chosen_pipeline.to_dict() == dlg._rec.best.pipeline.to_dict()


def test_panel_does_not_share_pipeline(qapp):
    from dic_app.gui.filter_panel import FilterPanelWidget
    panel = FilterPanelWidget()
    got = []
    panel.filter_applied.connect(got.append)
    p = FilterPipeline()
    p.add_step('CLAHE')
    panel.set_pipeline(p)
    assert len(p) == 1                       # caller's pipeline untouched
    panel._apply_pipeline()
    applied = got[-1]
    panel.set_pipeline(applied)              # e.g. restore from the project
    assert len(applied) == 1                 # was cleared before the fix
    panel._clear_all()
    assert len(applied) == 1                 # edits do not reach the project
    assert "CLAHE" in panel.applied_label.text()


def test_filter_tab_uses_original_images(qapp):
    from dic_app.gui.main_window import DICMainWindow
    from dic_app.io.image_loader import ImageData
    win = DICMainWindow()
    a, b = shot(), shot(4, 2, seed=1)
    for im in (a, b):
        win.project.images_data.append(ImageData(filepath="", image_gray=im,
                                                 width=N, height=N))
    p = FilterPipeline(); p.add_step('CLAHE')
    win.project.filter_pipeline = p
    win.project.preprocessed_images = [p.apply(a), p.apply(b)]
    win._on_tab_changed(1)
    assert win.filter_panel._reference_image is a
    assert win.filter_panel._deformed_image is b
    win.close()
