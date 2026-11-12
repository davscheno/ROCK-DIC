"""Smoke tests: every module imports, DIC engine recovers a known shift,
main window builds offscreen."""
import importlib
import os
import pkgutil
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import dic_app  # noqa: E402

MODULES = [m.name for m in pkgutil.walk_packages(dic_app.__path__, "dic_app.")]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def _speckle(h=300, w=300, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    img = rng.random((h, w)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)


@pytest.mark.parametrize("method", ["ICGN", "TEMPLATE_NCC", "OPTICAL_FLOW_FARNEBACK",
                                    "PHASE_CORRELATION", "FEATURE_MATCHING"])
def test_known_shift(method):
    from dic_app.core.dic_engine import DICEngine, DICParameters, DICMethod
    import cv2
    ref = _speckle()
    dx, dy = 3.4, -2.2
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    deff = cv2.warpAffine(ref, M, ref.shape[::-1], flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)
    p = DICParameters(method=getattr(DICMethod, method), subset_size=31,
                      step_size=15, search_radius_x=10, search_radius_y=10,
                      border_margin=40, max_displacement_px=8,
                      exclude_vegetation=False, exclude_water=False)
    res = DICEngine(p).run(ref, deff)
    u = np.nanmedian(res.u)
    v = np.nanmedian(res.v)
    assert abs(u - dx) < 0.3 and abs(v - dy) < 0.3, (u, v)


def test_main_window_builds():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from dic_app.gui.main_window import DICMainWindow
    w = DICMainWindow()
    w.show()
    app.processEvents()
    w.close()
