"""The project remembers which image pair produced the DIC results, so
reopening a project shows the results over the right deformed image."""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dic_app.core.dic_engine import DICEngine, DICParameters  # noqa: E402
from dic_app.io.image_loader import ImageLoader  # noqa: E402
from dic_app.io.project_manager import ProjectManager, ProjectState  # noqa: E402


def _images(tmp_path, n=3, h=240, w=320):
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.random((h + 40, w + 40, 3)).astype(np.float32), (0, 0), 2)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    paths = []
    for i in range(n):
        img = base[20 + i * 3:20 + i * 3 + h, 20:20 + w]      # each one shifted
        p = str(tmp_path / f"img{i}.bmp")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


def _project(paths, pair):
    st = ProjectState()
    for p in paths:
        st.images_data.append(ImageLoader.load(p))
        st.image_paths.append(p)
    st.reference_index = pair[0]
    st.dic_params = DICParameters(subset_size=31, step_size=20, border_margin=30,
                                  exclude_vegetation=False, exclude_water=False)
    ref = st.get_reference_image()
    deff = st.get_deformed_image(pair[1])
    st.results = [DICEngine(st.dic_params).run(ref, deff)]
    st.results_pair = pair
    return st


def test_pair_saved_and_restored(tmp_path):
    paths = _images(tmp_path)
    st = _project(paths, (0, 2))
    proj = str(tmp_path / "p.dicproj")
    ProjectManager.save(st, proj)
    assert json.load(open(os.path.join(proj, "project.json")))["results_pair"] == [0, 2]
    st2 = ProjectManager.load(proj)
    assert st2.results_pair == (0, 2) and len(st2.results) == 1


def test_missing_analysed_image_discards_results(tmp_path):
    paths = _images(tmp_path)
    st = _project(paths, (0, 2))
    proj = str(tmp_path / "p.dicproj")
    ProjectManager.save(st, proj)
    os.remove(paths[2])
    st2 = ProjectManager.load(proj)
    assert st2.results == [] and st2.results_pair is None


def test_index_remapped_when_other_image_missing(tmp_path):
    paths = _images(tmp_path)
    st = _project(paths, (0, 2))
    proj = str(tmp_path / "p.dicproj")
    ProjectManager.save(st, proj)
    os.remove(paths[1])
    st2 = ProjectManager.load(proj)
    assert st2.results_pair == (0, 1)
    assert st2.image_paths[1].endswith("img2.bmp")


def test_reopened_project_inspects_analysed_image(tmp_path):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from dic_app.gui.main_window import DICMainWindow
    paths = _images(tmp_path)
    proj = str(tmp_path / "p.dicproj")
    ProjectManager.save(_project(paths, (0, 2)), proj)

    win = DICMainWindow()
    win.project = ProjectManager.load(proj)
    win._restore_project_ui()
    shown = win.results_viewer._def_image
    expected = win.project.get_deformed_image(2)
    wrong = win.project.get_deformed_image(1)
    assert np.array_equal(shown, expected)
    assert not np.array_equal(shown, wrong)
    win.close()
