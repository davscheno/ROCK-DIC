"""Regular inspection grid (GridInspectionDialog): cells must tile the
image exactly and carry the right row/column labels."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dic_app.gui.results_viewer import GridInspectionDialog  # noqa: E402

SHAPES = [(1200, 1440), (1169, 1565), (350, 350), (120, 90), (4000, 6000)]


@pytest.mark.parametrize("h,w", SHAPES)
@pytest.mark.parametrize("cs", [100, 350, 700])
def test_cells_tile_image_exactly(h, w, cs):
    cells, rc, n_rows, n_cols = GridInspectionDialog.build_cells(h, w, cs)
    count = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in cells:
        assert x1 > x0 and y1 > y0
        count[y0:y1, x0:x1] += 1
    assert (count == 1).all()          # no gaps, no overlaps
    assert len(cells) == n_rows * n_cols


@pytest.mark.parametrize("h,w", SHAPES)
def test_labels_match_position(h, w):
    cells, rc, n_rows, n_cols = GridInspectionDialog.build_cells(h, w, 350)
    xs = sorted({c[0] for c in cells})
    ys = sorted({c[1] for c in cells})
    for (x0, y0, _, _), (r, c) in zip(cells, rc):
        assert ys.index(y0) == r and xs.index(x0) == c


def test_cells_have_similar_size():
    cells, *_ = GridInspectionDialog.build_cells(1169, 1565, 350)
    widths = {x1 - x0 for x0, _, x1, _ in cells}
    heights = {y1 - y0 for _, y0, _, y1 in cells}
    assert max(widths) - min(widths) <= 1
    assert max(heights) - min(heights) <= 1


def test_dialog_label_uses_real_row_col():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    img = np.zeros((1200, 1440), np.uint8)
    d = GridInspectionDialog(img, img, cell_size=350)
    d._show_cell(5)
    r, c = d._cell_rc[5]
    assert f"[riga {r + 1}, col {c + 1}]" in d.cell_label.text()
