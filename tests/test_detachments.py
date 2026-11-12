"""Real detachment perimeters: drawing in the three review dialogs,
statistics, map, project save/load, PDF report, CSV and GeoJSON."""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dic_app.core import detachment as dt  # noqa: E402

SQ = [(100, 100), (200, 100), (200, 150), (100, 150)]      # 100 x 50 px


def test_geometry():
    d = dt.make_detachment(SQ, gsd=0.02)
    assert d['area_px'] == pytest.approx(5000)
    assert d['area_m2'] == pytest.approx(5000 * 0.0004)
    assert d['perimeter_px'] == pytest.approx(300)
    assert d['centroid'] == pytest.approx([150, 125])


def test_union_does_not_double_count():
    a = {'id': 1, 'status': 'validated', 'detachments': [dt.make_detachment(SQ, 0.02)]}
    b = {'id': 2, 'kind': 'change', 'status': 'pending',
         'detachments': [dt.make_detachment(SQ, 0.02)]}
    rej = {'id': 3, 'status': 'rejected', 'detachments': [dt.make_detachment(SQ, 0.02)]}
    sm = dt.summarize([a, rej], [b], 0.02, (400, 400))
    assert sm['n_perimeters'] == 2                    # rejected zone excluded
    assert sm['sum_area_px'] == pytest.approx(10000)
    assert sm["union_area_px"] == pytest.approx(5000)           # same scar once


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def dic():
    from test_filter_advisor import shot
    from dic_app.core.dic_engine import DICEngine, DICParameters
    a = shot()[:500, :500].copy()
    b = shot(2, 1, seed=1)[:500, :500].copy()
    p = DICParameters(subset_size=31, step_size=15, border_margin=20,
                      exclude_vegetation=False, exclude_water=False)
    return a, b, DICEngine(p).run(a, b)


def test_drawer_offsets_and_overlay(qapp):
    from dic_app.gui.detachment_drawer import DetachmentDrawer
    from dic_app.gui.image_viewer import ImageViewer
    v = ImageViewer()
    v.set_image(np.zeros((120, 160), np.uint8))
    got = []
    dr = DetachmentDrawer([v])
    dr.changed.connect(got.append)
    dr.set_context((300, 200), (120, 160), gsd=0.01)
    dr.start()
    for p in [(10, 10), (60, 10), (60, 60), (60, 60), (10, 60)]:   # repeat = dbl-click
        dr.add_point(*p)
    dr.finish()
    d = got[-1][0]
    assert d['points'] == [[310, 210], [360, 210], [360, 260], [310, 260]]
    assert d['area_m2'] == pytest.approx(2500 * 1e-4)
    ov = dr.overlay()
    assert ov[10, 35, 3] > 0 and ov[35, 35, 3] == 0      # outline only
    dr.delete_last()
    assert got[-1] == []


def _rgb(a):
    return np.dstack([a] * 3)


def test_zone_validation_draw_validates_and_cancel_restores(qapp, dic):
    from dic_app.gui.results_viewer import ZoneValidationDialog
    a, b, res = dic
    zone = {'id': 1, 'bbox': (5, 5, 12, 12), 'area_points': 49,
            'max_displacement': 2.0, 'mean_displacement': 1.5, 'status': 'pending'}
    dlg = ZoneValidationDialog([zone], res, a, b, _rgb(a), _rgb(b), gsd=0.02)
    dr = dlg.drawer
    dr.start()
    for p in [(5, 5), (40, 5), (40, 40)]:
        dr.add_point(*p)
    dr.finish()
    assert zone['status'] == 'validated'
    assert len(zone['detachments']) == 1
    x0 = dr._offset[0]
    assert zone['detachments'][0]['points'][0][0] == 5 + x0
    assert zone['detachments'][0]['n_grid_points'] >= 0
    dlg._cancel()
    assert 'detachments' not in zone and zone['status'] == 'pending'


def test_grid_cell_drawing_marks_cell(qapp, dic):
    from dic_app.gui.results_viewer import GridInspectionDialog
    a, b, res = dic
    dlg = GridInspectionDialog(a, b, _rgb(a), _rgb(b), cell_size=250, gsd=0.02,
                               result=res)
    dlg._show_cell(3)
    dr = dlg.drawer
    dr.start()
    for p in [(10, 10), (80, 10), (80, 90)]:
        dr.add_point(*p)
    dr.finish()
    marked = dlg.get_marked_cells_with_detachments()
    assert len(marked) == 1
    bbox, dets = marked[0]
    assert bbox == dlg._cells[3]
    assert dets[0]['points'][0] == [bbox[0] + 10, bbox[1] + 10]


def test_viewer_stats_map_and_project_roundtrip(qapp, dic, tmp_path):
    from dic_app.gui.main_window import DICMainWindow
    from dic_app.io.image_loader import ImageData
    from dic_app.io.project_manager import ProjectManager
    a, b, res = dic
    pa, pb = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    cv2.imwrite(pa, a)
    cv2.imwrite(pb, b)
    win = DICMainWindow()
    pr = win.project
    for p in (pa, pb):
        from dic_app.io.image_loader import ImageLoader
        pr.add_image(ImageLoader.load(p), p)
    pr.gsd = 0.02
    pr.results, pr.results_pair = [res], (0, 1)
    win._show_results_with_images()
    rv = win.results_viewer
    det = dt.make_detachment([(150, 150), (260, 150), (260, 230), (150, 230)], 0.02, res)
    rv._add_manual_zone_from_bbox((140, 140, 270, 240), [det])
    assert rv.zone_table.item(0, 4).text().endswith("m²")
    assert "Distacchi reali" in rv.stats_text.toPlainText()
    ov = rv.viewer._overlay_buffer
    assert tuple(ov[190, 200, :3]) != (0, 0, 0) and ov[150, 200, 1] > 200   # outline

    proj = str(tmp_path / "p.dicproj")
    win._do_save(proj)
    meta = json.load(open(os.path.join(proj, "project.json")))
    z = meta['active_zones'][0]
    assert isinstance(z['bbox'][0], int)               # not strings
    assert z['detachments'][0]['area_m2'] == pytest.approx(det['area_m2'])
    win.close()

    win2 = DICMainWindow()
    win2.project = ProjectManager.load(proj)
    win2._restore_project_ui()
    zz = win2.results_viewer.get_active_zones()
    assert len(zz) == 1 and len(zz[0]['detachments']) == 1
    assert "Distacchi reali" in win2.results_viewer.stats_text.toPlainText()
    win2.close()


def test_report_pdf_csv_geojson(dic, tmp_path):
    from dic_app.io.report_generator import ReportGenerator
    a, b, res = dic
    zone = {'id': 1, 'bbox': (5, 5, 12, 12), 'area_points': 49,
            'max_displacement': 2.0, 'mean_displacement': 1.5, 'status': 'validated',
            'detachments': [dt.make_detachment([(150, 150), (260, 150), (260, 230)],
                                               0.02, res)]}
    rg = ReportGenerator(res, {'output_dir': str(tmp_path),
                               'formats': {'pdf': True, 'csv': False}},
                         base_image=_rgb(a), deformed_image=b, deformed_image_rgb=_rgb(b),
                         active_zones=[zone], gsd=0.02,
                         gps_info={'latitude': 44.4, 'longitude': 8.9},
                         geo_frame={'original_shape': (500, 500), 'pixel_offset': (0, 0)})
    rg.generate_all()
    assert os.path.getsize(tmp_path / "report_dic.pdf") > 10000
    gj = json.load(open(tmp_path / "distacchi_reali.geojson"))
    f = gj['features'][0]
    lon, lat = f['geometry']['coordinates'][0][0]
    assert 8.8 < lon < 9.0 and 44.3 < lat < 44.5
    assert f['properties']['area_m2'] == pytest.approx(zone['detachments'][0]['area_m2'])
    csv_txt = open(tmp_path / "distacchi_reali.csv", encoding="utf-8").read()
    assert "Z1-R1" in csv_txt


def test_outline_only_no_fill(qapp):
    from dic_app.gui.detachment_drawer import DetachmentDrawer
    from dic_app.gui.image_viewer import ImageViewer
    v = ImageViewer()
    v.set_image(np.zeros((120, 160), np.uint8))
    dr = DetachmentDrawer([v])
    dr.set_context((0, 0), (120, 160),
                   [dt.make_detachment([(10, 10), (110, 10), (110, 100), (10, 100)])])
    ov = dr.overlay()
    assert ov[55, 60, 3] == 0                       # inside left transparent
    assert ov[10, 60, 3] > 0 and ov[10, 60, 1] > 200  # yellow outline


def test_gsd_in_results_panel_updates_everything(qapp, dic, tmp_path):
    from dic_app.gui.main_window import DICMainWindow
    from dic_app.io.image_loader import ImageLoader
    a, b, res = dic
    pa, pb = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    cv2.imwrite(pa, a)
    cv2.imwrite(pb, b)
    win = DICMainWindow()
    pr = win.project
    for p in (pa, pb):
        pr.add_image(ImageLoader.load(p), p)
    pr.gsd = 0.02
    pr.results, pr.results_pair = [res], (0, 1)
    win._show_results_with_images()
    rv = win.results_viewer
    assert rv.gsd_spin.value() == pytest.approx(0.02)
    det = dt.make_detachment([(150, 150), (250, 150), (250, 250), (150, 250)], 0.02, res)
    rv._add_manual_zone_from_bbox((140, 140, 260, 260), [det])
    assert rv.zone_table.item(0, 4).text().startswith("4.00")

    rv.gsd_spin.setValue(0.05)                      # edited in the right bar
    z = rv.get_active_zones()[0]
    assert z['detachments'][0]['area_m2'] == pytest.approx(10000 * 0.0025)
    assert rv.zone_table.item(0, 4).text().startswith("25.00")
    assert "25.00 m" in rv.stats_text.toPlainText()
    assert pr.gsd == pytest.approx(0.05)
    assert win.gsd_spin.value() == pytest.approx(0.05)

    win.gsd_spin.setValue(0.01)                     # edited in the Project tab
    assert rv.gsd_spin.value() == pytest.approx(0.01)
    assert z['detachments'][0]['area_m2'] == pytest.approx(1.0)
    win.close()
