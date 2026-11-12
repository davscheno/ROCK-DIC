# ROCK-DIC — DIC Landslide Monitor

Desktop app (PyQt5) for rock-face / rockfall monitoring via Digital Image
Correlation on drone or fixed-camera photos. Version 1.1.0, MIT, entry point
`main.py`. UI strings are partly Italian (e.g. "Suggerisci parametri",
"Serie temporale").

## Environment
- Python 3.12 venv **outside OneDrive** (avoid syncing thousands of files):
  `%USERPROFILE%\.venvs\rock-dic`
- Run: `%USERPROFILE%\.venvs\rock-dic\Scripts\python.exe main.py`
- Tests: `%USERPROFILE%\.venvs\rock-dic\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`
  (`QT_QPA_PLATFORM=offscreen` is set by the tests)
- `rasterio` (GeoTIFF export) is optional and not installed; the app degrades
  gracefully without it.
- Verified with opencv-python 5.0, numpy 2.5, scipy 1.18, PyQt5 5.15.11.

## Layout
- `dic_app/core/` — computation, no Qt:
  - `dic_engine.py` — `DICEngine` / `DICParameters` / `DICResult`, 5 methods
    (`DICMethod`: ICGN default, TEMPLATE_NCC, OPTICAL_FLOW_FARNEBACK,
    PHASE_CORRELATION, FEATURE_MATCHING); `DICSequenceRunner`.
  - `subset_dic.py` — ZNCC search, pyramid guess, IC-GN refinement.
  - `image_registration.py` (alignment), `precrop.py` (crop of huge photos),
    `preprocessing.py` + `auto_filter.py` (filters), `mask_manager.py` (ROI),
    `param_advisor.py` (parameter suggestion / precision estimate),
    `change_detection.py` (vegetation/water masks, detachment classes),
    `postprocessing.py` (smoothing, strain, stats), `time_series.py`
    (multi-epoch), `geo_utils.py` (GPS/CRS/GeoTIFF).
- `dic_app/gui/` — Qt widgets; `main_window.py` holds the tabbed workflow,
  long jobs run in `QObject` workers on `QThread`.
- `dic_app/io/` — image + EXIF loading, `.dicproj` save/load
  (`project_manager.py`), PDF/CSV reports.
- `dic_app/utils/helpers.py` — logging, colormaps, NaN-safe filters.

## Workflow (pipeline)
Load images → (pre-crop) → align → filter → ROI/mask → DIC → validate
results → report / time series.

## Notes
- Image files and local data folders are git-ignored (see `.gitignore`).
- Imports use the `dic_app.` absolute prefix; `main.py` puts the repo root on
  `sys.path`.
