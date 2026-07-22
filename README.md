# STEM organizer (PySide6)

Organize, classify, and prepare multitrack music datasets. Automatically identify stem layouts, tag genres and vocals, align tracks, rename files, and build clean AI-ready datasets

PySide6 re-creation of the original [STEM organizer](https://github.com/gilliaangilliaan/STEM-organizer)
GUI (Tkinter / CustomTkinter). Same dark look, same four tabs, same backends —
only the GUI layer was rewritten.

## Tabs (unchanged from the original)

| Tab | Role |
|-----|------|
| **Classify**      | Demucs RMS classify → group → FLAC/WAV export; dedup, normalize, SI-SDR |
| **Match & Align** | Pair acapella / instrumental, organize, align to original timeline |
| **Genre & Gender**| MAEST genre/style + EffNet gender + dry/wet reverb (PaSST, ONNX) |
| **Rename**        | Rule-based sample rename + optional instrument Auto-detect (PaSST) |

Plus the **Stem Player** window (waveform + transport + per-stem mute/solo).

## Quick start (from source)

```bat
pip install -r requirements.txt
python run_stem_organizer.py
```

For the optional Genre & Gender and Rename Auto-detect models, run the bundled
installers inside `genre_gender_tagger\` and `instrument_tagger\` (or re-run
the original `install-deps.bat` from the Tk project — the venv folders are not
included here).

## Layout

```
STEM-organizer-Py6\
├── run_stem_organizer.py            # entry
├── classify_backend.py              # tk-free Classify (RMS) + SI-SDR worker layer
├── pair_matcher.py, stem_align.py   # copied verbatim from the Tk project
├── ffmpeg_bootstrap.py, deps_bootstrap.py, resource_monitor.py, …
├── track_renamer\engine\            # copied verbatim
├── track_renamer\folder_scanner.py, audio_preview.py, instrument_enrich.py, category_palette.py
├── genre_gender_tagger\, instrument_tagger\
├── stem_organizer\                  # PySide6 GUI
│   ├── app.py                       # QMainWindow (frameless + custom title bar)
│   ├── main_entry.py                # splash + single-instance + startup
│   ├── theme.py                     # COLORS / DARK tokens + QSS + QPalette
│   ├── settings_store.py            # JSON load / save / autosave
│   ├── splash.py                    # QSplashScreen + StartupWorker
│   ├── widgets\                     # title bar, action bar, log, status, sections, dialogs
│   ├── workers\                     # QThread adapters for Classify / Pair / Tagger
│   ├── tabs\                        # classify / pair_finder / genre_gender / rename
│   ├── player\                      # stem player window + audio engine + waveform widget
│   └── renamer\                     # rename GUI (rules + preview + audio bar)
└── logo.png, logo.ico, ffmpeg\, settings.json
```

## Architecture

- **Threading**: `QThread` subclasses + Qt signals (`Qt.QueuedConnection`) replace
  the original `queue.Queue` + `after(100, ...)` drain loop. The Classify workers
  reuse the verbatim `threading.Thread`-based `Worker` / `SdrWorker` classes
  (see `classify_backend.py`) inside a `BaseWorker` adapter that bridges tuples
  to Qt signals.
- **Waveforms**: `QPainterPath` filled polygon replaces the original
  `tk.Canvas.create_polygon`. Audio engine + TrackState are pure Python +
  numpy + sounddevice, ported near-verbatim.
- **Custom title bar**: frameless `QMainWindow` with `Qt.FramelessWindowHint`,
  drag + edge-resize reimplemented on `mousePress/Move`, Win11 rounded corners
  via `DwmSetWindowAttribute(DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)`.
- **Rename preview**: `QTableView` + `QAbstractTableModel` replaces the
  hand-rolled virtualized canvas list, with the same lazy-compute worker
  (priority visible rows first) and ANALYZE LOG sub-panel.
- **Theme**: `COLORS` / `DARK` dicts ported verbatim from `ui_theme.py`; built
  into a single QSS stylesheet + `QPalette`.

## License

MIT (same as the original).
