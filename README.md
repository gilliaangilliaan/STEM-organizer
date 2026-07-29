<p align="center">
  <img src="logo.png" alt="STEM organizer logo" width="320">
</p>

<div align="center">

# STEM organizer

Organize, classify, and prepare multitrack music datasets.<br>
Automatically create 2- or 4-stems, tag genre/style, gender/reverb, vocal type, and key, align tracks, check integrity, and export charts.

**By:** Gilliaan & Bas Curtiz  
**Repo:** [github.com/gilliaangilliaan/STEM-organizer](https://github.com/gilliaangilliaan/STEM-organizer)  
**Video:** [How to install & use](https://youtu.be/L6RbF4N99tE)

</div>

<p align="center">
  <img src="stem-organizer-screenshots.gif" alt="STEM organizer screenshots" width="800">
</p>

PySide6 desktop app for building and auditing stem libraries. Hover **?** in any tab for per-control help.

## Tabs

| Tab | What it does |
|-----|----------------|
| **Classify** | Demucs RMS classify → group stems; optional SI-SDR quality filter; export organized folders |
| **Genre & Gender** | **Genre** — MAEST genre/style tags · **Gender** — EffNet gender + reverb · **Vocal type** — PANNs (Singing/Speech/…) |
| **Key** | In-house KeyNet CNN → `KEY` / Initial key metadata |
| **Match & Align** | Pair instrumental/vocal folders, organize pairs, align stems to a reference |
| **Rename** | Rule-based sample rename + optional instrument Auto-detect (PaSST / hear21passt) |
| **Integrity** | **Compression** — FLAC Detective lossless/lossy · **Corruption** — fast/deep verify + fix · **Convert** — batch to FLAC |
| **Charts** | Scan library roots → KPIs, donuts, genre/style breakdown, SI-SDR bars; balance or export PNG/PDF |

Downstream tabs can auto-fill paths from Classify output (`*_organized`).

## Requirements

- **Windows**
- **Python 3.10 or 3.11** on PATH (for `install-deps.bat` / `build.bat`)
- Disk space for PyTorch + Demucs (`site-packages\` beside the `.exe`, or `.venv` / `genre_gender_tagger\venv` from source)
- **NVIDIA GPU** optional (CUDA 12.4 for RTX 20/30/40, CUDA 12.8 for RTX 50-series); CPU works everywhere

## Quick start (from source)

```bat
install-deps.bat
.venv\Scripts\python.exe run_stem_organizer.py
```

`install-deps.bat` at the project root:

1. Creates `.venv` and installs core GUI + audio stack (PySide6, Demucs, scipy, flac-detective, …)
2. Downloads **ffmpeg**, **mp3val**, and **flac** next to the project (for decode / corruption tools)
3. Installs tagger deps into the shared `genre_gender_tagger\venv`:
   - Genre & Gender (transformers, onnxruntime-directml, …)
   - Rename Auto-detect (hear21passt, timm)
   - PANNs vocal type (`panns-inference`)
   - Key Detect (librosa, soxr, more-itertools)

Pick **GPU or CPU PyTorch** once when prompted.

## Build `.exe`

```bat
build.bat
cd dist\STEM-organizer
install-deps.bat
STEM-organizer.exe
```

Run `install-deps.bat` **in the same folder as `STEM-organizer.exe`** so wheels land in `site-packages\`.

### Model assets

| Feature | Path | In repo? |
|---------|------|----------|
| Key Detect | `key_tagger\checkpoints\nf50-q05-221125.pt` | **Yes** (~11 MB) |
| Genre / Gender ONNX | `genre_gender_tagger\models\` | Optional — downloaded on first run if missing |
| PANNs Cnn14 | `panns_tagger\models\` or `%USERPROFILE%\panns_data\` | Optional — ~470 MB on first run |

`build.bat` copies tagger scripts and bundled checkpoints/models present at build time.

## Metadata tags (Charts sources)

Charts reads these custom tags from your library:

| Chart | Tag |
|-------|-----|
| SI-SDR | `SDR` |
| Genre / Style | `GENRE`, `STYLE` |
| Gender / Reverb | `GENDER`, `REVERB` |
| Vocal type | `VOCAL_TYPE` |
| Keys | `KEY` |
| Compression | `COMPRESSION` |

## Project layout

```
STEM-organizer-Py6/
├── run_stem_organizer.py      # Entry point
├── install-deps.bat           # One-shot dependency installer
├── build.bat                  # PyInstaller → dist\STEM-organizer\
├── stem_organizer/            # PySide6 UI + dataset tools
├── genre_gender_tagger/       # MAEST + EffNet tagger
├── instrument_tagger/         # PaSST Auto-detect
├── panns_tagger/              # PANNs vocal type
└── key_tagger/                # KeyNet key detection
```

## License

[MIT](LICENSE)
