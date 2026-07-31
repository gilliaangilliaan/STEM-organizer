<p align="center">
  <img src="logo.png" alt="STEM organizer logo" width="320">
</p>

<div align="center">

# STEM organizer

Organize, classify, prepare and balance audio datasets.<br>
Automatically create 2- or 4-stems, tag genre/style, gender/reverb, vocal type, and key, align tracks, check integrity, and export charts.

**By:** Gilliaan & Bas Curtiz  
**Video:** [How to install & use](https://youtu.be/9xvfCQVhs1Y)

</div>

<p align="center">
  <img src="screenshots-v107.gif" alt="STEM organizer screenshots" width="800">
</p>

PySide6 desktop app for building and auditing stem libraries. Hover **?** in any tab for per-control help.

## Tabs

| Tab | What it does |
|-----|----------------|
| **Classify** | [Demucs](https://github.com/facebookresearch/demucs) RMS classify → group stems<br>optional [SI-SDR](https://source-separation.github.io/tutorial/basics/evaluation.html#si-sdr) quality filter; export organized folders |
| **Genre & Gender** | **Genre** — [MAEST](https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e) genre/style tags<br>**Gender** — [EffNet gender](https://essentia.upf.edu/models.html#voice-gender) (male/female) + In-house trained reverb (dry/wet)<br>**Vocal type** — [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) (Singing/Speech/Rapping/Humming/Choir) |
| **Key** | In-house trained KeyNet CNN → `KEY` / Initial key — [outperforms](https://docs.google.com/spreadsheets/d/1asmBVlIjimZ9XAmK5JE42SX4vAvjGqjLflukYBgFSuE/edit?usp=sharing) [original model](https://github.com/a1ex90/MusicalKeyCNN/blob/main/checkpoints/keynet.pt) + [MIK](https://mixedinkey.com/) |
| **Match & Align** | Pair instrumental/vocal folders, organize pairs, align stems to a reference |
| **Rename** | Rule-based sample rename + optional instrument Auto-detect ([OpenMIC](https://research.atspotify.com/publications/openmic-2018-an-open-dataset-for-multiple-instrument-recognition) / [hear21passt](https://github.com/kkoutini/passt_hear21)) |
| **Integrity** | **Compression** — [FLAC Detective](https://pypi.org/project/flac-detective/) lossless/lossy<br>**Corruption** — fast/deep verify + fix ([AudioTester](http://www.vuplayer.com/other.php) / [foobar2k](https://www.foobar2000.org/) alike)<br>**Convert** — batch to FLAC |
| **Charts** | Scan library roots → donuts, genre/style breakdown, [SI-SDR](https://source-separation.github.io/tutorial/basics/evaluation.html#si-sdr) bars<br>Export PNG/PDF + output balanced dataset |

Downstream tabs can auto-fill paths from Classify output (`*_organized`).

## Requirements

- **Windows**
- **Python 3.10 or 3.11** on PATH (for `install-deps.bat` / `build.bat`)
- Disk space for PyTorch + Demucs (`site-packages\` beside the `.exe`, or `.venv` / `genre_gender_tagger\venv`)
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
