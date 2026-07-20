import gc
import json
import logging
import time
import os
import sys
import urllib.request
import warnings

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Silence TensorFlow C++ logs before any TF import (acapella path).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Harmless HF Hub noise on Windows VMs (no symlink support / no token).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")


APP_NAME = "Genre / Gender Tagger"
APP_VERSION = "1.0"

# Avoid Windows cp1252 crashes on non-ASCII filenames in print/tqdm.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _fmt_summary_elapsed(seconds: float) -> str:
    """SI-SDR-style elapsed for unified summaries (m:ss or h:mm:ss)."""
    total = max(0, int(round(float(seconds or 0))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def print_feature_summary(
    feature: str,
    *,
    elapsed: float,
    files: int,
    tagged: int | None = None,
    skipped: int | None = None,
    peak_vram_gb: float | None = None,
    results_path=None,
    extra_lines: list | None = None,
) -> None:
    """Unified LOG footer: === Feature Summary === … DONE."""
    n = max(0, int(files or 0))
    minutes = max(float(elapsed or 0) / 60.0, 1e-9)
    print(flush=True)
    print(f"=== {feature} Summary ===", flush=True)
    print(f"  Total time: {_fmt_summary_elapsed(elapsed)}", flush=True)
    print(f"  Files: {n}", flush=True)
    if n > 0:
        print(f"  Sec/file: {float(elapsed) / n:.3f}", flush=True)
        print(f"  Files/min: {n / minutes:.2f}", flush=True)
    if tagged is not None:
        if skipped is not None:
            print(f"  Tagged: {tagged} | Skipped: {skipped}", flush=True)
        else:
            print(f"  Tagged: {tagged}", flush=True)
    if peak_vram_gb is not None:
        print(f"  Peak VRAM: {float(peak_vram_gb):.2f} GB", flush=True)
    if extra_lines:
        for line in extra_lines:
            text = str(line).strip()
            if text:
                print(f"  {text}", flush=True)
    if results_path:
        print(f"  Results: {results_path}", flush=True)
    print(flush=True)
    print("DONE", flush=True)


_status(f"{APP_NAME} v{APP_VERSION}")
_status("Starting up...")

_status("  loading torch...")
import torch

_status("  loading audio / data helpers...")
import csv
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm as _tqdm_cls
from mutagen.flac import FLAC
from mutagen.id3 import COMM, ID3, TCON, TXXX

# Optional — not required for decode/resample (librosa). May be absent or
# mismatched after other packages (e.g. hear21passt) pull a CUDA wheel.
try:
    import torchaudio  # noqa: F401
except Exception:
    torchaudio = None


def _write_results_csv(path, rows) -> None:
    """Write list[dict] to CSV via stdlib (no pandas)."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    # Stable column order: keys from first row, then any extras.
    fieldnames: list[str] = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


_STEM_PROGRESS_INTERVAL = 0.1
_last_stem_progress_emit = 0.0
_last_gg_processed_emit = 0.0
_last_gg_processed_n = -1


def emit_gg_processed(n, total, *, force: bool = False) -> None:
    """One-line batch progress for STEM LOG: __gg_processed__ n total."""
    global _last_gg_processed_emit, _last_gg_processed_n
    total_i = int(total or 0)
    if total_i <= 0:
        return
    n_i = int(max(0, min(total_i, int(n or 0))))
    now = time.monotonic()
    if (
        not force
        and n_i != total_i
        and n_i == _last_gg_processed_n
    ):
        return
    if (
        not force
        and n_i != total_i
        and (now - _last_gg_processed_emit) < _STEM_PROGRESS_INTERVAL
    ):
        return
    _last_gg_processed_emit = now
    _last_gg_processed_n = n_i
    print(f"__gg_processed__\t{n_i}\t{total_i}", flush=True)


def emit_stem_progress(
    n,
    total,
    phase="",
    *,
    pct=None,
    eta="",
    force=False,
    display_n=None,
    display_total=None,
):
    """Machine progress for STEM host: __progress__ pct eta n total phase.

    ``n`` may be fractional for smooth bars; ``display_n`` / ``display_total``
    control the human count shown in the status bar (defaults from n/total).
    """
    global _last_stem_progress_emit
    total_i = int(total or 0)
    if total_i <= 0:
        return
    n_f = float(n or 0)
    show_total = int(display_total if display_total is not None else total_i)
    if display_n is not None:
        n_i = int(max(0, min(show_total, int(display_n))))
    else:
        n_i = int(max(0, min(show_total, round(n_f))))
    now = time.monotonic()
    if (
        not force
        and n_f < float(total_i)
        and (now - _last_stem_progress_emit) < _STEM_PROGRESS_INTERVAL
    ):
        return
    _last_stem_progress_emit = now
    if pct is None:
        pct = 100.0 * min(n_f, float(total_i)) / float(total_i)
    eta_s = "" if eta is None else str(eta)
    phase_s = (phase or "").replace("\t", " ").strip()
    print(
        f"__progress__\t{float(pct):.2f}\t{eta_s}\t{n_i}\t{show_total}\t{phase_s}",
        flush=True,
    )


def _fmt_confidence_pct(score) -> str:
    """Fraction 0–1 → whole percent for LOG, e.g. 0.723 → '72%'."""
    try:
        return f"{int(round(float(score) * 100.0))}%"
    except (TypeError, ValueError):
        return "0%"


def _log_gender_result(row):
    """Classify-style LOG: === file === + badge + dim pct (e.g. female 72%)."""
    name = Path(row.get("file") or "").name
    gender = str(row.get("gender") or "").strip().lower()
    reverb = str(row.get("reverb") or "").strip().lower()
    gconf = _fmt_confidence_pct(row.get("confidence", 0))
    rconf = _fmt_confidence_pct(row.get("reverb_confidence", 0))
    print(flush=True)
    print(f"=== {name} ===", flush=True)
    if gender in ("female", "male"):
        print(f"  {gender} {gconf}", flush=True)
    elif gender:
        print(f"GENDER: {gender}", flush=True)
        print(f"  {gconf}", flush=True)
    if reverb in ("dry", "wet"):
        print(f"  {reverb} {rconf}", flush=True)
    elif reverb and reverb != "?":
        print(f"REVERB: {reverb}", flush=True)
        print(f"  {rconf}", flush=True)


def _log_genre_result(path, genre, style, conf):
    """LOG block: === file ===, GENRE / STYLE, then dim pct."""
    print(flush=True)
    print(f"=== {Path(path).name} ===", flush=True)
    print("GENRE:", genre, flush=True)
    print("STYLE:", style or "", flush=True)
    print(f"  {_fmt_confidence_pct(conf)}", flush=True)


class _UiTqdm(_tqdm_cls):
    """tqdm that drives STEM-organizer progress bar when stdout is piped."""

    def __init__(self, *args, **kwargs):
        # Must exist before super().__init__ — tqdm refreshes/display during init.
        self._last_ui_emit = 0.0
        self._ui_piped = not sys.stdout.isatty()
        self._stem_phase = kwargs.pop("stem_phase", None)
        self._stem_pct_scale = float(kwargs.pop("stem_pct_scale", 1.0) or 1.0)
        self._stem_pct_offset = float(kwargs.pop("stem_pct_offset", 0.0) or 0.0)
        kwargs.setdefault("file", sys.stdout)
        kwargs.setdefault("ascii", True)
        kwargs.setdefault(
            "mininterval",
            _STEM_PROGRESS_INTERVAL if self._ui_piped else 0.35,
        )
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("ncols", 88)
        super().__init__(*args, **kwargs)

    def display(self, *args, **kwargs):
        if self._ui_piped:
            # Skip tqdm's \\r bar; only emit machine progress for the host.
            try:
                self._emit_stem_progress()
            except Exception:
                pass
            return None
        out = super().display(*args, **kwargs)
        try:
            self._emit_stem_progress()
        except Exception:
            pass
        return out

    def close(self):
        try:
            self._emit_stem_progress(force=True)
        except Exception:
            pass
        if self._ui_piped:
            # Avoid final bar line in the STEM LOG.
            try:
                self.disable = True
            except Exception:
                pass
        return super().close()

    def _emit_stem_progress(self, force: bool = False) -> None:
        total = getattr(self, "total", None) or 0
        if total <= 0:
            return
        n = int(getattr(self, "n", 0) or 0)
        now = time.monotonic()
        last = float(getattr(self, "_last_ui_emit", 0.0) or 0.0)
        if (
            not force
            and n < total
            and (now - last) < _STEM_PROGRESS_INTERVAL
        ):
            return
        self._last_ui_emit = now
        frac = float(n) / float(total)
        pct = 100.0 * (
            self._stem_pct_offset + self._stem_pct_scale * frac
        )
        eta = ""
        try:
            rate = self.format_dict.get("rate")
            if rate:
                eta = f"{(total - n) / float(rate):.1f}"
        except Exception:
            eta = ""
        phase = self._stem_phase
        if not phase:
            try:
                phase = (self.desc or "").strip()
            except Exception:
                phase = ""
        emit_stem_progress(
            n,
            total,
            phase or "",
            pct=pct,
            eta=eta,
            force=True,
        )


tqdm = _UiTqdm

from mutagen.mp3 import MP3
from mutagen.mp4 import MP4, MP4FreeForm
from mutagen.wave import WAVE

_status("Core libraries ready.")


# ==========================================================
# SETTINGS
# ==========================================================

INPUT_FOLDER = ""  # asked at runtime (see prompt below)

# CONTENT_TYPE selects the classifier path at runtime (see prompt).
#   "instrumental" -> Discogs MAEST genre/style -> GENRE(/STYLE) tags
#   "acapella"     -> gender + vocal mel-CNN reverb -> COMMENT/GENDER (+ REVERB)
CONTENT_TYPE = "instrumental"

OUTPUT_CSV = "genre_gender_results.csv"
OUTPUT_CSV_GENDER = "genre_gender_voice_results.csv"

# Metadata toggles.
# WRITE_METADATA=True  -> write tags at all. Set False to only export
#                        the CSV and leave every file untouched.
# OVERWRITE_TAGS=False -> skip files that already have genre/gender tags
#                        (resume-friendly). True forces re-tag everything.
# DRY_RUN              -> kept for parity with older versions, now only
#                        controls the final banner message.
WRITE_METADATA = True
OVERWRITE_TAGS = False

DRY_RUN = False


# TAG_WRITE_MODE controls HOW genre+style are stored in FLAC tags.
# Overridden at runtime by the prompt below, unless RUNTIME_PROMPTS
# is False.
#
#   "combined" -> GENRE field only, as "Genre/Style"
#                 e.g. GENRE = "Rock/Surf"
#   "split"    -> GENRE field = genre, STYLE field = style
#                 e.g. GENRE = "Rock", STYLE = "Surf"
#
# Both modes overwrite whatever is currently in those fields.
TAG_WRITE_MODE = "combined"


# GENDER_TAG_FIELD controls WHERE voice-gender is stored (acapella).
# Overridden at runtime by the prompt below when WRITE_METADATA is True.
#
#   "comment" -> COMMENT field, e.g. COMMENT = "female"
#   "gender"  -> GENDER field,  e.g. GENDER  = "female"
GENDER_TAG_FIELD = "comment"

# REVERB_TAG_MODE controls how vocal dry/wet reverb is stored with gender.
#
#   "combined" -> gender field only, as "gender/reverb"
#                 e.g. COMMENT = "female/wet"
#   "split"    -> gender field = gender, REVERB field = dry|wet
#                 e.g. COMMENT = "female", REVERB = "wet"
REVERB_TAG_MODE = "combined"


# BATCH_MODE toggles the two run styles.
# Overridden at runtime by the prompt below, unless RUNTIME_PROMPTS
# is False.
#
#   True  -> fast batched pipeline. Files are decoded in parallel
#           across AUDIO_WORKERS threads and pushed to the GPU in
#           BATCH_SIZE chunks. Fastest, but no per-file output.
#
#   False -> per-file style. Files are processed one at a time and each
#           result is printed live (GENRE / STYLE / CONF). Slower
#           (no batching), but you see every file's outcome as it
#           finishes. Also the recommended mode on CPU-only machines / VMs.
BATCH_MODE = True


# RUNTIME_PROMPTS=True  -> ask at startup which mode to run in and
#                          how to write tags. The answers override
#                          BATCH_MODE and TAG_WRITE_MODE above.
# RUNTIME_PROMPTS=False -> use BATCH_MODE and TAG_WRITE_MODE as-is,
#                          no questions asked.
RUNTIME_PROMPTS = True


MODEL_NAME = (
    "mtg-upf/"
    "discogs-maest-30s-pw-129e-519l"
)


SAMPLE_RATE = 16000

CLIP_LENGTH = 30

NUMBER_OF_CLIPS = 3


# RTX 5090 tuning
# v0.4 proved 64 is optimal. Kept.
# (GPU only. Ignored in per-file mode / on CPU.)
BATCH_SIZE = 64


# CPU workers
# v0.4 used 8. Workers now also do feature extraction,
# so they are busier. 8 still matches logical-core sweet spot.
# (Batch mode only.)
AUDIO_WORKERS = 8

# Gender batch: never queue the whole library at once. Each Future
# caches its mel-patch result until destroyed — with 80k files that
# OOMs Windows (ACCESS_VIOLATION / taskkill 0xc000012d). Process and
# tag in waves so RAM stays bounded and a crash keeps earlier tags.
GENDER_FILE_CHUNK = 256


# GPU timing debug
# OFF in v0.7: the per-batch synchronize blocks the CPU->GPU
# overlap that this version relies on. Total wall-clock timing
# at the end stays accurate without it.
MEASURE_GPU_TIME = False



AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a"
}

# Overridden by GG_RECURSIVE in non-interactive (STEM organizer) mode.
INCLUDE_SUBFOLDERS = True


def iter_audio_files(folder):
    """Yield audio files under folder (recursive when INCLUDE_SUBFOLDERS)."""
    root = Path(folder)
    it = root.rglob("*") if INCLUDE_SUBFOLDERS else root.iterdir()
    for path in it:
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def list_audio_files(folder):
    """Collect audio paths with periodic scan feedback for large libraries."""
    print("Scanning for audio files...", flush=True)
    files = []
    for i, path in enumerate(iter_audio_files(folder), 1):
        files.append(str(path))
        if i % 5000 == 0:
            print(f"  …found {i:,} audio files", flush=True)
    return files


# ----------------------------------------------------------
# Voice-gender / EffNet Discogs (acapella path)
# Essentia has no Windows wheels; we run official .pb weights
# with a MusiCNN-style mel front-end matching Essentia.
# ----------------------------------------------------------

GENDER_FRAME_SIZE = 512
GENDER_HOP_SIZE = 256
GENDER_N_MELS = 96
GENDER_PATCH_SIZE = 128
GENDER_PATCH_HOP = 62
GENDER_BATCH_SIZE = 64  # discogs-effnet-bs64 fixed batch

# Reverb (vocal mel-CNN): files decoded in parallel, crops stacked on GPU.
REVERB_FILE_CHUNK = 64
REVERB_GPU_BATCH = 32
GENDER_LABELS = ("female", "male")

GENDER_MODEL_DIR = Path(__file__).resolve().parent / "models"

GENDER_EFFNET_URL = (
    "https://essentia.upf.edu/models/feature-extractors/"
    "discogs-effnet/discogs-effnet-bs64-1.pb"
)
GENDER_HEAD_URL = (
    "https://essentia.upf.edu/models/classification-heads/"
    "gender/gender-discogs-effnet-1.pb"
)
GENDER_EFFNET_NAME = "discogs-effnet-bs64-1.pb"
GENDER_HEAD_NAME = "gender-discogs-effnet-1.pb"

# ONNX / DirectML (GPU on Windows — TF 2.11+ has no native Win CUDA).
GENDER_EFFNET_ONNX_NAME = "discogs-effnet-bsdynamic-1.onnx"
GENDER_HEAD_ONNX_NAME = "gender-discogs-effnet-1.onnx"
GENDER_EFFNET_ONNX_URL = (
    "https://essentia.upf.edu/models/feature-extractors/"
    "discogs-effnet/discogs-effnet-bsdynamic-1.onnx"
)
GENDER_HEAD_ONNX_URL = (
    "https://essentia.upf.edu/models/classification-heads/"
    "gender/gender-discogs-effnet-1.onnx"
)
GENDER_ORT_BATCH = 128
_INSTRUMENT_MODELS = (
    Path(__file__).resolve().parent.parent / "instrument_tagger" / "models"
)

_TF_SILENCED = False
_MEL_FILTERBANK = None


# ==========================================================
# DEVICE / DTYPE
# ==========================================================
#
# Works with or without an NVIDIA GPU. On a CPU-only machine
# (e.g. a VM) we fall back to fp32 and skip autocast, since the
# fp16/autocast path below is CUDA-only and would error on CPU.

_status("Detecting compute device...")

IS_GPU = torch.cuda.is_available()

device = (
    "cuda"
    if IS_GPU
    else "cpu"
)

# fp16 on GPU (proven, same as v0.4), fp32 on CPU.
MODEL_DTYPE = (
    torch.float16
    if IS_GPU
    else torch.float32
)


torch.set_grad_enabled(False)


if IS_GPU:

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    gpu_name = torch.cuda.get_device_name(0)
    _status(f"Device: CUDA — {gpu_name}")
else:
    _status("Device: CPU (no NVIDIA GPU detected)")

_status("Startup complete.\n")


# ==========================================================
# NON-INTERACTIVE MODE (env-driven, e.g. from STEM organizer)
# ==========================================================
#
# Set GG_MODE=genre or GG_MODE=gender plus the other GG_* vars
# to run without any interactive prompts.  When GG_MODE is
# empty the classic interactive CLI starts as usual.

_GG_MODE = os.environ.get("GG_MODE", "").strip().lower()

if _GG_MODE:
    _gg_input      = os.environ.get("GG_INPUT",        "").strip()
    _gg_batch      = os.environ.get("GG_BATCH",        "1").strip()
    _gg_tag_style  = os.environ.get("GG_TAG_STYLE",    "combined").strip().lower()
    _gg_gender_fld = os.environ.get("GG_GENDER_FIELD", "comment").strip().lower()
    _gg_reverb     = os.environ.get("GG_REVERB_MODE",  "combined").strip().lower()
    _gg_write_meta = os.environ.get("GG_WRITE_META",   "1").strip()
    _gg_overwrite  = os.environ.get("GG_OVERWRITE",    "0").strip()
    _gg_recursive  = os.environ.get("GG_RECURSIVE",    "1").strip()
    _gg_csv        = os.environ.get("GG_CSV",          "").strip()

    CONTENT_TYPE     = "acapella" if _GG_MODE == "gender" else "instrumental"
    INPUT_FOLDER     = _gg_input
    BATCH_MODE       = (_gg_batch != "0")
    TAG_WRITE_MODE   = _gg_tag_style if _gg_tag_style in ("combined", "split") else "combined"
    GENDER_TAG_FIELD = _gg_gender_fld if _gg_gender_fld in ("comment", "gender") else "comment"
    REVERB_TAG_MODE  = _gg_reverb if _gg_reverb in ("combined", "split") else "combined"
    WRITE_METADATA   = (_gg_write_meta != "0")
    OVERWRITE_TAGS   = (_gg_overwrite == "1")
    INCLUDE_SUBFOLDERS = (_gg_recursive != "0")
    RUNTIME_PROMPTS  = False

    if _gg_csv:
        OUTPUT_CSV        = _gg_csv
        OUTPUT_CSV_GENDER = _gg_csv

    _input_path = Path(INPUT_FOLDER) if INPUT_FOLDER else None
    if not _input_path or not _input_path.is_dir():
        print(
            f"GG_MODE error: INPUT_FOLDER does not exist: {INPUT_FOLDER!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    _has_audio_env = any(True for _ in iter_audio_files(_input_path))
    if not _has_audio_env:
        print(
            f"GG_MODE error: no supported audio files in {INPUT_FOLDER!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[GG] mode={CONTENT_TYPE}  folder={INPUT_FOLDER}"
        f"  batch={BATCH_MODE}  write_meta={WRITE_METADATA}"
        f"  overwrite={OVERWRITE_TAGS}"
        f"  subfolders={INCLUDE_SUBFOLDERS}"
        + (
            f"  reverb={REVERB_TAG_MODE}"
            if CONTENT_TYPE == "acapella"
            else ""
        ),
        flush=True,
    )


# ==========================================================
# ASK CONTENT TYPE (instrumental vs acapella)
# ==========================================================

if not _GG_MODE:

    print("==============================")
    print("Content type")
    print("==============================")
    print()
    print("  1 = Instrumental - genre/style (discogs-maest-30s-pw-129e-519l)")
    print("  2 = Acapella     - voice gender + dry/wet reverb (vocal mel-CNN)")
    print()

    while True:

        content_in = input(
            "Content type [1/2] (default 1): "
        ).strip()

        if content_in == "":

            content_in = "1"

        if content_in in ("1", "2"):

            CONTENT_TYPE = (
                "instrumental"
                if content_in == "1"
                else "acapella"
            )

            break

        print("Enter 1 or 2.")

    print()

    if CONTENT_TYPE == "instrumental":

        print("Selected: INSTRUMENTAL (discogs-maest-30s-pw-129e-519l)")

    else:

        print("Selected: ACAPELLA (gender-discogs-effnet)")

    print()


# ==========================================================
# ASK INPUT FOLDER
# ==========================================================

if not _GG_MODE:

    while True:

        user_input = input(
            "Enter input folder (drag in or paste path): "
        ).strip()


        # Strip surrounding quotes that Windows drag-and-drop adds.

        if (
            len(user_input) >= 2
            and user_input[0] == user_input[-1]
            and user_input[0] in "\"'"
        ):

            user_input = user_input[1:-1]


        candidate = Path(user_input)


        if not user_input:

            print("Empty input, try again.")

            continue


        if not candidate.exists():

            print(
                "Path does not exist:",
                user_input
            )

            continue


        if not candidate.is_dir():

            print(
                "Not a folder:",
                user_input
            )

            continue


        # Quick sanity: any supported audio in this tree?

        has_audio = any(True for _ in iter_audio_files(candidate))

        if not has_audio:

            print(
                "No supported audio files found in tree. Try again."
            )

            continue


        INPUT_FOLDER = str(candidate)

        break


    print()
    print(
        "Input folder:",
        INPUT_FOLDER
    )


# ==========================================================
# RUNTIME PROMPTS (mode + tag style)
# ==========================================================

if RUNTIME_PROMPTS and CONTENT_TYPE == "instrumental":

    # ---- Mode: batch vs per-file ----

    print()
    print("==============================")
    print("Run mode")
    print("==============================")
    print()
    print("  1 = Batch    - much faster, but no per-file overview")
    print("  2 = Per-file - slower, prints GENRE/STYLE/CONF per file")

    while True:

        mode_in = input(
            "Run mode [1/2] (default 1): "
        ).strip()

        if mode_in == "":

            mode_in = "1"

        if mode_in in ("1", "2"):

            BATCH_MODE = (mode_in == "1")

            break

        print("Enter 1 or 2.")

    print()

    if BATCH_MODE:

        print("Selected: BATCH")

    else:

        print("Selected: PER-FILE")


    # ---- Tag write style ----

    print()
    print("==============================")
    print("Tag writing")
    print("==============================")
    print()

    if WRITE_METADATA:

        print("  1 = GENRE field only, as \"Genre/Style\"")
        print("       e.g. GENRE = Rock/Metal")
        print("  2 = GENRE + STYLE fields, separated")
        print("       e.g. GENRE = Rock , STYLE = Metal")

        while True:

            tag_in = input(
                "Tag style [1/2] (default 1): "
            ).strip()

            if tag_in == "":

                tag_in = "1"

            if tag_in in ("1", "2"):

                TAG_WRITE_MODE = (
                    "combined"
                    if tag_in == "1"
                    else "split"
                )

                break

            print("Enter 1 or 2.")

    else:

        print("Tag writing is OFF (WRITE_METADATA=False).")
        print("Predictions will only be exported to CSV.")

    print()

    if WRITE_METADATA:

        if TAG_WRITE_MODE == "combined":

            print("Selected: GENRE field = \"Genre/Style\"")

        else:

            print("Selected: GENRE + STYLE fields, separated")

    print()


elif RUNTIME_PROMPTS and CONTENT_TYPE == "acapella":

    # ---- Mode: batch vs per-file ----

    if RUNTIME_PROMPTS:

        print()
        print("==============================")
        print("Run mode")
        print("==============================")
        print()
        print("  1 = Batch    - much faster, but no per-file overview")
        print("  2 = Per-file - slower, prints GENDER/CONF per file")

        while True:

            mode_in = input(
                "Run mode [1/2] (default 1): "
            ).strip()

            if mode_in == "":

                mode_in = "1"

            if mode_in in ("1", "2"):

                BATCH_MODE = (mode_in == "1")

                break

            print("Enter 1 or 2.")

        print()

        if BATCH_MODE:

            print("Selected: BATCH")

        else:

            print("Selected: PER-FILE")

    print()
    print("==============================")
    print("Tag writing")
    print("==============================")
    print()

    if WRITE_METADATA:

        print("  1 = COMMENT field")
        print("       e.g. COMMENT = female")
        print("  2 = GENDER field")
        print("       e.g. GENDER = female")

        while True:

            gender_tag_in = input(
                "Tag field [1/2] (default 1): "
            ).strip()

            if gender_tag_in == "":

                gender_tag_in = "1"

            if gender_tag_in in ("1", "2"):

                GENDER_TAG_FIELD = (
                    "comment"
                    if gender_tag_in == "1"
                    else "gender"
                )

                break

            print("Enter 1 or 2.")

        print()

        if GENDER_TAG_FIELD == "comment":

            print("Selected: COMMENT field")

        else:

            print("Selected: GENDER field")

        print()
        print("==============================")
        print("Reverb tagging (vocal mel-CNN)")
        print("==============================")
        print()
        print("  1 = Combined with gender")
        print("       e.g. COMMENT = female/wet")
        print("  2 = Separate REVERB field")
        print("       e.g. COMMENT = female, REVERB = wet")

        while True:

            reverb_tag_in = input(
                "Reverb mode [1/2] (default 1): "
            ).strip()

            if reverb_tag_in == "":

                reverb_tag_in = "1"

            if reverb_tag_in in ("1", "2"):

                REVERB_TAG_MODE = (
                    "combined"
                    if reverb_tag_in == "1"
                    else "split"
                )

                break

            print("Enter 1 or 2.")

        print()

        if REVERB_TAG_MODE == "combined":

            print("Selected: combined gender/reverb")

        else:

            print("Selected: split gender + REVERB")

    else:

        print("Tag writing is OFF (WRITE_METADATA=False).")
        print("Predictions will only be exported to CSV.")

    print()


# ==========================================================
# VOICE-GENDER HELPERS (gender-discogs-effnet via TF)
# ==========================================================

def _silence_tensorflow():
    """Hide TF C++ INFO spam and GraphDef deprecation warnings."""

    global _TF_SILENCED
    if _TF_SILENCED:
        return

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", message=r".*tf\.GraphDef.*")
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"tensorflow.*",
    )

    _TF_SILENCED = True


def _download_model_file(path, url, status=print):
    status(f"  downloading {path.name} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        raise SystemExit(
            f"\nERROR: could not download {path.name}\n"
            f"  reason: {exc}\n"
            f"  url:    {url}\n\n"
            f"Offline fix: place the file in:\n  {path.parent}\n"
        ) from exc


def ensure_gender_models(model_dir=None, status=print):
    """Download EffNet + gender .pb files if missing."""

    model_dir = Path(model_dir or GENDER_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    effnet = model_dir / GENDER_EFFNET_NAME
    gender = model_dir / GENDER_HEAD_NAME

    for path, url in ((effnet, GENDER_EFFNET_URL), (gender, GENDER_HEAD_URL)):
        if path.exists() and path.stat().st_size > 1000:
            continue
        _download_model_file(path, url, status=status)

    return effnet, gender


def ensure_gender_onnx_models(model_dir=None, status=print):
    """Download EffNet + gender .onnx files if missing."""

    model_dir = Path(model_dir or GENDER_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    effnet = model_dir / GENDER_EFFNET_ONNX_NAME
    gender = model_dir / GENDER_HEAD_ONNX_NAME

    if not effnet.exists() or effnet.stat().st_size < 1000:
        shared = _INSTRUMENT_MODELS / GENDER_EFFNET_ONNX_NAME
        if shared.exists() and shared.stat().st_size > 1000:
            status(f"  using shared {shared}")
            effnet = shared
        else:
            _download_model_file(effnet, GENDER_EFFNET_ONNX_URL, status=status)

    if not gender.exists() or gender.stat().st_size < 1000:
        _download_model_file(gender, GENDER_HEAD_ONNX_URL, status=status)

    return effnet, gender


def _wrap_frozen_graph(graph_path, input_names, output_names):
    _silence_tensorflow()
    import tensorflow as tf

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(Path(graph_path).read_bytes())

    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapped = tf.compat.v1.wrap_function(_imports_graph_def, [])

    return wrapped.prune(
        [wrapped.graph.as_graph_element(n) for n in input_names],
        [wrapped.graph.as_graph_element(n) for n in output_names],
    )


class _GenderTfBackend:
    name = "tensorflow-cpu"

    def __init__(self, embed_fn, gender_fn):
        self.embed_fn = embed_fn
        self.gender_fn = gender_fn

    def predict_batch(self, chunk):
        """chunk [N,128,96] -> probs [N,2]. Pads to 64 when N < 64 (TF graph)."""
        import tensorflow as tf

        valid = chunk.shape[0]
        if valid < GENDER_BATCH_SIZE:
            pad = np.zeros(
                (
                    GENDER_BATCH_SIZE - valid,
                    GENDER_PATCH_SIZE,
                    GENDER_N_MELS,
                ),
                dtype=np.float32,
            )
            chunk = np.concatenate([chunk, pad], axis=0)
        elif valid > GENDER_BATCH_SIZE:
            # Fixed bs64 graph — split.
            parts = []
            for start in range(0, valid, GENDER_BATCH_SIZE):
                parts.append(
                    self.predict_batch(chunk[start : start + GENDER_BATCH_SIZE])
                )
            return np.concatenate(parts, axis=0)

        x = tf.convert_to_tensor(chunk, dtype=tf.float32)
        embeddings = self.embed_fn(x)[0]
        probs = self.gender_fn(embeddings)[0].numpy()
        return probs[:valid]


class _GenderOrtBackend:
    def __init__(self, effnet_sess, head_sess, provider):
        self.effnet = effnet_sess
        self.head = head_sess
        self.name = f"onnxruntime:{provider}"
        self._mel_in = effnet_sess.get_inputs()[0].name
        out_names = [o.name for o in effnet_sess.get_outputs()]
        self._emb_out = "embeddings" if "embeddings" in out_names else out_names[-1]
        self._head_in = head_sess.get_inputs()[0].name
        self._head_out = head_sess.get_outputs()[0].name

    def predict_batch(self, chunk):
        """chunk [N,128,96] -> probs [N,2] (dynamic batch, no pad required)."""
        emb = self.effnet.run([self._emb_out], {self._mel_in: chunk})[0]
        return self.head.run([self._head_out], {self._head_in: emb})[0]


def _ort_providers():
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    ordered = []
    for name in (
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ):
        if name in available:
            ordered.append(name)
    return ordered or ["CPUExecutionProvider"]


def load_gender_ort_backend(model_dir=None, status=print):
    import onnxruntime as ort

    effnet_path, head_path = ensure_gender_onnx_models(model_dir, status=status)
    providers = _ort_providers()
    status(f"  ONNX Runtime providers: {', '.join(providers)}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    effnet = ort.InferenceSession(
        str(effnet_path), sess_options=so, providers=providers
    )
    head = ort.InferenceSession(
        str(head_path), sess_options=so, providers=providers
    )
    active = effnet.get_providers()[0]
    status(f"  using {active} for discogs-effnet + gender head")
    return _GenderOrtBackend(effnet, head, active)


def load_gender_tf_backend(model_dir=None, status=print):
    """Load EffNet + gender head via TensorFlow frozen graphs (CPU on Windows)."""

    _silence_tensorflow()
    status("  loading TensorFlow ...")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception:
        pass

    effnet_path, gender_path = ensure_gender_models(model_dir, status=status)

    status("  loading discogs-effnet embeddings ...")
    embed_fn = _wrap_frozen_graph(
        effnet_path,
        ["serving_default_melspectrogram:0"],
        ["PartitionedCall:1"],
    )

    status("  loading gender-discogs-effnet head ...")
    gender_fn = _wrap_frozen_graph(
        gender_path,
        ["model/Placeholder:0"],
        ["model/Softmax:0"],
    )

    return _GenderTfBackend(embed_fn, gender_fn)


def load_gender_models(model_dir=None, status=print):
    """
    Load gender backend (ONNX DirectML/CUDA preferred, TF CPU fallback).

    Returns a backend with .predict_batch(chunk) -> probs [N,2] and .name.
    """

    try:
        import onnxruntime  # noqa: F401

        return load_gender_ort_backend(model_dir, status=status)
    except Exception as exc:
        status(f"  ONNX Runtime unavailable ({exc}); falling back to TensorFlow")
        return load_gender_tf_backend(model_dir, status=status)


def load_mono_16k(filename):
    """Decode to mono float32 @ 16 kHz (Essentia MonoLoader target)."""
    import librosa

    data, sr = sf.read(filename, always_2d=True, dtype="float32")
    audio = data.mean(axis=1)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
            res_type="soxr_hq",
        )

    return audio.astype(np.float32, copy=False)


def _hz2mel_slaney(hz):
    """Essentia hz2melSlaney (Auditory Toolbox)."""

    hz = np.asarray(hz, dtype=np.float64)
    min_log_hz = 1000.0
    lin_slope = 3.0 / 200.0
    min_log_mel = min_log_hz * lin_slope
    log_step = np.log(6.4) / 27.0

    mel = np.empty_like(hz)
    linear = hz < min_log_hz
    mel[linear] = hz[linear] * lin_slope
    mel[~linear] = min_log_mel + np.log(hz[~linear] / min_log_hz) / log_step
    return mel


def _mel2hz_slaney(mel):
    """Essentia mel2hzSlaney."""

    mel = np.asarray(mel, dtype=np.float64)
    min_log_hz = 1000.0
    lin_slope = 3.0 / 200.0
    min_log_mel = min_log_hz * lin_slope
    log_step = np.log(6.4) / 27.0

    hz = np.empty_like(mel)
    linear = mel < min_log_mel
    hz[linear] = mel[linear] / lin_slope
    hz[~linear] = min_log_hz * np.exp((mel[~linear] - min_log_mel) * log_step)
    return hz


def _essentia_hann(size):
    """Essentia Windowing hann, symmetric=True, normalized=False."""

    i = np.arange(size, dtype=np.float64)
    return (0.5 - 0.5 * np.cos((2.0 * np.pi * i) / (size - 1.0))).astype(
        np.float32
    )


def _frame_cutter_essentia(signal, frame_size=None, hop_size=None):
    """
    Essentia FrameCutter: startFromZero=False, validFrameThresholdRatio=0.
    Returns [n_frames, frame_size].
    """

    if frame_size is None:
        frame_size = GENDER_FRAME_SIZE
    if hop_size is None:
        hop_size = GENDER_HOP_SIZE

    signal = np.asarray(signal, dtype=np.float32)
    n = int(signal.shape[0])
    start0 = -((frame_size + 1) // 2)

    starts = []
    start = start0
    while True:
        starts.append(start)
        if start + frame_size // 2 >= n:
            break
        start += hop_size

    starts = np.asarray(starts, dtype=np.int64)
    left = max(0, -int(starts[0]))
    right = max(0, int(starts[-1] + frame_size - n))
    padded = np.pad(signal, (left, right), mode="constant")
    starts_p = starts + left
    idx = starts_p[:, None] + np.arange(frame_size, dtype=np.int64)[None, :]
    return padded[idx]


def _build_essentia_mel_filterbank(
    spectrum_size=None,
    n_mels=None,
    sample_rate=None,
):
    """
    Essentia MelBands filterbank:
      warpingFormula=slaneyMel, weighting=linear, normalize=unit_tri.
    Returns [n_mels, spectrum_size].
    """

    if spectrum_size is None:
        spectrum_size = GENDER_FRAME_SIZE // 2 + 1
    if n_mels is None:
        n_mels = GENDER_N_MELS
    if sample_rate is None:
        sample_rate = SAMPLE_RATE

    low_hz = 0.0
    high_hz = sample_rate / 2.0

    low_mel = float(_hz2mel_slaney(np.array([low_hz]))[0])
    high_mel = float(_hz2mel_slaney(np.array([high_hz]))[0])
    mel_step = (high_mel - low_mel) / (n_mels + 1)

    mel_points = low_mel + mel_step * np.arange(n_mels + 2, dtype=np.float64)
    band_hz = _mel2hz_slaney(mel_points)

    frequency_scale = (sample_rate / 2.0) / (spectrum_size - 1)
    filters = np.zeros((n_mels, spectrum_size), dtype=np.float64)

    for i in range(n_mels):
        f_left, f_center, f_right = band_hz[i], band_hz[i + 1], band_hz[i + 2]
        fstep1 = f_center - f_left
        fstep2 = f_right - f_center

        jbegin = int(np.ceil(f_left / frequency_scale))
        jend = int(np.floor(f_right / frequency_scale))
        jend = min(jend, spectrum_size - 1)

        for j in range(jbegin, jend + 1):
            binfreq = j * frequency_scale
            if binfreq < f_center:
                coeff = (binfreq - f_left) / fstep1 if fstep1 else 0.0
            else:
                coeff = (f_right - binfreq) / fstep2 if fstep2 else 0.0
            filters[i, j] = max(coeff, 0.0)

        area = (fstep1 + fstep2) / 2.0
        if area > 0:
            filters[i] /= area

    return filters.astype(np.float32)


def _get_mel_filterbank():
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = _build_essentia_mel_filterbank()
    return _MEL_FILTERBANK


def musicnn_logmel(audio):
    """Essentia TensorflowInputMusiCNN equivalent. Returns [n_frames, 96]."""

    frames = _frame_cutter_essentia(audio)
    window = _essentia_hann(GENDER_FRAME_SIZE)
    mel_fb = _get_mel_filterbank()

    n = GENDER_FRAME_SIZE
    half = n // 2
    windowed = frames * window[np.newaxis, :]
    zp = np.empty_like(windowed)
    zp[:, : n - half] = windowed[:, half:]
    zp[:, n - half :] = windowed[:, :half]

    magnitude = np.abs(np.fft.rfft(zp, n=GENDER_FRAME_SIZE, axis=1)).astype(
        np.float32
    )
    power = magnitude * magnitude
    mel = power @ mel_fb.T

    return np.log10(1.0 + mel * 10000.0).astype(np.float32)


def mel_patches(mel):
    """
    Cut [n_frames, 96] into patches of 128 frames, hop 62.
    lastPatchMode=discard. Returns [n_patches, 128, 96].
    """

    n_frames = mel.shape[0]

    if n_frames < GENDER_PATCH_SIZE:
        pad = np.zeros(
            (GENDER_PATCH_SIZE - n_frames, GENDER_N_MELS),
            dtype=np.float32,
        )
        mel = np.concatenate([mel, pad], axis=0)
        return mel[np.newaxis, ...]

    patches = []
    start = 0
    while start + GENDER_PATCH_SIZE <= n_frames:
        patches.append(mel[start : start + GENDER_PATCH_SIZE])
        start += GENDER_PATCH_HOP

    if not patches:
        pad = np.zeros(
            (GENDER_PATCH_SIZE - n_frames, GENDER_N_MELS),
            dtype=np.float32,
        )
        mel = np.concatenate([mel, pad], axis=0)
        return mel[np.newaxis, ...]

    return np.stack(patches, axis=0).astype(np.float32)


def extract_patches(filename):
    """Decode + Mel + patch cut. Returns [n_patches, 128, 96]."""

    audio = load_mono_16k(filename)
    mel = musicnn_logmel(audio)
    return mel_patches(mel)


def predict_patches(patches, backend):
    """Run EffNet + gender head on patches [n, 128, 96] -> probs [n, 2]."""

    batch_size = (
        GENDER_ORT_BATCH
        if getattr(backend, "name", "").startswith("onnxruntime")
        else GENDER_BATCH_SIZE
    )
    n_patches = patches.shape[0]
    probs_all = []

    for batch_start in range(0, n_patches, batch_size):
        chunk = patches[batch_start : batch_start + batch_size]
        probs_all.append(backend.predict_batch(chunk))

    return np.concatenate(probs_all, axis=0)


def predict_fixed_batch(chunk64, backend, gender_fn=None):
    """
    One forward pass on a stacked patch batch.

    `backend` is a gender backend (.predict_batch). The unused gender_fn
    arg keeps older call sites that passed (embed_fn, gender_fn) working
    when they still unpack two values — prefer passing backend alone.
    """
    if gender_fn is not None and not hasattr(backend, "predict_batch"):
        # Legacy: (embed_fn, gender_fn) as two args.
        backend = _GenderTfBackend(backend, gender_fn)
    return backend.predict_batch(np.asarray(chunk64, dtype=np.float32))


def probs_to_result(probs):
    """Average patch probs -> gender label + confidence."""

    mean_prob = probs.mean(axis=0)
    best = int(mean_prob.argmax())

    return {
        "gender": GENDER_LABELS[best],
        "confidence": float(mean_prob[best]),
        "female": float(mean_prob[0]),
        "male": float(mean_prob[1]),
        "n_patches": int(probs.shape[0]),
    }


def calibrate_multiclass_confidence(top1, top2=None):
    """Map ~519-way top-1 softmax to a binary-like score.

    Renormalize top-1 vs top-2: p1/(p1+p2). Same reading as gender/reverb:
    ~0.5 = barely beats runner-up, higher = clearer win. Raw top-1 stays in top5.
    """
    t1 = float(top1)
    if top2 is None:
        return t1
    t2 = float(top2)
    denom = t1 + t2
    if denom <= 0.0:
        return 0.0
    return t1 / denom


def classify_gender_file(filename, backend, gender_fn=None):
    """Run gender-discogs-effnet on one file."""

    if gender_fn is not None and not hasattr(backend, "predict_batch"):
        backend = _GenderTfBackend(backend, gender_fn)
    patches = extract_patches(filename)
    probs = predict_patches(patches, backend)
    return probs_to_result(probs)


# ==========================================================
# METADATA WRITER (FLAC / MP3 / M4A / WAV)
# ==========================================================
#
# Genre TAG_WRITE_MODE:
#   "combined" -> GENRE only as "Genre/Style"
#   "split"    -> GENRE + STYLE
#
# Storage by format:
#   FLAC -> Vorbis comments (genre / style / comment / gender / reverb)
#   MP3 / WAV -> ID3 (TCON, COMM, TXXX:STYLE, TXXX:GENDER, TXXX:REVERB)
#   M4A -> MP4 atoms (©gen, ©cmt, iTunes freeform STYLE/GENDER/REVERB)

_MP4_STD = {
    "genre": "\xa9gen",
    "comment": "\xa9cmt",
}
_MP4_FREEFORM = {
    "style": "----:com.apple.iTunes:STYLE",
    "gender": "----:com.apple.iTunes:GENDER",
    "reverb": "----:com.apple.iTunes:REVERB",
}


def _id3_set_text(tags: ID3, frame_id: str, value: str | None, *, txxx_desc: str | None = None) -> None:
    if txxx_desc is not None:
        key = f"TXXX:{txxx_desc}"
        tags.delall(key)
        if value:
            tags.add(TXXX(encoding=3, desc=txxx_desc, text=[value]))
        return

    tags.delall(frame_id)
    if not value:
        return
    if frame_id == "TCON":
        tags.add(TCON(encoding=3, text=[value]))
    elif frame_id == "COMM":
        tags.add(COMM(encoding=3, lang="eng", desc="", text=[value]))


def _apply_id3_updates(tags: ID3, updates: dict) -> None:
    if "genre" in updates:
        _id3_set_text(tags, "TCON", updates["genre"])
    if "style" in updates:
        _id3_set_text(tags, "TXXX", updates["style"], txxx_desc="STYLE")
    if "comment" in updates:
        _id3_set_text(tags, "COMM", updates["comment"])
    if "gender" in updates:
        _id3_set_text(tags, "TXXX", updates["gender"], txxx_desc="GENDER")
    if "reverb" in updates:
        _id3_set_text(tags, "TXXX", updates["reverb"], txxx_desc="REVERB")


def _apply_mp4_updates(audio: MP4, updates: dict) -> None:
    for field, value in updates.items():
        if field in _MP4_STD:
            key = _MP4_STD[field]
            if value:
                audio[key] = [value]
            else:
                audio.pop(key, None)
        elif field in _MP4_FREEFORM:
            key = _MP4_FREEFORM[field]
            if value:
                audio[key] = [MP4FreeForm(value.encode("utf-8"))]
            else:
                audio.pop(key, None)


def apply_audio_tags(filename, updates: dict) -> bool:
    """Apply logical tag updates. None clears a field. Returns True on success."""

    ext = Path(filename).suffix.lower()

    try:
        if ext == ".flac":
            audio = FLAC(filename)
            for field, value in updates.items():
                if value:
                    audio[field] = value
                else:
                    audio.pop(field, None)
            audio.save()
            return True

        if ext == ".mp3":
            audio = MP3(filename)
            if audio.tags is None:
                audio.add_tags()
            _apply_id3_updates(audio.tags, updates)
            audio.save()
            return True

        if ext == ".wav":
            audio = WAVE(filename)
            if audio.tags is None:
                audio.add_tags()
            _apply_id3_updates(audio.tags, updates)
            audio.save()
            return True

        if ext in (".m4a", ".mp4"):
            audio = MP4(filename)
            _apply_mp4_updates(audio, updates)
            audio.save()
            return True

        print("SKIP (unsupported for tags):", filename)
        return False

    except Exception as exc:
        print(f"SKIP (tag error): {filename} ({exc})")
        return False


def _first_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
    return str(value).strip()


def read_tag_field(filename, field: str) -> str:
    """Read one logical tag field (genre/style/comment/gender/reverb)."""
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".flac":
            audio = FLAC(filename)
            return _first_str(audio.get(field))

        if ext in (".mp3", ".wav"):
            audio = MP3(filename) if ext == ".mp3" else WAVE(filename)
            tags = audio.tags
            if tags is None:
                return ""
            if field == "genre":
                frame = tags.get("TCON")
                return _first_str(getattr(frame, "text", None))
            if field == "comment":
                frames = tags.getall("COMM")
                return _first_str(frames[0].text if frames else "")
            txxx = {
                "style": "TXXX:STYLE",
                "gender": "TXXX:GENDER",
                "reverb": "TXXX:REVERB",
            }.get(field)
            if txxx:
                frames = tags.getall(txxx)
                return _first_str(frames[0].text if frames else "")
            return ""

        if ext in (".m4a", ".mp4"):
            audio = MP4(filename)
            if field in _MP4_STD:
                return _first_str(audio.get(_MP4_STD[field]))
            if field in _MP4_FREEFORM:
                raw = audio.get(_MP4_FREEFORM[field])
                if not raw:
                    return ""
                item = raw[0]
                if isinstance(item, (bytes, bytearray, MP4FreeForm)):
                    return bytes(item).decode("utf-8", errors="replace").strip()
                return _first_str(item)
            return ""

    except Exception:
        return ""
    return ""


def has_genre_tags(filename) -> bool:
    """True when GENRE already has a non-empty value."""
    return bool(read_tag_field(filename, "genre"))


def has_gender_tags(filename) -> bool:
    """True when target gender field already looks like female/male[+reverb]."""
    field = (
        GENDER_TAG_FIELD
        if GENDER_TAG_FIELD in ("comment", "gender")
        else "comment"
    )
    val = read_tag_field(filename, field).lower()
    if not val:
        return False
    head = val.split("/", 1)[0].strip()
    return head in ("female", "male")


def filter_untagged_files(files, *, kind: str):
    """Drop already-tagged files unless OVERWRITE_TAGS. Returns (kept, skipped)."""
    if OVERWRITE_TAGS:
        print(
            "Overwrite existing tags is on — tagging all files "
            "(not checking for existing tags).",
            flush=True,
        )
        return list(files), 0
    label = "gender" if kind == "gender" else "genre"
    print(
        f"Checking existing {label} tags "
        f"({len(files):,} file(s)) — already tagged files will be skipped "
        "(Overwrite existing tags is off)…",
        flush=True,
    )
    check = has_gender_tags if kind == "gender" else has_genre_tags
    kept = []
    skipped = 0
    iterator = files
    if len(files) >= 200:
        iterator = tqdm(files, desc="Checking tags")
    for path in iterator:
        if check(path):
            skipped += 1
        else:
            kept.append(path)
    return kept, skipped


def write_metadata(filename, genre, style):
    """Write genre/style tags. Returns True if written."""

    if TAG_WRITE_MODE == "split":
        updates = {
            "genre": genre or None,
            "style": style or None,
        }
    else:
        updates = {
            "genre": f"{genre}/{style}" if style else (genre or None),
            "style": None,
        }

    return apply_audio_tags(filename, updates)


def write_gender_metadata(filename, gender_value, reverb_value=None):
    """Write voice-gender (+ optional reverb) per GENDER_TAG_FIELD / REVERB_TAG_MODE."""

    field = (
        GENDER_TAG_FIELD
        if GENDER_TAG_FIELD in ("comment", "gender")
        else "comment"
    )
    reverb_value = (reverb_value or "").strip().lower() or None

    if REVERB_TAG_MODE == "combined" and reverb_value:
        return apply_audio_tags(
            filename,
            {field: f"{gender_value}/{reverb_value}"},
        )

    updates = {field: gender_value}
    if reverb_value:
        updates["reverb"] = reverb_value
    return apply_audio_tags(filename, updates)


# ==========================================================
# ACAPELLA PATH (gender-discogs-effnet + vocal mel-CNN reverb)
# ==========================================================

if CONTENT_TYPE == "acapella":

    from vocal_reverb import load_vocal_reverb

    tag_field_label = (
        "COMMENT"
        if GENDER_TAG_FIELD == "comment"
        else "GENDER"
    )
    reverb_mode_label = (
        f"{tag_field_label}=gender/reverb"
        if REVERB_TAG_MODE == "combined"
        else f"{tag_field_label}=gender + REVERB"
    )

    print()
    print("==============================")
    print(f"{APP_NAME} v{APP_VERSION}")
    print("Acapella / voice-gender + reverb")
    if BATCH_MODE:
        print("Batched pipeline")
    else:
        print("Per-file mode")
    print("gender-discogs-effnet + vocal_reverb.pt (mel-CNN)")
    print(f"Recursive + {reverb_mode_label} metadata")
    print("==============================")
    print()

    if BATCH_MODE:

        print("Audio workers:", AUDIO_WORKERS)
        print()

    print("Loading voice-gender models...")
    gender_backend = load_gender_models(status=_status)
    print(f"Models loaded ({gender_backend.name})")
    print()

    print("Loading vocal reverb classifier...")
    reverb_router = load_vocal_reverb(GENDER_MODEL_DIR, status=_status)
    print("Reverb classifier loaded")
    print()

    files = list_audio_files(INPUT_FOLDER)
    total_found = len(files)
    files, skipped_tagged = filter_untagged_files(files, kind="gender")

    print(
        "Found",
        total_found,
        "files (recursive)" if INCLUDE_SUBFOLDERS else "files (top-level only)",
    )
    if skipped_tagged:
        print(
            f"Skipping {skipped_tagged} already tagged "
            "(Overwrite existing tags is off)"
        )
    print(f"To process: {len(files)}")
    print()

    if not files:

        print(
            "No untagged audio files left. "
            "Enable overwrite to re-tag, or pick another folder."
        )
        sys.exit(0)

    start_time = time.perf_counter()
    results = [None] * len(files)

    def _empty_gender_row(filename, error=""):

        return {
            "file": filename,
            "gender": "",
            "confidence": 0.0,
            "female": 0.0,
            "male": 0.0,
            "reverb": "",
            "reverb_confidence": 0.0,
            "wet": 0.0,
            "dry": 0.0,
            "n_patches": 0,
            "error": error,
        }

    def _merge_reverb(row, filename):

        try:

            rev = reverb_router.predict(filename)

        except Exception as exc:

            row["error"] = (
                (row.get("error") or "") + f" | reverb: {exc}"
            ).strip(" |")
            return row

        row["reverb"] = rev["reverb"]
        row["reverb_confidence"] = round(rev["reverb_confidence"], 4)
        row["wet"] = round(rev["wet"], 4)
        row["dry"] = round(rev["dry"], 4)
        return row

    written = 0
    skipped = 0

    if BATCH_MODE:

        # Parallel mel/patch extract in bounded waves. Never submit the
        # whole library — Future objects cache patch arrays until GC.

        print("Processing gender (batch)...")
        print(
            f"  workers={AUDIO_WORKERS}  "
            f"file_chunk={GENDER_FILE_CHUNK}  "
            f"gpu_batch={GENDER_BATCH_SIZE}"
        )
        print(
            f"  reverb workers={AUDIO_WORKERS}  "
            f"file_chunk={REVERB_FILE_CHUNK}  "
            f"gpu_batch={REVERB_GPU_BATCH}  "
            f"device={getattr(reverb_router, 'device', '?')}"
        )
        print()

        def _extract_worker(args):

            index, filename = args

            try:

                patches = extract_patches(filename)

                return index, patches, None

            except Exception as exc:

                return index, None, str(exc)

        files_total = len(files)
        gender_started = time.perf_counter()

        def _gender_eta(done_n):
            if done_n <= 0:
                return ""
            rate = done_n / max(time.perf_counter() - gender_started, 1e-9)
            if rate <= 0:
                return ""
            return f"{(files_total - done_n) / rate:.1f}"

        with tqdm(total=files_total, desc="Gender") as pbar:

            for wave_start in range(0, files_total, GENDER_FILE_CHUNK):

                wave_end = min(wave_start + GENDER_FILE_CHUNK, files_total)
                wave_files = files[wave_start:wave_end]
                wave_len = wave_end - wave_start

                score_storage = {}
                errors = {}
                pending_patches = []
                pending_map = []

                def _flush_gender_batch(force=False):

                    while pending_patches and (
                        force
                        or len(pending_patches) >= GENDER_BATCH_SIZE
                    ):

                        take = min(
                            GENDER_BATCH_SIZE, len(pending_patches)
                        )
                        chunk = pending_patches[:take]
                        mapping = pending_map[:take]
                        del pending_patches[:take]
                        del pending_map[:take]

                        stacked = np.stack(chunk, axis=0)

                        if take < GENDER_BATCH_SIZE:

                            pad = np.zeros(
                                (
                                    GENDER_BATCH_SIZE - take,
                                    GENDER_PATCH_SIZE,
                                    GENDER_N_MELS,
                                ),
                                dtype=np.float32,
                            )
                            stacked = np.concatenate(
                                [stacked, pad], axis=0
                            )

                        probs = predict_fixed_batch(
                            stacked,
                            gender_backend,
                        )

                        for prob, idx in zip(probs[:take], mapping):

                            score_storage.setdefault(
                                idx, []
                            ).append(prob)

                extract_done = 0

                with ThreadPoolExecutor(
                    max_workers=AUDIO_WORKERS
                ) as executor:

                    futures = [
                        executor.submit(
                            _extract_worker,
                            (wave_start + i, filename),
                        )
                        for i, filename in enumerate(wave_files)
                    ]

                    for future in as_completed(futures):

                        index, patches, err = future.result()
                        extract_done += 1
                        # Soft bar during extract (40% of this wave).
                        soft_n = (
                            wave_start
                            + (extract_done / wave_len) * wave_len * 0.4
                        )
                        emit_stem_progress(
                            soft_n,
                            files_total,
                            "extracting",
                            eta=_gender_eta(wave_start),
                            display_n=wave_start + extract_done,
                        )

                        if err is not None:

                            errors[index] = err
                            continue

                        for patch in patches:

                            pending_patches.append(patch)
                            pending_map.append(index)

                        del patches

                        if len(pending_patches) >= GENDER_BATCH_SIZE:

                            _flush_gender_batch()

                    del futures

                _flush_gender_batch(force=True)

                for index in range(wave_start, wave_end):

                    filename = files[index]

                    if index in errors:

                        print("ERROR:", filename, flush=True)
                        print(" ", errors[index], flush=True)
                        results[index] = _empty_gender_row(
                            filename,
                            errors[index],
                        )
                        continue

                    probs = np.stack(score_storage[index], axis=0)
                    pred = probs_to_result(probs)

                    results[index] = {
                        "file": filename,
                        "gender": pred["gender"],
                        "confidence": round(pred["confidence"], 4),
                        "female": round(pred["female"], 4),
                        "male": round(pred["male"], 4),
                        "reverb": "",
                        "reverb_confidence": 0.0,
                        "wet": 0.0,
                        "dry": 0.0,
                        "n_patches": pred["n_patches"],
                        "error": "",
                    }

                reverb_jobs = [
                    (index, files[index])
                    for index in range(wave_start, wave_end)
                    if results[index] is not None
                    and results[index].get("gender")
                ]

                for rev_start in range(
                    0, len(reverb_jobs), REVERB_FILE_CHUNK
                ):

                    rev_chunk = reverb_jobs[
                        rev_start : rev_start + REVERB_FILE_CHUNK
                    ]
                    idxs = [item[0] for item in rev_chunk]
                    paths = [item[1] for item in rev_chunk]
                    outs = reverb_router.predict_many(
                        paths,
                        gpu_batch_size=REVERB_GPU_BATCH,
                        num_workers=AUDIO_WORKERS,
                    )

                    for index, rev in zip(idxs, outs):

                        row = results[index]

                        if isinstance(rev, BaseException):

                            row["error"] = (
                                (row.get("error") or "")
                                + f" | reverb: {rev}"
                            ).strip(" |")

                        else:

                            row["reverb"] = rev["reverb"]
                            row["reverb_confidence"] = round(
                                rev["reverb_confidence"], 4
                            )
                            row["wet"] = round(rev["wet"], 4)
                            row["dry"] = round(rev["dry"], 4)

                # Finalize per file: LOG + optional write + progress (60% of wave).
                # Use a list so nested helper can mutate without nonlocal
                # (this path runs at module scope, not inside a function).
                finalize_state = [0]

                def _advance_finalize(phase_name):
                    finalize_state[0] += 1
                    finalized_in_wave = finalize_state[0]
                    done_n = wave_start + finalized_in_wave
                    soft_n = (
                        wave_start
                        + wave_len * 0.4
                        + finalized_in_wave * 0.6
                    )
                    emit_stem_progress(
                        soft_n,
                        files_total,
                        phase_name,
                        eta=_gender_eta(done_n),
                        force=True,
                        display_n=done_n,
                    )
                    emit_gg_processed(done_n, files_total)
                    pbar.n = done_n
                    # Piped: host already got emit_stem_progress — avoid
                    # refresh overwriting soft pct with integer n/total.
                    if not getattr(pbar, "_ui_piped", False):
                        pbar.refresh()

                for index in range(wave_start, wave_end):

                    row = results[index]

                    if index in errors:

                        _advance_finalize("tagging")
                        continue

                    if WRITE_METADATA:

                        if not row or not row.get("gender"):

                            skipped += 1

                        else:

                            ok = write_gender_metadata(
                                row["file"],
                                row["gender"],
                                row.get("reverb") or None,
                            )

                            if ok:
                                written += 1
                            else:
                                skipped += 1

                    _advance_finalize(
                        "writing" if WRITE_METADATA else "tagging"
                    )

                del score_storage, errors, pending_patches, pending_map
                gc.collect()

        emit_gg_processed(files_total, files_total, force=True)

        print()

    else:

        # Per-file: live GENDER / REVERB / CONF

        print("Processing (one file at a time)...")
        print()

        for index, filename in enumerate(
            tqdm(files, desc="Gender")
        ):

            try:

                pred = classify_gender_file(
                    filename,
                    gender_backend,
                )

            except Exception as exc:

                print()
                print("ERROR:", filename)
                print(" ", exc)
                results[index] = _empty_gender_row(
                    filename,
                    str(exc),
                )
                continue

            row = {
                "file": filename,
                "gender": pred["gender"],
                "confidence": round(pred["confidence"], 4),
                "female": round(pred["female"], 4),
                "male": round(pred["male"], 4),
                "reverb": "",
                "reverb_confidence": 0.0,
                "wet": 0.0,
                "dry": 0.0,
                "n_patches": pred["n_patches"],
                "error": "",
            }
            row = _merge_reverb(row, filename)
            results[index] = row

            _log_gender_result(row)

            if WRITE_METADATA and row.get("gender"):

                ok = write_gender_metadata(
                    row["file"],
                    row["gender"],
                    row.get("reverb") or None,
                )

                if ok:
                    written += 1
                else:
                    skipped += 1

            elif WRITE_METADATA:

                skipped += 1

        print()

    elapsed = time.perf_counter() - start_time
    out_csv = OUTPUT_CSV_GENDER
    _write_results_csv(out_csv, results)

    peak = None
    if device == "cuda":
        try:
            peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        except Exception:
            peak = None

    extras = []
    if WRITE_METADATA:
        extras.append(f"{reverb_mode_label} tags written: {written}")
    else:
        extras.append("METADATA UNTOUCHED (WRITE_METADATA=False)")

    print_feature_summary(
        "Gender",
        elapsed=elapsed,
        files=len(files),
        tagged=written if WRITE_METADATA else None,
        skipped=skipped if WRITE_METADATA else None,
        peak_vram_gb=peak,
        results_path=out_csv,
        extra_lines=extras,
    )

    sys.exit(0)


# ==========================================================
# INSTRUMENTAL PATH (MAEST Discogs519)
# ==========================================================

_status("Loading transformers (MAEST)...")
# Quiet HF Hub "unauthenticated" / remote-code download chatter.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r".*cache-system uses symlinks.*",
    category=UserWarning,
    module=r"huggingface_hub.*",
)
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor
)
_status("Transformers ready.")
print()


print()
print("==============================")
print(f"{APP_NAME} v{APP_VERSION}")
if BATCH_MODE:
    print("Batched GPU pipeline")
else:
    print("Per-file mode")
print("Instrumental / Discogs genre")
print("Recursive + metadata write")
print("==============================")
print()

print(
    "Torch:",
    torch.__version__
)

print(
    "Torchaudio:",
    getattr(torchaudio, "__version__", None) or "not installed (librosa resample)",
)

print(
    "CUDA:",
    torch.version.cuda
)

print(
    "Device:",
    device
)



if device == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0)
                .total_memory
            /
            1024**3,
            2
        ),
        "GB"
    )


print()

if BATCH_MODE:

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Audio workers:",
        AUDIO_WORKERS
    )

print(
    "Clips/song:",
    NUMBER_OF_CLIPS
)

print(
    "Write metadata:",
    WRITE_METADATA
)

if WRITE_METADATA:

    print(
        "Tag style:",
        (
            "Genre/Style combined"
            if TAG_WRITE_MODE == "combined"
            else "GENRE + STYLE separated"
        )
    )

print()



# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading model...")


feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)



model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=MODEL_DTYPE
)


model.to(device)

model.eval()


print("Model loaded")
print()



# ==========================================================
# AUDIO FUNCTIONS
# ==========================================================
#
# v0.7 change: feature extraction has moved OUT of the main
# thread and INTO the worker. Each worker now returns ready
# input_values tensors, so the GPU never waits on serial
# feature extraction anymore.

def load_audio(filename):

    """
    Decode audio via SoundFile (faster than librosa load),
    downmix to mono, resample to SAMPLE_RATE via librosa
    (avoids brittle torchaudio native wheels on Windows).
    Returns a 1D float32 torch tensor.
    """

    data, sr = sf.read(
        filename,
        always_2d=True,
        dtype="float32"
    )

    # stereo -> mono (numpy), then optional resample
    audio_np = data.mean(axis=1)

    if sr != SAMPLE_RATE:
        audio_np = librosa.resample(
            audio_np,
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
            res_type="soxr_hq",
        )

    return torch.from_numpy(
        np.ascontiguousarray(audio_np, dtype=np.float32)
    )



def create_clips(audio):

    """
    Slice into NUMBER_OF_CLIPS equal windows of CLIP_LENGTH.
    Short tracks are zero-padded to one full clip.
    Every clip returned here is exactly CLIP_LENGTH * SAMPLE_RATE
    samples long, so downstream stacking needs no padding.
    """

    clip_samples = (
        CLIP_LENGTH
        *
        SAMPLE_RATE
    )

    length = audio.shape[0]


    if length <= clip_samples:

        audio = torch.nn.functional.pad(
            audio,
            (
                0,
                clip_samples - length
            )
        )

        return [
            audio.numpy()
        ]


    clips = []

    step = (
        length - clip_samples
    ) / (
        NUMBER_OF_CLIPS - 1
    )


    for i in range(NUMBER_OF_CLIPS):

        start = int(
            i * step
        )

        clips.append(
            audio[
                start:
                start + clip_samples
            ].numpy()
        )


    return clips



def load_and_extract(args):
    """
    Worker function. Runs on a CPU thread.

    Decodes the file, creates clips AND runs the feature
    extractor, so the main thread receives ready tensors.

    This is the key v0.7 change: the heavy CPU work that used
    to block the GPU (feature extraction in run_gpu_batch) now
    happens in parallel across AUDIO_WORKERS threads.
    """

    index, filename = args

    audio = load_audio(
        filename
    )

    clips = create_clips(
        audio
    )


    # All clips are identical length, so padding is a no-op.
    # We do not need the attention_mask (it would be all ones),
    # so it is omitted to save transfer bandwidth.

    inputs = feature_extractor(
        clips,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )


    return (
        index,
        inputs
    )




# ==========================================================
# GPU BATCH
# ==========================================================

def run_gpu_batch(batch):
    """
    batch: list of (index, inputs_dict)
    where inputs_dict has input_values [n_clips, 480000] on CPU.

    All feature extraction is already done by the workers,
    so this function only collates, pins memory, transfers
    asynchronously and runs the model.
    """

    input_values_list = []

    mapping = []


    for index, inputs in batch:

        iv = inputs["input_values"]

        input_values_list.append(
            iv
        )

        n_clips = iv.shape[0]

        mapping.extend(
            [index] * n_clips
        )


    # All clips are equal length -> plain cat, no padding needed.

    all_iv = torch.cat(
        input_values_list,
        dim=0
    )



    # Pinned memory + non_blocking transfer (GPU only).
    # Requires a pinned source to actually run async.
    # fp16 dtype matches v0.4 (proven). On CPU we keep fp32 and
    # a plain transfer.

    if IS_GPU:

        all_iv = (
            all_iv
            .pin_memory()
            .to(
                device,
                dtype=MODEL_DTYPE,
                non_blocking=True
            )
        )

    else:

        all_iv = all_iv.to(
            device,
            dtype=MODEL_DTYPE
        )



    if MEASURE_GPU_TIME and IS_GPU:

        torch.cuda.synchronize()

        gpu_start = time.perf_counter()


    with torch.inference_mode():

        # autocast is CUDA-only; on CPU we run straight fp32.

        if IS_GPU:

            with torch.autocast(
                device_type="cuda",
                dtype=MODEL_DTYPE
            ):

                output = model(
                    input_values=all_iv
                )

        else:

            output = model(
                input_values=all_iv
            )


    if MEASURE_GPU_TIME and device == "cuda":

        torch.cuda.synchronize()

        gpu_time = (
            time.perf_counter()
            -
            gpu_start
        )

        print(
            "GPU batch:",
            round(
                gpu_time,
                1
            ),
            "sec | clips:",
            len(mapping)
        )



    probs = torch.nn.functional.softmax(
        output.logits,
        dim=-1
    )


    del all_iv
    del output

    # No empty_cache: 6 GB / 32 GB leaves plenty of headroom,
    # and empty_cache forces a synchronize that kills throughput.


    return (
        probs.cpu(),
        mapping
    )




# ==========================================================
# FIND FILES
# ==========================================================

files = list_audio_files(INPUT_FOLDER)
total_found = len(files)
files, skipped_tagged = filter_untagged_files(files, kind="genre")

print(
    "Found",
    total_found,
    "files (recursive)" if INCLUDE_SUBFOLDERS else "files (top-level only)",
)
if skipped_tagged:
    print(
        f"Skipping {skipped_tagged} already tagged "
        "(Overwrite existing tags is off)"
    )
print(f"To process: {len(files)}")
print()

if not files:
    print(
        "No untagged audio files left. "
        "Enable overwrite to re-tag, or pick another folder."
    )
    sys.exit(0)


# ==========================================================
# TIMER START
# ==========================================================

start_time = time.perf_counter()



# ==========================================================
# PER-FILE INFERENCE (v0.1 style)
# ==========================================================
#
# Same audio pipeline as the batch path (load_audio + create_clips
# + feature_extractor), but one file at a time. Each file's top
# result is printed live.

def classify_one_file(filename):
    """
    Returns (scores_cpu, n_clips) where scores_cpu is a
    [n_clips, n_labels] tensor on CPU.
    """

    audio = load_audio(
        filename
    )

    clips = create_clips(
        audio
    )

    inputs = feature_extractor(
        clips,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    iv = inputs["input_values"]


    if IS_GPU:

        iv = (
            iv
            .pin_memory()
            .to(
                device,
                dtype=MODEL_DTYPE,
                non_blocking=True
            )
        )

    else:

        iv = iv.to(
            device,
            dtype=MODEL_DTYPE
        )


    with torch.inference_mode():

        if IS_GPU:

            with torch.autocast(
                device_type="cuda",
                dtype=MODEL_DTYPE
            ):

                output = model(
                    input_values=iv
                )

        else:

            output = model(
                input_values=iv
            )


    scores = torch.nn.functional.softmax(
        output.logits,
        dim=-1
    )


    return (
        scores.cpu(),
        scores.shape[0]
    )



# ==========================================================
# PROCESS
# ==========================================================

score_storage = {}

total_clips = 0


# When metadata write follows, audio owns 85% of the bar so Tagging
# never resets percent back to 0.
_GENRE_AUDIO_WEIGHT = 0.85 if WRITE_METADATA else 1.0
_GENRE_TAG_WEIGHT = 1.0 - _GENRE_AUDIO_WEIGHT


def _log_genre_from_scores(index, scores_tensor):
    """Compact LOG line from clip scores for one file."""
    avg = torch.mean(scores_tensor, dim=0)
    k = min(2, int(avg.numel()))
    top = torch.topk(avg, k)
    best_label = model.config.id2label[top.indices[0].item()]
    top1 = top.values[0].item()
    top2 = top.values[1].item() if k > 1 else None
    best_score = round(calibrate_multiclass_confidence(top1, top2), 4)
    parts = best_label.split("---")
    genre = parts[0]
    style = parts[1] if len(parts) > 1 else ""
    _log_genre_result(files[index], genre, style, best_score)


def _store_gpu_scores(scores, mapping):
    """Accumulate clip scores (batch mode: no per-file LOG spam)."""
    global total_clips
    total_clips += len(mapping)
    for score, idx in zip(scores, mapping):
        score_storage.setdefault(idx, []).append(score)
    emit_gg_processed(len(score_storage), len(files))


if BATCH_MODE:

    # ------------------------------------------------
    # BATCH PATH (streaming pipeline)
    # ------------------------------------------------

    batch = []


    print(
        "Processing..."
    )



    with ThreadPoolExecutor(
        max_workers=AUDIO_WORKERS
    ) as executor:


        futures = {

            executor.submit(
                load_and_extract,
                (i, f)
            ):
            i

            for i, f in enumerate(files)

        }



        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Audio",
            stem_phase="audio",
            stem_pct_scale=_GENRE_AUDIO_WEIGHT,
            stem_pct_offset=0.0,
        ):

            index, inputs = future.result()


            batch.append(
                (
                    index,
                    inputs
                )
            )



            if len(batch) >= BATCH_SIZE:

                scores, mapping = run_gpu_batch(
                    batch
                )
                _store_gpu_scores(scores, mapping)
                batch = []




    # remaining batch

    if batch:

        scores, mapping = run_gpu_batch(
            batch
        )
        _store_gpu_scores(scores, mapping)

    emit_gg_processed(len(files), len(files), force=True)

    print()

    print(
        "Total inference clips:",
        total_clips
    )



else:

    # ------------------------------------------------
    # PER-FILE PATH (v0.1 style, live output)
    # ------------------------------------------------

    print(
        "Processing (one file at a time)..."
    )

    print()


    for index, filename in enumerate(
        tqdm(
            files,
            desc="Analyzing",
            stem_phase="analyzing",
            stem_pct_scale=_GENRE_AUDIO_WEIGHT,
            stem_pct_offset=0.0,
        )
    ):

        scores, n_clips = classify_one_file(
            filename
        )

        total_clips += n_clips

        score_storage[index] = [
            scores[i]
            for i in range(n_clips)
        ]


        # Average the clips, then show the top hit, exactly like
        # v0.1's GENRE / STYLE / CONF block.

        avg = torch.mean(
            scores,
            dim=0
        )

        top = torch.topk(
            avg,
            min(2, int(avg.numel())),
        )

        best_label = model.config.id2label[
            top.indices[0].item()
        ]

        top1 = top.values[0].item()
        top2 = (
            top.values[1].item()
            if top.values.numel() > 1
            else None
        )
        best_score = calibrate_multiclass_confidence(
            top1,
            top2,
        )

        parts = best_label.split(
            "---"
        )

        genre = parts[0]

        style = (
            parts[1]
            if len(parts) > 1
            else ""
        )


        _log_genre_result(
            filename,
            genre,
            style,
            round(best_score, 4),
        )



    print()

    print(
        "Total inference clips:",
        total_clips
    )



# ==========================================================
# RESULTS
# ==========================================================

results = []


for index, filename in enumerate(files):

    avg = torch.mean(
        torch.stack(
            score_storage[index]
        ),
        dim=0
    )


    top5 = torch.topk(
        avg,
        5
    )


    predictions = []


    for score, label_index in zip(
        top5.values,
        top5.indices
    ):

        predictions.append(
            {
                "label":
                    model.config.id2label[
                        label_index.item()
                    ],

                "score":
                    round(
                        score.item(),
                        4
                    )
            }
        )


    best = predictions[0]
    runner = predictions[1] if len(predictions) > 1 else None

    parts = best["label"].split(
        "---"
    )

    genre = parts[0]

    style = (
        parts[1]
        if len(parts) > 1
        else ""
    )

    conf = calibrate_multiclass_confidence(
        best["score"],
        runner["score"] if runner else None,
    )

    results.append(
        {
            "file": filename,
            "genre": genre,
            "style": style,
            "confidence": round(conf, 4),
            "top5": json.dumps(predictions)
        }
    )



# ==========================================================
# WRITE METADATA
# ==========================================================

written = 0

skipped = 0


if WRITE_METADATA:

    print()
    print(
        "Writing metadata to",
        len(results),
        "files..."
    )


    for row in tqdm(
        results,
        desc="Tagging",
        stem_phase="tagging",
        stem_pct_scale=_GENRE_TAG_WEIGHT,
        stem_pct_offset=_GENRE_AUDIO_WEIGHT,
    ):

        ok = write_metadata(
            row["file"],
            row["genre"],
            row["style"]
        )

        if ok:

            written += 1

        else:

            skipped += 1


else:

    print()
    print(
        "Metadata writing OFF"
    )




# ==========================================================
# SUMMARY
# ==========================================================

elapsed = time.perf_counter() - start_time

peak = None
if device == "cuda":
    try:
        peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except Exception:
        peak = None

_write_results_csv(OUTPUT_CSV, results)

extras = []
if DRY_RUN:
    extras.append("DRY RUN flag set (banner only)")
if not WRITE_METADATA:
    extras.append("METADATA UNTOUCHED (WRITE_METADATA=False)")

print_feature_summary(
    "Genre",
    elapsed=elapsed,
    files=len(files),
    tagged=written if WRITE_METADATA else None,
    skipped=skipped if WRITE_METADATA else None,
    peak_vram_gb=peak,
    results_path=OUTPUT_CSV,
    extra_lines=extras,
)
