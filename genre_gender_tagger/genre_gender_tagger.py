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


def _status(msg):
    """Immediate console feedback during slow startup imports."""
    print(msg, flush=True)


_status(f"{APP_NAME} v{APP_VERSION}")
_status("Starting up...")

_status("  loading torch / torchaudio...")
import torch
import torchaudio

_status("  loading audio / data helpers...")
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from mutagen.flac import FLAC
from mutagen.id3 import COMM, ID3, TCON, TXXX
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
#   "acapella"     -> Essentia gender-discogs-effnet -> COMMENT or GENDER
CONTENT_TYPE = "instrumental"

OUTPUT_CSV = "genre_gender_results.csv"
OUTPUT_CSV_GENDER = "genre_gender_voice_results.csv"

# Metadata toggles.
# WRITE_METADATA=True  -> write tags at all. Set False to only export
#                        the CSV and leave every file untouched.
# DRY_RUN              -> kept for parity with older versions, now only
#                        controls the final banner message.
WRITE_METADATA = True

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
    _gg_write_meta = os.environ.get("GG_WRITE_META",   "1").strip()
    _gg_csv        = os.environ.get("GG_CSV",          "").strip()

    CONTENT_TYPE     = "acapella" if _GG_MODE == "gender" else "instrumental"
    INPUT_FOLDER     = _gg_input
    BATCH_MODE       = (_gg_batch != "0")
    TAG_WRITE_MODE   = _gg_tag_style if _gg_tag_style in ("combined", "split") else "combined"
    GENDER_TAG_FIELD = _gg_gender_fld if _gg_gender_fld in ("comment", "gender") else "comment"
    WRITE_METADATA   = (_gg_write_meta != "0")
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

    _has_audio_env = any(
        f.suffix.lower() in AUDIO_EXTENSIONS
        for f in _input_path.rglob("*")
        if f.is_file()
    )
    if not _has_audio_env:
        print(
            f"GG_MODE error: no supported audio files in {INPUT_FOLDER!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[GG] mode={CONTENT_TYPE}  folder={INPUT_FOLDER}"
        f"  batch={BATCH_MODE}  write_meta={WRITE_METADATA}",
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
    print("  2 = Acapella     - voice gender (gender-discogs-effnet)")
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

        has_audio = any(
            f.suffix.lower() in AUDIO_EXTENSIONS
            for f in candidate.rglob("*")
            if f.is_file()
        )

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


def ensure_gender_models(model_dir=None, status=print):
    """Download EffNet + gender .pb files if missing."""

    model_dir = Path(model_dir or GENDER_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    effnet = model_dir / GENDER_EFFNET_NAME
    gender = model_dir / GENDER_HEAD_NAME

    for path, url in ((effnet, GENDER_EFFNET_URL), (gender, GENDER_HEAD_URL)):
        if path.exists() and path.stat().st_size > 0:
            continue
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
                f"Offline / VM fix:\n"
                f"  1. On a machine with internet, run the tagger once (or copy\n"
                f"     from an existing install):\n"
                f"       models\\{GENDER_EFFNET_NAME}\n"
                f"       models\\{GENDER_HEAD_NAME}\n"
                f"  2. Place both files in:\n"
                f"       {model_dir}\n"
                f"  3. Re-run. Download is skipped when those files exist.\n"
            ) from exc

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


def load_gender_models(model_dir=None, status=print):
    """Load EffNet embedding + gender head callables."""

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

    return embed_fn, gender_fn


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


def predict_patches(patches, embed_fn, gender_fn):
    """Run EffNet + gender head on patches [n, 128, 96] -> probs [n, 2]."""
    import tensorflow as tf

    n_patches = patches.shape[0]
    probs_all = []

    for batch_start in range(0, n_patches, GENDER_BATCH_SIZE):
        chunk = patches[batch_start : batch_start + GENDER_BATCH_SIZE]
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

        embeddings = embed_fn(tf.constant(chunk))[0].numpy()
        probs = gender_fn(tf.constant(embeddings))[0].numpy()
        probs_all.append(probs[:valid])

    return np.concatenate(probs_all, axis=0)


def predict_fixed_batch(chunk64, embed_fn, gender_fn):
    """One forward pass. chunk64 must be [64, 128, 96]. Returns [64, 2]."""
    import tensorflow as tf

    embeddings = embed_fn(tf.constant(chunk64))[0].numpy()
    return gender_fn(tf.constant(embeddings))[0].numpy()


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


def classify_gender_file(filename, embed_fn, gender_fn):
    """Run gender-discogs-effnet on one file."""

    patches = extract_patches(filename)
    probs = predict_patches(patches, embed_fn, gender_fn)
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
#   FLAC -> Vorbis comments (genre / style / comment / gender)
#   MP3 / WAV -> ID3 (TCON, COMM, TXXX:STYLE, TXXX:GENDER)
#   M4A -> MP4 atoms (©gen, ©cmt, iTunes freeform STYLE/GENDER)

_MP4_STD = {
    "genre": "\xa9gen",
    "comment": "\xa9cmt",
}
_MP4_FREEFORM = {
    "style": "----:com.apple.iTunes:STYLE",
    "gender": "----:com.apple.iTunes:GENDER",
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


def write_gender_metadata(filename, gender_value):
    """Write voice-gender to COMMENT or GENDER (see GENDER_TAG_FIELD)."""

    field = (
        GENDER_TAG_FIELD
        if GENDER_TAG_FIELD in ("comment", "gender")
        else "comment"
    )
    return apply_audio_tags(filename, {field: gender_value})


# ==========================================================
# ACAPELLA PATH (Essentia gender-discogs-effnet via TF)
# ==========================================================

if CONTENT_TYPE == "acapella":

    tag_field_label = (
        "COMMENT"
        if GENDER_TAG_FIELD == "comment"
        else "GENDER"
    )

    print()
    print("==============================")
    print(f"{APP_NAME} v{APP_VERSION}")
    print("Acapella / voice-gender")
    if BATCH_MODE:
        print("Batched pipeline")
    else:
        print("Per-file mode")
    print("gender-discogs-effnet (TF)")
    print(f"Recursive + {tag_field_label} metadata")
    print("==============================")
    print()

    if BATCH_MODE:

        print("Audio workers:", AUDIO_WORKERS)
        print()

    print("Loading voice-gender models...")
    embed_fn, gender_fn = load_gender_models(status=_status)
    print("Models loaded")
    print()

    files = []

    for file in Path(INPUT_FOLDER).rglob("*"):

        if file.suffix.lower() in AUDIO_EXTENSIONS:

            files.append(str(file))

    print("Found", len(files), "files (recursive)")
    print()

    if not files:

        print("No audio files found. Exiting.")
        sys.exit(1)

    start_time = time.perf_counter()
    results = [None] * len(files)

    def _empty_gender_row(filename, error=""):

        return {
            "file": filename,
            "gender": "",
            "confidence": 0.0,
            "female": 0.0,
            "male": 0.0,
            "n_patches": 0,
            "error": error,
        }

    if BATCH_MODE:

        # Parallel mel/patch extract; pack patches across files into
        # EffNet's fixed 64-patch TF batches.

        print("Processing (batch)...")
        print()

        score_storage = {}
        errors = {}

        pending_patches = []
        pending_map = []

        def _flush_gender_batch(force=False):

            while pending_patches and (
                force or len(pending_patches) >= GENDER_BATCH_SIZE
            ):

                take = min(GENDER_BATCH_SIZE, len(pending_patches))
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
                    stacked = np.concatenate([stacked, pad], axis=0)

                probs = predict_fixed_batch(
                    stacked,
                    embed_fn,
                    gender_fn,
                )

                for prob, idx in zip(probs[:take], mapping):

                    score_storage.setdefault(idx, []).append(prob)

        def _extract_worker(args):

            index, filename = args

            try:

                patches = extract_patches(filename)

                return index, patches, None

            except Exception as exc:

                return index, None, str(exc)

        with ThreadPoolExecutor(
            max_workers=AUDIO_WORKERS
        ) as executor:

            futures = {
                executor.submit(
                    _extract_worker,
                    (i, f),
                ):
                i
                for i, f in enumerate(files)
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Audio",
            ):

                index, patches, err = future.result()

                if err is not None:

                    errors[index] = err
                    continue

                for patch in patches:

                    pending_patches.append(patch)
                    pending_map.append(index)

                if len(pending_patches) >= GENDER_BATCH_SIZE:

                    _flush_gender_batch()

        _flush_gender_batch(force=True)

        for index, filename in enumerate(files):

            if index in errors:

                print("ERROR:", filename)
                print(" ", errors[index])
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
                "n_patches": pred["n_patches"],
                "error": "",
            }

        print()

    else:

        # Per-file: live GENDER / CONF like instrumental per-file mode

        print("Processing (one file at a time)...")
        print()

        for index, filename in enumerate(
            tqdm(files, desc="Gender")
        ):

            try:

                pred = classify_gender_file(
                    filename,
                    embed_fn,
                    gender_fn,
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

            results[index] = {
                "file": filename,
                "gender": pred["gender"],
                "confidence": round(pred["confidence"], 4),
                "female": round(pred["female"], 4),
                "male": round(pred["male"], 4),
                "n_patches": pred["n_patches"],
                "error": "",
            }

            print()
            print(Path(filename).name)
            print("GENDER:", pred["gender"])
            print(
                "CONF:",
                round(pred["confidence"], 4),
            )

        print()

    written = 0
    skipped = 0

    if WRITE_METADATA:

        print(
            f"Writing {tag_field_label} metadata to",
            len(results),
            "files...",
        )

        for row in tqdm(results, desc="Tagging"):

            if not row["gender"]:

                skipped += 1
                continue

            ok = write_gender_metadata(row["file"], row["gender"])

            if ok:
                written += 1
            else:
                skipped += 1

        print("Tagged:", written, "| Skipped:", skipped)

    else:

        print("Metadata writing OFF")

    elapsed = time.perf_counter() - start_time
    minutes = max(elapsed / 60, 1e-9)

    print()
    print("==============================")
    print("PERFORMANCE")
    print("==============================")
    print("Time:", round(elapsed, 2), "sec")
    print("Files:", len(files))
    print("Sec/file:", round(elapsed / len(files), 3))
    print("Files/min:", round(len(files) / minutes, 2))

    out_csv = OUTPUT_CSV_GENDER

    pd.DataFrame(results).to_csv(
        out_csv,
        index=False,
        encoding="utf-8",
    )

    print()
    print("==============================")
    print("DONE")
    print("==============================")
    print(out_csv)

    if WRITE_METADATA:

        print(f"{tag_field_label} tags written:", written)

    else:

        print("METADATA UNTOUCHED (WRITE_METADATA=False)")

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
    torchaudio.__version__
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
    Decode audio via SoundFile (faster than librosa),
    downmix to mono, resample to SAMPLE_RATE.
    Returns a 1D float32 torch tensor.
    """

    data, sr = sf.read(
        filename,
        always_2d=True,
        dtype="float32"
    )

    audio = torch.from_numpy(
        data.T
    )


    # stereo -> mono

    if audio.shape[0] > 1:

        audio = torch.mean(
            audio,
            dim=0,
            keepdim=True
        )

    audio = audio.squeeze(0)


    if sr != SAMPLE_RATE:

        audio = torchaudio.functional.resample(
            audio,
            sr,
            SAMPLE_RATE
        )


    return audio



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
# FIND FILES (recursive through all subfolders)
# ==========================================================

files = []


for file in Path(INPUT_FOLDER).rglob("*"):

    if file.suffix.lower() in AUDIO_EXTENSIONS:

        files.append(
            str(file)
        )


print(
    "Found",
    len(files),
    "files (recursive)"
)

print()



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
            desc="Audio"
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

                total_clips += len(mapping)


                for score, idx in zip(
                    scores,
                    mapping
                ):

                    score_storage.setdefault(
                        idx,
                        []
                    ).append(
                        score
                    )


                batch = []




    # remaining batch

    if batch:

        scores, mapping = run_gpu_batch(
            batch
        )

        total_clips += len(mapping)


        for score, idx in zip(
            scores,
            mapping
        ):

            score_storage.setdefault(
                idx,
                []
            ).append(
                score
            )



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
            desc="Analyzing"
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
            1
        )

        best_label = model.config.id2label[
            top.indices[0].item()
        ]

        best_score = top.values[0].item()

        parts = best_label.split(
            "---"
        )

        genre = parts[0]

        style = (
            parts[1]
            if len(parts) > 1
            else ""
        )


        print()

        print(
            Path(filename).name
        )

        print(
            "GENRE:",
            genre
        )

        print(
            "STYLE:",
            style
        )

        print(
            "CONF:",
            round(
                best_score,
                4
            )
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

    parts = best["label"].split(
        "---"
    )

    genre = parts[0]

    style = (
        parts[1]
        if len(parts) > 1
        else ""
    )


    results.append(
        {
            "file": filename,
            "genre": genre,
            "style": style,
            "confidence": best["score"],
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
        desc="Tagging"
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


    print(
        "Tagged:",
        written,
        "| Skipped:",
        skipped
    )


else:

    print()
    print(
        "Metadata writing OFF"
    )




# ==========================================================
# PERFORMANCE
# ==========================================================

elapsed = (
    time.perf_counter()
    -
    start_time
)

minutes = elapsed / 60


print()

print("==============================")
print("PERFORMANCE")
print("==============================")

print(
    "Time:",
    round(
        elapsed,
        2
    ),
    "sec"
)


print(
    "Files:",
    len(files)
)


print(
    "Sec/file:",
    round(
        elapsed / len(files),
        3
    )
)


print(
    "Files/min:",
    round(
        len(files) / minutes,
        2
    )
)


if device == "cuda":

    print(
        "Peak VRAM:",
        round(
            torch.cuda.max_memory_allocated()
            /
            1024**3,
            2
        ),
        "GB"
    )




# ==========================================================
# SAVE
# ==========================================================

pd.DataFrame(
    results
).to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)


print()

print("==============================")
print("DONE")
print("==============================")

print(
    OUTPUT_CSV
)


if DRY_RUN:

    print(
        "DRY RUN flag set (banner only)"
    )


if not WRITE_METADATA:

    print(
        "METADATA UNTOUCHED (WRITE_METADATA=False)"
    )
