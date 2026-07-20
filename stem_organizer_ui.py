from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from ffmpeg_bootstrap import subprocess_kwargs

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk, filedialog, messagebox
from ui_theme import (  # noqa: F401
    ensure_ctk_dark, DARK, ctk_section, ctk_path_row,
    ctk_ui_font, ctk_section_font, ttk_ui_font, HEADER_DESC_FONT,
    ACTION_BTN_GAP, PATH_BTN_HEIGHT, ctk_pin_button_height,
)

# ML stack (torch, demucs, numpy, soundfile) — loaded via deps_bootstrap, not bundled in slim exe.
np = None
sf = None
torch = None
get_model = None
apply_model = None
AudioFile = None
_ML_INITIALIZED = False


AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.aif', '.aiff', '.ogg', '.m4a', '.opus')

MODELS = {
    'htdemucs (good)':                  'htdemucs',
    'htdemucs_ft (best, slowest)':               'htdemucs_ft',
    'htdemucs_6s (worst, fastest)':            'htdemucs_6s',
}

STEM_MODES = {
    '2-way (instrumental/vocals)': {
        'categories': ('instrumental', 'vocals'),
        'mapping':    {'vocals': 'vocals'},
        'fallback':   'instrumental',
    },
    '4-way (bass/drums/other/vocals)': {
        'categories': ('bass', 'drums', 'other', 'vocals'),
        'mapping':    {n: n for n in ('bass', 'drums', 'other', 'vocals')},
        'fallback':   'other',
    },
}

QUALITY_PRESETS = {
    'FLAC 16-bit':      {'ext': '.flac', 'subtype': 'PCM_16'},
    'FLAC 24-bit':      {'ext': '.flac', 'subtype': 'PCM_24'},
    'WAV 16-bit':       {'ext': '.wav',  'subtype': 'PCM_16'},
    'WAV 24-bit':       {'ext': '.wav',  'subtype': 'PCM_24'},
    'WAV 32-bit float': {'ext': '.wav',  'subtype': 'FLOAT'},
}

AMBIG_MODES = {
    'Skip ambiguous stem only': 'skip_stem',
    'Skip the entire song':     'skip_song',
}

STEM_FILE_EXTS = ('.flac', '.wav', '.mp3')
SDR_STEM_EXT_LABEL = '/'.join(STEM_FILE_EXTS)

SDR_DEFAULT_THRESHOLDS = {
    'bass': 25,
    'drums': 20,
    'other': 20,
    'vocals': 30,
    'instrumental': 30,
}

SCAN_MODES = {
    'Each subfolder (one level)': 'subfolders',
    'Each leaf folder (recursive)': 'recursive',
}

NAMING_MODES = {
    'Original folder name':            'preserve',
    'Folder name (simplified)':        'slug',
    'Sequential (song_0000, 0001, …)': 'sequential',
}

MANIFEST_FILENAME = 'index.json'
SETTINGS_FILENAME = 'settings.json'


def _is_frozen() -> bool:
    return getattr(sys, 'frozen', False)


def _resource_dir() -> Path:
    """Read-only bundled assets (icons, models, ffmpeg)."""
    if _is_frozen():
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _app_dir() -> Path:
    """Writable app directory (settings.json lives next to the exe when frozen)."""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = _app_dir()
RESOURCE_DIR = _resource_dir()
SETTINGS_PATH = APP_DIR / SETTINGS_FILENAME
ICON_ICO = RESOURCE_DIR / 'logo.ico'
LOGO_PNG = RESOURCE_DIR / 'logo.png'


def _configure_torch_home() -> None:
    for base in (APP_DIR, RESOURCE_DIR):
        torch_home = base / 'torch_home'
        checkpoints = torch_home / 'hub' / 'checkpoints'
        if checkpoints.is_dir() and any(checkpoints.glob('*.th')):
            os.environ['TORCH_HOME'] = str(torch_home)
            return


def _init_ml() -> None:
    global np, sf, torch, get_model, apply_model, AudioFile, FFMPEG, _ML_INITIALIZED
    if _ML_INITIALIZED:
        return
    from deps_bootstrap import init_external_deps, load_ml_deps

    init_external_deps()
    np, sf, torch, get_model, apply_model, AudioFile = load_ml_deps()
    _configure_torch_home()
    from ffmpeg_bootstrap import ffmpeg_path

    FFMPEG = ffmpeg_path()
    _ML_INITIALIZED = True


def load_demucs_model(model_id: str):
    return get_model(model_id)


def torch_cuda_built() -> bool:
    try:
        return bool(torch.backends.cuda.is_built())
    except Exception:
        return False


def cuda_device_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


_CUDA_USABLE: bool | None = None


def cuda_compute_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def cuda_arch_tag() -> str | None:
    cap = cuda_compute_capability()
    if not cap:
        return None
    major, minor = cap
    return f'sm_{major}{minor}'


def cuda_arch_listed() -> bool | None:
    """True when the GPU arch is in PyTorch's compiled arch list."""
    if not torch.cuda.is_available():
        return False
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        return None
    if not arch_list:
        return None
    tag = cuda_arch_tag()
    return bool(tag and tag in arch_list)


def cuda_usable(*, force: bool = False) -> bool:
    """Return whether CUDA kernels actually run on this GPU."""
    global _CUDA_USABLE
    if not force and _CUDA_USABLE is not None:
        return _CUDA_USABLE
    if not torch.cuda.is_available():
        _CUDA_USABLE = False
        return False
    listed = cuda_arch_listed()
    if listed is False:
        _CUDA_USABLE = False
        return False
    try:
        probe = torch.zeros(1, device='cuda')
        probe.add_(1)
        torch.cuda.synchronize()
        _CUDA_USABLE = True
    except RuntimeError:
        _CUDA_USABLE = False
    return _CUDA_USABLE


def cuda_effective() -> bool:
    return cuda_usable()


def cuda_incompatibility_hint() -> str | None:
    if not torch.cuda.is_available() or cuda_usable():
        return None
    name = cuda_device_name() or 'NVIDIA GPU'
    tag = cuda_arch_tag()
    cap = cuda_compute_capability()
    if cap and cap[0] >= 12:
        tag_text = f' ({tag})' if tag else ''
        return (
            f'{name}{tag_text} needs PyTorch with CUDA 12.8 (cu128). '
            'Re-run install-deps.bat and choose option 3 (RTX 50-series).'
        )
    if tag:
        return (
            f'{name} ({tag}) is not supported by the installed PyTorch build. '
            'Re-run install-deps.bat or disable CUDA to use CPU.'
        )
    return (
        f'{name} is visible to PyTorch but CUDA kernels failed. '
        'Re-run install-deps.bat or disable CUDA to use CPU.'
    )


def resolve_processing_device(use_cuda: bool) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not use_cuda:
        return 'cpu', warnings
    if not torch.cuda.is_available():
        warnings.append('CUDA not available, using CPU.')
        return 'cpu', warnings
    if not cuda_usable():
        hint = cuda_incompatibility_hint()
        warnings.append(hint or 'CUDA not usable on this GPU, using CPU.')
        return 'cpu', warnings
    return 'cuda', warnings


def is_cuda_kernel_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return 'no kernel image is available' in msg or 'cuda error' in msg
STATUS_LINK_URL = 'https://github.com/gilliaangilliaan/STEM-organizer'
DATASET_TYPES_URL = (
    'https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md'
)
DATASET_TYPE1_URL = f'{DATASET_TYPES_URL}#type-1-musdb'
DATASET_TYPE2_URL = f'{DATASET_TYPES_URL}#type-2-stems'
SI_SDR_URL = (
    'https://source-separation.github.io/tutorial/basics/evaluation.html'
    '?highlight=sdr#si-sdr'
)
APP_VERSION = '1.0.6'
SPLASH_SIZE = 512
SPLASH_PAD = 28
SPLASH_CHROMA = '#010101'
SPLASH_HOLD_MS = 1800
SPLASH_STATUS_FONT = ('Segoe UI', 9)
SPLASH_STATUS_BG = '#DEDEDE'
SPLASH_STATUS_COLOR = '#000000'
SPLASH_STATUS_GAP = 14
SPLASH_STATUS_PAD_Y = 48
_SEQ_RE = re.compile(r'^song_(\d+)$')
FFMPEG = None
SF_READ_EXTS = {'.wav', '.flac', '.aif', '.aiff', '.ogg', '.mp3', '.m4a', '.opus'}
_ALLOWED_NAME_CHARS = set('abcdefghijklmnopqrstuvwxyz0123456789')


def slugify(name: str) -> str:
    s = ''.join(c for c in name.lower() if c in _ALLOWED_NAME_CHARS)
    return s or 'folder'


def format_duration(seconds: float) -> str:
    """Format seconds for folder names. Windows-safe (no colons)."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_duration_log(seconds: float) -> str:
    """Format seconds as minutes:seconds:milliseconds for log output."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    mins, secs = divmod(total_s, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f'{hours}:{mins:02d}:{secs:02d}:{ms:03d}'
    return f'{mins:02d}:{secs:02d}:{ms:03d}'


def format_status_clock(seconds: float | None) -> str:
    """Format seconds as H:MM:SS for the status bar (hours grow as needed, no leading zero)."""
    if seconds is None or seconds < 0:
        return '--:--:--'
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f'{hours}:{minutes:02d}:{secs:02d}'


def format_eta(seconds: float | None) -> str:
    """Format remaining time as H:MM:SS for the status bar."""
    return format_status_clock(seconds)


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as M:SS or H:MM:SS."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{secs:02d}'
    return f'{minutes}:{secs:02d}'


class PhaseTimer:
    """Accumulate monotonic durations keyed by phase name."""

    def __init__(self) -> None:
        self._times: dict[str, float] = {}

    def add(self, phase: str, seconds: float) -> None:
        if seconds > 0:
            self._times[phase] = self._times.get(phase, 0.0) + seconds

    def get(self, phase: str) -> float:
        return self._times.get(phase, 0.0)

    def log_summary(
        self,
        log_fn,
        labels: dict[str, str],
        *,
        title: str = 'Phase timing',
        prefix: str = '  ',
    ) -> None:
        if not self._times:
            return
        log_fn(f'{prefix}{title}:')
        for phase, sec in sorted(self._times.items(), key=lambda kv: (-kv[1], kv[0])):
            label = labels.get(phase, phase)
            log_fn(f'{prefix}  {label}: {format_duration_log(sec)}')


ORGANIZE_PHASE_LABELS = {
    'model_load': 'Model load',
    'input_scan': 'Input scan',
    'dedup': 'De-dupe',
    'prescan': 'Pre-scan',
    'classification': 'RMS classification',
    'mixing': 'Mix stems',
    'export': 'Write output',
}

SDR_PHASE_LABELS = {
    'model_load': 'Model load',
    'target_scan': 'Target scan',
    'audio_load': 'Load audio',
    'separation': 'Separation (Demucs)',
    'sdr_compute': 'SDR computation',
}


def folder_name_with_duration(name: str, duration_sec: float | None, append: bool) -> str:
    if append and duration_sec and duration_sec > 0:
        return f"{name} [{format_duration(duration_sec)}]"
    return name


def folder_has_outputs(path: Path, categories: tuple[str, ...], ext: str) -> bool:
    if not path.is_dir():
        return False
    exts = (ext, '.flac', '.wav', '.mp3', '.ogg')
    seen_exts: list[str] = []
    for e in exts:
        if e not in seen_exts:
            seen_exts.append(e)
    for cat in categories:
        for e in seen_exts:
            if (path / f"{cat}{e}").is_file():
                return True
    return False


def display_path(path: str) -> str:
    """Normalize a filesystem path for display (backslashes on Windows)."""
    path = (path or '').strip()
    return os.path.normpath(path) if path else path


def find_existing_output_dir(
    out_dir: Path,
    rel: Path,
    naming_mode: str,
    categories: tuple[str, ...],
    ext: str,
    manifest: dict | None = None,
) -> Path | None:
    """Return an output folder that already contains stem files for this input song."""
    candidates: list[Path] = []

    if naming_mode == 'sequential' and manifest:
        rel_s = str(rel).replace('\\', '/')
        for folder_name, orig in manifest.items():
            if orig == rel_s:
                candidates.append(out_dir / folder_name)
    elif str(rel) == '.':
        candidates.append(out_dir)
    elif naming_mode == 'preserve':
        base_name = rel.name
        parent = out_dir.joinpath(*rel.parts[:-1]) if len(rel.parts) > 1 else out_dir
        candidates.append(out_dir / rel)
        if parent.is_dir():
            for child in sorted(parent.iterdir()):
                if child.is_dir() and (
                    child.name == base_name or child.name.startswith(base_name + ' [')
                ):
                    candidates.append(child)
    else:
        slug_parts = [slugify(pp) for pp in rel.parts if pp not in ('', '.')]
        if slug_parts:
            base = slug_parts[-1]
            parent = out_dir.joinpath(*slug_parts[:-1]) if len(slug_parts) > 1 else out_dir
            candidates.append(out_dir.joinpath(*slug_parts))
            if parent.is_dir():
                for child in sorted(parent.iterdir()):
                    if child.is_dir() and (
                        child.name == base or child.name.startswith(base + ' [')
                    ):
                        candidates.append(child)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        if folder_has_outputs(candidate, categories, ext):
            return candidate
    return None


def collect_song_groups(in_dir: Path, scan_mode: str) -> dict[Path, list[Path]]:
    """Group audio files into songs based on the selected scan mode."""
    groups: dict[Path, list[Path]] = {}

    if scan_mode == 'subfolders':
        for sub in sorted(in_dir.iterdir()):
            if not sub.is_dir():
                continue
            stems = [
                f for f in sub.rglob('*')
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS
            ]
            if stems:
                groups[sub] = sorted(stems)
        return groups

    for f in in_dir.rglob('*'):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            groups.setdefault(f.parent, []).append(f)
    for folder in groups:
        groups[folder] = sorted(groups[folder])
    return groups


def load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_manifest(out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (MANIFEST_FILENAME + '.tmp')
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(out_dir / MANIFEST_FILENAME)


def _valid_label(value: str | None, choices: dict, default: str) -> str:
    if value in choices:
        return value
    return default


def _safe_tk_int(var: tk.Variable, default: int) -> int:
    try:
        return int(var.get())
    except (tk.TclError, ValueError, TypeError):
        return default


def _safe_tk_float(var: tk.Variable, default: float) -> float:
    try:
        return float(var.get())
    except (tk.TclError, ValueError, TypeError):
        return default


def _tk_numeric_var_ready(var: tk.Variable) -> bool:
    try:
        var.get()
        return True
    except tk.TclError:
        return False


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    try:
        tmp = SETTINGS_PATH.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(SETTINGS_PATH)
    except OSError:
        pass


def send_to_recycle_bin(path: Path) -> None:
    """Move a file or folder to the system recycle bin when supported."""
    target = Path(path).resolve()
    if not target.exists():
        return

    if sys.platform == 'win32':
        from ctypes import wintypes

        FO_DELETE = 0x0003
        FOF_ALLOWUNDO = 0x0040
        FOF_NOCONFIRMATION = 0x0010
        FOF_SILENT = 0x0004

        class _SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ('hwnd', wintypes.HWND),
                ('wFunc', ctypes.c_uint),
                ('pFrom', wintypes.LPCWSTR),
                ('pTo', wintypes.LPCWSTR),
                ('fFlags', ctypes.c_ushort),
                ('fAnyOperationsAborted', wintypes.BOOL),
                ('hNameMappings', wintypes.LPVOID),
                ('lpszProgressTitle', wintypes.LPCWSTR),
            ]

        op = _SHFILEOPSTRUCTW()
        op.hwnd = None
        op.wFunc = FO_DELETE
        op.pFrom = str(target) + '\0\0'
        op.pTo = None
        op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
        op.hNameMappings = None
        op.lpszProgressTitle = None
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        if result != 0:
            raise OSError(f'recycle bin delete failed (code {result})')
        if op.fAnyOperationsAborted:
            raise OSError('recycle bin delete was aborted')
        return

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def _is_empty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return not any(path.iterdir())
    except OSError:
        return False


def remove_empty_parent_dirs(start: Path, root: Path) -> list[Path]:
    """Remove empty directories walking up from start; never removes root."""
    removed: list[Path] = []
    root = root.resolve()
    try:
        current = Path(start).resolve().parent
    except OSError:
        return removed

    while current != root:
        try:
            current.relative_to(root)
        except ValueError:
            break
        if not _is_empty_dir(current):
            break
        try:
            send_to_recycle_bin(current)
            removed.append(current)
        except OSError:
            break
        current = current.parent
    return removed


def prune_empty_dirs_under(root: Path) -> list[Path]:
    """Bottom-up sweep: remove every empty directory under root (not root itself)."""
    root = root.resolve()
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        current = Path(dirpath)
        if current == root or not _is_empty_dir(current):
            continue
        try:
            send_to_recycle_bin(current)
            removed.append(current)
        except OSError:
            pass
    return removed


def cleanup_empty_dirs_after_delete(deleted_paths: list[Path], root: Path) -> list[Path]:
    """Walk up from each deleted path and remove newly empty parent folders."""
    removed: list[Path] = []
    seen: set[Path] = set()
    for path in deleted_paths:
        for parent in remove_empty_parent_dirs(path, root):
            key = parent.resolve()
            if key not in seen:
                seen.add(key)
                removed.append(parent)
    return removed


def next_sequence_number(out_dir: Path, manifest: dict) -> int:
    n_max = -1
    if out_dir.exists():
        for d in out_dir.iterdir():
            if d.is_dir():
                m = _SEQ_RE.match(d.name)
                if m:
                    n_max = max(n_max, int(m.group(1)))
    for k in manifest:
        m = _SEQ_RE.match(k)
        if m:
            n_max = max(n_max, int(m.group(1)))
    return n_max + 1


def _normalize_audio(audio: np.ndarray, file_sr: int, sr: int, ch: int) -> np.ndarray:
    if audio.shape[0] == 1:
        audio = np.repeat(audio, ch, axis=0)
    elif audio.shape[0] > ch:
        audio = audio[:ch]
    if file_sr != sr:
        try:
            import resampy
            audio = resampy.resample(audio, file_sr, sr, axis=1)
        except ImportError:
            raise RuntimeError(
                f"Sample rate mismatch ({file_sr} Hz vs expected {sr} Hz) "
                "and resampy is not installed. Install it with: pip install resampy"
            )
    return audio.astype(np.float32)


def _read_soundfile(path: str, sr: int, ch: int) -> np.ndarray | None:
    try:
        data, file_sr = sf.read(path, dtype='float32', always_2d=True)
        return _normalize_audio(data.T, file_sr, sr, ch)
    except Exception:
        return None


def _read_via_ffmpeg(path: str, sr: int, ch: int) -> np.ndarray | None:
    if not FFMPEG:
        return None
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [FFMPEG, '-y', '-loglevel', 'error', '-i', path,
             '-ar', str(sr), '-ac', str(ch), tmp_path],
            check=True, capture_output=True,
            **subprocess_kwargs(),
        )
        return _read_soundfile(tmp_path, sr, ch)
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def format_load_error(exc: Exception) -> str:
    msg = str(exc)
    if 'CalledProcessError' in type(exc).__name__ or 'ffmpeg' in msg.lower():
        return 'decode failed'
    if len(msg) > 100:
        return msg[:97] + '...'
    return msg


def load_audio(path: str, sr: int, ch: int = 2) -> np.ndarray:
    """
    Load audio to (channels, samples) float32.
    Tries soundfile first (including MP3), then a direct ffmpeg decode, then demucs AudioFile.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in SF_READ_EXTS:
        audio = _read_soundfile(str(p), sr, ch)
        if audio is not None:
            return audio

    audio = _read_via_ffmpeg(str(p), sr, ch)
    if audio is not None:
        return audio

    if AudioFile is None:
        raise RuntimeError(
            f'Could not decode {p.name}. soundfile/ffmpeg failed and audio libraries are not initialized.'
        )
    return AudioFile(path).read(streams=0, samplerate=sr, channels=ch).numpy().astype(np.float32)


def write_audio(path: str, audio: np.ndarray, sr: int, subtype: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith('.flac') and FFMPEG:
        bps = 16 if subtype == 'PCM_16' else 24
        sample_fmt = 's16' if subtype == 'PCM_16' else 's32'
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio.T, sr, subtype='FLOAT')
            subprocess.run(
                [FFMPEG, '-y', '-loglevel', 'error', '-i', tmp_path,
                 '-c:a', 'flac', '-compression_level', '12',
                 '-sample_fmt', sample_fmt, '-bits_per_raw_sample', str(bps), path],
                check=True,
                **subprocess_kwargs(),
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        sf.write(path, audio.T, sr, subtype=subtype)


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2) + 1e-12))


def _tensor_rms(t: 'torch.Tensor') -> float:
    return float(torch.sqrt(torch.mean(t * t) + 1e-12).item())


# Dedup compares downsampled mono audio — full-rate fidelity is unnecessary for null-test matching.
_DEDUP_COMPARE_SR = 11025
# GPU dedup only on cards with at least this much VRAM; budget is a fraction of total VRAM.
_DEDUP_MIN_GPU_VRAM_BYTES = 8 * 1024 ** 3
_DEDUP_GPU_STACK_VRAM_FRACTION = 0.40


def _dedup_gpu_stack_limit_bytes() -> int | None:
    """Max GPU staging buffer for dedup, or None when GPU dedup should not be used."""
    if torch is None or not torch.cuda.is_available():
        return None
    vram = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    if vram < _DEDUP_MIN_GPU_VRAM_BYTES:
        return None
    return int(vram * _DEDUP_GPU_STACK_VRAM_FRACTION)


class _UnionFind:
    def __init__(self, n: int):
        self._p = list(range(n))

    def find(self, i: int) -> int:
        while self._p[i] != i:
            self._p[i] = self._p[self._p[i]]
            i = self._p[i]
        return i

    def union(self, i: int, j: int) -> None:
        self._p[self.find(i)] = self.find(j)

    def groups(self, items: list) -> dict[int, list]:
        result: dict[int, list] = {}
        for i, item in enumerate(items):
            result.setdefault(self.find(i), []).append(item)
        return result


def _dedup_pairs_union(
    items: list,
    uf: _UnionFind,
    threshold: float,
    *,
    device: str = 'cpu',
    log_fn=None,
) -> None:
    """Mark duplicate stem pairs in union-find (phase-inversion null test)."""
    n = len(items)
    use_cuda = device == 'cuda' and torch is not None and torch.cuda.is_available()
    gpu_stack = None
    lengths: list[int] = []

    if use_cuda:
        lengths = [a.shape[1] for _, a in items]
        max_len = max(lengths)
        n_ch = max(a.shape[0] for _, a in items)
        stack_bytes = n * n_ch * max_len * np.dtype(np.float32).itemsize
        stack_limit = _dedup_gpu_stack_limit_bytes()
        if stack_limit is None:
            use_cuda = False
            if log_fn:
                vram_gib = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
                log_fn(
                    f"  [dedup] GPU has {vram_gib:.1f} GiB VRAM (< 8 GiB); "
                    f"using CPU compare for {n} stems"
                )
        elif stack_bytes > stack_limit:
            use_cuda = False
            if log_fn:
                log_fn(
                    f"  [dedup] GPU stack needs {stack_bytes / (1024 ** 3):.1f} GiB; "
                    f"budget {stack_limit / (1024 ** 3):.1f} GiB; using CPU compare for {n} stems"
                )
        else:
            try:
                stack = np.zeros((n, n_ch, max_len), dtype=np.float32)
                for idx, (_, audio) in enumerate(items):
                    stack[idx, :audio.shape[0], :audio.shape[1]] = audio
                gpu_stack = torch.from_numpy(stack).to(device=device, dtype=torch.float32)
            except (MemoryError, RuntimeError):
                use_cuda = False
                gpu_stack = None
                if log_fn:
                    log_fn(f"  [dedup] GPU staging failed; using CPU compare for {n} stems")

    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            if use_cuda and gpu_stack is not None:
                length = min(lengths[i], lengths[j])
                ai = gpu_stack[i, :, :length]
                aj = gpu_stack[j, :, :length]
                diff = ai - aj
                denom = max(_tensor_rms(ai), _tensor_rms(aj), 1e-12)
                residual = _tensor_rms(diff) / denom
            else:
                ai, aj = items[i][1], items[j][1]
                length = min(ai.shape[1], aj.shape[1])
                denom = max(_rms(ai[:, :length]), _rms(aj[:, :length]), 1e-12)
                residual = _rms(ai[:, :length] - aj[:, :length]) / denom
            if residual < threshold:
                uf.union(i, j)

    if gpu_stack is not None:
        del gpu_stack
        if use_cuda:
            torch.cuda.empty_cache()


def find_duplicates(
    paths,
    sr: int,
    log_fn=None,
    threshold: float = 0.05,
    *,
    device: str = 'cpu',
):
    if len(paths) < 2:
        return list(paths)

    compare_sr = min(sr, _DEDUP_COMPARE_SR)
    min_samples = compare_sr

    audios = {}
    for p in paths:
        try:
            audios[p] = load_audio(str(p), sr=compare_sr, ch=1)
        except Exception:
            pass

    items = [(p, a) for p, a in audios.items() if a.shape[1] >= min_samples]
    n = len(items)
    if n < 2:
        return list(paths)

    uf = _UnionFind(n)
    _dedup_pairs_union(items, uf, threshold, device=device, log_fn=log_fn)

    keep = []
    for grp in uf.groups(items).values():
        if len(grp) == 1:
            keep.append(grp[0][0])
            continue
        best_path, _ = min(grp, key=lambda pa: float(np.max(np.abs(pa[1]))))
        keep.append(best_path)
        if log_fn:
            others = [g[0].name for g in grp if g[0] != best_path]
            log_fn(f"  [dedup] kept {best_path.name}; removed duplicates: {', '.join(others)}")

    keep.extend(p for p in paths if p not in audios)
    return keep


def prescan_stems(paths):
    """
    Fast pre-classification check for zero-byte or header-empty stems.
    Returns (issues, ok_paths) where issues is [(path, reason), ...].
    """
    issues = []
    ok = []
    for p in paths:
        path = Path(p)
        try:
            if path.stat().st_size == 0:
                issues.append((path, 'empty file (0 bytes)'))
                continue
        except OSError as exc:
            issues.append((path, f'unreadable: {exc}'))
            continue

        if path.suffix.lower() in SF_READ_EXTS and sf is not None:
            try:
                if sf.info(str(path)).frames == 0:
                    issues.append((path, 'empty audio (0 frames)'))
                    continue
            except Exception:
                pass

        ok.append(path)
    return issues, ok


def classify_batch(model, file_paths, device: str, batch_size: int = 4, stop_event=None):
    sr = model.samplerate
    sources = list(model.sources)

    for start in range(0, len(file_paths), batch_size):
        if stop_event and stop_event.is_set():
            if device == 'cuda':
                model.cpu()
                torch.cuda.empty_cache()
            return
        chunk = file_paths[start:start + batch_size]
        audios, lengths, valid = [], [], []
        for fp in chunk:
            try:
                a = load_audio(str(fp), sr=sr)
            except Exception as e:
                yield (fp, None, f'load failed: {format_load_error(e)}')
                continue
            n_samples = int(a.shape[-1])
            if n_samples == 0:
                yield (fp, None, 'empty audio (0 samples)')
                continue
            audios.append(a)
            lengths.append(n_samples)
            valid.append(fp)

        if not audios:
            continue

        max_len = max(lengths)
        batch = np.zeros((len(audios), 2, max_len), dtype=np.float32)
        for i, a in enumerate(audios):
            batch[i, :, :lengths[i]] = a

        try:
            with torch.no_grad():
                out = apply_model(
                    model, torch.from_numpy(batch).to(device),
                    device=device, progress=False, shifts=0,
                    split=True, overlap=0.25,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(valid) == 1:
                yield (valid[0], None, 'cuda OOM')
                continue
            for fp in valid:
                yield from classify_batch(model, [fp], device, batch_size=1)
            continue
        except RuntimeError as e:
            if device == 'cuda' and is_cuda_kernel_error(e):
                torch.cuda.empty_cache()
                for fp in valid:
                    yield (fp, None, 'CUDA kernels incompatible with this GPU (use cu128 for RTX 50-series)')
                continue
            for fp in valid:
                yield (fp, None, str(e))
            continue
        except AssertionError:
            if len(valid) == 1:
                yield (valid[0], None, 'audio too short for model')
                continue
            for fp in valid:
                yield from classify_batch(model, [fp], device, batch_size=1)
            continue

        out_np = out.cpu().numpy()
        for i, fp in enumerate(valid):
            energies = {n: _rms(out_np[i, j, :, :lengths[i]]) for j, n in enumerate(sources)}
            yield (fp, energies, None)

        if device == 'cuda':
            torch.cuda.empty_cache()


def classify_to_category(energies: dict, mode_cfg: dict, threshold: float, min_margin: float):
    total = sum(energies.values()) + 1e-12
    cat_shares = {c: 0.0 for c in mode_cfg['categories']}
    for src, e in energies.items():
        cat = mode_cfg['mapping'].get(src, mode_cfg['fallback'])
        if cat in cat_shares:
            cat_shares[cat] += e / total

    ranked = sorted(cat_shares, key=cat_shares.get, reverse=True)
    top_cat = ranked[0]
    runner_share = cat_shares[ranked[1]] if len(ranked) > 1 else 0.0
    top_share = cat_shares[top_cat]
    margin = top_share - runner_share

    if top_share < threshold and margin < min_margin:
        return "skip", top_cat, top_share, margin, 'both'
    if top_share < threshold:
        return "skip", top_cat, top_share, margin, 'confidence'
    if margin < min_margin:
        return "skip", top_cat, top_share, margin, 'margin'
    return (top_cat, top_cat, top_share, margin, None)


def mix_originals(paths, sr: int) -> np.ndarray:
    tracks = []
    for p in paths:
        try:
            tracks.append(load_audio(str(p), sr=sr))
        except Exception:
            pass
    if not tracks:
        return np.zeros((2, 0), dtype=np.float32)
    cut = min(t.shape[1] for t in tracks)
    mixed = np.zeros((2, cut), dtype=np.float32)
    for t in tracks:
        mixed += t[:, :cut]
    return mixed


def find_category_stem(folder: Path, category: str) -> Path | None:
    for ext in STEM_FILE_EXTS:
        p = folder / f'{category}{ext}'
        if p.is_file():
            return p
    return None


def collect_sdr_song_folders(root: Path, scan_mode: str) -> list[Path]:
    """Collect candidate song folders for SDR (organized stem outputs)."""
    if not root.is_dir():
        return []
    if scan_mode == 'subfolders':
        return sorted(p for p in root.iterdir() if p.is_dir())
    folders: set[Path] = set()
    for f in root.rglob('*'):
        if f.is_file() and f.suffix.lower() in STEM_FILE_EXTS:
            folders.add(f.parent)
    return sorted(folders)


def folder_has_all_stems(folder: Path, categories: tuple[str, ...]) -> bool:
    return all(find_category_stem(folder, c) is not None for c in categories)


SDR_LAYOUT_MUSDB = 'musdb'   # Type 1: one folder per song, stems named vocals.wav etc.
SDR_LAYOUT_STEMS = 'stems'   # Type 2: one folder per stem category, songs as files inside
SDR_LAYOUT_SINGLE_FLAT = 'single_flat'  # loose single-category files (vocals, instrumental, …)
SDR_LAYOUT_MIXED_FLAT = 'mixed_flat'  # loose vocals + instrumental files classified by name

SDR_VOCALS_ONLY_CATEGORIES = ('vocals',)
SDR_INSTRUMENTAL_ONLY_CATEGORIES = ('instrumental',)
SDR_MIXED_FLAT_CATEGORIES = ('instrumental', 'vocals')

SDR_SINGLE_FLAT_LAYOUTS = frozenset({
    SDR_LAYOUT_SINGLE_FLAT,
    SDR_LAYOUT_MIXED_FLAT,
})

SDR_STEM_PICK_ORDER = ('instrumental', 'vocals', 'bass', 'drums', 'other')

# Minimum keyword hits before offering to process every audio file in the folder.
SDR_SINGLE_STEM_ASK_MIN_MATCHES = 25

# Substrings that mark a file as a non-vocal stem (skip for vocals-only scan).
_SDR_NON_VOCAL_MARKERS = (
    '_instrumental', '-instrumental', '_drums', '-drums',
    '_bass', '-bass', '_other', '-other',
    '_no_vocals', '_novocal', '_no-vocals',
)

# Substrings that mark a file as a non-instrumental stem (skip for instrumental-only scan).
_SDR_NON_INSTRUMENTAL_MARKERS = (
    '_vocals', '-vocals', '_drums', '-drums',
    '_bass', '-bass', '_other', '-other',
    'acapella', 'acappella', 'a capella',
)


def is_sdr_audio_file(path: Path) -> bool:
    return path.suffix.lower() in STEM_FILE_EXTS


def iter_sdr_audio_files(root: Path, scan_mode: str):
    """Yield audio files under root for the current scan mode."""
    if not root.is_dir():
        return
    if scan_mode == 'subfolders':
        for f in root.iterdir():
            if f.is_file() and is_sdr_audio_file(f):
                yield f
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sub.iterdir():
                if f.is_file() and is_sdr_audio_file(f):
                    yield f
    else:
        for f in root.rglob('*'):
            if f.is_file() and is_sdr_audio_file(f):
                yield f


def is_vocals_only_stem_file(path: Path) -> bool:
    """True when a loose audio file looks like an isolated vocal stem."""
    if path.suffix.lower() not in STEM_FILE_EXTS:
        return False
    stem = path.stem.lower()
    if any(m in stem for m in _SDR_NON_VOCAL_MARKERS):
        return False
    if 'instrumental' in stem or '(instrumental)' in stem or 'inst.' in stem or '-inst' in stem:
        return False
    if stem == 'vocals' or stem.endswith('_vocals') or stem.endswith('-vocals'):
        return True
    if 'acapella' in stem or 'a capella' in stem or 'acappella' in stem:
        return True
    if 'vocal' in stem:
        return True
    return False


def is_instrumental_only_stem_file(path: Path) -> bool:
    """True when a loose audio file looks like an isolated instrumental stem."""
    if path.suffix.lower() not in STEM_FILE_EXTS:
        return False
    stem = path.stem.lower()
    if any(m in stem for m in _SDR_NON_INSTRUMENTAL_MARKERS):
        return False
    if 'vocal' in stem and 'instrumental' not in stem and '(instrumental)' not in stem:
        return False
    if stem == 'instrumental' or stem.endswith('_instrumental') or stem.endswith('-instrumental'):
        return True
    if '(instrumental)' in stem:
        return True
    if 'inst.' in stem or '-inst' in stem:
        return True
    if 'instrumental' in stem:
        return True
    return False


def _collect_single_stem_targets(
    root: Path, scan_mode: str, category: str, predicate,
) -> list[dict[str, Path]]:
    """Collect per-file targets for one stem category from flat or nested folders."""
    files = sorted(f for f in iter_sdr_audio_files(root, scan_mode) if predicate(f))
    return [{category: f} for f in files]


def collect_all_audio_targets(
    root: Path, scan_mode: str, category: str,
) -> list[dict[str, Path]]:
    """Treat every audio file in scope as the given single stem category."""
    files = sorted(iter_sdr_audio_files(root, scan_mode))
    return [{category: f} for f in files]


def sdr_single_flat_keyword_predicate(category: str):
    if category == 'vocals':
        return is_vocals_only_stem_file
    if category == 'instrumental':
        return is_instrumental_only_stem_file
    return None


def single_flat_category_patterns(category: str) -> str:
    return {
        'vocals': '*_vocals, acapella, vocal, …',
        'instrumental': '*_instrumental, instrumental, inst., -inst, (instrumental), …',
        'bass': '*_bass or bass in name',
        'drums': '*_drums or drums in name',
        'other': '*_other or other in name',
    }.get(category, category)


def build_single_stem_folder_hint(
    root: Path, scan_mode: str, category: str,
) -> dict[str, int | str | bool] | None:
    """Return stats when a flat folder has many keyword hits but not all files match."""
    predicate = sdr_single_flat_keyword_predicate(category)
    if predicate is None:
        return None
    patterns = single_flat_category_patterns(category)
    all_files = list(iter_sdr_audio_files(root, scan_mode))
    total = len(all_files)
    if total == 0:
        return None
    keyword_matches = sum(1 for f in all_files if predicate(f))
    unmatched = total - keyword_matches
    return {
        'kind': category,
        'patterns': patterns,
        'keyword_matches': keyword_matches,
        'total_audio': total,
        'should_ask_process_all': (
            keyword_matches >= SDR_SINGLE_STEM_ASK_MIN_MATCHES and unmatched > 0
        ),
    }


def single_stem_process_all_message(hint: dict[str, int | str | bool]) -> str:
    kind = str(hint['kind'])
    kind_title = kind.capitalize()
    return (
        f'Found {int(hint["keyword_matches"]):,} files with {kind} keywords\n'
        f'({hint["patterns"]})\n'
        f'out of {int(hint["total_audio"]):,} audio files in this folder.\n\n'
        f'Process all {int(hint["total_audio"]):,} files as {kind} for SI-SDR?'
    )


def collect_vocals_only_targets(root: Path, scan_mode: str) -> list[dict[str, Path]]:
    return _collect_single_stem_targets(
        root, scan_mode, 'vocals', is_vocals_only_stem_file,
    )


def collect_instrumental_only_targets(root: Path, scan_mode: str) -> list[dict[str, Path]]:
    return _collect_single_stem_targets(
        root, scan_mode, 'instrumental', is_instrumental_only_stem_file,
    )


def collect_mixed_flat_targets(
    root: Path, scan_mode: str,
) -> list[dict[str, Path]]:
    """Classify loose vocals/instrumental files independently by filename."""
    targets: list[dict[str, Path]] = []
    for path in sorted(iter_sdr_audio_files(root, scan_mode)):
        is_vocals = is_vocals_only_stem_file(path)
        is_instrumental = is_instrumental_only_stem_file(path)
        if is_vocals and not is_instrumental:
            targets.append({'vocals': path})
        elif is_instrumental and not is_vocals:
            targets.append({'instrumental': path})
    return targets


def find_category_folder(root: Path, category: str) -> Path | None:
    direct = root / category
    if direct.is_dir():
        return direct
    cat_lower = category.lower()
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == cat_lower:
            return p
    return None


def stem_file_song_key(path: Path, category: str) -> str:
    """Song identifier for a file inside a Type-2 stem category folder."""
    stem = path.stem
    cat = category.lower()
    lower = stem.lower()
    for prefix in (f'{cat}_', f'{cat}-', f'{cat} '):
        if lower.startswith(prefix):
            return stem[len(prefix):]
    for suffix in (f'_{cat}', f'-{cat}', f' {cat}'):
        if lower.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def collect_sdr_stem_songs(root: Path, categories: tuple[str, ...]) -> list[dict[str, Path]]:
    """Collect complete song stem sets from Type-2 (stem-per-folder) layout."""
    if not root.is_dir():
        return []
    cat_dirs: dict[str, Path] = {}
    for cat in categories:
        folder = find_category_folder(root, cat)
        if folder is not None:
            cat_dirs[cat] = folder
    if len(cat_dirs) != len(categories):
        return []

    songs: dict[str, dict[str, Path]] = {}
    for cat, folder in cat_dirs.items():
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in STEM_FILE_EXTS:
                key = stem_file_song_key(f, cat)
                songs.setdefault(key, {})[cat] = f

    return [paths for paths in songs.values() if all(c in paths for c in categories)]


def detect_sdr_layout(
    root: Path, categories: tuple[str, ...], scan_mode: str,
) -> str | None:
    """Auto-detect SDR input layout (Type 1 MUSDB vs Type 2 stem folders)."""
    type1 = [f for f in collect_sdr_song_folders(root, scan_mode)
             if folder_has_all_stems(f, categories)]
    type2 = collect_sdr_stem_songs(root, categories)
    if type1 and not type2:
        return SDR_LAYOUT_MUSDB
    if type2 and not type1:
        return SDR_LAYOUT_STEMS
    if type1 and type2:
        return SDR_LAYOUT_MUSDB if len(type1) >= len(type2) else SDR_LAYOUT_STEMS
    return None


_SDR_CATEGORY_ALTERNATES: dict[tuple[str, ...], tuple[str, ...]] = {
    STEM_MODES['2-way (instrumental/vocals)']['categories']: (
        STEM_MODES['4-way (bass/drums/other/vocals)']['categories']
    ),
    STEM_MODES['4-way (bass/drums/other/vocals)']['categories']: (
        STEM_MODES['2-way (instrumental/vocals)']['categories']
    ),
}

_ALL_SDR_CATEGORIES = ('bass', 'drums', 'other', 'vocals', 'instrumental')


def resolve_sdr_layout_and_categories(
    root: Path, scan_mode: str, preferred_categories: tuple[str, ...],
) -> tuple[tuple[str, ...] | None, str | None]:
    """Pick stem categories + layout that match folder contents.

    Tries the UI stem mode first, then the alternate layout (2-stem vs 4-stem),
    then vocals-only or instrumental-only files, or single-stem song folders.
    """
    layout = detect_sdr_layout(root, preferred_categories, scan_mode)
    if layout is not None:
        return preferred_categories, layout
    alt = _SDR_CATEGORY_ALTERNATES.get(preferred_categories)
    if alt is not None:
        layout = detect_sdr_layout(root, alt, scan_mode)
        if layout is not None:
            return alt, layout
    vocals = collect_vocals_only_targets(root, scan_mode)
    instrumental = collect_instrumental_only_targets(root, scan_mode)
    if vocals and not instrumental:
        return SDR_VOCALS_ONLY_CATEGORIES, SDR_LAYOUT_SINGLE_FLAT
    if instrumental and not vocals:
        return SDR_INSTRUMENTAL_ONLY_CATEGORIES, SDR_LAYOUT_SINGLE_FLAT
    if vocals and instrumental:
        return SDR_MIXED_FLAT_CATEGORIES, SDR_LAYOUT_MIXED_FLAT
    type1_vocals = [
        f for f in collect_sdr_song_folders(root, scan_mode)
        if folder_has_all_stems(f, SDR_VOCALS_ONLY_CATEGORIES)
    ]
    if type1_vocals:
        return SDR_VOCALS_ONLY_CATEGORIES, SDR_LAYOUT_MUSDB
    type1_inst = [
        f for f in collect_sdr_song_folders(root, scan_mode)
        if folder_has_all_stems(f, SDR_INSTRUMENTAL_ONLY_CATEGORIES)
    ]
    if type1_inst:
        return SDR_INSTRUMENTAL_ONLY_CATEGORIES, SDR_LAYOUT_MUSDB
    return None, None


def sdr_thresholds_for_categories(
    categories: tuple[str, ...], known: dict[str, float],
) -> dict[str, float]:
    return {
        cat: float(known.get(cat, SDR_DEFAULT_THRESHOLDS.get(cat, 30)))
        for cat in categories
    }


def describe_sdr_scan_failure(
    root: Path, scan_mode: str, preferred_categories: tuple[str, ...],
) -> str:
    folders = collect_sdr_song_folders(root, scan_mode)
    lines = ['No complete stem sets found.\n']
    if not folders:
        lines.append(f'No stem files ({SDR_STEM_EXT_LABEL}) found under:\n{root}')
        return '\n'.join(lines)

    lines.append(f'Scanned {len(folders)} folder(s) under:\n{root}\n')
    for label, cats in (
        ('2-stem (instrumental + vocals)', STEM_MODES['2-way (instrumental/vocals)']['categories']),
        ('4-stem (bass/drums/other/vocals)', STEM_MODES['4-way (bass/drums/other/vocals)']['categories']),
    ):
        n = sum(1 for f in folders if folder_has_all_stems(f, cats))
        if n:
            lines.append(f'  • {n} folder(s) with complete {label} sets')

    n_vocals = len(collect_vocals_only_targets(root, scan_mode))
    n_inst = len(collect_instrumental_only_targets(root, scan_mode))
    n_audio = sum(1 for _ in iter_sdr_audio_files(root, scan_mode))
    if n_audio:
        lines.append(f'  • {n_audio:,} audio file(s) total ({SDR_STEM_EXT_LABEL})')
    if n_vocals:
        lines.append(
            f'  • {n_vocals:,} vocals keyword match(es) '
            f'(*_vocals, acapella, or vocal in name)'
        )
    if n_inst:
        lines.append(
            f'  • {n_inst:,} instrumental keyword match(es) '
            f'(*_instrumental, instrumental, inst., -inst, or (instrumental) in name)'
        )

    sample = folders[0]
    found = [c for c in _ALL_SDR_CATEGORIES if find_category_stem(sample, c)]
    if found:
        lines.append(f'\nExample ({sample.name}): {", ".join(found)}')
    missing = [c for c in preferred_categories if c not in found]
    if missing and found:
        lines.append(
            f'\nStem mode expects {", ".join(preferred_categories)} — '
            f'missing in example: {", ".join(missing)}'
        )

    expected = ', '.join(f'{c}{SDR_STEM_EXT_LABEL}' for c in preferred_categories)
    lines.append(
        f'\nType 1 (MUSDB): each song folder contains:\n{expected}\n\n'
        f'Type 2 (Stems): each stem has its own folder:\n'
        f'{", ".join(preferred_categories)}/'
    )
    return '\n'.join(lines)


def sdr_process_order(categories: tuple[str, ...]) -> tuple[str, ...]:
    """Processing order for SI-SDR (2-stem: instrumental before vocals)."""
    if set(categories) == {'vocals', 'instrumental'}:
        return ('instrumental', 'vocals')
    return categories


def collect_sdr_targets(
    root: Path, categories: tuple[str, ...], scan_mode: str, layout: str, *,
    process_all: bool = False,
) -> list[Path | dict[str, Path]]:
    """Return song targets for the detected layout (folder paths or stem-path dicts)."""
    if layout == SDR_LAYOUT_MIXED_FLAT:
        return collect_mixed_flat_targets(root, scan_mode)
    if layout == SDR_LAYOUT_SINGLE_FLAT:
        cat = categories[0]
        if process_all:
            return collect_all_audio_targets(root, scan_mode, cat)
        predicate = sdr_single_flat_keyword_predicate(cat)
        if predicate is not None:
            return _collect_single_stem_targets(root, scan_mode, cat, predicate)
        return collect_all_audio_targets(root, scan_mode, cat)
    if layout == SDR_LAYOUT_STEMS:
        return collect_sdr_stem_songs(root, categories)
    return [f for f in collect_sdr_song_folders(root, scan_mode)
            if folder_has_all_stems(f, categories)]


def _audio_to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse (channels, samples) to mono float64 for SI-SDR."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if a.shape[0] == 1:
            return a[0]
        return a.mean(axis=0)
    raise ValueError(f'unsupported audio shape: {a.shape}')


def compute_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant SDR in dB (mono mixdown, DC removed)."""
    delta = 1e-7
    ref = _audio_to_mono(reference)
    est = _audio_to_mono(estimate)
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = np.dot(ref, ref)
    if ref_energy < delta:
        return float(-np.inf)
    alpha = np.dot(est, ref) / ref_energy
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + delta) / (np.sum(noise ** 2) + delta)))


def model_estimate_for_category(
    out_np: np.ndarray, sources: list[str], category: str,
    mixture: np.ndarray | None = None,
) -> np.ndarray:
    """Extract model output for a category (mix − vocals for instrumental, like UVR/--two-stems)."""
    if category in sources:
        return out_np[sources.index(category)]
    if category == 'instrumental':
        if mixture is not None and 'vocals' in sources:
            vocals = out_np[sources.index('vocals')]
            n = min(mixture.shape[1], vocals.shape[1])
            return mixture[:, :n] - vocals[:, :n]
        parts = [out_np[sources.index(s)] for s in ('drums', 'bass', 'other') if s in sources]
        if not parts:
            raise ValueError('instrumental: no drum/bass/other sources in model')
        est = np.zeros_like(parts[0])
        for p in parts:
            est += p
        return est
    raise ValueError(f'unknown category: {category}')


def separate_mixture(model, mixture: np.ndarray, device: str) -> np.ndarray:
    """Run Demucs on a single mixture; returns (sources, channels, samples).

    Applies the same mean/std normalization as demucs CLI and UVR before/after inference.
    """
    mix = mixture.astype(np.float32)
    ref = mix.mean(axis=0)
    mean = float(ref.mean())
    std = float(ref.std()) + 1e-8
    normalized = (mix - mean) / std
    with torch.no_grad():
        batch = torch.from_numpy(normalized[np.newaxis]).to(device)
        out = apply_model(
            model, batch, device=device, progress=False, split=True, overlap=0.25,
        )
    return out.cpu().numpy()[0] * std + mean


DONE_SENTINEL = object()
PROGRESS_TAG = '__progress__'
PAIR_LOG_TAG = '__pair_log__'
SDR_LOG_TAG = '__sdr_line__'
GG_PROCESSED_TAG = '__gg_processed__'


def _play_done_sound() -> None:
    from done_sound import play_done_sound

    play_done_sound()


class Worker(threading.Thread):
    def __init__(self, params: dict, log_q: queue.Queue):
        super().__init__(daemon=True)
        self.p = params
        self.q = log_q
        self._stop = threading.Event()
        self._total_stems = 0
        self._completed_stems = 0
        self._run_started_at = 0.0
        self._stats: dict = {}
        self._phase_timer = PhaseTimer()

    def _reset_stats(self, folders_total: int) -> None:
        self._stats = {
            'folders_total': folders_total,
            'folder_outcomes': {},
            'stems_skipped': {'confidence': 0, 'margin': 0, 'both': 0, 'error': 0},
            'folder_names': {
                'deleted_short': [],
                'deleted_incomplete': [],
                'deleted_both': [],
            },
            'stem_skip_details': [],
        }

    @staticmethod
    def _folder_display(rel: Path) -> str:
        s = str(rel).replace('\\', '/')
        return s if s and s != '.' else '.'

    def _record_folder_outcome(self, outcome: str, folder_name: str | None = None) -> None:
        oc = self._stats['folder_outcomes']
        oc[outcome] = oc.get(outcome, 0) + 1
        if not folder_name:
            return
        names = self._stats['folder_names']
        if outcome == 'deleted_both':
            names['deleted_short'].append(folder_name)
            names['deleted_incomplete'].append(folder_name)
            names['deleted_both'].append(folder_name)
        elif outcome in names:
            names[outcome].append(folder_name)

    def _record_stem_skip(self, folder_rel: Path, stem_path: Path, reason: str) -> None:
        self._stats['stems_skipped'][reason] = self._stats['stems_skipped'].get(reason, 0) + 1
        folder = self._folder_display(folder_rel)
        stem = stem_path.name
        self._stats['stem_skip_details'].append({
            'folder': folder,
            'stem': stem,
            'reason': reason,
        })

    def _log_run_summary(self, elapsed: float) -> None:
        oc = self._stats['folder_outcomes']
        folders_total = self._stats['folders_total']
        processed = sum(oc.values())
        self.log('')
        self.log('=== RMS Summary ===')
        self.log(f'  Total time: {format_elapsed(elapsed)}')
        if folders_total:
            self.log(
                f'  Avg per folder: {format_elapsed(elapsed / folders_total)} '
                f'({folders_total} folder(s))'
            )
        for key in FOLDER_OUTCOME_ORDER:
            if key in ('deleted_short', 'deleted_incomplete', 'deleted_both'):
                continue
            n = oc.get(key, 0)
            if n:
                self.log(f'  {FOLDER_OUTCOME_LABELS[key]}: {n}')
        names = self._stats['folder_names']
        short_names = names['deleted_short']
        if short_names:
            self.log(f'  {FOLDER_OUTCOME_LABELS["deleted_short"]}: {len(short_names)}')
            for name in short_names:
                self.log(f'    {name}')
        incomplete_names = names['deleted_incomplete']
        if incomplete_names:
            self.log(f'  {FOLDER_OUTCOME_LABELS["deleted_incomplete"]}: {len(incomplete_names)}')
            for name in incomplete_names:
                self.log(f'    {name}')
        if processed < folders_total:
            self.log(f'  Not processed: {folders_total - processed}')
        details = self._stats['stem_skip_details']
        if details:
            self.log('  Stems skipped during classification:')
            by_reason: dict[str, list[str]] = {}
            for item in details:
                folder, stem = item['folder'], item['stem']
                label = stem if folder == '.' else f'{folder}/{stem}'
                by_reason.setdefault(item['reason'], []).append(label)
            for reason in ('confidence', 'margin', 'both', 'error'):
                items = by_reason.get(reason, [])
                if items:
                    self.log(f'    {_skip_reason_label(reason)}: {len(items)}')
                    for label in items:
                        self.log(f'      {label}')
        self._phase_timer.log_summary(self.log, ORGANIZE_PHASE_LABELS)
        self.log('')
        self.log('DONE')

    def stop(self):
        self._stop.set()

    def log(self, msg: str):
        self.q.put(msg)

    def _report_progress(self) -> None:
        total = self._total_stems
        done = self._completed_stems
        if total <= 0:
            pct = 0.0
            eta = None
        else:
            pct = min(100.0, done / total * 100.0)
            elapsed = time.monotonic() - self._run_started_at
            eta = (elapsed / done * (total - done)) if done > 0 else None
        self.q.put((PROGRESS_TAG, pct, eta))

    def _mark_stems_done(self, count: int) -> None:
        if count <= 0:
            return
        self._completed_stems = min(self._total_stems, self._completed_stems + count)
        self._report_progress()

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.log(f'[ERROR] {e}')
            self.log(traceback.format_exc())
        finally:
            self.q.put(DONE_SENTINEL)

    def _resolve_output_dir(self, out_dir: Path, rel: Path,
                            manifest: dict, next_n_ref: list,
                            duration_sec: float | None = None) -> tuple[Path, dict, int]:
        append_duration = self.p.get('append_duration', False)
        naming_mode = self.p['naming_mode']
        if naming_mode == 'sequential':
            name = folder_name_with_duration(f"song_{next_n_ref[0]:04d}", duration_sec, append_duration)
            target_dir = out_dir / name
            manifest[name] = str(rel).replace('\\', '/')
            save_manifest(out_dir, manifest)
            next_n_ref[0] += 1
            self.log(f"  -> {name}  (original: {rel})")
        elif naming_mode == 'preserve':
            if str(rel) == '.':
                target_dir = out_dir
                display = '.'
            else:
                parts = list(rel.parts)
                parts[-1] = folder_name_with_duration(parts[-1], duration_sec, append_duration)
                target_dir = out_dir.joinpath(*parts)
                display = Path(*parts)
            self.log(f"  -> {display}")
        else:
            slug_parts = [slugify(pp) for pp in rel.parts if pp not in ('', '.')]
            if slug_parts:
                slug_parts[-1] = folder_name_with_duration(slug_parts[-1], duration_sec, append_duration)
            target_dir = out_dir.joinpath(*slug_parts) if slug_parts else out_dir
            self.log(f"  -> {Path(*slug_parts) if slug_parts else '.'}")
        return target_dir, manifest, next_n_ref[0]

    def _compute_gain(self, mixes: dict, cut: int) -> float:
        if not self.p['peak_norm'] or cut == 0:
            return 1.0
        total = sum(m[:, :cut] for m in mixes.values())
        peak = float(np.max(np.abs(total)))
        target_lin = 10 ** (-1.0 / 20.0)
        return target_lin / peak if peak > 0 else 1.0

    def _write_category_mixes(self, mixes: dict, buckets: dict, mode_cfg: dict,
                              target_dir: Path, ext: str, subtype: str, sr: int,
                              gain: float, cut: int) -> tuple[set[str], bool]:
        written: set[str] = set()
        had_errors = False
        for cat in mode_cfg['categories']:
            if cat not in mixes:
                self.log(f"  ({cat}: no stems)")
                continue
            scaled = mixes[cat][:, :cut] * gain
            out_path = target_dir / f"{cat}{ext}"
            try:
                write_audio(str(out_path), scaled, sr, subtype)
                written.add(cat)
                self.log(f"  wrote {cat}{ext}  ({len(buckets[cat])} stems, {format_duration_log(cut / sr)})")
            except Exception as e:
                had_errors = True
                self.log(f"  [export error] {cat}: {e}")
        return written, had_errors

    def _maybe_cleanup_output_folder(self, target_dir: Path, duration_sec: float,
                                     written_cats: set[str], mode_cfg: dict,
                                     manifest: dict, out_dir: Path) -> tuple[dict, str | None]:
        reasons: list[str] = []
        short = False
        incomplete = False

        if self.p.get('delete_if_short'):
            min_sec = float(self.p.get('min_duration_sec', 8))
            if duration_sec < min_sec:
                short = True
                reasons.append(
                    f'duration {format_duration(duration_sec)} < {format_duration(min_sec)}'
                )

        if self.p.get('delete_if_incomplete'):
            expected = set(mode_cfg['categories'])
            missing = sorted(expected - written_cats)
            if missing:
                incomplete = True
                reasons.append(f'missing: {", ".join(missing)}')

        if not reasons:
            return manifest, None

        folder_name = target_dir.name
        if target_dir.exists():
            try:
                send_to_recycle_bin(target_dir)
                self.log(f"  [deleted] {folder_name}: {'; '.join(reasons)}")
            except OSError as e:
                self.log(f"  [delete error] {folder_name}: {e}")
                return manifest, 'delete_failed'

        if self.p['naming_mode'] == 'sequential' and folder_name in manifest:
            del manifest[folder_name]
            save_manifest(out_dir, manifest)

        if short and incomplete:
            return manifest, 'deleted_both'
        if short:
            return manifest, 'deleted_short'
        return manifest, 'deleted_incomplete'

    def _process_folder(self, folder: Path, stems: list, model, device: str,
                        mode_cfg: dict, ext: str, subtype: str, sr: int,
                        out_dir: Path, manifest: dict, next_n_ref: list) -> tuple[dict, int]:
        in_dir = Path(self.p['input_dir'])
        rel = folder.relative_to(in_dir) if folder != in_dir else Path('.')

        if self.p.get('skip_existing'):
            existing = find_existing_output_dir(
                out_dir, rel, self.p['naming_mode'],
                mode_cfg['categories'], ext, manifest,
            )
            if existing:
                try:
                    display = existing.relative_to(out_dir)
                except ValueError:
                    display = existing
                self.log(f"  [skip existing] {display}")
                self._mark_stems_done(len(stems))
                self._record_folder_outcome('skip_existing')
                return manifest, next_n_ref[0]

        if self.p['dedup']:
            self.log('Starting de-duping...')
            t0 = time.monotonic()
            before = len(stems)
            stems = find_duplicates(stems, sr=sr, log_fn=self.log, device=device)
            dedup_dt = time.monotonic() - t0
            self._phase_timer.add('dedup', dedup_dt)
            self.log(f'  [dedup] finished in {format_duration_log(dedup_dt)}')
            if len(stems) < before:
                self.log(f"  [dedup] {before} -> {len(stems)} stems after deduplication")

        buckets = {c: [] for c in mode_cfg['categories']}
        skipped = 0
        had_ambig = False
        folder_had_errors = False
        skip_reasons: dict[str, int] = {'confidence': 0, 'margin': 0, 'both': 0, 'error': 0}

        self.log('Scanning stems...')
        t0 = time.monotonic()
        prescan_issues, stems = prescan_stems(stems)
        prescan_dt = time.monotonic() - t0
        self._phase_timer.add('prescan', prescan_dt)
        self.log(f'  [prescan] finished in {format_duration_log(prescan_dt)}')
        for path, reason in prescan_issues:
            tag = '[empty]' if reason.startswith('empty') else '[skip]'
            self.log(f"  {tag} {path.name} — {reason}")
            self._mark_stems_done(1)
            skipped += 1
            skip_reasons['error'] += 1
            folder_had_errors = True
            self._record_stem_skip(rel, path, 'error')

        self.log('Starting RMS classification...')
        t0 = time.monotonic()
        for path, energies, err in classify_batch(
                model, stems, device, batch_size=int(self.p['batch_size']), stop_event=self._stop):
            self._mark_stems_done(1)
            if self._stop.is_set():
                self.log('Stopped by user.')
                if device == 'cuda':
                    model.cpu()
                    torch.cuda.empty_cache()
                return manifest, next_n_ref[0]
            if err:
                self.log(f"  [skip] {path.name}: {err}")
                skipped += 1
                skip_reasons['error'] += 1
                folder_had_errors = True
                self._record_stem_skip(rel, path, 'error')
                continue
            label, _, top_share, _margin, skip_reason = classify_to_category(
                energies, mode_cfg, float(self.p['threshold']), float(self.p['min_margin']))
            self.log(
                f"  {label} {top_share:.0%}  →  {path.name}"
            )
            if label == 'skip':
                skipped += 1
                had_ambig = True
                if skip_reason:
                    skip_reasons[skip_reason] += 1
                    self._record_stem_skip(rel, path, skip_reason)
            else:
                buckets[label].append(path)
        class_dt = time.monotonic() - t0
        self._phase_timer.add('classification', class_dt)
        self.log(f'  [classification] finished in {format_duration_log(class_dt)}')

        if had_ambig and self.p['ambig_mode'] == 'skip_song':
            self.log('  [skip song] ambiguous stem(s) detected; skipping entire song')
            self._record_folder_outcome('skip_song')
            return manifest, next_n_ref[0]

        t0 = time.monotonic()
        mixes = {}
        for cat, paths in buckets.items():
            if not paths:
                continue
            m = mix_originals(paths, sr=sr)
            if m.shape[1] == 0:
                self.log(f"  [error] {cat}: all stems failed to load")
                folder_had_errors = True
                continue
            mixes[cat] = m
        mix_dt = time.monotonic() - t0
        if mixes:
            self._phase_timer.add('mixing', mix_dt)
            self.log(f'  [mixing] finished in {format_duration_log(mix_dt)}')

        cut = min((m.shape[1] for m in mixes.values()), default=0)
        if cut == 0:
            self.log('  [skip] no stems to export')
            self._record_folder_outcome('skip_no_stems')
            return manifest, next_n_ref[0]

        duration_sec = cut / sr
        target_dir, manifest, _ = self._resolve_output_dir(
            out_dir, rel, manifest, next_n_ref, duration_sec=duration_sec,
        )
        self.log('Starting to write output...')
        t0 = time.monotonic()
        target_dir.mkdir(parents=True, exist_ok=True)
        gain = self._compute_gain(mixes, cut)

        written_cats, export_errors = self._write_category_mixes(
            mixes, buckets, mode_cfg, target_dir, ext, subtype, sr, gain, cut,
        )
        folder_had_errors = folder_had_errors or export_errors

        if self.p['make_mixture'] and ext == '.wav' and cut > 0:
            total = sum(m[:, :cut] for m in mixes.values()) * gain
            mix_path = target_dir / f"mixture{ext}"
            try:
                write_audio(str(mix_path), total, sr, subtype)
                n = sum(len(buckets[c]) for c in mixes)
                peak_db = 20 * np.log10(max(float(np.max(np.abs(total))), 1e-12))
                self.log(f"  wrote mixture{ext}  ({n} stems, peak {peak_db:+.2f} dBFS)")
            except Exception as e:
                folder_had_errors = True
                self.log(f"  [export error] mixture: {e}")
        export_dt = time.monotonic() - t0
        self._phase_timer.add('export', export_dt)
        self.log(f'  [export] finished in {format_duration_log(export_dt)}')

        manifest, delete_outcome = self._maybe_cleanup_output_folder(
            target_dir, duration_sec, written_cats, mode_cfg, manifest, out_dir,
        )
        folder_name = self._folder_display(rel)
        if delete_outcome:
            self._record_folder_outcome(delete_outcome, folder_name)
        elif folder_had_errors:
            self._record_folder_outcome('success_with_errors')
        else:
            self._record_folder_outcome('success')

        if skipped:
            summary = format_skip_summary(skip_reasons)
            if summary:
                self.log(summary)

        return manifest, next_n_ref[0]

    def _run(self):
        model = None
        device = 'cpu'
        self._phase_timer = PhaseTimer()
        try:
            p = self.p
            device, device_warnings = resolve_processing_device(bool(p['use_cuda']))
            for warning in device_warnings:
                self.log(f'  [warn] {warning}')
            if FFMPEG:
                from ffmpeg_bootstrap import ffmpeg_folder_path

                self.log(f'  ffmpeg: {ffmpeg_folder_path()}')
            else:
                from ffmpeg_bootstrap import ffmpeg_missing_message

                self.log(f'  [warn] {ffmpeg_missing_message()}')
            self.log(f"  Device: {device}")
            from deps_bootstrap import demucs_models_present

            if not demucs_models_present():
                self.log('  Downloading Demucs model weights (~450 MB)...')
            self.log(f"  Loading model '{p['model_id']}' ...")
            t0 = time.monotonic()
            model = load_demucs_model(p['model_id']).eval().to(device)
            model_load_dt = time.monotonic() - t0
            self._phase_timer.add('model_load', model_load_dt)
            self.log(f'  [model load] finished in {format_duration_log(model_load_dt)}')
            self.log(f"  Model sources: {list(model.sources)}  (sr={model.samplerate})")

            in_dir, out_dir = Path(p['input_dir']), Path(p['output_dir'])
            mode_cfg = STEM_MODES[p['stem_mode']]
            ext, subtype = QUALITY_PRESETS[p['quality']].values()
            sr = model.samplerate

            self.log('  Scanning input folders...')
            t0 = time.monotonic()
            groups = collect_song_groups(in_dir, p['scan_mode'])
            input_scan_dt = time.monotonic() - t0
            self._phase_timer.add('input_scan', input_scan_dt)
            self.log(f'  [input scan] finished in {format_duration_log(input_scan_dt)}')

            if not groups:
                self.log(f"[skip] no audio files found under {in_dir}")
                return

            self._total_stems = sum(len(v) for v in groups.values())
            self._completed_stems = 0
            self._run_started_at = time.monotonic()
            self._reset_stats(len(groups))
            self._report_progress()

            self.log(f"  Found {sum(len(v) for v in groups.values())} stem(s) across {len(groups)} folder(s).")

            manifest: dict = {}
            next_n_ref = [0]
            if p['naming_mode'] == 'sequential' or p.get('skip_existing'):
                manifest = load_manifest(out_dir)
            if p['naming_mode'] == 'sequential':
                next_n_ref[0] = next_sequence_number(out_dir, manifest)
                self.log(f"  Naming: sequential; resuming at song_{next_n_ref[0]:04d}")
            elif p['naming_mode'] == 'preserve':
                self.log('  Naming: original folder name')
            else:
                self.log('  Naming: simplified folder name')

            scan_label = next(k for k, v in SCAN_MODES.items() if v == p['scan_mode'])
            self.log(f"  Scan: {scan_label}")
            if p.get('skip_existing'):
                self.log('  Resume: skipping songs that already have output stems')

            for fi, (folder, stems) in enumerate(sorted(groups.items()), 1):
                if self._stop.is_set():
                    self.log('Stopped by user.')
                    if device == 'cuda':
                        model.cpu()
                        torch.cuda.empty_cache()
                    return
                rel = folder.relative_to(in_dir) if folder != in_dir else Path('.')
                self.log('')
                self.log(f"=== [{fi}/{len(groups)}] {rel}  ({len(stems)} stems) ===")
                manifest, next_n_ref[0] = self._process_folder(
                    folder, stems, model, device, mode_cfg, ext, subtype, sr,
                    out_dir, manifest, next_n_ref,
                )

            elapsed = time.monotonic() - self._run_started_at
            self._log_run_summary(elapsed)
            self._completed_stems = self._total_stems
            self._report_progress()
        finally:
            if model is not None and device == 'cuda':
                model.cpu()
                del model
                torch.cuda.empty_cache()


class SdrWorker(threading.Thread):
    def __init__(self, params: dict, log_q: queue.Queue):
        super().__init__(daemon=True)
        self.p = params
        self.q = log_q
        self._stop = threading.Event()
        self._total_folders = 0
        self._completed_folders = 0
        self._run_started_at = 0.0
        self._stats: dict = {}
        self._phase_timer = PhaseTimer()

    def _reset_stats(self, folders_total: int) -> None:
        self._stats = {
            'folders_total': folders_total,
            'passed': 0,
            'skipped_incomplete': 0,
            'deleted_whole_folder': [],
            'deleted_stem_files': [],
        }

    def stop(self):
        self._stop.set()

    def log(self, msg):
        self.q.put(msg)

    def _log_sdr_line(self, filename: str, score: float, threshold: float):
        self.q.put((SDR_LOG_TAG, filename, score, threshold))

    def _report_progress(self) -> None:
        total = self._total_folders
        done = self._completed_folders
        if total <= 0:
            pct, eta = 0.0, None
        else:
            pct = min(100.0, done / total * 100.0)
            elapsed = time.monotonic() - self._run_started_at
            eta = (elapsed / done * (total - done)) if done > 0 else None
        self.q.put((PROGRESS_TAG, pct, eta))

    def _mark_folder_done(self) -> None:
        self._completed_folders = min(self._total_folders, self._completed_folders + 1)
        self._report_progress()

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.log(f'[ERROR] {e}')
            self.log(traceback.format_exc())
        finally:
            self.q.put(DONE_SENTINEL)

    def _folder_display(self, folder: Path, root: Path) -> str:
        try:
            rel = folder.relative_to(root)
            s = str(rel).replace('\\', '/')
            return s if s and s != '.' else folder.name
        except ValueError:
            return folder.name

    def _cleanup_empty_dirs_after_delete(self, deleted_paths: list[Path]) -> None:
        root = getattr(self, '_sdr_root', None)
        if root is None or not deleted_paths:
            return
        cleanup_empty_dirs_after_delete(deleted_paths, root)

    def _prune_empty_sdr_dirs(self) -> None:
        root = getattr(self, '_sdr_root', None)
        if root is None:
            return
        pruned = prune_empty_dirs_under(root)
        if pruned:
            self.log(f'  [cleanup] Removed {len(pruned)} empty folder(s).')

    def _log_sdr_summary(self, elapsed: float) -> None:
        st = self._stats
        self.log('')
        self.log('=== SI-SDR Summary ===')
        self.log(f'  Total time: {format_elapsed(elapsed)}')
        if st['folders_total']:
            self.log(
                f'  Avg per folder: {format_elapsed(elapsed / st["folders_total"])} '
                f'({st["folders_total"]} folder(s))'
            )
        if st['passed']:
            self.log(f'  Passed: {st["passed"]}')
        if st['skipped_incomplete']:
            self.log(f'  Skipped (incomplete stems): {st["skipped_incomplete"]}')
        whole = st['deleted_whole_folder']
        if whole:
            self.log(f'  Deleted (whole folder): {len(whole)}')
            for name in whole:
                self.log(f'    {name}')
        stems = st['deleted_stem_files']
        if stems:
            self.log(f'  Deleted (stem file): {len(stems)}')
            for name in stems:
                self.log(f'    {name}')
        self._phase_timer.log_summary(self.log, SDR_PHASE_LABELS)
        self.log('')
        self.log('DONE')

    def _process_song(
        self, stem_paths: dict[str, Path], display: str,
        categories: tuple[str, ...],
        thresholds: dict[str, float], model, device: str, sources: list[str], sr: int,
        fi: int, total: int, *,
        delete_whole: Path | None = None,
        layout: str = SDR_LAYOUT_MUSDB,
    ) -> None:
        self.log('')
        self.log(f'=== [{fi:02d}/{total}] {display} ===')

        missing = [cat for cat in categories if cat not in stem_paths]
        if missing:
            self.log(f'  [skip] missing expected stem(s): {", ".join(missing)}')
            self._stats['skipped_incomplete'] += 1
            return

        audios: dict[str, np.ndarray] = {}
        t0 = time.monotonic()
        for cat, path in stem_paths.items():
            try:
                audios[cat] = load_audio(str(path), sr=sr)
            except Exception as e:
                self.log(f'  [skip] {path.name}: load failed: {format_load_error(e)}')
                self._stats['skipped_incomplete'] += 1
                return
        load_dt = time.monotonic() - t0
        self._phase_timer.add('audio_load', load_dt)

        cut = min(a.shape[1] for a in audios.values())
        if cut == 0:
            self.log('  [skip] empty audio')
            self._stats['skipped_incomplete'] += 1
            return

        scores: dict[str, float] = {}
        sep_dt = 0.0
        sdr_dt = 0.0
        for cat in sdr_process_order(categories):
            ref = audios[cat][:, :cut]
            t0 = time.monotonic()
            try:
                out_np = separate_mixture(model, ref, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.log(f'  [error] {cat}: cuda OOM during separation')
                self._stats['skipped_incomplete'] += 1
                return
            except Exception as e:
                self.log(f'  [error] {cat}: separation failed: {e}')
                self._stats['skipped_incomplete'] += 1
                return
            sep_dt += time.monotonic() - t0

            if device == 'cuda':
                torch.cuda.empty_cache()

            t0 = time.monotonic()
            try:
                est = model_estimate_for_category(out_np[:, :, :cut], sources, cat, mixture=ref)
            except ValueError as e:
                self.log(f'  [error] {cat}: {e}')
                self._stats['skipped_incomplete'] += 1
                return
            scores[cat] = compute_si_sdr(ref, est)
            sdr_dt += time.monotonic() - t0
            self._log_sdr_line(stem_paths[cat].name, scores[cat], thresholds[cat])

        self._phase_timer.add('separation', sep_dt)
        self._phase_timer.add('sdr_compute', sdr_dt)
        self.log(
            f'  [timing] load {format_duration_log(load_dt)} | '
            f'separation {format_duration_log(sep_dt)} | '
            f'SDR {format_duration_log(sdr_dt)}'
        )

        failed = [(cat, stem_paths[cat], scores[cat]) for cat in categories
                  if scores[cat] < thresholds[cat]]

        if not failed:
            self._stats['passed'] += 1
            return

        delete_folder = self.p.get('sdr_delete_folder', True)
        deleted_paths: list[Path] = []
        if delete_folder:
            try:
                if layout == SDR_LAYOUT_MUSDB and delete_whole is not None:
                    send_to_recycle_bin(delete_whole)
                    deleted_paths.append(delete_whole)
                else:
                    for path in stem_paths.values():
                        send_to_recycle_bin(path)
                        deleted_paths.append(path)
                names = ', '.join(p.name for _, p, _ in failed)
                self.log(f'[deleted] {display}: Threshold not met: {names}')
                if layout == SDR_LAYOUT_MUSDB and delete_whole is not None:
                    self._stats['deleted_whole_folder'].append(display)
                elif layout in SDR_SINGLE_FLAT_LAYOUTS:
                    self._stats['deleted_stem_files'].extend(p.name for _, p, _ in failed)
                else:
                    self._stats['deleted_whole_folder'].append(display)
            except OSError as e:
                self.log(f'[delete error] {display}: {e}')
            else:
                self._cleanup_empty_dirs_after_delete(deleted_paths)
        else:
            for cat, path, _score in failed:
                try:
                    send_to_recycle_bin(path)
                    deleted_paths.append(path)
                    if layout == SDR_LAYOUT_STEMS:
                        loc = f'{path.parent.name}/{path.name}'
                    elif layout in SDR_SINGLE_FLAT_LAYOUTS:
                        loc = path.name
                    else:
                        loc = f'{display}\\{path.name}'
                    self.log(f'[deleted] {loc}: Threshold not met.')
                    self._stats['deleted_stem_files'].append(loc)
                except OSError as e:
                    self.log(f'[delete error] {path}: {e}')
            if deleted_paths:
                self._cleanup_empty_dirs_after_delete(deleted_paths)

    def _process_folder(
        self, folder: Path, root: Path, categories: tuple[str, ...],
        thresholds: dict[str, float], model, device: str, sources: list[str], sr: int,
        fi: int, total: int,
    ) -> None:
        display = self._folder_display(folder, root)
        stem_paths: dict[str, Path] = {}
        missing = []
        for cat in categories:
            p = find_category_stem(folder, cat)
            if p is None:
                missing.append(cat)
            else:
                stem_paths[cat] = p

        if missing:
            self.log('')
            self.log(f'=== [{fi:02d}/{total}] {display} ===')
            self.log(f'  [skip] missing expected stem(s): {", ".join(missing)}')
            self._stats['skipped_incomplete'] += 1
            return

        self._process_song(
            stem_paths, display, categories, thresholds, model, device, sources, sr,
            fi, total, delete_whole=folder, layout=SDR_LAYOUT_MUSDB,
        )

    def _run(self):
        model = None
        device = 'cpu'
        self._phase_timer = PhaseTimer()
        try:
            p = self.p
            device, device_warnings = resolve_processing_device(bool(p['use_cuda']))
            for warning in device_warnings:
                self.log(f'  [warn] {warning}')
            if FFMPEG:
                from ffmpeg_bootstrap import ffmpeg_folder_path

                self.log(f'  ffmpeg: {ffmpeg_folder_path()}')
            else:
                from ffmpeg_bootstrap import ffmpeg_missing_message

                self.log(f'  [warn] {ffmpeg_missing_message()}')
            self.log(f'  Device: {device}')
            from deps_bootstrap import demucs_models_present

            if not demucs_models_present():
                self.log('  Downloading Demucs model weights (~450 MB)...')
            self.log(f"  Loading model '{p['model_id']}' ...")
            t0 = time.monotonic()
            model = load_demucs_model(p['model_id']).eval().to(device)
            model_load_dt = time.monotonic() - t0
            self._phase_timer.add('model_load', model_load_dt)
            self.log(f'  [model load] finished in {format_duration_log(model_load_dt)}')
            sources = list(model.sources)
            sr = model.samplerate
            self.log(f'  Model sources: {sources}  (sr={sr})')

            mode_cfg = STEM_MODES[p['stem_mode']]
            preferred = mode_cfg['categories']
            root = Path(p['target_dir'])
            self._sdr_root = root.resolve()
            scan_mode = p['scan_mode']

            self.log('Starting SI-SDR determination...')
            t0 = time.monotonic()

            if 'sdr_categories' in p and 'sdr_layout' in p:
                categories = tuple(p['sdr_categories'])
                layout = p['sdr_layout']
            else:
                categories, layout = resolve_sdr_layout_and_categories(
                    root, scan_mode, preferred,
                )
            if layout is None or categories is None:
                self.log('[error] ' + describe_sdr_scan_failure(root, scan_mode, preferred))
                return

            if layout == SDR_LAYOUT_MIXED_FLAT:
                self.log(
                    '[info] Mixed loose files detected; classifying each filename '
                    'as instrumental or vocals.'
                )
            elif categories != preferred:
                self.log(
                    f'[info] Folders contain {len(categories)}-stem sets '
                    f'({", ".join(categories)}); Stem mode is {p["stem_mode"]}.'
                )
            elif layout == SDR_LAYOUT_SINGLE_FLAT:
                if p.get('sdr_user_picked_category'):
                    self.log(
                        f'[info] User-selected {categories[0]}-only folder; '
                        f'processing all audio files as {categories[0]}.'
                    )
                else:
                    patterns = single_flat_category_patterns(categories[0])
                    self.log(f'[info] {categories[0]}-only files detected ({patterns}).')

            process_all = bool(p.get('sdr_flat_process_all', False))
            if process_all and layout in SDR_SINGLE_FLAT_LAYOUTS:
                self.log(
                    f'[info] Processing all audio files in folder as {categories[0]}.'
                )

            thresholds = sdr_thresholds_for_categories(categories, p['sdr_thresholds'])
            targets = collect_sdr_targets(
                root, categories, scan_mode, layout, process_all=process_all,
            )
            if layout == SDR_LAYOUT_MUSDB:
                layout_label = 'Type 1 (MUSDB)'
            elif layout == SDR_LAYOUT_STEMS:
                layout_label = 'Type 2 (Stems)'
            elif layout == SDR_LAYOUT_SINGLE_FLAT:
                layout_label = f'Type 3 ({categories[0]}-only files)'
            elif layout == SDR_LAYOUT_MIXED_FLAT:
                layout_label = 'Type 3 (mixed vocals/instrumental files)'
            else:
                layout_label = layout
            self.log(f'  Detected layout: {layout_label}')

            if not targets:
                expected = ', '.join(f'{c}{SDR_STEM_EXT_LABEL}' for c in categories)
                self.log(
                    f'[error] No folders with all expected stems found under {root}.\n'
                    f'  Expected each song folder to contain: {expected}'
                )
                return

            self._phase_timer.add('target_scan', time.monotonic() - t0)
            self.log(f'  [target scan] finished in {format_duration_log(self._phase_timer.get("target_scan"))}')

            self._total_folders = len(targets)
            self._completed_folders = 0
            self._run_started_at = time.monotonic()
            self._reset_stats(len(targets))
            self._report_progress()

            scanned = len(collect_sdr_song_folders(root, scan_mode)) if layout == SDR_LAYOUT_MUSDB else len(targets)
            if layout == SDR_LAYOUT_SINGLE_FLAT:
                self.log(f'  Found {len(targets)} {categories[0]} file(s) to check.')
            elif layout == SDR_LAYOUT_MIXED_FLAT:
                counts = {
                    category: sum(category in target for target in targets)
                    for category in categories
                }
                self.log(
                    f'  Found {counts["instrumental"]} instrumental and '
                    f'{counts["vocals"]} vocals file(s) to check.'
                )
                unmatched = (
                    sum(1 for _ in iter_sdr_audio_files(root, scan_mode))
                    - len(targets)
                )
                if unmatched:
                    self.log(
                        f'  [skip] {unmatched} file(s) had no recognizable '
                        'vocals/instrumental filename marker.'
                    )
            else:
                unit = 'folder(s)' if layout == SDR_LAYOUT_MUSDB else 'song(s)'
                self.log(
                    f'  Found {len(targets)} {unit} with complete '
                    f'{len(categories)}-stem sets (of {scanned} scanned).'
                )

            try:
                for fi, target in enumerate(targets, 1):
                    if self._stop.is_set():
                        self.log('Stopped by user.')
                        if device == 'cuda':
                            model.cpu()
                            torch.cuda.empty_cache()
                        return
                    if layout == SDR_LAYOUT_MUSDB:
                        self._process_folder(
                            target, root, categories, thresholds, model, device, sources, sr,
                            fi, len(targets),
                        )
                    elif layout in SDR_SINGLE_FLAT_LAYOUTS:
                        stem_paths: dict[str, Path] = target
                        cat = next(iter(stem_paths))
                        display = stem_paths[cat].stem
                        target_categories = (
                            (cat,) if layout == SDR_LAYOUT_MIXED_FLAT else categories
                        )
                        self._process_song(
                            stem_paths, display, target_categories, thresholds,
                            model, device, sources, sr,
                            fi, len(targets), layout=layout,
                        )
                    else:
                        stem_paths: dict[str, Path] = target
                        cat = next(iter(stem_paths))
                        display = stem_file_song_key(stem_paths[cat], cat)
                        self._process_song(
                            stem_paths, display, categories, thresholds, model, device, sources, sr,
                            fi, len(targets), layout=SDR_LAYOUT_STEMS,
                        )
                    self._mark_folder_done()
            finally:
                self._prune_empty_sdr_dirs()

            elapsed = time.monotonic() - self._run_started_at
            self._log_sdr_summary(elapsed)
            self._completed_folders = self._total_folders
            self._report_progress()
        finally:
            if model is not None and device == 'cuda':
                model.cpu()
                del model
                torch.cuda.empty_cache()


COLORS = {
    'bg':         '#1e1f26',
    'panel':      '#262833',
    'panel2':     '#2e3140',
    'fg':         '#e6e8ef',
    'fg_dim':     '#9aa0b4',
    'accent':     '#7c5cff',
    'accent_hov': '#9077ff',
    'danger':     '#e25c5c',
    'log_bg':     '#15161c',
    'log_fg':     '#d6dae8',
    'border':     '#3a3d4d',
    'status_trough': '#343647',
    'status_pct':    '#ffffff',
}


def _blend_hex(fg: str, bg: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    fg = fg.lstrip('#')
    bg = bg.lstrip('#')
    fr, fg_g, fb = (int(fg[i:i + 2], 16) for i in (0, 2, 4))
    br, bg_g, bb = (int(bg[i:i + 2], 16) for i in (0, 2, 4))
    return (
        f'#{int(fr + (br - fr) * t):02x}'
        f'{int(fg_g + (bg_g - fg_g) * t):02x}'
        f'{int(fb + (bb - fb) * t):02x}'
    )


def _entry_select_colors() -> tuple[str, str]:
    c = COLORS
    return (
        _blend_hex(c['accent'], c['panel2'], 0.36),
        _blend_hex(c['accent'], c['panel2'], 0.58),
    )

# tk.Label only — ttk TLabel styles ignore small font sizes on Windows/clam.
# Matches Rename Files subtitle (size 12, dim text).
HEADER_DESC_FONT = ('Segoe UI', 12)
HEADER_DESC_COLOR = '#9aa0b4'  # DARK['text_dim']
ACTION_BTN_FONT = ('Segoe UI Semibold', 10)
ACTION_BTN_PADX = 14
ACTION_BTN_PADY = 4
CTRL_FIELD_PAD = 3
CTRL_ROW_PADY = 2
CTRL_BTN_FONT = ('Segoe UI', 10)
CTRL_BTN_PADX = 12
CTRL_BTN_PADY = 4
PATH_BTN_FONT = ('Segoe UI', 10)
PATH_BTN_PADX = 8
PATH_BTN_PADY = 4
PATH_COMBO_WIDTH = 24
STATUS_FONT = ('Segoe UI', 9)
RESOURCE_BAR_HEIGHT = 10
RESOURCE_BAR_WIDTH = 52
RESOURCE_ROW_HEIGHT = 16
STATUS_PROGRESS_ROW_HEIGHT = 14
STATUS_PAD_TOP = 3
STATUS_ROW_GAP = 7
STATUS_PAD_BOTTOM = 26
STATUS_FRAME_HEIGHT = (
    STATUS_PAD_TOP + RESOURCE_ROW_HEIGHT + STATUS_ROW_GAP
    + STATUS_PROGRESS_ROW_HEIGHT + STATUS_PAD_BOTTOM
)
STATUS_IDLE_Y = STATUS_PAD_TOP + RESOURCE_ROW_HEIGHT + STATUS_ROW_GAP
STATUS_PAD_X = 10
SDR_THRESH_NOTE_FONT = ('Segoe UI', 8)
STATUS_PCT_FONT = ('Segoe UI Semibold', 9)
STATUS_PROGRESS_HEIGHT = 14
STATUS_PROGRESS_Y_PAD = 1
STATUS_TOP_PAD = 0
STATUS_BOTTOM_PAD = 2

LOG_FONT = ('Consolas', 10)
LOG_FONT_BOLD = ('Consolas', 10, 'bold')
# Confidence % — dim + slightly smaller than body LOG text.
LOG_PCT_FONT = ('Consolas', 8)
# Log chips: Arial (clean sans-serif on Windows); shared with About legend.
LOG_STEM_CHIP_FONT_SIZE = 9
LOG_STEM_GAP_TAG = 'log_stem_gap'
LOG_STEM_GAP_FONT = ('Consolas', 4)
LOG_FOLDER_STEM_GAP_TAG = 'log_folder_stem_gap'
LOG_FOLDER_STEM_GAP_FONT = ('Consolas', 7)
FOLDER_TITLE_RE = re.compile(r'^=== \[\d+/\d+\]')
STEM_BLOCK_GAP_AFTER = frozenset({
    'Starting RMS classification...',
})
# 4-way: bass / drums / other / vocals — 2-way: instrumental / vocals
LOG_STEM_COLORS = {
    'bass':         '#ef4444',
    'drums':        '#f59e0b',
    'other':        '#10b981',
    'vocals':       '#a855f7',
    'instrumental': '#60A5FA',
}
# Genre & Gender badges (Classify-style chips).
# Soft chip fill / wet text = button text (COLORS['log_fg']).
LOG_GG_COLORS = {
    'female': '#ec4899',  # pink
    'male':   '#60A5FA',  # blue (same family as instrumental)
    'dry':    COLORS['log_fg'],  # genre + dry bg
    'wet':    '#262833',  # wet + style chip bg
}
LOG_GG_FG = {
    'dry': '#262833',  # dark on soft fill
    'wet': COLORS['log_fg'],  # soft white on wet/style
}
LOG_SKIP_COLOR = '#636b7a'
LOG_MARGIN_COLOR = '#9aa0b4'
LOG_DELETED_COLOR = '#e89292'
LOG_WARN_COLOR = '#ecc990'
STEM_CLASSIFY_RE = re.compile(
    # Optional pct; legacy margin ignored if present.
    r'^(\s+)([a-z_]+)(?: (\d+%))?(?: \(margin [^)]+\))?(  →  .+)$'
)
GG_HEADER_RE = re.compile(r'^=== .+ ===\s*$')
GG_BADGE_RE = re.compile(
    r'^(\s*)(female|male|dry|wet)'
    r'(?: \(confidence [^)]+\)| (\d+%))?\s*$',
    re.IGNORECASE,
)
GG_PCT_ONLY_RE = re.compile(r'^(\s*)(\d+%)\s*$')


_stem_chip_font_obj: tkfont.Font | None = None
_log_stem_chip_font_spec: tuple[str, ...] | None = None
_stem_chip_cache: dict[str, str] = {}
_stem_chip_width_px: int = 0


def _resolve_log_stem_chip_font(text: tk.Misc) -> tuple[str, ...]:
    families = {name.lower(): name for name in tkfont.families(text)}
    arial = families.get('arial')
    if arial:
        return arial, LOG_STEM_CHIP_FONT_SIZE, 'bold'
    return 'Segoe UI Semibold', LOG_STEM_CHIP_FONT_SIZE


def _pad_chip_to_width(
    font: tkfont.Font, label: str, width_px: int, *, lower: bool = False,
) -> str:
    """Pad with spaces to at least width_px; longer labels keep ~1-space sides (bg grows)."""
    text = label.strip()
    if lower:
        text = text.lower()
    left = right = 1
    while font.measure((' ' * left) + text + (' ' * right)) < width_px:
        if left <= right:
            left += 1
        else:
            right += 1
    return (' ' * left) + text + (' ' * right)


def _pad_stem_chip(font: tkfont.Font, label: str, width_px: int) -> str:
    return _pad_chip_to_width(font, label, width_px, lower=True)


def _stem_chip_color(label: str) -> str:
    key = label.strip().lower()
    if key == 'skip':
        return LOG_SKIP_COLOR
    if key in LOG_GG_COLORS:
        return LOG_GG_COLORS[key]
    return LOG_STEM_COLORS.get(key, COLORS['panel2'])


def _chip_label_set() -> list[str]:
    return list(LOG_STEM_COLORS.keys()) + list(LOG_GG_COLORS.keys()) + ['skip']


def _init_stem_chip_layout(text: tk.Misc) -> None:
    """Precompute equal-width chip strings (Arial + pixel padding)."""
    global _stem_chip_font_obj, _log_stem_chip_font_spec, _stem_chip_cache, _stem_chip_width_px
    labels = _chip_label_set()
    if _stem_chip_cache and set(labels).issubset(_stem_chip_cache):
        return
    spec = _resolve_log_stem_chip_font(text)
    _log_stem_chip_font_spec = spec
    _stem_chip_font_obj = tkfont.Font(root=text, font=spec)
    font = _stem_chip_font_obj
    longest = max(labels, key=len)
    target = max(font.measure(longest), font.measure('n' * 9)) + 16
    provisional = {lb: _pad_stem_chip(font, lb, target) for lb in labels}
    _stem_chip_width_px = max(font.measure(s) for s in provisional.values())
    _stem_chip_cache = {
        lb: _pad_stem_chip(font, lb, _stem_chip_width_px) for lb in labels
    }


def _format_stem_chip_text(label: str) -> str:
    """Fixed-width chip text so tag backgrounds align."""
    text = label.strip().lower()
    if text in _stem_chip_cache:
        return _stem_chip_cache[text]
    if _stem_chip_font_obj is not None and _stem_chip_width_px > 0:
        return _pad_stem_chip(_stem_chip_font_obj, text, _stem_chip_width_px)
    return f' {text} '


def _format_gg_value_chip(label: str) -> str:
    """
    Genre/style chip: same min width as gender/reverb badges; keep case.
    Longer text → chip grows (no clipping).
    """
    text = (label or '?').strip() or '?'
    if _stem_chip_font_obj is not None and _stem_chip_width_px > 0:
        return _pad_chip_to_width(
            _stem_chip_font_obj, text, _stem_chip_width_px, lower=False,
        )
    return f' {text} '


def _gg_confidence_tag(text: str) -> str:
    """Green if percent > 70 (or fraction > 0.7), pale orange otherwise."""
    try:
        m = re.search(r'(-?\d+(?:\.\d+)?)', text or '')
        if not m:
            return 'gg_conf_low'
        v = float(m.group(1))
        # Whole percent (72) vs legacy fraction (0.72).
        if v <= 1.0 and '%' not in (text or ''):
            v *= 100.0
        if v > 70.0:
            return 'gg_conf'
    except ValueError:
        pass
    return 'gg_conf_low'


def _gg_insert_confidence(text_widget: tk.Text, conf_text: str) -> None:
    """Insert '(confidence 72%)' with only the percent value colored."""
    raw = conf_text or ''
    # Keep leading gap after badge (strip() was eating the space).
    lead = raw[: len(raw) - len(raw.lstrip())]
    if lead:
        text_widget.insert('end', lead, 'log_margin')
    body = raw.strip()
    m = re.match(
        r'^(\(confidence\s+)(-?\d+(?:\.\d+)?)(%?)(\))\s*$',
        body,
        flags=re.IGNORECASE,
    )
    if not m:
        text_widget.insert('end', body or raw, 'log_margin')
        return
    text_widget.insert('end', m.group(1), 'log_margin')
    value = m.group(2) + (m.group(3) or '')
    text_widget.insert('end', value, _gg_confidence_tag(value))
    text_widget.insert('end', m.group(4), 'log_margin')


def _ensure_stem_chip_layout(parent: tk.Misc) -> None:
    _init_stem_chip_layout(parent)


def _stem_chip_font_spec(parent: tk.Misc) -> tuple[str, ...]:
    _ensure_stem_chip_layout(parent)
    return _log_stem_chip_font_spec or _resolve_log_stem_chip_font(parent)


def _stem_log_tag(label: str) -> str:
    key = label.lower()
    if key == 'skip':
        return 'log_stem_skip'
    if key in LOG_STEM_COLORS or key in LOG_GG_COLORS:
        return f'log_stem_{key}'
    return 'log_stem_unknown'


def _configure_stem_log_tags(text: tk.Text) -> None:
    """Colored stem labels via text tags (scales; embedded Label chips stop after ~N lines)."""
    _ensure_stem_chip_layout(text)
    chip_font = _stem_chip_font_spec(text)
    for stem, color in {**LOG_STEM_COLORS, **LOG_GG_COLORS}.items():
        text.tag_configure(
            f'log_stem_{stem}',
            foreground=LOG_GG_FG.get(stem, 'white'),
            background=color,
            font=chip_font,
        )
    text.tag_configure(
        'log_stem_skip',
        foreground='white',
        background=LOG_SKIP_COLOR,
        font=chip_font,
    )
    text.tag_configure(
        'log_stem_unknown',
        foreground='white',
        background=COLORS['panel2'],
        font=chip_font,
    )


def _skip_reason_label(reason: str) -> str:
    return {
        'confidence': "didn't match confidence",
        'margin': "didn't match min. margin",
        'both': "didn't match confidence + min. margin",
        'error': 'error',
    }.get(reason, reason)


def format_skip_summary(counts: dict[str, int]) -> str:
    active = [(k, counts[k]) for k in ('confidence', 'margin', 'both', 'error') if counts.get(k)]
    total = sum(counts.values())
    if not total:
        return ''
    if len(active) == 1:
        detail = _skip_reason_label(active[0][0])
    else:
        detail = ', '.join(f"{n} {_skip_reason_label(k)}" for k, n in active)
    return f"  ({total} stem(s) skipped: {detail})"


FOLDER_OUTCOME_LABELS = {
    'success': 'Successful',
    'success_with_errors': 'Successful (with errors)',
    'skip_existing': 'Skipped - output already exists',
    'skip_song': 'Skipped - ambiguous song',
    'skip_no_stems': 'Skipped - no stems to export',
    'deleted_short': 'Deleted - shorter than minimum duration',
    'deleted_incomplete': 'Deleted - missing expected stem(s)',
    'deleted_both': 'Deleted - too short and incomplete',
    'delete_failed': 'Delete failed',
}

FOLDER_OUTCOME_ORDER = (
    'success', 'success_with_errors', 'skip_existing', 'skip_song', 'skip_no_stems',
    'deleted_short', 'deleted_incomplete', 'deleted_both', 'delete_failed',
)


def _rgb_from_hex(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _tint_hex(fg: str, base: str, amount: float) -> str:
    """Tint base toward fg — for light text on dark log backgrounds."""
    r1, g1, b1 = _rgb_from_hex(fg)
    r2, g2, b2 = _rgb_from_hex(base)
    return (
        f'#{int(r2 + (r1 - r2) * amount):02x}'
        f'{int(g2 + (g1 - g2) * amount):02x}'
        f'{int(b2 + (b1 - b2) * amount):02x}'
    )


SDR_PASS_COLOR = '#7ee0a0'
SDR_FAIL_COLOR = '#ff7a7a'
SDR_LABEL_COLOR = '#d6dae8'
SDR_DONE_LINE = 'DONE'

INFO_ICON_SIZE = 16
INFO_ICON_FONT = ('Segoe UI Semibold', 10)
INFO_ICON_CX_NUDGE = 0.65  # optical center — shift "?" slightly right
INFO_ICON_CY_NUDGE = -0.75
INFO_ICON_OPACITY_DIM = 0.42
INFO_ICON_OPACITY_FULL = 1.0
ABOUT_DIALOG_W = 760
ABOUT_DIALOG_H = 1030
ABOUT_PAD_X = 32
ABOUT_COL_GAP = 32
ABOUT_ICON_SIZE = 112
ABOUT_TITLE_FONT = ('Segoe UI Semibold', 15)
ABOUT_SECTION_FONT = ('Segoe UI', 10, 'bold')
ABOUT_BODY_FONT = ('Segoe UI', 9)
ABOUT_LEGEND_FONT = ('Segoe UI Semibold', 9)
ABOUT_LEGEND_SUB_FONT = ('Segoe UI', 9, 'bold')
ABOUT_COL_W = (ABOUT_DIALOG_W - ABOUT_PAD_X * 2 - ABOUT_COL_GAP) // 2
ABOUT_FULL_W = ABOUT_DIALOG_W - ABOUT_PAD_X * 2
ABOUT_LEGEND_WRAP = ABOUT_COL_W - 100
ABOUT_LEGEND_ROW_H = 36
ABOUT_LEGEND_ROW_PAD = 4
ABOUT_LEGEND_SLOT = ABOUT_LEGEND_ROW_H + ABOUT_LEGEND_ROW_PAD
ABOUT_LEGEND_SECTION_H = 20
ABOUT_LEGEND_SECTION_TOP = 12
ABOUT_LEGEND_SECTION_BOTTOM = 8

ABOUT_BULLETS = (
    'Scans folders of audio stems and classifies each one via Demucs (vocals, drums, bass, other)',
    'Mixes original files into cleanly-grouped outputs per folder',
    'Skips stems with ambiguous classification (e.g. background vocals + guitar)',
    'Supports 2-way (instrumental/vocals) and 4-way (bass/drums/other/vocals) mixing modes',
)

ABOUT_HOW_IT_WORKS_TAIL = (
    'Accepted stems are summed from their original files (not AI-separated). '
    'You can filter short or incomplete outputs, '
    'resume by skipping existing results, and export an optional mixture.wav per song.\n\n'
    'Additionally, you can play 2-stem or 4-stem folders to audition mixes, '
    'using the STEM player with the Play button.'
)

ABOUT_SDR_BULLETS = (
    'Optional SI-SDR quality check on organized stem folders.',
    'Each stem file is processed individually through Demucs and compared to the model output.',
    'Set per-stem thresholds (dB) — stems scoring below are moved to the Recycle Bin.',
    'Optionally delete the whole folder (Type 1) or all stems for a song (Type 2) when any stem fails.',
    'Uses the same 2-way or 4-way stem mode and thresholds as the main classification settings.',
)

ABOUT_SDR_LAYOUTS = (
    ('Type 1', DATASET_TYPE1_URL,
     ': one folder per song containing vocals.wav, bass.wav, etc.'),
    ('Type 2', DATASET_TYPE2_URL,
     ': one folder per stem category (vocals/, bass/, …) with a file per song inside'),
)

ABOUT_LEGEND_2WAY = (
    ('instrumental', 'Non-vocal content — drums, bass, keys, synths, and other instruments combined.'),
    ('vocals', 'Lead vocals, backing vocals, and vocal FX.'),
)

ABOUT_LEGEND_4WAY = (
    ('bass', 'Bass guitar, synth bass, low-end.'),
    ('drums', 'Kick, snare, hats, percussion.'),
    ('other', 'Keys, guitars, synths, and everything else non-vocal.'),
    ('vocals', 'Lead vocals, backing vocals, and vocal FX.'),
)

ABOUT_LEGEND_SKIP = (
    'Ambiguous or rejected stem — below confidence/margin thresholds, or classification error.'
)


def _device_notice_message() -> str:
    if cuda_effective():
        name = cuda_device_name()
        return f'GPU acceleration is available ({name}).' if name else 'GPU acceleration is available.'

    incompatible = cuda_incompatibility_hint()
    if incompatible:
        return '\n'.join([
            'STEM organizer detected your NVIDIA GPU, but the installed PyTorch build cannot run on it.',
            '',
            incompatible,
            '',
            'Until you reinstall with the matching PyTorch build, processing will use CPU.',
        ])

    lines = [
        'STEM organizer is running in CPU mode.',
        '',
    ]
    if torch_cuda_built():
        lines += [
            'PyTorch was built with CUDA support, but no usable NVIDIA GPU was detected.',
            '',
            'To enable GPU acceleration (optional):',
            '  • NVIDIA GPU with up-to-date drivers',
            '  • Restart the app after installing drivers',
            '',
            'CPU mode works on any PC — processing is just slower.',
        ]
    elif _is_frozen():
        lines += [
            'PyTorch is installed separately (run install-deps.bat once).',
            'This build does not bundle GPU libraries.',
            '',
            'install-deps.bat asks whether you have an NVIDIA GPU',
            'and installs CPU or CUDA PyTorch automatically.',
            '',
            'You do not need a GPU to use this app.',
        ]
    else:
        lines += [
            'For optional GPU acceleration, install CUDA PyTorch via install-deps.bat.',
            '  RTX 30/40 series: cu124',
            '  RTX 50-series (5090, etc.): cu128',
            '',
            'CPU mode works on any PC — processing is just slower.',
        ]
    return '\n'.join(lines)


def show_device_notice_dialog(parent: tk.Misc) -> bool:
    """Show CPU/GPU info dialog. Returns True when 'Don't show again' is checked."""
    dlg = tk.Toplevel(parent)
    dlg.title('Processing device')
    dlg.configure(bg=COLORS['panel'])
    apply_window_icon(dlg)
    dlg.resizable(False, False)
    dlg.transient(parent)
    dlg.attributes('-topmost', True)

    outer = tk.Frame(dlg, bg=COLORS['panel'])
    outer.pack(fill='both', expand=True, padx=24, pady=(20, 16))

    tk.Label(
        outer,
        text=_device_notice_message(),
        font=HEADER_DESC_FONT,
        fg=COLORS['fg'],
        bg=COLORS['panel'],
        justify='left',
        wraplength=460,
        anchor='w',
    ).pack(fill='x')

    dont_show = tk.BooleanVar(value=False)
    tk.Checkbutton(
        outer,
        text="Don't show this message again",
        variable=dont_show,
        font=HEADER_DESC_FONT,
        fg=COLORS['fg_dim'],
        bg=COLORS['panel'],
        activebackground=COLORS['panel'],
        activeforeground=COLORS['fg'],
        selectcolor=COLORS['panel2'],
        highlightthickness=0,
        bd=0,
        anchor='w',
    ).pack(fill='x', pady=(16, 0))

    dismissed = {'value': False}

    def close_dialog() -> None:
        dismissed['value'] = bool(dont_show.get())
        dlg.destroy()

    tk.Button(
        outer,
        text='OK',
        command=close_dialog,
        font=ACTION_BTN_FONT,
        bg=COLORS['accent'],
        fg='white',
        activebackground=COLORS['accent_hov'],
        activeforeground='white',
        relief='flat',
        padx=18,
        pady=6,
        cursor='hand2',
        bd=0,
    ).pack(pady=(16, 0))

    dlg.bind('<Return>', lambda _e: close_dialog())
    dlg.bind('<Escape>', lambda _e: close_dialog())
    dlg.protocol('WM_DELETE_WINDOW', close_dialog)

    dlg.update_idletasks()
    width = max(480, outer.winfo_reqwidth() + 48)
    height = outer.winfo_reqheight() + 40
    # Center on parent + dark title bar — same as About / help dialogs.
    top = parent.winfo_toplevel()
    top.update_idletasks()
    x = top.winfo_rootx() + max(0, (top.winfo_width() - width) // 2)
    y = top.winfo_rooty() + max(0, (top.winfo_height() - height) // 2)
    dlg.geometry(f'{width}x{height}+{x}+{y}')
    dlg.update_idletasks()
    _win_apply_dwm_rounded_corners(dlg)
    dlg.after(20, lambda: _win_apply_dwm_rounded_corners(dlg))
    dlg.grab_set()
    dlg.focus_force()
    parent.wait_window(dlg)
    return dismissed['value']


def _center_toplevel(win: tk.Toplevel, width: int, height: int) -> None:
    win.update_idletasks()
    x, y, w, h = _place_window_centered(win, width, height)
    win.geometry(f'{w}x{h}+{x}+{y}')


def _legend_chip(parent: tk.Misc, label: str) -> tk.Label:
    _ensure_stem_chip_layout(parent)
    return tk.Label(
        parent,
        text=_format_stem_chip_text(label),
        font=_stem_chip_font_spec(parent),
        foreground='white',
        background=_stem_chip_color(label),
        pady=0,
    )


def _legend_skip_chip(parent: tk.Misc) -> tk.Label:
    return _legend_chip(parent, 'skip')


_SDR_STEM_PICK_LABELS = {
    'instrumental': 'Instrumental',
    'vocals': 'Vocals',
    'bass': 'Bass',
    'drums': 'Drums',
    'other': 'Other',
}


def ask_sdr_stem_category(parent: tk.Misc) -> str | None:
    """Ask which single stem type a folder contains when auto-detect fails."""
    choice: list[str | None] = [None]
    dlg = tk.Toplevel(parent)
    dlg.title('Stem type')
    dlg.configure(bg=COLORS['panel'])
    dlg.transient(parent)
    dlg.grab_set()
    dlg.resizable(False, False)

    tk.Label(
        dlg,
        text='Could not auto-detect the stem layout.\n'
             'What type of stem is in this folder?',
        bg=COLORS['panel'], fg=COLORS['fg'],
        font=('Segoe UI', 10), justify='left',
    ).pack(anchor='w', padx=24, pady=(20, 12))

    btn_frm = tk.Frame(dlg, bg=COLORS['panel'])
    btn_frm.pack(padx=24, pady=(0, 8), fill='x')

    def pick(cat: str) -> None:
        choice[0] = cat
        dlg.destroy()

    for cat in SDR_STEM_PICK_ORDER:
        row = tk.Frame(btn_frm, bg=COLORS['panel'])
        row.pack(fill='x', pady=3)
        chip = _legend_chip(row, cat)
        chip.pack(side='left')
        for widget in (chip, row):
            widget.bind('<Button-1>', lambda _e, c=cat: pick(c))
        tk.Button(
            row, text=_SDR_STEM_PICK_LABELS[cat], anchor='w',
            bg=COLORS['panel2'], fg=COLORS['fg'],
            activebackground=COLORS['accent'], activeforeground='white',
            relief='flat', borderwidth=0, padx=12, pady=6, cursor='hand2',
            command=lambda c=cat: pick(c),
        ).pack(side='left', fill='x', expand=True, padx=(8, 0))

    tk.Button(
        dlg, text='Cancel', command=dlg.destroy,
        bg=COLORS['panel2'], fg=COLORS['fg_dim'],
        activebackground=COLORS['panel'], relief='flat', borderwidth=0,
        padx=16, pady=6, cursor='hand2',
    ).pack(pady=(4, 16))

    dlg.protocol('WM_DELETE_WINDOW', dlg.destroy)
    dlg.bind('<Escape>', lambda _e: dlg.destroy())
    dlg.update_idletasks()
    pw, ph = parent.winfo_width(), parent.winfo_height()
    px, py = parent.winfo_rootx(), parent.winfo_rooty()
    w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
    dlg.geometry(f'+{px + max(0, (pw - w) // 2)}+{py + max(0, (ph - h) // 2)}')
    parent.wait_window(dlg)
    return choice[0]


def _legend_grid_row(
    grid: tk.Misc,
    row: int,
    col_offset: int,
    desc: str,
    *,
    label: str | None = None,
    skip: bool = False,
    padx_chip: tuple[int, int] = (0, 12),
    minsize: int = ABOUT_LEGEND_SLOT,
) -> None:
    grid.rowconfigure(row, minsize=minsize, weight=0)
    chip = _legend_skip_chip(grid) if skip else _legend_chip(grid, label or '')
    chip.grid(row=row, column=col_offset, sticky='nw', padx=padx_chip, pady=(2, 0))
    tk.Label(
        grid, text=desc, justify='left', wraplength=ABOUT_LEGEND_WRAP,
        font=ABOUT_LEGEND_FONT, fg=COLORS['fg_dim'], bg=COLORS['panel'], anchor='nw',
        pady=0,
    ).grid(row=row, column=col_offset + 1, sticky='nw', pady=(2, 0))


def _legend_section_header(grid: tk.Misc, row: int, text: str) -> None:
    grid.rowconfigure(
        row,
        minsize=ABOUT_LEGEND_SECTION_H + ABOUT_LEGEND_SECTION_TOP + ABOUT_LEGEND_SECTION_BOTTOM,
        weight=0,
    )
    tk.Label(
        grid, text=text,
        font=ABOUT_LEGEND_SUB_FONT, fg=COLORS['fg'], bg=COLORS['panel'], anchor='w',
    ).grid(
        row=row, column=0, columnspan=2, sticky='sw',
        pady=(ABOUT_LEGEND_SECTION_TOP, ABOUT_LEGEND_SECTION_BOTTOM),
    )


def _build_about_legend(parent: tk.Misc) -> None:
    grid = tk.Frame(parent, bg=COLORS['panel'])
    grid.pack(fill='x', pady=(0, 16))

    grid.columnconfigure(0, weight=0)
    grid.columnconfigure(1, weight=1, uniform='leg_col')
    grid.columnconfigure(2, weight=0)
    grid.columnconfigure(3, weight=1, uniform='leg_col')

    # Row 0: Section headers
    # Left: Stem legend | Right: 2-stem
    grid.rowconfigure(0, minsize=ABOUT_LEGEND_SECTION_H + ABOUT_LEGEND_SECTION_BOTTOM, weight=0)
    tk.Label(
        grid, text='Stem legend',
        font=ABOUT_SECTION_FONT, fg=COLORS['fg'], bg=COLORS['panel'], anchor='w',
    ).grid(
        row=0, column=0, columnspan=2, sticky='sw',
        pady=(0, ABOUT_LEGEND_SECTION_BOTTOM),
    )
    tk.Label(
        grid, text='2-stem',
        font=ABOUT_LEGEND_SUB_FONT, fg=COLORS['fg'], bg=COLORS['panel'], anchor='w',
    ).grid(
        row=0, column=2, columnspan=2, sticky='sw',
        padx=(ABOUT_COL_GAP, 0),
        pady=(0, ABOUT_LEGEND_SECTION_BOTTOM),
    )

    # Row 1: Left: skip | Right: instrumental & vocals nested Frame
    _legend_grid_row(grid, 1, 0, ABOUT_LEGEND_SKIP, skip=True)

    right_top = tk.Frame(grid, bg=COLORS['panel'])
    right_top.grid(row=1, column=2, columnspan=2, sticky='nw')

    # instrumental (2-stem)
    inst_row = tk.Frame(right_top, bg=COLORS['panel'])
    inst_row.pack(fill='x', anchor='nw', pady=(2, 0))
    _legend_chip(inst_row, ABOUT_LEGEND_2WAY[0][0]).pack(side='left', anchor='nw', padx=(ABOUT_COL_GAP, 12))
    tk.Label(
        inst_row, text=ABOUT_LEGEND_2WAY[0][1], justify='left', wraplength=ABOUT_LEGEND_WRAP,
        font=ABOUT_LEGEND_FONT, fg=COLORS['fg_dim'], bg=COLORS['panel'], anchor='nw',
        pady=0,
    ).pack(side='left', anchor='nw')

    # vocals (2-stem)
    voc_row = tk.Frame(right_top, bg=COLORS['panel'])
    voc_row.pack(fill='x', anchor='nw', pady=(6, 0))
    _legend_chip(voc_row, ABOUT_LEGEND_2WAY[1][0]).pack(side='left', anchor='nw', padx=(ABOUT_COL_GAP, 12))
    tk.Label(
        voc_row, text=ABOUT_LEGEND_2WAY[1][1], justify='left', wraplength=ABOUT_LEGEND_WRAP,
        font=ABOUT_LEGEND_FONT, fg=COLORS['fg_dim'], bg=COLORS['panel'], anchor='nw',
        pady=0,
    ).pack(side='left', anchor='nw')

    # Row 2: divider above 4-stem
    grid.rowconfigure(2, minsize=0, weight=0)
    tk.Frame(grid, bg=COLORS['border'], height=1).grid(
        row=2, column=0, columnspan=4, sticky='ew',
        pady=(20, 10),
    )

    # Row 3: Left: 4-stem header | Right: (empty)
    grid.rowconfigure(
        3,
        minsize=ABOUT_LEGEND_SECTION_H + ABOUT_LEGEND_SECTION_BOTTOM,
        weight=0,
    )
    tk.Label(
        grid, text='4-stem',
        font=ABOUT_LEGEND_SUB_FONT, fg=COLORS['fg'], bg=COLORS['panel'], anchor='w',
    ).grid(
        row=3, column=0, columnspan=2, sticky='sw',
        pady=(0, ABOUT_LEGEND_SECTION_BOTTOM),
    )

    # Row 4: Left: bass (4-stem) | Right: other (4-stem)
    _legend_grid_row(grid, 4, 0, ABOUT_LEGEND_4WAY[0][1], label=ABOUT_LEGEND_4WAY[0][0])
    _legend_grid_row(
        grid, 4, 2, ABOUT_LEGEND_4WAY[2][1], label=ABOUT_LEGEND_4WAY[2][0],
        padx_chip=(ABOUT_COL_GAP, 12),
    )

    # Row 5: Left: drums (4-stem) | Right: vocals (4-stem)
    _legend_grid_row(grid, 5, 0, ABOUT_LEGEND_4WAY[1][1], label=ABOUT_LEGEND_4WAY[1][0], minsize=0)
    _legend_grid_row(
        grid, 5, 2, ABOUT_LEGEND_4WAY[3][1], label=ABOUT_LEGEND_4WAY[3][0],
        padx_chip=(ABOUT_COL_GAP, 12), minsize=0,
    )



def _build_about_how_it_works(parent: tk.Misc) -> tk.Text:
    body = tk.Text(
        parent, wrap='word', width=1, height=1,
        font=ABOUT_BODY_FONT, fg=COLORS['fg_dim'], bg=COLORS['panel'],
        relief='flat', highlightthickness=0, bd=0,
        padx=0, pady=0, cursor='arrow', spacing1=2, spacing2=0, spacing3=1,
    )
    body.bind('<Key>', lambda _e: 'break')

    def add_link(text: str, url: str) -> None:
        tag = f'link_{id(text)}'
        body.insert('end', text, tag)
        body.tag_configure(tag, foreground=COLORS['accent'], underline=True)
        body.tag_bind(
            tag, '<Enter>',
            lambda _e, t=tag: (
                body.configure(cursor='hand2'),
                body.tag_configure(t, foreground=COLORS['accent_hov']),
            ),
        )
        body.tag_bind(
            tag, '<Leave>',
            lambda _e, t=tag: (
                body.configure(cursor='arrow'),
                body.tag_configure(t, foreground=COLORS['accent']),
            ),
        )
        body.tag_bind(tag, '<Button-1>', lambda _e, u=url: webbrowser.open(u))

    body.insert(
        'end',
        'A GUI tool for organizing messy stem collections using Demucs-powered '
        'classification (handy for ',
    )
    add_link('datasets type 1/2/4', DATASET_TYPES_URL)
    body.insert('end', ').\n')
    for bullet in ABOUT_BULLETS:
        body.insert('end', f' • {bullet}\n')
    body.insert(
        'end',
        '\nIdeal for organizing ripped stems, unsorted libraries, or building training '
        'datasets — without having to audition everything manually.\n',
    )
    body.insert('end', ABOUT_HOW_IT_WORKS_TAIL)
    return body


def _build_about_sdr(parent: tk.Misc) -> tk.Text:
    body = tk.Text(
        parent, wrap='word', width=1, height=1,
        font=ABOUT_BODY_FONT, fg=COLORS['fg_dim'], bg=COLORS['panel'],
        relief='flat', highlightthickness=0, bd=0,
        padx=0, pady=0, cursor='arrow', spacing1=2, spacing2=0, spacing3=1,
    )
    body.bind('<Key>', lambda _e: 'break')

    def add_link(text: str, url: str) -> None:
        tag = f'link_{id(text)}'
        body.insert('end', text, tag)
        body.tag_configure(tag, foreground=COLORS['accent'], underline=True)
        body.tag_bind(
            tag, '<Enter>',
            lambda _e, t=tag: (
                body.configure(cursor='hand2'),
                body.tag_configure(t, foreground=COLORS['accent_hov']),
            ),
        )
        body.tag_bind(
            tag, '<Leave>',
            lambda _e, t=tag: (
                body.configure(cursor='arrow'),
                body.tag_configure(t, foreground=COLORS['accent']),
            ),
        )
        body.tag_bind(tag, '<Button-1>', lambda _e, u=url: webbrowser.open(u))

    body.insert(
        'end',
        'After organizing stems (or on an existing library), you can filter out '
        'low-quality results using scale-invariant SDR (',
    )
    add_link('SI-SDR', SI_SDR_URL)
    body.insert('end', '):\n')
    for bullet in ABOUT_SDR_BULLETS:
        body.insert('end', f' • {bullet}\n')
    body.insert('end', '\nTwo input layouts are supported:\n')
    for i, (label, url, suffix) in enumerate(ABOUT_SDR_LAYOUTS):
        body.insert('end', ' • ')
        add_link(label, url)
        if i < len(ABOUT_SDR_LAYOUTS) - 1:
            body.insert('end', f'{suffix}\n')
        else:
            body.insert('end', suffix)
    return body


def _about_divider(parent: tk.Misc, *, pady: tuple[int, int] = (0, 12)) -> None:
    tk.Frame(parent, bg=COLORS['border'], height=1).pack(fill='x', pady=pady)


def _about_header_label(parent: tk.Misc, text: str, *, font: tuple, fg: str, pady: tuple[int, int] = (0, 0)) -> None:
    tk.Label(
        parent, text=text, font=font, fg=fg, bg=COLORS['panel'], anchor='center',
    ).pack(anchor='center', pady=pady)


def show_about_dialog(parent: tk.Tk, icon: tk.PhotoImage | None = None) -> None:
    dlg = tk.Toplevel(parent)
    dlg.title('About STEM organizer')
    dlg.configure(bg=COLORS['panel'])
    apply_window_icon(dlg)
    dlg.resizable(False, False)
    dlg.transient(parent)

    outer = tk.Frame(dlg, bg=COLORS['panel'])
    outer.pack(fill='both', expand=True, padx=ABOUT_PAD_X, pady=(4, 6))

    header = tk.Frame(outer, bg=COLORS['panel'])
    header.pack(fill='x', pady=(0, 2))

    if icon is not None:
        tk.Label(header, image=icon, bg=COLORS['panel']).pack(anchor='center', pady=(0, 0))
        dlg._about_icon = icon  # keep reference

    _about_header_label(
        header, 'by Gilliaan & Bas Curtiz',
        font=ABOUT_BODY_FONT, fg=COLORS['fg_dim'], pady=(0, 0),
    )
    _about_header_label(
        header, f'Version {APP_VERSION}',
        font=ABOUT_LEGEND_FONT, fg=COLORS['fg_dim'], pady=(0, 2),
    )

    link = tk.Label(
        header, text='View on GitHub',
        font=ABOUT_BODY_FONT, fg=COLORS['accent'], bg=COLORS['panel'],
        cursor='hand2',
    )
    link.pack(anchor='center', pady=(0, 2))
    link.bind('<Button-1>', lambda _e: webbrowser.open(STATUS_LINK_URL))
    link.bind(
        '<Enter>',
        lambda _e: link.configure(fg=COLORS['accent_hov'], font=ABOUT_BODY_FONT + ('underline',)),
    )
    link.bind(
        '<Leave>',
        lambda _e: link.configure(fg=COLORS['accent'], font=ABOUT_BODY_FONT),
    )

    _about_divider(outer, pady=(4, 8))

    # How it works (Full width)
    tk.Label(
        outer, text='How it works',
        font=ABOUT_SECTION_FONT, fg=COLORS['fg'], bg=COLORS['panel'], anchor='w',
    ).pack(fill='x', pady=(0, 2))
    how = _build_about_how_it_works(outer)
    how.pack(fill='x', pady=(0, 0))
    how.configure(width=max(60, ABOUT_FULL_W // 7))

    _about_divider(outer, pady=(4, 8))

    # Stem legend (2 columns inside)
    _build_about_legend(outer)

    _about_divider(outer, pady=(8, 6))

    # (Optional) Calculate SI-SDR (Full width)
    tk.Label(
        outer, text='(Optional) Calculate SI-SDR',
        font=ABOUT_SECTION_FONT, fg=COLORS['fg'], bg=COLORS['panel'], anchor='w',
    ).pack(fill='x', pady=(0, 2))
    sdr = _build_about_sdr(outer)
    sdr.pack(fill='x', pady=(0, 0))
    sdr.configure(width=max(60, ABOUT_FULL_W // 7))
    dlg.geometry(f'{ABOUT_DIALOG_W}x{ABOUT_DIALOG_H}')
    dlg.update_idletasks()

    _fit_about_text_height(how)
    _fit_about_text_height(sdr)
    outer.update_idletasks()
    dlg.update_idletasks()

    dialog_h = _about_toplevel_height(dlg, outer, ABOUT_DIALOG_W)
    # Center on parent window — same as Match / Genre / Rename help dialogs.
    top = parent.winfo_toplevel()
    top.update_idletasks()
    width = ABOUT_DIALOG_W
    x = top.winfo_rootx() + max(0, (top.winfo_width() - width) // 2)
    y = top.winfo_rooty() + max(0, (top.winfo_height() - dialog_h) // 2)
    dlg.geometry(f'{width}x{dialog_h}+{x}+{y}')
    dlg.update_idletasks()

    # Same dark title bar / DWM chrome as Match / Genre / Rename help.
    _win_apply_dwm_rounded_corners(dlg)
    dlg.after(20, lambda: _win_apply_dwm_rounded_corners(dlg))

    dlg.grab_set()
    dlg.focus_force()
    dlg.bind('<Escape>', lambda _e: dlg.destroy())
    dlg.protocol('WM_DELETE_WINDOW', dlg.destroy)
    parent.wait_window(dlg)


def _about_toplevel_height(dlg: tk.Toplevel, outer: tk.Frame, width: int) -> int:
    """Window height that fits outer content plus non-client chrome."""
    outer.update_idletasks()
    dlg.update_idletasks()
    content_h = outer.winfo_reqheight()
    dlg.geometry(f'{width}x{content_h}')
    dlg.update_idletasks()
    chrome = dlg.winfo_height() - content_h
    if chrome <= 0:
        chrome = 39
    return min(ABOUT_DIALOG_H, max(520, content_h + chrome))


def _fit_about_text_height(text: tk.Text) -> None:
    """Size Text height to wrapped display lines so nothing is clipped."""
    text.update_idletasks()
    # height=1 leaves the widget 1px wide; display-line count then explodes.
    text.configure(height=max(int(text.cget('height')), 8))
    text.winfo_toplevel().update_idletasks()
    text.update_idletasks()
    try:
        lines = int(text.tk.call(text._w, 'count', '-displaylines', '1.0', 'end'))
        if text.get('end-1c') == '\n' and lines > 1:
            lines -= 1
        text.configure(height=max(lines + 1, 1))
    except tk.TclError:
        text.configure(height=18)


class InfoIcon(tk.Canvas):
    """Small (?) icon — dim by default, full opacity on hover."""

    def __init__(self, master: tk.Misc, on_click, **kw):
        super().__init__(
            master,
            width=INFO_ICON_SIZE,
            height=INFO_ICON_SIZE,
            highlightthickness=0,
            bd=0,
            bg=COLORS['bg'],
            cursor='hand2',
            **kw,
        )
        self._on_click = on_click
        self._opacity = INFO_ICON_OPACITY_DIM
        self._redraw()
        self.bind('<Enter>', self._enter, add='+')
        self.bind('<Leave>', self._leave, add='+')
        self.bind('<Button-1>', self._click, add='+')
        Tooltip(self, 'Show more info/help.')

    def _color(self) -> str:
        # Match tab description dim color; brighten to fg on hover.
        if self._opacity >= INFO_ICON_OPACITY_FULL:
            return COLORS['fg']
        return HEADER_DESC_COLOR

    def _redraw(self) -> None:
        self.delete('all')
        c = self._color()
        s = INFO_ICON_SIZE
        pad = 1.5
        self.create_oval(pad, pad, s - pad, s - pad, outline=c, width=1)
        self.create_text(
            s / 2 + INFO_ICON_CX_NUDGE,
            s / 2 + INFO_ICON_CY_NUDGE,
            text='?',
            fill=c,
            font=INFO_ICON_FONT,
            anchor='center',
        )

    def _enter(self, _e=None) -> None:
        self._opacity = INFO_ICON_OPACITY_FULL
        self._redraw()

    def _leave(self, _e=None) -> None:
        self._opacity = INFO_ICON_OPACITY_DIM
        self._redraw()

    def _click(self, _e=None) -> None:
        self._on_click()


def _photoimage_from_ppm(root: tk.Misc, ppm: bytes) -> tk.PhotoImage:
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as tmp:
        tmp.write(ppm)
        tmp_path = tmp.name
    return tk.PhotoImage(master=root, file=tmp_path)


def _load_ico_bmp_photoimage(
    root: tk.Misc, data: bytes, size: int, bg: tuple[int, int, int],
) -> tk.PhotoImage | None:
    count = int.from_bytes(data[4:6], 'little')
    best_offset = None
    best_w = 0
    best_diff = 10**9
    for i in range(count):
        entry = data[6 + i * 16: 6 + (i + 1) * 16]
        w = entry[0] or 256
        h = entry[1] or 256
        diff = abs(w - size) + abs(h - size)
        if diff < best_diff:
            best_diff = diff
            best_w = w
            best_offset = int.from_bytes(entry[12:16], 'little')
    if best_offset is None:
        return None

    img = data[best_offset:]
    if len(img) < 40 or img[:4] != b'\x28\x00\x00\x00':
        return None
    bi_width = int.from_bytes(img[4:8], 'little', signed=True)
    bi_height = int.from_bytes(img[8:12], 'little', signed=True)
    bi_bit_count = int.from_bytes(img[14:16], 'little')
    if bi_bit_count != 32:
        return None

    h = bi_height // 2
    w = bi_width
    header_size = int.from_bytes(img[0:4], 'little')
    row_bytes = ((w * 4 + 3) // 4) * 4
    pixels = bytearray()
    for y in range(h):
        row_off = header_size + (h - 1 - y) * row_bytes
        for x in range(w):
            b = img[row_off + x * 4]
            g = img[row_off + x * 4 + 1]
            r = img[row_off + x * 4 + 2]
            a = img[row_off + x * 4 + 3]
            af = a / 255.0
            if a == 0:
                pixels.extend(bg)
            else:
                pixels.extend(
                    int(bg[i] * (1 - af) + c * af)
                    for i, c in enumerate((r, g, b))
                )
    ppm = f'P6\n{w} {h}\n255\n'.encode('ascii') + bytes(pixels)
    photo = _photoimage_from_ppm(root, ppm)
    if best_w > size:
        factor = max(1, round(best_w / size))
        photo = photo.subsample(factor, factor)
    return photo


def load_ico_photoimage(
    root: tk.Misc, path: Path, size: int = 16,
    bg: tuple[int, int, int] | None = None,
) -> tk.PhotoImage | None:
    """Load an .ico for use in Tk widgets (PNG or BMP payloads)."""
    if not path.exists():
        return None
    bg = bg or _rgb_from_hex(COLORS['panel'])
    try:
        data = path.read_bytes()
        if data[:4] != b'\x00\x00\x01\x00':
            return None
        count = int.from_bytes(data[4:6], 'little')
        best_png = None
        best_diff = 10**9
        best_w = size
        header = 6
        for i in range(count):
            entry = data[header + i * 16: header + (i + 1) * 16]
            w = entry[0] or 256
            h = entry[1] or 256
            nbytes = int.from_bytes(entry[8:12], 'little')
            offset = int.from_bytes(entry[12:16], 'little')
            payload = data[offset:offset + nbytes]
            if not payload.startswith(b'\x89PNG\r\n\x1a\n'):
                continue
            diff = abs(w - size) + abs(h - size)
            if diff < best_diff:
                best_diff = diff
                best_png = payload
                best_w = w
        if best_png is not None:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(best_png)
                tmp_path = tmp.name
            photo = tk.PhotoImage(master=root, file=tmp_path)
            if best_w > size:
                factor = max(1, round(best_w / size))
                photo = photo.subsample(factor, factor)
            return photo

        photo = _load_ico_bmp_photoimage(root, data, size, bg)
        if photo is not None:
            return photo
    except Exception:
        pass

    if LOGO_PNG.exists():
        try:
            return scale_photoimage(tk.PhotoImage(master=root, file=str(LOGO_PNG)), size)
        except Exception:
            pass
    return None


def apply_window_icon(win: tk.Misc) -> tk.PhotoImage | None:
    """Set title-bar / taskbar icon from logo.ico."""
    photo = load_ico_photoimage(
        win, ICON_ICO, TITLE_ICON_SIZE, bg=_rgb_from_hex(COLORS['panel']),
    )
    if photo is not None:
        try:
            taskbar = load_ico_photoimage(
                win, ICON_ICO, 22, bg=_rgb_from_hex(COLORS['panel']),
            )
            win.iconphoto(True, taskbar or photo)
        except Exception:
            pass
    if ICON_ICO.exists():
        try:
            win.iconbitmap(str(ICON_ICO))
        except Exception:
            pass
    return photo


def apply_app_icon(root: tk.Tk) -> tk.PhotoImage | None:
    return apply_window_icon(root)


def scale_photoimage(photo: tk.PhotoImage, target: int) -> tk.PhotoImage:
    w, h = photo.width(), photo.height()
    if w == target and h == target:
        return photo
    out = photo
    w, h = out.width(), out.height()
    if w < target:
        z = max(1, round(target / w))
        out = out.zoom(z, z)
        w = out.width()
    if w > target:
        s = max(1, round(w / target))
        out = out.subsample(s, s)
    return out


def load_about_icon(root: tk.Misc) -> tk.PhotoImage | None:
    return load_ico_photoimage(
        root, ICON_ICO, ABOUT_ICON_SIZE, bg=_rgb_from_hex(COLORS['panel']),
    )


def load_splash_photoimage(root: tk.Misc) -> tk.PhotoImage | None:
    try:
        if LOGO_PNG.exists():
            photo = tk.PhotoImage(master=root, file=str(LOGO_PNG))
            if photo.width() != SPLASH_SIZE or photo.height() != SPLASH_SIZE:
                photo = scale_photoimage(photo, SPLASH_SIZE)
            return photo
        photo = load_ico_photoimage(root, ICON_ICO, 256)
        if photo is not None:
            return scale_photoimage(photo, SPLASH_SIZE)
    except Exception:
        pass
    return None


def show_splash_screen(on_done, *, run_startup=None) -> None:
    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(True)
    transparent = False
    startup_state = {'done': False, 'error': None, 'cancelled': False}

    def set_status(msg: str) -> None:
        if startup_state['cancelled']:
            return

        def _apply() -> None:
            if status_lbl.winfo_exists():
                status_lbl.configure(text=msg)

        root.after(0, _apply)

    def _run_startup_thread() -> None:
        try:
            if run_startup is not None:
                run_startup(set_status)
        except Exception as exc:
            startup_state['error'] = exc
        finally:
            startup_state['done'] = True

    try:
        root.wm_attributes('-topmost', True)
    except tk.TclError:
        pass

    photo = load_splash_photoimage(root)
    if photo is None:
        root.destroy()
        if run_startup is not None:
            run_startup(lambda _msg: None)
        on_done(startup_state['error'])
        return

    if sys.platform == 'win32':
        root.configure(bg=SPLASH_CHROMA)
        try:
            root.wm_attributes('-transparentcolor', SPLASH_CHROMA)
            transparent = True
        except tk.TclError:
            root.configure(bg=COLORS['bg'])
    else:
        root.configure(bg=COLORS['bg'])

    win_w = photo.width()
    win_h = photo.height() + SPLASH_STATUS_PAD_Y
    if not transparent:
        win_w += SPLASH_PAD * 2 + 2
        win_h += SPLASH_PAD * 2 + 2
    x, y, win_w, win_h = _place_window_centered(root, win_w, win_h)
    root.geometry(f'{win_w}x{win_h}+{x}+{y}')

    content_bg = SPLASH_CHROMA if transparent else COLORS['panel']
    shell = tk.Frame(root, bg=content_bg, bd=0, highlightthickness=0)
    shell.pack(fill='both', expand=True)

    if transparent:
        tk.Label(
            shell, image=photo, bg=SPLASH_CHROMA, bd=0, highlightthickness=0,
        ).pack()
        strip_bg = SPLASH_CHROMA
    else:
        outer = tk.Frame(shell, bg=COLORS['border'], bd=0, highlightthickness=0)
        outer.pack(fill='both', expand=True, padx=1, pady=1)
        card = tk.Frame(outer, bg=COLORS['panel'], bd=0, highlightthickness=0)
        card.pack(fill='both', expand=True)
        tk.Label(card, image=photo, bg=COLORS['panel'], bd=0).pack(
            padx=SPLASH_PAD, pady=(SPLASH_PAD, 4),
        )
        strip_bg = COLORS['panel']

    status_strip = tk.Frame(shell, bg=strip_bg, bd=0, highlightthickness=0)
    status_strip.pack(side='bottom', pady=(SPLASH_STATUS_GAP, 8))

    status_lbl = tk.Label(
        status_strip, text='Starting…',
        bg=SPLASH_STATUS_BG, fg=SPLASH_STATUS_COLOR,
        font=SPLASH_STATUS_FONT, justify='center',
        padx=14, pady=4,
    )
    status_lbl.pack()
    root._splash_photo = photo

    root.update_idletasks()
    needed_h = photo.height() + SPLASH_STATUS_GAP + status_lbl.winfo_reqheight() + 8
    if needed_h > win_h:
        win_h = needed_h
        x, y, win_w, win_h = _place_window_centered(root, win_w, win_h)
        root.geometry(f'{win_w}x{win_h}+{x}+{y}')

    alpha_ok = True
    try:
        root.wm_attributes('-alpha', 0.0)
    except tk.TclError:
        alpha_ok = False

    root.deiconify()
    root.update_idletasks()

    if run_startup is not None:
        threading.Thread(target=_run_startup_thread, daemon=True).start()

    min_visible_until = time.monotonic() + (SPLASH_HOLD_MS / 1000.0)

    def finish():
        startup_state['cancelled'] = True
        err = startup_state['error']
        try:
            root.destroy()
        except tk.TclError:
            pass
        on_done(err)

    def try_finish():
        if not startup_state['done']:
            root.after(80, try_finish)
            return
        if time.monotonic() < min_visible_until:
            root.after(80, try_finish)
            return
        fade_out()

    def fade_out(step=0):
        if not alpha_ok:
            finish()
            return
        step += 1
        alpha = max(0.0, 1.0 - step / 12)
        root.wm_attributes('-alpha', alpha)
        if alpha > 0:
            root.after(20, lambda: fade_out(step))
        else:
            finish()

    def fade_in(step=0):
        if not alpha_ok:
            root.after(SPLASH_HOLD_MS, try_finish)
            return
        step += 1
        alpha = min(1.0, step / 16)
        root.wm_attributes('-alpha', alpha)
        if alpha < 1.0:
            root.after(24, lambda: fade_in(step))
        else:
            try_finish()

    root.after(40, fade_in)
    root.mainloop()


TITLE_BAR_HEIGHT = 36
TITLE_ICON_SIZE = 22
TITLE_BAR_CONTENT_PAD_Y = 5
RESIZE_BORDER = 6
# Outer window clip radius (custom title bar / overrideredirect).
WINDOW_CORNER_RADIUS = 12
_USE_CUSTOM_TITLE_BAR = sys.platform == 'win32'
WIN_DEFAULT_W = 1200
WIN_DEFAULT_H = 1018
WIN_MIN_W = 1000
WIN_MIN_H = 720
CONTENT_PAD = 18
CONTENT_PAD_Y = 10
HEADER_TOP_PAD = 4
ACTIONS_BOTTOM_PAD = 12
# Extra air between dark tab panels and the action button row.
# Keeps Align (tallest) from sitting flush against Play / Start / etc.
ACTIONS_TOP_GAP = 20
SECTION_INNER_PAD = 10
CLASS_FRAME_PAD = (SECTION_INNER_PAD, 3, SECTION_INNER_PAD, SECTION_INNER_PAD)
CLASS_TAB_CONTENT_PAD = (0, 3)
SECTION_GAP = 8
SECTION_SIDE_PAD_LEFT = 14
SECTION_SIDE_PAD_RIGHT = 4
LOG_PAD_BOTTOM = ACTIONS_BOTTOM_PAD
LOG_INNER_PAD = 14
SECTION_PADX = (SECTION_SIDE_PAD_LEFT, SECTION_SIDE_PAD_RIGHT)


def _win_toplevel_hwnd(root: tk.Misc, *, flush: bool = False) -> int:
    if flush:
        root.update_idletasks()
    user32 = ctypes.windll.user32
    wid = int(root.winfo_id())
    return int(user32.GetParent(wid) or wid)


def _win_window_rect(root: tk.Misc) -> tuple[int, int, int, int] | None:
    """Native outer window bounds as (x, y, width, height)."""
    try:
        user32 = ctypes.windll.user32
        hwnd = _win_toplevel_hwnd(root, flush=True)
        rect = _WIN_RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return (
                int(rect.left),
                int(rect.top),
                int(rect.right - rect.left),
                int(rect.bottom - rect.top),
            )
    except Exception:
        pass
    return None


def _clamp_window_bounds(
    root: tk.Misc, x: int, y: int, w: int, h: int,
) -> tuple[int, int, int, int]:
    w = max(WIN_MIN_W, int(w))
    h = max(WIN_MIN_H, int(h))
    ax, ay, aw, ah = _placement_bounds(root)
    w = min(w, max(1, aw))
    h = min(h, max(1, ah))
    x = max(ax, min(int(x), ax + aw - w))
    y = max(ay, min(int(y), ay + ah - h))
    return x, y, w, h


class _WIN_RECT(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_long),
        ('top', ctypes.c_long),
        ('right', ctypes.c_long),
        ('bottom', ctypes.c_long),
    ]


class _WIN_MONITORINFO(ctypes.Structure):
    _fields_ = [
        ('cbSize', ctypes.c_ulong),
        ('rcMonitor', _WIN_RECT),
        ('rcWork', _WIN_RECT),
        ('dwFlags', ctypes.c_ulong),
    ]


def _win_work_area(root: tk.Misc) -> tuple[int, int, int, int]:
    """Monitor work area (excludes taskbar) as (x, y, width, height)."""
    try:
        user32 = ctypes.windll.user32
        hwnd = _win_toplevel_hwnd(root)
        hmon = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        mi = _WIN_MONITORINFO()
        mi.cbSize = ctypes.sizeof(_WIN_MONITORINFO)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            work = mi.rcWork
            return (
                int(work.left),
                int(work.top),
                int(work.right - work.left),
                int(work.bottom - work.top),
            )
    except Exception:
        pass
    return 0, 0, int(root.winfo_screenwidth()), int(root.winfo_screenheight())


def _placement_bounds(root: tk.Misc) -> tuple[int, int, int, int]:
    """Area available for window placement as (x, y, width, height)."""
    if sys.platform == 'win32':
        return _win_work_area(root)
    return 0, 0, int(root.winfo_screenwidth()), int(root.winfo_screenheight())


def _place_window_centered(root: tk.Misc, width: int, height: int) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) centered in the usable screen area."""
    ax, ay, aw, ah = _placement_bounds(root)
    w = min(max(1, width), max(1, aw))
    h = min(max(1, height), max(1, ah))
    x = ax + max(0, (aw - w) // 2)
    y = ay + max(0, (ah - h) // 2)
    return x, y, w, h


def _win_move_resize(root: tk.Misc, x: int, y: int, w: int, h: int) -> bool:
    try:
        user32 = ctypes.windll.user32
        hwnd = _win_toplevel_hwnd(root)
        SWP_NOZORDER = 0x0004
        SWP_NOACTIVATE = 0x0010
        return bool(
            user32.SetWindowPos(
                hwnd, None, int(x), int(y), int(w), int(h),
                SWP_NOZORDER | SWP_NOACTIVATE,
            )
        )
    except Exception:
        return False


def _sync_tk_geometry(root: tk.Misc, x: int, y: int, w: int, h: int) -> None:
    try:
        root.geometry(f'{w}x{h}+{x}+{y}')
    except tk.TclError:
        pass


def _win_show_window(root: tk.Misc, cmd: int) -> bool:
    """ShowWindow codes for custom title bar (overrideredirect blocks iconify)."""
    try:
        user32 = ctypes.windll.user32
        return bool(user32.ShowWindow(_win_toplevel_hwnd(root), cmd))
    except Exception:
        return False


def apply_native_window_frame(root: tk.Misc) -> None:
    """Keep taskbar entry after overrideredirect(True); resize uses manual edge drag."""
    if not _USE_CUSTOM_TITLE_BAR:
        return
    try:
        GWL_EXSTYLE = -20
        WS_EX_APPWINDOW = 0x00040000
        WS_EX_TOOLWINDOW = 0x00000080

        user32 = ctypes.windll.user32
        hwnd = _win_toplevel_hwnd(root, flush=True)

        root.configure(highlightthickness=0, bd=0)

        ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ex = (ex | WS_EX_APPWINDOW) & ~WS_EX_TOOLWINDOW
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)
    except Exception:
        pass


def _win_colorref_from_hex(hex_color: str) -> int:
    """COLORREF 0x00BBGGRR from #RRGGBB."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b << 16) | (g << 8) | r


def _win_apply_rounded_corners(
    root: tk.Misc,
    *,
    maximized: bool = False,
    radius: int = WINDOW_CORNER_RADIUS,
) -> None:
    """Clip frameless (overrideredirect) window via SetWindowRgn.

    Skips no-op updates — SetWindowRgn loops make Windows show the busy cursor.
    Do not use this on normal decorated Toplevels (help / STEM player) — that
    leaves a white square frame. Those use `_win_apply_dwm_rounded_corners`.
    """
    if sys.platform != 'win32':
        return
    if getattr(root, '_rounding_corners', False):
        return
    try:
        w = max(int(root.winfo_width()), 1)
        h = max(int(root.winfo_height()), 1)
        if w < 2 or h < 2:
            return
        key = (w, h, bool(maximized), int(radius), 'rgn')
        if getattr(root, '_round_corner_key', None) == key:
            return

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        # No flush — update_idletasks here prolongs the busy cursor on startup.
        hwnd = _win_toplevel_hwnd(root, flush=False)
        root._rounding_corners = True  # type: ignore[attr-defined]
        try:
            if maximized or radius <= 0:
                user32.SetWindowRgn(hwnd, 0, True)
            else:
                # CreateRoundRectRgn: right/bottom exclusive; ellipse = 2*r.
                hrgn = gdi32.CreateRoundRectRgn(
                    0, 0, w + 1, h + 1, radius * 2, radius * 2,
                )
                if not hrgn:
                    return
                # On success the system owns hrgn — do not DeleteObject.
                if not user32.SetWindowRgn(hwnd, hrgn, True):
                    gdi32.DeleteObject(hrgn)
                    return
            root._round_corner_key = key  # type: ignore[attr-defined]
        finally:
            root._rounding_corners = False  # type: ignore[attr-defined]
    except Exception:
        try:
            root._rounding_corners = False  # type: ignore[attr-defined]
        except Exception:
            pass


def _win_apply_dwm_rounded_corners(
    root: tk.Misc,
    *,
    maximized: bool = False,
) -> None:
    """Native Win11 rounded corners + dark border for decorated Toplevels."""
    if sys.platform != 'win32':
        return
    try:
        user32 = ctypes.windll.user32
        dwmapi = ctypes.windll.dwmapi
        hwnd = _win_toplevel_hwnd(root, flush=False)
        # Drop any prior SetWindowRgn clip (causes white square halo on framed wins).
        user32.SetWindowRgn(hwnd, 0, True)

        # DWMWA_WINDOW_CORNER_PREFERENCE = 33
        # DWMWCP_DONOTROUND = 1, DWMWCP_ROUND = 2
        pref = ctypes.c_int(1 if maximized else 2)
        dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(pref), ctypes.sizeof(pref))

        # DWMWA_BORDER_COLOR = 34 — match app border (kills light/white frame).
        border = ctypes.c_uint(_win_colorref_from_hex(COLORS['border']))
        dwmapi.DwmSetWindowAttribute(
            hwnd, 34, ctypes.byref(border), ctypes.sizeof(border),
        )
        # DWMWA_CAPTION_COLOR = 35 — dark title bar on STEM player / help.
        caption = ctypes.c_uint(_win_colorref_from_hex(COLORS['panel']))
        dwmapi.DwmSetWindowAttribute(
            hwnd, 35, ctypes.byref(caption), ctypes.sizeof(caption),
        )
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        dark = ctypes.c_int(1)
        dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark),
        )
        root._round_corner_key = (bool(maximized), 'dwm')  # type: ignore[attr-defined]
    except Exception:
        pass


class DarkScrollbar(tk.Canvas):
    """Canvas-drawn scrollbar; native Windows scrollbars ignore dark theme colors."""

    WIDTH = 14

    def __init__(self, master, command=None, **kwargs):
        super().__init__(
            master, width=self.WIDTH, highlightthickness=0, borderwidth=0,
            background=COLORS['log_bg'], relief='flat', **kwargs,
        )
        self._command = command
        self._top = 0.0
        self._bottom = 1.0
        self._thumb_y0 = 0.0
        self._thumb_y1 = 0.0
        self._drag_offset = 0.0
        self._visible = False
        self.bind('<Configure>', lambda _e: self._redraw())
        self.bind('<Button-1>', self._press)
        self.bind('<B1-Motion>', self._drag)
        self.bind('<ButtonRelease-1>', lambda _e: None)

    def set(self, first, last):
        self._top = float(first)
        self._bottom = float(last)
        self._redraw()

    def _redraw(self):
        self.delete('all')
        height = max(int(self.winfo_height()), 1)
        if self._bottom - self._top >= 1.0:
            self._visible = False
            return
        self._visible = True
        y0 = height * self._top
        y1 = height * self._bottom
        min_thumb = 28
        if y1 - y0 < min_thumb:
            y1 = min(y0 + min_thumb, height)
        self._thumb_y0, self._thumb_y1 = y0, y1
        self.create_rectangle(
            2, y0, self.WIDTH - 2, y1,
            fill=COLORS['panel2'], outline=COLORS['border'], width=1,
        )

    def _press(self, event):
        if not self._visible:
            return
        if self._thumb_y0 <= event.y <= self._thumb_y1:
            self._drag_offset = event.y - self._thumb_y0
        else:
            self._drag_offset = (self._thumb_y1 - self._thumb_y0) / 2
            self._jump(event.y - self._drag_offset)

    def _drag(self, event):
        if self._visible:
            self._jump(event.y - self._drag_offset)

    def _jump(self, thumb_top):
        height = max(int(self.winfo_height()), 1)
        thumb_h = max(self._thumb_y1 - self._thumb_y0, 1)
        span = max(height - thumb_h, 1)
        frac = max(0.0, min(1.0, thumb_top / span))
        if self._command:
            self._command('moveto', frac)


def bind_mousewheel(widget: tk.Widget, yview) -> None:
    def on_wheel(event):
        yview('scroll', int(-event.delta / 120), 'units')
        return 'break'

    def on_enter(_event):
        widget.bind_all('<MouseWheel>', on_wheel)

    def on_leave(_event):
        widget.unbind_all('<MouseWheel>')

    widget.bind('<Enter>', on_enter, add='+')
    widget.bind('<Leave>', on_leave, add='+')


def _patch_hand_cursor_controls(*, include_ctk: bool = False) -> None:
    """Default hand cursor on radio/check/slider controls (ttk, tk, and optionally CTk).

    Do not import customtkinter during early theme setup — that can interact badly
    with the splash→main window handoff on Windows (withdrawn / invisible UI).
    """

    def _wrap(cls, name: str):
        if getattr(cls, "_stem_hand_cursor", False):
            return cls

        class Wrapped(cls):  # type: ignore[valid-type, misc]
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("cursor", "hand2")
                super().__init__(*args, **kwargs)

        Wrapped.__name__ = name
        Wrapped.__qualname__ = name
        Wrapped._stem_hand_cursor = True  # type: ignore[attr-defined]
        return Wrapped

    ttk.Radiobutton = _wrap(ttk.Radiobutton, "Radiobutton")  # type: ignore[misc, assignment]
    ttk.Checkbutton = _wrap(ttk.Checkbutton, "Checkbutton")  # type: ignore[misc, assignment]
    ttk.Scale = _wrap(ttk.Scale, "Scale")  # type: ignore[misc, assignment]
    tk.Radiobutton = _wrap(tk.Radiobutton, "Radiobutton")  # type: ignore[misc, assignment]
    tk.Checkbutton = _wrap(tk.Checkbutton, "Checkbutton")  # type: ignore[misc, assignment]
    tk.Scale = _wrap(tk.Scale, "Scale")  # type: ignore[misc, assignment]

    if not include_ctk:
        return

    try:
        import customtkinter as ctk
    except ImportError:
        return

    ctk.CTkCheckBox = _wrap(ctk.CTkCheckBox, "CTkCheckBox")  # type: ignore[misc, assignment]
    if hasattr(ctk, "CTkRadioButton"):
        ctk.CTkRadioButton = _wrap(ctk.CTkRadioButton, "CTkRadioButton")  # type: ignore[misc, assignment]
    if hasattr(ctk, "CTkSlider"):
        ctk.CTkSlider = _wrap(ctk.CTkSlider, "CTkSlider")  # type: ignore[misc, assignment]


def apply_theme(root: tk.Tk) -> None:
    _patch_hand_cursor_controls(include_ctk=False)
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass

    base  = ('Segoe UI', 10)
    bold  = ('Segoe UI Semibold', 10)
    section = ('Segoe UI Semibold', 8)
    title = ('Segoe UI Semibold', 16)
    C = COLORS
    select_active, select_inactive = _entry_select_colors()
    action_pad = (ACTION_BTN_PADX, ACTION_BTN_PADY)
    action_bw = 1
    action_pad_states = [
        ('disabled', action_pad), ('pressed', action_pad), ('active', action_pad),
        ('!disabled', action_pad),
    ]
    action_bw_states = [
        ('disabled', action_bw), ('pressed', action_bw), ('active', action_bw),
        ('!disabled', action_bw),
    ]

    root.configure(bg=C['bg'])
    root.option_add('*Font', base)
    root.option_add('*selectBackground', select_active)
    root.option_add('*selectForeground', C['fg'])
    root.option_add('*inactiveSelectBackground', select_inactive)

    style.configure('.', background=C['bg'], foreground=C['fg'],
                    fieldbackground=C['panel2'], bordercolor=C['border'],
                    lightcolor=C['panel'], darkcolor=C['panel'],
                    troughcolor=C['panel'], focuscolor=C['accent'])

    cfgs = {
        'TFrame':                    {'background': C['bg']},
        'Card.TFrame':               {'background': C['panel']},
        'TLabel':                    {'background': C['bg'], 'foreground': C['fg']},
        'Dim.TLabel':                {'background': C['bg'], 'foreground': C['fg_dim']},
        'Title.TLabel':              {'background': C['bg'], 'foreground': C['fg'], 'font': title},
        'Subtitle.TLabel':           {'background': C['bg'], 'foreground': C['fg_dim']},
        'Status.TLabel':             {'background': C['panel'], 'foreground': C['fg_dim'], 'padding': (10, 1)},
        'TLabelframe':               {'background': C['bg'], 'foreground': C['fg'],
                                      'bordercolor': C['border'], 'relief': 'solid', 'borderwidth': 1},
        'TLabelframe.Label':         {'background': C['bg'], 'foreground': C['fg_dim'], 'font': section},
        'TEntry':                    {'fieldbackground': C['panel2'], 'foreground': C['fg'],
                                      'bordercolor': C['border'], 'insertcolor': C['fg'],
                                      'selectbackground': select_active,
                                      'selectforeground': C['fg'],
                                      'padding': CTRL_FIELD_PAD},
        'TCombobox':                 {'fieldbackground': C['panel2'], 'background': C['panel2'],
                                      'foreground': C['fg'], 'arrowcolor': C['fg_dim'],
                                      'bordercolor': C['border'], 'padding': CTRL_FIELD_PAD,
                                      'selectbackground': select_active, 'selectforeground': C['fg'],
                                      'insertcolor': C['fg']},
        'TCheckbutton':              {
            'background': C['bg'], 'foreground': C['fg'],
            'focuscolor': C['bg'], 'indicatorcolor': C['panel2'],
            'indicatorbackground': C['panel2'], 'indicatorforeground': C['fg'],
        },
        'TRadiobutton':              {
            'background': C['bg'], 'foreground': C['fg'],
            'focuscolor': C['bg'],
            'indicatorbackground': C['panel2'], 'indicatorforeground': C['fg'],
        },
        'TButton':                   {'background': C['panel2'], 'foreground': C['fg'],
                                      'bordercolor': C['border'], 'padding': (14, 8), 'borderwidth': 1},
        'Path.TButton':              {'background': C['panel2'], 'foreground': C['fg'],
                                      'bordercolor': C['border'],
                                      'padding': (PATH_BTN_PADX, PATH_BTN_PADY), 'borderwidth': 1,
                                      'font': PATH_BTN_FONT},
        'Accent.TButton':            {'background': C['accent'], 'foreground': 'white',
                                      'bordercolor': C['accent'], 'padding': action_pad,
                                      'borderwidth': action_bw, 'font': bold, 'anchor': 'center'},
        'AccentMuted.TButton':       {'background': C['accent'], 'foreground': C['fg_dim'],
                                      'bordercolor': C['accent'], 'padding': action_pad,
                                      'borderwidth': action_bw, 'font': bold, 'anchor': 'center'},
        'Danger.TButton':            {'background': C['panel2'], 'foreground': C['danger'],
                                      'bordercolor': C['border'], 'padding': action_pad,
                                      'borderwidth': action_bw, 'font': bold, 'anchor': 'center'},
        'DangerMuted.TButton':       {'background': C['panel2'], 'foreground': C['fg_dim'],
                                      'bordercolor': C['border'], 'padding': action_pad,
                                      'borderwidth': action_bw, 'font': bold, 'anchor': 'center'},
        'Horizontal.TProgressbar':   {'background': C['accent'], 'troughcolor': C['panel2'],
                                      'bordercolor': C['panel2'],
                                      'lightcolor': C['accent'], 'darkcolor': C['accent']},
        'Status.Horizontal.TProgressbar': {
                                      'background': C['accent'], 'troughcolor': C['panel2'],
                                      'bordercolor': C['panel2'],
                                      'lightcolor': C['accent'], 'darkcolor': C['accent'],
                                      'thickness': 6},
        'Horizontal.TScale':         {'background': C['bg'], 'troughcolor': C['panel2'],
                                      'bordercolor': C['border'],
                                      'lightcolor': C['accent'], 'darkcolor': C['accent']},
        'TSpinbox':                  {'fieldbackground': C['panel2'], 'foreground': C['fg'],
                                      'background': C['panel2'], 'bordercolor': C['border'],
                                      'arrowcolor': C['fg_dim'], 'insertcolor': C['fg'],
                                      'selectbackground': select_active,
                                      'selectforeground': C['fg'],
                                      'padding': CTRL_FIELD_PAD,
                                      'font': ttk_ui_font()},
        'Vertical.TScrollbar':       {'background': C['panel2'], 'troughcolor': C['log_bg'],
                                      'bordercolor': C['border'], 'arrowcolor': C['fg_dim'],
                                      'darkcolor': C['panel'], 'lightcolor': C['panel2']},
        'Horizontal.TScrollbar':     {'background': C['panel2'], 'troughcolor': C['log_bg'],
                                      'bordercolor': C['border'], 'arrowcolor': C['fg_dim'],
                                      'darkcolor': C['panel'], 'lightcolor': C['panel2']},
        'Class.TNotebook':           {'background': C['bg'], 'borderwidth': 0,
                                      'tabmargins': [2, 0, 2, 0]},
        'Class.TNotebook.Tab':       {'background': C['panel2'], 'foreground': C['fg_dim'],
                                      'padding': (14, 3), 'font': bold, 'borderwidth': 1,
                                      'bordercolor': C['border'],
                                      'lightcolor': C['panel2'], 'darkcolor': C['panel2']},
        'Mode.TNotebook':            {'background': C['bg'], 'borderwidth': 0,
                                      'tabmargins': [2, 0, 2, 0],
                                      'bordercolor': C['bg'], 'lightcolor': C['bg'],
                                      'darkcolor': C['bg']},
        'Mode.TNotebook.Tab':        {'background': C['panel2'], 'foreground': C['fg_dim'],
                                      'padding': (14, 4), 'font': bold, 'borderwidth': 1,
                                      'bordercolor': C['panel2'],
                                      'lightcolor': C['panel2'], 'darkcolor': C['panel2']},
        'Sub.TNotebook':             {'background': C['bg'], 'borderwidth': 0,
                                      'tabmargins': [2, 0, 2, 0],
                                      'bordercolor': C['bg'], 'lightcolor': C['bg'],
                                      'darkcolor': C['bg']},
        'Sub.TNotebook.Tab':         {'background': C['panel2'], 'foreground': C['fg_dim'],
                                      'padding': (14, 4), 'font': bold, 'borderwidth': 1,
                                      'bordercolor': C['panel2'],
                                      'lightcolor': C['panel2'], 'darkcolor': C['panel2']},
    }
    for name, opts in cfgs.items():
        style.configure(name, **opts)

    _btn_layout = [
        ('Button.border', {'sticky': 'nswe', 'children': [
            ('Button.focus', {'sticky': 'nswe', 'children': [
                ('Button.padding', {'sticky': 'nswe', 'children': [
                    ('Button.label', {'sticky': 'nswe'}),
                ]}),
            ]}),
        ]}),
    ]
    for btn_style in ('TButton', 'Path.TButton'):
        style.layout(btn_style, _btn_layout)

    # clam wraps radio/check labels in a Focus element that paints a solid white
    # hover slab on Windows — drop that wrapper so hover stays readable.
    style.layout(
        'TRadiobutton',
        [
            (
                'Radiobutton.padding',
                {
                    'sticky': 'nswe',
                    'children': [
                        ('Radiobutton.indicator', {'side': 'left', 'sticky': ''}),
                        ('Radiobutton.label', {'side': 'left', 'sticky': 'nswe'}),
                    ],
                },
            )
        ],
    )
    style.layout(
        'TCheckbutton',
        [
            (
                'Checkbutton.padding',
                {
                    'sticky': 'nswe',
                    'children': [
                        ('Checkbutton.indicator', {'side': 'left', 'sticky': ''}),
                        ('Checkbutton.label', {'side': 'left', 'sticky': 'nswe'}),
                    ],
                },
            )
        ],
    )

    style.layout(
        'Mode.TNotebook.Tab',
        [('Notebook.tab', {
            'sticky': 'nswe',
            'children': [
                ('Notebook.padding', {
                    'side': 'top', 'sticky': 'nswe',
                    'children': [
                        ('Notebook.label', {'side': 'top', 'sticky': ''}),
                    ],
                }),
            ],
        })],
    )

    style.layout(
        'Sub.TNotebook.Tab',
        [('Notebook.tab', {
            'sticky': 'nswe',
            'children': [
                ('Notebook.padding', {
                    'side': 'top', 'sticky': 'nswe',
                    'children': [
                        ('Notebook.label', {'side': 'top', 'sticky': ''}),
                    ],
                }),
            ],
        })],
    )

    style.map('TEntry',
              bordercolor=[('focus', C['accent'])],
              selectbackground=[('focus', select_active), ('!focus', select_inactive)],
              selectforeground=[('focus', C['fg']), ('!focus', C['fg_dim'])])
    style.map('TSpinbox',
              bordercolor=[('focus', C['accent'])],
              selectbackground=[('focus', select_active), ('!focus', select_inactive)],
              selectforeground=[('focus', C['fg']), ('!focus', C['fg_dim'])])
    style.map('TCombobox',
              fieldbackground=[('readonly', C['panel2']), ('!disabled', C['panel2'])],
              background=[('readonly', C['panel2']), ('active', C['panel2'])],
              foreground=[('readonly', C['fg']), ('hover', C['fg']),
                          ('focus', C['fg']), ('active', C['fg'])],
              selectbackground=[('focus', select_active), ('!focus', select_inactive)],
              selectforeground=[('focus', C['fg']), ('!focus', C['fg_dim'])],
              arrowcolor=[('hover', C['fg']), ('active', C['fg'])],
              bordercolor=[('focus', C['accent'])])
    style.map('TCheckbutton',
              background=[
                  ('active', C['bg']), ('pressed', C['bg']),
                  ('selected', C['bg']), ('hover', C['bg']),
              ],
              foreground=[
                  ('disabled', C['fg_dim']), ('active', C['fg']),
                  ('hover', C['fg']), ('focus', C['fg']), ('selected', C['fg']),
              ],
              indicatorbackground=[
                  ('selected', C['accent']), ('pressed', C['accent']),
                  ('active', C['panel']), ('hover', C['panel']),
                  ('!disabled', C['panel2']),
              ],
              indicatorcolor=[
                  ('selected', C['accent']),
                  ('active', C['panel2']), ('hover', C['panel2']),
              ])
    style.map('TRadiobutton',
              background=[
                  ('active', C['bg']), ('pressed', C['bg']),
                  ('selected', C['bg']), ('hover', C['bg']),
              ],
              foreground=[
                  ('disabled', C['fg_dim']), ('active', C['fg']),
                  ('hover', C['fg']), ('focus', C['fg']), ('selected', C['fg']),
              ],
              indicatorbackground=[
                  ('selected', C['accent']), ('pressed', C['accent']),
                  ('active', C['panel']), ('hover', C['panel']),
                  ('!disabled', C['panel2']),
              ],
              indicatorcolor=[
                  ('selected', C['accent']),
                  ('active', C['panel2']), ('hover', C['panel2']),
              ])
    style.map('TButton',
              background=[('active', C['panel']), ('disabled', C['panel'])],
              foreground=[('disabled', C['fg_dim']), ('active', C['fg']), ('hover', C['fg'])])
    style.map('Path.TButton',
              background=[('active', C['panel']), ('disabled', C['panel'])],
              foreground=[('disabled', C['fg_dim']), ('active', C['fg']), ('hover', C['fg'])])
    style.map('Accent.TButton',
              background=[('active', C['accent_hov']), ('pressed', C['accent_hov'])],
              foreground=[('active', 'white'), ('!disabled', 'white')],
              padding=action_pad_states, borderwidth=action_bw_states)
    style.map('AccentMuted.TButton',
              background=[('active', C['accent_hov']), ('pressed', C['accent_hov'])],
              foreground=[('active', C['fg_dim']), ('!disabled', C['fg_dim'])],
              padding=action_pad_states, borderwidth=action_bw_states)
    style.map('Danger.TButton',
              background=[('active', C['panel'])],
              foreground=[('active', C['danger']), ('!disabled', C['danger'])],
              padding=action_pad_states, borderwidth=action_bw_states)
    style.map('DangerMuted.TButton',
              background=[('active', C['panel'])],
              foreground=[('active', C['fg_dim']), ('!disabled', C['fg_dim'])],
              padding=action_pad_states, borderwidth=action_bw_states)
    style.map('Vertical.TScrollbar',
              background=[('active', C['accent']), ('pressed', C['accent_hov'])],
              arrowcolor=[('active', C['fg']), ('pressed', C['fg'])])
    style.map('Horizontal.TScrollbar',
              background=[('active', C['accent']), ('pressed', C['accent_hov'])],
              arrowcolor=[('active', C['fg']), ('pressed', C['fg'])])
    style.map('Class.TNotebook.Tab',
              background=[('selected', C['bg']), ('active', C['panel']), ('!selected', C['panel2'])],
              foreground=[('selected', C['fg']), ('active', C['fg']), ('!selected', C['fg_dim'])],
              lightcolor=[('selected', C['bg']), ('active', C['panel']), ('!selected', C['panel2'])],
              darkcolor=[('selected', C['bg']), ('active', C['panel']), ('!selected', C['panel2'])],
              bordercolor=[('selected', C['border']), ('active', C['border'])],
              expand=[('selected', [1, 1, 1, 0])])
    style.map('Mode.TNotebook.Tab',
              background=[('selected', C['bg']), ('active', C['panel']), ('!selected', C['panel2'])],
              foreground=[('selected', C['fg']), ('active', C['fg']), ('!selected', C['fg_dim'])],
              lightcolor=[('selected', C['border']), ('active', C['border']),
                          ('!selected', C['panel2'])],
              darkcolor=[('selected', C['border']), ('active', C['border']),
                         ('!selected', C['panel2'])],
              bordercolor=[('selected', C['border']), ('active', C['border']),
                           ('!selected', C['panel2'])])
    style.map('Mode.TNotebook',
              bordercolor=[('active', C['bg']), ('!active', C['bg'])],
              lightcolor=[('active', C['bg']), ('!active', C['bg'])],
              darkcolor=[('active', C['bg']), ('!active', C['bg'])])
    style.map('Sub.TNotebook.Tab',
              background=[('selected', C['bg']), ('active', C['panel']), ('!selected', C['panel2'])],
              foreground=[('selected', C['fg']), ('active', C['fg']), ('!selected', C['fg_dim'])],
              lightcolor=[('selected', C['border']), ('active', C['border']),
                          ('!selected', C['panel2'])],
              darkcolor=[('selected', C['border']), ('active', C['border']),
                         ('!selected', C['panel2'])],
              bordercolor=[('selected', C['border']), ('active', C['border']),
                           ('!selected', C['panel2'])])
    style.map('Sub.TNotebook',
              bordercolor=[('active', C['bg']), ('!active', C['bg'])],
              lightcolor=[('active', C['bg']), ('!active', C['bg'])],
              darkcolor=[('active', C['bg']), ('!active', C['bg'])])

    for k, v in (('background', C['panel2']), ('foreground', C['fg']),
                 ('selectBackground', select_active), ('selectForeground', C['fg'])):
        root.option_add(f'*TCombobox*Listbox.{k}', v)


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str, delay: int = 550, wrap: int = 340):
        self.w, self.text, self.delay, self.wrap = widget, text, delay, wrap
        self._after = None
        self._tip = None
        widget.bind('<Enter>', self._schedule, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _schedule(self, _e=None):
        self._cancel()
        self._after = self.w.after(self.delay, self._show)

    def _cancel(self):
        if self._after is not None:
            try:
                self.w.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None

    def _show(self):
        if self._tip is not None:
            return
        try:
            x = self.w.winfo_rootx() + 14
            y = self.w.winfo_rooty() + self.w.winfo_height() + 6
        except tk.TclError:
            return
        tw = tk.Toplevel(self.w)
        tw.wm_overrideredirect(True)
        try:
            tw.wm_attributes('-topmost', True)
        except tk.TclError:
            pass
        tw.wm_geometry(f'+{x}+{y}')
        border = tk.Frame(tw, background=COLORS['border'])
        border.pack()
        tk.Label(border, text=self.text, justify='left',
                 background=COLORS['log_bg'], foreground=COLORS['log_fg'],
                 padx=10, pady=7, wraplength=self.wrap,
                 font=('Segoe UI', 9)).pack(padx=1, pady=1)
        self._tip = tw

    def _hide(self, _e=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None

    def set_text(self, text: str) -> None:
        self.text = text
        self._hide()


def tip(*widgets, text: str) -> None:
    for w in widgets:
        Tooltip(w, text)


TIPS = {
    'input':      "Root folder to scan.\nWith 'Each subfolder', every direct child folder is one song and all audio inside it is processed together.\nExample: E:\\Audio\\Multitracks\\0\\100 Gecs - 745 Sticky [wav]\\*.wav",
    'output':     "Where the grouped mixes are written. Subfolder names from the input are recreated here.\nExample: T:\\Multitracks 4-stem\\0\\100 Gecs - 745 Sticky [wav]\\drums.flac + bass.flac + …",
    'open_path':  'Open this folder in File Explorer.',
    'scan':       "How songs are discovered:\n• Each subfolder (one level) - one song per direct child of the input folder; audio is collected recursively inside each child.\n• Each leaf folder (recursive) - any nested folder that directly contains audio files is treated as its own song.",
    'cuda':       "Optional GPU acceleration. Uses CUDA when PyTorch supports your GPU; otherwise runs on CPU.\nRTX 50-series (5090, etc.) needs cu128 from install-deps.bat option 3.\nAuto-falls back to CPU on out-of-memory or incompatible builds.",
    'model':      "Demucs model used as the classifier.",
    'stems':      "Output category layout:\n• 2-way (instrumental/vocals) - non-vocal stems → instrumental.flac, vocal stems → vocals.flac\n• 4-way - bass / drums / other / vocals each get their own file.",
    'quality':    "Output file format:\n• FLAC 16-bit - lossless, CD quality, smallest\n• FLAC 24-bit - lossless, studio quality\n• WAV 16-bit - uncompressed PCM, CD quality\n• WAV 24-bit - uncompressed PCM, studio quality\n• WAV 32-bit float - uncompressed float, best for further processing\nFLAC uses ffmpeg compression level 12 when ffmpeg is on PATH.",
    'confidence': "Minimum share of total energy the dominant CATEGORY must reach for a stem to be accepted.\nExample at 40%: the winning category (vocals, or drums+bass+other combined in 2-stem mode) must hold ≥40% of total energy.",
    'margin':     "Minimum lead the dominant CATEGORY must have over the runner-up CATEGORY.\nIn 2-stem mode this measures vocals vs instrumental (drums+bass+other combined). In 4-way mode it measures the winning stem vs the next-loudest stem.\nA small margin means the stem is contaminated with content from another category.",
    'ambig':      "What to do when a stem is contaminated - i.e., more than one CATEGORY has significant energy (e.g., vocals mixed with instruments in 2-stem mode, or drums mixed with bass in 4-way mode):\n• Skip ambiguous stem only - drop just that stem, keep the rest.\n• Skip the entire song - abort this folder; no outputs are written.",
    'batch':      "Stems processed per GPU pass. Higher = faster, more VRAM.\nAuto-shrinks on out-of-memory.",
    'peak_norm':  "Apply a single gain to every category output so that, when summed back together, the mixture peaks at exactly -1 dBFS.\nDisable to keep raw summed levels (may clip).",
    'mixture':    "Also write 'mixture.wav' per folder - the sum of every accepted stem (skipped stems excluded).\nUseful for AI training datasets.\nAvailable only when output quality is a WAV format.",
    'dedup':      "Detect duplicate stems within each folder via phase-inversion null test.\nIf two stems cancel out when one is inverted (residual < 5% RMS), they're treated as the same content; only the one with the lowest peak dBFS is kept.\nUses GPU when CUDA is enabled and the card has at least 8 GiB VRAM; otherwise CPU.",
    'naming':     "Output folder naming:\n• Original folder name - keeps the input subfolder name exactly (e.g. '100 Gecs - 745 Sticky [wav]').\n• Folder name (simplified) - sanitizes names to a–z and 0–9 only.\n• Sequential - names folders song_0000, song_0001, … and continues past any existing numbered folders already in the output (no overwrite).\nIn sequential mode an 'index.json' is written at the output root, mapping each number to the original folder name so you can trace back later.",
    'duration':   "Append the output stem length to the folder name.\nBased on the shortest accepted stem (same length used for all output files).\nUses Windows-safe format: [3m12s], [18s], [1h03m12s].\nExample: '100 Gecs - 745 Sticky [wav] [3m11s]'.",
    'delete_short': "After export, move the output folder to the Recycle Bin if the stem length is shorter than the minimum.\nUseful for filtering broken or truncated multitracks.",
    'min_duration': "Minimum output length to keep. Folders shorter than this are recycled when the option above is enabled.",
    'delete_incomplete': "After export, move the output folder to the Recycle Bin if any expected stem file is missing.\nFor 4-way mode, all of bass, drums, other, and vocals must be present.\nFor 2-way (instrumental/vocals), both files must be present.",
    'skip_existing':    "Skip songs whose output folder already exists and contains stem files.\nMatches folder name with or without a duration suffix (e.g. 'Song [wav]' or 'Song [wav] [3m12s]').\nUseful for resuming a partially completed batch.",
    'start':      "Begin classifying and mixing. The UI stays responsive during the run.",
    'stop':       "Request a clean stop after the current folder finishes.",
    'save_log':   "Save the current log to a text file.",
    'clear_log':  "Clear the log panel for a fresh run.",
    'play_stems': "Open the stem preview player.\nLoad a folder with bass/drums/other/vocals or instrumental/vocals to audition mixes.",
    'start_sdr':  "Run SI-SDR quality check on organized stem folders.\nEach stem file is processed through Demucs individually and compared to the model output.\nSupports Type 1 (one folder per song) and Type 2 (one folder per stem category); layout is auto-detected.",
    'stop_sdr':   "Request a clean stop after the current folder finishes.",
    'save_sdr_log': "Save the current log to a text file.",
    'sdr_delete_folder': "After SI-SDR check, move the entire folder to the Recycle Bin if any stem falls below its threshold.\nWhen disabled, only the failing stem file is recycled.",
    'sdr_threshold': "Minimum SI-SDR (dB) for this stem. Files scoring below are moved to the Recycle Bin.",
    'status_link': 'View source code on GitHub',
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title('STEM organizer')
        x, y, w, h = _place_window_centered(self, WIN_DEFAULT_W, WIN_DEFAULT_H)
        self.geometry(f'{w}x{h}+{x}+{y}')
        self.minsize(WIN_MIN_W, WIN_MIN_H)

        self.input_dir    = tk.StringVar()
        self.output_dir   = tk.StringVar()
        self.use_cuda     = tk.BooleanVar(value=cuda_effective())
        self.model_label  = tk.StringVar(value=next(iter(MODELS)))
        self.stem_mode    = tk.StringVar(value=next(iter(STEM_MODES)))
        self.quality      = tk.StringVar(value='FLAC 16-bit')
        self.threshold    = tk.DoubleVar(value=0.40)
        self.min_margin   = tk.DoubleVar(value=0.20)
        self.batch_size   = tk.IntVar(value=4)
        self.peak_norm    = tk.BooleanVar(value=True)
        self.make_mixture = tk.BooleanVar(value=False)
        self.dedup        = tk.BooleanVar(value=False)
        self.ambig_label  = tk.StringVar(value=next(iter(AMBIG_MODES)))
        self.scan_label   = tk.StringVar(value=next(iter(SCAN_MODES)))
        self.naming_label = tk.StringVar(value=next(iter(NAMING_MODES)))
        self.append_duration = tk.BooleanVar(value=False)
        self.delete_if_short = tk.BooleanVar(value=True)
        self.min_duration_sec = tk.IntVar(value=8)
        self.delete_if_incomplete = tk.BooleanVar(value=True)
        self.skip_existing = tk.BooleanVar(value=True)
        self.sdr_delete_folder = tk.BooleanVar(value=True)
        self.hide_device_notice = False
        self.sdr_thresholds: dict[str, tk.IntVar] = {}
        self._class_tab = tk.StringVar(value='rms')
        self._sdr_use_output_dir = False
        self._worker_kind = 'rms'
        self._sdr_thresholds_settings: dict[str, int] = {}
        self._rms_saw_done = False
        self._pending_stem_block_gap = False
        self.status_var   = tk.StringVar(value='Idle')
        self.device_var   = tk.StringVar(value=self._device_status_text())
        self.elapsed_var = tk.StringVar(value='Elapsed: 0:00:00')
        self.eta_var = tk.StringVar(value='ETA --:--:--')
        self._progress_pct_value = 0.0
        self._progress_started_at = 0.0
        self._progress_tick_id: str | None = None
        self._resource_tick_id: str | None = None
        self._resource_monitor = None
        self._resource_visible = False
        self._stopping = False
        self._pair_busy = False
        self._rename_busy = False

        self.log_queue: queue.Queue = queue.Queue()
        self.worker = None
        self.sdr_worker = None

        apply_theme(self)
        self._title_icon = apply_app_icon(self)
        if _USE_CUSTOM_TITLE_BAR:
            self.configure(bg=COLORS['panel'])
            self.overrideredirect(True)
            self._is_maximized = False
            self._restore_geometry = ''
            self._drag_x = 0
            self._drag_y = 0
            self._resize_info = None
            self._resize_active = False
            self._resize_after_id: str | None = None
            self._resize_pending: tuple[int, int] | None = None
            self._resize_cursor_edge = ''
            self._pre_minimize_bounds: tuple[int, int, int, int] | None = None
            self._restore_after_minimize = False
            self._minimize_restore_job: str | None = None
            self._corner_after_id: str | None = None
        self._build_ui()
        if _USE_CUSTOM_TITLE_BAR:
            self._enable_edge_resize()
        self._bind_settings_autosave()
        self._load_settings()
        self._center_on_screen()
        if _USE_CUSTOM_TITLE_BAR:
            apply_native_window_frame(self)
            self._refresh_window_corners()
            self.bind('<Map>', self._on_window_map, add='+')
            self.bind('<Configure>', self._on_window_configure_corners, add='+')
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self.update_idletasks()
        self._force_main_window_visible()
        self.after(100, self._drain_log)
        self.after(300, self._maybe_show_device_notice)
        # Re-assert visibility after any deferred widget init (e.g. CTk).
        self.after(250, self._force_main_window_visible)
        # Build Rename off-screen so first tab click doesn't paint widget-by-widget.
        self.after(400, self._preload_renamer_panel)

    def _refresh_window_corners(self) -> None:
        """Re-apply rounded clip for current size / maximize state."""
        if not _USE_CUSTOM_TITLE_BAR:
            return
        _win_apply_rounded_corners(
            self, maximized=bool(getattr(self, '_is_maximized', False)),
        )

    def _schedule_window_corners(self, delay_ms: int = 80) -> None:
        """Debounce corner region updates during resize."""
        if not _USE_CUSTOM_TITLE_BAR:
            return
        job = getattr(self, '_corner_after_id', None)
        if job is not None:
            try:
                self.after_cancel(job)
            except (tk.TclError, ValueError):
                pass

        def _apply() -> None:
            self._corner_after_id = None
            if getattr(self, '_resize_active', False):
                return
            if getattr(self, '_rounding_corners', False):
                return
            self._refresh_window_corners()

        self._corner_after_id = self.after(delay_ms, _apply)

    def _on_window_configure_corners(self, event) -> None:
        if event.widget is not self:
            return
        if getattr(self, '_resize_active', False):
            return
        if getattr(self, '_rounding_corners', False):
            return
        self._schedule_window_corners()

    def _force_main_window_visible(self) -> None:
        """Ensure the main window is not left withdrawn/off-screen after splash."""
        try:
            self.deiconify()
            self.state('normal')
            self.lift()
            self.focus_force()
            if sys.platform == 'win32':
                try:
                    hwnd = _win_toplevel_hwnd(self, flush=True)
                    # SW_SHOW / SW_RESTORE
                    ctypes.windll.user32.ShowWindow(hwnd, 9)
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                except Exception:
                    pass
            # One deferred corner pass — sync refresh here caused busy-cursor flicker.
            self._schedule_window_corners(120)
        except tk.TclError:
            pass

    @staticmethod
    def _device_status_text() -> str:
        if cuda_effective():
            name = cuda_device_name()
            return f'Device: GPU ({name})' if name else 'Device: GPU'
        if torch.cuda.is_available():
            tag = cuda_arch_tag()
            suffix = f', {tag} unsupported' if tag else ' unsupported'
            return f'Device: CPU (GPU{suffix})'
        if torch_cuda_built():
            return 'Device: CPU (CUDA not detected)'
        return 'Device: CPU'

    def _maybe_show_device_notice(self) -> None:
        if cuda_effective():
            return
        if self.hide_device_notice:
            return
        if show_device_notice_dialog(self):
            self.hide_device_notice = True
            self._save_settings()

    def _classify_mode_active(self) -> bool:
        return self._active_mode() == 'classify'

    def _match_mode_active(self) -> bool:
        return self._active_mode() == 'match'

    def _gg_mode_active(self) -> bool:
        return self._active_mode() == 'genre_gender'

    def _rename_mode_active(self) -> bool:
        return self._active_mode() == 'rename'

    def _active_mode(self) -> str:
        name = self.mode_notebook.get()
        return {
            'Classify':      'classify',
            'Match & Align': 'match',
            'Genre & Gender': 'genre_gender',
            'Rename':        'rename',
        }[name]

    def _renamer_destructive_busy(self) -> bool:
        panel = getattr(self, 'renamer_panel', None)
        return panel is not None and bool(panel.destructive_busy)

    def _show_standard_mode_layout(self) -> None:
        self._left_frame.configure(width=540)
        self._left_frame.pack_propagate(False)
        self._content_frame.columnconfigure(0, weight=0, minsize=540)
        self._content_frame.columnconfigure(1, weight=1)
        self._left_frame.grid_configure(
            row=0, column=0, columnspan=1, sticky='nsw', padx=(0, 14),
        )
        self._right_frame.grid()
        self._restore_actions_frame()

    def _restore_actions_frame(self) -> None:
        """Show action bar again after Rename (parent must stay mapped to avoid CTk shrink)."""
        af = self._actions_frame
        try:
            af.pack_propagate(True)
        except tk.TclError:
            pass
        # Undo Rename collapse (height=1); let packed buttons define height.
        try:
            if hasattr(af, '_set_dimensions'):
                af._set_dimensions(height=PATH_BTN_HEIGHT)
            else:
                af.configure(height=PATH_BTN_HEIGHT)
            af.pack_propagate(True)
        except tk.TclError:
            pass
        pack_kw = dict(
            side='bottom', fill='x', padx=SECTION_PADX,
            pady=(ACTIONS_TOP_GAP, ACTIONS_BOTTOM_PAD),
        )
        if not af.winfo_manager():
            af.pack(**pack_kw)
        else:
            af.pack_configure(**pack_kw)

    def _show_rename_mode_layout(self) -> None:
        self._left_frame.pack_propagate(True)
        # Hide buttons only — do NOT pack_forget the actions frame (that
        # was shrinking CTk buttons after returning from Rename).
        self._hide_organize_action_bar()
        if hasattr(self, 'pair_panel'):
            self.pair_panel.hide_action_bar()
        if hasattr(self, 'gg_panel'):
            self.gg_panel.hide_action_bar()
        # Collapse empty bar so Rename keeps full height.
        # Shrink pad by 22px vs standard (20+12) so rules/preview grow and
        # wavebar/Cancel/Rename shift down; Idle/Device status stays put.
        af = self._actions_frame
        af.configure(height=1)
        try:
            af.pack_propagate(False)
        except tk.TclError:
            pass
        pack_kw = dict(
            side='bottom', fill='x', padx=SECTION_PADX,
            pady=(0, 10),
        )
        if not af.winfo_manager():
            af.pack(**pack_kw)
        else:
            af.pack_configure(**pack_kw)
        self._right_frame.grid_remove()
        self._content_frame.columnconfigure(0, weight=1, minsize=540)
        self._content_frame.columnconfigure(1, weight=0)
        self._left_frame.grid_configure(
            row=0, column=0, columnspan=2, sticky='nsew', padx=0,
        )

    def _show_organize_action_bar(self) -> None:
        self._restore_actions_frame()
        self.start_btn.pack(side='left')
        self.stop_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
        self.save_log_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
        self.clear_log_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
        self.play_btn.pack(side='right')
        self._pin_action_bar_heights()

    def _hide_organize_action_bar(self) -> None:
        for widget in self._organize_action_widgets:
            widget.pack_forget()

    def _pin_action_bar_heights(self) -> None:
        """Keep shared bottom-bar buttons at fixed height across tab switches."""
        buttons = list(self._organize_action_widgets)
        pair = getattr(self, 'pair_panel', None)
        if pair is not None:
            buttons.extend((
                getattr(pair, 'find_btn', None),
                getattr(pair, 'organize_btn', None),
                getattr(pair, 'play_stems_btn', None),
                getattr(pair, 'export_list_btn', None),
                getattr(pair, 'distribute_btn', None),
                getattr(pair, 'sort_folders_btn', None),
                getattr(pair, 'align_btn', None),
            ))
        gg = getattr(self, 'gg_panel', None)
        if gg is not None:
            buttons.extend((
                getattr(gg, 'genre_btn', None),
                getattr(gg, 'gender_btn', None),
                getattr(gg, 'stop_btn', None),
            ))
        ctk_pin_button_height(*buttons)

    def _pin_action_bar_heights_later(self) -> None:
        """Pin now and again after layout settles (post-Rename return)."""
        self._pin_action_bar_heights()
        self.after_idle(self._pin_action_bar_heights)
        self.after(50, self._pin_action_bar_heights)

    def _show_rename_veil(self, text: str = 'Loading Rename…') -> None:
        """Opaque cover over Rename tab body only (not mode tabs / action pad)."""
        # Tab content frame sits below the mode segmented button and above the
        # collapsed action bar — smaller at top and bottom than left_frame.
        parent = self._rename_tab
        veil = getattr(self, '_rename_veil', None)
        if veil is not None:
            try:
                if veil.master is not parent:
                    veil.destroy()
                    veil = None
            except tk.TclError:
                veil = None
            if veil is None:
                self._rename_veil = None
                self._rename_veil_label = None
        if veil is None:
            ctk = ensure_ctk_dark()
            veil = ctk.CTkFrame(
                parent, fg_color=DARK['bg'], corner_radius=0,
            )
            lbl = ctk.CTkLabel(
                veil, text=text,
                font=ctk.CTkFont(family='Segoe UI', size=14),
                text_color=DARK['text_dim'],
            )
            lbl.place(relx=0.5, rely=0.45, anchor='center')
            self._rename_veil = veil
            self._rename_veil_label = lbl
        else:
            try:
                self._rename_veil_label.configure(text=text)
            except tk.TclError:
                pass
        veil.place(relx=0, rely=0, relwidth=1, relheight=1)
        veil.lift()

    def _hide_rename_veil(self) -> None:
        veil = getattr(self, '_rename_veil', None)
        if veil is None:
            return
        try:
            veil.place_forget()
        except tk.TclError:
            pass

    def _unmap_rename_holder(self) -> None:
        """Keep Rename content unmapped so return visits don't flash before the veil."""
        holder = getattr(self, '_rename_holder', None)
        if holder is None:
            return
        try:
            if holder.winfo_ismapped():
                holder.pack_forget()
        except tk.TclError:
            pass

    def _renamer_compute_ready(self) -> bool:
        panel = getattr(self, 'renamer_panel', None)
        if panel is None:
            return False
        preview = getattr(panel, 'preview_panel', None)
        if preview is None or not hasattr(preview, 'lazy_compute_complete'):
            return True
        try:
            return bool(preview.lazy_compute_complete())
        except Exception:
            return True

    def _preload_renamer_panel(self) -> None:
        """Build Rename UI into an unmapped holder so widgets never paint mid-build."""
        if self.renamer_panel is not None or getattr(self, '_renamer_building', False):
            return
        holder = getattr(self, '_rename_holder', None)
        if holder is None:
            return
        self._renamer_building = True
        try:
            from track_renamer_panel import TrackRenamerPanel
            _patch_hand_cursor_controls(include_ctk=True)
            panel = TrackRenamerPanel(holder, host=self)
            panel.pack(fill='both', expand=True)
            self.renamer_panel = panel
        finally:
            self._renamer_building = False
        self._poll_renamer_compute()

    def _poll_renamer_compute(self) -> None:
        if not self._renamer_compute_ready():
            self.after(40, self._poll_renamer_compute)
            return
        self._renamer_ready = True
        if getattr(self, '_renamer_reveal_pending', False):
            self._reveal_renamer_panel(getattr(self, '_renamer_reveal_gen', 0))

    def _reveal_renamer_panel(self, gen: int | None = None) -> None:
        """Map prebuilt Rename UI under the veil, then uncover after paint settles."""
        if gen is None:
            gen = getattr(self, '_renamer_reveal_gen', 0)
        if not self._renamer_ready or self.renamer_panel is None:
            self._renamer_reveal_pending = True
            if self.renamer_panel is None and not getattr(self, '_renamer_building', False):
                self._preload_renamer_panel()
            elif not self._renamer_ready:
                self._poll_renamer_compute()
            return
        self._renamer_reveal_pending = False
        holder = self._rename_holder
        try:
            if not holder.winfo_ismapped():
                holder.pack(fill='both', expand=True)
        except tk.TclError:
            holder.pack(fill='both', expand=True)
        # Layout + first preview paint while still covered by veil.
        try:
            self.update_idletasks()
            preview = getattr(self.renamer_panel, 'preview_panel', None)
            if preview is not None and hasattr(preview, '_render_visible'):
                preview._render_visible()
            self.update_idletasks()
        except Exception:
            pass
        self._show_rename_veil()  # keep cover on top after pack
        self.after(200, lambda g=gen: self._finish_renamer_reveal(g))

    def _finish_renamer_reveal(self, gen: int | None = None) -> None:
        if gen is not None and gen != getattr(self, '_renamer_reveal_gen', 0):
            return
        if self._active_mode() != 'rename':
            self._hide_rename_veil()
            return
        self._hide_rename_veil()
        panel = self.renamer_panel
        if panel is not None:
            panel.on_tab_shown()

    def _begin_rename_tab_show(self) -> None:
        """Every Rename visit: layout → veil → remount under cover → uncover."""
        self._renamer_reveal_gen = getattr(self, '_renamer_reveal_gen', 0) + 1
        gen = self._renamer_reveal_gen
        self.pair_panel.hide_action_bar()
        if hasattr(self, 'gg_panel'):
            self.gg_panel.hide_action_bar()
        self._hide_organize_action_bar()
        # Unmap first so CTk tab switch can't flash the dense Rename tree.
        self._unmap_rename_holder()
        # Reclaim pad / expand column BEFORE cover so Loading matches final size.
        self._show_rename_mode_layout()
        self.update_idletasks()
        self._show_rename_veil()
        self.update_idletasks()
        try:
            self.update()  # paint veil at final Rename footprint
        except tk.TclError:
            pass
        self._renamer_reveal_pending = True
        if self.renamer_panel is None:
            self._preload_renamer_panel()
        self._reveal_renamer_panel(gen)

    def _on_mode_tab_changed(self, _event=None) -> None:
        mode = self._active_mode()
        rename_visible = mode == 'rename'
        if rename_visible != self._renamer_tab_visible:
            panel = getattr(self, 'renamer_panel', None)
            if panel is not None and self._renamer_ready and not rename_visible:
                panel.on_tab_hidden()
            self._renamer_tab_visible = rename_visible

        if mode == 'rename':
            self._begin_rename_tab_show()
        else:
            self._renamer_reveal_gen = getattr(self, '_renamer_reveal_gen', 0) + 1
            self._renamer_reveal_pending = False
            self._unmap_rename_holder()
            self._hide_rename_veil()
            if mode == 'classify':
                self._show_standard_mode_layout()
                self.pair_panel.hide_action_bar()
                if hasattr(self, 'gg_panel'):
                    self.gg_panel.hide_action_bar()
                self._show_organize_action_bar()
                self._update_action_buttons_for_tab()
            elif mode == 'match':
                self._show_standard_mode_layout()
                self._hide_organize_action_bar()
                if hasattr(self, 'gg_panel'):
                    self.gg_panel.hide_action_bar()
                self.pair_panel.show_action_bar()
                if not self._pair_busy:
                    self.pair_panel.set_buttons_state('normal')
            elif mode == 'genre_gender':
                self._show_standard_mode_layout()
                self._hide_organize_action_bar()
                self.pair_panel.hide_action_bar()
                self.gg_panel.show_action_bar()
                if not self._pair_busy:
                    self.gg_panel.set_buttons_state('normal')
            # Rename unmaps the action bar parent; re-pin after layout settles.
            self._pin_action_bar_heights_later()

    def _set_action_buttons(self, running: bool, *, sdr: bool = False) -> None:
        """Only swap colors — never disabled/style changes (ttk shrinks on both)."""
        if hasattr(self, 'pair_panel'):
            self.pair_panel.set_buttons_state('disabled' if running else 'normal')
        if hasattr(self, 'gg_panel'):
            self.gg_panel.set_buttons_state('disabled' if running else 'normal')
        if not self._classify_mode_active():
            return
        if running:
            self.start_btn.configure(
                text_color=DARK['text_dim'], cursor='arrow', height=PATH_BTN_HEIGHT,
            )
            self.stop_btn.configure(
                text_color=DARK['danger'], cursor='hand2', height=PATH_BTN_HEIGHT,
            )
            for btn in (self.save_log_btn, self.clear_log_btn, self.play_btn):
                btn.configure(
                    text_color=DARK['text_dim'], cursor='arrow', height=PATH_BTN_HEIGHT,
                )
        else:
            self.start_btn.configure(
                text_color='#ffffff', cursor='hand2', height=PATH_BTN_HEIGHT,
            )
            self.stop_btn.configure(
                text_color=DARK['text_dim'], cursor='arrow', height=PATH_BTN_HEIGHT,
            )
            for btn in (self.save_log_btn, self.clear_log_btn, self.play_btn):
                btn.configure(
                    text_color=DARK['text'], cursor='hand2', height=PATH_BTN_HEIGHT,
                )
        self._pin_action_bar_heights()

    def _update_action_buttons_for_tab(self) -> None:
        if not self._classify_mode_active():
            return
        sdr = self._class_tab.get() == 'sdr'
        running = (
            (self.sdr_worker is not None and self.sdr_worker.is_alive()) if sdr
            else (self.worker is not None and self.worker.is_alive())
        )
        # Never toggle state= — CTk buttons shrink on Windows when re-enabled.
        if sdr:
            self.start_btn.configure(
                text='▶  Start SI-SDR', command=self._start_sdr, width=108,
                height=PATH_BTN_HEIGHT,
            )
            self.stop_btn.configure(command=self._stop_sdr, height=PATH_BTN_HEIGHT)
            self.save_log_btn.configure(
                text='Save SI-SDR log', command=self._save_sdr_log, width=108,
                height=PATH_BTN_HEIGHT,
            )
            self._action_tooltips['start'].set_text(TIPS['start_sdr'])
            self._action_tooltips['stop'].set_text(TIPS['stop_sdr'])
            self._action_tooltips['save'].set_text(TIPS['save_sdr_log'])
        else:
            self.start_btn.configure(
                text='▶  Start RMS', command=self._start, width=96,
                height=PATH_BTN_HEIGHT,
            )
            self.stop_btn.configure(command=self._stop, height=PATH_BTN_HEIGHT)
            self.save_log_btn.configure(
                text='Save RMS log', command=self._save_log, width=96,
                height=PATH_BTN_HEIGHT,
            )
            self._action_tooltips['start'].set_text(TIPS['start'])
            self._action_tooltips['stop'].set_text(TIPS['stop'])
            self._action_tooltips['save'].set_text(TIPS['save_log'])
        self._set_action_buttons(running, sdr=sdr)
        self._pin_action_bar_heights()

    def _on_class_tab_changed(self, _event=None) -> None:
        tab_name = self.cls_notebook.get()
        self._class_tab.set('sdr' if tab_name == 'SI-SDR' else 'rms')
        self._update_action_buttons_for_tab()
        self._refresh_cls_frame()

    def _path_button(self, parent: tk.Misc, text: str, command) -> tk.Button:
        C = COLORS
        return tk.Button(
            parent, text=text, command=command,
            font=PATH_BTN_FONT, bg=C['panel2'], fg=C['fg'],
            activebackground=C['panel'], activeforeground=C['fg'],
            relief='flat', borderwidth=0,
            highlightthickness=0,
            takefocus=0,
            padx=PATH_BTN_PADX, pady=PATH_BTN_PADY, cursor='hand2',
        )

    def _normalize_path_var(self, var: tk.StringVar) -> None:
        normalized = display_path(var.get())
        if normalized != var.get():
            var.set(normalized)

    def _path_row(self, parent, row, label, var, picker, opener, tip_text):
        ctk = ensure_ctk_dark()
        t = DARK
        _font = ctk_ui_font()
        lbl = ctk.CTkLabel(
            parent, text=label, text_color=t['label'], font=_font,
        )
        lbl.grid(row=row, column=0, sticky='w', padx=(0, 10), pady=CTRL_ROW_PADY)
        ent = ctk.CTkEntry(
            parent, textvariable=var,
            fg_color=t['control_bg'], border_color=t['border'],
            text_color=t['entry_text'], font=_font, height=30,
        )
        ent.grid(row=row, column=1, sticky='ew', pady=CTRL_ROW_PADY)
        ent.bind('<FocusOut>', lambda _e, v=var: self._normalize_path_var(v))
        browse_btn = ctk.CTkButton(
            parent, text='Browse', width=72, height=30,
            fg_color=t['btn'], hover_color=t['btn_hover'], text_color=t['text'],
            font=_font, command=picker,
        )
        browse_btn.grid(row=row, column=2, padx=(4, 0), pady=CTRL_ROW_PADY)
        open_btn = ctk.CTkButton(
            parent, text='Open', width=64, height=30,
            fg_color=t['btn'], hover_color=t['btn_hover'], text_color=t['text'],
            font=_font, command=opener,
        )
        open_btn.grid(row=row, column=3, padx=(4, 0), pady=CTRL_ROW_PADY)
        tip(lbl, ent, browse_btn, text=tip_text)
        Tooltip(open_btn, TIPS['open_path'])

    def _combo_field(self, parent, row, col, label, var, values, tip_text, *,
                     sticky='ew', width=None):
        ctk = ensure_ctk_dark()
        t = DARK
        _font = ctk_ui_font()
        lbl = ctk.CTkLabel(
            parent, text=label, text_color=t['label'], font=_font,
        )
        lbl.grid(row=row, column=col, sticky='w', padx=(0, 10), pady=CTRL_ROW_PADY)
        cb_width = (width * 8 + 24) if width is not None else 200
        cb = ctk.CTkOptionMenu(
            parent,
            variable=var,
            values=list(values),
            fg_color=t['control_bg'],
            button_color=t['btn'],
            button_hover_color=t['btn_hover'],
            text_color=t['text'],
            dropdown_fg_color=t['panel'],
            dropdown_hover_color=t['btn_hover'],
            dropdown_text_color=t['text'],
            font=_font,
            dropdown_font=_font,
            width=cb_width,
            height=30,
        )
        cb.grid(row=row, column=col + 1, sticky=sticky,
                padx=(0, 16) if col == 0 else 0, pady=CTRL_ROW_PADY)
        tip(lbl, cb, text=tip_text)

    def _slider_field(self, parent, row, col, label, var, lo, hi, fmt, tip_text):
        ctk = ensure_ctk_dark()
        t = DARK
        _font = ctk_ui_font()
        lbl = ctk.CTkLabel(
            parent, text=label, text_color=t['label'], font=_font,
        )
        lbl.grid(row=row, column=col, sticky='w', padx=(0, 10), pady=3)
        row_frm = ctk.CTkFrame(parent, fg_color='transparent')
        # Left column: gap before next field. Right column: small inset from border.
        row_frm.grid(
            row=row, column=col + 1, sticky='ew',
            padx=(0, 16) if col == 0 else (0, 6), pady=3,
        )
        row_frm.columnconfigure(0, weight=1)
        readout = ctk.CTkLabel(
            row_frm, text=fmt(var.get()), text_color=t['text'],
            width=40, anchor='e', font=_font,
        )
        scale = ctk.CTkSlider(
            row_frm, from_=lo, to=hi, variable=var,
            button_color=t['accent'], button_hover_color=t['accent_hover'],
            progress_color=t['accent'], fg_color=t['control_bg'],
            height=16,
            command=lambda _v: readout.configure(text=fmt(var.get())),
        )
        scale.grid(row=0, column=0, sticky='ew')
        readout.grid(row=0, column=1, padx=(8, 0))
        tip(lbl, scale, readout, text=tip_text)

    def _center_on_screen(self) -> None:
        self.update_idletasks()
        w = WIN_DEFAULT_W
        h = WIN_DEFAULT_H
        x, y, w, h = _place_window_centered(self, w, h)
        if _USE_CUSTOM_TITLE_BAR and _win_move_resize(self, x, y, w, h):
            _sync_tk_geometry(self, x, y, w, h)
        else:
            self.geometry(f'{w}x{h}+{x}+{y}')

    def _on_window_map(self, _event=None) -> None:
        if getattr(self, '_resize_active', False):
            return
        apply_native_window_frame(self)
        self._refresh_window_corners()
        if not self._restore_after_minimize or self._pre_minimize_bounds is None:
            return
        self._restore_after_minimize = False
        bounds = self._pre_minimize_bounds
        if getattr(self, '_minimize_restore_job', None) is not None:
            try:
                self.after_cancel(self._minimize_restore_job)
            except tk.TclError:
                pass

        def restore_bounds() -> None:
            self._minimize_restore_job = None
            x, y, w, h = _clamp_window_bounds(self, *bounds)
            if _USE_CUSTOM_TITLE_BAR:
                _win_move_resize(self, x, y, w, h)
            _sync_tk_geometry(self, x, y, w, h)
            self._refresh_window_corners()

        # Windows completes SW_RESTORE after the Map event is dispatched.
        self._minimize_restore_job = self.after(30, restore_bounds)

    def _bind_title_drag(self, widget: tk.Widget) -> None:
        widget.bind('<Button-1>', self._title_start_move, add='+')
        widget.bind('<B1-Motion>', self._title_drag_window, add='+')
        widget.bind('<Double-Button-1>', lambda _e: self._toggle_maximize(), add='+')

    def _title_start_move(self, event) -> None:
        if self._is_maximized:
            return
        self._drag_x = event.x_root - self.winfo_x()
        self._drag_y = event.y_root - self.winfo_y()

    def _title_drag_window(self, event) -> None:
        if self._is_maximized:
            return
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        rect = _win_window_rect(self)
        if rect is None:
            try:
                self.geometry(f'+{x}+{y}')
            except tk.TclError:
                pass
            return
        _, _, w, h = rect
        x, y, w, h = _clamp_window_bounds(self, x, y, w, h)
        if _USE_CUSTOM_TITLE_BAR:
            _win_move_resize(self, x, y, w, h)
        else:
            _sync_tk_geometry(self, x, y, w, h)

    def _enable_edge_resize(self) -> None:
        cursors = {
            'n': 'top_side', 's': 'bottom_side', 'e': 'right_side', 'w': 'left_side',
            'ne': 'top_right_corner', 'nw': 'top_left_corner',
            'se': 'bottom_right_corner', 'sw': 'bottom_left_corner',
        }

        def _hit_test(x: int, y: int) -> str:
            if self._is_maximized:
                return ''
            w, h = self.winfo_width(), self.winfo_height()
            if w < 2 or h < 2:
                return ''
            b = RESIZE_BORDER
            left, right = x < b, x >= w - b
            top, bottom = y < b, y >= h - b
            if top and left:
                return 'nw'
            if top and right:
                return 'ne'
            if bottom and left:
                return 'sw'
            if bottom and right:
                return 'se'
            if left:
                return 'w'
            if right:
                return 'e'
            if top:
                return 'n'
            if bottom:
                return 's'
            return ''

        def _event_pos_in_window(event) -> tuple[int, int]:
            # Motion/press bubble from children; event.x/y are child-local.
            # Hit-test must use coords relative to the toplevel window.
            return (
                int(event.x_root - self.winfo_rootx()),
                int(event.y_root - self.winfo_rooty()),
            )

        def _on_motion(event) -> None:
            x, y = _event_pos_in_window(event)
            edge = _hit_test(x, y)
            if edge == self._resize_cursor_edge:
                return
            self._resize_cursor_edge = edge
            try:
                self.configure(cursor=cursors.get(edge, ''))
            except tk.TclError:
                pass

        def _window_bounds() -> tuple[int, int, int, int]:
            rect = _win_window_rect(self)
            if rect is not None:
                return rect
            return (
                self.winfo_x(), self.winfo_y(),
                self.winfo_width(), self.winfo_height(),
            )

        def _on_press(event) -> None:
            x, y = _event_pos_in_window(event)
            edge = _hit_test(x, y)
            if not edge:
                return
            ox, oy, ow, oh = _window_bounds()
            self._resize_active = True
            self._resize_info = (
                edge, event.x_root, event.y_root, ox, oy, ow, oh,
            )

        def _apply_resize_drag() -> None:
            self._resize_after_id = None
            pending = self._resize_pending
            if not self._resize_info or pending is None:
                return
            event_x, event_y = pending
            self._resize_pending = None
            edge, sx, sy, ox, oy, ow, oh = self._resize_info
            dx, dy = event_x - sx, event_y - sy
            x, y, w, h = ox, oy, ow, oh
            if 'e' in edge:
                w = ow + dx
            if 's' in edge:
                h = oh + dy
            if 'w' in edge:
                w = ow - dx
                x = ox + ow - w
            if 'n' in edge:
                h = oh - dy
                y = oy + oh - h
            x, y, w, h = _clamp_window_bounds(self, x, y, w, h)
            try:
                if _USE_CUSTOM_TITLE_BAR:
                    if not _win_move_resize(self, x, y, w, h):
                        _sync_tk_geometry(self, x, y, w, h)
                else:
                    _sync_tk_geometry(self, x, y, w, h)
            except tk.TclError:
                pass

        def _on_drag(event) -> None:
            if not self._resize_info:
                return
            self._resize_pending = (event.x_root, event.y_root)
            if self._resize_after_id is not None:
                return
            self._resize_after_id = self.after_idle(_apply_resize_drag)

        def _on_release(_event) -> None:
            if self._resize_after_id is not None:
                try:
                    self.after_cancel(self._resize_after_id)
                except ValueError:
                    pass
                self._resize_after_id = None
            self._resize_pending = None
            self._resize_info = None
            self._resize_active = False
            self._resize_cursor_edge = ''
            try:
                self.configure(cursor='')
            except tk.TclError:
                pass
            rect = _win_window_rect(self)
            if rect is not None:
                _sync_tk_geometry(self, *rect)
            self._refresh_window_corners()

        self.bind('<Motion>', _on_motion, add='+')
        self.bind('<ButtonPress-1>', _on_press, add='+')
        self.bind('<B1-Motion>', _on_drag, add='+')
        self.bind('<ButtonRelease-1>', _on_release, add='+')

    def _minimize_window(self) -> None:
        if _USE_CUSTOM_TITLE_BAR:
            self.update_idletasks()
            bounds = _win_window_rect(self)
            if bounds is None:
                bounds = (
                    self.winfo_x(),
                    self.winfo_y(),
                    self.winfo_width(),
                    self.winfo_height(),
                )
            self._pre_minimize_bounds = bounds
            self._restore_after_minimize = True
            if _win_show_window(self, 6):  # SW_MINIMIZE
                return
        self.iconify()

    def _toggle_maximize(self) -> None:
        if self._is_maximized:
            if self._restore_geometry:
                self.geometry(self._restore_geometry)
            self._is_maximized = False
            self.update_idletasks()
            self._refresh_window_corners()
            return
        self.update_idletasks()
        self._restore_geometry = self.geometry()
        x, y, w, h = _win_work_area(self)
        x, y, w, h = _clamp_window_bounds(self, x, y, w, h)
        if _USE_CUSTOM_TITLE_BAR:
            _win_move_resize(self, x, y, w, h)
        _sync_tk_geometry(self, x, y, w, h)
        self._is_maximized = True
        self._refresh_window_corners()

    def _title_button(self, parent, text, command, hover_bg, hover_fg=None):
        fg = COLORS['fg_dim']
        hover_fg = hover_fg or COLORS['fg']
        btn = tk.Label(
            parent, text=text, width=4,
            bg=COLORS['panel'], fg=fg,
            font=('Segoe UI', 11), cursor='hand2',
        )
        btn.pack(side='left')

        def on_enter(_e):
            btn.configure(bg=hover_bg, fg=hover_fg)

        def on_leave(_e):
            btn.configure(bg=COLORS['panel'], fg=fg)

        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        btn.bind('<Button-1>', lambda _e: command())
        return btn

    def _build_custom_title_bar(self) -> None:
        bar = tk.Frame(
            self, bg=COLORS['panel'], height=TITLE_BAR_HEIGHT,
            highlightthickness=0, bd=0,
        )
        bar.grid(row=0, column=0, sticky='ew')
        bar.grid_propagate(False)
        bar.columnconfigure(1, weight=1)

        col = 0
        if self._title_icon is not None:
            icon = tk.Label(bar, image=self._title_icon, bg=COLORS['panel'])
            icon.grid(
                row=0, column=col, sticky='w',
                padx=(10, 2), pady=(TITLE_BAR_CONTENT_PAD_Y, 0),
            )
            self._bind_title_drag(icon)
            col += 1

        title = tk.Label(
            bar, text='STEM organizer',
            bg=COLORS['panel'], fg=COLORS['fg_dim'],
            font=('Segoe UI', 10), anchor='w',
        )
        title.grid(
            row=0, column=col, sticky='w',
            pady=(TITLE_BAR_CONTENT_PAD_Y, 0),
        )

        controls = tk.Frame(bar, bg=COLORS['panel'])
        controls.grid(row=0, column=col + 1, sticky='e')
        self._title_button(controls, '\u2212', self._minimize_window, COLORS['panel2'])
        self._title_button(controls, '\u25a1', self._toggle_maximize, COLORS['panel2'])
        self._title_button(controls, '\u00d7', self._on_close, COLORS['danger'], hover_fg='white')

        for widget in (bar, title):
            self._bind_title_drag(widget)

        self._title_bar = bar

    def _build_ui(self):
        ctk = ensure_ctk_dark()
        content_row = 1 if _USE_CUSTOM_TITLE_BAR else 0
        bottom_row = 2 if _USE_CUSTOM_TITLE_BAR else 1

        self.columnconfigure(0, weight=1)
        self.rowconfigure(content_row, weight=1)
        if _USE_CUSTOM_TITLE_BAR:
            self.rowconfigure(0, weight=0)
            self._build_custom_title_bar()

        content = ctk.CTkFrame(self, fg_color='transparent')
        content.grid(row=content_row, column=0, sticky='nsew',
                     padx=CONTENT_PAD, pady=(0, CONTENT_PAD_Y))
        self._content_frame = content
        content.columnconfigure(0, weight=0, minsize=540)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left = ctk.CTkFrame(content, fg_color='transparent', width=540)
        left.grid(row=0, column=0, sticky='nsw', padx=(0, 14))
        left.pack_propagate(False)
        self._left_frame = left

        actions = ctk.CTkFrame(left, fg_color='transparent')
        actions.pack(
            side='bottom', fill='x', padx=SECTION_PADX,
            pady=(ACTIONS_TOP_GAP, ACTIONS_BOTTOM_PAD),
        )
        self._actions_frame = actions
        t = DARK
        _btn_font = ctk_ui_font()
        self.start_btn = ctk.CTkButton(
            actions, text='▶  Start RMS', command=self._start,
            font=_btn_font, width=96, height=PATH_BTN_HEIGHT,
            fg_color=t['accent'], hover_color=t['accent_hover'],
            text_color='#ffffff', cursor='hand2',
        )
        self.start_btn.pack(side='left')
        self.stop_btn = ctk.CTkButton(
            actions, text='■  Stop', command=self._stop,
            font=_btn_font, width=64, height=PATH_BTN_HEIGHT,
            fg_color=t['btn'], hover_color=t['btn_hover'],
            text_color=t['text_dim'], cursor='arrow',
        )
        self.stop_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
        self.save_log_btn = ctk.CTkButton(
            actions, text='Save RMS log', command=self._save_log,
            font=_btn_font, width=96, height=PATH_BTN_HEIGHT,
            fg_color=t['btn'], hover_color=t['btn_hover'],
            text_color=t['text'], cursor='hand2',
        )
        self.save_log_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
        self.clear_log_btn = ctk.CTkButton(
            actions, text='Clear log', command=self._clear_log,
            font=_btn_font, width=76, height=PATH_BTN_HEIGHT,
            fg_color=t['btn'], hover_color=t['btn_hover'],
            text_color=t['text'], cursor='hand2',
        )
        self.clear_log_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
        self._action_tooltips = {
            'start': Tooltip(self.start_btn, TIPS['start']),
            'stop': Tooltip(self.stop_btn, TIPS['stop']),
            'save': Tooltip(self.save_log_btn, TIPS['save_log']),
            'clear': Tooltip(self.clear_log_btn, TIPS['clear_log']),
        }
        self.play_btn = ctk.CTkButton(
            actions, text='♫  Play', command=self._open_stem_player,
            font=_btn_font, width=72, height=PATH_BTN_HEIGHT,
            fg_color=t['btn'], hover_color=t['accent'],
            text_color=t['text'], cursor='hand2',
        )
        self.play_btn.pack(side='right')
        Tooltip(self.play_btn, TIPS['play_stems'])
        self._organize_action_widgets = (
            self.start_btn, self.stop_btn, self.save_log_btn,
            self.clear_log_btn, self.play_btn,
        )

        left_body = ctk.CTkFrame(left, fg_color='transparent')
        left_body.pack(side='top', fill='both', expand=True)

        self.mode_notebook = ctk.CTkTabview(
            left_body,
            fg_color=DARK['bg'],
            segmented_button_fg_color=DARK['panel'],
            segmented_button_selected_color=DARK['accent'],
            segmented_button_selected_hover_color=DARK['accent_hover'],
            segmented_button_unselected_color=DARK['panel'],
            segmented_button_unselected_hover_color=DARK['btn_hover'],
            text_color=DARK['text'],
            text_color_disabled=DARK['text_dim'],
            command=self._on_mode_tab_changed,
        )
        self.mode_notebook.pack(fill='both', expand=True)
        for _tab_name in ('Classify', 'Match & Align', 'Genre & Gender', 'Rename'):
            self.mode_notebook.add(_tab_name)
        self.mode_notebook._segmented_button.configure(font=ctk_ui_font())
        organize_tab = self.mode_notebook.tab('Classify')
        pair_finder_tab = self.mode_notebook.tab('Match & Align')
        genre_gender_tab = self.mode_notebook.tab('Genre & Gender')
        rename_tab = self.mode_notebook.tab('Rename')
        self._organize_tab = organize_tab
        self._pair_finder_tab = pair_finder_tab
        self._genre_gender_tab = genre_gender_tab
        self._rename_tab = rename_tab

        from pair_finder_panel import PairFinderPanel
        self.pair_panel = PairFinderPanel(
            self, pair_finder_tab, info_icon_factory=InfoIcon,
        )
        self.pair_panel.pack(fill='both', expand=True)
        self.pair_panel.attach_action_bar(actions)

        from genre_gender_panel import GenreGenderPanel
        self.gg_panel = GenreGenderPanel(
            self, genre_gender_tab, info_icon_factory=InfoIcon,
        )
        self.gg_panel.pack(fill='both', expand=True)
        self.gg_panel.attach_action_bar(actions)

        self.renamer_panel = None
        self._renamer_ready = False
        self._renamer_building = False
        self._renamer_reveal_pending = False
        self._renamer_reveal_gen = 0
        self._rename_veil = None
        self._rename_veil_label = None
        # Unmapped until reveal — build widgets here so nothing paints mid-construction.
        self._rename_holder = ctk.CTkFrame(rename_tab, fg_color=DARK['bg'], corner_radius=0)
        self._renamer_tab_visible = None

        right = ctk.CTkFrame(content, fg_color='transparent')
        right.grid(row=0, column=1, sticky='nsew')
        self._right_frame = right
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        header = ctk.CTkFrame(organize_tab, fg_color='transparent')
        header.pack(fill='x', padx=SECTION_PADX, pady=(HEADER_TOP_PAD, 6))
        desc_row = ctk.CTkFrame(header, fg_color='transparent')
        desc_row.pack(fill='x', anchor='w')
        ctk.CTkLabel(
            desc_row,
            text='Classifies stems, mixes originals into 2/4-stem folders.',
            font=ctk_ui_font(),
            text_color=DARK['text_dim'],
            wraplength=470,
            justify='left',
        ).pack(side='left')
        # CTkLabel is taller than the glyph; pin icon to top so it sits with the text.
        InfoIcon(desc_row, self._show_about_dialog).pack(
            side='left', padx=(4, 0), anchor='n', pady=(5, 0),
        )
        if not FFMPEG:
            from ffmpeg_bootstrap import ffmpeg_missing_message

            ctk.CTkLabel(
                header,
                text=ffmpeg_missing_message(),
                font=ctk.CTkFont(family='Segoe UI', size=9),
                text_color=DARK['text_dim'],
                wraplength=500,
                justify='left',
            ).pack(anchor='w', pady=(4, 0))

        paths = ctk_section(organize_tab, 'PATHS')
        paths.columnconfigure(1, weight=1)
        self._path_row(paths, 0, 'Input',  self.input_dir,  self._pick_input,
                       self._open_input, TIPS['input'])
        self._path_row(paths, 1, 'Output', self.output_dir, self._pick_output,
                       self._open_output, TIPS['output'])
        self._combo_field(paths, 2, 0, 'Scan',   self.scan_label,   list(SCAN_MODES),   TIPS['scan'],
                          sticky='w', width=PATH_COMBO_WIDTH)
        self._combo_field(paths, 3, 0, 'Naming', self.naming_label, list(NAMING_MODES), TIPS['naming'],
                          sticky='w', width=PATH_COMBO_WIDTH)
        _ui = ctk_ui_font()

        filters = ctk_section(organize_tab, 'OUTPUT FILTERS')
        filters.columnconfigure(0, weight=1)
        filter_row0 = ctk.CTkFrame(filters, fg_color='transparent')
        filter_row0.grid(row=0, column=0, sticky='ew', pady=2)
        filter_left = ctk.CTkFrame(filter_row0, fg_color='transparent')
        filter_left.pack(side='left')
        short_chk = ctk.CTkCheckBox(
            filter_left,
            text='Delete folder if shorter than',
            variable=self.delete_if_short,
            onvalue=True, offvalue=False,
            fg_color=DARK['accent'],
            hover_color=DARK['accent_hover'],
            text_color=DARK['text'],
            border_color=DARK['border'],
            checkmark_color='#ffffff',
            font=_ui,
            command=self._update_filter_state,
        )
        short_chk.pack(side='left')
        Tooltip(short_chk, TIPS['delete_short'])
        dur_ctrl = ctk.CTkFrame(filter_left, fg_color='transparent')
        dur_ctrl.pack(side='left', padx=(8, 0))
        self.min_dur_sp = ttk.Spinbox(
            dur_ctrl, from_=1, to=600, textvariable=self.min_duration_sec, width=6,
            font=ttk_ui_font(),
        )
        self.min_dur_sp.pack(side='left')
        ctk.CTkLabel(
            dur_ctrl, text='seconds', text_color=DARK['text_dim'], font=_ui,
        ).pack(side='left', padx=(8, 0))
        tip(self.min_dur_sp, text=TIPS['min_duration'])
        skip_chk = ctk.CTkCheckBox(
            filter_row0,
            text='Skip if output already exists',
            variable=self.skip_existing,
            onvalue=True, offvalue=False,
            fg_color=DARK['accent'],
            hover_color=DARK['accent_hover'],
            text_color=DARK['text'],
            border_color=DARK['border'],
            checkmark_color='#ffffff',
            font=_ui,
        )
        skip_chk.pack(side='right')
        Tooltip(skip_chk, TIPS['skip_existing'])
        incomplete_chk = ctk.CTkCheckBox(
            filters,
            text='Delete folder if any expected stem is missing',
            variable=self.delete_if_incomplete,
            onvalue=True, offvalue=False,
            fg_color=DARK['accent'],
            hover_color=DARK['accent_hover'],
            text_color=DARK['text'],
            border_color=DARK['border'],
            checkmark_color='#ffffff',
            font=_ui,
        )
        incomplete_chk.grid(row=1, column=0, sticky='w', pady=(2, 0))
        Tooltip(incomplete_chk, TIPS['delete_incomplete'])

        opts = ctk_section(organize_tab, 'OPTIONS')
        opts.columnconfigure(1, weight=1)
        opts.columnconfigure(3, weight=1)
        self._combo_field(opts, 0, 0, 'Model',   self.model_label, list(MODELS),          TIPS['model'])
        self._combo_field(opts, 0, 2, 'Stems',   self.stem_mode,   list(STEM_MODES),      TIPS['stems'])
        self._combo_field(opts, 1, 0, 'Quality', self.quality,     list(QUALITY_PRESETS), TIPS['quality'])
        if cuda_effective():
            cuda_text = 'Use CUDA (GPU)'
            cuda_enabled = True
        elif torch.cuda.is_available():
            cuda_text = 'Use CUDA (GPU)   ·   incompatible PyTorch build'
            cuda_enabled = False
        elif torch_cuda_built():
            cuda_text = 'Use CUDA (GPU)   ·   no GPU detected'
            cuda_enabled = False
        else:
            cuda_text = 'Use CUDA (GPU)   ·   unavailable'
            cuda_enabled = False
        cuda_chk = ctk.CTkCheckBox(
            opts,
            text=cuda_text,
            variable=self.use_cuda,
            onvalue=True, offvalue=False,
            fg_color=DARK['accent'],
            hover_color=DARK['accent_hover'],
            text_color=DARK['text'] if cuda_enabled else DARK['text_dim'],
            border_color=DARK['border'],
            checkmark_color='#ffffff',
            font=_ui,
            state='normal' if cuda_enabled else 'disabled',
        )
        cuda_chk.grid(row=1, column=2, columnspan=2, sticky='w', pady=3)
        Tooltip(cuda_chk, TIPS['cuda'])
        self._combo_field(opts, 2, 0, 'On ambiguous', self.ambig_label, list(AMBIG_MODES), TIPS['ambig'])

        # CLASSIFICATION: RMS/SI-SDR hang on the card top border (CTk overhang).
        cls_wrap = ctk.CTkFrame(organize_tab, fg_color='transparent')
        cls_wrap.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        ctk.CTkLabel(
            cls_wrap,
            text='CLASSIFICATION',
            font=ctk_section_font(),
            text_color=DARK['text_dim'],
            anchor='w',
        ).pack(anchor='w', pady=(0, 3))

        cls_slot = ctk.CTkFrame(cls_wrap, fg_color='transparent')
        cls_slot.pack(fill='x')

        # Shorter card so bottom border isn't clipped by the tab body / action bar.
        _cls_h = 248
        _hang = 6
        self.cls_notebook = ctk.CTkTabview(
            cls_slot,
            fg_color=DARK['panel'],
            bg_color=DARK['bg'],
            border_color=DARK['border'],
            border_width=1,
            corner_radius=8,
            segmented_button_fg_color=DARK['panel2'],
            segmented_button_selected_color=DARK['accent'],
            segmented_button_selected_hover_color=DARK['accent_hover'],
            segmented_button_unselected_color=DARK['panel2'],
            segmented_button_unselected_hover_color=DARK['btn_hover'],
            text_color=DARK['text'],
            text_color_disabled=DARK['text_dim'],
            anchor='n',
            height=_cls_h,
            command=self._on_class_tab_changed,
        )
        self.cls_notebook._segmented_button.configure(font=_ui)
        self.cls_notebook._outer_spacing = 0
        self.cls_notebook._outer_button_overhang = _hang
        self.cls_notebook._configure_grid()
        self.cls_notebook._set_grid_canvas()
        self.cls_notebook._set_grid_segmented_button()
        self.cls_notebook._configure_segmented_button_background_corners()
        self.cls_notebook.place(x=0, y=0, relwidth=1)
        # Slot a few px taller than the card so the bottom border isn't clipped.
        cls_slot.configure(height=_cls_h + 4)
        cls_slot.pack_propagate(False)
        self.cls_frame = self.cls_notebook
        self.cls_notebook.add('RMS')
        self.cls_notebook.add('SI-SDR')
        self.cls_rms_tab = self.cls_notebook.tab('RMS')
        self.cls_sdr_tab = self.cls_notebook.tab('SI-SDR')

        self.cls_rms_tab.columnconfigure(1, weight=1)
        self.cls_rms_tab.columnconfigure(3, weight=1)
        pct = lambda v: f"{v:.0%}"
        self._slider_field(self.cls_rms_tab, 0, 0, 'Confidence', self.threshold,  0.10, 0.90, pct, TIPS['confidence'])
        self._slider_field(self.cls_rms_tab, 0, 2, 'Min. margin', self.min_margin, 0.00, 0.50, pct, TIPS['margin'])
        batch_lbl = ctk.CTkLabel(
            self.cls_rms_tab, text='Batch size', text_color=DARK['label'], font=_ui,
        )
        batch_lbl.grid(row=1, column=0, sticky='w', padx=(0, 10), pady=3)
        batch_sp = ttk.Spinbox(
            self.cls_rms_tab, from_=1, to=16, textvariable=self.batch_size, width=6,
            font=ttk_ui_font(),
        )
        batch_sp.grid(row=1, column=1, sticky='w', pady=3)
        tip(batch_lbl, batch_sp, text=TIPS['batch'])
        dedup_chk = ctk.CTkCheckBox(
            self.cls_rms_tab,
            text='Remove duplicate stems (keep quietest)',
            variable=self.dedup,
            onvalue=True, offvalue=False,
            fg_color=DARK['accent'],
            hover_color=DARK['accent_hover'],
            text_color=DARK['text'],
            border_color=DARK['border'],
            checkmark_color='#ffffff',
            font=_ui,
        )
        dedup_chk.grid(row=2, column=0, columnspan=4, sticky='w', pady=3)
        Tooltip(dedup_chk, TIPS['dedup'])
        peak_chk = ctk.CTkCheckBox(
            self.cls_rms_tab,
            text='Normalize so summed mixture peaks at -1 dB',
            variable=self.peak_norm,
            onvalue=True, offvalue=False,
            fg_color=DARK['accent'],
            hover_color=DARK['accent_hover'],
            text_color=DARK['text'],
            border_color=DARK['border'],
            checkmark_color='#ffffff',
            font=_ui,
        )
        peak_chk.grid(row=3, column=0, columnspan=4, sticky='w', pady=3)
        Tooltip(peak_chk, TIPS['peak_norm'])
        self.mix_chk = ctk.CTkCheckBox(
            self.cls_rms_tab,
            text='Also write mixture.wav (WAV quality only)',
            variable=self.make_mixture,
            onvalue=True, offvalue=False,
            fg_color=DARK['accent'],
            hover_color=DARK['accent_hover'],
            text_color=DARK['text'],
            border_color=DARK['border'],
            checkmark_color='#ffffff',
            font=_ui,
        )
        self.mix_chk.grid(row=4, column=0, columnspan=4, sticky='w', pady=3)
        Tooltip(self.mix_chk, TIPS['mixture'])

        self.cls_sdr_tab.columnconfigure(1, weight=1)
        self._sdr_thresh_frame = ctk.CTkFrame(self.cls_sdr_tab, fg_color='transparent')
        self._sdr_thresh_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        self._sdr_thresh_frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(
            self.cls_sdr_tab,
            text='Threshold (deletes anything below stated value)',
            font=_ui,
            text_color=DARK['text_dim'],
        ).grid(row=1, column=0, columnspan=2, sticky='w', pady=(6, 4))
        sdr_del_chk = ctk.CTkCheckBox(
            self.cls_sdr_tab,
            text='Delete folder if any expected stem is missing after SI-SDR determination',
            variable=self.sdr_delete_folder,
            onvalue=True, offvalue=False,
            fg_color=DARK['accent'],
            hover_color=DARK['accent_hover'],
            text_color=DARK['text'],
            border_color=DARK['border'],
            checkmark_color='#ffffff',
            font=_ui,
        )
        sdr_del_chk.grid(row=2, column=0, columnspan=2, sticky='w', pady=(0, 0))
        Tooltip(sdr_del_chk, TIPS['sdr_delete_folder'])
        self.stem_mode.trace_add('write', lambda *_: self._rebuild_sdr_thresholds())
        self._rebuild_sdr_thresholds()

        self.quality.trace_add('write', lambda *_: self._update_mixture_state())
        self._update_mixture_state()
        self._update_filter_state()

        _log_wrap = ctk.CTkFrame(
            right,
            fg_color=DARK['panel'],
            border_color=DARK['border'],
            border_width=1,
            corner_radius=8,
        )
        _log_wrap.grid(row=0, column=0, sticky='nsew',
                       pady=(HEADER_TOP_PAD, LOG_PAD_BOTTOM),
                       padx=(0, SECTION_SIDE_PAD_LEFT))
        _log_wrap.rowconfigure(1, weight=1)
        _log_wrap.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            _log_wrap, text='LOG',
            font=ctk_section_font(),
            text_color=DARK['text_dim'], anchor='w',
        ).grid(row=0, column=0, columnspan=2, sticky='w',
               padx=LOG_INNER_PAD, pady=(LOG_INNER_PAD - 4, 4))
        log_frame = ctk.CTkFrame(_log_wrap, fg_color='transparent')
        log_frame.grid(row=1, column=0, columnspan=2, sticky='nsew',
                       padx=LOG_INNER_PAD, pady=(0, LOG_INNER_PAD))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, wrap='word', state='disabled',
                                background=COLORS['log_bg'], foreground=COLORS['log_fg'],
                                insertbackground=COLORS['fg'], relief='flat', borderwidth=0,
                                font=LOG_FONT, padx=10, pady=8)
        self.log_text.grid(row=0, column=0, sticky='nsew')
        # Same CTk thumb style as IGNORE WHEN MATCHING (border / accent hover).
        # Shown only when content overflows (see _on_log_scroll).
        scroll = ctk.CTkScrollbar(
            log_frame,
            command=self.log_text.yview,
            fg_color=COLORS['log_bg'],
            button_color=DARK['scrollbar'],
            button_hover_color=DARK['scrollbar_hover'],
        )
        self._log_scrollbar = scroll
        self.log_text.configure(yscrollcommand=self._on_log_scroll)
        bind_mousewheel(self.log_text, self.log_text.yview)
        bind_mousewheel(log_frame, self.log_text.yview)
        for tag, color in (('err', '#ff7a7a'), ('warn', LOG_WARN_COLOR),
                           ('ok', '#7ee0a0'), ('info', COLORS['fg_dim']),
                           ('detail', COLORS['log_fg']),
                           ('sdr_pass', SDR_PASS_COLOR), ('sdr_fail', SDR_FAIL_COLOR),
                           ('sdr_label', SDR_LABEL_COLOR),
                           # Genre & Gender result blocks
                           ('gg_file', COLORS['fg_dim']),
                           ('gg_key', COLORS['fg_dim']),
                           ('gg_val', COLORS['log_fg']),
                           ('gg_conf', '#7ee0a0'),
                           ('gg_conf_low', '#e0b07a'),
                           ('gg_result', COLORS['log_fg'])):
            self.log_text.tag_configure(tag, foreground=color)
        self.log_text.tag_configure(
            'log_pct',
            foreground=COLORS['fg_dim'],
            font=LOG_PCT_FONT,
        )
        self.log_text.tag_configure('deleted', foreground=LOG_DELETED_COLOR)
        self.log_text.tag_configure('log_margin', foreground=LOG_MARGIN_COLOR)
        _configure_stem_log_tags(self.log_text)
        self.log_text.tag_configure(
            LOG_STEM_GAP_TAG,
            font=LOG_STEM_GAP_FONT,
            foreground=COLORS['log_bg'],
            background=COLORS['log_bg'],
        )
        self.log_text.tag_configure(
            LOG_FOLDER_STEM_GAP_TAG,
            font=LOG_FOLDER_STEM_GAP_FONT,
            foreground=COLORS['log_bg'],
            background=COLORS['log_bg'],
        )

        bottom = ctk.CTkFrame(self, fg_color='transparent')
        bottom.grid(row=bottom_row, column=0, sticky='ew',
                    padx=CONTENT_PAD, pady=(STATUS_TOP_PAD, STATUS_BOTTOM_PAD))
        bottom.columnconfigure(0, weight=1)

        self.status_frame = tk.Frame(
            bottom, bg=COLORS['panel'], height=STATUS_FRAME_HEIGHT,
            highlightthickness=0, bd=0,
        )
        self.status_frame.grid(row=0, column=0, sticky='ew')
        self.status_frame.grid_propagate(False)
        self.status_frame.pack_propagate(False)
        self.status_frame.columnconfigure(0, weight=1)

        self.status_idle_left = tk.Label(
            self.status_frame, textvariable=self.status_var,
            bg=COLORS['panel'], fg=COLORS['fg_dim'], font=STATUS_FONT,
            padx=STATUS_PAD_X, pady=0,
        )
        self.status_idle_left.place(
            relx=0, rely=0, anchor='nw', x=0, y=STATUS_IDLE_Y,
        )

        self.status_idle_right = tk.Label(
            self.status_frame, textvariable=self.device_var,
            bg=COLORS['panel'], fg=COLORS['fg_dim'], font=STATUS_FONT,
            padx=STATUS_PAD_X, pady=0,
        )
        self.status_idle_right.place(
            relx=1.0, rely=0, anchor='ne', x=0, y=STATUS_IDLE_Y,
        )

        self.status_center = tk.Frame(self.status_frame, bg=COLORS['panel'])
        self.status_credit = tk.Label(
            self.status_center,
            text=f'v{APP_VERSION}  by Gilliaan & Bas Curtiz',
            bg=COLORS['panel'],
            fg=COLORS['fg_dim'],
            font=STATUS_FONT,
            cursor='hand2',
        )
        self.status_credit.pack()
        self.status_credit.bind(
            '<Button-1>', lambda _e: webbrowser.open(STATUS_LINK_URL),
        )
        self.status_credit.bind(
            '<Enter>', lambda _e: self.status_credit.configure(fg=COLORS['fg']),
        )
        self.status_credit.bind(
            '<Leave>', lambda _e: self.status_credit.configure(fg=COLORS['fg_dim']),
        )
        self.status_center.place(
            relx=0.5, rely=0, anchor='n', y=STATUS_IDLE_Y,
        )
        Tooltip(self.status_credit, TIPS['status_link'])

        self.status_run = tk.Frame(
            self.status_frame, bg=COLORS['panel'],
            height=STATUS_FRAME_HEIGHT, highlightthickness=0, bd=0,
        )
        self.status_run.pack_propagate(False)
        self.status_run.columnconfigure(1, weight=1)
        for row_idx in range(5):
            self.status_run.rowconfigure(row_idx, weight=0)
        self.status_run.rowconfigure(1, minsize=RESOURCE_ROW_HEIGHT)
        self.status_run.rowconfigure(3, minsize=STATUS_PROGRESS_ROW_HEIGHT)

        self._status_pad_top = tk.Frame(
            self.status_run, bg=COLORS['panel'], height=STATUS_PAD_TOP,
        )
        self._status_pad_top.grid(row=0, column=0, columnspan=3, sticky='ew')
        self._status_pad_top.grid_propagate(False)

        self._status_row_gap = tk.Frame(
            self.status_run, bg=COLORS['panel'], height=STATUS_ROW_GAP,
        )
        self._status_row_gap.grid(row=2, column=0, columnspan=3, sticky='ew')
        self._status_row_gap.grid_propagate(False)

        self._status_pad_bottom = tk.Frame(
            self.status_run, bg=COLORS['panel'], height=STATUS_PAD_BOTTOM,
        )
        self._status_pad_bottom.grid(row=4, column=0, columnspan=3, sticky='ew')
        self._status_pad_bottom.grid_propagate(False)

        self.status_elapsed_lbl = tk.Label(
            self.status_run, textvariable=self.elapsed_var,
            bg=COLORS['panel'], fg=COLORS['fg_dim'], font=STATUS_FONT, anchor='w',
        )
        self.status_elapsed_lbl.grid(
            row=3, column=0, sticky='wn',
            padx=(STATUS_PAD_X, 10), pady=0,
        )

        self._progress_trough = tk.Frame(
            self.status_run, bg=COLORS['status_trough'],
            height=STATUS_PROGRESS_HEIGHT, highlightthickness=0, bd=0,
        )
        self._progress_trough.grid(
            row=3, column=1, sticky='ew', pady=(STATUS_PROGRESS_Y_PAD, 0),
        )
        self._progress_trough.grid_propagate(False)
        self._progress_trough.bind('<Configure>', self._on_progress_trough_resize)

        self._progress_fill = tk.Frame(self._progress_trough, bg=COLORS['accent'], highlightthickness=0, bd=0)
        self._progress_pct_lbl = tk.Label(
            self._progress_fill, text='0%',
            bg=COLORS['accent'], fg=COLORS['status_pct'],
            font=STATUS_PCT_FONT, anchor='e',
        )
        self._progress_pct_lbl.pack(side='right', padx=(0, 8))

        self.status_eta_lbl = tk.Label(
            self.status_run, textvariable=self.eta_var,
            bg=COLORS['panel'], fg=COLORS['fg_dim'], font=STATUS_FONT, anchor='e',
        )
        self.status_eta_lbl.grid(
            row=3, column=2, sticky='ne',
            padx=(10, STATUS_PAD_X), pady=0,
        )

        self._build_resource_bars(self.status_run)
        self._on_mode_tab_changed()

    def _build_resource_bars(self, parent: tk.Misc) -> None:
        row = tk.Frame(parent, bg=COLORS['panel'], height=RESOURCE_ROW_HEIGHT)
        row.grid(row=1, column=0, columnspan=3, sticky='ew', padx=STATUS_PAD_X, pady=0)
        row.grid_propagate(False)
        self._resource_bars: dict[str, dict] = {}
        specs = (
            ('cpu', 'CPU:'),
            ('gpu', 'GPU:'),
            ('ram', 'RAM:'),
            ('disk_read', 'HDD (r):'),
            ('disk_write', 'HDD (w):'),
        )
        n = len(specs)
        for idx, (key, label) in enumerate(specs):
            relx = idx / (n - 1) if n > 1 else 0.0
            if idx == 0:
                anchor = 'nw'
            elif idx == n - 1:
                anchor = 'ne'
            else:
                anchor = 'n'
            cell = tk.Frame(row, bg=COLORS['panel'])
            cell.place(relx=relx, rely=0, anchor=anchor)
            tk.Label(
                cell, text=label, bg=COLORS['panel'], fg=COLORS['fg_dim'],
                font=STATUS_FONT, anchor='w',
            ).pack(side='left')
            trough = tk.Frame(
                cell, bg=COLORS['status_trough'], width=RESOURCE_BAR_WIDTH,
                height=RESOURCE_BAR_HEIGHT, highlightthickness=0, bd=0,
            )
            trough.pack(side='left', padx=(4, 4))
            trough.pack_propagate(False)
            fill = tk.Frame(trough, bg=COLORS['accent'], height=RESOURCE_BAR_HEIGHT, highlightthickness=0, bd=0)
            pct_lbl = tk.Label(
                cell, text='0%', bg=COLORS['panel'], fg=COLORS['fg'],
                font=STATUS_FONT, width=4, anchor='w',
            )
            pct_lbl.pack(side='left')
            self._resource_bars[key] = {'trough': trough, 'fill': fill, 'pct': pct_lbl}
        self._resource_row = row
        self._resource_row.grid_remove()
        self._resource_visible = False

    def _set_resource_bar(self, key: str, pct: float) -> None:
        widgets = self._resource_bars.get(key)
        if widgets is None:
            return
        pct = max(0.0, min(100.0, float(pct)))
        widgets['pct'].configure(text=f'{pct:.0f}%')
        trough = widgets['trough']
        fill = widgets['fill']
        fill_w = max(0, int(round(RESOURCE_BAR_WIDTH * pct / 100.0)))
        fill.place(x=0, y=0, width=fill_w, height=RESOURCE_BAR_HEIGHT)

    def _cancel_resource_tick(self) -> None:
        if self._resource_tick_id is not None:
            try:
                self.after_cancel(self._resource_tick_id)
            except ValueError:
                pass
            self._resource_tick_id = None

    def _start_resource_monitor(self) -> None:
        from resource_monitor import ResourceMonitor

        self._cancel_resource_tick()
        if self._resource_monitor is None:
            self._resource_monitor = ResourceMonitor()
        if not self._resource_visible:
            self._resource_row.grid()
            self._resource_visible = True
        self._sample_resources()
        self._resource_tick_id = self.after(1000, self._tick_resources)

    def _sample_resources(self) -> None:
        if self._resource_monitor is None:
            return
        snap = self._resource_monitor.sample()
        self._set_resource_bar('cpu', snap.cpu)
        self._set_resource_bar('gpu', snap.gpu)
        self._set_resource_bar('ram', snap.ram)
        self._set_resource_bar('disk_read', snap.disk_read)
        self._set_resource_bar('disk_write', snap.disk_write)

    def _stop_resource_monitor(self) -> None:
        self._cancel_resource_tick()
        if getattr(self, '_resource_visible', False):
            self._resource_row.grid_remove()
            self._resource_visible = False
        for key in self._resource_bars:
            self._set_resource_bar(key, 0.0)

    def _status_work_active(self) -> bool:
        return (
            (self.worker is not None and self.worker.is_alive())
            or (self.sdr_worker is not None and self.sdr_worker.is_alive())
            or self._pair_busy
            or getattr(self, '_rename_busy', False)
        )

    def _tick_resources(self) -> None:
        self._resource_tick_id = None
        if not self._status_work_active():
            self._stop_resource_monitor()
            return
        self._sample_resources()
        self._resource_tick_id = self.after(1000, self._tick_resources)

    def _maybe_show_status_idle(self) -> None:
        if self._status_work_active():
            return
        self._show_status_idle()

    def _set_rename_busy(
        self,
        busy: bool,
        status: str = '',
        *,
        pct: float | None = None,
        eta: float | None = None,
    ) -> None:
        """Drive bottom status bar (progress / ETA / CPU·GPU) from Rename tab."""
        was_busy = bool(getattr(self, '_rename_busy', False))
        self._rename_busy = bool(busy)
        if status:
            self.status_var.set(status)
        if busy:
            if not was_busy:
                self._show_status_progress()
            if pct is not None:
                self._update_progress(float(pct), eta)
        else:
            if was_busy:
                self._update_progress(100.0, None)
                self.after(300, self._maybe_show_status_idle)

    def _rebuild_sdr_thresholds(self) -> None:
        saved = {
            k: _safe_tk_int(v, SDR_DEFAULT_THRESHOLDS.get(k, 30))
            for k, v in self.sdr_thresholds.items()
        }
        settings_saved = getattr(self, '_sdr_thresholds_settings', {})
        for w in self._sdr_thresh_frame.winfo_children():
            w.destroy()
        mode_cfg = STEM_MODES[self.stem_mode.get()]
        self.sdr_thresholds = {}
        for i, cat in enumerate(mode_cfg['categories']):
            default = SDR_DEFAULT_THRESHOLDS.get(cat, 30)
            if cat in saved:
                default = saved[cat]
            elif cat in settings_saved:
                default = settings_saved[cat]
            var = tk.IntVar(value=int(default))
            self.sdr_thresholds[cat] = var
            ctk = ensure_ctk_dark()
            _ui = ctk_ui_font()
            lbl = ctk.CTkLabel(
                self._sdr_thresh_frame, text=f'{str(cat).capitalize()}:',
                text_color=DARK['label'], font=_ui,
            )
            lbl.grid(row=i, column=0, sticky='w', padx=(0, 10), pady=2)
            ctrl = ctk.CTkFrame(self._sdr_thresh_frame, fg_color='transparent')
            ctrl.grid(row=i, column=1, sticky='w', pady=2)
            sp = ttk.Spinbox(
                ctrl, from_=0, to=100, textvariable=var, width=6,
                font=ttk_ui_font(),
            )
            sp.pack(side='left')
            ctk.CTkLabel(
                ctrl, text='SI-SDR', text_color=DARK['text'], font=_ui,
            ).pack(side='left', padx=(8, 0))
            tip(lbl, sp, text=TIPS['sdr_threshold'])
            if not getattr(self, '_loading_settings', False):
                sp.bind('<FocusOut>', self._autosave_sdr_threshold, add='+')
                sp.bind('<Return>', self._autosave_sdr_threshold, add='+')
        self._refresh_cls_frame()

    def _refresh_cls_frame(self) -> None:
        """Re-sync frame after classification tab content changes."""
        if not hasattr(self, 'cls_frame'):
            return
        self.cls_frame.update_idletasks()

    def _autosave_sdr_threshold(self, _event=None) -> None:
        if getattr(self, '_loading_settings', False):
            return
        for var in self.sdr_thresholds.values():
            try:
                int(var.get())
            except (tk.TclError, ValueError, TypeError):
                return
        self._save_settings()

    def _on_progress_trough_resize(self, _event=None) -> None:
        self._redraw_progress_fill()

    def _redraw_progress_fill(self) -> None:
        if not hasattr(self, '_progress_trough'):
            return
        self._progress_trough.update_idletasks()
        w = self._progress_trough.winfo_width()
        if w < 2:
            self.after(50, self._redraw_progress_fill)
            return
        h = self._progress_trough.winfo_height()
        fill_w = max(0, min(w, int(round(w * self._progress_pct_value / 100.0))))
        self._progress_fill.place(x=0, y=0, width=fill_w, height=h)

    def _cancel_progress_tick(self) -> None:
        if self._progress_tick_id is not None:
            try:
                self.after_cancel(self._progress_tick_id)
            except ValueError:
                pass
            self._progress_tick_id = None

    def _tick_status_clock(self) -> None:
        self._progress_tick_id = None
        if not self._progress_started_at:
            return
        if self._stopping:
            self.elapsed_var.set('Stopping…')
        else:
            elapsed = time.monotonic() - self._progress_started_at
            self.elapsed_var.set(f'Elapsed: {format_status_clock(elapsed)}')
        self._progress_tick_id = self.after(250, self._tick_status_clock)

    def _place_status_idle(self) -> None:
        self.status_idle_left.place(
            relx=0, rely=0, anchor='nw', x=0, y=STATUS_IDLE_Y,
        )
        self.status_idle_right.place(
            relx=1.0, rely=0, anchor='ne', x=0, y=STATUS_IDLE_Y,
        )
        self.status_center.place(
            relx=0.5, rely=0, anchor='n', y=STATUS_IDLE_Y,
        )

    def _show_status_idle(self) -> None:
        self._cancel_progress_tick()
        self._stop_resource_monitor()
        self._progress_started_at = 0.0
        self.status_run.place_forget()
        self._place_status_idle()
        self.status_var.set('Idle')
        self._stopping = False

    def _show_status_progress(self) -> None:
        self.status_idle_left.place_forget()
        self.status_idle_right.place_forget()
        self.status_center.place_forget()
        self.status_run.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._stopping = False
        self._progress_pct_value = 0.0
        self._progress_started_at = time.monotonic()
        self._progress_pct_lbl.configure(text='0%')
        self.elapsed_var.set('Elapsed: 0:00:00')
        self.eta_var.set('ETA --:--:--')
        self._cancel_progress_tick()
        self._tick_status_clock()
        self._start_resource_monitor()
        self.after(50, self._redraw_progress_fill)

    def _update_progress(
        self,
        pct: float,
        eta: float | None,
        n: int | None = None,
        total: int | None = None,
        phase: str | None = None,
    ) -> None:
        self._progress_pct_value = float(pct)
        # Always show percent on the bar (n/total/phase stay optional for callers).
        self._progress_pct_lbl.configure(text=f'{pct:.0f}%')
        if n is not None and total is not None and int(total) > 0:
            detail = f'{int(n):,}/{int(total):,}'
            if phase:
                detail = f'{phase} {detail}'
            self.status_var.set(detail)
        self.eta_var.set(f'ETA {format_eta(eta)}')
        self._redraw_progress_fill()

    def _update_mixture_state(self):
        is_wav = self.quality.get().startswith('WAV')
        self.mix_chk.configure(state='normal' if is_wav else 'disabled')
        if not is_wav:
            self.make_mixture.set(False)

    def _update_filter_state(self):
        state = 'normal' if self.delete_if_short.get() else 'disabled'
        self.min_dur_sp.configure(state=state)

    def _settings_snapshot(self) -> dict:
        snap = {
            'input_dir':    display_path(self.input_dir.get()),
            'output_dir':   display_path(self.output_dir.get()),
            'use_cuda':     self.use_cuda.get(),
            'model_label':  self.model_label.get(),
            'stem_mode':    self.stem_mode.get(),
            'quality':      self.quality.get(),
            'threshold':    _safe_tk_float(self.threshold, 0.40),
            'min_margin':   _safe_tk_float(self.min_margin, 0.20),
            'batch_size':   _safe_tk_int(self.batch_size, 4),
            'peak_norm':    self.peak_norm.get(),
            'make_mixture': self.make_mixture.get(),
            'dedup':        self.dedup.get(),
            'ambig_label':  self.ambig_label.get(),
            'scan_label':   self.scan_label.get(),
            'naming_label': self.naming_label.get(),
            'append_duration': False,
            'delete_if_short': self.delete_if_short.get(),
            'min_duration_sec': _safe_tk_int(self.min_duration_sec, 8),
            'delete_if_incomplete': self.delete_if_incomplete.get(),
            'skip_existing': self.skip_existing.get(),
            'sdr_delete_folder': self.sdr_delete_folder.get(),
            'sdr_thresholds': {
                k: _safe_tk_int(v, SDR_DEFAULT_THRESHOLDS.get(k, 30))
                for k, v in self.sdr_thresholds.items()
            },
            'hide_device_notice': self.hide_device_notice,
        }
        if hasattr(self, 'pair_panel'):
            snap.update(self.pair_panel.settings_snapshot())
        return snap

    def _load_settings(self) -> None:
        data = load_settings()
        if not data:
            return

        self._loading_settings = True
        try:
            self.input_dir.set(display_path(data.get('input_dir', '')))
            self.output_dir.set(display_path(data.get('output_dir', '')))
            if cuda_effective():
                self.use_cuda.set(bool(data.get('use_cuda', True)))
            self.model_label.set(_valid_label(data.get('model_label'), MODELS, next(iter(MODELS))))
            stem_mode = data.get('stem_mode')
            if stem_mode in ('Vocals + Instrumental', '2-way (Instrumental/Vocals)'):
                stem_mode = '2-way (instrumental/vocals)'
            if stem_mode == '4-way (drums/bass/other/vocals)':
                stem_mode = '4-way (bass/drums/other/vocals)'
            self.stem_mode.set(_valid_label(stem_mode, STEM_MODES, next(iter(STEM_MODES))))
            self.quality.set(_valid_label(data.get('quality'), QUALITY_PRESETS, 'FLAC 16-bit'))
            self.threshold.set(float(data.get('threshold', self.threshold.get())))
            self.min_margin.set(float(data.get('min_margin', self.min_margin.get())))
            self.batch_size.set(int(data.get('batch_size', _safe_tk_int(self.batch_size, 4))))
            self.peak_norm.set(bool(data.get('peak_norm', self.peak_norm.get())))
            self.make_mixture.set(bool(data.get('make_mixture', self.make_mixture.get())))
            self.dedup.set(bool(data.get('dedup', self.dedup.get())))
            self.ambig_label.set(_valid_label(data.get('ambig_label'), AMBIG_MODES, next(iter(AMBIG_MODES))))
            self.scan_label.set(_valid_label(data.get('scan_label'), SCAN_MODES, next(iter(SCAN_MODES))))
            self.naming_label.set(_valid_label(data.get('naming_label'), NAMING_MODES, next(iter(NAMING_MODES))))
            self.append_duration.set(False)
            self.delete_if_short.set(bool(data.get('delete_if_short', self.delete_if_short.get())))
            self.min_duration_sec.set(int(data.get('min_duration_sec', _safe_tk_int(self.min_duration_sec, 8))))
            self.delete_if_incomplete.set(
                bool(data.get('delete_if_incomplete', self.delete_if_incomplete.get()))
            )
            self.skip_existing.set(bool(data.get('skip_existing', self.skip_existing.get())))
            self.sdr_delete_folder.set(
                bool(data.get('sdr_delete_folder', self.sdr_delete_folder.get()))
            )
            sdr_thr = data.get('sdr_thresholds')
            if isinstance(sdr_thr, dict):
                self._sdr_thresholds_settings = {
                    k: int(v) for k, v in sdr_thr.items()
                    if k in SDR_DEFAULT_THRESHOLDS
                }
            else:
                self._sdr_thresholds_settings = {}
            self._rebuild_sdr_thresholds()
            self._update_mixture_state()
            self._update_filter_state()
            self.hide_device_notice = bool(data.get('hide_device_notice', False))
            if hasattr(self, 'pair_panel'):
                self.pair_panel._load_settings()
        finally:
            self._loading_settings = False

    def _bind_settings_autosave(self) -> None:
        self._loading_settings = False

        def _autosave(*_):
            if self._loading_settings:
                return
            for var in (
                self.threshold, self.min_margin, self.batch_size, self.min_duration_sec,
            ):
                if not _tk_numeric_var_ready(var):
                    return
            self._save_settings()

        for var in (
            self.input_dir, self.output_dir, self.use_cuda, self.model_label,
            self.stem_mode, self.quality, self.threshold, self.min_margin,
            self.batch_size, self.peak_norm, self.make_mixture, self.dedup,
            self.ambig_label, self.scan_label, self.naming_label,
            self.delete_if_short, self.min_duration_sec, self.delete_if_incomplete, self.skip_existing,
            self.sdr_delete_folder,
        ):
            var.trace_add('write', _autosave)

    def _save_settings(self) -> None:
        save_settings(self._settings_snapshot())

    def _on_close(self) -> None:
        if self._renamer_destructive_busy():
            messagebox.showwarning(
                'Rename in progress',
                'Files are currently being renamed or moved.\n\n'
                'Wait for the operation to finish before closing.',
                parent=self,
            )
            return
        self._save_settings()
        self._stop_resource_monitor()
        if self._resource_monitor is not None:
            self._resource_monitor.close()
            self._resource_monitor = None
        if self.worker and self.worker.is_alive():
            self.worker.stop()
        if self.sdr_worker and self.sdr_worker.is_alive():
            self.sdr_worker.stop()
        if getattr(self, '_minimize_restore_job', None) is not None:
            try:
                self.after_cancel(self._minimize_restore_job)
            except tk.TclError:
                pass
            self._minimize_restore_job = None
        panel = getattr(self, 'renamer_panel', None)
        if panel is not None:
            panel.shutdown()
        self.destroy()

    def _open_folder(self, var: tk.StringVar, label: str) -> None:
        path = var.get().strip()
        if not path:
            messagebox.showinfo('Open folder', f'No {label} path set.')
            return
        target = Path(path)
        if not target.exists():
            messagebox.showerror('Open folder', f'Path does not exist:\n{path}')
            return
        try:
            if sys.platform == 'win32':
                os.startfile(str(target))
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(target)], check=False)
            else:
                subprocess.run(['xdg-open', str(target)], check=False)
        except OSError as e:
            messagebox.showerror('Open folder', str(e))

    def _open_input(self) -> None:
        self._open_folder(self.input_dir, 'input')

    def _open_output(self) -> None:
        self._open_folder(self.output_dir, 'output')

    def _show_about_dialog(self) -> None:
        if not hasattr(self, '_about_icon') or self._about_icon is None:
            self._about_icon = load_about_icon(self)
        show_about_dialog(self, self._about_icon)

    def _pick_input(self):
        d = filedialog.askdirectory(title='Select input directory',
                                    initialdir=self.input_dir.get() or None)
        if d:
            self.input_dir.set(display_path(d))
            if not self.output_dir.get():
                self.output_dir.set(display_path(
                    str(Path(d).parent / (Path(d).name + '_organized'))
                ))
            self._save_settings()

    def _pick_output(self):
        d = filedialog.askdirectory(title='Select output directory',
                                    initialdir=self.output_dir.get() or None)
        if d:
            self.output_dir.set(display_path(d))
            self._save_settings()

    def _start(self):
        if self._pair_busy or self._renamer_destructive_busy():
            return
        if self.worker and self.worker.is_alive():
            return
        if self.sdr_worker and self.sdr_worker.is_alive():
            return
        self.worker = None
        if not self.input_dir.get() or not os.path.isdir(self.input_dir.get()):
            messagebox.showerror('Missing input', 'Please select a valid input directory.')
            return
        if not self.output_dir.get():
            messagebox.showerror('Missing output', 'Please select an output directory.')
            return
        self._normalize_path_var(self.input_dir)
        self._normalize_path_var(self.output_dir)
        self._save_settings()
        self._worker_kind = 'rms'
        self._rms_saw_done = False
        self._sdr_use_output_dir = False
        params = {
            'input_dir':    display_path(self.input_dir.get()),
            'output_dir':   display_path(self.output_dir.get()),
            'use_cuda':     self.use_cuda.get(),
            'model_id':     MODELS[self.model_label.get()],
            'stem_mode':    self.stem_mode.get(),
            'quality':      self.quality.get(),
            'threshold':    _safe_tk_float(self.threshold, 0.40),
            'min_margin':   _safe_tk_float(self.min_margin, 0.20),
            'batch_size':   _safe_tk_int(self.batch_size, 4),
            'peak_norm':    self.peak_norm.get(),
            'make_mixture': self.make_mixture.get(),
            'dedup':        self.dedup.get(),
            'ambig_mode':   AMBIG_MODES[self.ambig_label.get()],
            'scan_mode':    SCAN_MODES[self.scan_label.get()],
            'naming_mode':  NAMING_MODES[self.naming_label.get()],
            'append_duration': False,
            'delete_if_short': self.delete_if_short.get(),
            'min_duration_sec': _safe_tk_int(self.min_duration_sec, 8),
            'delete_if_incomplete': self.delete_if_incomplete.get(),
            'skip_existing': self.skip_existing.get(),
        }
        self._append_log('=== Starting job ===')
        for k, v in params.items():
            self._append_log(f"  {k}: {v}")
        self._set_action_buttons(running=True)
        self._show_status_progress()
        self.log_queue = queue.Queue()
        self.worker = Worker(params, self.log_queue)
        self.worker.start()

    def _start_sdr(self):
        if self._pair_busy or self._renamer_destructive_busy():
            return
        if self.sdr_worker and self.sdr_worker.is_alive():
            return
        if self.worker and self.worker.is_alive():
            return
        self.sdr_worker = None

        use_output = self._sdr_use_output_dir
        if use_output:
            target = display_path(self.output_dir.get().strip())
            label = 'output'
        else:
            target = display_path(self.input_dir.get().strip())
            label = 'input'

        if not target or not os.path.isdir(target):
            messagebox.showerror(
                'Missing folder',
                f'Please select a valid {label} directory for SI-SDR processing.',
            )
            return

        mode_cfg = STEM_MODES[self.stem_mode.get()]
        preferred = mode_cfg['categories']
        root = Path(target)
        scan_mode = SCAN_MODES[self.scan_label.get()]
        categories, layout = resolve_sdr_layout_and_categories(
            root, scan_mode, preferred,
        )
        process_all = False
        user_picked = False

        if layout is None or categories is None:
            n_audio = sum(1 for _ in iter_sdr_audio_files(root, scan_mode))
            if n_audio == 0:
                messagebox.showerror(
                    'Missing stems',
                    describe_sdr_scan_failure(root, scan_mode, preferred),
                    parent=self,
                )
                return
            chosen = ask_sdr_stem_category(self)
            if not chosen:
                return
            categories = (chosen,)
            layout = SDR_LAYOUT_SINGLE_FLAT
            process_all = True
            user_picked = True

        if layout == SDR_LAYOUT_SINGLE_FLAT and not user_picked:
            hint = build_single_stem_folder_hint(root, scan_mode, categories[0])
            if hint and hint['should_ask_process_all']:
                title = f'{str(hint["kind"]).capitalize()}-only folder?'
                if messagebox.askyesno(
                    title, single_stem_process_all_message(hint), parent=self,
                ):
                    process_all = True

        valid = collect_sdr_targets(
            root, categories, scan_mode, layout, process_all=process_all,
        )
        if not valid:
            messagebox.showerror(
                'Missing stems',
                describe_sdr_scan_failure(root, scan_mode, preferred),
                parent=self,
            )
            return

        self._save_settings()
        self._worker_kind = 'sdr'
        known_thresholds = {
            k: float(_safe_tk_int(v, SDR_DEFAULT_THRESHOLDS.get(k, 30)))
            for k, v in self.sdr_thresholds.items()
        }
        params = {
            'target_dir': target,
            'use_cuda': self.use_cuda.get(),
            'model_id': MODELS[self.model_label.get()],
            'stem_mode': self.stem_mode.get(),
            'scan_mode': SCAN_MODES[self.scan_label.get()],
            'sdr_categories': categories,
            'sdr_layout': layout,
            'sdr_flat_process_all': process_all,
            'sdr_user_picked_category': user_picked,
            'sdr_thresholds': sdr_thresholds_for_categories(categories, known_thresholds),
            'sdr_delete_folder': self.sdr_delete_folder.get(),
        }
        self._append_log('=== Starting SI-SDR job ===')
        for k, v in params.items():
            self._append_log(f"  {k}: {v}")
        self._set_action_buttons(running=True, sdr=True)
        self._show_status_progress()
        self.log_queue = queue.Queue()
        self.sdr_worker = SdrWorker(params, self.log_queue)
        self.sdr_worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self._append_log('[stopping] ...')
            self._stopping = True
            self._update_progress(self._progress_pct_value, None)

    def _stop_sdr(self):
        if self.sdr_worker:
            self.sdr_worker.stop()
            self._append_log('[stopping] ...')
            self._stopping = True
            self._update_progress(self._progress_pct_value, None)

    def _organize_worker_active(self) -> bool:
        return (
            (self.worker is not None and self.worker.is_alive())
            or (self.sdr_worker is not None and self.sdr_worker.is_alive())
            or self._renamer_destructive_busy()
            or self._pair_busy
            or getattr(self, '_rename_busy', False)
        )

    def _set_pair_busy(self, busy: bool, status: str, panel) -> None:
        self._pair_busy = busy
        self.status_var.set(status)
        panel.set_buttons_state('disabled' if busy else 'normal')
        if busy:
            if self._classify_mode_active():
                # Dim only — state=disabled shrinks CTk button height on Windows.
                for btn in (
                    self.start_btn, self.stop_btn,
                    self.save_log_btn, self.clear_log_btn, self.play_btn,
                ):
                    btn.configure(
                        text_color=DARK['text_dim'],
                        cursor='arrow',
                        height=PATH_BTN_HEIGHT,
                    )
            self._show_status_progress()
        else:
            if self._classify_mode_active():
                self._update_action_buttons_for_tab()
            self._update_progress(100.0, None)
            self.after(300, self._maybe_show_status_idle)
        self._pin_action_bar_heights()

    def _on_log_scroll(self, first, last) -> None:
        """Update LOG scrollbar; show only when content overflows."""
        sb = getattr(self, '_log_scrollbar', None)
        if sb is None:
            return
        try:
            first_f, last_f = float(first), float(last)
        except (TypeError, ValueError):
            return
        sb.set(first_f, last_f)
        need = (last_f - first_f) < 0.999
        try:
            mapped = bool(sb.winfo_ismapped())
        except tk.TclError:
            return
        if need and not mapped:
            sb.grid(row=0, column=1, sticky='ns')
        elif not need and mapped:
            sb.grid_forget()

    def _gg_insert_dim_pct(self, pct: str, *, indent: str = '  ') -> None:
        """Dim percentage — same color as === filename === (info / fg_dim)."""
        text = (pct or '').strip()
        if not text:
            return
        if not text.endswith('%'):
            text = f'{text}%'
        self.log_text.insert('end', indent)
        self.log_text.insert('end', text, 'log_pct')
        self.log_text.insert('end', '\n')
        self.log_text.insert('end', '\u200b\n', LOG_FOLDER_STEM_GAP_TAG)

    def _gg_flush_genre_style_row(
        self,
        genre: str | None,
        style: str | None,
        conf_pct: str | None = None,
    ) -> None:
        """Genre chip, style chip below, optional dim pct (Rename-style)."""
        _ensure_stem_chip_layout(self.log_text)
        genre = (genre or '').strip()
        style = (style or '').strip()
        conf = (conf_pct or '').strip()
        if not genre and not style and not conf:
            return
        if genre:
            self.log_text.insert('end', '  ')
            self.log_text.insert(
                'end',
                _format_gg_value_chip(genre),
                _stem_log_tag('dry'),
            )
            if conf:
                self.log_text.insert('end', f'  {conf}', 'log_pct')
                conf = ''
            self.log_text.insert('end', '\n')
            self.log_text.insert('end', '\u200b\n', LOG_FOLDER_STEM_GAP_TAG)
        if style:
            self.log_text.insert('end', '  ')
            self.log_text.insert(
                'end',
                _format_gg_value_chip(style),
                _stem_log_tag('wet'),
            )
            self.log_text.insert('end', '\n')
            self.log_text.insert('end', '\u200b\n', LOG_FOLDER_STEM_GAP_TAG)
        if conf:
            self._gg_insert_dim_pct(conf)

    def _gg_flush_pending_genre(self) -> None:
        pending = getattr(self, '_gg_pending_genre', None)
        style = getattr(self, '_gg_pending_style', None)
        if pending or style:
            self._gg_pending_genre = None
            self._gg_pending_style = None
            self._gg_flush_genre_style_row(pending, style)

    def _append_pair_log(self, message: str, tag: str = 'info') -> None:
        self.log_text.configure(state='normal')
        line = message or ''
        # Gender/reverb:  female 72%   /   dry 55%
        badge_m = GG_BADGE_RE.match(line)
        if badge_m:
            self._gg_flush_pending_genre()
            indent, label, pct = (
                badge_m.group(1), badge_m.group(2), badge_m.group(3),
            )
            self.log_text.insert('end', indent or '  ')
            self.log_text.insert(
                'end',
                _format_stem_chip_text(label),
                _stem_log_tag(label),
            )
            if pct:
                self.log_text.insert('end', f'  {pct}', 'log_pct')
            self.log_text.insert('end', '\n')
            # Same gap height as under === filename === (equal air top/bottom).
            self.log_text.insert('end', '\u200b\n', LOG_FOLDER_STEM_GAP_TAG)
        elif GG_HEADER_RE.match(line.strip()):
            self._gg_flush_pending_genre()
            # Same dim as "Starting genre tagger:" (info / fg_dim).
            self.log_text.insert('end', line.strip() + '\n', 'info')
            # Air before first badge — match gap after bottom badge.
            self.log_text.insert('end', '\u200b\n', LOG_FOLDER_STEM_GAP_TAG)
        else:
            # Genre: buffer GENRE + STYLE, then CONF / bare 72%.
            key_m = re.match(
                r'^(GENRE|STYLE|CONF|GENDER|REVERB):\s*(.*)$',
                line,
                flags=re.IGNORECASE,
            )
            if key_m:
                key = key_m.group(1).upper()
                val = (key_m.group(2) or '').strip()
                if key == 'GENRE' and val:
                    self._gg_pending_genre = val
                elif key == 'STYLE':
                    self._gg_pending_style = val
                elif key == 'CONF' and val:
                    genre = getattr(self, '_gg_pending_genre', None)
                    style = getattr(self, '_gg_pending_style', None)
                    self._gg_pending_genre = None
                    self._gg_pending_style = None
                    conf_val = val
                    try:
                        if not conf_val.endswith('%'):
                            conf_val = (
                                f"{int(round(float(conf_val) * 100.0))}%"
                            )
                    except ValueError:
                        pass
                    self._gg_flush_genre_style_row(genre, style, conf_val)
                else:
                    self._gg_flush_pending_genre()
                    # Fallback (unknown gender/reverb strings, …).
                    self.log_text.insert('end', f'{key}: ', 'gg_key')
                    self.log_text.insert('end', val + '\n', 'gg_val')
            else:
                pct_only = GG_PCT_ONLY_RE.match(line)
                conf_legacy = re.match(
                    r'^(\s*)\(confidence\s+([^)]+)\)\s*$',
                    line,
                    flags=re.IGNORECASE,
                )
                if pct_only or conf_legacy:
                    genre = getattr(self, '_gg_pending_genre', None)
                    style = getattr(self, '_gg_pending_style', None)
                    self._gg_pending_genre = None
                    self._gg_pending_style = None
                    if pct_only:
                        conf_val = pct_only.group(2)
                    else:
                        conf_val = (conf_legacy.group(2) or '').strip()
                        try:
                            if not conf_val.endswith('%'):
                                conf_val = (
                                    f"{int(round(float(conf_val) * 100.0))}%"
                                )
                        except ValueError:
                            pass
                    self._gg_flush_genre_style_row(genre, style, conf_val)
                else:
                    self._gg_flush_pending_genre()
                    allowed = {
                        'err', 'warn', 'ok', 'info', 'detail', 'log_pct',
                        'gg_file', 'gg_key', 'gg_val', 'gg_conf', 'gg_conf_low',
                        'gg_result',
                    }
                    use = tag if tag in allowed else 'info'
                    if line.strip() == 'DONE':
                        use = 'ok'
                    self.log_text.insert('end', line + '\n', use)
                    if line.strip() == 'DONE':
                        _play_done_sound()
        # Huge batch runs: trim oldest lines so the widget stays responsive.
        try:
            end_line = int(float(str(self.log_text.index('end-1c')).split('.')[0]))
            if end_line > 1500:
                self.log_text.delete('1.0', f'{end_line - 1200}.0')
        except (tk.TclError, ValueError, TypeError):
            pass
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _open_stem_player(self) -> None:
        from stem_player import open_stem_player

        open_stem_player(self)

    def _save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title='Save RMS log',
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
            initialfile=f'stem-organizer-log-{time.strftime("%Y%m%d-%H%M%S")}.txt',
        )
        if not path:
            return
        self.log_text.configure(state='normal')
        content = self.log_text.get('1.0', 'end-1c')
        self.log_text.configure(state='disabled')
        try:
            Path(path).write_text(content, encoding='utf-8')
        except OSError as e:
            messagebox.showerror('Save RMS log', str(e))

    def _clear_log(self) -> None:
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        # window_create chip labels are not always removed by delete() on Windows.
        for child in self.log_text.winfo_children():
            child.destroy()
        self.log_text.yview_moveto(0)
        self.log_text.configure(state='disabled')
        self._pending_stem_block_gap = False
        try:
            self.log_text.mark_unset('_gg_processed')
        except tk.TclError:
            pass

    def _gg_update_processed_line(self, n: int, total: int) -> None:
        """Update single Batch progress line in LOG (Processed: n/total)."""
        line = f'Processed: {int(n):,}/{int(total):,}'
        self.log_text.configure(state='normal')
        try:
            start = self.log_text.index('_gg_processed')
        except tk.TclError:
            start = None
        if start is not None:
            self.log_text.delete(start, f'{start} lineend')
            self.log_text.insert(start, line, 'info')
        else:
            self.log_text.insert('end', line, 'info')
            self.log_text.mark_set('_gg_processed', 'end-1c linestart')
            self.log_text.mark_gravity('_gg_processed', 'left')
            self.log_text.insert('end', '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _save_sdr_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title='Save SI-SDR log',
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
            initialfile=f'stem-organizer-sisdr-log-{time.strftime("%Y%m%d-%H%M%S")}.txt',
        )
        if not path:
            return
        self.log_text.configure(state='normal')
        content = self.log_text.get('1.0', 'end-1c')
        self.log_text.configure(state='disabled')
        try:
            Path(path).write_text(content, encoding='utf-8')
        except OSError as e:
            messagebox.showerror('Save SI-SDR log', str(e))

    def _offer_sdr_after_rms(self) -> None:
        self._sdr_use_output_dir = True
        self.cls_notebook.set('SI-SDR')
        # CTkTabview.set() does not fire command= — sync StringVar + bottom bar.
        self._class_tab.set('sdr')
        self._update_action_buttons_for_tab()
        self._refresh_cls_frame()
        # Quiet dialog — MessageBox.showinfo also plays SystemAsterisk and
        # would double the DONE sound from _append_log.
        self._show_info_quiet(
            'Calculate SI-SDR?',
            'RMS classification is complete.\n\n'
            'Check the SI-SDR thresholds and settings before you hit Start SI-SDR calc.',
        )

    def _show_info_quiet(self, title: str, message: str) -> None:
        """Modal info without Windows MessageBox system sound / light chrome."""
        from ui_theme import show_info_dark

        show_info_dark(self, title, message)

    def _job_finished(self):
        kind = self._worker_kind
        saw_rms_done = self._rms_saw_done
        if kind == 'sdr':
            self.sdr_worker = None
        else:
            self.worker = None
            self._rms_saw_done = False
        self._update_action_buttons_for_tab()
        self._show_status_idle()
        if kind == 'rms' and saw_rms_done:
            self._offer_sdr_after_rms()

    def _drain_log(self):
        try:
            while True:
                try:
                    msg = self.log_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if msg is DONE_SENTINEL:
                        self._job_finished()
                    elif isinstance(msg, tuple) and msg and msg[0] == PROGRESS_TAG:
                        pct = msg[1] if len(msg) > 1 else 0.0
                        eta = msg[2] if len(msg) > 2 else None
                        n = msg[3] if len(msg) > 3 else None
                        total = msg[4] if len(msg) > 4 else None
                        phase = msg[5] if len(msg) > 5 else None
                        self._update_progress(
                            float(pct),
                            eta,
                            n=n,
                            total=total,
                            phase=phase or None,
                        )
                    elif isinstance(msg, tuple) and len(msg) == 3 and msg[0] == PAIR_LOG_TAG:
                        self._append_pair_log(msg[1], msg[2])
                    elif isinstance(msg, tuple) and len(msg) == 3 and msg[0] == GG_PROCESSED_TAG:
                        self._gg_update_processed_line(msg[1], msg[2])
                    elif isinstance(msg, tuple) and len(msg) == 4 and msg[0] == SDR_LOG_TAG:
                        self._append_sdr_log_line(msg[1], msg[2], msg[3])
                    else:
                        self._append_log(msg)
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()
        self.after(100, self._drain_log)

    def _append_sdr_log_line(self, filename: str, score: float, threshold: float) -> None:
        self.log_text.configure(state='normal')
        self.log_text.insert('end', '  ')
        
        # Try to show the stem name as a chip if it matches a known category
        stem_name = Path(filename).stem.lower()
        if stem_name in LOG_STEM_COLORS or stem_name == 'instrumental':
            self.log_text.insert('end', '  ')
            self.log_text.insert('end', _format_stem_chip_text(stem_name), _stem_log_tag(stem_name))
            self.log_text.insert('end', '  →  ')
        else:
            self.log_text.insert('end', f'{filename}  →  ')

        self.log_text.insert('end', 'SI-SDR: ', 'sdr_label')
        score_str = f'{score:.1f}'
        score_tag = 'sdr_pass' if score >= threshold else 'sdr_fail'
        self.log_text.insert('end', score_str, score_tag)
        self.log_text.insert('end', '\n')
        self.log_text.insert('end', '\u200b\n', LOG_STEM_GAP_TAG)
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _append_log(self, msg: str):
        line = msg.rstrip()
        self.log_text.configure(state='normal')
        if '[deleted]' in line or '[delete error]' in line:
            self.log_text.insert('end', line + '\n', 'deleted')
        elif m := STEM_CLASSIFY_RE.match(line):
            if self._pending_stem_block_gap:
                self.log_text.insert('end', '\u200b\n', LOG_FOLDER_STEM_GAP_TAG)
                self._pending_stem_block_gap = False
            indent, label, pct, suffix = (
                m.group(1), m.group(2), m.group(3), m.group(4),
            )
            self.log_text.insert('end', indent)
            self.log_text.insert('end', _format_stem_chip_text(label), _stem_log_tag(label))
            if pct:
                self.log_text.insert('end', f'  {pct}', 'log_pct')
            self.log_text.insert('end', suffix)
            self.log_text.insert('end', '\n')
            self.log_text.insert('end', '\u200b\n', LOG_STEM_GAP_TAG)
        else:
            s = line.strip()
            low = line.lower()
            if '[error]' in low:
                tag = 'err'
            elif s == 'DONE':
                tag = 'ok'
            elif '[warn]' in low or 'cuda oom' in low:
                tag = 'warn'
            elif s.startswith('Done') or '    wrote ' in line or line.lstrip().startswith('wrote '):
                tag = 'ok'
            elif line.startswith('  Successful'):
                tag = 'ok'
            elif line.startswith(('  Deleted', '  Delete failed')):
                tag = 'deleted'
            elif line.startswith(('  Skipped', '  Not processed')):
                tag = 'warn'
            elif line.startswith(('  Total time:', '  Avg per folder:', '  Stems skipped', '  Files:', '  Sec/file:', '  Files/min:', '  Peak VRAM:', '  Results:', '  Tagged:')):
                tag = 'info'
            elif re.match(r'^=== .+ Summary ===\s*$', s):
                tag = 'info'
            elif line.startswith('=== SI-SDR Summary') or line.startswith('=== RMS Summary'):
                tag = 'info'
            elif line.startswith('  Passed:'):
                tag = 'ok'
            elif line.startswith(('  Deleted (whole folder):', '  Deleted (stem file):')):
                tag = 'deleted'
            elif line.startswith('  Skipped (incomplete stems):'):
                tag = 'warn'
            elif line.startswith('      '):
                tag = 'warn'
            elif line.startswith('    ') and ':' in line:
                tag = 'warn'
            elif line.startswith('    '):
                tag = 'deleted'
            elif '[skip existing]' in line:
                tag = 'warn'
            elif s.startswith('===') or s.startswith('['):
                tag = 'info'
            else:
                # Intro / status chatter — same dim as === filename === (fg_dim).
                tag = 'info'
            self.log_text.insert('end', line + '\n', (tag,))
            if line.strip() in STEM_BLOCK_GAP_AFTER:
                self._pending_stem_block_gap = True
            if FOLDER_TITLE_RE.match(line):
                self.log_text.insert('end', '\u200b\n', LOG_FOLDER_STEM_GAP_TAG)
        self.log_text.see('end')
        self.log_text.configure(state='disabled')
        if s := line.strip():
            if s in ('DONE', 'Done.'):
                self._rms_saw_done = True
                _play_done_sound()
            elif s == SDR_DONE_LINE and SDR_DONE_LINE != 'DONE':
                _play_done_sound()


def _startup_tasks(set_status) -> None:
    from deps_bootstrap import ensure_ml_deps

    if not ensure_ml_deps(show_dialog=False, set_status=set_status):
        raise RuntimeError('Missing Python packages. Run install-deps.bat beside this app.')

    set_status('Initializing application…')
    _init_ml()
    set_status('Preparing interface…')
    time.sleep(0.15)


def main():
    from single_instance import acquire_single_instance

    if not acquire_single_instance():
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(
            'Already running',
            'STEM organizer is already open.\n\n'
            'Close the existing window before starting another copy.',
            parent=root,
        )
        root.destroy()
        raise SystemExit(0)

    def launch(startup_error):
        if startup_error is not None:
            root = tk.Tk()
            root.withdraw()
            if isinstance(startup_error, RuntimeError):
                from deps_bootstrap import ensure_ml_deps
                ensure_ml_deps(show_dialog=True)
            else:
                messagebox.showerror('Startup failed', str(startup_error), parent=root)
            root.destroy()
            raise SystemExit(1)

        from update_checker import run_check_in_thread

        try:
            app = App()
        except Exception as exc:
            err_text = traceback.format_exc()
            try:
                (APP_DIR / 'startup_error.log').write_text(
                    err_text, encoding='utf-8',
                )
            except OSError:
                pass
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                'Startup failed',
                f'Could not open the main window.\n\n{exc}\n\n'
                'Details were written to startup_error.log beside the exe.\n'
                'End any leftover STEM-organizer process in Task Manager, then retry.',
                parent=root,
            )
            root.destroy()
            raise SystemExit(1) from exc

        run_check_in_thread(APP_VERSION, app)
        app.mainloop()

    show_splash_screen(launch, run_startup=_startup_tasks)


if __name__ == '__main__':
    main()