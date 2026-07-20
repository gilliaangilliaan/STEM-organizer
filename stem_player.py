"""Multi-track stem preview player for Match & Align."""
from __future__ import annotations

import gc
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox

from ffmpeg_bootstrap import subprocess_kwargs

if sys.platform == 'win32':
    import ctypes
else:
    ctypes = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

_sf = None
_np = None
_ffmpeg: str | None = None
_audio_deps_ready = False

SF_READ_EXTS = {'.wav', '.flac', '.aif', '.aiff', '.ogg', '.mp3', '.m4a', '.opus'}

PLAYER_SR = 44100
SEEK_JUMP_SEC = 15
UI_TICK_MS = 33

STEM_ORDER_4 = ('bass', 'drums', 'other', 'vocals')  # demucs filename stems
DEMUCS_LAYOUT_STEMS = ('other', 'drums', 'bass')
FILE_STEM_NAMES = STEM_ORDER_4 + ('instrumental',)

AUDIO_EXTS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aiff', '.aif', '.opus')

PLAYER_WIN_W = 1180
PLAYER_WIN_H = 960
PLAYER_MIN_W = 900
PLAYER_MIN_H = 520

TRACK_ROW_H = 80  # minimum fallback when layout not yet measured
TRACK_ROW_GAP = 6
CONTROLS_W = 168
TIMELINE_H = 28
HEADER_H = 52
METER_W = 72
SHORTCUTS_FOOTER_H = 42  # reserved chrome height for default window layout
WAVE_ZOOM_MIN = 1.0
WAVE_ZOOM_MAX = 64.0
WAVE_ZOOM_STEP = 1.22
WAVE_PEAK_BINS_FULL = 4096
WAVE_FOLLOW_POS = 0.22  # keep playhead ~22% from left while playing zoomed-in

BACKUP_DIR_NAME = '_backup_before_align'

STEM_COLORS = {
    'bass': '#ef4444',
    'drums': '#f59e0b',
    'other': '#10b981',
    'vocals': '#a855f7',
    'instrumental': '#60A5FA',
    'acapella': '#a855f7',
    'original': '#7c5cff',
}

STEM_LABELS = {
    'bass': 'Bass',
    'drums': 'Drums',
    'other': 'Other',
    'vocals': 'Vocals',
    'instrumental': 'Instrumental',
    'acapella': 'Acapella',
    'original': 'Original',
}


def _stem_row_label(name: str, stem_roles: set[str]) -> str:
    """Show Vocals for demucs 4-stem folders; Acapella for pair-finder layouts."""
    if name not in ('acapella', 'vocals'):
        return STEM_LABELS.get(name, name.title())
    if 'instrumental' in stem_roles or 'original' in stem_roles:
        return 'Acapella'
    if {'other', 'drums', 'bass'}.issubset(stem_roles):
        return 'Vocals'
    return STEM_LABELS.get(name, name.title())

SHORTCUT_FONT = ('Segoe UI', 8)
BUSY_DOT_CYCLE_SEC = 1.0
_BUSY_DOT_FRAMES = ('.', '..', '...')
BUSY_FONT = ('Segoe UI', 10)
BUSY_WORD_WIDTH = 7   # "Loading"
BUSY_DOTS_WIDTH = 3
MIN_BUSY_BADGE_SEC = 0.5
FOLDER_CACHE_MAX = 8


def _blend_hex(fg: str, bg: str, t: float) -> str:
    """Blend fg toward bg; t=0 keeps fg, t=1 returns bg."""
    t = max(0.0, min(1.0, t))
    fg = fg.lstrip('#')
    bg = bg.lstrip('#')
    fr, fg_g, fb = (int(fg[i:i + 2], 16) for i in (0, 2, 4))
    br, bg_g, bb = (int(bg[i:i + 2], 16) for i in (0, 2, 4))
    r = int(fr + (br - fr) * t)
    g = int(fg_g + (bg_g - fg_g) * t)
    b = int(fb + (bb - fb) * t)
    return f'#{r:02x}{g:02x}{b:02x}'


TITLE_HOVER_BG = '#1e2029'
TITLE_FLASH_PASS_BG = _blend_hex('#7ee0a0', '#262833', 0.55)
TITLE_FLASH_FAIL_BG = _blend_hex('#ff7a7a', '#262833', 0.55)
TITLE_FLASH_MS = 1500
WAVEFORM_DIM_BLEND = 0.68


def _bind_tooltip(widget: tk.Misc, text: str) -> None:
    from ui_theme import Tooltip

    Tooltip(widget, text)


def _shortcut_key_cap(parent: tk.Misc, text: str, colors: dict) -> tk.Frame:
    cap = tk.Frame(
        parent, bg=colors['panel2'],
        highlightthickness=1, highlightbackground=colors['border'],
    )
    tk.Label(
        cap, text=text, bg=colors['panel2'], fg=colors['fg'],
        font=SHORTCUT_FONT, padx=5, pady=1,
    ).pack()
    return cap


def _shortcut_keys(
    parent: tk.Misc, keys: tuple[str, ...], join: str, colors: dict,
) -> tk.Frame:
    keys_frm = tk.Frame(parent, bg=colors['bg'])
    for i, key in enumerate(keys):
        if i > 0:
            if join == 'plus':
                tk.Label(
                    keys_frm, text='+', bg=colors['bg'], fg=colors['fg_dim'],
                    font=SHORTCUT_FONT,
                ).pack(side='left', padx=3)
            else:
                tk.Frame(keys_frm, bg=colors['bg'], width=6).pack(side='left')
        _shortcut_key_cap(keys_frm, key, colors).pack(side='left')
    return keys_frm


def _shortcut_group(
    parent: tk.Misc, label: str, keys: tuple[str, ...], join: str, colors: dict,
) -> tk.Frame:
    grp = tk.Frame(parent, bg=colors['bg'])
    tk.Label(
        grp, text=label, bg=colors['bg'], fg=colors['fg_dim'],
        font=SHORTCUT_FONT,
    ).pack(side='left', padx=(0, 8))
    _shortcut_keys(grp, keys, join, colors).pack(side='left')
    return grp


def _populate_shortcuts_bar(bar: tk.Frame, colors: dict, *, stem_count: int = 0) -> None:
    has_stems = stem_count > 0
    n_cols = 6 if has_stems else 4
    for col in range(n_cols):
        bar.columnconfigure(col, weight=1)

    def _place_group(
        column: int, label: str, keys: tuple[str, ...], join: str, *, anchor: str = 'center',
    ) -> None:
        cell = tk.Frame(bar, bg=colors['bg'])
        cell.grid(row=0, column=column, sticky='ew')
        _shortcut_group(cell, label, keys, join, colors).pack(anchor=anchor)

    _place_group(0, 'Play / pause', ('Space',), 'plus', anchor='w')

    if has_stems:
        n = min(stem_count, 3)
        solo_keys = tuple(str(i + 1) for i in range(n))
        mute_keys = tuple(f'⇧{i + 1}' for i in range(n))
        _place_group(1, 'Solo stem', solo_keys, 'gap')
        _place_group(2, 'Mute stem', mute_keys, 'gap')
        _place_group(3, 'Seek ±15 seconds', ('←', '→'), 'gap')
        _place_group(4, 'Prev / next song', ('[', ']'), 'gap')
        _place_group(5, 'Pass / Fail folder', ('P', 'F'), 'gap', anchor='e')
    else:
        _place_group(1, 'Zoom in / out', ('Ctrl', 'scroll ↑↓'), 'plus')
        _place_group(2, 'Prev / next song', ('[', ']'), 'gap')
        _place_group(3, 'Pass / Fail folder', ('P', 'F'), 'gap', anchor='e')


def _parse_window_size(widget: tk.Misc) -> tuple[int, int]:
    widget.update_idletasks()
    geo = widget.geometry()
    if geo and 'x' in geo:
        try:
            wh = geo.split('+', 1)[0]
            w, h = wh.split('x', 1)
            return int(w), int(h)
        except ValueError:
            pass
    return max(1, widget.winfo_width()), max(1, widget.winfo_height())


def _win_toplevel_hwnd(root: tk.Misc) -> int:
    user32 = ctypes.windll.user32
    wid = int(root.winfo_id())
    return int(user32.GetParent(wid) or wid)


def _win_window_rect(root: tk.Misc) -> tuple[int, int, int, int] | None:
    if sys.platform != 'win32' or ctypes is None:
        return None
    try:
        class _RECT(ctypes.Structure):
            _fields_ = [
                ('left', ctypes.c_long),
                ('top', ctypes.c_long),
                ('right', ctypes.c_long),
                ('bottom', ctypes.c_long),
            ]

        user32 = ctypes.windll.user32
        hwnd = _win_toplevel_hwnd(root)
        rect = _RECT()
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


def _parent_title_bar_bottom(parent: tk.Misc) -> int:
    """Screen Y coordinate immediately below the parent's custom title bar."""
    parent.update_idletasks()
    bar = getattr(parent, '_title_bar', None)
    if bar is not None:
        try:
            bar.update_idletasks()
            bottom = bar.winfo_rooty() + bar.winfo_height()
            if bottom > parent.winfo_rooty():
                return bottom
        except tk.TclError:
            pass
    return parent.winfo_rooty() + _parent_title_bar_height(parent)


def _parent_content_rect(parent: tk.Misc) -> tuple[int, int, int, int]:
    """Return (x, y, width, height) for the STEM player, bottom-aligned under the title bar."""
    parent.update_idletasks()
    px = parent.winfo_rootx()
    pw = max(1, parent.winfo_width())
    parent_bottom = parent.winfo_rooty() + max(1, parent.winfo_height())
    title_bottom = _parent_title_bar_bottom(parent)
    ph = max(PLAYER_MIN_H, parent_bottom - title_bottom)
    py = parent_bottom - ph
    if py < title_bottom:
        py = title_bottom
        ph = max(PLAYER_MIN_H, parent_bottom - py)
    return px, py, pw, ph


def _parent_title_bar_height(parent: tk.Misc) -> int:
    try:
        bar = getattr(parent, '_title_bar', None)
        if bar is not None:
            parent.update_idletasks()
            bar.update_idletasks()
            h = bar.winfo_height()
            if h > 0:
                return h
    except (AttributeError, tk.TclError):
        pass
    try:
        from stem_organizer_ui import TITLE_BAR_HEIGHT, _USE_CUSTOM_TITLE_BAR

        if _USE_CUSTOM_TITLE_BAR:
            return TITLE_BAR_HEIGHT
    except ImportError:
        pass
    return 0


def _fit_toplevel_outer_bounds(
    win: tk.Misc, x: int, y: int, w: int, h: int,
) -> None:
    """Position a decorated Toplevel so its outer window bounds match the target rect."""
    target_right = x + w
    target_bottom = y + h
    gx, gy, gw, gh = x, y, max(PLAYER_MIN_W, w), max(PLAYER_MIN_H, h)
    for _ in range(8):
        try:
            win.geometry(f'{int(gw)}x{int(gh)}+{int(gx)}+{int(gy)}')
        except tk.TclError:
            return
        win.update_idletasks()
        outer = _win_window_rect(win)
        if outer is None:
            rx, ry = win.winfo_rootx(), win.winfo_rooty()
            gw += w - win.winfo_width()
            gh += h - win.winfo_height()
            gx += x - rx
            gy += y - ry
            try:
                win.geometry(f'{int(gw)}x{int(gh)}+{int(gx)}+{int(gy)}')
            except tk.TclError:
                pass
            return
        ox, oy, ow, oh = outer
        gx += x - ox
        gy += y - oy
        gw += w - ow
        gh += h - oh
        if (
            abs(ox - x) <= 1 and abs(oy - y) <= 1
            and abs(ox + ow - target_right) <= 1
            and abs(oy + oh - target_bottom) <= 1
        ):
            break
        gw = max(PLAYER_MIN_W, gw)
        gh = max(PLAYER_MIN_H, gh)


def _place_over_parent(
    parent: tk.Misc, child: tk.Misc, width: int, height: int,
) -> None:
    """Cover the parent below its title bar, aligned to the main window bottom."""
    parent.update_idletasks()
    child.update_idletasks()
    px, py, pw, ph = _parent_content_rect(parent)
    width = max(PLAYER_MIN_W, int(pw))
    height = max(PLAYER_MIN_H, min(int(height), int(ph)))
    _fit_toplevel_outer_bounds(child, px, py, width, height)
    child.lift(parent)


def _player_app_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _ensure_player_audio_deps() -> None:
    global _sf, _np, _ffmpeg, _audio_deps_ready
    if _audio_deps_ready:
        return
    from ffmpeg_bootstrap import ffmpeg_path

    import numpy as np
    import soundfile as sf

    _np = np
    _sf = sf
    _ffmpeg = ffmpeg_path()
    _audio_deps_ready = True


def _normalize_player_audio(audio, file_sr: int, sr: int, ch: int):
    _ensure_player_audio_deps()
    if audio.shape[0] == 1:
        audio = _np.repeat(audio, ch, axis=0)
    elif audio.shape[0] > ch:
        audio = audio[:ch]
    if file_sr == sr:
        return audio.astype(_np.float32)
    try:
        import resampy
        audio = resampy.resample(audio, file_sr, sr, axis=1)
    except ImportError:
        raise RuntimeError(
            f'Sample rate mismatch ({file_sr} Hz vs {sr} Hz) and resampy is not installed.'
        )
    return audio.astype(_np.float32)


def _read_soundfile_player(path: str, sr: int, ch: int):
    _ensure_player_audio_deps()
    try:
        data, file_sr = _sf.read(path, dtype='float32', always_2d=True, mmap=True)
        return _normalize_player_audio(data.T, file_sr, sr, ch)
    except Exception:
        try:
            data, file_sr = _sf.read(path, dtype='float32', always_2d=True)
            return _normalize_player_audio(data.T, file_sr, sr, ch)
        except Exception:
            return None


def _read_via_ffmpeg_player(path: str, sr: int, ch: int):
    _ensure_player_audio_deps()
    if not _ffmpeg:
        return None
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [_ffmpeg, '-y', '-loglevel', 'error', '-i', path,
             '-ar', str(sr), '-ac', str(ch), tmp_path],
            check=True, capture_output=True,
            **subprocess_kwargs(),
        )
        return _read_soundfile_player(tmp_path, sr, ch)
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def load_player_audio(path: str, sr: int = PLAYER_SR, ch: int = 2):
    """Load audio for playback — soundfile, then ffmpeg, then demucs AudioFile."""
    _ensure_player_audio_deps()
    p = Path(path)
    ext = p.suffix.lower()

    if ext in SF_READ_EXTS:
        audio = _read_soundfile_player(str(p), sr, ch)
        if audio is not None:
            return audio

    audio = _read_via_ffmpeg_player(str(p), sr, ch)
    if audio is not None:
        return audio

    try:
        from demucs.audio import AudioFile
        return AudioFile(path).read(
            streams=0, samplerate=sr, channels=ch,
        ).numpy().astype(_np.float32)
    except Exception as exc:
        hint = (
            'Re-run install-deps.bat if packages are missing. '
            'For FLAC without ffmpeg, ensure soundfile/libsndfile supports FLAC.'
        )
        raise RuntimeError(f'Could not decode audio ({p.name}): {exc}\n{hint}') from exc


def _rgb_from_hex(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_to_rgb_str(hex_color: str) -> str:
    r, g, b = _rgb_from_hex(hex_color)
    return f'#{r:02x}{g:02x}{b:02x}'


def _lerp_hex(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = _rgb_from_hex(a)
    r2, g2, b2 = _rgb_from_hex(b)
    return (
        f'#{int(r1 + (r2 - r1) * t):02x}'
        f'{int(g1 + (g2 - g1) * t):02x}'
        f'{int(b1 + (b2 - b1) * t):02x}'
    )


def _meter_gradient_color(t: float) -> str:
    """Map 0..1 meter position to green → yellow → red."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return _lerp_hex('#22c55e', '#eab308', t / 0.5)
    return _lerp_hex('#eab308', '#ef4444', (t - 0.5) / 0.5)


def find_stem_file(folder: Path, stem: str) -> Path | None:
    for ext in AUDIO_EXTS:
        p = folder / f'{stem}{ext}'
        if p.is_file():
            return p
    stem_l = stem.lower()
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS and f.stem.lower() == stem_l:
            return f
    return None


def _collect_stem_roles(folder: Path) -> dict[str, Path]:
    """Map stem role names to audio files in a song folder."""
    from stem_align import classify_audio_file

    roles: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.name == BACKUP_DIR_NAME or not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        role = classify_audio_file(path)
        if role and role not in roles:
            roles[role] = path
    for stem in FILE_STEM_NAMES:
        if stem not in roles:
            found = find_stem_file(folder, stem)
            if found is not None:
                roles[stem] = found
    return roles


def _vocal_stem_role(roles: dict[str, Path]) -> str | None:
    if 'acapella' in roles:
        return 'acapella'
    if 'vocals' in roles:
        return 'vocals'
    return None


def _order_stem_roles(roles: dict[str, Path]) -> list[str]:
    """Top-to-bottom display order for loaded stems."""
    vocal = _vocal_stem_role(roles)
    names = set(roles)

    if vocal and all(k in names for k in DEMUCS_LAYOUT_STEMS):
        return [vocal, 'other', 'drums', 'bass']

    if vocal and 'instrumental' in names and 'original' in names:
        return [vocal, 'instrumental', 'original']

    if vocal and 'instrumental' in names:
        return [vocal, 'instrumental']

    if vocal and names.intersection(DEMUCS_LAYOUT_STEMS):
        order = [vocal]
        for stem in DEMUCS_LAYOUT_STEMS:
            if stem in names:
                order.append(stem)
        return order

    fallback = (
        'acapella', 'vocals', 'instrumental',
        'other', 'drums', 'bass', 'original',
    )
    order = [name for name in fallback if name in names]
    for name in sorted(names):
        if name not in order:
            order.append(name)
    return order


def detect_stem_folder(folder: Path) -> list[tuple[str, Path]]:
    """Return ordered (stem_name, path) pairs for a song folder."""
    roles = _collect_stem_roles(folder)
    if len(roles) < 2:
        return []
    return [(name, roles[name]) for name in _order_stem_roles(roles)]


def list_player_song_folders(library_root: Path) -> list[Path]:
    if not library_root.is_dir():
        return []
    folders = [
        path for path in library_root.iterdir()
        if path.is_dir() and path.name != BACKUP_DIR_NAME
    ]
    return sorted(folders, key=lambda path: _strip_review_tag(path.name).casefold())


def _strip_review_tag(name: str) -> str:
    """Remove pass/fail review suffix or legacy prefix from a folder name."""
    import re

    text = name.strip()
    text = re.sub(r'_(?:\[pass\]|\[fail\])\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\[(?:pass|fail)\]\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def rename_folder_review(folder: Path, verdict: str) -> Path:
    verdict = verdict.strip().lower()
    if verdict not in {'pass', 'fail'}:
        raise ValueError(f'Invalid verdict: {verdict}')
    clean = _strip_review_tag(folder.name)
    new_name = f'{clean}_[{verdict}]'
    dest = folder.parent / new_name
    if folder.name == new_name:
        return folder.resolve()
    if dest.exists():
        raise FileExistsError(f'Folder already exists: {new_name}')
    last_exc: OSError | None = None
    for attempt in range(8):
        try:
            folder.rename(dest)
            return dest.resolve()
        except OSError as exc:
            last_exc = exc
            denied = (
                getattr(exc, 'winerror', None) == 5
                or exc.errno in {13, 32}  # EACCES, EBUSY
            )
            if not denied or attempt >= 7:
                raise
            gc.collect()
            time.sleep(0.05 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f'Could not rename folder: {folder.name}')


def _path_same(a: Path, b: Path) -> bool:
    try:
        if a.resolve() == b.resolve():
            return True
    except OSError:
        pass
    return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _index_song_folder(folders: list[Path], folder: Path) -> int:
    """Locate a song folder in the library list after renames (resolved paths)."""
    for i, candidate in enumerate(folders):
        if _path_same(candidate, folder):
            return i
    clean = _strip_review_tag(folder.name).casefold()
    if clean:
        for i, candidate in enumerate(folders):
            if _strip_review_tag(candidate.name).casefold() == clean:
                return i
    raise ValueError(folder)


def _folder_in_song_library(folder: Path, library_root: Path) -> bool:
    if not library_root.is_dir():
        return False
    try:
        _index_song_folder(list_player_song_folders(library_root), folder)
        return True
    except ValueError:
        return False


def _library_root_containing(folder: Path) -> Path | None:
    """Return parent directory when it is a song library that contains folder."""
    parent = folder.parent
    if not parent.is_dir():
        return None
    if not _folder_in_song_library(folder, parent):
        return None
    return parent


def format_time_ms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    mins, secs = divmod(total_s, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f'{hours:02d}:{mins:02d}:{secs:02d}:{ms:03d}'
    return f'{mins:02d}:{secs:02d}:{ms:03d}'


def format_ruler_time(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f'{hours}:{mins:02d}:{secs:02d}'
    return f'{mins:02d}:{secs:02d}'


def compute_waveform_peaks(mono, num_bins: int):
    _ensure_player_audio_deps()
    if num_bins < 1:
        return _np.zeros(1, dtype=_np.float32)
    mono = _np.asarray(mono, dtype=_np.float32).ravel()
    if mono.size == 0:
        return _np.zeros(num_bins, dtype=_np.float32)
    n = mono.size
    step = max(1, n // num_bins)
    count = min(num_bins, (n + step - 1) // step)
    usable = count * step
    if usable <= 0:
        return _np.zeros(num_bins, dtype=_np.float32)
    peaks = _np.max(_np.abs(mono[:usable].reshape(count, step)), axis=1)
    if count < num_bins:
        peaks = _np.pad(peaks, (0, num_bins - count))
    return peaks.astype(_np.float32)


_PEAKS_FAST_MAX_SAMPLES = 600_000


def _compute_peaks_full_fast(mono) -> 'np.ndarray':
    """Build full-zoom peaks from downsampled audio (much faster on long songs)."""
    _ensure_player_audio_deps()
    mono = _np.asarray(mono, dtype=_np.float32).ravel()
    if mono.size > _PEAKS_FAST_MAX_SAMPLES:
        step = max(1, mono.size // _PEAKS_FAST_MAX_SAMPLES)
        mono = mono[::step]
    return compute_waveform_peaks(mono, WAVE_PEAK_BINS_FULL)


def _to_stereo(audio):
    _ensure_player_audio_deps()
    a = _np.asarray(audio, dtype=_np.float32)
    if a.ndim == 1:
        return _np.stack([a, a], axis=0)
    if a.shape[0] == 1:
        return _np.repeat(a, 2, axis=0)
    if a.shape[0] > 2:
        return a[:2]
    return a


class _TrackState:
    __slots__ = (
        'name', 'path', 'audio', 'peaks', 'peaks_full', 'volume', 'muted', 'solo',
        'color', 'row', 'wave_canvas', '_solo_btn', '_mute_btn',
    )

    def __init__(self, name: str, path: Path, audio, color: str):
        self.name = name
        self.path = path
        self.audio = _to_stereo(audio)
        self.peaks = None
        self.peaks_full = None
        self.volume = 1.0
        self.muted = False
        self.solo = False
        self.color = color
        self.row: tk.Frame | None = None
        self.wave_canvas: tk.Canvas | None = None
        self._solo_btn = None
        self._mute_btn = None


def _load_one_stem(name: str, path: Path) -> _TrackState:
    try:
        audio = load_player_audio(str(path), PLAYER_SR, 2)
    except Exception as exc:
        raise RuntimeError(f'Failed to load {path.name}:\n{exc}') from exc
    color = STEM_COLORS.get(name, '#9aa0b4')
    track = _TrackState(name, path, audio, color)
    track.peaks_full = _compute_peaks_full_fast(track.audio.mean(axis=0))
    return track


def _load_tracks_from_stems(stems: list[tuple[str, Path]]) -> list[_TrackState]:
    if not stems:
        return []
    if len(stems) == 1:
        name, path = stems[0]
        return [_load_one_stem(name, path)]
    tracks: list[_TrackState | None] = [None] * len(stems)
    workers = min(len(stems), 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_load_one_stem, stems[i][0], stems[i][1]): i
            for i in range(len(stems))
        }
        for fut in as_completed(futures):
            tracks[futures[fut]] = fut.result()
    return tracks  # type: ignore[return-value]


def _folder_cache_key(folder: Path, stems: list[tuple[str, Path]]) -> str:
    parts = [str(folder.resolve())]
    for name, path in sorted(stems, key=lambda item: item[0]):
        st = path.stat()
        parts.append(f'{name}:{st.st_mtime_ns}:{st.st_size}')
    return '|'.join(parts)


def _tracks_cache_snapshot(tracks: list[_TrackState]) -> list[_TrackState]:
    snap: list[_TrackState] = []
    for track in tracks:
        copy = _TrackState(track.name, track.path, track.audio, track.color)
        copy.peaks_full = track.peaks_full
        copy.volume = track.volume
        copy.muted = track.muted
        copy.solo = track.solo
        snap.append(copy)
    return snap


def _tracks_from_cache(cached: list[_TrackState], stems: list[tuple[str, Path]]) -> list[_TrackState]:
    path_by_name = {name: path for name, path in stems}
    tracks: list[_TrackState] = []
    for track in cached:
        path = path_by_name.get(track.name, track.path)
        copy = _TrackState(track.name, path, track.audio, track.color)
        copy.peaks_full = track.peaks_full
        copy.volume = track.volume
        copy.muted = track.muted
        copy.solo = track.solo
        tracks.append(copy)
    return tracks


class _AudioEngine:
    def __init__(self, tracks: list[_TrackState], sr: int = PLAYER_SR):
        self.tracks = tracks
        self.sr = sr
        self.duration = 0.0
        if tracks:
            self.duration = max(t.audio.shape[1] for t in tracks) / sr
        self._position = 0
        self._playing = False
        self.master_volume = 0.85
        self._lock = threading.Lock()
        self._meter = 0.0
        self._stream: sd.OutputStream | None = None

    @property
    def position(self) -> float:
        with self._lock:
            return self._position / self.sr

    @position.setter
    def position(self, seconds: float) -> None:
        with self._lock:
            self._position = int(max(0.0, min(seconds, self.duration)) * self.sr)

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    def set_playing(self, playing: bool) -> None:
        with self._lock:
            self._playing = playing

    def meter_level(self) -> float:
        with self._lock:
            return self._meter

    def _any_solo(self) -> bool:
        return any(t.solo for t in self.tracks)

    def _track_audible(self, track: _TrackState) -> bool:
        if track.muted:
            return False
        if self._any_solo():
            return track.solo
        return True

    def _callback(self, outdata, frames, _time_info, _status) -> None:
        _ensure_player_audio_deps()
        with self._lock:
            pos = self._position
            playing = self._playing
            master = self.master_volume

        out = _np.zeros((frames, 2), dtype=_np.float32)
        if not playing or not self.tracks:
            outdata[:] = out
            with self._lock:
                self._meter = 0.0
            return

        end_pos = pos + frames
        peak = 0.0
        for track in self.tracks:
            if not self._track_audible(track):
                continue
            audio = track.audio
            n = audio.shape[1]
            if pos >= n:
                continue
            take = min(frames, n - pos)
            chunk = audio[:, pos:pos + take].T * (track.volume * master)
            out[:take] += chunk
            peak = max(peak, float(_np.max(_np.abs(out[:take]))))

        outdata[:] = out
        with self._lock:
            self._position = pos + frames
            if self._position >= int(self.duration * self.sr):
                self._playing = False
                self._position = int(self.duration * self.sr)
            self._meter = peak

    def start_stream(self) -> None:
        if sd is None:
            raise RuntimeError('sounddevice is not installed')
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            dtype='float32',
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stream = None


class StemPlayerWindow(tk.Toplevel):
    def __init__(self, parent: tk.Misc, *, colors: dict):
        super().__init__(parent)
        self.withdraw()
        self._colors = colors
        self._engine: _AudioEngine | None = None
        self._tracks: list[_TrackState] = []
        self._folder: Path | None = None
        self._library_root: Path | None = None
        self._song_folders: list[Path] = []
        self._folder_index = -1
        self._wave_w = 0
        self._view_zoom = WAVE_ZOOM_MIN
        self._view_start = 0.0
        self._redraw_sig: tuple | None = None
        self._tick_id: str | None = None
        self._resize_after: str | None = None
        self._playhead_ids: list[int] = []
        self._title_var = tk.StringVar(value='No folder loaded')
        self._time_var = tk.StringVar(value='00:00:000')
        self._master_var = tk.DoubleVar(value=85.0)
        self._title_flash_after_id: str | None = None
        self._title_flash_verdict: str | None = None
        self._folder_nav_keys_bound = False
        self._default_w = PLAYER_MIN_W
        self._default_h = PLAYER_WIN_H
        self._resize_guard = False
        self._was_maximized = False
        self._busy_generation = 0
        self._busy_dot_last = 0.0
        self._busy_word = 'Loading'
        self._busy_dot_i = 0
        self._busy_mode = 'loading'
        self._busy_started_at = 0.0
        self._busy_hide_after_id: str | None = None
        self._folder_job_active = False
        self._folder_cache: OrderedDict[str, list[_TrackState]] = OrderedDict()
        self._folder_cache_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()
        self._prefetch_inflight: set[str] = set()
        self._main_jobs: queue.Queue = queue.Queue()
        self._pending_open: tuple[Path, int | None] | None = None

        self.title('STEM player')
        self.configure(bg=colors['bg'])
        self._window_icon = None
        try:
            from ui_theme import apply_toplevel_icon
            self._window_icon = apply_toplevel_icon(self)
        except Exception:
            pass
        self._corner_after_id: str | None = None

        self._build_ui()
        self._bind_keys()
        self._bind_keys_on_widget(self)
        self._bind_zoom_wheel(self)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        self._parent_ref = parent

        self.bind('<Configure>', self._on_window_resize, add='+')
        self.after(100, self._redraw_all)

    def _is_maximized(self) -> bool:
        try:
            return self.state() == 'zoomed'
        except tk.TclError:
            return False

    def _init_window_geometry(self, width: int, height: int) -> None:
        self._default_w = width
        self._default_h = height
        self.minsize(PLAYER_MIN_W, PLAYER_MIN_H)
        self.resizable(True, True)
        self._allow_maximized_size()
        self._was_maximized = False

    def _allow_maximized_size(self) -> None:
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.maxsize(sw, sh)

    def _default_open_size(self, parent: tk.Misc | None = None) -> tuple[int, int]:
        parent = parent or self._parent_ref
        _, _, pw, ph = _parent_content_rect(parent)
        if pw < PLAYER_MIN_W:
            pw, _ = _parse_window_size(parent)
        if pw < PLAYER_MIN_W:
            try:
                from ui_theme import WIN_DEFAULT_W

                pw = WIN_DEFAULT_W
            except ImportError:
                pw = PLAYER_WIN_W
        return max(PLAYER_MIN_W, pw), max(PLAYER_MIN_H, ph)

    def _apply_default_open_geometry(self) -> None:
        if self._is_maximized():
            return
        parent = self._parent_ref
        win_w, win_h = self._default_open_size(parent)
        _place_over_parent(parent, self, win_w, win_h)
        self._default_w = win_w
        self._default_h = win_h
        self.update_idletasks()

    def _refresh_rounded_corners(self) -> None:
        try:
            from ui_theme import apply_toplevel_rounded_corners
            apply_toplevel_rounded_corners(
                self, maximized=self._is_maximized(),
            )
        except Exception:
            pass

    def _schedule_rounded_corners(self, delay_ms: int = 40) -> None:
        job = getattr(self, '_corner_after_id', None)
        if job is not None:
            try:
                self.after_cancel(job)
            except (tk.TclError, ValueError):
                pass

        def _apply() -> None:
            self._corner_after_id = None
            self._refresh_rounded_corners()

        self._corner_after_id = self.after(delay_ms, _apply)

    def _place_and_show(self) -> None:
        parent = self._parent_ref
        parent.update_idletasks()
        self.update_idletasks()
        self.deiconify()
        self._apply_default_open_geometry()
        self._init_window_geometry(self._default_w, self._default_h)
        self.lift(parent)
        self.focus_force()
        self.after_idle(self._apply_default_open_geometry)
        self.after_idle(self._refresh_rounded_corners)

    def _ctk_icon_btn(self, parent, text: str, command, *, text_color: str | None = None, width: int = 36):
        """Compact CTk button matching Browse/Open chrome."""
        from ui_theme import DARK, ctk_ui_font, ensure_ctk_dark

        ctk = ensure_ctk_dark()
        t = DARK
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            height=30,
            fg_color=t['btn'],
            hover_color=t['btn_hover'],
            text_color=text_color or t['text'],
            font=ctk_ui_font(),
            cursor='hand2',
        )

    def _ctk_volume_slider(self, parent, variable, command, *, width: int):
        from ui_theme import DARK, ensure_ctk_dark

        ctk = ensure_ctk_dark()
        t = DARK
        return ctk.CTkSlider(
            parent,
            from_=0,
            to=100,
            variable=variable,
            command=command,
            width=width,
            height=16,
            fg_color=t['control_bg'],
            progress_color=t['accent'],
            button_color=t['accent'],
            button_hover_color=t['accent_hover'],
        )

    def _build_ui(self) -> None:
        C = self._colors
        from ui_theme import ctk_action_button

        self._shortcuts_footer = tk.Frame(self, bg=C['bg'])
        self._shortcuts_footer.pack(side='bottom', fill='x')
        self._shortcuts_bar = tk.Frame(self._shortcuts_footer, bg=C['bg'])
        self._shortcuts_bar.pack(fill='x', padx=16, pady=(2, 8))
        self._refresh_shortcuts_footer()

        header = tk.Frame(self, bg=C['panel'], height=HEADER_H)
        header.pack(fill='x', padx=12, pady=(12, 6))
        header.pack_propagate(False)

        load_btn = ctk_action_button(header, '📁  Load', self._load_folder, width=88)
        load_btn.pack(side='left', padx=(8, 12))
        _bind_tooltip(load_btn, 'Choose a song folder to load stems. Use [ and ] for prev/next song.')

        self._title_frame = tk.Frame(header, bg=C['panel'])
        self._title_frame.pack(side='left', fill='x', expand=True)
        self._title_lbl = tk.Label(
            self._title_frame, textvariable=self._title_var, bg=C['panel'], fg=C['fg'],
            font=('Segoe UI', 10), anchor='w',
        )
        self._title_lbl.pack(fill='both', expand=True, anchor='w')
        for widget in (self._title_frame, self._title_lbl):
            widget.bind('<Enter>', self._on_title_enter)
            widget.bind('<Leave>', self._on_title_leave)
            widget.bind('<Button-1>', self._on_title_click)

        time_lbl = tk.Label(
            header, textvariable=self._time_var, bg=C['panel'], fg=C['fg'],
            font=('Consolas', 14), width=12,
        )
        time_lbl.pack(side='left', padx=(8, 16))

        transport = tk.Frame(header, bg=C['panel'])
        transport.pack(side='left', padx=(0, 12))

        nav = tk.Frame(header, bg=C['panel'])
        nav.pack(side='left', padx=(0, 8))
        self._btn_prev_song = self._ctk_icon_btn(nav, '◀', self._prev_song_folder)
        self._btn_prev_song.pack(side='left', padx=1)
        self._btn_next_song = self._ctk_icon_btn(nav, '▶', self._next_song_folder)
        self._btn_next_song.pack(side='left', padx=1)

        self._btn_skip_back = self._ctk_icon_btn(
            transport, '⏮', lambda: self._seek_relative(-SEEK_JUMP_SEC),
        )
        self._btn_skip_back.pack(side='left', padx=2)

        self._btn_play = self._ctk_icon_btn(transport, '⏵', self._toggle_play)
        self._btn_play.pack(side='left', padx=2)

        self._btn_stop = self._ctk_icon_btn(
            transport, '■', self._stop, text_color=C['fg_dim'],
        )
        self._btn_stop.pack(side='left', padx=2)

        self._btn_skip_fwd = self._ctk_icon_btn(
            transport, '⏭', lambda: self._seek_relative(SEEK_JUMP_SEC),
        )
        self._btn_skip_fwd.pack(side='left', padx=2)

        vol_frame = tk.Frame(header, bg=C['panel'])
        vol_frame.pack(side='right', padx=(8, 0))

        tk.Label(vol_frame, text='Master', bg=C['panel'], fg=C['fg_dim'],
                 font=('Segoe UI', 9)).pack(side='left', padx=(0, 6))

        self._master_scale = self._ctk_volume_slider(
            vol_frame, self._master_var, self._on_master_volume, width=120,
        )
        self._master_scale.pack(side='left')

        self._meter_canvas = tk.Canvas(
            vol_frame, width=METER_W, height=18, bg=C['log_bg'],
            highlightthickness=1, highlightbackground=C['border'],
            highlightcolor=C['border'], takefocus=0,
        )
        self._meter_canvas.pack(side='left', padx=(10, 8))

        body = tk.Frame(self, bg=C['bg'])
        body.pack(fill='both', expand=True, padx=12, pady=(0, 4))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        timeline_row = tk.Frame(body, bg=C['bg'])
        timeline_row.grid(row=0, column=0, sticky='ew', pady=(0, 4))

        timeline_spacer = tk.Frame(timeline_row, bg=C['bg'], width=CONTROLS_W)
        timeline_spacer.pack(side='left')
        timeline_spacer.pack_propagate(False)

        self._timeline_canvas = tk.Canvas(
            timeline_row, height=TIMELINE_H, bg=C['log_bg'],
            highlightthickness=1, highlightbackground=C['border'],
            highlightcolor=C['border'], takefocus=0,
        )
        self._timeline_canvas.pack(side='left', fill='x', expand=True, padx=(8, 0))
        self._timeline_canvas.bind('<Button-1>', self._on_timeline_click)
        self._bind_zoom_wheel(self._timeline_canvas)

        self._tracks_outer = tk.Frame(body, bg=C['bg'])
        self._tracks_outer.grid(row=1, column=0, sticky='nsew')
        self._tracks_outer.columnconfigure(0, weight=1)
        self._tracks_outer.rowconfigure(0, weight=1)
        self._tracks_outer.bind('<Configure>', self._on_tracks_area_resize, add='+')
        self._bind_zoom_wheel(self._tracks_outer)

        self._tracks_inner = tk.Frame(self._tracks_outer, bg=C['bg'])
        self._tracks_inner.place(relx=0, rely=0, relwidth=1, relheight=1)

        busy_bg = _blend_hex(C['log_bg'], C['bg'], 0.35)
        busy_border = _blend_hex(C['border'], C['panel'], 0.5)
        self._busy_badge = tk.Frame(
            self, bg=busy_bg,
            highlightthickness=1, highlightbackground=busy_border,
        )
        busy_inner = tk.Frame(self._busy_badge, bg=busy_bg)
        busy_inner.pack(padx=(16, 12), pady=6)
        self._busy_word_lbl = tk.Label(
            busy_inner, text='Loading', bg=busy_bg, fg=C['fg_dim'],
            font=BUSY_FONT, width=BUSY_WORD_WIDTH, anchor='e',
        )
        self._busy_dots_lbl = tk.Label(
            busy_inner, text='.', bg=busy_bg, fg=C['fg_dim'],
            font=BUSY_FONT, width=BUSY_DOTS_WIDTH, anchor='w',
        )
        self._busy_word_lbl.pack(side='left')
        self._busy_dots_lbl.pack(side='left')

        self._start_ui_tick()

    def _title_bar_bg(self) -> str:
        if self._title_flash_verdict == 'pass':
            return TITLE_FLASH_PASS_BG
        if self._title_flash_verdict == 'fail':
            return TITLE_FLASH_FAIL_BG
        return self._colors['panel']

    def _cancel_title_flash(self) -> None:
        if self._title_flash_after_id is not None:
            try:
                self.after_cancel(self._title_flash_after_id)
            except tk.TclError:
                pass
            self._title_flash_after_id = None

    def _end_title_flash(self) -> None:
        self._title_flash_after_id = None
        self._title_flash_verdict = None
        panel_bg = self._colors['panel']
        for widget in (self._title_frame, self._title_lbl):
            widget.configure(bg=panel_bg)

    def _flash_title_bar(self, verdict: str) -> None:
        self._cancel_title_flash()
        self._title_flash_verdict = verdict
        flash_bg = TITLE_FLASH_PASS_BG if verdict == 'pass' else TITLE_FLASH_FAIL_BG
        for widget in (self._title_frame, self._title_lbl):
            widget.configure(bg=flash_bg)
        self._title_flash_after_id = self.after(TITLE_FLASH_MS, self._end_title_flash)

    def _begin_busy(self, mode: str) -> int:
        self._busy_generation += 1
        gen = self._busy_generation
        self._busy_mode = mode
        self._busy_started_at = time.monotonic()
        if self._busy_hide_after_id is not None:
            try:
                self.after_cancel(self._busy_hide_after_id)
            except tk.TclError:
                pass
            self._busy_hide_after_id = None
        self._busy_word = 'Saving' if mode == 'saving' else 'Loading'
        self._busy_dot_i = 0
        self._busy_word_lbl.configure(text=self._busy_word)
        self._busy_dots_lbl.configure(text=_BUSY_DOT_FRAMES[0])
        self._busy_badge.place(relx=0.5, rely=0.5, anchor='center')
        self.tk.call('raise', self._busy_badge._w)
        self._busy_dot_last = time.monotonic()
        self.update_idletasks()
        return gen

    def _enqueue_main(self, callback, /, *args) -> None:
        """Schedule work on the Tk main thread from a background worker."""
        self._main_jobs.put((callback, args))

    def _drain_main_jobs(self, *, max_jobs: int = 1) -> None:
        for _ in range(max_jobs):
            try:
                callback, args = self._main_jobs.get_nowait()
            except queue.Empty:
                break
            callback(*args)

    def _poll_busy_dots(self) -> None:
        if not self._busy_badge.winfo_ismapped():
            return
        now = time.monotonic()
        step_sec = BUSY_DOT_CYCLE_SEC / len(_BUSY_DOT_FRAMES)
        if now - self._busy_dot_last < step_sec:
            return
        self._busy_dot_last = now
        self._busy_dot_i = (self._busy_dot_i + 1) % len(_BUSY_DOT_FRAMES)
        self._busy_dots_lbl.configure(text=_BUSY_DOT_FRAMES[self._busy_dot_i])

    def _end_busy(self, gen: int) -> None:
        if gen != self._busy_generation:
            return
        remaining = MIN_BUSY_BADGE_SEC - (time.monotonic() - self._busy_started_at)
        if remaining > 0:
            delay_ms = int(remaining * 1000) + 1
            if self._busy_hide_after_id is not None:
                try:
                    self.after_cancel(self._busy_hide_after_id)
                except tk.TclError:
                    pass
            self._busy_hide_after_id = self.after(
                delay_ms,
                lambda g=gen: self._finish_end_busy(g),
            )
            return
        self._finish_end_busy(gen)

    def _finish_end_busy(self, gen: int) -> None:
        self._busy_hide_after_id = None
        if gen != self._busy_generation:
            return
        self._busy_badge.place_forget()

    def _set_folder_metadata(
        self,
        folder: Path,
        *,
        library_index: int | None,
        refresh_library: bool = True,
    ) -> None:
        self._folder = folder
        self._resolve_library_root(folder)
        if refresh_library and self._library_root is not None:
            self._refresh_song_library(keep_folder=folder)
        if library_index is not None:
            self._folder_index = library_index
        elif self._song_folders:
            try:
                self._folder_index = _index_song_folder(self._song_folders, folder)
            except ValueError:
                self._folder_index = -1
        self._title_var.set(folder.name)
        self._set_title_interactive(True)

    def _get_folder_cache(self, key: str) -> list[_TrackState] | None:
        with self._folder_cache_lock:
            cached = self._folder_cache.get(key)
            if cached is None:
                return None
            self._folder_cache.move_to_end(key)
            return cached

    def _put_folder_cache(self, key: str, tracks: list[_TrackState]) -> None:
        with self._folder_cache_lock:
            self._folder_cache[key] = _tracks_cache_snapshot(tracks)
            while len(self._folder_cache) > FOLDER_CACHE_MAX:
                self._folder_cache.popitem(last=False)

    def _prefetch_folder(self, folder: Path) -> None:
        stems = detect_stem_folder(folder)
        if not stems:
            return
        try:
            cache_key = _folder_cache_key(folder, stems)
        except OSError:
            return
        with self._folder_cache_lock:
            if cache_key in self._folder_cache:
                return
        with self._prefetch_lock:
            if cache_key in self._prefetch_inflight:
                return
            self._prefetch_inflight.add(cache_key)

        def worker() -> None:
            try:
                tracks = _load_tracks_from_stems(stems)
                self._put_folder_cache(cache_key, tracks)
            except Exception:
                pass
            finally:
                with self._prefetch_lock:
                    self._prefetch_inflight.discard(cache_key)

        threading.Thread(target=worker, daemon=True).start()

    def _prefetch_adjacent_songs(self) -> None:
        if not self._song_folders or self._folder_index < 0:
            return
        idx = self._folder_index
        if idx > 0:
            self._prefetch_folder(self._song_folders[idx - 1])
        if idx + 1 < len(self._song_folders):
            self._prefetch_folder(self._song_folders[idx + 1])

    def _restart_engine(self) -> bool:
        if not self._tracks:
            return False
        self._engine = _AudioEngine(self._tracks, PLAYER_SR)
        self._on_master_volume()
        try:
            self._engine.start_stream()
        except Exception as exc:
            messagebox.showerror('STEM player', f'Audio output failed:\n{exc}', parent=self)
            self._engine = None
            return False
        return True

    def _finish_open_folder_ui(self, gen: int) -> None:
        if gen != self._busy_generation:
            return
        try:
            self._resize_guard = True
            self._clear_track_rows()
            self._tracks_inner.columnconfigure(0, weight=1)
            self._pending_stem_roles = {track.name for track in self._tracks}
            self.after(0, lambda g=gen: self._rebuild_track_rows_step(g, 0))
        except Exception:
            self._resize_guard = False
            self._folder_job_active = False
            self._end_busy(gen)
            raise

    def _rebuild_track_rows_step(self, gen: int, index: int) -> None:
        if gen != self._busy_generation:
            return
        n = len(self._tracks)
        if index < n:
            self._build_one_track_row(
                self._tracks[index], index, n, self._pending_stem_roles,
            )
            self.after(0, lambda g=gen, i=index + 1: self._rebuild_track_rows_step(g, i))
            return
        self._resize_guard = False
        self._bind_keys_on_widget(self._tracks_inner)
        self._refresh_shortcuts_footer()
        self.focus_set()
        self._redraw_sig = None
        self.after(0, lambda g=gen: self._draw_loaded_waveforms_step(g, 0))

    def _draw_loaded_waveforms_step(self, gen: int, index: int) -> None:
        if gen != self._busy_generation:
            return
        if not self._tracks or self._engine is None:
            self._folder_job_active = False
            self._end_busy(gen)
            return

        if index == 0:
            self._tracks_outer.update_idletasks()
            wave_w = self._waveform_width()
            self._wave_w = wave_w
            self._pending_draw_bins = max(100, min(wave_w, 1024))

        if index < len(self._tracks):
            track = self._tracks[index]
            track.peaks = self._peaks_for_view(track, self._pending_draw_bins)
            self._draw_waveform(track)
            self.after(0, lambda g=gen, i=index + 1: self._draw_loaded_waveforms_step(g, i))
            return

        self._draw_timeline(self._wave_w, self._duration())
        self._update_playhead(self._engine.position)
        self._folder_job_active = False
        self._end_busy(gen)
        self._prefetch_adjacent_songs()

    def _install_loaded_folder(
        self,
        folder: Path,
        *,
        library_index: int | None,
        tracks: list[_TrackState],
    ) -> bool:
        self._set_folder_metadata(folder, library_index=library_index)
        self._tracks = tracks
        self._view_zoom = WAVE_ZOOM_MIN
        self._view_start = 0.0
        if not self._restart_engine():
            return False

        self._resize_guard = True
        try:
            self._rebuild_track_rows()
            self._refresh_shortcuts_footer()
            self.focus_set()
            self._redraw_sig = None
        finally:
            self._resize_guard = False
        self.after_idle(self._schedule_redraw)
        return True

    def _set_title_interactive(self, active: bool) -> None:
        cursor = 'hand2' if active else ''
        for widget in (self._title_frame, self._title_lbl):
            widget.configure(cursor=cursor)
        if active and not getattr(self, '_title_tooltip_bound', False):
            for widget in (self._title_frame, self._title_lbl):
                _bind_tooltip(widget, 'Click to open folder')
            self._title_tooltip_bound = True

    def _on_title_enter(self, _event=None) -> None:
        if self._folder is None:
            return
        for widget in (self._title_frame, self._title_lbl):
            widget.configure(bg=TITLE_HOVER_BG)

    def _on_title_leave(self, _event=None) -> None:
        bg = self._title_bar_bg()
        for widget in (self._title_frame, self._title_lbl):
            widget.configure(bg=bg)

    def _on_title_click(self, _event=None) -> None:
        if self._folder is None or not self._folder.is_dir():
            return
        try:
            if sys.platform == 'win32':
                os.startfile(self._folder)  # type: ignore[attr-defined]
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(self._folder)], check=False)
            else:
                subprocess.run(['xdg-open', str(self._folder)], check=False)
        except Exception as exc:
            messagebox.showerror('STEM player', f'Could not open folder:\n{exc}', parent=self)

    def _refresh_shortcuts_footer(self) -> None:
        for child in self._shortcuts_bar.winfo_children():
            child.destroy()
        _populate_shortcuts_bar(
            self._shortcuts_bar, self._colors, stem_count=len(self._tracks),
        )

    def _handle_stem_shortcut(self, event) -> str | None:
        idx: int | None = None
        mute = False

        if event.keysym in ('1', '2', '3', '4'):
            idx = int(event.keysym) - 1
            mute = bool(event.state & 0x1)
        elif event.keysym in ('exclam', 'at', 'numbersign', 'dollar'):
            idx = {'exclam': 0, 'at': 1, 'numbersign': 2, 'dollar': 3}[event.keysym]
            mute = True
        elif len(event.char) == 1 and event.char in '!@#$':
            idx = '!@#$'.index(event.char)
            mute = True

        if idx is None or idx >= len(self._tracks):
            return None
        if mute:
            self._toggle_mute(self._tracks[idx])
        else:
            self._toggle_solo(self._tracks[idx])
        return 'break'

    def _bind_keys(self) -> None:
        def _space(_e=None):
            self._toggle_play()
            return 'break'

        def _left(_e=None):
            self._seek_relative(-SEEK_JUMP_SEC)
            return 'break'

        def _right(_e=None):
            self._seek_relative(SEEK_JUMP_SEC)
            return 'break'

        def _pass(_e=None):
            self._mark_folder_review('pass')
            return 'break'

        def _fail(_e=None):
            self._mark_folder_review('fail')
            return 'break'

        def _prev(_e=None):
            self._prev_song_folder()
            return 'break'

        def _next(_e=None):
            self._next_song_folder()
            return 'break'

        self._key_handlers = (_space, _left, _right)
        self._stem_key_bindings: list[tuple[str, object]] = []
        for i in range(4):
            self._stem_key_bindings.append((f'<KeyPress-{i + 1}>', self._handle_stem_shortcut))
        for keysym in ('exclam', 'at', 'numbersign', 'dollar'):
            self._stem_key_bindings.append((f'<KeyPress-{keysym}>', self._handle_stem_shortcut))

        for seq, handler in (
            ('<KeyPress-space>', _space),
            ('<Left>', _left),
            ('<Right>', _right),
            ('<KeyPress-p>', _pass),
            ('<KeyPress-P>', _pass),
            ('<KeyPress-f>', _fail),
            ('<KeyPress-F>', _fail),
            ('<KeyPress-bracketleft>', _prev),
            ('<KeyPress-bracketright>', _next),
            ('<bracketleft>', _prev),
            ('<bracketright>', _next),
        ):
            self.bind(seq, handler, add='+')
        for seq, handler in self._stem_key_bindings:
            self.bind(seq, handler, add='+')
        self._bind_folder_nav_keys_global()

    def _bind_folder_nav_keys_global(self) -> None:
        if self._folder_nav_keys_bound:
            return

        def _route_prev(event=None):
            if event is not None:
                try:
                    if event.widget.winfo_toplevel() != self:
                        return
                except tk.TclError:
                    return
            self._prev_song_folder()
            return 'break'

        def _route_next(event=None):
            if event is not None:
                try:
                    if event.widget.winfo_toplevel() != self:
                        return
                except tk.TclError:
                    return
            self._next_song_folder()
            return 'break'

        self._folder_nav_prev_handler = _route_prev
        self._folder_nav_next_handler = _route_next
        self.bind_all('<KeyPress-bracketleft>', _route_prev, add='+')
        self.bind_all('<KeyPress-bracketright>', _route_next, add='+')
        self.bind_all('<bracketleft>', _route_prev, add='+')
        self.bind_all('<bracketright>', _route_next, add='+')
        self._folder_nav_keys_bound = True

    def _unbind_folder_nav_keys_global(self) -> None:
        if not self._folder_nav_keys_bound:
            return
        for seq in (
            '<KeyPress-bracketleft>', '<KeyPress-bracketright>',
            '<bracketleft>', '<bracketright>',
        ):
            try:
                self.unbind_all(seq)
            except tk.TclError:
                pass
        self._folder_nav_keys_bound = False

    def _bind_keys_on_widget(self, widget: tk.Misc) -> None:
        bindings: list[tuple[str, object]] = [
            ('<KeyPress-space>', self._key_handlers[0]),
            ('<Left>', self._key_handlers[1]),
            ('<Right>', self._key_handlers[2]),
            ('<KeyPress-p>', lambda _e: self._mark_folder_review('pass') or 'break'),
            ('<KeyPress-P>', lambda _e: self._mark_folder_review('pass') or 'break'),
            ('<KeyPress-f>', lambda _e: self._mark_folder_review('fail') or 'break'),
            ('<KeyPress-F>', lambda _e: self._mark_folder_review('fail') or 'break'),
            ('<KeyPress-bracketleft>', lambda _e: self._prev_song_folder() or 'break'),
            ('<KeyPress-bracketright>', lambda _e: self._next_song_folder() or 'break'),
            ('<bracketleft>', lambda _e: self._prev_song_folder() or 'break'),
            ('<bracketright>', lambda _e: self._next_song_folder() or 'break'),
        ] + self._stem_key_bindings
        for seq, handler in bindings:
            widget.bind(seq, handler, add='+')
        for child in widget.winfo_children():
            self._bind_keys_on_widget(child)

    def _start_ui_tick(self) -> None:
        self._drain_main_jobs(max_jobs=1)
        self._poll_busy_dots()
        self._ui_tick()
        self._tick_id = self.after(UI_TICK_MS, self._start_ui_tick)

    def _ui_tick(self) -> None:
        if self._engine is None:
            return
        pos = self._engine.position
        self._time_var.set(format_time_ms(pos))
        if self._follow_playhead(pos):
            self._redraw_wave_view(force=True)
        else:
            self._update_playhead(pos)
        self._draw_meter(self._engine.meter_level())
        if self._engine.playing:
            self._btn_play.configure(text='⏸')
        else:
            self._btn_play.configure(text='⏵')

    def _draw_meter(self, level: float) -> None:
        c = self._meter_canvas
        w = max(1, int(c.winfo_width() or METER_W))
        h = max(1, int(c.winfo_height() or 18))
        inset = 1
        inner_w = max(1, w - inset * 2)
        fill_w = int(min(1.0, level) * inner_w)
        cache = (w, h)
        bg = self._colors['log_bg']

        if getattr(self, '_meter_bg_cache', None) != cache:
            c.delete('meter_bg')
            for i in range(inner_w):
                t = i / max(inner_w - 1, 1)
                dim = _lerp_hex(bg, _meter_gradient_color(t), 0.4)
                x0 = inset + i
                c.create_rectangle(
                    x0, inset, x0 + 1, h - inset,
                    fill=dim, outline=dim, tags='meter_bg',
                )
            self._meter_bg_cache = cache

        c.delete('meter')
        for i in range(fill_w):
            t = i / max(inner_w - 1, 1)
            color = _meter_gradient_color(t)
            x0 = inset + i
            c.create_rectangle(
                x0, inset, x0 + 1, h - inset,
                fill=color, outline=color, tags='meter',
            )

    def _on_master_volume(self, _val: str | None = None) -> None:
        if self._engine is not None:
            self._engine.master_volume = float(self._master_var.get()) / 100.0

    def _organize_library_from_parent(self) -> Path | None:
        parent = self._parent_ref
        try:
            var = getattr(parent, 'output_dir', None)
            if var is None:
                return None
            text = var.get().strip()
            if not text:
                return None
            path = Path(text)
            if path.is_dir():
                return path
        except (AttributeError, tk.TclError, OSError):
            pass
        return None

    def _align_library_from_parent(self) -> Path | None:
        parent = self._parent_ref
        panel = getattr(parent, 'pair_panel', None)
        attrs_host = panel if panel is not None else parent
        try:
            from stem_align import default_with_original_dir, resolve_with_original_dir

            for attr in ('align_with_original_dir', 'align_stems_root'):
                var = getattr(attrs_host, attr, None)
                if var is None:
                    continue
                text = var.get().strip()
                if not text:
                    continue
                path = Path(text)
                if attr == 'align_stems_root' and path.is_dir():
                    resolved = resolve_with_original_dir(path)
                    if resolved.is_dir():
                        return resolved
                elif path.is_dir():
                    return path
            stems_root = getattr(attrs_host, 'align_stems_root', None)
            if stems_root is not None:
                text = stems_root.get().strip()
                if text:
                    candidate = default_with_original_dir(Path(text))
                    if candidate.is_dir():
                        return candidate
        except (AttributeError, tk.TclError, OSError):
            pass
        return None

    def _library_from_parent(self) -> Path | None:
        parent = self._parent_ref
        if hasattr(parent, '_classify_mode_active') and parent._classify_mode_active():
            return self._organize_library_from_parent()
        return self._align_library_from_parent()

    def _refresh_song_library(self, *, keep_folder: Path | None = None) -> None:
        if self._library_root is None or not self._library_root.is_dir():
            self._song_folders = []
            self._folder_index = -1
            return
        self._song_folders = list_player_song_folders(self._library_root)
        if keep_folder is not None:
            try:
                self._folder_index = _index_song_folder(self._song_folders, keep_folder)
                return
            except ValueError:
                pass
        if self._folder is not None:
            try:
                self._folder_index = _index_song_folder(self._song_folders, self._folder)
                return
            except ValueError:
                pass
        if self._folder_index >= len(self._song_folders):
            self._folder_index = max(0, len(self._song_folders) - 1)

    def _sync_folder_index(self, folder: Path | None = None) -> None:
        target = folder if folder is not None else self._folder
        if target is None or not self._song_folders:
            return
        try:
            self._folder_index = _index_song_folder(self._song_folders, target)
        except ValueError:
            self._folder_index = -1

    def _resolve_library_root(self, folder: Path | None = None) -> Path | None:
        probe = folder if folder is not None else self._folder
        configured = self._library_from_parent()

        if configured is not None and configured.is_dir():
            if probe is None or _folder_in_song_library(probe, configured):
                self._library_root = configured
                return configured

        if self._library_root is not None and self._library_root.is_dir():
            if probe is None or _folder_in_song_library(probe, self._library_root):
                return self._library_root

        if probe is not None:
            containing = _library_root_containing(probe)
            if containing is not None:
                self._library_root = containing
                return containing
            parent = probe.parent
            if parent.is_dir() and any(
                child.is_dir() and child.name != BACKUP_DIR_NAME
                for child in parent.iterdir()
            ):
                self._library_root = parent
                return parent

        if configured is not None and configured.is_dir():
            self._library_root = configured
            return configured
        return None

    def _ensure_song_library(self) -> bool:
        library = self._resolve_library_root(self._folder)
        if library is None:
            return False
        if self._library_root != library or not self._song_folders:
            self._library_root = library
            self._refresh_song_library(keep_folder=self._folder)
        if self._folder is not None:
            self._sync_folder_index()
        elif self._folder_index >= len(self._song_folders):
            self._folder_index = max(-1, len(self._song_folders) - 1)
        return bool(self._song_folders)

    def _prepare_library(self, library_root: Path) -> None:
        """Remember library root and song list without loading any folder."""
        if not library_root.is_dir():
            return
        self._library_root = library_root
        keep = self._folder if self._folder is not None else None
        self._refresh_song_library(keep_folder=keep)
        if self._folder is None:
            self._folder_index = -1

    def _open_library(self, library_root: Path, *, start_index: int = 0) -> None:
        self._prepare_library(library_root)
        if not self._song_folders:
            messagebox.showwarning(
                'Stem player',
                f'No song folders found in:\n{library_root}',
                parent=self,
            )
            return
        index = max(0, min(start_index, len(self._song_folders) - 1))
        self._open_folder(self._song_folders[index], library_index=index)

    def _prev_song_folder(self) -> None:
        if self._folder_job_active:
            return
        if not self._ensure_song_library():
            return
        idx = self._folder_index
        if idx <= 0:
            return
        self._open_folder(
            self._song_folders[idx - 1],
            library_index=idx - 1,
        )

    def _next_song_folder(self) -> None:
        if self._folder_job_active:
            return
        if not self._ensure_song_library():
            return
        idx = self._folder_index
        if idx < 0:
            if self._folder is None and self._song_folders:
                self._open_folder(self._song_folders[0], library_index=0)
            return
        if idx >= len(self._song_folders) - 1:
            return
        self._open_folder(
            self._song_folders[idx + 1],
            library_index=idx + 1,
        )

    def _stop_playback_only(self) -> None:
        """Stop audio output but keep decoded stems in memory."""
        self._stop()
        if self._engine is not None:
            self._engine.set_playing(False)
            self._engine.stop_stream()
            self._engine = None
        if sys.platform == 'win32':
            self.update_idletasks()
            time.sleep(0.05)

    def _mark_folder_review(self, verdict: str) -> None:
        if self._folder is None or self._folder_job_active:
            return
        if not self._tracks:
            return
        folder = self._folder
        library_index = self._folder_index if self._folder_index >= 0 else None

        self._folder_job_active = True
        gen = self._begin_busy('saving')
        self._stop_playback_only()

        def worker() -> None:
            result: dict = {'new_path': None, 'err': None, 'kind': None}
            try:
                new_path = rename_folder_review(folder, verdict)
                result['new_path'] = new_path
            except FileExistsError as exc:
                result['err'] = exc
                result['kind'] = 'exists'
            except OSError as exc:
                result['err'] = exc
                result['kind'] = 'os'
            except Exception as exc:
                result['err'] = exc
                result['kind'] = 'load'
            self._enqueue_main(
                self._apply_mark_review, gen, folder, library_index, verdict, result,
            )

        threading.Thread(target=worker, daemon=True).start()

    def _apply_mark_review(
        self,
        gen: int,
        folder: Path,
        library_index: int | None,
        verdict: str,
        result: dict,
    ) -> None:
        reload_folder = False
        try:
            if gen != self._busy_generation:
                return
            err = result['err']
            if err is not None:
                reload_folder = folder.is_dir()
                if result['kind'] == 'exists':
                    messagebox.showerror('Stem player', str(err), parent=self)
                elif result['kind'] == 'os':
                    msg = f'Could not rename folder:\n{err}'
                    if getattr(err, 'winerror', None) == 5:
                        msg += (
                            '\n\nClose any File Explorer window showing this folder, '
                            'then try again.'
                        )
                    messagebox.showerror('Stem player', msg, parent=self)
                else:
                    messagebox.showerror('STEM player', str(err), parent=self)
                return

            new_path = result['new_path']
            for track in self._tracks:
                track.path = new_path / track.path.name

            self._folder = new_path
            self._title_var.set(new_path.name)
            self._set_title_interactive(True)
            if self._library_root is None:
                library = self._library_from_parent()
                if library is not None:
                    self._prepare_library(library)
            self._refresh_song_library(keep_folder=new_path)
            self._sync_folder_index(new_path)

            if not self._restart_engine():
                return
            self.after_idle(self._schedule_redraw)
            self._flash_title_bar(verdict)
        finally:
            self._folder_job_active = False
            self._end_busy(gen)
        if reload_folder:
            self._open_folder(folder, library_index=library_index)

    def _output_folder_from_parent(self) -> Path | None:
        return self._library_from_parent()

    def _load_folder(self) -> None:
        if sd is None:
            messagebox.showerror(
                'Stem player',
                'Audio playback requires the sounddevice package.\n\n'
                'Re-run install-deps.bat to install it, then restart the app.',
                parent=self,
            )
            return

        library = self._library_from_parent()

        initial = self._folder or library or self._library_root
        initial_dir = str(initial) if initial is not None and Path(initial).is_dir() else None

        folder = filedialog.askdirectory(
            title='Load stem folder',
            initialdir=initial_dir,
            parent=self,
        )
        if not folder:
            return
        folder_path = Path(folder)
        containing = _library_root_containing(folder_path)
        if containing is not None:
            self._library_root = containing
        elif library is not None and library.is_dir():
            self._library_root = library
        self._open_folder(folder_path)

    def _open_folder(
        self,
        folder: Path,
        *,
        library_index: int | None = None,
        busy_mode: str = 'loading',
    ) -> None:
        if self._folder_job_active:
            return
        self._end_title_flash()
        self._folder_job_active = True
        self._pending_open = (folder, library_index)
        gen = self._begin_busy(busy_mode)
        self.after(0, lambda g=gen: self._open_folder_begin(g))

    def _open_folder_begin(self, gen: int) -> None:
        if gen != self._busy_generation or self._pending_open is None:
            return
        folder, library_index = self._pending_open
        self._pending_open = None
        stems = detect_stem_folder(folder)
        if not stems:
            self._folder_job_active = False
            self._end_busy(gen)
            messagebox.showwarning(
                'Stem player',
                'No stem files found in this folder.\n\n'
                'Expected instrumental, acapella, and/or (original song) audio files.',
                parent=self,
            )
            return

        self._stop()
        if self._engine is not None:
            self._engine.stop_stream()
            self._engine = None

        target = folder
        lib_idx = library_index
        cache_key = _folder_cache_key(target, stems)
        cached = self._get_folder_cache(cache_key)

        def worker() -> None:
            err: Exception | None = None
            tracks: list[_TrackState] | None = None
            try:
                if cached is not None:
                    tracks = _tracks_from_cache(cached, stems)
                else:
                    tracks = _load_tracks_from_stems(stems)
                    self._put_folder_cache(cache_key, tracks)
            except Exception as exc:
                err = exc
            self._enqueue_main(
                self._apply_open_folder, gen, target, lib_idx, tracks, err,
            )

        threading.Thread(target=worker, daemon=True).start()

    def _apply_open_folder(
        self,
        gen: int,
        folder: Path,
        library_index: int | None,
        tracks: list[_TrackState] | None,
        err: Exception | None,
    ) -> None:
        if gen != self._busy_generation:
            return
        if err is not None or tracks is None:
            messagebox.showerror(
                'STEM player',
                str(err) if err is not None else 'Load failed.',
                parent=self,
            )
            self._folder_job_active = False
            self._end_busy(gen)
            return

        self._set_folder_metadata(
            folder,
            library_index=library_index,
            refresh_library=library_index is None,
        )
        self._tracks = tracks
        self._view_zoom = WAVE_ZOOM_MIN
        self._view_start = 0.0
        if not self._restart_engine():
            self._folder_job_active = False
            self._end_busy(gen)
            return

        self.after(0, lambda g=gen: self._finish_open_folder_ui(g))

    def _clear_track_rows(self) -> None:
        for child in self._tracks_inner.winfo_children():
            child.destroy()
        self._playhead_ids.clear()
        for i in range(8):
            self._tracks_inner.rowconfigure(i, weight=0, uniform='')

    def _build_one_track_row(
        self,
        track: _TrackState,
        index: int,
        total: int,
        stem_roles: set[str],
    ) -> None:
        C = self._colors
        self._tracks_inner.rowconfigure(index, weight=1, uniform='track')
        pad_bottom = TRACK_ROW_GAP if index < total - 1 else 0
        row = tk.Frame(self._tracks_inner, bg=C['bg'])
        row.grid(row=index, column=0, sticky='nsew', pady=(0, pad_bottom))
        track.row = row

        ctrl = tk.Frame(row, bg=C['bg'], width=CONTROLS_W)
        ctrl.pack(side='left', fill='y', padx=(0, 8))
        ctrl.pack_propagate(False)

        tk.Label(
            ctrl, text=_stem_row_label(track.name, stem_roles),
            bg=C['bg'], fg=track.color, font=('Segoe UI Semibold', 10),
            anchor='w', width=12,
        ).pack(anchor='w', padx=(4, 0))

        btn_row = tk.Frame(ctrl, bg=C['bg'])
        btn_row.pack(anchor='w', pady=(4, 0), padx=2)

        solo_btn = self._ctk_icon_btn(
            btn_row, 'S', lambda t=track: self._toggle_solo(t),
            text_color=C['fg_dim'], width=28,
        )
        solo_btn.pack(side='left', padx=(0, 4))
        _bind_tooltip(solo_btn, 'Solo')
        mute_btn = self._ctk_icon_btn(
            btn_row, 'M', lambda t=track: self._toggle_mute(t),
            text_color=C['fg_dim'], width=28,
        )
        mute_btn.pack(side='left')
        _bind_tooltip(mute_btn, 'Mute')

        vol_var = tk.DoubleVar(value=100.0)

        def _vol_cb(_v, t=track, vv=vol_var):
            t.volume = float(vv.get()) / 100.0

        vol_scale = self._ctk_volume_slider(
            ctrl, vol_var, _vol_cb, width=CONTROLS_W - 16,
        )
        vol_scale.pack(fill='x', padx=4, pady=(6, 0))

        wave = tk.Canvas(
            row, bg=C['log_bg'],
            highlightthickness=1, highlightbackground=C['border'],
            highlightcolor=C['border'], takefocus=0,
        )
        wave.pack(side='left', fill='both', expand=True)
        wave.bind('<Button-1>', lambda e, t=track: self._on_waveform_click(e, t))
        self._bind_zoom_wheel(wave)
        track.wave_canvas = wave

        track._solo_btn = solo_btn
        track._mute_btn = mute_btn

    def _rebuild_track_rows(self) -> None:
        self._clear_track_rows()
        n = len(self._tracks)
        self._tracks_inner.columnconfigure(0, weight=1)
        stem_roles = {track.name for track in self._tracks}
        for i, track in enumerate(self._tracks):
            self._build_one_track_row(track, i, n, stem_roles)
        self._bind_keys_on_widget(self._tracks_inner)

    def _update_solo_btn(self, track: _TrackState) -> None:
        btn = track._solo_btn
        if btn is None:
            return
        C = self._colors
        if track.solo:
            btn.configure(
                fg_color=C['accent'],
                hover_color=C['accent_hov'],
                text_color='#ffffff',
            )
        else:
            btn.configure(
                fg_color=C['panel2'],
                hover_color=C['border'],
                text_color=C['fg_dim'],
            )

    def _update_mute_btn(self, track: _TrackState) -> None:
        btn = track._mute_btn
        if btn is None:
            return
        C = self._colors
        if track.muted:
            btn.configure(
                fg_color=C['danger'],
                hover_color=C['danger'],
                text_color='#ffffff',
            )
        else:
            btn.configure(
                fg_color=C['panel2'],
                hover_color=C['border'],
                text_color=C['fg_dim'],
            )

    def _toggle_solo(self, track: _TrackState) -> None:
        track.solo = not track.solo
        if track.solo:
            track.muted = False
        self._update_solo_btn(track)
        self._update_mute_btn(track)
        self._redraw_track_waveforms()

    def _toggle_mute(self, track: _TrackState) -> None:
        track.muted = not track.muted
        if track.muted:
            track.solo = False
        self._update_mute_btn(track)
        self._update_solo_btn(track)
        self._redraw_track_waveforms()

    def _any_solo(self) -> bool:
        return any(t.solo for t in self._tracks)

    def _waveform_dimmed(self, track: _TrackState) -> bool:
        if track.muted:
            return True
        if self._any_solo() and not track.solo:
            return True
        return False

    def _waveform_display_color(self, track: _TrackState) -> str:
        if not self._waveform_dimmed(track):
            return track.color
        return _blend_hex(track.color, self._colors['log_bg'], WAVEFORM_DIM_BLEND)

    def _redraw_track_waveforms(self) -> None:
        for track in self._tracks:
            if track.peaks is not None:
                self._draw_waveform(track)
        if self._engine is not None:
            self._update_playhead(self._engine.position)

    def _duration(self) -> float:
        return self._engine.duration if self._engine else 0.0

    def _view_duration(self) -> float:
        dur = self._duration()
        if dur <= 0:
            return 0.0
        return dur / max(WAVE_ZOOM_MIN, self._view_zoom)

    def _view_end(self) -> float:
        return min(self._duration(), self._view_start + self._view_duration())

    def _clamp_wave_view(self) -> None:
        dur = self._duration()
        if dur <= 0:
            self._view_zoom = WAVE_ZOOM_MIN
            self._view_start = 0.0
            return
        vis = self._view_duration()
        if vis >= dur - 1e-9:
            self._view_zoom = WAVE_ZOOM_MIN
            self._view_start = 0.0
            return
        max_start = max(0.0, dur - vis)
        self._view_start = max(0.0, min(self._view_start, max_start))

    def _peaks_for_view(self, track: _TrackState, bins: int):
        _ensure_player_audio_deps()
        full = track.peaks_full
        dur = self._duration()
        if full is None or dur <= 0:
            return _np.zeros(max(1, bins), dtype=_np.float32)
        n = len(full)
        i0 = int(max(0, min(n - 1, (self._view_start / dur) * n)))
        i1 = int(max(i0 + 1, min(n, (self._view_end() / dur) * n)))
        region = full[i0:i1]
        if region.size == 0:
            return _np.zeros(max(1, bins), dtype=_np.float32)
        if region.size == bins:
            return region
        src_x = _np.arange(region.size, dtype=_np.float32)
        dst_x = _np.linspace(0, region.size - 1, bins, dtype=_np.float32)
        return _np.interp(dst_x, src_x, region).astype(_np.float32)

    def _follow_playhead(self, pos: float) -> bool:
        if self._view_zoom <= WAVE_ZOOM_MIN + 1e-9:
            return False
        if self._engine is None or not self._engine.playing:
            return False
        vis = self._view_duration()
        dur = self._duration()
        if vis <= 0 or dur <= 0:
            return False
        target = pos - vis * WAVE_FOLLOW_POS
        max_start = max(0.0, dur - vis)
        new_start = max(0.0, min(target, max_start))
        if abs(new_start - self._view_start) < 1e-4:
            return False
        self._view_start = new_start
        return True

    def _time_to_x(self, seconds: float, width: float) -> float:
        vis = self._view_duration()
        if width <= 0 or vis <= 0:
            return 0.0
        return ((seconds - self._view_start) / vis) * width

    def _bind_zoom_wheel(self, widget: tk.Misc) -> None:
        widget.bind('<Control-MouseWheel>', self._on_zoom_wheel, add='+')
        if sys.platform != 'win32':
            widget.bind('<Control-Button-4>', self._on_zoom_wheel, add='+')
            widget.bind('<Control-Button-5>', self._on_zoom_wheel, add='+')

    def _on_zoom_wheel(self, event) -> str:
        if not self._tracks or self._engine is None:
            return 'break'

        if getattr(event, 'delta', 0):
            direction = 1 if event.delta > 0 else -1
        elif getattr(event, 'num', None) == 4:
            direction = 1
        elif getattr(event, 'num', None) == 5:
            direction = -1
        else:
            return 'break'

        try:
            widget = event.widget
            w = max(1, widget.winfo_width())
            x = float(event.x)
        except tk.TclError:
            return 'break'

        anchor_t = self._x_to_time(x, w)
        old_zoom = self._view_zoom
        new_zoom = old_zoom * (WAVE_ZOOM_STEP ** direction)
        new_zoom = max(WAVE_ZOOM_MIN, min(WAVE_ZOOM_MAX, new_zoom))
        if abs(new_zoom - old_zoom) < 1e-6:
            return 'break'

        self._view_zoom = new_zoom
        vis = self._view_duration()
        self._view_start = anchor_t - (x / w) * vis
        self._clamp_wave_view()
        self._redraw_sig = None
        self._redraw_wave_view(force=True)
        return 'break'

    def _seek_to(self, seconds: float) -> None:
        if self._engine is None:
            return
        self._engine.position = seconds
        self._time_var.set(format_time_ms(self._engine.position))
        self._update_playhead(self._engine.position)

    def _seek_relative(self, delta: float) -> None:
        if self._engine is None:
            return
        self._seek_to(self._engine.position + delta)

    def _toggle_play(self) -> None:
        if self._engine is None:
            return
        if self._engine.playing:
            self._engine.set_playing(False)
            self._btn_play.configure(text='⏵')
        else:
            if self._engine.position >= self._duration() - 0.01:
                self._engine.position = 0.0
            self._engine.set_playing(True)
            self._btn_play.configure(text='⏸')

    def _stop(self) -> None:
        if self._engine is None:
            return
        self._engine.set_playing(False)
        self._engine.position = 0.0
        self._btn_play.configure(text='⏵')
        self._time_var.set(format_time_ms(0))
        self._update_playhead(0)

    def _x_to_time(self, x: float, width: float) -> float:
        vis = self._view_duration()
        if width <= 0 or vis <= 0:
            return 0.0
        frac = max(0.0, min(1.0, x / width))
        t = self._view_start + frac * vis
        return max(0.0, min(self._duration(), t))

    def _on_timeline_click(self, event) -> None:
        w = self._timeline_canvas.winfo_width()
        self._seek_to(self._x_to_time(event.x, w))

    def _on_waveform_click(self, event, _track: _TrackState) -> None:
        w = event.widget.winfo_width()
        self._seek_to(self._x_to_time(event.x, w))

    def _track_row_height(self) -> int:
        for track in self._tracks:
            c = track.wave_canvas
            if c is not None:
                h = c.winfo_height()
                if h > 2:
                    return h
        outer_h = self._tracks_outer.winfo_height()
        n = max(1, len(self._tracks))
        if outer_h > 2:
            gaps = TRACK_ROW_GAP * max(0, n - 1)
            return max(TRACK_ROW_H, (outer_h - gaps) // n)
        return TRACK_ROW_H

    def _on_tracks_area_resize(self, event) -> None:
        if event.widget is not self._tracks_outer:
            return
        self._schedule_redraw()

    def _waveform_width(self) -> int:
        for track in self._tracks:
            c = track.wave_canvas
            if c is not None:
                w = c.winfo_width()
                if w > 2:
                    return w
        tw = self._timeline_canvas.winfo_width()
        if tw > 2:
            return tw
        return max(200, self.winfo_width() - CONTROLS_W - 48)

    def _redraw_wave_view(self, *, force: bool = False) -> None:
        if not self._tracks or self._engine is None:
            return

        wave_w = self._waveform_width()
        sig = (
            wave_w,
            self._view_zoom,
            round(self._view_start, 5),
            tuple((t.name, t.muted, t.solo) for t in self._tracks),
        )
        if not force and sig == self._redraw_sig:
            self._update_playhead(self._engine.position)
            return
        self._redraw_sig = sig
        self._wave_w = wave_w

        bins = max(100, min(wave_w, 1024))
        for track in self._tracks:
            track.peaks = self._peaks_for_view(track, bins)
            self._draw_waveform(track)

        self._draw_timeline(wave_w, self._duration())
        self._update_playhead(self._engine.position)

    def _schedule_redraw(self, _event=None) -> None:
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = self.after(60, lambda: self._redraw_all(force=True))

    def _redraw_all(self, *, force: bool = False) -> None:
        if not self._tracks or self._engine is None:
            return
        self._tracks_outer.update_idletasks()
        self._redraw_wave_view(force=force)

    def _on_window_resize(self, event) -> None:
        if event.widget is not self or getattr(self, '_resize_guard', False):
            return
        maximized = self._is_maximized()
        if maximized:
            if not self._was_maximized:
                self._allow_maximized_size()
            self._was_maximized = True
            self._schedule_rounded_corners()
            self._schedule_redraw()
            return
        if getattr(self, '_was_maximized', False):
            self._was_maximized = False
            self._allow_maximized_size()
            self._schedule_rounded_corners()
            self.after_idle(lambda: self._redraw_all(force=True))
            return
        self._schedule_rounded_corners()
        self._schedule_redraw()

    def _draw_timeline(self, width: int, duration: float) -> None:
        c = self._timeline_canvas
        width = max(1, int(c.winfo_width()) or width)
        c.delete('all')
        h = TIMELINE_H
        c.configure(scrollregion=(0, 0, width, h))
        c.create_line(0, h - 1, width, h - 1, fill=self._colors['border'])

        if duration <= 0:
            return

        vis = self._view_duration()
        vis_start = self._view_start
        vis_end = self._view_end()
        if vis <= 0:
            return

        interval = 30.0
        if vis > 600:
            interval = 60.0
        elif vis < 60:
            interval = 10.0
        if vis < 15:
            interval = 5.0
        if vis < 5:
            interval = 1.0

        t = vis_start
        while t <= vis_end + 0.01:
            x = self._time_to_x(t, width)
            c.create_line(x, h - 12, x, h - 1, fill=self._colors['fg_dim'])
            c.create_text(
                x + 2, 4, text=format_ruler_time(t), anchor='nw',
                fill=self._colors['fg_dim'], font=('Segoe UI', 8),
            )
            t += interval

    def _draw_waveform(self, track: _TrackState) -> None:
        c = track.wave_canvas
        if c is None or track.peaks is None:
            return
        c.delete('all')
        w = max(1, self._wave_w)
        h = max(1, int(c.winfo_height()) or self._track_row_height())
        peaks = _np.asarray(track.peaks, dtype=_np.float32)
        n = len(peaks)
        if n == 0:
            return

        mid = h / 2
        color = self._waveform_display_color(track)
        bar_w = w / n
        max_amp = mid - 6
        amps = _np.minimum(peaks, 1.0) * max_amp
        x_left = _np.arange(n, dtype=_np.float32) * bar_w
        x_right = x_left + max(1.0, bar_w - 0.5)
        y_top = mid - amps
        y_bot = mid + amps

        top = _np.column_stack([x_left, y_top])
        bot = _np.column_stack([x_right[::-1], y_bot[::-1]])
        pts = _np.vstack([top, bot]).reshape(-1).tolist()
        c.create_polygon(pts, fill=color, outline='')
        self._draw_waveform_filename(c, track, w, h)

    def _draw_waveform_filename(
        self, canvas: tk.Canvas, track: _TrackState, width: int, height: int,
    ) -> None:
        name = track.path.name
        font = ('Segoe UI', 8)
        pad_x, pad_y = 10, 8
        text_x = width - pad_x
        text_y = height - pad_y

        measure = canvas.create_text(0, 0, text=name, font=font, anchor='se')
        bbox = canvas.bbox(measure)
        canvas.delete(measure)
        if not bbox:
            return

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        inset = 4
        x0 = max(2, text_x - tw - inset)
        y0 = text_y - th - inset
        x1 = min(width - 2, text_x + inset)
        y1 = min(height - 2, text_y + inset)
        bg = self._colors['log_bg']
        canvas.create_rectangle(
            x0, y0, x1, y1, fill=bg, outline=bg, tags='filename',
        )
        canvas.create_text(
            text_x, text_y, text=name, anchor='se',
            fill=self._colors['fg_dim'], font=font, tags='filename',
        )

    def _update_playhead(self, position: float) -> None:
        dur = self._duration()
        if dur <= 0:
            return

        for track in self._tracks:
            c = track.wave_canvas
            if c is None:
                continue
            w = max(1, self._wave_w)
            h = max(1, int(c.winfo_height()) or self._track_row_height())
            x = self._time_to_x(position, w)
            c.delete('playhead')
            c.create_line(x, 0, x, h, fill='#ffffff', width=1, tags='playhead')

        tc = self._timeline_canvas
        tw = max(1, int(tc.winfo_width()) or self._wave_w)
        tx = self._time_to_x(position, tw)
        tc.delete('playhead')
        tc.create_line(tx, 0, tx, TIMELINE_H, fill='#ffffff', width=1, tags='playhead')

    def _on_close(self) -> None:
        self._unbind_folder_nav_keys_global()
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        if self._tick_id is not None:
            try:
                self.after_cancel(self._tick_id)
            except tk.TclError:
                pass
        if self._busy_hide_after_id is not None:
            try:
                self.after_cancel(self._busy_hide_after_id)
            except tk.TclError:
                pass
        if self._engine is not None:
            self._engine.set_playing(False)
            self._engine.stop_stream()
        self.destroy()


def open_stem_player(parent: tk.Misc) -> None:
    """Open (or focus) the stem preview player window."""
    from ui_theme import COLORS

    _ensure_player_audio_deps()

    existing = getattr(parent, '_stem_player_window', None)
    if existing is not None and existing.winfo_exists():
        existing._place_and_show()
        library = existing._library_from_parent()
        if library is not None:
            if library != existing._library_root:
                existing._prepare_library(library)
            else:
                existing._refresh_song_library(keep_folder=existing._folder)
                if existing._folder is not None:
                    existing._sync_folder_index()
        return

    win = StemPlayerWindow(parent, colors=COLORS)
    parent._stem_player_window = win  # type: ignore[attr-defined]
    library = win._library_from_parent()
    if library is not None:
        win._prepare_library(library)
    win.after_idle(win._place_and_show)
