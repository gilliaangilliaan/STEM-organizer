from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import traceback
import webbrowser
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile


AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.aif', '.aiff', '.ogg', '.m4a', '.opus')

MODELS = {
    'htdemucs (good)':                  'htdemucs',
    'htdemucs_ft (best, slowest)':               'htdemucs_ft',
    'htdemucs_6s (worst, fastest)':            'htdemucs_6s',
}

STEM_MODES = {
    'Vocals + Instrumental': {
        'categories': ('vocals', 'instrumental'),
        'mapping':    {'vocals': 'vocals'},
        'fallback':   'instrumental',
    },
    '4-way (drums/bass/other/vocals)': {
        'categories': ('drums', 'bass', 'other', 'vocals'),
        'mapping':    {n: n for n in ('drums', 'bass', 'other', 'vocals')},
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

NAMING_MODES = {
    'Folder name (simplified)':        'slug',
    'Sequential (song_0000, 0001, …)': 'sequential',
}

MANIFEST_FILENAME = 'index.json'
_SEQ_RE = re.compile(r'^song_(\d+)$')
FFMPEG = shutil.which('ffmpeg')
_ALLOWED_NAME_CHARS = set('abcdefghijklmnopqrstuvwxyz0123456789')


def slugify(name: str) -> str:
    s = ''.join(c for c in name.lower() if c in _ALLOWED_NAME_CHARS)
    return s or 'folder'


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


def load_audio(path: str, sr: int, ch: int = 2) -> np.ndarray:
    p = Path(path)
    sf_exts = {'.wav', '.flac', '.aif', '.aiff', '.ogg'}
    if p.suffix.lower() in sf_exts:
        try:
            data, file_sr = sf.read(str(p), dtype='float32', always_2d=True)
            # sf.read returns (samples, channels); convert to (channels, samples)
            audio = data.T
            # Normalise channel count
            if audio.shape[0] == 1:
                audio = np.repeat(audio, ch, axis=0)
            elif audio.shape[0] > ch:
                audio = audio[:ch]
            # Resample if needed
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
        except Exception:
            pass  # fall through to AudioFile / ffmpeg
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
                yield (fp, None, f'load failed: {e}')
                continue
            audios.append(a)
            lengths.append(a.shape[1])
            valid.append(fp)

        if not audios:
            continue

        max_len = max(lengths)
        batch = np.zeros((len(audios), 2, max_len), dtype=np.float32)
        for i, a in enumerate(audios):
            batch[i, :, :a.shape[1]] = a

        try:
            with torch.no_grad():
                out = apply_model(
                    model, torch.from_numpy(batch).to(device),
                    device=device, progress=False, split=True, overlap=0.25,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(valid) == 1:
                yield (valid[0], None, 'cuda OOM')
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

    if top_share < threshold or margin < min_margin:
        return ('SKIP', top_cat, top_share, margin)
    return (top_cat, top_cat, top_share, margin)


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


def find_duplicates(paths, sr: int, log_fn=None, threshold: float = 0.05):
    if len(paths) < 2:
        return list(paths)

    audios = {}
    for p in paths:
        try:
            audios[p] = load_audio(str(p), sr=sr)
        except Exception:
            pass

    items = [(p, a) for p, a in audios.items() if a.shape[1] >= sr]
    n = len(items)
    if n < 2:
        return list(paths)

    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            ai, aj = items[i][1], items[j][1]
            L = min(ai.shape[1], aj.shape[1])
            denom = max(_rms(ai[:, :L]), _rms(aj[:, :L]), 1e-12)
            if _rms(ai[:, :L] - aj[:, :L]) / denom < threshold:
                uf.union(i, j)

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


DONE_SENTINEL = object()


class Worker(threading.Thread):
    def __init__(self, params: dict, log_q: queue.Queue):
        super().__init__(daemon=True)
        self.p = params
        self.q = log_q
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def log(self, msg: str):
        self.q.put(msg)

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.log(f'[ERROR] {e}')
            self.log(traceback.format_exc())
        finally:
            self.q.put(DONE_SENTINEL)

    def _resolve_output_dir(self, out_dir: Path, rel: Path,
                            manifest: dict, next_n_ref: list) -> tuple[Path, dict, int]:
        naming_mode = self.p['naming_mode']
        if naming_mode == 'sequential':
            name = f"song_{next_n_ref[0]:04d}"
            target_dir = out_dir / name
            manifest[name] = str(rel).replace('\\', '/')
            save_manifest(out_dir, manifest)
            next_n_ref[0] += 1
            self.log(f"  -> {name}  (original: {rel})")
        else:
            slug_parts = [slugify(pp) for pp in rel.parts if pp not in ('', '.')]
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
                              gain: float, cut: int) -> None:
        for cat in mode_cfg['categories']:
            if cat not in mixes:
                self.log(f"  ({cat}: no stems)")
                continue
            scaled = mixes[cat][:, :cut] * gain
            out_path = target_dir / f"{cat}{ext}"
            try:
                write_audio(str(out_path), scaled, sr, subtype)
                self.log(f"  wrote {cat}{ext}  ({len(buckets[cat])} stems, {cut/sr:.2f}s)")
            except Exception as e:
                self.log(f"  [export error] {cat}: {e}")

    def _process_folder(self, folder: Path, stems: list, model, device: str,
                        mode_cfg: dict, ext: str, subtype: str, sr: int,
                        out_dir: Path, manifest: dict, next_n_ref: list) -> tuple[dict, int]:
        if self.p['dedup']:
            before = len(stems)
            stems = find_duplicates(stems, sr=sr, log_fn=self.log)
            if len(stems) < before:
                self.log(f"  [dedup] {before} -> {len(stems)} stems after deduplication")

        buckets = {c: [] for c in mode_cfg['categories']}
        skipped = 0
        had_ambig = False

        for path, energies, err in classify_batch(
                model, stems, device, batch_size=int(self.p['batch_size']), stop_event=self._stop):
            if self._stop.is_set():
                self.log('Stopped by user.')
                if device == 'cuda':
                    model.cpu()
                    torch.cuda.empty_cache()
                return manifest, next_n_ref[0]
            if err:
                self.log(f"  [skip] {path.name}: {err}")
                skipped += 1
                continue
            label, top_cat, top_share, margin = classify_to_category(
                energies, mode_cfg, float(self.p['threshold']), float(self.p['min_margin']))
            bar = '#' * int(top_share * 20)
            self.log(f"  {path.name}")
            self.log(f"    top={top_cat:<12} {top_share:5.1%} {bar}  margin={margin:+.1%}  -> {label}")
            if label == 'SKIP':
                skipped += 1
                had_ambig = True
            else:
                buckets[label].append(path)

        if had_ambig and self.p['ambig_mode'] == 'skip_song':
            self.log('  [skip song] ambiguous stem(s) detected; skipping entire song')
            return manifest, next_n_ref[0]

        in_dir = Path(self.p['input_dir'])
        rel = folder.relative_to(in_dir) if folder != in_dir else Path('.')
        target_dir, manifest, _ = self._resolve_output_dir(out_dir, rel, manifest, next_n_ref)

        mixes = {}
        for cat, paths in buckets.items():
            if not paths:
                continue
            m = mix_originals(paths, sr=sr)
            if m.shape[1] == 0:
                self.log(f"  [error] {cat}: all stems failed to load")
                continue
            mixes[cat] = m

        cut = min((m.shape[1] for m in mixes.values()), default=0)
        gain = self._compute_gain(mixes, cut)

        self._write_category_mixes(mixes, buckets, mode_cfg, target_dir, ext, subtype, sr, gain, cut)

        if self.p['make_mixture'] and ext == '.wav' and cut > 0:
            total = sum(m[:, :cut] for m in mixes.values()) * gain
            mix_path = target_dir / f"mixture{ext}"
            try:
                write_audio(str(mix_path), total, sr, subtype)
                n = sum(len(buckets[c]) for c in mixes)
                peak_db = 20 * np.log10(max(float(np.max(np.abs(total))), 1e-12))
                self.log(f"  wrote mixture{ext}  ({n} stems, peak {peak_db:+.2f} dBFS)")
            except Exception as e:
                self.log(f"  [export error] mixture: {e}")

        if skipped:
            self.log(f"  ({skipped} stem(s) skipped)")

        return manifest, next_n_ref[0]

    def _run(self):
        model = None
        device = 'cpu'
        try:
            p = self.p
            device = 'cuda' if (p['use_cuda'] and torch.cuda.is_available()) else 'cpu'
            if p['use_cuda'] and not torch.cuda.is_available():
                self.log('[warn] CUDA not available, using CPU.')
            self.log(f"Device: {device}")
            self.log(f"Loading model '{p['model_id']}' ...")
            model = get_model(p['model_id']).eval().to(device)
            self.log(f"Model sources: {list(model.sources)}  (sr={model.samplerate})")

            in_dir, out_dir = Path(p['input_dir']), Path(p['output_dir'])
            mode_cfg = STEM_MODES[p['stem_mode']]
            ext, subtype = QUALITY_PRESETS[p['quality']].values()
            sr = model.samplerate

            groups: dict[Path, list[Path]] = {}
            for f in in_dir.rglob('*'):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    groups.setdefault(f.parent, []).append(f)

            if not groups:
                self.log(f"[skip] no audio files found under {in_dir}")
                return
            self.log(f"Found {sum(len(v) for v in groups.values())} stem(s) across {len(groups)} folder(s).")

            manifest: dict = {}
            next_n_ref = [0]
            if p['naming_mode'] == 'sequential':
                manifest = load_manifest(out_dir)
                next_n_ref[0] = next_sequence_number(out_dir, manifest)
                self.log(f"Naming: sequential; resuming at song_{next_n_ref[0]:04d}")
            else:
                self.log('Naming: simplified folder name')

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

            self.log('')
            self.log('Done.')
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
}


def apply_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass

    base  = ('Segoe UI', 10)
    bold  = ('Segoe UI Semibold', 10)
    title = ('Segoe UI Semibold', 16)
    C = COLORS

    root.configure(bg=C['bg'])
    root.option_add('*Font', base)

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
        'Status.TLabel':             {'background': C['panel'], 'foreground': C['fg_dim'], 'padding': (10, 6)},
        'TLabelframe':               {'background': C['bg'], 'foreground': C['fg'],
                                      'bordercolor': C['border'], 'relief': 'solid', 'borderwidth': 1},
        'TLabelframe.Label':         {'background': C['bg'], 'foreground': C['fg_dim'], 'font': bold},
        'TEntry':                    {'fieldbackground': C['panel2'], 'foreground': C['fg'],
                                      'bordercolor': C['border'], 'insertcolor': C['fg'], 'padding': 6},
        'TCombobox':                 {'fieldbackground': C['panel2'], 'background': C['panel2'],
                                      'foreground': C['fg'], 'arrowcolor': C['fg_dim'],
                                      'bordercolor': C['border'], 'padding': 4,
                                      'selectbackground': C['panel2'], 'selectforeground': C['fg'],
                                      'insertcolor': C['fg']},
        'TCheckbutton':              {'background': C['bg'], 'foreground': C['fg'],
                                      'indicatorcolor': C['panel2']},
        'TButton':                   {'background': C['panel2'], 'foreground': C['fg'],
                                      'bordercolor': C['border'], 'padding': (14, 8), 'borderwidth': 1},
        'Accent.TButton':            {'background': C['accent'], 'foreground': 'white',
                                      'bordercolor': C['accent'], 'padding': (18, 9), 'font': bold},
        'Danger.TButton':            {'background': C['panel2'], 'foreground': C['danger'],
                                      'bordercolor': C['border'], 'padding': (14, 8)},
        'Horizontal.TProgressbar':   {'background': C['accent'], 'troughcolor': C['panel2'],
                                      'bordercolor': C['panel2'],
                                      'lightcolor': C['accent'], 'darkcolor': C['accent']},
        'Horizontal.TScale':         {'background': C['bg'], 'troughcolor': C['panel2'],
                                      'bordercolor': C['border'],
                                      'lightcolor': C['accent'], 'darkcolor': C['accent']},
        'TSpinbox':                  {'fieldbackground': C['panel2'], 'foreground': C['fg'],
                                      'background': C['panel2'], 'bordercolor': C['border'],
                                      'arrowcolor': C['fg_dim'], 'insertcolor': C['fg']},
    }
    for name, opts in cfgs.items():
        style.configure(name, **opts)

    style.map('TEntry',    bordercolor=[('focus', C['accent'])])
    style.map('TSpinbox',  bordercolor=[('focus', C['accent'])])
    style.map('TCombobox',
              fieldbackground=[('readonly', C['panel2']), ('!disabled', C['panel2'])],
              background=[('readonly', C['panel2']), ('active', C['panel2'])],
              foreground=[('readonly', C['fg']), ('hover', C['fg']),
                          ('focus', C['fg']), ('active', C['fg'])],
              selectbackground=[('readonly', C['panel2']), ('focus', C['panel2'])],
              selectforeground=[('readonly', C['fg']), ('focus', C['fg'])],
              arrowcolor=[('hover', C['fg']), ('active', C['fg'])],
              bordercolor=[('focus', C['accent'])])
    style.map('TCheckbutton',
              background=[('active', C['bg']), ('hover', C['bg'])],
              foreground=[('disabled', C['fg_dim']), ('active', C['fg']),
                          ('hover', C['fg']), ('focus', C['fg']), ('selected', C['fg'])],
              indicatorcolor=[('selected', C['accent']),
                              ('active', C['panel2']), ('hover', C['panel2'])])
    style.map('TButton',
              background=[('active', C['panel']), ('disabled', C['panel'])],
              foreground=[('disabled', C['fg_dim']), ('active', C['fg']), ('hover', C['fg'])])
    style.map('Accent.TButton',
              background=[('active', C['accent_hov']), ('disabled', C['panel2'])],
              foreground=[('disabled', C['fg_dim'])])
    style.map('Danger.TButton',
              background=[('active', C['panel'])],
              foreground=[('disabled', C['fg_dim'])])

    for k, v in (('background', C['panel2']), ('foreground', C['fg']),
                 ('selectBackground', C['accent']), ('selectForeground', C['fg'])):
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
                 background=COLORS['panel2'], foreground=COLORS['fg'],
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


def tip(*widgets, text: str) -> None:
    for w in widgets:
        Tooltip(w, text)


TIPS = {
    'input':      "Folder to scan. Every leaf folder containing audio files is treated as one 'song'.\nSubfolders are walked recursively.",
    'output':     "Where the grouped mixes are written. Input layout is mirrored:\ninput/song_01/* → output/song_01/vocals.flac + instrumental.flac",
    'cuda':       "Run the Demucs classifier on your GPU. Much faster than CPU.\nAuto-falls back to CPU on out-of-memory.",
    'model':      "Demucs model used as the classifier.",
    'stems':      "Output category layout:\n• Vocals + Instrumental - vocal stems → vocals.flac, rest → instrumental.flac\n• 4-way - drums / bass / other / vocals each get their own file.",
    'quality':    "Output file format:\n• FLAC 16-bit - lossless, CD quality, smallest\n• FLAC 24-bit - lossless, studio quality\n• WAV 16-bit - uncompressed PCM, CD quality\n• WAV 24-bit - uncompressed PCM, studio quality\n• WAV 32-bit float - uncompressed float, best for further processing\nFLAC uses ffmpeg compression level 12 when ffmpeg is on PATH.",
    'confidence': "Minimum share of total energy the dominant CATEGORY must reach for a stem to be accepted.\nExample at 35%: the winning category (vocals, or drums+bass+other combined in 2-stem mode) must hold ≥35% of total energy.",
    'margin':     "Minimum lead the dominant CATEGORY must have over the runner-up CATEGORY.\nIn 2-stem mode this measures vocals vs instrumental (drums+bass+other combined). In 4-way mode it measures the winning stem vs the next-loudest stem.\nA small margin means the stem is contaminated with content from another category.",
    'ambig':      "What to do when a stem is contaminated - i.e., more than one CATEGORY has significant energy (e.g., vocals mixed with instruments in 2-stem mode, or drums mixed with bass in 4-way mode):\n• Skip ambiguous stem only - drop just that stem, keep the rest.\n• Skip the entire song - abort this folder; no outputs are written.",
    'batch':      "Stems processed per GPU pass. Higher = faster, more VRAM.\nAuto-shrinks on out-of-memory.",
    'peak_norm':  "Apply a single gain to every category output so that, when summed back together, the mixture peaks at exactly -1 dBFS.\nDisable to keep raw summed levels (may clip).",
    'mixture':    "Also write 'mixture.wav' per folder - the sum of every accepted stem (skipped stems excluded).\nUseful for AI training datasets.\nAvailable only when output quality is a WAV format.",
    'dedup':      "Detect duplicate stems within each folder via phase-inversion null test.\nIf two stems cancel out when one is inverted (residual < 5% RMS), they're treated as the same content; only the one with the lowest peak dBFS is kept.\nRuns before classification, so duplicates never waste GPU time.",
    'naming':     "Output folder naming:\n• Folder name (simplified) - uses the input folder name, sanitized to a–z and 0–9.\n• Sequential - names folders song_0000, song_0001, … and continues past any existing numbered folders already in the output (no overwrite).\nIn sequential mode an 'index.json' is written at the output root, mapping each number to the original folder name so you can trace back later.",
    'start':      "Begin classifying and mixing. The UI stays responsive during the run.",
    'stop':       "Request a clean stop after the current folder finishes.",
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Demucs Stem Organizer')
        self.geometry('720x860')
        self.minsize(640, 720)

        self.input_dir    = tk.StringVar()
        self.output_dir   = tk.StringVar()
        self.use_cuda     = tk.BooleanVar(value=torch.cuda.is_available())
        self.model_label  = tk.StringVar(value=next(iter(MODELS)))
        self.stem_mode    = tk.StringVar(value=next(iter(STEM_MODES)))
        self.quality      = tk.StringVar(value='FLAC 16-bit')
        self.threshold    = tk.DoubleVar(value=0.35)
        self.min_margin   = tk.DoubleVar(value=0.15)
        self.batch_size   = tk.IntVar(value=4)
        self.peak_norm    = tk.BooleanVar(value=True)
        self.make_mixture = tk.BooleanVar(value=False)
        self.dedup        = tk.BooleanVar(value=False)
        self.ambig_label  = tk.StringVar(value=next(iter(AMBIG_MODES)))
        self.naming_label = tk.StringVar(value=next(iter(NAMING_MODES)))
        self.status_var   = tk.StringVar(value='Idle')

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker = None

        apply_theme(self)
        self._build_ui()
        self.after(100, self._drain_log)

    def _path_row(self, parent, row, label, var, picker, tip_text):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky='w', padx=(0, 10), pady=4)
        ent = ttk.Entry(parent, textvariable=var)
        ent.grid(row=row, column=1, sticky='ew', pady=4)
        btn = ttk.Button(parent, text='Browse…', command=picker)
        btn.grid(row=row, column=2, padx=(8, 0), pady=4)
        tip(lbl, ent, btn, text=tip_text)

    def _combo_field(self, parent, row, col, label, var, values, tip_text):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=col, sticky='w', padx=(0, 10), pady=6)
        cb = ttk.Combobox(parent, textvariable=var, values=values, state='readonly')
        cb.grid(row=row, column=col + 1, sticky='ew',
                padx=(0, 16) if col == 0 else 0, pady=6)
        for seq in ('<Control-a>', '<Control-A>', '<KeyPress>'):
            cb.bind(seq, lambda e: 'break')
        cb.bind('<<ComboboxSelected>>', lambda e: cb.selection_clear())
        cb.bind('<FocusIn>',            lambda e: cb.selection_clear())
        tip(lbl, cb, text=tip_text)

    def _slider_field(self, parent, row, col, label, var, lo, hi, fmt, tip_text):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=col, sticky='w', padx=(0, 10), pady=6)
        row_frm = ttk.Frame(parent)
        row_frm.grid(row=row, column=col + 1, sticky='ew',
                     padx=(0, 16) if col == 0 else 0, pady=6)
        row_frm.columnconfigure(0, weight=1)
        readout = ttk.Label(row_frm, text=fmt(var.get()), style='Dim.TLabel', width=5)
        scale = ttk.Scale(row_frm, from_=lo, to=hi, orient='horizontal', variable=var,
                          command=lambda _v: readout.configure(text=fmt(var.get())))
        scale.grid(row=0, column=0, sticky='ew')
        readout.grid(row=0, column=1, padx=(8, 0))
        tip(lbl, scale, readout, text=tip_text)

    def _build_ui(self):
        outer = ttk.Frame(self)
        outer.pack(fill='both', expand=True, padx=18, pady=14)

        header = ttk.Frame(outer)
        header.pack(fill='x', pady=(0, 12))
        ttk.Label(header, text='Demucs Stem Organizer', style='Title.TLabel').pack(anchor='w')
        ttk.Label(header,
                  text='Classifies each stem with Demucs, then mixes the ORIGINAL files into one cleanly-grouped output per folder.',
                  style='Subtitle.TLabel').pack(anchor='w', pady=(2, 0))
        if not FFMPEG:
            ttk.Label(header,
                      text='ffmpeg not found on PATH - FLAC will use libsndfile defaults (lower compression)',
                      style='Subtitle.TLabel').pack(anchor='w', pady=(4, 0))

        paths = ttk.LabelFrame(outer, text='  Paths  ', padding=12)
        paths.pack(fill='x', pady=(0, 10))
        paths.columnconfigure(1, weight=1)
        self._path_row(paths, 0, 'Input',  self.input_dir,  self._pick_input,  TIPS['input'])
        self._path_row(paths, 1, 'Output', self.output_dir, self._pick_output, TIPS['output'])

        opts = ttk.LabelFrame(outer, text='  Options  ', padding=12)
        opts.pack(fill='x', pady=(0, 10))
        opts.columnconfigure(1, weight=1)
        opts.columnconfigure(3, weight=1)
        self._combo_field(opts, 0, 0, 'Model',   self.model_label, list(MODELS),          TIPS['model'])
        self._combo_field(opts, 0, 2, 'Stems',   self.stem_mode,   list(STEM_MODES),      TIPS['stems'])
        self._combo_field(opts, 1, 0, 'Quality', self.quality,     list(QUALITY_PRESETS), TIPS['quality'])
        cuda_text = 'Use CUDA (GPU)' + ('' if torch.cuda.is_available() else '   ·   unavailable')
        cuda_chk = ttk.Checkbutton(opts, text=cuda_text, variable=self.use_cuda,
                                   state='normal' if torch.cuda.is_available() else 'disabled')
        cuda_chk.grid(row=1, column=2, columnspan=2, sticky='w', pady=6)
        Tooltip(cuda_chk, TIPS['cuda'])
        self._combo_field(opts, 2, 0, 'On ambiguous', self.ambig_label,  list(AMBIG_MODES),  TIPS['ambig'])
        self._combo_field(opts, 2, 2, 'Naming',       self.naming_label, list(NAMING_MODES), TIPS['naming'])

        cls = ttk.LabelFrame(outer, text='  Classification  ', padding=12)
        cls.pack(fill='x', pady=(0, 10))
        cls.columnconfigure(1, weight=1)
        cls.columnconfigure(3, weight=1)
        pct = lambda v: f"{v:.0%}"
        self._slider_field(cls, 0, 0, 'Confidence', self.threshold,  0.10, 0.90, pct, TIPS['confidence'])
        self._slider_field(cls, 0, 2, 'Min margin', self.min_margin, 0.00, 0.50, pct, TIPS['margin'])
        batch_lbl = ttk.Label(cls, text='Batch size')
        batch_lbl.grid(row=1, column=0, sticky='w', padx=(0, 10), pady=6)
        batch_sp = ttk.Spinbox(cls, from_=1, to=16, textvariable=self.batch_size, width=6)
        batch_sp.grid(row=1, column=1, sticky='w', pady=6)
        tip(batch_lbl, batch_sp, text=TIPS['batch'])
        peak_chk = ttk.Checkbutton(cls, text='Normalize so summed mixture peaks at -1 dBFS',
                                   variable=self.peak_norm)
        peak_chk.grid(row=1, column=2, columnspan=2, sticky='w', pady=6)
        Tooltip(peak_chk, TIPS['peak_norm'])
        dedup_chk = ttk.Checkbutton(cls, text='Remove duplicate stems (keep quietest)',
                                    variable=self.dedup)
        dedup_chk.grid(row=2, column=0, columnspan=4, sticky='w', pady=6)
        Tooltip(dedup_chk, TIPS['dedup'])
        self.mix_chk = ttk.Checkbutton(cls, text='Also write mixture.wav (WAV quality only)',
                                       variable=self.make_mixture)
        self.mix_chk.grid(row=3, column=0, columnspan=4, sticky='w', pady=6)
        Tooltip(self.mix_chk, TIPS['mixture'])
        self.quality.trace_add('write', lambda *_: self._update_mixture_state())
        self._update_mixture_state()

        actions = ttk.Frame(outer)
        actions.pack(fill='x', pady=(2, 10))
        self.start_btn = ttk.Button(actions, text='▶  Start', style='Accent.TButton', command=self._start)
        self.start_btn.pack(side='left')
        Tooltip(self.start_btn, TIPS['start'])
        self.stop_btn = ttk.Button(actions, text='■  Stop', style='Danger.TButton',
                                   command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=(8, 0))
        Tooltip(self.stop_btn, TIPS['stop'])
        self.progress = ttk.Progressbar(actions, mode='indeterminate', length=160)

        hf_url = 'https://huggingface.co/gilliaaan'
        footer = ttk.Frame(outer)
        footer.pack(side='bottom', fill='x', pady=(8, 0))
        link = tk.Label(footer, text='huggingface.co/gilliaaan',
                        background=COLORS['bg'], foreground=COLORS['accent'],
                        cursor='hand2', font=('Segoe UI', 9, 'underline'))
        link.pack(anchor='center')
        link.bind('<Button-1>', lambda _e: webbrowser.open(hf_url))
        Tooltip(link, f'Open {hf_url} in your browser')

        status = ttk.Frame(outer, style='Card.TFrame')
        status.pack(side='bottom', fill='x', pady=(10, 0))
        ttk.Label(status, textvariable=self.status_var, style='Status.TLabel').pack(side='left')
        ttk.Label(status,
                  text=f"Device: {'CUDA available' if torch.cuda.is_available() else 'CPU only'}",
                  style='Status.TLabel').pack(side='right')

        log_frame = ttk.LabelFrame(outer, text='  Log  ', padding=10)
        log_frame.pack(side='top', fill='both', expand=True)
        self.log_text = tk.Text(log_frame, wrap='word', state='disabled', height=20,
                                background=COLORS['log_bg'], foreground=COLORS['log_fg'],
                                insertbackground=COLORS['fg'], relief='flat', borderwidth=0,
                                font=('Consolas', 10), padx=10, pady=8)
        self.log_text.pack(side='left', fill='both', expand=True)
        scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        scroll.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=scroll.set)
        for tag, color in (('err', '#ff7a7a'), ('warn', '#ffb86b'),
                           ('ok', '#7ee0a0'), ('info', COLORS['fg_dim'])):
            self.log_text.tag_configure(tag, foreground=color)

    def _update_mixture_state(self):
        is_wav = self.quality.get().startswith('WAV')
        self.mix_chk.configure(state='normal' if is_wav else 'disabled')
        if not is_wav:
            self.make_mixture.set(False)

    def _pick_input(self):
        d = filedialog.askdirectory(title='Select input directory')
        if d:
            self.input_dir.set(d)
            if not self.output_dir.get():
                self.output_dir.set(str(Path(d).parent / (Path(d).name + '_organized')))

    def _pick_output(self):
        d = filedialog.askdirectory(title='Select output directory')
        if d:
            self.output_dir.set(d)

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.input_dir.get() or not os.path.isdir(self.input_dir.get()):
            messagebox.showerror('Missing input', 'Please select a valid input directory.')
            return
        if not self.output_dir.get():
            messagebox.showerror('Missing output', 'Please select an output directory.')
            return
        params = {
            'input_dir':    self.input_dir.get(),
            'output_dir':   self.output_dir.get(),
            'use_cuda':     self.use_cuda.get(),
            'model_id':     MODELS[self.model_label.get()],
            'stem_mode':    self.stem_mode.get(),
            'quality':      self.quality.get(),
            'threshold':    self.threshold.get(),
            'min_margin':   self.min_margin.get(),
            'batch_size':   self.batch_size.get(),
            'peak_norm':    self.peak_norm.get(),
            'make_mixture': self.make_mixture.get(),
            'dedup':        self.dedup.get(),
            'ambig_mode':   AMBIG_MODES[self.ambig_label.get()],
            'naming_mode':  NAMING_MODES[self.naming_label.get()],
        }
        self._append_log('=== Starting job ===')
        for k, v in params.items():
            self._append_log(f"  {k}: {v}")
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set('Running…')
        self.progress.pack(side='right')
        self.progress.start(12)
        self.worker = Worker(params, self.log_queue)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self._append_log('[stopping] ...')
            self.status_var.set('Stopping…')

    def _job_finished(self):
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress.stop()
        self.progress.pack_forget()
        self.status_var.set('Idle')
        self.worker = None

    def _drain_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg is DONE_SENTINEL:
                    self._job_finished()
                else:
                    self._append_log(msg)
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _append_log(self, msg: str):
        s = msg.strip()
        low = msg.lower()
        if '[error]' in low:
            tag = 'err'
        elif '[warn]' in low or 'oom' in low:
            tag = 'warn'
        elif s.startswith('Done') or '    wrote ' in msg or msg.lstrip().startswith('wrote '):
            tag = 'ok'
        elif s.startswith(('===', '[', 'Device:', 'Loading', 'Found', 'Model sources')):
            tag = 'info'
        else:
            tag = None
        self.log_text.configure(state='normal')
        self.log_text.insert('end', msg.rstrip() + '\n', tag or ())
        self.log_text.see('end')
        self.log_text.configure(state='disabled')


def main():
    App().mainloop()


if __name__ == '__main__':
    main()