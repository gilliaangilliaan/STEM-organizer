#!/usr/bin/env python3
"""
OpenMIC-2018 instrument classifier (PaSST).

20-class multi-label model (hear21passt / kkoutini PaSST openmic checkpoint).

Phase-1 CLI: classify files / folder → JSON lines on stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

# Avoid Windows cp1252 crashes on non-ASCII paths / names in print().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ----------------------------------------------------------
# Constants (PaSST OpenMIC — 32 kHz, ~10 s clips)
# ----------------------------------------------------------

SAMPLE_RATE = 32000
MAX_AUDIO_SECONDS = 10.0
# PaSST OpenMIC expects 998 mel frames @ hop=320.
# Preemphasis conv1d (k=2, no pad) drops 1 sample → add +1 so STFT still yields 998.
CLIP_SAMPLES = 997 * 320 + 1

# If synthesizer is #1, demote to runner-up when raw sigmoid gap ≤ this.
# Larger = fewer Synth prefixes (more aggressive demote).
SYNTH_DEMOTE_MAX_GAP = 0.35

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"

# Alphabetical OpenMIC-2018 instrument vocabulary (official order).
OPENMIC_INSTRUMENTS = (
    "accordion",
    "banjo",
    "bass",
    "cello",
    "clarinet",
    "cymbals",
    "drums",
    "flute",
    "guitar",
    "mallet_percussion",
    "mandolin",
    "organ",
    "piano",
    "saxophone",
    "synthesizer",
    "trombone",
    "trumpet",
    "ukulele",
    "violin",
    "voice",
)

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".aif",
    ".aiff",
}


def _status(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _print_json(obj: dict) -> None:
    """Emit one JSON line on stdout; survive Windows console encoding."""
    try:
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    except UnicodeEncodeError:
        print(json.dumps(obj, ensure_ascii=True), flush=True)


def load_mono_32k(filename: str | Path) -> np.ndarray:
    """Mono float32 @ 32 kHz, first MAX_AUDIO_SECONDS."""
    import librosa

    max_src = None
    try:
        info = sf.info(str(filename))
        if info.samplerate > 0:
            max_src = int(MAX_AUDIO_SECONDS * info.samplerate) + info.samplerate
    except Exception:
        max_src = None

    data, sr = sf.read(
        str(filename),
        always_2d=True,
        dtype="float32",
        frames=max_src if max_src else -1,
    )
    audio = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
            res_type="soxr_hq",
        )
    if audio.shape[0] > CLIP_SAMPLES:
        audio = audio[:CLIP_SAMPLES]
    # Peak normalize lightly — PaSST expects roughly [-1, 1].
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


class _PasstOpenmicBackend:
    """PaSST OpenMIC-2018 via hear21passt (logits → sigmoid)."""

    name = "passt-openmic"

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def predict(self, audio: np.ndarray) -> np.ndarray:
        import torch

        if audio.size == 0:
            return np.zeros(len(OPENMIC_INSTRUMENTS), dtype=np.float32)
        # PaSST OpenMIC was trained on 10 s → ~998 mel frames. Always pad/trim
        # to that length so positional encodings match the checkpoint.
        if audio.shape[0] < CLIP_SAMPLES:
            audio = np.pad(audio, (0, CLIP_SAMPLES - audio.shape[0]))
        elif audio.shape[0] > CLIP_SAMPLES:
            audio = audio[:CLIP_SAMPLES]
        tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.get_scene_embeddings(tensor)
            # mode=logits → (1, 20)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()[0]
        return probs.astype(np.float32, copy=False)


def load_backend(status=_status) -> _PasstOpenmicBackend:
    """Load PaSST OpenMIC weights (auto-download on first use)."""
    import torch

    try:
        from hear21passt.models.passt import get_model as get_model_passt
        from hear21passt.wrapper import PasstBasicWrapper
        # hear21passt prints tensor shapes on first forward — silence that spam.
        import hear21passt.models.passt as _passt_mod

        _passt_mod.first_RUN = False
    except ImportError as exc:
        raise SystemExit(
            "\nERROR: hear21passt not installed.\n"
            "  Run instrument_tagger\\install-deps.bat\n"
            f"  detail: {exc}\n"
        ) from exc

    # Local mel (no torchaudio) — GG venv often has broken torchaudio wheels.
    from passt_mel import PasstMelSTFT

    status("  loading PaSST OpenMIC-2018 (first run downloads ~330 MB)...")
    mel = PasstMelSTFT(
        n_mels=128,
        sr=SAMPLE_RATE,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        fmin=0.0,
        fmax=None,
    )
    # hear21passt openmic2008.py used arch="openmic2008" (broken);
    # correct arch key is "openmic". Silence get_model's print(model) spam.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = get_model_passt(arch="openmic", n_classes=20)
    model = PasstBasicWrapper(
        mel=mel,
        net=net,
        mode="logits",
        scene_embedding_size=20,
        timestamp_embedding_size=20,
        max_model_window=10000,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    status(f"  device: {device}")
    return _PasstOpenmicBackend(model, device)


def probs_to_result(
    probs: np.ndarray,
    *,
    top_k: int = 5,
    threshold: float = 0.0,
) -> dict:
    """
    Sigmoid multi-label scores → primary label + top-k list.

    Bias: OpenMIC ``synthesizer`` is a loose electronic bucket. If it is
    argmax *and* the runner-up raw score is within SYNTH_DEMOTE_MAX_GAP,
    use the runner-up instead. Clear synth wins stay synthesizer.
    ``top`` still lists raw scores (synth may appear first there).
    """
    order = np.argsort(-probs)
    best_i = int(order[0])
    demoted_synth = False
    if OPENMIC_INSTRUMENTS[best_i] == "synthesizer" and order.size > 1:
        runner_i = int(order[1])
        gap = float(probs[best_i]) - float(probs[runner_i])
        if gap <= SYNTH_DEMOTE_MAX_GAP:
            best_i = runner_i
            demoted_synth = True

    top = []
    for i in order[: max(1, top_k)]:
        score = float(probs[i])
        if score < threshold and top:
            break
        top.append([OPENMIC_INSTRUMENTS[int(i)], score])

    above = [
        [OPENMIC_INSTRUMENTS[int(i)], float(probs[i])]
        for i in order
        if float(probs[i]) >= threshold
    ]

    # Confidence: chosen label vs strongest other (often synthesizer after demote).
    p1 = float(probs[best_i])
    others = [int(i) for i in order if int(i) != best_i]
    p2 = float(probs[others[0]]) if others else 0.0
    denom = p1 + p2
    calibrated = (p1 / denom) if denom > 0 else p1

    return {
        "label": OPENMIC_INSTRUMENTS[best_i],
        "score": float(calibrated),
        "score_raw": p1,
        "top": top,
        "above": above,
        "n_patches": 1,
        "model": "passt-openmic",
        "demoted_synth": demoted_synth,
    }


def classify_file(
    filename: str | Path,
    backend: _PasstOpenmicBackend,
    *,
    top_k: int = 5,
    threshold: float = 0.0,
) -> dict:
    audio = load_mono_32k(filename)
    probs = backend.predict(audio)
    result = probs_to_result(probs, top_k=top_k, threshold=threshold)
    result["path"] = str(Path(filename).resolve())
    return result


def iter_audio_files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify audio with PaSST OpenMIC-2018 instruments.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="Single audio file")
    src.add_argument("--folder", type=Path, help="Folder (recursive)")
    src.add_argument(
        "--files-from",
        type=Path,
        help="Text file with one audio path per line (use - for stdin)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Top-k labels in output (default 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Min raw sigmoid for above[] list (default 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files from folder (0 = all)",
    )
    args = parser.parse_args(argv)

    files: list[Path] = []
    if args.file is not None:
        if not args.file.is_file():
            _status(f"ERROR: not a file: {args.file}")
            return 1
        files = [args.file]
    elif args.files_from is not None:
        if str(args.files_from) == "-":
            raw_lines = sys.stdin.read().splitlines()
        else:
            if not args.files_from.is_file():
                _status(f"ERROR: not a file: {args.files_from}")
                return 1
            raw_lines = args.files_from.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        for line in raw_lines:
            line = line.lstrip("\ufeff").strip().strip('"')
            if not line:
                continue
            path = Path(line)
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(path)
            if args.limit and len(files) >= args.limit:
                break
        if not files:
            _status("ERROR: no valid audio paths in --files-from list")
            return 1
    else:
        if not args.folder.is_dir():
            _status(f"ERROR: not a folder: {args.folder}")
            return 1
        for i, path in enumerate(iter_audio_files(args.folder), 1):
            files.append(path)
            if args.limit and i >= args.limit:
                break
        if not files:
            _status(f"ERROR: no audio files under {args.folder}")
            return 1

    _status(f"Instrument tagger (PaSST OpenMIC) — {len(files)} file(s)")
    backend = load_backend(status=_status)
    _status(f"  backend: {backend.name}")

    errors = 0
    for i, path in enumerate(files, 1):
        _status(f"[{i}/{len(files)}] {path.name}")
        try:
            result = classify_file(
                path,
                backend,
                top_k=args.top,
                threshold=args.threshold,
            )
            _print_json(result)
        except Exception as exc:
            errors += 1
            _print_json(
                {
                    "path": str(path.resolve()),
                    "error": str(exc),
                }
            )

    _status(f"done. ok={len(files) - errors} err={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
