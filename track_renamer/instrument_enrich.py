"""Fill Track.instrument from PaSST OpenMIC worker (subprocess)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from track_renamer.engine.defaults import map_instrument_to_category
from track_renamer.engine.models import OpRule, Rule, Track


def _app_root() -> Path:
    """Folder that holds instrument_tagger\\ + genre_gender_tagger\\ (exe dir when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


TAGGER_DIR = _app_root() / "instrument_tagger"
TAGGER_SCRIPT = TAGGER_DIR / "instrument_tagger.py"

# Bump when model/label set / primary-pick policy changes so stale cache dies.
_CACHE_MODEL = "passt-openmic-nosynth-g35"

# path → (mtime_ns, label, score, second_score, model_id)
_CACHE: dict[str, tuple] = {}

ResultCallback = Callable[[dict[str, Any]], None]
ProgressCallback = Callable[[int, int], None]


def resolve_tagger_python() -> Path | None:
    """Prefer shared genre_gender_tagger venv (one torch for both taggers)."""
    root = _app_root()
    candidates = (
        root / "genre_gender_tagger" / "venv" / "Scripts" / "python.exe",
        root / "genre_gender_tagger" / "venv" / "bin" / "python",
        # Legacy dedicated instrument venv (pre-slim).
        TAGGER_DIR / "venv" / "Scripts" / "python.exe",
        TAGGER_DIR / "venv" / "bin" / "python",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def rules_need_instrument_ml(rules: list[Rule]) -> bool:
    for rule in rules:
        if isinstance(rule, OpRule) and rule.op == "categoryBundle":
            source = str(rule.params.get("source", "filename")).lower()
            if source in ("model", "combo"):
                return True
    return False


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def classify_decision(
    label: str,
    score: float = 0.0,
    *,
    second_score: float = 0.0,
) -> tuple[str, str]:
    """
    Return (action, category_name).

    action: 'apply' | 'skip_unmap'
    Always apply when label maps to a Category Macro row.
    score / second_score ignored (kept for call-site compatibility).
    """
    _ = (score, second_score)
    category = map_instrument_to_category(label)
    if not category:
        return "skip_unmap", category
    return "apply", category


def _unpack_cache(cached: tuple) -> tuple[int, str, float, float] | None:
    """Return (mtime, label, score, second) if entry matches current model."""
    if len(cached) < 5:
        return None  # legacy cache — force re-infer
    mtime, label, score, second, model_id = cached[:5]
    if model_id != _CACHE_MODEL:
        return None
    return int(mtime), str(label), float(score), float(second)


def apply_cached_labels(tracks: list[Track]) -> int:
    """Apply cache hits onto tracks. Returns number filled from cache."""
    filled = 0
    for track in tracks:
        path = track.file_path
        if path is None or not path.is_file():
            continue
        key = str(path.resolve())
        cached = _CACHE.get(key)
        if not cached:
            continue
        unpacked = _unpack_cache(cached)
        if not unpacked:
            continue
        mtime, label, score, second = unpacked
        if mtime != _mtime_ns(path):
            continue
        track.instrument = label
        track.instrument_score = score
        track.instrument_second = float(second)
        track.category = map_instrument_to_category(label)
        filled += 1
    return filled


def _paths_needing_infer(tracks: list[Track]) -> list[Path]:
    needed: list[Path] = []
    for track in tracks:
        path = track.file_path
        if path is None or not path.is_file():
            continue
        key = str(path.resolve())
        cached = _CACHE.get(key)
        unpacked = _unpack_cache(cached) if cached else None
        if unpacked and unpacked[0] == _mtime_ns(path):
            continue
        needed.append(path)
    return needed


def _second_from_row(row: dict) -> float:
    """Runner-up share. Worker score is calibrated p1/(p1+p2) → second = 1-score."""
    try:
        score = float(row.get("score") or 0.0)
        if 0.0 < score <= 1.0:
            return max(0.0, 1.0 - score)
        top = row.get("top") or []
        if isinstance(top, list) and len(top) >= 2:
            return float(top[1][1])
    except (TypeError, ValueError, IndexError):
        pass
    return 0.0


def _emit_result(
    on_result: ResultCallback | None,
    *,
    path: Path,
    label: str,
    score: float,
    second_score: float,
    error: str = "",
) -> None:
    if on_result is None:
        return
    category = map_instrument_to_category(label) if not error else ""
    on_result(
        {
            "path": path,
            "name": path.name,
            "label": label,
            "score": score,
            "second_score": second_score,
            "category": category,
            "error": error,
        }
    )


def enrich_tracks(
    tracks: list[Track],
    *,
    status: Callable[[str], None] | None = None,
    on_progress: ProgressCallback | None = None,
    on_result: ResultCallback | None = None,
) -> tuple[int, str | None]:
    """
    Classify tracks missing cache entries via instrument_tagger.

    on_result receives one dict per file (cache hit or fresh infer).
    Returns (classified_count, error_message_or_None).
    """
    apply_cached_labels(tracks)
    pending = _paths_needing_infer(tracks)
    pending_keys = {str(p.resolve()) for p in pending}

    # Emit cache hits first so the analyze log fills immediately.
    cached_n = 0
    for track in tracks:
        path = track.file_path
        if path is None or not path.is_file():
            continue
        key = str(path.resolve())
        if key in pending_keys:
            continue
        cached = _CACHE.get(key)
        if not cached:
            continue
        unpacked = _unpack_cache(cached)
        if not unpacked:
            continue
        _mtime, label, score, second = unpacked
        cached_n += 1
        _emit_result(
            on_result,
            path=path,
            label=label,
            score=score,
            second_score=second,
        )
        if on_progress:
            on_progress(cached_n, cached_n + len(pending))

    status = status or (lambda _msg: None)
    if cached_n and pending:
        status(f"Cache hit {cached_n:,} — inferring {len(pending):,}…")
    elif cached_n and not pending:
        status(f"All {cached_n:,} from cache.")
        return cached_n, None
    elif not pending:
        return cached_n, None

    py = resolve_tagger_python()
    if py is None or not TAGGER_SCRIPT.is_file():
        return cached_n, (
            "Instrument tagger not installed.\n"
            "Run dist\\install-deps.bat and answer Yes to Rename Auto-detect\n"
            "(shared venv: genre_gender_tagger\\venv — not under instrument_tagger\\)."
        )

    total = cached_n + len(pending)
    status(f"Starting tagger for {len(pending):,} file(s)…")

    list_path: Path | None = None
    classified = 0
    done = cached_n
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as handle:
            list_path = Path(handle.name)
            for path in pending:
                handle.write(f"{path.resolve()}\n")

        cmd = [
            str(py),
            "-u",
            str(TAGGER_SCRIPT),
            "--files-from",
            str(list_path),
            "--top",
            "2",
        ]
        # Merge stderr→stdout so status/TF logs cannot fill the stderr pipe
        # and deadlock the worker (classic hang ~dozens of files in).
        from ffmpeg_bootstrap import subprocess_kwargs

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(TAGGER_DIR),
            **subprocess_kwargs(),  # hide console window on Windows
        )
        assert proc.stdout is not None

        log_tail: list[str] = []
        for line in proc.stdout:
            raw = line.rstrip("\n")
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                log_tail.append(stripped)
                if len(log_tail) > 40:
                    log_tail = log_tail[-40:]
                # Forward warmup lines; skip per-file "[n/total] name" spam
                # and hear21passt / torch noise (model dump, shapes, warnings).
                if (
                    stripped.startswith("[")
                    and "/" in stripped[:12]
                    and "]" in stripped[:16]
                ):
                    continue
                low = stripped.lower()
                if (
                    "torch.size" in low
                    or "userwarning" in low
                    or "warnings.warn" in low
                    or "input image size" in low
                    or stripped.startswith(
                        (
                            "X flattened",
                            "forward_features",
                            "head ",
                            " self.",
                            "patch_embed",
                            "Loading PASST",
                            "Loading PaSST",
                            "(1): Linear",
                            "(head_dist):",
                            "Sequential(",
                            "  (",
                        )
                    )
                ):
                    continue
                status(stripped.lstrip())
                continue

            done += 1
            path = Path(str(row.get("path") or ""))
            if "error" in row or not path.name:
                _emit_result(
                    on_result,
                    path=path if path.name else Path("unknown"),
                    label="",
                    score=0.0,
                    second_score=0.0,
                    error=str(row.get("error") or "error"),
                )
                if on_progress:
                    on_progress(done, total)
                continue

            _store_result(path, row)
            classified += 1
            label = str(row.get("label") or "")
            try:
                score = float(row.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            second = _second_from_row(row)
            _emit_result(
                on_result,
                path=path,
                label=label,
                score=score,
                second_score=second,
            )
            if on_progress:
                on_progress(done, total)

        returncode = proc.wait()
        if returncode != 0 and classified == 0:
            # Prefer real traceback / ERROR lines over model-dump noise.
            useful = [
                ln
                for ln in log_tail
                if any(
                    k in ln
                    for k in (
                        "Error",
                        "ERROR",
                        "Traceback",
                        "Exception",
                        "UnicodeEncode",
                        "not installed",
                    )
                )
            ]
            err = "\n".join(useful or log_tail[-8:]).strip() or "tagger failed"
            return cached_n, err[:500]
    except OSError as exc:
        return cached_n, str(exc)
    finally:
        if list_path is not None:
            try:
                list_path.unlink()
            except OSError:
                pass

    apply_cached_labels(tracks)
    return cached_n + classified, None


def _store_result(path: Path, row: dict) -> None:
    label = str(row.get("label") or "")
    try:
        score = float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    second = _second_from_row(row)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    key = str(resolved)
    _CACHE[key] = (
        _mtime_ns(resolved),
        label,
        score,
        second,
        _CACHE_MODEL,
    )


def clear_instrument_cache() -> None:
    _CACHE.clear()


def tagger_available() -> bool:
    return resolve_tagger_python() is not None and TAGGER_SCRIPT.is_file()
