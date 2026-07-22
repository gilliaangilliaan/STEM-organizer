"""Resolve a real Python for genre/gender + instrument tagger subprocesses.

Frozen builds ship tagger scripts beside the exe (no nested venv). ML wheels
live in ``site-packages\\`` from root ``install-deps.bat``. ``sys.executable``
is the .exe, so we spawn a matching system / ``py`` launcher interpreter with
``PYTHONPATH`` pointing at that folder.

Source / legacy: prefer ``genre_gender_tagger\\venv\\Scripts\\python.exe``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from deps_bootstrap import (
    APP_VERSION_FILE,
    SITE_PACKAGES_MARKER,
    SUPPORTED_PYTHON,
    app_dir,
    external_site_dirs,
    is_frozen,
)
from ffmpeg_bootstrap import subprocess_kwargs


def _parse_version_tag(text: str) -> tuple[int, int] | None:
    text = text.strip()
    parts = text.replace(",", ".").split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def tagger_app_root() -> Path:
    """Folder that holds ``genre_gender_tagger\\`` / ``instrument_tagger\\``."""
    return app_dir()


def genre_gender_dir() -> Path:
    """``genre_gender_tagger`` beside the exe, or under ``_internal`` when frozen."""
    root = tagger_app_root()
    candidates = [root / "genre_gender_tagger"]
    if is_frozen():
        candidates.append(root / "_internal" / "genre_gender_tagger")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "genre_gender_tagger")
    for path in candidates:
        if (path / "genre_gender_tagger.py").is_file():
            return path
    return candidates[0]


def genre_gender_script() -> Path:
    return genre_gender_dir() / "genre_gender_tagger.py"


def instrument_tagger_dir() -> Path:
    root = tagger_app_root()
    candidates = [root / "instrument_tagger"]
    if is_frozen():
        candidates.append(root / "_internal" / "instrument_tagger")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "instrument_tagger")
    for path in candidates:
        if (path / "instrument_tagger.py").is_file():
            return path
    return candidates[0]


def instrument_tagger_script() -> Path:
    return instrument_tagger_dir() / "instrument_tagger.py"


def _site_packages() -> Path | None:
    for path in external_site_dirs():
        if path.is_dir():
            return path
    return None


def _expected_python() -> tuple[int, int]:
    root = tagger_app_root()
    for marker in (
        root / "site-packages" / SITE_PACKAGES_MARKER,
        root / APP_VERSION_FILE,
    ):
        if not marker.is_file():
            continue
        parsed = _parse_version_tag(marker.read_text(encoding="utf-8"))
        if parsed in SUPPORTED_PYTHON:
            return parsed
    if is_frozen():
        return sys.version_info[:2]
    return sys.version_info[:2]


def _python_version(exe: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            [str(exe), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
            **subprocess_kwargs(),
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_version_tag(out)


def resolve_host_python() -> Path | None:
    """System / py-launcher Python matching site-packages / build version."""
    major, minor = _expected_python()
    candidates: list[Path] = []

    try:
        out = subprocess.check_output(
            ["py", f"-{major}.{minor}", "-c", "import sys; print(sys.executable)"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
            **subprocess_kwargs(),
        ).strip()
        if out:
            candidates.append(Path(out))
    except (OSError, subprocess.SubprocessError):
        pass

    for name in ("python", "python.exe"):
        try:
            out = subprocess.check_output(
                ["where" if sys.platform == "win32" else "which", name],
                text=True,
                timeout=10,
                stderr=subprocess.DEVNULL,
                **subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            line = line.strip()
            if line:
                candidates.append(Path(line))

    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        ver = _python_version(path)
        if ver == (major, minor):
            return path
    return None


def _venv_python_candidates(root: Path) -> tuple[Path, ...]:
    gg = root / "genre_gender_tagger"
    inst = root / "instrument_tagger"
    return (
        gg / "venv" / "Scripts" / "python.exe",
        gg / "venv" / "bin" / "python",
        inst / "venv" / "Scripts" / "python.exe",
        inst / "venv" / "bin" / "python",
    )


def resolve_tagger_python() -> Path | None:
    """Interpreter for tagger subprocesses (venv or host python)."""
    root = tagger_app_root()
    if is_frozen():
        # Prefer site-packages + real Python (no nested venv in dist).
        site = _site_packages()
        host = resolve_host_python()
        if host is not None and site is not None:
            return host
        for path in _venv_python_candidates(root):
            if path.is_file():
                return path
        return None
    for path in _venv_python_candidates(root):
        if path.is_file():
            return path
    return None


def tagger_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for tagger spawn; sets PYTHONPATH to site-packages when frozen.

    Host Python (not the .exe) must see hear21passt / onnxruntime wheels
    installed by root install-deps.bat into ``site-packages\\``.
    """
    env = dict(base if base is not None else os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if not is_frozen():
        return env
    site = _site_packages()
    if site is None:
        return env
    try:
        entry = str(site.resolve())
    except OSError:
        entry = str(site)
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    # Always put site-packages first so Auto-detect finds hear21passt.
    parts = [p for p in parts if os.path.normcase(p) != os.path.normcase(entry)]
    env["PYTHONPATH"] = entry + (os.pathsep + os.pathsep.join(parts) if parts else "")
    return env


def missing_tagger_python_hint() -> str:
    if is_frozen():
        return (
            "No Python found for Genre & Gender / Rename Auto-detect.\n"
            "Run install-deps.bat beside STEM-organizer.exe once "
            "(installs into site-packages\\).\n"
            "Need Python 3.10 or 3.11 on PATH (or: py -3.11)."
        )
    return (
        "Genre & Gender venv not found.\n"
        "Run genre_gender_tagger\\install-deps.bat once "
        "(or root install-deps.bat)."
    )
