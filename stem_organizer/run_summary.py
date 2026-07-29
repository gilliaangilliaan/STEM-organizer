"""Unified LOG run-summary footer (Genre / Gender / Vocal / Compression / …)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

LogFn = Callable[[str, str], None]


def fmt_elapsed(seconds: float) -> str:
    """m:ss or h:mm:ss — same reading as Classify / Genre summaries."""
    total = max(0, int(round(float(seconds or 0))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def emit_run_summary(
    log: LogFn,
    feature: str,
    *,
    elapsed: Optional[float] = 0.0,
    files: Optional[int] = None,
    stat_lines: Optional[Sequence[tuple[str, str]]] = None,
    extra_lines: Optional[Sequence[str]] = None,
) -> None:
    """Blank line · === Feature Summary === · stats · blank line · DONE."""
    log("", "info")
    log(f"=== {feature} Summary ===", "info")
    if elapsed is not None:
        log(f"  Total time: {fmt_elapsed(elapsed)}", "info")
    if files is not None:
        n = max(0, int(files))
        log(f"  Files: {n:,}", "info")
        if n > 0:
            minutes = max(float(elapsed) / 60.0, 1e-9)
            log(f"  Sec/file: {float(elapsed) / n:.3f}", "info")
            log(f"  Files/min: {n / minutes:.2f}", "info")
    for text, tag in stat_lines or ():
        line = text if text.startswith("  ") else f"  {text}"
        log(line, tag)
    for line in extra_lines or ():
        stripped = str(line).strip()
        if stripped:
            log(f"  {stripped}", "info")
    log("", "info")
    log("DONE", "ok")
