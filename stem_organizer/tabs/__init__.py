"""Tab registry. Each tab module exposes a ``register(window, settings)`` function.

Stage 1 (skeleton): no tabs registered. As each tab lands it is added here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..app import MainWindow
    from ..settings_store import SettingsStore


def register_all_tabs(window: "MainWindow", settings: "SettingsStore") -> None:
    """Register every tab that is currently implemented."""
    try:
        from .classify_tab import register as register_classify

        register_classify(window, settings)
    except Exception as exc:  # tab not ported yet
        print(f"[stem_organizer] classify tab unavailable: {exc}")

    try:
        from .pair_finder_tab import register as register_pair

        register_pair(window, settings)
    except Exception as exc:
        print(f"[stem_organizer] pair-finder tab unavailable: {exc}")

    try:
        from .genre_gender_tab import register as register_gg

        register_gg(window, settings)
    except Exception as exc:
        print(f"[stem_organizer] genre/gender tab unavailable: {exc}")

    try:
        from .rename_tab import register as register_rename

        register_rename(window, settings)
    except Exception as exc:
        print(f"[stem_organizer] rename tab unavailable: {exc}")
