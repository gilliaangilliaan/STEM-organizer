"""Theme tokens — dark from shared ui_theme; light keeps sample-library palette."""

from track_renamer.category_palette import DEFAULT_CATEGORY_COLORS
from ui_theme import (
    DARK,
    PREVIEW_LOG_FONT_FAMILY,
    PREVIEW_LOG_FONT_SIZE,
    PREVIEW_LOG_PCT_FONT_SIZE,
)

# Re-export for existing imports
__all__ = (
    "DARK",
    "LIGHT",
    "PREVIEW_LOG_FONT_FAMILY",
    "PREVIEW_LOG_FONT_SIZE",
    "PREVIEW_LOG_PCT_FONT_SIZE",
)

_STEM_ACCENT = DARK["accent"]
_STEM_ACCENT_HOV = DARK["accent_hover"]
_STEM_DANGER = DARK["danger"]

# Light — sample-library spreadsheet palette (sampled from reference)
LIGHT = {
    "bg": "#ffffff",
    "panel": "#f7f8fa",
    "panel_2": "#ffffff",
    "card": "#ffffff",
    "input": "#ffffff",
    "control_bg": "#f7f8fa",
    "border": "#e2e5ea",
    "border_soft": "#eceef2",
    "text": "#1a1a1a",
    "text_dim": "#4a4a4a",
    "text_mute": "#6b7280",
    "accent": _STEM_ACCENT,
    "accent_hover": _STEM_ACCENT_HOV,
    "accent_soft": "#ede9fe",
    "active_row": "#e3e4e8",
    "changed": _STEM_ACCENT,
    "unchanged": "#9ca3af",
    "list_bg": "#ffffff",
    "list_fg": "#1a1a1a",
    "waveform_bg": "#ffffff",
    "waveform_axis": "#d9dce3",
    "waveform_playhead": "#111827",
    "audio": "#b8d4a8",
    "midi": "#a4c2f4",
    "group": "#b4a6d9",
    "danger": _STEM_DANGER,
    "badge_fg": "#1a1a1a",
    "btn": "#f0f2f5",
    "btn_hover": "#e4e7ec",
    "row_even": "#ffffff",
    "row_odd": "#f7f8fa",
    "category_colors": dict(DEFAULT_CATEGORY_COLORS),
}
