"""Theme tokens — dark palette matches STEM organizer; light uses sample-library spreadsheet colors."""

from track_renamer.category_palette import DEFAULT_CATEGORY_COLORS

# STEM organizer LOG panel font (stem_organizer_ui.LOG_FONT)
PREVIEW_LOG_FONT_FAMILY = "Consolas"
PREVIEW_LOG_FONT_SIZE = 12

# STEM organizer COLORS (stem_organizer_ui.py) — navy/charcoal + purple accent
_STEM = {
    "bg": "#1e1f26",
    "panel": "#262833",
    "panel2": "#2e3140",
    "fg": "#e6e8ef",
    "fg_dim": "#9aa0b4",
    "accent": "#7c5cff",
    "accent_hov": "#9077ff",
    "danger": "#e25c5c",
    "log_bg": "#15161c",
    "log_fg": "#d6dae8",
    "border": "#3a3d4d",
    "status_trough": "#343647",
}

# Default dark — STEM organizer look
DARK = {
    "bg": _STEM["bg"],
    "panel": _STEM["panel"],
    "panel_2": _STEM["panel2"],
    "card": _STEM["panel2"],
    "input": _STEM["panel"],
    "control_bg": _STEM["panel2"],
    "border": _STEM["border"],
    "border_soft": _STEM["status_trough"],
    "text": _STEM["fg"],
    "text_dim": _STEM["fg_dim"],
    "text_mute": "#7a8199",
    "accent": _STEM["accent"],
    "accent_hover": _STEM["accent_hov"],
    "accent_soft": "#2a2540",
    "active_row": "#44485f",
    "loading_bg": "#2a2540",
    "changed": _STEM["accent_hov"],
    "unchanged": _STEM["fg_dim"],
    "list_bg": _STEM["log_bg"],
    "list_fg": _STEM["log_fg"],
    "waveform_bg": _STEM["log_bg"],
    "waveform_axis": "#343647",
    "waveform_playhead": "#ffffff",
    "audio": "#10b981",
    "midi": _STEM["accent"],
    "group": "#a855f7",
    "danger": _STEM["danger"],
    "badge_fg": "#ffffff",
    "btn": _STEM["panel2"],
    "btn_hover": _STEM["border"],
    "row_even": "",
    "row_odd": "",
    "category_colors": {},
}

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
    "accent": _STEM["accent"],
    "accent_hover": _STEM["accent_hov"],
    "accent_soft": "#ede9fe",
    "active_row": "#e3e4e8",
    "changed": _STEM["accent"],
    "unchanged": "#9ca3af",
    "list_bg": "#ffffff",
    "list_fg": "#1a1a1a",
    "waveform_bg": "#ffffff",
    "waveform_axis": "#d9dce3",
    "waveform_playhead": "#111827",
    "audio": "#b8d4a8",
    "midi": "#a4c2f4",
    "group": "#b4a6d9",
    "danger": _STEM["danger"],
    "badge_fg": "#1a1a1a",
    "btn": "#f0f2f5",
    "btn_hover": "#e4e7ec",
    "row_even": "#ffffff",
    "row_odd": "#f7f8fa",
    "category_colors": dict(DEFAULT_CATEGORY_COLORS),
}
