"""Themed help window for Track Renamer."""

from __future__ import annotations

import tkinter as tk
import webbrowser

from ui_theme import (
    apply_toplevel_icon,
    apply_toplevel_rounded_corners,
    ctk_action_button,
)


DOCS_URL = "https://enableton-renamer-docs.vercel.app/"


def show_rename_help_dialog(parent: tk.Misc, theme: dict) -> None:
    t = theme
    dialog = tk.Toplevel(parent)
    dialog.title("Track Renamer help")
    dialog.configure(bg=t["panel"])
    dialog.resizable(False, False)
    dialog.transient(parent.winfo_toplevel())
    apply_toplevel_icon(dialog)

    outer = tk.Frame(dialog, bg=t["panel"])
    outer.pack(fill="both", expand=True, padx=22, pady=18)

    tk.Label(
        outer,
        text="Track Renamer",
        font=("Segoe UI Semibold", 18),
        fg=t["text"],
        bg=t["panel"],
    ).pack(anchor="w")
    tk.Label(
        outer,
        text="Build a rule stack, verify every filename, then rename safely.",
        font=("Segoe UI", 10),
        fg=t["text_dim"],
        bg=t["panel"],
    ).pack(anchor="w", pady=(2, 14))

    def section(title: str, body: str) -> None:
        card = tk.Frame(
            outer,
            bg=t["card"],
            highlightbackground=t["border"],
            highlightthickness=1,
        )
        card.pack(fill="x", pady=(0, 10))
        tk.Label(
            card,
            text=title.upper(),
            font=("Segoe UI Semibold", 9),
            fg=t["accent_hover"],
            bg=t["card"],
        ).pack(anchor="w", padx=14, pady=(10, 4))
        tk.Label(
            card,
            text=body,
            font=("Segoe UI", 10),
            fg=t["text_dim"],
            bg=t["card"],
            justify="left",
            anchor="w",
            wraplength=600,
        ).pack(fill="x", padx=14, pady=(0, 11))

    section(
        "Workflow",
        "1. Open a folder and choose whether subfolders are included.\n"
        "2. Add, reorder, enable, or edit rules on the left.\n"
        "3. Click Apply only when rules have changed, then inspect the preview.\n"
        "4. Select the files you want and click Rename.",
    )
    section(
        "Preview & audio",
        "Original names are struck through; the resulting category badge and filename "
        "appear after the arrow. Click a row to load its waveform. Space toggles "
        "playback, Left/Right seeks 3 seconds, and Up/Down changes the active file.",
    )
    section(
        "Safe file operations",
        "Extensions are preserved. Naming conflicts receive _1, _2, and so on. "
        "After renaming, you can optionally choose a destination where BASS, DRUMS, "
        "VOCALS, and other prefix folders are created automatically.",
    )

    shortcuts = tk.Frame(
        outer,
        bg=t["card"],
        highlightbackground=t["border"],
        highlightthickness=1,
    )
    shortcuts.pack(fill="x", pady=(0, 12))
    tk.Label(
        shortcuts,
        text="KEYBOARD",
        font=("Segoe UI Semibold", 9),
        fg=t["accent_hover"],
        bg=t["card"],
    ).grid(row=0, column=0, columnspan=4, sticky="w", padx=14, pady=(10, 6))
    pairs = (
        ("↑ / ↓", "Select file"),
        ("Page Up / Down", "Scroll preview"),
        ("Space", "Play / pause"),
        ("← / →", "Seek 3 seconds"),
    )
    for index, (key, action) in enumerate(pairs):
        column = (index % 2) * 2
        row = 1 + index // 2
        tk.Label(
            shortcuts,
            text=key,
            font=("Consolas", 9, "bold"),
            fg=t["text"],
            bg=t["border_soft"],
            padx=7,
            pady=3,
        ).grid(row=row, column=column, sticky="w", padx=(14, 8), pady=(0, 8))
        tk.Label(
            shortcuts,
            text=action,
            font=("Segoe UI", 9),
            fg=t["text_dim"],
            bg=t["card"],
        ).grid(row=row, column=column + 1, sticky="w", padx=(0, 24), pady=(0, 8))

    footer = tk.Frame(outer, bg=t["panel"])
    footer.pack(fill="x")
    docs = tk.Label(
        footer,
        text="Open full rule reference",
        font=("Segoe UI", 10),
        fg=t["accent"],
        bg=t["panel"],
        cursor="hand2",
    )
    docs.pack(side="left")
    docs.bind("<Button-1>", lambda _event: webbrowser.open(DOCS_URL))
    docs.bind("<Enter>", lambda _event: docs.configure(fg=t["accent_hover"]))
    docs.bind("<Leave>", lambda _event: docs.configure(fg=t["accent"]))

    close = ctk_action_button(footer, "Close", dialog.destroy, width=72)
    close.pack(side="right")

    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.update_idletasks()
    width = 680
    height = max(560, outer.winfo_reqheight() + 36)
    top = parent.winfo_toplevel()
    x = top.winfo_rootx() + max(0, (top.winfo_width() - width) // 2)
    y = top.winfo_rooty() + max(0, (top.winfo_height() - height) // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")
    dialog.update_idletasks()
    apply_toplevel_rounded_corners(dialog)
    dialog.after(20, lambda: apply_toplevel_rounded_corners(dialog))
    dialog.grab_set()
    close.focus_set()
