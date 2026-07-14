"""Hover tooltips for CustomTkinter widgets."""

from __future__ import annotations

import tkinter as tk


class ToolTip:
    def __init__(
        self,
        widget,
        text: str,
        *,
        delay: int = 500,
        bg: str = "#1e1f26",
        border: str = "#3a3d4d",
        text_color: str = "#e6e8ef",
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.bg = bg
        self.border = border
        self.text_color = text_color
        self._tip_window: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._hide()
        self._after_id = self.widget.after(self.delay, self._show)

    def _show(self) -> None:
        self._after_id = None
        if self._tip_window or not self.text:
            return
        if not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        frame = tk.Frame(tw, bg=self.border, padx=1, pady=1)
        frame.pack()
        label = tk.Label(
            frame,
            text=self.text,
            justify="left",
            background=self.bg,
            foreground=self.text_color,
            relief="flat",
            font=("Segoe UI", 9),
            wraplength=300,
            padx=10,
            pady=6,
        )
        label.pack()
        self._tip_window = tw

    def _hide(self, _event=None) -> None:
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._tip_window:
            try:
                if self._tip_window.winfo_exists():
                    self._tip_window.destroy()
            except tk.TclError:
                pass
            self._tip_window = None


def bind_tooltip(widget, text: str, theme: dict | None = None, delay: int = 500) -> ToolTip:
    t = theme or {}
    return ToolTip(
        widget,
        text,
        delay=delay,
        bg=t.get("bg", "#1e1f26"),
        border=t.get("border", "#3a3d4d"),
        text_color=t.get("text", "#e6e8ef"),
    )
