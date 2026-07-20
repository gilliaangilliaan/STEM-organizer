"""Compact category-colored audio preview strip."""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path

import customtkinter as ctk

from track_renamer.audio_preview import AudioPreviewService, WaveformPeaks
from track_renamer.category_palette import (
    default_category_color,
    parse_category_prefix_display,
)
from track_renamer.engine.models import PreviewRow, Track
from ui_theme import PATH_BTN_HEIGHT, ctk_action_button


def preview_waveform_color(
    row: PreviewRow | None,
    track: Track,
    theme: dict,
    category_colors: dict[str, str] | None = None,
) -> str:
    display = row.new_display if row is not None else track.display_name
    parsed = parse_category_prefix_display(
        display, known=category_colors or None,
    )
    if parsed:
        if parsed[0] == "FX":
            return "#e6e8ef"
        return (category_colors or {}).get(
            parsed[0],
            default_category_color(parsed[0]),
        )
    return theme["accent"]


class AudioPlayerBar(ctk.CTkFrame):
    def __init__(self, master, theme: dict, **kwargs) -> None:
        super().__init__(
            master, fg_color="transparent", height=PATH_BTN_HEIGHT, **kwargs,
        )
        self.theme = theme
        self.service = AudioPreviewService()
        self.active_track: Track | None = None
        self.active_row: PreviewRow | None = None
        self.peaks: WaveformPeaks = ()
        self.duration = 0.0
        self.waveform_color = theme["accent"]
        self.category_colors: dict[str, str] = {}
        self.status_text = "Select an audio file to preview"
        self._poll_job: str | None = None
        self._last_state = "stopped"
        self._build()
        self._schedule_poll()

    def _build(self) -> None:
        t = self.theme
        # Action-bar height/style; keep compact icon width (was 32).
        self.play_btn = ctk_action_button(
            self, "▶", self.toggle_playback, width=32,
        )
        self.play_btn.configure(state="disabled")
        self.play_btn.pack(side="left", padx=(0, 10))

        self.waveform = tk.Canvas(
            self,
            height=PATH_BTN_HEIGHT,
            highlightthickness=0,
            borderwidth=0,
            bg=t["waveform_bg"],
        )
        self.waveform.pack(side="left", fill="both", expand=True)
        self.waveform.bind("<Configure>", lambda _event: self._draw_waveform())

    def set_active(self, track: Track | None, row: PreviewRow | None) -> None:
        if track is None:
            self.reset()
            return

        self.waveform_color = preview_waveform_color(
            row,
            track,
            self.theme,
            self.category_colors,
        )
        same_file = (
            self.active_track is not None
            and self.active_track.file_path == track.file_path
        )
        self.active_track = track
        self.active_row = row
        self._draw_waveform()
        if same_file:
            state = self.service.playback_state()
            if state in ("playing", "paused"):
                self._apply_playback_state(state)
            elif self.peaks:
                self._set_status(track.display_name)
            return

        self.peaks = ()
        self.duration = 0.0
        self._last_state = "stopped"
        self.play_btn.configure(text="▶")
        path = track.file_path
        if not track.is_audio or path is None or not path.is_file():
            self.service.reset()
            self.play_btn.configure(state="disabled")
            self._set_status("Audio preview unavailable")
            self._draw_waveform()
            return
        if not self.service.available:
            self.service.reset()
            self.play_btn.configure(state="disabled")
            self._set_status(self.service.unavailable_message)
            self._draw_waveform()
            return

        self.play_btn.configure(state="normal")
        self._set_status(f"Loading {path.name}…")
        self.service.load(path)
        self._draw_waveform()

    def toggle_playback(self) -> None:
        state = self.service.play_pause()
        self._apply_playback_state(state)

    def seek(self, seconds: float) -> None:
        if self.active_track is None:
            return
        self.service.seek(seconds)
        self._draw_waveform()

    def _apply_playback_state(self, state: str) -> None:
        self._last_state = state
        self.play_btn.configure(text="⏸" if state == "playing" else "▶")
        if self.active_track is None:
            return
        name = self.active_track.display_name
        if state in ("playing", "paused") or self.peaks:
            self._set_status(name)

    def _schedule_poll(self) -> None:
        if self._poll_job is None and self.winfo_exists():
            self._poll_job = self.after(100, self._poll)

    def _poll(self) -> None:
        self._poll_job = None
        while True:
            try:
                generation, event_type, payload = self.service.events.get_nowait()
            except queue.Empty:
                break
            if generation != self.service.generation:
                continue
            if event_type == "waveform":
                self.peaks = payload  # type: ignore[assignment]
                if self.active_track is not None:
                    self._set_status(self.active_track.display_name)
                self._draw_waveform()
            elif event_type == "duration":
                self.duration = float(payload)
            else:
                message = str(payload).splitlines()[-1]
                self._set_status(message[:120])

        state = self.service.playback_state()
        if state != self._last_state:
            self._apply_playback_state(state)
            self._draw_waveform()
        if state in ("playing", "paused"):
            self._draw_waveform()
        self._schedule_poll()

    def _draw_waveform(self) -> None:
        canvas = self.waveform
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        center = height / 2
        canvas.create_line(
            0,
            center,
            width,
            center,
            fill=self.theme["waveform_axis"],
        )
        if self.peaks:
            peak_count = len(self.peaks)
            for x in range(width):
                peak_index = min(peak_count - 1, int(x * peak_count / width))
                low, high = self.peaks[peak_index]
                y1 = center - high * (center - 2)
                y2 = center - low * (center - 2)
                canvas.create_line(x, y1, x, y2, fill=self.waveform_color)
            if self.duration > 0:
                progress = min(1.0, self.service.playback_position() / self.duration)
                playhead_x = int(progress * max(width - 1, 1))
                canvas.create_line(
                    playhead_x,
                    1,
                    playhead_x,
                    height - 1,
                    fill=self.theme["waveform_playhead"],
                    width=1,
                )
        subdued_state = (
            self.active_track is None
            or self.status_text == "Audio preview unavailable"
        )
        filename_state = (
            self.active_track is not None
            and self.status_text == self.active_track.display_name
        )
        text_id = canvas.create_text(
            width - 4 if filename_state else width / 2,
            height if filename_state else height / 2,
            text=self.status_text,
            anchor="se" if filename_state else "center",
            fill="#4a4e62" if subdued_state else self.theme["text_dim"],
            font=("Segoe UI", 8 if filename_state else 9),
        )
        bounds = canvas.bbox(text_id)
        if bounds:
            background_id = canvas.create_rectangle(
                bounds[0] - 4,
                bounds[1] - 1,
                bounds[2] + 3,
                bounds[3] + 1,
                fill=self.theme["waveform_bg"],
                outline="",
            )
            canvas.tag_raise(text_id, background_id)

    def _set_status(self, text: str) -> None:
        self.status_text = text
        self._draw_waveform()

    def set_theme(self, theme: dict) -> None:
        self.theme = theme
        self.configure(fg_color="transparent")
        self.play_btn.configure(
            fg_color=theme["btn"],
            hover_color=theme["btn_hover"],
            text_color=theme["text"],
        )
        self.waveform.configure(bg=theme["waveform_bg"])
        if self.active_track is not None:
            self.waveform_color = preview_waveform_color(
                self.active_row,
                self.active_track,
                theme,
                self.category_colors,
            )
        self._draw_waveform()

    def set_category_colors(self, colors: dict[str, str]) -> None:
        self.category_colors = dict(colors)
        if self.active_track is not None:
            self.waveform_color = preview_waveform_color(
                self.active_row,
                self.active_track,
                self.theme,
                self.category_colors,
            )
        self._draw_waveform()

    def reset(self) -> None:
        self.service.reset()
        self.active_track = None
        self.active_row = None
        self.peaks = ()
        self.duration = 0.0
        self._last_state = "stopped"
        self.play_btn.configure(text="▶", state="disabled")
        self.status_text = "Select an audio file to preview"
        self._draw_waveform()

    def shutdown(self) -> None:
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None
        self.service.shutdown()
