"""Preview panel — virtualized file list for large folders."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import customtkinter as ctk

from track_renamer.category_palette import (
    CATEGORY_BADGE_TEXT,
    applied_category_colors,
    category_badge_label,
    default_category_color,
    parse_category_prefix_display,
)
from track_renamer.engine.models import PreviewRow, Track
from track_renamer.engine.processor import PreparedRulePlan, compute_preview_row, prepare_rules
from track_renamer.gui.theme import PREVIEW_LOG_FONT_FAMILY, PREVIEW_LOG_FONT_SIZE
from track_renamer.gui.tips import TIPS
from track_renamer.gui.tooltip import bind_tooltip

ROW_HEIGHT = 28
RENDER_BUFFER = 10
PREVIEW_BATCH_SIZE = 200
LAZY_BUFFER_ROWS = 60
RESULT_BATCH_SIZE = 64
RESULT_POLL_MS = 40


@dataclass(slots=True)
class _PreviewJob:
    generation: int
    tracks: list[Track]
    rules: PreparedRulePlan
    cancel: threading.Event = field(default_factory=threading.Event)
    priority: queue.SimpleQueue[int] = field(default_factory=queue.SimpleQueue)
    results: queue.SimpleQueue[tuple[list[tuple[int, PreviewRow]], bool]] = field(
        default_factory=queue.SimpleQueue
    )
    requested: set[int] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _FolderHeader:
    label: str


class _PreviewRowWidget:
    """Reusable viewport row; reconfigured instead of destroyed while scrolling."""

    def __init__(
        self,
        canvas: tk.Canvas,
        theme: dict,
        fonts: dict[str, ctk.CTkFont],
        on_toggle: Callable[[int, PreviewRow | None, Track, bool], None],
        on_activate: Callable[[int, PreviewRow | None, Track], None],
    ) -> None:
        self.theme = theme
        self.fonts = fonts
        self.on_toggle = on_toggle
        self.on_activate = on_activate
        self.index = -1
        self.display_index = -1
        self.row: PreviewRow | None = None
        self.track: Track | None = None
        self._render_selected = True
        self._render_active = False
        self._render_header: str | None = None
        self._render_grouped = False
        self.category_colors: dict[str, str] = {}

        self.frame = ctk.CTkFrame(
            canvas,
            fg_color="transparent",
            corner_radius=0,
            height=ROW_HEIGHT,
        )
        self.frame.pack_propagate(False)
        self.var = ctk.BooleanVar(value=True)
        self.checkbox = ctk.CTkCheckBox(
            self.frame,
            text="",
            width=18,
            checkbox_width=16,
            checkbox_height=16,
            variable=self.var,
            command=self._toggle,
            fg_color=theme["accent"],
            hover_color=theme["accent_hover"],
            border_color=theme["border"],
        )
        self.checkbox.pack(side="left", padx=(6, 4))

        self.old_group = self._make_name_group()
        self.old_group["frame"].pack(side="left", anchor="w")
        self.arrow = ctk.CTkLabel(
            self.frame, text="→", text_color=theme["changed"], font=fonts["normal"]
        )
        self.arrow.pack(side="left", padx=4)
        self.new_group = self._make_name_group()
        self.new_group["frame"].pack(side="left", anchor="w")
        self.status = ctk.CTkLabel(
            self.frame,
            text="",
            font=fonts["italic"],
            text_color=theme["unchanged"],
        )
        self.status.pack(side="right", padx=8)
        self.folder_label = ctk.CTkLabel(
            self.frame,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=theme["text_mute"],
        )
        self.window_id = canvas.create_window(
            0, 0, window=self.frame, anchor="nw", width=1, state="hidden"
        )
        self._bind_activation(self.frame)
        self._bind_activation(self.old_group["frame"])
        self._bind_activation(self.old_group["badge"])
        self._bind_activation(self.old_group["separator"])
        self._bind_activation(self.old_group["text"])
        self._bind_activation(self.arrow)
        self._bind_activation(self.new_group["frame"])
        self._bind_activation(self.new_group["badge"])
        self._bind_activation(self.new_group["separator"])
        self._bind_activation(self.new_group["text"])
        self._bind_activation(self.status)

    def bind_header(self, label: str, theme: dict) -> None:
        self.index = -1
        self.track = None
        self.row = None
        self._render_header = label
        self.checkbox.pack_forget()
        self.old_group["frame"].pack_forget()
        self.arrow.pack_forget()
        self.new_group["frame"].pack_forget()
        self.status.pack_forget()
        self.folder_label.configure(
            text=f"  {label}",
            text_color=theme["text_mute"],
        )
        self.folder_label.pack(fill="both", expand=True)
        self.frame.configure(fg_color=theme["panel_2"])

    def _bind_activation(self, widget) -> None:
        targets = [widget]
        for attr in ("_canvas", "_text_label"):
            target = getattr(widget, attr, None)
            if target is not None:
                targets.append(target)
        for target in targets:
            try:
                target.bind("<Button-1>", self._activate, add="+")
                target.configure(cursor="hand2")
            except (tk.TclError, ValueError):
                pass

    def _make_name_group(self) -> dict[str, ctk.CTkBaseClass]:
        group = ctk.CTkFrame(self.frame, fg_color="transparent")
        badge = ctk.CTkLabel(
            group,
            text="",
            width=64,
            height=20,
            font=self.fonts["badge"],
            text_color=CATEGORY_BADGE_TEXT,
            corner_radius=4,
        )
        badge.pack(side="left", padx=(0, 2))
        separator = ctk.CTkLabel(group, text="", anchor="w")
        separator.pack(side="left")
        text = ctk.CTkLabel(group, text="", anchor="w", justify="left")
        text.pack(side="left")
        return {"frame": group, "badge": badge, "separator": separator, "text": text}

    def _set_name(
        self,
        group: dict[str, ctk.CTkBaseClass],
        value: str,
        font: ctk.CTkFont,
        color: str,
    ) -> None:
        parsed = parse_category_prefix_display(value)
        badge = group["badge"]
        separator = group["separator"]
        label = group["text"]
        label.configure(font=font, text_color=color)
        separator.configure(font=font, text_color=color)
        if parsed:
            category, remainder = parsed
            badge.configure(
                text=category_badge_label(category),
                fg_color=self.category_colors.get(
                    category,
                    default_category_color(category),
                ),
            )
            if not badge.winfo_manager():
                badge.pack(side="left", padx=(0, 2), before=separator)
            separator.configure(text=" - ")
            label.configure(text=remainder)
        else:
            badge.pack_forget()
            separator.configure(text="")
            label.configure(text=value)

    def bind(
        self,
        index: int,
        track: Track,
        row: PreviewRow | None,
        theme: dict,
        width: int,
        active: bool = False,
        category_colors: dict[str, str] | None = None,
        grouped: bool = False,
    ) -> None:
        self.index = index
        self.track = track
        self.row = row
        self._render_header = None
        self.theme = theme
        self._render_selected = track.selected
        self._render_active = active
        self._render_grouped = grouped
        self.category_colors = category_colors if category_colors is not None else {}
        self.var.set(track.selected)
        self.folder_label.pack_forget()
        for widget in (
            self.checkbox,
            self.old_group["frame"],
            self.arrow,
            self.new_group["frame"],
            self.status,
        ):
            widget.pack_forget()
        self.checkbox.pack(side="left", padx=(6, 4))
        self.old_group["frame"].pack(side="left", anchor="w")
        self.status.pack(side="right", padx=8)
        self.checkbox.configure(
            fg_color=theme["accent"],
            hover_color=theme["accent_hover"],
            border_color=theme["border"],
        )
        base_indent = 18 if grouped else 6
        self.checkbox.pack_configure(padx=(10 * track.depth + base_indent, 4))

        effective_changed = bool(row is not None and row.changed and track.selected)
        if active:
            self.frame.configure(fg_color=theme["active_row"])
        elif effective_changed:
            self.frame.configure(fg_color=theme["accent_soft"])
        elif theme.get("row_odd"):
            self.frame.configure(
                fg_color=theme["row_even"] if index % 2 == 0 else theme["row_odd"]
            )
        else:
            self.frame.configure(fg_color="transparent")

        if row is None:
            self._set_name(
                self.old_group,
                track.display_name,
                self.fonts["normal"],
                theme.get("list_fg", theme["text"]),
            )
            self.arrow.pack_forget()
            self.new_group["frame"].pack_forget()
            self.status.configure(text="…", text_color=theme["text_mute"])
        elif effective_changed:
            self._set_name(
                self.old_group, row.original_display, self.fonts["strike"], theme["text_dim"]
            )
            if not self.arrow.winfo_manager():
                self.arrow.pack(side="left", padx=4, before=self.status)
            self.arrow.configure(text_color=theme["changed"])
            if not self.new_group["frame"].winfo_manager():
                self.new_group["frame"].pack(side="left", anchor="w", before=self.status)
            self._set_name(
                self.new_group,
                row.new_display,
                self.fonts["bold"],
                theme.get("list_fg", theme["text"]),
            )
            self.status.configure(text="")
        else:
            self._set_name(
                self.old_group,
                row.original_display,
                self.fonts["normal"],
                theme.get("list_fg", theme["text"]),
            )
            self.arrow.pack_forget()
            self.new_group["frame"].pack_forget()
            self.status.configure(
                text="unchanged" if not row.changed else "",
                text_color=theme["unchanged"],
            )

    def show(
        self,
        canvas: tk.Canvas,
        display_index: int,
        y: int,
        width: int,
    ) -> None:
        self.display_index = display_index
        canvas.coords(self.window_id, 0, y)
        canvas.itemconfigure(self.window_id, width=width, state="normal")

    def hide(self, canvas: tk.Canvas) -> None:
        self.display_index = -1
        canvas.itemconfigure(self.window_id, state="hidden")

    def destroy(self, canvas: tk.Canvas) -> None:
        canvas.delete(self.window_id)
        self.frame.destroy()

    def _toggle(self) -> None:
        if self.track is not None and self.index >= 0:
            self.on_toggle(self.index, self.row, self.track, bool(self.var.get()))

    def _activate(self, _event=None) -> None:
        if self.track is not None and self.index >= 0:
            self.on_activate(self.index, self.row, self.track)


class PreviewPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        theme: dict,
        on_change,
        on_active=None,
        on_play_pause=None,
        on_seek=None,
        **kwargs,
    ):
        super().__init__(master, fg_color=theme["panel"], corner_radius=12, **kwargs)
        self.theme = theme
        self.on_change = on_change
        self.on_active = on_active or (lambda _track, _row: None)
        self.on_play_pause = on_play_pause or (lambda: None)
        self.on_seek = on_seek or (lambda _seconds: None)
        self._active_track_id: str | None = None
        self.rows: list[PreviewRow] = []
        self.only_changed = ctk.BooleanVar(value=False)
        self._loading = False
        self._preview_pending = False
        self._stats_text = "0 will change · 0 unchanged"
        self._changed_count = 0
        self._last_batch_len = 0
        self._filtered_dirty = True
        self._filtered_cache: list[PreviewRow] = []
        self._lazy_enabled = False
        self._lazy_tracks = []
        self._lazy_rules = []
        self._lazy_rows: list[PreviewRow | None] = []
        self._lazy_done = 0
        self._lazy_changed = 0
        self._lazy_selected_changed = 0
        self._lazy_generation = 0
        self._preview_job: _PreviewJob | None = None
        self._result_poll_job: str | None = None
        self._lazy_dirty = False
        self._scroll_idle_job: str | None = None
        self._is_scrolling = False
        self._row_pool: list[_PreviewRowWidget] = []
        self._pool_first = -1
        self._pool_last = -1
        self._visible_source_indices: set[int] = set()
        self._lazy_view_entries: list[int | _FolderHeader] = []
        self._folder_header_positions: list[int] = []
        self._root_folder_label = "ROOT"
        self._category_colors: dict[str, str] = {}
        self._render_job: str | None = None
        self._canvas_width = 400
        self._init_log_fonts()
        self._build()

    def _init_log_fonts(self) -> None:
        family = PREVIEW_LOG_FONT_FAMILY
        size = PREVIEW_LOG_FONT_SIZE
        self._log_font = ctk.CTkFont(family=family, size=size)
        self._log_font_bold = ctk.CTkFont(family=family, size=size, weight="bold")
        self._log_font_strike = ctk.CTkFont(family=family, size=size, overstrike=True)
        self._log_font_italic = ctk.CTkFont(family=family, size=size, slant="italic")
        self._category_badge_font = ctk.CTkFont(size=10, weight="bold")

    def _tip(self, widget, key: str) -> None:
        bind_tooltip(widget, TIPS[key], self.theme)

    def _list_bg(self, theme: dict | None = None) -> str:
        t = theme or self.theme
        return t.get("list_bg", t["card"])

    def _build(self) -> None:
        t = self.theme
        list_bg = self._list_bg(t)
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 8))

        ctk.CTkLabel(
            header,
            text="PREVIEW",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=t["text_mute"],
        ).pack(side="left")

        self.stats_label = ctk.CTkLabel(
            header,
            text="0 will change · 0 unchanged",
            font=self._log_font,
            text_color=t["text_dim"],
        )
        self.stats_label.pack(side="left", padx=(12, 0))

        tools = ctk.CTkFrame(self, fg_color="transparent")
        tools.pack(fill="x", padx=16, pady=(0, 6))

        select_all_btn = ctk.CTkButton(
            tools, text="Select all", width=80, height=26, fg_color=t["btn"],
            hover_color=t["btn_hover"], text_color=t["text_dim"], command=self._select_all,
        )
        select_all_btn.pack(side="left", padx=(0, 6))
        self.select_all_btn = select_all_btn
        self._tip(select_all_btn, "select_all")

        deselect_btn = ctk.CTkButton(
            tools, text="Deselect all", width=90, height=26, fg_color=t["btn"],
            hover_color=t["btn_hover"], text_color=t["text_dim"], command=self._deselect_all,
        )
        deselect_btn.pack(side="left", padx=(0, 12))
        self.deselect_btn = deselect_btn
        self._tip(deselect_btn, "deselect_all")

        only_changed_cb = ctk.CTkCheckBox(
            tools,
            text="Only changed",
            variable=self.only_changed,
            command=self._on_filter_changed,
            font=ctk.CTkFont(size=12),
            text_color=t["text_dim"],
            fg_color=t["accent"],
            hover_color=t["accent_hover"],
        )
        only_changed_cb.pack(side="left")
        self.only_changed_cb = only_changed_cb
        self._tip(only_changed_cb, "only_changed")

        list_outer = ctk.CTkFrame(self, fg_color=list_bg, corner_radius=8)
        list_outer.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        list_outer.grid_rowconfigure(0, weight=1)
        list_outer.grid_columnconfigure(0, weight=1)
        self.list_outer = list_outer

        self.canvas = tk.Canvas(
            list_outer,
            highlightthickness=0,
            borderwidth=0,
            bg=list_bg,
            takefocus=True,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Up>", lambda _event: self._keyboard_move(-1))
        self.canvas.bind("<Down>", lambda _event: self._keyboard_move(1))
        self.canvas.bind("<Prior>", lambda _event: self._keyboard_page(-1))
        self.canvas.bind("<Next>", lambda _event: self._keyboard_page(1))
        self.canvas.bind("<Left>", lambda _event: self._keyboard_seek(-3.0))
        self.canvas.bind("<Right>", lambda _event: self._keyboard_seek(3.0))
        self.canvas.bind("<space>", self._keyboard_play_pause)

        self.scrollbar = ctk.CTkScrollbar(list_outer, command=self._on_scroll)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        self.sticky_folder_label = ctk.CTkLabel(
            list_outer,
            text="",
            anchor="w",
            height=ROW_HEIGHT,
            corner_radius=0,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=t["text_mute"],
            fg_color=t["panel_2"],
        )
        self.sticky_folder_label.bind("<Enter>", self._bind_mousewheel)
        self.sticky_folder_label.bind("<Leave>", self._unbind_mousewheel)

        self.status_label = ctk.CTkLabel(
            list_outer,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=t["text_mute"],
            fg_color=t.get("loading_bg", "transparent"),
            corner_radius=8,
        )
        self.status_label.place(relx=0.5, rely=0.5, anchor="center")

        self.spinner = ctk.CTkProgressBar(
            list_outer,
            width=160,
            height=8,
            corner_radius=8,
            mode="indeterminate",
            progress_color=t["accent"],
            fg_color=t.get("loading_bg", t["panel_2"]),
        )
        self.spinner.place(relx=0.5, rely=0.56, anchor="center")
        self.spinner.lower()
        self.spinner.stop()

    def _bind_mousewheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")
        self._note_scroll_activity()
        self._schedule_render()

    def _on_scroll(self, *args) -> None:
        self.canvas.yview(*args)
        self._note_scroll_activity()
        self._schedule_render()

    def _note_scroll_activity(self) -> None:
        self._is_scrolling = True

        if self._scroll_idle_job:
            try:
                self.after_cancel(self._scroll_idle_job)
            except Exception:
                pass

        def clear() -> None:
            self._scroll_idle_job = None
            self._is_scrolling = False
            if self._lazy_dirty:
                self._lazy_dirty = False
                self._schedule_render(immediate=True)

        self._scroll_idle_job = self.after(140, clear)

    def _on_canvas_configure(self, event) -> None:
        self._canvas_width = max(event.width, 1)
        self._update_scroll_region()
        self._schedule_render()

    def _on_filter_changed(self) -> None:
        if self._lazy_enabled:
            if not self.lazy_compute_complete():
                self.only_changed.set(False)
                return
            source_indices = (
                (i for i, row in enumerate(self._lazy_rows) if row and row.changed)
                if self.only_changed.get()
                else range(len(self._lazy_tracks))
            )
            self._set_folder_view(source_indices)
            self.canvas.yview_moveto(0)
            self._clear_row_widgets()
            self._update_scroll_region()
            self._schedule_render(immediate=True)
            return
        self.canvas.yview_moveto(0)
        self._clear_row_widgets()
        self._filtered_dirty = True
        self._update_scroll_region()
        self._schedule_render(immediate=True)

    def _build_folder_view(self, source_indices) -> list[int | _FolderHeader]:
        indices = list(source_indices)
        if not indices or not any(
            self._lazy_tracks[index].relative_path for index in indices
        ):
            return indices

        entries: list[int | _FolderHeader] = []
        previous_folder: str | None = None
        for index in indices:
            relative_path = self._lazy_tracks[index].relative_path
            parent = Path(relative_path).parent
            if str(parent) == ".":
                folder = self._root_folder_label
            else:
                folder = " › ".join(parent.parts)
            if folder != previous_folder:
                entries.append(_FolderHeader(folder))
                previous_folder = folder
            entries.append(index)
        return entries

    def _set_folder_view(self, source_indices) -> None:
        self._lazy_view_entries = self._build_folder_view(source_indices)
        self._folder_header_positions = [
            index
            for index, entry in enumerate(self._lazy_view_entries)
            if isinstance(entry, _FolderHeader)
        ]

    def set_theme(self, theme: dict) -> None:
        self.theme = theme
        list_bg = self._list_bg(theme)
        self.configure(fg_color=theme["panel"])
        self.list_outer.configure(fg_color=list_bg)
        self.canvas.configure(bg=list_bg)
        self.stats_label.configure(text_color=theme["text_dim"])
        self.status_label.configure(text_color=theme["text_mute"])
        self.status_label.configure(fg_color=theme.get("loading_bg", "transparent"))
        self.sticky_folder_label.configure(
            text_color=theme["text_mute"],
            fg_color=theme["panel_2"],
        )
        self.spinner.configure(
            progress_color=theme["accent"],
            fg_color=theme.get("loading_bg", theme["panel_2"]),
        )
        for btn in (self.select_all_btn, self.deselect_btn):
            btn.configure(
                fg_color=theme["btn"],
                hover_color=theme["btn_hover"],
                text_color=theme["text_dim"],
            )
        self.only_changed_cb.configure(
            text_color=theme["text_dim"],
            fg_color=theme["accent"],
            hover_color=theme["accent_hover"],
        )
        if self._preview_pending:
            self.stats_label.configure(text="Rules changed — click Apply", text_color=theme["accent"])
        else:
            self.stats_label.configure(text=self._stats_text, text_color=theme["text_dim"])
        self._clear_row_widgets()
        self._schedule_render(immediate=True)

    def begin_viewport_lazy(self, tracks, rules, root_label: str = "") -> None:
        """Start a cancellable full preview with viewport-priority scheduling."""
        try:
            prev_y = self.canvas.yview()[0]
        except Exception:
            prev_y = 0.0

        # Entering lazy mode supersedes any prior blocking "loading" overlay.
        self._loading = False
        self.status_label.configure(text="")
        self.status_label.lower()
        self.spinner.stop()
        self.spinner.lower()

        self._lazy_generation += 1
        self._lazy_enabled = True
        self.rows = []
        self._lazy_tracks = list(tracks)
        self._root_folder_label = root_label or "ROOT"
        self._category_colors = applied_category_colors(list(rules))
        self._lazy_rules = prepare_rules(rules)
        self._lazy_rows = [None] * len(self._lazy_tracks)
        self._lazy_done = 0
        self._lazy_changed = 0
        self._lazy_selected_changed = 0
        self._set_folder_view(range(len(self._lazy_tracks)))

        active_track = next(
            (track for track in self._lazy_tracks if track.id == self._active_track_id),
            None,
        )
        if self._active_track_id is not None and active_track is None:
            self.clear_active()
        elif active_track is not None:
            self.on_active(active_track, None)

        if self._preview_job is not None:
            self._preview_job.cancel.set()
        job = _PreviewJob(
            generation=self._lazy_generation,
            tracks=self._lazy_tracks,
            rules=self._lazy_rules,
        )
        self._preview_job = job

        self.only_changed.set(False)
        self.only_changed_cb.configure(state="disabled")

        total = len(self._lazy_tracks)
        self._stats_text = f"0 will change · 0 / {total:,} computed"
        if not self._preview_pending:
            self.stats_label.configure(
                text=self._stats_text,
                text_color=self.theme["text_dim"],
            )
        self.status_label.configure(text="")
        self.status_label.lower()
        self._clear_row_widgets()
        self._update_scroll_region()
        try:
            self.canvas.yview_moveto(prev_y)
        except Exception:
            pass
        first, last = self._visible_bounds(total)
        self._queue_job_priority(job, range(first, min(total, last + LAZY_BUFFER_ROWS)))
        self._schedule_render(immediate=True)

        threading.Thread(
            target=self._preview_worker,
            args=(job,),
            daemon=True,
        ).start()
        self._schedule_result_poll()

    def lazy_compute_complete(self) -> bool:
        if not self._lazy_enabled:
            return True
        return self._lazy_done >= len(self._lazy_tracks)

    def lazy_compute_progress(self) -> tuple[int, int]:
        if not self._lazy_enabled:
            return 0, 0
        return self._lazy_done, len(self._lazy_tracks)

    def end_lazy_mode(self) -> None:
        if self._preview_job is not None:
            self._preview_job.cancel.set()
            self._preview_job = None
        if self._result_poll_job is not None:
            try:
                self.after_cancel(self._result_poll_job)
            except Exception:
                pass
            self._result_poll_job = None
        self._lazy_enabled = False
        self.only_changed_cb.configure(state="normal")

    @staticmethod
    def _preview_worker(job: _PreviewJob) -> None:
        total = len(job.tracks)
        completed = bytearray(total)
        done = 0
        next_index = 0
        batch: list[tuple[int, PreviewRow]] = []
        last_emit = time.perf_counter()

        while done < total and not job.cancel.is_set():
            try:
                idx = job.priority.get_nowait()
            except queue.Empty:
                while next_index < total and completed[next_index]:
                    next_index += 1
                if next_index >= total:
                    break
                idx = next_index
                next_index += 1

            if idx < 0 or idx >= total or completed[idx]:
                continue
            row = compute_preview_row(job.tracks[idx], job.rules, index=idx + 1)
            completed[idx] = 1
            done += 1
            batch.append((idx, row))

            now = time.perf_counter()
            if len(batch) >= RESULT_BATCH_SIZE or now - last_emit >= 0.025:
                job.results.put((batch, False))
                batch = []
                last_emit = now

        if batch:
            job.results.put((batch, False))
        job.results.put(([], True))

    def _schedule_result_poll(self) -> None:
        if self._result_poll_job is None:
            self._result_poll_job = self.after(RESULT_POLL_MS, self._drain_preview_results)

    def _queue_job_priority(self, job: _PreviewJob, indices) -> None:
        for index in indices:
            if (
                0 <= index < len(job.tracks)
                and index not in job.requested
                and self._lazy_rows[index] is None
            ):
                job.requested.add(index)
                job.priority.put(index)

    def _drain_preview_results(self) -> None:
        self._result_poll_job = None
        job = self._preview_job
        if job is None or job.generation != self._lazy_generation:
            return
        finished = False
        visible_changed = False
        active_update: tuple[Track, PreviewRow] | None = None
        while True:
            try:
                batch, batch_finished = job.results.get_nowait()
            except queue.Empty:
                break
            finished = finished or batch_finished
            for index, row in batch:
                if self._lazy_rows[index] is not None:
                    continue
                self._lazy_rows[index] = row
                self._lazy_done += 1
                if row.changed:
                    self._lazy_changed += 1
                    if row.track.selected:
                        self._lazy_selected_changed += 1
                if index in self._visible_source_indices:
                    visible_changed = True
                if row.track.id == self._active_track_id:
                    active_update = (row.track, row)

        done = self._lazy_done
        total = len(self._lazy_tracks)
        changed = self._lazy_changed
        if finished and done == total:
            unchanged = total - changed
            self._stats_text = f"{changed:,} will change · {unchanged:,} unchanged"
            self.only_changed_cb.configure(state="normal")
        else:
            self._stats_text = f"{changed:,} will change · {done:,} / {total:,} computed"
        if not self._preview_pending:
            self.stats_label.configure(
                text=self._stats_text,
                text_color=self.theme["text_dim"],
            )

        if visible_changed and not self._is_scrolling:
            self._schedule_render(immediate=True)
        if active_update is not None:
            self.on_active(*active_update)
        self.on_change()
        if not finished:
            self._schedule_result_poll()

    def cancel_preview_work(self) -> None:
        """Stop obsolete background work while retaining the current visible preview."""
        if self._preview_job is not None:
            self._preview_job.cancel.set()

    def shutdown(self) -> None:
        self.cancel_preview_work()
        for attr in ("_render_job", "_result_poll_job", "_scroll_idle_job"):
            job = getattr(self, attr, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except (tk.TclError, ValueError):
                    pass
                setattr(self, attr, None)

    def set_preview_pending(self, pending: bool) -> None:
        self._preview_pending = pending
        if pending:
            self.stats_label.configure(
                text="Rules changed — click Apply",
                text_color=self.theme["accent"],
            )
        else:
            self.stats_label.configure(
                text=self._stats_text,
                text_color=self.theme["text_dim"],
            )

    def begin_incremental_load(self, total: int) -> None:
        self._loading = True
        self.end_lazy_mode()
        try:
            prev_y = self.canvas.yview()[0]
        except Exception:
            prev_y = 0.0
        self.rows = []
        self._changed_count = 0
        self._last_batch_len = 0
        self._filtered_dirty = True
        self._filtered_cache = []
        self._incremental_total = total
        self._stats_text = f"0 will change · 0 / {total:,} scanned"
        self.stats_label.configure(text=self._stats_text, text_color=self.theme["text_dim"])
        self.status_label.configure(text="Updating preview…")
        self.status_label.lift()
        self._clear_row_widgets()
        self._update_scroll_region()
        try:
            self.canvas.yview_moveto(prev_y)
        except Exception:
            pass

    def update_rows_batch(self, rows: list[PreviewRow], done: int, total: int) -> None:
        # Incremental accounting avoids O(n) on every batch/scroll.
        new_len = len(rows)
        if new_len > self._last_batch_len:
            self._changed_count += sum(1 for r in rows[self._last_batch_len : new_len] if r.changed)
            self._last_batch_len = new_len
        self.rows = rows
        self._filtered_dirty = True
        changed = self._changed_count
        if done < total:
            self._stats_text = f"{changed:,} will change · {done:,} / {total:,} scanned"
            self.stats_label.configure(text=self._stats_text, text_color=self.theme["text_dim"])
            if done >= PREVIEW_BATCH_SIZE:
                self._loading = False
                self.status_label.configure(text="")
                self.status_label.lower()
        else:
            unchanged = len(rows) - changed
            self._loading = False
            self._preview_pending = False
            self._stats_text = f"{changed:,} will change · {unchanged:,} unchanged"
            self.stats_label.configure(text=self._stats_text, text_color=self.theme["text_dim"])
            self.status_label.configure(text="")
            self.status_label.lower()
        self._update_scroll_region()
        self._schedule_render(immediate=True)

    def set_loading(self, loading: bool) -> None:
        self._loading = loading
        if loading:
            self.status_label.configure(text="Updating preview…")
            self.status_label.lift()
            self.spinner.lift()
            self.spinner.start()
        else:
            self.status_label.configure(text="")
            self.status_label.lower()
            self.spinner.stop()
            self.spinner.lower()

    def set_rows(self, rows: list[PreviewRow]) -> None:
        self._loading = False
        self.end_lazy_mode()
        try:
            prev_y = self.canvas.yview()[0]
        except Exception:
            prev_y = 0.0
        self.rows = rows
        self._changed_count = sum(1 for r in rows if r.changed)
        self._last_batch_len = len(rows)
        self._filtered_dirty = True
        changed = self._changed_count
        unchanged = len(rows) - changed
        self._stats_text = f"{changed:,} will change · {unchanged:,} unchanged"
        self._preview_pending = False
        self.stats_label.configure(text=self._stats_text, text_color=self.theme["text_dim"])
        self.status_label.configure(text="")
        self.status_label.lower()
        self._clear_row_widgets()
        self._update_scroll_region()
        try:
            self.canvas.yview_moveto(prev_y)
        except Exception:
            pass
        self._schedule_render(immediate=True)

    def _filtered_rows(self) -> list[PreviewRow]:
        if self._lazy_enabled:
            # Lazy mode doesn't support "only changed" filtering.
            return []
        if not self._filtered_dirty:
            return self._filtered_cache
        if self.only_changed.get():
            self._filtered_cache = [r for r in self.rows if r.changed]
        else:
            self._filtered_cache = self.rows
        self._filtered_dirty = False
        return self._filtered_cache

    def _update_scroll_region(self) -> None:
        if self._lazy_enabled:
            count = len(self._lazy_view_entries)
        else:
            count = len(self._filtered_rows())
        height = max(count * ROW_HEIGHT, self.canvas.winfo_height(), 1)
        self.canvas.configure(scrollregion=(0, 0, self._canvas_width, height))

    def _schedule_render(self, immediate: bool = False) -> None:
        if self._render_job:
            if not immediate:
                return
            try:
                self.after_cancel(self._render_job)
            except Exception:
                pass
        delay = 0 if immediate else 16
        self._render_job = self.after(delay, self._render_visible)

    def _clear_row_widgets(self) -> None:
        for slot in self._row_pool:
            slot.hide(self.canvas)
        self.sticky_folder_label.place_forget()
        self._pool_first = -1
        self._pool_last = -1
        self._visible_source_indices.clear()

    def _invalidate_row_widget(self, index: int) -> None:
        """Pooled rows are rebound on the next render."""
        if self._pool_first <= index < self._pool_last:
            self._schedule_render(immediate=True)

    def _ensure_row_pool(self, size: int) -> None:
        fonts = {
            "normal": self._log_font,
            "bold": self._log_font_bold,
            "strike": self._log_font_strike,
            "italic": self._log_font_italic,
            "badge": self._category_badge_font,
        }
        while len(self._row_pool) < size:
            self._row_pool.append(
                _PreviewRowWidget(
                    self.canvas,
                    self.theme,
                    fonts,
                    self._handle_pool_toggle,
                    self._handle_row_activate,
                )
            )

    def _handle_pool_toggle(
        self,
        index: int,
        row: PreviewRow | None,
        track: Track,
        selected: bool,
    ) -> None:
        previous = track.selected
        if previous == selected:
            return
        track.selected = selected
        if self._lazy_enabled and row is not None and row.changed:
            self._lazy_selected_changed += 1 if selected else -1
        self._schedule_render(immediate=True)
        self.on_change()

    def _handle_row_activate(
        self,
        _index: int,
        row: PreviewRow | None,
        track: Track,
    ) -> None:
        if not track.is_audio or not track.is_file:
            return
        self.canvas.focus_set()
        if self._active_track_id == track.id:
            self.on_active(track, row)
            return
        self._active_track_id = track.id
        self._schedule_render(immediate=True)
        self.on_active(track, row)

    def _display_source_indices(self):
        if self._lazy_enabled:
            return self._lazy_view_entries
        return range(len(self._filtered_rows()))

    def _active_display_index(self, indices) -> int | None:
        if self._active_track_id is None:
            return None
        tracks = self._lazy_tracks if self._lazy_enabled else [
            row.track for row in self._filtered_rows()
        ]
        for display_index, source_index in enumerate(indices):
            if isinstance(source_index, _FolderHeader):
                continue
            if tracks[source_index].id == self._active_track_id:
                return display_index
        return None

    def _keyboard_move(self, direction: int):
        indices = self._display_source_indices()
        total = len(indices)
        if total == 0:
            return "break"
        active = self._active_display_index(indices)
        target = (0 if direction > 0 else total - 1) if active is None else active + direction
        self._activate_keyboard_target(indices, target, direction)
        return "break"

    def _keyboard_page(self, direction: int):
        indices = self._display_source_indices()
        total = len(indices)
        if total == 0:
            return "break"
        page_rows = max(1, int(self.canvas.winfo_height() / ROW_HEIGHT) - 1)
        active = self._active_display_index(indices)
        target = (
            0 if direction > 0 else total - 1
        ) if active is None else active + direction * page_rows
        target = max(0, min(total - 1, target))
        self._activate_keyboard_target(indices, target, direction)
        return "break"

    def _activate_keyboard_target(self, indices, target: int, direction: int) -> None:
        tracks = self._lazy_tracks if self._lazy_enabled else [
            row.track for row in self._filtered_rows()
        ]
        rows = self._lazy_rows if self._lazy_enabled else list(self._filtered_rows())
        total = len(indices)
        while 0 <= target < total:
            source_index = indices[target]
            if isinstance(source_index, _FolderHeader):
                target += direction
                continue
            track = tracks[source_index]
            if track.is_audio and track.is_file:
                row = rows[source_index]
                self._active_track_id = track.id
                self._ensure_display_visible(target, total)
                self._schedule_render(immediate=True)
                self.on_active(track, row)
                return
            target += direction

    def _ensure_display_visible(self, display_index: int, total: int) -> None:
        viewport = max(self.canvas.winfo_height(), ROW_HEIGHT)
        top = self.canvas.canvasy(0)
        row_top = display_index * ROW_HEIGHT
        row_bottom = row_top + ROW_HEIGHT
        content_height = max(total * ROW_HEIGHT, viewport)
        if row_top < top:
            self.canvas.yview_moveto(row_top / content_height)
        elif row_bottom > top + viewport:
            target_top = max(0, row_bottom - viewport)
            self.canvas.yview_moveto(target_top / content_height)

    def _keyboard_play_pause(self, _event=None):
        if self._active_track_id is not None:
            self.on_play_pause()
        return "break"

    def _keyboard_seek(self, seconds: float):
        if self._active_track_id is not None:
            self.on_seek(seconds)
        return "break"

    def clear_active(self) -> None:
        if self._active_track_id is None:
            return
        self._active_track_id = None
        self._schedule_render(immediate=True)
        self.on_active(None, None)

    def _visible_bounds(self, count: int) -> tuple[int, int]:
        top = self.canvas.canvasy(0)
        viewport = max(self.canvas.winfo_height(), ROW_HEIGHT)
        visible_count = int(viewport // ROW_HEIGHT) + RENDER_BUFFER * 2
        first = max(0, int(top // ROW_HEIGHT) - RENDER_BUFFER)
        return first, min(count, first + visible_count)

    def _update_sticky_folder(self, total: int) -> None:
        if not self._folder_header_positions or total <= 0:
            self.sticky_folder_label.place_forget()
            return
        top_display = max(
            0,
            min(total - 1, int(self.canvas.canvasy(0) // ROW_HEIGHT)),
        )
        position_index = bisect_right(
            self._folder_header_positions, top_display
        ) - 1
        if position_index < 0:
            self.sticky_folder_label.place_forget()
            return
        header_position = self._folder_header_positions[position_index]
        header = self._lazy_view_entries[header_position]
        if not isinstance(header, _FolderHeader):
            self.sticky_folder_label.place_forget()
            return
        self.sticky_folder_label.configure(
            text=f"  {header.label}",
            width=self._canvas_width,
            height=ROW_HEIGHT,
        )
        self.sticky_folder_label.place(x=0, y=0)
        self.sticky_folder_label.lift()

    def _render_pool(
        self,
        first: int,
        last: int,
        tracks: list[Track],
        rows: list[PreviewRow | None],
        index_map: list[int | _FolderHeader] | None = None,
    ) -> None:
        count = max(0, last - first)
        self._ensure_row_pool(count)
        self._pool_first = first
        self._pool_last = last
        self._visible_source_indices = set()

        needed = set(range(first, last))
        slots_by_display = {
            slot.display_index: slot
            for slot in self._row_pool
            if slot.display_index in needed
        }
        free_slots = [
            slot for slot in self._row_pool if slot.display_index not in needed
        ]
        for slot in free_slots:
            slot.hide(self.canvas)

        for display_index in range(first, last):
            slot = slots_by_display.get(display_index)
            if slot is None:
                slot = free_slots.pop()
            entry = (
                index_map[display_index] if index_map is not None else display_index
            )
            if isinstance(entry, _FolderHeader):
                if (
                    slot._render_header != entry.label
                    or slot.theme is not self.theme
                ):
                    slot.bind_header(entry.label, self.theme)
                    slot.theme = self.theme
                slot.show(
                    self.canvas,
                    display_index,
                    display_index * ROW_HEIGHT,
                    self._canvas_width,
                )
                continue
            source_index = entry
            self._visible_source_indices.add(source_index)
            track = tracks[source_index]
            row = rows[source_index]
            needs_bind = (
                slot.index != source_index
                or slot.row is not row
                or slot.track is not track
                or slot._render_selected != track.selected
                or slot._render_active != (track.id == self._active_track_id)
                or slot.category_colors is not self._category_colors
                or slot.theme is not self.theme
                or slot._render_grouped != bool(self._folder_header_positions)
            )
            if needs_bind:
                slot.bind(
                    source_index,
                    track,
                    row,
                    self.theme,
                    self._canvas_width,
                    track.id == self._active_track_id,
                    self._category_colors,
                    bool(self._folder_header_positions),
                )
            slot.show(
                self.canvas,
                display_index,
                display_index * ROW_HEIGHT,
                self._canvas_width,
            )

    def _render_visible(self) -> None:
        self._render_job = None
        if self._lazy_enabled:
            self._render_visible_lazy()
            return

        rows = self._filtered_rows()
        if not rows:
            self._clear_row_widgets()
            return

        first, last = self._visible_bounds(len(rows))
        self._render_pool(first, last, [row.track for row in rows], list(rows))

    def _render_visible_lazy(self) -> None:
        view_entries = self._lazy_view_entries
        total = len(view_entries)
        if total == 0:
            self._clear_row_widgets()
            return

        self._update_sticky_folder(total)
        first, last = self._visible_bounds(total)

        visible_sources = [
            entry
            for entry in view_entries[first:last]
            if isinstance(entry, int)
        ]
        missing_in_view = sum(
            1 for source_index in visible_sources
            if self._lazy_rows[source_index] is None
        )
        visible_total = max(len(visible_sources), 1)
        computed_in_view = visible_total - missing_in_view

        if computed_in_view == 0 and missing_in_view:
            self.status_label.configure(text="Loading…")
            self.status_label.lift()
            self.spinner.lift()
            self.spinner.start()
        else:
            # Hide the overlay once we have anything meaningful to show.
            self.status_label.configure(text="")
            self.status_label.lower()
            self.spinner.stop()
            self.spinner.lower()

        job = self._preview_job
        if job is not None:
            priority_indices = [
                view_entries[i]
                for i in range(first, last)
                if isinstance(view_entries[i], int)
            ]
            priority_indices.extend(
                view_entries[i]
                for i in range(max(0, first - LAZY_BUFFER_ROWS), first)
                if isinstance(view_entries[i], int)
            )
            priority_indices.extend(
                view_entries[i]
                for i in range(last, min(total, last + LAZY_BUFFER_ROWS))
                if isinstance(view_entries[i], int)
            )
            self._queue_job_priority(job, priority_indices)

        self._render_pool(
            first,
            last,
            self._lazy_tracks,
            self._lazy_rows,
            view_entries,
        )

    def _select_all(self) -> None:
        if self._lazy_enabled:
            for track in self._lazy_tracks:
                track.selected = True
            self._lazy_selected_changed = self._lazy_changed
        else:
            for row in self._filtered_rows():
                row.track.selected = True
        self._clear_row_widgets()
        self._schedule_render(immediate=True)
        self.on_change()

    def _deselect_all(self) -> None:
        if self._lazy_enabled:
            for track in self._lazy_tracks:
                track.selected = False
            self._lazy_selected_changed = 0
        else:
            for row in self._filtered_rows():
                row.track.selected = False
        self._clear_row_widgets()
        self._schedule_render(immediate=True)
        self.on_change()

    def selected_renames(self) -> dict[str, str]:
        if self._lazy_enabled:
            if not self.lazy_compute_complete():
                return {}
            return {
                row.track.id: row.new_name
                for row in self._lazy_rows
                if row is not None and row.changed and row.track.selected
            }

        return {
            row.track.id: row.new_name
            for row in self.rows
            if row.changed and row.track.selected
        }

    def rename_count(self) -> int:
        if self._lazy_enabled:
            return self._lazy_selected_changed
        return sum(1 for row in self.rows if row.changed and row.track.selected)
