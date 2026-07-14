"""Embedded Track Renamer panel for the STEM Organizer host."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import customtkinter as ctk

from track_renamer.category_palette import normalize_rules_category_colors
from track_renamer.engine.defaults import make_default_rules, make_demo_tracks
from track_renamer.engine.models import Rule, rule_from_dict, rule_to_dict
from track_renamer.folder_scanner import (
    apply_file_renames_detailed,
    move_files_to_prefix_folders,
)
from track_renamer.gui.app import PRESETS_DIR, TrackRenamerApp
from track_renamer.gui.help_dialog import show_rename_help_dialog
from track_renamer.gui.theme import DARK


class TrackRenamerPanel(ctk.CTkFrame):
    """The standalone TrackRenamerApp controller hosted inside a CTkFrame."""

    def __init__(self, master, host=None) -> None:
        super().__init__(master, fg_color=DARK["bg"])
        self.host = host
        self.dark_mode = True
        self.theme = DARK.copy()
        self.folder_path: Path | None = None
        self.recursive = True
        self.tracks = make_demo_tracks()
        self.rules: list[Rule] = make_default_rules()
        self.demo_mode = True
        self._scan_generation = 0
        self._preview_generation = 0
        self._preview_stale = False
        self._busy = False
        self._applied_rules_fingerprint = self._rules_fingerprint(self.rules)
        self._destructive_busy = False
        self._shutdown = False

        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        self._build_ui()
        self._apply_preview()
        self._update_footer()

    @property
    def destructive_busy(self) -> bool:
        """True only while renaming or moving renamed files."""
        return self._destructive_busy

    def _dialog_parent(self):
        candidate = self.host
        if isinstance(candidate, tk.Misc):
            try:
                if candidate.winfo_exists():
                    return candidate
            except tk.TclError:
                pass
        try:
            return self.winfo_toplevel()
        except tk.TclError:
            return self

    def on_tab_shown(self) -> None:
        """Notify the panel that its host tab is visible."""
        if not self._shutdown:
            self._update_footer()
            self.after_idle(self.focus_workspace)

    def focus_workspace(self) -> None:
        """Keep tab activation from placing an insertion cursor in rule fields."""
        if self._shutdown:
            return
        try:
            self.preview_panel.canvas.focus_set()
        except tk.TclError:
            pass

    def on_tab_hidden(self) -> None:
        """Stop audible playback while retaining rules, selection, and preview."""
        if self._shutdown:
            return
        self.audio_player.service.stop()
        self.audio_player._apply_playback_state("stopped")
        self.audio_player._draw_waveform()

    def shutdown(self) -> None:
        """Release workers and audio without destroying the containing window."""
        if self._shutdown:
            return
        self._shutdown = True
        self._scan_generation += 1
        self._preview_generation += 1
        self.preview_panel.shutdown()
        self.audio_player.shutdown()

    def _close(self) -> None:
        # Embedded panels must never destroy their host.
        self.on_tab_hidden()

    def _delete_preset(self) -> None:
        parent = self._dialog_parent()
        name = self.preset_menu.get()
        if name == "Default":
            messagebox.showinfo(
                "Delete template",
                "The Default template can’t be deleted.",
                parent=parent,
            )
            return
        path = PRESETS_DIR / f"{name}.json"
        if not path.exists():
            messagebox.showwarning(
                "Delete template",
                "Template file not found.",
                parent=parent,
            )
            return
        if not messagebox.askyesno(
            "Delete template",
            f"Delete template '{name}'?",
            parent=parent,
        ):
            return
        try:
            path.unlink()
        except Exception as exc:
            messagebox.showerror("Delete template", str(exc), parent=parent)
            return
        self.preset_menu.configure(values=self._preset_names())
        self.preset_menu.set("Default")
        self._load_preset("Default")

    def _on_scan_done(self, path: Path, tracks, error: Exception | None) -> None:
        parent = self._dialog_parent()
        if error:
            self._set_busy(False)
            messagebox.showerror(
                "Error scanning folder",
                str(error),
                parent=parent,
            )
            return

        self.tracks = tracks
        self.folder_path = path
        self.demo_mode = False
        self.source_label.configure(text=str(path))

        if not tracks:
            self._set_busy(False)
            self.preview_panel.set_rows([])
            messagebox.showwarning(
                "No files found",
                "No audio/MIDI files found in this folder.\n\n"
                "Supported: wav, mp3, aiff, flac, ogg, m4a, mid, midi",
                parent=parent,
            )
            self._update_footer()
            return
        self._apply_preview()

    def _open_folder(self) -> None:
        if self._busy:
            return
        path = filedialog.askdirectory(
            title="Select folder to scan",
            parent=self._dialog_parent(),
        )
        if path:
            self._scan_folder(Path(path))

    def _apply_renames(self) -> None:
        if self._busy:
            return
        parent = self._dialog_parent()
        renames = self.preview_panel.selected_renames()
        if not renames:
            messagebox.showinfo(
                "Nothing to rename",
                "No selected files will change.",
                parent=parent,
            )
            return
        if self.demo_mode:
            messagebox.showinfo(
                "Demo mode",
                f"{len(renames)} files would be renamed.\n\n"
                "Open a folder to rename real files on disk.",
                parent=parent,
            )
            return
        if not messagebox.askyesno(
            "Confirm rename",
            f"Rename {len(renames)} file(s) on disk?\n\n"
            "This cannot be undone automatically.",
            parent=parent,
        ):
            return

        self.preview_panel.clear_active()
        self.audio_player.reset()
        self._destructive_busy = True
        self._set_busy(True, "Renaming files…")

        def work() -> None:
            success, errors, renamed_paths = apply_file_renames_detailed(renames)
            self.after(
                0,
                lambda: self._on_rename_done(success, errors, renamed_paths),
            )

        threading.Thread(target=work, daemon=True).start()

    def _on_rename_done(
        self,
        success: int,
        errors: list[str],
        renamed_paths: list[Path],
    ) -> None:
        self._destructive_busy = False
        root = self.folder_path
        if not root or not renamed_paths:
            self._show_file_operation_result(
                "Rename completed",
                f"Renamed {success} file(s).",
                errors,
            )
            self._finish_file_operation()
            return

        error_note = (
            f"\n{len(errors)} file(s) could not be renamed." if errors else ""
        )
        organize = messagebox.askyesno(
            "Organize by prefix",
            f"Renamed {success} file(s).{error_note}\n\n"
            "Move the renamed files into folders based on their prefix?\n\n"
            "After selecting Yes, choose the parent destination folder.\n"
            "BASS, DRUMS, VOCALS, and other prefix folders will be created inside it.\n\n"
            "Existing filename conflicts will receive _1, _2, and so on.",
            parent=self._dialog_parent(),
        )
        if not organize:
            if errors:
                self._show_file_operation_result(
                    "Rename completed with errors",
                    f"Renamed {success} file(s).",
                    errors,
                )
            self._finish_file_operation()
            return

        destination = filedialog.askdirectory(
            title="Select destination for prefix folders",
            initialdir=str(root),
            mustexist=True,
            parent=self._dialog_parent(),
        )
        if not destination:
            self._finish_file_operation()
            return
        destination_root = Path(destination)
        self._destructive_busy = True
        self._set_busy(True, "Organizing renamed files…")

        def work() -> None:
            moved, skipped, move_errors = move_files_to_prefix_folders(
                renamed_paths,
                destination_root,
            )
            self.after(
                0,
                lambda: self._on_organize_done(
                    success,
                    errors,
                    moved,
                    skipped,
                    move_errors,
                ),
            )

        threading.Thread(target=work, daemon=True).start()

    def _show_file_operation_result(
        self,
        title: str,
        summary: str,
        errors: list[str],
    ) -> None:
        parent = self._dialog_parent()
        if errors:
            details = "\n".join(errors[:10])
            if len(errors) > 10:
                details += f"\n…and {len(errors) - 10} more."
            messagebox.showwarning(
                title,
                f"{summary}\n\n{details}",
                parent=parent,
            )
        else:
            messagebox.showinfo(title, summary, parent=parent)

    def _finish_file_operation(self) -> None:
        self._destructive_busy = False
        if self.folder_path:
            self._scan_folder(self.folder_path)
        else:
            self._set_busy(False)

    def _save_preset(self) -> None:
        parent = self._dialog_parent()
        name = simpledialog.askstring(
            "Save preset",
            "Preset name:",
            parent=parent,
        )
        if not name:
            return
        data = {"rules": [rule_to_dict(rule) for rule in self.rules]}
        path = PRESETS_DIR / f"{name}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.preset_menu.configure(values=self._preset_names())
        self.preset_menu.set(name)
        messagebox.showinfo(
            "Saved",
            f"Preset saved as {name}",
            parent=parent,
        )

    def _load_preset(self, name: str) -> None:
        if name == "Default":
            self.rules = make_default_rules()
        else:
            path = PRESETS_DIR / f"{name}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self.rules = [
                    rule_from_dict(rule)
                    for rule in data.get("rules", [])
                ]
        normalize_rules_category_colors(self.rules)
        self.rules_panel.set_rules(self.rules)
        self._apply_preview()

    def _show_help(self) -> None:
        show_rename_help_dialog(self._dialog_parent(), self.theme)


# Reuse the tested standalone controller methods that are window-agnostic. Panel
# overrides above replace every operation that needs embedded lifecycle/dialog
# behavior, while the original package remains a faithful vendored runtime.
for _method_name, _method in TrackRenamerApp.__dict__.items():
    if (
        _method_name.startswith("__")
        or _method_name == "main"
        or hasattr(TrackRenamerPanel, _method_name)
    ):
        continue
    setattr(TrackRenamerPanel, _method_name, _method)

del _method_name, _method
