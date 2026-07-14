"""Main application window."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from track_renamer.category_palette import (
    applied_category_colors,
    normalize_rules_category_colors,
    sort_rule_category_keywords,
)
from track_renamer.engine.defaults import make_default_rules, make_demo_tracks
from track_renamer.engine.models import Rule, rule_from_dict, rule_to_dict
from track_renamer.folder_scanner import (
    apply_file_renames_detailed,
    move_files_to_prefix_folders,
    scan_folder,
)
from track_renamer.gui.audio_player import AudioPlayerBar
from track_renamer.gui.help_dialog import show_rename_help_dialog
from track_renamer.gui.preview_panel import PreviewPanel
from track_renamer.gui.rules_panel import RulesPanel
from track_renamer.gui.theme import DARK
from track_renamer.gui.tips import TIPS
from track_renamer.gui.tooltip import bind_tooltip

PRESETS_DIR = Path.home() / ".track_renamer" / "presets"
class TrackRenamerApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sample Renamer")
        self.geometry("1280x820")
        self.minsize(1000, 700)

        self.dark_mode = True
        self.theme = DARK.copy()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

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

        PRESETS_DIR.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._apply_preview()
        self._update_footer()

    def _tip(self, widget, key: str) -> None:
        bind_tooltip(widget, TIPS[key], self.theme)

    def _build_ui(self) -> None:
        t = self.theme
        self.configure(fg_color=t["bg"])

        header = ctk.CTkFrame(self, fg_color=t["bg"], height=72)
        header.pack(fill="x", padx=20, pady=(16, 8))
        header.pack_propagate(False)
        self.header_frame = header

        title_block = ctk.CTkFrame(header, fg_color="transparent")
        title_block.pack(side="left", fill="y")
        title_label = ctk.CTkLabel(
            title_block,
            text="Rename Files",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=t["text"],
        )
        title_label.pack(anchor="w")
        self.title_label = title_label

        subtitle = ctk.CTkLabel(
            title_block,
            text="Scan a folder, build rules, and preview filename changes before applying.",
            font=ctk.CTkFont(size=12),
            text_color=t["text_dim"],
        )
        subtitle.pack(anchor="w")
        self.subtitle_label = subtitle

        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.pack(side="right")

        self.preset_menu = ctk.CTkOptionMenu(
            actions,
            values=self._preset_names(),
            command=self._load_preset,
            width=140,
            fg_color=t["btn"],
            button_color=t["border"],
            button_hover_color=t["accent"],
            dropdown_fg_color=t["panel_2"],
            text_color=t["text"],
        )
        self.preset_menu.set("Default")
        self.preset_menu.pack(side="left", padx=4)
        self._tip(self.preset_menu, "preset")

        self.save_btn = ctk.CTkButton(
            actions,
            text="+",
            width=36,
            height=28,
            command=self._save_preset,
            fg_color=t["btn"],
            hover_color=t["btn_hover"],
        )
        self.save_btn.pack(side="left", padx=4)
        self._tip(self.save_btn, "save_preset")

        self.delete_preset_btn = ctk.CTkButton(
            actions,
            text="-",
            width=36,
            height=28,
            command=self._delete_preset,
            fg_color=t["btn"],
            hover_color=t["danger"],
            text_color=t["text"],
        )
        self.delete_preset_btn.pack(side="left", padx=(0, 4))
        self._tip(self.delete_preset_btn, "delete_preset")

        self.open_btn = ctk.CTkButton(
            actions,
            text="Open folder",
            width=100,
            command=self._open_folder,
            fg_color=t["btn"],
            hover_color=t["btn_hover"],
            text_color=t["text"],
        )
        self.open_btn.pack(side="left", padx=4)
        self._tip(self.open_btn, "open_folder")

        self.help_btn = ctk.CTkButton(
            actions,
            text="Help",
            width=60,
            command=self._show_help,
            fg_color=t["btn"],
            hover_color=t["btn_hover"],
            text_color=t["text"],
        )
        self.help_btn.pack(side="left", padx=4)
        self._tip(self.help_btn, "help")

        options = ctk.CTkFrame(self, fg_color=t["bg"], height=36)
        options.pack(fill="x", padx=20, pady=(0, 4))
        options.pack_propagate(False)
        self.options_frame = options

        self.recursive_var = ctk.BooleanVar(value=True)
        self.recursive_cb = ctk.CTkCheckBox(
            options,
            text="Include subfolders",
            variable=self.recursive_var,
            command=self._on_recursive_toggle,
            font=ctk.CTkFont(size=12),
            text_color=t["text_dim"],
            fg_color=t["accent"],
            hover_color=t["accent_hover"],
        )
        self.recursive_cb.pack(side="left")
        self._tip(self.recursive_cb, "recursive")

        self.source_label = ctk.CTkLabel(
            options,
            text="Demo files (open a folder to scan real files)",
            font=ctk.CTkFont(size=12),
            text_color=t["text_mute"],
        )
        self.source_label.pack(side="left", padx=(16, 0))

        body = ctk.CTkFrame(self, fg_color=t["bg"])
        body.pack(fill="both", expand=True, padx=20, pady=8)
        self.body_frame = body
        body.grid_columnconfigure(0, weight=1, uniform="col")
        body.grid_columnconfigure(1, weight=1, uniform="col")
        body.grid_rowconfigure(0, weight=1)

        self.rules_panel = RulesPanel(
            body, theme=t, on_change=self._on_rules_changed, on_apply=self._apply_preview
        )
        self.rules_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.rules_panel.set_rules(self.rules)

        self.preview_panel = PreviewPanel(
            body,
            theme=t,
            on_change=self._update_footer,
            on_active=self._on_active_preview,
            on_play_pause=self._toggle_audio_preview,
            on_seek=self._seek_audio_preview,
        )
        self.preview_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        footer = ctk.CTkFrame(self, fg_color=t["bg"], height=50)
        footer.pack(fill="x", padx=20, pady=(8, 16))
        footer.pack_propagate(False)
        self.footer_frame = footer

        self.file_count_label = ctk.CTkLabel(
            footer,
            text=f"{len(self.tracks)} files",
            font=ctk.CTkFont(size=13),
            text_color=t["text_dim"],
        )
        self.file_count_label.pack(side="left")

        self.audio_player = AudioPlayerBar(footer, theme=t)
        self.audio_player.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(28, 24),
        )

        btn_row = ctk.CTkFrame(footer, fg_color="transparent")
        btn_row.pack(side="right")

        self.cancel_btn = ctk.CTkButton(
            btn_row,
            text="Cancel",
            width=100,
            height=36,
            fg_color=t["btn"],
            hover_color=t["btn_hover"],
            text_color=t["text"],
            command=self._close,
        )
        self.cancel_btn.pack(side="left", padx=(0, 8))
        self._tip(self.cancel_btn, "cancel")

        self.rename_btn = ctk.CTkButton(
            btn_row,
            text="Rename 0",
            width=120,
            height=36,
            fg_color=t["accent"],
            hover_color=t["accent_hover"],
            command=self._apply_renames,
        )
        self.rename_btn.pack(side="left")
        self._tip(self.rename_btn, "rename")

    def _preset_names(self) -> list[str]:
        names = ["Default"]
        if PRESETS_DIR.exists():
            names.extend(sorted(p.stem for p in PRESETS_DIR.glob("*.json")))
        return names

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.open_btn.configure(state=state)
        if message:
            self.source_label.configure(text=message)
        elif self.folder_path:
            self.source_label.configure(text=str(self.folder_path))
        elif self.demo_mode:
            self.source_label.configure(text="Demo files (open a folder to scan real files)")
        if hasattr(self, "preview_panel"):
            self._update_footer()

    def _on_active_preview(
        self,
        track,
        row,
    ) -> None:
        self.audio_player.set_active(track, row)

    def _toggle_audio_preview(self) -> None:
        self.audio_player.toggle_playback()

    def _seek_audio_preview(self, seconds: float) -> None:
        self.audio_player.seek(seconds)

    @staticmethod
    def _rules_fingerprint(rules: list[Rule]) -> str:
        return json.dumps(
            [rule_to_dict(rule) for rule in rules],
            sort_keys=True,
            separators=(",", ":"),
        )

    def _on_rules_changed(self) -> None:
        self.rules = self.rules_panel.get_rules()
        pending = (
            self._rules_fingerprint(self.rules)
            != self._applied_rules_fingerprint
        )
        if pending == self._preview_stale:
            return
        self._preview_stale = pending
        if pending:
            self.preview_panel.cancel_preview_work()
        self._set_preview_pending(pending)
        self._update_footer()

    def _set_preview_pending(self, pending: bool) -> None:
        self.rules_panel.set_apply_pending(pending)
        self.preview_panel.set_preview_pending(pending)

    def _apply_preview(self) -> None:
        self.rules = self.rules_panel.get_rules()
        if sort_rule_category_keywords(self.rules):
            self.rules_panel.set_rules(self.rules)
        self._applied_rules_fingerprint = self._rules_fingerprint(self.rules)
        self._preview_stale = False
        self._set_preview_pending(False)
        self._refresh_preview()

    def _on_recursive_toggle(self) -> None:
        self.recursive = self.recursive_var.get()
        if self.folder_path:
            self._scan_folder(self.folder_path)

    def _refresh_preview(self) -> None:
        self._preview_generation += 1
        tracks = self.tracks
        rules = self.rules
        self.audio_player.set_category_colors(applied_category_colors(rules))
        root_label = self.folder_path.name if self.folder_path else "ROOT"
        self.preview_panel.begin_viewport_lazy(tracks, rules, root_label)
        self._set_busy(False, f"Preparing preview ({len(tracks):,} files)")
        self._update_footer()

    def _update_footer(self) -> None:
        count = self.preview_panel.rename_count()
        complete = self.preview_panel.lazy_compute_complete()
        if complete:
            self.rename_btn.configure(text=f"Rename {count:,}")
        else:
            done, total = self.preview_panel.lazy_compute_progress()
            self.rename_btn.configure(text=f"Preparing {done:,}/{total:,}")
        can_rename = complete and count > 0 and not self._busy and not self._preview_stale
        self.rename_btn.configure(state="normal" if can_rename else "disabled")
        self.file_count_label.configure(text=f"{len(self.tracks):,} files")
        if complete and not self._busy:
            if self.folder_path:
                self.source_label.configure(text=str(self.folder_path))
            elif self.demo_mode:
                self.source_label.configure(
                    text="Demo files (open a folder to scan real files)"
                )

    def _delete_preset(self) -> None:
        name = self.preset_menu.get()
        if name == "Default":
            messagebox.showinfo("Delete template", "The Default template can’t be deleted.")
            return
        path = PRESETS_DIR / f"{name}.json"
        if not path.exists():
            messagebox.showwarning("Delete template", "Template file not found.")
            return
        if not messagebox.askyesno("Delete template", f"Delete template '{name}'?"):
            return
        try:
            path.unlink()
        except Exception as exc:
            messagebox.showerror("Delete template", str(exc))
            return
        self.preset_menu.configure(values=self._preset_names())
        self.preset_menu.set("Default")
        self._load_preset("Default")

    def _apply_theme(self) -> None:
        t = self.theme
        self.configure(fg_color=t["bg"])
        self.header_frame.configure(fg_color=t["bg"])
        self.options_frame.configure(fg_color=t["bg"])
        self.body_frame.configure(fg_color=t["bg"])
        self.footer_frame.configure(fg_color=t["bg"])

        self.title_label.configure(text_color=t["text"])
        self.subtitle_label.configure(text_color=t["text_dim"])
        self.source_label.configure(text_color=t["text_mute"])
        self.file_count_label.configure(text_color=t["text_dim"])

        for widget in (
            self.open_btn,
            self.help_btn,
            self.cancel_btn,
        ):
            widget.configure(
                fg_color=t["btn"],
                hover_color=t["btn_hover"],
                text_color=t["text"],
            )
        self.save_btn.configure(fg_color=t["btn"], hover_color=t["btn_hover"])
        self.delete_preset_btn.configure(fg_color=t["btn"], hover_color=t["danger"], text_color=t["text"])
        self.preset_menu.configure(
            fg_color=t["btn"],
            text_color=t["text"],
            button_color=t["border"],
            button_hover_color=t["accent"],
            dropdown_fg_color=t["panel_2"],
        )
        self.rename_btn.configure(
            fg_color=t["accent"],
            hover_color=t["accent_hover"],
            text_color="#ffffff",
        )
        self.recursive_cb.configure(
            text_color=t["text_dim"],
            fg_color=t["accent"],
            hover_color=t["accent_hover"],
        )

        self.rules_panel.set_theme(t)
        self.preview_panel.set_theme(t)
        self.audio_player.set_theme(t)
        self._set_preview_pending(self._preview_stale)

    def _scan_folder(self, path: Path) -> None:
        self.preview_panel.clear_active()
        self.audio_player.reset()
        self._scan_generation += 1
        generation = self._scan_generation
        self._set_busy(True, f"Scanning {path.name}…")
        self.preview_panel.set_loading(True)
        recursive = self.recursive

        def work() -> None:
            def progress(count: int) -> None:
                if generation != self._scan_generation:
                    return
                self.after(0, lambda c=count: self._set_busy(True, f"Scanning… {c:,} files found"))

            try:
                tracks = scan_folder(path, recursive=recursive, progress=progress)
                error = None
            except Exception as exc:
                tracks = []
                error = exc

            if generation != self._scan_generation:
                return
            self.after(0, lambda: self._on_scan_done(path, tracks, error))

        threading.Thread(target=work, daemon=True).start()

    def _on_scan_done(self, path: Path, tracks, error: Exception | None) -> None:
        if error:
            self._set_busy(False)
            messagebox.showerror("Error scanning folder", str(error))
            return

        self.tracks = tracks
        self.folder_path = path
        self.demo_mode = False
        self.title(f"Sample Renamer — {path.name}")
        self.source_label.configure(text=str(path))

        if not tracks:
            self._set_busy(False)
            self.preview_panel.set_rows([])
            messagebox.showwarning(
                "No files found",
                "No audio/MIDI files found in this folder.\n\n"
                "Supported: wav, mp3, aiff, flac, ogg, m4a, mid, midi",
            )
            self._update_footer()
            return

        self._apply_preview()

    def _open_folder(self) -> None:
        if self._busy:
            return
        path = filedialog.askdirectory(title="Select folder to scan")
        if path:
            self._scan_folder(Path(path))

    def _apply_renames(self) -> None:
        if self._busy:
            return
        renames = self.preview_panel.selected_renames()
        if not renames:
            messagebox.showinfo("Nothing to rename", "No selected files will change.")
            return

        if self.demo_mode:
            messagebox.showinfo(
                "Demo mode",
                f"{len(renames)} files would be renamed.\n\nOpen a folder to rename real files on disk.",
            )
            return

        if not messagebox.askyesno(
            "Confirm rename",
            f"Rename {len(renames)} file(s) on disk?\n\nThis cannot be undone automatically.",
        ):
            return

        self.preview_panel.clear_active()
        self.audio_player.reset()
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
            f"\n{len(errors)} file(s) could not be renamed."
            if errors
            else ""
        )
        organize = messagebox.askyesno(
            "Organize by prefix",
            f"Renamed {success} file(s).{error_note}\n\n"
            "Move the renamed files into folders based on their prefix?\n\n"
            "After selecting Yes, choose the parent destination folder.\n"
            "BASS, DRUMS, VOCALS, and other prefix folders will be created inside it.\n\n"
            "Existing filename conflicts will receive _1, _2, and so on.",
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
        )
        if not destination:
            self._finish_file_operation()
            return
        destination_root = Path(destination)

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

    def _on_organize_done(
        self,
        renamed: int,
        rename_errors: list[str],
        moved: int,
        skipped: int,
        move_errors: list[str],
    ) -> None:
        summary = f"Renamed {renamed} file(s).\nMoved {moved} file(s)."
        if skipped:
            summary += f"\nSkipped {skipped} file(s) without a PREFIX - name."
        self._show_file_operation_result(
            "Organization completed",
            summary,
            rename_errors + move_errors,
        )
        self._finish_file_operation()

    def _show_file_operation_result(
        self,
        title: str,
        summary: str,
        errors: list[str],
    ) -> None:
        if errors:
            details = "\n".join(errors[:10])
            if len(errors) > 10:
                details += f"\n…and {len(errors) - 10} more."
            messagebox.showwarning(title, f"{summary}\n\n{details}")
        else:
            messagebox.showinfo(title, summary)

    def _finish_file_operation(self) -> None:
        if self.folder_path:
            self._scan_folder(self.folder_path)
        else:
            self._set_busy(False)

    def _save_preset(self) -> None:
        from tkinter import simpledialog

        name = simpledialog.askstring("Save preset", "Preset name:")
        if not name:
            return
        data = {"rules": [rule_to_dict(r) for r in self.rules]}
        path = PRESETS_DIR / f"{name}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.preset_menu.configure(values=self._preset_names())
        self.preset_menu.set(name)
        messagebox.showinfo("Saved", f"Preset saved as {name}")

    def _load_preset(self, name: str) -> None:
        if name == "Default":
            self.rules = make_default_rules()
        else:
            path = PRESETS_DIR / f"{name}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self.rules = [rule_from_dict(r) for r in data.get("rules", [])]
        normalize_rules_category_colors(self.rules)
        self.rules_panel.set_rules(self.rules)
        self._apply_preview()

    def _show_help(self) -> None:
        show_rename_help_dialog(self, self.theme)

    def _close(self) -> None:
        self.preview_panel.shutdown()
        self.audio_player.shutdown()
        self.destroy()


def main() -> None:
    app = TrackRenamerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
