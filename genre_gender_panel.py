from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ffmpeg_bootstrap import subprocess_kwargs
from ui_theme import (
    ACTIONS_BOTTOM_PAD,
    ACTION_BTN_FONT,
    ACTION_BTN_PADX,
    ACTION_BTN_PADY,
    COLORS,
    CTRL_ROW_PADY,
    HEADER_DESC_FONT,
    HEADER_TOP_PAD,
    PATH_BTN_FONT,
    PATH_BTN_PADX,
    PATH_BTN_PADY,
    SECTION_GAP,
    SECTION_INNER_PAD,
    SECTION_PADX,
    Tooltip,
    display_path,
    tip,
)

APP_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
SETTINGS_PATH = APP_DIR / "settings.json"
PAIR_LOG_TAG = "__pair_log__"
PANEL_TITLE = "Genre & Gender"
# Bundled beside the app (source tree or next to STEM-organizer.exe).
TAGGER_DIR = APP_DIR / "genre_gender_tagger"
TAGGER_SCRIPT = TAGGER_DIR / "genre_gender_tagger.py"
TAGGER_PYTHON = TAGGER_DIR / "venv" / "Scripts" / "python.exe"

TIPS = {
    "genre_input": "Folder containing instrumental FLAC/MP3/WAV files to tag with genre.",
    "gender_input": "Folder containing acapella FLAC/MP3/WAV files to tag with voice gender.",
    "batch_mode": (
        "Batch: fast on NVIDIA GPU; on CPU/VM it can sit on one long progress "
        "bar for a while (still working). Per-file: recommended for CPU/VM — "
        "slower overall but prints each file as it finishes."
    ),
    "tag_style": (
        "Combined writes a single GENRE tag as 'Genre/Style'. "
        "Split writes separate GENRE and STYLE tags."
    ),
    "tag_field": "Comment writes gender to the COMMENT tag. Gender writes to a GENDER tag.",
    "write_meta": "Write tags to FLAC/MP3/M4A/WAV. Disable to only generate the CSV.",
    "csv_path": "Optional path for the output CSV. Leave empty to use the tagger default.",
    "open_path": "Open this folder in Explorer.",
}


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    try:
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(SETTINGS_PATH)
    except OSError:
        pass


def show_genre_gender_help_dialog(parent: tk.Misc, mode: str) -> None:
    is_genre = mode == "genre"
    title = "Genre help" if is_genre else "Gender help"
    heading = (
        "Tag instrumentals with genre & style"
        if is_genre
        else "Tag acapellas with voice gender"
    )
    intro = (
        "Classify instrumental tracks and write Discogs-style GENRE / STYLE tags."
        if is_genre
        else "Estimate singing voice gender and write it to COMMENT or GENDER tags."
    )
    sections = (
        (
            "Workflow",
            "1. Choose an input folder of instrumental FLAC/MP3/WAV files.\n"
            "2. Pick run mode (Batch or Per-file) and tag style.\n"
            "3. Click ▶ Tag genre and watch progress in the LOG panel.\n"
            "4. Check FLAC tags and/or the CSV export when the run finishes.",
        ),
        (
            "Model",
            "Uses Hugging Face MAEST "
            "mtg-upf/discogs-maest-30s-pw-129e-519l "
            "(Discogs519 genre/style). The model downloads into the Hugging Face "
            "cache on first genre run.",
        ),
        (
            "Options",
            "Batch is faster (best with a GPU). Per-file prints each prediction live. "
            "Combined writes one GENRE tag as Genre/Style; Split writes separate "
            "GENRE and STYLE tags. Tags are written to FLAC, MP3, M4A, and WAV "
            "(ID3 / Vorbis / MP4 atoms as appropriate). A CSV export is always written.",
        ),
        (
            "Setup",
            "Genre & Gender uses the bundled folder genre_gender_tagger\\ with its "
            "own venv. Run genre_gender_tagger\\install-deps.bat once "
            "(or accept it at the end of STEM install-deps.bat).",
        ),
    ) if is_genre else (
        (
            "Workflow",
            "1. Choose an input folder of acapella FLAC/MP3/WAV files.\n"
            "2. Pick run mode and which tag field to write.\n"
            "3. Click ▶ Tag gender and watch progress in the LOG panel.\n"
            "4. Check FLAC tags and/or the CSV export when the run finishes.",
        ),
        (
            "Model",
            "Uses Essentia gender-discogs-effnet "
            "(discogs-effnet-bs64 embeddings + gender classification head) "
            "via TensorFlow .pb models. The .pb files ship in "
            "genre_gender_tagger\\models\\; no download is needed when those "
            "files are present.",
        ),
        (
            "Options",
            "Batch is faster; Per-file prints each GENDER/CONF live. "
            "COMMENT writes e.g. COMMENT=female; GENDER writes a dedicated "
            "GENDER tag. Tags are written to FLAC, MP3, M4A, and WAV "
            "(ID3 / Vorbis / MP4 atoms as appropriate). A CSV export is always written.",
        ),
        (
            "Setup",
            "Genre & Gender uses the bundled folder genre_gender_tagger\\ with its "
            "own venv. Run genre_gender_tagger\\install-deps.bat once "
            "(or accept it at the end of STEM install-deps.bat).",
        ),
    )

    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.configure(bg=COLORS["panel"])
    dialog.resizable(False, False)
    dialog.transient(parent.winfo_toplevel())

    outer = tk.Frame(dialog, bg=COLORS["panel"])
    outer.pack(fill="both", expand=True, padx=22, pady=18)
    tk.Label(
        outer, text=heading, font=("Segoe UI Semibold", 18),
        fg=COLORS["fg"], bg=COLORS["panel"],
    ).pack(anchor="w")
    tk.Label(
        outer, text=intro, font=("Segoe UI", 10),
        fg=COLORS["fg_dim"], bg=COLORS["panel"],
    ).pack(anchor="w", pady=(2, 14))

    for section_title, body in sections:
        card = tk.Frame(
            outer, bg=COLORS["panel2"],
            highlightbackground=COLORS["border"], highlightthickness=1,
        )
        card.pack(fill="x", pady=(0, 10))
        tk.Label(
            card, text=section_title.upper(), font=("Segoe UI Semibold", 9),
            fg=COLORS["accent_hov"], bg=COLORS["panel2"],
        ).pack(anchor="w", padx=14, pady=(10, 4))
        tk.Label(
            card, text=body, font=("Segoe UI", 10),
            fg=COLORS["fg_dim"], bg=COLORS["panel2"],
            justify="left", anchor="w", wraplength=600,
        ).pack(fill="x", padx=14, pady=(0, 11))

    footer = tk.Frame(outer, bg=COLORS["panel"])
    footer.pack(fill="x", pady=(2, 0))
    tk.Label(
        footer,
        text="Hover over individual controls for more detail.",
        font=("Segoe UI", 9), fg=COLORS["fg_dim"], bg=COLORS["panel"],
    ).pack(side="left")
    close = tk.Button(
        footer, text="Close", command=dialog.destroy,
        font=("Segoe UI Semibold", 10),
        bg=COLORS["accent"], fg="#ffffff",
        activebackground=COLORS["accent_hov"], activeforeground="#ffffff",
        relief="flat", borderwidth=0, highlightthickness=0,
        padx=18, pady=5, cursor="hand2",
    )
    close.pack(side="right")

    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.update_idletasks()
    width = 680
    height = max(520, outer.winfo_reqheight() + 36)
    top = parent.winfo_toplevel()
    x = top.winfo_rootx() + max(0, (top.winfo_width() - width) // 2)
    y = top.winfo_rooty() + max(0, (top.winfo_height() - height) // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")
    dialog.grab_set()
    close.focus_set()


class GenreGenderPanel(ttk.Frame):
    def __init__(
        self,
        host: tk.Misc,
        parent: tk.Misc,
        info_icon_factory=None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._info_icon_factory = info_icon_factory
        self._busy = False
        self._stop_requested = False
        self._proc: subprocess.Popen | None = None
        self._worker: threading.Thread | None = None

        # ---- Genre tab vars ----
        self.genre_input_dir = tk.StringVar()
        self.genre_batch_mode = tk.BooleanVar(value=True)
        self.genre_tag_style = tk.StringVar(value="combined")
        self.genre_write_meta = tk.BooleanVar(value=True)
        self.genre_csv_path = tk.StringVar()

        # ---- Gender tab vars ----
        self.gender_input_dir = tk.StringVar()
        self.gender_batch_mode = tk.BooleanVar(value=True)
        self.gender_tag_field = tk.StringVar(value="comment")
        self.gender_write_meta = tk.BooleanVar(value=True)
        self.gender_csv_path = tk.StringVar()

        self._build_ui()
        self._bind_settings_autosave()
        self._load_settings()

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self, style="Sub.TNotebook", takefocus=0)
        notebook.grid(row=0, column=0, sticky="nsew")
        self._notebook = notebook

        genre_tab = ttk.Frame(notebook)
        gender_tab = ttk.Frame(notebook)
        notebook.add(genre_tab, text="  Genre  ")
        notebook.add(gender_tab, text="  Gender  ")
        notebook.bind("<<NotebookTabChanged>>", self._on_subtab_changed)

        self._build_genre_tab(genre_tab)
        self._build_gender_tab(gender_tab)

    def _make_info_icon(self, parent: tk.Misc, command) -> None:
        if self._info_icon_factory is not None:
            self._info_icon_factory(parent, command).pack(side="left", padx=(4, 0))
            return
        fallback = tk.Label(
            parent, text="?", font=("Segoe UI Semibold", 9),
            fg=COLORS["fg_dim"], bg=COLORS["bg"], cursor="hand2",
        )
        fallback.pack(side="left", padx=(5, 0))
        fallback.bind("<Button-1>", lambda _event: command())
        Tooltip(fallback, "Show more info/help.")

    def _show_genre_help(self) -> None:
        show_genre_gender_help_dialog(self._host, "genre")

    def _show_gender_help(self) -> None:
        show_genre_gender_help_dialog(self._host, "gender")

    def _description_with_info(
        self,
        parent: tk.Misc,
        text: str,
        command,
    ) -> None:
        desc_row = tk.Frame(parent, bg=COLORS["bg"])
        desc_row.pack(fill="x", anchor="w")
        tk.Label(
            desc_row,
            text=text,
            font=HEADER_DESC_FONT,
            fg=COLORS["fg"],
            bg=COLORS["bg"],
            wraplength=470,
            justify="left",
        ).pack(side="left")
        self._make_info_icon(desc_row, command)

    def _build_genre_tab(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill="x", padx=SECTION_PADX, pady=(HEADER_TOP_PAD, 12))
        self._description_with_info(
            header,
            "Tag instrumental FLAC/MP3 files with Discogs genre and style.",
            self._show_genre_help,
        )

        paths = ttk.LabelFrame(parent, text="  PATHS  ", padding=SECTION_INNER_PAD)
        paths.pack(fill="x", padx=SECTION_PADX, pady=(0, SECTION_GAP))
        paths.columnconfigure(1, weight=1)
        self._path_row(
            paths, 0,
            "Input folder", self.genre_input_dir,
            lambda: self._pick_dir(self.genre_input_dir, "Select input folder"),
            lambda: self._open_folder(self.genre_input_dir),
            TIPS["genre_input"],
        )

        opts = ttk.LabelFrame(parent, text="  OPTIONS  ", padding=SECTION_INNER_PAD)
        opts.pack(fill="x", padx=SECTION_PADX, pady=(0, SECTION_GAP))

        mode_row = ttk.Frame(opts)
        mode_row.pack(anchor="w", pady=(0, 6))
        mode_lbl = ttk.Label(mode_row, text="Run mode")
        mode_lbl.pack(side="left", padx=(0, 12))
        batch_rb = ttk.Radiobutton(
            mode_row, text="Batch (fast)",
            variable=self.genre_batch_mode, value=True, cursor="hand2",
        )
        batch_rb.pack(side="left", padx=(0, 10))
        perfile_rb = ttk.Radiobutton(
            mode_row, text="Per-file (live results)",
            variable=self.genre_batch_mode, value=False, cursor="hand2",
        )
        perfile_rb.pack(side="left")
        tip(mode_lbl, batch_rb, perfile_rb, text=TIPS["batch_mode"])

        style_row = ttk.Frame(opts)
        style_row.pack(anchor="w", pady=(0, 6))
        style_lbl = ttk.Label(style_row, text="Tag style")
        style_lbl.pack(side="left", padx=(0, 12))
        combined_rb = ttk.Radiobutton(
            style_row, text="Combined  (GENRE=Rock/Surf)",
            variable=self.genre_tag_style, value="combined", cursor="hand2",
        )
        combined_rb.pack(side="left", padx=(0, 10))
        split_rb = ttk.Radiobutton(
            style_row, text="Split  (GENRE=Rock, STYLE=Surf)",
            variable=self.genre_tag_style, value="split", cursor="hand2",
        )
        split_rb.pack(side="left")
        tip(style_lbl, combined_rb, split_rb, text=TIPS["tag_style"])

        meta_chk = ttk.Checkbutton(
            opts, text="Write metadata to files",
            variable=self.genre_write_meta, cursor="hand2",
        )
        meta_chk.pack(anchor="w", pady=(0, 4))
        tip(meta_chk, text=TIPS["write_meta"])

        tk.Frame(parent, bg=COLORS["bg"], height=ACTIONS_BOTTOM_PAD).pack(side="bottom")

    def _build_gender_tab(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill="x", padx=SECTION_PADX, pady=(HEADER_TOP_PAD, 12))
        self._description_with_info(
            header,
            "Tag acapella FLAC/MP3 files with singing voice gender (female / male).",
            self._show_gender_help,
        )

        paths = ttk.LabelFrame(parent, text="  PATHS  ", padding=SECTION_INNER_PAD)
        paths.pack(fill="x", padx=SECTION_PADX, pady=(0, SECTION_GAP))
        paths.columnconfigure(1, weight=1)
        self._path_row(
            paths, 0,
            "Input folder", self.gender_input_dir,
            lambda: self._pick_dir(self.gender_input_dir, "Select input folder"),
            lambda: self._open_folder(self.gender_input_dir),
            TIPS["gender_input"],
        )

        opts = ttk.LabelFrame(parent, text="  OPTIONS  ", padding=SECTION_INNER_PAD)
        opts.pack(fill="x", padx=SECTION_PADX, pady=(0, SECTION_GAP))

        mode_row = ttk.Frame(opts)
        mode_row.pack(anchor="w", pady=(0, 6))
        mode_lbl = ttk.Label(mode_row, text="Run mode")
        mode_lbl.pack(side="left", padx=(0, 12))
        batch_rb = ttk.Radiobutton(
            mode_row, text="Batch (fast)",
            variable=self.gender_batch_mode, value=True, cursor="hand2",
        )
        batch_rb.pack(side="left", padx=(0, 10))
        perfile_rb = ttk.Radiobutton(
            mode_row, text="Per-file (live results)",
            variable=self.gender_batch_mode, value=False, cursor="hand2",
        )
        perfile_rb.pack(side="left")
        tip(mode_lbl, batch_rb, perfile_rb, text=TIPS["batch_mode"])

        field_row = ttk.Frame(opts)
        field_row.pack(anchor="w", pady=(0, 6))
        field_lbl = ttk.Label(field_row, text="Tag field")
        field_lbl.pack(side="left", padx=(0, 12))
        comment_rb = ttk.Radiobutton(
            field_row, text="COMMENT  (COMMENT=female)",
            variable=self.gender_tag_field, value="comment", cursor="hand2",
        )
        comment_rb.pack(side="left", padx=(0, 10))
        gender_rb = ttk.Radiobutton(
            field_row, text="GENDER  (GENDER=female)",
            variable=self.gender_tag_field, value="gender", cursor="hand2",
        )
        gender_rb.pack(side="left")
        tip(field_lbl, comment_rb, gender_rb, text=TIPS["tag_field"])

        meta_chk = ttk.Checkbutton(
            opts, text="Write metadata to files",
            variable=self.gender_write_meta, cursor="hand2",
        )
        meta_chk.pack(anchor="w", pady=(0, 4))
        tip(meta_chk, text=TIPS["write_meta"])

        tk.Frame(parent, bg=COLORS["bg"], height=ACTIONS_BOTTOM_PAD).pack(side="bottom")

    # ------------------------------------------------------------------
    # Action bar
    # ------------------------------------------------------------------

    def attach_action_bar(self, parent: tk.Misc) -> None:
        # Same pattern as Classify: accent Start + dim ■ Stop beside it.
        self.genre_btn = self._action_button(
            parent, "▶  Tag genre", self._start_genre, accent=True,
        )
        self.gender_btn = self._action_button(
            parent, "▶  Tag gender", self._start_gender, accent=True,
        )
        C = COLORS
        self.stop_btn = tk.Button(
            parent,
            text="■  Stop",
            command=self._request_stop,
            font=ACTION_BTN_FONT,
            bg=C["panel2"],
            fg=C["fg_dim"],
            activebackground=C["panel"],
            activeforeground=C["danger"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=ACTION_BTN_PADX,
            pady=ACTION_BTN_PADY,
            cursor="arrow",
        )
        tip(self.stop_btn, text="Stop the running tagger.")

    def _genre_tab_active(self) -> bool:
        return self._notebook.index(self._notebook.select()) == 0

    def _on_subtab_changed(self, _event=None) -> None:
        host = self._host
        if hasattr(host, "_gg_mode_active") and not host._gg_mode_active():
            return
        self.show_action_bar()
        if not self._busy:
            self.set_buttons_state("normal")

    def show_action_bar(self) -> None:
        self.genre_btn.pack_forget()
        self.gender_btn.pack_forget()
        self.stop_btn.pack_forget()
        if self._genre_tab_active():
            self.genre_btn.pack(side="left")
        else:
            self.gender_btn.pack(side="left")
        self.stop_btn.pack(side="left", padx=(8, 0))
        self._refresh_run_buttons()

    def hide_action_bar(self) -> None:
        self.genre_btn.pack_forget()
        self.gender_btn.pack_forget()
        self.stop_btn.pack_forget()

    def set_buttons_state(self, state: str) -> None:
        # Match Classify: only swap colors/cursors — never state=disabled
        # (avoids button size jumps on Windows).
        if self._busy:
            self._refresh_run_buttons(running=True)
        elif state == "disabled":
            # Another mode is busy — dim Tag + Stop.
            self.genre_btn.configure(fg=COLORS["fg_dim"], cursor="arrow")
            self.gender_btn.configure(fg=COLORS["fg_dim"], cursor="arrow")
            self.stop_btn.configure(fg=COLORS["fg_dim"], cursor="arrow")
        else:
            self._refresh_run_buttons(running=False)

    def _refresh_run_buttons(self, *, running: bool | None = None) -> None:
        if running is None:
            running = self._busy
        if running:
            self.genre_btn.configure(fg=COLORS["fg_dim"], cursor="arrow")
            self.gender_btn.configure(fg=COLORS["fg_dim"], cursor="arrow")
            self.stop_btn.configure(fg=COLORS["danger"], cursor="hand2")
        else:
            self.genre_btn.configure(fg="white", cursor="hand2")
            self.gender_btn.configure(fg="white", cursor="hand2")
            self.stop_btn.configure(fg=COLORS["fg_dim"], cursor="arrow")

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def _start_genre(self) -> None:
        if self._busy or self._host._organize_worker_active():
            return
        input_dir = self.genre_input_dir.get().strip()
        if not input_dir or not Path(input_dir).is_dir():
            messagebox.showerror(PANEL_TITLE, "Input folder is missing or invalid.")
            return
        self._clear_log()
        self._log(f"Starting genre tagger on: {input_dir}", "info")
        self._stop_requested = False
        self._set_busy(True, "Tagging genre…")
        self._worker = threading.Thread(
            target=self._run_tagger,
            args=(
                "genre",
                input_dir,
                bool(self.genre_batch_mode.get()),
                self.genre_tag_style.get(),
                "comment",
                bool(self.genre_write_meta.get()),
                self.genre_csv_path.get().strip(),
            ),
            daemon=True,
        )
        self._worker.start()

    def _start_gender(self) -> None:
        if self._busy or self._host._organize_worker_active():
            return
        input_dir = self.gender_input_dir.get().strip()
        if not input_dir or not Path(input_dir).is_dir():
            messagebox.showerror(PANEL_TITLE, "Input folder is missing or invalid.")
            return
        self._clear_log()
        self._log(f"Starting gender tagger on: {input_dir}", "info")
        self._stop_requested = False
        self._set_busy(True, "Tagging gender…")
        self._worker = threading.Thread(
            target=self._run_tagger,
            args=(
                "gender",
                input_dir,
                bool(self.gender_batch_mode.get()),
                "combined",
                self.gender_tag_field.get(),
                bool(self.gender_write_meta.get()),
                self.gender_csv_path.get().strip(),
            ),
            daemon=True,
        )
        self._worker.start()

    def _request_stop(self) -> None:
        if not self._busy:
            return
        self._stop_requested = True
        self._log("Stop requested…", "warn")
        self.stop()

    def stop(self) -> None:
        """Kill the running tagger process if any."""
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def _run_tagger(
        self,
        mode: str,
        input_dir: str,
        batch_mode: bool,
        tag_style: str,
        gender_field: str,
        write_meta: bool,
        csv_path: str,
    ) -> None:
        if not TAGGER_SCRIPT.is_file():
            self.after(
                0,
                lambda: messagebox.showerror(
                    PANEL_TITLE,
                    f"Bundled tagger not found:\n{TAGGER_SCRIPT}\n\n"
                    "Expected folder: genre_gender_tagger\\ beside STEM organizer.",
                ),
            )
            self._finish_worker("Failed — tagger missing")
            return

        if not TAGGER_PYTHON.is_file():
            self.after(
                0,
                lambda: messagebox.showerror(
                    PANEL_TITLE,
                    f"Genre & Gender venv not found:\n{TAGGER_PYTHON}\n\n"
                    "Run genre_gender_tagger\\install-deps.bat once "
                    "(or choose Yes when STEM install-deps.bat offers it).",
                ),
            )
            self._finish_worker("Failed — venv missing")
            return

        env = os.environ.copy()
        env["GG_MODE"]         = mode
        env["GG_INPUT"]        = input_dir
        env["GG_BATCH"]        = "1" if batch_mode else "0"
        env["GG_TAG_STYLE"]    = tag_style
        env["GG_GENDER_FIELD"] = gender_field
        env["GG_WRITE_META"]   = "1" if write_meta else "0"
        # Hidden console is not a TTY — without this, tqdm/log lines buffer
        # and the UI looks frozen mid-progress (e.g. stuck at Audio 61/100).
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if csv_path:
            env["GG_CSV"] = csv_path

        try:
            proc = subprocess.Popen(
                [str(TAGGER_PYTHON), "-u", str(TAGGER_SCRIPT)],
                env=env,
                cwd=str(TAGGER_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **subprocess_kwargs(),  # hide console window on Windows
            )
            self._proc = proc

            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line:
                    self._host.log_queue.put((PAIR_LOG_TAG, line, "info"))

            proc.wait()
            if self._stop_requested:
                self._host.log_queue.put((PAIR_LOG_TAG, "[tagger stopped]", "warn"))
                self._finish_worker("Stopped")
            elif proc.returncode == 0:
                self._finish_worker("Done")
            else:
                self._host.log_queue.put(
                    (PAIR_LOG_TAG, f"[tagger exited with code {proc.returncode}]", "warn")
                )
                self._finish_worker(f"Failed (exit {proc.returncode})")

        except Exception as exc:
            if self._stop_requested:
                self._host.log_queue.put((PAIR_LOG_TAG, "[tagger stopped]", "warn"))
                self._finish_worker("Stopped")
            else:
                self._host.log_queue.put((PAIR_LOG_TAG, str(exc), "err"))
                self._finish_worker("Failed")
        finally:
            self._proc = None
            self._stop_requested = False

    # ------------------------------------------------------------------
    # Busy / status
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        self._host._set_pair_busy(busy, status, self)

    def _finish_worker(self, status: str) -> None:
        self.after(0, lambda: self._set_busy(False, status))

    # ------------------------------------------------------------------
    # Log helpers (delegated to host)
    # ------------------------------------------------------------------

    def _clear_log(self) -> None:
        self._host._clear_log()

    def _log(self, message: str, tag: str = "info") -> None:
        self._host.log_queue.put((PAIR_LOG_TAG, message, tag))

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _pick_dir(self, var: tk.StringVar, title: str) -> None:
        initial = var.get().strip()
        path = filedialog.askdirectory(title=title, initialdir=initial or None)
        if path:
            var.set(display_path(path))

    def _open_folder(self, var: tk.StringVar) -> None:
        path = Path(var.get().strip())
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            messagebox.showwarning(PANEL_TITLE, "Folder does not exist.")
            return
        import sys as _sys
        if _sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _normalize_path_var(self, var: tk.StringVar) -> None:
        normalized = display_path(var.get())
        if normalized != var.get():
            var.set(normalized)

    # ------------------------------------------------------------------
    # Widget factories (mirrored from pair_finder_panel)
    # ------------------------------------------------------------------

    def _action_button(
        self,
        parent: tk.Misc,
        text: str,
        command,
        *,
        accent: bool = False,
    ) -> tk.Button:
        C = COLORS
        if accent:
            return tk.Button(
                parent, text=text, command=command,
                font=ACTION_BTN_FONT, bg=C["accent"], fg="white",
                activebackground=C["accent_hov"], activeforeground="white",
                relief="flat", borderwidth=0, highlightthickness=0,
                padx=ACTION_BTN_PADX, pady=ACTION_BTN_PADY, cursor="hand2",
            )
        return tk.Button(
            parent, text=text, command=command,
            font=ACTION_BTN_FONT, bg=C["panel2"], fg=C["fg"],
            activebackground=C["panel"], activeforeground=C["fg"],
            relief="flat", borderwidth=0, highlightthickness=0,
            padx=ACTION_BTN_PADX, pady=ACTION_BTN_PADY, cursor="hand2",
        )

    def _path_button(self, parent: tk.Misc, text: str, command) -> tk.Button:
        C = COLORS
        return tk.Button(
            parent, text=text, command=command,
            font=PATH_BTN_FONT, bg=C["panel2"], fg=C["fg"],
            activebackground=C["panel"], activeforeground=C["fg"],
            relief="flat", borderwidth=0, highlightthickness=0,
            padx=PATH_BTN_PADX, pady=PATH_BTN_PADY, cursor="hand2",
        )

    def _path_row(
        self,
        parent: tk.Misc,
        row: int,
        label: str,
        var: tk.StringVar,
        picker,
        opener,
        tip_text: str,
    ) -> None:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=CTRL_ROW_PADY)
        ent = ttk.Entry(parent, textvariable=var)
        ent.grid(row=row, column=1, sticky="ew", pady=CTRL_ROW_PADY)
        ent.bind("<FocusOut>", lambda _e, v=var: self._normalize_path_var(v))
        browse_btn = self._path_button(parent, "Browse", picker)
        browse_btn.grid(row=row, column=2, padx=(4, 0), pady=CTRL_ROW_PADY)
        open_btn = self._path_button(parent, "Open", opener)
        open_btn.grid(row=row, column=3, padx=(4, 0), pady=CTRL_ROW_PADY)
        tip(lbl, ent, browse_btn, text=tip_text)
        Tooltip(open_btn, TIPS["open_path"])

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def settings_snapshot(self) -> dict:
        return {
            "gg_genre_input_dir":   self.genre_input_dir.get(),
            "gg_genre_batch_mode":  bool(self.genre_batch_mode.get()),
            "gg_genre_tag_style":   self.genre_tag_style.get(),
            "gg_genre_write_meta":  bool(self.genre_write_meta.get()),
            "gg_genre_csv_path":    self.genre_csv_path.get(),
            "gg_gender_input_dir":  self.gender_input_dir.get(),
            "gg_gender_batch_mode": bool(self.gender_batch_mode.get()),
            "gg_gender_tag_field":  self.gender_tag_field.get(),
            "gg_gender_write_meta": bool(self.gender_write_meta.get()),
            "gg_gender_csv_path":   self.gender_csv_path.get(),
        }

    def _load_settings(self) -> None:
        data = load_settings()

        self.genre_input_dir.set(
            display_path(str(data.get("gg_genre_input_dir", "")))
        )
        self.genre_batch_mode.set(bool(data.get("gg_genre_batch_mode", True)))
        tag_style = data.get("gg_genre_tag_style", "combined")
        if tag_style in ("combined", "split"):
            self.genre_tag_style.set(tag_style)
        self.genre_write_meta.set(bool(data.get("gg_genre_write_meta", True)))
        self.genre_csv_path.set(str(data.get("gg_genre_csv_path", "")))

        self.gender_input_dir.set(
            display_path(str(data.get("gg_gender_input_dir", "")))
        )
        self.gender_batch_mode.set(bool(data.get("gg_gender_batch_mode", True)))
        tag_field = data.get("gg_gender_tag_field", "comment")
        if tag_field in ("comment", "gender"):
            self.gender_tag_field.set(tag_field)
        self.gender_write_meta.set(bool(data.get("gg_gender_write_meta", True)))
        self.gender_csv_path.set(str(data.get("gg_gender_csv_path", "")))

    def _save_settings(self) -> None:
        data = load_settings()
        data.pop("gg_tagger_dir", None)
        data.update(self.settings_snapshot())
        save_settings(data)

    def _bind_settings_autosave(self) -> None:
        for var in (
            self.genre_input_dir,
            self.genre_batch_mode,
            self.genre_tag_style,
            self.genre_write_meta,
            self.genre_csv_path,
            self.gender_input_dir,
            self.gender_batch_mode,
            self.gender_tag_field,
            self.gender_write_meta,
            self.gender_csv_path,
        ):
            var.trace_add("write", lambda *_: self._save_settings())
