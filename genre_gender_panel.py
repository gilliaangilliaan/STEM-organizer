from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

from ffmpeg_bootstrap import subprocess_kwargs
from ui_theme import (
    ACTION_BTN_GAP,
    ACTIONS_BOTTOM_PAD,
    COLORS,
    DARK,
    HEADER_DESC_COLOR,
    HEADER_TOP_PAD,
    PATH_BTN_HEIGHT,
    SECTION_GAP,
    SECTION_INNER_PAD,
    SECTION_PADX,
    Tooltip,
    ctk_action_button,
    ctk_path_row,
    ctk_pin_button_height,
    ctk_section,
    ctk_size_subtab_buttons,
    ctk_ui_font,
    display_path,
    ensure_ctk_dark,
    show_ctk_help_dialog,
    tip,
)

APP_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
SETTINGS_PATH = APP_DIR / "settings.json"
PAIR_LOG_TAG = "__pair_log__"
PROGRESS_TAG = "__progress__"
PANEL_TITLE = "Genre & Gender"
_TQDM_PCT_RE = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)%\|.*?\|?\s*(?P<cur>\d+)/(?P<total>\d+)",
)
# "Checking tags: 12718/81601 (16%)" / tqdm leftovers — progress bar only, not LOG.
_PROGRESS_LOG_RE = re.compile(
    r"^.+?:\s*\d[\d,]*/\d[\d,]*\s*\(\d+(?:\.\d+)?%\)$"
    r"|^\s*\d+(?:\.\d+)?%\|"
    r"|it/s\]",
)
_GG_RESULT_KEY_RE = re.compile(
    r"^(GENRE|STYLE|CONF|GENDER|REVERB):\s*(.*)$",
    re.IGNORECASE,
)
_GG_AUDIO_NAME_RE = re.compile(
    r"\.(flac|mp3|wav|m4a|aiff?|ogg|opus)\s*$",
    re.IGNORECASE,
)


_GG_BADGE_LINE_RE = re.compile(
    r"^\s*(female|male|dry|wet)"
    r"(?:\s+\(confidence\s+[^)]+\)|\s+\d+%)?\s*$",
    re.IGNORECASE,
)


def _gg_log_tag(line: str) -> str:
    """Pick LOG color tag from a tagger stdout line."""
    s = (line or "").strip()
    if not s:
        return "info"
    low = s.lower()
    if low.startswith("error") or low.startswith("[tagger exited"):
        return "err"
    if low.startswith("[tagger") or low.startswith("stop requested"):
        return "warn"
    if s.startswith("===") and s.endswith("==="):
        return "detail"
    if _GG_BADGE_LINE_RE.match(s) or low.startswith("(confidence"):
        return "gg_result"
    if _GG_RESULT_KEY_RE.match(s):
        return "gg_result"
    if _GG_AUDIO_NAME_RE.search(s) and ":" not in s.split()[0]:
        return "gg_file"
    if s.startswith("Processing") or s.startswith("Tagged:"):
        return "detail"
    return "info"
# Bundled beside the app (source tree or next to STEM-organizer.exe).
TAGGER_DIR = APP_DIR / "genre_gender_tagger"
TAGGER_SCRIPT = TAGGER_DIR / "genre_gender_tagger.py"
TAGGER_PYTHON = TAGGER_DIR / "venv" / "Scripts" / "python.exe"

TIPS = {
    "genre_input": "Folder containing instrumental FLAC/MP3/WAV files to tag with genre.",
    "gender_input": "Folder containing acapella FLAC/MP3/WAV files to tag with voice gender.",
    "include_subfolders": "Scan audio files in subfolders too, not just the selected folder itself.",
    "batch_mode": (
        "Batch: fastest — many files share GPU batches; status bar shows "
        "live n/total; LOG uses === file === plus genre/style badges "
        "(dry/wet chip colors). "
        "Per-file: same LOG shape, slower (no multi-file GPU packing)."
    ),
    "gender_batch_mode": (
        "Batch: fastest — patches from many files share GPU batches "
        "(ONNX DirectML EffNet + PyTorch reverb); status bar stays live "
        "through extract/reverb/write. LOG uses Classify-style badges "
        "(female/male, wet/dry). Per-file: same LOG, slower (still GPU)."
    ),
    "tag_style": (
        "Combined writes a single GENRE tag as 'Genre/Style'. "
        "Split writes separate GENRE and STYLE tags."
    ),
    "tag_field": "Comment writes gender to the COMMENT tag. Gender writes to a GENDER tag.",
    "reverb_mode": (
        "Dry/wet from the bundled vocal mel-CNN (models\\vocal_reverb.pt). "
        "Combined writes gender/reverb into the chosen field "
        "(e.g. COMMENT=female/wet). Split writes gender alone and "
        "REVERB=wet|dry as a separate custom field."
    ),
    "write_meta": "Write tags to FLAC/MP3/M4A/WAV. Disable to only generate the CSV.",
    "overwrite_tags": (
        "Off (default): skip files that already have genre/gender tags "
        "(resume-friendly). On: re-tag every file."
    ),
    "csv_path": "Optional path for the output CSV. Leave empty to use the tagger default.",
    "open_path": "Open this folder in Explorer.",
}


def _format_tagger_exit(code: int | None) -> str:
    """Human-readable tagger exit (Windows NTSTATUS often unsigned)."""
    if code is None:
        return "unknown"
    code_u = code & 0xFFFFFFFF if code < 0 else int(code)
    if code_u == 0xC0000005:
        return (
            f"{code} ACCESS_VIOLATION (usually RAM exhausted). "
            "Batch now tags in 256-file waves — retry; already-tagged "
            "files from a partial run are kept."
        )
    if code_u == 0xC000012D:
        return f"{code} out of system resources / memory"
    return str(code)


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
        else "Tag acapellas with voice gender + dry/wet reverb"
    )
    intro = (
        "Classify instrumental tracks and write Discogs-style GENRE / STYLE tags."
        if is_genre
        else "Estimate singing voice gender and dry/wet reverb, then write tags."
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
            "2. Pick run mode, tag field, and reverb write style.\n"
            "3. Click ▶ Tag gender and watch progress in the LOG panel.\n"
            "4. Check tags and/or the CSV export when the run finishes.",
        ),
        (
            "Model",
            "Gender: Essentia gender-discogs-effnet — at least detects "
            "high- and low-pitched vocals.\n"
            "Reverb: lightweight mel-CNN — the world's first FOSS reverb "
            "classifier trained on singing vocals — 4.8k dry and 4.8k wet "
            "(sample packs, remix packs, multitracks).",
        ),
        (
            "Options",
            "Batch is faster; only reverb can use the GPU (gender stays on CPU). "
            "Per-file prints each GENDER/REVERB/CONF live. COMMENT writes e.g. "
            "COMMENT=female; GENDER writes a dedicated GENDER tag. Combined "
            "reverb mode writes e.g. COMMENT=female/wet; Split writes "
            "COMMENT=female and REVERB=wet. Tags are written to FLAC, MP3, M4A, "
            "and WAV. A CSV export is always written.",
        ),
        (
            "Setup",
            "Genre & Gender uses the bundled folder genre_gender_tagger\\ with its "
            "own venv. Run genre_gender_tagger\\install-deps.bat once "
            "(or accept it at the end of STEM install-deps.bat).",
        ),
    )

    show_ctk_help_dialog(
        parent,
        title=title,
        heading=heading,
        intro=intro,
        sections=list(sections),
    )


_ctk = ensure_ctk_dark()


class GenreGenderPanel(_ctk.CTkFrame):
    def __init__(
        self,
        host: tk.Misc,
        parent: tk.Misc,
        info_icon_factory=None,
    ) -> None:
        ctk = ensure_ctk_dark()
        super().__init__(parent, fg_color=DARK["bg"])
        self._ctk = ctk
        self._host = host
        self._info_icon_factory = info_icon_factory
        self._busy = False
        self._stop_requested = False
        self._proc: subprocess.Popen | None = None
        self._worker: threading.Thread | None = None

        # ---- Genre tab vars ----
        self.genre_input_dir = tk.StringVar()
        self.genre_include_subfolders = tk.BooleanVar(value=True)
        self.genre_batch_mode = tk.BooleanVar(value=True)
        self.genre_tag_style = tk.StringVar(value="combined")
        self.genre_write_meta = tk.BooleanVar(value=True)
        self.genre_overwrite_tags = tk.BooleanVar(value=False)
        self.genre_csv_path = tk.StringVar()

        # ---- Gender tab vars ----
        self.gender_input_dir = tk.StringVar()
        self.gender_include_subfolders = tk.BooleanVar(value=True)
        self.gender_batch_mode = tk.BooleanVar(value=True)
        self.gender_tag_field = tk.StringVar(value="comment")
        self.gender_reverb_mode = tk.StringVar(value="combined")
        self.gender_write_meta = tk.BooleanVar(value=True)
        self.gender_overwrite_tags = tk.BooleanVar(value=False)
        self.gender_csv_path = tk.StringVar()

        self._build_ui()
        self._bind_settings_autosave()
        self._load_settings()

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        ctk = self._ctk
        t = DARK
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        tabview = ctk.CTkTabview(
            self,
            fg_color=t["bg"],
            segmented_button_fg_color=t["panel"],
            segmented_button_selected_color=t["accent"],
            segmented_button_selected_hover_color=t["accent_hover"],
            segmented_button_unselected_color=t["panel_2"],
            segmented_button_unselected_hover_color=t["btn_hover"],
            text_color=t["text"],
            anchor="w",
            command=self._on_subtab_changed,
        )
        tabview.grid(row=0, column=0, sticky="nsew")
        self._tabview = tabview

        genre_tab = tabview.add("Genre")
        gender_tab = tabview.add("Gender")
        ctk_size_subtab_buttons(tabview)

        self._build_genre_tab(genre_tab)
        self._build_gender_tab(gender_tab)

    def _make_info_icon(self, parent: tk.Misc, command) -> None:
        # CTkLabel is taller than the glyph; pin icon to top next to the text.
        pack_kw = dict(side="left", padx=(4, 0), anchor="n", pady=(5, 0))
        if self._info_icon_factory is not None:
            self._info_icon_factory(parent, command).pack(**pack_kw)
            return
        fallback = tk.Label(
            parent, text="?", font=("Segoe UI Semibold", 9),
            fg=COLORS["fg_dim"], bg=DARK["bg"], cursor="hand2",
        )
        fallback.pack(**pack_kw)
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
        ctk = self._ctk
        desc_row = ctk.CTkFrame(parent, fg_color="transparent")
        desc_row.pack(fill="x", anchor="w")
        ctk.CTkLabel(
            desc_row,
            text=text,
            font=ctk_ui_font(),
            text_color=HEADER_DESC_COLOR,
            wraplength=470,
            justify="left",
            anchor="w",
        ).pack(side="left")
        self._make_info_icon(desc_row, command)

    def _radio_row(
        self,
        parent,
        label: str,
        variable: tk.Variable,
        options: list[tuple[str, object]],
        tip_text: str,
        *,
        show_label: bool = True,
    ) -> None:
        ctk = self._ctk
        t = DARK
        _font = ctk_ui_font()
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(anchor="w", pady=(0, 6))
        widgets: list = []
        if show_label and label:
            lbl = ctk.CTkLabel(
                row,
                text=label,
                text_color=t["label"],
                font=_font,
            )
            lbl.pack(side="left", padx=(0, 12))
            widgets.append(lbl)
        for idx, (text, value) in enumerate(options):
            # Main = body color; parenthetical = title color. Circle-only
            # radio so short labels (Split/Batch) don't get CTk width gaps.
            paren = text.find("(")
            if paren > 0:
                main = text[:paren].rstrip()
                hint = text[paren:]
            else:
                main, hint = text, None
            opt = ctk.CTkFrame(row, fg_color="transparent")
            opt.pack(side="left", padx=(0, 28 if idx < len(options) - 1 else 0))
            rb = ctk.CTkRadioButton(
                opt,
                text="",
                variable=variable,
                value=value,
                fg_color=t["accent"],
                border_color=t["border"],
                hover_color=t["accent_hover"],
                font=_font,
                width=22,
                radiobutton_width=22,
                radiobutton_height=22,
            )
            rb.pack(side="left")
            widgets.append(rb)

            def _pick(_event=None, v=value, var=variable) -> None:
                var.set(v)

            main_lbl = ctk.CTkLabel(
                opt, text=main, text_color=t["text"], font=_font, cursor="hand2",
            )
            main_lbl.pack(side="left", padx=(6, 0))
            main_lbl.bind("<Button-1>", _pick)
            widgets.append(main_lbl)
            if hint:
                hint_lbl = ctk.CTkLabel(
                    opt, text=hint, text_color=t["text_dim"], font=_font, cursor="hand2",
                )
                hint_lbl.pack(side="left", padx=(6, 0))
                hint_lbl.bind("<Button-1>", _pick)
                widgets.append(hint_lbl)
        tip(*widgets, text=tip_text)

    def _tag_options_row(
        self,
        parent,
        write_meta_var: tk.BooleanVar,
        overwrite_var: tk.BooleanVar,
    ) -> None:
        """Write-metadata + overwrite checkboxes side by side under TAG OPTIONS."""
        ctk = self._ctk
        t = DARK
        _font = ctk_ui_font()
        card = ctk_section(parent, "Tag options")
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(anchor="w", fill="x")
        meta_chk = ctk.CTkCheckBox(
            row,
            text="Write metadata to files",
            variable=write_meta_var,
            fg_color=t["accent"],
            hover_color=t["accent_hover"],
            border_color=t["border"],
            text_color=t["text"],
            font=_font,
        )
        meta_chk.pack(side="left")
        tip(meta_chk, text=TIPS["write_meta"])
        overwrite_chk = ctk.CTkCheckBox(
            row,
            text="Overwrite existing tags",
            variable=overwrite_var,
            fg_color=t["accent"],
            hover_color=t["accent_hover"],
            border_color=t["border"],
            text_color=t["text"],
            font=_font,
        )
        overwrite_chk.pack(side="left", padx=(28, 0))
        tip(overwrite_chk, text=TIPS["overwrite_tags"])

    def _build_genre_tab(self, parent) -> None:
        ctk = self._ctk
        t = DARK
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", padx=SECTION_PADX, pady=(HEADER_TOP_PAD, 12))
        self._description_with_info(
            header,
            "Tag instrumental/music files with Discogs genre and style tags.",
            self._show_genre_help,
        )

        paths = ctk_section(parent, "Paths")
        paths.columnconfigure(1, weight=1)
        self._path_row(
            paths, 0,
            "Input folder", self.genre_input_dir,
            lambda: self._pick_dir(self.genre_input_dir, "Select input folder"),
            lambda: self._open_folder(self.genre_input_dir),
            TIPS["genre_input"],
        )
        genre_sub_chk = ctk.CTkCheckBox(
            paths,
            text="Include subfolders",
            variable=self.genre_include_subfolders,
            fg_color=t["accent"],
            hover_color=t["accent_hover"],
            border_color=t["border"],
            text_color=t["text"],
            font=ctk_ui_font(),
        )
        genre_sub_chk.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        tip(genre_sub_chk, text=TIPS["include_subfolders"])

        run_mode = ctk_section(parent, "Run mode")
        self._radio_row(
            run_mode,
            "Run mode",
            self.genre_batch_mode,
            [
                ("Batch (fast)", True),
                ("Per-file (live results)", False),
            ],
            TIPS["batch_mode"],
            show_label=False,
        )

        tag_style = ctk_section(parent, "Tag style")
        self._radio_row(
            tag_style,
            "Tag style",
            self.genre_tag_style,
            [
                ("Combined  (GENRE=Rock/Surf)", "combined"),
                ("Split  (GENRE=Rock, STYLE=Surf)", "split"),
            ],
            TIPS["tag_style"],
            show_label=False,
        )
        self._tag_options_row(
            parent,
            self.genre_write_meta,
            self.genre_overwrite_tags,
        )

        tk.Frame(parent, bg=DARK["bg"], height=ACTIONS_BOTTOM_PAD).pack(side="bottom")

    def _build_gender_tab(self, parent) -> None:
        ctk = self._ctk
        t = DARK
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", padx=SECTION_PADX, pady=(HEADER_TOP_PAD, 12))
        self._description_with_info(
            header,
            "Tag acapella audio files with voice gender (female/male) and reverb (wet/dry).",
            self._show_gender_help,
        )

        paths = ctk_section(parent, "Paths")
        paths.columnconfigure(1, weight=1)
        self._path_row(
            paths, 0,
            "Input folder", self.gender_input_dir,
            lambda: self._pick_dir(self.gender_input_dir, "Select input folder"),
            lambda: self._open_folder(self.gender_input_dir),
            TIPS["gender_input"],
        )
        gender_sub_chk = ctk.CTkCheckBox(
            paths,
            text="Include subfolders",
            variable=self.gender_include_subfolders,
            fg_color=t["accent"],
            hover_color=t["accent_hover"],
            border_color=t["border"],
            text_color=t["text"],
            font=ctk_ui_font(),
        )
        gender_sub_chk.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        tip(gender_sub_chk, text=TIPS["include_subfolders"])

        run_mode = ctk_section(parent, "Run mode")
        self._radio_row(
            run_mode,
            "Run mode",
            self.gender_batch_mode,
            [
                ("Batch (fast)", True),
                ("Per-file (live results)", False),
            ],
            TIPS["gender_batch_mode"],
            show_label=False,
        )

        tag_field = ctk_section(parent, "Voice gender")
        self._radio_row(
            tag_field,
            "Voice gender",
            self.gender_tag_field,
            [
                ("COMMENT  (COMMENT=female)", "comment"),
                ("GENDER  (GENDER=female)", "gender"),
            ],
            TIPS["tag_field"],
            show_label=False,
        )

        reverb = ctk_section(parent, "Reverb")
        self._radio_row(
            reverb,
            "Reverb",
            self.gender_reverb_mode,
            [
                ("Combined  (e.g. female/wet)", "combined"),
                ("Split  (gender + REVERB=wet)", "split"),
            ],
            TIPS["reverb_mode"],
            show_label=False,
        )
        self._tag_options_row(
            parent,
            self.gender_write_meta,
            self.gender_overwrite_tags,
        )

        tk.Frame(parent, bg=DARK["bg"], height=ACTIONS_BOTTOM_PAD).pack(side="bottom")

    # ------------------------------------------------------------------
    # Action bar
    # ------------------------------------------------------------------

    def attach_action_bar(self, parent: tk.Misc) -> None:
        # Same pattern as Classify: accent Start + dim ■ Stop beside it.
        self.genre_btn = ctk_action_button(
            parent, "▶  Tag genre", self._start_genre, accent=True, width=96,
        )
        self.gender_btn = ctk_action_button(
            parent, "▶  Tag gender", self._start_gender, accent=True, width=100,
        )
        self.stop_btn = ctk_action_button(
            parent, "■  Stop", self._request_stop,
            width=64, text_color=DARK["text_dim"], cursor="arrow",
        )
        tip(self.stop_btn, text="Stop the running tagger.")

    def _genre_tab_active(self) -> bool:
        return self._tabview.get() == "Genre"

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
        self.stop_btn.pack(side="left", padx=(ACTION_BTN_GAP, 0))
        self._refresh_run_buttons()
        host = self._host
        if hasattr(host, "_pin_action_bar_heights"):
            host._pin_action_bar_heights()

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
            self.genre_btn.configure(text_color=DARK["text_dim"], cursor="arrow")
            self.gender_btn.configure(text_color=DARK["text_dim"], cursor="arrow")
            self.stop_btn.configure(text_color=DARK["text_dim"], cursor="arrow")
        else:
            self._refresh_run_buttons(running=False)

    def _refresh_run_buttons(self, *, running: bool | None = None) -> None:
        if running is None:
            running = self._busy
        h = PATH_BTN_HEIGHT
        if running:
            self.genre_btn.configure(text_color=DARK["text_dim"], cursor="arrow", height=h)
            self.gender_btn.configure(text_color=DARK["text_dim"], cursor="arrow", height=h)
            self.stop_btn.configure(text_color=DARK["danger"], cursor="hand2", height=h)
        else:
            self.genre_btn.configure(text_color="#ffffff", cursor="hand2", height=h)
            self.gender_btn.configure(text_color="#ffffff", cursor="hand2", height=h)
            self.stop_btn.configure(text_color=DARK["text_dim"], cursor="arrow", height=h)
        ctk_pin_button_height(self.genre_btn, self.gender_btn, self.stop_btn)

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
                bool(self.genre_include_subfolders.get()),
                bool(self.genre_overwrite_tags.get()),
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
                self.gender_reverb_mode.get(),
                self.gender_tag_field.get(),
                bool(self.gender_write_meta.get()),
                self.gender_csv_path.get().strip(),
                bool(self.gender_include_subfolders.get()),
                bool(self.gender_overwrite_tags.get()),
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
        include_subfolders: bool = True,
        overwrite_tags: bool = False,
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
        env["GG_GENDER_FIELD"] = gender_field
        env["GG_WRITE_META"]   = "1" if write_meta else "0"
        env["GG_OVERWRITE"]    = "1" if overwrite_tags else "0"
        env["GG_RECURSIVE"]    = "1" if include_subfolders else "0"
        if mode == "gender":
            # For gender runs, tag_style carries combined|split reverb write mode.
            env["GG_REVERB_MODE"] = (
                tag_style if tag_style in ("combined", "split") else "combined"
            )
            env["GG_TAG_STYLE"] = "combined"
        else:
            env["GG_TAG_STYLE"] = tag_style
            env["GG_REVERB_MODE"] = "combined"
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
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **subprocess_kwargs(),  # hide console window on Windows
            )
            self._proc = proc

            for line in self._iter_tagger_lines(proc.stdout):
                self._handle_tagger_line(line)

            proc.wait()
            if self._stop_requested:
                self._host.log_queue.put((PAIR_LOG_TAG, "[tagger stopped]", "warn"))
                self._finish_worker("Stopped")
            elif proc.returncode == 0:
                self._host.log_queue.put((PROGRESS_TAG, 100.0, 0.0))
                self._finish_worker("Done")
            else:
                detail = _format_tagger_exit(proc.returncode)
                self._host.log_queue.put(
                    (PAIR_LOG_TAG, f"[tagger exited: {detail}]", "warn")
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

    @staticmethod
    def _iter_tagger_lines(stream):
        """Yield stdout lines; treat tqdm \\r updates as separate lines."""
        buf = ""
        while True:
            chunk = stream.read(1024)
            if not chunk:
                break
            buf += chunk.replace("\r\n", "\n").replace("\r", "\n")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line:
                    yield line
        rem = buf.strip()
        if rem:
            yield rem

    def _handle_tagger_line(self, line: str) -> None:
        """Forward log lines; drive host progress bar from tagger updates."""
        if line.startswith("__progress__\t") or line.startswith("__progress__ "):
            parts = line.split("\t") if "\t" in line else line.split()
            try:
                pct = float(parts[1])
            except (IndexError, ValueError):
                return
            eta = None
            if len(parts) >= 3 and parts[2] not in ("", "?"):
                try:
                    eta = float(parts[2])
                except ValueError:
                    eta = None
            n = total = None
            phase = ""
            if len(parts) >= 5:
                try:
                    n = int(float(parts[3]))
                    total = int(float(parts[4]))
                except ValueError:
                    n = total = None
            if len(parts) >= 6:
                phase = str(parts[5] or "").strip()
            self._host.log_queue.put(
                (PROGRESS_TAG, pct, eta, n, total, phase)
            )
            return

        parsed = _TQDM_PCT_RE.search(line)
        if parsed:
            try:
                pct = float(parsed.group("pct"))
            except ValueError:
                pct = None
            n = total = None
            try:
                n = int(parsed.group("cur"))
                total = int(parsed.group("total"))
            except (ValueError, IndexError):
                pass
            if pct is not None:
                self._host.log_queue.put(
                    (PROGRESS_TAG, pct, None, n, total, "")
                )
            return  # never dump tqdm bars into LOG

        if _PROGRESS_LOG_RE.search(line):
            return

        self._host.log_queue.put((PAIR_LOG_TAG, line, _gg_log_tag(line)))

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
        parent.columnconfigure(1, weight=1)
        _lbl, ent, _browse, _open_btn = ctk_path_row(
            parent,
            row,
            label,
            var,
            picker,
            opener,
            tip_text=tip_text,
            open_tip=TIPS["open_path"],
        )
        ent.bind("<FocusOut>", lambda _e, v=var: self._normalize_path_var(v))

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def settings_snapshot(self) -> dict:
        return {
            "gg_genre_input_dir":   self.genre_input_dir.get(),
            "gg_genre_include_subfolders": bool(self.genre_include_subfolders.get()),
            "gg_genre_batch_mode":  bool(self.genre_batch_mode.get()),
            "gg_genre_tag_style":   self.genre_tag_style.get(),
            "gg_genre_write_meta":  bool(self.genre_write_meta.get()),
            "gg_genre_overwrite_tags": bool(self.genre_overwrite_tags.get()),
            "gg_genre_csv_path":    self.genre_csv_path.get(),
            "gg_gender_input_dir":  self.gender_input_dir.get(),
            "gg_gender_include_subfolders": bool(self.gender_include_subfolders.get()),
            "gg_gender_batch_mode": bool(self.gender_batch_mode.get()),
            "gg_gender_tag_field":  self.gender_tag_field.get(),
            "gg_gender_reverb_mode": self.gender_reverb_mode.get(),
            "gg_gender_write_meta": bool(self.gender_write_meta.get()),
            "gg_gender_overwrite_tags": bool(self.gender_overwrite_tags.get()),
            "gg_gender_csv_path":   self.gender_csv_path.get(),
        }

    def _load_settings(self) -> None:
        data = load_settings()

        self.genre_input_dir.set(
            display_path(str(data.get("gg_genre_input_dir", "")))
        )
        self.genre_include_subfolders.set(
            bool(data.get("gg_genre_include_subfolders", True))
        )
        self.genre_batch_mode.set(bool(data.get("gg_genre_batch_mode", True)))
        tag_style = data.get("gg_genre_tag_style", "combined")
        if tag_style in ("combined", "split"):
            self.genre_tag_style.set(tag_style)
        self.genre_write_meta.set(bool(data.get("gg_genre_write_meta", True)))
        self.genre_overwrite_tags.set(
            bool(data.get("gg_genre_overwrite_tags", False))
        )
        self.genre_csv_path.set(str(data.get("gg_genre_csv_path", "")))

        self.gender_input_dir.set(
            display_path(str(data.get("gg_gender_input_dir", "")))
        )
        self.gender_include_subfolders.set(
            bool(data.get("gg_gender_include_subfolders", True))
        )
        self.gender_batch_mode.set(bool(data.get("gg_gender_batch_mode", True)))
        tag_field = data.get("gg_gender_tag_field", "comment")
        if tag_field in ("comment", "gender"):
            self.gender_tag_field.set(tag_field)
        reverb_mode = data.get("gg_gender_reverb_mode", "combined")
        if reverb_mode in ("combined", "split"):
            self.gender_reverb_mode.set(reverb_mode)
        self.gender_write_meta.set(bool(data.get("gg_gender_write_meta", True)))
        self.gender_overwrite_tags.set(
            bool(data.get("gg_gender_overwrite_tags", False))
        )
        self.gender_csv_path.set(str(data.get("gg_gender_csv_path", "")))

    def _save_settings(self) -> None:
        data = load_settings()
        data.pop("gg_tagger_dir", None)
        data.update(self.settings_snapshot())
        save_settings(data)

    def _bind_settings_autosave(self) -> None:
        for var in (
            self.genre_input_dir,
            self.genre_include_subfolders,
            self.genre_batch_mode,
            self.genre_tag_style,
            self.genre_write_meta,
            self.genre_overwrite_tags,
            self.genre_csv_path,
            self.gender_input_dir,
            self.gender_include_subfolders,
            self.gender_batch_mode,
            self.gender_tag_field,
            self.gender_reverb_mode,
            self.gender_write_meta,
            self.gender_overwrite_tags,
            self.gender_csv_path,
        ):
            var.trace_add("write", lambda *_: self._save_settings())
