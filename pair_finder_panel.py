from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pair_matcher import (
    IgnoreRules,
    find_pairs,
    organize_matched_folder,
    strictness_to_threshold,
)
from stem_align import (
    align_all_songs,
    default_with_original_dir,
    default_without_original_dir,
    distribute_originals,
    export_song_list,
    sort_folders_by_original,
)
from stem_player import open_stem_player
from ui_theme import (
    ACTIONS_BOTTOM_PAD,
    ACTION_BTN_FONT,
    ACTION_BTN_PADX,
    ACTION_BTN_PADY,
    COLORS,
    CONTENT_PAD,
    CTRL_ROW_PADY,
    DarkScrollbar,
    HEADER_DESC_FONT,
    HEADER_TOP_PAD,
    LOG_FONT,
    LOG_INNER_PAD,
    LOG_WARN_COLOR,
    LEFT_PANEL_WIDTH,
    PATH_BTN_FONT,
    PATH_BTN_PADX,
    PATH_BTN_PADY,
    SECTION_GAP,
    SECTION_INNER_PAD,
    SECTION_PADX,
    SECTION_SIDE_PAD_LEFT,
    STATUS_BOTTOM_PAD,
    STATUS_FONT,
    STATUS_FRAME_HEIGHT,
    STATUS_IDLE_Y,
    STATUS_PAD_BOTTOM,
    STATUS_PAD_TOP,
    STATUS_PAD_X,
    STATUS_PCT_FONT,
    STATUS_PROGRESS_HEIGHT,
    STATUS_PROGRESS_ROW_HEIGHT,
    STATUS_PROGRESS_Y_PAD,
    STATUS_ROW_GAP,
    STATUS_TOP_PAD,
    RESOURCE_BAR_HEIGHT,
    RESOURCE_BAR_WIDTH,
    RESOURCE_ROW_HEIGHT,
    Tooltip,
    WIN_DEFAULT_H,
    WIN_DEFAULT_W,
    apply_theme,
    bind_mousewheel,
    display_path,
    format_eta,
    format_status_clock,
    place_window_centered,
    tip,
)

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
SETTINGS_PATH = APP_DIR / 'settings.json'
PAIR_LOG_TAG = '__pair_log__'
PANEL_TITLE = 'Match & Align'

TIPS = {
    'acapella': 'Folder containing acapella FLAC or MP3 files.',
    'instrumental': 'Folder containing instrumental FLAC or MP3 files.',
    'reference': 'Matching starts from the reference folder; each file looks for a partner in the other folder.',
    'include_subfolders': 'Scan FLAC/MP3 files in subfolders too, not just the selected folder itself.',
    'pairs_output': 'Matched pairs are moved here (flat folder).',
    'organize': 'Folder of matched FLAC/MP3 files to group into Artist - Title subfolders.',
    'strictness': 'Higher = stricter tag matching. Compares artist and title separately; both must meet the threshold.',
    'filename_fallback': 'When artist or title tags are missing, parse them from the filename (e.g. Artist - Title (Acapella).mp3).',
    'ignore_parentheses': 'Strip text inside (parentheses) before comparing tags.',
    'ignore_square': 'Strip text inside [square brackets] before comparing tags.',
    'ignore_all_brackets': 'Strip both (parentheses) and [square brackets] before comparing.',
    'ignore_spaces': 'Collapse repeated spaces and trim edges before comparing.',
    'ignore_custom': 'Remove these words or phrases from tags (case-insensitive) before comparing.',
    'add_keyword': 'Add a custom keyword to ignore.',
    'open_path': 'Open this folder in Explorer.',
    'align_stems_root': 'Root folder with one subfolder per song (each contains acapella + instrumental).',
    'align_inbox': 'Folder where you drop downloaded originals before distributing.',
    'align_export': 'Text file listing subfolder names to look up on Soundiiz / YouTube / Deezer.',
    'align_backup': 'Copy instrumental and acapella to _backup_before_align before overwriting.',
    'align_skip_existing': 'Skip song folders that already contain _backup_before_align (from a previous align run).',
    'align_analysis': 'Seconds of audio used to cross-correlate stems against the original song.',
    'align_sort_after': 'After distributing, move song folders into with_original and without_original subfolders.',
    'align_with_original': 'Folders that received an original song (auto-filled as stems root / with_original).',
    'align_play_stems': 'Open stem player on the with_original library (instrumental, acapella, original).',
    'align_without_original': 'Folders still missing an original (auto-filled as stems root / without_original).',
}


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    try:
        tmp = SETTINGS_PATH.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(SETTINGS_PATH)
    except OSError:
        pass


class PairFinderPanel(ttk.Frame):
    def __init__(self, host: tk.Misc, parent: tk.Misc) -> None:
        super().__init__(parent)
        self._host = host

        self.acapella_dir = tk.StringVar()
        self.instrumental_dir = tk.StringVar()
        self.pairs_output_dir = tk.StringVar()
        self.organize_dir = tk.StringVar()
        self.reference_side = tk.StringVar(value='acapella')
        self.strictness = tk.DoubleVar(value=75.0)
        self.use_filename_fallback = tk.BooleanVar(value=True)
        self.include_subfolders = tk.BooleanVar(value=False)
        self.align_stems_root = tk.StringVar()
        self.align_originals_inbox = tk.StringVar()
        self.align_export_file = tk.StringVar(value=str(APP_DIR / 'songs_to_download.txt'))
        self.align_backup = tk.BooleanVar(value=True)
        self.align_skip_existing = tk.BooleanVar(value=True)
        self.align_analysis_sec = tk.IntVar(value=30)
        self.align_sort_after = tk.BooleanVar(value=True)
        self.align_with_original_dir = tk.StringVar()
        self.align_without_original_dir = tk.StringVar()
        self.ignore_parentheses = tk.BooleanVar(value=True)
        self.ignore_square_brackets = tk.BooleanVar(value=True)
        self.ignore_all_brackets = tk.BooleanVar(value=True)
        self.ignore_extra_spaces = tk.BooleanVar(value=True)
        self._custom_keyword_vars: list[tk.StringVar] = []
        self._busy = False

        self._worker: threading.Thread | None = None

        self._build_ui()
        self._bind_settings_autosave()
        self._load_settings()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self, style='Sub.TNotebook', takefocus=0)
        notebook.grid(row=0, column=0, sticky='nsew')
        self._notebook = notebook
        pair_tab = ttk.Frame(notebook)
        align_tab = ttk.Frame(notebook)
        notebook.add(pair_tab, text='  Match  ')
        notebook.add(align_tab, text='  Align  ')
        notebook.bind('<<NotebookTabChanged>>', self._on_pair_subtab_changed)
        self._build_pair_tab(pair_tab)
        self._build_align_tab(align_tab)

    def _build_pair_tab(self, left: ttk.Frame) -> None:
        header = ttk.Frame(left)
        header.pack(fill='x', padx=SECTION_PADX, pady=(HEADER_TOP_PAD, 12))
        tk.Label(
            header,
            text='Match acapella and instrumental FLAC/MP3 files by artist/title tags, then organize pairs into song folders.',
            font=HEADER_DESC_FONT, fg=COLORS['fg'], bg=COLORS['bg'],
            wraplength=520, justify='left',
        ).pack(anchor='w')

        paths = ttk.LabelFrame(left, text='  FOLDERS  ', padding=SECTION_INNER_PAD)
        paths.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        paths.columnconfigure(1, weight=1)
        self._path_row(paths, 0, 'Acapella', self.acapella_dir, self._pick_acapella, self._open_acapella, TIPS['acapella'])
        self._path_row(paths, 1, 'Instrumental', self.instrumental_dir, self._pick_instrumental, self._open_instrumental, TIPS['instrumental'])
        self._path_row(paths, 2, 'Pairs output', self.pairs_output_dir, self._pick_pairs_output, self._open_pairs_output, TIPS['pairs_output'])
        self._path_row(paths, 3, 'Organize in', self.organize_dir, self._pick_organize, self._open_organize, TIPS['organize'])

        ref_frame = ttk.Frame(paths)
        ref_frame.grid(row=4, column=0, columnspan=4, sticky='w', pady=(8, 0))
        ttk.Label(ref_frame, text='Reference folder').pack(side='left', padx=(0, 12))
        acap_ref = ttk.Radiobutton(ref_frame, text='Acapella', variable=self.reference_side, value='acapella')
        acap_ref.pack(side='left', padx=(0, 10))
        inst_ref = ttk.Radiobutton(ref_frame, text='Instrumental', variable=self.reference_side, value='instrumental')
        inst_ref.pack(side='left')
        tip(acap_ref, inst_ref, text=TIPS['reference'])

        subfolder_chk = ttk.Checkbutton(
            paths, text='Include subfolders',
            variable=self.include_subfolders,
        )
        subfolder_chk.grid(row=5, column=0, columnspan=4, sticky='w', pady=(8, 0))
        tip(subfolder_chk, text=TIPS['include_subfolders'])

        match_opts = ttk.LabelFrame(left, text='  MATCHING  ', padding=SECTION_INNER_PAD)
        match_opts.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        match_opts.columnconfigure(1, weight=1)
        strict_lbl = ttk.Label(match_opts, text='Tag strictness')
        strict_lbl.grid(row=0, column=0, sticky='w', padx=(0, 10), pady=6)
        strict_row = ttk.Frame(match_opts)
        strict_row.grid(row=0, column=1, sticky='ew', pady=6)
        strict_row.columnconfigure(0, weight=1)
        self.strictness_readout = ttk.Label(strict_row, text=self._strictness_label(), style='Dim.TLabel', width=16)
        self.strictness_scale = ttk.Scale(
            strict_row, from_=0, to=100, orient='horizontal', variable=self.strictness,
            command=lambda _v: self.strictness_readout.configure(text=self._strictness_label()),
        )
        self.strictness_scale.grid(row=0, column=0, sticky='ew')
        self.strictness_readout.grid(row=0, column=1, padx=(8, 0))
        tip(strict_lbl, self.strictness_scale, self.strictness_readout, text=TIPS['strictness'])
        ttk.Label(
            match_opts,
            text='Loose accepts minor tag differences (extra artists, & vs and, spacing). Strict requires near-exact tags.',
            style='Dim.TLabel', wraplength=520,
        ).grid(row=1, column=0, columnspan=2, sticky='w', pady=(2, 0))
        fallback_chk = ttk.Checkbutton(
            match_opts, text='Use filename when tags are missing',
            variable=self.use_filename_fallback,
        )
        fallback_chk.grid(row=2, column=0, columnspan=2, sticky='w', pady=(8, 0))
        tip(fallback_chk, text=TIPS['filename_fallback'])

        ignore_opts = ttk.LabelFrame(left, text='  IGNORE WHEN MATCHING  ', padding=SECTION_INNER_PAD)
        ignore_opts.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        ignore_opts.columnconfigure(1, weight=1)
        paren_chk = ttk.Checkbutton(
            ignore_opts, text='Text in (parentheses)',
            variable=self.ignore_parentheses, command=self._sync_bracket_checks,
        )
        paren_chk.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 4))
        tip(paren_chk, text=TIPS['ignore_parentheses'])
        square_chk = ttk.Checkbutton(
            ignore_opts, text='Text in [square brackets]',
            variable=self.ignore_square_brackets, command=self._sync_bracket_checks,
        )
        square_chk.grid(row=1, column=0, columnspan=2, sticky='w', pady=(0, 4))
        tip(square_chk, text=TIPS['ignore_square'])
        both_chk = ttk.Checkbutton(
            ignore_opts, text='Both ( ) and [ ]',
            variable=self.ignore_all_brackets, command=self._on_all_brackets_toggle,
        )
        both_chk.grid(row=2, column=0, columnspan=2, sticky='w', pady=(0, 4))
        tip(both_chk, text=TIPS['ignore_all_brackets'])
        spaces_chk = ttk.Checkbutton(
            ignore_opts, text='Extra spaces', variable=self.ignore_extra_spaces,
        )
        spaces_chk.grid(row=3, column=0, columnspan=2, sticky='w', pady=(0, 8))
        tip(spaces_chk, text=TIPS['ignore_spaces'])
        ttk.Label(ignore_opts, text='Custom keywords', style='Dim.TLabel').grid(
            row=4, column=0, columnspan=2, sticky='w', pady=(0, 4),
        )
        self._custom_keywords_frame = ttk.Frame(ignore_opts)
        self._custom_keywords_frame.grid(row=5, column=0, columnspan=2, sticky='ew')
        self._custom_keywords_frame.columnconfigure(0, weight=1)
        keyword_actions = tk.Frame(ignore_opts, bg=COLORS['bg'])
        keyword_actions.grid(row=6, column=0, columnspan=2, sticky='w', pady=(6, 0))
        self._add_keyword_btn = self._icon_button(
            keyword_actions, '+', self._add_custom_keyword_row, accent=True, width=3,
        )
        self._add_keyword_btn.pack(side='left')
        tip(self._add_keyword_btn, text=TIPS['add_keyword'])

    def attach_action_bar(self, parent: tk.Misc) -> None:
        """Place Match & Align actions in the host bottom bar (shared with Classify)."""
        self.find_btn = self._action_button(
            parent, '▶  Find pairs', self._start_find_pairs, accent=True,
        )
        self.organize_btn = self._action_button(
            parent, '▶  Organize folder', self._start_organize,
        )
        C = COLORS
        self.play_stems_btn = tk.Button(
            parent, text='♫  Play', command=self._open_stem_player,
            font=ACTION_BTN_FONT, bg=C['panel2'], fg=C['fg_dim'],
            activebackground=C['accent_hov'], activeforeground='white',
            relief='flat', borderwidth=0, highlightthickness=0,
            padx=ACTION_BTN_PADX, pady=ACTION_BTN_PADY, cursor='hand2',
        )
        tip(self.play_stems_btn, text=TIPS['align_play_stems'])

        def _play_enter(_e=None):
            self.play_stems_btn.configure(bg=C['accent'], fg='white')

        def _play_leave(_e=None):
            self.play_stems_btn.configure(bg=C['panel2'], fg=C['fg_dim'])

        self.play_stems_btn.bind('<Enter>', _play_enter, add='+')
        self.play_stems_btn.bind('<Leave>', _play_leave, add='+')

    def _align_tab_active(self) -> bool:
        return self._notebook.index(self._notebook.select()) == 1

    def _on_pair_subtab_changed(self, _event=None) -> None:
        host = self._host
        if hasattr(host, '_classify_mode_active') and not host._classify_mode_active():
            self.show_action_bar()
            if not self._busy:
                self.set_buttons_state('normal')

    def show_action_bar(self) -> None:
        self.find_btn.pack_forget()
        self.organize_btn.pack_forget()
        self.play_stems_btn.pack_forget()
        if self._align_tab_active():
            self.play_stems_btn.pack(side='right')
        else:
            self.find_btn.pack(side='left')
            self.organize_btn.pack(side='left', padx=(8, 0))

    def hide_action_bar(self) -> None:
        self.find_btn.pack_forget()
        self.organize_btn.pack_forget()
        self.play_stems_btn.pack_forget()

    def _build_align_tab(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill='x', padx=SECTION_PADX, pady=(HEADER_TOP_PAD, 12))
        tk.Label(
            header,
            text='Align instrumental and acapella to the original song. The original is the master timeline; silence is added or the start is trimmed so stems line up with it.',
            font=HEADER_DESC_FONT, fg=COLORS['fg'], bg=COLORS['bg'],
            wraplength=520, justify='left',
        ).pack(anchor='w')

        library = ttk.LabelFrame(parent, text='  STEM LIBRARY  ', padding=SECTION_INNER_PAD)
        library.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        library.columnconfigure(1, weight=1)
        self._path_row(
            library, 0, 'Stems root', self.align_stems_root,
            self._pick_align_stems_root, self._open_align_stems_root, TIPS['align_stems_root'],
        )

        step_export = ttk.LabelFrame(parent, text='  1  EXPORT  ', padding=SECTION_INNER_PAD)
        step_export.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        step_export.columnconfigure(1, weight=1)
        self._file_row(
            step_export, 0, 'Export list', self.align_export_file,
            self._pick_align_export, TIPS['align_export'],
            self._open_align_export,
        )
        export_actions = tk.Frame(step_export, bg=COLORS['bg'])
        export_actions.grid(row=1, column=0, columnspan=4, sticky='w', pady=(8, 0))
        self.export_list_btn = self._action_button(
            export_actions, 'Export song list', self._start_export_list,
        )
        self.export_list_btn.pack(side='left')

        step_distribute = ttk.LabelFrame(parent, text='  2  DISTRIBUTE  ', padding=SECTION_INNER_PAD)
        step_distribute.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        step_distribute.columnconfigure(1, weight=1)
        self._path_row(
            step_distribute, 0, 'Originals inbox', self.align_originals_inbox,
            self._pick_align_inbox, self._open_align_inbox, TIPS['align_inbox'],
        )
        distribute_opts = ttk.Frame(step_distribute)
        distribute_opts.grid(row=1, column=0, columnspan=4, sticky='w', pady=(4, 0))
        sort_chk = ttk.Checkbutton(distribute_opts, text='Sort after distribute', variable=self.align_sort_after)
        sort_chk.pack(side='left')
        tip(sort_chk, text=TIPS['align_sort_after'])
        distribute_actions = tk.Frame(step_distribute, bg=COLORS['bg'])
        distribute_actions.grid(row=2, column=0, columnspan=4, sticky='w', pady=(8, 0))
        self.distribute_btn = self._action_button(
            distribute_actions, 'Distribute originals', self._start_distribute_originals,
        )
        self.distribute_btn.pack(side='left')

        step_sort = ttk.LabelFrame(parent, text='  3  SORT  ', padding=SECTION_INNER_PAD)
        step_sort.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        step_sort.columnconfigure(1, weight=1)
        self._path_row(
            step_sort, 0, 'With original', self.align_with_original_dir,
            lambda: self._pick_align_sort_dir(self.align_with_original_dir, 'With original folder'),
            lambda: self._open_folder(self.align_with_original_dir),
            TIPS['align_with_original'],
        )
        self._path_row(
            step_sort, 1, 'Without original', self.align_without_original_dir,
            lambda: self._pick_align_sort_dir(self.align_without_original_dir, 'Without original folder'),
            lambda: self._open_folder(self.align_without_original_dir),
            TIPS['align_without_original'],
        )
        sort_actions = tk.Frame(step_sort, bg=COLORS['bg'])
        sort_actions.grid(row=2, column=0, columnspan=4, sticky='w', pady=(8, 0))
        self.sort_folders_btn = self._action_button(
            sort_actions, 'Sort folders', self._start_sort_folders,
        )
        self.sort_folders_btn.pack(side='left')

        step_align = ttk.LabelFrame(parent, text='  4  ALIGN  ', padding=SECTION_INNER_PAD)
        step_align.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        align_opts = ttk.Frame(step_align)
        align_opts.pack(anchor='w')
        backup_chk = ttk.Checkbutton(align_opts, text='Backup stems before align', variable=self.align_backup)
        backup_chk.pack(side='left', padx=(0, 16))
        tip(backup_chk, text=TIPS['align_backup'])
        skip_chk = ttk.Checkbutton(
            align_opts, text='Skip if output already exists', variable=self.align_skip_existing,
        )
        skip_chk.pack(side='left', padx=(0, 16))
        tip(skip_chk, text=TIPS['align_skip_existing'])
        ttk.Label(align_opts, text='Analysis').pack(side='left', padx=(0, 6))
        analysis_sp = ttk.Spinbox(align_opts, from_=10, to=120, textvariable=self.align_analysis_sec, width=5)
        analysis_sp.pack(side='left')
        ttk.Label(align_opts, text='seconds', style='Dim.TLabel').pack(side='left', padx=(8, 0))
        tip(analysis_sp, text=TIPS['align_analysis'])
        align_actions = tk.Frame(step_align, bg=COLORS['bg'])
        align_actions.pack(anchor='w', pady=(8, 0))
        self.align_btn = self._action_button(
            align_actions, 'Align stems', self._start_align_stems, accent=True,
        )
        self.align_btn.pack(side='left')

        tk.Frame(parent, bg=COLORS['bg'], height=ACTIONS_BOTTOM_PAD).pack(side='bottom')

    def _file_row(self, parent, row, label, var, picker, tip_text, opener=None):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky='w', padx=(0, 10), pady=CTRL_ROW_PADY)
        ent = ttk.Entry(parent, textvariable=var)
        ent.grid(row=row, column=1, sticky='ew', pady=CTRL_ROW_PADY)
        ent.bind('<FocusOut>', lambda _e, v=var: self._normalize_path_var(v))
        browse_btn = self._path_button(parent, 'Browse', picker)
        browse_btn.grid(row=row, column=2, padx=(4, 0), pady=CTRL_ROW_PADY)
        widgets = [lbl, ent, browse_btn]
        if opener is not None:
            open_btn = self._path_button(parent, 'Open', opener)
            open_btn.grid(row=row, column=3, padx=(4, 0), pady=CTRL_ROW_PADY)
            widgets.append(open_btn)
            Tooltip(open_btn, TIPS['open_path'])
        tip(*widgets, text=tip_text)

    def _strictness_label(self) -> str:
        value = float(self.strictness.get())
        threshold = strictness_to_threshold(value)
        if value >= 90:
            label = 'Strict'
        elif value >= 65:
            label = 'Balanced'
        elif value >= 35:
            label = 'Loose'
        else:
            label = 'Very loose'
        return f'{label} ({threshold:.0%})'

    def _icon_button(
        self,
        parent,
        text,
        command,
        *,
        accent: bool = False,
        width: int | None = None,
    ) -> tk.Button:
        C = COLORS
        if accent:
            btn = tk.Button(
                parent, text=text, command=command,
                font=('Segoe UI Semibold', 11), bg=C['accent'], fg='white',
                activebackground=C['accent_hov'], activeforeground='white',
                relief='flat', borderwidth=0, highlightthickness=0,
                padx=8, pady=2, cursor='hand2',
            )
        else:
            btn = tk.Button(
                parent, text=text, command=command,
                font=('Segoe UI', 10), bg=C['panel2'], fg=C['fg_dim'],
                activebackground=C['panel'], activeforeground=C['danger'],
                relief='flat', borderwidth=0, highlightthickness=0,
                padx=6, pady=2, cursor='hand2',
            )
        if width is not None:
            btn.configure(width=width)
        return btn

    def _on_all_brackets_toggle(self) -> None:
        if self.ignore_all_brackets.get():
            self.ignore_parentheses.set(True)
            self.ignore_square_brackets.set(True)

    def _sync_bracket_checks(self) -> None:
        if self.ignore_parentheses.get() and self.ignore_square_brackets.get():
            self.ignore_all_brackets.set(True)
        else:
            self.ignore_all_brackets.set(False)

    def _add_custom_keyword_row(self, value: str = '') -> None:
        row = ttk.Frame(self._custom_keywords_frame)
        row.pack(fill='x', pady=2)
        row.columnconfigure(0, weight=1)
        var = tk.StringVar(value=value)
        entry = ttk.Entry(row, textvariable=var)
        entry.grid(row=0, column=0, sticky='ew')
        var.trace_add('write', lambda *_: self._save_settings())
        remove_btn = self._icon_button(
            row, '×', lambda r=row, v=var: self._remove_custom_keyword_row(r, v),
        )
        remove_btn.grid(row=0, column=1, padx=(4, 0))
        self._custom_keyword_vars.append(var)

    def _remove_custom_keyword_row(self, row: ttk.Frame, var: tk.StringVar) -> None:
        if var in self._custom_keyword_vars:
            self._custom_keyword_vars.remove(var)
        row.destroy()
        self._save_settings()

    def _clear_custom_keyword_rows(self) -> None:
        for widget in self._custom_keywords_frame.winfo_children():
            widget.destroy()
        self._custom_keyword_vars.clear()

    def _load_custom_keyword_rows(self, keywords: list[str]) -> None:
        self._clear_custom_keyword_rows()
        for keyword in keywords:
            if keyword.strip():
                self._add_custom_keyword_row(keyword.strip())

    def _get_ignore_rules(self) -> IgnoreRules:
        keywords = tuple(
            var.get().strip()
            for var in self._custom_keyword_vars
            if var.get().strip()
        )
        return IgnoreRules(
            ignore_parentheses=bool(self.ignore_parentheses.get()),
            ignore_square_brackets=bool(self.ignore_square_brackets.get()),
            ignore_all_brackets=bool(self.ignore_all_brackets.get()),
            ignore_extra_spaces=bool(self.ignore_extra_spaces.get()),
            custom_keywords=keywords,
        )

    def _action_button(self, parent, text, command, *, accent: bool = False) -> tk.Button:
        C = COLORS
        if accent:
            return tk.Button(
                parent, text=text, command=command,
                font=ACTION_BTN_FONT, bg=C['accent'], fg='white',
                activebackground=C['accent_hov'], activeforeground='white',
                relief='flat', borderwidth=0, highlightthickness=0,
                padx=ACTION_BTN_PADX, pady=ACTION_BTN_PADY, cursor='hand2',
            )
        return tk.Button(
            parent, text=text, command=command,
            font=ACTION_BTN_FONT, bg=C['panel2'], fg=C['fg'],
            activebackground=C['panel'], activeforeground=C['fg'],
            relief='flat', borderwidth=0, highlightthickness=0,
            padx=ACTION_BTN_PADX, pady=ACTION_BTN_PADY, cursor='hand2',
        )

    def _path_button(self, parent: tk.Misc, text: str, command) -> tk.Button:
        C = COLORS
        return tk.Button(
            parent, text=text, command=command,
            font=PATH_BTN_FONT, bg=C['panel2'], fg=C['fg'],
            activebackground=C['panel'], activeforeground=C['fg'],
            relief='flat', borderwidth=0, highlightthickness=0,
            padx=PATH_BTN_PADX, pady=PATH_BTN_PADY, cursor='hand2',
        )

    def _path_row(self, parent, row, label, var, picker, opener, tip_text):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky='w', padx=(0, 10), pady=CTRL_ROW_PADY)
        ent = ttk.Entry(parent, textvariable=var)
        ent.grid(row=row, column=1, sticky='ew', pady=CTRL_ROW_PADY)
        ent.bind('<FocusOut>', lambda _e, v=var: self._normalize_path_var(v))
        browse_btn = self._path_button(parent, 'Browse', picker)
        browse_btn.grid(row=row, column=2, padx=(4, 0), pady=CTRL_ROW_PADY)
        open_btn = self._path_button(parent, 'Open', opener)
        open_btn.grid(row=row, column=3, padx=(4, 0), pady=CTRL_ROW_PADY)
        tip(lbl, ent, browse_btn, text=tip_text)
        Tooltip(open_btn, TIPS['open_path'])

    def _normalize_path_var(self, var: tk.StringVar) -> None:
        normalized = display_path(var.get())
        if normalized != var.get():
            var.set(normalized)

    def _pick_dir(self, var: tk.StringVar, title: str) -> None:
        initial = var.get().strip()
        path = filedialog.askdirectory(title=title, initialdir=initial or None)
        if path:
            var.set(display_path(path))

    def _pick_acapella(self) -> None:
        self._pick_dir(self.acapella_dir, 'Select acapella folder')

    def _pick_instrumental(self) -> None:
        self._pick_dir(self.instrumental_dir, 'Select instrumental folder')

    def _pick_pairs_output(self) -> None:
        self._pick_dir(self.pairs_output_dir, 'Select pairs output folder')

    def _pick_organize(self) -> None:
        self._pick_dir(self.organize_dir, 'Select folder to organize')

    def _open_folder(self, var: tk.StringVar) -> None:
        path = Path(var.get().strip())
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            messagebox.showwarning(PANEL_TITLE, 'Folder does not exist.')
            return
        if sys.platform == 'win32':
            os.startfile(path)  # noqa: S606
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(path)], check=False)
        else:
            subprocess.run(['xdg-open', str(path)], check=False)

    def _open_acapella(self) -> None:
        self._open_folder(self.acapella_dir)

    def _open_instrumental(self) -> None:
        self._open_folder(self.instrumental_dir)

    def _open_pairs_output(self) -> None:
        self._open_folder(self.pairs_output_dir)

    def _open_organize(self) -> None:
        self._open_folder(self.organize_dir)

    def _pick_align_stems_root(self) -> None:
        self._pick_dir(self.align_stems_root, 'Select stems root folder')

    def _sync_align_sort_dirs(self, *_args) -> None:
        root_s = self.align_stems_root.get().strip()
        if not root_s:
            return
        root = Path(root_s)
        self.align_with_original_dir.set(display_path(str(default_with_original_dir(root))))
        self.align_without_original_dir.set(display_path(str(default_without_original_dir(root))))

    def _pick_align_inbox(self) -> None:
        self._pick_dir(self.align_originals_inbox, 'Select originals inbox folder')

    def _pick_align_export(self) -> None:
        path = filedialog.asksaveasfilename(
            title='Export song list',
            defaultextension='.txt',
            initialfile='songs_to_download.txt',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
        )
        if path:
            self.align_export_file.set(display_path(path))

    def _pick_align_sort_dir(self, var: tk.StringVar, title: str) -> None:
        self._pick_dir(var, title)

    def _align_sort_dirs(self, root: Path) -> tuple[Path, Path]:
        with_raw = self.align_with_original_dir.get().strip()
        without_raw = self.align_without_original_dir.get().strip()
        return (
            Path(with_raw) if with_raw else default_with_original_dir(root),
            Path(without_raw) if without_raw else default_without_original_dir(root),
        )

    def _open_align_stems_root(self) -> None:
        self._open_folder(self.align_stems_root)

    def _open_stem_player(self) -> None:
        open_stem_player(self._host)

    def _open_align_inbox(self) -> None:
        self._open_folder(self.align_originals_inbox)

    def _open_align_export(self) -> None:
        self._open_folder(self.align_export_file)

    def _start_export_list(self) -> None:
        if self._busy or self._host._organize_worker_active():
            return
        root = Path(self.align_stems_root.get().strip())
        export_path = Path(self.align_export_file.get().strip())
        if not root.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Stems root folder is missing or invalid.')
            return
        if not export_path.parent.exists():
            messagebox.showerror(PANEL_TITLE, 'Export list folder does not exist.')
            return
        self._clear_log()
        self._set_busy(True, 'Exporting song list…')
        self._worker = threading.Thread(
            target=self._run_export_list, args=(root, export_path), daemon=True,
        )
        self._worker.start()

    def _run_export_list(self, root: Path, export_path: Path) -> None:
        try:
            count = export_song_list(root, export_path)
            self._log(f'Exported {count:,} song folder name(s) to {export_path}', 'ok')
            self._finish_worker(f'Done · {count:,} name(s) exported')
        except Exception:
            self._log(traceback.format_exc(), 'err')
            self._finish_worker('Failed')

    def _start_distribute_originals(self) -> None:
        if self._busy or self._host._organize_worker_active():
            return
        inbox = Path(self.align_originals_inbox.get().strip())
        root = Path(self.align_stems_root.get().strip())
        if not inbox.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Originals inbox folder is missing or invalid.')
            return
        if not root.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Stems root folder is missing or invalid.')
            return
        self._clear_log()
        self._set_busy(True, 'Distributing originals…')
        self._worker = threading.Thread(
            target=self._run_distribute_originals,
            args=(
                inbox, root,
                bool(self.align_sort_after.get()),
                *self._align_sort_dirs(root),
            ),
            daemon=True,
        )
        self._worker.start()

    def _run_distribute_originals(
        self,
        inbox: Path,
        root: Path,
        sort_after: bool,
        with_dir: Path,
        without_dir: Path,
    ) -> None:
        try:
            moved, skipped, unmatched, rejected, sorted_with, sorted_without = distribute_originals(
                inbox, root,
                on_log=self._report_log,
                on_progress=self._report_progress,
                sort_after=sort_after,
                with_original_dir=with_dir,
                without_original_dir=without_dir,
            )
            self._log(
                f'Done · moved {moved:,} · skipped {skipped:,} · unmatched {unmatched:,} · '
                f'rejected {rejected:,}',
                'info',
            )
            if sort_after:
                self._log(
                    f'Sorted · with original {sorted_with:,} · without {sorted_without:,}',
                    'info',
                )
            self._finish_worker(f'Done · {moved:,} moved')
        except Exception:
            self._log(traceback.format_exc(), 'err')
            self._finish_worker('Failed')

    def _start_sort_folders(self) -> None:
        if self._busy or self._host._organize_worker_active():
            return
        root = Path(self.align_stems_root.get().strip())
        if not root.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Stems root folder is missing or invalid.')
            return
        self._clear_log()
        self._set_busy(True, 'Sorting folders…')
        with_dir, without_dir = self._align_sort_dirs(root)
        self._worker = threading.Thread(
            target=self._run_sort_folders,
            args=(root, with_dir, without_dir),
            daemon=True,
        )
        self._worker.start()

    def _run_sort_folders(
        self,
        root: Path,
        with_dir: Path,
        without_dir: Path,
    ) -> None:
        try:
            moved_with, moved_without, skipped = sort_folders_by_original(
                root,
                with_original_dir=with_dir,
                without_original_dir=without_dir,
                on_log=self._report_log,
                on_progress=self._report_progress,
            )
            self._log(
                f'Done · with original {moved_with:,} · without {moved_without:,} · skipped {skipped:,}',
                'info',
            )
            self._finish_worker(f'Done · {moved_with + moved_without:,} sorted')
        except Exception:
            self._log(traceback.format_exc(), 'err')
            self._finish_worker('Failed')

    def _start_align_stems(self) -> None:
        if self._busy or self._host._organize_worker_active():
            return
        root = Path(self.align_stems_root.get().strip())
        if not root.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Stems root folder is missing or invalid.')
            return
        try:
            analysis_sec = float(self.align_analysis_sec.get())
        except (tk.TclError, ValueError, TypeError):
            messagebox.showerror(PANEL_TITLE, 'Analysis seconds must be a number.')
            return
        self._clear_log()
        self._set_busy(True, 'Aligning stems…')
        with_dir, _without_dir = self._align_sort_dirs(root)
        self._worker = threading.Thread(
            target=self._run_align_stems,
            args=(root, with_dir, analysis_sec, bool(self.align_backup.get()), bool(self.align_skip_existing.get())),
            daemon=True,
        )
        self._worker.start()

    def _run_align_stems(
        self,
        root: Path,
        with_dir: Path,
        analysis_sec: float,
        backup: bool,
        skip_existing: bool,
    ) -> None:
        try:
            results, skipped = align_all_songs(
                root,
                with_original_dir=with_dir,
                analysis_sec=analysis_sec,
                backup=backup,
                skip_existing=skip_existing,
                on_log=self._report_log,
                on_progress=self._report_progress,
            )
            self._log(f'Aligned {len(results):,} folder(s)', 'ok')
            if skipped:
                self._log(f'Skipped {skipped:,} folder(s) (already aligned)', 'warn')
            self._finish_worker(f'Done · {len(results):,} aligned')
        except Exception:
            self._log(traceback.format_exc(), 'err')
            self._finish_worker('Failed')

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        self._host._set_pair_busy(busy, status, self)

    def set_buttons_state(self, state: str) -> None:
        for btn in (
            self.find_btn, self.organize_btn,
            self.export_list_btn, self.distribute_btn, self.sort_folders_btn,
            self.align_btn, self.play_stems_btn,
        ):
            btn.configure(state=state)

    def _start_find_pairs(self) -> None:
        if self._busy or self._host._organize_worker_active():
            return
        acapella = Path(self.acapella_dir.get().strip())
        instrumental = Path(self.instrumental_dir.get().strip())
        output = Path(self.pairs_output_dir.get().strip())
        if not acapella.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Acapella folder is missing or invalid.')
            return
        if not instrumental.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Instrumental folder is missing or invalid.')
            return
        if not output.exists():
            try:
                output.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror(PANEL_TITLE, f'Could not create output folder:\n{exc}')
                return
        ref_acapella = self.reference_side.get() == 'acapella'
        strictness = float(self.strictness.get())
        use_filename_fallback = bool(self.use_filename_fallback.get())
        include_subfolders = bool(self.include_subfolders.get())
        ignore_rules = self._get_ignore_rules()
        self._clear_log()
        self._log('Finding pairs…', 'info')
        self._set_busy(True, 'Finding pairs…')
        self._worker = threading.Thread(
            target=self._run_find_pairs,
            args=(
                acapella, instrumental, output, ref_acapella, strictness,
                use_filename_fallback, include_subfolders, ignore_rules,
            ),
            daemon=True,
        )
        self._worker.start()

    def _run_find_pairs(
        self,
        acapella: Path,
        instrumental: Path,
        output: Path,
        ref_acapella: bool,
        strictness: float,
        use_filename_fallback: bool,
        include_subfolders: bool,
        ignore_rules: IgnoreRules,
    ) -> None:
        try:
            if ref_acapella:
                reference_dir, partner_dir = acapella, instrumental
                ref_label, partner_label = 'acapella', 'instrumental'
            else:
                reference_dir, partner_dir = instrumental, acapella
                ref_label, partner_label = 'instrumental', 'acapella'

            result = find_pairs(
                reference_dir,
                partner_dir,
                reference_is_acapella=ref_acapella,
                strictness=strictness,
                use_filename_fallback=use_filename_fallback,
                ignore_rules=ignore_rules,
                include_subfolders=include_subfolders,
                move_to=output,
                on_progress=self._report_progress,
                on_log=self._report_log,
            )
            threshold = strictness_to_threshold(strictness)
            fallback_note = 'tags + filename' if use_filename_fallback else 'tags only'
            self._log(
                f'Done · {ref_label} reference · {fallback_note} · threshold {threshold:.0%} · '
                f'{len(result.pairs):,} pair(s) · '
                f'{len(result.unmatched_reference):,} unmatched reference · '
                f'{len(result.unmatched_partner):,} unmatched partner',
                'info',
            )
            if result.pairs:
                show = min(25, len(result.pairs))
                for match in result.pairs[:show]:
                    self._log(
                        f'✓ {match.reference.display_name}  ↔  {match.partner.path.name}  ({match.score:.0%})',
                        'ok',
                    )
                if len(result.pairs) > show:
                    self._log(f'… and {len(result.pairs) - show:,} more pair(s)', 'info')

            if result.unmatched_reference:
                self._log(f'Unmatched {ref_label}: {len(result.unmatched_reference)}', 'warn')
                for track in result.unmatched_reference[:20]:
                    self._log(f'  · {track.path.name}', 'warn')
                if len(result.unmatched_reference) > 20:
                    self._log(f'  … and {len(result.unmatched_reference) - 20} more', 'warn')

            if result.unmatched_partner:
                self._log(f'Unmatched {partner_label}: {len(result.unmatched_partner)}', 'warn')
                for track in result.unmatched_partner[:20]:
                    self._log(f'  · {track.path.name}', 'warn')
                if len(result.unmatched_partner) > 20:
                    self._log(f'  … and {len(result.unmatched_partner) - 20} more', 'warn')

            self._finish_worker(f'Done · {len(result.pairs)} pair(s) moved')
        except Exception:
            self._log(traceback.format_exc(), 'err')
            self._finish_worker('Failed')

    def _start_organize(self) -> None:
        if self._busy or self._host._organize_worker_active():
            return
        folder = Path(self.organize_dir.get().strip())
        if not folder.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Organize folder is missing or invalid.')
            return
        strictness = float(self.strictness.get())
        use_filename_fallback = bool(self.use_filename_fallback.get())
        include_subfolders = bool(self.include_subfolders.get())
        ignore_rules = self._get_ignore_rules()
        self._clear_log()
        self._log('Organizing matched files…', 'info')
        self._set_busy(True, 'Organizing…')
        self._worker = threading.Thread(
            target=self._run_organize,
            args=(folder, strictness, use_filename_fallback, include_subfolders, ignore_rules),
            daemon=True,
        )
        self._worker.start()

    def _run_organize(
        self,
        folder: Path,
        strictness: float,
        use_filename_fallback: bool,
        include_subfolders: bool,
        ignore_rules: IgnoreRules,
    ) -> None:
        try:
            moved = organize_matched_folder(
                folder,
                strictness=strictness,
                use_filename_fallback=use_filename_fallback,
                ignore_rules=ignore_rules,
                include_subfolders=include_subfolders,
                on_progress=self._report_progress,
                on_log=self._report_log,
            )
            if not moved:
                self._log('No FLAC or MP3 files found in the selected folder.', 'warn')
                self._finish_worker('Nothing to organize')
                return
            self._log(f'Created {len(moved):,} folder(s)', 'ok')
            show = min(25, len(moved))
            for dest_dir, files in moved[:show]:
                names = ', '.join(p.name for p in files)
                self._log(f'✓ {dest_dir.name}/  →  {names}', 'ok')
            if len(moved) > show:
                self._log(f'… and {len(moved) - show:,} more folder(s)', 'info')
            self._finish_worker(f'Done · {len(moved)} folder(s) created')
        except Exception:
            self._log(traceback.format_exc(), 'err')
            self._finish_worker('Failed')

    def _finish_worker(self, status: str) -> None:
        self.after(0, lambda: self._set_busy(False, status))

    def _clear_log(self) -> None:
        self._host._clear_log()

    def _report_log(self, message: str, tag: str = 'info') -> None:
        self._host.log_queue.put((PAIR_LOG_TAG, message, tag))

    def _log(self, message: str, tag: str = 'info') -> None:
        self._host.log_queue.put((PAIR_LOG_TAG, message, tag))

    def _report_progress(self, done: int, total: int, message: str) -> None:
        pct = (done / total * 100.0) if total else 0.0
        eta = None
        if done > 0 and total > done:
            started = getattr(self._host, '_progress_started_at', 0.0)
            if started:
                elapsed = time.monotonic() - started
                eta = elapsed / done * (total - done)
        self._host.log_queue.put(('__progress__', pct, eta))
        if message:
            self._host.after(0, lambda m=message: self._host.status_var.set(m))

    def settings_snapshot(self) -> dict:
        return {
            'acapella_dir': self.acapella_dir.get(),
            'instrumental_dir': self.instrumental_dir.get(),
            'pairs_output_dir': self.pairs_output_dir.get(),
            'organize_dir': self.organize_dir.get(),
            'reference_side': self.reference_side.get(),
            'strictness': float(self.strictness.get()),
            'use_filename_fallback': bool(self.use_filename_fallback.get()),
            'include_subfolders': bool(self.include_subfolders.get()),
            'ignore_rules': self._get_ignore_rules().to_dict(),
            'align_stems_root': self.align_stems_root.get(),
            'align_originals_inbox': self.align_originals_inbox.get(),
            'align_export_file': self.align_export_file.get(),
            'align_backup': bool(self.align_backup.get()),
            'align_skip_existing': bool(self.align_skip_existing.get()),
            'align_analysis_sec': int(self.align_analysis_sec.get()),
            'align_sort_after': bool(self.align_sort_after.get()),
            'align_with_original_dir': self.align_with_original_dir.get(),
            'align_without_original_dir': self.align_without_original_dir.get(),
        }

    def _load_settings(self) -> None:
        data = load_settings()
        self.acapella_dir.set(display_path(str(data.get('acapella_dir', ''))))
        self.instrumental_dir.set(display_path(str(data.get('instrumental_dir', ''))))
        self.pairs_output_dir.set(display_path(str(data.get('pairs_output_dir', ''))))
        self.organize_dir.set(display_path(str(data.get('organize_dir', ''))))
        ref = data.get('reference_side', 'acapella')
        if ref in ('acapella', 'instrumental'):
            self.reference_side.set(ref)
        try:
            self.strictness.set(float(data.get('strictness', 75.0)))
        except (TypeError, ValueError):
            pass
        self.use_filename_fallback.set(bool(data.get('use_filename_fallback', True)))
        self.include_subfolders.set(bool(data.get('include_subfolders', False)))
        rules = IgnoreRules.from_dict(data.get('ignore_rules'))
        self.ignore_parentheses.set(rules.ignore_parentheses)
        self.ignore_square_brackets.set(rules.ignore_square_brackets)
        self.ignore_all_brackets.set(rules.ignore_all_brackets)
        self.ignore_extra_spaces.set(rules.ignore_extra_spaces)
        self._load_custom_keyword_rows(list(rules.custom_keywords))
        self.strictness_readout.configure(text=self._strictness_label())
        self.align_stems_root.set(display_path(str(data.get('align_stems_root', ''))))
        self._sync_align_sort_dirs()
        self.align_originals_inbox.set(display_path(str(data.get('align_originals_inbox', ''))))
        export = data.get('align_export_file')
        if export:
            self.align_export_file.set(display_path(str(export)))
        self.align_backup.set(bool(data.get('align_backup', True)))
        self.align_skip_existing.set(bool(data.get('align_skip_existing', True)))
        try:
            self.align_analysis_sec.set(int(data.get('align_analysis_sec', 30)))
        except (TypeError, ValueError):
            pass
        self.align_sort_after.set(bool(data.get('align_sort_after', True)))

    def _save_settings(self) -> None:
        data = load_settings()
        data.update(self.settings_snapshot())
        save_settings(data)

    def _bind_settings_autosave(self) -> None:
        self.align_stems_root.trace_add('write', self._sync_align_sort_dirs)
        for var in (
            self.acapella_dir, self.instrumental_dir, self.pairs_output_dir,
            self.organize_dir, self.reference_side, self.strictness,
            self.use_filename_fallback, self.include_subfolders,
            self.ignore_parentheses,
            self.ignore_square_brackets, self.ignore_all_brackets,
            self.ignore_extra_spaces,
            self.align_stems_root, self.align_originals_inbox, self.align_export_file,
            self.align_backup, self.align_skip_existing, self.align_analysis_sec, self.align_sort_after,
            self.align_with_original_dir, self.align_without_original_dir,
        ):
            var.trace_add('write', lambda *_: self._save_settings())
