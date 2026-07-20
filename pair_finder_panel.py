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
from tkinter import filedialog, messagebox

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
    resolve_export_list_path,
    sort_folders_by_original,
)
from stem_player import open_stem_player
from ui_theme import (
    ACTION_BTN_GAP,
    COLORS,
    CTRL_ROW_PADY,
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
    ctk_section_font,
    ctk_size_subtab_buttons,
    ctk_ui_font,
    display_path,
    ensure_ctk_dark,
    show_ctk_help_dialog,
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
    'pairs_output': (
        'Matched pairs are moved here (flat). Organize folder then groups them '
        'into Artist - Title subfolders inside this same folder.'
    ),
    'strictness': 'Higher = stricter tag matching. Compares artist and title separately; both must meet the threshold.',
    'filename_fallback': (
        'Parse artist/title from the filename only (e.g. Artist - Title (Acapella).mp3), '
        'even when metadata tags exist. Off = use tags only.'
    ),
    'ignore_parentheses': 'Strip text inside (parentheses) before comparing tags.',
    'ignore_square': 'Strip text inside [square brackets] before comparing tags.',
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
    'align_with_original': 'Folders that received an original song (auto-filled as stems root / with_original).',
    'align_play_stems': 'Open stem player on the with_original library (instrumental, acapella, original).',
    'align_without_original': 'Folders still missing an original (auto-filled as stems root / without_original).',
    'export_list_btn': 'Write song-folder names from Stems root into the export list file.',
    'distribute_btn': 'Copy originals from the inbox into matching song folders under Stems root.',
    'sort_folders_btn': 'Move song folders into with_original / without_original based on whether an original exists.',
    'align_stems_btn': 'Align acapella and instrumental stems to the original song timeline.',
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


ctk = ensure_ctk_dark()


class PairFinderPanel(ctk.CTkFrame):
    def __init__(self, host: tk.Misc, parent: tk.Misc, info_icon_factory=None) -> None:
        ensure_ctk_dark()
        super().__init__(parent, fg_color=DARK['bg'])
        self._host = host
        self._info_icon_factory = info_icon_factory
        self._ctk = ctk

        self.acapella_dir = tk.StringVar()
        self.instrumental_dir = tk.StringVar()
        self.pairs_output_dir = tk.StringVar()
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
        self.align_with_original_dir = tk.StringVar()
        self.align_without_original_dir = tk.StringVar()
        self.ignore_parentheses = tk.BooleanVar(value=True)
        self.ignore_square_brackets = tk.BooleanVar(value=True)
        self.ignore_extra_spaces = tk.BooleanVar(value=True)
        self._custom_keyword_vars: list[tk.StringVar] = []
        self._busy = False

        self._worker: threading.Thread | None = None

        self._build_ui()
        self._bind_settings_autosave()
        self._load_settings()

    def _build_ui(self) -> None:
        ctk = self._ctk
        t = DARK
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        tabview = ctk.CTkTabview(
            self,
            fg_color=t['bg'],
            segmented_button_fg_color=t['panel'],
            segmented_button_selected_color=t['accent'],
            segmented_button_selected_hover_color=t['accent_hover'],
            segmented_button_unselected_color=t['panel_2'],
            segmented_button_unselected_hover_color=t['border'],
            text_color=t['text'],
            anchor='w',
            command=self._on_pair_subtab_changed,
        )
        tabview.grid(row=0, column=0, sticky='nsew')
        self._tabview = tabview

        pair_tab = tabview.add('Match')
        align_tab = tabview.add('Align')
        ctk_size_subtab_buttons(tabview)
        self._build_pair_tab(pair_tab)
        self._build_align_tab(align_tab)

    def _make_info_icon(self, parent: tk.Misc, command) -> None:
        # CTkLabel is taller than the glyph; pin icon to top next to the text.
        pack_kw = dict(side='left', padx=(4, 0), anchor='n', pady=(5, 0))
        if self._info_icon_factory is not None:
            self._info_icon_factory(parent, command).pack(**pack_kw)
            return
        fallback = tk.Label(
            parent, text='?', font=('Segoe UI Semibold', 9),
            fg=COLORS['fg_dim'], bg=DARK['bg'], cursor='hand2',
        )
        fallback.pack(**pack_kw)
        fallback.bind('<Button-1>', lambda _event: command())
        Tooltip(fallback, 'Show more info/help.')

    def _show_match_help(self) -> None:
        show_ctk_help_dialog(
            self._host,
            title='Match help',
            heading='Match acapellas & instrumentals',
            intro=(
                'Find matching versions by artist and title, then turn them into an organized library.'
            ),
            sections=(
                (
                    'Workflow',
                    '1. Choose the Acapella and Instrumental source folders.\n'
                    '2. Choose where matched files should be moved.\n'
                    '3. Set the reference side and matching strictness.\n'
                    '4. Click Find pairs, review the log, then Organize folder.',
                ),
                (
                    'How matching works',
                    'Artist and title tags are compared separately. The reference folder drives the scan: '
                    'each reference file searches for its best partner in the other folder. '
                    '"Use filename instead" parses artist/title from the filename only, ignoring tags.',
                ),
                (
                    'Tune the result',
                    'Lower strictness accepts small differences such as extra artists, spacing, or “&” versus '
                    '“and”. Ignore rules remove brackets, extra spaces, and custom words before comparison; '
                    'they do not rename the source files.',
                ),
                (
                    'File safety',
                    'Find pairs moves only confirmed matches into Pairs output. Organize folder then groups '
                    'those files into Artist - Title subfolders inside Pairs output. '
                    'Unmatched files remain in their source folder.',
                ),
            ),
        )

    def _show_align_help(self) -> None:
        show_ctk_help_dialog(
            self._host,
            title='Align help',
            heading='Align stems to the original',
            intro=(
                'Use the original song as the master timeline for its instrumental and acapella.'
            ),
            sections=(
                (
                    'Required layout',
                    'Stems root contains one folder per song, with an instrumental and acapella inside. '
                    'Downloaded original songs first go into the Originals inbox.',
                ),
                (
                    'Four-step workflow',
                    '1. Export the song-folder names as a download list.\n'
                    '2. Put downloaded originals in the inbox and distribute them.\n'
                    '3. Sort song folders into with_original and without_original.\n'
                    '4. Align folders that have an original.',
                ),
                (
                    'How alignment works',
                    'The original song is the master timeline. Audio is analyzed to estimate the offset; silence '
                    'is added or the beginning is trimmed so the instrumental and acapella start at the correct time. '
                    'A longer analysis window can help difficult material but takes more time.',
                ),
                (
                    'File safety',
                    'Keep Backup stems before align enabled to preserve the untouched files in '
                    '_backup_before_align. Skip if output already exists makes interrupted batches safe to resume.',
                ),
            ),
        )

    def _description_with_info(
        self,
        parent: tk.Misc,
        text: str,
        command,
        *,
        final_line: str | None = None,
    ) -> None:
        """Description + ? icon snug after the last word.

        Single-line (final_line=None): shrink-wrapped text, icon immediately after.
        Two-line: first line alone; final_line + icon on the second row so ? sits
        after the last word even when the blurb is long (Align).
        """
        ctk = self._ctk
        desc_font = ctk_ui_font()
        if final_line is None:
            desc_row = ctk.CTkFrame(parent, fg_color='transparent')
            desc_row.pack(fill='x', anchor='w')
            ctk.CTkLabel(
                desc_row,
                text=text,
                font=desc_font,
                text_color=HEADER_DESC_COLOR,
                anchor='w',
            ).pack(side='left')
            self._make_info_icon(desc_row, command)
            return

        ctk.CTkLabel(
            parent,
            text=text,
            font=desc_font,
            text_color=HEADER_DESC_COLOR,
            anchor='w',
        ).pack(anchor='w')
        final_row = ctk.CTkFrame(parent, fg_color='transparent')
        final_row.pack(fill='x', anchor='w')
        ctk.CTkLabel(
            final_row,
            text=final_line,
            font=desc_font,
            text_color=HEADER_DESC_COLOR,
            anchor='w',
        ).pack(side='left')
        self._make_info_icon(final_row, command)

    def _build_pair_tab(self, left: tk.Misc) -> None:
        ctk = self._ctk
        t = DARK

        header = ctk.CTkFrame(left, fg_color='transparent')
        header.pack(fill='x', padx=SECTION_PADX, pady=(HEADER_TOP_PAD, 12))
        self._description_with_info(
            header,
            'Match acapella/instrumental files by artist/title tags, then organize pairs into song folders.',
            self._show_match_help,
        )

        paths = ctk_section(left, 'Folders')
        paths.grid_columnconfigure(1, weight=1)
        self._path_row(paths, 0, 'Acapella', self.acapella_dir, self._pick_acapella, self._open_acapella, TIPS['acapella'])
        self._path_row(paths, 1, 'Instrumental', self.instrumental_dir, self._pick_instrumental, self._open_instrumental, TIPS['instrumental'])
        self._path_row(paths, 2, 'Pairs output', self.pairs_output_dir, self._pick_pairs_output, self._open_pairs_output, TIPS['pairs_output'])

        ref_frame = ctk.CTkFrame(paths, fg_color='transparent')
        ref_frame.grid(row=3, column=0, columnspan=4, sticky='w', pady=(8, 0))
        _ui = ctk_ui_font()
        ctk.CTkLabel(
            ref_frame, text='Reference folder', text_color=t['label'], font=_ui,
        ).pack(side='left', padx=(0, 24))
        ref_widgets: list = []
        for idx, (label, value) in enumerate((
            ('Acapella', 'acapella'),
            ('Instrumental', 'instrumental'),
        )):
            opt = ctk.CTkFrame(ref_frame, fg_color='transparent')
            opt.pack(side='left', padx=(0, 28 if idx == 0 else 0))
            rb = ctk.CTkRadioButton(
                opt,
                text='',
                variable=self.reference_side,
                value=value,
                fg_color=t['accent'],
                border_color=t['border'],
                hover_color=t['accent_hover'],
                font=_ui,
                width=22,
                radiobutton_width=22,
                radiobutton_height=22,
            )
            rb.pack(side='left')
            ref_widgets.append(rb)

            def _pick(_event=None, v=value) -> None:
                self.reference_side.set(v)

            lbl = ctk.CTkLabel(
                opt, text=label, text_color=t['text'], font=_ui, cursor='hand2',
            )
            lbl.pack(side='left', padx=(6, 0))
            lbl.bind('<Button-1>', _pick)
            ref_widgets.append(lbl)
        tip(*ref_widgets, text=TIPS['reference'])

        subfolder_chk = ctk.CTkCheckBox(
            paths, text='Include subfolders', variable=self.include_subfolders,
            fg_color=t['accent'], hover_color=t['accent_hover'], text_color=t['text'],
            font=_ui,
        )
        subfolder_chk.grid(row=5, column=0, columnspan=4, sticky='w', pady=(8, 0))
        tip(subfolder_chk, text=TIPS['include_subfolders'])

        match_opts = ctk_section(left, 'Matching')
        match_opts.grid_columnconfigure(1, weight=1)
        strict_lbl = ctk.CTkLabel(
            match_opts, text='Tag strictness', text_color=t['label'], font=_ui,
        )
        strict_lbl.grid(row=0, column=0, sticky='w', padx=(0, 10), pady=6)
        strict_row = ctk.CTkFrame(match_opts, fg_color='transparent')
        strict_row.grid(row=0, column=1, sticky='ew', pady=6)
        strict_row.grid_columnconfigure(0, weight=1)
        self.strictness_readout = ctk.CTkLabel(
            strict_row, text=self._strictness_label(), text_color=t['text'],
            width=120, font=_ui,
        )
        self.strictness_scale = ctk.CTkSlider(
            strict_row, from_=0, to=100, variable=self.strictness,
            progress_color=t['accent'], button_color=t['accent'],
            button_hover_color=t['accent_hover'],
            command=lambda _v: self.strictness_readout.configure(text=self._strictness_label()),
        )
        self.strictness_scale.grid(row=0, column=0, sticky='ew')
        self.strictness_readout.grid(row=0, column=1, padx=(8, 0))
        tip(strict_lbl, self.strictness_scale, self.strictness_readout, text=TIPS['strictness'])
        ctk.CTkLabel(
            match_opts,
            text=(
                'Loose accepts minor tag differences (extra artists, & vs and, spacing).\n'
                'Strict requires near-exact tags.'
            ),
            text_color=t['text_dim'],
            font=_ui,
            wraplength=520,
            justify='left',
            anchor='w',
        ).grid(row=1, column=0, columnspan=2, sticky='w', pady=(2, 0))
        fallback_chk = ctk.CTkCheckBox(
            match_opts, text='Use filename instead',
            variable=self.use_filename_fallback,
            fg_color=t['accent'], hover_color=t['accent_hover'], text_color=t['text'],
            font=_ui,
        )
        fallback_chk.grid(row=2, column=0, columnspan=2, sticky='w', pady=(8, 0))
        tip(fallback_chk, text=TIPS['filename_fallback'])

        # Full IGNORE card scrolls (checkboxes + keywords). Height grows with
        # content up to max; scrollbar only when content overflows.
        self._ignore_card_max_h = 240
        ignore_wrap = ctk.CTkFrame(left, fg_color='transparent')
        ignore_wrap.pack(fill='x', padx=SECTION_PADX, pady=(0, SECTION_GAP))
        ctk.CTkLabel(
            ignore_wrap,
            text='IGNORE WHEN MATCHING',
            font=ctk_section_font(),
            text_color=t['text_dim'],
            anchor='w',
        ).pack(anchor='w', pady=(0, 3))
        self._ignore_scroll = ctk.CTkScrollableFrame(
            ignore_wrap,
            height=self._ignore_card_max_h,
            fg_color=t['panel'],
            border_color=t['border'],
            border_width=1,
            corner_radius=8,
            scrollbar_button_color=t['scrollbar'],
            scrollbar_button_hover_color=t['scrollbar_hover'],
        )
        self._ignore_scroll.pack(fill='x')
        self._ignore_scroll._scrollbar.grid_forget()
        ignore_inner = ctk.CTkFrame(self._ignore_scroll, fg_color='transparent')
        ignore_inner.pack(fill='x', padx=SECTION_INNER_PAD, pady=SECTION_INNER_PAD)
        self._ignore_inner = ignore_inner
        ignore_inner.grid_columnconfigure(1, weight=1)
        paren_chk = ctk.CTkCheckBox(
            ignore_inner, text='Text in (parentheses)',
            variable=self.ignore_parentheses,
            fg_color=t['accent'], hover_color=t['accent_hover'], text_color=t['text'],
            font=_ui,
        )
        paren_chk.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 4))
        tip(paren_chk, text=TIPS['ignore_parentheses'])
        square_chk = ctk.CTkCheckBox(
            ignore_inner, text='Text in [square brackets]',
            variable=self.ignore_square_brackets,
            fg_color=t['accent'], hover_color=t['accent_hover'], text_color=t['text'],
            font=_ui,
        )
        square_chk.grid(row=1, column=0, columnspan=2, sticky='w', pady=(0, 4))
        tip(square_chk, text=TIPS['ignore_square'])
        spaces_chk = ctk.CTkCheckBox(
            ignore_inner, text='Extra spaces', variable=self.ignore_extra_spaces,
            fg_color=t['accent'], hover_color=t['accent_hover'], text_color=t['text'],
            font=_ui,
        )
        spaces_chk.grid(row=2, column=0, columnspan=2, sticky='w', pady=(0, 8))
        tip(spaces_chk, text=TIPS['ignore_spaces'])
        kw_header = ctk.CTkFrame(ignore_inner, fg_color='transparent')
        kw_header.grid(row=3, column=0, columnspan=2, sticky='w', pady=(0, 10))
        ctk.CTkLabel(
            kw_header, text='Custom keywords', text_color=t['text_dim'],
            font=_ui,
        ).pack(side='left', padx=(0, 8))
        self._add_keyword_btn = ctk.CTkButton(
            kw_header, text='+', width=36, height=28,
            fg_color=t['accent'], hover_color=t['accent_hover'], text_color='#ffffff',
            font=_ui, command=self._add_custom_keyword_row,
        )
        self._add_keyword_btn.pack(side='left')
        tip(self._add_keyword_btn, text=TIPS['add_keyword'])
        self._custom_keywords_frame = ctk.CTkFrame(ignore_inner, fg_color='transparent')
        self._custom_keywords_frame.grid(row=4, column=0, columnspan=2, sticky='ew')
        self._custom_keywords_frame.grid_columnconfigure(0, weight=1)
        self.after_idle(self._sync_ignore_card_scroll)

    def attach_action_bar(self, parent: tk.Misc) -> None:
        """Place Match & Align actions in the host bottom bar (shared with Classify)."""
        self.find_btn = ctk_action_button(
            parent, '▶  Find pairs', self._start_find_pairs, accent=True, width=100,
        )
        self.organize_btn = ctk_action_button(
            parent, '▶  Organize folder', self._start_organize, width=128,
        )
        self.export_list_btn = ctk_action_button(
            parent, 'Export list', self._start_export_list, width=86,
        )
        self.distribute_btn = ctk_action_button(
            parent, 'Distribute originals', self._start_distribute_originals, width=128,
        )
        self.sort_folders_btn = ctk_action_button(
            parent, 'Sort folders', self._start_sort_folders, width=94,
        )
        self.align_btn = ctk_action_button(
            parent, 'Align stems', self._start_align_stems, accent=True, width=96,
        )
        self.play_stems_btn = ctk_action_button(
            parent, '♫  Play', self._open_stem_player,
            width=72, hover_color=DARK['accent'],
        )
        tip(self.export_list_btn, text=TIPS['export_list_btn'])
        tip(self.distribute_btn, text=TIPS['distribute_btn'])
        tip(self.sort_folders_btn, text=TIPS['sort_folders_btn'])
        tip(self.align_btn, text=TIPS['align_stems_btn'])
        tip(self.play_stems_btn, text=TIPS['align_play_stems'])

    def _align_tab_active(self) -> bool:
        return self._tabview.get() == 'Align'

    def _on_pair_subtab_changed(self, _event=None) -> None:
        host = self._host
        if hasattr(host, '_classify_mode_active') and not host._classify_mode_active():
            self.show_action_bar()
            if not self._busy:
                self.set_buttons_state('normal')

    def _align_action_buttons(self):
        return (
            self.export_list_btn, self.distribute_btn,
            self.sort_folders_btn, self.align_btn,
        )

    def show_action_bar(self) -> None:
        for btn in (
            self.find_btn, self.organize_btn, self.play_stems_btn,
            *self._align_action_buttons(),
        ):
            btn.pack_forget()
        if self._align_tab_active():
            self.export_list_btn.pack(side='left')
            self.distribute_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
            self.sort_folders_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
            self.align_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
            self.play_stems_btn.pack(side='right')
        else:
            self.find_btn.pack(side='left')
            self.organize_btn.pack(side='left', padx=(ACTION_BTN_GAP, 0))
        ctk_pin_button_height(
            self.find_btn, self.organize_btn, self.play_stems_btn,
            *self._align_action_buttons(),
        )
        host = self._host
        if hasattr(host, '_pin_action_bar_heights'):
            host._pin_action_bar_heights()

    def hide_action_bar(self) -> None:
        for btn in (
            self.find_btn, self.organize_btn, self.play_stems_btn,
            *self._align_action_buttons(),
        ):
            btn.pack_forget()

    def _build_align_tab(self, parent: tk.Misc) -> None:
        ctk = self._ctk
        t = DARK
        _ui = ctk_ui_font()
        body = parent

        header = ctk.CTkFrame(body, fg_color='transparent')
        header.pack(fill='x', padx=SECTION_PADX, pady=(HEADER_TOP_PAD, 12))
        self._description_with_info(
            header,
            'Align acapella/instrumental files to the original song. The original is the master timeline;',
            self._show_align_help,
            final_line=(
                'silence is added or the start is trimmed so stems line up with it.'
            ),
        )

        library = ctk_section(body, 'Stem library')
        library.grid_columnconfigure(1, weight=1)
        self._path_row(
            library, 0, 'Stems root', self.align_stems_root,
            self._pick_align_stems_root, self._open_align_stems_root, TIPS['align_stems_root'],
        )

        step_export = ctk_section(body, '1  Export')
        step_export.grid_columnconfigure(1, weight=1)
        self._file_row(
            step_export, 0, 'Export list', self.align_export_file,
            self._pick_align_export, TIPS['align_export'],
            self._open_align_export,
        )

        step_distribute = ctk_section(body, '2  Distribute')
        step_distribute.grid_columnconfigure(1, weight=1)
        self._path_row(
            step_distribute, 0, 'Originals inbox', self.align_originals_inbox,
            self._pick_align_inbox, self._open_align_inbox, TIPS['align_inbox'],
        )

        step_sort = ctk_section(body, '3  Sort')
        step_sort.grid_columnconfigure(1, weight=1)
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

        step_align = ctk_section(body, '4  Align')
        analysis_row = ctk.CTkFrame(step_align, fg_color='transparent')
        analysis_row.pack(anchor='w')
        ctk.CTkLabel(
            analysis_row, text='Analysis', text_color=t['label'],
            font=_ui,
        ).pack(side='left', padx=(0, 6))
        analysis_entry = ctk.CTkEntry(
            analysis_row, textvariable=self.align_analysis_sec, width=56, height=30,
            fg_color=t['control_bg'], border_color=t['border'], text_color=t['entry_text'],
            font=_ui,
        )
        analysis_entry.pack(side='left')
        ctk.CTkLabel(
            analysis_row, text='seconds', text_color=t['text_dim'],
            font=_ui,
        ).pack(side='left', padx=(8, 0))
        tip(analysis_entry, text=TIPS['align_analysis'])
        align_opts = ctk.CTkFrame(step_align, fg_color='transparent')
        align_opts.pack(fill='x', pady=(8, 0))
        backup_chk = ctk.CTkCheckBox(
            align_opts, text='Backup stems before align', variable=self.align_backup,
            fg_color=t['accent'], hover_color=t['accent_hover'], text_color=t['text'],
            font=_ui,
        )
        backup_chk.pack(side='left', padx=(0, 16))
        tip(backup_chk, text=TIPS['align_backup'])
        skip_chk = ctk.CTkCheckBox(
            align_opts, text='Skip if output already exists', variable=self.align_skip_existing,
            fg_color=t['accent'], hover_color=t['accent_hover'], text_color=t['text'],
            font=_ui,
        )
        skip_chk.pack(side='left')
        tip(skip_chk, text=TIPS['align_skip_existing'])

    def _file_row(self, parent, row, label, var, picker, tip_text, opener=None):
        ctk = self._ctk
        t = DARK
        lbl = ctk.CTkLabel(
            parent, text=label, text_color=t['label'],
            font=ctk_ui_font(),
        )
        lbl.grid(row=row, column=0, sticky='w', padx=(0, 10), pady=CTRL_ROW_PADY)
        ent = ctk.CTkEntry(
            parent,
            textvariable=var,
            fg_color=t['control_bg'],
            border_color=t['border'],
            text_color=t['entry_text'],
            height=30,
        )
        ent.grid(row=row, column=1, sticky='ew', pady=CTRL_ROW_PADY)
        ent.bind('<FocusOut>', lambda _e, v=var: self._normalize_path_var(v))
        browse_btn = ctk.CTkButton(
            parent,
            text='Browse',
            width=72,
            height=30,
            fg_color=t['btn'],
            hover_color=t['btn_hover'],
            text_color=t['text'],
            command=picker,
        )
        browse_btn.grid(row=row, column=2, padx=(4, 0), pady=CTRL_ROW_PADY)
        widgets = [lbl, ent, browse_btn]
        if opener is not None:
            open_btn = ctk.CTkButton(
                parent,
                text='Open',
                width=64,
                height=30,
                fg_color=t['btn'],
                hover_color=t['btn_hover'],
                text_color=t['text'],
                command=opener,
            )
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

    def _add_custom_keyword_row(self, value: str = '', *, sync: bool = True) -> None:
        ctk = self._ctk
        t = DARK
        row = ctk.CTkFrame(self._custom_keywords_frame, fg_color='transparent')
        row.pack(fill='x', pady=2)
        row.grid_columnconfigure(0, weight=1)
        var = tk.StringVar(value=value)
        entry = ctk.CTkEntry(
            row, textvariable=var,
            fg_color=t['control_bg'], border_color=t['border'],
            text_color=t['entry_text'],
            height=30, font=ctk_ui_font(),
        )
        entry.grid(row=0, column=0, sticky='ew')
        var.trace_add('write', lambda *_: self._save_settings())
        remove_btn = ctk.CTkButton(
            row, text='×', width=32, height=30,
            fg_color=t['btn'], hover_color=t['danger'], text_color=t['text'],
            font=ctk_ui_font(),
            command=lambda r=row, v=var: self._remove_custom_keyword_row(r, v),
        )
        remove_btn.grid(row=0, column=1, padx=(4, 0))
        self._custom_keyword_vars.append(var)
        if sync:
            self.after_idle(self._sync_ignore_card_scroll)

    def _remove_custom_keyword_row(self, row: tk.Misc, var: tk.StringVar) -> None:
        if var in self._custom_keyword_vars:
            self._custom_keyword_vars.remove(var)
        row.destroy()
        self._save_settings()
        # Keep one empty field so the control stays discoverable.
        if not self._custom_keyword_vars:
            self._add_custom_keyword_row('', sync=False)
        self.after_idle(self._sync_ignore_card_scroll)

    def _clear_custom_keyword_rows(self) -> None:
        frame = getattr(self, '_custom_keywords_frame', None)
        if frame is not None:
            for widget in frame.winfo_children():
                widget.destroy()
        self._custom_keyword_vars.clear()

    def _load_custom_keyword_rows(self, keywords: list[str]) -> None:
        self._clear_custom_keyword_rows()
        for keyword in keywords:
            if keyword.strip():
                self._add_custom_keyword_row(keyword.strip(), sync=False)
        if not self._custom_keyword_vars:
            self._add_custom_keyword_row('', sync=False)
        self.after_idle(self._sync_ignore_card_scroll)

    def _sync_ignore_card_scroll(self) -> None:
        """Grow IGNORE card with content; scrollbar as soon as a keyword row exists."""
        sf = getattr(self, '_ignore_scroll', None)
        inner = getattr(self, '_ignore_inner', None)
        if sf is None or inner is None:
            return
        inner.update_idletasks()
        content_px = max(inner.winfo_reqheight(), 1)
        max_logical = int(getattr(self, '_ignore_card_max_h', 240))
        try:
            max_px = int(sf._apply_widget_scaling(max_logical))
            view_logical = max(1, int(round(sf._reverse_widget_scaling(content_px))))
        except Exception:
            max_px = max_logical
            view_logical = content_px
        has_keywords = len(self._custom_keyword_vars) > 0
        if content_px > max_px:
            sf.configure(height=max_logical)
        else:
            sf.configure(height=view_logical)
        self._show_ignore_card_scrollbar(has_keywords)
        try:
            canvas = sf._parent_canvas
            canvas.configure(scrollregion=canvas.bbox('all'))
        except Exception:
            pass

    def _show_ignore_card_scrollbar(self, show: bool) -> None:
        sf = getattr(self, '_ignore_scroll', None)
        if sf is None:
            return
        try:
            bs = sf._apply_widget_scaling(
                sf._parent_frame.cget('corner_radius')
                + sf._parent_frame.cget('border_width')
            )
        except Exception:
            bs = 0
        if show:
            sf._create_grid()
        else:
            sf._scrollbar.grid_forget()
            sf._parent_canvas.grid(
                row=1, column=0, sticky='nsew', padx=bs, pady=bs,
            )

    def _get_ignore_rules(self) -> IgnoreRules:
        keywords = tuple(
            var.get().strip()
            for var in self._custom_keyword_vars
            if var.get().strip()
        )
        return IgnoreRules(
            ignore_parentheses=bool(self.ignore_parentheses.get()),
            ignore_square_brackets=bool(self.ignore_square_brackets.get()),
            ignore_extra_spaces=bool(self.ignore_extra_spaces.get()),
            custom_keywords=keywords,
        )

    def _path_row(self, parent, row, label, var, picker, opener, tip_text):
        _, ent, _, _ = ctk_path_row(
            parent, row, label, var, picker, opener,
            tip_text=tip_text, open_tip=TIPS['open_path'],
        )
        ent.bind('<FocusOut>', lambda _e, v=var: self._normalize_path_var(v))

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
        export_path = resolve_export_list_path(
            Path(self.align_export_file.get().strip())
        )
        if not root.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Stems root folder is missing or invalid.')
            return
        if not export_path.parent.is_dir():
            messagebox.showerror(PANEL_TITLE, 'Export list folder does not exist.')
            return
        # Persist resolved file path if user pasted a folder (e.g. same as stems root).
        self.align_export_file.set(display_path(str(export_path)))
        self._clear_log()
        self._set_busy(True, 'Exporting song list…')
        self._worker = threading.Thread(
            target=self._run_export_list, args=(root, export_path), daemon=True,
        )
        self._worker.start()

    def _run_export_list(self, root: Path, export_path: Path) -> None:
        try:
            count = export_song_list(root, export_path)
            self._log_feature_summary(
                'Export list',
                lines=[
                    (f'Names: {count:,}', 'ok'),
                    (f'File: {export_path}', 'info'),
                ],
            )
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
            args=(inbox, root, *self._align_sort_dirs(root)),
            daemon=True,
        )
        self._worker.start()

    def _run_distribute_originals(
        self,
        inbox: Path,
        root: Path,
        with_dir: Path,
        without_dir: Path,
    ) -> None:
        try:
            moved, skipped, unmatched, rejected, _sorted_with, _sorted_without = distribute_originals(
                inbox, root,
                on_log=self._report_log,
                on_progress=self._report_progress,
                sort_after=False,
                with_original_dir=with_dir,
                without_original_dir=without_dir,
            )
            self._log_feature_summary(
                'Distribute originals',
                lines=[
                    (f'Moved: {moved:,}', 'ok'),
                    (f'Skipped: {skipped:,}', 'warn' if skipped else 'info'),
                    (f'Unmatched: {unmatched:,}', 'warn' if unmatched else 'info'),
                    (f'Rejected: {rejected:,}', 'warn' if rejected else 'info'),
                ],
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
            self._log_feature_summary(
                'Sort folders',
                lines=[
                    (f'With original: {moved_with:,}', 'ok'),
                    (f'Without original: {moved_without:,}', 'info'),
                    (f'Skipped: {skipped:,}', 'warn' if skipped else 'info'),
                ],
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
            self._log_feature_summary(
                'Align stems',
                lines=[
                    (f'Aligned: {len(results):,}', 'ok'),
                    (f'Skipped (already aligned): {skipped:,}', 'warn' if skipped else 'info'),
                ],
            )
            self._finish_worker(f'Done · {len(results):,} aligned')
        except Exception:
            self._log(traceback.format_exc(), 'err')
            self._finish_worker('Failed')

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        self._host._set_pair_busy(busy, status, self)

    def set_buttons_state(self, state: str) -> None:
        # Color/cursor only — never state=disabled (CTk shrinks buttons on Windows).
        disabled = state != 'normal'
        dim = DARK['text_dim']
        normal = DARK['text']
        accent_btns = {self.find_btn, self.align_btn}
        buttons = (
            self.find_btn, self.organize_btn, self.play_stems_btn,
            *self._align_action_buttons(),
        )
        for btn in buttons:
            if disabled:
                btn.configure(
                    text_color=dim, cursor='arrow', height=PATH_BTN_HEIGHT,
                )
            elif btn in accent_btns:
                btn.configure(
                    text_color='#ffffff', cursor='hand2', height=PATH_BTN_HEIGHT,
                )
            else:
                btn.configure(
                    text_color=normal, cursor='hand2', height=PATH_BTN_HEIGHT,
                )
        ctk_pin_button_height(*buttons)

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
            fallback_note = 'filename only' if use_filename_fallback else 'tags only'
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

            self._log_feature_summary(
                'Find pairs',
                lines=[
                    (f'Reference: {ref_label} · {fallback_note}', 'info'),
                    (f'Threshold: {threshold:.0%}', 'info'),
                    (f'Pairs: {len(result.pairs):,}', 'ok'),
                    (f'Unmatched reference: {len(result.unmatched_reference):,}', 'warn' if result.unmatched_reference else 'info'),
                    (f'Unmatched partner: {len(result.unmatched_partner):,}', 'warn' if result.unmatched_partner else 'info'),
                ],
            )
            self._finish_worker(f'Done · {len(result.pairs)} pair(s) moved')
        except Exception:
            self._log(traceback.format_exc(), 'err')
            self._finish_worker('Failed')

    def _start_organize(self) -> None:
        if self._busy or self._host._organize_worker_active():
            return
        folder = Path(self.pairs_output_dir.get().strip())
        if not folder.is_dir():
            messagebox.showerror(
                PANEL_TITLE,
                'Pairs output folder is missing or invalid.\n'
                'Find pairs writes there; Organize folder groups files inside it.',
            )
            return
        strictness = float(self.strictness.get())
        use_filename_fallback = bool(self.use_filename_fallback.get())
        include_subfolders = bool(self.include_subfolders.get())
        ignore_rules = self._get_ignore_rules()
        self._clear_log()
        self._log(f'Organizing matched files in:\n  {folder}', 'info')
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
            for dest_dir, _files in moved:
                self._log(f'✓ {dest_dir.name}/', 'ok')
            self._log_feature_summary(
                'Organize folder',
                lines=[(f'Folders created: {len(moved):,}', 'ok')],
            )
            self._finish_worker(f'Done · {len(moved)} folder(s) created')
        except Exception:
            self._log(traceback.format_exc(), 'err')
            self._finish_worker('Failed')

    def _log_feature_summary(
        self,
        feature: str,
        *,
        lines: list[tuple[str, str]] | None = None,
    ) -> None:
        """Unified Match & Align footer: === Feature Summary === … DONE."""
        self._log(f'=== {feature} Summary ===', 'info')
        for text, tag in (lines or ()):
            self._log(text if text.startswith('  ') else f'  {text}', tag)
        self._log('', 'info')
        self._log('DONE', 'ok')

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
            'align_with_original_dir': self.align_with_original_dir.get(),
            'align_without_original_dir': self.align_without_original_dir.get(),
        }

    def _load_settings(self) -> None:
        data = load_settings()
        self.acapella_dir.set(display_path(str(data.get('acapella_dir', ''))))
        self.instrumental_dir.set(display_path(str(data.get('instrumental_dir', ''))))
        self.pairs_output_dir.set(display_path(str(data.get('pairs_output_dir', ''))))
        # Legacy: older builds had a separate organize_dir; prefer pairs_output.
        if not self.pairs_output_dir.get().strip():
            legacy = str(data.get('organize_dir', '') or '').strip()
            if legacy:
                self.pairs_output_dir.set(display_path(legacy))
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

    def _save_settings(self) -> None:
        data = load_settings()
        data.update(self.settings_snapshot())
        save_settings(data)

    def _bind_settings_autosave(self) -> None:
        self.align_stems_root.trace_add('write', self._sync_align_sort_dirs)
        for var in (
            self.acapella_dir, self.instrumental_dir, self.pairs_output_dir,
            self.reference_side, self.strictness,
            self.use_filename_fallback, self.include_subfolders,
            self.ignore_parentheses,
            self.ignore_square_brackets,
            self.ignore_extra_spaces,
            self.align_stems_root, self.align_originals_inbox, self.align_export_file,
            self.align_backup, self.align_skip_existing, self.align_analysis_sec,
            self.align_with_original_dir, self.align_without_original_dir,
        ):
            var.trace_add('write', lambda *_: self._save_settings())
