"""Genre & Gender tab — port of genre_gender_panel.GenreGenderPanel.

Two sub-tabs:
  Genre  — Paths (input + include subfolders), Run mode (Batch/Per-file),
           Tag style (Combined/Split), Tag options (Write metadata, Overwrite).
  Gender — same shape + Voice gender field (Comment/Gender) + Reverb mode
           (Combined/Split).

Action buttons: ▶ Tag genre / ▶ Tag gender / ■ Stop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    RadioButton,
    ScrollArea,
    TabWidget,
)

from .. import theme
from ..settings_store import SettingsStore, display_path
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from ..widgets.dialogs import help_dialog, show_info
from ..widgets.info_icon import InfoIcon
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from ..workers.tagger_worker import TaggerWorker


PANEL_TITLE = "Genre & Gender"

# Hint text beside radios — QLabel + stylesheet (BodyLabel polish resets to white)
_HINT_FONT_PX = theme.BODY_FONT_PX


def _split_option_label(text: str) -> tuple[str, str | None]:
    """'Batch (fast)' → ('Batch', 'fast'); no paren → (text, None)."""
    paren = text.find("(")
    if paren > 0:
        hint = text[paren:].strip().removeprefix("(").removesuffix(")").strip()
        return text[:paren].rstrip(), hint or None
    return text.strip(), None


def _style_radio_hint(lbl: QLabel) -> None:
    dim = theme.DARK["text_dim"]
    lbl.setObjectName("RadioHint")
    lbl.setStyleSheet(
        f"""
        QLabel#RadioHint {{
            color: {dim};
            font-family: "{theme.FONT_FAMILY}";
            font-size: {_HINT_FONT_PX}px;
            background: transparent;
        }}
        """
    )


class _RadioRow(QWidget):
    """Two radio options split evenly across the full card width."""

    def __init__(self, parent: QWidget, options, value: str, *, tooltip: str = "") -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._buttons: dict[str, RadioButton] = {}
        self._extra: list[QWidget] = []

        for label, val in options:
            main, hint = _split_option_label(label)
            key = str(val)

            cell = QWidget()
            cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            cell_lay = QHBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(6)

            rb = RadioButton(main)
            rb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            rb.setCursor(Qt.PointingHandCursor)
            if tooltip:
                rb.setToolTip(tooltip)
            self._group.addButton(rb)
            self._buttons[key] = rb
            cell_lay.addWidget(rb, 0, Qt.AlignVCenter)

            if hint:
                hint_lbl = QLabel(hint)
                hint_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                _style_radio_hint(hint_lbl)
                hint_lbl.setCursor(Qt.PointingHandCursor)
                if tooltip:
                    hint_lbl.setToolTip(tooltip)
                hint_lbl.mousePressEvent = (  # type: ignore[method-assign]
                    lambda _e, v=key: self.set_value(v)
                )
                cell_lay.addWidget(hint_lbl, 0, Qt.AlignVCenter)
                self._extra.append(hint_lbl)

            cell_lay.addStretch(1)
            layout.addWidget(cell, 1)  # equal stretch → half-split
            self._extra.append(cell)

        self.set_value(value)
        self.valueChanged = self._group.buttonClicked

    def set_value(self, value: str) -> None:
        rb = self._buttons.get(str(value))
        if rb is not None:
            rb.setChecked(True)

    def value(self) -> str:
        for val, rb in self._buttons.items():
            if rb.isChecked():
                return val
        return ""

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 Qt name
        super().setEnabled(enabled)
        for rb in self._buttons.values():
            rb.setEnabled(enabled)
        for w in self._extra:
            w.setEnabled(enabled)


class GenreGenderTab(QWidget):
    """Genre + Gender sub-tabs sharing an action bar."""

    request_status_running = Signal()
    request_status_idle = Signal(str)
    request_progress = Signal(float, object, int, int, str)
    request_status_text = Signal(str)
    request_log = Signal(str, str)
    request_clear_log = Signal()
    request_gg_processed = Signal(int, int)

    def __init__(self, settings: SettingsStore) -> None:
        super().__init__()
        self._settings = settings
        self._worker: Optional[TaggerWorker] = None
        self._busy = False
        self._loading = False

        self._build_ui()
        self.load_settings()
        self._bind_autosave()

    # ----- UI -----

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._tabview = TabWidget()
        theme.configure_tab_widget(self._tabview)
        self._tabview.addTab(self._build_genre_tab(), "Genre")
        self._tabview.addTab(self._build_gender_tab(), "Gender")
        theme.configure_tab_widget(self._tabview)
        theme.inset_tab_bar(self._tabview)
        self._tabview.currentChanged.connect(self._on_subtab_changed)
        layout.addWidget(self._tabview)

    def _build_genre_tab(self) -> QWidget:
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(ScrollArea.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(
            theme.PAGE_CONTENT_INSET, 0, theme.PAGE_CONTENT_INSET, 0
        )
        v.setSpacing(theme.SECTION_GAP)

        header = QHBoxLayout()
        header.setContentsMargins(0, 8, 0, 10)
        header.setSpacing(6)
        title = BodyLabel("Tag instrumental/music files with Discogs genre and style tags")
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=lambda: self._show_help("genre")))
        header.addStretch(1)
        v.addLayout(header)

        paths = Section(inner, "Path")
        paths.body.layout().setSpacing(12)
        self.genre_input_row = PathRow(paths.body, "Input folder", label_width=80)
        self.genre_include_subfolders = CheckBox("Include subfolders")
        self.genre_include_subfolders.setChecked(True)
        paths.body.layout().addWidget(self.genre_include_subfolders)
        v.addWidget(paths)

        run_card = Section(inner, "Run mode")
        self.genre_run_mode = _RadioRow(
            run_card.body,
            [("Batch (fast)", "batch"), ("Per-file (live results)", "per_file")],
            "batch",
            tooltip="Batch shares GPU batches across files (fastest). Per-file logs each prediction live.",
        )
        run_card.body.layout().addWidget(self.genre_run_mode)
        v.addWidget(run_card)

        style_card = Section(inner, "Tag style")
        self.genre_tag_style = _RadioRow(
            style_card.body,
            [("Combined  (GENRE=Rock/Surf)", "combined"), ("Split  (GENRE=Rock, STYLE=Surf)", "split")],
            "combined",
            tooltip="Combined writes a single GENRE tag as Genre/Style. Split writes separate GENRE and STYLE tags.",
        )
        style_card.body.layout().addWidget(self.genre_tag_style)
        v.addWidget(style_card)

        opts_card = Section(inner, "Tag options")
        opts_lay = opts_card.body.layout()
        m = opts_lay.contentsMargins()
        opts_lay.setContentsMargins(m.left(), 14, m.right(), m.bottom())
        opts_lay.setSpacing(12)
        self.genre_write_meta = CheckBox("Write metadata tags")
        self.genre_write_meta.setChecked(True)
        self.genre_write_meta.setToolTip("Write tags to FLAC/MP3/M4A/WAV. Disable to only generate the CSV.")
        self.genre_overwrite_tags = CheckBox("Overwrite existing tags")
        self.genre_overwrite_tags.setToolTip("Off (default): skip files that already have genre tags (resume-friendly). On: re-tag every file.")
        opts_lay.addWidget(self.genre_write_meta)
        opts_lay.addWidget(self.genre_overwrite_tags)
        v.addWidget(opts_card)
        v.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    def _build_gender_tab(self) -> QWidget:
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(ScrollArea.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(
            theme.PAGE_CONTENT_INSET, 0, theme.PAGE_CONTENT_INSET, 0
        )
        v.setSpacing(theme.SECTION_GAP)

        header = QHBoxLayout()
        header.setContentsMargins(0, 8, 0, 10)
        header.setSpacing(6)
        title = BodyLabel("Tag acapella audio files with voice gender (female/male) and reverb (wet/dry)")
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=lambda: self._show_help("gender")))
        header.addStretch(1)
        v.addLayout(header)

        paths = Section(inner, "Path")
        paths.body.layout().setSpacing(12)
        self.gender_input_row = PathRow(paths.body, "Input folder", label_width=80)
        self.gender_include_subfolders = CheckBox("Include subfolders")
        self.gender_include_subfolders.setChecked(True)
        paths.body.layout().addWidget(self.gender_include_subfolders)
        v.addWidget(paths)

        run_card = Section(inner, "Run mode")
        self.gender_run_mode = _RadioRow(
            run_card.body,
            [("Batch (fast)", "batch"), ("Per-file (live results)", "per_file")],
            "batch",
        )
        run_card.body.layout().addWidget(self.gender_run_mode)
        v.addWidget(run_card)

        field_card = Section(inner, "Write gender to")



        self.gender_tag_field = _RadioRow(
            field_card.body,
            [("Comment", "comment"), ("Gender (custom)", "gender")],
            "comment",
            tooltip="Comment writes gender to the COMMENT tag. Gender writes to a GENDER tag.",
        )
        field_card.body.layout().addWidget(self.gender_tag_field)
        v.addWidget(field_card)

        rev_card = Section(inner, "Write reverb to")



        self.gender_reverb_mode = _RadioRow(
            rev_card.body,
            [("Combined  (COMMENT=female/wet)", "combined"), ("Split  (GENDER=female, REVERB=wet)", "split")],
            "combined",
            tooltip=(
                "Dry/wet from the bundled vocal mel-CNN. "
                "Combined writes gender/reverb into the chosen field. "
                "Split writes gender alone and REVERB=wet|dry as a separate custom field."
            ),
        )
        rev_card.body.layout().addWidget(self.gender_reverb_mode)
        v.addWidget(rev_card)

        opts_card = Section(inner, "Tag options")
        opts_lay = opts_card.body.layout()
        m = opts_lay.contentsMargins()
        opts_lay.setContentsMargins(m.left(), 14, m.right(), m.bottom())
        opts_lay.setSpacing(12)
        self.gender_write_meta = CheckBox("Write metadata tags")
        self.gender_write_meta.setChecked(True)
        self.gender_overwrite_tags = CheckBox("Overwrite existing tags")
        opts_lay.addWidget(self.gender_write_meta)
        opts_lay.addWidget(self.gender_overwrite_tags)
        v.addWidget(opts_card)
        v.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    def _on_subtab_changed(self, _idx: int) -> None:
        if not hasattr(self, "_action_page"):
            return
        is_genre = self._tabview.currentIndex() == 0
        self.genre_btn.setVisible(is_genre)
        self.gender_btn.setVisible(not is_genre)

    # ----- action bar -----

    def attach_action_bar(self, page: ActionBarPage, window) -> None:
        self._action_page = page
        self.genre_btn = action_button("▶ Tag genre", on_click=self._start_genre, accent=True)
        self.genre_btn.setFixedWidth(120)
        self.gender_btn = action_button("▶ Tag gender", on_click=self._start_gender, accent=True)
        self.gender_btn.setFixedWidth(120)
        self.stop_btn = action_button("■ Stop", on_click=self._stop)
        self.stop_btn.setEnabled(False)
        # Match Classify: content-sized Stop (don't let it expand across the bar)
        self.stop_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.stop_btn.setFixedWidth(max(72, self.stop_btn.sizeHint().width()))
        page.add_button(self.genre_btn)
        page.add_button(self.gender_btn)
        page.add_button(self.stop_btn)
        page.add_stretch()
        self._on_subtab_changed(0)

    # ----- worker lifecycle -----

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = busy
        self.genre_btn.setEnabled(not busy)
        self.gender_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        if busy:
            self.request_status_running.emit()
            if status:
                self.request_status_text.emit(status)
        else:
            self.request_status_idle.emit(status or "Idle")

    def _on_worker_done(self, status: str) -> None:
        self._worker = None
        self._set_busy(False, status)

    def _stop(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()
        self.request_log.emit("[stopping] ...", "warn")

    def _start_genre(self) -> None:
        if self._busy:
            return
        input_dir = self.genre_input_row.text().strip()
        if not input_dir or not Path(input_dir).is_dir():
            show_info(self, PANEL_TITLE, "Input folder is missing or invalid.")
            return
        self.request_clear_log.emit()
        # Startup/config indented like Classify; === file headers stay flush.
        self.request_log.emit("  Starting genre tagger:", "info")
        self.request_log.emit(f"  {input_dir}", "info")
        worker = TaggerWorker(
            "genre", input_dir,
            batch_mode=self.genre_run_mode.value() == "batch",
            tag_style=self.genre_tag_style.value(),
            gender_field="comment",
            write_meta=self.genre_write_meta.isChecked(),
            csv_path="",
            include_subfolders=self.genre_include_subfolders.isChecked(),
            overwrite_tags=self.genre_overwrite_tags.isChecked(),
            parent=self,
        )
        self._wire(worker)
        self._set_busy(True, "Tagging genre…")
        worker.start()

    def _start_gender(self) -> None:
        if self._busy:
            return
        input_dir = self.gender_input_row.text().strip()
        if not input_dir or not Path(input_dir).is_dir():
            show_info(self, PANEL_TITLE, "Input folder is missing or invalid.")
            return
        self.request_clear_log.emit()
        # Startup/config indented like Classify; === file headers stay flush.
        self.request_log.emit("  Starting gender tagger:", "info")
        self.request_log.emit(f"  {input_dir}", "info")
        worker = TaggerWorker(
            "gender", input_dir,
            batch_mode=self.gender_run_mode.value() == "batch",
            tag_style=self.gender_reverb_mode.value(),
            gender_field=self.gender_tag_field.value(),
            write_meta=self.gender_write_meta.isChecked(),
            csv_path="",
            include_subfolders=self.gender_include_subfolders.isChecked(),
            overwrite_tags=self.gender_overwrite_tags.isChecked(),
            parent=self,
        )
        self._wire(worker)
        self._set_busy(True, "Tagging gender…")
        worker.start()

    def _wire(self, worker: TaggerWorker) -> None:
        self._worker = worker
        worker.log_line.connect(self.request_log)
        worker.progress.connect(self.request_progress)
        worker.processed.connect(self.request_gg_processed)
        worker.status.connect(self.request_status_text)
        worker.finished_ok.connect(self._on_worker_done)

    # ----- help -----

    def _show_help(self, mode: str) -> None:
        is_genre = mode == "genre"
        help_dialog(
            self,
            title="Genre help" if is_genre else "Gender help",
            heading=(
                "Tag instrumentals with genre & style"
                if is_genre
                else "Tag acapellas with voice gender + dry/wet reverb"
            ),
            intro=(
                "Classify instrumental tracks and write Discogs-style GENRE / STYLE tags"
                if is_genre
                else "Estimate singing voice gender and dry/wet reverb, then write tags"
            ),
            sections=[
                ("Workflow", [
                    "1. Choose an input folder of instrumental FLAC/MP3/WAV files.",
                    "2. Pick run mode (Batch or Per-file) and tag style.",
                    "3. Click ▶ Tag genre (or Tag gender) and watch progress in the LOG panel.",
                    "4. Check FLAC tags and/or the CSV export when the run finishes.",
                ]),
                ("Model", [
                    "Genre: Hugging Face MAEST mtg-upf/discogs-maest-30s-pw-129e-519l (Discogs519)."
                    if is_genre
                    else "Gender: Discogs-EffNet (ONNX Runtime DirectML). Reverb: in-house trained vocal mel-CNN.",
                ]),
                ("Options", [
                    "Batch is faster (best with a GPU). Per-file prints each prediction live. "
                    "Combined writes one tag as Genre/Style (or gender/reverb). Split writes separate tags.",
                    "Tags are written to FLAC, MP3, M4A, and WAV (ID3 / Vorbis / MP4 atoms as appropriate). "
                    "A CSV export is always written.",
                ]),
                ("Setup", [
                    "Genre & Gender uses the bundled folder genre_gender_tagger\\ with its own venv. "
                    "Run genre_gender_tagger\\install-deps.bat once (or accept it at the end of STEM install-deps.bat).",
                ]),
            ],
        )

    # ----- settings -----

    def settings_snapshot(self) -> dict:
        return {
            "gg_genre_input_dir": display_path(self.genre_input_row.text()),
            "gg_genre_include_subfolders": bool(self.genre_include_subfolders.isChecked()),
            "gg_genre_batch_mode": self.genre_run_mode.value() == "batch",
            "gg_genre_tag_style": self.genre_tag_style.value(),
            "gg_genre_write_meta": bool(self.genre_write_meta.isChecked()),
            "gg_genre_overwrite_tags": bool(self.genre_overwrite_tags.isChecked()),
            "gg_gender_input_dir": display_path(self.gender_input_row.text()),
            "gg_gender_include_subfolders": bool(self.gender_include_subfolders.isChecked()),
            "gg_gender_batch_mode": self.gender_run_mode.value() == "batch",
            "gg_gender_tag_field": self.gender_tag_field.value(),
            "gg_gender_reverb_mode": self.gender_reverb_mode.value(),
            "gg_gender_write_meta": bool(self.gender_write_meta.isChecked()),
            "gg_gender_overwrite_tags": bool(self.gender_overwrite_tags.isChecked()),
        }

    def load_settings(self) -> None:
        self._loading = True
        try:
            d = self._settings.data
            if d.get("gg_genre_input_dir"):
                self.genre_input_row.set_text(d["gg_genre_input_dir"])
            self.genre_include_subfolders.setChecked(bool(d.get("gg_genre_include_subfolders", True)))
            self.genre_run_mode.set_value("batch" if d.get("gg_genre_batch_mode", True) else "per_file")
            self.genre_tag_style.set_value(d.get("gg_genre_tag_style", "combined"))
            self.genre_write_meta.setChecked(bool(d.get("gg_genre_write_meta", True)))
            self.genre_overwrite_tags.setChecked(bool(d.get("gg_genre_overwrite_tags", False)))
            if d.get("gg_gender_input_dir"):
                self.gender_input_row.set_text(d["gg_gender_input_dir"])
            self.gender_include_subfolders.setChecked(bool(d.get("gg_gender_include_subfolders", True)))
            self.gender_run_mode.set_value("batch" if d.get("gg_gender_batch_mode", True) else "per_file")
            self.gender_tag_field.set_value(d.get("gg_gender_tag_field", "comment"))
            self.gender_reverb_mode.set_value(d.get("gg_gender_reverb_mode", "combined"))
            self.gender_write_meta.setChecked(bool(d.get("gg_gender_write_meta", True)))
            self.gender_overwrite_tags.setChecked(bool(d.get("gg_gender_overwrite_tags", False)))
        finally:
            self._loading = False

    def _bind_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(200)
        self._autosave_timer.timeout.connect(self._flush_settings)
        for sig in (
            self.genre_input_row.entry.textChanged,
            self.genre_include_subfolders.toggled,
            self.genre_run_mode.valueChanged,
            self.genre_tag_style.valueChanged,
            self.genre_write_meta.toggled,
            self.genre_overwrite_tags.toggled,
            self.gender_input_row.entry.textChanged,
            self.gender_include_subfolders.toggled,
            self.gender_run_mode.valueChanged,
            self.gender_tag_field.valueChanged,
            self.gender_reverb_mode.valueChanged,
            self.gender_write_meta.toggled,
            self.gender_overwrite_tags.toggled,
        ):
            sig.connect(self._schedule_save)

    def _schedule_save(self, *_) -> None:
        if self._loading:
            return
        if not hasattr(self, "_autosave_timer"):
            return
        self._autosave_timer.start()

    def _flush_settings(self) -> None:
        if self._loading:
            return
        self._settings.merge(self.settings_snapshot())
        self._settings.flush()

    def flush_settings(self) -> None:
        """Immediate persist (used on app close)."""
        if hasattr(self, "_autosave_timer"):
            self._autosave_timer.stop()
        self._flush_settings()


def register(window, settings: SettingsStore) -> None:
    tab = GenreGenderTab(settings)
    window.register_tab("Genre & Gender", tab)
    page = window.action_bar.make_page("Genre & Gender")
    tab.attach_action_bar(page, window)
    tab.request_status_running.connect(lambda: window.status_bar.show_running())
    tab.request_status_idle.connect(window.status_bar.show_idle)
    tab.request_progress.connect(window.status_bar.update_progress)
    tab.request_status_text.connect(window.status_bar.set_status)
    tab.request_log.connect(window.append_log)
    tab.request_clear_log.connect(window.clear_log)
    tab.request_gg_processed.connect(window.update_gg_processed)
