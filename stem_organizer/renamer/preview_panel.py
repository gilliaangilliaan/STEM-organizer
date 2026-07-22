"""Preview panel — QTableView + QAbstractTableModel + lazy-compute worker.

Replaces the hand-rolled virtualized canvas list from
track_renamer.gui.preview_panel with idiomatic Qt model/view.

Columns: ✓ | Category | Original | New
ANALYZE LOG (instrument ML output) lives in a hidden QTextEdit swapped in via
QStackedLayout during an enrich run.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QPersistentModelIndex,
    QRect,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeySequence, QPainter, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QStackedLayout,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    PushButton,
    TableItemDelegate,
    TableView,
    TextEdit,
    ToggleButton,
)

from track_renamer.category_palette import default_category_color, parse_category_prefix_display
from track_renamer.engine.models import PreviewRow, Rule, Track
from track_renamer.engine.processor import compute_preview_row, prepare_rules

from .. import theme
from ..widgets.log_panel import LOG_INDENT


# Columns: ✓ | Category | Original | New
COL_CHECK = 0
COL_CATEGORY = 1
COL_ORIGINAL = 2
COL_NEW = 3
N_COLS = 4

RESULT_BATCH_SIZE = 64
RESULT_POLL_MS = 40
LAZY_BUFFER_ROWS = 60


@dataclass
class _PreviewJob:
    generation: int
    tracks: List[Track]
    rules: object  # PreparedRulePlan
    cancel: threading.Event = field(default_factory=threading.Event)
    priority: "queue.SimpleQueue[int]" = field(default_factory=queue.SimpleQueue)
    results: "queue.SimpleQueue" = field(default_factory=queue.SimpleQueue)
    requested: set = field(default_factory=set)


class PreviewModel(QAbstractTableModel):
    """Table model: one row per Track, lazily filled with PreviewRow."""

    def __init__(self) -> None:
        super().__init__()
        self._tracks: List[Track] = []
        self._rows: List[Optional[PreviewRow]] = []
        self._category_colors: dict = {}

    # ----- model API -----

    def set_tracks(self, tracks: List[Track]) -> None:
        self.beginResetModel()
        self._tracks = list(tracks)
        self._rows = [None] * len(self._tracks)
        self.endResetModel()

    def set_category_colors(self, colors: dict) -> None:
        self._category_colors = dict(colors or {})

    def track_at(self, row: int) -> Optional[Track]:
        if 0 <= row < len(self._tracks):
            return self._tracks[row]
        return None

    def row_at(self, row: int) -> Optional[PreviewRow]:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def set_row(self, row: int, preview: PreviewRow) -> None:
        if 0 <= row < len(self._rows):
            self._rows[row] = preview
            ix = self.index(row, 0)
            ix2 = self.index(row, N_COLS - 1)
            self.dataChanged.emit(ix, ix2, [Qt.DisplayRole, Qt.BackgroundRole, Qt.FontRole])

    def update_selection(self, row: int, value: bool) -> None:
        track = self.track_at(row)
        if track is None:
            return
        track.selected = bool(value)
        # CheckState + name styling (strikethrough clears when deselected)
        left = self.index(row, COL_CHECK)
        right = self.index(row, COL_NEW)
        self.dataChanged.emit(
            left,
            right,
            [Qt.CheckStateRole, Qt.FontRole, Qt.ForegroundRole, Qt.DisplayRole],
        )

    # ----- QAbstractTableModel -----

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._tracks)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else N_COLS

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if orientation != Qt.Horizontal:
            return None
        if role == Qt.TextAlignmentRole:
            if section == COL_CHECK:
                return Qt.AlignHCenter | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter
        if role != Qt.DisplayRole:
            return None
        return ("✓", "Category", "Original", "New")[section]

    def data(self, index, role=Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        track = self.track_at(row)
        if track is None:
            return None
        preview = self.row_at(row)

        if role == Qt.CheckStateRole and col == COL_CHECK:
            return Qt.Checked if track.selected else Qt.Unchecked
        if role == Qt.TextAlignmentRole:
            if col == COL_CHECK:
                return Qt.AlignHCenter | Qt.AlignVCenter
            if col == COL_CATEGORY:
                return Qt.AlignCenter
            return Qt.AlignLeft | Qt.AlignVCenter
        if role == Qt.BackgroundRole:
            # Active-row highlight handled by selection model; just alternate rows
            if row % 2 == 1:
                return QColor(theme.COLORS["panel"])
            return None
        if role == Qt.ForegroundRole:
            # COL_CATEGORY is painted by _CategoryBadgeDelegate (badge bg + white text);
            # no per-cell foreground needed there.
            # Strikethrough / emphasis only while selected — deselected rows look normal.
            active_change = (
                preview is not None and preview.changed and track.selected
            )
            if col == COL_ORIGINAL and active_change:
                return QColor(theme.DARK["text_mute"])
            if col == COL_NEW and active_change:
                return QColor("#ffffff")
            if col == COL_NEW:
                # Deselected or not changing — same dim as former "unchanged" status
                return QColor(theme.DARK["unchanged"])
            if col == COL_ORIGINAL:
                return QColor("#ffffff")
            return QColor(theme.DARK["text"])
        if role == Qt.FontRole:
            font = QFont(theme.FONT_FAMILY)
            font.setPixelSize(theme.BODY_FONT_PX)
            active_change = (
                preview is not None and preview.changed and track.selected
            )
            if col == COL_NEW and active_change:
                font.setBold(True)
            if col == COL_NEW and (
                preview is None or not track.selected or not preview.changed
            ):
                font.setItalic(True)
            if col == COL_ORIGINAL and active_change:
                font.setStrikeOut(True)
            return font
        if role == Qt.DisplayRole:
            if col == COL_ORIGINAL:
                return track.display_name
            if col == COL_NEW:
                if preview is None:
                    return "…"
                # Deselected → treat as not renaming (dimmed "unchanged" label)
                if not track.selected:
                    return "unchanged"
                if not preview.changed:
                    return "unchanged"
                return preview.new_display
            if col == COL_CATEGORY:
                if preview is None:
                    return ""
                parsed = parse_category_prefix_display(preview.new_display)
                return parsed[0].upper() if parsed else ""
        return None

    def flags(self, index):  # noqa: N802
        if not index.isValid():
            return Qt.NoItemFlags
        # Check column is toggled by PreviewPanel click handler — Fluent's painted
        # checkbox sits outside Qt's ItemIsUserCheckable hit box, so relying on
        # setData alone makes uncheck work and re-check fail.
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def setData(self, index, value, role=Qt.EditRole):  # noqa: N802
        if role == Qt.CheckStateRole and index.column() == COL_CHECK:
            checked = value in (
                Qt.Checked,
                getattr(Qt.CheckState, "Checked", Qt.Checked),
                2,
            )
            self.update_selection(index.row(), bool(checked))
            return True
        return False


class _CenteredCheckDelegate(TableItemDelegate):
    """Fluent TableItemDelegate pins the checkbox at rect.x+15; center it.

    Active-row left indicator is white (Fluent default falls back to accent).
    Checkbox checked fill stays theme accent via light/darkCheckedColor.
    """

    def _drawIndicator(self, painter: QPainter, option, index) -> None:  # noqa: N802
        # Thin left bar for the focused/selected row — white, not accent lavender.
        y, h = option.rect.y(), option.rect.height()
        ph = round(0.35 * h if self.pressedRow == index.row() else 0.257 * h)
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(4, ph + y, 3, h - 2 * ph, 1.5, 1.5)

    def _drawCheckBox(self, painter: QPainter, option, index) -> None:  # noqa: N802
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QStyleOptionViewItem

        size = 19
        # Fluent draws at option.rect.x() + 15 — shift rect so that lands centered.
        centered = option.rect.x() + (option.rect.width() - size) / 2.0
        opt = QStyleOptionViewItem(option)
        opt.rect = QRect(
            int(round(centered - 15)),
            option.rect.y(),
            option.rect.width(),
            option.rect.height(),
        )
        super()._drawCheckBox(painter, opt, index)


class _CategoryBadgeDelegate(QStyledItemDelegate):
    """Paints COL_CATEGORY with the same ChipRenderer pixmap as Rename/Classify
    log chips: Arial Bold 11px, white text, solid fill, fixed height, sharp
    corners (no radius — log pixmaps are filled rects). Centered with inset so
    dark row/grid lines stay visible between adjacent cells.
    """

    # Keep a dark gap around the chip so row separators remain visible.
    _BADGE_INSET_H = 2
    _BADGE_INSET_V = 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from ..widgets.log_panel import ChipRenderer

        self._chips = ChipRenderer()

    @staticmethod
    def _chip_font() -> QFont:
        from ..widgets.log_panel import LOG_CHIP_FONT_PX

        font = QFont("Arial")
        font.setBold(True)
        font.setPixelSize(LOG_CHIP_FONT_PX)
        if not QFontMetrics(font).inFont(ord("A")):
            font = QFont(theme.FONT_FAMILY_SEMIBOLD)
            font.setBold(True)
            font.setPixelSize(LOG_CHIP_FONT_PX)
        return font

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802
        label = (index.data(Qt.DisplayRole) or "").strip()
        if not label:
            return
        model = index.model()
        color_hex = ""
        if hasattr(model, "_category_colors"):
            preview = model.row_at(index.row())
            if preview is not None:
                parsed = parse_category_prefix_display(preview.new_display)
                if parsed:
                    cat = parsed[0]
                    color_hex = model._category_colors.get(cat) or default_category_color(cat)
        if not color_hex:
            color_hex = theme.DARK.get("text_mute", "#6b7080")

        chip_text = label.lower()
        pix = self._chips.category_pixmap(chip_text, color_hex)
        cell = option.rect
        avail = cell.adjusted(
            self._BADGE_INSET_H,
            self._BADGE_INSET_V,
            -self._BADGE_INSET_H,
            -self._BADGE_INSET_V,
        )
        # Log chips are fixed compact height/width — center them; do not stretch
        # into a full-cell bar (that read as sharper/blockier than the log).
        x = avail.x() + max(0, (avail.width() - pix.width()) // 2)
        y = avail.y() + max(0, (avail.height() - pix.height()) // 2)

        painter.save()
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setClipRect(avail)
        painter.drawPixmap(x, y, pix)
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802
        from PySide6.QtCore import QSize

        label = (index.data(Qt.DisplayRole) or "").strip().lower() or "xxxx"
        pix = self._chips.category_pixmap(
            label, theme.DARK.get("text_mute", "#6b7080")
        )
        return QSize(
            pix.width() + 2 * self._BADGE_INSET_H,
            pix.height() + 2 * self._BADGE_INSET_V,
        )


class PreviewPanel(QWidget):
    """Right side: preview table + tools + analyze log."""

    on_change: Optional[Callable[[], None]] = None
    on_active: Optional[Callable[[Optional[Track], Optional[PreviewRow]], None]] = None
    on_play_pause: Optional[Callable[[], None]] = None
    on_seek: Optional[Callable[[float], None]] = None

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("PreviewPanel")
        self._tracks: List[Track] = []
        self._root_label: str = ""
        self._lazy_generation = 0
        self._lazy_done = 0
        self._lazy_total = 0
        self._lazy_changed = 0
        self._lazy_selected_changed = 0
        self._job: Optional[_PreviewJob] = None
        self._only_changed = False
        self._active_index: Optional[int] = None
        self._pending_state = False

        self._build_ui()

        # Result poll timer
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(RESULT_POLL_MS)
        self._poll_timer.timeout.connect(self._drain_results)

    # ----- build -----

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        # Top 0 — RULES title is the vertical anchor; TrackRenamerApp pads above us
        layout.setContentsMargins(8, 0, 8, 8)
        layout.setSpacing(6)

        # Header — match RULES row height / dim section title
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        title = CaptionLabel("PREVIEW")
        title.setObjectName("SectionTitle")
        title.setStyleSheet(
            f"color: {theme.DARK['text_dim']}; font-size: {theme.SECTION_TITLE_PX}px; "
            f'font-family: "{theme.FONT_FAMILY}"; font-weight: 600; background: transparent;'
        )
        title.setFixedHeight(theme.ACTION_BTN_HEIGHT)
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.section_title = title
        # Explicit VCenter: SectionTitle QSS max-height vs fixed 30px otherwise
        # shifts this label a few px below the stats CaptionLabel in the row.
        header.addWidget(title, 0, Qt.AlignVCenter)
        header.addStretch(1)
        self.stats_label = CaptionLabel("0 files")
        self.stats_label.setStyleSheet(
            f"color: {theme.DARK['text_dim']}; font-size: {theme.SECTION_TITLE_PX}px; "
            f'font-family: "{theme.FONT_FAMILY}"; background: transparent; padding: 0px;'
        )
        self.stats_label.setFixedHeight(theme.ACTION_BTN_HEIGHT)
        self.stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(self.stats_label, 0, Qt.AlignVCenter)
        layout.addLayout(header)

        # Tools
        tools = QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        self.select_all_btn = PushButton("Select all")
        self.select_all_btn.clicked.connect(self._select_all)
        self.deselect_all_btn = PushButton("Deselect all")
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        self.only_changed_btn = ToggleButton("✓ Only changed")
        self.only_changed_btn.setCheckable(True)
        self.only_changed_btn.toggled.connect(self._on_only_changed_toggled)
        for b in (self.select_all_btn, self.deselect_all_btn, self.only_changed_btn):
            tools.addWidget(b)
        tools.addStretch(1)
        layout.addLayout(tools)

        # Stacked: table / analyze log
        stacked_host = QWidget()
        self._stack = QStackedLayout(stacked_host)

        self.table = TableView()
        self.model = PreviewModel()
        self.table.setModel(self.model)
        # Replace Fluent default so ✓ checkboxes center in the narrow first column
        # (keeps TableBase.delegate / hover+selection wiring via setItemDelegate).
        self.table.setItemDelegate(_CenteredCheckDelegate(self.table))
        # TableView subclasses QTableView; prefer TableView enums when present
        _select_rows = getattr(TableView, "SelectRows", QAbstractItemView.SelectRows)
        _single = getattr(TableView, "SingleSelection", QAbstractItemView.SingleSelection)
        _no_edit = getattr(TableView, "NoEditTriggers", QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(_select_rows)
        self.table.setSelectionMode(_single)
        self.table.setEditTriggers(_no_edit)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionsClickable(True)
        hdr.setMinimumSectionSize(48)
        # Check Fixed; Category + Original Interactive (drag handles); New Stretch
        # fills leftover so viewport stays full without a horizontal scrollbar.
        # Dragging Original|New resizes both (Interactive next to Stretch).
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setSectionResizeMode(COL_CHECK, QHeaderView.Fixed)
        hdr.resizeSection(COL_CHECK, 34)
        hdr.resizeSection(COL_CATEGORY, 80)
        hdr.resizeSection(COL_ORIGINAL, 240)  # seed; New Stretch takes remaining
        hdr.setSectionResizeMode(COL_CATEGORY, QHeaderView.Interactive)
        hdr.setSectionResizeMode(COL_ORIGINAL, QHeaderView.Interactive)
        hdr.setSectionResizeMode(COL_NEW, QHeaderView.Stretch)
        hdr.setStretchLastSection(False)
        hdr.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hdr.sectionDoubleClicked.connect(self._on_header_double_clicked)
        self.table.verticalHeader().setDefaultSectionSize(26)
        # Category cell → log-style chip fill (inset so row lines show)
        self.table.setItemDelegateForColumn(COL_CATEGORY, _CategoryBadgeDelegate(self.table))
        self.table.clicked.connect(self._on_table_clicked)
        # Painted ✓ cells are not QCheckBox widgets — hand cursor on COL_CHECK hover.
        self.table.setMouseTracking(True)
        self.table.viewport().setMouseTracking(True)
        self.table.viewport().installEventFilter(self)
        self._stack.addWidget(self.table)
        QTimer.singleShot(0, self._fit_category_column)

        # Auto-detect log — TextEdit (not PlainTextEdit) so painted chip images
        # render (QPlainTextEdit only shows a placeholder glyph for insertImage).
        # Same LogView styling + Fluent focus-layer suppression as widgets.LogPanel.
        self.analyze_log = TextEdit()
        self.analyze_log.setObjectName("LogView")
        self.analyze_log.setReadOnly(True)
        self.analyze_log.setFont(theme.F_LOG)
        layer = getattr(self.analyze_log, "layer", None)
        if layer is not None:
            layer.hide()
            layer.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        
        t = theme.DARK
        c = theme.COLORS
        log_qss = f"""
            TextEdit#LogView,
            TextEdit#LogView:hover,
            TextEdit#LogView:focus {{
                background-color: {c['log_bg']};
                color: {c['log_fg']};
                font-family: "{theme.FONT_FAMILY_MONO}";
                font-size: {theme.LOG_FONT_PX}px;
                border: 1px solid {t['border']};
                border-bottom: 1px solid {t['border']};
                border-radius: {theme.LOG_VIEW_CORNER_RADIUS}px;
                padding: 8px;
                selection-background-color: {c['accent']};
                selection-color: #ffffff;
            }}
        """
        self.analyze_log.setStyleSheet(log_qss)
        try:
            from qfluentwidgets import setCustomStyleSheet
            setCustomStyleSheet(self.analyze_log, log_qss, log_qss)
        except Exception:
            pass
        
        # Shared chip / format / line-spacing renderer — identical to Classify LOG.
        from ..widgets.log_panel import ChipRenderer

        self._chips = ChipRenderer()
        self._stack.addWidget(self.analyze_log)
        self._stack.setCurrentWidget(self.table)

        layout.addWidget(stacked_host, stretch=1)

        # Keyboard shortcuts (documented in the Rename help dialog).
        # QShortcut with WidgetWithChildrenShortcut works regardless of which
        # child (table, header) currently holds focus — a bare keyPressEvent on
        # the panel never fires because the QTableView eats Up/Down/PageUp/Down.
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        QShortcut(QKeySequence(Qt.Key_Up), self, self._kb_prev, context=ctx)
        QShortcut(QKeySequence(Qt.Key_Down), self, self._kb_next, context=ctx)
        QShortcut(QKeySequence(Qt.Key_PageUp), self, self._kb_page_up, context=ctx)
        QShortcut(QKeySequence(Qt.Key_PageDown), self, self._kb_page_down, context=ctx)
        QShortcut(QKeySequence(Qt.Key_Space), self, self._kb_toggle_play, context=ctx)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._kb_seek_back, context=ctx)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._kb_seek_fwd, context=ctx)
        # Enter toggles ✓ for the active row — scoped to the table so line edits
        # / dialogs / tool buttons keep their own Enter behavior.
        table_ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        QShortcut(QKeySequence(Qt.Key_Return), self.table, self._kb_toggle_check, context=table_ctx)
        QShortcut(QKeySequence(Qt.Key_Enter), self.table, self._kb_toggle_check, context=table_ctx)

    # ----- public API (port method names) -----

    def set_rows(self, rows: List[PreviewRow]) -> None:
        self.model.set_tracks([r.track for r in rows])
        for i, r in enumerate(rows):
            self.model.set_row(i, r)
        self._update_stats()
        self._fit_category_column()

    def _fit_category_column(self) -> None:
        """Widen Category so the longest chip label (and header) fit."""
        from track_renamer.category_palette import DEFAULT_CATEGORY_COLORS

        fm = QFontMetrics(_CategoryBadgeDelegate._chip_font())
        labels = {"category"}  # header baseline (lowercase measure)
        for name in DEFAULT_CATEGORY_COLORS:
            labels.add(name.lower())
        for i in range(self.model.rowCount()):
            preview = self.model.row_at(i)
            if preview is None:
                continue
            parsed = parse_category_prefix_display(preview.new_display)
            if parsed:
                labels.add(parsed[0].lower())
        text_w = max((fm.horizontalAdvance(t) for t in labels), default=40)
        try:
            hdr = self.table.horizontalHeader()
        except RuntimeError:
            return
        header_w = hdr.fontMetrics().horizontalAdvance("Category")
        # Same geometry as ChipRenderer.category_pixmap (shared min width, grows).
        chip_w = max(getattr(self._chips, "chip_width_px", 0), text_w + 16)
        width = max(chip_w + 2 * _CategoryBadgeDelegate._BADGE_INSET_H, header_w + 20, 56)
        hdr.resizeSection(COL_CATEGORY, width)

    def set_loading(self, loading: bool) -> None:
        # Simple: just dim the stats label
        self.stats_label.setText("Loading…" if loading else self.stats_label.text())

    def set_preview_pending(self, pending: bool) -> None:
        self._pending_state = pending
        if pending:
            self.stats_label.setText("Rules changed — click Apply")
        else:
            self._update_stats()

    def clear_active(self) -> None:
        self._active_index = None
        self.table.clearSelection()
        if self.on_active:
            self.on_active(None, None)

    def cancel_preview_work(self) -> None:
        if self._job is not None:
            self._job.cancel.set()

    def begin_viewport_lazy(self, tracks: List[Track], rules: List[Rule], root_label: str = "") -> None:
        self.cancel_preview_work()
        self._tracks = list(tracks)
        self._root_label = root_label
        self._lazy_generation += 1
        gen = self._lazy_generation
        self._lazy_done = 0
        self._lazy_total = len(tracks)
        self._lazy_changed = 0
        self._lazy_selected_changed = 0

        try:
            prepared = prepare_rules(rules)
        except Exception:
            prepared = rules

        self.model.set_tracks(tracks)
        self._fit_category_column()

        # Seed priority with first ~LAZY_BUFFER_ROWS
        self._job = _PreviewJob(generation=gen, tracks=self._tracks, rules=prepared)
        # Priority seed only — do NOT mark requested yet or the worker skips them
        for i in range(min(LAZY_BUFFER_ROWS, len(tracks))):
            self._job.priority.put(i)

        worker = threading.Thread(target=self._preview_worker, args=(self._job,), daemon=True)
        worker.start()
        if not self._poll_timer.isActive():
            self._poll_timer.start()
        self._update_stats()

    def lazy_compute_complete(self) -> bool:
        return self._lazy_done >= self._lazy_total

    def lazy_compute_progress(self):
        return (self._lazy_done, self._lazy_total)

    def selected_renames(self) -> dict:
        if not self.lazy_compute_complete():
            return {}
        result = {}
        for i, track in enumerate(self._tracks):
            row = self.model.row_at(i)
            if row is None or not row.changed or not track.selected:
                continue
            result[track.id] = row.new_name
        return result

    def rename_count(self) -> int:
        if not self.lazy_compute_complete():
            return self._lazy_selected_changed
        n = 0
        for i, track in enumerate(self._tracks):
            row = self.model.row_at(i)
            if row is not None and row.changed and track.selected:
                n += 1
        return n

    def shutdown(self) -> None:
        self._poll_timer.stop()
        if self._job is not None:
            self._job.cancel.set()

    # ----- ANALYZE LOG -----
    # Rendering mirrors widgets.LogPanel (Classify LOG): painted chip pixmaps,
    # Consolas mono tag colors, LOG_LINE_EXTRA_PX line spacing. Categories use
    # the same color source as the preview table badge (model._category_colors
    # / default_category_color) so log chips and table chips stay in sync.

    def begin_analyze_log(self, total: int) -> None:
        self._stack.setCurrentWidget(self.analyze_log)
        self.analyze_log.clear()
        # Startup/config indented like Classify; === Summary stays flush.
        self._append_line(f"{LOG_INDENT}Starting Auto-detect (PaSST OpenMIC)…", "info")
        self._append_line(f"{LOG_INDENT}Selected {total:,} file(s).", "info")
        self._append_line(f"{LOG_INDENT}Checking cache / starting tagger…", "info")

    def append_analyze_status(self, message: str) -> None:
        text = message or ""
        if text and not text.startswith((" ", "=", "-")):
            text = f"{LOG_INDENT}{text}"
        self._append_line(text, "info")

    def append_analyze_log(self, *, filename: str, action: str, category: str = "",
                           score: float = 0.0, label: str = "", total: int = 0,
                           done: int = 0, reason: str = "") -> None:
        # Build one chip row: "  <chip>  <pct>  <filename>[  (reason · label)]"
        cursor = self.analyze_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._chips.apply_line_spacing(cursor)
        self._chips.insert(cursor, "  ")

        if action == "apply":
            cat = (category or "—").strip()
            self._chips.insert_category_chip(cursor, cat, self._category_color(cat))
            self._chips.insert(cursor, f"  {score * 100:>3.0f}%", "log_pct")
            self._chips.insert(cursor, f"  {filename}", "detail")
        elif action == "error":
            self._chips.insert_chip(cursor, "skip")
            self._chips.insert(cursor, "   —   ", "log_pct")
            self._chips.insert(cursor, f"  {filename}", "err")
            if reason:
                self._chips.insert(cursor, f"  ({reason})", "err")
        else:
            self._chips.insert_chip(cursor, "skip")
            self._chips.insert(cursor, f"  {score * 100:>3.0f}%", "log_pct")
            self._chips.insert(cursor, f"  {filename}", "warn")
            extra = reason or ""
            if label:
                extra = f"{extra} · {label}" if extra else label
            if extra:
                self._chips.insert(cursor, f"  ({extra})", "warn")

        self._chips.insert(cursor, "\n")
        self.analyze_log.setTextCursor(cursor)
        self.analyze_log.ensureCursorVisible()

    def append_analyze_summary(self, *, elapsed_sec: float, total: int) -> None:
        mins = int(elapsed_sec // 60)
        secs = int(elapsed_sec % 60)
        self._append_line("=== Instrument Summary ===", "info")
        self._append_line(f"  Total time: {mins}:{secs:02d}", "info")
        self._append_line(f"  Files: {total}", "info")
        self._append_line("DONE", "ok")
        try:
            from done_sound import play_done_sound
            play_done_sound()
        except Exception:
            pass

    def end_analyze_log(self) -> None:
        self._stack.setCurrentWidget(self.table)

    # ----- internals -----

    def _category_color(self, category: str) -> str:
        """Color for a category chip — same source as the table badge."""
        colors = getattr(self.model, "_category_colors", {}) or {}
        hex_val = colors.get(category)
        if hex_val:
            return hex_val
        return default_category_color(category)

    def _append_line(self, text: str, tag: str) -> None:
        """Append one status/summary line with Classify-LOG styling + spacing."""
        cursor = self.analyze_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._chips.apply_line_spacing(cursor)
        self._chips.insert(cursor, text + "\n", tag)
        self.analyze_log.setTextCursor(cursor)
        self.analyze_log.ensureCursorVisible()

    @staticmethod
    def _preview_worker(job: _PreviewJob) -> None:
        batch: List[tuple] = []
        next_idx = 0
        last_flush = time.monotonic()
        while not job.cancel.is_set():
            try:
                idx = job.priority.get_nowait()
            except queue.Empty:
                if next_idx >= len(job.tracks):
                    break
                idx = next_idx
                next_idx += 1
                if idx in job.requested:
                    continue
            if idx >= len(job.tracks) or idx in job.requested:
                continue
            try:
                row = compute_preview_row(job.tracks[idx], job.rules, index=idx + 1)
            except Exception:
                # Mark done so we don't spin forever; leave an empty slot in the model.
                job.requested.add(idx)
                continue
            job.requested.add(idx)
            batch.append((idx, row))
            if len(batch) >= RESULT_BATCH_SIZE or (time.monotonic() - last_flush) > 0.025:
                job.results.put((list(batch), False))
                batch.clear()
                last_flush = time.monotonic()
        if batch:
            job.results.put((list(batch), False))
        job.results.put(([], True))

    def _drain_results(self) -> None:
        if self._job is None:
            self._poll_timer.stop()
            return
        finished = False
        drained = 0
        while drained < 64:
            try:
                batch, done = self._job.results.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if done:
                finished = True
                continue
            for idx, row in batch:
                if self._job.generation != self._lazy_generation:
                    continue
                self.model.set_row(idx, row)
                self._lazy_done += 1
                if row.changed:
                    self._lazy_changed += 1
                    if row.track.selected:
                        self._lazy_selected_changed += 1
                # Active-row callback: deliver the freshly-computed PreviewRow
                if idx == self._active_index and self.on_active is not None:
                    self.on_active(row.track, row)
        # Re-seed priority for the visible viewport
        self._seed_visible_priority()
        self._update_stats()
        if self.on_change:
            self.on_change()
        job = self._job
        if job is None:
            self._poll_timer.stop()
            return
        # Only stop when *this* generation's worker is done (avoid killing a fresh job)
        if job.generation != self._lazy_generation:
            return
        if finished or self._lazy_done >= self._lazy_total:
            self._poll_timer.stop()
            self._fit_category_column()
            # Worker finished but some rows were skipped — treat as complete for UI
            if self._lazy_done < self._lazy_total and finished:
                self._lazy_done = self._lazy_total
                self._update_stats()

    def _seed_visible_priority(self) -> None:
        if self._job is None or not self._tracks:
            return
        first = self.table.rowAt(0)
        last = self.table.rowAt(self.table.viewport().height() - 1)
        if first < 0:
            first = 0
        if last < 0:
            last = min(first + LAZY_BUFFER_ROWS, len(self._tracks))
        for i in range(max(0, first - 5), min(len(self._tracks), last + LAZY_BUFFER_ROWS)):
            if i not in self._job.requested:
                self._job.priority.put(i)
                # requested is filled by the worker when it actually computes the row

    def _update_stats(self) -> None:
        if self._lazy_done >= self._lazy_total:
            total = len(self._tracks)
            changed = sum(
                1 for i in range(total)
                if self.model.row_at(i) is not None and self.model.row_at(i).changed
            )
            self.stats_label.setText(
                f"{changed} will change · {total - changed} unchanged"
            )
        else:
            self.stats_label.setText(
                f"Computing preview… {self._lazy_done}/{self._lazy_total}"
            )

    # ----- selection / interaction -----

    def eventFilter(self, obj, event):  # noqa: N802
        try:
            viewport = self.table.viewport()
        except RuntimeError:
            return super().eventFilter(obj, event)

        if obj is viewport and event.type() == QEvent.Type.MouseMove:
            idx = self.table.indexAt(event.pos())
            if idx.isValid() and idx.column() == COL_CHECK:
                viewport.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                viewport.unsetCursor()
        return super().eventFilter(obj, event)

    def _on_header_double_clicked(self, logical_index: int) -> None:
        """Auto-fit column to contents (Excel-style double-click on header)."""
        if logical_index == COL_CHECK:
            return
        hdr = self.table.horizontalHeader()
        hint = hdr.sectionSizeFromContents(logical_index).width()
        hdr.resizeSection(logical_index, max(48, min(hint + 12, 720)))

    def _on_table_clicked(self, index) -> None:
        if not index.isValid():
            return
        if index.column() == COL_CHECK:
            self._toggle_row_selected(index.row())
            return
        self._set_active(index.row())

    def _toggle_row_selected(self, row: int) -> None:
        track = self.model.track_at(row)
        if track is None:
            return
        preview = self.model.row_at(row)
        was = bool(track.selected)
        now = not was
        self.model.update_selection(row, now)
        if preview is not None and preview.changed:
            if now and not was:
                self._lazy_selected_changed += 1
            elif was and not now:
                self._lazy_selected_changed = max(0, self._lazy_selected_changed - 1)
        if self.on_change:
            self.on_change()

    def _set_active(self, row: int) -> None:
        if row < 0 or row >= len(self._tracks):
            return
        self._active_index = row
        track = self._tracks[row]
        preview = self.model.row_at(row)
        if self.on_active is not None:
            self.on_active(track, preview)

    def _select_all(self) -> None:
        for i, track in enumerate(self._tracks):
            if not track.selected:
                track.selected = True
                self.model.update_selection(i, True)
        self._lazy_selected_changed = self._lazy_changed
        if self.on_change:
            self.on_change()

    def _deselect_all(self) -> None:
        for i, track in enumerate(self._tracks):
            if track.selected:
                track.selected = False
                self.model.update_selection(i, False)
        self._lazy_selected_changed = 0
        if self.on_change:
            self.on_change()

    def _on_only_changed_toggled(self, checked: bool) -> None:
        # Simple row hide for unchanged
        for i in range(len(self._tracks)):
            preview = self.model.row_at(i)
            hide = checked and preview is not None and not preview.changed
            self.table.setRowHidden(i, hide)

    # ----- keyboard -----
    # Handled via QShortcut (see _build_ui) — the panel's keyPressEvent never
    # fires because the embedded QTableView consumes navigation keys first.

    def _kb_step(self, delta: int) -> None:
        base = self._active_index if self._active_index is not None else -1 if delta > 0 else 0
        new_idx = base + delta
        if 0 <= new_idx < len(self._tracks):
            self._set_active(new_idx)
            self.table.selectRow(new_idx)
            self.table.scrollTo(self.model.index(new_idx, 0))
            # So Enter (table-scoped) can toggle ✓ after arrow navigation.
            self.table.setFocus(Qt.ShortcutFocusReason)

    def _kb_prev(self) -> None:
        self._kb_step(-1)

    def _kb_next(self) -> None:
        self._kb_step(1)

    def _kb_page_up(self) -> None:
        # Scroll roughly a viewport of rows.
        step = max(1, self.table.viewport().height() // max(1, self.table.rowHeight(0)))
        self._kb_step(-step)

    def _kb_page_down(self) -> None:
        step = max(1, self.table.viewport().height() // max(1, self.table.rowHeight(0)))
        self._kb_step(step)

    def _kb_toggle_play(self) -> None:
        if self.on_play_pause is not None:
            self.on_play_pause()

    def _kb_toggle_check(self) -> None:
        """Toggle include/✓ for the current preview row (Enter / Return)."""
        row = self._active_index
        if row is None:
            idx = self.table.currentIndex()
            if not idx.isValid():
                return
            row = idx.row()
        self._toggle_row_selected(row)

    def _kb_seek_back(self) -> None:
        if self.on_seek is not None:
            self.on_seek(-3.0)

    def _kb_seek_fwd(self) -> None:
        if self.on_seek is not None:
            self.on_seek(3.0)
