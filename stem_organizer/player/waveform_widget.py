"""Waveform widget — QPainter polygon + playhead + filename overlay.

Replaces stem_player._draw_waveform / _update_playhead / _draw_waveform_filename
(the tk.Canvas calls). The window owns the math (view_start, view_zoom, peaks);
this widget just paints whatever the window tells it to via set_peaks().
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .. import theme


class WaveformWidget(QWidget):
    """Paints one track's waveform + playhead + filename overlay."""

    # Fraction 0..1 across the drawable width (avoids stale cached pixel widths).
    clicked = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self._peaks: Optional[np.ndarray] = None
        self._color: QColor = QColor(theme.COLORS["accent"])
        self._dimmed_color: QColor = QColor(theme.COLORS["fg_dim"])
        self._dimmed: bool = False
        self._playhead_x: Optional[float] = None
        self._filename: str = ""

    # ----- configuration -----

    def set_peaks(self, peaks: Optional[np.ndarray]) -> None:
        self._peaks = peaks
        self.update()

    def set_color(self, hex_color: str, *, dimmed: bool = False) -> None:
        self._color = QColor(hex_color)
        self._dimmed = dimmed
        # Pre-blend dimmed variant with background for a softer look
        if dimmed:
            self._color = self._blend(QColor(hex_color), QColor(theme.COLORS["log_bg"]), 0.68)
        self.update()

    def set_playhead(self, frac: Optional[float]) -> None:
        """Playhead as 0..1 across the drawable width, or None to hide."""
        self._playhead_x = frac
        self.update()

    def set_filename(self, text: str) -> None:
        self._filename = text
        self.update()

    @staticmethod
    def _blend(fg: QColor, bg: QColor, t: float) -> QColor:
        r = int(fg.red() * (1 - t) + bg.red() * t)
        g = int(fg.green() * (1 - t) + bg.green() * t)
        b = int(fg.blue() * (1 - t) + bg.blue() * t)
        return QColor(r, g, b)

    # ----- paint -----

    def mousePressEvent(self, event) -> None:  # noqa: N802 Qt name
        if event.button() == Qt.LeftButton:
            # Map click against this widget's live width (CTk uses event.widget
            # winfo_width). Emitting a fraction avoids stale _wave_w caches and
            # any DPI/margin mismatch between click X and a foreign width.
            rect = self.contentsRect()
            w = max(1.0, float(rect.width()))
            x = float(event.position().x()) - float(rect.x())
            self.clicked.emit(max(0.0, min(1.0, x / w)))

    def paintEvent(self, event) -> None:  # noqa: N802 Qt name
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        rect = self.contentsRect()
        w = max(1, rect.width())
        h = max(1, rect.height())
        ox, oy = rect.x(), rect.y()

        # Background (full widget)
        p.fillRect(self.rect(), QColor(theme.COLORS["log_bg"]))

        peaks = self._peaks
        if peaks is None or len(peaks) == 0:
            # No data: dim center line
            p.setPen(QPen(QColor(theme.COLORS["border"]), 1))
            p.drawLine(ox, oy + h // 2, ox + w, oy + h // 2)
            self._draw_filename(p, ox, oy, w, h)
            self._draw_playhead(p, ox, oy, w, h)
            return

        peaks = np.asarray(peaks, dtype=np.float32)
        n = len(peaks)
        mid = oy + h / 2.0
        max_amp = h / 2.0 - 6
        amps = np.minimum(peaks, 1.0) * max_amp
        bar_w = w / n
        x_left = ox + np.arange(n, dtype=np.float32) * bar_w
        x_right = x_left + max(1.0, bar_w - 0.5)
        y_top = mid - amps
        y_bot = mid + amps

        path = QPainterPath()
        path.moveTo(float(x_left[0]), float(y_top[0]))
        for i in range(n):
            path.lineTo(float(x_right[i]), float(y_top[i]))
        for i in range(n - 1, -1, -1):
            path.lineTo(float(x_left[i]), float(y_bot[i]))
        path.closeSubpath()
        p.fillPath(path, self._color)

        self._draw_filename(p, ox, oy, w, h)
        self._draw_playhead(p, ox, oy, w, h)

    def _draw_filename(self, p: QPainter, ox: int, oy: int, w: int, h: int) -> None:
        if not self._filename:
            return
        p.setRenderHint(QPainter.TextAntialiasing, True)
        font = p.font()
        font.setPointSize(7)
        p.setFont(font)
        metrics = p.fontMetrics()
        text_w = metrics.horizontalAdvance(self._filename)
        text_h = metrics.height()
        pad_x, pad_y = 6, 3
        rect_x = ox + w - text_w - pad_x * 2 - 4
        rect_y = oy + h - text_h - pad_y * 2 - 4
        if rect_x < ox or rect_y < oy:
            return
        bg = QColor(theme.COLORS["panel"])
        bg.setAlpha(180)
        p.fillRect(QRectF(rect_x, rect_y, text_w + pad_x * 2, text_h + pad_y * 2), bg)
        p.setPen(QColor(theme.COLORS["fg"]))
        p.drawText(
            QRectF(rect_x + pad_x, rect_y + pad_y, text_w, text_h),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._filename,
        )

    def _draw_playhead(self, p: QPainter, ox: int, oy: int, w: int, h: int) -> None:
        if self._playhead_x is None:
            return
        # _playhead_x is 0..1 fraction of the drawable width
        x = ox + float(self._playhead_x) * w
        if ox <= x <= ox + w:
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawLine(int(round(x)), oy, int(round(x)), oy + h)
