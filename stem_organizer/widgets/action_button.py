"""Action button factory — Fluent PushButton / PrimaryPushButton variants."""
from __future__ import annotations

from typing import Callable, Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import PrimaryPushButton, PushButton, setCustomStyleSheet

from .. import theme

FluentButton = Union[PushButton, PrimaryPushButton]

# Fluent paints PrimaryPushButton labels dark unless this sheet is applied
# (same tokens as theme.style_fluent_subtree / app QSS).
_PRIMARY_LABEL_SHEET = """
PrimaryPushButton {
    color: #ffffff;
}
PrimaryPushButton:hover {
    color: #ffffff;
}
PrimaryPushButton:pressed {
    color: #ffffff;
}
PrimaryPushButton:focus {
    color: #ffffff;
}
"""


def action_button(
    text: str,
    *,
    on_click: Optional[Callable[[], None]] = None,
    parent: Optional[QWidget] = None,
    accent: bool = False,
    danger: bool = False,
    width: Optional[int] = None,
    height: int = theme.ACTION_BTN_HEIGHT,
    tip: str = "",
) -> FluentButton:
    """Build a Fluent action button (primary / normal / danger)."""
    if accent:
        btn: FluentButton = PrimaryPushButton(text, parent)
        # Dialogs (help Close, preset OK, …) are not walked by style_fluent_subtree.
        setCustomStyleSheet(btn, "", _PRIMARY_LABEL_SHEET)
    else:
        btn = PushButton(text, parent)
        if danger:
            btn.setStyleSheet(
                f"PushButton {{ color: {theme.DARK['danger']}; }}"
                f"PushButton:hover {{ background-color: {theme.DARK['danger']}; color: #ffffff; }}"
            )
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFixedHeight(height)
    btn.setFont(theme.F_ACTION)
    if width is not None:
        btn.setFixedWidth(width)
    if on_click is not None:
        btn.clicked.connect(on_click)
    if tip:
        btn.setToolTip(tip)
    return btn
