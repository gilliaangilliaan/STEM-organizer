from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

COLORS = {
    'bg':         '#1e1f26',
    'panel':      '#262833',
    'panel2':     '#2e3140',
    'fg':         '#e6e8ef',
    'fg_dim':     '#9aa0b4',
    'accent':     '#7c5cff',
    'accent_hov': '#9077ff',
    'danger':     '#e25c5c',
    'log_bg':     '#15161c',
    'log_fg':     '#d6dae8',
    'border':     '#3a3d4d',
    'status_trough': '#343647',
    'status_pct':    '#ffffff',
}

# Renamer-compatible dark tokens (shared CTk look across all tabs).
DARK = {
    "bg": COLORS["bg"],
    "panel": COLORS["panel"],
    "panel_2": COLORS["panel2"],
    "card": COLORS["panel2"],
    "input": COLORS["panel"],
    "control_bg": COLORS["panel2"],
    "border": COLORS["border"],
    "border_soft": COLORS["status_trough"],
    # Scrollbar thumb: dim = border; hover = neutral gray lift (no accent tint).
    "scrollbar": COLORS["border"],
    "scrollbar_hover": "#565968",
    # Labels / buttons: softer. Inputs: brighter.
    "text": COLORS["log_fg"],
    "text_dim": COLORS["fg_dim"],
    # Field labels (Stems root, Export list, …) — same as button text.
    "label": COLORS["log_fg"],
    "text_mute": "#7a8199",
    "entry_text": COLORS["fg"],
    "accent": COLORS["accent"],
    "accent_hover": COLORS["accent_hov"],
    "accent_soft": "#2a2540",
    "active_row": "#44485f",
    "loading_bg": "#2a2540",
    "changed": COLORS["accent_hov"],
    "unchanged": COLORS["fg_dim"],
    "list_bg": COLORS["log_bg"],
    "list_fg": COLORS["log_fg"],
    "waveform_bg": COLORS["log_bg"],
    "waveform_axis": "#343647",
    "waveform_playhead": "#ffffff",
    "audio": "#10b981",
    "midi": COLORS["accent"],
    "group": "#a855f7",
    "danger": COLORS["danger"],
    "badge_fg": "#ffffff",
    "btn": COLORS["panel2"],
    "btn_hover": COLORS["border"],
    "row_even": "",
    "row_odd": "",
    "category_colors": {},
    # STEM aliases used by older panels
    "fg": COLORS["log_fg"],
    "fg_dim": COLORS["fg_dim"],
    "panel2": COLORS["panel2"],
    "accent_hov": COLORS["accent_hov"],
    "log_bg": COLORS["log_bg"],
    "log_fg": COLORS["log_fg"],
    "status_trough": COLORS["status_trough"],
}

PREVIEW_LOG_FONT_FAMILY = "Consolas"
PREVIEW_LOG_FONT_SIZE = 12
PREVIEW_LOG_PCT_FONT_SIZE = 10  # confidence % in ANALYZE LOG — slightly smaller

_CTK_READY = False

# Matches Rename Files subtitle / Classify tab description.
HEADER_DESC_FONT = ('Segoe UI', 12)
HEADER_DESC_COLOR = DARK['text_dim']
# Section titles (PATHS, OPTIONS, …) — smaller than body/description.
SECTION_TITLE_SIZE = 10


def ensure_ctk_dark():
    """Import customtkinter once and force dark appearance (Renamer look)."""
    global _CTK_READY
    import customtkinter as ctk

    if not _CTK_READY:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        # Default CTk theme is Roboto 13 — sync body widgets to description size.
        from customtkinter.windows.widgets.theme import ThemeManager

        font_theme = ThemeManager.theme.get("CTkFont")
        if isinstance(font_theme, dict):
            font_theme["family"] = HEADER_DESC_FONT[0]
            font_theme["size"] = HEADER_DESC_FONT[1]
        entry_theme = ThemeManager.theme.get("CTkEntry")
        if isinstance(entry_theme, dict):
            entry_theme["text_color"] = DARK["entry_text"]
        for key in ("CTkLabel", "CTkButton", "CTkCheckBox", "CTkRadioButton", "CTkOptionMenu"):
            widget_theme = ThemeManager.theme.get(key)
            if isinstance(widget_theme, dict) and "text_color" in widget_theme:
                widget_theme["text_color"] = DARK["text"]
        _CTK_READY = True
    return ctk


def ctk_ui_font(*, weight: str = "normal"):
    """Body / label font — same size as tab description line."""
    ctk = ensure_ctk_dark()
    return ctk.CTkFont(
        family=HEADER_DESC_FONT[0],
        size=HEADER_DESC_FONT[1],
        weight=weight,
    )


def ttk_ui_font():
    """Tk/ttk font matching ctk_ui_font() (CTk size is pixels → negative Tk size)."""
    return (HEADER_DESC_FONT[0], -HEADER_DESC_FONT[1])


def ctk_section_font():
    """Section titles (PATHS, OPTIONS, …) — smaller than body."""
    ctk = ensure_ctk_dark()
    return ctk.CTkFont(
        family=HEADER_DESC_FONT[0], size=SECTION_TITLE_SIZE, weight="bold",
    )


# Path-row button sizes (ctk_path_row Browse / Open).
PATH_BTN_WIDTH_BROWSE = 72
PATH_BTN_WIDTH_OPEN = 64
PATH_BTN_HEIGHT = 30
# Compact CTAs (Rename Apply / in-card compact buttons).
COMPACT_BTN_HEIGHT = 24
# Shared bottom action-bar gap between buttons.
ACTION_BTN_GAP = 6


def ctk_size_subtab_buttons(tabview, widths=None, height: int = PATH_BTN_HEIGHT):
    """Size CTkTabview segments like Browse / Open (fixed width, height 30)."""
    sb = tabview._segmented_button
    names = list(sb._buttons_dict.keys())
    if not names:
        return
    if widths is None:
        widths = [PATH_BTN_WIDTH_BROWSE, PATH_BTN_WIDTH_OPEN]
    while len(widths) < len(names):
        widths = list(widths) + [PATH_BTN_WIDTH_BROWSE]
    font = ctk_ui_font()
    total = 0
    for name, w in zip(names, widths):
        sb._buttons_dict[name].configure(width=w, height=height, font=font)
        total += w
    sb.configure(height=height, width=total, dynamic_resizing=False)


def ctk_action_button(
    parent,
    text: str,
    command,
    *,
    accent: bool = False,
    width: int | None = None,
    height: int = PATH_BTN_HEIGHT,
    text_color: str | None = None,
    hover_color: str | None = None,
    cursor: str = "hand2",
):
    """CTk button matching Browse / Classify action-bar style."""
    ctk = ensure_ctk_dark()
    t = DARK
    if width is None:
        width = max(72, int(len(text) * 7.6) + 28)
    if accent:
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            font=ctk_ui_font(),
            width=width,
            height=height,
            fg_color=t["accent"],
            hover_color=hover_color or t["accent_hover"],
            text_color=text_color or "#ffffff",
            cursor=cursor,
        )
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        font=ctk_ui_font(),
        width=width,
        height=height,
        fg_color=t["btn"],
        hover_color=hover_color or t["btn_hover"],
        text_color=text_color or t["text"],
        cursor=cursor,
    )


def ctk_pin_button_height(*buttons, height: int = PATH_BTN_HEIGHT) -> None:
    """Re-apply fixed height after configure/pack (CTk can shrink on Windows)."""
    for btn in buttons:
        if btn is None:
            continue
        try:
            # configure() alone is not always enough after parent pack_forget
            # (Rename layout) — force CTk internal size + redraw.
            if hasattr(btn, "_set_dimensions"):
                btn._set_dimensions(height=height)
            else:
                btn.configure(height=height)
            btn.configure(height=height)
        except Exception:
            pass


def ctk_fixed_btn_slot(parent, *, width: int, height: int = PATH_BTN_HEIGHT):
    """Fixed-size shell so pack overflow cannot shrink a CTkButton canvas."""
    ctk = ensure_ctk_dark()
    slot = ctk.CTkFrame(
        parent, fg_color="transparent", width=width, height=height,
    )
    try:
        slot.pack_propagate(False)
    except Exception:
        pass
    return slot


def ctk_section(
    parent,
    title: str,
    *,
    padx=None,
    pady=None,
    inner_padx=None,
    inner_pady=None,
    title_pady=(0, 3),
):
    """Labeled CTk section matching Renamer card/panel look."""
    ctk = ensure_ctk_dark()
    t = DARK
    if padx is None:
        padx = SECTION_PADX
    if pady is None:
        pady = (0, SECTION_GAP)
    if inner_padx is None:
        inner_padx = SECTION_INNER_PAD
    if inner_pady is None:
        inner_pady = SECTION_INNER_PAD
    wrap = ctk.CTkFrame(parent, fg_color="transparent")
    wrap.pack(fill="x", padx=padx, pady=pady)
    ctk.CTkLabel(
        wrap,
        text=title.upper(),
        font=ctk_section_font(),
        text_color=t["text_dim"],
        anchor="w",
    ).pack(anchor="w", pady=title_pady)
    body = ctk.CTkFrame(
        wrap,
        fg_color=t["panel"],
        border_color=t["border"],
        border_width=1,
        corner_radius=8,
    )
    body.pack(fill="x")
    inner = ctk.CTkFrame(body, fg_color="transparent")
    inner.pack(fill="x", padx=inner_padx, pady=inner_pady)
    return inner


def ctk_path_row(
    parent,
    row: int,
    label: str,
    var: tk.Variable,
    browse_cmd,
    open_cmd,
    *,
    tip_text: str = "",
    open_tip: str = "Open this folder in Explorer.",
):
    """Grid path row: label + entry + Browse + Open (CTk)."""
    ctk = ensure_ctk_dark()
    t = DARK
    _font = ctk_ui_font()
    lbl = ctk.CTkLabel(
        parent, text=label, text_color=t["label"], font=_font,
    )
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=CTRL_ROW_PADY)
    ent = ctk.CTkEntry(
        parent,
        textvariable=var,
        fg_color=t["control_bg"],
        border_color=t["border"],
        text_color=t["entry_text"],
        font=_font,
        height=30,
    )
    ent.grid(row=row, column=1, sticky="ew", pady=CTRL_ROW_PADY)
    browse = ctk.CTkButton(
        parent,
        text="Browse",
        width=72,
        height=30,
        fg_color=t["btn"],
        hover_color=t["btn_hover"],
        text_color=t["text"],
        font=_font,
        command=browse_cmd,
    )
    browse.grid(row=row, column=2, padx=(4, 0), pady=CTRL_ROW_PADY)
    open_btn = ctk.CTkButton(
        parent,
        text="Open",
        width=64,
        height=30,
        fg_color=t["btn"],
        hover_color=t["btn_hover"],
        text_color=t["text"],
        font=_font,
        command=open_cmd,
    )
    open_btn.grid(row=row, column=3, padx=(4, 0), pady=CTRL_ROW_PADY)
    if tip_text:
        tip(lbl, ent, browse, text=tip_text)
    tip(open_btn, text=open_tip)
    return lbl, ent, browse, open_btn


def apply_toplevel_icon(win: tk.Misc):
    """Apply logo.ico to a Toplevel. Lazy-import avoids cycle with stem_organizer_ui."""
    try:
        from stem_organizer_ui import apply_window_icon
        photo = apply_window_icon(win)
        if photo is not None:
            # Keep reference so Tk does not GC the PhotoImage.
            win._app_icon_photo = photo  # type: ignore[attr-defined]
        return photo
    except Exception:
        return None


def apply_toplevel_rounded_corners(win: tk.Misc, *, maximized: bool = False):
    """Round decorated Toplevels via DWM (Win11). Not SetWindowRgn — that whites out the frame."""
    try:
        from stem_organizer_ui import _win_apply_dwm_rounded_corners
        _win_apply_dwm_rounded_corners(win, maximized=maximized)
    except Exception:
        pass


def _center_toplevel(win: tk.Misc, parent: tk.Misc) -> None:
    win.update_idletasks()
    try:
        pw = parent.winfo_toplevel()
        x = pw.winfo_rootx() + (pw.winfo_width() - win.winfo_reqwidth()) // 2
        y = pw.winfo_rooty() + (pw.winfo_height() - win.winfo_reqheight()) // 2
        win.geometry(f'+{max(0, x)}+{max(0, y)}')
    except tk.TclError:
        pass


def show_info_dark(parent: tk.Misc, title: str, message: str) -> None:
    """Dark themed OK dialog (no Windows MessageBox sound / light chrome)."""
    import customtkinter as ctk

    root = parent.winfo_toplevel()
    win = ctk.CTkToplevel(root)
    win.title(title)
    win.transient(root)
    win.resizable(False, False)
    apply_toplevel_icon(win)
    win.grab_set()

    frame = ctk.CTkFrame(win, fg_color='transparent')
    frame.pack(fill='both', expand=True, padx=20, pady=16)
    ctk.CTkLabel(
        frame,
        text=message,
        justify='left',
        wraplength=420,
        font=ctk.CTkFont(family='Segoe UI', size=13),
    ).pack(anchor='w')
    btn_row = ctk.CTkFrame(frame, fg_color='transparent')
    btn_row.pack(fill='x', pady=(16, 0))

    def _close() -> None:
        try:
            win.grab_release()
        except tk.TclError:
            pass
        win.destroy()

    ctk.CTkButton(btn_row, text='OK', width=88, command=_close).pack(side='right')
    win.protocol('WM_DELETE_WINDOW', _close)
    win.after(10, lambda: apply_toplevel_icon(win))
    _center_toplevel(win, parent)
    win.focus_force()
    win.wait_window()


def ask_yes_no_dark(
    parent: tk.Misc,
    title: str,
    message: str,
    *,
    yes_text: str = 'Yes',
    no_text: str = 'No',
) -> bool:
    """Dark themed Yes/No dialog. Returns True if Yes."""
    import customtkinter as ctk

    root = parent.winfo_toplevel()
    result = {'ok': False}
    win = ctk.CTkToplevel(root)
    win.title(title)
    win.transient(root)
    win.resizable(False, False)
    apply_toplevel_icon(win)
    win.grab_set()

    frame = ctk.CTkFrame(win, fg_color='transparent')
    frame.pack(fill='both', expand=True, padx=20, pady=16)
    ctk.CTkLabel(
        frame,
        text=message,
        justify='left',
        wraplength=420,
        font=ctk.CTkFont(family='Segoe UI', size=13),
    ).pack(anchor='w')
    btn_row = ctk.CTkFrame(frame, fg_color='transparent')
    btn_row.pack(fill='x', pady=(16, 0))

    def _finish(ok: bool) -> None:
        result['ok'] = ok
        try:
            win.grab_release()
        except tk.TclError:
            pass
        win.destroy()

    ctk.CTkButton(
        btn_row, text=no_text, width=88, fg_color=DARK['panel_2'],
        hover_color=DARK['border'], command=lambda: _finish(False),
    ).pack(side='right', padx=(8, 0))
    ctk.CTkButton(
        btn_row, text=yes_text, width=88, command=lambda: _finish(True),
    ).pack(side='right')
    win.protocol('WM_DELETE_WINDOW', lambda: _finish(False))
    win.after(10, lambda: apply_toplevel_icon(win))
    _center_toplevel(win, parent)
    win.focus_force()
    win.wait_window()
    return bool(result['ok'])


def show_ctk_help_dialog(
    parent: tk.Misc,
    *,
    title: str,
    heading: str,
    intro: str,
    sections: list[tuple[str, str]],
    footer_note: str = "Hover over individual controls for more detail.",
) -> None:
    """Shared dark help dialog (tk cards; Close = Browse/Open CTk style)."""
    t = DARK
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.configure(bg=t["panel"])
    dialog.resizable(False, False)
    dialog.transient(parent.winfo_toplevel())
    apply_toplevel_icon(dialog)

    outer = tk.Frame(dialog, bg=t["panel"])
    outer.pack(fill="both", expand=True, padx=22, pady=18)
    tk.Label(
        outer, text=heading, font=("Segoe UI Semibold", 18),
        fg=t["text"], bg=t["panel"],
    ).pack(anchor="w")
    tk.Label(
        outer, text=intro, font=("Segoe UI", 10),
        fg=t["text_dim"], bg=t["panel"],
    ).pack(anchor="w", pady=(2, 14))

    for section_title, body in sections:
        card = tk.Frame(
            outer, bg=t["card"],
            highlightbackground=t["border"], highlightthickness=1,
        )
        card.pack(fill="x", pady=(0, 10))
        tk.Label(
            card, text=section_title.upper(), font=("Segoe UI Semibold", 9),
            fg=t["accent_hover"], bg=t["card"],
        ).pack(anchor="w", padx=14, pady=(10, 4))
        tk.Label(
            card, text=body, font=("Segoe UI", 10),
            fg=t["text_dim"], bg=t["card"],
            justify="left", anchor="w", wraplength=600,
        ).pack(fill="x", padx=14, pady=(0, 11))

    footer = tk.Frame(outer, bg=t["panel"])
    footer.pack(fill="x", pady=(2, 0))
    tk.Label(
        footer, text=footer_note, font=("Segoe UI", 9),
        fg=t["text_dim"], bg=t["panel"],
    ).pack(side="left")
    close = ctk_action_button(footer, "Close", dialog.destroy, width=72)
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
    dialog.update_idletasks()
    apply_toplevel_rounded_corners(dialog)
    # DWM corner preference sticks more reliably after the window is mapped.
    dialog.after(20, lambda: apply_toplevel_rounded_corners(dialog))
    dialog.grab_set()
    close.focus_set()


def _blend_hex(fg: str, bg: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    fg = fg.lstrip('#')
    bg = bg.lstrip('#')
    fr, fg_g, fb = (int(fg[i:i + 2], 16) for i in (0, 2, 4))
    br, bg_g, bb = (int(bg[i:i + 2], 16) for i in (0, 2, 4))
    return (
        f'#{int(fr + (br - fr) * t):02x}'
        f'{int(fg_g + (bg_g - fg_g) * t):02x}'
        f'{int(fb + (bb - fb) * t):02x}'
    )


def _entry_select_colors() -> tuple[str, str]:
    c = COLORS
    return (
        _blend_hex(c['accent'], c['panel2'], 0.36),
        _blend_hex(c['accent'], c['panel2'], 0.58),
    )

ACTION_BTN_FONT = ('Segoe UI Semibold', 10)
ACTION_BTN_PADX = 14
ACTION_BTN_PADY = 4
CTRL_FIELD_PAD = 3
CTRL_ROW_PADY = 2
PATH_BTN_FONT = ('Segoe UI', 10)
PATH_BTN_PADX = 8
PATH_BTN_PADY = 4
STATUS_FONT = ('Segoe UI', 9)
LOG_FONT = ('Consolas', 10)
LOG_WARN_COLOR = '#ecc990'

CTRL_BTN_PADY = 4
WIN_DEFAULT_W = 1290
TITLE_BAR_HEIGHT = 0
_USE_CUSTOM_TITLE_BAR = False
WIN_DEFAULT_H = 1020
WIN_MIN_W = 860
WIN_MIN_H = 620
LEFT_PANEL_WIDTH = 540
CONTENT_PAD = 18
HEADER_TOP_PAD = 4
SECTION_INNER_PAD = 10
SECTION_GAP = 8
SECTION_SIDE_PAD_LEFT = 14
SECTION_SIDE_PAD_RIGHT = 4
SECTION_PADX = (SECTION_SIDE_PAD_LEFT, SECTION_SIDE_PAD_RIGHT)
LOG_INNER_PAD = 14
ACTIONS_BOTTOM_PAD = 12

RESOURCE_BAR_HEIGHT = 10
RESOURCE_BAR_WIDTH = 52
RESOURCE_ROW_HEIGHT = 16
STATUS_PROGRESS_ROW_HEIGHT = 14
STATUS_PAD_TOP = 3
STATUS_ROW_GAP = 7
STATUS_PAD_BOTTOM = 26
STATUS_FRAME_HEIGHT = (
    STATUS_PAD_TOP + RESOURCE_ROW_HEIGHT + STATUS_ROW_GAP
    + STATUS_PROGRESS_ROW_HEIGHT + STATUS_PAD_BOTTOM
)
STATUS_IDLE_Y = STATUS_PAD_TOP + RESOURCE_ROW_HEIGHT + STATUS_ROW_GAP
STATUS_PAD_X = 10
STATUS_PCT_FONT = ('Segoe UI Semibold', 9)
STATUS_PROGRESS_HEIGHT = 14
STATUS_PROGRESS_Y_PAD = 1
STATUS_TOP_PAD = 0
STATUS_BOTTOM_PAD = 2


def format_status_clock(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return '--:--:--'
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f'{hours}:{minutes:02d}:{secs:02d}'


def format_eta(seconds: float | None) -> str:
    return format_status_clock(seconds)


class DarkScrollbar(tk.Canvas):
    WIDTH = 14

    def __init__(self, master, command=None, **kwargs):
        super().__init__(
            master, width=self.WIDTH, highlightthickness=0, borderwidth=0,
            background=COLORS['log_bg'], relief='flat', **kwargs,
        )
        self._command = command
        self._top = 0.0
        self._bottom = 1.0
        self._thumb_y0 = 0.0
        self._thumb_y1 = 0.0
        self._drag_offset = 0.0
        self._visible = False
        self.bind('<Configure>', lambda _e: self._redraw())
        self.bind('<Button-1>', self._press)
        self.bind('<B1-Motion>', self._drag)

    def set(self, first, last):
        self._top = float(first)
        self._bottom = float(last)
        self._redraw()

    def _redraw(self):
        self.delete('all')
        height = max(int(self.winfo_height()), 1)
        if self._bottom - self._top >= 1.0:
            self._visible = False
            return
        self._visible = True
        y0 = height * self._top
        y1 = height * self._bottom
        min_thumb = 28
        if y1 - y0 < min_thumb:
            y1 = min(y0 + min_thumb, height)
        self._thumb_y0, self._thumb_y1 = y0, y1
        self.create_rectangle(
            2, y0, self.WIDTH - 2, y1,
            fill=COLORS['panel2'], outline=COLORS['border'], width=1,
        )

    def _press(self, event):
        if not self._visible:
            return
        if self._thumb_y0 <= event.y <= self._thumb_y1:
            self._drag_offset = event.y - self._thumb_y0
        else:
            self._drag_offset = (self._thumb_y1 - self._thumb_y0) / 2
            self._jump(event.y - self._drag_offset)

    def _drag(self, event):
        if self._visible:
            self._jump(event.y - self._drag_offset)

    def _jump(self, thumb_top):
        height = max(int(self.winfo_height()), 1)
        thumb_h = max(self._thumb_y1 - self._thumb_y0, 1)
        span = max(height - thumb_h, 1)
        frac = max(0.0, min(1.0, thumb_top / span))
        if self._command:
            self._command('moveto', frac)


def bind_mousewheel(widget: tk.Widget, yview) -> None:
    def on_wheel(event):
        yview('scroll', int(-event.delta / 120), 'units')
        return 'break'

    def on_enter(_event):
        widget.bind_all('<MouseWheel>', on_wheel)

    def on_leave(_event):
        widget.unbind_all('<MouseWheel>')

    widget.bind('<Enter>', on_enter, add='+')
    widget.bind('<Leave>', on_leave, add='+')


def apply_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass

    base = ('Segoe UI', 10)
    bold = ('Segoe UI Semibold', 10)
    section = ('Segoe UI Semibold', 8)
    title = ('Segoe UI Semibold', 16)
    C = COLORS
    select_active, select_inactive = _entry_select_colors()
    action_pad = (ACTION_BTN_PADX, ACTION_BTN_PADY)

    root.configure(bg=C['bg'])
    root.option_add('*Font', base)
    root.option_add('*selectBackground', select_active)
    root.option_add('*selectForeground', C['fg'])
    root.option_add('*inactiveSelectBackground', select_inactive)

    ui_fg = C['log_fg']  # softer labels/buttons; entries keep brighter C['fg']
    style.configure('.', background=C['bg'], foreground=ui_fg,
                    fieldbackground=C['panel2'], bordercolor=C['border'],
                    lightcolor=C['panel'], darkcolor=C['panel'],
                    troughcolor=C['panel'], focuscolor=C['accent'])

    cfgs = {
        'TFrame':                    {'background': C['bg']},
        'TLabel':                    {'background': C['bg'], 'foreground': ui_fg},
        'Dim.TLabel':                {'background': C['bg'], 'foreground': C['fg_dim']},
        'Title.TLabel':              {'background': C['bg'], 'foreground': ui_fg, 'font': title},
        'TLabelframe':               {'background': C['bg'], 'foreground': ui_fg,
                                      'bordercolor': C['border'], 'relief': 'solid', 'borderwidth': 1},
        'TLabelframe.Label':         {'background': C['bg'], 'foreground': C['fg_dim'], 'font': section},
        'TEntry':                    {'fieldbackground': C['panel2'], 'foreground': C['fg'],
                                      'bordercolor': C['border'], 'insertcolor': C['fg'],
                                      'selectbackground': select_active,
                                      'selectforeground': C['fg'],
                                      'padding': CTRL_FIELD_PAD},
        'TCombobox':                 {'fieldbackground': C['panel2'], 'background': C['panel2'],
                                      'foreground': C['fg'], 'arrowcolor': C['fg_dim'],
                                      'bordercolor': C['border'], 'padding': CTRL_FIELD_PAD,
                                      'selectbackground': select_active,
                                      'selectforeground': C['fg']},
        'TCheckbutton':              {
            'background': C['bg'], 'foreground': ui_fg,
            'focuscolor': C['bg'], 'indicatorbackground': C['panel2'],
            'indicatorforeground': ui_fg,
        },
        'TRadiobutton':              {
            'background': C['bg'], 'foreground': ui_fg,
            'focuscolor': C['bg'], 'indicatorbackground': C['panel2'],
            'indicatorforeground': ui_fg,
        },
        'TButton':                   {'background': C['panel2'], 'foreground': ui_fg,
                                      'bordercolor': C['border'], 'padding': (14, 8), 'borderwidth': 1},
        'Horizontal.TScale':         {'background': C['bg'], 'troughcolor': C['panel2'],
                                      'bordercolor': C['border'],
                                      'lightcolor': C['accent'], 'darkcolor': C['accent']},
        'Horizontal.TProgressbar':   {'background': C['accent'], 'troughcolor': C['panel2'],
                                      'bordercolor': C['panel2'],
                                      'lightcolor': C['accent'], 'darkcolor': C['accent']},
        'Class.TNotebook':           {'background': C['bg'], 'borderwidth': 0,
                                      'tabmargins': [2, 0, 2, 0]},
        'Class.TNotebook.Tab':       {'background': C['panel2'], 'foreground': C['fg_dim'],
                                      'padding': (14, 4), 'font': bold, 'borderwidth': 1,
                                      'bordercolor': C['border']},
    }
    for name, opts in cfgs.items():
        style.configure(name, **opts)

    # clam wraps the label in a Focus element that paints a solid white hover slab
    # on Windows — drop that wrapper so hover keeps readable dark-theme text.
    style.layout(
        'TRadiobutton',
        [
            (
                'Radiobutton.padding',
                {
                    'sticky': 'nswe',
                    'children': [
                        ('Radiobutton.indicator', {'side': 'left', 'sticky': ''}),
                        ('Radiobutton.label', {'side': 'left', 'sticky': 'nswe'}),
                    ],
                },
            )
        ],
    )
    style.layout(
        'TCheckbutton',
        [
            (
                'Checkbutton.padding',
                {
                    'sticky': 'nswe',
                    'children': [
                        ('Checkbutton.indicator', {'side': 'left', 'sticky': ''}),
                        ('Checkbutton.label', {'side': 'left', 'sticky': 'nswe'}),
                    ],
                },
            )
        ],
    )

    style.map('TEntry',
              bordercolor=[('focus', C['accent'])],
              selectbackground=[('focus', select_active), ('!focus', select_inactive)],
              selectforeground=[('focus', C['fg']), ('!focus', C['fg_dim'])])
    style.map('TCombobox',
              fieldbackground=[('readonly', C['panel2']), ('!disabled', C['panel2'])],
              background=[('readonly', C['panel2']), ('active', C['panel2'])],
              bordercolor=[('focus', C['accent'])],
              selectbackground=[('focus', select_active), ('!focus', select_inactive)],
              selectforeground=[('focus', C['fg']), ('!focus', C['fg_dim'])])
    style.map('TCheckbutton',
              background=[
                  ('active', C['bg']), ('pressed', C['bg']),
                  ('selected', C['bg']), ('hover', C['bg']),
              ],
              foreground=[
                  ('active', ui_fg), ('selected', ui_fg),
                  ('disabled', C['fg_dim']),
              ],
              indicatorbackground=[
                  ('selected', C['accent']), ('pressed', C['accent']),
                  ('active', C['panel']), ('!disabled', C['panel2']),
              ],
              indicatorcolor=[('selected', C['accent']), ('pressed', C['accent'])])
    style.map('TRadiobutton',
              background=[
                  ('active', C['bg']), ('pressed', C['bg']),
                  ('selected', C['bg']), ('hover', C['bg']),
              ],
              foreground=[
                  ('active', ui_fg), ('selected', ui_fg),
                  ('disabled', C['fg_dim']),
              ],
              indicatorbackground=[
                  ('selected', C['accent']), ('pressed', C['accent']),
                  ('active', C['panel']), ('!disabled', C['panel2']),
              ],
              indicatorcolor=[('selected', C['accent']), ('pressed', C['accent'])])
    style.map('TButton',
              background=[('active', C['panel'])],
              foreground=[('active', ui_fg)])
    style.map('Horizontal.TScale',
              background=[('active', C['bg'])])
    style.map('Class.TNotebook.Tab',
              background=[('selected', C['bg']), ('active', C['panel']), ('!selected', C['panel2'])],
              foreground=[('selected', ui_fg), ('active', ui_fg), ('!selected', C['fg_dim'])],
              expand=[('selected', [1, 1, 1, 0])])

    for k, v in (('background', C['panel2']), ('foreground', ui_fg),
                 ('selectBackground', select_active), ('selectForeground', C['fg'])):
        root.option_add(f'*TCombobox*Listbox.{k}', v)


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str, delay: int = 550, wrap: int = 340):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wrap = wrap
        self._after = None
        self._tip = None
        widget.bind('<Enter>', self._schedule, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None

    def _show(self):
        if self._tip is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except tk.TclError:
            return
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f'+{x}+{y}')
        border = tk.Frame(tw, background=COLORS['border'])
        border.pack()
        tk.Label(
            border, text=self.text, justify='left', wraplength=self.wrap,
            background=COLORS['log_bg'], foreground=COLORS['log_fg'],
            padx=8, pady=6, font=('Segoe UI', 9),
        ).pack(padx=1, pady=1)

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


def tip(*widgets: tk.Widget, text: str) -> None:
    for widget in widgets:
        Tooltip(widget, text)


def display_path(path: str) -> str:
    text = (path or '').strip()
    if not text:
        return ''
    if sys.platform == 'win32':
        return text.replace('/', '\\')
    return text


def place_window_centered(root: tk.Misc, width: int, height: int) -> tuple[int, int, int, int]:
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    width = max(WIN_MIN_W, width)
    height = max(WIN_MIN_H, height)
    x = max(0, (sw - width) // 2)
    y = max(0, (sh - height) // 2)
    return x, y, width, height
