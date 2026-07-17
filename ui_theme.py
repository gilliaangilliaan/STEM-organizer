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

HEADER_DESC_FONT = ('Segoe UI', 10)
ACTION_BTN_FONT = ('Segoe UI Semibold', 10)
ACTION_BTN_PADX = 14
ACTION_BTN_PADY = 4
CTRL_FIELD_PAD = 3
CTRL_ROW_PADY = 4
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
HEADER_TOP_PAD = 8
SECTION_INNER_PAD = 16
SECTION_GAP = 14
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

    style.configure('.', background=C['bg'], foreground=C['fg'],
                    fieldbackground=C['panel2'], bordercolor=C['border'],
                    lightcolor=C['panel'], darkcolor=C['panel'],
                    troughcolor=C['panel'], focuscolor=C['accent'])

    cfgs = {
        'TFrame':                    {'background': C['bg']},
        'TLabel':                    {'background': C['bg'], 'foreground': C['fg']},
        'Dim.TLabel':                {'background': C['bg'], 'foreground': C['fg_dim']},
        'Title.TLabel':              {'background': C['bg'], 'foreground': C['fg'], 'font': title},
        'TLabelframe':               {'background': C['bg'], 'foreground': C['fg'],
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
            'background': C['bg'], 'foreground': C['fg'],
            'focuscolor': C['bg'], 'indicatorbackground': C['panel2'],
            'indicatorforeground': C['fg'],
        },
        'TRadiobutton':              {
            'background': C['bg'], 'foreground': C['fg'],
            'focuscolor': C['bg'], 'indicatorbackground': C['panel2'],
            'indicatorforeground': C['fg'],
        },
        'TButton':                   {'background': C['panel2'], 'foreground': C['fg'],
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
                  ('active', C['fg']), ('selected', C['fg']),
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
                  ('active', C['fg']), ('selected', C['fg']),
                  ('disabled', C['fg_dim']),
              ],
              indicatorbackground=[
                  ('selected', C['accent']), ('pressed', C['accent']),
                  ('active', C['panel']), ('!disabled', C['panel2']),
              ],
              indicatorcolor=[('selected', C['accent']), ('pressed', C['accent'])])
    style.map('TButton',
              background=[('active', C['panel'])],
              foreground=[('active', C['fg'])])
    style.map('Horizontal.TScale',
              background=[('active', C['bg'])])
    style.map('Class.TNotebook.Tab',
              background=[('selected', C['bg']), ('active', C['panel']), ('!selected', C['panel2'])],
              foreground=[('selected', C['fg']), ('active', C['fg']), ('!selected', C['fg_dim'])],
              expand=[('selected', [1, 1, 1, 0])])

    for k, v in (('background', C['panel2']), ('foreground', C['fg']),
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
