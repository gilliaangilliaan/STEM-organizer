"""Entry point: QApplication + splash + single-instance + main window.

Stage 9 wires together:
  - single_instance.acquire_single_instance() gate
  - QSplashScreen + StartupWorker (deps_bootstrap + classify_backend._init_ml)
  - MainWindow construction after startup completes
  - update_checker background thread
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from . import theme
from .app import MainWindow
from .settings_store import SettingsStore, app_dir
from .splash import show_splash_and_startup
from .widgets.dialogs import show_info


def _startup_error_report(exc: BaseException) -> str:
    """Full diagnostics for startup_error.log (exc may no longer be 'active')."""
    parts: list[str] = [
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "",
        f"executable: {sys.executable}",
        f"frozen: {getattr(sys, 'frozen', False)}",
        f"_MEIPASS: {getattr(sys, '_MEIPASS', None)}",
        f"cwd: {os.getcwd()}",
        f"python: {sys.version}",
        "",
        "sys.path:",
    ]
    parts.extend(f"  {p}" for p in sys.path)
    parts.append("")
    parts.append("site-packages candidates:")
    try:
        from deps_bootstrap import app_dir as deps_app_dir
        from deps_bootstrap import external_site_dirs

        parts.append(f"  deps app_dir: {deps_app_dir()}")
        for path in external_site_dirs():
            parts.append(f"  {path}  exists={path.is_dir()}")
    except Exception as list_exc:
        parts.append(f"  (could not list: {list_exc})")
    return "\n".join(parts)


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("STEM organizer")
    app.setApplicationDisplayName("STEM organizer")
    app.setApplicationVersion(theme.APP_VERSION)

    icon_path = Path(__file__).resolve().parent.parent / "logo.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    theme.apply_theme(app)

    # Single instance gate
    try:
        from single_instance import acquire_single_instance

        if not acquire_single_instance():
            show_info(None, "STEM organizer", "Another instance is already running.")
            return 0
    except Exception:
        # single_instance optional — continue if missing
        pass

    settings = SettingsStore()

    # Splash + startup
    splash_shown = False
    try:
        from .splash import Splash  # noqa: F401 — verify import works

        splash_shown = True
    except Exception:
        splash_shown = False

    if splash_shown:
        splash, worker = show_splash_and_startup(on_ready=lambda exc: _on_ready(app, settings, exc))
        return app.exec()
    else:
        _construct_and_show(app, settings, None)
        return app.exec()


def _on_ready(app: QApplication, settings: SettingsStore, startup_error) -> None:
    if startup_error is not None:
        report = _startup_error_report(startup_error)
        try:
            sys.stderr.write(report + "\n")
        except Exception:
            pass
        try:
            log_path = app_dir() / "startup_error.log"
            log_path.write_text(report, encoding="utf-8")
        except OSError:
            pass
        # Keep dialog readable; full traceback + sys.path are in the log.
        msg = str(startup_error).strip() or repr(startup_error)
        if len(msg) > 1800:
            msg = msg[:1800] + "\n…"
        show_info(
            None,
            "STEM organizer — startup failed",
            f"Startup failed:\n{msg}\n\nDetails were written to startup_error.log.",
        )
        sys.exit(1)
    _construct_and_show(app, settings, startup_error)


def _construct_and_show(app: QApplication, settings: SettingsStore, startup_error) -> None:
    window = MainWindow(settings)

    from .tabs import register_all_tabs

    register_all_tabs(window, settings)

    theme.polish_fluent_controls(window)
    # Rename Apply clones Clear width after Clear gets polished fonts/padding
    from PySide6.QtCore import QTimer

    from .renamer.rules_panel import RulesPanel

    def _sync_rename_apply() -> None:
        for panel in window.findChildren(RulesPanel):
            panel.match_apply_to_clear()

    _sync_rename_apply()
    window.show()
    QTimer.singleShot(0, _sync_rename_apply)

    from .widgets.titlebar import apply_window_corner_preference

    # Region clip needs a mapped HWND + final size (same as CTk after show)
    apply_window_corner_preference(window, theme.WINDOW_CORNER_RADIUS)

    # Background update check
    try:
        from update_checker import run_check_in_thread

        run_check_in_thread(theme.APP_VERSION, window)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(run())
