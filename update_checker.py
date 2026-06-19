# update_checker.py
import threading
import webbrowser

import requests
import tkinter as tk
from packaging.version import parse as parse_version

GITHUB_REPO_OWNER = 'gilliaangilliaan'
GITHUB_REPO_NAME = 'STEM-organizer'
GITHUB_API_URL = (
    f'https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/latest'
)
RELEASES_PAGE_URL = f'https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases'


def get_latest_release_info():
    """Fetch the latest release information from the GitHub API."""
    try:
        headers = {'Accept': 'application/vnd.github.v3+json'}
        response = requests.get(GITHUB_API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        print(f'[Update Check] Network or API error: {exc}')
        return None
    except Exception as exc:
        print(f'[Update Check] Unexpected error fetching release info: {exc}')
        return None


def compare_versions(current_version_str, latest_version_str):
    """Return True when the latest release tag is newer than the current version."""
    try:
        current_version_str = current_version_str.lstrip('v').lstrip('.')
        latest_version_str = latest_version_str.lstrip('v').lstrip('.')

        current_version = parse_version(current_version_str)
        latest_version = parse_version(latest_version_str)
        return latest_version > current_version
    except Exception as exc:
        print(
            f"[Update Check] Error comparing versions "
            f"('{current_version_str}' vs '{latest_version_str}'): {exc}"
        )
        return False


def show_update_dialog(new_version_tag, parent_window):
    """Show a simple modal dialog when a newer release is available."""
    if parent_window is None or not isinstance(parent_window, (tk.Tk, tk.Toplevel)):
        print('ERROR: Cannot show update dialog, invalid parent window.')
        return
    try:
        if not parent_window.winfo_exists():
            return
    except tk.TclError:
        return

    dlg = tk.Toplevel(parent_window)
    dlg.title('Update Available')
    dlg.resizable(False, False)
    dlg.attributes('-topmost', True)
    dlg.transient(parent_window)

    message = (
        f'A new version ({new_version_tag}) is available.\n'
        f'Visit the Releases page on GitHub to download.'
    )
    tk.Label(dlg, text=message, justify='left', wraplength=380).pack(padx=20, pady=(20, 12))

    btn_row = tk.Frame(dlg)
    btn_row.pack(pady=(0, 20))

    def open_release_page():
        try:
            webbrowser.open(RELEASES_PAGE_URL, new=2)
        except Exception as exc:
            print(f'[Update Check] Failed to open browser: {exc}')
        dlg.destroy()

    tk.Button(btn_row, text='Download Update', width=16, command=open_release_page).pack(
        side='left', padx=6
    )
    tk.Button(btn_row, text='Later', width=10, command=dlg.destroy).pack(side='left', padx=6)

    dlg.update_idletasks()
    pw, ph = parent_window.winfo_width(), parent_window.winfo_height()
    px, py = parent_window.winfo_x(), parent_window.winfo_y()
    dw, dh = dlg.winfo_width(), dlg.winfo_height()
    dlg.geometry(f'+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}')

    dlg.grab_set()
    dlg.focus_force()
    dlg.bind('<Return>', lambda _e: open_release_page())
    dlg.bind('<Escape>', lambda _e: dlg.destroy())
    dlg.protocol('WM_DELETE_WINDOW', dlg.destroy)
    parent_window.wait_window(dlg)


def run_check_in_thread(current_version, root_window):
    """Check GitHub releases in a background thread; show dialog on the UI thread."""
    def threaded_task():
        release_info = get_latest_release_info()
        if not release_info or 'tag_name' not in release_info:
            print('[Update Check] Could not retrieve valid release information from thread.')
            return

        latest_tag = release_info['tag_name']
        print(f'[Update Check] Latest: {latest_tag}, Current: {current_version}')

        if compare_versions(current_version, latest_tag):
            print(f'[Update Check] New version found: {latest_tag}. Scheduling dialog.')
            root_window.after(
                0,
                lambda: show_update_dialog(latest_tag, parent_window=root_window),
            )
        else:
            print('[Update Check] Application is up-to-date.')

    threading.Thread(target=threaded_task, daemon=True).start()
