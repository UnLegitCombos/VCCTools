"""Open the VCC Tools app in its own window and stop the server when the window closes.

Starts Streamlit without a browser tab and shows it in a native window (pywebview, which
uses the WebView2 runtime that ships with Windows 11). Falls back to the normal browser if
the window cannot be created. Run with start_app.bat.
"""

import ctypes
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(tempfile.gettempdir(), "vcc_tools_app.log")
TITLE = "VCC Tools"
ICON = os.path.join(ROOT, "teamMaker", "assets", "vcc-logo.ico")
APP_ID = "VCCTools.App"  # Windows taskbar identity
START_TIMEOUT_S = 90


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_up(server, port):
    """Wait for Streamlit's health check; return False if it exits or never answers."""
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.time() + START_TIMEOUT_S
    while time.time() < deadline and server.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def stop(server):
    if server.poll() is None:
        # /T also ends Streamlit's child processes.
        subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], check=False, capture_output=True)


def show_error(message):
    print(message)
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(0, message, TITLE, 0x10)


def show_info(message):
    print(message)
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(0, message, TITLE, 0x40)
    else:
        input("Press Enter to stop the app.")


def set_taskbar_icon(title):
    """Put the VCC icon on the window at the sizes the title bar and taskbar use."""
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.LoadImageW.restype = ctypes.c_void_p
    user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return
    for which, metric in ((1, 11), (0, 49)):  # ICON_BIG at SM_CXICON, ICON_SMALL at SM_CXSMICON
        size = user32.GetSystemMetrics(metric)
        icon = user32.LoadImageW(None, ICON, 1, size, size, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
        if icon:
            user32.SendMessageW(hwnd, 0x80, which, icon)  # WM_SETICON


def open_window(url):
    """Show the app in a native window until it is closed; False if that is not possible."""
    try:
        import webview

        window = webview.create_window(TITLE, url, width=1500, height=950, min_size=(1000, 650))
        if os.name == "nt" and os.path.isfile(ICON):
            # Its own taskbar button (not grouped under Python), showing the window's icon.
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
            window.events.shown += lambda: set_taskbar_icon(TITLE)
        webview.start(icon=ICON)
    except Exception as exc:  # noqa: BLE001 - any failure here falls back to the browser
        print(f"Could not open the app window ({exc}).")
        return False
    return True


def main():
    port = free_port()
    with open(LOG, "w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", str(port),
                "--browser.gatherUsageStats", "false",
            ],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
    try:
        if not wait_until_up(server, port):
            show_error(f"The app did not start. Details: {LOG}")
            return 1
        url = f"http://127.0.0.1:{port}"
        if not open_window(url):
            webbrowser.open(url)
            show_info(
                f"The app window is not available, so it opened in your browser at {url}.\n\n"
                "Click OK to stop the app."
            )
        return 0
    finally:
        stop(server)


if __name__ == "__main__":
    sys.exit(main())
