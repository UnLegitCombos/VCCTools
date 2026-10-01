import io
import sys


def setup_console():
    """Make stdout/stderr UTF-8 safe (Windows consoles and redirected output)."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
