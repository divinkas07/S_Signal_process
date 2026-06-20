"""
🛰️ LEO Ku-Band Dynamic Reality Simulator (RES_env)
Entry point for the simulation application.
"""

import sys
import os

# ── DLL Path Setup (Windows) ──────────────────────────────────────────
if sys.platform == "win32":
    try:
        import PySide6
        possible_paths = []
        if hasattr(PySide6, "__path__"):
            possible_paths.extend(PySide6.__path__)
        if hasattr(PySide6, "__file__") and PySide6.__file__:
            possible_paths.append(os.path.dirname(PySide6.__file__))

        for path in possible_paths:
            qt_bin = os.path.join(path, "Qt", "bin")
            if os.path.exists(qt_bin):
                os.add_dll_directory(qt_bin)
                break
            if os.path.isdir(path) and any(
                f.endswith(".dll") for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ):
                os.add_dll_directory(path)
                break
    except Exception:
        pass

# ── Launch Application ────────────────────────────────────────────────
from gui.main_window import main

if __name__ == "__main__":
    main()
