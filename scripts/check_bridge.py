"""End-to-end check of the live GUI bridge, without the GUI.

The bridge runs inside Glaxnimate, so the pytest suite (which lives in the venv and
has no PyQt6) can only test the *client* half against a stand-in server. This goes
further: it loads the real installed plugin, starts its real QTcpServer against a
real Glaxnimate document, drives it with the real client, and confirms an edit sent
over the socket actually mutates the document.

What it does NOT cover: Glaxnimate's menu wiring and canvas repaint. Those are the
app's job; click "Start AI Bridge" in a real window to exercise them.

Run with the SYSTEM interpreter (it needs both glaxnimate and system PyQt6, which
the venv deliberately does not have):

    MOD=~/src/glaxnimate/build/bin/plugin/python/build/lib
    PYTHONPATH="$MOD:src" QT_QPA_PLATFORM=offscreen python3 scripts/check_bridge.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading

PLUGIN = os.path.expanduser(
    "~/.local/share/stalefiles/glaxnimate/plugins/AiBridge/ai_bridge.py"
)


def main() -> int:
    import glaxnimate
    from PyQt6.QtCore import QCoreApplication, QTimer

    from glaxnimate_ai.engine.live import LiveBridge

    spec = importlib.util.spec_from_file_location("ai_bridge", PLUGIN)
    ai_bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ai_bridge)

    class FakeWindow:
        def status(self, m):
            print("  [gui]", m)

        def warning(self, m):
            print("  [gui warning]", m)

    app = QCoreApplication(sys.argv)
    out: dict = {}

    with glaxnimate.environment.Headless():
        doc = glaxnimate.model.Document("")
        comp = doc.assets.add_composition()
        comp.width, comp.height = 640, 480

        ai_bridge.start(FakeWindow(), doc, {})

        def drive():
            b = LiveBridge(timeout=5)
            out["ping"] = b.ping()
            out["edit"] = b.run("comp.width = 1280\nresult = comp.width")
            out["err"] = b.run("undefined_name")
            QTimer.singleShot(0, app.quit)

        threading.Thread(target=drive, daemon=True).start()
        QTimer.singleShot(5000, app.quit)  # safety net if the client hangs
        app.exec()

        checks = [
            ("ping returns pong", out.get("ping", {}).get("result") == "pong"),
            ("edit succeeded", out.get("edit", {}).get("ok") is True),
            ("document actually changed over the socket", comp.width == 1280),
            ("error path stays clean", out.get("err", {}).get("ok") is False),
        ]

    ok = all(passed for _, passed in checks)
    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    print("\nbridge OK" if ok else "\nbridge FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
