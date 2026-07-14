"""Live bridge: let an AI edit the document you are looking at, as you look at it.

Runs *inside* Glaxnimate. Click **Plugins > Start AI Bridge** and the app listens on
127.0.0.1:9123; the MCP server sends Python, this executes it against the open
document, and the canvas updates immediately.

Two things make this safe, and both are worth stating because the obvious
implementations are wrong:

**Threading.** Qt objects may only be touched from the thread that owns them. The
tempting design — a background socket thread that pokes the document — corrupts
state or crashes outright. Instead this uses `QTcpServer`, whose signals are
delivered by Qt's *own* event loop on the main thread. So every document edit
happens on the main thread by construction. No locks, no marshalling, no
`QTimer.singleShot` dance: the correctness falls out of the design rather than
being bolted on.

**Which Qt.** PyQt6 from apt links the same system Qt6 that Glaxnimate already
loaded, so there is exactly one Qt in the process. PySide6 from pip would drag a
*second* copy of Qt into the same address space, which is undefined behaviour and
fails in ways that are miserable to debug. Install `python3-pyqt6`, never
`pip install PySide6`.

Every edit is wrapped in `document.macro()`, so Ctrl+Z undoes what the AI did in
one step. If you are going to let a model edit your work, you want the undo stack
to have your back.
"""

import json
import traceback

import glaxnimate

try:
    from PyQt6.QtCore import QObject
    from PyQt6.QtNetwork import QHostAddress, QTcpServer
except ImportError as e:  # pragma: no cover - only ever hit inside the GUI
    raise ImportError(
        "The AI Bridge needs PyQt6 bound to the *system* Qt:\n"
        "    sudo apt install python3-pyqt6 python3-pyqt6.qtnetwork\n"
        "Do NOT `pip install PySide6` — it bundles a second Qt into this process."
    ) from e

HOST, PORT = "127.0.0.1", 9123

_server = None
_conns = []


class _Bridge(QObject):
    """Owns the socket. Lives on the main thread; so does every edit it makes."""

    def __init__(self, window, document):
        super().__init__()
        self.window = window
        self.document = document
        self.server = QTcpServer(self)
        self.server.newConnection.connect(self._accept)

    def listen(self) -> bool:
        return self.server.listen(QHostAddress(HOST), PORT)

    def _accept(self):
        sock = self.server.nextPendingConnection()
        _conns.append(sock)
        sock.readyRead.connect(lambda s=sock: self._read(s))
        sock.disconnected.connect(lambda s=sock: _conns.remove(s) if s in _conns else None)

    def _read(self, sock):
        # Newline-delimited JSON. A short read is normal; wait for the terminator
        # rather than assuming one packet == one message.
        while sock.canReadLine():
            line = bytes(sock.readLine()).decode("utf-8", "replace").strip()
            if not line:
                continue
            sock.write((json.dumps(self._handle(line)) + "\n").encode())
            sock.flush()

    def _handle(self, line: str) -> dict:
        try:
            msg = json.loads(line)
        except ValueError as e:
            return {"ok": False, "error": f"bad JSON: {e}"}

        if msg.get("op") == "ping":
            comp = self._composition()
            return {
                "ok": True,
                "result": "pong",
                "size": [comp.width, comp.height] if comp else None,
            }

        code = msg.get("code")
        if not code:
            return {"ok": False, "error": "expected {'code': ...} or {'op': 'ping'}"}

        ns = {
            "glaxnimate": glaxnimate,
            "model": glaxnimate.model,
            "utils": glaxnimate.utils,
            "document": self.document,
            "window": self.window,
            "comp": self._composition(),
        }
        try:
            # One macro == one Ctrl+Z. If a model is editing your document, undo
            # must put it back the way it was in a single step.
            with self.document.macro("AI edit"):
                exec(compile(code, "<ai>", "exec"), ns)  # noqa: S102 - the entire point
            return {"ok": True, "result": str(ns.get("result", "ok"))}
        except Exception:
            return {"ok": False, "error": traceback.format_exc(limit=4)}

    def _composition(self):
        comps = list(self.document.assets.compositions.values)
        return comps[0] if comps else None


def start(window, document, settings):
    global _server
    if _server is not None:
        window.warning(f"AI Bridge already listening on {HOST}:{PORT}")
        return

    bridge = _Bridge(window, document)
    if not bridge.listen():
        window.warning(f"AI Bridge could not bind {HOST}:{PORT} (already in use?)")
        return

    _server = bridge
    window.status(f"AI Bridge listening on {HOST}:{PORT} - the AI can now edit this document")


def stop(window, document, settings):
    global _server
    if _server is None:
        window.status("AI Bridge is not running")
        return
    for c in list(_conns):
        c.close()
    _conns.clear()
    _server.server.close()
    _server = None
    window.status("AI Bridge stopped")
