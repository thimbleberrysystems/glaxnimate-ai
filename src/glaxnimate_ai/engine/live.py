"""Client for the live GUI bridge.

The plugin (`plugin/AiBridge/`) runs inside Glaxnimate and listens on
127.0.0.1:9123. This talks to it, so the model can edit the document you are
actually looking at and watch the canvas update.

The protocol is newline-delimited JSON, deliberately: it is trivial to debug with
`nc`, and there is no framing to get wrong.

Headless remains the default and is the only thing the tests use. The bridge is
for the moment you want to *watch*, or to hand a half-finished scene to the AI and
say "fix the arms".
"""

from __future__ import annotations

import json
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LiveBridge", "BridgeUnavailable", "open_in_glaxnimate"]

HOST, PORT = "127.0.0.1", 9123


class BridgeUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class LiveBridge:
    host: str = HOST
    port: int = PORT
    timeout: float = 10.0

    def _rpc(self, msg: dict) -> dict:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
                s.sendall((json.dumps(msg) + "\n").encode())
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(8192)
                    if not chunk:
                        break
                    buf += chunk
        except OSError as e:
            raise BridgeUnavailable(
                f"nothing listening on {self.host}:{self.port} - open Glaxnimate and "
                f"click Plugins > Start AI Bridge ({e})"
            ) from e

        if not buf.strip():
            raise BridgeUnavailable("bridge closed the connection without replying")
        return json.loads(buf.decode())

    def ping(self) -> dict:
        return self._rpc({"op": "ping"})

    def run(self, code: str) -> dict:
        """Execute Python inside the running Glaxnimate against the open document.

        In scope there: `document`, `comp`, `window`, `model`, `utils`, `glaxnimate`.
        Every call is one undo step.
        """
        return self._rpc({"code": code})


def open_in_glaxnimate(path: str | Path) -> str:
    """Launch the GUI on a file. The simple, always-works path — no plugin needed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    subprocess.Popen(
        ["glaxnimate", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"opened {path} in Glaxnimate"
