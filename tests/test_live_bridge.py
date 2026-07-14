"""The live bridge protocol, tested against a stand-in server.

The real bridge runs inside Glaxnimate and needs PyQt6 and a human clicking
"Start AI Bridge", so it cannot be tested here. What *can* be tested is our half:
that the client frames messages correctly, waits for the terminator, parses
replies, and fails with a message that tells the user what to do.

Standing up a fake server is not a cop-out — the framing is exactly where a
line-delimited protocol goes wrong, and a short read looks like success right up
until it doesn't.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

from glaxnimate_ai.engine.live import BridgeUnavailable, LiveBridge


def _fake_bridge(handler, *, chunked: bool = False) -> tuple[int, threading.Thread]:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        with conn:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf += chunk
            reply = (json.dumps(handler(json.loads(buf))) + "\n").encode()
            if chunked:
                # Dribble the reply out one byte at a time. A client that assumes
                # one recv() == one message passes against a fast local server and
                # then fails in the wild; this makes that bug impossible to miss.
                for b in reply:
                    conn.send(bytes([b]))
            else:
                conn.sendall(reply)
        srv.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return port, t


def test_ping_round_trips():
    port, _ = _fake_bridge(lambda msg: {"ok": True, "result": "pong", "size": [960, 540]})
    r = LiveBridge(port=port, timeout=5).ping()
    assert r["result"] == "pong"
    assert r["size"] == [960, 540]


def test_code_is_sent_and_result_returned():
    seen = {}

    def handler(msg):
        seen.update(msg)
        return {"ok": True, "result": "done"}

    port, _ = _fake_bridge(handler)
    r = LiveBridge(port=port, timeout=5).run("comp.width = 1280")
    assert seen["code"] == "comp.width = 1280"
    assert r["ok"] and r["result"] == "done"


def test_reply_split_across_packets_is_reassembled():
    """The failure mode a fast localhost server hides."""
    port, _ = _fake_bridge(
        lambda m: {"ok": True, "result": "x" * 500}, chunked=True
    )
    r = LiveBridge(port=port, timeout=5).run("pass")
    assert r["result"] == "x" * 500


def test_errors_from_the_gui_come_back_intact():
    port, _ = _fake_bridge(
        lambda m: {"ok": False, "error": "NameError: name 'wat' is not defined"}
    )
    r = LiveBridge(port=port, timeout=5).run("wat")
    assert not r["ok"]
    assert "NameError" in r["error"]


def test_no_gui_gives_an_actionable_message():
    """Not just 'connection refused' — tell the user what to click."""
    with pytest.raises(BridgeUnavailable) as e:
        LiveBridge(port=1, timeout=1).ping()
    msg = str(e.value)
    assert "Start AI Bridge" in msg, msg
