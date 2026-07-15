"""Drive the tools through the real MCP protocol, not direct function calls.

This test class exists because of a bug that unit tests could not see: v1's
SIGALRM timeout worked when tests called Session.run() from the main thread and
would have crashed the moment the server ran tools anywhere else. "Works in unit
tests, breaks over MCP" is a class of bug, and the antidote is exercising the
same wire path a model uses: client → protocol → FastMCP → worker thread → Qt.
"""

from __future__ import annotations

import anyio
import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _client():
    from mcp.shared.memory import create_connected_server_and_client_session

    from glaxnimate_ai.mcp.server import mcp as server

    return create_connected_server_and_client_session(server._mcp_server)


async def test_the_whole_loop_over_the_wire(tmp_path, monkeypatch):
    monkeypatch.setenv("GLAXNIMATE_AI_PROJECTS", str(tmp_path / "projects"))
    async with await _client() as client:
        r = await client.call_tool("new_document",
                                   {"width": 640, "height": 360, "frames": 24})
        assert not r.isError
        doc = r.content[0].text.split(":")[0]

        r = await client.call_tool("run_script", {"doc_id": doc, "code": (
            "scenery('sky')\nscenery('ground')\n"
            "man = human()\n"
            "ch = add_character(man, make_gait(man, 'walk', cycle_frames=24),"
            " x=80, name='man', face='human')\n"
            "set_expression(ch, 'happy', 12)\n"
        )})
        assert not r.isError, r.content[0].text

        r = await client.call_tool("lint_animation", {"doc_id": doc})
        assert "clean" in r.content[0].text

        r = await client.call_tool("diagnose_animation", {"doc_id": doc})
        assert "metrics" in r.content[0].text

        r = await client.call_tool("describe_scene", {"doc_id": doc})
        assert "man" in r.content[0].text and "sky" in r.content[0].text

        # a render must come back as actual image content
        r = await client.call_tool("render_frame", {"doc_id": doc, "frame": 12})
        kinds = [c.type for c in r.content]
        assert "image" in kinds, f"expected image content, got {kinds}"


async def test_script_errors_surface_as_text_not_protocol_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("GLAXNIMATE_AI_PROJECTS", str(tmp_path / "projects"))
    async with await _client() as client:
        r = await client.call_tool("new_document", {"width": 320, "height": 200})
        doc = r.content[0].text.split(":")[0]
        r = await client.call_tool("run_script",
                                   {"doc_id": doc, "code": "make_gait(human(), 'moonwalk')"})
        # the model needs the teaching error IN BAND, not a protocol failure
        assert "moonwalk" in r.content[0].text and "walk" in r.content[0].text


async def test_assets_flow_over_the_wire(tmp_path, monkeypatch):
    monkeypatch.setenv("GLAXNIMATE_AI_ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("GLAXNIMATE_AI_PROJECTS", str(tmp_path / "projects"))
    async with await _client() as client:
        bad = '{"version": 1, "kind": "prop", "shapes": []}'
        r = await client.call_tool("save_asset",
                                   {"kind": "prop", "name": "empty", "data": bad})
        assert "rejected" in r.content[0].text

        good = ('{"version": 1, "kind": "prop", "shapes": '
                '[{"type": "rect", "x": -10, "y": -20, "w": 20, "h": 20, '
                '"color": "#808080"}]}')
        r = await client.call_tool("save_asset",
                                   {"kind": "prop", "name": "crate", "data": good})
        assert "saved" in r.content[0].text

        r = await client.call_tool("list_assets", {})
        assert "crate" in r.content[0].text


async def test_the_event_loop_stays_free_during_a_slow_script(tmp_path, monkeypatch):
    """The v1 failure mode: any tool blocked the whole server. Now a slow script
    runs on the Qt worker while the loop keeps answering other requests."""
    monkeypatch.setenv("GLAXNIMATE_AI_PROJECTS", str(tmp_path / "projects"))
    async with await _client() as client:
        r = await client.call_tool("new_document", {"width": 320, "height": 200})
        doc = r.content[0].text.split(":")[0]

        slow = "import time\nfor _ in range(20):\n    time.sleep(0.05)\n"
        results = {}

        async def run_slow():
            results["slow"] = await client.call_tool(
                "run_script", {"doc_id": doc, "code": slow})

        async def poke_while_busy():
            await anyio.sleep(0.2)  # let the slow script get going
            with anyio.fail_after(2):  # would deadlock on a blocked loop
                results["poke"] = await client.call_tool("cartoon_api", {})

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_slow)
            tg.start_soon(poke_while_busy)

        assert not results["slow"].isError
        assert "GAITS" in results["poke"].content[0].text
