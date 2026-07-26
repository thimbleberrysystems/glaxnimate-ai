"""Rhythm & social format: land motion on the beat, and fit the platform.

auto_sfx puts sound on the motion; this is the inverse — the tempo grid, so cuts and
hits snap to the beat. Plus a loop check for seamless shorts/stickers.
"""

from __future__ import annotations

import pytest

from glaxnimate_ai.cartoon.rhythm import beat_frames, frames_per_beat, snap_to_beat
from glaxnimate_ai.engine.session import SessionStore


def test_beat_grid_math():
    # 120 bpm at 24 fps -> 12 frames per beat
    assert frames_per_beat(120, 24) == 12
    assert beat_frames(120, 24, 48) == [0, 12, 24, 36, 48]
    # eighth-note subdivision doubles the density
    assert beat_frames(120, 24, 24, division=2) == [0, 6, 12, 18, 24]


def test_snap_to_beat_rounds_onto_the_grid():
    assert snap_to_beat(14, 120, 24) == 12   # nearer to beat 1 (12) than beat 2 (24)
    assert snap_to_beat(19, 120, 24) == 24
    assert snap_to_beat(0, 120, 24) == 0


def test_beats_reads_the_scene_tempo():
    s = SessionStore().create(width=480, height=480, frames=24)
    s.run("music(seed=3, bpm=120)\nman = stick()\n"
          "add_character(man, make_gait(man, 'walk', cycle_frames=24), x=240, name='m')")
    assert s._beats() == [0, 12, 24]
    assert s._snap_to_beat(10) == 12


def test_beats_without_a_tempo_teaches():
    s = SessionStore().create(width=480, height=480, frames=24)
    with pytest.raises(ValueError, match="no tempo"):
        s._beats()


def test_loop_report_knows_a_cycle_from_a_mid_cycle():
    # frames == cycle -> the pose loops
    s = SessionStore().create(width=480, height=480, frames=24)
    s.run("man = stick()\n"
          "add_character(man, make_gait(man, 'walk', cycle_frames=24), x=240, name='m')")
    assert "cycle loops" in s._loop_report()

    # frames != cycle -> it lands mid-stride, so it does not loop
    s2 = SessionStore().create(width=480, height=480, frames=30)
    s2.run("man = stick()\n"
           "add_character(man, make_gait(man, 'walk', cycle_frames=24), x=240, name='m')")
    assert "does NOT loop" in s2._loop_report()


def test_format_presets_are_the_platform_shapes():
    from glaxnimate_ai.mcp.server import _FORMATS

    assert _FORMATS["portrait"] == (540, 960)   # 9:16 vertical
    assert _FORMATS["square"][0] == _FORMATS["square"][1]
    assert _FORMATS["sticker"] == (512, 512)
