"""Audio: synthesis, mixing, and the numeric report the model reads instead of ears."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from glaxnimate_ai.audio.mix import Cue, mix_report, render_cues, resolve_patch, write_wav
from glaxnimate_ai.audio.synth import BUILTIN, render_patch, sfx_validate


# ---------------------------------------------------------------- synthesis
def test_rendering_is_deterministic():
    """Same patch, same samples — noise included (fixed-seed generator)."""
    a = render_patch(BUILTIN["thud"])
    b = render_patch(BUILTIN["thud"])
    assert np.array_equal(a, b)


def test_every_builtin_renders_audible_and_clickless():
    for name, patch in BUILTIN.items():
        buf = render_patch(patch)
        peak = float(np.max(np.abs(buf)))
        assert 0.05 < peak <= 1.0, f"{name} peaks at {peak}"
        assert abs(float(buf[0])) < 1e-6, f"{name} starts with a click"
        assert abs(float(buf[-1])) < 0.01, f"{name} ends with a click"


def test_patch_validation_teaches():
    with pytest.raises(ValueError, match="osc"):
        sfx_validate({"version": 1, "kind": "sfx", "layers": [{"osc": "theremin"}]})
    with pytest.raises(ValueError, match="f0"):
        sfx_validate({"version": 1, "kind": "sfx",
                      "layers": [{"osc": "sine", "dur": 0.2}]})
    with pytest.raises(ValueError, match="layers"):
        sfx_validate({"version": 1, "kind": "sfx", "layers": []})


def test_sfx_assets_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("GLAXNIMATE_AI_ASSETS", str(tmp_path))
    from glaxnimate_ai.cartoon.assets import load_asset, save_asset

    save_asset("sfx", "custom_beep",
               {"version": 1, "kind": "sfx",
                "layers": [{"osc": "sine", "f0": 660, "dur": 0.1}]})
    assert resolve_patch("custom_beep") == load_asset("sfx", "custom_beep")


def test_unknown_sfx_lists_the_vocabulary():
    with pytest.raises(ValueError, match="boing"):
        resolve_patch("kaboom")


# ------------------------------------------------------------------- mixing
def test_limiter_makes_clipping_impossible():
    """Forty simultaneous full-gain boings: the sum runs hot, the output never
    exceeds 1.0, and the report says so instead of pretending it was fine."""
    cues = [Cue(frame=0, sfx="boing", gain=2.0) for _ in range(40)]
    res = render_cues(cues, frames=24, fps=24)
    assert float(np.max(np.abs(res.buffer))) <= 1.0
    assert res.pre_peak > 1.5
    report = mix_report(cues, res, frames=24, fps=24)
    assert "WARNING" in report and "hot" in report


def test_pileups_are_flagged():
    cues = [Cue(frame=10, sfx="pop") for _ in range(6)]
    res = render_cues(cues, frames=48, fps=24)
    report = mix_report(cues, res, frames=48, fps=24)
    assert "overlap" in report


def test_a_clean_mix_reports_clean():
    cues = [Cue(frame=f, sfx="step", gain=0.8) for f in (4, 16, 28, 40)]
    res = render_cues(cues, frames=48, fps=24)
    report = mix_report(cues, res, frames=48, fps=24)
    assert "WARNING" not in report
    assert "4 cue(s)" in report and "step" in report


def test_panning_actually_pans():
    hard_left = render_cues([Cue(0, "ding", pan=-1.0)], frames=24, fps=24)
    l_energy = float(np.sum(hard_left.buffer[:, 0] ** 2))
    r_energy = float(np.sum(hard_left.buffer[:, 1] ** 2))
    assert l_energy > 100 * max(r_energy, 1e-9)


def test_wav_writes_and_reads_back(tmp_path):
    cues = [Cue(frame=0, sfx="ding")]
    res = render_cues(cues, frames=24, fps=24)
    p = write_wav(res, tmp_path / "out.wav")
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 2
        assert w.getframerate() == res.sr
        assert w.getnframes() == len(res.buffer)


def test_empty_cue_sheet_says_so():
    res = render_cues([], frames=24, fps=24)
    assert "empty" in mix_report([], res, frames=24, fps=24)


# ---------------------------------------------------- motion events -> cues
def _session(script, frames=48, width=760):
    from glaxnimate_ai.engine.session import SessionStore

    s = SessionStore().create(width=width, height=400, frames=frames)
    res = s.run(script)
    assert res.ok, res.format()
    return s


def test_a_walk_gets_footsteps_on_its_plants():
    """Two cycles, two feet: about four plant onsets, each panned to the foot."""
    s = _session(
        "man = human()\n"
        "add_character(man, make_gait(man, 'walk', cycle_frames=24), x=90, name='man')\n"
        "print(auto_sfx())\n"
    )
    cues = s.doc["audio"]["cues"]
    steps = [c for c in cues if c["sfx"] == "step"]
    assert 3 <= len(steps) <= 5, f"expected ~4 footfalls, got {len(steps)}"
    assert all(c["frame"] > 0 for c in steps), "frame 0 is a stance, not a stomp"


def test_a_bouncing_ball_gets_boings_on_its_hits():
    s = _session(
        "ball = motion.bounce(x0=100, x1=600, ground_y=ground, apex=170,"
        " frames=frames, bounces=4, radius=24)\n"
        "add_object(ball, size=Vec2(48, 48))\n"
        "auto_sfx()\n"
    )
    boings = [c for c in s.doc["audio"]["cues"] if c["sfx"] == "boing"]
    assert 3 <= len(boings) <= 5, f"expected ~4 ground hits, got {len(boings)}"


def test_a_jump_gets_a_whoosh_and_a_thud():
    s = _session(
        "man = human()\n"
        "add_action(man, actions.jump(man, ground_y=ground, x=300, height=150,"
        " frames=frames), name='jumper')\n"
        "auto_sfx()\n",
        frames=36,
    )
    kinds = [c["sfx"] for c in s.doc["audio"]["cues"]]
    assert kinds.count("whoosh") == 1, kinds
    assert kinds.count("thud") == 1, kinds


def test_a_run_does_not_whoosh_every_stride():
    """A run's brief flight phases are locomotion, not jumps — the min-airborne
    threshold must keep them silent."""
    s = _session(
        "man = human()\n"
        "add_character(man, make_gait(man, 'run', cycle_frames=16), x=60, name='man')\n"
        "auto_sfx()\n"
    )
    kinds = [c["sfx"] for c in s.doc["audio"]["cues"]]
    assert kinds.count("whoosh") == 0, f"a run whooshed: {kinds}"


def test_expression_swaps_sting():
    s = _session(
        "man = human()\n"
        "ch = add_character(man, make_gait(man, 'walk', cycle_frames=24),"
        " x=90, name='man', face='human')\n"
        "set_expression(ch, 'surprised', 20)\n"
        "auto_sfx({'plant': None})\n"   # silence the steps; isolate the sting
    )
    kinds = [c["sfx"] for c in s.doc["audio"]["cues"]]
    assert kinds == ["pop"], kinds


def test_manual_cues_and_the_mix_report():
    s = _session("add_sound('ding', 10)\nadd_sound('splat', 30, gain=0.7, pan=0.5)\n")
    result, report = s.audio_mix()
    assert "2 cue(s)" in report and "ding" in report
    assert s.has_audio


def test_bad_sfx_name_fails_at_placement_not_export():
    from glaxnimate_ai.engine.session import SessionStore

    s = SessionStore().create(width=300, height=200, frames=8)
    res = s.run("add_sound('kaboom', 0)")
    assert not res.ok and "boing" in res.format()


def test_cues_persist_and_replay():
    from glaxnimate_ai.engine.session import Session

    s = _session(
        "man = human()\n"
        "add_character(man, make_gait(man, 'walk', cycle_frames=24), x=90, name='man')\n"
        "auto_sfx()\nadd_sound('ding', 40)\n"
    )
    reborn = Session.replay(s.doc_id)
    assert reborn.doc["audio"]["cues"] == s.doc["audio"]["cues"]
    _, report = reborn.audio_mix()
    assert "cue(s)" in report


# ---------------------------------------------------------------- mux to MP4
def test_exported_mp4_carries_the_soundtrack(tmp_path):
    """The full chain: scene -> silent MP4 -> mixed cues muxed in, probed back."""
    from glaxnimate import io as gio

    from glaxnimate_ai.audio.mux import audio_duration, has_audio_stream, mux_audio

    s = _session(
        "man = human()\n"
        "add_character(man, make_gait(man, 'walk', cycle_frames=24), x=80, name='man')\n"
        "auto_sfx()\n"
    )
    fmt = gio.registry.from_extension("mp4", gio.Direction.Export)
    silent = tmp_path / "silent.mp4"
    silent.write_bytes(fmt.save(s.scene.comp))
    assert not has_audio_stream(silent), "glaxnimate export should be video-only"

    mix, _ = s.audio_mix()
    final = mux_audio(silent, mix, tmp_path / "with_sound.mp4")
    assert has_audio_stream(final)
    scene_seconds = s.frames / s.scene.fps
    assert abs(audio_duration(final) - scene_seconds) < 0.5


# ------------------------------------------------------------------- music
def test_music_is_deterministic_per_seed():
    from glaxnimate_ai.audio.music import render_music

    a = render_music({"seed": 7}, duration_s=2.0)
    b = render_music({"seed": 7}, duration_s=2.0)
    c = render_music({"seed": 8}, duration_s=2.0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(a) == int(2.0 * 44100)


def test_music_spec_validation_teaches():
    from glaxnimate_ai.audio.music import music_validate

    with pytest.raises(ValueError, match="bpm"):
        music_validate({"bpm": 500})
    with pytest.raises(ValueError, match="gain"):
        music_validate({"gain": 0})


def test_music_joins_the_mix_and_persists():
    from glaxnimate_ai.engine.session import Session

    s = _session("add_sound('ding', 4)\nmusic(seed=3, bpm=110, gain=0.2)\n")
    result, report = s.audio_mix()
    # the bed fills the scene: well over half the samples are non-silent
    active = float(np.mean(np.abs(result.buffer) > 1e-4))
    assert active > 0.5, f"music bed covers only {active:.0%} of the timeline"

    reborn = Session.replay(s.doc_id)
    assert reborn.doc["audio"]["music"]["seed"] == 3
    r2, _ = reborn.audio_mix()
    assert np.array_equal(result.buffer, r2.buffer), "music must replay identically"
