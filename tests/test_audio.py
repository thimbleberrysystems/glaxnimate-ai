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
