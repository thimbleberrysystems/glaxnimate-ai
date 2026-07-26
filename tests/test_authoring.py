"""Authoring features from the dogfood pass: clear audio, floating props, particles,
and a consistent object API — the gaps that made diverse scenes fail or read badly.
"""

from __future__ import annotations

import numpy as np
import pytest

from glaxnimate_ai.audio.mix import duck
from glaxnimate_ai.engine.session import Session, SessionStore


# ------------------------------------------------------------------ audio clarity
def test_duck_drops_the_bed_under_a_span_with_ramps():
    bed = np.ones(44100 * 3, dtype=np.float32)
    out = duck(bed, [(1.0, 2.0)], level=0.3)
    rms = lambda a: float(np.sqrt(np.mean(a * a)))
    assert rms(out[int(0.2 * 44100):int(0.7 * 44100)]) > 0.9, "untouched outside the span"
    assert rms(out[int(1.35 * 44100):int(1.65 * 44100)]) < 0.4, "ducked under the span"
    # ramped, not a hard step: the sample right at the edge is between the two levels
    edge = out[int(1.0 * 44100)]
    assert 0.3 < edge < 1.0, "the dip should ramp, not click"


def test_music_is_ducked_under_dialogue_in_the_mix():
    s = SessionStore().create(width=320, height=320, frames=48)
    s.run("man = stick()\n"
          "add_action(man, actions.idle(man, ground_y=ground, x=160), name='m', face='stick')\n"
          "music(seed=3, bpm=120, gain=0.5)\n"
          "say('m', 'hello there everyone', 12)")
    # the first extra track is the (ducked) music bed
    start, bed, gain, pan = s._extra_audio()[0]
    fps, sr = s.scene.fps, 44100
    line = s.doc["audio"]["dialogue"][0]
    a = int(line["frame"] / fps * sr)
    b = int((line["frame"] + line["dur"]) / fps * sr)
    rms = lambda x: float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    before = rms(bed[max(0, a - 8000):max(0, a - 2000)])   # music just before the line
    during = rms(bed[a + 2000:b - 2000])                   # music under the line
    assert before > 0 and during < before * 0.5, \
        f"music should duck under dialogue (before {before:.3f}, during {during:.3f})"
    # deterministic sanity: mixing twice is identical (no RNG on the bus)
    r1, _ = s.audio_mix()
    r2, _ = s.audio_mix()
    assert np.array_equal(r1.buffer, r2.buffer)


# ------------------------------------------------------------------ floating props
def test_add_shape_places_a_floating_pulsing_prop():
    s = SessionStore().create(width=360, height=320, frames=24)
    heart = [{"type": "ellipse", "cx": 0, "cy": 0, "w": 24, "h": 22, "color": "#e74c3c"}]
    r = s.run(f"add_shape({heart!r}, x=180, y=150, pulse=(0.8, 1.2, 3))")
    assert r.ok, r.format()
    assert len(s.doc["shapes"]) == 1
    lay = next(sh for sh in s.scene.comp.shapes if sh.name.startswith("shape."))
    g = lay.shapes[0]
    assert g.transform.scale.keyframe_count > 2, "a pulse should key the scale"
    # the beat sweeps the full range across the timeline (min near 0.8, max near 1.2)
    sizes = [g.transform.scale.value_at_time(float(f)).x for f in range(0, 25)]
    assert max(sizes) - min(sizes) > 0.25, "the heart should visibly pulse"


def test_add_shape_reveal_is_hidden_until_its_frame():
    s = SessionStore().create(width=300, height=300, frames=20)
    s.run("add_shape([{'type':'ellipse','cx':0,'cy':0,'w':20,'h':20,'color':'#111'}],"
          " x=150, y=150, appear=8)")
    lay = next(sh for sh in s.scene.comp.shapes if sh.name.startswith("shape."))
    assert lay.opacity.value_at_time(0) == 0 and lay.opacity.value_at_time(8) > 0.9


# ----------------------------------------------------------------------- particles
def test_emit_burst_scatters_effects():
    s = SessionStore().create(width=400, height=360, frames=24)
    r = s.run("emit('spark', x=200, y=140, count=10, spread=80, start=4, seed=1)")
    assert r.ok, r.format()
    layers = [sh for sh in s.scene.comp.shapes if sh.name.startswith("emit.")]
    assert len(layers) == 10, "one layer per particle"


def test_emit_rain_falls_and_persists():
    st = SessionStore()
    s = st.create(width=480, height=360, frames=30)
    r = s.run("emit(None, x=240, y=60, count=16, spread=220, drop=280, over=18,"
              " color='#4a90d9', seed=2)")
    assert r.ok, r.format()
    assert len(s.doc["emits"]) == 1
    s.save()
    replayed = Session.replay(s.doc_id)
    assert len(replayed.doc["emits"]) == 1
    assert len([sh for sh in replayed.scene.comp.shapes if sh.name.startswith("emit.")]) == 16
    assert replayed.scene.comp.render_image(15).width == 480


# ------------------------------------------------------------------- object API
def test_add_object_accepts_name_like_add_character():
    s = SessionStore().create(width=400, height=300, frames=20)
    r = s.run("ball = motion.bounce(x0=60, x1=340, ground_y=ground, apex=90, frames=20)\n"
              "add_object(ball, shape='Ellipse', size=Vec2(30,30), color='#e74c3c', name='ball')")
    assert r.ok, r.format()
    assert any(n == "ball" for n, _, _ in s.objects)


def test_add_object_rejects_a_typo_with_a_teaching_error():
    s = SessionStore().create(width=400, height=300, frames=20)
    r = s.run("ball = motion.bounce(x0=60, x1=340, ground_y=ground, apex=90, frames=20)\n"
              "add_object(ball, colour='#e74c3c')")  # British spelling typo
    assert not r.ok
    assert "unexpected argument" in r.format() and "colour" in r.format()
