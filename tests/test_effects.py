"""Effects: visual juice as data, placed from the same motion events as foley.

The claims: an effect is a validated document (shapes + a grow/fade envelope); it
bakes to a layer that is invisible except across its lifespan; auto_fx derives dust
/ impact / speed lines from the SAME Timeline events that auto_sfx reads; and every
placed effect survives a save + replay.
"""

from __future__ import annotations

import pytest

from glaxnimate import environment

from glaxnimate_ai.cartoon import assets as A
from glaxnimate_ai.cartoon.effects import fx_names, resolve_fx
from glaxnimate_ai.engine.bake import Scene, bake_effect
from glaxnimate_ai.engine.session import Session, SessionStore


def test_builtin_effects_validate():
    assert set(fx_names()) == {"dust", "impact", "speed_lines", "spark"}
    for name in fx_names():
        d = resolve_fx(name)
        assert d["kind"] == "fx" and d["shapes"] and d["lifespan"] > 0


def test_unknown_effect_teaches():
    with pytest.raises(ValueError, match="unknown effect"):
        resolve_fx("kapow")


def test_fx_validate_rejects_malformed():
    with pytest.raises(ValueError, match="non-empty 'shapes'"):
        A.fx_validate({"version": 1, "kind": "fx", "shapes": []})
    with pytest.raises(ValueError, match="lifespan"):
        A.fx_validate({"version": 1, "kind": "fx",
                       "shapes": [{"type": "ellipse", "cx": 0, "cy": 0, "w": 4, "h": 4}],
                       "lifespan": 0})


def test_effect_is_invisible_outside_its_lifespan():
    with environment.Headless():
        sc = Scene.create(width=200, height=200, frames=20)
        lay = bake_effect(sc, resolve_fx("impact"), x=100, y=100, start=4)
        op = lay.opacity
        assert op.value_at_time(0) == 0, "invisible before it starts"
        assert op.value_at_time(4) > 0.9, "full on the frame it pops in"
        # lifespan 5 -> gone by frame 9
        assert op.value_at_time(9) < 0.05, "faded to nothing by the end"
        # and the grow envelope actually keys scale (only possible post binding fix)
        assert lay.shapes[0].transform.scale.keyframe_count == 2


def test_auto_fx_derives_effects_from_the_same_events_as_foley():
    s = SessionStore().create(width=400, height=320, frames=24)
    r = s.run(
        "man = stick()\n"
        "add_character(man, make_gait(man, 'walk'), x=200, name='m')\n"
        "auto_fx()"
    )
    assert r.ok, r.format()
    autos = [e for e in s.doc["effects"] if e.get("auto")]
    assert autos, "a walk should spawn foot-plant dust"
    assert all(e["fx"] == "dust" for e in autos), "foot plants map to dust"


def test_manual_effect_and_persistence_round_trip():
    st = SessionStore()
    s = st.create(width=360, height=300, frames=20)
    r = s.run(
        "man = stick()\n"
        "add_action(man, actions.punch(man, ground_y=ground, x=180, frames=16), name='f')\n"
        "add_effect('impact', x=250, y=150, frame=10)"
    )
    assert r.ok, r.format()
    assert len(s.doc["effects"]) == 1
    s.save()

    replayed = Session.replay(s.doc_id)
    assert len(replayed.doc["effects"]) == 1
    # the replayed scene renders (effect layer rebuilt from the record)
    img = replayed.scene.comp.render_image(10)
    assert img.width == 360


def test_auto_fx_clear_hides_prior_effects_on_rerun():
    s = SessionStore().create(width=400, height=320, frames=24)
    s.run("man = stick()\nadd_character(man, make_gait(man, 'walk'), x=200, name='m')")
    s.run("auto_fx()")
    first = len([e for e in s.doc["effects"] if e.get("auto")])
    s.run("auto_fx()")  # re-run with default clear=True
    second = len([e for e in s.doc["effects"] if e.get("auto")])
    assert first > 0 and second == first, "re-running auto_fx must not accumulate"
