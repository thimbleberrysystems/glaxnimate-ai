"""Two-figure interaction and props-as-toys: fights, hand-offs, using the world.

The claims: a wielded prop rides a hand bone (so a sword follows a swing); a thrown
prop flies a ballistic arc and is invisible until it leaves the hand; a clash lands
the impact bundle (flash + sfx + camera) on one contact frame; and all of it
survives save + replay.
"""

from __future__ import annotations

import pytest

from glaxnimate_ai.engine.session import Session, SessionStore

SWORD = {"shapes": [{"type": "rect", "x": 0, "y": -3, "w": 70, "h": 6, "color": "#9aa7b5"}]}
ROCK = {"shapes": [{"type": "ellipse", "cx": 0, "cy": 0, "w": 22, "h": 20, "color": "#6b6b6b"}]}


def test_wielded_prop_rides_the_hand_bone():
    s = SessionStore().create(width=340, height=320, frames=18)
    r = s.run(
        "man = stick()\n"
        "add_action(man, actions.swing(man, ground_y=ground, x=170, frames=18), name='k')\n"
        f"wield('k', {SWORD!r}, bone='arm_lower')"
    )
    assert r.ok, r.format()
    assert len(s.doc["wields"]) == 1
    ch = s.characters[0]
    # the sword group was added into the forearm bone's layer (so it moves with it)
    hand = ch.bone_layers["arm_lower"]
    assert any(sh.type_name == "Group" for sh in hand.shapes), "sword not on the hand"


def test_wield_rejects_a_missing_bone_with_a_teaching_error():
    s = SessionStore().create(width=300, height=300, frames=8)
    s.run("man = stick()\nadd_action(man, actions.idle(man, ground_y=ground), name='m')")
    r = s.run(f"wield('m', {SWORD!r}, bone='tentacle')")
    assert not r.ok
    assert "no bone" in r.format() and "arm_lower" in r.format()


def test_thrown_prop_is_hidden_until_release_and_flies_an_arc():
    s = SessionStore().create(width=420, height=300, frames=20)
    r = s.run(f"throw({ROCK!r}, x0=60, y0=180, x1=360, y1=200, apex=130, release=3, spin=540)")
    assert r.ok, r.format()
    assert len(s.doc["throws"]) == 1
    # find the thrown layer and check it is invisible before release, visible after
    lay = next(sh for sh in s.scene.comp.shapes if sh.name.startswith("thrown."))
    assert lay.opacity.value_at_time(0) == 0, "invisible before release"
    assert lay.opacity.value_at_time(3) > 0.9, "visible once released"
    # arc: the group is higher on screen (smaller y) at mid-flight than at the ends
    g = lay.shapes[0]
    y_start = g.transform.position.value_at_time(3).y
    y_mid = g.transform.position.value_at_time(11).y
    assert y_mid < y_start - 40, "the throw should arc up over the middle"


def test_clash_lands_the_full_bundle_on_one_frame():
    s = SessionStore().create(width=420, height=320, frames=24)
    r = s.run(
        "a = stick(); b = stick()\n"
        "add_action(a, actions.hitstop(actions.punch(a, ground_y=ground, x=150, frames=16), at=10), name='a')\n"
        "add_action(b, actions.sequence((actions.idle(b, ground_y=ground, x=250), 10),"
        " (actions.knockback(b, ground_y=ground, x=250, facing=-1), 14)), name='b')\n"
        "clash(10, x=250, y=120)"
    )
    assert r.ok, r.format()
    # one impact effect on the contact frame, one hit cue there, one camera op
    fx = [e for e in s.doc["effects"] if e["frame"] == 10]
    cues = [c for c in s.doc["audio"]["cues"] if c["frame"] == 10]
    assert fx and cues and s.doc.get("camera"), "clash should fire flash + sfx + camera"


def test_interaction_survives_save_and_replay():
    st = SessionStore()
    s = st.create(width=360, height=320, frames=18)
    s.run(
        "man = stick()\n"
        "add_action(man, actions.swing(man, ground_y=ground, x=170, frames=18), name='k')\n"
        f"wield('k', {SWORD!r}, bone='arm_lower')\n"
        f"throw({ROCK!r}, x0=60, y0=180, x1=300, y1=200, apex=100, release=4)"
    )
    s.save()
    replayed = Session.replay(s.doc_id)
    assert len(replayed.doc["wields"]) == 1
    assert len(replayed.doc["throws"]) == 1
    assert replayed.scene.comp.render_image(9).width == 360
