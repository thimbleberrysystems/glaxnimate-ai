"""The v2 primitive round: a background that beats the black export default,
appear/vanish shape timing, and the acting ergonomics (freeze, posture,
sequence auto-hold, motion.drop) that a scene used to hand-roll every time.
"""

from __future__ import annotations

import numpy as np

from glaxnimate_ai.cartoon import actions, motion
from glaxnimate_ai.cartoon.assets import load_face, prop_validate
from glaxnimate_ai.cartoon.presets import stick
from glaxnimate_ai.engine.session import Session, SessionStore
from glaxnimate_ai.feedback.render import render_frame

GROUND = 300.0


# ---------------------------------------------------------------- background
def test_background_fills_records_and_replays():
    st = SessionStore()
    s = st.create(width=320, height=200, frames=6, ground_y=GROUND)
    r = s.run("background('#ffffff')")
    assert r.ok, r.format()
    assert s.doc["background"] == {"color": "#ffffff"}
    assert any(sh.name == "background" for sh in s.scene.comp.shapes)
    s.save()
    assert Session.replay(s.doc_id).doc["background"] == {"color": "#ffffff"}


def test_background_defaults_light():
    s = SessionStore().create(width=200, height=120, frames=2)
    assert s.run("background()").ok
    assert s.doc["background"]["color"] == "#f6f6f7"


# ------------------------------------------------------------ shape timing
def _named(scene, name):
    return next(sh for sh in scene.comp.shapes if sh.name == name)


def test_add_shape_vanish_toggles_opacity():
    s = SessionStore().create(width=200, height=200, frames=30)
    s.run("add_shape([{'type':'ellipse','cx':0,'cy':0,'w':20,'h':20,'color':'#111'}],"
          " 100, 100, name='dot', appear=6, vanish=18)")
    op = _named(s.scene, "dot").opacity
    assert op.value_at_time(0.0) < 0.5      # hidden before appear
    assert op.value_at_time(10.0) > 0.5     # shown after appear
    assert op.value_at_time(24.0) < 0.5     # gone after vanish
    assert s.doc["shapes"][0]["vanish"] == 18


def test_shape_vanish_survives_replay():
    s = SessionStore().create(width=200, height=200, frames=30)
    s.run("add_shape([{'type':'rect','x':-5,'y':-5,'w':10,'h':10,'color':'#111'}],"
          " 50, 50, name='box', vanish=12)")
    s.save()
    back = Session.replay(s.doc_id)
    assert _named(back.scene, "box").opacity.value_at_time(20.0) < 0.5


# ---------------------------------------------------------------- freeze
def test_freeze_holds_one_pose_forever():
    body = stick()
    src = actions.celebrate(body, ground_y=GROUND, x=100, frames=20)
    held = actions.freeze(src, at=8)
    a, b = held(0.0), held(500.0)
    assert a.root.x == b.root.x and a.root.y == b.root.y
    assert a.angles == b.angles == src(8.0).angles


# ---------------------------------------------------------------- posture
def test_posture_leans_without_moving_the_feet():
    body = stick()
    base = actions.idle(body, ground_y=GROUND, x=120)
    leaned = actions.posture(base, lean=18.0, droop=8.0)
    fb, fl = base(4.0), leaned(4.0)
    assert fl.angles["spine"] == fb.angles.get("spine", 0.0) + 18.0
    # feet are IK'd in the base pose; leaning the spine must not move them
    for foot in ("shin_l", "shin_r"):
        p0 = body.rig.solve(fb)[foot].tip
        p1 = body.rig.solve(fl)[foot].tip
        assert abs(p0.x - p1.x) < 1e-6 and abs(p0.y - p1.y) < 1e-6


def test_posture_stiffen_kills_the_sway():
    body = stick()
    base = actions.idle(body, ground_y=GROUND, x=120)
    stiff = actions.posture(base, stiffen=True)
    assert stiff(3.0).angles["spine"] == 0.0
    assert stiff(9.0).angles["spine"] == 0.0


# ------------------------------------------------------- sequence auto-hold
def test_sequence_holds_the_final_pose_past_its_end():
    body = stick()
    combo = actions.sequence(
        (actions.idle(body, ground_y=GROUND, x=100), 10),
        (actions.celebrate(body, ground_y=GROUND, x=100, frames=12), 12),
    )
    total = 22
    end = combo(float(total))
    # anything past the end settles on the same held pose, not a wrap or drift
    assert combo(float(total + 40)).angles == end.angles
    assert combo(float(total + 200)).root.x == end.root.x


# ---------------------------------------------------------------- drop
def test_drop_parks_then_lands_then_settles():
    s = motion.drop(x=50.0, y_top=-120.0, y_land=140.0, frames=10, delay=5, settle=4)
    assert len(s) == 5 + 11 + 4
    assert all(v.pos.y == -120.0 for v in s[:5])         # parked (off-screen) first
    assert s[0].frame == 0 and s[-1].frame == 5 + 10 + 4
    land = next(v for v in s if v.frame == 15)           # delay+frames
    assert abs(land.pos.y - 140.0) < 1e-6
    # accelerating: later steps of the fall are bigger than earlier ones
    fall = [v.pos.y for v in s[5:16]]
    assert (fall[-1] - fall[-2]) > (fall[1] - fall[0])


# ---------------------------------------------------------------- text
def test_prop_validator_accepts_text():
    prop_validate({"version": 1, "kind": "prop",
                   "shapes": [{"type": "text", "x": 0, "y": 0, "text": "hi"}]})


def test_text_shape_renders_real_glyphs():
    # The Glaxnimate fork now downcasts TextShape, so text/font/position write and
    # the glyphs actually rasterise instead of the shape being a no-op ShapeElement.
    s = SessionStore().create(width=200, height=80, frames=2, ground_y=70)
    r = s.run('background("#ffffff")\n'
              'add_shape([{"type":"text","x":0,"y":0,"text":"OK","size":40,'
              '"color":"#000000","anchor":"middle"}], 100, 45, name="t")')
    assert r.ok, r.format()
    a = np.asarray(render_frame(s.scene, 0).convert("L"))
    assert (a < 80).sum() > 150, "text should put real ink on the canvas"


def test_text_survives_replay():
    s = SessionStore().create(width=200, height=80, frames=2, ground_y=70)
    s.run('add_shape([{"type":"text","x":0,"y":0,"text":"HELLO","size":20,'
          '"color":"#111"}], 40, 40, name="t")')
    s.save()
    back = Session.replay(s.doc_id)
    assert back.doc["shapes"][0]["shapes"][0]["type"] == "text"
    a = np.asarray(render_frame(back.scene, 0).convert("L"))
    assert (a < 80).sum() > 80


# ---------------------------------------------------------------- facing
def _char_layer(scene, name):
    return next(sh for sh in scene.comp.shapes if sh.name == name)


def test_facing_left_mirrors_the_puppet():
    s = SessionStore().create(width=300, height=340, frames=4, ground_y=300)
    s.run('a=stick(); b=stick()\n'
          'add_action(a, actions.idle(a, ground_y=ground, x=110), name="a", face="stick")\n'
          'add_action(b, actions.idle(b, ground_y=ground, x=210), name="b", face="stick", facing=-1)')
    assert _char_layer(s.scene, "a").transform.scale.value.x == 1.0
    assert _char_layer(s.scene, "b").transform.scale.value.x == -1.0
    # flipping is in place: the mirrored figure keeps its root x, it doesn't slide off
    assert _char_layer(s.scene, "b").transform.position.value_at_time(0.0).x > 150


def test_scale_makes_a_giant_and_composes_with_facing():
    s = SessionStore().create(width=300, height=340, frames=4, ground_y=300)
    s.run('a=stick(); b=stick()\n'
          'add_action(a, actions.idle(a, ground_y=ground, x=90), name="a", face="stick", scale=1.6)\n'
          'add_action(b, actions.idle(b, ground_y=ground, x=210), name="b", face="stick", facing=-1, scale=1.6)')
    sa = _char_layer(s.scene, "a").transform.scale.value
    sb = _char_layer(s.scene, "b").transform.scale.value
    assert (round(sa.x, 2), round(sa.y, 2)) == (1.6, 1.6)      # giant, facing right
    assert (round(sb.x, 2), round(sb.y, 2)) == (-1.6, 1.6)     # giant, mirrored


def test_facing_persists_and_replays():
    s = SessionStore().create(width=300, height=340, frames=4, ground_y=300)
    s.run('b=stick()\n'
          'add_action(b, actions.idle(b, ground_y=ground, x=210), name="b", '
          'face="stick", facing=-1)')
    assert s.doc["characters"][0]["facing"] == -1
    s.save()
    back = Session.replay(s.doc_id)
    assert _char_layer(back.scene, "b").transform.scale.value.x == -1.0


# ---------------------------------------------------------------- emotions
def test_face_has_the_expanded_expression_set():
    atts = load_face("stick")["attachments"]
    for e in ("neutral", "happy", "sad", "surprised", "angry", "tired", "excited"):
        assert e in atts, f"stick face is missing the {e!r} expression"


def test_set_expression_changes_the_rendered_face():
    def face_ink(exp):
        s = SessionStore().create(width=140, height=360, frames=3, ground_y=330)
        s.run(f'background("#fff")\nm=stick(ink="#111")\n'
              f'add_action(m, actions.idle(m, ground_y=ground, x=70), name="m", face="stick")\n'
              f'set_expression("m","{exp}",0)')
        a = np.asarray(render_frame(s.scene, 1).convert("L").crop((30, 40, 110, 116)))
        return (a < 100).sum()
    # a heavy-lidded tired face and a wide excited face put different ink on the head
    assert face_ink("tired") != face_ink("excited")
