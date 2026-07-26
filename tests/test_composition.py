"""Composition & connectors: the last dogfood gaps — locomotion composing with
actions, smoothed sequence joins, and ropes/leashes that span moving things.
"""

from __future__ import annotations

import math

from glaxnimate_ai.cartoon import actions
from glaxnimate_ai.cartoon.presets import make_gait, stick
from glaxnimate_ai.cartoon.timeline import from_pose_fn
from glaxnimate_ai.feedback.lint import lint_timeline
from glaxnimate_ai.engine.session import Session, SessionStore

GROUND = 300.0


def _foot_x(body, pose_fn, f):
    return body.rig.solve(pose_fn(float(f)))["shin_l"].tip.x


def test_locomote_lets_a_gait_compose_in_a_sequence():
    body = stick()
    gait = make_gait(body, "walk", cycle_frames=14)
    combo = actions.sequence(
        (actions.locomote(body, gait, ground_y=GROUND, x0=100), 14),
        (actions.fall(body, ground_y=GROUND, x=200, frames=20), 20),
    )
    # it walks (root travels) then falls, and the whole thing lints clean
    x_start = combo(0.0).root.x
    x_walked = combo(13.0).root.x
    assert x_walked > x_start + 20, "the gait segment should actually travel"
    tl = from_pose_fn(body, combo, frames=34)
    assert lint_timeline(tl, ground_y=GROUND).ok


def test_blend_smooths_a_pose_join_without_changing_length():
    body = stick()
    seg_a = actions.celebrate(body, ground_y=GROUND, x=150, frames=16)
    seg_b = actions.idle(body, ground_y=GROUND, x=150)
    hard = actions.sequence((seg_a, 16), (seg_b, 14))
    soft = actions.sequence((seg_a, 16), (seg_b, 14), blend=6)
    # near the seam, the biggest arm jump between consecutive frames is smaller
    def worst_arm_jump(fn):
        prev, worst = None, 0.0
        for f in range(31):
            a = fn(float(f)).angles.get("arm_upper", 0.0)
            if prev is not None:
                worst = max(worst, abs(a - prev))
            prev = a
        return worst
    assert worst_arm_jump(soft) < worst_arm_jump(hard), "blend should ease the join"
    # same-position feet stay clean under blend (no new skating)
    assert lint_timeline(from_pose_fn(body, soft, frames=30), ground_y=GROUND).ok


def test_rope_draws_a_static_line():
    s = SessionStore().create(width=400, height=300, frames=10)
    r = s.run("rope(60, 120, 340, 140, sag=20)")
    assert r.ok, r.format()
    assert len(s.doc["ropes"]) == 1
    assert any(sh.name.startswith("rope.") for sh in s.scene.comp.shapes)


def test_leash_spans_two_moving_characters_and_persists():
    st = SessionStore()
    s = st.create(width=640, height=340, frames=24)
    r = s.run(
        "person = stick(); dog = quadruped()\n"
        "add_character(person, make_gait(person,'walk',cycle_frames=24), x=150, name='person')\n"
        "add_character(dog, make_gait(dog,'trot',cycle_frames=16), x=330, name='dog', style='lineart')\n"
        "leash('person','arm_lower','dog','neck')")
    assert r.ok, r.format()
    assert len(s.doc["leashes"]) == 1
    lay = next(sh for sh in s.scene.comp.shapes if sh.name.startswith("leash."))
    g = lay.shapes[0]  # the group carries the animated span transform
    # the connector re-aims every frame: its length (scale x) changes as they move
    lens = {round(g.transform.scale.value_at_time(float(f)).x) for f in range(0, 24, 4)}
    assert len(lens) > 1, "the leash should stretch as the gap changes"
    s.save()
    assert len(Session.replay(s.doc_id).doc["leashes"]) == 1


def test_fly_stays_airborne_and_travels():
    body = stick()
    fn = actions.fly(body, ground_y=GROUND, x0=60, x1=300, height=120, frames=30)
    # never plants a foot (always well above the ground) and moves across
    for f in range(31):
        feet = body.rig.solve(fn(float(f)))
        for shin in ("shin_l", "shin_r"):
            assert GROUND - feet[shin].tip.y > 20, "flight should keep the feet up"
    assert fn(30.0).root.x > fn(0.0).root.x + 100, "it should travel"
