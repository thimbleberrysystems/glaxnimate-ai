"""The Timeline IR: the v2 seam everything consumes.

The point of the IR is that the critic reads *data*, not closures — so these tests
check both directions: rigs sample into timelines that lint identically to the old
path, and things that were invisible to the v1 critic (balls) are now checkable.
"""

from __future__ import annotations

from glaxnimate_ai.cartoon import motion, timeline
from glaxnimate_ai.cartoon.gait import pose_at
from glaxnimate_ai.cartoon.presets import biped, make_gait
from glaxnimate_ai.feedback.lint import lint_object, lint_rig, lint_timeline

GROUND = 470.0


def _walk():
    body = biped()
    gait = make_gait(body, "walk", cycle_frames=24)

    def pose_fn(t: float):
        return pose_at(body.rig, gait, t, ground_y=GROUND, body_x0=90)

    return body, gait, pose_fn


# ------------------------------------------------------------- construction
def test_timeline_carries_the_metadata_the_critic_needs():
    body, gait, pose_fn = _walk()
    tl = timeline.from_pose_fn(body, pose_fn, frames=24, limbs=gait.limbs)

    assert tl.root == "hips"
    assert tl.leg_length == body.leg_length
    assert ("thigh_l", "shin_l") in tl.limbs
    assert "shin_l" in tl.contacts
    # every node densely sampled: frames+1 entries
    assert all(len(t.origin) == 25 for t in tl.nodes.values())


def test_timeline_lints_identically_to_the_wrapper():
    body, gait, pose_fn = _walk()
    tl = timeline.from_pose_fn(body, pose_fn, frames=48, limbs=gait.limbs)

    via_ir = lint_timeline(tl, ground_y=GROUND, canvas=(960, 540))
    via_wrapper = lint_rig(body, pose_fn, frames=48, ground_y=GROUND,
                           limbs=gait.limbs, canvas=(960, 540))
    assert via_ir.ok and via_wrapper.ok
    assert len(via_ir.issues) == len(via_wrapper.issues)


# ---------------------------------------------- objects are now first-class
def test_a_clean_bounce_lints_clean():
    ball = motion.bounce(x0=100, x1=700, ground_y=GROUND, apex=200,
                         frames=48, bounces=4, radius=30)
    rep = lint_object("ball", ball, ground_y=GROUND, radius=30, canvas=(960, 540))
    assert rep.ok, rep.format()


def test_a_ball_through_the_floor_is_caught():
    """v1 could not see this at all — objects had no pose_fn to inspect."""
    ball = motion.bounce(x0=100, x1=700, ground_y=GROUND + 40, apex=200,
                         frames=48, bounces=4, radius=30)  # ground 40px too low
    rep = lint_object("ball", ball, ground_y=GROUND, radius=30)
    assert not rep.ok
    assert any(i.check == "ground_penetration" for i in rep.issues), rep.format()


def test_a_bounce_does_not_false_positive_on_slip():
    """A ball travels horizontally through its contacts — that is physics, not a
    skating bug, and the slip check must not fire on it."""
    ball = motion.bounce(x0=100, x1=800, ground_y=GROUND, apex=150,
                         frames=60, bounces=6, radius=26)
    rep = lint_object("ball", ball, ground_y=GROUND, radius=26)
    assert not any(i.check == "contact_slip" for i in rep.issues), rep.format()


def test_nan_in_samples_is_caught():
    from glaxnimate_ai.cartoon.geometry import Vec2
    from glaxnimate_ai.cartoon.motion import Sample

    samples = [Sample(0, Vec2(0, 0)), Sample(1, Vec2(float("nan"), 0))]
    rep = lint_object("thing", samples, ground_y=GROUND)
    assert any(i.check == "nan" for i in rep.issues)
