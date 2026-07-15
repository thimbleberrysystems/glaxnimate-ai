"""Actions compose the principles — so each test asserts the principle is present.

A jump that doesn't anticipate isn't a jump, it's a levitation. A trail that
doesn't lag isn't follow-through. These check the mechanics that make each action
read as the principle it claims to embody.
"""

from __future__ import annotations

from glaxnimate_ai.cartoon import actions
from glaxnimate_ai.cartoon.gait import pose_at
from glaxnimate_ai.cartoon.presets import human, make_gait, quadruped
from glaxnimate_ai.feedback.lint import lint_rig

GROUND = 420.0


def _root_heights(body, pose_fn, frames):
    return [pose_fn(float(f)).root.y for f in range(frames + 1)]


# ------------------------------------------------------------------- jump
def test_jump_anticipates_then_leaves_the_ground():
    """Anticipation: the body must dip DOWN before it goes up."""
    man = human()
    jump = actions.jump(man, ground_y=GROUND, x=200, height=150, frames=36)
    ys = _root_heights(man, jump, 36)

    stand = ys[0]
    lowest_early = max(ys[:10])         # y grows downward, so max = lowest point
    highest = min(ys)                   # min y = top of the arc
    assert lowest_early > stand + 5, "no anticipation crouch before the jump"
    assert highest < stand - 80, "the jump never got off the ground"


def test_jump_lands_with_feet_on_the_ground():
    man = human()
    jump = actions.jump(man, ground_y=GROUND, x=200, height=150, frames=36)
    end = man.rig.solve(jump(36.0))
    for c in man.rig.contacts:
        assert abs(end[c].tip.y - GROUND) < 6, "feet are not on the ground at landing"


def test_jump_lints_clean():
    man = human()
    jump = actions.jump(man, ground_y=GROUND, x=200, height=150, frames=36)
    rep = lint_rig(man, jump, frames=36, ground_y=GROUND, canvas=(600, 480))
    assert rep.ok, rep.format()


# ------------------------------------------------------------------- trail
def test_trail_streams_behind_the_motion():
    """Follow-through: a loose part deflects opposite to the direction of travel."""
    dog = quadruped()
    gait = make_gait(dog, "trot", cycle_frames=16, stride=120)
    base = lambda t: pose_at(dog.rig, gait, t, ground_y=GROUND, body_x0=0)  # noqa: E731

    trailed = actions.trail(base, dog, ["tail"], lag=3, swing=30)
    moving = trailed(20.0).angles["tail"] - base(20.0).angles["tail"]
    assert moving < 0, "tail should stream backward while the body moves right"


def test_trail_is_still_when_the_body_is_still():
    """Overlapping action is velocity-driven: no motion, no swing."""
    man = human()
    still = actions.idle(man, ground_y=GROUND, x=100, cycle_frames=48)
    trailed = actions.trail(still, man, ["head"], lag=3, swing=30)
    # idle barely moves horizontally, so the trailing deflection must be tiny
    dev = abs(trailed(24.0).angles.get("head", 0.0) - still(24.0).angles.get("head", 0.0))
    assert dev < 2.0, "a stationary character's trail should be near zero"


# ------------------------------------------------------------------- idle / wave
def test_idle_breathes_but_keeps_feet_planted():
    man = human()
    idle = actions.idle(man, ground_y=GROUND, x=150, cycle_frames=48)
    ys = _root_heights(man, idle, 48)
    assert max(ys) - min(ys) > 1.0, "an idle with no motion at all reads as dead"
    assert max(ys) - min(ys) < 20.0, "an idle should breathe, not bounce"
    end = man.rig.solve(idle(24.0))
    for c in man.rig.contacts:
        assert abs(end[c].tip.y - GROUND) < 6, "idle feet drifted off the ground"


def test_wave_raises_the_arm():
    man = human()
    w = actions.wave(man, ground_y=GROUND, x=100, frames=48)
    # the waving arm should be well away from its rest hang partway through
    mid = w(30.0)
    assert abs(mid.angles.get("arm_upper", 0.0)) > 40, "the arm never lifted to wave"


# ------------------------------------------------------------------- sequence
def test_sequence_plays_actions_back_to_back():
    man = human()
    idle = actions.idle(man, ground_y=GROUND, x=100)
    jump = actions.jump(man, ground_y=GROUND, x=100, height=120, frames=30)
    combined = actions.sequence((idle, 12), (jump, 30))

    # first segment is the idle (near standing), second is the jump (leaves ground)
    idle_y = combined(6.0).root.y
    jump_peak = min(combined(float(f)).root.y for f in range(12, 42))
    assert jump_peak < idle_y - 60, "the jump segment never fired"
