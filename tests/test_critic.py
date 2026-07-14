"""The critic stack must catch faults we deliberately introduce.

This is the test that earns the right to say "we mostly don't need images". A
linter that only ever passes is worthless — so every check here is fed a broken
animation and required to notice.
"""

from __future__ import annotations

import math

import pytest

from glaxnimate_ai.cartoon.gait import pose_at
from glaxnimate_ai.cartoon.geometry import Vec2
from glaxnimate_ai.cartoon.presets import biped, make_gait, quadruped
from glaxnimate_ai.feedback.diagnose import arc_quality, diagnose_rig, spacing_chart
from glaxnimate_ai.feedback.lint import lint_rig

GROUND = 470.0
FRAMES = 48


def good_walk(make_body=biped, gait_name="walk"):
    body = make_body()
    gait = make_gait(body, gait_name, cycle_frames=24)

    def pose_fn(t: float):
        return pose_at(body.rig, gait, t, ground_y=GROUND, body_x0=90,
                       hip_height=body.hip_height)

    return body, gait, pose_fn


# --------------------------------------------------------- the happy path
@pytest.mark.parametrize("make_body,gait_name", [(biped, "walk"), (quadruped, "trot")])
def test_clean_walk_passes_the_linter(make_body, gait_name):
    body, gait, pose_fn = good_walk(make_body, gait_name)
    rep = lint_rig(body, pose_fn, frames=FRAMES, ground_y=GROUND,
                   limbs=gait.limbs, canvas=(960, 540))
    assert rep.ok, rep.format()


def test_clean_walk_passes_the_diagnostics():
    body, _, pose_fn = good_walk()
    d = diagnose_rig(body, pose_fn, frames=FRAMES, ground_y=GROUND)
    assert d.ok, d.format()


# --------------------------------------------------- deliberately broken input
def test_sliding_foot_is_caught():
    """Drag the whole figure sideways: the planted feet must be seen to skate."""
    body, gait, pose_fn = good_walk()

    def sliding(t: float):
        p = pose_fn(t)
        p.root = p.root + Vec2(3.0 * t, 0.0)  # feet stay put; body runs away
        return p

    rep = lint_rig(body, sliding, frames=FRAMES, ground_y=GROUND, limbs=gait.limbs)
    checks = {i.check for i in rep.issues}
    assert not rep.ok
    # Either the foot visibly slid, or the leg tore free trying to keep up.
    assert "contact_slip" in checks or "ik_overextension" in checks, rep.format()


def test_overextended_leg_is_caught():
    """Ask the hips to ride above full leg extension while the feet stay on the floor.

    Note what does *not* work: translating the whole pose upward. That lifts the
    rig bodily, legs and all — the feet leave the ground still perfectly bent and
    nothing is strained. The leg only tears when the hips move *and the feet are
    held*, which is precisely the situation `make_gait`'s reach guard exists to
    prevent. Here we go around that guard on purpose, to prove the linter is a
    real second line of defence rather than decoration.
    """
    body = biped()
    gait = make_gait(body, "walk", cycle_frames=24)

    def stilts(t: float):
        # Hips above the leg's full reach: the feet are still pinned to GROUND by
        # the IK targets, so the legs must stretch further than they physically can.
        return pose_at(body.rig, gait, t, ground_y=GROUND, body_x0=90,
                       hip_height=body.leg_length * 1.05)

    rep = lint_rig(body, stilts, frames=FRAMES, ground_y=GROUND, limbs=gait.limbs)
    assert not rep.ok
    assert any(i.check == "ik_overextension" for i in rep.issues), rep.format()


def test_ground_penetration_is_caught():
    body, gait, pose_fn = good_walk()

    def sunken(t: float):
        p = pose_fn(t)
        p.root = p.root + Vec2(0.0, 40.0)  # hips dropped; feet go through the floor
        return p

    rep = lint_rig(body, sunken, frames=FRAMES, ground_y=GROUND, limbs=gait.limbs)
    assert any(i.check == "ground_penetration" for i in rep.issues), rep.format()


def test_nan_is_caught():
    body, gait, pose_fn = good_walk()

    def poisoned(t: float):
        p = pose_fn(t)
        if t > 10:
            p.root = Vec2(float("nan"), 0.0)
        return p

    rep = lint_rig(body, poisoned, frames=FRAMES, ground_y=GROUND, limbs=gait.limbs)
    assert any(i.check == "nan" for i in rep.issues), rep.format()


def test_strobing_is_caught():
    body, gait, pose_fn = good_walk()

    def teleport(t: float):
        p = pose_fn(t)
        if int(t) % 2 == 0:
            p.root = p.root + Vec2(300.0, 0.0)  # flickers back and forth
        return p

    rep = lint_rig(body, teleport, frames=FRAMES, ground_y=GROUND, limbs=gait.limbs)
    assert any(i.check == "strobing" for i in rep.issues), rep.format()


# ------------------------------------------------- diagnostics on known-bad input
def test_linear_interpolation_reads_as_robotic():
    """Constant spacing is the signature of animation nobody timed."""
    linear = [Vec2(10.0 * i, 0.0) for i in range(20)]
    assert spacing_chart(linear)["variation"] < 0.01

    eased = [Vec2(100.0 * (i / 19) ** 2, 0.0) for i in range(20)]
    assert spacing_chart(eased)["variation"] > 0.2, "an eased move must not look linear"


def test_zigzag_path_fails_the_arc_check():
    """A hand that jitters instead of sweeping must be caught."""
    arc = [Vec2(i * 10.0, -40.0 * math.sin(math.pi * i / 19)) for i in range(20)]
    assert arc_quality(arc)["reversals"] <= 1, "a smooth arc should not reverse"

    zigzag = [Vec2(i * 10.0, 30.0 * (-1) ** i) for i in range(20)]
    assert arc_quality(zigzag)["reversals"] > 4, "a zigzag must be flagged"


def test_standing_figure_leaning_past_its_feet_is_caught():
    """A *stationary* figure whose mass is outside its feet would fall over.

    Two traps here, both of which I walked into:

    1. Translating the root does not unbalance anything — the feet are children of
       the root, so they move with it and the figure stays perfectly over them.
       The torso has to lean while the legs stay planted.
    2. Balance is statics, so it only means anything when the figure is still. On a
       walk the centre of mass overhangs the support on every step by design, and
       the check cannot separate that from a genuine topple. Hence `stride=0`: the
       figure marks time on the spot, and now balance is a fair question.
    """
    body = biped()
    standing = make_gait(body, "walk", cycle_frames=24, stride=0.0)

    def upright(t: float):
        return pose_at(body.rig, standing, t, ground_y=GROUND, body_x0=300,
                       hip_height=body.hip_height)

    def bent_double(t: float):
        p = upright(t)
        p.angles["spine"] = 80.0  # torso almost horizontal; legs stay planted
        return p

    ok = diagnose_rig(body, upright, frames=FRAMES, ground_y=GROUND)
    assert next(f for f in ok.findings if f.metric == "off_balance").verdict == "good", ok.format()

    bad = diagnose_rig(body, bent_double, frames=FRAMES, ground_y=GROUND)
    off = next(f for f in bad.findings if f.metric == "off_balance")
    assert off.verdict == "poor", bad.format()
