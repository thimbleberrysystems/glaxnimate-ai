"""The load-bearing test: one engine, many creatures, zero contact slip.

If this fails, the project's central claim is false.
"""
import math

import pytest

from glaxnimate_ai.cartoon import motion
from glaxnimate_ai.cartoon.gait import foot_target, is_planted, pose_at
from glaxnimate_ai.cartoon.presets import biped, make_gait, quadruped


def _max_slip(body, gait, ground_y=500.0, frames=96):
    """Largest world-space movement of any foot while it is planted."""
    worst = 0.0
    for limb in gait.limbs:
        prev = None
        for f in range(frames):
            t = float(f)
            if is_planted(gait, limb, t):
                p = foot_target(gait, limb, t, ground_y, 0.0)
                if prev is not None:
                    worst = max(worst, p.distance_to(prev))
                prev = p
            else:
                prev = None  # airborne: a new plant starts fresh
    return worst


@pytest.mark.parametrize(
    "make_body,gait_name",
    [
        (biped, "walk"),
        (biped, "run"),
        (quadruped, "walk"),
        (quadruped, "trot"),
        (quadruped, "gallop"),
    ],
)
def test_planted_feet_do_not_slip(make_body, gait_name):
    body = make_body()
    gait = make_gait(body, gait_name)
    assert _max_slip(body, gait) < 1e-9, "a planted foot moved: the character is skating"


@pytest.mark.parametrize("make_body,gait_name", [(biped, "walk"), (quadruped, "trot")])
def test_ik_puts_the_foot_exactly_on_target(make_body, gait_name):
    """The legs must actually reach the targets the gait asks for."""
    body = make_body()
    gait = make_gait(body, gait_name)
    ground_y = 500.0

    for f in range(0, 48):
        t = float(f)
        pose = pose_at(body.rig, gait, t, ground_y=ground_y, hip_height=body.hip_height)
        frames = body.rig.solve(pose)
        for limb in gait.limbs:
            want = foot_target(gait, limb, t, ground_y, 0.0)
            got = frames[limb.lower].tip
            assert got.distance_to(want) < 1e-6, f"IK missed at frame {f}"


def test_gait_table_is_a_real_trot():
    """Diagonal pairs move together — that is what makes a trot a trot."""
    body = quadruped()
    trot = make_gait(body, "trot")
    hind_l, hind_r, fore_l, fore_r = trot.limbs
    assert hind_l.phase == fore_r.phase
    assert hind_r.phase == fore_l.phase
    assert hind_l.phase != hind_r.phase


def test_wheel_does_not_skate():
    """No-slip: distance travelled must equal radius * angle."""
    r = 40.0
    s = motion.roll(x0=0.0, x1=600.0, y=0.0, radius=r, frames=60)
    for a, b in zip(s, s[1:], strict=False):
        travelled = b.pos.x - a.pos.x
        rolled = math.radians(b.angle - a.angle) * r
        assert abs(travelled - rolled) < 1e-9


def test_ball_bounce_decays_and_preserves_area():
    s = motion.bounce(x0=0, x1=800, ground_y=500, apex=300, frames=90, bounces=5)
    heights = [500 - 40 - smp.pos.y for smp in s]
    first = max(heights[:20])
    last = max(heights[-20:])
    assert last < first * 0.6, "bounces should lose height"
    for smp in s:
        assert abs(smp.scale.x * smp.scale.y - 1.0) < 1e-9, "squash must preserve area"
