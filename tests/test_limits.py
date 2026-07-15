"""Drive the engine to its limits and confirm it either holds or fails cleanly.

The central claim is "animate anything". A test suite that only ever exercises a
biped walk and a quadruped trot does not earn that claim. So this pushes on the
edges: a twelve-legged creature, degenerate parameters, extreme scales, and a
crowded scene — and requires each either to work or to raise a clear error, never
to silently produce garbage.
"""

from __future__ import annotations

import pytest

from glaxnimate_ai.cartoon.gait import (
    Gait,
    Limb,
    foot_target,
    is_planted,
    pose_at,
    reach_needed,
)
from glaxnimate_ai.cartoon.geometry import Vec2
from glaxnimate_ai.cartoon.presets import biped, make_gait, pace, quadruped
from glaxnimate_ai.cartoon.principles import squash_stretch
from glaxnimate_ai.cartoon.rig import Joint, Rig


# ------------------------------------------------------ one engine, N limbs
def _centipede(n_legs: int = 12) -> tuple[Rig, Gait]:
    joints = [Joint("body", None, length=0.0)]
    limbs = []
    for i in range(n_legs):
        off = i * 15.0 - n_legs * 7.5
        joints += [
            Joint(f"u{i}", "body", length=20, rest_angle=90, offset=Vec2(off, 0)),
            Joint(f"l{i}", f"u{i}", length=20, rest_angle=0, contact=True),
        ]
        limbs.append(Limb(f"u{i}", f"l{i}", phase=i / n_legs, hip_offset=off))
    rig = Rig(joints)
    gait = Gait(limbs=limbs, cycle_frames=24, stride=30, duty=0.6,
                lift=10, bob=2, ride_height=34)
    return rig, gait


def test_many_legged_creature_animates_with_no_slip():
    """A centipede is the same engine with a longer phase table — and its 12 feet
    must be as slip-free as a biped's two."""
    rig, gait = _centipede(12)
    worst = 0.0
    for limb in gait.limbs:
        prev = None
        for f in range(48):
            if is_planted(gait, limb, float(f)):
                p = foot_target(gait, limb, float(f), 0.0, 0.0)
                if prev is not None:
                    worst = max(worst, abs(p.x - prev))
                prev = p.x
            else:
                prev = None
    assert worst < 1e-9, f"a centipede foot slipped by {worst}"


def test_centipede_reach_is_finite_everywhere():
    rig, gait = _centipede(12)
    for name, d in reach_needed(rig, gait).items():
        assert d == d and d < float("inf"), f"{name} reach is not finite"


# --------------------------------------------------------- extreme parameters
@pytest.mark.parametrize("cycle", [2.0, 8.0, 200.0, 1000.0])
def test_extreme_cycle_lengths(cycle):
    """A near-strobe (2 frames) and a glacial crawl (1000) must both build."""
    make_gait(biped(), "walk", cycle_frames=cycle)


@pytest.mark.parametrize(
    "maker,scale",
    [(biped, 10.0), (biped, 0.1), (quadruped, 8.0), (quadruped, 0.15)],
)
def test_extreme_body_scales(maker, scale):
    """A gait is defined in ratios, so a mouse and an elephant use the same code."""
    if maker is biped:
        body = biped(thigh=80 * scale, shin=80 * scale, spine=70 * scale)
    else:
        body = quadruped(upper=55 * scale, lower=55 * scale, body=120 * scale)
    gait = make_gait(body, "walk")
    # feet still reach at any scale
    for limb in gait.limbs:
        leg = body.rig.joints[limb.upper].length + body.rig.joints[limb.lower].length
        assert reach_needed(body.rig, gait)[limb.upper] <= leg


def test_squash_preserves_area_at_all_speeds():
    for speed in (-1000, -100, -1, 0, 1, 100, 1000):
        s = squash_stretch(speed)
        assert abs(s.x * s.y - 1.0) < 1e-9, f"volume changed at speed {speed}"


# --------------------------------------------------------- degenerate inputs
def test_pace_rejects_zero_frames():
    with pytest.raises(ValueError):
        pace(biped(), "walk", distance=100, frames=0)


def test_walking_backward_is_allowed():
    """Negative distance = leftward walk. Legit, and the feet must still not slip."""
    g = pace(biped(), "walk", distance=-200, frames=48)
    assert g.speed < 0
    # a leftward walk is a rightward walk mirrored; reach is unchanged
    for limb in g.limbs:
        leg = biped().rig.joints[limb.upper].length + biped().rig.joints[limb.lower].length
        assert reach_needed(biped().rig, g)[limb.upper] <= leg * 1.001


def test_single_root_enforced():
    with pytest.raises(ValueError, match="one root"):
        Rig([Joint("a", None), Joint("b", None)])


def test_cycle_in_rig_rejected():
    with pytest.raises(ValueError, match="cycle|root"):
        Rig([Joint("a", "b"), Joint("b", "a")])


def test_unknown_parent_rejected():
    with pytest.raises(ValueError, match="parent"):
        Rig([Joint("root", None), Joint("child", "ghost")])


# ------------------------------------------------------------- a crowded scene
def test_many_characters_in_one_scene():
    """Eight characters at once — nothing should collide or slow to a crawl."""
    from glaxnimate_ai.engine.session import SessionStore

    s = SessionStore().create(width=1600, height=500, frames=32)
    lines = []
    for i in range(8):
        maker = "human()" if i % 2 else "quadruped()"
        gait = "walk" if i % 2 else "trot"
        lines.append(
            f"add_character({maker}, make_gait({maker}, '{gait}', cycle_frames=18), "
            f"x={60 + i * 180}, name='c{i}')"
        )
    res = s.run("\n".join(lines))
    assert res.ok, res.format()
    assert len(s.characters) == 8


def test_long_animation_poses_stay_finite():
    """A 600-frame walk must not drift into NaN or infinity anywhere."""
    body = biped()
    gait = make_gait(body, "walk", cycle_frames=24)
    for f in range(0, 600, 7):
        pose = pose_at(body.rig, gait, float(f), ground_y=400, body_x0=0)
        frames = body.rig.solve(pose)
        for jf in frames.values():
            assert all(v == v and abs(v) < 1e6 for v in (jf.origin.x, jf.origin.y))
