"""Combat / stunt beats: the acting vocabulary that makes a stick figure fight.

These test *geometry*, not pixels — a beat is a `pose_fn(t)`, so we solve the rig
at the telling frames and assert the fist reaches, the foot rises, the flip turns,
grounded beats keep their feet down, and every beat stays finite. Cheap and exact,
the way the critic stack prefers.
"""

from __future__ import annotations

import math

import pytest

from glaxnimate_ai.cartoon import actions
from glaxnimate_ai.cartoon.presets import quadruped, stick

GROUND = 300.0


def _solved(body, pose_fn, t):
    return body.rig.solve(pose_fn(t))


ALL_BEATS = {
    "punch": lambda b: actions.punch(b, ground_y=GROUND, x=170, frames=16),
    "kick": lambda b: actions.kick(b, ground_y=GROUND, x=170, frames=18),
    "dash": lambda b: actions.dash(b, ground_y=GROUND, x0=60, x1=250, frames=14),
    "flip": lambda b: actions.flip(b, ground_y=GROUND, x=120, frames=26),
    "swing": lambda b: actions.swing(b, ground_y=GROUND, x=170, frames=18),
    "block": lambda b: actions.block(b, ground_y=GROUND, x=170, frames=12),
    "knockback": lambda b: actions.knockback(b, ground_y=GROUND, x=200, frames=16),
    "land": lambda b: actions.land(b, ground_y=GROUND, x=170, frames=14),
    # everyday acting verbs
    "celebrate": lambda b: actions.celebrate(b, ground_y=GROUND, x=170, frames=26),
    "fall": lambda b: actions.fall(b, ground_y=GROUND, x=200, frames=24),
    "sit": lambda b: actions.sit(b, ground_y=GROUND, x=170, frames=18),
    "tap": lambda b: actions.tap(b, ground_y=GROUND, x=170, frames=24),
}
FRAMES = {"punch": 16, "kick": 18, "dash": 14, "flip": 26, "swing": 18,
          "block": 12, "knockback": 16, "land": 14,
          "celebrate": 26, "fall": 24, "sit": 18, "tap": 24}


@pytest.mark.parametrize("name", list(ALL_BEATS))
def test_every_beat_lints_clean(name):
    """No skating, no feet through the floor — even the translating beats. A dash and
    a knockback lift the feet for the move; a kick chambers above the ground."""
    from glaxnimate_ai.cartoon.timeline import from_pose_fn
    from glaxnimate_ai.feedback.lint import lint_timeline

    body = stick()
    tl = from_pose_fn(body, ALL_BEATS[name](body), frames=FRAMES[name])
    rep = lint_timeline(tl, ground_y=GROUND)
    errors = [i for i in rep.issues if i.severity == "error"]
    assert not errors, f"{name}: {[(i.check, i.frame) for i in errors[:4]]}"


@pytest.mark.parametrize("name", list(ALL_BEATS))
def test_every_beat_stays_finite(name):
    body = stick()
    fn = ALL_BEATS[name](body)
    n = FRAMES[name]
    for f in range(n + 1):
        frames = _solved(body, fn, float(f))
        for jf in frames.values():
            assert math.isfinite(jf.origin.x) and math.isfinite(jf.origin.y)
            assert math.isfinite(jf.tip.x) and math.isfinite(jf.tip.y)


def test_punch_drives_the_fist_forward_at_contact():
    body = stick()
    fn = actions.punch(body, ground_y=GROUND, x=170, frames=16)
    # hit lands ~ frame 0.60 * 16
    hit = _solved(body, fn, 9.6)
    start = _solved(body, fn, 0.0)
    fist = hit["arm_lower"].tip
    shoulder = hit["arm_upper"].origin  # the actual shoulder, not the elbow
    # the fist is well forward of the shoulder (a straight jab, near full reach),
    # and further out than it began (it was chambered).
    assert fist.x - shoulder.x > 70, "the punch did not extend forward"
    assert fist.x > start["arm_lower"].tip.x + 20, "the fist did not travel"


def test_kick_lifts_the_foot_and_keeps_the_support_planted():
    body = stick()
    fn = actions.kick(body, ground_y=GROUND, x=170, frames=18)
    hit = _solved(body, fn, 10.4)  # ~0.58 * 18
    # legs: one kicks (foot rises well above the ground), one stays planted.
    foot_l = hit["shin_l"].tip.y
    foot_r = hit["shin_r"].tip.y
    high, low = min(foot_l, foot_r), max(foot_l, foot_r)
    assert GROUND - high > 60, "the kicking foot should leave the ground"
    assert abs(low - GROUND) < 25, "the support foot should stay planted"


@pytest.mark.parametrize("name", ["punch", "block", "land", "knockback"])
def test_grounded_beats_keep_feet_near_the_ground(name):
    body = stick()
    fn = ALL_BEATS[name](body)
    for f in range(FRAMES[name] + 1):
        frames = _solved(body, fn, float(f))
        for shin in ("shin_l", "shin_r"):
            foot_y = frames[shin].tip.y
            assert foot_y <= GROUND + 6, f"{name}: foot punched through the floor"
            assert GROUND - foot_y < 60, f"{name}: foot floated off the ground"


def test_flip_turns_a_full_circle():
    body = stick()
    fn = actions.flip(body, ground_y=GROUND, x=120, frames=26)
    spins = [fn(float(f)).root_angle for f in range(27)]
    assert max(abs(s) for s in spins) > 300, "a flip should approach a full turn"
    assert abs(spins[0]) < 1 and abs(spins[-1]) < 1, "it should start and end upright"


def test_beats_compose_in_a_sequence():
    body = stick()
    combo = actions.sequence(
        (actions.dash(body, ground_y=GROUND, x0=40, x1=150, frames=12), 12),
        (actions.punch(body, ground_y=GROUND, x=150, frames=14), 14),
    )
    for f in range(26):
        frames = body.rig.solve(combo(float(f)))
        assert all(math.isfinite(jf.tip.x) for jf in frames.values())


def test_hitstop_freezes_on_impact_then_resumes():
    body = stick()
    fn = actions.punch(body, ground_y=GROUND, x=170, frames=16)
    hs = actions.hitstop(fn, at=10.0, freeze=3.0)
    fist = lambda f: body.rig.solve(hs(float(f)))["arm_lower"].tip.x
    # the whole freeze window holds the contact pose dead-still
    assert fist(10) == fist(11) == fist(12), "hitstop should freeze the pose"
    # then it resumes exactly where it paused (frame 13 == original frame 10)
    resumed = body.rig.solve(hs(13.0))["arm_lower"].tip.x
    original = body.rig.solve(fn(10.0))["arm_lower"].tip.x
    assert abs(resumed - original) < 1e-9, "hitstop should resume, not skip"


def test_hitstop_beat_bakes_and_lints_clean_over_its_extra_length():
    from glaxnimate_ai.engine.session import SessionStore
    from glaxnimate_ai.feedback.lint import lint_rig

    s = SessionStore().create(width=360, height=340, frames=19)  # 16 + 3 freeze
    r = s.run(
        "man = stick()\n"
        "beat = actions.hitstop(actions.punch(man, ground_y=ground, x=180, frames=16),"
        " at=10, freeze=3)\n"
        "add_action(man, beat, name='fighter')"
    )
    assert r.ok, r.format()
    ch = s.characters[0]
    rep = lint_rig(ch.body, ch.pose_fn, frames=s.frames, ground_y=s.ground_y,
                   limbs=ch.limb_pairs)
    assert rep.ok, rep.format()


def test_retime_changes_spacing_not_endpoints():
    from glaxnimate_ai.cartoon.principles import ease_in
    body = stick()
    fn = actions.punch(body, ground_y=GROUND, x=170, frames=16)
    rt = actions.retime(fn, 16, ease_in)
    end = lambda g, f: body.rig.solve(g(float(f)))["arm_lower"].tip.x
    # endpoints are preserved; only the interior spacing (the timing) changes
    assert abs(end(rt, 0) - end(fn, 0)) < 1e-9
    assert abs(end(rt, 16) - end(fn, 16)) < 1e-9
    # ease_in holds the start back: at the midpoint the retimed beat lags the linear
    assert end(rt, 8) != end(fn, 8)


def test_beats_are_generic_not_humanoid_only():
    """A beat must not crash on a non-biped: _set guards missing arm joints, and the
    leg helpers are rig-generic. Scope is 'animate anything', not 'animate people'."""
    dog = quadruped()
    fn = actions.knockback(dog, ground_y=GROUND, x=200, frames=16)
    for f in range(17):
        frames = dog.rig.solve(fn(float(f)))
        assert all(math.isfinite(jf.tip.y) for jf in frames.values())
