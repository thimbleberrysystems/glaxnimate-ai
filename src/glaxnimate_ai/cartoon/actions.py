"""Actions: the principles composed into things a character *does*.

Gaits handle locomotion. This handles the rest of the acting vocabulary — a jump,
an idle, a wave, a loose cape trailing behind — by composing the animation
principles the library already has.

Two of the twelve principles had the maths but nothing that used them, which is
the same as not having them:

* **Anticipation** — you wind up before you go. `jump` crouches before it launches.
* **Follow-through / overlapping action** — loose parts (a tail, a cape, an ear,
  hair) keep moving after the body stops, and lag behind while it moves. `trail`
  drives a chain from a delayed, damped copy of its own base, so it swings and
  settles instead of moving rigidly with the body.

Everything here returns a `pose_fn(t) -> Pose`, the same currency `bake_rig` and
the linter already speak, so an action drops into the pipeline exactly like a gait.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .geometry import Vec2, clamp
from .presets import Body
from .principles import anticipate, ease_in_out, ease_out
from .rig import Pose, solve_two_bone

__all__ = ["jump", "idle", "wave", "trail", "sequence"]

PoseFn = Callable[[float], Pose]


def _ground_legs(body: Body, pose: Pose, frames_solved, foot_y: float) -> None:
    """IK both legs of a biped down to a fixed ground height. Used by standing actions."""
    for upper_name, lower_name in _leg_pairs(body):
        upper = body.rig.joints[upper_name]
        lower = body.rig.joints[lower_name]
        hip = frames_solved[upper_name].origin
        target = Vec2(hip.x, foot_y)
        wu, ll = solve_two_bone(hip, target, upper.length, lower.length, bend_positive=True)
        parent = frames_solved[upper.parent].angle if upper.parent else pose.root_angle
        pose.angles[upper_name] = wu - parent - upper.rest_angle
        pose.angles[lower_name] = ll - lower.rest_angle


def _leg_pairs(body: Body) -> list[tuple[str, str]]:
    """(upper, lower) for each contact leg — works for any rig, not just the biped."""
    pairs = []
    for name, j in body.rig.joints.items():
        if j.contact and j.parent is not None:
            pairs.append((j.parent, name))
    return pairs


def jump(
    body: Body,
    *,
    ground_y: float,
    x: float = 0.0,
    height: float = 140.0,
    distance: float = 0.0,
    frames: int = 36,
    anticip: float = 0.22,
    land: float = 0.16,
) -> PoseFn:
    """A jump: anticipation, launch, an arc through the air, a squash landing.

    This is the canonical demonstration of the principles working together — three
    of the twelve in one move (anticipation, arcs, squash-and-stretch) — and the
    single most-requested thing a stick figure should be able to do.

    The squash is done honestly for a cut-out rig: the *legs bend*. A deep crouch
    reads as a squash and full extension at launch reads as a stretch, which is how
    real cut-out animation fakes volume without a deformable mesh. `anticip` and
    `land` are the fractions of the cycle spent winding up and absorbing impact.
    """
    hip = body.hip_height
    launch = anticip
    touchdown = 1.0 - land

    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)

        if p < launch:
            # Anticipation: sink into a crouch. `anticipate` dips below zero first,
            # which is the tiny counter-settle before the spring.
            s = anticipate(p / launch)
            rise = -0.28 * hip * s
            gx = x
        elif p < touchdown:
            # Airborne: a parabolic arc. Peak height at the midpoint of the flight.
            s = (p - launch) / (touchdown - launch)
            rise = height * 4.0 * s * (1.0 - s) + 0.02 * hip
            gx = x + distance * ease_in_out(s)
        else:
            # Landing: absorb into a crouch, then ease back up to standing.
            s = (p - touchdown) / (1.0 - touchdown)
            rise = -0.24 * hip * math.sin(math.pi * s) * (1.0 - ease_out(s) * 0.4)
            gx = x + distance

        pose = Pose(root=Vec2(gx, ground_y - hip - rise), root_angle=0.0)

        # Secondary: arms lift through the launch and reach up at the apex — a real
        # jump throws its arms up.
        arm_phase = clamp((p - launch * 0.5) / max(touchdown - launch * 0.5, 1e-6), 0, 1)
        arm = -60.0 * math.sin(math.pi * arm_phase)
        for j in ("arm_upper", "arm_upper_far"):
            if j in body.rig.joints:
                pose.angles[j] = arm

        frames_solved = body.rig.solve(pose)

        airborne = launch <= p < touchdown and rise > 0.12 * hip
        if airborne:
            # Tuck the feet up under the body.
            tuck = min((rise) / max(height, 1e-6), 1.0)
            for upper_name, lower_name in _leg_pairs(body):
                pose.angles[upper_name] = -40.0 * tuck
                pose.angles[lower_name] = 80.0 * tuck
        else:
            _ground_legs(body, pose, frames_solved, ground_y)

        return pose

    return pose_fn


def idle(body: Body, *, ground_y: float, x: float = 0.0, cycle_frames: float = 48.0) -> PoseFn:
    """A living hold: a slow breathing rise and fall, feet planted.

    A character that is perfectly still reads as dead. The smallest amount of life —
    a breath — is the difference between a pause and a freeze.
    """
    hip = body.hip_height
    amp = 0.02 * hip

    def pose_fn(t: float) -> Pose:
        rise = amp * math.sin(2.0 * math.pi * t / cycle_frames)
        pose = Pose(root=Vec2(x, ground_y - hip - rise), root_angle=0.0)
        if "spine" in body.rig.joints:
            pose.angles["spine"] = 1.5 * math.sin(2.0 * math.pi * t / cycle_frames)
        frames_solved = body.rig.solve(pose)
        _ground_legs(body, pose, frames_solved, ground_y)
        return pose

    return pose_fn


def wave(
    body: Body,
    *,
    ground_y: float,
    x: float = 0.0,
    arm: str = "arm_upper",
    forearm: str = "arm_lower",
    cycles: float = 3.0,
    frames: int = 48,
) -> PoseFn:
    """Stand and wave: the near arm raises and the forearm oscillates.

    Anticipation on the way up (the arm swings back slightly first), then a
    sinusoidal wave — a gesture, i.e. a secondary action the character does on
    purpose, as distinct from an involuntary follow-through.
    """
    base = idle(body, ground_y=ground_y, x=x)
    have_arm = arm in body.rig.joints and forearm in body.rig.joints

    def pose_fn(t: float) -> Pose:
        pose = base(t)
        if have_arm:
            p = clamp(t / frames, 0.0, 1.0)
            up = clamp(p / 0.25, 0, 1)  # arm reaches full height in the first quarter
            raise_amt = ease_out(up) - 0.12 * math.sin(math.pi * up)  # tiny anticipation dip
            pose.angles[arm] = -150.0 * raise_amt
            if p > 0.2:
                pose.angles[forearm] = 30.0 * math.sin(2.0 * math.pi * cycles * (p - 0.2))
        return pose

    return pose_fn


def trail(
    pose_fn: PoseFn,
    body: Body,
    chain: list[str],
    *,
    lag: float = 2.5,
    damping: float = 0.7,
    swing: float = 26.0,
) -> PoseFn:
    """Wrap a pose function so a loose chain (tail, cape, ear, hair) follows through.

    Follow-through and overlapping action: when the body moves, the tip of a loose
    chain lags; when the body stops, the chain keeps swinging and settles. Each
    joint down the chain lags a little more than the one above it (that stagger is
    the *overlapping* part), and the swing is driven by how fast the base is moving,
    so it is silent when the character is still and whips when it darts.

    `lag` is how many frames the motion trails; `damping` how quickly it settles;
    `swing` the maximum deflection in degrees.
    """
    def wrapped(t: float) -> Pose:
        pose = pose_fn(t)
        base_now = pose.root.x
        base_past = pose_fn(max(t - lag, 0.0)).root.x
        velocity = base_now - base_past  # how fast the body moved over the lag window

        for depth, joint in enumerate(chain):
            if joint not in body.rig.joints:
                continue
            # Each link trails the one above it: an extra half-frame of lag per link.
            local_lag = lag * (1.0 + depth * 0.5)
            past = pose_fn(max(t - local_lag, 0.0)).root.x
            v = base_now - past
            # Deflect opposite to travel (the tail streams *behind* the motion),
            # damped by depth so the tip moves most.
            deflect = clamp(-v * 0.4, -swing, swing) * (damping ** depth) * (1.0 + depth * 0.25)
            pose.angles[joint] = pose.angles.get(joint, 0.0) + deflect
            _ = velocity  # base velocity kept for callers that want the read
        return pose

    return wrapped


def sequence(*segments: tuple[PoseFn, int]) -> PoseFn:
    """Play pose functions back to back: [(action, frames), ...].

    Cartoon acting is one beat after another — crouch, then jump, then wave. This
    stitches actions into a timeline, each running in its own local frame count.
    """
    if not segments:
        raise ValueError("sequence needs at least one (pose_fn, frames)")
    bounds = []
    acc = 0
    for fn, n in segments:
        bounds.append((acc, acc + n, fn))
        acc += n

    def pose_fn(t: float) -> Pose:
        for start, end, fn in bounds:
            if t < end or end == acc:
                return fn(t - start)
        return bounds[-1][2](t - bounds[-1][0])

    return pose_fn
