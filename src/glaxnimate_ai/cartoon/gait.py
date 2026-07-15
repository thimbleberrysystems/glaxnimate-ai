"""One locomotion engine for every creature.

A gait is a **phase table**: N limbs, each offset around a unit cycle, each
running the same stance/swing curve. A biped walk is two limbs half a cycle
apart; a quadruped walk is four limbs at quarter-cycle offsets; a trot is the
same four limbs in diagonal pairs; a centipede is forty. The maths does not
change — only the table and the joint graph do.

That is the whole reason this project can claim to "animate anything" rather
than "animate people".

The load-bearing idea is in `foot_target`: during stance the foot is pinned to a
fixed world position. Not eased toward one, not approximately — *pinned*. The
body moves; the foot does not. Contact slip is therefore zero by construction
rather than by tuning, and the linter's job is to confirm that nothing else in
the pipeline broke it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Vec2
from .principles import ease_in_out
from .rig import Pose, Rig, solve_two_bone

__all__ = ["Limb", "Swing", "Gait", "gait_phases", "GAIT_TABLE", "pose_at", "foot_target",
           "reach_needed", "body_pose", "is_planted"]


@dataclass(slots=True)
class Limb:
    """A two-bone leg: upper bone, lower bone, and where it sits in the cycle."""

    upper: str
    lower: str
    phase: float = 0.0
    #: Which way the knee folds. A knee and an elbow differ by this flag alone —
    #: as do a horse's hind leg and its foreleg.
    bend_positive: bool = True
    #: Horizontal offset of this limb's attachment from the rig root, at rest.
    #:
    #: A quadruped's forelegs hang off the shoulders, well forward of the pelvis.
    #: Plant every foot relative to the *root* and the front feet get told to land
    #: under the hips — a reach they cannot make. Each limb must step under its
    #: own attachment. A biped hides this bug entirely (both legs attach at the
    #: root), which is exactly why the quadruped is in the test suite.
    hip_offset: float = 0.0


@dataclass(slots=True)
class Swing:
    """Secondary motion driven by the gait cycle: arms, tail, ears, head.

    Arms counter-swinging the legs is not decoration — a walk without it reads
    as a shop mannequin being pushed along.
    """

    joint: str
    phase: float = 0.0
    amplitude: float = 20.0


@dataclass(slots=True)
class Gait:
    limbs: list[Limb]
    swings: list[Swing] = field(default_factory=list)

    cycle_frames: float = 24.0
    #: Distance the body travels in one full cycle. Everything else derives from
    #: this, which is why the feet cannot slip: speed is not an independent knob.
    stride: float = 120.0
    #: Fraction of the cycle each limb spends on the ground. > 0.5 means there is
    #: always a foot down (a walk); < 0.5 means there is an airborne phase (a run).
    duty: float = 0.62
    lift: float = 30.0
    bob: float = 6.0
    lean: float = 0.0
    #: How high the hips ride, in absolute units. Set by `make_gait` from the body
    #: and the gait's crouch: a galloping animal *gathers* — it rides lower than a
    #: walking one — which is both physically true and what gives a fast gait's
    #: legs the reach margin they need. 0 means "caller supplies hip_height".
    ride_height: float = 0.0

    @property
    def speed(self) -> float:
        """Body units per frame. Derived — never set independently of stride."""
        return self.stride / self.cycle_frames


#: Phase offsets per limb count. This table *is* the difference between animals.
GAIT_TABLE: dict[str, dict[int, list[float]]] = {
    # Two feet, half a cycle apart.
    "walk": {2: [0.0, 0.5], 4: [0.0, 0.5, 0.25, 0.75]},
    "run": {2: [0.0, 0.5], 4: [0.0, 0.5, 0.25, 0.75]},
    # Diagonal pairs move together: front-left with hind-right.
    "trot": {4: [0.0, 0.5, 0.5, 0.0]},
    # Front pair, then hind pair — the rocking-horse gait.
    "gallop": {4: [0.0, 0.1, 0.5, 0.6]},
    "bound": {4: [0.0, 0.0, 0.5, 0.5]},
    # Hop: everything together.
    "hop": {2: [0.0, 0.0], 4: [0.0, 0.0, 0.0, 0.0]},
}

#: Per-gait defaults. Stride, lift and bob are **fractions of hip height**, not
#: pixels: a dachshund and a giraffe take very different strides, and the number
#: that is actually constant across animals is the ratio. `presets.make_gait`
#: turns these into absolute units for a given body.
# Stride and lift are tuned so every gait *reaches* on its default body — the
# reach guard in `make_gait` now enforces this, and run/gallop were originally set
# a touch too aggressive (they over-extended by 5-10px). A faster look comes from a
# shorter cycle_frames, not a longer stride: speed = stride / cycle_frames, and only
# stride is bounded by leg length.
# `crouch` lowers the hips for fast gaits. A galloping animal gathers its body
# closer to the ground than a standing one — this is real locomotion, and it is
# also what gives a fast gait's legs the reach margin they need. Without it, the
# reach is dominated by hip height and a gallop cannot fit on any leg (the whole
# rig scales together, so longer legs do not help — only a lower ride does).
# keys: duty, stride, lift, bob (fractions of hip height), lean (deg), crouch.
GAIT_DEFAULTS: dict[str, dict[str, float]] = {
    "walk":   {"duty": 0.62, "stride": 0.85, "lift": 0.18, "bob": 0.04, "lean": 0.0, "crouch": 1.00},  # noqa: E501
    "run":    {"duty": 0.35, "stride": 1.08, "lift": 0.28, "bob": 0.08, "lean": -8.0, "crouch": 0.86},  # noqa: E501
    "trot":   {"duty": 0.45, "stride": 1.00, "lift": 0.22, "bob": 0.05, "lean": 0.0, "crouch": 0.96},  # noqa: E501
    "gallop": {"duty": 0.30, "stride": 1.20, "lift": 0.28, "bob": 0.08, "lean": -6.0, "crouch": 0.84},  # noqa: E501
    "bound":  {"duty": 0.35, "stride": 1.18, "lift": 0.30, "bob": 0.09, "lean": 0.0, "crouch": 0.86},  # noqa: E501
    "hop":    {"duty": 0.50, "stride": 0.76, "lift": 0.30, "bob": 0.11, "lean": 0.0, "crouch": 0.90},  # noqa: E501
}

#: Hips ride at this fraction of full leg length. A standing figure does *not*
#: lock its legs straight — and if it did, a foot planted even slightly ahead of
#: the hip would be out of reach and the IK would silently fall short. This one
#: number is the difference between a walk and a stilt-walk.
HIP_HEIGHT_RATIO = 0.85


def gait_phases(name: str, n_limbs: int) -> list[float]:
    """Phase offsets for `n_limbs` legs in the named gait."""
    try:
        table = GAIT_TABLE[name]
    except KeyError:
        raise ValueError(f"unknown gait {name!r}; have {sorted(GAIT_TABLE)}") from None
    if n_limbs in table:
        return list(table[n_limbs])
    # Unlisted limb counts (a centipede, a hexapod) get evenly spread phases —
    # a travelling wave, which is what many-legged things actually do.
    return [i / n_limbs for i in range(n_limbs)]


# --------------------------------------------------------------------- feet
def foot_target(gait: Gait, limb: Limb, t: float, ground_y: float, body_x0: float) -> Vec2:
    """Where this foot is, in world space, at time `t`.

    Stance: the foot is at a fixed plant position — literally constant, so it
    cannot slide. Swing: it arcs forward by exactly one stride, which lands it
    on the next plant position, so the cycle closes seamlessly.
    """
    g = t / gait.cycle_frames + limb.phase
    n = math.floor(g)
    p = g - n

    # Chosen so that at mid-stance the foot sits directly under *its own* hip —
    # hence hip_offset. That is what makes the plant look supported rather than
    # staggered, and what stops a quadruped's forelegs reaching for the pelvis.
    x0 = body_x0 + limb.hip_offset + gait.stride * (gait.duty / 2.0 - limb.phase)
    plant = x0 + n * gait.stride

    if p < gait.duty:
        return Vec2(plant, ground_y)  # planted. Exactly. This is the point.

    s = (p - gait.duty) / (1.0 - gait.duty)
    x = plant + gait.stride * ease_in_out(s)
    y = ground_y - gait.lift * math.sin(math.pi * s)
    return Vec2(x, y)


def is_planted(gait: Gait, limb: Limb, t: float) -> bool:
    g = t / gait.cycle_frames + limb.phase
    return (g - math.floor(g)) < gait.duty


def body_position(gait: Gait, t: float, ground_y: float, body_x0: float, hip_height: float) -> Vec2:
    """The hips: forward at constant speed, bobbing at twice the cycle rate.

    Twice, because the body rises over each supporting leg — once per step, and
    there are two steps per cycle. Getting this frequency wrong is the classic
    tell of a hand-faked walk.
    """
    x = body_x0 + gait.speed * t
    y = ground_y - hip_height - gait.bob * math.sin(4.0 * math.pi * t / gait.cycle_frames)
    return Vec2(x, y)


# --------------------------------------------------------------------- posing
def body_pose(gait: Gait, t: float, ground_y: float, body_x0: float, hip_height: float) -> Pose:
    """The pose before the legs are solved: body placement + secondary motion.

    Split out from `pose_at` so the reach check can find where each hip *is*
    without triggering the leg IK (which clamps, and so would hide the very
    over-extension we are trying to measure).
    """
    pose = Pose(
        root=body_position(gait, t, ground_y, body_x0, hip_height),
        root_angle=gait.lean,
    )
    cyc = t / gait.cycle_frames
    for s in gait.swings:
        pose.angles[s.joint] = s.amplitude * math.sin(2.0 * math.pi * (cyc + s.phase))
    return pose


def _ride(gait: Gait, hip_height: float | None) -> float:
    """Resolve the hip height: an explicit override wins, else the gait's own."""
    if hip_height is not None:
        return hip_height
    if gait.ride_height > 0:
        return gait.ride_height
    raise ValueError("no hip height: pass hip_height, or build the gait with make_gait")


def reach_needed(rig: Rig, gait: Gait, hip_height: float | None = None) -> dict[str, float]:
    """Largest hip→foot-target distance for each limb, sampled over one whole cycle.

    This is the reach the IK is actually asked for. If it exceeds a leg's length,
    that leg straightens and falls short and the foot drags — the quiet skating
    artefact `lint.ik_overextension` catches per frame.

    Sampling the **whole cycle** is the entire point. The foot reaches farthest
    during swing, not during stance, so a stance-only estimate (what the old guard
    used) passes a gallop at construction and then fails at lint. Distances are
    translation-invariant, so the sample origin is arbitrary.
    """
    h = _ride(gait, hip_height)
    n = max(int(gait.cycle_frames), 8)
    worst: dict[str, float] = {}
    for i in range(n):
        t = i / n * gait.cycle_frames
        frames = rig.solve(body_pose(gait, t, 0.0, 0.0, h))
        for limb in gait.limbs:
            hip = frames[limb.upper].origin
            d = hip.distance_to(foot_target(gait, limb, t, 0.0, 0.0))
            worst[limb.upper] = max(worst.get(limb.upper, 0.0), d)
    return worst


def pose_at(
    rig: Rig,
    gait: Gait,
    t: float,
    *,
    ground_y: float,
    body_x0: float = 0.0,
    hip_height: float | None = None,
) -> Pose:
    """The rig's full pose at time `t`.

    Order matters: the body is solved first by forward kinematics, which fixes
    where each hip *is*; only then are the legs solved backwards from their foot
    targets. Doing it the other way round would make the hips depend on the legs
    and the feet would drift.

    `hip_height` defaults to the gait's own `ride_height` (set by `make_gait`), so
    a crouched gallop stays crouched without the caller having to remember.
    """
    pose = body_pose(gait, t, ground_y, body_x0, _ride(gait, hip_height))

    # Pass 1: FK. The hips do not depend on the legs, so this settles them.
    frames = rig.solve(pose)

    # Pass 2: IK each leg onto its foot target.
    for limb in gait.limbs:
        upper = rig.joints[limb.upper]
        lower = rig.joints[limb.lower]
        hip = frames[limb.upper].origin
        target = foot_target(gait, limb, t, ground_y, body_x0)

        world_upper, local_lower = solve_two_bone(
            hip, target, upper.length, lower.length, bend_positive=limb.bend_positive
        )

        parent_angle = frames[upper.parent].angle if upper.parent else pose.root_angle
        pose.angles[limb.upper] = world_upper - parent_angle - upper.rest_angle
        pose.angles[limb.lower] = local_lower - lower.rest_angle

    return pose
