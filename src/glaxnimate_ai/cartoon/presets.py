"""Bodies. A human is one preset among many, never the design centre.

Screen coordinates: +x right, +y **down**. So "up" is -90 degrees. This trips
everyone up once; it is called out here so it only happens once.

Limb order is a contract: **[hind_left, hind_right, fore_left, fore_right]** for
quadrupeds, **[left, right]** for bipeds. `gait.GAIT_TABLE` is written against
that order, which is what lets one phase table produce a correct lateral-sequence
walk *and* a correct diagonal trot without either knowing about the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .gait import GAIT_DEFAULTS, HIP_HEIGHT_RATIO, Gait, Limb, Swing, gait_phases
from .geometry import Vec2
from .rig import Joint, Pose, Rig

__all__ = ["Body", "Part", "biped", "human", "quadruped", "make_gait"]


@dataclass(slots=True)
class Part:
    """How one bone is *drawn*. Pure data — `engine/bake.py` does the drawing.

    Skinning is what turns a rig into a character. The rig was already correct as
    a stick figure; a torso being thicker than a forearm, a head being an ellipse
    rather than a rod, and a hand being a dot at the end of the arm are all it
    takes to read as a person instead of a diagram. None of it touches the maths.
    """

    width: float = 14.0
    color: str = "#33333c"
    #: Draw an ellipse at the bone's tip instead of a capsule — a head.
    head: tuple[float, float] | None = None
    #: A dot at the bone's tip: a hand, a paw, a nose.
    tip: float = 0.0
    #: Painting order. Low numbers go behind, so a far arm sits under the torso.
    z: int = 0


@dataclass(slots=True)
class Body:
    """A rig plus the metadata a gait needs to drive it."""

    rig: Rig
    limbs: list[Limb]
    swings: list[Swing]
    #: Full extension of one leg (upper + lower bone).
    leg_length: float
    #: Joints to draw as thick strokes, in draw order.
    bones: list[str]
    #: How each bone is skinned. Missing joints fall back to a plain capsule.
    parts: dict[str, Part] = field(default_factory=dict)

    @property
    def hip_height(self) -> float:
        """Where the hips ride. Deliberately below full leg extension — see
        `gait.HIP_HEIGHT_RATIO`."""
        return self.leg_length * HIP_HEIGHT_RATIO


def biped(
    *,
    thigh: float = 80.0,
    shin: float = 80.0,
    spine: float = 70.0,
    arm: float = 45.0,
    forearm: float = 40.0,
    head: float = 34.0,
) -> Body:
    """Side-view two-legged figure. `human()` is this with default proportions."""
    joints = [
        Joint("hips", None, length=0.0),
        Joint("spine", "hips", length=spine, rest_angle=-90.0, offset=Vec2(0, 0), mass=45.0),
        Joint("head", "spine", length=head, rest_angle=0.0, mass=8.0),
        # Arms hang from the top of the spine (its tip = the shoulders).
        Joint("arm_upper", "spine", length=arm, rest_angle=180.0, mass=3.0),
        Joint("arm_lower", "arm_upper", length=forearm, rest_angle=0.0, mass=2.0),
        Joint("arm_upper_far", "spine", length=arm, rest_angle=180.0, mass=3.0),
        Joint("arm_lower_far", "arm_upper_far", length=forearm, rest_angle=0.0, mass=2.0),
    ]
    for side in ("l", "r"):
        joints += [
            Joint(f"thigh_{side}", "hips", length=thigh, rest_angle=90.0,
                  offset=Vec2(0, 0), mass=10.0),
            # The shin's tip is the foot: this is the ground contact the linter polices.
            Joint(f"shin_{side}", f"thigh_{side}", length=shin, rest_angle=0.0,
                  contact=True, mass=4.0),
            Joint(f"foot_{side}", f"shin_{side}", length=22.0, rest_angle=-90.0, mass=1.0),
        ]

    limbs = [
        Limb("thigh_l", "shin_l", bend_positive=True),
        Limb("thigh_r", "shin_r", bend_positive=True),
    ]
    # Arms counter-swing the legs: half a cycle out of phase with the same side.
    swings = [
        Swing("arm_upper", phase=0.5, amplitude=22.0),
        Swing("arm_upper_far", phase=0.0, amplitude=22.0),
        Swing("head", phase=0.0, amplitude=3.0),
    ]
    # Far side first, near side last: the limbs on the far side of the body must
    # paint *behind* the torso or the figure reads as flat.
    bones = [
        "arm_upper_far", "arm_lower_far",
        "thigh_r", "shin_r", "foot_r",
        "spine", "head",
        "thigh_l", "shin_l", "foot_l",
        "arm_upper", "arm_lower",
    ]

    SHIRT, TROUSER, SKIN, SHOE = "#4a6fa5", "#2f3542", "#f0c39a", "#22242c"
    FAR = "#3d5c8a"  # the far limbs sit in shadow, which is most of the depth cue
    parts = {
        "spine": Part(width=36, color=SHIRT),
        "head": Part(color=SKIN, head=(46, 52)),
        "arm_upper": Part(width=17, color=SHIRT),
        "arm_lower": Part(width=13, color=SKIN, tip=8),      # tip = the hand
        "arm_upper_far": Part(width=17, color=FAR),
        "arm_lower_far": Part(width=13, color=FAR, tip=8),
        "thigh_l": Part(width=24, color=TROUSER),
        "shin_l": Part(width=18, color=TROUSER),
        "foot_l": Part(width=12, color=SHOE, tip=6),
        "thigh_r": Part(width=24, color=FAR),
        "shin_r": Part(width=18, color=FAR),
        "foot_r": Part(width=12, color="#191b21", tip=6),
    }
    return Body(Rig(joints), limbs, swings, leg_length=thigh + shin, bones=bones, parts=parts)


def human(**kw) -> Body:
    return biped(**kw)


def quadruped(
    *,
    upper: float = 55.0,
    lower: float = 55.0,
    body: float = 120.0,
    neck: float = 42.0,
    head: float = 20.0,   # short: the head is the *ellipse*, not the bone
    tail: float = 58.0,
) -> Body:
    """Side-view four-legged animal: dog, cat, horse, deer.

    Forelegs bend the opposite way to hind legs — an elbow, not a knee. That one
    flag is most of what separates a convincing dog from a table with fur.
    """
    joints = [
        Joint("pelvis", None, length=0.0),
        Joint("spine", "pelvis", length=body, rest_angle=0.0, offset=Vec2(0, 0)),
        Joint("neck", "spine", length=neck, rest_angle=-50.0),
        Joint("head", "neck", length=head, rest_angle=40.0),
        Joint("tail", "pelvis", length=tail, rest_angle=200.0, offset=Vec2(0, 0)),
    ]
    for side in ("l", "r"):
        joints += [
            # Hind legs hang off the pelvis.
            Joint(f"hind_upper_{side}", "pelvis", length=upper,
                  rest_angle=90.0, offset=Vec2(0, 0)),
            Joint(f"hind_lower_{side}", f"hind_upper_{side}", length=lower,
                  rest_angle=0.0, contact=True),
            # Forelegs hang off the shoulders (the spine's tip).
            Joint(f"fore_upper_{side}", "spine", length=upper, rest_angle=90.0),
            Joint(f"fore_lower_{side}", f"fore_upper_{side}", length=lower,
                  rest_angle=0.0, contact=True),
        ]

    # Contract: hind_l, hind_r, fore_l, fore_r.
    limbs = [
        Limb("hind_upper_l", "hind_lower_l", bend_positive=True),
        Limb("hind_upper_r", "hind_lower_r", bend_positive=True),
        Limb("fore_upper_l", "fore_lower_l", bend_positive=False),
        Limb("fore_upper_r", "fore_lower_r", bend_positive=False),
    ]
    swings = [
        Swing("tail", phase=0.0, amplitude=10.0),
        Swing("neck", phase=0.25, amplitude=4.0),
    ]
    bones = [
        "hind_upper_r", "hind_lower_r", "fore_upper_r", "fore_lower_r",
        "tail", "spine", "neck", "head",
        "hind_upper_l", "hind_lower_l", "fore_upper_l", "fore_lower_l",
    ]

    COAT, FAR, NOSE = "#a5734a", "#8a5f3d", "#33333c"
    parts = {
        "spine": Part(width=44, color=COAT),
        "neck": Part(width=26, color=COAT),
        "head": Part(width=22, color=COAT, head=(42, 34)),
        "tail": Part(width=10, color=COAT, tip=7),
        "hind_upper_l": Part(width=20, color=COAT),
        "hind_lower_l": Part(width=13, color=COAT, tip=7),
        "fore_upper_l": Part(width=18, color=COAT),
        "fore_lower_l": Part(width=12, color=COAT, tip=7),
        "hind_upper_r": Part(width=20, color=FAR),
        "hind_lower_r": Part(width=13, color=FAR, tip=7),
        "fore_upper_r": Part(width=18, color=FAR),
        "fore_lower_r": Part(width=12, color=FAR, tip=7),
    }
    del NOSE
    return Body(Rig(joints), limbs, swings, leg_length=upper + lower, bones=bones, parts=parts)


def make_gait(body: Body, name: str = "walk", **overrides) -> Gait:
    """Bind a named gait to a body: phases from the table, proportions from the body.

    Stride, lift and bob are stored in `GAIT_DEFAULTS` as fractions of hip height
    and scaled here, so the same gait reads correctly on a terrier and on a horse.

    Then it checks the legs can actually *reach*. A leg that cannot reach its
    target does not error — the IK just straightens and falls short, and the foot
    slides. Catching it here turns a subtle skating artefact into a loud failure.
    """
    phases = gait_phases(name, len(body.limbs))

    # Where each limb actually attaches, measured at rest. A quadruped's forelegs
    # sit a whole body-length forward of its hind legs; each must step under
    # itself, not under the root.
    rest = body.rig.solve(Pose())
    limbs = [
        Limb(
            limb.upper,
            limb.lower,
            phase=p,
            bend_positive=limb.bend_positive,
            hip_offset=rest[limb.upper].origin.x,
        )
        for limb, p in zip(body.limbs, phases, strict=True)
    ]

    d = dict(GAIT_DEFAULTS.get(name, GAIT_DEFAULTS["walk"]))
    h = body.hip_height
    params = {
        "duty": d["duty"],
        "lean": d["lean"],
        "stride": d["stride"] * h,
        "lift": d["lift"] * h,
        "bob": d["bob"] * h,
    }
    params.update(overrides)

    gait = Gait(limbs=limbs, swings=list(body.swings), **params)

    # Worst case: hip at the top of its bob, foot at the far end of its stance.
    reach = math.hypot(gait.stride * gait.duty / 2.0, h + gait.bob)
    leg = body.leg_length
    if reach > leg:
        raise ValueError(
            f"{name!r} needs a reach of {reach:.1f} but the leg is only {leg:.1f} long. "
            f"Shorten the stride, lower the bob, or lengthen the leg."
        )
    return gait
