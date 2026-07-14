"""A rig is a graph of joints. Everything articulated is one.

This is the generic core the whole library rests on: a biped, a quadruped, a
bird, a snake, a car with wheels and a tree with branches differ only in the
shape of the joint graph — never in the code that solves it.

Two solvers:

* **Forward kinematics** (`Rig.solve`) — given joint angles, where is everything?
  Used for the body, spine, head, tails, anything driven by rotation.
* **Inverse kinematics** (`solve_two_bone`) — given a *target*, what angles put
  the limb tip there? Used for legs.

Why legs are IK and not FK: if you rotate a thigh and shin forward and hope the
foot lands right, it lands wherever the maths puts it and slides along the
ground. If you instead decide where the foot goes and solve backwards, the
planted foot is world-stationary *by construction* and contact slip is zero.
That single choice is what makes a walk cycle read as walking rather than
skating, and it is why `feedback/lint.py` can assert slip == 0 rather than
merely measure it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Vec2, clamp

__all__ = ["Joint", "Rig", "JointFrame", "Pose", "solve_two_bone"]


@dataclass(slots=True)
class Joint:
    """One bone.

    `offset` is where this joint's origin sits in its parent's local frame. It
    defaults to the parent's tip, which is what you want for a limb chain
    (shin hangs off the end of the thigh); override it for things that branch
    from partway along a bone, like a shoulder off a spine.
    """

    name: str
    parent: str | None = None
    length: float = 0.0
    rest_angle: float = 0.0
    offset: Vec2 | None = None

    #: Touches the ground. The linter requires these to be world-stationary
    #: while planted — a foot, a paw, a hand doing a handstand.
    contact: bool = False

    #: Rolls along the ground (a wheel). No-slip couples spin to travel:
    #: distance == radius * angle. Checked by the linter just like a foot.
    rolling: bool = False
    radius: float = 0.0


@dataclass(frozen=True, slots=True)
class JointFrame:
    """Where a joint ended up, in world space."""

    origin: Vec2
    angle: float  # world, degrees
    tip: Vec2


@dataclass(slots=True)
class Pose:
    """A rig's state at one instant: root placement + local joint angles."""

    root: Vec2 = field(default_factory=Vec2)
    root_angle: float = 0.0
    angles: dict[str, float] = field(default_factory=dict)

    def copy(self) -> Pose:
        return Pose(self.root, self.root_angle, dict(self.angles))


class Rig:
    """An ordered joint graph. Parents are always solved before children."""

    def __init__(self, joints: list[Joint]) -> None:
        self.joints: dict[str, Joint] = {j.name: j for j in joints}
        if len(self.joints) != len(joints):
            raise ValueError("duplicate joint names")

        roots = [j.name for j in joints if j.parent is None]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root joint, got {roots}")
        self.root_name = roots[0]

        self._order = self._topological_order(joints)

    def _topological_order(self, joints: list[Joint]) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(name: str, trail: tuple[str, ...]) -> None:
            if name in seen:
                return
            if name in trail:
                raise ValueError(f"cycle in rig: {' -> '.join((*trail, name))}")
            j = self.joints[name]
            if j.parent is not None:
                if j.parent not in self.joints:
                    raise ValueError(f"joint {name!r} has unknown parent {j.parent!r}")
                visit(j.parent, (*trail, name))
            seen.add(name)
            order.append(name)

        for j in joints:
            visit(j.name, ())
        return order

    # ---------------------------------------------------------------- queries
    def chain(self, name: str) -> list[str]:
        """Root-to-`name` path, for IK and for the linter's joint-integrity check."""
        out: list[str] = []
        cur: str | None = name
        while cur is not None:
            out.append(cur)
            cur = self.joints[cur].parent
        return list(reversed(out))

    @property
    def contacts(self) -> list[str]:
        return [j.name for j in self.joints.values() if j.contact or j.rolling]

    # ------------------------------------------------------------------- FK
    def solve(self, pose: Pose) -> dict[str, JointFrame]:
        """Forward kinematics: joint angles in, world positions out."""
        frames: dict[str, JointFrame] = {}

        for name in self._order:
            j = self.joints[name]
            local = pose.angles.get(name, 0.0) + j.rest_angle

            if j.parent is None:
                origin = pose.root
                angle = pose.root_angle + local
            else:
                p = frames[j.parent]
                pj = self.joints[j.parent]
                # Default attachment is the parent's tip — the natural thing for
                # a limb chain. An explicit offset branches off mid-bone instead.
                off = j.offset if j.offset is not None else Vec2(pj.length, 0.0)
                origin = p.origin + off.rotated(p.angle)
                angle = p.angle + local

            tip = origin + Vec2(j.length, 0.0).rotated(angle)
            frames[name] = JointFrame(origin, angle, tip)

        return frames


def solve_two_bone(
    root: Vec2,
    target: Vec2,
    l1: float,
    l2: float,
    *,
    bend_positive: bool = True,
) -> tuple[float, float]:
    """Place a two-bone limb's tip on `target`. Returns (world angle, local angle).

    Law of cosines. `bend_positive` picks which way the knee/elbow folds — the
    single flag that distinguishes a knee from an elbow, or a dog's foreleg from
    its hind leg.

    If the target is out of reach the limb straightens toward it rather than
    raising: a leg reaching too far should stretch, not explode. That keeps the
    solver total, so a gait never has to special-case its own geometry.
    """
    d = target - root
    dist = clamp(d.length(), 1e-6, l1 + l2 - 1e-9)

    # Interior angle at the root between bone 1 and the root->target line.
    cos_a = clamp((l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist), -1.0, 1.0)
    a = math.degrees(math.acos(cos_a))

    # Interior angle at the joint between the two bones.
    cos_b = clamp((l1 * l1 + l2 * l2 - dist * dist) / (2 * l1 * l2), -1.0, 1.0)
    b = math.degrees(math.acos(cos_b))

    sign = 1.0 if bend_positive else -1.0
    upper = d.angle() - sign * a
    lower = sign * (180.0 - b)  # local to the upper bone
    return upper, lower
