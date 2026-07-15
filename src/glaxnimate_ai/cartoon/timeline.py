"""The Timeline IR: an animation, sampled into plain data.

This is the seam the v2 architecture pivots on. v1 passed Python closures
(`pose_fn`) around — which meant the critic could only inspect things that came
through one registration path, nothing was serializable, and the keyframe reducer
had nothing to chew on. A `Timeline` is the same information as pure data: for
every node, its world-space origin, tip and angle on every frame, plus the small
amount of metadata the critic needs (masses, lengths, which nodes touch ground).

Everything downstream consumes this one structure:

* the **critic** (`feedback/lint.py`, `feedback/diagnose.py`) — so a bouncing
  ball, a prop, or a character loaded from a file is exactly as checkable as a
  rig built in-process;
* the **keyframe reducer** (`engine/reduce.py`) — which fits sparse bezier keys
  to these dense samples;
* the **renderer/baker** — which needs the same world transforms.

Build one with `from_pose_fn` (rigs) or `from_samples` (ball/wheel/leaf motion).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .geometry import Vec2
from .presets import Body
from .rig import Pose

__all__ = ["NodeTrack", "Timeline", "from_pose_fn", "from_samples"]


@dataclass(slots=True)
class NodeTrack:
    """One node's world-space motion, densely sampled.

    `contact` marks nodes the ground checks apply to. `check_slip` is separate on
    purpose: a planted foot must be world-stationary (slip is the cardinal sin),
    but a bouncing ball legitimately travels *through* its ground contacts — same
    ground rules, different slip rules.
    """

    origin: list[Vec2]
    tip: list[Vec2]
    angle: list[float]
    length: float = 0.0
    mass: float = 0.0
    contact: bool = False
    check_slip: bool = False


@dataclass(slots=True)
class Timeline:
    """A whole animation as data. The critic and the reducer both read this."""

    frames: int
    nodes: dict[str, NodeTrack] = field(default_factory=dict)
    #: Name of the body/root node, for spacing and balance checks.
    root: str | None = None
    #: (upper, lower) joint pairs for the IK over-extension check.
    limbs: list[tuple[str, str]] = field(default_factory=list)
    #: Longest leg (upper+lower), used to scale posture thresholds.
    leg_length: float = 0.0

    @property
    def contacts(self) -> list[str]:
        return [n for n, t in self.nodes.items() if t.contact]


def from_pose_fn(
    body: Body,
    pose_fn: Callable[[float], Pose],
    *,
    frames: int,
    limbs: Iterable | None = None,
) -> Timeline:
    """Sample a rig-driven animation into a Timeline.

    `limbs` accepts gait `Limb` objects or plain (upper, lower) name pairs; they
    feed the over-extension check. Sampling errors propagate — the caller decides
    whether a broken pose is fatal or a lint finding (`lint_rig` reports it).
    """
    rig = body.rig
    tl = Timeline(frames=frames, root=rig.root_name, leg_length=body.leg_length)

    for name in rig.joints:
        j = rig.joints[name]
        tl.nodes[name] = NodeTrack(
            origin=[], tip=[], angle=[],
            length=j.length, mass=j.mass,
            contact=j.contact or j.rolling,
            check_slip=j.contact or j.rolling,
        )

    for f in range(frames + 1):
        world = rig.solve(pose_fn(float(f)))
        for name, jf in world.items():
            t = tl.nodes[name]
            t.origin.append(jf.origin)
            t.tip.append(jf.tip)
            t.angle.append(jf.angle)

    if limbs:
        for limb in limbs:
            pair = (limb.upper, limb.lower) if hasattr(limb, "upper") else tuple(limb)
            tl.limbs.append(pair)  # type: ignore[arg-type]

    return tl


def from_samples(name: str, samples, *, radius: float = 0.0) -> Timeline:
    """A non-rig object (ball, wheel, leaf) as a Timeline.

    The node's *tip* is its lowest point — centre plus radius scaled by the squash
    (a squashed ball is shallower than a round one) — which is what the ground
    checks care about. `check_slip` stays off: a ball moves through its contacts.
    """
    track = NodeTrack(
        origin=[], tip=[], angle=[],
        length=radius, contact=radius > 0, check_slip=False,
    )
    frames = 0
    for smp in samples:
        frames = max(frames, smp.frame)
        track.origin.append(smp.pos)
        track.tip.append(smp.pos + Vec2(0.0, radius * smp.scale.y))
        track.angle.append(smp.angle)

    return Timeline(frames=frames, nodes={name: track}, root=name)
