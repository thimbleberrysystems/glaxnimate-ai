"""Tier 0 of the critic stack: *is it broken?*

Free, instant, no LLM, no pixels. Most animation faults are not matters of taste —
they are measurable, and this catches them before anything is rendered.

v2: the checks run on the **Timeline IR** (`cartoon/timeline.py`) — plain sampled
data — not on Python closures. That one change means *everything* on stage is
checkable: rig characters, bouncing balls, wheels, and eventually scenes loaded
from disk. `lint_rig` remains as a thin wrapper for the rig-and-pose-fn call sites.

Everything here is generic over the rig. "Contact slip" is not a check about feet;
it is a check about **contact points**, so the one rule catches a sliding foot, a
skating paw and a wheel spinning on a stationary car.

The most valuable check is `ik_overextension`. A limb that cannot reach its target
does not raise — the solver just straightens and falls short, and the foot slides a
little. That is a *quiet* bug: the animation still plays, still exports, and simply
looks subtly wrong forever. Naming it here turns it into a loud one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from ..cartoon import timeline as tlmod
from ..cartoon.gait import Limb
from ..cartoon.presets import Body
from ..cartoon.rig import Pose
from ..cartoon.timeline import Timeline

__all__ = ["Issue", "Report", "lint_timeline", "lint_rig", "lint_object"]


@dataclass(slots=True)
class Issue:
    check: str
    severity: str  # "error" | "warning"
    detail: str
    frame: int | None = None
    value: float | None = None

    def __str__(self) -> str:
        at = f" @f{self.frame}" if self.frame is not None else ""
        v = f" ({self.value:.2f})" if self.value is not None else ""
        return f"[{self.severity}] {self.check}{at}: {self.detail}{v}"


@dataclass(slots=True)
class Report:
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def add(self, *a, **kw) -> None:
        self.issues.append(Issue(*a, **kw))

    def format(self, limit: int = 12) -> str:
        if not self.issues:
            return "clean: no issues found"
        errs = [i for i in self.issues if i.severity == "error"]
        warns = [i for i in self.issues if i.severity == "warning"]
        head = f"{len(errs)} error(s), {len(warns)} warning(s)"
        # Collapse repeats: one bad frame usually means fifty bad frames, and the
        # model does not need to read all fifty.
        seen: dict[str, int] = {}
        lines: list[str] = []
        for i in (*errs, *warns):
            seen[i.check] = seen.get(i.check, 0) + 1
            if seen[i.check] <= 3 and len(lines) < limit:
                lines.append("  " + str(i))
        for check, n in seen.items():
            if n > 3:
                lines.append(f"  ... {check}: {n} occurrences total")
        return head + "\n" + "\n".join(lines)


def lint_timeline(
    tl: Timeline,
    *,
    ground_y: float,
    canvas: tuple[int, int] | None = None,
    contact_tol: float = 1.5,
    slip_tol: float = 0.5,
    strobe_px: float = 90.0,
) -> Report:
    """Check sampled animation data for the faults that are arithmetic, not opinion."""
    rep = Report()

    # --- non-finite values. Poison everything downstream; check first.
    for name, t in tl.nodes.items():
        for f, (o, a) in enumerate(zip(t.origin, t.angle, strict=True)):
            if not all(map(math.isfinite, (o.x, o.y, a))):
                rep.add("nan", "error", f"{name} is not finite", frame=f)

    contacts = tl.contacts
    if not contacts:
        rep.add("no_contacts", "warning",
                "nothing declares a contact point; ground checks are limited")

    for c in contacts:
        t = tl.nodes[c]
        prev_x: float | None = None
        for f, p in enumerate(t.tip):
            planted = abs(p.y - ground_y) <= contact_tol

            # --- contact slip: a planted contact must not move. The cardinal sin.
            # Skipped for nodes that legitimately travel through their contacts
            # (a bouncing ball), which declare check_slip=False.
            if t.check_slip:
                if planted:
                    if prev_x is not None:
                        d = abs(p.x - prev_x)
                        if d > slip_tol:
                            rep.add(
                                "contact_slip", "error",
                                f"{c} slid while planted (the character is skating)",
                                frame=f, value=d,
                            )
                    prev_x = p.x
                else:
                    prev_x = None

            # --- ground penetration applies to every contact node.
            if p.y > ground_y + contact_tol:
                rep.add(
                    "ground_penetration", "error", f"{c} is below the ground line",
                    frame=f, value=p.y - ground_y,
                )

    # --- IK over-extension: the quiet one. The limb could not reach, so it
    # straightened and fell short, and the foot dragged.
    for upper, lower in tl.limbs:
        if upper not in tl.nodes or lower not in tl.nodes:
            continue
        reach = tl.nodes[upper].length + tl.nodes[lower].length
        for f, (hip, foot) in enumerate(
            zip(tl.nodes[upper].origin, tl.nodes[lower].tip, strict=True)
        ):
            d = hip.distance_to(foot)
            if d > reach - 1e-6:
                rep.add(
                    "ik_overextension", "error",
                    f"{upper} is fully extended and cannot reach its target "
                    f"(needs {d:.1f}, has {reach:.1f})",
                    frame=f, value=d - reach,
                )

    # --- strobing: motion so fast between frames that the eye loses the object.
    for name, t in tl.nodes.items():
        for f in range(1, len(t.origin)):
            d = t.origin[f].distance_to(t.origin[f - 1])
            if d > strobe_px:
                rep.add(
                    "strobing", "warning",
                    f"{name} jumps {d:.0f}px in one frame; it will read as a flicker",
                    frame=f, value=d,
                )

    # --- out of canvas
    if canvas:
        w, h = canvas
        n_frames = max((len(t.origin) for t in tl.nodes.values()), default=0)
        for f in range(n_frames):
            for name, t in tl.nodes.items():
                if f >= len(t.origin):
                    continue
                o = t.origin[f]
                if not (-1 <= o.x <= w + 1 and -1 <= o.y <= h + 1):
                    rep.add("out_of_canvas", "warning",
                            f"{name} has left the frame", frame=f)
                    break  # one report per frame is plenty

    return rep


# ------------------------------------------------------------------ wrappers
def lint_rig(
    body: Body,
    pose_fn: Callable[[float], Pose],
    *,
    frames: int,
    ground_y: float,
    limbs: list[Limb] | None = None,
    canvas: tuple[int, int] | None = None,
    contact_tol: float = 1.5,
    slip_tol: float = 0.5,
    strobe_px: float = 90.0,
) -> Report:
    """Sample a rig animation and lint it. A broken pose is itself the finding."""
    try:
        tl = tlmod.from_pose_fn(body, pose_fn, frames=frames, limbs=limbs)
    except Exception as e:  # noqa: BLE001 - a raising pose_fn is a lint result
        rep = Report()
        rep.add("pose_error", "error", f"pose_fn raised: {e}")
        return rep
    return lint_timeline(
        tl, ground_y=ground_y, canvas=canvas,
        contact_tol=contact_tol, slip_tol=slip_tol, strobe_px=strobe_px,
    )


def lint_object(
    name: str,
    samples,
    *,
    ground_y: float,
    radius: float = 0.0,
    canvas: tuple[int, int] | None = None,
    **kw,
) -> Report:
    """Lint a non-rig object (a bounce, a roll, a drift) — new in v2.

    v1 could not check these at all: the critic took pose functions, and a ball
    has none. Now a ball that sinks through the floor is caught like a foot is.
    """
    tl = tlmod.from_samples(name, samples, radius=radius)
    return lint_timeline(tl, ground_y=ground_y, canvas=canvas, **kw)
