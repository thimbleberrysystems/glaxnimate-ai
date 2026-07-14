"""Tier 0 of the critic stack: *is it broken?*

Free, instant, no LLM, no pixels. Most animation faults are not matters of taste —
they are measurable, and this catches them before anything is rendered.

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

from ..cartoon.gait import Limb
from ..cartoon.presets import Body
from ..cartoon.rig import Pose

__all__ = ["Issue", "Report", "lint_rig"]


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
    """Check a rig animation for the faults that are arithmetic, not opinion."""
    rep = Report()
    rig = body.rig

    world = []
    for f in range(frames + 1):
        try:
            world.append(rig.solve(pose_fn(float(f))))
        except Exception as e:  # noqa: BLE001 - a broken pose is itself the finding
            rep.add("pose_error", "error", f"pose_fn raised: {e}", frame=f)
            return rep

    # --- non-finite values. Poison everything downstream; check first.
    for f, fr in enumerate(world):
        for name, jf in fr.items():
            if not all(map(math.isfinite, (jf.origin.x, jf.origin.y, jf.angle))):
                rep.add("nan", "error", f"{name} is not finite", frame=f)

    contacts = rig.contacts
    if not contacts:
        rep.add("no_contacts", "warning", "rig declares no contact points; slip cannot be checked")

    # --- contact slip: a planted contact must not move. The cardinal sin.
    for c in contacts:
        prev_x: float | None = None
        for f, fr in enumerate(world):
            p = fr[c].tip
            planted = abs(p.y - ground_y) <= contact_tol
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

            # --- ground penetration
            if p.y > ground_y + contact_tol:
                rep.add(
                    "ground_penetration", "error", f"{c} is below the ground line",
                    frame=f, value=p.y - ground_y,
                )

    # --- IK over-extension: the quiet one. The limb could not reach, so it
    # straightened and fell short, and the foot dragged.
    if limbs:
        for limb in limbs:
            reach = rig.joints[limb.upper].length + rig.joints[limb.lower].length
            for f, fr in enumerate(world):
                d = fr[limb.upper].origin.distance_to(fr[limb.lower].tip)
                if d > reach - 1e-6:
                    rep.add(
                        "ik_overextension", "error",
                        f"{limb.upper} is fully extended and cannot reach its target "
                        f"(needs {d:.1f}, has {reach:.1f})",
                        frame=f, value=d - reach,
                    )

    # --- strobing: motion so fast between frames that the eye loses the object.
    for name in rig.joints:
        for f in range(1, len(world)):
            d = world[f][name].origin.distance_to(world[f - 1][name].origin)
            if d > strobe_px:
                rep.add(
                    "strobing", "warning",
                    f"{name} jumps {d:.0f}px in one frame; it will read as a flicker",
                    frame=f, value=d,
                )

    # --- out of canvas
    if canvas:
        w, h = canvas
        for f, fr in enumerate(world):
            for name, jf in fr.items():
                if not (-1 <= jf.origin.x <= w + 1 and -1 <= jf.origin.y <= h + 1):
                    rep.add(
                        "out_of_canvas", "warning", f"{name} has left the frame",
                        frame=f,
                    )
                    break  # one report per frame is plenty

    return rep
