"""Motion for things without legs: balls, wheels, leaves, logos.

"Animate anything" has to mean more than "animate more animals". A bouncing
ball, a driving car and a swaying sign are not rigs, and forcing them through a
gait engine would be silly — so they live here, and the animation principles
(`principles.py`) apply to them unchanged.

The wheel is worth a look: `roll()` couples spin to travel with distance ==
radius * angle. That is the *same* no-slip law that pins a planted foot, which
is why one linter check covers a sliding foot, a skating paw and a spinning
wheel on a car that is not moving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Vec2, lerp
from .principles import ease_in_out, squash_stretch

__all__ = ["Sample", "bounce", "roll", "spring", "drift", "sway"]


@dataclass(slots=True)
class Sample:
    """One object's state on one frame."""

    frame: int
    pos: Vec2
    scale: Vec2 = field(default_factory=lambda: Vec2(1.0, 1.0))
    angle: float = 0.0


def bounce(
    *,
    x0: float,
    x1: float,
    ground_y: float,
    apex: float,
    frames: int,
    bounces: int = 5,
    restitution: float = 0.62,
    radius: float = 40.0,
) -> list[Sample]:
    """A ball bouncing across the screen, losing height each time.

    Successive arcs are both lower *and shorter*: a bounce's airtime goes with
    the square root of its height, so equal-length arcs would look wrong no
    matter how the heights decay. Squash on contact, stretch in flight, area
    preserved throughout.
    """
    if bounces < 1:
        raise ValueError("need at least one bounce")

    apexes = [apex * restitution**i for i in range(bounces)]
    # Airtime ~ sqrt(height). Normalised, then scaled to fill `frames`.
    durations = [math.sqrt(a / apex) if apex > 0 else 1.0 for a in apexes]
    total = sum(durations)
    durations = [d / total * frames for d in durations]

    bounds: list[tuple[float, float]] = []
    acc = 0.0
    for d in durations:
        bounds.append((acc, acc + d))
        acc += d

    out: list[Sample] = []
    for f in range(frames + 1):
        t = min(float(f), float(frames))

        i = next((k for k, (a, b) in enumerate(bounds) if a <= t < b), bounces - 1)
        a, b = bounds[i]
        s = (t - a) / (b - a) if b > a else 1.0
        s = min(max(s, 0.0), 1.0)

        # Parabola: on the ground at s=0 and s=1, peaking halfway.
        h = apexes[i] * 4.0 * s * (1.0 - s)
        y = ground_y - radius - h
        x = lerp(x0, x1, t / frames)

        # Vertical speed drives the deformation; near a contact it is an impact.
        vy = apexes[i] * 4.0 * (1.0 - 2.0 * s) / max(b - a, 1e-6)
        near_contact = s < 0.06 or s > 0.94
        norm = vy / max(apex * 0.25, 1e-6)
        scale = squash_stretch(-abs(norm) if near_contact else abs(norm) * 0.5)

        out.append(Sample(f, Vec2(x, y), scale))
    return out


def roll(
    *,
    x0: float,
    x1: float,
    y: float,
    radius: float,
    frames: int,
    ease: bool = True,
) -> list[Sample]:
    """A wheel rolling without slipping.

    The spin is *derived* from the distance travelled, never authored alongside
    it. Author them separately and they drift apart, and the wheel visibly
    skates — the single most common tell in amateur vehicle animation.
    """
    out: list[Sample] = []
    for f in range(frames + 1):
        t = f / frames
        s = ease_in_out(t) if ease else t
        x = lerp(x0, x1, s)
        angle = math.degrees((x - x0) / radius)  # <- the no-slip law
        out.append(Sample(f, Vec2(x, y), angle=angle))
    return out


def spring(
    *,
    start: Vec2,
    end: Vec2,
    frames: int,
    stiffness: float = 0.18,
    damping: float = 0.72,
) -> list[Sample]:
    """Overshoot and settle. Anything with mass arrives this way."""
    out: list[Sample] = []
    pos, vel = start, Vec2(0.0, 0.0)
    for f in range(frames + 1):
        force = (end - pos) * stiffness
        vel = (vel + force) * damping
        pos = pos + vel
        out.append(Sample(f, pos))
    return out


def drift(*, start: Vec2, end: Vec2, frames: int, sway_amount: float = 30.0) -> list[Sample]:
    """A falling leaf: descends while swinging side to side."""
    out: list[Sample] = []
    for f in range(frames + 1):
        t = f / frames
        base = Vec2(lerp(start.x, end.x, t), lerp(start.y, end.y, t))
        offset = math.sin(t * math.pi * 4.0) * sway_amount
        out.append(Sample(f, base + Vec2(offset, 0.0), angle=offset * 0.6))
    return out


def sway(*, pivot: Vec2, frames: int, amplitude: float = 8.0, cycles: float = 2.0) -> list[Sample]:
    """A gentle rock in place — a sign, a branch, an idle character."""
    return [
        Sample(f, pivot, angle=amplitude * math.sin(2.0 * math.pi * cycles * f / frames))
        for f in range(frames + 1)
    ]
