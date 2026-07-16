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

__all__ = ["Sample", "bounce", "roll", "spring", "attract", "drift", "sway"]


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


def attract(
    *,
    start: Vec2,
    end: Vec2,
    frames: int,
    power: float = 2.0,
    contact_gap: float = 0.0,
    ring: float = 0.0,
) -> list[Sample]:
    """Inverse-power attraction: a magnet's snatch, gravity's fall, a charge's grab.

    NOT `spring()`, and the difference is the whole effect. A spring pulls hardest
    when it is *furthest* away, so it lunges early, sails past the target and
    wobbles back — measured on a 300px trip it covers 112% of the distance by
    frame 5 and overshoots to 371. That reads as elastic, which is precisely what
    a magnet is not. An inverse-power law does the opposite: almost nothing
    happens at range, then the force runs away with itself and the last stretch is
    violent. That late snatch is what makes a magnet look magnetic.

    Two things then have to be reconciled, and reconciling them is the point:

    * **The physics has no sense of timing.** Integrated honestly, `1/r^2` reaches
      contact whenever it reaches contact — a slightly different gap or strength
      and the snap lands frames from where the cut needs it.
    * **An animator needs it to land on a chosen frame.** So the trip is simulated
      once, then *resampled in time* onto exactly `frames`. The curve keeps the
      shape the physics gave it; the contact lands where you asked. This is the
      bargain `pace()` already strikes for gaits — real motion, on the beat.

    `contact_gap` stops the body at a surface instead of a point: things collide,
    they do not pass through each other. `ring` adds a small damped rattle after
    contact — the clack of two magnets settling. Set `power=2` for charge and
    gravity, higher (3-4) for a dipole's steeper, snappier grab.
    """
    if frames < 1:
        raise ValueError("attract needs frames >= 1")
    span = end - start
    d0 = span.length()
    if d0 <= contact_gap:
        raise ValueError(
            f"start is already within contact_gap ({d0:.1f} <= {contact_gap})"
        )

    # --- simulate the honest physics, NORMALISED: 1.0 at the start, `stop` at
    # contact. Working in pixels with a fixed step is a trap -- the force at
    # range is 1/d0**power, so a 400px trip at power=4 starts at ~4e-11 and the
    # integrator crawls for millions of steps before anything moves. Normalised,
    # the force starts at 1.0 and the whole fall takes O(1) time for ANY input.
    # The rescale below discards absolute time anyway, which is what buys this.
    stop = contact_gap / d0
    travel = [0.0]  # fraction of the available gap closed
    u, v = 1.0, 0.0
    dt = 2e-5
    guard = 0
    while u > stop and guard < 2_000_000:
        v += (1.0 / max(u, 1e-9) ** power) * dt
        u -= v * dt
        travel.append(min((1.0 - max(u, stop)) / (1.0 - stop), 1.0))
        guard += 1
    if guard >= 2_000_000:  # pragma: no cover - unreachable for sane inputs
        raise RuntimeError("attract failed to reach contact; check power/contact_gap")

    # --- resample that trip onto the frames we were given
    reach = d0 - contact_gap
    unit = span / d0
    out: list[Sample] = []
    n = len(travel) - 1
    for f in range(frames + 1):
        t = (f / frames) * n
        i = min(int(t), n - 1) if n else 0
        d = lerp(travel[i], travel[min(i + 1, n)], t - i) if n else travel[0]
        out.append(Sample(f, start + unit * (d * reach)))

    if ring > 0.0 and frames >= 2:
        # the clack: a fast damped rattle along the line of approach, decaying to
        # nothing. It starts AT contact, so it never delays the impact.
        settle = max(2, int(frames * 0.18))
        for k in range(1, settle + 1):
            f = frames - settle + k
            if f < 1 or f > frames:
                continue
            decay = math.exp(-5.0 * k / settle)
            wobble = math.sin(k / settle * math.pi * 3.0) * ring * decay
            out[f] = Sample(f, out[f].pos - unit * wobble)
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
