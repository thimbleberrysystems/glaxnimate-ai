"""The animation principles, as functions.

This is the craft the LLM does not have. Everything here is universal: squash and
stretch applies to a bouncing ball exactly as it does to a person landing, and an
arc is an arc whether it is a hand or a thrown hat.

Timing is the whole game. Linear interpolation is what makes computer animation
look like computer animation — real motion accelerates and decelerates, and the
`ease_*` family is how you say so. `feedback/diagnose.py` measures exactly this
via the spacing chart, so these functions and that check are two views of one
idea.
"""

from __future__ import annotations

import math

from .geometry import Vec2, clamp

__all__ = [
    "linear",
    "ease_in",
    "ease_out",
    "ease_in_out",
    "anticipate",
    "overshoot",
    "bounce_decay",
    "squash_stretch",
    "arc",
    "follow_through",
]


# --------------------------------------------------------------------- easing
# Each takes t in [0, 1] and returns an eased t in [0, 1].


def linear(t: float) -> float:
    return clamp(t, 0.0, 1.0)


def ease_in(t: float, power: float = 2.0) -> float:
    """Slow start. Weight getting under way."""
    return clamp(t, 0.0, 1.0) ** power


def ease_out(t: float, power: float = 2.0) -> float:
    """Slow stop. Settling."""
    return 1.0 - (1.0 - clamp(t, 0.0, 1.0)) ** power


def ease_in_out(t: float) -> float:
    """Slow in, slow out — the default for almost any deliberate move."""
    t = clamp(t, 0.0, 1.0)
    return 3 * t * t - 2 * t * t * t  # smoothstep


def anticipate(t: float, amount: float = 0.15) -> float:
    """Wind up the opposite way before moving.

    Nobody jumps without crouching first. Returns values below 0 early on, which
    is the point: the object pulls *back* before it goes.
    """
    t = clamp(t, 0.0, 1.0)
    return ease_in_out(t) - amount * math.sin(math.pi * t) * (1.0 - t)


def overshoot(t: float, amount: float = 0.12) -> float:
    """Sail past the target and settle back. Weight has momentum."""
    t = clamp(t, 0.0, 1.0)
    return ease_out(t) + amount * math.sin(math.pi * t) * t


def bounce_decay(n: int, restitution: float = 0.6) -> list[float]:
    """Peak heights of successive bounces, as fractions of the first."""
    return [restitution**i for i in range(n)]


# ---------------------------------------------------------------- deformation
def squash_stretch(speed: float, *, amount: float = 0.35, cap: float = 0.6) -> Vec2:
    """Scale factors that preserve area: fast = stretched, impact = squashed.

    Volume conservation is what sells it — a ball that only flattens reads as
    melting. Pass a negative `speed` for impact (squash), positive for flight
    (stretch). Returns (x_scale, y_scale) around 1.0.
    """
    k = clamp(speed * amount, -cap, cap)
    stretch = 1.0 + k
    return Vec2(1.0 / stretch, stretch)  # x * y == 1: area preserved


# ----------------------------------------------------------------------- arcs
def arc(a: Vec2, b: Vec2, t: float, height: float) -> Vec2:
    """A point along a parabolic arc from `a` to `b`.

    Living things move in arcs; only machines move in straight lines. `height`
    is the sag (positive) or lift (negative) perpendicular to the chord.
    """
    t = clamp(t, 0.0, 1.0)
    straight = Vec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
    bulge = math.sin(math.pi * t) * height
    chord = b - a
    n = Vec2(-chord.y, chord.x)
    ln = n.length()
    if ln < 1e-9:
        return straight
    return straight + (n / ln) * bulge


def follow_through(t: float, lag: float = 0.15) -> float:
    """Sample a driver's motion slightly in the past.

    Loose parts — hair, a coat, a tail, an ear — do not stop when the body
    stops. Drive them with a delayed copy of the body and they trail correctly.
    """
    return clamp(t - lag, 0.0, 1.0)
