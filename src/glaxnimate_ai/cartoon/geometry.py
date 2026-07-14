"""Minimal 2D vector math.

Deliberately dependency-free: the pose engine is pure Python so that rig maths,
gaits and the linter can be tested without Glaxnimate (or Qt) present at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Vec2", "clamp", "lerp"]


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, o: Vec2) -> Vec2:
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: Vec2) -> Vec2:
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> Vec2:
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> Vec2:
        return Vec2(self.x / k, self.y / k)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def rotated(self, degrees: float) -> Vec2:
        r = math.radians(degrees)
        c, s = math.cos(r), math.sin(r)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def angle(self) -> float:
        """Direction in degrees, measured like the screen: +x right, +y down."""
        return math.degrees(math.atan2(self.y, self.x))

    def distance_to(self, o: Vec2) -> float:
        return math.hypot(self.x - o.x, self.y - o.y)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
