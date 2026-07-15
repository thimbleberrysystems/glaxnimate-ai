"""Keyframe reduction: dense samples in, sparse bezier keys out.

v1 baked every bone on every frame — 2,332 keyframes and 227 KB for one walking
man, and a file that was uneditable in the GUI because dragging any key means
fighting the 95 next to it. This module is the fix: fit a handful of keys with
bezier easing to the sampled curves, verified to stay within the linter's own
tolerances.

Two ideas, both boring on purpose:

* **Seed at extrema.** The local minima/maxima of a channel are the *poses* — a
  knee's extreme bend is the contact pose, an arm's peak swing is its reversal.
  Seeding keys there means the sparse keys land where an animator would have put
  them, which is what makes the output *editable*, not merely small.
* **Greedy refinement.** After seeding, reconstruct and insert a key at the worst
  error until everything is within tolerance. No cleverness; provably within
  budget because the loop does not exit until it is.

The timing curves use fixed x-handles at 1/3 and 2/3. That makes the bezier's
time function the identity (x(s) = s), so fitting the y-handles is a closed-form
least squares — and evaluation needs no root-finding. The expressive loss is
minor (you can't represent extreme time-warps in one segment), and the refinement
loop compensates by adding a key where it matters.

`evaluate_*` deliberately mirror Glaxnimate's interpolation; a test compares them
against `value_at_time` so the fidelity proof means something.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cartoon.geometry import Vec2

__all__ = [
    "ScalarKey", "PointKey",
    "reduce_scalar", "reduce_point",
    "evaluate_scalar", "evaluate_point",
]

# Fixed bezier x-handles: x(s) == s, so time never needs solving.
_CX1, _CX2 = 1.0 / 3.0, 2.0 / 3.0


@dataclass(slots=True)
class ScalarKey:
    frame: int
    value: float
    #: Timing-curve control points for the segment AFTER this key, in [0,1]².
    cy1: float = _CX1
    cy2: float = _CX2


@dataclass(slots=True)
class PointKey:
    frame: int
    value: Vec2
    cy1: float = _CX1
    cy2: float = _CX2


def _timing(u: float, cy1: float, cy2: float) -> float:
    """Value-fraction for time-fraction `u` on the (1/3, cy1)(2/3, cy2) bezier."""
    inv = 1.0 - u
    return 3.0 * cy1 * u * inv * inv + 3.0 * cy2 * u * u * inv + u * u * u


def _fit_segment(norm: list[float]) -> tuple[float, float]:
    """Least-squares (cy1, cy2) for normalized samples norm[i] = v(u_i), u_i = i/(n-1).

    y(u) = 3·cy1·u(1-u)² + 3·cy2·u²(1-u) + u³ is linear in (cy1, cy2): a plain
    2×2 normal-equation solve. Handles clamped to [-0.5, 1.5] — enough for real
    overshoot, not enough to write garbage that renders as a glitch.
    """
    n = len(norm)
    if n <= 2:
        return _CX1, _CX2
    a11 = a12 = a22 = b1 = b2 = 0.0
    for i in range(1, n - 1):
        u = i / (n - 1)
        inv = 1.0 - u
        f1 = 3.0 * u * inv * inv
        f2 = 3.0 * u * u * inv
        target = norm[i] - u ** 3
        a11 += f1 * f1
        a12 += f1 * f2
        a22 += f2 * f2
        b1 += f1 * target
        b2 += f2 * target
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return _CX1, _CX2
    cy1 = (b1 * a22 - b2 * a12) / det
    cy2 = (a11 * b2 - a12 * b1) / det
    clamp = lambda v: max(-0.5, min(1.5, v))  # noqa: E731
    return clamp(cy1), clamp(cy2)


def _extrema(values: list[float]) -> list[int]:
    """Indices of local minima/maxima — the poses an animator would key."""
    out = []
    for i in range(1, len(values) - 1):
        a, b, c = values[i - 1], values[i], values[i + 1]
        if (b - a) * (c - b) < 0:
            out.append(i)
    return out


# ------------------------------------------------------------------- scalar
def evaluate_scalar(keys: list[ScalarKey], frame: float) -> float:
    """Reconstruct a reduced scalar channel — mirrors Glaxnimate's interpolation."""
    if not keys:
        return 0.0
    if frame <= keys[0].frame:
        return keys[0].value
    for k0, k1 in zip(keys, keys[1:], strict=False):
        if frame <= k1.frame:
            u = (frame - k0.frame) / (k1.frame - k0.frame)
            f = _timing(u, k0.cy1, k0.cy2)
            return k0.value + (k1.value - k0.value) * f
    return keys[-1].value


def reduce_scalar(values: list[float], *, tol: float, ease: bool = True) -> list[ScalarKey]:
    """Fit sparse eased keys to dense per-frame samples, max error ≤ `tol`.

    `ease=False` keeps every segment linear-timed (identity handles). Used for
    channels whose transitions cannot be written — see the scale note in bake.py —
    at the cost of a few extra keys from the refinement loop.
    """
    n = len(values)
    if n == 0:
        return []
    if max(values) - min(values) < tol * 0.5:
        return [ScalarKey(0, values[0])]  # effectively static

    frames = sorted({0, n - 1, *_extrema(values)})

    def build(frames_: list[int]) -> list[ScalarKey]:
        keys = []
        for i, f in enumerate(frames_):
            k = ScalarKey(f, values[f])
            if i + 1 < len(frames_):
                f1 = frames_[i + 1]
                span = values[f1] - values[f]
                if ease and abs(span) > 1e-9 and f1 - f > 1:
                    norm = [(values[j] - values[f]) / span for j in range(f, f1 + 1)]
                    k.cy1, k.cy2 = _fit_segment(norm)
            keys.append(k)
        return keys

    while True:
        keys = build(frames)
        worst_f, worst_e = -1, tol
        for f in range(n):
            e = abs(evaluate_scalar(keys, f) - values[f])
            if e > worst_e:
                worst_f, worst_e = f, e
        if worst_f < 0:
            return keys
        frames = sorted({*frames, worst_f})


# -------------------------------------------------------------------- point
def evaluate_point(keys: list[PointKey], frame: float) -> Vec2:
    if not keys:
        return Vec2()
    if frame <= keys[0].frame:
        return keys[0].value
    for k0, k1 in zip(keys, keys[1:], strict=False):
        if frame <= k1.frame:
            u = (frame - k0.frame) / (k1.frame - k0.frame)
            f = _timing(u, k0.cy1, k0.cy2)
            return k0.value + (k1.value - k0.value) * f
    return keys[-1].value


def reduce_point(points: list[Vec2], *, tol: float, ease: bool = True) -> list[PointKey]:
    """Sparse keys for a 2D channel.

    Between keys the *path* is a straight chord (eased in time), so curvature is
    handled by splitting: wherever the true path bows more than `tol` away from
    the chord, a key is inserted. Timing along each chord is then fitted the same
    way as scalars. Extrema of x and y seed the split, for the same editability
    reason.
    """
    n = len(points)
    if n == 0:
        return []
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span < tol * 0.5:
        return [PointKey(0, points[0])]

    frames = sorted({0, n - 1, *_extrema(xs), *_extrema(ys)})

    def build(frames_: list[int]) -> list[PointKey]:
        keys = []
        for i, f in enumerate(frames_):
            k = PointKey(f, points[f])
            if i + 1 < len(frames_):
                f1 = frames_[i + 1]
                chord = points[f1] - points[f]
                clen2 = chord.x * chord.x + chord.y * chord.y
                if ease and clen2 > 1e-12 and f1 - f > 1:
                    # progress along the chord as the timing signal
                    norm = []
                    for j in range(f, f1 + 1):
                        d = points[j] - points[f]
                        norm.append((d.x * chord.x + d.y * chord.y) / clen2)
                    k.cy1, k.cy2 = _fit_segment(norm)
            keys.append(k)
        return keys

    while True:
        keys = build(frames)
        worst_f, worst_e = -1, tol
        for f in range(n):
            e = evaluate_point(keys, f).distance_to(points[f])
            if e > worst_e:
                worst_f, worst_e = f, e
        if worst_f < 0:
            return keys
        frames = sorted({*frames, worst_f})
