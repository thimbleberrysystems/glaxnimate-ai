"""Diagrams: the vocabulary an *explainer* needs, as data.

Educational animation is not a different engine — it is the same "content is data"
bet pointed at a whiteboard instead of a character. Every element here is a pure
function that returns a **prop-schema shape list** (the same `rect`/`ellipse`/
`polygon`/`text`/`polyline` a script already draws with `add_shape`), computed
rather than drawn. So an axis, a plotted curve, a labelled arrow or a bar chart is
authored, rendered and *revealed* (see `write_on`) exactly like any other prop.

The set is chosen to cover the didactic roles animation actually serves — connect
objects (`arrow`, `label`), covary and show process (`axes` + `plot`, `number_line`,
`counter`), reveal structure (`brace`, `grid`), and compare (`bar_chart`) — not to
mirror any one tool's API.

Coordinates are local, origin at the element's natural anchor, y **down** (screen
space), so a list drops straight into `add_shape(shapes, X, Y)`. For a graph, the
anchor is the plot box's bottom-left corner and data maps up-and-right via
`mapper(...)`, which a script also uses to place a dot *on* the curve.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

__all__ = ["arrow", "label", "bracket", "brace", "underline", "axes", "mapper",
           "plot", "number_line", "grid", "bar_chart"]

INK = "#1f2430"


def _fmt(v: float) -> str:
    r = round(v, 3)
    return str(int(r)) if r == int(r) else f"{r:g}"


# --------------------------------------------------------------- annotations
def arrow(x0: float, y0: float, x1: float, y1: float, *, color: str = INK,
          width: float = 3.0, head: float = 12.0) -> list[dict]:
    """A straight arrow from (x0,y0) to (x1,y1): a stroked shaft + a solid head.
    The connector that points at a thing or links two things."""
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy) or 1.0
    ux, uy = dx / d, dy / d
    px, py = -uy, ux                      # perpendicular
    base = (x1 - ux * head, y1 - uy * head)
    return [
        {"type": "polyline", "points": [[x0, y0], [base[0], base[1]]],
         "color": color, "width": width},
        {"type": "polygon", "color": color, "points": [
            [x1, y1],
            [base[0] + px * head * 0.5, base[1] + py * head * 0.5],
            [base[0] - px * head * 0.5, base[1] - py * head * 0.5]]},
    ]


def label(text: str, x: float, y: float, *, to: tuple[float, float] | None = None,
          size: float = 18.0, color: str = INK, anchor: str = "middle",
          leader: str = INK) -> list[dict]:
    """A text callout at (x,y). If `to` is given, a thin leader line runs from the
    text to that point — the classic "this bit here" annotation."""
    sh: list[dict] = []
    if to is not None:
        sh.append({"type": "polyline", "points": [[x, y + size * 0.2], list(to)],
                   "color": leader, "width": 1.5})
    sh.append({"type": "text", "x": x, "y": y, "text": text, "size": size,
               "color": color, "anchor": anchor})
    return sh


def underline(x: float, w: float, y: float, *, color: str = "#e0533d",
              width: float = 3.0) -> list[dict]:
    """A stroke under a span [x, x+w] at height y — emphasis on a term."""
    return [{"type": "polyline", "points": [[x, y], [x + w, y]],
             "color": color, "width": width}]


def bracket(x0: float, y0: float, x1: float, y1: float, *, color: str = INK,
            width: float = 2.5, depth: float = 10.0) -> list[dict]:
    """A square bracket spanning (x0,y0)->(x1,y1), lipping toward its content by
    `depth` — groups a set of things."""
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy) or 1.0
    px, py = -dy / d * depth, dx / d * depth
    return [{"type": "polyline", "color": color, "width": width, "points": [
        [x0 + px, y0 + py], [x0, y0], [x1, y1], [x1 + px, y1 + py]]}]


def brace(x0: float, y0: float, x1: float, y1: float, *, color: str = INK,
          width: float = 2.5, depth: float = 14.0, text: str | None = None,
          size: float = 16.0) -> list[dict]:
    """A curly-ish brace spanning the two points, pointing at its midpoint, with an
    optional label past the tip — 'all of this is X'."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy) or 1.0
    px, py = -dy / d, dx / d               # toward the label side
    q0 = (x0 + dx * 0.25, y0 + dy * 0.25)
    q1 = (x1 - dx * 0.25, y1 - dy * 0.25)
    tip = (mx + px * depth, my + py * depth)
    sh = [{"type": "polyline", "color": color, "width": width, "points": [
        [x0, y0],
        [q0[0] + px * depth * 0.6, q0[1] + py * depth * 0.6],
        [mx + px * depth * 0.6, my + py * depth * 0.6], list(tip),
        [mx + px * depth * 0.6, my + py * depth * 0.6],
        [q1[0] + px * depth * 0.6, q1[1] + py * depth * 0.6],
        [x1, y1]]}]
    if text is not None:
        sh += label(text, tip[0] + px * (size + 2), tip[1] + py * (size + 2),
                    size=size, color=color)
    return sh


# ------------------------------------------------------------------- graphs
def mapper(w: float, h: float, xrange: tuple[float, float],
           yrange: tuple[float, float]) -> Callable[[float, float], tuple[float, float]]:
    """Data (dx,dy) -> local (lx,ly) for a plot box `w`x`h` with the bottom-left at
    the origin and y up. Use it to drop a dot or a label *onto* a graph the same
    `axes`/`plot` drew, so they line up."""
    x0, x1 = xrange
    y0, y1 = yrange
    sx = w / ((x1 - x0) or 1.0)
    sy = h / ((y1 - y0) or 1.0)

    def to(dx: float, dy: float) -> tuple[float, float]:
        return ((dx - x0) * sx, -(dy - y0) * sy)
    return to


def axes(w: float, h: float, *, xrange: tuple[float, float] = (0.0, 10.0),
         yrange: tuple[float, float] = (0.0, 10.0), xstep: float = 1.0,
         ystep: float = 1.0, color: str = INK, width: float = 2.5,
         numbers: bool = True, xlabel: str | None = None,
         ylabel: str | None = None, tick: float = 5.0,
         number_size: float = 12.0) -> list[dict]:
    """A 2D coordinate frame: x and y axes (arrow-tipped), tick marks and numbers.
    Origin at data (0,0) when it is in range, else clamped to the box edge. Draw
    `plot(...)` with the SAME w/h/xrange/yrange and place both at one point."""
    to = mapper(w, h, xrange, yrange)
    x0, x1 = xrange
    y0, y1 = yrange
    axis_y = min(max(0.0, y0), y1)         # where the x-axis sits (data y)
    axis_x = min(max(0.0, x0), x1)
    ox, _ = to(axis_x, 0)
    _, oy = to(0, axis_y)
    sh: list[dict] = []
    sh += arrow(*to(x0, axis_y), *to(x1, axis_y), color=color, width=width, head=11)
    sh += arrow(*to(axis_x, y0), *to(axis_x, y1), color=color, width=width, head=11)

    def frange(a, b, s):
        n = int(round((b - a) / s))
        return [a + i * s for i in range(n + 1)]

    for xv in frange(x0, x1, xstep):
        lx, ly = to(xv, axis_y)
        if abs(xv - axis_x) < 1e-9:
            continue
        sh.append({"type": "polyline", "color": color, "width": 1.6,
                   "points": [[lx, ly - tick], [lx, ly + tick]]})
        if numbers:
            sh.append({"type": "text", "x": lx, "y": ly + tick + number_size,
                       "text": _fmt(xv), "size": number_size, "color": color,
                       "anchor": "middle"})
    for yv in frange(y0, y1, ystep):
        lx, ly = to(axis_x, yv)
        if abs(yv - axis_y) < 1e-9:
            continue
        sh.append({"type": "polyline", "color": color, "width": 1.6,
                   "points": [[lx - tick, ly], [lx + tick, ly]]})
        if numbers:
            sh.append({"type": "text", "x": lx - tick - 4, "y": ly + number_size * 0.35,
                       "text": _fmt(yv), "size": number_size, "color": color,
                       "anchor": "end"})
    if xlabel:
        lx, ly = to(x1, axis_y)
        sh.append({"type": "text", "x": lx + 6, "y": ly + 20, "text": xlabel,
                   "size": number_size + 4, "color": color, "anchor": "start"})
    if ylabel:
        lx, ly = to(axis_x, y1)
        sh.append({"type": "text", "x": lx - 8, "y": ly - 8, "text": ylabel,
                   "size": number_size + 4, "color": color, "anchor": "end"})
    return sh


def plot(fn: Callable[[float], float], *, w: float, h: float,
         xrange: tuple[float, float] = (0.0, 10.0),
         yrange: tuple[float, float] = (0.0, 10.0), samples: int = 96,
         color: str = "#1a6cff", width: float = 3.0) -> list[dict]:
    """Sample y=fn(x) across `xrange` into a stroked curve, in the same local frame
    as `axes(w,h,xrange,yrange)`. Points that fall outside `yrange` break the curve
    (so an asymptote doesn't draw a spurious vertical), rather than clamping flat."""
    to = mapper(w, h, xrange, yrange)
    x0, x1 = xrange
    y0, y1 = yrange
    segs: list[list[list[float]]] = [[]]
    for i in range(samples + 1):
        xv = x0 + (x1 - x0) * i / samples
        try:
            yv = fn(xv)
        except (ValueError, ZeroDivisionError):
            yv = None
        if yv is None or not (y0 - 1e-9 <= yv <= y1 + 1e-9) or yv != yv:
            if segs[-1]:
                segs.append([])
            continue
        lx, ly = to(xv, yv)
        segs[-1].append([round(lx, 2), round(ly, 2)])
    return [{"type": "polyline", "points": s, "color": color, "width": width}
            for s in segs if len(s) >= 2]


def number_line(w: float, *, xrange: tuple[float, float] = (0.0, 10.0),
                step: float = 1.0, color: str = INK, width: float = 2.5,
                numbers: bool = True, tick: float = 6.0,
                number_size: float = 13.0) -> list[dict]:
    """A single arrow-tipped axis with ticks and numbers — the number line."""
    x0, x1 = xrange
    sx = w / ((x1 - x0) or 1.0)
    sh = arrow(0, 0, w, 0, color=color, width=width, head=11)
    n = int(round((x1 - x0) / step))
    for i in range(n + 1):
        xv = x0 + i * step
        lx = (xv - x0) * sx
        sh.append({"type": "polyline", "color": color, "width": 1.6,
                   "points": [[lx, -tick], [lx, tick]]})
        if numbers:
            sh.append({"type": "text", "x": lx, "y": tick + number_size, "text": _fmt(xv),
                       "size": number_size, "color": color, "anchor": "middle"})
    return sh


def grid(w: float, h: float, *, nx: int = 10, ny: int = 10,
         color: str = "#c7ccd6", width: float = 1.0) -> list[dict]:
    """A faint reference grid over a w×h box (origin at bottom-left, y up)."""
    sh: list[dict] = []
    for i in range(nx + 1):
        lx = w * i / nx
        sh.append({"type": "polyline", "color": color, "width": width,
                   "points": [[lx, 0], [lx, -h]]})
    for j in range(ny + 1):
        ly = -h * j / ny
        sh.append({"type": "polyline", "color": color, "width": width,
                   "points": [[0, ly], [w, ly]]})
    return sh


def bar_chart(values: Sequence[float], *, w: float, h: float,
              labels: Sequence[str] | None = None, color: str = "#1a6cff",
              baseline: str = INK, gap: float = 0.35, vmax: float | None = None,
              value_labels: bool = False, number_size: float = 13.0) -> list[dict]:
    """Bars for `values` in a w×h box (bottom-left origin, y up), with a baseline
    and optional category / value labels — the compare-quantities diagram."""
    n = len(values)
    vmax = vmax if vmax is not None else (max(values) or 1.0)
    slot = w / n
    bw = slot * (1 - gap)
    sh: list[dict] = [{"type": "polyline", "color": baseline, "width": 2.5,
                       "points": [[0, 0], [w, 0]]}]
    for i, v in enumerate(values):
        cx = slot * (i + 0.5)
        bh = h * (v / vmax)
        sh.append({"type": "rect", "x": cx - bw / 2, "y": -bh, "w": bw, "h": bh,
                   "color": color})
        if labels is not None:
            sh.append({"type": "text", "x": cx, "y": number_size + 4,
                       "text": str(labels[i]), "size": number_size,
                       "color": baseline, "anchor": "middle"})
        if value_labels:
            sh.append({"type": "text", "x": cx, "y": -bh - 6, "text": _fmt(v),
                       "size": number_size, "color": color, "anchor": "middle"})
    return sh
