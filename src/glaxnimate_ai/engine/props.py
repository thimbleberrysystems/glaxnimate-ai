"""Scenery: ground, sky, buildings, trees, clouds.

These live in `engine/` rather than `cartoon/` on purpose. They draw *into* a
Glaxnimate document, whereas everything under `cartoon/` is pure pose maths that
must stay free of Qt so the tests and the critic can run without it.

They are all geometric — a house is a rectangle and a triangle — which is exactly
the class of thing an LLM *can* draw. It cannot draw a person (hence the
procedural rigs), but it can absolutely place a roof on a box, look at the render,
and fix it. So this module is a starting kit, not a closed set: scripts are free
to build their own props out of the same primitives.

`parallax` is the one piece of real craft here. Distant things drift slowly, near
things rush past; that difference is most of what sells depth in a side-scroller,
and it costs nothing but a multiplier.
"""

from __future__ import annotations

from glaxnimate import model, utils

from .bake import Scene

__all__ = ["sky", "ground", "house", "school", "tree", "cloud", "sun", "parallax"]


def _group(parent, color: str, name: str = "") -> model.shapes.Group:
    g = parent.add_shape("Group")
    if name:
        g.name = name
    g.add_shape("Fill").color.value = color
    return g


def _rect(parent, x: float, y: float, w: float, h: float, color: str, round_: float = 0.0):
    """A rect placed by its *top-left*, which is how anyone actually thinks about it."""
    g = _group(parent, color)
    r = g.add_shape("Rect")
    r.size.value = utils.Size(w, h)
    r.position.value = utils.Point(x + w / 2, y + h / 2)
    if round_:
        r.rounded.value = round_
    return g


def _ellipse(parent, cx: float, cy: float, w: float, h: float, color: str):
    g = _group(parent, color)
    e = g.add_shape("Ellipse")
    e.size.value = utils.Size(w, h)
    e.position.value = utils.Point(cx, cy)
    return g


def _poly(parent, pts: list[tuple[float, float]], color: str):
    """A sharp-cornered polygon.

    `Bezier.add_point` wants (position, in_tangent, out_tangent) — it is a *curve*
    point. For straight edges, seed the path with one zero-tangent point and use
    `line_to` for the rest, which is what keeps the corners crisp.
    """
    g = _group(parent, color)
    p = g.add_shape("Path")
    bez = p.shape.value
    zero = utils.Point(0, 0)
    x0, y0 = pts[0]
    bez.add_point(utils.Point(x0, y0), zero, zero)
    for x, y in pts[1:]:
        bez.line_to(utils.Point(x, y))
    bez.close()
    p.shape.value = bez
    return g


def sky(scene: Scene, layer, *, top: str = "#bfe3f5", horizon_y: float | None = None) -> None:
    h = horizon_y if horizon_y is not None else int(scene.comp.height)
    _rect(layer, 0, 0, int(scene.comp.width), h, top)


def ground(scene: Scene, layer, y: float, *, color: str = "#7fb069", road: str | None = "#9a9a9f"):
    w, h = int(scene.comp.width), int(scene.comp.height)
    _rect(layer, 0, y, w, h - y, color)
    if road:
        _rect(layer, 0, y, w, 14, road)  # a pavement strip to walk on


def house(layer, x: float, ground_y: float, *, w: float = 150.0, h: float = 120.0,
          wall: str = "#e8c07d", roof: str = "#b5533c", door: str = "#6b4226"):
    top = ground_y - h
    _rect(layer, x, top, w, h, wall)
    _poly(layer, [(x - 14, top), (x + w + 14, top), (x + w / 2, top - 55)], roof)
    _rect(layer, x + w * 0.42, ground_y - h * 0.45, w * 0.18, h * 0.45, door)
    _rect(layer, x + w * 0.12, top + h * 0.18, w * 0.2, h * 0.2, "#cfe8f2")
    _rect(layer, x + w * 0.68, top + h * 0.18, w * 0.2, h * 0.2, "#cfe8f2")


def school(layer, x: float, ground_y: float, *, w: float = 260.0, h: float = 170.0,
           wall: str = "#d9d2c5", roof: str = "#7a6a5d", door: str = "#5a4632"):
    top = ground_y - h
    _rect(layer, x, top, w, h, wall)
    _rect(layer, x - 10, top - 18, w + 20, 18, roof)
    # A clock tower, so it reads as a school and not a warehouse.
    _rect(layer, x + w / 2 - 26, top - 70, 52, 70, wall)
    _rect(layer, x + w / 2 - 34, top - 88, 68, 18, roof)
    _ellipse(layer, x + w / 2, top - 42, 30, 30, "#ffffff")
    _rect(layer, x + w * 0.44, ground_y - h * 0.38, w * 0.12, h * 0.38, door)
    for i in range(4):
        wx = x + w * (0.08 + i * 0.22)
        _rect(layer, wx, top + h * 0.15, w * 0.12, h * 0.22, "#cfe8f2")


def tree(layer, x: float, ground_y: float, *, h: float = 120.0,
         trunk: str = "#7a5c3e", leaves: str = "#4f8f4a"):
    _rect(layer, x - 8, ground_y - h * 0.45, 16, h * 0.45, trunk)
    _ellipse(layer, x, ground_y - h * 0.72, h * 0.75, h * 0.62, leaves)


def cloud(layer, x: float, y: float, *, w: float = 90.0, color: str = "#ffffff"):
    _ellipse(layer, x, y, w, w * 0.55, color)
    _ellipse(layer, x - w * 0.35, y + w * 0.08, w * 0.6, w * 0.42, color)
    _ellipse(layer, x + w * 0.35, y + w * 0.08, w * 0.65, w * 0.45, color)


def sun(layer, x: float, y: float, *, r: float = 46.0, color: str = "#ffd76e"):
    _ellipse(layer, x, y, r * 2, r * 2, color)


def parallax(layer, *, distance: float, frames: int, camera_speed: float) -> None:
    """Scroll a layer at a fraction of the camera's speed.

    `distance` 0 = glued to the camera (the far background barely moves),
    1 = moves with the world (the ground the character walks on).

    Distant things drifting slowly while near things rush past is most of what
    creates depth in a side-scroller, and it is a single multiplier.
    """
    shift = -camera_speed * distance
    for f in (0, frames):
        layer.transform.position.set_keyframe(f, utils.Point(shift * f, 0.0))
