"""Line-art render mode: the stick-figure look is skinning, not a new engine.

The claims under test: (1) the stroke render fields survive the JSON round-trip;
(2) `stick()` is a biped skinned as uniform pen strokes with a ring head; (3) a
baked stick figure reaches the canvas as real `Stroke` shapes, not filled capsules;
(4) `style="lineart"` reskins *any* body — a dog, not just a person — because the
look is a render flag over the shared rig ("style, not species").
"""

from __future__ import annotations

from glaxnimate import environment

from glaxnimate_ai.cartoon import assets as A
from glaxnimate_ai.cartoon.gait import pose_at
from glaxnimate_ai.cartoon.presets import lineart, make_gait, quadruped, stick
from glaxnimate_ai.engine.bake import Scene, bake_rig
from glaxnimate_ai.engine.session import SessionStore


def _count(shape, kind: str) -> int:
    n = 1 if shape.type_name == kind else 0
    for ch in getattr(shape, "shapes", []):
        n += _count(ch, kind)
    return n


def _shape_kinds(comp, kind: str) -> int:
    return sum(_count(s, kind) for s in comp.shapes)


def test_part_stroke_fields_round_trip():
    """stroke / head_style / z must survive template -> data -> Body."""
    body = stick()
    rebuilt = A.body_from_data(A.body_to_data(body))
    for name, p in body.parts.items():
        r = rebuilt.parts[name]
        assert (r.stroke, r.head_style, r.z) == (p.stroke, p.head_style, p.z)
    assert all(p.stroke for p in rebuilt.parts.values())
    assert rebuilt.parts["head"].head_style == "ring"


def test_stick_is_a_uniform_inked_biped():
    body = stick()
    widths = {p.width for p in body.parts.values()}
    colors = {p.color for p in body.parts.values()}
    assert len(widths) == 1, "a stick figure is one uniform weight"
    assert len(colors) == 1, "a stick figure is one ink"
    assert body.parts["head"].head is not None, "the head is still a ring, not a rod"


def test_stick_bakes_as_strokes_and_reaches_the_canvas():
    SessionStore()  # ensures the shared Headless environment exists
    body = stick()
    sc = Scene.create(width=360, height=360, frames=12)
    gait = make_gait(body, "walk")
    pose_fn = lambda t: pose_at(body.rig, gait, t, ground_y=300.0, body_x0=180.0)
    bake_rig(sc, body, pose_fn, frames=12, layer_name="stick")

    assert _shape_kinds(sc.comp, "Stroke") == len(body.bones), \
        "every bone should skin to one open stroke"
    assert _shape_kinds(sc.comp, "Rect") == 0, "no filled capsules in line-art mode"

    img = sc.comp.render_image(0)
    px = img.load()
    w, h = int(sc.comp.width), int(sc.comp.height)
    opaque = sum(1 for y in range(h) for x in range(w) if px[x, y][3] > 0)
    assert opaque > 200, "the stick figure did not actually draw"


def test_lineart_reskins_any_body_not_just_people():
    """style, not species: a dog drawn as line-art is still strokes over the same rig."""
    SessionStore()
    dog = quadruped()  # a filled-capsule body, no stroke parts of its own
    assert not any(p.stroke for p in dog.parts.values())

    stick_dog = lineart(dog)
    assert stick_dog.rig is dog.rig or set(stick_dog.rig.joints) == set(dog.rig.joints)
    assert all(p.stroke for p in stick_dog.parts.values())

    sc = Scene.create(width=480, height=360, frames=12)
    gait = make_gait(stick_dog, "trot")
    pose_fn = lambda t: pose_at(stick_dog.rig, gait, t, ground_y=300.0, body_x0=200.0)
    bake_rig(sc, stick_dog, pose_fn, frames=12, layer_name="dog")

    assert _shape_kinds(sc.comp, "Stroke") == len(stick_dog.bones), \
        "lineart should stroke every bone of a non-stick body"
    assert _shape_kinds(sc.comp, "Rect") == 0
