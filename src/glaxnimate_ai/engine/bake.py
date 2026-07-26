"""Turn animation into a Glaxnimate document — as a rig, not a bake.

v1 wrote every bone's *world* transform on every frame: 2,332 keyframes for one
walking man, uneditable output, all easing flattened to per-frame linear. v2
writes what an animator would have built by hand:

* **One layer per bone, parented** (`Layer.parent`) to its parent bone's layer —
  transform inheritance does the forward kinematics, so a bone's local position
  is *static* (its attachment offset) and only its **local rotation** animates.
  Draw order comes from layer stacking (last-added on top), independent of the
  parent hierarchy — the same split Spine uses, and the reason a far arm can be
  parented to the spine yet still paint behind it.
* **Sparse keys with real easing** (`engine/reduce.py`): keys seeded at the
  channel's extrema (the poses), refined until reconstruction error is inside
  tolerance, bezier timing fitted per segment and written through
  `KeyframeTransition`.

The result opens in the GUI as an actual puppet: rotate a thigh and the leg
follows; drag one of a handful of keys and the motion retimes.

The pose engine stays pure Python; this module remains the only place poses meet
Glaxnimate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from glaxnimate import model, utils

from ..cartoon.geometry import Vec2
from ..cartoon.presets import Body, Part
from ..cartoon.rig import Pose
from .reduce import PointKey, ScalarKey, reduce_point, reduce_scalar

__all__ = ["Scene", "bake_rig", "bake_samples"]

#: Reduction tolerances. Position error under a pixel and angle error under a
#: degree are both below what the linter allows and what an eye can see at 1x.
TOL_PX = 0.75
TOL_DEG = 0.75
#: Contact chains (a foot and the leg above it) get a tighter angular tolerance.
#: Interpolation wiggle composes down the chain into horizontal drift of planted
#: feet; 0.3 deg holds the residual slip in the *baked* output under 0.8px/frame
#: (measured), which is invisible under anti-aliasing. The IR itself — what the
#: linter certifies — remains exactly zero-slip; this bound is purely about how
#: faithfully sparse keys replay it.
TOL_DEG_LEG = 0.3
TOL_SCALE = 0.01


@dataclass(slots=True)
class Scene:
    """A Glaxnimate document plus the composition everything goes into."""

    document: model.Document
    comp: model.Composition
    fps: float = 24.0

    @classmethod
    def create(
        cls, width: int = 960, height: int = 540, frames: int = 48, fps: float = 24.0
    ) -> Scene:
        doc = model.Document("")
        comp = doc.assets.add_composition()
        comp.width, comp.height = width, height
        comp.animation.first_frame = 0
        comp.animation.last_frame = frames
        comp.fps = fps
        return cls(doc, comp, fps)

    def layer(self, name: str = "layer") -> model.shapes.Layer:
        """A layer whose visibility actually spans the animation.

        `Layer.animation.last_frame` defaults to **-1**, which makes the layer
        invisible and every frame render blank — with no error and no warning.
        Never add a layer without setting this. See docs/glaxnimate-api.md.
        """
        lay = self.comp.add_shape("Layer")
        lay.name = name
        lay.animation.first_frame = self.comp.animation.first_frame
        lay.animation.last_frame = self.comp.animation.last_frame
        return lay


# ----------------------------------------------------------- channel writers
_LINEAR = (1.0 / 3.0, 2.0 / 3.0)


def _write_scalar(prop, keys: list[ScalarKey], *, transitions: bool = True,
                  offset: float = 0.0) -> int:
    """Write a reduced scalar channel; returns how many keys it cost.

    `transitions=False` writes plain linear keys, for channels that ship linear
    and let the reducer compensate with a few extra keys instead of timing curves.

    `offset` shifts every key: the reducer counts from 0 because it only sees a
    list, so a clip that starts partway through the film says so here.
    """
    if len(keys) <= 1:
        prop.value = keys[0].value if keys else 0.0
        return 0
    for k in keys:
        prop.set_keyframe(float(k.frame) + offset, k.value)
    if transitions:
        for k in keys[:-1]:
            if (k.cy1, k.cy2) == _LINEAR:
                continue  # identity timing: the default, nothing to write
            tr = model.KeyframeTransition()
            tr.before = utils.Point(1.0 / 3.0, k.cy1)
            tr.after = utils.Point(2.0 / 3.0, k.cy2)
            prop.set_transition(float(k.frame) + offset, tr)
    return len(keys)


def _write_point(prop, keys: list[PointKey], *, transitions: bool = True,
                 offset: float = 0.0) -> int:
    if len(keys) <= 1:
        v = keys[0].value if keys else Vec2()
        prop.value = utils.Point(v.x, v.y)
        return 0
    for k in keys:
        prop.set_keyframe(float(k.frame) + offset, utils.Point(k.value.x, k.value.y))
    if transitions:
        for k in keys[:-1]:
            if (k.cy1, k.cy2) == _LINEAR:
                continue
            tr = model.KeyframeTransition()
            tr.before = utils.Point(1.0 / 3.0, k.cy1)
            tr.after = utils.Point(2.0 / 3.0, k.cy2)
            prop.set_transition(float(k.frame) + offset, tr)
    return len(keys)


def _write_scale(prop, keys: list[PointKey], *, transitions: bool = True,
                 offset: float = 0.0) -> int:
    """Write a reduced scale channel. Returns how many keys it cost.

    Scale is a **QVector2D** property, so it must be written with `utils.Vector2D`:
    a `utils.Point` (QPointF) silently fails to write it, which is why the historical
    `_write_point`-into-scale calls never actually squashed anything. With the
    patched binding, eased scale via `set_transition` also works, so this can carry
    timing like position and rotation. Empty channel falls back to identity (1, 1).
    """
    if len(keys) <= 1:
        v = keys[0].value if keys else Vec2(1.0, 1.0)
        prop.value = utils.Vector2D(v.x, v.y)
        return 0
    for k in keys:
        prop.set_keyframe(float(k.frame) + offset, utils.Vector2D(k.value.x, k.value.y))
    if transitions:
        for k in keys[:-1]:
            if (k.cy1, k.cy2) == _LINEAR:
                continue
            tr = model.KeyframeTransition()
            tr.before = utils.Point(1.0 / 3.0, k.cy1)
            tr.after = utils.Point(2.0 / 3.0, k.cy2)
            prop.set_transition(float(k.frame) + offset, tr)
    return len(keys)


def _clip_offset(samples) -> float:
    """The frame a sample list actually starts on.

    The reducer only ever sees a list of values, so its keys count from 0. That
    is invisible while every `motion.*` generator starts at frame 0 -- index and
    frame coincide and `Sample.frame` is decoration. The moment a clip starts
    partway through a film (beat four of six), the index is a lie and the whole
    clip silently plays at the top of the timeline. So the offset is read back
    from the samples, and a list that is not one contiguous run is refused
    rather than quietly mis-timed.
    """
    if not samples:
        return 0.0
    first = samples[0].frame
    for i, smp in enumerate(samples):
        if smp.frame != first + i:
            raise ValueError(
                f"samples must be one contiguous run of frames; samples[{i}].frame "
                f"is {smp.frame}, expected {first + i}. Baking keys off the list "
                f"index would put this clip on the wrong frames."
            )
    return float(first)


# ----------------------------------------------------------------- skinning
def _stroke_styler(g, color: str, width: float):
    """A round-capped, round-joined stroke — the pen that draws a stick figure.

    Round caps matter: each bone is its own layer, so a rounded line end is what
    makes one bone meet the next cleanly at a bend — the stroke-mode equivalent of
    the capsule's joint disc, at zero extra cost.
    """
    st = g.add_shape("Stroke")
    st.color.value = color
    st.width.value = width
    st.cap = st.Cap.RoundCap
    st.join = st.Join.RoundJoin
    return st


def _draw_part(layer, length: float, part: Part, *, marker: float = 0.0) -> None:
    """Draw one bone's skin into its layer, in local space (origin = the joint).

    Two looks. **Filled** (default): a capsule plus a disc at the joint — the disc
    stops two capsules meeting at an angle from leaving a notch on every bend. A
    `head` part is a filled ellipse at the tip; a `tip` is a hand/paw dot.
    **Stroke** (`part.stroke`, the stick-figure look): an open pen line from origin
    to tip, `width` its weight; a `head` becomes a ring (outline), a dot (solid) or
    a fill, per `head_style`. Either way it is static — the transform does the moving.
    """
    g = layer.add_shape("Group")

    if part.stroke:
        if part.head:
            hw, hh = part.head
            if part.head_style == "ring":
                _stroke_styler(g, part.color, part.width)
            else:  # "dot" / "fill": a solid head
                g.add_shape("Fill").color.value = part.color
            e = g.add_shape("Ellipse")
            e.size.value = utils.Size(hw, hh)
            e.position.value = utils.Point(length, 0.0)
        else:
            _stroke_styler(g, part.color, part.width)
            p = g.add_shape("Path")
            bez = p.shape.value
            zero = utils.Point(0.0, 0.0)
            bez.add_point(utils.Point(0.0, 0.0), zero, zero)
            bez.line_to(utils.Point(max(length, 1.0), 0.0))  # open path: no close()
            p.shape.value = bez
            if part.tip > 0:  # a hand/paw dot wider than the line
                dot = layer.add_shape("Group")
                dot.add_shape("Fill").color.value = part.color
                t = dot.add_shape("Ellipse")
                t.size.value = utils.Size(part.tip * 2, part.tip * 2)
                t.position.value = utils.Point(length, 0.0)
        if marker > 0:
            _draw_marker(layer, length, marker)
        return

    g.add_shape("Fill").color.value = part.color

    if part.head:
        hw, hh = part.head
        e = g.add_shape("Ellipse")
        e.size.value = utils.Size(hw, hh)
        e.position.value = utils.Point(length, 0.0)
        return

    w = part.width
    r = g.add_shape("Rect")
    r.size.value = utils.Size(max(length, 1.0), w)
    r.position.value = utils.Point(length / 2.0, 0.0)
    r.rounded.value = w / 2.0

    cap = g.add_shape("Ellipse")
    cap.size.value = utils.Size(w, w)
    cap.position.value = utils.Point(0.0, 0.0)

    if part.tip > 0:
        t = g.add_shape("Ellipse")
        t.size.value = utils.Size(part.tip * 2, part.tip * 2)
        t.position.value = utils.Point(length, 0.0)

    if marker > 0:
        _draw_marker(layer, length, marker)


def _draw_marker(layer, length: float, marker: float) -> None:
    """A contact marker dot riding the bone tip — static, zero keys."""
    mg = layer.add_shape("Group")
    mg.add_shape("Fill").color.value = "#e8543f"
    e = mg.add_shape("Ellipse")
    e.size.value = utils.Size(marker * 2, marker * 2)
    e.position.value = utils.Point(length, 0.0)


def bake_rig(
    scene: Scene,
    body: Body,
    pose_fn: Callable[[float], Pose],
    *,
    frames: int,
    first: int = 0,
    color: str | None = None,
    thickness: float | None = None,
    joint_color: str | None = "#e8543f",
    joint_radius: float = 0.0,
    layer_name: str = "character",
    stats: dict | None = None,
    layers_out: dict | None = None,
) -> model.shapes.Layer:
    """Build a parented bone-layer rig and key it sparsely.

    `layers_out`, if given, is filled with {joint_name: layer} so callers can
    parent extra art (face attachments, props held in a hand) to specific bones.

    Sampling happens once; every local channel (root path, per-bone local
    rotation) is reduced independently. Local channels are the whole trick: in
    world space every bone moves every frame because the body does, but in
    parent space a walking skeleton is nearly periodic rotations over *static*
    offsets — which is what makes ~10 keys describe what took ~200.

    `first` bakes only `first..frames`, for a character that appears partway
    through. It is not a micro-optimisation: a film of six shots bakes every
    creature across the *whole* timeline otherwise, and the reducer's greedy
    refinement is worse than linear in the sample count — a four-creature reel
    with onion-skin ghosts blew a three-minute budget baking frames nobody can
    see. Keys are written at `first + i`, so the clip lands where it belongs.
    """
    rig = body.rig

    # ---- sample local channels (no Qt in this part)
    root_pos: list[Vec2] = []
    root_rot: list[float] = []
    local_rot: dict[str, list[float]] = {name: [] for name in rig.joints}
    for f in range(first, frames + 1):
        pose = pose_fn(float(f))
        root_pos.append(pose.root)
        j_root = rig.joints[rig.root_name]
        root_rot.append(pose.root_angle + pose.angles.get(rig.root_name, 0.0) + j_root.rest_angle)
        for name, j in rig.joints.items():
            if name != rig.root_name:
                local_rot[name].append(pose.angles.get(name, 0.0) + j.rest_angle)

    # ---- create layers in painter's order (last-added paints on top)
    root_layer = scene.layer(layer_name)
    layer_of: dict[str, model.shapes.Layer] = {rig.root_name: root_layer}

    skinless = [n for n in rig.joints
                if n != rig.root_name and n not in body.bones]
    for name in (*skinless, *body.bones):
        if name == rig.root_name:
            continue
        lay = scene.layer(f"{layer_name}.{name}")
        layer_of[name] = lay

    # ---- parent the hierarchy (order-independent; draw order stays stacking)
    for name, lay in layer_of.items():
        parent = rig.joints[name].parent
        if parent is not None:
            lay.parent = layer_of[parent]

    # ---- skins (static, local space)
    # replace() keeps stroke/head/tip/z/head_style intact under a colour/thickness
    # override — a plain Part() reconstruction used to silently drop them, which
    # turned a stick figure back into filled capsules the moment you recoloured it.
    for name in body.bones:
        j = rig.joints[name]
        part = body.parts.get(name, Part())
        if color is not None or thickness is not None:
            part = replace(
                part,
                width=thickness if thickness is not None else part.width,
                color=color if color is not None else part.color,
            )
        marker = joint_radius if (joint_color and joint_radius > 0
                                  and (j.contact or j.rolling)) else 0.0
        _draw_part(layer_of[name], j.length, part, marker=marker)

    # ---- keys
    n_keys = 0
    n_keys += _write_point(root_layer.transform.position,
                           reduce_point(root_pos, tol=TOL_PX), offset=first)
    n_keys += _write_scalar(root_layer.transform.rotation,
                            reduce_scalar(root_rot, tol=TOL_DEG), offset=first)

    # A contact chain is the contact joint and everything above it to the root —
    # the bones whose angles decide where a planted foot actually lands.
    contact_chain: set[str] = set()
    for name, j in rig.joints.items():
        if j.contact or j.rolling:
            contact_chain.update(rig.chain(name))

    for name, lay in layer_of.items():
        if name == rig.root_name:
            continue
        j = rig.joints[name]
        off = j.offset if j.offset is not None else Vec2(rig.joints[j.parent].length, 0.0)
        lay.transform.position.value = utils.Point(off.x, off.y)  # static: the whole point
        tol = TOL_DEG_LEG if name in contact_chain else TOL_DEG
        n_keys += _write_scalar(lay.transform.rotation,
                                reduce_scalar(local_rot[name], tol=tol),
                                offset=first)

    if stats is not None:
        stats["keyframes"] = n_keys
        stats["layers"] = len(layer_of)
    if layers_out is not None:
        layers_out.update(layer_of)

    return root_layer


def bake_samples(
    scene: Scene,
    samples,
    *,
    shape: str = "Ellipse",
    size: Vec2 | None = None,
    color: str = "#e8543f",
    layer_name: str = "object",
    stats: dict | None = None,
) -> model.shapes.Layer:
    """Bake a `motion.Sample` list (ball, wheel, leaf) with reduced keys."""
    samples = list(samples)
    size = size or Vec2(80.0, 80.0)
    lay = scene.layer(layer_name)
    g = lay.add_shape("Group")
    g.add_shape("Fill").color.value = color
    s = g.add_shape(shape)
    s.size.value = utils.Size(size.x, size.y)

    pos = [smp.pos for smp in samples]
    rot = [smp.angle for smp in samples]
    scl = [smp.scale for smp in samples]

    off = _clip_offset(samples)
    n = _write_point(g.transform.position, reduce_point(pos, tol=TOL_PX), offset=off)
    n += _write_scalar(g.transform.rotation, reduce_scalar(rot, tol=TOL_DEG),
                       offset=off)
    # scale: written as Vector2D so it actually lands (a Point no-ops on a
    # QVector2D property). Linear keys — the ease=False reducer adds keys until
    # linear segments are within tolerance; eased scale is a later refinement.
    n += _write_scale(g.transform.scale,
                      reduce_point(scl, tol=TOL_SCALE, ease=False),
                      transitions=False, offset=off)
    if stats is not None:
        stats["keyframes"] = n

    return lay


def bake_prop_samples(
    scene: Scene,
    prop_data: dict,
    samples,
    *,
    scale: float | tuple[float, float] = 1.0,
    layer_name: str = "prop_object",
    stats: dict | None = None,
) -> model.shapes.Layer:
    """Bake a *multi-shape* prop onto a motion path.

    `bake_samples` animates one ellipse; `add_prop` draws many shapes but nails
    them to the ground. Anything made of more than one shape that also moves — a
    two-tone bar magnet, a paperclip, a compass needle, a car — falls between
    them, so it lands here.

    The prop's shapes are drawn once into a group in LOCAL coordinates and the
    group's transform carries the motion, which is the cut-out puppet trick the
    bone baker already uses: N shapes cost one animated transform, not N. Local
    origin is the prop's anchor, so `Sample.angle` rotates about it — author the
    shapes around y=0 for a thing that spins about its middle (a needle), or
    above it for a thing that pivots on the ground.
    """
    from .props import draw_prop

    samples = list(samples)
    lay = scene.layer(layer_name)
    g = lay.add_shape("Group")
    draw_prop(g, prop_data, x=0.0, ground_y=0.0, scale=scale)

    off = _clip_offset(samples)
    n = _write_point(g.transform.position, reduce_point([s.pos for s in samples],
                                                        tol=TOL_PX), offset=off)
    n += _write_scalar(g.transform.rotation, reduce_scalar([s.angle for s in samples],
                                                           tol=TOL_DEG), offset=off)
    n += _write_scale(g.transform.scale,
                      reduce_point([s.scale for s in samples], tol=TOL_SCALE,
                                   ease=False),
                      transitions=False, offset=off)
    if stats is not None:
        stats["keyframes"] = n
    return lay


def bake_effect(
    scene: Scene,
    fx_data: dict,
    *,
    x: float,
    y: float,
    start: float,
    layer_name: str = "fx",
) -> model.shapes.Layer:
    """Bake a short-lived effect: shapes drawn once, then grown, spun and faded.

    The effect pops in on frame `start`, plays its `lifespan`, and is opacity-zero
    before and after — so it costs nothing on every other frame and never lingers.
    The grow envelope rides `transform.scale` (a `Vector2D`, only writable since the
    binding fix); the fade rides the layer opacity. Position is static: an effect
    happens *at* a spot (a foot plant, a fist's contact), it does not travel.
    """
    from .props import draw_prop

    life = float(fx_data.get("lifespan", 6))
    g0, g1 = fx_data.get("grow", [1.0, 1.0])
    fade = bool(fx_data.get("fade", True))
    spin = float(fx_data.get("spin", 0.0))
    end = float(start) + life

    lay = scene.layer(layer_name)
    g = lay.add_shape("Group")
    draw_prop(g, {"shapes": fx_data["shapes"]}, x=0.0, ground_y=0.0)

    g.transform.position.value = utils.Point(float(x), float(y))
    g.transform.scale.set_keyframe(float(start), utils.Vector2D(float(g0), float(g0)))
    g.transform.scale.set_keyframe(end, utils.Vector2D(float(g1), float(g1)))
    if spin:
        g.transform.rotation.set_keyframe(float(start), 0.0)
        g.transform.rotation.set_keyframe(end, spin)

    # Opacity: invisible until `start`, pop to full, fade (or hold) to nothing by
    # `end`, and the final keyframe holds it at zero forever after.
    op = lay.opacity
    pre = max(float(start) - 1.0, 0.0)
    if pre < float(start):
        op.set_keyframe(pre, 0.0)
    op.set_keyframe(float(start), 1.0)
    op.set_keyframe(end, 0.0 if fade else 1.0)
    return lay
