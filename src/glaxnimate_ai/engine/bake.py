"""Turn poses into a Glaxnimate document.

The pose engine (`cartoon/`) is pure Python and knows nothing about Glaxnimate.
This module is the only place the two meet: it samples a pose per frame and
writes keyframes. Keeping the seam this thin is deliberate — it means the rig
maths, the gaits and the linter are all testable with no Qt in the room, and
swapping the renderer later would touch one file.

A bone is drawn as a rounded rect in a Group whose transform carries the joint's
world position and angle. The rect is offset by half its length so the group's
origin sits on the joint, which makes rotation happen about the joint rather
than about the middle of the bone — the difference between an elbow and a
propeller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from glaxnimate import model, utils

from ..cartoon.geometry import Vec2
from ..cartoon.presets import Body, Part
from ..cartoon.rig import Pose

__all__ = ["Scene", "bake_rig", "bake_samples"]


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


def _bone_group(parent, length: float, part: Part) -> model.shapes.Group:
    """Draw one skinned bone, anchored at the joint and running along local +x.

    A capsule plus a disc at the joint. The disc is what makes an elbow look like
    an elbow: without it, two rounded rects meeting at an angle leave a visible
    notch on the outside of every bend.

    If the part declares a `head`, it is an ellipse at the bone's tip instead —
    a head is not a rod.
    """
    g = parent.add_shape("Group")
    g.add_shape("Fill").color.value = part.color

    if part.head:
        hw, hh = part.head
        e = g.add_shape("Ellipse")
        e.size.value = utils.Size(hw, hh)
        e.position.value = utils.Point(length, 0.0)
        return g

    w = part.width
    r = g.add_shape("Rect")
    r.size.value = utils.Size(max(length, 1.0), w)
    # Offset by half the length so the group's origin is the joint, not the bone's
    # centre — otherwise rotation swings the bone like a propeller.
    r.position.value = utils.Point(length / 2.0, 0.0)
    r.rounded.value = w / 2.0

    joint_cap = g.add_shape("Ellipse")
    joint_cap.size.value = utils.Size(w, w)
    joint_cap.position.value = utils.Point(0.0, 0.0)

    if part.tip > 0:  # a hand, a paw, a nose
        t = g.add_shape("Ellipse")
        t.size.value = utils.Size(part.tip * 2, part.tip * 2)
        t.position.value = utils.Point(length, 0.0)

    return g


def bake_rig(
    scene: Scene,
    body: Body,
    pose_fn: Callable[[float], Pose],
    *,
    frames: int,
    color: str | None = None,
    thickness: float | None = None,
    joint_color: str | None = "#e8543f",
    joint_radius: float = 0.0,
    layer_name: str = "character",
) -> model.shapes.Layer:
    """Sample `pose_fn` once per frame and keyframe every bone.

    Bones are skinned from `body.parts`. `color`/`thickness` override the skin
    entirely, which is what you want for a silhouette or a debug pass.
    """
    lay = scene.layer(layer_name)
    groups: dict[str, model.shapes.Group] = {}

    for name in body.bones:
        joint = body.rig.joints[name]
        part = body.parts.get(name, Part())
        if color is not None or thickness is not None:
            part = Part(
                width=thickness if thickness is not None else part.width,
                color=color if color is not None else part.color,
                head=part.head,
                tip=part.tip,
            )
        groups[name] = _bone_group(lay, joint.length, part)
        groups[name].name = name

    if joint_color and joint_radius > 0:
        for name in body.rig.contacts:
            g = lay.add_shape("Group")
            g.name = f"{name}__contact"
            g.add_shape("Fill").color.value = joint_color
            e = g.add_shape("Ellipse")
            e.size.value = utils.Size(joint_radius * 2, joint_radius * 2)
            groups[f"{name}__contact"] = g

    for f in range(frames + 1):
        world = body.rig.solve(pose_fn(float(f)))
        for name, g in groups.items():
            if name.endswith("__contact"):
                jf = world[name.removesuffix("__contact")]
                g.transform.position.set_keyframe(f, utils.Point(jf.tip.x, jf.tip.y))
            else:
                jf = world[name]
                g.transform.position.set_keyframe(f, utils.Point(jf.origin.x, jf.origin.y))
                g.transform.rotation.set_keyframe(f, jf.angle)

    return lay


def bake_samples(
    scene: Scene,
    samples,
    *,
    shape: str = "Ellipse",
    size: Vec2 | None = None,
    color: str = "#e8543f",
    layer_name: str = "object",
) -> model.shapes.Layer:
    """Keyframe a list of `motion.Sample` (position, scale, rotation) onto one shape.

    This is the non-character path: balls, wheels, leaves, logos.
    """
    size = size or Vec2(80.0, 80.0)
    lay = scene.layer(layer_name)
    g = lay.add_shape("Group")
    g.add_shape("Fill").color.value = color
    s = g.add_shape(shape)
    s.size.value = utils.Size(size.x, size.y)

    for smp in samples:
        g.transform.position.set_keyframe(smp.frame, utils.Point(smp.pos.x, smp.pos.y))
        g.transform.rotation.set_keyframe(smp.frame, smp.angle)
        g.transform.scale.set_keyframe(smp.frame, utils.Point(smp.scale.x, smp.scale.y))

    return lay
