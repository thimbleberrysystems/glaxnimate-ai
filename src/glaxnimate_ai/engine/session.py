"""A live animation session: a document, the things in it, and a sandbox to build them.

The LLM writes short Python against the `cartoon` library rather than making one
tool call per keyframe. Animation is repetitive by nature — a gait is one limb
motion looped with offsets, a bounce is an arc repeated with decay — so it is
*code-shaped*. One script saying "six bounces, each 15% lower" replaces about a
hundred and twenty `set_keyframe` round-trips.

The sandbox is a guardrail against model **mistakes** — runaway loops, missing
imports — and not a security boundary against a malicious actor. This is a local,
single-user tool and pretending otherwise would be theatre.

The one thing scripts must do is *register* what they build (`add_character`,
`add_object`). That is what lets the critic stack run automatically afterwards:
the session knows there is a dog with these limbs on this ground line, so it can
check its feet without being told twice.
"""

from __future__ import annotations

import io
import signal
import traceback
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from glaxnimate import environment

from ..cartoon import actions, assets, geometry, motion, presets, principles, rig
from ..cartoon.gait import Gait, pose_at
from ..cartoon.presets import Body
from . import scene_doc as SD
from .bake import Scene, bake_rig, bake_samples

__all__ = ["Character", "Session", "SessionStore", "ScriptResult"]


@dataclass(slots=True)
class Character:
    """A rig the script put on stage, remembered so the critic can inspect it."""

    name: str
    body: Body
    gait: Gait | None
    pose_fn: Any
    #: (upper, lower) joint pairs for the over-extension lint. Derived from the
    #: gait at creation but stored flat, so a scene reloaded from disk (where
    #: gait objects no longer exist) keeps the check.
    limb_pairs: list = field(default_factory=list)
    #: attachment name -> its layer, when the character has a face.
    face_layers: dict = field(default_factory=dict)
    #: the attachment visible from frame 0.
    face_default: str = ""
    #: (frame, attachment) history, for inspection and (later) the scene doc.
    expressions: list = field(default_factory=list)
    _face_keyed: bool = False


@dataclass(slots=True)
class ScriptResult:
    ok: bool
    stdout: str
    error: str | None = None

    def format(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.error:
            parts.append(self.error.rstrip())
        return "\n".join(parts) or ("ok" if self.ok else "failed")


class _Timeout(Exception):
    pass


@dataclass
class Session:
    """One document under construction."""

    doc_id: str
    scene: Scene
    ground_y: float
    frames: int
    characters: list[Character] = field(default_factory=list)
    #: (name, samples, radius) for every non-rig object, so the critic sees them.
    objects: list[tuple] = field(default_factory=list)
    #: The scene as data — everything needed to rebuild this session from disk.
    doc: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        doc_id: str,
        *,
        width: int = 960,
        height: int = 540,
        frames: int = 48,
        fps: float = 24.0,
        ground_y: float | None = None,
    ) -> Session:
        scene = Scene.create(width, height, frames=frames, fps=fps)
        _pinned_scenes.append(scene)  # see note on _pinned_scenes above
        gy = ground_y if ground_y is not None else height * 0.87
        doc = SD.empty_doc(width=width, height=height, frames=frames, fps=fps, ground_y=gy)
        return cls(doc_id, scene, gy, frames, doc=doc)

    # ------------------------------------------------------ the script's API
    def _add_character(
        self,
        body: Body,
        gait: Gait,
        *,
        x: float = 80.0,
        name: str = "character",
        color: str | None = None,
        thickness: float | None = None,
        face: str | dict | None = None,
    ) -> Character:
        """Bake a rig into the document and register it with the critic.

        `color`/`thickness` default to None so the body's own skin is used. Pass
        them to flatten the character to a single colour — useful for a silhouette
        check, useless for a cartoon.
        """

        def pose_fn(t: float):
            # No hip_height here on purpose: pose_at reads the gait's own
            # ride_height, so a crouched gallop stays crouched. Passing
            # body.hip_height would stand the figure back up and over-extend the
            # legs — the exact bug that made the gallop face-plant.
            return pose_at(body.rig, gait, t, ground_y=self.ground_y, body_x0=x)

        return self._bake_character(body, pose_fn, gait=gait, name=name,
                                    color=color, thickness=thickness, face=face)

    def _bake_character(
        self, body: Body, pose_fn, *, gait: Gait | None, name: str,
        color: str | None = None, thickness: float | None = None,
        face: str | dict | None = None, record: bool = True,
        poses: list | None = None, face_data: dict | None = None,
        expressions: list | None = None,
    ) -> Character:
        """Bake + register + record. Every character path funnels through here so
        the scene document always matches what is on the canvas."""
        layers: dict = {}
        bake_rig(
            self.scene, body, pose_fn, frames=self.frames,
            color=color, thickness=thickness, layer_name=name, layers_out=layers,
        )
        limb_pairs = [(li.upper, li.lower) for li in gait.limbs] if gait else []
        ch = Character(name, body, gait, pose_fn, limb_pairs=limb_pairs)

        if face_data is None and face is not None:
            face_data = assets.load_face(face) if isinstance(face, str)                 else assets.face_validate(face)
        if face_data is not None:
            self._attach_face(ch, layers, face_data)
        self.characters.append(ch)

        if record:
            self.doc["characters"].append({
                "name": name,
                "body": assets.body_to_data(body),
                "poses": poses if poses is not None
                         else SD.sample_poses(pose_fn, self.frames),
                "limbs": [list(x) for x in limb_pairs],
                "color": color, "thickness": thickness,
                "face": face_data,
                "expressions": [],
            })
        if expressions:
            replaying = not record
            for frame, att in expressions:
                self._set_expression(ch, att, frame, _record=not replaying)
        return ch

    def _attach_face(self, ch: Character, layers: dict, data: dict) -> None:
        """Mount a face's attachments on the character's slot bone.

        One layer per attachment, parented to the slot's bone so it rides the
        head for free. The layer's rotation compensates the bone's rest-pose
        world angle, so face art is authored screen-aligned (x = facing
        direction, y = down) and reads correctly on an upright human head and a
        tilted dog head alike. Visibility is a radio button flipped by keying
        opacity with hold steps — see set_expression.
        """
        from glaxnimate import utils

        from ..cartoon.rig import Pose
        from . import props as P

        slot_name = data["slot"]
        slot = ch.body.slots.get(slot_name)
        if slot is None:
            raise ValueError(
                f"body has no slot {slot_name!r}; its slots are {sorted(ch.body.slots)}"
            )
        bone = slot["bone"]
        rest = ch.body.rig.solve(Pose())

        for i, (att, shapes) in enumerate(data["attachments"].items()):
            lay = self.scene.layer(f"{ch.name}.face.{att}")
            lay.parent = layers[bone]
            off = slot.get("offset", [0, 0])
            lay.transform.position.value = utils.Point(off[0], off[1])
            lay.transform.rotation.value = -rest[bone].angle
            P.draw_prop(lay, {"shapes": shapes}, x=0.0, ground_y=0.0)
            lay.opacity.value = 1.0 if i == 0 else 0.0
            ch.face_layers[att] = lay
            if i == 0:
                ch.face_default = att

    def _set_expression(self, character, attachment: str, frame: float,
                        _record: bool = True) -> str:
        """Swap the visible face attachment at `frame` (a hold key — no crossfade).

        Radio-button semantics by construction: every attachment layer is keyed
        at the frame, exactly one at full opacity. The linter's one-visible-per-
        slot rule can therefore only be violated by hand-editing the file.
        """
        from glaxnimate import model

        ch = character if isinstance(character, Character) else next(
            (c for c in self.characters if c.name == character), None
        )
        if ch is None:
            raise ValueError(
                f"no character {character!r}; have {[c.name for c in self.characters]}"
            )
        if not ch.face_layers:
            raise ValueError(
                f"{ch.name} has no face - pass face=... to add_character first"
            )
        if attachment not in ch.face_layers:
            raise ValueError(
                f"no attachment {attachment!r}; this face has {sorted(ch.face_layers)}"
            )

        def key(lay, f, v):
            lay.opacity.set_keyframe(float(f), v)
            tr = model.KeyframeTransition()
            tr.hold = True
            lay.opacity.set_transition(float(f), tr)

        if not ch._face_keyed:
            # Backfill frame 0 first: a property's value before its first keyframe
            # is that keyframe's value, so keying only at `frame` would silently
            # change what was visible from the start.
            for att, lay in ch.face_layers.items():
                key(lay, 0.0, 1.0 if att == ch.face_default else 0.0)
            ch._face_keyed = True

        for att, lay in ch.face_layers.items():
            key(lay, frame, 1.0 if att == attachment else 0.0)
        ch.expressions.append((float(frame), attachment))
        if _record:
            for rec in self.doc.get("characters", []):
                if rec["name"] == ch.name:
                    rec["expressions"].append([float(frame), attachment])
                    break
        return f"{ch.name} shows {attachment!r} from frame {frame:g}"

    def _add_action(
        self,
        body: Body,
        pose_fn,
        *,
        name: str = "character",
        color: str | None = None,
        thickness: float | None = None,
    ) -> Character:
        """Bake a character driven by an arbitrary pose function (a jump, a wave).

        Actions aren't locomotion, so there is no gait — but the character is still
        registered so the linter and diagnostics can inspect it (contact slip,
        joint integrity, bounds all still apply).
        """
        return self._bake_character(body, pose_fn, gait=None, name=name,
                                    color=color, thickness=thickness)

    _SCENERY = ("sky", "ground", "house", "school", "tree", "cloud", "sun")

    def _scenery(self, template: str, *, layer_name: str = "backdrop",
                 record: bool = True, **params):
        """Draw a parametric backdrop template (sky, ground, house, tree ...).

        v1's run_script had no scenery access at all — the examples drew their
        backdrops in Python *around* the session, which meant an MCP-driven model
        could animate a man but never give him a street to walk down. Recorded in
        the scene doc like everything else.
        """
        from . import props as P

        if template not in self._SCENERY:
            raise ValueError(f"unknown scenery {template!r}; have {self._SCENERY}")
        if record:
            self.doc["scenery"].append(
                {"template": template, "params": dict(params), "layer": layer_name}
            )
        lay = self.scene.layer(layer_name)
        fn = getattr(P, template)
        params = dict(params)  # consumed below; the doc kept the original
        if template == "sky":
            fn(self.scene, lay, **params)
        elif template == "ground":
            fn(self.scene, lay, params.pop("y", self.ground_y), **params)
        elif template in ("house", "school", "tree"):
            fn(lay, params.pop("x", 100.0), params.pop("ground_y", self.ground_y), **params)
        else:  # cloud, sun
            fn(lay, params.pop("x", 100.0), params.pop("y", 80.0), **params)
        return lay

    def _add_prop(self, prop, *, x: float = 0.0, scale: float = 1.0,
                  layer_name: str = "props"):
        """Place a data prop (a dict or an asset name) on the ground line."""
        from . import props as P

        data = assets.load_prop(prop) if isinstance(prop, str) else assets.prop_validate(prop)
        lay = self.scene.layer(layer_name)
        P.draw_prop(lay, data, x=x, ground_y=self.ground_y, scale=scale)
        self.doc["props"].append({"data": data, "x": x, "scale": scale,
                                  "layer": layer_name})
        return lay

    def _add_object(self, samples, *, record: bool = True, **kw):
        # Register the samples so the critic can see the object too — in v1 only
        # rig characters were checkable and a ball through the floor was invisible.
        samples = list(samples)
        size = kw.get("size")
        radius = (size.y / 2.0) if size is not None else 40.0
        name = kw.get("layer_name", f"object{len(self.objects)}")
        self.objects.append((name, samples, radius))
        if record:
            self.doc["objects"].append({
                "name": name,
                "samples": SD.samples_to_data(samples),
                "shape": kw.get("shape", "Ellipse"),
                "size": [size.x, size.y] if size is not None else None,
                "color": kw.get("color", "#e8543f"),
            })
        return bake_samples(self.scene, samples, **kw)

    def _add_chaser(
        self,
        body: Body,
        gait_name: str,
        target,
        *,
        x: float = 60.0,
        gap: float = 40.0,
        cycle_frames: float = 16.0,
        name: str = "chaser",
        **char_kw,
    ) -> Character:
        """A character paced to chase a moving target and end `gap` px behind it.

        This is the fix for the coordination gap: no per-character metric can see
        that a chaser is losing the race, because each character is individually
        fine — it is the *relationship* that is wrong. `pace` sizes the gait so the
        character actually arrives where the target ends up.

        `target` is anything with `.pos.x` samples (a `motion.*` result) or a plain
        final x-coordinate.
        """
        if hasattr(target, "__iter__") and not isinstance(target, (int, float)):
            target_end = max(smp.pos.x for smp in target)
        else:
            target_end = float(target)

        distance = (target_end - gap) - x
        if distance <= 0:
            raise ValueError(
                f"the chaser starts at x={x} but only needs to reach "
                f"{target_end - gap:.0f} — it is already there. Start it further back."
            )
        gait = presets.pace(
            body, gait_name, distance=distance, frames=self.frames, cycle_frames=cycle_frames
        )
        return self._add_character(body, gait, x=x, name=name, **char_kw)

    def namespace(self) -> dict[str, Any]:
        """What a script sees. Deliberately small — this is the LLM's vocabulary."""
        ns: dict[str, Any] = {
            # the cartoon library
            "presets": presets,
            "motion": motion,
            "actions": actions,
            "principles": principles,
            "geometry": geometry,
            "rig": rig,
            "Vec2": geometry.Vec2,
            "human": presets.human,
            "biped": presets.biped,
            "quadruped": presets.quadruped,
            "make_gait": presets.make_gait,
            "pace": presets.pace,
            "pose_at": pose_at,  # build a base pose_fn to wrap with actions.trail
            # the asset library: the vocabulary that grows without code changes
            "assets": assets,
            "load_body": assets.load_body,
            "save_body": assets.save_body,
            "body_from_data": assets.body_from_data,
            "load_gait": assets.load_gait,
            "register_gait": assets.register_gait,
            "load_prop": assets.load_prop,
            "add_prop": self._add_prop,
            "load_face": assets.load_face,
            "set_expression": self._set_expression,
            "scenery": self._scenery,
            # the stage
            "add_character": self._add_character,
            "add_object": self._add_object,
            "add_chaser": self._add_chaser,
            "add_action": self._add_action,
            "scene": self.scene,
            "ground": self.ground_y,
            "frames": self.frames,
            "width": self.scene.comp.width,
            "height": self.scene.comp.height,
        }
        return ns

    # ------------------------------------------------------------ persistence
    def save(self) -> None:
        """Write the scene document. Called automatically after every successful
        script — a crash or restart after this point loses nothing."""
        SD.save_doc(self.doc_id, self.doc)

    @classmethod
    def replay(cls, doc_id: str) -> Session:
        """Rebuild a live session from its saved document.

        Characters come back as sampled-pose lookups (exact — poses are only ever
        read at integer frames), objects from their sample rows, scenery and props
        from their recorded calls. The result bakes, lints and renders identically
        to the session that saved it.
        """
        doc = SD.load_doc(doc_id)
        c = doc["canvas"]
        session = cls.create(
            doc_id, width=c["width"], height=c["height"],
            frames=c["frames"], fps=c["fps"], ground_y=doc["ground_y"],
        )
        session.doc = doc  # the replayed doc IS the doc; don't re-record

        for sc in doc["scenery"]:
            session._scenery(sc["template"], layer_name=sc["layer"],
                             record=False, **sc["params"])
        for pr in doc["props"]:
            lay = session.scene.layer(pr.get("layer", "props"))
            from . import props as P
            P.draw_prop(lay, pr["data"], x=pr["x"],
                        ground_y=session.ground_y, scale=pr.get("scale", 1.0))
        for rec in doc["characters"]:
            body = assets.body_from_data(rec["body"])
            pose_fn = SD.pose_lookup(rec["poses"])
            ch = session._bake_character(
                body, pose_fn, gait=None, name=rec["name"],
                color=rec.get("color"), thickness=rec.get("thickness"),
                face_data=rec.get("face"), record=False,
                expressions=[(f, a) for f, a in rec.get("expressions", [])],
            )
            ch.limb_pairs = [tuple(x) for x in rec.get("limbs", [])]
        for ob in doc["objects"]:
            from ..cartoon.geometry import Vec2
            size = Vec2(*ob["size"]) if ob.get("size") else None
            session._add_object(
                SD.samples_from_data(ob["samples"]), record=False,
                shape=ob.get("shape", "Ellipse"), size=size,
                color=ob.get("color", "#e8543f"), layer_name=ob["name"],
            )
        return session

    # ---------------------------------------------------------------- running
    def run(self, code: str, *, timeout: int = 20) -> ScriptResult:
        """Execute a script against this session, capturing output and tracebacks.

        The traceback goes straight back to the model, which is the point: it reads
        the error and fixes itself, which is far cheaper than a round of rendering.
        """

        def _alarm(_sig, _frm):
            raise _Timeout(f"script exceeded {timeout}s")

        ns = self.namespace()
        buf = io.StringIO()
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
        try:
            with redirect_stdout(buf):
                exec(compile(code, "<script>", "exec"), ns)  # noqa: S102 - by design
            self.save()
            return ScriptResult(True, buf.getvalue())
        except _Timeout as e:
            return ScriptResult(False, buf.getvalue(), f"TimeoutError: {e}")
        except Exception:
            # Trim our own frames: the model only wants to see its own mistake.
            tb = traceback.format_exc(limit=6)
            return ScriptResult(False, buf.getvalue(), tb)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


# The one Glaxnimate environment for the whole process. This must be a process
# singleton, not per-store: entering Headless() tears down and re-creates Qt's
# application object, and any still-alive document from an earlier context — in
# particular one with parented layers, which hold QObject connections between
# them — dangles and segfaults on the next touch. Found the hard way: three
# SessionStores in one pytest run crashed the interpreter.
_env_stack: ExitStack | None = None

# Documents are pinned for the life of the process. Layer parenting creates
# QObject connections *between* layers, and letting Python's GC destroy a
# document at an arbitrary moment — possibly while another document is pushing
# onto its undo stack — segfaults inside Qt (observed in QUndoStack::push).
# Qt object lifetime is not something to outsmart from Python; a scene is a few
# MB and sessions per process are few, so we simply never free them.
_pinned_scenes: list = []


def _shared_environment() -> None:
    global _env_stack
    if _env_stack is None:
        _env_stack = ExitStack()
        _env_stack.enter_context(environment.Headless())


class SessionStore:
    """Every document in play, sharing the process-wide Glaxnimate environment."""

    def __init__(self) -> None:
        _shared_environment()
        self._sessions: dict[str, Session] = {}
        self._n = 0

    def create(self, **kw) -> Session:
        self._n += 1
        doc_id = f"doc{self._n}"
        s = Session.create(doc_id, **kw)
        self._sessions[doc_id] = s
        return s

    def get(self, doc_id: str) -> Session:
        try:
            return self._sessions[doc_id]
        except KeyError:
            pass
        try:
            session = Session.replay(doc_id)  # a restart loses nothing
        except FileNotFoundError:
            known = ", ".join(self._sessions) or "none"
            raise KeyError(
                f"no document {doc_id!r} (open: {known}; "
                f"saved: {SD.list_docs() or 'none'})"
            ) from None
        self._sessions[doc_id] = session
        return session

    def close(self) -> None:
        """Kept for API compatibility. The Qt environment is process-wide and
        deliberately never torn down: Qt shutdown with live documents is exactly
        the crash this design removes."""
