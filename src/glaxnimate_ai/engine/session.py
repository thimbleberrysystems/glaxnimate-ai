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

from ..cartoon import actions, geometry, motion, presets, principles, rig
from ..cartoon.gait import Gait, pose_at
from ..cartoon.presets import Body
from .bake import Scene, bake_rig, bake_samples

__all__ = ["Character", "Session", "SessionStore", "ScriptResult"]


@dataclass(slots=True)
class Character:
    """A rig the script put on stage, remembered so the critic can inspect it."""

    name: str
    body: Body
    gait: Gait | None
    pose_fn: Any


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
        return cls(doc_id, scene, ground_y if ground_y is not None else height * 0.87, frames)

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

        bake_rig(
            self.scene, body, pose_fn, frames=self.frames,
            color=color, thickness=thickness, layer_name=name,
        )
        ch = Character(name, body, gait, pose_fn)
        self.characters.append(ch)
        return ch

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
        bake_rig(self.scene, body, pose_fn, frames=self.frames,
                 color=color, thickness=thickness, layer_name=name)
        ch = Character(name, body, None, pose_fn)
        self.characters.append(ch)
        return ch

    def _add_object(self, samples, **kw):
        # Register the samples so the critic can see the object too — in v1 only
        # rig characters were checkable and a ball through the floor was invisible.
        samples = list(samples)
        size = kw.get("size")
        radius = (size.y / 2.0) if size is not None else 40.0
        self.objects.append((kw.get("layer_name", f"object{len(self.objects)}"),
                             samples, radius))
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
            known = ", ".join(self._sessions) or "none"
            raise KeyError(f"no document {doc_id!r} (open: {known})") from None

    def close(self) -> None:
        """Kept for API compatibility. The Qt environment is process-wide and
        deliberately never torn down: Qt shutdown with live documents is exactly
        the crash this design removes."""
