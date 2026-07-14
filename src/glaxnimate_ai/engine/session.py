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

from ..cartoon import geometry, motion, presets, principles, rig
from ..cartoon.gait import Gait, pose_at
from ..cartoon.presets import Body
from .bake import Scene, bake_rig, bake_samples

__all__ = ["Character", "Session", "SessionStore", "ScriptResult"]


@dataclass(slots=True)
class Character:
    """A rig the script put on stage, remembered so the critic can inspect it."""

    name: str
    body: Body
    gait: Gait
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
        return cls(doc_id, scene, ground_y if ground_y is not None else height * 0.87, frames)

    # ------------------------------------------------------ the script's API
    def _add_character(
        self,
        body: Body,
        gait: Gait,
        *,
        x: float = 80.0,
        name: str = "character",
        color: str = "#3b3b46",
        thickness: float = 14.0,
    ) -> Character:
        """Bake a rig into the document and register it with the critic."""

        def pose_fn(t: float):
            return pose_at(
                body.rig, gait, t,
                ground_y=self.ground_y, body_x0=x, hip_height=body.hip_height,
            )

        bake_rig(
            self.scene, body, pose_fn, frames=self.frames,
            color=color, thickness=thickness, layer_name=name,
        )
        ch = Character(name, body, gait, pose_fn)
        self.characters.append(ch)
        return ch

    def _add_object(self, samples, **kw):
        return bake_samples(self.scene, samples, **kw)

    def namespace(self) -> dict[str, Any]:
        """What a script sees. Deliberately small — this is the LLM's vocabulary."""
        ns: dict[str, Any] = {
            # the cartoon library
            "presets": presets,
            "motion": motion,
            "principles": principles,
            "geometry": geometry,
            "rig": rig,
            "Vec2": geometry.Vec2,
            "human": presets.human,
            "biped": presets.biped,
            "quadruped": presets.quadruped,
            "make_gait": presets.make_gait,
            # the stage
            "add_character": self._add_character,
            "add_object": self._add_object,
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


class SessionStore:
    """Every document in play, plus the one global Glaxnimate environment.

    `Headless()` is entered **once** for the life of the process. It brings up a
    Qt application object; doing that per call would be slow and, worse, unstable.
    """

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._stack.enter_context(environment.Headless())
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
        self._stack.close()
