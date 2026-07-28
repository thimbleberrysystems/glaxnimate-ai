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
import math
import sys
import time
import traceback
from pathlib import Path
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from glaxnimate import environment

from ..cartoon import (
    actions, assets, diagram, geometry, motion, physics, presets, principles,
    rig, scene_style)
from ..cartoon.gait import Gait, pose_at
from ..cartoon.presets import Body, apply_style
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
    #: bone/joint name -> its baked layer, so props can be parented to a hand.
    bone_layers: dict = field(default_factory=dict)
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
    #: Baked effect layers placed by auto_fx, so a re-run can hide them (baked Qt
    #: layers cannot be removed cleanly; opacity 0 is the honest clear).
    fx_layers: list = field(default_factory=list)
    #: The world/camera container layer; content re-parents under it so its
    #: transform acts as a camera. Created lazily on the first camera call.
    camera_layer: object = None
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
        style: str | None = None,
        face: str | dict | None = None,
        facing: float = 1.0,
        scale: float = 1.0,
    ) -> Character:
        """Bake a rig into the document and register it with the critic.

        `color`/`thickness` default to None so the body's own skin is used. Pass
        them to flatten the character to a single colour — useful for a silhouette
        check, useless for a cartoon.

        `style="lineart"` reskins the body as a stick figure before baking, so any
        creature — not just `stick()` — can be drawn as line art, and the look
        serialises into the scene document (it lives in the body's parts).

        `facing=-1` mirrors the character to look left (its profile face and limbs
        turn) — so two figures can face each other. Pair it with a leftward gait to
        walk left; on its own it just turns a standing/talking figure around.
        """
        if style:
            body = apply_style(body, style)

        def pose_fn(t: float):
            # No hip_height here on purpose: pose_at reads the gait's own
            # ride_height, so a crouched gallop stays crouched. Passing
            # body.hip_height would stand the figure back up and over-extend the
            # legs — the exact bug that made the gallop face-plant.
            return pose_at(body.rig, gait, t, ground_y=self.ground_y, body_x0=x)

        return self._bake_character(body, pose_fn, gait=gait, name=name,
                                    color=color, thickness=thickness, face=face,
                                    facing=facing, scale=scale)

    def _bake_character(
        self, body: Body, pose_fn, *, gait: Gait | None, name: str,
        color: str | None = None, thickness: float | None = None,
        face: str | dict | None = None, record: bool = True,
        poses: list | None = None, face_data: dict | None = None,
        expressions: list | None = None, first: int = 0, facing: float = 1.0,
        scale: float = 1.0,
    ) -> Character:
        """Bake + register + record. Every character path funnels through here so
        the scene document always matches what is on the canvas."""
        layers: dict = {}
        bake_rig(
            self.scene, body, pose_fn, frames=self.frames, first=first,
            color=color, thickness=thickness, layer_name=name, layers_out=layers,
            facing=facing, scale=scale,
        )
        limb_pairs = [(li.upper, li.lower) for li in gait.limbs] if gait else []
        ch = Character(name, body, gait, pose_fn, limb_pairs=limb_pairs,
                       bone_layers=dict(layers))

        if face_data is None and face is not None:
            face_data = (assets.load_face(face) if isinstance(face, str)
                         else assets.face_validate(face))
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
                "facing": facing,
                "scale": scale,
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
        style: str | None = None,
        face: str | dict | None = None,
        first: int = 0,
        facing: float = 1.0,
        scale: float = 1.0,
    ) -> Character:
        """Bake a character driven by an arbitrary pose function (a jump, a punch).

        Actions aren't locomotion, so there is no gait — but the character is still
        registered so the linter and diagnostics can inspect it (contact slip,
        joint integrity, bounds all still apply). `style="lineart"` reskins as a
        stick figure, `face=` mounts a face, `facing=-1` turns it to look left,
        `scale=1.6` makes a giant (or 0.6 a sprite), exactly as `add_character` does.
        """
        if style:
            body = apply_style(body, style)
        return self._bake_character(body, pose_fn, gait=None, name=name,
                                    color=color, thickness=thickness, face=face,
                                    first=first, facing=facing, scale=scale)

    # ------------------------------------------------------------------ audio
    def _add_sound(self, sfx, frame: float, *, gain: float = 1.0,
                   pan: float | None = None) -> str:
        """Place one sound cue. `sfx` is a name (boing, thud, step, pop, whoosh,
        slide_up, slide_down, splat, ding, or a saved sfx asset) or an inline
        patch dict. `pan=None` centres it."""
        from ..audio.mix import resolve_patch

        resolve_patch(sfx)  # teaching error now, not at export time
        self.doc["audio"]["cues"].append({
            "frame": float(frame), "sfx": sfx, "gain": float(gain),
            "pan": 0.0 if pan is None else float(pan),
        })
        return f"cue: {sfx if isinstance(sfx, str) else 'inline patch'} at f{frame:g}"

    #: what auto_sfx plays for each motion event; override per call
    _SFX_MAP = {"plant": "step", "hit": "boing", "launch": "whoosh",
                "land": "thud", "expression": "pop"}

    def _auto_sfx(self, mapping: dict | None = None, *, clear: bool = True,
                  gain: float = 1.0) -> str:
        """Place cues from the motion itself — the foley pass.

        The same Timeline data the linter reads yields foot plants, ball-ground
        hits, jump launches/landings and expression swaps; each becomes a cue,
        panned to where it happens on screen. Map an event kind to None to
        silence it: auto_sfx({"plant": None}).
        """
        from ..audio import events as EV
        from ..cartoon import timeline as tlmod

        sfx_map = {**self._SFX_MAP, **(mapping or {})}
        width = float(self.scene.comp.width)
        found: list[EV.MotionEvent] = []

        for ch in self.characters:
            tl = tlmod.from_pose_fn(ch.body, ch.pose_fn, frames=self.frames)
            found += EV.plant_onsets(tl, ground_y=self.ground_y)
            found += EV.airborne_spans(tl, ground_y=self.ground_y)
        for name, samples, radius in self.objects:
            found += EV.object_hits(samples, radius=radius,
                                    ground_y=self.ground_y, name=name)
        found += EV.expression_stings(self.doc)

        if clear:
            self.doc["audio"]["cues"] = [
                c for c in self.doc["audio"]["cues"] if not c.get("auto")
            ]
        placed = 0
        for ev in sorted(found, key=lambda e: e.frame):
            sfx = sfx_map.get(ev.kind)
            if sfx is None:
                continue
            pan = max(-0.6, min(0.6, (ev.x / width) * 1.2 - 0.6))
            self.doc["audio"]["cues"].append({
                "frame": float(ev.frame), "sfx": sfx, "gain": gain,
                "pan": round(pan, 3), "auto": True, "event": ev.kind,
            })
            placed += 1
        kinds = {}
        for ev in found:
            kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
        return f"placed {placed} cue(s) from motion ({summary or 'no events found'})"

    # ----------------------------------------------------------------- effects
    def _add_effect(self, fx, x: float, y: float, frame: float, *,
                    layer_name: str | None = None, record: bool = True) -> str:
        """Bake one visual effect at (x, y), popping in on `frame`.

        `fx` is a builtin name (impact, dust, speed_lines, spark), an inline fx
        document, or a saved fx asset dict. Effects are short-lived: they grow, fade
        and vanish, costing nothing on every other frame.
        """
        from ..cartoon.effects import resolve_fx
        from .bake import bake_effect

        data = resolve_fx(fx)
        lname = layer_name or f"fx.{len(self.doc['effects'])}"
        bake_effect(self.scene, data, x=float(x), y=float(y), start=float(frame),
                    layer_name=lname)
        if record:
            self.doc["effects"].append({
                "fx": fx if isinstance(fx, str) else data,
                "x": float(x), "y": float(y), "frame": float(frame), "layer": lname,
            })
        label = fx if isinstance(fx, str) else "inline fx"
        return f"effect {label} at ({x:g}, {y:g}) f{frame:g}"

    #: what auto_fx spawns for each motion event; override per call
    _FX_MAP = {"plant": "dust", "hit": "impact", "launch": "speed_lines",
               "land": "dust"}

    def _auto_fx(self, mapping: dict | None = None, *, clear: bool = True) -> str:
        """Place visual effects from the motion itself — the picture twin of auto_sfx.

        The SAME Timeline events that foley reads — foot plants, ground hits, jump
        launches and landings — spawn dust, impact flashes and speed lines, on the
        same frames and at the same screen positions. One derivation, two channels:
        a landing gets its thud *and* its dust from the same event. Map a kind to
        None to skip it: auto_fx({"plant": None}).
        """
        from ..audio import events as EV
        from ..cartoon import timeline as tlmod
        from ..cartoon.effects import resolve_fx
        from .bake import bake_effect

        fx_map = {**self._FX_MAP, **(mapping or {})}
        found: list[EV.MotionEvent] = []
        for ch in self.characters:
            tl = tlmod.from_pose_fn(ch.body, ch.pose_fn, frames=self.frames)
            found += EV.plant_onsets(tl, ground_y=self.ground_y)
            found += EV.airborne_spans(tl, ground_y=self.ground_y)
        for name, samples, radius in self.objects:
            found += EV.object_hits(samples, radius=radius,
                                    ground_y=self.ground_y, name=name)

        if clear:  # hide prior auto layers (baked layers can't be removed) + drop records
            for lay in self.fx_layers:
                lay.opacity.clear_keyframes()
                lay.opacity.value = 0.0
            self.fx_layers = []
            self.doc["effects"] = [e for e in self.doc["effects"] if not e.get("auto")]

        placed = 0
        for ev in sorted(found, key=lambda e: e.frame):
            name = fx_map.get(ev.kind)
            if name is None:
                continue
            lname = f"fx.auto.{ev.kind}.{placed}"
            lay = bake_effect(self.scene, resolve_fx(name), x=ev.x,
                              y=self.ground_y, start=float(ev.frame), layer_name=lname)
            self.fx_layers.append(lay)
            self.doc["effects"].append({
                "fx": name, "x": ev.x, "y": self.ground_y, "frame": float(ev.frame),
                "layer": lname, "auto": True, "event": ev.kind,
            })
            placed += 1
        kinds: dict[str, int] = {}
        for ev in found:
            kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
        return f"placed {placed} effect(s) from motion ({summary or 'no events found'})"

    def audio_mix(self):
        """Render the scene's cue sheet to a stereo buffer + report. Used by the
        export path and the sound_report tool."""
        from ..audio.mix import Cue, mix_report, render_cues

        cues = [Cue(c["frame"], c["sfx"], c.get("gain", 1.0), c.get("pan", 0.0))
                for c in self.doc["audio"]["cues"]]
        extra = self._extra_audio()
        result = render_cues(cues, frames=self.frames, fps=self.scene.fps,
                             extra=extra)
        report = mix_report(cues, result, frames=self.frames, fps=self.scene.fps)
        faults = self._dialogue_faults()
        if faults:
            report += "\n" + "\n".join(faults)
        return result, report

    def _dialogue_faults(self) -> list[str]:
        """Dialogue defects a peak meter cannot show.

        Two lines at once is not loud, it is unintelligible — the mix report is
        happy, the peak is fine, and the film is broken. Same for a line still
        talking when the picture ends. Neither is audible to the model, and
        neither shows up in dBFS, so they are checked as arithmetic here: the
        lines' spans are known exactly, because TTS was rendered and measured
        before it was ever cached.
        """
        fps = self.scene.fps
        lines = sorted((e for e in (self.doc.get("audio") or {}).get("dialogue", [])),
                       key=lambda e: e["frame"])
        out: list[str] = []
        for a, b in zip(lines, lines[1:], strict=False):
            a_end = a["frame"] + a["dur"] * fps
            if b["frame"] < a_end:
                out.append(
                    f"  ! two lines overlap by {a_end - b['frame']:.0f}f: "
                    f"{a['text'][:28]!r} is still talking at f{b['frame']:g} when "
                    f"{b['text'][:28]!r} starts"
                )
        for e in lines:
            end = e["frame"] + e["dur"] * fps
            if end > self.frames:
                out.append(
                    f"  ! {e['text'][:28]!r} runs {end - self.frames:.0f}f past the "
                    f"last frame — it will be cut off mid-sentence"
                )
        return out

    def _music(self, seed: int = 0, *, bpm: float = 96, gain: float = 0.25) -> str:
        """A seeded chiptune underscore beneath the whole scene.

        Deterministic per seed — if it sounds naff, change the seed and listen
        again; the spec is data in the scene doc like everything else. Keep the
        gain low: it is a bed, not the show. music(seed=None) removes it.
        """
        from ..audio.music import music_validate

        if seed is None:
            self.doc["audio"]["music"] = None
            return "music removed"
        spec = music_validate({"seed": seed, "bpm": bpm, "gain": gain})
        self.doc["audio"]["music"] = spec
        return f"music: seed {spec['seed']}, {spec['bpm']:g} bpm, gain {spec['gain']:g}"

    # -------------------------------------------------- rhythm & social format
    def _tempo(self, bpm: float | None) -> float:
        b = bpm if bpm is not None else (self.doc.get("audio") or {}).get("music", {})
        b = b.get("bpm") if isinstance(b, dict) else b
        if not b:
            raise ValueError("no tempo yet — pass bpm= or call music(bpm=...) first")
        return float(b)

    def _beats(self, division: int = 1, *, bpm: float | None = None) -> list[int]:
        """The frames the beat lands on, so cuts/hits/effects can snap to the music.

        The inverse of auto_sfx: that puts sound on the motion, this puts motion on
        the beat. `division` 1=quarters, 2=eighths, 4=sixteenths. Uses the scene's
        music() bpm unless you pass one."""
        from ..cartoon.rhythm import beat_frames

        return beat_frames(self._tempo(bpm), self.scene.fps, self.frames,
                           division=division)

    def _snap_to_beat(self, frame: float, division: int = 1, *,
                      bpm: float | None = None) -> int:
        """Round a frame to the nearest beat — drop a cut or a clash onto the grid."""
        from ..cartoon.rhythm import snap_to_beat

        return snap_to_beat(frame, self._tempo(bpm), self.scene.fps, division=division)

    def _loop_report(self) -> str:
        """Does the animation loop seamlessly? Reports how far each character's pose
        drifts between the first frame and the last — a seamless loop (for a sticker
        or a short) wants that near zero. A walk baked to exactly its cycle loops; an
        odd number of frames usually does not."""
        if not self.characters:
            return "loop: no characters to check"
        lines = []
        worst = 0.0
        for ch in self.characters:
            a = ch.body.rig.solve(ch.pose_fn(0.0))
            b = ch.body.rig.solve(ch.pose_fn(float(self.frames)))
            root = ch.body.rig.root_name
            ra, rb = a[root].origin, b[root].origin
            # pose drift relative to the root: does the *cycle* repeat? (a forward
            # walk translates but its limbs still loop — that is what we score.)
            drift = max((a[j].tip - ra).distance_to(b[j].tip - rb) for j in a)
            travel = ra.distance_to(rb)
            note = "seamless" if drift < 4 else f"cycle jumps {drift:.0f}px"
            if travel > 4:
                note += f" (travels {travel:.0f}px — loops in place, scrolls on screen)"
            worst = max(worst, drift)
            lines.append(f"{ch.name}: {note}")
        head = "cycle loops" if worst < 4 else f"cycle does NOT loop (worst {worst:.0f}px)"
        return f"loop [{head}]: " + "; ".join(lines)

    def _say(self, character, text: str, frame: float,
             *, voice: str | None = None, gain: float = 1.0,
             bubble: bool = True, lipsync: bool = True) -> str:
        """A character speaks. TTS renders now and is CACHED to the project dir,
        so the scene replays its dialogue without piper installed — the same
        persist-the-samples rule the doc uses for poses. A speech bubble shows
        above the speaker for the duration (bubble=False for off-screen voice).

        `lipsync` (when the speaker's face has the say_* mouths) flaps the mouth from
        the audio's own RMS envelope — the model cannot hear, but the mouth is
        arithmetic over the WAV it just rendered.
        """
        from ..audio import voice as V

        ch = character if isinstance(character, Character) else next(
            (c for c in self.characters if c.name == character), None
        )
        if ch is None and isinstance(character, str) and character:
            raise ValueError(
                f"no character {character!r}; have {[c.name for c in self.characters]}"
            )

        vname = voice or V.DEFAULT_VOICE
        samples = V.synthesize(text, vname)
        dur_s = len(samples) / 44100.0

        n = len(self.doc["audio"]["dialogue"])
        wav_path = SD.doc_path(self.doc_id).parent / "audio" / f"line{n}.wav"
        V.save_line(samples, wav_path)

        entry = {
            "frame": float(frame), "character": ch.name if ch else None,
            "text": text, "voice": vname, "gain": gain,
            "wav": wav_path.name, "dur": dur_s, "bubble": bool(bubble and ch),
        }
        self.doc["audio"]["dialogue"].append(entry)
        if entry["bubble"]:
            self._draw_bubble(ch, frame, dur_s)

        synced = False
        if lipsync and ch is not None:
            from ..audio.lipsync import LIPSYNC_SLOTS, mouth_levels

            if all(slot in ch.face_layers for slot in LIPSYNC_SLOTS):
                for f, level in mouth_levels(samples, fps=self.scene.fps,
                                             start_frame=frame):
                    self._set_expression(ch, LIPSYNC_SLOTS[level], f)
                synced = True
        tail = " +lip-sync" if synced else ""
        return (f"{ch.name if ch else 'voice'} says {text!r} at f{frame:g} "
                f"({dur_s:.1f}s){tail}")

    def _draw_bubble(self, ch: Character, frame: float, dur_s: float) -> None:
        """A minimal speech bubble above the speaker's head, held for the line."""
        from glaxnimate import model, utils

        f = min(int(frame), self.frames)
        rec = next((r for r in self.doc["characters"] if r["name"] == ch.name), None)
        if rec and rec["poses"]:
            root = rec["poses"][min(f, len(rec["poses"]) - 1)]["root"]
        else:
            pose = ch.pose_fn(float(f))
            root = [pose.root.x, pose.root.y]
        x = root[0] + 60
        y = root[1] - ch.body.leg_length * 0.9

        lay = self.scene.layer(f"{ch.name}.bubble")
        g = lay.add_shape("Group")
        g.add_shape("Fill").color.value = "#ffffff"
        e = g.add_shape("Ellipse")
        e.size.value = utils.Size(110, 64)
        e.position.value = utils.Point(x, y)
        dots = lay.add_shape("Group")
        dots.add_shape("Fill").color.value = "#44464f"
        for i in range(3):
            d = dots.add_shape("Ellipse")
            d.size.value = utils.Size(10, 10)
            d.position.value = utils.Point(x - 22 + i * 22, y)

        end = min(frame + dur_s * self.scene.fps, self.frames)
        for prop_frame, value in ((0.0, 0.0), (float(frame), 1.0), (float(end), 0.0)):
            lay.opacity.set_keyframe(prop_frame, value)
            tr = model.KeyframeTransition()
            tr.hold = True
            lay.opacity.set_transition(prop_frame, tr)

    def _extra_audio(self):
        """Music beds and dialogue lines, rendered onto the same bus as cues.

        The music bed is ducked under every dialogue span so speech stays clear —
        an underscore at full level through a line is the single biggest reason a
        mix reads as muddy."""
        extra = []
        audio = self.doc.get("audio") or {}
        fps = self.scene.fps
        spans = [(e["frame"] / fps, e["frame"] / fps + e["dur"])
                 for e in audio.get("dialogue", [])]
        music = audio.get("music")
        if music:
            from ..audio.mix import duck
            from ..audio.music import render_music

            duration = self.frames / self.scene.fps
            bed = render_music(music, duration_s=duration)
            bed = duck(bed, spans, level=0.28)  # drop under dialogue
            extra.append((0.0, bed, music.get("gain", 0.25), 0.0))
        for entry in audio.get("dialogue", []):
            from ..audio.voice import load_line

            wav = SD.doc_path(self.doc_id).parent / "audio" / entry["wav"]
            if not wav.exists():
                continue  # cache lost; the report will show fewer spans
            samples, sr = load_line(wav)
            if sr != 44100:
                import numpy as np
                n_out = int(len(samples) * 44100 / sr)
                samples = np.interp(np.linspace(0, len(samples) - 1, n_out),
                                    np.arange(len(samples)), samples)
            extra.append((entry["frame"] / self.scene.fps, samples,
                          entry.get("gain", 1.0), 0.0))
        return extra

    @property
    def has_audio(self) -> bool:
        a = self.doc.get("audio") or {}
        return bool(a.get("cues") or a.get("music") or a.get("dialogue"))

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

    def _add_prop(self, prop, *, x: float = 0.0, y: float | None = None,
                  scale: float | tuple[float, float] = 1.0,
                  layer_name: str = "props"):
        """Place a data prop (a dict or an asset name) on the ground line.

        `y` overrides the ground line for something that is not standing on it —
        a diagram's slab, a sign on a wall. The prop's own origin still sits at
        whatever y it lands on; only the default changes.
        """
        from . import props as P

        data = assets.load_prop(prop) if isinstance(prop, str) else assets.prop_validate(prop)
        anchor = self.ground_y if y is None else float(y)
        lay = self.scene.layer(layer_name)
        P.draw_prop(lay, data, x=x, ground_y=anchor, scale=scale)
        self.doc["props"].append({"data": data, "x": x, "y": y, "scale": scale,
                                  "layer": layer_name})
        return lay

    def _add_object(self, samples, *, record: bool = True, **kw):
        # Register the samples so the critic can see the object too — in v1 only
        # rig characters were checkable and a ball through the floor was invisible.
        if "name" in kw:  # accept name= like add_character, not just layer_name=
            kw.setdefault("layer_name", kw.pop("name"))
        allowed = {"shape", "size", "color", "layer_name", "stats"}
        bad = set(kw) - allowed
        if bad:
            raise TypeError(f"add_object got unexpected argument(s) {sorted(bad)}; "
                            f"valid: name/layer_name, shape, size, color")
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

    def _shot(self, prefix: str, start: float, end: float, *, record: bool = True) -> str:
        """Show a layer, and everything named under it, only during [start, end].

        This is how one document holds several shots. There is no camera and no
        cut — the timeline is continuous — so a "shot" is just visibility: name
        a beat's layers `beat3.magnet`, `beat3.iron`, and gate the lot with
        `shot("beat3", 336, 720)`.

        It gates by name *prefix* rather than by layer, because a character is
        not one layer: `bake_rig` writes a root plus one layer per bone, and
        they are siblings that use `Layer.parent` for transforms only. Gating
        the root alone leaves the entire skeleton on screen — measured, 4981
        stray pixels of a man who should have left the shot.

        Opacity keys are HOLD keys: a shot cuts, it does not dissolve.
        """
        from glaxnimate import model

        want = [sh for sh in self.scene.comp.shapes
                if sh.name == prefix or sh.name.startswith(prefix + ".")]
        if not want:
            names = sorted({sh.name.split(".")[0] for sh in self.scene.comp.shapes})
            raise ValueError(f"no layers named {prefix!r} or {prefix!r}.*; have {names}")

        # A broad prefix silently re-gating layers that already have their own,
        # narrower shot is a real trap: `shot("s5", ...)` matches `s5.tick0` and
        # overwrites the gate that made the mark appear on its own frame. It
        # looks like nothing happened and everything turns on at once.
        already = [sh.name for sh in want if sh.opacity.keyframe_count > 0]
        if already:
            raise ValueError(
                f"shot({prefix!r}) would overwrite the gate already on "
                f"{', '.join(already[:3])}"
                f"{'...' if len(already) > 3 else ''}. Gate the parts or the "
                f"whole, not both — name sub-shots outside {prefix!r}.* if they "
                f"need their own timing."
            )

        keys: list[tuple[float, float]] = []
        if start > 0:
            keys.append((0.0, 0.0))
        keys.append((float(start), 1.0))
        if end < self.frames:
            keys.append((float(end), 0.0))

        for sh in want:
            for f, v in keys:
                sh.opacity.set_keyframe(f, v)
                tr = model.KeyframeTransition()
                tr.hold = True
                sh.opacity.set_transition(f, tr)
        if record:
            self.doc.setdefault("shots", []).append(
                {"prefix": prefix, "start": float(start), "end": float(end)}
            )
        return f"shot {prefix!r}: f{start:g}-{end:g} ({len(want)} layer(s))"

    # ------------------------------------------------------- ropes & leashes
    def _bone_tip(self, ch: Character, bone: str, frame: float):
        """World position of a character's bone tip at a frame — for connectors."""
        from ..cartoon.geometry import Vec2
        frames = ch.body.rig.solve(ch.pose_fn(float(frame)))
        jf = frames.get(bone) or frames[ch.body.rig.root_name]
        return Vec2(jf.tip.x, jf.tip.y)

    def _rope(self, x0: float, y0: float, x1: float, y1: float, *, sag: float = 0.0,
              color: str = "#6b5637", width: float = 3.0, name: str | None = None,
              record: bool = True) -> str:
        """A static line between two points — a tightrope, a clothesline, a fuse, a
        hanging rope, a fixed leash. `sag` dips the middle (a slack rope)."""
        from glaxnimate import utils

        lname = name or f"rope.{len(self.doc.setdefault('ropes', []))}"
        lay = self.scene.layer(lname)
        g = lay.add_shape("Group")
        st = g.add_shape("Stroke")
        st.color.value = color
        st.width.value = width
        st.cap = st.Cap.RoundCap
        p = g.add_shape("Path")
        bez = p.shape.value
        z = utils.Point(0.0, 0.0)
        bez.add_point(utils.Point(x0, y0), z, z)
        if sag > 0:  # a slack middle (a shallow V reads as a dip)
            bez.line_to(utils.Point((x0 + x1) / 2, (y0 + y1) / 2 + sag))
        bez.line_to(utils.Point(x1, y1))
        p.shape.value = bez
        if record:
            self.doc["ropes"].append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                      "sag": sag, "color": color, "width": width,
                                      "name": lname})
        return f"rope ({x0:g},{y0:g})->({x1:g},{y1:g})"

    def _leash(self, a, a_bone: str, b, b_bone: str, *, color: str = "#6b5637",
               width: float = 3.0, name: str | None = None, record: bool = True) -> str:
        """A live connector between two characters' bones — a dog's leash, a tow-rope,
        holding hands. A unit stroke whose transform is re-aimed every frame to span
        bone A -> bone B (position + rotation + a real x-scale, only possible since the
        binding fix), so it tracks both ends as they move."""
        from glaxnimate import utils

        cha, chb = self._char(a), self._char(b)
        lname = name or f"leash.{len(self.doc.setdefault('leashes', []))}"
        lay = self.scene.layer(lname)
        g = lay.add_shape("Group")
        st = g.add_shape("Stroke")
        st.color.value = color
        st.width.value = width
        st.cap = st.Cap.RoundCap
        p = g.add_shape("Path")
        bez = p.shape.value
        z = utils.Point(0.0, 0.0)
        bez.add_point(utils.Point(0.0, 0.0), z, z)
        bez.line_to(utils.Point(1.0, 0.0))  # unit line, stretched by the transform
        p.shape.value = bez
        for f in range(self.frames + 1):
            pa, pb = self._bone_tip(cha, a_bone, f), self._bone_tip(chb, b_bone, f)
            dx, dy = pb.x - pa.x, pb.y - pa.y
            dist = math.hypot(dx, dy) or 1.0
            g.transform.position.set_keyframe(float(f), utils.Point(pa.x, pa.y))
            g.transform.rotation.set_keyframe(float(f), math.degrees(math.atan2(dy, dx)))
            g.transform.scale.set_keyframe(float(f), utils.Vector2D(dist, 1.0))
        if record:
            self.doc["leashes"].append({"a": cha.name, "a_bone": a_bone, "b": chb.name,
                                        "b_bone": b_bone, "color": color, "width": width,
                                        "name": lname})
        return f"leash {cha.name}.{a_bone} <-> {chb.name}.{b_bone}"

    # ------------------------------------------------- fight FX (smear/aura/beam)
    def _gate_layers(self, prefix: str, keys: list[tuple]) -> None:
        """Set opacity keyframes on every layer named `prefix` or `prefix.*`."""
        for sh in self.scene.comp.shapes:
            if sh.name == prefix or sh.name.startswith(prefix + "."):
                for f, v in keys:
                    sh.opacity.set_keyframe(float(f), float(v))

    def _afterimage(self, character, at: float, *, count: int = 3, gap: float = 2.0,
                    opacity: float = 0.34, color: str = "#93a6c4",
                    name: str | None = None, record: bool = True) -> str:
        """Speed ghosts: faded frozen copies of the figure at recent poses, flashing
        around frame `at` — the trailing after-images of a dash, a teleport, a blur-
        fast strike. `count` ghosts, each `gap` frames further back and fainter."""
        from .bake import bake_rig

        ch = self._char(character)
        base = name or f"ghost.{len([o for o in self.doc['overlays'] if o['kind'] == 'afterimage'])}"
        for i in range(1, count + 1):
            gf = max(float(at) - gap * i, 0.0)
            gname = f"{base}.{i}"
            frozen = (lambda g: (lambda t: ch.pose_fn(g)))(gf)   # constant pose at gf
            bake_rig(self.scene, ch.body, frozen, frames=self.frames, color=color,
                     layer_name=gname, joint_color=None, joint_radius=0.0)
            fade = opacity * (1.0 - (i - 1) / (count + 1))        # older = fainter
            self._gate_layers(gname, [(0.0, 0.0), (gf + 0.5, fade),
                                      (float(at) + 1, fade), (float(at) + 3, 0.0)])
        if record:
            self.doc["overlays"].append({
                "kind": "afterimage", "character": ch.name, "at": float(at),
                "count": count, "gap": gap, "opacity": opacity, "color": color,
                "name": base})
        return f"afterimage x{count} on {ch.name} @f{at:g}"


    def _smear(self, character, bone: str, at: float, *, span: int = 4,
               color: str = "#e6efff", spread: float = 16.0, opacity: float = 0.55,
               name: str | None = None, record: bool = True) -> str:
        """A motion smear: the translucent swoosh a fast limb or blade leaves on its
        strike frame — THE defining look of a stick fight. Samples the bone's path
        over the last `span` frames and draws a tapered arc through it, flashing on
        frame `at` and gone a few frames later."""
        from glaxnimate import utils

        from . import props as P

        ch = self._char(character)
        pts = [self._bone_tip(ch, bone, max(float(at) - span + i, 0.0))
               for i in range(span + 1)]
        top, bot = [], []
        for i, p in enumerate(pts):
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, span)]
            tx, ty = b.x - a.x, b.y - a.y
            L = math.hypot(tx, ty) or 1.0
            perx, pery = -ty / L, tx / L
            hw = spread * math.sin(math.pi * i / span) + 1.5   # fat middle, thin ends
            top.append([p.x + perx * hw, p.y + pery * hw])
            bot.append([p.x - perx * hw, p.y - pery * hw])
        poly = {"type": "polygon", "points": top + bot[::-1], "color": color}
        lname = name or f"smear.{len([o for o in self.doc['overlays'] if o['kind'] == 'smear'])}"
        lay = self.scene.layer(lname)
        P.draw_prop(lay.add_shape("Group"), {"shapes": [poly]}, x=0.0, ground_y=0.0)
        op = lay.opacity
        op.set_keyframe(max(float(at) - 1, 0.0), 0.0)
        op.set_keyframe(float(at), opacity)
        op.set_keyframe(float(at) + 3, 0.0)
        if record:
            self.doc["overlays"].append({
                "kind": "smear", "character": ch.name, "bone": bone, "at": float(at),
                "span": span, "color": color, "spread": spread, "opacity": opacity,
                "name": lname})
        return f"smear on {ch.name}.{bone} @f{at:g}"

    def _aura(self, character, *, color: str = "#ffd23f", radius: float = 48.0,
              bone: str = "spine", start: float = 0.0, pulse: float = 6.0,
              opacity: float = 0.4, name: str | None = None, record: bool = True) -> str:
        """A charge-up aura: a translucent glow that rides a character and pulses —
        the power-up halo of every energy fight. Follows `bone` each frame, swells in
        from `start`, and beats `pulse` times."""
        from glaxnimate import utils

        from . import props as P

        ch = self._char(character)
        bone = bone if bone in ch.body.rig.joints else ch.body.rig.root_name
        lname = name or f"aura.{len([o for o in self.doc['overlays'] if o['kind'] == 'aura'])}"
        lay = self.scene.layer(lname)
        g = lay.add_shape("Group")
        P.draw_prop(g, {"shapes": [
            {"type": "ellipse", "cx": 0, "cy": 0, "w": radius * 2, "h": radius * 2.4,
             "color": color},
            {"type": "ellipse", "cx": 0, "cy": 0, "w": radius, "h": radius * 1.3,
             "color": "#ffffff"}]}, x=0.0, ground_y=0.0)
        for f in range(self.frames + 1):
            p = self._bone_tip(ch, bone, f)
            beat = 0.85 + 0.28 * (0.5 + 0.5 * math.sin(2.0 * math.pi * pulse * f / max(self.frames, 1)))
            g.transform.position.set_keyframe(float(f), utils.Point(p.x, p.y))
            g.transform.scale.set_keyframe(float(f), utils.Vector2D(beat, beat))
        op = lay.opacity                      # swell in from `start`
        op.set_keyframe(max(float(start) - 1, 0.0), 0.0)
        op.set_keyframe(float(start) + 4, opacity)
        if record:
            self.doc["overlays"].append({
                "kind": "aura", "character": ch.name, "color": color, "radius": radius,
                "bone": bone, "start": float(start), "pulse": pulse, "opacity": opacity,
                "name": lname})
        return f"aura on {ch.name}"

    def _beam(self, x0: float, y0: float, x1: float, y1: float, start: float,
              end: float, *, color: str = "#7fd4ff", core: str = "#ffffff",
              width: float = 20.0, opacity: float = 0.85, name: str | None = None,
              record: bool = True) -> str:
        """An energy beam from (x0,y0) to (x1,y1), lit during [start,end] — a glowing
        outer layer with a bright core. Two beams aimed at each other with an impact
        flash at the meeting point make the classic beam struggle."""
        from glaxnimate import utils

        from . import props as P

        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy) or 1.0
        lname = name or f"beam.{len([o for o in self.doc['overlays'] if o['kind'] == 'beam'])}"
        lay = self.scene.layer(lname)
        g = lay.add_shape("Group")
        P.draw_prop(g, {"shapes": [
            {"type": "rect", "x": 0, "y": -width / 2, "w": 1, "h": width, "color": color},
            {"type": "rect", "x": 0, "y": -width / 6, "w": 1, "h": width / 3, "color": core}]},
            x=0.0, ground_y=0.0)
        g.transform.position.value = utils.Point(x0, y0)
        g.transform.rotation.value = math.degrees(math.atan2(dy, dx))
        g.transform.scale.value = utils.Vector2D(dist, 1.0)   # stretch the unit beam
        op = lay.opacity
        op.set_keyframe(max(float(start) - 1, 0.0), 0.0)
        op.set_keyframe(float(start), opacity)
        op.set_keyframe(float(end), opacity)
        op.set_keyframe(float(end) + 2, 0.0)
        if record:
            self.doc["overlays"].append({
                "kind": "beam", "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "start": float(start), "end": float(end), "color": color, "core": core,
                "width": width, "opacity": opacity, "name": lname})
        return f"beam ({x0:g},{y0:g})->({x1:g},{y1:g}) f{start:g}-{end:g}"

    # ------------------------------------------------------------- particles
    def _emit(self, fx, x: float, y: float, *, count: int = 12, spread: float = 120.0,
              drop: float = 0.0, start: float = 0.0, over: float = 8.0,
              size: float = 1.0, color: str = "#8fbfe0", seed: int = 0,
              record: bool = True) -> str:
        """Emit a shower of many small particles — confetti, sparks, rain, snow.

        `drop=0` bursts them in place, scattered within `spread` (confetti pop,
        sparkle, a puff of smoke — pass an fx name like "spark"). `drop>0` makes them
        FALL that many px from a band `spread` wide (rain, snow, falling confetti —
        `fx` is then an inline shape or a colour). Particles stagger over `over`
        frames. Deterministic in `seed`, so it saves and replays identically.
        """
        import random

        from ..cartoon.effects import resolve_fx
        from ..cartoon.geometry import Vec2
        from .bake import bake_effect, bake_prop_samples

        rng = random.Random(seed)
        base = f"emit.{len(self.doc.setdefault('emits', []))}"
        if drop > 0.0:  # falling particles
            shape = fx if isinstance(fx, dict) else {"shapes": [
                {"type": "ellipse", "cx": 0, "cy": 0, "w": 3.5 * size,
                 "h": 9 * size, "color": color}]}
            for i in range(count):
                px = x + rng.uniform(-spread, spread)
                rel = int(start + rng.uniform(0, over))
                span = max(self.frames - rel, 1)
                spd = rng.uniform(0.75, 1.25)
                from ..cartoon.motion import Sample
                samples = []
                for f in range(int(span) + 1):
                    s = min(f / span * spd, 1.0)
                    samples.append(Sample(int(rel + f), Vec2(px, y + drop * s),
                                          scale=Vec2(1, 1), angle=0.0))
                bake_prop_samples(self.scene, shape, samples, layer_name=f"{base}.{i}")
        else:  # scattered burst of an effect
            data = fx if isinstance(fx, dict) else resolve_fx(fx)
            for i in range(count):
                ang = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(0, spread)
                px, py = x + math.cos(ang) * r, y + math.sin(ang) * r
                bake_effect(self.scene, data, x=px, y=py,
                            start=start + rng.uniform(0, over), layer_name=f"{base}.{i}")
        if record:
            self.doc["emits"].append({
                "fx": fx if isinstance(fx, str) else (fx if isinstance(fx, dict) else str(fx)),
                "x": x, "y": y, "count": count, "spread": spread, "drop": drop,
                "start": start, "over": over, "size": size, "color": color, "seed": seed,
            })
        return f"emitted {count} particle(s) at ({x:g}, {y:g})"

    # ---------------------------------------------------- free-floating shapes
    def _add_shape(self, shapes, x: float, y: float, *, name: str | None = None,
                   scale: float = 1.0, pulse: tuple | None = None, spin: float = 0.0,
                   appear: float = 0.0, vanish: float = 0.0, record: bool = True) -> str:
        """Place a free-floating prop at (x, y) — not pinned to the ground, not held.

        This is how a scene gets a heart, a sign, a chart, a hat, a drum, a speech
        placard: anything that is neither scenery nor a puppet. `shapes` is an inline
        list / {"shapes": [...]} of prop-schema shapes, or a saved prop name. Options:
        `pulse=(lo, hi, cycles)` beats the scale (a heart); `spin` rotates it;
        `appear` pops it in at a frame (a reveal); `vanish` pops it back out at a
        frame (a thought that clears, a flash that's gone). Screen coords, y down.
        """
        from glaxnimate import model, utils

        from . import props as P

        if isinstance(shapes, list):
            data = {"shapes": shapes}
        elif isinstance(shapes, dict):
            data = shapes
        else:
            data = assets.load_prop(shapes)
        lname = name or f"shape.{len(self.doc.setdefault('shapes', []))}"
        lay = self.scene.layer(lname)
        g = lay.add_shape("Group")
        P.draw_prop(g, data, x=0.0, ground_y=0.0)
        g.transform.position.value = utils.Point(float(x), float(y))

        base = float(scale)
        if pulse:
            lo, hi, cycles = pulse
            for f in range(0, self.frames + 1, 2):  # sample the beat
                phase = 0.5 + 0.5 * math.sin(2.0 * math.pi * cycles * f / max(self.frames, 1))
                sc = base * (lo + (hi - lo) * phase)
                g.transform.scale.set_keyframe(float(f), utils.Vector2D(sc, sc))
        elif base != 1.0:
            g.transform.scale.value = utils.Vector2D(base, base)
        if spin:
            g.transform.rotation.set_keyframe(0.0, 0.0)
            g.transform.rotation.set_keyframe(float(self.frames), float(spin))
        if (appear and appear > 0) or (vanish and vanish > 0):
            start_vis = 0.0 if (appear and appear > 0) else 1.0
            lay.opacity.set_keyframe(0.0, start_vis)
            tr = model.KeyframeTransition()
            tr.hold = True
            lay.opacity.set_transition(0.0, tr)
            if appear and appear > 0:
                lay.opacity.set_keyframe(float(appear), 1.0)
                tr2 = model.KeyframeTransition()
                tr2.hold = True
                lay.opacity.set_transition(float(appear), tr2)
            if vanish and vanish > 0:
                lay.opacity.set_keyframe(float(vanish), 0.0)

        if record:
            self.doc["shapes"].append({
                "shapes": data["shapes"], "x": x, "y": y, "scale": scale,
                "pulse": list(pulse) if pulse else None, "spin": spin,
                "appear": appear, "vanish": vanish, "name": lname,
            })
        return f"shape {lname!r} at ({x:g}, {y:g})"

    # ------------------------------------------------------------ explainer hooks
    def _highlight(self, x: float, y: float, *, w: float = 70.0, h: float = 70.0,
                   shape: str = "ring", color: str = "#e0533d", width: float = 4.0,
                   appear: float = 0.0, vanish: float = 0.0,
                   pulse: bool = True, name: str | None = None) -> str:
        """Draw attention to a spot: a ring or box that pops in (and optionally out)
        and breathes. The single most-used explainer move — 'look *here*'. `shape` is
        "ring" or "box"; pair `appear`/`vanish` with a narration beat."""
        if shape == "box":
            hw, hh = w / 2, h / 2
            pts = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]]
            shapes = [{"type": "polyline", "points": pts, "color": color, "width": width}]
        else:
            n = 40
            pts = [[round(w / 2 * math.cos(2 * math.pi * i / n), 2),
                    round(h / 2 * math.sin(2 * math.pi * i / n), 2)] for i in range(n + 1)]
            shapes = [{"type": "polyline", "points": pts, "color": color, "width": width}]
        lname = name or f"highlight.{len(self.doc.setdefault('shapes', []))}"
        return self._add_shape(shapes, x, y, name=lname, appear=appear, vanish=vanish,
                               pulse=(0.9, 1.08, 3) if pulse else None)

    def _write_on(self, shapes, x: float, y: float, *, start: float = 0.0,
                  end: float | None = None, name: str | None = None,
                  record: bool = True) -> str:
        """Reveal a diagram as if hand-drawn: stroked paths (`polyline`) are *drawn*
        with a travelling pen (a keyed Trim), fills and text fade in — each shape in
        turn across [start, end]. The whiteboard build-up, from any shape list a
        diagram function returns."""
        from glaxnimate import model, utils

        from . import props as P

        items = shapes["shapes"] if isinstance(shapes, dict) else list(shapes)
        n = max(len(items), 1)
        end = end if end is not None else start + max(n * 8, 12)
        span = (end - start) / n
        base = name or f"writeon.{len(self.doc.setdefault('writeons', []))}"

        for i, sh in enumerate(items):
            s0 = start + i * span
            s1 = s0 + span
            lay = self.scene.layer(f"{base}.{i}")
            if sh.get("type") == "polyline":
                g = lay.add_shape("Group")
                st = g.add_shape("Stroke")
                st.color.value = sh.get("color", "#1f2430")
                st.width.value = float(sh.get("width", 3.0))
                st.cap = st.Cap.RoundCap
                st.join = st.Join.RoundJoin
                tr = g.add_shape("Trim")
                p = g.add_shape("Path")
                bez = p.shape.value
                z = utils.Point(0, 0)
                px0, py0 = sh["points"][0]
                bez.add_point(utils.Point(x + px0, y + py0), z, z)
                for px, py in sh["points"][1:]:
                    bez.line_to(utils.Point(x + px, y + py))
                if sh.get("close"):
                    bez.close()
                p.shape.value = bez
                tr.end.set_keyframe(0.0, 0.0)
                tr.end.set_keyframe(float(s0), 0.0)
                tr.end.set_keyframe(float(s1), 1.0)
            else:
                P.draw_prop(lay, {"shapes": [sh]}, x=x, ground_y=y)
                lay.opacity.set_keyframe(0.0, 0.0)
                hold = model.KeyframeTransition(); hold.hold = True
                lay.opacity.set_transition(0.0, hold)
                lay.opacity.set_keyframe(float(s0), 0.0)
                lay.opacity.set_keyframe(float(s1), 1.0)
        if record:
            self.doc.setdefault("writeons", []).append(
                {"shapes": items, "x": x, "y": y, "start": start, "end": end, "name": base})
        return f"write_on {base!r}: {n} shapes over f{start:g}-{end:g}"

    def _counter(self, x: float, y: float, *, v0: float = 0.0, v1: float = 100.0,
                 start: float = 0.0, end: float = 48.0, size: float = 28.0,
                 color: str = "#1f2430", prefix: str = "", suffix: str = "",
                 decimals: int = 0, anchor: str = "middle", ease: bool = True,
                 step: int = 1, name: str | None = None, record: bool = True) -> str:
        """A number that ticks from `v0` to `v1` across [start, end] — a tally, a
        percentage, a running total, a physics readout. Text is not animatable, so
        this lays one label per step and shows each in its window, holding `v1` after."""
        from . import props as P

        base = name or f"counter.{len(self.doc.setdefault('counters', []))}"
        end = float(end); start = float(start)
        frames = [f for f in range(int(start), int(end) + 1, max(step, 1))]
        if not frames or frames[-1] != int(end):
            frames.append(int(end))

        def val_at(f):
            t = 0.0 if end == start else (f - start) / (end - start)
            t = principles.ease_out(t) if ease else t
            return v0 + (v1 - v0) * max(0.0, min(1.0, t))

        from glaxnimate import model

        def key(lay, f, v):                         # a stepped (held) opacity key
            lay.opacity.set_keyframe(float(f), float(v))
            tr = model.KeyframeTransition(); tr.hold = True
            lay.opacity.set_transition(float(f), tr)

        for i, f in enumerate(frames):
            txt = f"{prefix}{val_at(f):.{decimals}f}{suffix}"
            lay = self.scene.layer(f"{base}.{i}")
            P.draw_prop(lay, {"shapes": [{"type": "text", "x": 0, "y": 0, "text": txt,
                                          "size": size, "color": color, "anchor": anchor}]},
                        x=x, ground_y=y)
            # exactly one label visible at a time: on at its frame, off when the next
            # arrives (the last one holds `v1` to the end). Held keys => no cross-fade.
            key(lay, 0.0, 0.0)
            key(lay, float(f), 1.0)
            if i + 1 < len(frames):
                key(lay, float(frames[i + 1]), 0.0)
        if record:
            self.doc.setdefault("counters", []).append({
                "x": x, "y": y, "v0": v0, "v1": v1, "start": start, "end": end,
                "size": size, "color": color, "prefix": prefix, "suffix": suffix,
                "decimals": decimals, "anchor": anchor, "ease": ease, "step": step,
                "name": base})
        return f"counter {base!r}: {v0:g}->{v1:g} over f{start:g}-{end:g}"

    # ------------------------------------------------------------------ camera
    def _world(self):
        """The camera container: content re-parents under one layer so its transform
        is a camera (pan=position, zoom=scale). Created lazily and populated with
        everything on screen so far — so call camera AFTER building the scene (and
        after any shot() gating, which reads top-level layers by name)."""
        if self.camera_layer is None:
            w = self.scene.layer("camera")
            for sh in list(self.scene.comp.shapes):
                if sh is not w and sh.parent is None:
                    sh.parent = w
            self.camera_layer = w
        return self.camera_layer

    def _apply_camera(self, keys: list[tuple]) -> None:
        """Write world-transform keyframes. Each key is (frame, zoom, focal_x,
        focal_y, shake_x, shake_y); the position that keeps `focal` centred under a
        `zoom` is center - zoom*focal, plus the shake offset."""
        from glaxnimate import utils

        w = self._world()
        cw, ch = float(self.scene.comp.width), float(self.scene.comp.height)
        for f, z, fx, fy, sx, sy in keys:
            w.transform.scale.set_keyframe(float(f), utils.Vector2D(z, z))
            w.transform.position.set_keyframe(
                float(f), utils.Point(cw / 2 - z * fx + sx, ch / 2 - z * fy + sy))

    def _impact_camera(self, frame: float, x: float | None = None,
                       y: float | None = None, *, zoom: float = 1.2,
                       shake: float = 9.0, ramp: int = 2, hold: int = 3,
                       shake_frames: int = 6, record: bool = True) -> str:
        """The hit-seller: a fast punch-in toward (x, y) plus a decaying screen shake,
        both fired on the contact `frame`. Fire it on the same frame as the impact
        flash, the hit sfx and the hitstop and the whole blow lands as one moment.

        Deterministic (no RNG): the shake is a damped two-axis oscillation, so a scene
        replays and lints identically."""
        cw, ch = float(self.scene.comp.width), float(self.scene.comp.height)
        fx = cw / 2 if x is None else float(x)
        fy = ch / 2 if y is None else float(y)
        f0 = float(frame)
        total = 2 * ramp + hold
        window = max(total, shake_frames)

        def zoom_at(df: float) -> float:
            if df <= 0 or df >= total:
                return 1.0
            if df < ramp:
                return 1.0 + (zoom - 1.0) * principles.ease_out(df / ramp)
            if df < ramp + hold:
                return zoom
            return 1.0 + (zoom - 1.0) * (1.0 - principles.ease_in((df - ramp - hold) / ramp))

        keys: list[tuple] = [(0.0, 1.0, cw / 2, ch / 2, 0.0, 0.0)]
        span = zoom - 1.0 or 1.0
        for k in range(int(window) + 1):
            z = zoom_at(float(k))
            # focal blends to centre as the zoom returns to 1, so z==1 is neutral
            tz = (z - 1.0) / span
            efx = cw / 2 + (fx - cw / 2) * tz
            efy = ch / 2 + (fy - ch / 2) * tz
            if k < shake_frames:
                d = 1.0 - k / shake_frames
                sx, sy = shake * d * math.cos(k * 2.3), shake * d * math.sin(k * 3.1)
            else:
                sx = sy = 0.0
            keys.append((f0 + k, z, efx, efy, sx, sy))
        keys.append((f0 + window, 1.0, cw / 2, ch / 2, 0.0, 0.0))
        self._apply_camera(keys)
        if record:
            self.doc.setdefault("camera", []).append({
                "op": "impact", "frame": f0, "x": x, "y": y, "zoom": zoom,
                "shake": shake, "ramp": ramp, "hold": hold, "shake_frames": shake_frames,
            })
        return f"impact camera at f{f0:g}: zoom {zoom:g}x + shake {shake:g}px"

    def _camera_move(self, start: float, end: float, *, zoom: float = 1.0,
                     focus_x: float | None = None, focus_y: float | None = None,
                     record: bool = True) -> str:
        """A held camera move: ease from the current neutral framing to `zoom` centred
        on (focus_x, focus_y) between `start` and `end` frames, for staging and slow
        push-ins. Pass zoom=1 with a focus to pan; zoom>1 to push in."""
        cw, ch = float(self.scene.comp.width), float(self.scene.comp.height)
        fx = cw / 2 if focus_x is None else float(focus_x)
        fy = ch / 2 if focus_y is None else float(focus_y)
        keys = [
            (0.0, 1.0, cw / 2, ch / 2, 0.0, 0.0),
            (float(start), 1.0, cw / 2, ch / 2, 0.0, 0.0),
            (float(end), zoom, fx, fy, 0.0, 0.0),
        ]
        self._apply_camera(keys)
        if record:
            self.doc.setdefault("camera", []).append({
                "op": "move", "start": float(start), "end": float(end),
                "zoom": zoom, "x": focus_x, "y": focus_y,
            })
        return f"camera move f{start:g}-{end:g}: zoom {zoom:g}x"

    # ------------------------------------------------- screenshot backdrop (AvA)
    def _backdrop(self, image_path, *, fit: bool = True, opacity: float = 1.0,
                  record: bool = True) -> str:
        """Set a screenshot (or any image) as the backdrop — the Animator-vs-Animation
        canvas: a desktop, a webpage, a game, over which the stick figures act and
        which they fight *with* (drag its windows, ride the cursor). Call it FIRST so
        it sits behind everything. `fit` scales it to fill the canvas."""
        from glaxnimate import utils

        bmp = self.scene.document.assets.add_image_file(str(image_path), True)
        lay = self.scene.layer("backdrop")
        im = lay.add_shape("Image")
        im.image = bmp
        cw, ch = float(self.scene.comp.width), float(self.scene.comp.height)
        iw, ih = float(getattr(bmp, "width", cw)), float(getattr(bmp, "height", ch))
        # Image origin is its top-left; anchor at (0,0) and stretch to fill the canvas.
        im.transform.position.value = utils.Point(0.0, 0.0)
        if fit and iw > 0 and ih > 0:
            im.transform.scale.value = utils.Vector2D(cw / iw, ch / ih)
        if opacity < 1.0:
            lay.opacity.value = opacity
        if record:
            self.doc["backdrop"] = {"path": str(image_path), "fit": fit,
                                    "opacity": opacity}
        return f"backdrop {image_path}"

    def _background(self, color: str = "#f6f6f7", *, record: bool = True) -> str:
        """Fill the canvas with a flat colour behind everything.

        This is not decoration — it is what a rendered scene needs to *not* be
        black. Glaxnimate's video export composites transparency onto black, so a
        scene with no full-canvas art exports on a black field; a stark white or
        light-grey card is the stick-figure default. The card is drawn oversized so
        a camera shake or a zoom-out never reveals an edge. Call it *first*, before
        any content: layers draw in creation order, so the earliest one sits behind
        everything.
        """
        from glaxnimate import utils

        from . import props as P

        comp = self.scene.comp
        lay = self.scene.layer("background")
        cw, ch = float(comp.width), float(comp.height)
        m = 120.0
        g = lay.add_shape("Group")
        P.draw_prop(g, {"shapes": [
            {"type": "rect", "x": -cw / 2 - m, "y": -ch / 2 - m,
             "w": cw + 2 * m, "h": ch + 2 * m, "color": color}]},
            x=0.0, ground_y=0.0)
        g.transform.position.value = utils.Point(cw / 2, ch / 2)
        if record:
            self.doc["background"] = {"color": color}
        return f"background {color}"

    def _backdrop_style(self, name: str, *, record: bool = True) -> str:
        """Set a whole themed world behind the action — a gradient sky, a night
        starfield, a blueprint grid, a chalkboard — instead of a flat card. Call it
        FIRST (it sits at the very back). Returns the ink/accent that read well in
        the theme, so a figure can be coloured to match (see theme_palette)."""
        from . import props as P

        comp = self.scene.comp
        lay = self.scene.layer("backdrop_style")
        shapes = scene_style.theme_backdrop(name, float(comp.width),
                                            float(comp.height), self.ground_y)
        P.draw_prop(lay.add_shape("Group"), {"shapes": shapes}, x=0.0, ground_y=0.0)
        if record:
            self.doc["backdrop_style"] = {"name": name}
        pal = scene_style.theme_palette(name)
        return f"backdrop_style {name!r} (ink {pal['ink']}, accent {pal['accent']})"

    def _path_fn(self, points, span: float):
        """A polyline sampler: point at fraction f/span along `points` (a (x,y) list)."""
        from ..cartoon.geometry import Vec2
        pts = [Vec2(float(a), float(b)) for a, b in points]

        def at(f):
            if len(pts) == 1:
                return pts[0]
            s = max(0.0, min(1.0, f / max(span, 1))) * (len(pts) - 1)
            i = int(s)
            if i >= len(pts) - 1:
                return pts[-1]
            a, b, frac = pts[i], pts[i + 1], s - i
            return Vec2(a.x + (b.x - a.x) * frac, a.y + (b.y - a.y) * frac)
        return at

    def _cursor(self, points, *, start: float = 0.0, frames: int | None = None,
                name: str = "cursor", record: bool = True) -> str:
        """A mouse cursor that sweeps along `points` — the antagonist of Animator vs
        Animation. Pair with drag() so it grabs and hauls the figure around."""
        from ..cartoon.geometry import Vec2
        from ..cartoon.motion import Sample

        n = frames or self.frames
        span = n - start
        path = self._path_fn(points, span)
        arrow = [
            {"type": "polygon", "points": [[0, 0], [0, 27], [7, 20], [12, 31],
                                            [18, 29], [12, 18], [21, 18]], "color": "#111"},
            {"type": "polygon", "points": [[2, 4], [2, 22], [7, 17], [11, 26],
                                            [14, 25], [10, 15], [16, 15]], "color": "#fff"},
        ]
        samples = [Sample(int(start + i), path(i), scale=Vec2(1, 1), angle=0.0)
                   for i in range(int(span) + 1)]
        self._add_moving_prop(arrow, samples, name=name, record=record)
        return f"cursor along {len(points)} point(s)"

    def _drag(self, body, points, *, grab: str = "head", frames: int | None = None,
              start: float = 0.0):
        """The figure grabbed and dragged by the cursor: a ragdoll pinned at `grab`
        (a joint, default the head) to the cursor path, dangling and flailing as it is
        hauled around — the Animator-vs-Animation manhandling. Returns a pose_fn for
        add_action; drive the matching cursor() along the SAME points."""
        from ..cartoon.geometry import Vec2
        from ..cartoon.rig import Pose
        from ..cartoon import physics

        n = frames or self.frames
        span = n - start
        path = self._path_fn(points, span)

        def pin_path(f):
            return path(f - start) if f >= start else path(0)

        p0 = path(0)
        pose0 = Pose(root=Vec2(p0.x, p0.y + body.hip_height * 0.5), root_angle=0.0)
        return physics.ragdoll(body, pose0, ground_y=self.ground_y, frames=n,
                               pin=grab, pin_path=pin_path)

    # ----------------------------------------------------------- gun & clones
    def _shoot(self, shooter, target_x: float, target_y: float, frame: float, *,
               bone: str = "arm_lower", color: str = "#ffd766") -> str:
        """Fire a shot from a character's hand at (target_x, target_y): a muzzle flash
        at the barrel, a bright tracer to the target, an impact where it lands, and a
        crack of sound — all on `frame`. Pair with actions.aim so the arm points the
        right way."""
        ch = self._char(shooter)
        bp = self._bone_tip(ch, bone, frame)
        self._add_effect("spark", bp.x, bp.y, frame)                 # muzzle flash
        self._beam(bp.x, bp.y, target_x, target_y, frame, frame + 1,   # tracer
                   color=color, core="#ffffff", width=6, opacity=0.9)
        self._add_effect("impact", target_x, target_y, frame + 1)     # hit
        self._add_sound("pop", frame)
        return f"{ch.name} shoots @f{frame:g} -> ({target_x:g},{target_y:g})"

    def _clones(self, body, factory, count: int, *, name: str = "clone",
                face: str | dict | None = None) -> str:
        """The duplication technique: spawn `count` copies of a character, each driven
        by factory(i) -> pose_fn (so each clone can be at its own place, phase or
        beat). Every clone is a real registered character the critic can inspect."""
        for i in range(int(count)):
            self._add_action(body, factory(i), name=f"{name}{i}", face=face)
        return f"{count} clone(s) '{name}0..{int(count) - 1}'"

    # -------------------------------------------------------- physics ragdoll
    def _ragdoll(self, body, x: float, *, y: float | None = None,
                 launch: tuple = (0.0, 0.0), spin: float = 0.0,
                 frames: int | None = None, from_pose=None):
        """A simulated ragdoll pose_fn: the figure goes limp and is thrown by `launch`
        (vx, vy) + `spin`, tumbling and settling under real gravity + ground collision.

        Hand it to add_action. `launch` is the velocity at the moment of losing
        control — a hit or a throw. Feet genuinely slide as it tumbles (it is physics,
        not a controlled walk), so unlike the acting verbs a ragdoll is not slip-clean.
        """
        from ..cartoon import physics
        from ..cartoon.geometry import Vec2
        from ..cartoon.rig import Pose

        pose0 = from_pose or Pose(root=Vec2(x, self.ground_y - body.hip_height),
                                  root_angle=0.0)
        return physics.ragdoll(body, pose0, ground_y=self.ground_y,
                               frames=frames or self.frames, launch=launch, spin=spin)

    # ----------------------------------------- two-figure & props-as-toys (WS4)
    def _char(self, character) -> Character:
        """Resolve a character by name or object, with a teaching error."""
        ch = character if isinstance(character, Character) else next(
            (c for c in self.characters if c.name == character), None)
        if ch is None:
            raise ValueError(
                f"no character {character!r}; have {[c.name for c in self.characters]}")
        return ch

    def _wield(self, character, prop, *, bone: str = "arm_lower",
               offset: tuple | None = None, scale: float = 1.0, spin: float = 0.0,
               record: bool = True) -> str:
        """Put a prop in a character's hand: it rides the bone for the whole clip.

        The Animator-vs-Animation signature — a stick figure *using* an object. The
        prop is drawn once into the hand bone's layer, so the bone's transform swings
        it for free (a sword follows a `swing`, a torch bobs with a walk). `bone`
        defaults to the near forearm; `offset` defaults to the bone tip (the hand).
        `spin` (total degrees over the clip) twirls the prop in the hand — a staff or
        nunchuck spin. Pair with a `swing` beat for a weapon strike, or `throw`.
        """
        from glaxnimate import utils

        ch = self._char(character)
        if bone not in ch.bone_layers:
            raise ValueError(
                f"{ch.name} has no bone {bone!r}; bones: {sorted(ch.bone_layers)}")
        data = ({"shapes": prop} if isinstance(prop, list)
                else prop if isinstance(prop, dict) else assets.load_prop(prop))
        hand = ch.bone_layers[bone]
        blen = ch.body.rig.joints[bone].length
        ox, oy = offset if offset is not None else (blen, 0.0)

        g = hand.add_shape("Group")
        from . import props as P
        P.draw_prop(g, data, x=0.0, ground_y=0.0, scale=scale)
        g.transform.position.value = utils.Point(float(ox), float(oy))
        if spin:  # twirl the prop in the hand (staff/nunchuck)
            g.transform.rotation.set_keyframe(0.0, 0.0)
            g.transform.rotation.set_keyframe(float(self.frames), float(spin))
        if record:
            self.doc.setdefault("wields", []).append({
                "character": ch.name, "prop": prop if isinstance(prop, str) else data,
                "bone": bone, "offset": [float(ox), float(oy)], "scale": scale,
                "spin": spin,
            })
        return f"{ch.name} wields a prop on {bone}"

    def _throw(self, prop, *, x0: float, y0: float, x1: float, y1: float,
               apex: float = 120.0, release: float = 0.0, frames: float | None = None,
               spin: float = 360.0, scale: float = 1.0, record: bool = True) -> str:
        """A thrown object: a prop on a ballistic arc, appearing at the `release` frame.

        The other half of props-as-toys — a wielded thing let go. It flies from
        (x0, y0) to (x1, y1) over a parabola `apex` px tall, spinning `spin` degrees,
        invisible until `release`. Reuses the prop-on-a-motion-path baker; the arc is
        the same parabola `jump` uses.
        """
        from glaxnimate import model

        from ..cartoon import motion
        from ..cartoon.geometry import Vec2
        from .bake import bake_prop_samples

        data = ({"shapes": prop} if isinstance(prop, list)
                else prop if isinstance(prop, dict) else assets.load_prop(prop))
        end = float(self.frames if frames is None else frames)
        span = max(end - release, 1.0)
        samples = []
        for i in range(int(span) + 1):
            s = i / span
            x = x0 + (x1 - x0) * s
            y = y0 + (y1 - y0) * s - apex * 4.0 * s * (1.0 - s)  # parabola, up = -y
            samples.append(motion.Sample(int(release + i), Vec2(x, y),
                                         scale=Vec2(1.0, 1.0), angle=spin * s))
        lname = f"thrown.{len(self.doc.setdefault('throws', []))}"
        lay = bake_prop_samples(self.scene, data, samples, scale=scale, layer_name=lname)
        if release > 0:  # invisible until it leaves the hand
            lay.opacity.set_keyframe(0.0, 0.0)
            tr = model.KeyframeTransition()
            tr.hold = True
            lay.opacity.set_transition(0.0, tr)
            lay.opacity.set_keyframe(float(release), 1.0)
        if record:
            self.doc["throws"].append({
                "prop": prop if isinstance(prop, str) else data,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1, "apex": apex,
                "release": release, "frames": frames, "spin": spin, "scale": scale,
                "layer": lname,
            })
        return f"thrown prop f{release:g}: ({x0:g},{y0:g})->({x1:g},{y1:g})"

    def _clash(self, frame: float, x: float, y: float, *, sfx: str | None = "splat",
               fx: str | None = "impact", camera: bool = True, zoom: float = 1.3,
               shake: float = 10.0) -> str:
        """The 'it connected' bundle on one contact frame: an impact flash, a hit sfx,
        and a camera punch-in + shake — the convergence in a single call.

        The poses are composed separately (the attacker's `hitstop`, the defender's
        `knockback`); this lands everything else on the frame they meet. A two-figure
        exchange is: bake the attacker with a hitstopped strike, bake the defender
        with sequence((idle, contact), (knockback, ...)), then clash() on contact.
        """
        msgs = []
        if fx:
            msgs.append(self._add_effect(fx, x, y, frame))
        if sfx:
            msgs.append(self._add_sound(sfx, frame))
        if camera:
            msgs.append(self._impact_camera(frame, x, y, zoom=zoom, shake=shake))
        return " | ".join(msgs)

    def _add_moving_prop(self, prop, samples, *, name: str | None = None,
                         scale: float | tuple[float, float] = 1.0,
                         radius: float | None = None, record: bool = True):
        """Put a data prop (many shapes) on a motion path (`motion.*` samples).

        `add_prop` pins a prop to the ground; `add_object` moves a single shape.
        A two-tone bar magnet, a paperclip, a compass needle — anything with more
        than one shape that also moves — needs both, which is this.

        It registers with the critic like `add_object` does, so a prop that flies
        off-canvas is caught. `radius` is only the critic's idea of the prop's
        size (bounds and ground checks); it does not affect drawing.
        """
        from .bake import bake_prop_samples

        if isinstance(prop, list):        # accept a bare shapes list, like add_shape
            prop = {"shapes": prop}
        data = (assets.load_prop(prop) if isinstance(prop, str)
                else assets.prop_validate({"version": 1, "kind": "prop", **prop}))
        samples = list(samples)
        name = name or f"prop{len(self.objects)}"
        self.objects.append((name, samples, radius if radius is not None else 8.0))
        if record:
            self.doc["objects"].append({
                "name": name,
                "samples": SD.samples_to_data(samples),
                "prop": data,          # <- the discriminator on replay
                "scale": scale,
                "radius": radius,
            })
        return bake_prop_samples(self.scene, data, samples, scale=scale,
                                 layer_name=name)

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
            # the explainer toolkit: diagrams as data
            "diagram": diagram,
            "arrow": diagram.arrow,
            "axes": diagram.axes,
            "plot": diagram.plot,
            "number_line": diagram.number_line,
            "grid": diagram.grid,
            "bar_chart": diagram.bar_chart,
            "brace": diagram.brace,
            "label": diagram.label,
            "human": presets.human,
            "biped": presets.biped,
            "quadruped": presets.quadruped,
            "stick": presets.stick,
            "lineart": presets.lineart,
            # render-skins — reskin any body in a named style
            "silhouette": presets.silhouette,
            "comic": presets.comic,
            "flat": presets.flat,
            "neon": presets.neon,
            "sketch": presets.sketch,
            "blocky": presets.blocky,
            "rubber_hose": presets.rubber_hose,
            "apply_style": presets.apply_style,
            "style_names": presets.style_names,
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
            "backdrop": self._backdrop,
            "background": self._background,
            "backdrop_style": self._backdrop_style,
            # scene-style building blocks (gradients, patterns, scenery, themes)
            "scene_style": scene_style,
            "theme_palette": scene_style.theme_palette,
            "theme_names": scene_style.theme_names,
            "gradient_bg": scene_style.gradient_bg,
            "radial_bg": scene_style.radial_bg,
            "cursor": self._cursor,
            "drag": self._drag,
            "add_sound": self._add_sound,
            "add_effect": self._add_effect,
            "add_shape": self._add_shape,
            "highlight": self._highlight,
            "write_on": self._write_on,
            "counter": self._counter,
            "emit": self._emit,
            "rope": self._rope,
            "leash": self._leash,
            "smear": self._smear,
            "aura": self._aura,
            "beam": self._beam,
            "afterimage": self._afterimage,
            "auto_fx": self._auto_fx,
            "add_moving_prop": self._add_moving_prop,
            "shot": self._shot,
            "impact_camera": self._impact_camera,
            "camera_move": self._camera_move,
            "auto_sfx": self._auto_sfx,
            "music": self._music,
            "beats": self._beats,
            "snap_to_beat": self._snap_to_beat,
            "loop_report": self._loop_report,
            "say": self._say,
            # the stage
            "add_character": self._add_character,
            "add_object": self._add_object,
            "add_chaser": self._add_chaser,
            "add_action": self._add_action,
            "wield": self._wield,
            "throw": self._throw,
            "clash": self._clash,
            "ragdoll": self._ragdoll,
            "shoot": self._shoot,
            "clones": self._clones,
            "physics": physics,
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

        bg = doc.get("background")           # very first, so it sits behind all
        if bg:
            session._background(bg["color"], record=False)
        bs = doc.get("backdrop_style")
        if bs:
            session._backdrop_style(bs["name"], record=False)
        bd = doc.get("backdrop")            # then the image backdrop
        if bd and Path(bd["path"]).exists():
            session._backdrop(bd["path"], fit=bd["fit"], opacity=bd["opacity"],
                              record=False)
        for sc in doc["scenery"]:
            session._scenery(sc["template"], layer_name=sc["layer"],
                             record=False, **sc["params"])
        for pr in doc["props"]:
            lay = session.scene.layer(pr.get("layer", "props"))
            from . import props as P
            y = pr.get("y")
            sc = pr.get("scale", 1.0)
            P.draw_prop(lay, pr["data"], x=pr["x"],
                        ground_y=session.ground_y if y is None else float(y),
                        scale=tuple(sc) if isinstance(sc, list) else sc)
        for rec in doc["characters"]:
            body = assets.body_from_data(rec["body"])
            pose_fn = SD.pose_lookup(rec["poses"])
            ch = session._bake_character(
                body, pose_fn, gait=None, name=rec["name"],
                color=rec.get("color"), thickness=rec.get("thickness"),
                face_data=rec.get("face"), record=False,
                expressions=[(f, a) for f, a in rec.get("expressions", [])],
                facing=rec.get("facing", 1.0), scale=rec.get("scale", 1.0),
            )
            ch.limb_pairs = [tuple(x) for x in rec.get("limbs", [])]
        for ob in doc["objects"]:
            from ..cartoon.geometry import Vec2
            smp = SD.samples_from_data(ob["samples"])
            if ob.get("prop"):
                sc = ob.get("scale", 1.0)
                session._add_moving_prop(
                    ob["prop"], smp, name=ob["name"],
                    scale=tuple(sc) if isinstance(sc, list) else sc,
                    radius=ob.get("radius"), record=False,
                )
                continue
            size = Vec2(*ob["size"]) if ob.get("size") else None
            session._add_object(
                SD.samples_from_data(ob["samples"]), record=False,
                shape=ob.get("shape", "Ellipse"), size=size,
                color=ob.get("color", "#e8543f"), layer_name=ob["name"],
            )
        for e in doc.get("effects", []):
            from ..cartoon.effects import resolve_fx
            from .bake import bake_effect
            lay = bake_effect(session.scene, resolve_fx(e["fx"]), x=e["x"], y=e["y"],
                              start=e["frame"], layer_name=e.get("layer", "fx"))
            if e.get("auto"):
                session.fx_layers.append(lay)
        for sp in doc.get("shapes", []):
            session._add_shape(
                sp["shapes"], sp["x"], sp["y"], name=sp["name"], scale=sp["scale"],
                pulse=tuple(sp["pulse"]) if sp["pulse"] else None, spin=sp["spin"],
                appear=sp["appear"], vanish=sp.get("vanish", 0.0), record=False)
        for wo in doc.get("writeons", []):
            session._write_on(wo["shapes"], wo["x"], wo["y"], start=wo["start"],
                              end=wo["end"], name=wo["name"], record=False)
        for co in doc.get("counters", []):
            session._counter(
                co["x"], co["y"], v0=co["v0"], v1=co["v1"], start=co["start"],
                end=co["end"], size=co["size"], color=co["color"], prefix=co["prefix"],
                suffix=co["suffix"], decimals=co["decimals"], anchor=co["anchor"],
                ease=co["ease"], step=co["step"], name=co["name"], record=False)
        for em in doc.get("emits", []):
            session._emit(
                em["fx"], em["x"], em["y"], count=em["count"], spread=em["spread"],
                drop=em["drop"], start=em["start"], over=em["over"], size=em["size"],
                color=em["color"], seed=em["seed"], record=False)
        for w in doc.get("wields", []):
            ch = next((c for c in session.characters if c.name == w["character"]), None)
            if ch:
                session._wield(ch, w["prop"], bone=w["bone"],
                               offset=tuple(w["offset"]), scale=w["scale"],
                               spin=w.get("spin", 0.0), record=False)
        for rp in doc.get("ropes", []):
            session._rope(rp["x0"], rp["y0"], rp["x1"], rp["y1"], sag=rp["sag"],
                          color=rp["color"], width=rp["width"], name=rp["name"], record=False)
        for ls in doc.get("leashes", []):
            if any(c.name == ls["a"] for c in session.characters) and \
               any(c.name == ls["b"] for c in session.characters):
                session._leash(ls["a"], ls["a_bone"], ls["b"], ls["b_bone"],
                               color=ls["color"], width=ls["width"], name=ls["name"],
                               record=False)
        for ov in doc.get("overlays", []):   # smear / aura / beam
            k = ov["kind"]
            if k == "smear":
                session._smear(ov["character"], ov["bone"], ov["at"], span=ov["span"],
                               color=ov["color"], spread=ov["spread"],
                               opacity=ov["opacity"], name=ov["name"], record=False)
            elif k == "aura":
                session._aura(ov["character"], color=ov["color"], radius=ov["radius"],
                              bone=ov["bone"], start=ov["start"], pulse=ov["pulse"],
                              opacity=ov["opacity"], name=ov["name"], record=False)
            elif k == "beam":
                session._beam(ov["x0"], ov["y0"], ov["x1"], ov["y1"], ov["start"],
                              ov["end"], color=ov["color"], core=ov["core"],
                              width=ov["width"], opacity=ov["opacity"], name=ov["name"],
                              record=False)
            elif k == "afterimage":
                session._afterimage(ov["character"], ov["at"], count=ov["count"],
                                    gap=ov["gap"], opacity=ov["opacity"],
                                    color=ov["color"], name=ov["name"], record=False)
        for th in doc.get("throws", []):
            session._throw(th["prop"], x0=th["x0"], y0=th["y0"], x1=th["x1"],
                           y1=th["y1"], apex=th["apex"], release=th["release"],
                           frames=th["frames"], spin=th["spin"], scale=th["scale"],
                           record=False)
        # after every layer exists, or a prefix would match nothing
        for sh in doc.get("shots", []):
            session._shot(sh["prefix"], sh["start"], sh["end"], record=False)
        # camera last of all: it re-parents every content layer under the world layer
        for cam in doc.get("camera", []):
            if cam["op"] == "impact":
                session._impact_camera(
                    cam["frame"], cam.get("x"), cam.get("y"), zoom=cam["zoom"],
                    shake=cam["shake"], ramp=cam["ramp"], hold=cam["hold"],
                    shake_frames=cam["shake_frames"], record=False)
            elif cam["op"] == "move":
                session._camera_move(
                    cam["start"], cam["end"], zoom=cam["zoom"],
                    focus_x=cam.get("x"), focus_y=cam.get("y"), record=False)
        for entry in (doc.get("audio") or {}).get("dialogue", []):
            if entry.get("bubble") and entry.get("character"):
                ch = next((c for c in session.characters
                           if c.name == entry["character"]), None)
                if ch:
                    session._draw_bubble(ch, entry["frame"], entry["dur"])
        return session

    # ---------------------------------------------------------------- running
    def run(self, code: str, *, timeout: float = 20.0) -> ScriptResult:
        """Execute a script against this session, capturing output and tracebacks.

        The traceback goes straight back to the model, which is the point: it reads
        the error and fixes itself, which is far cheaper than a round of rendering.

        The timeout is a trace hook, not SIGALRM. SIGALRM only works on the main
        thread — under the MCP server, scripts run on the dedicated Qt worker
        thread, where signal.signal raises ValueError (found by calling the tool
        the way the server actually does, not the way the unit tests did). The
        tracer interrupts any Python-level runaway (the realistic failure: an LLM
        writes an infinite loop); a hang inside a C call would not be caught,
        which is an accepted limit, not an oversight.
        """
        ns = self.namespace()
        buf = io.StringIO()
        deadline = time.monotonic() + timeout

        def tracer(frame, event, arg):  # noqa: ARG001
            if time.monotonic() > deadline:
                raise _Timeout(f"script exceeded {timeout:g}s")
            return tracer

        old_trace = sys.gettrace()
        sys.settrace(tracer)
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
            sys.settrace(old_trace)


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
