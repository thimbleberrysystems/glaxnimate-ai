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
import sys
import time
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
        return result, report

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

    def _say(self, character, text: str, frame: float,
             *, voice: str | None = None, gain: float = 1.0,
             bubble: bool = True) -> str:
        """A character speaks. TTS renders now and is CACHED to the project dir,
        so the scene replays its dialogue without piper installed — the same
        persist-the-samples rule the doc uses for poses. A speech bubble shows
        above the speaker for the duration (bubble=False for off-screen voice).
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
        return f"{ch.name if ch else 'voice'} says {text!r} at f{frame:g} ({dur_s:.1f}s)"

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
        """Music beds and dialogue lines, rendered onto the same bus as cues."""
        extra = []
        audio = self.doc.get("audio") or {}
        music = audio.get("music")
        if music:
            from ..audio.music import render_music

            duration = self.frames / self.scene.fps
            bed = render_music(music, duration_s=duration)
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

    def _add_moving_prop(self, prop, samples, *, name: str | None = None,
                         scale: float = 1.0, radius: float | None = None,
                         record: bool = True):
        """Put a data prop (many shapes) on a motion path (`motion.*` samples).

        `add_prop` pins a prop to the ground; `add_object` moves a single shape.
        A two-tone bar magnet, a paperclip, a compass needle — anything with more
        than one shape that also moves — needs both, which is this.

        It registers with the critic like `add_object` does, so a prop that flies
        off-canvas is caught. `radius` is only the critic's idea of the prop's
        size (bounds and ground checks); it does not affect drawing.
        """
        from .bake import bake_prop_samples

        data = (assets.load_prop(prop) if isinstance(prop, str)
                else assets.prop_validate(prop))
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
            "add_sound": self._add_sound,
            "add_moving_prop": self._add_moving_prop,
            "shot": self._shot,
            "auto_sfx": self._auto_sfx,
            "music": self._music,
            "say": self._say,
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
            smp = SD.samples_from_data(ob["samples"])
            if ob.get("prop"):
                session._add_moving_prop(
                    ob["prop"], smp, name=ob["name"], scale=ob.get("scale", 1.0),
                    radius=ob.get("radius"), record=False,
                )
                continue
            size = Vec2(*ob["size"]) if ob.get("size") else None
            session._add_object(
                SD.samples_from_data(ob["samples"]), record=False,
                shape=ob.get("shape", "Ellipse"), size=size,
                color=ob.get("color", "#e8543f"), layer_name=ob["name"],
            )
        # after every layer exists, or a prefix would match nothing
        for sh in doc.get("shots", []):
            session._shot(sh["prefix"], sh["start"], sh["end"], record=False)
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
