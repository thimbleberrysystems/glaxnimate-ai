"""The scene document: a session as data, on disk, replayable.

v1 kept scenes as Python closures in process memory — unserializable, gone on
restart, invisible to anything but the process that made them. The scene document
fixes that with one idea: **persist the sampled poses, not the program.**

A pose function — a gait, a jump, an arbitrary closure a script composed — is
only ever evaluated at integer frames. So sampling it once per frame captures it
*exactly*: the record replays into an identical bake, an identical timeline, an
identical lint report. No attempt is made to serialize code, which is why any
pose_fn a script can write is persistable, not just the ones we anticipated.

The document is versioned JSON:

    {"version": 1, "kind": "scene",
     "canvas": {...}, "ground_y": ...,
     "scenery": [{"template": "house", "params": {...}, "layer": ...}],
     "props":   [{"data": {...}, "x": ..., "scale": ...}],
     "characters": [{"name", "body": <body data>, "poses": [<pose per frame>],
                     "limbs": [[upper, lower]...], "face": <face data>|null,
                     "face_default": ..., "expressions": [[frame, att]...]}],
     "objects": [{"name", "samples": [[frame,x,y,angle,sx,sy]...],
                  "shape", "size", "color"}]}

`Session` records into this as helpers run and autosaves after every successful
script; `replay` rebuilds a live session from it. Restarting the server loses
nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

VERSION = 1

__all__ = ["VERSION", "empty_doc", "projects_root", "doc_path", "save_doc",
           "load_doc", "list_docs", "pose_lookup", "describe"]


def projects_root() -> Path:
    root = os.environ.get("GLAXNIMATE_AI_PROJECTS")
    return Path(root) if root else Path("projects")


def doc_path(doc_id: str) -> Path:
    return projects_root() / doc_id / "scene.json"


def empty_doc(*, width: int, height: int, frames: int, fps: float,
              ground_y: float) -> dict:
    return {
        "version": VERSION, "kind": "scene",
        "canvas": {"width": width, "height": height, "frames": frames, "fps": fps},
        "ground_y": ground_y,
        "scenery": [], "props": [], "characters": [], "objects": [], "shots": [],
        "audio": {"cues": [], "music": None, "dialogue": []},
    }


def save_doc(doc_id: str, doc: dict) -> Path:
    p = doc_path(doc_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def load_doc(doc_id: str) -> dict:
    p = doc_path(doc_id)
    if not p.exists():
        raise FileNotFoundError(
            f"no saved scene {doc_id!r}; saved scenes: {list_docs()}"
        )
    doc = json.loads(p.read_text())
    doc.setdefault("audio", {"cues": [], "music": None, "dialogue": []})
    if doc.get("version") != VERSION:
        raise ValueError(
            f"scene {doc_id!r} has version {doc.get('version')!r}; "
            f"this build reads version {VERSION}"
        )
    return doc


def list_docs() -> list[str]:
    root = projects_root()
    if not root.is_dir():
        return []
    return sorted(p.parent.name for p in root.glob("*/scene.json"))


# ------------------------------------------------------------------- poses
def sample_poses(pose_fn, frames: int) -> list[dict]:
    """Freeze a pose function into per-frame data. Exact, because the pipeline
    only ever evaluates poses at integer frames."""
    out = []
    for f in range(frames + 1):
        p = pose_fn(float(f))
        out.append({
            "root": [p.root.x, p.root.y],
            "root_angle": p.root_angle,
            "angles": dict(p.angles),
        })
    return out


def pose_lookup(poses: list[dict]):
    """The inverse of sample_poses: a pose_fn backed by recorded data."""
    from ..cartoon.geometry import Vec2
    from ..cartoon.rig import Pose

    def pose_fn(t: float):
        f = min(max(int(round(t)), 0), len(poses) - 1)
        rec = poses[f]
        return Pose(
            root=Vec2(*rec["root"]),
            root_angle=rec["root_angle"],
            angles=dict(rec["angles"]),
        )

    return pose_fn


def samples_to_data(samples) -> list[list[float]]:
    return [[s.frame, s.pos.x, s.pos.y, s.angle, s.scale.x, s.scale.y]
            for s in samples]


def samples_from_data(rows: list[list[float]]):
    from ..cartoon.geometry import Vec2
    from ..cartoon.motion import Sample

    return [Sample(int(r[0]), Vec2(r[1], r[2]),
                   scale=Vec2(r[4], r[5]), angle=r[3]) for r in rows]


# ---------------------------------------------------------------- describe
def describe(doc: dict) -> str:
    """A human/LLM-readable summary of what is in a scene."""
    c = doc["canvas"]
    lines = [
        f"canvas {c['width']}x{c['height']}, {c['frames']} frames @ {c['fps']}fps, "
        f"ground_y={doc['ground_y']:.0f}",
    ]
    if doc["scenery"]:
        items = ", ".join(f"{s['template']}@{s['params'].get('x', 0):.0f}"
                          for s in doc["scenery"])
        lines.append(f"scenery: {items}")
    if doc["props"]:
        lines.append(f"props: {len(doc['props'])} placed")
    for ch in doc["characters"]:
        joints = len(ch["body"]["joints"])
        face = f", face ({len(ch['face']['attachments'])} expressions)" if ch.get("face") else ""
        expr = f", swaps at {[f for f, _ in ch['expressions']]}" if ch.get("expressions") else ""
        lines.append(f"character {ch['name']!r}: {joints} joints{face}{expr}")
    for ob in doc["objects"]:
        lines.append(f"object {ob['name']!r}: {len(ob['samples'])} motion samples")
    audio = doc.get("audio") or {}
    bits = []
    if audio.get("cues"):
        bits.append(f"{len(audio['cues'])} sfx cue(s)")
    if audio.get("music"):
        bits.append(f"music (seed {audio['music'].get('seed')})")
    if audio.get("dialogue"):
        bits.append(f"{len(audio['dialogue'])} line(s) of dialogue")
    if bits:
        lines.append("audio: " + ", ".join(bits))
    if len(lines) == 1:
        lines.append("(empty scene)")
    return "\n".join(lines)
