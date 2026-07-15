"""Assets as data: bodies, gaits, props and actions as validated JSON.

This is the scalability fix v2 exists for. In v1 every creature, gesture and prop
was Python *we* had to write — two bodies cost 321 lines of `presets.py`, and the
LLM (supposedly the cartoon creator) could not grow its own vocabulary. Now a new
creature is a JSON document: authorable by the model through `save_asset`,
validated on load by the same checks the code path uses (`Rig.__init__` rejects
cycles and orphans; `make_gait`'s reach guard rejects impossible gaits), stored in
`assets/` as a growing, shareable library.

The parametric template functions (`biped()`, `quadruped()`) remain — they are
*generators* that emit body data, which is a legitimate job for code. What is gone
is the requirement that new content pass through us.

Every document carries `{"version": 1}`; loaders reject documents from the future
rather than mis-reading them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .gait import GAIT_DEFAULTS, GAIT_TABLE, Limb, Swing
from .geometry import Vec2
from .presets import Body, Part
from .rig import Joint, Rig

__all__ = [
    "assets_root", "asset_path", "list_assets",
    "body_to_data", "body_from_data", "save_body", "load_body",
    "prop_validate", "save_prop", "load_prop",
    "register_gait", "save_gait", "load_gait",
    "face_validate", "save_face", "load_face",
    "save_asset", "load_asset",
]

VERSION = 1
KINDS = ("body", "gait", "prop", "action", "face")
_DIRS = {"body": "bodies", "gait": "gaits", "prop": "props",
         "action": "actions", "face": "faces"}


def assets_root() -> Path:
    """Where the library lives. Env-overridable so tests and projects can isolate."""
    root = os.environ.get("GLAXNIMATE_AI_ASSETS")
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[3] / "assets"


def asset_path(kind: str, name: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown asset kind {kind!r}; have {KINDS}")
    return assets_root() / _DIRS[kind] / f"{name}.{kind}.json"


def list_assets() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for kind in KINDS:
        d = assets_root() / _DIRS[kind]
        if d.is_dir():
            out[kind] = sorted(p.name.removesuffix(f".{kind}.json")
                               for p in d.glob(f"*.{kind}.json"))
    return out


def _check_version(data: dict, kind: str) -> None:
    v = data.get("version")
    if v != VERSION:
        raise ValueError(
            f"{kind} document has version {v!r}; this build reads version {VERSION}. "
            f"Re-export the asset or update the library."
        )
    if data.get("kind") != kind:
        raise ValueError(f"expected a {kind!r} document, got kind={data.get('kind')!r}")


# --------------------------------------------------------------------- body
def body_to_data(body: Body) -> dict:
    """Serialize a Body — templates call this to emit shareable assets."""
    joints = []
    for j in body.rig.joints.values():
        joints.append({
            "name": j.name, "parent": j.parent, "length": j.length,
            "rest_angle": j.rest_angle,
            "offset": [j.offset.x, j.offset.y] if j.offset is not None else None,
            "contact": j.contact, "rolling": j.rolling, "radius": j.radius,
            "mass": j.mass,
        })
    parts = {
        name: {"width": p.width, "color": p.color,
               "head": list(p.head) if p.head else None, "tip": p.tip}
        for name, p in body.parts.items()
    }
    return {
        "version": VERSION, "kind": "body",
        "joints": joints,
        "limbs": [{"upper": limb.upper, "lower": limb.lower,
                   "bend_positive": limb.bend_positive} for limb in body.limbs],
        "swings": [{"joint": s.joint, "phase": s.phase, "amplitude": s.amplitude}
                   for s in body.swings],
        "bones": list(body.bones),
        "leg_length": body.leg_length,
        "parts": parts,
        "slots": {k: dict(v) for k, v in body.slots.items()},
    }


def body_from_data(data: dict) -> Body:
    """Build a Body from a document. Validation IS construction: `Rig.__init__`
    rejects cycles, duplicate names and unknown parents with teaching errors."""
    _check_version(data, "body")

    try:
        joints = [
            Joint(
                name=j["name"], parent=j.get("parent"),
                length=float(j.get("length", 0.0)),
                rest_angle=float(j.get("rest_angle", 0.0)),
                offset=Vec2(*j["offset"]) if j.get("offset") is not None else None,
                contact=bool(j.get("contact", False)),
                rolling=bool(j.get("rolling", False)),
                radius=float(j.get("radius", 0.0)),
                mass=float(j.get("mass", 0.0)),
            )
            for j in data["joints"]
        ]
    except KeyError as e:
        raise ValueError(f"every joint needs a {e.args[0]!r} field") from None

    rig = Rig(joints)

    limbs = []
    for entry in data.get("limbs", []):
        for k in ("upper", "lower"):
            if entry.get(k) not in rig.joints:
                raise ValueError(
                    f"limb references unknown joint {entry.get(k)!r}; "
                    f"joints are {sorted(rig.joints)}"
                )
        limbs.append(Limb(entry["upper"], entry["lower"],
                          bend_positive=bool(entry.get("bend_positive", True))))

    swings = [Swing(s["joint"], phase=float(s.get("phase", 0.0)),
                    amplitude=float(s.get("amplitude", 20.0)))
              for s in data.get("swings", [])]
    for s in swings:
        if s.joint not in rig.joints:
            raise ValueError(f"swing references unknown joint {s.joint!r}")

    bones = data.get("bones") or [n for n in rig.joints if rig.joints[n].length > 0]
    for b in bones:
        if b not in rig.joints:
            raise ValueError(f"bones lists unknown joint {b!r}")

    leg_length = float(data.get("leg_length") or 0.0)
    if not leg_length and limbs:
        leg_length = max(
            rig.joints[limb.upper].length + rig.joints[limb.lower].length
            for limb in limbs
        )

    parts = {}
    for name, p in (data.get("parts") or {}).items():
        if name not in rig.joints:
            raise ValueError(f"parts references unknown joint {name!r}")
        parts[name] = Part(
            width=float(p.get("width", 14.0)), color=p.get("color", "#33333c"),
            head=tuple(p["head"]) if p.get("head") else None,
            tip=float(p.get("tip", 0.0)),
        )

    slots = {}
    for name, sl in (data.get("slots") or {}).items():
        if sl.get("bone") not in rig.joints:
            raise ValueError(f"slot {name!r} references unknown bone {sl.get('bone')!r}")
        slots[name] = {"bone": sl["bone"], "offset": list(sl.get("offset", [0, 0]))}

    return Body(rig=rig, limbs=limbs, swings=swings,
                leg_length=leg_length, bones=list(bones), parts=parts, slots=slots)


def save_body(body: Body | dict, name: str) -> Path:
    data = body if isinstance(body, dict) else body_to_data(body)
    body_from_data(data)  # never persist an asset that would not load back
    p = asset_path("body", name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    return p


def load_body(name: str) -> Body:
    p = asset_path("body", name)
    if not p.exists():
        have = list_assets().get("body", [])
        raise FileNotFoundError(f"no body asset {name!r}; have {have}")
    return body_from_data(json.loads(p.read_text()))


# --------------------------------------------------------------------- gait
def register_gait(data: dict) -> str:
    """Add a custom gait to the live tables so `make_gait(body, name)` finds it.

    A gait document is a phase table plus ratio parameters:
        {"version": 1, "kind": "gait", "name": "scuttle",
         "phases": {"4": [0, 0.5, 0.25, 0.75]},
         "duty": 0.6, "stride": 0.8, "lift": 0.2, "bob": 0.05,
         "lean": 0, "crouch": 0.95}
    Ratios are fractions of hip height, exactly like the builtins — and the reach
    guard still applies when the gait is bound to a body, so an impossible gait
    fails at make_gait with the usual advice.
    """
    _check_version(data, "gait")
    name = data["name"]
    phases = {int(k): [float(x) for x in v] for k, v in data["phases"].items()}
    for n, ph in phases.items():
        if len(ph) != n:
            raise ValueError(f"phase table for {n} limbs has {len(ph)} entries")
        if not all(0.0 <= x < 1.0 for x in ph):
            raise ValueError("phase offsets must be in [0, 1)")
    for field in ("duty", "stride", "lift", "bob"):
        if field not in data:
            raise ValueError(f"gait document is missing {field!r}")
    GAIT_TABLE[name] = phases
    GAIT_DEFAULTS[name] = {
        "duty": float(data["duty"]), "stride": float(data["stride"]),
        "lift": float(data["lift"]), "bob": float(data["bob"]),
        "lean": float(data.get("lean", 0.0)), "crouch": float(data.get("crouch", 1.0)),
    }
    return name


def save_gait(data: dict, name: str | None = None) -> Path:
    name = name or data.get("name")
    register_gait(data)  # validates
    p = asset_path("gait", name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    return p


def load_gait(name: str) -> str:
    p = asset_path("gait", name)
    if not p.exists():
        raise FileNotFoundError(f"no gait asset {name!r}; have {list_assets().get('gait', [])}")
    return register_gait(json.loads(p.read_text()))


# --------------------------------------------------------------------- prop
_PROP_TYPES = ("rect", "ellipse", "polygon")


def prop_validate(data: dict) -> dict:
    """A prop is a declarative shape list in local coords, origin at its ground
    anchor. Exactly the class of thing an LLM can author, render, and fix."""
    _check_version(data, "prop")
    shapes = data.get("shapes")
    if not shapes:
        raise ValueError("a prop needs a non-empty 'shapes' list")
    for i, sh in enumerate(shapes):
        t = sh.get("type")
        if t not in _PROP_TYPES:
            raise ValueError(f"shapes[{i}].type is {t!r}; must be one of {_PROP_TYPES}")
        if t == "rect":
            missing = [k for k in ("x", "y", "w", "h") if k not in sh]
        elif t == "ellipse":
            missing = [k for k in ("cx", "cy", "w", "h") if k not in sh]
        else:
            missing = [] if sh.get("points") and len(sh["points"]) >= 3 else ["points (>=3)"]
        if missing:
            raise ValueError(f"shapes[{i}] ({t}) is missing {missing}")
    return data


def save_prop(data: dict, name: str) -> Path:
    prop_validate(data)
    p = asset_path("prop", name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    return p


def load_prop(name: str) -> dict:
    p = asset_path("prop", name)
    if not p.exists():
        raise FileNotFoundError(f"no prop asset {name!r}; have {list_assets().get('prop', [])}")
    return prop_validate(json.loads(p.read_text()))


# --------------------------------------------------------------------- face
def face_validate(data: dict) -> dict:
    """A face is a set of swappable attachments for one slot.

    Shapes use the prop schema, authored screen-aligned around the slot point at
    the body's rest pose (x = facing direction, y = down): the baker compensates
    for the slot bone's rest angle, so the same face document reads correctly on
    an upright human head and a tilted dog head.
    """
    _check_version(data, "face")
    if not data.get("slot"):
        raise ValueError("a face document needs a 'slot' name (usually 'face')")
    atts = data.get("attachments")
    if not atts:
        raise ValueError("a face needs a non-empty 'attachments' mapping")
    for shapes in atts.values():
        prop_validate({"version": VERSION, "kind": "prop", "shapes": shapes})
    return data


def save_face(data: dict, name: str) -> Path:
    face_validate(data)
    p = asset_path("face", name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    return p


def load_face(name: str) -> dict:
    p = asset_path("face", name)
    if not p.exists():
        raise FileNotFoundError(f"no face asset {name!r}; have {list_assets().get('face', [])}")
    return face_validate(json.loads(p.read_text()))


# ------------------------------------------------------------------ generic
_VALIDATORS = {"body": body_from_data, "gait": register_gait, "prop": prop_validate,
               "face": face_validate}


def save_asset(kind: str, name: str, data: dict) -> Path:
    """The LLM-facing entry point: validate, then persist. Never the reverse."""
    validator = _VALIDATORS.get(kind)
    if validator is None:
        raise ValueError(f"cannot save assets of kind {kind!r} yet; have {sorted(_VALIDATORS)}")
    validator(data)
    p = asset_path(kind, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    return p


def load_asset(kind: str, name: str) -> dict:
    p = asset_path(kind, name)
    if not p.exists():
        raise FileNotFoundError(
            f"no {kind} asset {name!r}; have {list_assets().get(kind, [])}"
        )
    return json.loads(p.read_text())
