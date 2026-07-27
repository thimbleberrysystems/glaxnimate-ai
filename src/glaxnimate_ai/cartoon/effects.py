"""Effects: the visual juice that separates stick content from a moving diagram.

Content is data, exactly as sound is. An effect is a small declarative document —
prop-schema shapes drawn in local space plus a short animation envelope (grow the
scale, fade the opacity, maybe spin) that plays out over a `lifespan` in frames.
The engine bakes it at a position on a frame (`engine/bake.py::bake_effect`); the
envelope uses real `transform.scale`, which only became writable once the binding
was fixed.

The builtins below are the ones an action reel leans on — a dust puff where a foot
or body lands, an impact starburst where a blow connects, speed lines off a dash,
sparks off a clash, a fire burst on an explosion, a screen flash and a rainbow
glitch. New effects are `fx` JSON assets in the same shape; nothing here is
privileged over an authored one.
"""

from __future__ import annotations

import math

__all__ = ["BUILTIN_FX", "resolve_fx", "fx_names"]


def _star(points: int, r_out: float, r_in: float, color: str) -> dict:
    """A 2N-vertex star polygon centred on the origin — the classic 'pow' burst."""
    pts = []
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = math.pi * i / points - math.pi / 2
        pts.append([round(r * math.cos(a), 2), round(r * math.sin(a), 2)])
    return {"type": "polygon", "points": pts, "color": color}


#: Each builtin is a complete fx document (minus version/kind, added on resolve).
#: `shapes` are prop-schema shapes around the origin; `grow` scales from->to across
#: `lifespan` frames; `fade` ramps opacity 1->0; `spin` is total degrees.
BUILTIN_FX: dict[str, dict] = {
    # a soft ground puff: a low cluster of pale circles that balloons and fades
    "dust": {
        "lifespan": 7, "grow": [0.4, 1.6], "fade": True, "spin": 0,
        "shapes": [
            {"type": "ellipse", "cx": 0, "cy": -6, "w": 26, "h": 20, "color": "#c4c4c8"},
            {"type": "ellipse", "cx": -18, "cy": -2, "w": 18, "h": 14, "color": "#d2d2d6"},
            {"type": "ellipse", "cx": 18, "cy": -2, "w": 18, "h": 14, "color": "#d2d2d6"},
        ],
    },
    # a hard hit: a two-tone starburst that snaps out and vanishes fast
    "impact": {
        "lifespan": 5, "grow": [0.35, 1.5], "fade": True, "spin": 12,
        "shapes": [
            _star(10, 46, 17, "#ff5a3c"),
            _star(8, 30, 12, "#ffd23f"),
        ],
    },
    # motion streaks: a few thin ink lines trailing behind a fast move
    "speed_lines": {
        "lifespan": 4, "grow": [1.15, 0.75], "fade": True, "spin": 0,
        "shapes": [
            {"type": "rect", "x": -60, "y": -22, "w": 46, "h": 3, "color": "#1a1a1a"},
            {"type": "rect", "x": -70, "y": -4, "w": 60, "h": 4, "color": "#1a1a1a"},
            {"type": "rect", "x": -58, "y": 14, "w": 42, "h": 3, "color": "#1a1a1a"},
        ],
    },
    # a clash: small sparks flung outward
    "spark": {
        "lifespan": 5, "grow": [0.5, 1.7], "fade": True, "spin": 0,
        "shapes": [
            {"type": "polygon", "points": [[0, -20], [3, 0], [0, 20], [-3, 0]],
             "color": "#ffcf3f"},
            {"type": "polygon", "points": [[-20, 0], [0, 3], [20, 0], [0, -3]],
             "color": "#ffcf3f"},
            {"type": "polygon", "points": [[-14, -14], [2, 2], [14, 14], [-2, -2]],
             "color": "#ffd98a"},
        ],
    },
    # an ignition burst: layered flame tongues that leap up and fade. For a fire
    # that *keeps* burning, add the `flame` prop asset with a scale `pulse` instead;
    # this is the FWOOSH on the frame it catches.
    "fire": {
        "lifespan": 9, "grow": [0.5, 1.4], "fade": True, "spin": 0,
        "shapes": [
            {"type": "polygon", "points": [[-34, 22], [-18, -40], [-6, 10],
                                            [4, -64], [16, 6], [30, -34], [40, 22]],
             "color": "#ff5a1a"},
            {"type": "polygon", "points": [[-22, 22], [-8, -28], [2, 8],
                                            [11, -42], [24, 22]], "color": "#ffab26"},
            {"type": "polygon", "points": [[-11, 22], [0, -16], [11, 22]],
             "color": "#ffe870"},
        ],
    },
    # a screen flash: one bright panel that pops full and fades in three frames —
    # for a hit-frame whiteout, a camera pop, a lightning strike.
    "flash": {
        "lifespan": 3, "grow": [1.0, 1.0], "fade": True, "spin": 0,
        "shapes": [
            {"type": "rect", "x": -90, "y": -60, "w": 180, "h": 120, "color": "#ffffff"},
        ],
    },
    # a digital glitch: three saturated colour bars that flash over a screen and
    # vanish — a corruption, a hallucinated write, a broken render.
    "glitch": {
        "lifespan": 4, "grow": [1.0, 1.0], "fade": True, "spin": 0,
        "shapes": [
            {"type": "rect", "x": -66, "y": -44, "w": 44, "h": 88, "color": "#ff0055"},
            {"type": "rect", "x": -22, "y": -44, "w": 44, "h": 88, "color": "#00e5ff"},
            {"type": "rect", "x": 22, "y": -44, "w": 45, "h": 88, "color": "#ffe600"},
        ],
    },
}


def fx_names() -> list[str]:
    return sorted(BUILTIN_FX)


def resolve_fx(fx: str | dict) -> dict:
    """A builtin name or an inline/loaded fx document -> a validated fx document.

    Mirrors `audio.mix.resolve_patch`: the same call accepts `"impact"` and a full
    dict, so `add_effect` and `auto_fx` never have to care which they were given.
    """
    from .assets import fx_validate

    if isinstance(fx, str):
        if fx not in BUILTIN_FX:
            raise ValueError(
                f"unknown effect {fx!r}; builtins are {fx_names()} "
                f"(or pass an fx document / a saved fx asset name via load_asset)"
            )
        data = {"version": 1, "kind": "fx", **BUILTIN_FX[fx]}
    else:
        data = dict(fx)
        data.setdefault("version", 1)
        data.setdefault("kind", "fx")
    return fx_validate(data)
