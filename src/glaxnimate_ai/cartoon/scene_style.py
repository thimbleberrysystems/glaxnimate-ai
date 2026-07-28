"""Scene styles: the *world* a character stands in, as data.

A render-skin (`presets.STYLES`) changes how the figure is drawn; this changes
everything behind it — the thing that actually made every clip so far read as
"the same style" (a flat card). Backdrops are built from the same prop-schema
shapes as everything else (gradients are colour bands, a starfield is dots), so
they compose, persist and replay with no new engine machinery.

`THEMES` bundles a coordinated look — a backdrop plus the ink/accent a figure
should wear to sit in it — so one name ("night", "blueprint", "sunset") restyles
a whole scene. Deterministic: a starfield with the same seed lays identical stars,
so a themed scene renders the same every time.
"""

from __future__ import annotations

import math

__all__ = ["gradient_bg", "radial_bg", "dots", "grid_bg", "stripes", "stars",
           "hills", "THEMES", "theme_names", "theme_backdrop", "theme_palette"]

_M = 120.0  # oversize so camera shake/zoom never reveals an edge


def _lerp_hex(a: str, b: str, t: float) -> str:
    a, b = a.lstrip("#"), b.lstrip("#")
    ca = tuple(int(a[i:i + 2], 16) for i in (0, 2, 4))
    cb = tuple(int(b[i:i + 2], 16) for i in (0, 2, 4))
    return "#" + "".join(f"{int(ca[i] + (cb[i] - ca[i]) * t):02x}" for i in range(3))


def _rng(seed: int):
    s = seed & 0x7FFFFFFF or 1

    def nxt() -> float:
        nonlocal s
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        return s / 0x7FFFFFFF
    return nxt


# --------------------------------------------------------------- backdrops
def gradient_bg(w: float, h: float, top: str, bottom: str, *, bands: int = 60) -> list[dict]:
    """A vertical gradient as horizontal colour bands (no gradient binding needed,
    so it always renders and replays)."""
    sh = []
    bh = (h + 2 * _M) / bands
    for i in range(bands):
        y = -_M + i * bh
        sh.append({"type": "rect", "x": -_M, "y": y, "w": w + 2 * _M, "h": bh + 1,
                   "color": _lerp_hex(top, bottom, i / (bands - 1))})
    return sh


def radial_bg(w: float, h: float, inner: str, outer: str, *, cx: float | None = None,
              cy: float | None = None, rings: int = 44) -> list[dict]:
    """A radial glow — concentric ellipses from `inner` at the centre to `outer`.
    A spotlight, a sunrise, a portal."""
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    rmax = math.hypot(w + 2 * _M, h + 2 * _M)
    sh = [{"type": "rect", "x": -_M, "y": -_M, "w": w + 2 * _M, "h": h + 2 * _M,
           "color": outer}]
    for i in range(rings, 0, -1):
        t = i / rings
        r = rmax * t
        sh.append({"type": "ellipse", "cx": cx, "cy": cy, "w": r, "h": r,
                   "color": _lerp_hex(inner, outer, t)})
    return sh


def dots(w: float, h: float, *, color: str = "#d0d4dc", gap: float = 34.0,
         r: float = 3.0) -> list[dict]:
    """A polka / halftone dot field — comic and flat-design backdrops."""
    sh = []
    y = -_M
    while y < h + _M:
        x = -_M
        while x < w + _M:
            sh.append({"type": "ellipse", "cx": x, "cy": y, "w": r * 2, "h": r * 2,
                       "color": color})
            x += gap
        y += gap
    return sh


def grid_bg(w: float, h: float, *, color: str = "#3a5a8c", gap: float = 42.0,
            width: float = 1.2) -> list[dict]:
    """A ruled grid — blueprint, graph paper, a HUD."""
    sh = []
    x = 0.0
    while x < w + _M:
        sh.append({"type": "polyline", "points": [[x, -_M], [x, h + _M]],
                   "color": color, "width": width})
        x += gap
    y = 0.0
    while y < h + _M:
        sh.append({"type": "polyline", "points": [[-_M, y], [w + _M, y]],
                   "color": color, "width": width})
        y += gap
    return sh


def stripes(w: float, h: float, *, a: str = "#ffd76a", b: str = "#ffcf4a",
            band: float = 46.0) -> list[dict]:
    """Alternating diagonal bands — a sunburst backdrop, a circus, a retro sky."""
    sh = [{"type": "rect", "x": -_M, "y": -_M, "w": w + 2 * _M, "h": h + 2 * _M, "color": a}]
    x = -_M
    i = 0
    while x < w + _M:
        if i % 2:
            sh.append({"type": "polygon", "color": b, "points": [
                [x, -_M], [x + band, -_M], [x + band - h - 2 * _M, h + _M],
                [x - h - 2 * _M, h + _M]]})
        x += band
        i += 1
    return sh


def stars(w: float, h: float, *, n: int = 90, color: str = "#eef2ff",
          seed: int = 7, up_to: float | None = None) -> list[dict]:
    """A deterministic starfield in the top band of the canvas (night skies)."""
    rnd = _rng(seed)
    top = up_to if up_to is not None else h * 0.7
    sh = []
    for _ in range(n):
        x = rnd() * (w + 2 * _M) - _M
        y = rnd() * (top + _M) - _M
        s = 1.2 + rnd() * 2.6
        sh.append({"type": "ellipse", "cx": x, "cy": y, "w": s, "h": s, "color": color})
    return sh


def hills(w: float, h: float, ground_y: float, *, colors=("#7bc47f", "#5aa860"),
          layers: int = 2) -> list[dict]:
    """Rolling silhouette hills rising to `ground_y` — a landscape floor."""
    sh = []
    for li in range(layers):
        base = ground_y - li * 26
        amp = 26 + li * 10
        pts = [[-_M, h + _M], [-_M, base]]
        x = -_M
        while x <= w + _M:
            pts.append([x, base - amp * (0.5 + 0.5 * math.sin(x / 130.0 + li * 1.7))])
            x += 40
        pts += [[w + _M, base], [w + _M, h + _M]]
        sh.append({"type": "polygon", "points": pts, "color": colors[li % len(colors)]})
    return sh


# ------------------------------------------------------------------- themes
def _night(w, h, gy):
    return radial_bg(w, h, "#243b6b", "#0b1026") + stars(w, h, n=110)


def _sunset(w, h, gy):
    return (gradient_bg(w, h, "#ffd27a", "#c34d6e")
            + [{"type": "ellipse", "cx": w * 0.5, "cy": gy - 40, "w": 150, "h": 150,
                "color": "#fff1c9"}]
            + hills(w, h, gy, colors=("#7a4a63", "#5c3550")))


def _blueprint(w, h, gy):
    return [{"type": "rect", "x": -_M, "y": -_M, "w": w + 2 * _M, "h": h + 2 * _M,
             "color": "#0f3a67"}] + grid_bg(w, h, color="#2f6ba6", gap=40)


def _notebook(w, h, gy):
    sh = [{"type": "rect", "x": -_M, "y": -_M, "w": w + 2 * _M, "h": h + 2 * _M,
           "color": "#fbf8ef"}]
    y = 40.0
    while y < h + _M:
        sh.append({"type": "polyline", "points": [[-_M, y], [w + _M, y]],
                   "color": "#cfe0ea", "width": 1.4})
        y += 34
    sh.append({"type": "polyline", "points": [[70, -_M], [70, h + _M]],
               "color": "#e7a6a0", "width": 1.6})
    return sh


def _chalkboard(w, h, gy):
    return ([{"type": "rect", "x": -_M, "y": -_M, "w": w + 2 * _M, "h": h + 2 * _M,
              "color": "#1f3b32"}]
            + grid_bg(w, h, color="#2c4b41", gap=60, width=1.0))


def _vaporwave(w, h, gy):
    return (gradient_bg(w, h, "#2b1055", "#ff6ac1")
            + [{"type": "ellipse", "cx": w * 0.5, "cy": gy - 70, "w": 170, "h": 170,
                "color": "#ffd36e"}]
            + grid_bg(w, h, color="#c86bff", gap=44, width=1.4))


def _comic(w, h, gy):
    return [{"type": "rect", "x": -_M, "y": -_M, "w": w + 2 * _M, "h": h + 2 * _M,
             "color": "#fff6e0"}] + dots(w, h, color="#ffd25a", gap=30, r=4)


def _spotlight(w, h, gy):
    return radial_bg(w, h, "#3a3a44", "#0e0e12", cx=w / 2, cy=gy - 120)


def _sky(w, h, gy):
    return (gradient_bg(w, h, "#bfe3f5", "#eaf6fb")
            + [{"type": "ellipse", "cx": w * 0.78, "cy": 90, "w": 90, "h": 90,
                "color": "#ffe9a8"}]
            + hills(w, h, gy, colors=("#8fd08a", "#69b56a")))


def _paper(w, h, gy):
    return [{"type": "rect", "x": -_M, "y": -_M, "w": w + 2 * _M, "h": h + 2 * _M,
             "color": "#ece3d0"}]


#: name -> (backdrop builder, palette hints for the figures that stand in it)
THEMES = {
    "night":      (_night,      {"ink": "#e7ecff", "accent": "#8b5cff", "ground": "#141a33"}),
    "sunset":     (_sunset,     {"ink": "#2a1622", "accent": "#ffd27a", "ground": "#5c3550"}),
    "blueprint":  (_blueprint,  {"ink": "#eaf4ff", "accent": "#8fd0ff", "ground": "#0c3057"}),
    "notebook":   (_notebook,   {"ink": "#28407a", "accent": "#e0533d", "ground": "#cfe0ea"}),
    "chalkboard": (_chalkboard, {"ink": "#f3f6ee", "accent": "#ffe08a", "ground": "#16302a"}),
    "vaporwave":  (_vaporwave,  {"ink": "#101024", "accent": "#00f5d4", "ground": "#3a1a66"}),
    "comic":      (_comic,      {"ink": "#181818", "accent": "#e0533d", "ground": "#f6c23c"}),
    "spotlight":  (_spotlight,  {"ink": "#f2f2f6", "accent": "#ffd23f", "ground": "#101015"}),
    "sky":        (_sky,        {"ink": "#22303a", "accent": "#e0533d", "ground": "#69b56a"}),
    "paper":      (_paper,      {"ink": "#3a3630", "accent": "#b5533c", "ground": "#c9bfa6"}),
}


def theme_names() -> list[str]:
    return sorted(THEMES)


def theme_backdrop(name: str, w: float, h: float, ground_y: float) -> list[dict]:
    """The full backdrop shape list for a theme, sized to the canvas."""
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; have {theme_names()}")
    return THEMES[name][0](w, h, ground_y)


def theme_palette(name: str) -> dict:
    """The ink / accent / ground colours that read well in a theme — so a figure
    dropped into "night" can wear light ink without the author guessing."""
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; have {theme_names()}")
    return dict(THEMES[name][1])
