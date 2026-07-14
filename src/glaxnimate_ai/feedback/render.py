"""Tier 2 of the critic stack: pictures, for what numbers cannot judge.

Used sparingly and on purpose. The linter and the numeric diagnostics answer
"is it broken?" and "is it good?" for free; images are for the questions that
resist arithmetic — does this read as a character, is the composition any good.

**An LLM cannot watch a GIF.** It sees still images. So the useful render is a
*contact sheet*: a grid of frames with their numbers burned in, the whole motion
legible at a glance for roughly the token cost of one picture. Sixteen frames in
one 1024px sheet costs ~1,400 tokens; the same sixteen sent individually costs
~5,600 for strictly less context.

GIFs are for the human.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from ..engine.bake import Scene

__all__ = ["render_frame", "contact_sheet", "motion_trail", "save_gif"]

_BG = (250, 250, 250, 255)


def _flatten(img: Image.Image) -> Image.Image:
    """Composite onto a light background: RGBA transparency reads as black in a sheet."""
    bg = Image.new("RGBA", img.size, _BG)
    return Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")


def render_frame(scene: Scene, frame: int) -> Image.Image:
    return _flatten(scene.comp.render_image(frame))


def contact_sheet(
    scene: Scene,
    *,
    frames: list[int] | None = None,
    count: int = 12,
    cols: int = 4,
    width: int = 1024,
) -> Image.Image:
    """A grid of numbered frames — the primary review artefact.

    Frame numbers are drawn on each cell so a critique can name the frame it
    means ("the arm pops at 18") instead of gesturing at the motion in general.
    """
    last = int(scene.comp.animation.last_frame)
    if frames is None:
        # Sample within the animation: the final frame is exclusive and renders empty.
        frames = [round(i * (last - 1) / max(count - 1, 1)) for i in range(count)]

    shots = [_flatten(scene.comp.render_image(f)) for f in frames]
    rows = (len(shots) + cols - 1) // cols

    cw = width // cols
    ch = round(cw * shots[0].height / shots[0].width)

    sheet = Image.new("RGB", (cw * cols, ch * rows), (232, 232, 236))
    draw = ImageDraw.Draw(sheet)

    for i, (f, shot) in enumerate(zip(frames, shots, strict=True)):
        x, y = (i % cols) * cw, (i // cols) * ch
        sheet.paste(shot.resize((cw, ch), Image.LANCZOS), (x, y))
        draw.rectangle([x, y, x + cw - 1, y + ch - 1], outline=(200, 200, 206))
        draw.rectangle([x + 4, y + 4, x + 34, y + 20], fill=(30, 30, 36))
        draw.text((x + 9, y + 8), f"{f:>3}", fill=(255, 255, 255))

    return sheet


def motion_trail(
    scene: Scene,
    *,
    count: int = 10,
    width: int = 1024,
) -> Image.Image:
    """Onion skin: successive frames ghosted over one another.

    This is how an animator checks *arcs*. A limb whose path zigzags instead of
    sweeping is instantly visible here and essentially invisible in any single
    frame — which is why a contact sheet alone is not enough.
    """
    last = int(scene.comp.animation.last_frame)
    frames = [round(i * (last - 1) / max(count - 1, 1)) for i in range(count)]

    base = Image.new("RGBA", (scene.comp.width, scene.comp.height), _BG)
    for i, f in enumerate(frames):
        shot = scene.comp.render_image(f).convert("RGBA")
        # Older frames fade out, so the direction of travel is legible.
        alpha = int(40 + 215 * (i / max(len(frames) - 1, 1)))
        faded = shot.copy()
        faded.putalpha(shot.getchannel("A").point(lambda a, k=alpha: a * k // 255))
        base = Image.alpha_composite(base, faded)

    out = base.convert("RGB")
    h = round(width * out.height / out.width)
    return out.resize((width, h), Image.LANCZOS)


def save_gif(scene: Scene, path: str, *, step: int = 1) -> str:
    """A GIF for the human. Glaxnimate has no GIF exporter, so build it from frames."""
    last = int(scene.comp.animation.last_frame)
    shots = [_flatten(scene.comp.render_image(f)) for f in range(0, last, step)]
    ms = int(1000 / scene.fps * step)
    shots[0].save(path, save_all=True, append_images=shots[1:], duration=ms, loop=0, optimize=True)
    return path
