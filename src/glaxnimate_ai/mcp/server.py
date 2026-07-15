"""The MCP surface. Deliberately small.

The tools are ordered the way the loop should run:

    new_document -> run_script -> lint -> diagnose -> (render) -> export

`lint` and `diagnose` are free and frame-precise; `render` costs ~1,400 tokens and
says "hmm". So the tool descriptions push the model down the cheap tiers first and
only reach for pictures when numbers cannot answer the question. That ordering is
the product, not an optimisation.

This server is the smallest part of the codebase. The library and the critic stack
are the product; this just exposes them.
"""

from __future__ import annotations

import io
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image as MCPImage
from PIL import Image

from ..engine.session import SessionStore
from ..feedback import render as R
from ..feedback.diagnose import diagnose_rig
from ..feedback.lint import lint_rig

mcp = FastMCP("glaxnimate-ai")
store = SessionStore()

OUT = Path("out")


def _png(img: Image.Image, max_px: int = 1024) -> MCPImage:
    """Hand an image to the model, capped in size.

    Image tokens go as (w x h) / 750, so a 2048px sheet costs 4x a 1024px one for
    no extra legibility. Cap it.
    """
    if img.width > max_px:
        img = img.resize((max_px, round(max_px * img.height / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return MCPImage(data=buf.getvalue(), format="png")


# --------------------------------------------------------------------- build
@mcp.tool()
def new_document(
    width: int = 960, height: int = 540, frames: int = 48, fps: float = 24.0
) -> str:
    """Start a new animation. Returns a doc_id to pass to every other tool.

    `frames` is the length. At 24fps a walk cycle is ~24 frames, so 48 gives two.
    The ground line defaults to 87% of the height.
    """
    s = store.create(width=width, height=height, frames=frames, fps=fps)
    return (
        f"{s.doc_id}: {width}x{height}, {frames} frames @ {fps}fps, "
        f"ground_y={s.ground_y:.0f}"
    )


@mcp.tool()
def run_script(doc_id: str, code: str) -> str:
    """Run Python against the cartoon library to build the animation. The workhorse.

    Available without import: `human`, `biped`, `quadruped`, `make_gait`,
    `add_character`, `add_object`, `motion`, `principles`, `presets`, `Vec2`,
    plus `ground`, `frames`, `width`, `height`.

    Call `cartoon_api()` first if you have not seen the library. Example:

        man = human()
        walk = make_gait(man, "walk", cycle_frames=24)
        add_character(man, walk, x=90, name="man")

        ball = motion.bounce(x0=60, x1=880, ground_y=ground,
                             apex=220, frames=frames, bounces=5)
        add_object(ball, color="#e8543f")

    Errors come back as a traceback — read it and fix the script. Do NOT reach for
    a render to find out what went wrong; run `lint_animation` first, it is free.
    """
    res = store.get(doc_id).run(code)
    return res.format()


@mcp.tool()
def cartoon_api() -> str:
    """The cartoon library's vocabulary. Read this before writing your first script."""
    return _API


# ------------------------------------------------------- the critic (cheap first)
@mcp.tool()
def lint_animation(doc_id: str) -> str:
    """TIER 0 - is it BROKEN? Free, instant, no image. Run this after every script.

    Catches the faults that are arithmetic rather than opinion: a planted foot that
    slides (the character is skating), a limb that cannot reach and so drags, feet
    below the ground, strobing, NaNs, anything off-canvas.
    """
    s = store.get(doc_id)
    if not s.characters:
        return "no characters registered; nothing to lint (use add_character)"

    out = []
    for ch in s.characters:
        rep = lint_rig(
            ch.body, ch.pose_fn, frames=s.frames, ground_y=s.ground_y,
            limbs=ch.gait.limbs, canvas=(s.scene.comp.width, s.scene.comp.height),
        )
        out.append(f"{ch.name}: {rep.format()}")
    return "\n".join(out)


@mcp.tool()
def diagnose_animation(doc_id: str, track: str | None = None) -> str:
    """TIER 1 - is it GOOD? ~500 tokens, frame-precise, still no image.

    The animator's instruments as numbers: the spacing chart (even spacing means
    dead-linear interpolation, i.e. nobody timed it), arc reversals on a tracked
    limb (a zigzag instead of a sweep), balance, and silhouette readability.

    Prefer this over rendering. It tells you *which* frame and *by how much*; a
    picture only tells you something looks off.
    """
    s = store.get(doc_id)
    if not s.characters:
        return "no characters registered; nothing to diagnose"

    out = []
    for ch in s.characters:
        d = diagnose_rig(
            ch.body, ch.pose_fn, frames=s.frames, ground_y=s.ground_y, track=track
        )
        out.append(f"{ch.name}:\n{d.format()}")
    return "\n".join(out)


# ------------------------------------------------------------ vision (last resort)
@mcp.tool()
def render_contact_sheet(doc_id: str, count: int = 8, cols: int = 4) -> MCPImage:
    """TIER 2 - LOOK at it. ~1,400 tokens. Use only for what numbers cannot judge.

    A grid of numbered frames: the whole motion in one image. Good for "does this
    read as a character", "is the composition any good". Bad for finding bugs — the
    linter already did that, for free.
    """
    s = store.get(doc_id)
    return _png(R.contact_sheet(s.scene, count=count, cols=cols))


@mcp.tool()
def render_frame(doc_id: str, frame: int) -> MCPImage:
    """TIER 2 - one frame, full size. For inspecting a specific moment in detail."""
    s = store.get(doc_id)
    return _png(R.render_frame(s.scene, frame))


@mcp.tool()
def render_motion_trail(doc_id: str, count: int = 10) -> MCPImage:
    """TIER 2 - onion skin: successive frames ghosted over each other.

    This is how you check ARCS. A limb whose path zigzags instead of sweeping is
    obvious here and invisible in any single frame.
    """
    s = store.get(doc_id)
    return _png(R.motion_trail(s.scene, count=count))


# -------------------------------------------------------------------- output
@mcp.tool()
def export(doc_id: str, filename: str, format: str = "json") -> str:
    """Export the animation. Formats: json (Lottie), rawr (Glaxnimate), svg, mp4,
    webm, webp, tgs (Telegram sticker), gif.

    Lottie and .rawr both open in the Glaxnimate GUI.
    """
    from glaxnimate import io as gio

    s = store.get(doc_id)
    OUT.mkdir(exist_ok=True)
    path = OUT / filename

    if format == "gif":
        # Glaxnimate has no GIF exporter, so build one from rendered frames.
        R.save_gif(s.scene, str(path))
        return f"wrote {path}"

    fmt = gio.registry.from_extension(format, gio.Direction.Export)
    if fmt is None:
        return f"no exporter for {format!r}"
    path.write_bytes(fmt.save(s.scene.comp))
    return f"wrote {path} ({path.stat().st_size:,} bytes)"


@mcp.tool()
def open_in_gui(doc_id: str, filename: str = "scene.rawr") -> str:
    """Open this animation in the Glaxnimate GUI so the user can see and edit it.

    Always works; needs no plugin. Use it when you want the user to look at the
    real thing rather than a contact sheet.
    """
    from glaxnimate import io as gio

    from ..engine.live import open_in_glaxnimate

    s = store.get(doc_id)
    OUT.mkdir(exist_ok=True)
    path = OUT / filename
    fmt = gio.registry.from_extension("rawr", gio.Direction.Export)
    path.write_bytes(fmt.save(s.scene.comp))
    return open_in_glaxnimate(path)


@mcp.tool()
def gui_live_run(code: str) -> str:
    """Edit the document open in a RUNNING Glaxnimate window, live.

    Requires the user to have clicked **Plugins > Start AI Bridge**. In scope:
    `document`, `comp`, `window`, `model`, `utils`. Each call is one undo step, so
    the user can Ctrl+Z anything you do.

    Use this to tweak a scene the user is already looking at. To build one from
    scratch, use `run_script` — it is headless, faster, and has the cartoon library.
    """
    from ..engine.live import BridgeUnavailable, LiveBridge

    try:
        r = LiveBridge().run(code)
    except BridgeUnavailable as e:
        return f"bridge not available: {e}"
    return r.get("result", "ok") if r.get("ok") else r.get("error", "failed")


@mcp.tool()
def gui_live_status() -> str:
    """Is a Glaxnimate window listening for live edits?"""
    from ..engine.live import BridgeUnavailable, LiveBridge

    try:
        r = LiveBridge(timeout=2.0).ping()
    except BridgeUnavailable as e:
        return f"no: {e}"
    return f"yes - live document is {r.get('size')}"


@mcp.tool()
def preview_for_human(doc_id: str, filename: str = "preview.gif") -> str:
    """TIER 4 - write a GIF for the *user* to watch.

    A human plus one sentence ("legs too stiff") is the highest-signal feedback in
    this whole system, and it costs no tokens at all. When you have taken it as far
    as the numbers can, hand it over.
    """
    s = store.get(doc_id)
    OUT.mkdir(exist_ok=True)
    path = OUT / filename
    R.save_gif(s.scene, str(path), step=2)
    return f"wrote {path} - ask the user to watch it and say what is wrong"


_API = """\
CARTOON LIBRARY
===============
Screen coords: +x right, +y DOWN. "Up" is negative y. Ground is a y value.

BODIES (rigs). A human is one preset among many; the engine animates anything.
  human() / biped(thigh=, shin=, spine=, arm=, forearm=, head=)  -> Body
  quadruped(upper=, lower=, body=, neck=, head=, tail=)          -> Body  (dog/cat/horse)
  body.hip_height, body.leg_length

GAITS. A gait is a phase table: N limbs offset around one cycle. Same code for
every creature.
  make_gait(body, name, cycle_frames=24, **overrides) -> Gait
  names: walk, run, trot, gallop, bound, hop
         (bipeds: walk/run/hop. quadrupeds: all six.)
  overrides: stride, duty, lift, bob, lean  (defaults scale with body size)
  A FASTER move comes from a SHORTER cycle_frames, not a longer stride — stride
  is bounded by leg length, and make_gait will reject a stride the legs cannot
  reach (it tells you by how much). Fast gaits crouch automatically.

  pace(body, name, distance=, frames=, cycle_frames=16) -> Gait
    A gait tuned to travel exactly `distance` px in `frames`. Use this when a
    character must ARRIVE somewhere — a door, a mark, another character.

STAGE
  add_character(body, gait, x=80, name="...", color=None, thickness=None)
     color/thickness default to None = use the body's own skin (a person looks
     like a person). Pass them only to flatten to one colour.
  add_object(samples, shape="Ellipse", size=Vec2(w,h), color="#e8543f")
  add_chaser(body, gait_name, target, x=60, gap=40, cycle_frames=16, name="...")
     A character PACED to chase `target` (a motion.* result) and end `gap` px
     behind it. Solves "the chaser lost the race" in one call — no per-frame
     metric catches that, because each character is individually fine; it is the
     relationship that is wrong.

MOTION (things without legs; no rig needed)
  motion.bounce(x0=, x1=, ground_y=, apex=, frames=, bounces=5, restitution=.62, radius=40)
  motion.roll(x0=, x1=, y=, radius=, frames=)      # wheel: spin locked to travel
  motion.spring(start=Vec2, end=Vec2, frames=)     # overshoot and settle
  motion.drift(start=, end=, frames=, sway_amount=)  # falling leaf
  motion.sway(pivot=, frames=, amplitude=, cycles=)  # rock in place

PRINCIPLES (apply to anything: a ball, a person, a logo)
  principles.ease_in / ease_out / ease_in_out / linear
  principles.anticipate(t)   # wind up before you go
  principles.overshoot(t)    # sail past, settle back
  principles.squash_stretch(speed)  -> Vec2 scale, area preserved
  principles.arc(a, b, t, height)   # living things move in arcs

IN SCOPE WITHOUT IMPORT: ground, frames, width, height, scene, Vec2

WORKFLOW - obey this order, it saves you tokens and time:
  1. run_script       build it
  2. lint_animation   is it broken?  FREE. always.
  3. diagnose_animation  is it good? ~500 tokens, names the frame.
  4. render_contact_sheet  only for what numbers cannot judge. ~1,400 tokens.
  5. preview_for_human  hand the human a GIF; their one sentence beats everything.
"""


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
