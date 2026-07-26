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

import asyncio
import functools
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image as MCPImage
from PIL import Image

from ..engine.session import SessionStore
from ..feedback import render as R
from ..feedback.diagnose import diagnose_rig
from ..feedback.lint import lint_object, lint_rig

mcp = FastMCP("glaxnimate-ai")

# ONE thread owns Qt. Glaxnimate documents (and the Headless environment itself)
# live exclusively on this worker: FastMCP calls sync tools inline on the event
# loop, so in v1 a render or a 20-second script froze the whole server — and the
# moment anyone "fixed" that with a thread pool, Qt objects would be touched from
# many threads, which is the segfault class phase B dug out of QUndoStack::push.
# A single-thread executor gives both properties at once: the event loop stays
# free (list_tools and the gui_live_* tools answer while a bake runs), and every
# Qt object only ever sees one thread.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glax")
_store: SessionStore | None = None


class _Store:
    """Deferred store: created on first use, ON the worker thread, so the Qt
    environment is born where it will live."""

    def __getattr__(self, name):
        global _store
        if _store is None:
            _store = SessionStore()
        return getattr(_store, name)


store = _Store()


def qt_tool(fn):
    """Run a sync tool body on the Qt worker thread; the event loop stays free.

    `functools.wraps` preserves the signature FastMCP introspects for the tool
    schema, and the wrapper being async is what moves execution off the loop.
    """

    @functools.wraps(fn)
    async def wrapper(*a, **kw):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_worker, functools.partial(fn, *a, **kw))

    return wrapper


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
#: Social-format canvas presets. `preset` overrides width/height on new_document.
_FORMATS = {
    "landscape": (960, 540),   # 16:9 — YouTube, the default
    "portrait": (540, 960),    # 9:16 — TikTok / Reels / Shorts
    "square": (720, 720),      # 1:1 — feed posts
    "sticker": (512, 512),     # Telegram tgs sticker
}


@mcp.tool()
@qt_tool
def new_document(
    width: int = 960, height: int = 540, frames: int = 48, fps: float = 24.0,
    preset: str | None = None,
) -> str:
    """Start a new animation. Returns a doc_id to pass to every other tool.

    `frames` is the length. At 24fps a walk cycle is ~24 frames, so 48 gives two.
    The ground line defaults to 87% of the height. `preset` sets the canvas for a
    platform and overrides width/height: portrait (9:16, TikTok/Shorts), square
    (1:1), sticker (512, Telegram tgs), landscape (16:9, the default). For a
    seamless short/sticker, make `frames` a whole motion cycle and check loop_report.
    """
    if preset is not None:
        if preset not in _FORMATS:
            raise ValueError(f"unknown preset {preset!r}; have {sorted(_FORMATS)}")
        width, height = _FORMATS[preset]
    s = store.create(width=width, height=height, frames=frames, fps=fps)
    tag = f" [{preset}]" if preset else ""
    return (
        f"{s.doc_id}: {width}x{height}{tag}, {frames} frames @ {fps}fps, "
        f"ground_y={s.ground_y:.0f}. Scenes autosave and survive restarts - "
        f"describe_scene(doc_id) shows what is in one."
    )


@mcp.tool()
@qt_tool
def run_script(doc_id: str, code: str) -> str:
    """Run Python against the cartoon library to build the animation. The workhorse.

    Available without import: `human`, `biped`, `quadruped`, `make_gait`,
    `add_character`, `add_object`, `motion`, `principles`, `presets`, `Vec2`,
    sound (`auto_sfx`, `add_sound`, `music`, `say`), plus `ground`, `frames`,
    `width`, `height`.

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
@qt_tool
def lint_animation(doc_id: str) -> str:
    """TIER 0 - is it BROKEN? Free, instant, no image. Run this after every script.

    Catches the faults that are arithmetic rather than opinion: a planted foot that
    slides (the character is skating), a limb that cannot reach and so drags, feet
    below the ground, strobing, NaNs, anything off-canvas.
    """
    s = store.get(doc_id)
    if not s.characters and not s.objects:
        return "nothing registered to lint (use add_character / add_object)"

    canvas = (int(s.scene.comp.width), int(s.scene.comp.height))
    out = []
    for ch in s.characters:
        rep = lint_rig(
            ch.body, ch.pose_fn, frames=s.frames, ground_y=s.ground_y,
            limbs=ch.limb_pairs or None, canvas=canvas,
        )
        out.append(f"{ch.name}: {rep.format()}")
    for name, samples, radius in s.objects:
        rep = lint_object(name, samples, ground_y=s.ground_y, radius=radius, canvas=canvas)
        out.append(f"{name}: {rep.format()}")
    return "\n".join(out)


@mcp.tool()
@qt_tool
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


@mcp.tool()
@qt_tool
def describe_scene(doc_id: str) -> str:
    """What is in this scene, as data: canvas, scenery, characters (with faces and
    expression swaps), objects. Scenes persist to disk and survive restarts —
    passing a doc_id from a previous session reloads it transparently."""
    from ..engine import scene_doc as SD

    s = store.get(doc_id)
    return SD.describe(s.doc)


@mcp.tool()
@qt_tool
def loop_report(doc_id: str) -> str:
    """Does it loop seamlessly? For a sticker or a short that plays on repeat, the
    last frame should land back on the first. Reports each character's pose drift
    between frame 0 and the final frame — near zero is a clean loop. Fix a jump by
    making `frames` a whole motion cycle (a walk loops at exactly its cycle length)."""
    return store.get(doc_id)._loop_report()


# -------------------------------------------------------------------- sound
@mcp.tool()
@qt_tool
def auto_sfx(doc_id: str, gain: float = 1.0) -> str:
    """The foley pass: derive sound cues FROM the motion. Zero guessing.

    The same Timeline data the linter reads yields foot plants, ball-ground
    hits, jump launches/landings and expression swaps; each becomes a cue on
    the exact frame, panned to where it happens on screen. Defaults:
    plant→step, hit→boing, launch→whoosh, land→thud, expression→pop.
    Run it AFTER the animation lints clean (cues are placed from current
    motion; re-run it if you change the motion). For overrides, call
    `auto_sfx({...})` inside run_script.
    """
    return store.get(doc_id)._auto_sfx(gain=gain)


@mcp.tool()
@qt_tool
def add_sound(doc_id: str, sfx: str, frame: float,
              gain: float = 1.0, pan: float = 0.0) -> str:
    """Place one sound cue by hand (auto_sfx covers motion-driven sounds).

    `sfx` is a builtin patch (boing, thud, step, pop, whoosh, slide_up,
    slide_down, splat, ding) or a saved sfx asset name. pan is -1 (left)
    to 1 (right). New sounds are sfx assets: JSON synth patches saved via
    save_asset — author a patch, save it, cue it by name.
    """
    return store.get(doc_id)._add_sound(sfx, frame, gain=gain, pan=pan)


@mcp.tool()
@qt_tool
def auto_fx(doc_id: str) -> str:
    """The visual-juice pass: spawn effects FROM the motion — the picture twin of
    auto_sfx. Zero guessing.

    The same Timeline events that drive foley — foot plants, ball-ground hits, jump
    launches and landings — spawn dust puffs, impact flashes and speed lines, on the
    exact frames, at the right screen positions. So a landing gets its thud AND its
    dust from one event. Defaults: plant→dust, hit→impact, launch→speed_lines,
    land→dust. Run it AFTER the motion lints clean; for overrides or contact-frame
    hits (a punch landing), use add_effect / auto_fx({...}) inside run_script.
    """
    return store.get(doc_id)._auto_fx()


@mcp.tool()
@qt_tool
def add_effect(doc_id: str, fx: str, x: float, y: float, frame: float) -> str:
    """Place one visual effect by hand (auto_fx covers motion-driven ones).

    `fx` is a builtin (impact, dust, speed_lines, spark) or a saved fx asset name.
    The effect pops in on `frame` at (x, y) — screen coords, y down — then grows and
    fades over its lifespan. Put an `impact` on a strike's contact frame (beats land
    ~60% through) with a hit sfx on the same frame. New effects are fx assets: shapes
    plus a grow/fade envelope, saved via save_asset.
    """
    return store.get(doc_id)._add_effect(fx, x, y, frame)


@mcp.tool()
@qt_tool
def impact_camera(doc_id: str, frame: float, x: float | None = None,
                  y: float | None = None, zoom: float = 1.2,
                  shake: float = 9.0) -> str:
    """Sell a hit: a fast camera punch-in toward (x, y) plus a decaying screen shake,
    both on the contact `frame`. Fire it on the SAME frame as the impact flash, the
    hit sfx and the beat's hitstop and the whole blow lands as one moment — this is
    the single biggest 'it feels like it connected' lever. Returns to neutral after.

    Call it AFTER the scene is built (it re-parents all content under a camera layer)
    and after any shot() gating. Deterministic — replays identically."""
    return store.get(doc_id)._impact_camera(frame, x, y, zoom=zoom, shake=shake)


@mcp.tool()
@qt_tool
def camera_move(doc_id: str, start: float, end: float, zoom: float = 1.0,
                focus_x: float | None = None, focus_y: float | None = None) -> str:
    """A held camera move for staging: ease from neutral framing to `zoom` centred on
    (focus_x, focus_y) between `start` and `end`. zoom>1 pushes in; zoom=1 with a
    focus pans. Call after the scene is built (re-parents content under the camera)."""
    return store.get(doc_id)._camera_move(start, end, zoom=zoom,
                                          focus_x=focus_x, focus_y=focus_y)


@mcp.tool()
@qt_tool
def clash(doc_id: str, frame: float, x: float, y: float,
          sfx: str = "splat", fx: str = "impact", zoom: float = 1.3,
          shake: float = 10.0) -> str:
    """The 'it connected' bundle on one contact frame: impact flash + hit sfx + camera
    punch-in and shake, in a single call. Use it for the moment two figures meet.

    The poses are composed in run_script: bake the attacker with a hitstopped strike
    (actions.hitstop(actions.punch(...), at=<contact>)), bake the defender with
    actions.sequence((actions.idle(...), <contact>), (actions.knockback(...), ...)),
    then clash() on the contact frame. For a wielded weapon or a thrown object, use
    wield()/throw() inside run_script (they need a prop and exact coordinates)."""
    return store.get(doc_id)._clash(frame, x, y, sfx=sfx or None, fx=fx or None,
                                    zoom=zoom, shake=shake)


@mcp.tool()
@qt_tool
def say(doc_id: str, character: str, text: str, frame: float,
        voice: str | None = None) -> str:
    """A character speaks the line, with a speech bubble for the duration.

    Local neural TTS (piper). The rendered audio is cached inside the project,
    so the scene replays its dialogue forever without re-synthesis. If the
    voice model is missing, the error contains the exact download command —
    relay it to the user (one ~60MB download, network needed once).
    """
    return store.get(doc_id)._say(character, text, frame, voice=voice)


@mcp.tool()
@qt_tool
def sound_report(doc_id: str) -> str:
    """TIER 0 for AUDIO - the mix as numbers, no listening required. Free.

    You cannot hear your own soundtrack; this is how you check it anyway:
    the cue sheet (what plays when), peak dBFS (quiet mixes read below -20;
    the limiter guarantees no clipping), and pile-up warnings when too many
    cues land together. The human ear is the final tier - preview_for_human
    writes an MP4 with the soundtrack when audio exists.
    """
    s = store.get(doc_id)
    if not s.has_audio:
        return ("no audio in this scene - auto_sfx derives cues from the "
                "motion; add_sound/music/say add more")
    _, report = s.audio_mix()
    return report


# ------------------------------------------------------------------- assets
@mcp.tool()
@qt_tool
def save_asset(kind: str, name: str, data: str) -> str:
    """Save a new asset (body/gait/prop) to the library as JSON. THE growth path.

    This is how you add a creature, gait or prop that does not exist yet: author
    the JSON, save it, then use it by name in run_script (`load_body("bird")`,
    `make_gait(body, "scuttle")`, `add_prop("bench")`). Assets are validated
    before saving — a body with a joint cycle, a gait whose legs cannot reach, or
    a malformed prop is rejected with an error that says what to fix.

    Schemas: body = {version:1, kind:"body", joints:[{name,parent,length,
    rest_angle,offset,contact,mass}], limbs:[{upper,lower,bend_positive}],
    swings:[...], bones:[draw order], parts:{joint:{width,color,head,tip}}}.
    gait = {version:1, kind:"gait", name, phases:{"2":[0,.5]}, duty, stride,
    lift, bob, lean, crouch} (ratios of hip height). prop = {version:1,
    kind:"prop", shapes:[{type:rect|ellipse|polygon, ...}]} with origin at the
    ground anchor, negative y up. Look at an existing asset first: load_asset.
    """
    import json as _json

    from ..cartoon import assets as A

    try:
        path = A.save_asset(kind, name, _json.loads(data))
    except (ValueError, KeyError) as e:
        return f"rejected: {e}"
    return f"saved {path.name} - use it by name in run_script"


@mcp.tool()
@qt_tool
def list_assets() -> str:
    """Everything in the asset library, by kind."""
    from ..cartoon import assets as A

    listing = A.list_assets()
    if not listing:
        return "the asset library is empty"
    return "\n".join(f"{kind}: {', '.join(names)}" for kind, names in listing.items())


@mcp.tool()
@qt_tool
def load_asset(kind: str, name: str) -> str:
    """Read an asset's JSON — the fastest way to learn a schema is a real example."""
    import json as _json

    from ..cartoon import assets as A

    try:
        return _json.dumps(A.load_asset(kind, name), indent=1)
    except (FileNotFoundError, ValueError) as e:
        return str(e)


# ------------------------------------------------------------ vision (last resort)
@mcp.tool()
@qt_tool
def render_contact_sheet(doc_id: str, count: int = 8, cols: int = 4) -> MCPImage:
    """TIER 2 - LOOK at it. ~1,400 tokens. Use only for what numbers cannot judge.

    A grid of numbered frames: the whole motion in one image. Good for "does this
    read as a character", "is the composition any good". Bad for finding bugs — the
    linter already did that, for free.
    """
    s = store.get(doc_id)
    return _png(R.contact_sheet(s.scene, count=count, cols=cols))


@mcp.tool()
@qt_tool
def render_frame(doc_id: str, frame: int) -> MCPImage:
    """TIER 2 - one frame, full size. For inspecting a specific moment in detail."""
    s = store.get(doc_id)
    return _png(R.render_frame(s.scene, frame))


@mcp.tool()
@qt_tool
def render_motion_trail(doc_id: str, count: int = 10) -> MCPImage:
    """TIER 2 - onion skin: successive frames ghosted over each other.

    This is how you check ARCS. A limb whose path zigzags instead of sweeping is
    obvious here and invisible in any single frame.
    """
    s = store.get(doc_id)
    return _png(R.motion_trail(s.scene, count=count))


# -------------------------------------------------------------------- output
@mcp.tool()
@qt_tool
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

    # Sound rides along automatically: if the scene has cues (auto_sfx or
    # add_sound), video formats get the mixed track muxed in.
    if format in ("mp4", "webm") and s.has_audio:
        from ..audio.mux import mux_audio

        mix, report = s.audio_mix()
        mux_audio(path, mix, path)
        return (f"wrote {path} ({path.stat().st_size:,} bytes) with audio\n"
                f"{report}")
    return f"wrote {path} ({path.stat().st_size:,} bytes)"


@mcp.tool()
@qt_tool
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
@qt_tool
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
    if s.has_audio:
        # GIFs are mute; give the human an MP4 with the soundtrack too.
        from glaxnimate import io as gio

        from ..audio.mux import mux_audio

        mp4 = path.with_suffix(".mp4")
        fmt = gio.registry.from_extension("mp4", gio.Direction.Export)
        mp4.write_bytes(fmt.save(s.scene.comp))
        mix, _ = s.audio_mix()
        mux_audio(mp4, mix, mp4)
        return (f"wrote {path} and {mp4} (with sound) - "
                f"ask the user to watch/listen and say what is wrong")
    return f"wrote {path} - ask the user to watch it and say what is wrong"


_API = """\
CARTOON LIBRARY
===============
Screen coords: +x right, +y DOWN. "Up" is negative y. Ground is a y value.

BODIES (rigs). A human is one preset among many; the engine animates anything.
  human() / biped(thigh=, shin=, spine=, arm=, forearm=, head=)  -> Body
  quadruped(upper=, lower=, body=, neck=, head=, tail=)          -> Body  (dog/cat/horse)
  stick(ink=, weight=, head_d=)                                  -> Body  (line-art figure)
  body.hip_height, body.leg_length

LINE-ART (the stick-figure look -- STYLE, not species):
  stick() is biped() drawn as uniform pen strokes with a ring head.
  lineart(body) reskins ANY body the same way -- lineart(quadruped()) is a stick dog.
  Or add_character(body, gait, style="lineart") to stroke it at bake time.
  The look lives in the body's parts, so it saves and replays like any other skin.

THE ASSET LIBRARY (data, not code -- this is how the vocabulary GROWS):
  load_body("bird") -> Body            a creature saved as JSON
  save_body(body, "name")              persist one (validated first)
  body_from_data({...}) -> Body        build straight from a dict
  load_gait("scuttle") / register_gait({...})   custom gaits by name
  add_prop("bench", x=200) / load_prop(name)    data props on the ground line
  New creature? Author body JSON (see save_asset tool for the schema), save it,
  load it by name. It gets the same linter and reach guard as the builtins.

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

FACES (swappable expressions on a slot; stepped, like cut-out animation)
  add_character(..., face="stick")     mount a face asset (stick, human, dog, yours)
  set_expression(char_or_name, "happy", frame)   hold-swap at that frame
  stick face: neutral, happy, sad, surprised, blink, angry, determined (+ say_*).
  human face: neutral, happy, sad, surprised, blink.  dog: normal, happy.
  New faces are face.json assets: attachments of prop-schema shapes, authored
  screen-aligned around the slot point (x = facing, y = down).

VOICE-OVER (local neural TTS, cached in the project; the model can't hear, so
placement and lip-sync are arithmetic, not guesswork)
  say(char, "line", frame)   speaks + a speech bubble; auto lip-sync flaps the
     mouth from the audio's own RMS envelope when the face has say_* mouths (the
     stick face does). say(..., lipsync=False) to hold the mouth still.

SCENERY (backdrops, from scripts)
  scenery("sky") / scenery("ground") / scenery("house", x=520)
  scenery("school", x=40) / scenery("tree", x=300, h=120)
  scenery("cloud", x=140, y=70) / scenery("sun", x=880, y=66)
  Draw back-to-front: sky first, ground last before characters.

STAGE
  add_character(body, gait, x=80, name="...", color=None, thickness=None, face=None)
     color/thickness default to None = use the body's own skin (a person looks
     like a person). Pass them only to flatten to one colour.
  add_object(samples, shape="Ellipse", size=Vec2(w,h), color="#e8543f")
  add_chaser(body, gait_name, target, x=60, gap=40, cycle_frames=16, name="...")
     A character PACED to chase `target` (a motion.* result) and end `gap` px
     behind it. Solves "the chaser lost the race" in one call — no per-frame
     metric catches that, because each character is individually fine; it is the
     relationship that is wrong.

ACTIONS (things a character DOES; not locomotion). Each returns a pose function
you hand to add_action(body, pose_fn, name="..."):
  actions.jump(body, ground_y=, x=, height=, distance=0, frames=36)
     anticipation (crouch) -> launch -> arc -> squash landing. Three principles.
  actions.idle(body, ground_y=, x=, cycle_frames=48)   # breathing; a living hold
  actions.wave(body, ground_y=, x=, frames=48)          # raise arm and wave
  actions.trail(pose_fn, body, chain=[joints], lag=2.5, swing=26)
     FOLLOW-THROUGH: wrap any pose_fn so a loose chain (tail, cape, ear, hair)
     lags behind the motion and settles when the body stops. chain is ordered
     base->tip. Silent when the character is still, whips when it darts.
  actions.sequence((action1, frames1), (action2, frames2), ...)  # beats in a row

EVERYDAY acting verbs (the non-combat ones a scene needs constantly):
  actions.celebrate(body, ground_y=, x=, pumps=2)   # arms up, pumping — joy/victory
  actions.fall(body, ground_y=, x=, facing=1)       # a pratfall, lands sitting
  actions.sit(body, ground_y=, x=, seat=)           # sit down (desk/bench); + tap = typing
  actions.tap(body, ground_y=, x=, hits=4)          # hands strike down — drum/type/keyboard

COMBAT / STUNT beats (a stick figure that fights, not just walks). facing=+1 faces
right, -1 left. Each is anticipation -> fast strike -> settle; the blow lands ~60%
through, so put an fx/sfx cue there. String them with actions.sequence:
  actions.punch(body, ground_y=, x=, facing=1, frames=16)    # straight jab
  actions.kick(body, ground_y=, x=, facing=1, frames=18)     # front kick, foot snaps out
  actions.dash(body, ground_y=, x0=, x1=, frames=14)         # explosive lunge across
  actions.flip(body, ground_y=, x=, distance=90, height=150, facing=1, frames=26)
  actions.swing(body, ground_y=, x=, facing=1, frames=18)    # overhead sword/club arc
  actions.block(body, ground_y=, x=, facing=1, frames=12)    # guard up (caught, not hit)
  actions.knockback(body, ground_y=, x=, facing=1, distance=70, frames=16)  # take a hit
  actions.land(body, ground_y=, x=, frames=14)               # hero crouch landing
  actions.line_of_action(pose_fn, body, curve=1.0)  # exaggerate the pose's read

SNAP TIMING (snappy beats weigh more than floaty ones; the viral tell):
  actions.hitstop(pose_fn, at=<contact frame>, freeze=3)  # freeze on impact for
     weight; adds `freeze` frames -> bake/add_action over original length + freeze.
  actions.hold(pose_fn, at, frames)      # a moving hold: stop dead, then resume
  actions.retime(pose_fn, span, ease)    # re-time a beat: ease=principles.ease_in
     (snappy, sits then bursts) or ease_out (floaty). Same length, new spacing.
  principles.hold_snap(t, hold=0.4)      # a "sit still then snap" easing curve

EFFECTS (the visual juice: dust, impact flashes, speed lines, sparks)
  auto_fx()               spawn effects FROM motion (twin of auto_sfx): foot
     plants->dust, ground hits->impact, jump launch->speed_lines, land->dust.
     Same events as the foley pass; run after the motion lints clean.
  add_effect(fx, x, y, frame)   place one by hand. builtins: impact, dust,
     speed_lines, spark. Put an impact on a strike's contact frame (~60% through)
     with a hit sfx there and hitstop on the beat -> the whole hit lands on one frame.
  New effects are fx assets (save_asset(\"fx\",...)): prop-schema shapes + an
  envelope {lifespan, grow:[from,to], fade, spin}. Grows via real transform.scale.

FLOATING SHAPES (a prop that is neither scenery nor held: heart, sign, chart, hat)
  add_shape(shapes, x, y, pulse=(lo,hi,cycles), spin=, appear=)  # place at (x,y),
     not ground-pinned. pulse beats the scale (a heart); appear pops it in (reveal).
     shapes is an inline [{...}] prop-schema list or a saved prop name.

PARTICLES (showers of many small bits: confetti, sparks, rain, snow, smoke)
  emit(fx, x, y, count=, spread=, start=, over=)          # burst in place (drop=0):
     confetti/sparkle/smoke. fx is an effect name like \"spark\".
  emit(None, x, y, count=, spread=, drop=300, color=)     # FALLING (drop>0): rain,
     snow, falling confetti across a band `spread` wide. Deterministic per seed.

CAMERA (call LAST — after the scene is built and after any shot() gating; it
re-parents all content under one camera layer whose transform is the camera):
  impact_camera(frame, x, y, zoom=1.2, shake=9)  # SELL THE HIT: fast punch-in
     toward (x,y) + a decaying shake on the contact frame. Fire it on the SAME
     frame as the impact flash + hit sfx + the beat's hitstop -> one big moment.
  camera_move(start, end, zoom=1, focus_x, focus_y)  # held push-in or pan for
     staging. zoom>1 pushes in; zoom=1 with a focus pans. Returns to neutral.

TWO FIGURES & PROPS-AS-TOYS (fights, hand-offs, using the world):
  A two-figure hit is composed from beats you already have:
    add_action(a, actions.hitstop(actions.punch(a, ground_y=ground, x=150), at=10), name=\"a\")
    add_action(b, actions.sequence((actions.idle(b, ground_y=ground, x=250), 10),
                                   (actions.knockback(b, ground_y=ground, x=250,
                                                      facing=-1), 14)), name=\"b\")
    clash(10, x=250, y=120)   # impact flash + hit sfx + camera punch-in, one call
  wield(char, prop, bone=\"arm_lower\")   # a prop rides the hand: a sword on a swing,
     a torch on a walk. prop is a name or an inline {shapes:[...]} dict.
  throw(prop, x0=, y0=, x1=, y1=, apex=120, release=, spin=360)  # let it go: a
     ballistic arc, invisible until the release frame. The AvA move: use the world.

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

AUDIO (sound is data on the same doc; exports mux it into mp4/webm)
  auto_sfx()                         the foley pass: footsteps, hits, whooshes,
                                     landings, expression pops - derived from
                                     the motion, on the right frames, panned to
                                     where they happen. Run after motion is final.
  auto_sfx({"plant": None})          silence a kind; or remap {"hit": "splat"}
  add_sound("boing", frame, gain=1, pan=0)   one manual cue
     builtins: boing thud step pop whoosh slide_up slide_down splat ding
     new sounds = sfx assets: JSON synth patches (see save_asset), cued by name
  music(seed=7, bpm=104, gain=0.2)   seeded chiptune bed; bad? change the seed.

BEAT SYNC & FORMAT (make it land on the beat and fit the platform):
  beats(division=1)          the frames the music beat lands on (2=eighths, 4=16ths).
  snap_to_beat(frame)        round a frame onto the grid — put a cut/clash on a beat.
     A montage feels deliberate when cuts and hits sit on beats(): after music(),
     place clash()/shot()/add_effect on a frame from beats() or snap_to_beat().
  new_document(preset=\"portrait\")  9:16 for TikTok/Shorts (also square, sticker).
  loop_report()   for a looping short/sticker: is frame 0 == the last frame? Make
     `frames` a whole motion cycle so a walk loops seamlessly.
     music(seed=None) removes it. Keep gain low - it is a bed, not the show.
  say("man", "Hello!", frame)        neural TTS + speech bubble for the line's
     duration. Cached to the project; replays without the TTS installed.
  You cannot hear - sound_report is your ears-as-numbers (cue sheet, peak dBFS,
  pile-ups), and preview_for_human writes an MP4 with the soundtrack.

IN SCOPE WITHOUT IMPORT: ground, frames, width, height, scene, Vec2

WORKFLOW - obey this order, it saves you tokens and time:
  1. run_script       build it
  2. lint_animation   is it broken?  FREE. always.
  3. diagnose_animation  is it good? ~500 tokens, names the frame.
  4. render_contact_sheet  only for what numbers cannot judge. ~1,400 tokens.
  5. auto_sfx + sound_report  foley from the motion, checked as numbers.
  6. preview_for_human  hand the human the result; with audio you get an MP4
     with sound - their one sentence beats everything.
"""


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
