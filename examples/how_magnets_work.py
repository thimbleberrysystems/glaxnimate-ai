"""*How do magnets work?* — a narrated science explainer, ~69 seconds.

Run:  .venv/bin/python examples/how_magnets_work.py

This is the project's second end-to-end goal, and it stresses different muscles
than `walk_home_from_school.py`. There is no walk cycle in it. What it needs
instead is *physics that reads*, six shots on one timeline, and an explanation
carried entirely by the voice — because the bindings expose no text shape, so
this film cannot draw a single label. That constraint turns out to be a gift: a
science film where the eye watches the physics and the ear takes the words is
better than one plastered with captions.

Two engine pieces exist because of this film, and neither is about magnets:

* `motion.attract` — inverse-power attraction. The intuitive choice was
  `spring()`, which is elastic: it lunges when far away, overshoots and wobbles.
  A magnet does the opposite, and that late snatch is the entire effect.
* `shot()` — six beats, one document, no camera. A shot is just visibility.

Everything else is data: six JSON props, nine synth patches, a seeded music bed,
and dialogue rendered by a local neural voice and cached in the project.
"""

from __future__ import annotations

from pathlib import Path

from glaxnimate import io as gio

from glaxnimate_ai.audio.mux import audio_duration, has_audio_stream, mux_audio
from glaxnimate_ai.engine.session import SessionStore

W, H, FPS = 960, 540, 24
OUT = Path("out")

# ---------------------------------------------------------------- the cut list
# One document, one continuous timeline; a "shot" is a gated set of layers.
B1 = (0, 216)        # 9s   hook: two magnets snap together
B2 = (216, 384)      # 7s   flip one over: they fight back
B3 = (384, 864)      # 20s  inside the iron: domains align
B4 = (864, 1152)     # 12s  filings make the field visible
B5 = (1152, 1392)    # 10s  cut it in half: still two poles
B6 = (1392, 1704)    # 13s  Earth is a giant magnet
FRAMES = B6[1]

TABLE_Y = 470.0      # the session's ground line (beats 1-2 draw a table above it)
TABLE_TOP = 380.0    # where the tabletop actually sits, so the action is centred
MAG_S = 2.0          # bar magnets read at 240x72 -- big enough to see the poles
MAG_W = 120.0 * MAG_S    # two touching magnets stop this far apart, centre to centre
N_RED, S_BLUE, STEEL, DARK = "#d94f3d", "#3d6fd9", "#b9c0cc", "#3a3f4a"


def _tabletop(x: float, w: float, color: str = "#6d4f38") -> dict:
    return {"version": 1, "kind": "prop", "shapes": [
        {"type": "rect", "x": -w / 2, "y": -14, "w": w, "h": 14, "color": "#8a6547"},
        {"type": "rect", "x": -w / 2, "y": 0, "w": w, "h": 90, "color": color},
    ]}


def _slab(w: float, h: float, color: str) -> dict:
    """A plain rectangle centred on its own origin."""
    return {"version": 1, "kind": "prop", "shapes": [
        {"type": "rect", "x": -w / 2, "y": -h / 2, "w": w, "h": h, "color": color}]}


SCRIPT = f"""
import math, random
from glaxnimate_ai.cartoon.principles import ease_in_out, ease_out

random.seed(7)          # the film must render identically every time
scenery('sky', top='#dfeaf5')

N_RED, S_BLUE, STEEL, DARK = '{N_RED}', '{S_BLUE}', '{STEEL}', '{DARK}'
CY = {TABLE_TOP} - 14 - 36                   # a magnet's centre, resting on the table

def hold(samples, until):
    '''Extend a clip by freezing its last pose — the beat holds after the action.'''
    last = samples[-1]
    return samples + [motion.Sample(f, last.pos, last.scale, last.angle)
                      for f in range(last.frame + 1, until + 1)]

def still(pos, f0, f1, angle=0.0):
    return [motion.Sample(f, pos, Vec2(1, 1), angle) for f in range(f0, f1 + 1)]

def field_angle(px, py, mx, my):
    '''A 2D dipole's direction at a point — the formula the filings obey.

    B = (3(m.r^)r^ - m)/r^3 for a moment m along +x. Only the DIRECTION is
    wanted, so every constant out front drops away.
    '''
    dx, dy = px - mx, py - my
    r2 = dx * dx + dy * dy
    if r2 < 1.0:
        return 0.0
    r = math.sqrt(r2)
    ux, uy = dx / r, dy / r
    bx = 3.0 * ux * ux - 1.0
    by = 3.0 * ux * uy
    return math.degrees(math.atan2(by, bx))

# ============================================================ BEAT 1: the hook
add_prop({_tabletop(0, 960)!r}, x=480, y={TABLE_TOP}, layer_name='b1.table')
add_moving_prop('bar_magnet', still(Vec2(700, CY), {B1[0]}, {B1[1]}),
                name='b1.fixed', scale={MAG_S}, radius=30)

# contact_gap = one magnet's width: their faces meet instead of overlapping.
snap = motion.attract(start=Vec2(240, CY), end=Vec2(700, CY), frames=160,
                      power=3.0, contact_gap={MAG_W}, ring=9.0)
add_moving_prop('bar_magnet', hold(snap, {B1[1]}), name='b1.free',
                scale={MAG_S}, radius=30)
add_sound('thud', 160, gain=0.95)
say(None, "Watch what happens when these two magnets get close.", 10)
say(None, "They pull themselves together.", 172)
print(shot('b1', {B1[0]}, {B1[1]}))

# ========================================================== BEAT 2: they repel
add_prop({_tabletop(0, 960)!r}, x=480, y={TABLE_TOP}, layer_name='b2.table')
add_moving_prop('bar_magnet', still(Vec2(700, CY), {B2[0]}, {B2[1]}),
                name='b2.fixed', scale={MAG_S}, radius=30)

# Flip the free magnet end for end, then shove it in. Now like poles face each
# other. Repulsion is attraction run backwards in time: from rest at the closest
# point the force flings it away hardest at first and eases as the gap opens --
# the attract curve, reversed. One primitive, both signs. It must be built from
# the FAR point inward and then reversed, or the magnet flies at its twin
# instead of away from it.
flip = []
for f in range({B2[0]}, 265):
    t = ease_in_out(min(max((f - 228) / 34.0, 0.0), 1.0))
    flip.append(motion.Sample(f, Vec2(240, CY - 62 * math.sin(t * math.pi)),
                              Vec2(1, 1), 180.0 * t))
push = [motion.Sample(f, Vec2(240 + (f - 265) * 6.4, CY), Vec2(1, 1), 180.0)
        for f in range(265, 301)]
inward = motion.attract(start=Vec2(150, CY), end=Vec2(470, CY), frames=83, power=3.0)
repel = [motion.Sample(301 + i, s.pos, Vec2(1, 1), 180.0)
         for i, s in enumerate(reversed(inward))]
add_moving_prop('bar_magnet', hold(flip + push + repel, {B2[1]}),
                name='b2.free', scale={MAG_S}, radius=30)
add_sound('whoosh', 302, gain=0.7)
say(None, "But turn one of them around, and they shove each other away.", 224)
print(shot('b2', {B2[0]}, {B2[1]}))

# ================================================= BEAT 3: inside the iron bar
IRON_X, IRON_Y, IRON_W, IRON_H = 470, 300, 420, 210
add_prop({_slab(520, 210, STEEL)!r}, x=IRON_X, y=IRON_Y, layer_name='b3.iron')

# The magnet that comes to visit, and the domains it orders.
mag3 = (still(Vec2(120, IRON_Y), {B3[0]}, 560)
        + [motion.Sample(f, Vec2(120 + ease_in_out(min((f - 560) / 90.0, 1.0)) * 120,
                                 IRON_Y)) for f in range(561, 651)])
add_moving_prop('bar_magnet', hold(mag3, {B3[1]}), name='b3.magnet', scale=1.1, radius=22)

for row in range(3):
    for col in range(8):
        ax = IRON_X - IRON_W / 2 + 40 + col * 48
        ay = IRON_Y - IRON_H / 2 + 45 + row * 60
        a0 = random.uniform(0, 360)
        # Domains nearest the magnet turn first, so the order visibly SWEEPS
        # through the metal rather than snapping on all at once.
        t0 = 610 + (ax - (IRON_X - IRON_W / 2)) / IRON_W * 110
        smp = []
        for f in range({B3[0]}, {B3[1]} + 1):
            t = ease_in_out(min(max((f - t0) / 80.0, 0.0), 1.0))
            wobble = 2.5 * math.sin(f * 0.22 + col) * (1.0 - t)   # jitter, then still
            smp.append(motion.Sample(f, Vec2(ax, ay), Vec2(1, 1),
                                     a0 * (1.0 - t) + wobble))
        add_moving_prop('domain_arrow', smp, name=f'b3.dom{{row}}_{{col}}', radius=6)

# The payoff: the iron is a magnet now, so it can pick up a paperclip.
clip3 = motion.attract(start=Vec2(945, IRON_Y + 52),
                       end=Vec2(IRON_X + IRON_W / 2, IRON_Y + 52),
                       frames=90, power=3.0, contact_gap=42.0)
clip3 = [motion.Sample(760 + i, s.pos) for i, s in enumerate(clip3)]
add_moving_prop('paperclip', hold(clip3, {B3[1]}), name='b3.clip',
                scale=3.0, radius=16)
add_sound('pop', 850, gain=0.6)
say(None, "So why does iron stick to a magnet at all? Look inside the metal.", 392)
say(None, "It is already full of tiny magnets, all facing different ways, cancelling out.", 500)
say(None, "Bring a magnet close, and they swing into line. Now the iron is a magnet too.", 640)
print(shot('b3', {B3[0]}, {B3[1]}))

# ================================================== BEAT 4: the field revealed
PAPER_Y = 300
add_prop({_slab(880, 380, '#fbfbf8')!r}, x=480, y=PAPER_Y, layer_name='b4.paper')
MX, MY = 480.0, float(PAPER_Y)
add_moving_prop('bar_magnet', still(Vec2(MX, MY), {B4[0]}, {B4[1]}),
                name='b4.magnet', scale=1.15, radius=22)

# Filings sprinkle down, tumbling, then each swings to the local field direction.
for i in range(78):
    fx = random.uniform(80, 880)
    fy = random.uniform(130, 470)
    if abs(fx - MX) < 76 and abs(fy - MY) < 26:
        continue                       # not on top of the magnet itself
    drop_at = {B4[0]} + 6 + int(random.uniform(0, 60))
    tumble = random.uniform(0, 360)
    target = field_angle(fx, fy, MX, MY)
    smp = []
    for f in range({B4[0]}, {B4[1]} + 1):
        fall = min(max((f - drop_at) / 30.0, 0.0), 1.0)
        y = fy - 220 * (1.0 - ease_out(fall))
        align = ease_in_out(min(max((f - (drop_at + 34)) / 46.0, 0.0), 1.0))
        ang = tumble + (f - drop_at) * 6.0 * (1.0 - align)
        smp.append(motion.Sample(f, Vec2(fx, y), Vec2(1, 1),
                                 ang * (1.0 - align) + target * align))
    add_moving_prop('filing', smp, name=f'b4.fil{{i}}', radius=4)

say(None, "You cannot see a magnetic field. But iron filings can.", 872)
say(None, "Every speck turns to point along the field, and the shape appears: "
          "out of one end, around, and into the other.", 980)
print(shot('b4', {B4[0]}, {B4[1]}))

# ================================================== BEAT 5: cut it in half
CUT_AT = 1248
# The whole bar is 288x86; each half must be 144x86 -- half the WIDTH only. A
# uniform scale would halve the height too and the halves would read as
# different objects. Sample.scale cannot do it: transform.scale is unwritable
# through these bindings (see docs/glaxnimate-api.md). The per-axis DRAW scale
# can, because it multiplies coordinates before Qt ever sees them.
add_moving_prop('bar_magnet', still(Vec2(480, 300), {B5[0]}, CUT_AT),
                name='b5.whole', scale=(2.4, 2.4), radius=36)
add_prop({_slab(5, 200, DARK)!r}, x=480, y=300, layer_name='b5.blade')

# After the cut, two independent magnets, each with a full red N and blue S.
# The new poles appear AT the cut face, which is the entire point.
for side, x0, x1 in (('l', 408, 352), ('r', 552, 608)):
    smp = [motion.Sample(f, Vec2(x0 + (x1 - x0) * ease_out(min((f - CUT_AT) / 70.0, 1.0)),
                                 300))
           for f in range(CUT_AT, {B5[1]} + 1)]
    add_moving_prop('bar_magnet', smp, name=f'b5.half.{{side}}',
                    scale=(1.2, 2.4), radius=36)

add_sound('splat', CUT_AT, gain=0.8)
say(None, "Cut a magnet in half to separate its poles.", 1160)
say(None, "You just get two magnets. Nobody has ever found a lone pole.", 1256)
# Three sub-shots, not one: the whole bar and its halves must never share a
# frame, or the cut reads as a smear of overlapping bars.
print(shot('b5.whole', {B5[0]}, CUT_AT))
print(shot('b5.blade', 1216, 1262))
print(shot('b5.half', CUT_AT, {B5[1]}))

# ================================================== BEAT 6: Earth is a magnet
EX, EY = 330.0, 300.0
add_prop({{'version': 1, 'kind': 'prop', 'shapes': [
    {{'type': 'ellipse', 'cx': 0, 'cy': 0, 'w': 300, 'h': 300, 'color': '#5b8f6a'}},
    {{'type': 'ellipse', 'cx': 0, 'cy': 0, 'w': 276, 'h': 276, 'color': '#6fa87c'}}]}},
    x=EX, y=EY, layer_name='b6.earth')
# the bar magnet buried inside the planet, tilted like the real field
add_moving_prop('bar_magnet', still(Vec2(EX, EY), {B6[0]}, {B6[1]}, angle=-101.0),
                name='b6.core', scale=1.5, radius=22)

add_prop({{'version': 1, 'kind': 'prop', 'shapes': [
    {{'type': 'ellipse', 'cx': 0, 'cy': 0, 'w': 190, 'h': 190, 'color': '#cfd6e2'}},
    {{'type': 'ellipse', 'cx': 0, 'cy': 0, 'w': 168, 'h': 168, 'color': '#f7f9fc'}}]}},
    x=760, y=300, layer_name='b6.dial')

# A compass needle hunts north and settles: a decaying oscillation, which is
# what any damped pointer does. sway() rocks forever, so this decays it.
needle = []
for f in range({B6[0]}, {B6[1]} + 1):
    t = (f - {B6[0]}) / 60.0
    swing = 60.0 * math.exp(-0.30 * t) * math.cos(2.2 * t)
    needle.append(motion.Sample(f, Vec2(760, 300), Vec2(1, 1), -95.0 + swing))
add_moving_prop('compass_needle', needle, name='b6.needle', scale=1.6, radius=8)
add_sound('ding', 1500, gain=0.5)

say(None, "And the biggest magnet you will ever stand on is the Earth itself.", 1400)
say(None, "Its molten core turns the whole planet into a bar magnet — which is "
          "exactly what a compass needle has been finding all along.", 1500)
print(shot('b6', {B6[0]}, {B6[1]}))

music(seed=23, bpm=96, gain=0.13)
"""


def main() -> None:
    s = SessionStore().create(width=W, height=H, frames=FRAMES, fps=FPS,
                              ground_y=TABLE_Y)
    res = s.run(SCRIPT)
    print(res.format())
    if not res.ok:
        raise SystemExit("script failed")

    print("\n--- sound (tier 0: the mix as numbers, no listening) ---")
    mix, report = s.audio_mix()
    print(report)

    OUT.mkdir(exist_ok=True)
    mp4 = OUT / "how_magnets_work.mp4"
    fmt = gio.registry.from_extension("mp4", gio.Direction.Export)
    mp4.write_bytes(fmt.save(s.scene.comp))
    mux_audio(mp4, mix, mp4)
    print(f"\nwrote {mp4} ({mp4.stat().st_size:,} bytes) | audio stream: "
          f"{has_audio_stream(mp4)} ({audio_duration(mp4):.1f}s) | "
          f"{FRAMES / FPS:.0f}s of film")


if __name__ == "__main__":
    main()
