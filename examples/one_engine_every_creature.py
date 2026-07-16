"""*One engine, every creature* — the capability reel. ~30 seconds, square.

Run:  .venv/bin/python examples/one_engine_every_creature.py

Four bodies walk the same code. A two-legged human, a four-legged dog whose
forelegs bend the opposite way to its hind legs, a bird with a beak and stubby
backwards-looking legs, and an eight-legged tardigrade doing a *metachronal
wave* — the ripple real millipedes and water bears actually walk with.

The engine does not know what any of them are. A rig is a graph of joints; a
gait is a table of phase offsets. Eight-limb coordination, the thing that looks
hardest here, is eight numbers in a JSON file:

    "phases": {"8": [0.75, 0.25, 0.5, 0.0, 0.25, 0.75, 0.0, 0.5]}

The last shot is the one animators care about. Ghost copies of the walk are
drawn at earlier moments in time, and every planted foot lands on the same spot
in every ghost while the body moves on over it. That is not tuned: legs are
solved with inverse kinematics from a fixed foot target, so a planted foot is
world-stationary *by construction* and the linter merely confirms it.

The film is silent on purpose. It is built for a feed, where video autoplays
muted, and the bindings expose no text shape — so the claim lives in the post
copy and the film is only the evidence.
"""

from __future__ import annotations

from pathlib import Path

from glaxnimate import io as gio

from glaxnimate_ai.engine.session import SessionStore

W = H = 900
FPS = 24
GROUND = 720.0

# --- the cut list. One document, one timeline; a shot is gated visibility.
S1 = (0, 120)       # 5s   human      2 legs
S2 = (120, 240)     # 5s   dog        4 legs, elbow != knee
S3 = (240, 360)     # 5s   bird       2 legs, odd proportions
S4 = (360, 510)     # 6s   tardigrade 8 legs, metachronal wave
S5 = (510, 744)     # 10s  the feet: onion skin, zero slip
FRAMES = S5[1]

OUT = Path("out")

SCRIPT = f"""
import copy
load_gait('trundle')

def scaled(body, k):
    '''Resize a body by scaling its data, not the render.

    Four real creatures differ wildly in size -- a bird's legs are 52px, a
    human's 160 -- and a reel needs each to FILL its shot. Bodies are documents,
    so resizing one is arithmetic on the document: lengths, offsets, skin widths
    and slots all scale together, and the rig it builds is a first-class body
    with the same reach guard and the same linter.
    '''
    d = copy.deepcopy(assets.body_to_data(body) if not isinstance(body, dict) else body)
    for j in d['joints']:
        j['length'] = j.get('length', 0.0) * k
        if j.get('offset'):
            j['offset'] = [j['offset'][0] * k, j['offset'][1] * k]
    for part in (d.get('parts') or {{}}).values():
        part['width'] = part.get('width', 14.0) * k
        if part.get('head'):
            part['head'] = [part['head'][0] * k, part['head'][1] * k]
        if part.get('tip'):
            part['tip'] = part['tip'] * k
    for sl in (d.get('slots') or {{}}).values():
        sl['offset'] = [sl['offset'][0] * k, sl['offset'][1] * k]
    if d.get('leg_length'):
        d['leg_length'] = d['leg_length'] * k
    return body_from_data(d)

# Each creature scaled to roughly fill the frame. They are wildly different
# animals; the reel is about the ENGINE, so the framing is normalised, not them.
MAN  = scaled(human(), 1.9)
DOG  = scaled(quadruped(), 1.9)
BIRD = scaled(load_body('bird'), 3.3)
TARD = scaled(load_body('tardigrade'), 1.5)

def crossing(body, gait_name, *, cycle, dist, t0=0, x0=-150, span=120):
    '''A creature that crosses the frame during its own shot.

    `pace` sizes the gait so it travels exactly x1-x0 in the shot's length --
    speed is a SHORTER cycle, never a longer stride, which is why nothing here
    has to be nudged by hand. The pose function is then shifted by t0 so the
    creature enters on its shot's first frame rather than the film's.
    '''
    # `span` is the SHOT's length, not the film's: pace it to cross in the
    # time it is actually on screen, or it exits early and leaves dead frames.
    g = pace(body, gait_name, distance=dist, frames=span, cycle_frames=cycle)
    def pf(t, _b=body, _g=g, _x0=x0, _t0=t0):
        return pose_at(_b.rig, _g, t - _t0, ground_y={GROUND}, body_x0=_x0)
    return pf

def ground_line(name, color='#c8d2de'):
    add_prop({{'version': 1, 'kind': 'prop', 'shapes': [
        {{'type': 'rect', 'x': -480, 'y': 0, 'w': 960, 'h': 7, 'color': color}}]}},
        x=450, y={GROUND}, layer_name=name)

scenery('sky', top='#eaf1f8')

# ---- SHOT 1: a human. The expected thing, so the rest has a baseline.
ground_line('s1.ground')
add_action(MAN, crossing(MAN, 'walk', cycle=26, dist=1240, x0=-290,
                         t0={S1[0]}), name='s1.man', first={S1[0]})
print(shot('s1', {S1[0]}, {S1[1]}))

# ---- SHOT 2: a dog. Forelegs bend the opposite way to hind legs -- an elbow,
# not a knee. That one flag is most of what separates a dog from a table.
ground_line('s2.ground')
add_action(DOG, crossing(DOG, 'trot', cycle=16, dist=1330, x0=-380,
                         t0={S2[0]}), name='s2.dog', first={S2[0]})
print(shot('s2', {S2[0]}, {S2[1]}))

# ---- SHOT 3: a bird. Authored as raw JSON, derived from no template.
ground_line('s3.ground')
add_action(BIRD, crossing(BIRD, 'walk', cycle=14, dist=1400, x0=-300,
                          t0={S3[0]}), name='s3.bird', first={S3[0]})
print(shot('s3', {S3[0]}, {S3[1]}))

# ---- SHOT 4: eight legs. The gait is a phase table; the wave is arithmetic.
ground_line('s4.ground')
add_action(TARD, crossing(TARD, 'trundle', cycle=24, dist=700, x0=-430,
                          t0={S4[0]}), name='s4.tardi', first={S4[0]})
print(shot('s4', {S4[0]}, {S4[1]}))

# ---- SHOT 5: the feet. The claim animators actually care about.
# A mark is stamped on the ground the instant a foot plants, and the foot then
# stays welded to that mark while the body walks on over it. Nothing is tuned to
# make that happen: legs are solved with IK from a FIXED foot target, so a
# planted foot is world-stationary by construction. The plants are found the
# same way the linter finds them -- solve the rig, ask which contact tips are
# on the ground line -- so the marks cannot disagree with what is drawn.
ground_line('s5.ground', color='#9fb0c4')
foot_pf = crossing(MAN, 'walk', cycle=30, dist=1090, x0=-180,
                   t0={S5[0]}, span={S5[1]} - {S5[0]})
add_action(MAN, foot_pf, name='s5.man', first={S5[0]})

TICK = {{'version': 1, 'kind': 'prop', 'shapes': [
    {{'type': 'rect', 'x': -4, 'y': -34, 'w': 8, 'h': 34, 'color': '#e8543f'}},
    {{'type': 'ellipse', 'cx': 0, 'cy': -40, 'w': 15, 'h': 15, 'color': '#e8543f'}}]}}

was_down = {{}}
stamped = 0
for f in range({S5[0]}, {S5[1]} + 1):
    fr = MAN.rig.solve(foot_pf(float(f)))
    for jname in ('shin_l', 'shin_r'):
        tip = fr[jname].tip
        down = abs(tip.y - {GROUND}) <= 2.0
        if down and not was_down.get(jname, True):
            # deliberately NOT under 's5.': the beat-wide shot('s5', ...)
            # matches by prefix and would clobber each mark's own gate.
            nm = f'mark{{stamped}}'
            add_prop(TICK, x=tip.x, y={GROUND}, layer_name=nm)
            # the mark appears as the foot lands and stays for the rest of the shot
            shot(nm, f, {S5[1]})
            stamped += 1
        was_down[jname] = down
print(f'stamped {{stamped}} foot plants')
print(shot('s5', {S5[0]}, {S5[1]}))
"""


def main() -> None:
    s = SessionStore().create(width=W, height=H, frames=FRAMES, fps=FPS,
                              ground_y=GROUND)
    # 20s is the MCP guardrail against runaway loops, not a budget for real
    # work: nine rigs (four creatures + five onion-skin ghosts) over 744
    # frames is a lot of honest IK.
    res = s.run(SCRIPT, timeout=180.0)
    print(res.format())
    if not res.ok:
        raise SystemExit("script failed")

    print("\n--- tier 0: is any of it broken? (free, no image) ---")
    from glaxnimate_ai.feedback.lint import lint_rig
    for ch in s.characters:
        rep = lint_rig(ch.body, ch.pose_fn, frames=FRAMES, ground_y=s.ground_y,
                       limbs=ch.limb_pairs or None, canvas=(W, H))
        print(f"  {ch.name:12s} {rep.format()}")

    OUT.mkdir(exist_ok=True)
    mp4 = OUT / "one_engine_every_creature.mp4"
    fmt = gio.registry.from_extension("mp4", gio.Direction.Export)
    mp4.write_bytes(fmt.save(s.scene.comp))
    print(f"\nwrote {mp4} ({mp4.stat().st_size:,} bytes) | "
          f"{W}x{H} | {FRAMES / FPS:.0f}s | silent by design")


if __name__ == "__main__":
    main()
