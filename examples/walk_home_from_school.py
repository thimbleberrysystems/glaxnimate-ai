"""The project's goal, end to end: *a man walks home from school*.

Run:  PYTHONPATH=src .venv/bin/python examples/walk_home_from_school.py

This is what the LLM is meant to produce from that one sentence. It is written
the way a script from `run_script` would be, and it goes through the same loop:
build, lint, diagnose, render, export.

Note what the script does *not* do: it never draws the man. It cannot — nobody can
place bezier points for a human figure blind. It picks a body and a gait, and the
library does the drawing. The scenery, being geometric, it does build directly.
That division is the whole idea.
"""

from __future__ import annotations

from pathlib import Path

from glaxnimate import io as gio

from glaxnimate_ai.engine import props
from glaxnimate_ai.engine.session import SessionStore
from glaxnimate_ai.feedback import render as R
from glaxnimate_ai.feedback.diagnose import diagnose_rig
from glaxnimate_ai.feedback.lint import lint_rig

W, H, FRAMES, FPS = 960, 460, 96, 24
OUT = Path("out")


def main() -> None:
    store = SessionStore()
    s = store.create(width=W, height=H, frames=FRAMES, fps=FPS)
    ground_y = s.ground_y
    comp = s.scene

    # The world is built back to front, so nearer things paint over farther ones.
    sky_l = comp.layer("sky")
    props.sky(comp, sky_l, top="#bfe3f5")
    props.sun(sky_l, 830, 70)

    far = comp.layer("far")
    for x in (120, 380, 700):
        props.cloud(far, x, 70 + (x % 40))
    # Clouds sit almost still: they are miles away.
    props.parallax(far, distance=0.12, frames=FRAMES, camera_speed=4.6)

    mid = comp.layer("buildings")
    props.school(mid, 40, ground_y)          # he starts here...
    props.house(mid, 690, ground_y)          # ...and ends here
    for x in (370, 520, 640):
        props.tree(mid, x, ground_y, h=110)
    props.parallax(mid, distance=0.55, frames=FRAMES, camera_speed=4.6)

    near = comp.layer("ground")
    props.ground(comp, near, ground_y)

    # The man. Never drawn — chosen, then walked.
    s.run(
        "man = human()\n"
        "walk = make_gait(man, 'walk', cycle_frames=24)\n"
        "add_character(man, walk, x=150, name='man', color='#33333c')\n"
        "print(f'walk: stride {walk.stride:.0f}px, {walk.speed:.1f}px/frame, "
        "{walk.speed * frames:.0f}px travelled')\n"
    )

    ch = s.characters[0]

    print("\nLINT   (tier 0 - is it broken?  free)")
    rep = lint_rig(ch.body, ch.pose_fn, frames=FRAMES, ground_y=ground_y,
                   limbs=ch.gait.limbs, canvas=(W, H))
    print("  " + rep.format())

    print("\nDIAGNOSE (tier 1 - is it good?  ~500 tokens)")
    print(diagnose_rig(ch.body, ch.pose_fn, frames=FRAMES, ground_y=ground_y,
                       track="arm_lower").format())

    OUT.mkdir(exist_ok=True)
    R.contact_sheet(comp, count=8, cols=4, width=1200).save(OUT / "walk_home.png")
    R.save_gif(comp, str(OUT / "walk_home.gif"), step=2)
    fmt = gio.registry.from_extension("json", gio.Direction.Export)
    (OUT / "walk_home.json").write_bytes(fmt.save(comp.comp))

    print(f"\nwrote {OUT}/walk_home.png (contact sheet)")
    print(f"      {OUT}/walk_home.gif (watch this)")
    print(f"      {OUT}/walk_home.json (Lottie - opens in Glaxnimate)")


if __name__ == "__main__":
    main()
