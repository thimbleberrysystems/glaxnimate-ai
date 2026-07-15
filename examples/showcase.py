"""Everything at once: locomotion, an action, follow-through, and physics.

Run:  .venv/bin/python examples/showcase.py

Four things share one scene, each exercising a different part of the library and
each linting clean: a man walks, a dog trots with a tail that trails behind it, a
figure jumps (anticipation, arc, squash landing), and a ball bounces. If the whole
system works, this is what "do it all" looks like in one frame.
"""

from __future__ import annotations

from pathlib import Path

from glaxnimate_ai.engine import props
from glaxnimate_ai.engine.session import SessionStore
from glaxnimate_ai.feedback import render as R
from glaxnimate_ai.feedback.lint import lint_rig

OUT = Path("out")


def main() -> None:
    s = SessionStore().create(width=1200, height=460, frames=48)
    g = s.ground_y

    sky = s.scene.layer("sky")
    props.sky(s.scene, sky)
    props.sun(sky, 1100, 60)
    for x in (140, 500, 950):
        props.cloud(s.scene.layer("far"), x, 70)
    for x in (330, 760):
        props.tree(s.scene.layer("mid"), x, g, h=120)
    props.ground(s.scene, s.scene.layer("ground"), g)

    s.run(
        # a man walking
        "add_character(human(), make_gait(human(), 'walk', cycle_frames=24),"
        " x=60, name='walker')\n"
        # a dog trotting, its tail trailing behind it (follow-through)
        "dog = quadruped()\n"
        "trot = make_gait(dog, 'trot', cycle_frames=16)\n"
        "base = lambda t: pose_at(dog.rig, trot, t, ground_y=ground, body_x0=330)\n"
        "add_action(dog, actions.trail(base, dog, ['tail'], lag=3, swing=34), name='dog')\n"
        # a figure jumping: anticipation, arc, squash landing
        "add_action(human(), actions.jump(human(), ground_y=ground, x=720,"
        " height=150, frames=frames), name='jumper')\n"
        # a bouncing ball
        "add_object(motion.bounce(x0=900, x1=1140, ground_y=ground, apex=170,"
        " frames=frames, bounces=4), color='#e8543f')\n"
    )

    for ch in s.characters:
        rep = lint_rig(ch.body, ch.pose_fn, frames=48, ground_y=g,
                       limbs=ch.gait.limbs if ch.gait else None, canvas=(1200, 460))
        print(f"lint {ch.name}: {rep.format().splitlines()[0]}")

    OUT.mkdir(exist_ok=True)
    R.contact_sheet(s.scene, count=6, cols=3, width=1200).save(OUT / "showcase.png")
    R.save_gif(s.scene, str(OUT / "showcase.gif"), step=2)
    print(f"wrote {OUT}/showcase.png and .gif")


if __name__ == "__main__":
    main()
