"""A dog chases a bouncing ball — the coordination fix, end to end.

Run:  .venv/bin/python examples/dog_chases_ball.py

Driving this by hand took four iterations: the ball kept outrunning the dog, and a
too-violent gallop face-planted. Neither failure is visible in any per-character
metric — each character is individually fine; it is the *relationship* that is
wrong, and the fix is `add_chaser`, which paces the dog to the ball.
"""

from __future__ import annotations

from pathlib import Path

from glaxnimate_ai.engine import props
from glaxnimate_ai.engine.session import SessionStore
from glaxnimate_ai.feedback import render as R
from glaxnimate_ai.feedback.diagnose import diagnose_rig
from glaxnimate_ai.feedback.lint import lint_rig

OUT = Path("out")


def main() -> None:
    s = SessionStore().create(width=1000, height=440, frames=72, fps=24)
    g = s.ground_y

    sky = s.scene.layer("sky")
    props.sky(s.scene, sky, top="#bfe3f5")
    props.sun(sky, 880, 66)
    mid = s.scene.layer("mid")
    for x in (120, 300, 820):
        props.tree(mid, x, g, h=130)
    props.ground(s.scene, s.scene.layer("ground"), g)

    s.run(
        "ball = motion.bounce(x0=140, x1=660, ground_y=ground, apex=185,\n"
        "                     frames=frames, bounces=6, restitution=0.68, radius=26)\n"
        "add_object(ball, size=Vec2(52,52), color='#e8543f')\n"
        "add_chaser(quadruped(), 'trot', ball, x=70, gap=40, cycle_frames=16, name='dog')\n"
    )

    ch = s.characters[0]
    print("LINT:", lint_rig(ch.body, ch.pose_fn, frames=72, ground_y=g,
                            limbs=ch.gait.limbs, canvas=(1000, 440)).format())
    print(diagnose_rig(ch.body, ch.pose_fn, frames=72, ground_y=g).format())

    OUT.mkdir(exist_ok=True)
    R.contact_sheet(s.scene, count=8, cols=4, width=1200).save(OUT / "dog_chases_ball.png")
    R.save_gif(s.scene, str(OUT / "dog_chases_ball.gif"), step=2)
    print(f"\nwrote {OUT}/dog_chases_ball.png and .gif")


if __name__ == "__main__":
    main()
