"""Advanced techniques from the real-animation benchmark: a physics ragdoll,
wall-run/wall-jump, gunplay, clones, and screenshot-backdrop (Animator vs
Animation) with a cursor that ragdoll-drags the figure.
"""

from __future__ import annotations

import math

from PIL import Image

from glaxnimate_ai.cartoon import actions, physics
from glaxnimate_ai.cartoon.geometry import Vec2
from glaxnimate_ai.cartoon.presets import stick
from glaxnimate_ai.cartoon.rig import Pose
from glaxnimate_ai.engine.session import Session, SessionStore

GROUND = 340.0


# ------------------------------------------------------------------- ragdoll
def test_ragdoll_falls_settles_and_stays_finite():
    body = stick()
    pose0 = Pose(root=Vec2(120, GROUND - body.hip_height), root_angle=0.0)
    fn = physics.ragdoll(body, pose0, ground_y=GROUND, frames=40,
                         launch=(200, -240), spin=180)
    for f in range(41):                       # never blows up
        for jf in body.rig.solve(fn(float(f))).values():
            assert math.isfinite(jf.tip.x) and math.isfinite(jf.tip.y)
    # it launches up then comes to rest low, near the ground, and stops moving
    root_y = [fn(float(f)).root.x for f in range(41)]
    assert fn(40.0).root.x > fn(0.0).root.x + 80, "it should travel from the throw"
    late = [body.rig.solve(fn(float(f)))["shin_l"].tip.y for f in range(34, 41)]
    assert max(late) >= GROUND - 40, "it should end up down near the ground"
    assert max(late) - min(late) < 30, "and settle (stop tumbling)"


def test_ragdoll_pin_follows_a_path():
    body = stick()
    pose0 = Pose(root=Vec2(100, GROUND - body.hip_height), root_angle=0.0)
    path = lambda f: Vec2(100 + f * 4, 120)      # a moving grab point
    fn = physics.ragdoll(body, pose0, ground_y=GROUND, frames=20,
                         pin="head", pin_path=path)
    # the pinned head tracks the path (roughly), the body dangling below
    head = body.rig.solve(fn(10.0))["head"].tip
    assert abs(head.x - path(10).x) < 40 and abs(head.y - path(10).y) < 40


# ------------------------------------------------------------------- parkour
def test_wall_run_climbs_and_stays_off_the_ground():
    body = stick()
    fn = actions.wall_run(body, wall_x=300, y0=GROUND - 100, y1=120, frames=18)
    assert fn(17.0).root.y < fn(0.0).root.y - 100, "it should climb the wall"
    for f in range(19):                       # feet never touch the ground plane
        for shin in ("shin_l", "shin_r"):
            assert body.rig.solve(fn(float(f)))[shin].tip.y < GROUND - 5


def test_aim_points_the_arm_at_the_target():
    body = stick()
    fn = actions.aim(body, ground_y=GROUND, x=120, target_x=460, target_y=150)
    frames = body.rig.solve(fn(11.0))
    hand = frames["arm_lower"].tip
    shoulder = frames["arm_upper"].origin
    # the hand ends up on the target side of the shoulder (pointing right/forward)
    assert hand.x > shoulder.x + 20


# ------------------------------------------------------------------- gunplay
def test_shoot_places_muzzle_flash_tracer_and_impact():
    s = SessionStore().create(width=560, height=340, frames=20)
    s.run("g = stick()\n"
          "add_action(g, actions.aim(g, ground_y=ground, x=120, target_x=440, target_y=150), name='g')\n"
          "shoot('g', 430, 150, 10)")
    assert any(e["frame"] == 10 for e in s.doc["effects"]), "muzzle flash"
    assert any(o["kind"] == "beam" for o in s.doc["overlays"]), "tracer beam"
    assert any(c["frame"] == 10 for c in s.doc["audio"]["cues"]), "gunshot sfx"


# ------------------------------------------------------------------- clones
def test_clones_register_n_characters():
    s = SessionStore().create(width=640, height=340, frames=16)
    s.run("f = stick()\n"
          "clones(f, lambda i: actions.punch(f, ground_y=ground, x=110+i*120, frames=16), 4)")
    assert len(s.characters) == 4
    assert {c.name for c in s.characters} == {"clone0", "clone1", "clone2", "clone3"}


# ------------------------------------------------- screenshot backdrop (AvA)
def _make_png(tmp_path):
    p = tmp_path / "shot.png"
    Image.new("RGB", (760, 400), "#2d6cdf").save(p)
    return p


def test_backdrop_loads_an_image_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("GLAXNIMATE_AI_PROJECTS", str(tmp_path / "proj"))
    png = _make_png(tmp_path)
    s = SessionStore().create(width=760, height=400, frames=12)
    r = s.run(f"backdrop({str(png)!r})\n"
              "man = stick()\n"
              "add_character(man, make_gait(man,'walk',cycle_frames=16), x=200, name='man')")
    assert r.ok, r.format()
    assert s.doc["backdrop"] and s.doc["backdrop"]["path"] == str(png)
    assert any(sh.name == "backdrop" for sh in s.scene.comp.shapes)
    s.save()
    replayed = Session.replay(s.doc_id)
    assert replayed.doc["backdrop"]["path"] == str(png)
    assert replayed.scene.comp.render_image(4).width == 760


def test_cursor_drag_hauls_a_ragdoll_along_the_path(tmp_path, monkeypatch):
    monkeypatch.setenv("GLAXNIMATE_AI_PROJECTS", str(tmp_path / "proj"))
    png = _make_png(tmp_path)
    s = SessionStore().create(width=760, height=400, frames=30)
    r = s.run(
        f"backdrop({str(png)!r})\n"
        "man = stick()\n"
        "path = [(210,150),(420,110),(600,280)]\n"
        "add_action(man, drag(man, path, grab='head', frames=30), name='victim')\n"
        "cursor(path, frames=30)")
    assert r.ok, r.format()
    ch = s.characters[0]
    # the grabbed head tracks the cursor: it ends near the path's end
    head_end = ch.body.rig.solve(ch.pose_fn(29.0))["head"].tip
    assert abs(head_end.x - 600) < 60 and abs(head_end.y - 280) < 80
    assert any(sh.name == "cursor" for sh in s.scene.comp.shapes)
