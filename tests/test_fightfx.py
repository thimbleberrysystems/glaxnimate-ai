"""Fight FX: the effects language real stick animations run on — motion smears,
speed ghosts, charge auras, energy beams, ragdoll throws, weapon twirls.
"""

from __future__ import annotations

import math

from glaxnimate_ai.cartoon import actions
from glaxnimate_ai.cartoon.presets import stick
from glaxnimate_ai.engine.session import Session, SessionStore

GROUND = 300.0


def _mk(frames=20):
    s = SessionStore().create(width=520, height=340, frames=frames)
    s.run("f = stick()\n"
          "add_action(f, actions.punch(f, ground_y=ground, x=170, frames=16), name='f')")
    return s


def test_smear_flashes_a_translucent_arc_on_the_strike():
    s = _mk()
    r = s.run("smear('f', 'arm_lower', 10, span=5)")
    assert r.ok, r.format()
    lay = next(sh for sh in s.scene.comp.shapes if sh.name.startswith("smear."))
    # invisible before the strike, translucent on it, gone after
    assert lay.opacity.value_at_time(6) == 0
    assert 0.2 < lay.opacity.value_at_time(10) < 0.9
    assert lay.opacity.value_at_time(14) == 0


def test_aura_follows_the_character_and_pulses():
    s = _mk()
    r = s.run("aura('f', color='#ffd23f', radius=48, pulse=5)")
    assert r.ok, r.format()
    g = next(sh for sh in s.scene.comp.shapes if sh.name.startswith("aura.")).shapes[0]
    # it tracks the figure (position keys) and beats (scale varies)
    assert g.transform.position.keyframe_count > 2
    sizes = [g.transform.scale.value_at_time(float(f)).x for f in range(20)]
    assert max(sizes) - min(sizes) > 0.1, "the aura should pulse"


def test_beam_spans_its_endpoints_and_lights_on_cue():
    s = _mk()
    r = s.run("beam(180, 150, 480, 150, start=4, end=14, width=24)")
    assert r.ok, r.format()
    lay = next(sh for sh in s.scene.comp.shapes if sh.name.startswith("beam."))
    g = lay.shapes[0]
    assert round(g.transform.scale.value_at_time(6).x) == 300, "beam length == span"
    assert lay.opacity.value_at_time(2) == 0 and lay.opacity.value_at_time(6) > 0.5


def test_afterimage_bakes_faded_ghosts_that_flash_and_fade():
    s = _mk()
    r = s.run("afterimage('f', 12, count=3, gap=2)")
    assert r.ok, r.format()
    # 3 ghosts, each a faded rig; their layers are gated invisible at frame 0
    ghost_layers = [sh for sh in s.scene.comp.shapes if sh.name.startswith("ghost.")]
    assert len(ghost_layers) >= 3, "three ghost rigs baked"
    assert all(sh.opacity.value_at_time(0) == 0 for sh in ghost_layers), \
        "ghosts start invisible and only flash near the strike"


def test_tossed_is_a_spinning_airborne_arc():
    body = stick()
    fn = actions.tossed(body, ground_y=GROUND, x0=120, x1=460, apex=120, spins=1.5, frames=24)
    xs = [fn(float(f)).root.x for f in range(25)]
    ys = [fn(float(f)).root.y for f in range(25)]
    assert xs[-1] > xs[0] + 200, "it should fly across"
    assert min(ys) < ys[0] - 60, "it should arc up over the throw"
    assert abs(fn(24.0).root_angle) > 300, "it should spin"
    for f in range(25):  # every joint stays finite
        for jf in body.rig.solve(fn(float(f))).values():
            assert math.isfinite(jf.tip.x) and math.isfinite(jf.tip.y)


def test_wield_spin_twirls_the_prop():
    s = SessionStore().create(width=340, height=320, frames=24)
    r = s.run("f = stick()\n"
              "add_action(f, actions.idle(f, ground_y=ground, x=170), name='f')\n"
              "wield('f', [{'type':'rect','x':0,'y':-2,'w':50,'h':4,'color':'#888'}],"
              " bone='arm_lower', spin=720)")
    assert r.ok, r.format()
    assert s.doc["wields"][0]["spin"] == 720


def test_fight_fx_survive_save_and_replay():
    s = _mk()
    s.run("smear('f','arm_lower',10)\naura('f')\nbeam(180,150,480,150,start=4,end=14)\n"
          "afterimage('f',12,count=2)")
    assert len(s.doc["overlays"]) == 4
    s.save()
    replayed = Session.replay(s.doc_id)
    assert len(replayed.doc["overlays"]) == 4
    assert replayed.scene.comp.render_image(10).width == 520
