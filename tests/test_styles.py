"""Render styles: the same rig, many looks. A style is a per-bone skin, so it
reskins *any* body — humanoid or not — and serialises with the scene.
"""

from __future__ import annotations

import numpy as np

from glaxnimate_ai.cartoon import assets, presets
from glaxnimate_ai.engine.session import Session, SessionStore
from glaxnimate_ai.feedback.render import render_frame

STYLES = ["capsule", "lineart", "comic", "flat", "silhouette", "rubber_hose",
          "neon", "sketch", "blocky"]


def test_registry_lists_every_style():
    for s in STYLES:
        assert s in presets.style_names()


def test_apply_style_sets_the_render_mode():
    b = presets.human()
    assert presets.apply_style(b, "silhouette").parts["spine"].render == "capsule"
    assert presets.apply_style(b, "comic").parts["spine"].render == "outline"
    assert presets.apply_style(b, "neon").parts["spine"].render == "glow"
    assert presets.apply_style(b, "sketch").parts["spine"].render == "rough"
    assert presets.apply_style(b, "blocky").parts["spine"].render == "blocky"
    # the source body is untouched (reskin returns a new Body)
    assert b.parts["spine"].render == "capsule"


def _ink(style, body_expr="human()", face="human"):
    s = SessionStore().create(width=200, height=320, frames=24, ground_y=300)
    bg = "#101018" if style == "neon" else "#ffffff"
    r = s.run(f'background("{bg}")\nb = {body_expr}\n'
              f'add_action(b, actions.celebrate(b, ground_y=ground, x=100, frames=20), '
              f'name="c", style={style!r}, face={face!r})')
    assert r.ok, r.format()
    a = np.asarray(render_frame(s.scene, 10).convert("L"))
    return a


def test_every_style_draws_something():
    for style in STYLES:
        a = _ink(style)
        assert (a < 200).sum() > 400 or (a > 60).sum() > 400, f"{style} rendered blank"


def test_silhouette_is_darker_and_flatter_than_capsule():
    cap = _ink("capsule")
    sil = _ink("silhouette")
    # the silhouette is one dark ink -> more near-black pixels than the coloured body
    assert (sil < 60).sum() > (cap < 60).sum()


def test_style_reskins_a_non_humanoid_rig():
    # "animate anything": a dog takes the same neon treatment
    s = SessionStore().create(width=360, height=240, frames=24, ground_y=210)
    r = s.run('background("#101018")\n'
              'd = quadruped()\n'
              'add_character(d, make_gait(d, "trot", cycle_frames=16), x=120, '
              'name="dog", style="neon")')
    assert r.ok, r.format()
    assert np.asarray(render_frame(s.scene, 6).convert("L")).std() > 5


def test_style_serialises_and_replays():
    b = presets.apply_style(presets.human(), "comic")
    back = assets.body_from_data(assets.body_to_data(b))
    assert back.parts["spine"].render == "outline"
    assert back.parts["spine"].outline == b.parts["spine"].outline

    s = SessionStore().create(width=200, height=320, frames=24, ground_y=300)
    s.run('add_action(rubber_hose(human()), '
          'actions.idle(human(), ground_y=ground, x=100), name="c", face="human")')
    s.save()
    replayed = Session.replay(s.doc_id)
    assert np.asarray(render_frame(replayed.scene, 5).convert("L")).std() > 5
