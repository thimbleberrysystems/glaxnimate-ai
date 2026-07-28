"""Scene styles: themed worlds behind the action. A backdrop is shape data, so it
composes and replays; a theme also hands back the ink that reads well in it.
"""

from __future__ import annotations

import numpy as np

from glaxnimate_ai.cartoon import scene_style as SS
from glaxnimate_ai.engine.session import Session, SessionStore
from glaxnimate_ai.feedback.render import render_frame

THEMES = ["night", "sunset", "blueprint", "notebook", "chalkboard", "vaporwave",
          "comic", "spotlight", "sky", "paper"]


def test_registry_and_palette():
    for t in THEMES:
        assert t in SS.theme_names()
        pal = SS.theme_palette(t)
        assert {"ink", "accent", "ground"} <= set(pal)


def test_gradient_bands_and_starfield_are_deterministic():
    g = SS.gradient_bg(200, 100, "#000000", "#ffffff", bands=10)
    assert len(g) == 10 and g[0]["color"] != g[-1]["color"]
    assert SS.stars(200, 100, n=30, seed=3) == SS.stars(200, 100, n=30, seed=3)


def test_each_theme_builds_a_backdrop():
    for t in THEMES:
        sh = SS.theme_backdrop(t, 400, 300, 260)
        assert sh and all("type" in s and "color" in s for s in sh)


def test_backdrop_style_fills_the_frame_and_differs_by_theme():
    def corner(theme):
        s = SessionStore().create(width=200, height=160, frames=4, ground_y=140)
        assert s.run(f'backdrop_style("{theme}")').ok
        return render_frame(s.scene, 0).convert("RGB").getpixel((6, 6))
    night = corner("night")
    sky = corner("sky")
    assert sum(night) < 220 and sum(sky) > 380      # night is dark, sky is light
    assert night != sky


def test_backdrop_style_persists_and_replays():
    s = SessionStore().create(width=200, height=160, frames=4, ground_y=140)
    s.run('backdrop_style("blueprint")')
    assert s.doc["backdrop_style"] == {"name": "blueprint"}
    s.save()
    back = Session.replay(s.doc_id)
    assert back.doc["backdrop_style"] == {"name": "blueprint"}
    # the blueprint's deep blue still fills the corner after replay
    assert render_frame(back.scene, 0).convert("RGB").getpixel((6, 6))[2] > 60


def test_theme_and_figure_compose():
    s = SessionStore().create(width=260, height=300, frames=24, ground_y=278)
    r = s.run('backdrop_style("night")\n'
              'pal = theme_palette("night")\n'
              'b = biped()\n'
              'add_action(b, actions.wave(b, ground_y=ground, x=130, cycles=2, frames=24), '
              'name="c", style="neon", face="human")')
    assert r.ok, r.format()
    assert np.asarray(render_frame(s.scene, 10).convert("L")).std() > 8
