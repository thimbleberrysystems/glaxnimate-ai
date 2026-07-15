"""Assets as data — the acceptance test for the whole v2 thesis.

The claim: a new creature is a JSON document, authorable with zero Python
changes, validated on load, usable by name. The star witness is the bird below:
it exists nowhere in the code — no template, no preset — and it must load, walk,
and pass the same linter as the shipped bodies.
"""

from __future__ import annotations

import pytest

from glaxnimate_ai.cartoon import assets as A
from glaxnimate_ai.cartoon.gait import pose_at
from glaxnimate_ai.cartoon.presets import human, make_gait, quadruped
from glaxnimate_ai.feedback.lint import lint_rig

GROUND = 420.0

#: A bird, written as raw data the way an LLM would author it. Not derived from
#: any template: stubby legs, a horizontal body, a neck, a beak and a tail.
BIRD = {
    "version": 1,
    "kind": "body",
    "joints": [
        {"name": "hips", "parent": None, "length": 0},
        {"name": "body", "parent": "hips", "length": 70, "rest_angle": -12,
         "offset": [0, 0], "mass": 55},
        {"name": "neck", "parent": "body", "length": 34, "rest_angle": -78, "mass": 8},
        {"name": "head", "parent": "neck", "length": 14, "rest_angle": 66, "mass": 7},
        {"name": "tail", "parent": "hips", "length": 34, "rest_angle": 195,
         "offset": [0, 0], "mass": 6},
        {"name": "thigh_l", "parent": "hips", "length": 26, "rest_angle": 90,
         "offset": [0, 0], "mass": 8},
        {"name": "shin_l", "parent": "thigh_l", "length": 26, "contact": True, "mass": 4},
        {"name": "thigh_r", "parent": "hips", "length": 26, "rest_angle": 90,
         "offset": [0, 0], "mass": 8},
        {"name": "shin_r", "parent": "thigh_r", "length": 26, "contact": True, "mass": 4},
    ],
    "limbs": [
        {"upper": "thigh_l", "lower": "shin_l"},
        {"upper": "thigh_r", "lower": "shin_r"},
    ],
    "swings": [{"joint": "tail", "phase": 0.0, "amplitude": 8.0}],
    "bones": ["thigh_r", "shin_r", "tail", "body", "neck", "head",
              "thigh_l", "shin_l"],
    "parts": {
        "body": {"width": 34, "color": "#4a8fb5"},
        "neck": {"width": 14, "color": "#4a8fb5"},
        "head": {"width": 10, "color": "#4a8fb5", "head": [22, 18], "tip": 4},
        "tail": {"width": 9, "color": "#3a7295", "tip": 5},
        "thigh_l": {"width": 8, "color": "#d9a441"},
        "shin_l": {"width": 6, "color": "#d9a441", "tip": 4},
        "thigh_r": {"width": 8, "color": "#b5893a"},
        "shin_r": {"width": 6, "color": "#b5893a", "tip": 4},
    },
}


@pytest.fixture
def library(tmp_path, monkeypatch):
    """An isolated asset library so tests never touch the shipped one."""
    monkeypatch.setenv("GLAXNIMATE_AI_ASSETS", str(tmp_path))
    return tmp_path


# ------------------------------------------------------- the acceptance test
def test_a_bird_authored_as_json_walks_clean(library):
    """No template, no preset, no Python — a JSON creature must simply work."""
    A.save_asset("body", "bird", BIRD)
    bird = A.load_body("bird")
    gait = make_gait(bird, "walk", cycle_frames=16)

    def pf(t):
        return pose_at(bird.rig, gait, t, ground_y=GROUND, body_x0=80)

    rep = lint_rig(bird, pf, frames=32, ground_y=GROUND,
                   limbs=gait.limbs, canvas=(800, 480))
    assert rep.ok, rep.format()


def test_a_custom_gait_authored_as_json_binds_and_reaches(library):
    """A gait document registers into the live tables and the reach guard still
    protects it — an impossible one fails with the usual advice."""
    A.save_asset("gait", "scuttle", {
        "version": 1, "kind": "gait", "name": "scuttle",
        "phases": {"2": [0.0, 0.5], "4": [0.0, 0.5, 0.25, 0.75]},
        "duty": 0.7, "stride": 0.6, "lift": 0.12, "bob": 0.03,
        "lean": 0, "crouch": 0.98,
    })
    dog = quadruped()
    gait = make_gait(dog, "scuttle", cycle_frames=20)
    assert gait.duty == 0.7

    with pytest.raises(ValueError, match="reach"):
        make_gait(dog, "scuttle", stride=500.0)


# ------------------------------------------------------------- round-tripping
def test_shipped_bodies_round_trip():
    """template -> data -> Body must preserve everything the pipeline reads."""
    for maker in (human, quadruped):
        original = maker()
        rebuilt = A.body_from_data(A.body_to_data(original))
        assert set(rebuilt.rig.joints) == set(original.rig.joints)
        assert rebuilt.leg_length == original.leg_length
        assert rebuilt.bones == original.bones
        assert len(rebuilt.parts) == len(original.parts)
        for name, j in original.rig.joints.items():
            r = rebuilt.rig.joints[name]
            assert (r.parent, r.length, r.contact, r.mass) == \
                   (j.parent, j.length, j.contact, j.mass)


# ------------------------------------------------------- validation teaches
def test_body_with_a_cycle_is_rejected_with_a_teaching_error(library):
    bad = {"version": 1, "kind": "body",
           "joints": [{"name": "a", "parent": "b"}, {"name": "b", "parent": "a"}]}
    with pytest.raises(ValueError, match="cycle|root"):
        A.body_from_data(bad)


def test_limb_referencing_a_ghost_joint_is_rejected(library):
    bad = dict(BIRD)
    bad["limbs"] = [{"upper": "thigh_l", "lower": "wing"}]
    with pytest.raises(ValueError, match="wing"):
        A.body_from_data(bad)


def test_wrong_version_is_rejected_not_misread(library):
    with pytest.raises(ValueError, match="version"):
        A.body_from_data({"version": 99, "kind": "body", "joints": []})


def test_save_never_persists_an_invalid_asset(library):
    with pytest.raises(ValueError):
        A.save_asset("prop", "broken", {"version": 1, "kind": "prop", "shapes": []})
    assert not A.asset_path("prop", "broken").exists()


# ------------------------------------------------------------------ props
def test_a_data_prop_draws_into_a_scene(library):
    A.save_asset("prop", "bench", {
        "version": 1, "kind": "prop",
        "shapes": [
            {"type": "rect", "x": -50, "y": -34, "w": 100, "h": 8, "color": "#8a6642"},
            {"type": "rect", "x": -44, "y": -26, "w": 8, "h": 26, "color": "#6e5236"},
            {"type": "rect", "x": 36, "y": -26, "w": 8, "h": 26, "color": "#6e5236"},
        ],
    })
    from glaxnimate_ai.engine.session import SessionStore

    s = SessionStore().create(width=400, height=300, frames=8)
    res = s.run("add_prop('bench', x=200)")
    assert res.ok, res.format()
    img = s.scene.comp.render_image(0)
    # the bench seat should put opaque pixels near (200, ground-30)
    px = img.load()
    assert px[200, int(s.ground_y) - 30][3] > 0, "the bench did not draw where placed"
