"""Scenes are data on disk: kill the process, reload, get the same animation.

The mechanism under test is "persist the sampled poses, not the program": a pose
function is only ever evaluated at integer frames, so its samples replay exactly —
no attempt to serialize code, which is why *any* script-built scene survives.
"""

from __future__ import annotations

from glaxnimate_ai.engine import scene_doc as SD
from glaxnimate_ai.engine.session import Session, SessionStore
from glaxnimate_ai.feedback.lint import lint_rig

SCRIPT = """
scenery("sky")
scenery("ground")
scenery("house", x=520)
man = human()
ch = add_character(man, make_gait(man, 'walk', cycle_frames=24),
                   x=80, name='man', face='human')
set_expression(ch, 'happy', 0)
set_expression(ch, 'surprised', 30)
ball = motion.bounce(x0=120, x1=600, ground_y=ground, apex=160,
                     frames=frames, bounces=4, radius=24)
add_object(ball, size=Vec2(48, 48), color='#e8543f')
"""


def _build():
    store = SessionStore()
    s = store.create(width=720, height=400, frames=48)
    res = s.run(SCRIPT)
    assert res.ok, res.format()
    return s


def test_a_scene_survives_a_restart_pixel_for_pixel():
    original = _build()
    doc_id = original.doc_id
    before = original.scene.comp.render_image(24).tobytes()

    # A fresh store simulates a restarted server: its in-memory dict is empty,
    # so get() must come back from disk alone.
    reborn = SessionStore().get(doc_id)
    after = reborn.scene.comp.render_image(24).tobytes()

    assert before == after, "replayed scene renders differently from the original"


def test_a_replayed_scene_is_fully_inspectable_and_lintable():
    original = _build()
    reborn = Session.replay(original.doc_id)

    # characters, faces, expressions, objects all made the trip
    ch = reborn.characters[0]
    assert ch.name == "man"
    assert ch.limb_pairs == [("thigh_l", "shin_l"), ("thigh_r", "shin_r")]
    assert set(ch.face_layers) == {"neutral", "happy", "sad", "surprised", "blink"}
    assert ch.expressions == [(0.0, "happy"), (30.0, "surprised")]
    assert len(reborn.objects) == 1

    # and the critic still has everything it needs — including over-extension
    rep = lint_rig(ch.body, ch.pose_fn, frames=reborn.frames,
                   ground_y=reborn.ground_y, limbs=ch.limb_pairs)
    assert rep.ok, rep.format()


def test_describe_reads_like_a_scene():
    s = _build()
    text = SD.describe(s.doc)
    assert "man" in text and "house" in text
    assert "expressions" in text or "swaps" in text


def test_missing_scene_error_lists_what_exists():
    _build()  # ensures at least one saved scene exists
    store = SessionStore()
    try:
        store.get("doc999")
        raise AssertionError("should have raised")
    except KeyError as e:
        assert "doc999" in str(e) and "saved" in str(e)


def test_documents_from_the_future_are_rejected():
    s = _build()
    s.doc["version"] = 99
    s.save()
    try:
        Session.replay(s.doc_id)
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "version" in str(e)
