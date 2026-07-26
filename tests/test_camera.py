"""Camera: pan/zoom/shake via a world container layer, neutral outside its window.

The camera is one layer that all content re-parents under, so its transform is the
camera and the fix to transform.scale (WS0) is what makes zoom possible at all. The
claims: an impact camera pushes in and shakes on the hit and returns to neutral;
a camera move eases and holds; and both survive save + replay.
"""

from __future__ import annotations

from glaxnimate_ai.engine.session import Session, SessionStore


def _build(frames=20):
    s = SessionStore().create(width=360, height=340, frames=frames)
    s.run("scenery('ground')\nman = stick()\n"
          "add_character(man, make_gait(man, 'walk'), x=180, name='m')")
    return s


def test_impact_camera_pushes_in_and_returns_to_neutral():
    s = _build()
    r = s.run("impact_camera(10, x=250, y=120, zoom=1.4, shake=8)")
    assert r.ok, r.format()
    cam = s.camera_layer
    assert cam is not None, "a camera layer should have been created"
    # neutral before, zoomed at the peak, neutral again after
    z_before = cam.transform.scale.value_at_time(0).x
    z_peak = cam.transform.scale.value_at_time(13).x   # ramp(2)+hold(3) -> ~peak
    z_after = cam.transform.scale.value_at_time(19).x
    assert abs(z_before - 1.0) < 1e-6, "camera starts neutral"
    assert z_peak > 1.25, "camera should push in on the hit"
    assert abs(z_after - 1.0) < 0.02, "camera returns to neutral"


def test_camera_reparents_top_level_content_under_one_layer():
    s = _build()
    # the layers that are top-level (parent None) before the camera exists
    tops_before = [sh for sh in s.scene.comp.shapes if sh.parent is None]
    s.run("impact_camera(10, x=180, y=150)")
    cam = s.camera_layer
    # each of those now hangs off the camera layer (bone sub-layers keep their
    # character-root parent, which is itself now under the camera)
    for sh in tops_before:
        assert sh is cam or sh.parent is cam, \
            f"{sh.name} was not re-parented under the camera"
    assert len(tops_before) >= 2


def test_camera_move_eases_and_holds():
    s = _build()
    r = s.run("camera_move(4, 12, zoom=1.5, focus_x=250, focus_y=100)")
    assert r.ok, r.format()
    cam = s.camera_layer
    assert abs(cam.transform.scale.value_at_time(4).x - 1.0) < 1e-6, "neutral at start"
    assert cam.transform.scale.value_at_time(12).x > 1.4, "reaches the push-in"


def test_camera_survives_save_and_replay():
    s = _build()
    s.run("impact_camera(10, x=250, y=120, zoom=1.3, shake=9)")
    s.save()
    replayed = Session.replay(s.doc_id)
    assert len(replayed.doc.get("camera", [])) == 1
    assert replayed.camera_layer is not None
    peak = replayed.camera_layer.transform.scale.value_at_time(13).x
    assert peak > 1.2, "the replayed camera pushes in identically"
    # and it renders
    assert replayed.scene.comp.render_image(11).width == 360
