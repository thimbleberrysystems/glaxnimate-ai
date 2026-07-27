"""The explainer toolkit: diagrams-as-data (pure generators) and the timing hooks
(write_on / counter / highlight) that reveal and animate them.
"""

from __future__ import annotations

import numpy as np

from glaxnimate_ai.cartoon import diagram as D
from glaxnimate_ai.engine.session import Session, SessionStore
from glaxnimate_ai.feedback.render import render_frame


# --------------------------------------------------------------- pure generators
def test_arrow_is_a_shaft_plus_a_head():
    sh = D.arrow(0, 0, 100, 0)
    kinds = [s["type"] for s in sh]
    assert kinds == ["polyline", "polygon"]           # shaft, then arrowhead
    assert sh[0]["points"][0] == [0, 0]


def test_mapper_places_data_in_the_box():
    to = D.mapper(200, 100, (0, 10), (0, 5))
    assert to(0, 0) == (0.0, -0.0)
    assert to(10, 5) == (200.0, -100.0)               # bottom-left origin, y up
    assert to(5, 0) == (100.0, 0.0)


def test_axes_has_lines_and_numbers():
    sh = D.axes(200, 200, xrange=(-2, 2), yrange=(0, 4), xlabel="x")
    assert any(s["type"] == "polyline" for s in sh)
    texts = [s["text"] for s in sh if s["type"] == "text"]
    assert "x" in texts and "2" in texts and "-2" in texts


def test_plot_traces_a_curve_and_breaks_outside_range():
    good = D.plot(lambda x: x, w=100, h=100, xrange=(0, 10), yrange=(0, 10))
    assert len(good) == 1 and good[0]["type"] == "polyline"
    # 1/x blows past the range around 0 -> the curve must split, not draw a spike
    split = D.plot(lambda x: 1.0 / x, w=100, h=100, xrange=(-2, 2), yrange=(-4, 4))
    assert len(split) >= 2


def test_bar_chart_makes_one_bar_per_value():
    sh = D.bar_chart([3, 1, 4], w=120, h=80, labels=["a", "b", "c"])
    assert sum(1 for s in sh if s["type"] == "rect") == 3
    assert sum(1 for s in sh if s["type"] == "text") == 3


# ------------------------------------------------------------------ write_on
def test_write_on_draws_progressively_and_replays():
    s = SessionStore().create(width=240, height=140, frames=40, ground_y=120)
    r = s.run('background("#fff")\n'
              'write_on(axes(160, 90, xrange=(0,8), yrange=(0,8)), 40, 110, start=0, end=30)')
    assert r.ok, r.format()

    def ink(f):
        return int((np.asarray(render_frame(s.scene, f).convert("L")) < 170).sum())
    assert ink(2) < ink(28), "the diagram should still be drawing on"
    assert len(s.doc["writeons"]) == 1
    s.save()
    assert len(Session.replay(s.doc_id).doc["writeons"]) == 1


# ------------------------------------------------------------------ counter
def test_counter_shows_one_value_at_a_time_and_replays():
    s = SessionStore().create(width=200, height=100, frames=40, ground_y=80)
    s.run('background("#fff")\ncounter(100, 55, v0=0, v1=50, start=4, end=36, size=30)')
    assert len(s.doc["counters"]) == 1
    # ink is present at the end (the held final value renders)
    assert int((np.asarray(render_frame(s.scene, 38).convert("L")) < 120).sum()) > 40
    s.save()
    back = Session.replay(s.doc_id)
    assert len(back.doc["counters"]) == 1
    assert int((np.asarray(render_frame(back.scene, 38).convert("L")) < 120).sum()) > 40


# ------------------------------------------------------------------ highlight
def test_highlight_ring_renders_and_persists():
    s = SessionStore().create(width=200, height=200, frames=20, ground_y=180)
    s.run('background("#fff")\nhighlight(100, 100, w=80, h=80, appear=0, color="#e0533d")')
    assert int((np.asarray(render_frame(s.scene, 5).convert("L")) < 170).sum()) > 80
    assert any(sh.name.startswith("highlight.") for sh in s.scene.comp.shapes)
