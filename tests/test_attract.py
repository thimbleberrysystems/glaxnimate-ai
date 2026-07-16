"""Inverse-power attraction: the curve that makes a magnet look magnetic.

These tests exist because the obvious primitive is the wrong one. `spring()` was
the intuitive choice for "two magnets snap together" and it is elastic, not
magnetic — it lunges when far, overshoots the target and wobbles back. Every
assertion below is a property that separates the two, so nobody (including me)
reaches for the springy one again.
"""
from __future__ import annotations

import pytest

from glaxnimate_ai.cartoon.geometry import Vec2
from glaxnimate_ai.cartoon.motion import attract, spring


def _xs(samples):
    return [s.pos.x for s in samples]


def test_the_snatch_is_late_where_a_springs_lunge_is_early():
    """The defining property. A magnet does almost nothing at range, then takes
    the last stretch violently; a spring does the exact opposite."""
    a = _xs(attract(start=Vec2(0, 0), end=Vec2(300, 0), frames=40, contact_gap=20))
    s = _xs(spring(start=Vec2(0, 0), end=Vec2(300, 0), frames=40))
    travel = 280.0

    assert a[10] / travel < 0.10, "attract moved too much, too early — that reads elastic"
    assert a[20] / travel < 0.35, "the snatch must still be ahead at the halfway mark"
    assert a[-1] / travel == pytest.approx(1.0, abs=0.01)

    # the same measurement convicts the spring
    assert s[10] / 300.0 > 1.0, "spring should already have overshot by frame 10"


def test_it_stops_at_the_surface_and_never_passes_through():
    """Two magnets collide. `spring` sails 24% past its target; a solid object
    cannot, and a pass-through is the ugliest possible tell."""
    contact = 20.0
    a = attract(start=Vec2(0, 0), end=Vec2(300, 0), frames=40, contact_gap=contact,
                ring=8.0)
    limit = 300.0 - contact
    assert max(_xs(a)) <= limit + 1e-6, "passed through the surface"
    assert _xs(a)[-1] == pytest.approx(limit, abs=0.5)


def test_speed_rises_toward_contact():
    a = _xs(attract(start=Vec2(0, 0), end=Vec2(300, 0), frames=40, contact_gap=20))
    v = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    assert v[30] > v[10] * 3, "the force must run away with itself, not die out"
    assert all(x >= -1e-9 for x in v[:35]), "approach must be monotonic (no wobble)"


@pytest.mark.parametrize("frames", [12, 40, 90])
@pytest.mark.parametrize("power", [2.0, 3.0, 4.0])
def test_contact_lands_on_the_frame_you_asked_for(frames, power):
    """The bargain: honest physics has no sense of timing, but a cut does. The
    curve keeps its shape; the impact lands on the beat regardless of power."""
    a = attract(start=Vec2(0, 0), end=Vec2(400, 0), frames=frames, power=power,
                contact_gap=25)
    assert len(a) == frames + 1
    assert _xs(a)[-1] == pytest.approx(375.0, abs=1.0)
    assert _xs(a)[0] == pytest.approx(0.0, abs=1e-6)


def test_a_steeper_power_snaps_later_and_harder():
    """Why `power` is exposed: a dipole grabs more abruptly than gravity."""
    gravity = _xs(attract(start=Vec2(0, 0), end=Vec2(300, 0), frames=40, power=2.0))
    dipole = _xs(attract(start=Vec2(0, 0), end=Vec2(300, 0), frames=40, power=4.0))
    assert dipole[25] < gravity[25], "power=4 should still be hanging back at f25"


def test_it_works_diagonally_not_just_along_x():
    a = attract(start=Vec2(0, 0), end=Vec2(300, 400), frames=30, contact_gap=50)
    last = a[-1].pos
    # 500px hypotenuse, stopping 50 short => 450 along the same line
    assert last.length() == pytest.approx(450.0, abs=1.0)
    assert last.y / last.x == pytest.approx(400 / 300, rel=1e-3), "drifted off the line"


def test_the_ring_rattles_without_delaying_the_impact():
    quiet = attract(start=Vec2(0, 0), end=Vec2(300, 0), frames=40, contact_gap=20)
    loud = attract(start=Vec2(0, 0), end=Vec2(300, 0), frames=40, contact_gap=20,
                   ring=10.0)
    assert _xs(loud)[:30] == pytest.approx(_xs(quiet)[:30]), "ring touched the approach"
    assert _xs(loud)[-1] == pytest.approx(_xs(quiet)[-1], abs=0.5), "must settle at rest"
    assert any(abs(a - b) > 1.0
               for a, b in zip(_xs(loud)[30:], _xs(quiet)[30:], strict=True)), \
        "ring produced no rattle at all"


def test_teaching_errors():
    with pytest.raises(ValueError, match="contact_gap"):
        attract(start=Vec2(0, 0), end=Vec2(10, 0), frames=20, contact_gap=30)
    with pytest.raises(ValueError, match="frames"):
        attract(start=Vec2(0, 0), end=Vec2(300, 0), frames=0)
