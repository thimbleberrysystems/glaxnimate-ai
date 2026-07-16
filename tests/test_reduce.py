"""The keyframe reducer: sparse keys must provably replay the dense truth.

Three layers of proof, because "the file got smaller" is not the claim — the
claim is *smaller and faithful and editable*:

1. **Parity** — our pure-Python evaluator matches Glaxnimate's `value_at_time`
   for the same keys and transitions, so every other test here means something.
2. **Fidelity** — reconstructing the world through the reduced channels stays
   within tolerance of the dense Timeline IR, including the hard case: residual
   drift of a planted foot.
3. **Budget** — a full walk bakes in a bounded number of keys, an order of
   magnitude below v1's frame-by-frame bake.
"""

from __future__ import annotations

import math

import pytest

from glaxnimate_ai.cartoon import timeline as T
from glaxnimate_ai.cartoon.gait import pose_at
from glaxnimate_ai.cartoon.geometry import Vec2
from glaxnimate_ai.cartoon.presets import biped, make_gait
from glaxnimate_ai.engine.reduce import (
    evaluate_point,
    evaluate_scalar,
    reduce_point,
    reduce_scalar,
)

FRAMES, GROUND = 96, 470.0


def _walk_channels():
    """Sample the walk's local channels exactly as the baker does."""
    body = biped()
    gait = make_gait(body, "walk", cycle_frames=24)
    rig = body.rig

    def pf(t):
        return pose_at(rig, gait, t, ground_y=GROUND, body_x0=90)

    root_pos, root_rot = [], []
    local = {n: [] for n in rig.joints if n != rig.root_name}
    for f in range(FRAMES + 1):
        p = pf(float(f))
        root_pos.append(p.root)
        jr = rig.joints[rig.root_name]
        root_rot.append(p.root_angle + p.angles.get(rig.root_name, 0) + jr.rest_angle)
        for n in local:
            local[n].append(p.angles.get(n, 0) + rig.joints[n].rest_angle)
    return body, gait, pf, root_pos, root_rot, local


def _reduced_world(body, root_pos, root_rot, local):
    """World transforms reconstructed from reduced channels — the layer math in
    pure Python, contact chains at the baker's tight tolerance."""
    rig = body.rig
    contact_chain: set[str] = set()
    for name, j in rig.joints.items():
        if j.contact or j.rolling:
            contact_chain.update(rig.chain(name))

    kp = reduce_point(root_pos, tol=0.75)
    kr = reduce_scalar(root_rot, tol=0.75)
    kl = {
        n: reduce_scalar(v, tol=(0.3 if n in contact_chain else 0.75))
        for n, v in local.items()
    }
    n_keys = len(kp) + len(kr) + sum(len(k) for k in kl.values())

    def world_at(f):
        out = {rig.root_name: (evaluate_point(kp, f), evaluate_scalar(kr, f))}
        for n in rig._order:
            j = rig.joints[n]
            if j.parent is None:
                continue
            po, pa = out[j.parent]
            off = j.offset if j.offset is not None else Vec2(rig.joints[j.parent].length, 0)
            out[n] = (po + off.rotated(pa), pa + evaluate_scalar(kl[n], f))
        return out

    return world_at, n_keys


# ------------------------------------------------------------------ synthetic
def test_sine_curve_reduces_within_tolerance():
    vals = [30 * math.sin(2 * math.pi * f / 24) for f in range(97)]
    keys = reduce_scalar(vals, tol=0.5)
    assert len(keys) < len(vals) / 3, "a sinusoid should reduce dramatically"
    worst = max(abs(evaluate_scalar(keys, f) - vals[f]) for f in range(97))
    assert worst <= 0.5


def test_linear_channel_costs_two_keys():
    vals = [2.5 * f for f in range(97)]
    keys = reduce_scalar(vals, tol=0.5)
    assert len(keys) == 2, "a straight line is two keys, full stop"


def test_static_channel_costs_zero_animated_keys():
    keys = reduce_scalar([12.0] * 97, tol=0.5)
    assert len(keys) == 1  # written as a static value, not a keyframe


def test_point_channel_bows_get_split():
    pts = [Vec2(f * 5, 40 * math.sin(math.pi * f / 48)) for f in range(97)]
    keys = reduce_point(pts, tol=0.75)
    worst = max(evaluate_point(keys, f).distance_to(pts[f]) for f in range(97))
    assert worst <= 0.75
    assert len(keys) < 30


def test_ease_false_still_meets_tolerance():
    """Linear-only mode (the scale-channel workaround) trades keys, not error."""
    vals = [30 * math.sin(2 * math.pi * f / 24) for f in range(97)]
    eased = reduce_scalar(vals, tol=0.5)
    linear = reduce_scalar(vals, tol=0.5, ease=False)
    worst = max(abs(evaluate_scalar(linear, f) - vals[f]) for f in range(97))
    assert worst <= 0.5
    assert len(linear) >= len(eased), "linear mode may not be cheaper than eased"


# --------------------------------------------------------------------- parity
def test_evaluator_matches_glaxnimate_interpolation():
    """If this fails, every fidelity number in this file is fiction."""
    from glaxnimate import model, utils

    from glaxnimate_ai.engine.reduce import ScalarKey
    from glaxnimate_ai.engine.session import SessionStore

    SessionStore()  # ensures the shared Headless environment exists
    d = model.Document("")
    c = d.assets.add_composition()
    c.animation.last_frame = 48
    lay = c.add_shape("Layer")
    lay.animation.last_frame = 48
    g = lay.add_shape("Group")

    keys = [ScalarKey(0, 0.0, 0.55, 0.2), ScalarKey(20, 80.0, 0.1, 0.9),
            ScalarKey(48, -30.0)]
    rot = g.transform.rotation
    for k in keys:
        rot.set_keyframe(float(k.frame), k.value)
    for k in keys[:-1]:
        tr = model.KeyframeTransition()
        tr.before = utils.Point(1 / 3, k.cy1)
        tr.after = utils.Point(2 / 3, k.cy2)
        rot.set_transition(float(k.frame), tr)

    worst = max(abs(evaluate_scalar(keys, f) - rot.value_at_time(f)) for f in range(49))
    assert worst < 1e-3, f"evaluator diverges from glaxnimate by {worst}"


# ------------------------------------------------------------------- fidelity
def test_reduced_walk_stays_close_to_the_dense_truth():
    body, gait, pf, root_pos, root_rot, local = _walk_channels()
    world_at, n_keys = _reduced_world(body, root_pos, root_rot, local)
    tl = T.from_pose_fn(body, pf, frames=FRAMES)

    worst = 0.0
    for f in range(FRAMES + 1):
        w = world_at(f)
        for n, j in body.rig.joints.items():
            o, a = w[n]
            worst = max(worst, o.distance_to(tl.nodes[n].origin[f]))
            tip = o + Vec2(j.length, 0).rotated(a)
            worst = max(worst, tip.distance_to(tl.nodes[n].tip[f]))
    assert worst <= 2.0, f"reduced walk drifts {worst:.2f}px from the dense truth"


def test_reduced_walk_keeps_planted_feet_still_enough():
    """The IR has exactly zero slip; the sparse replay must stay under a pixel.

    Sub-pixel per-frame drift disappears under anti-aliasing. This bound is what
    the tightened contact-chain tolerance buys — remove TOL_DEG_LEG and this
    fails at ~1.24px/frame.
    """
    body, gait, pf, root_pos, root_rot, local = _walk_channels()
    world_at, _ = _reduced_world(body, root_pos, root_rot, local)

    worst = 0.0
    for c in ("shin_l", "shin_r"):
        j = body.rig.joints[c]
        prev = None
        for f in range(FRAMES + 1):
            o, a = world_at(f)[c]
            tip = o + Vec2(j.length, 0).rotated(a)
            if abs(tip.y - GROUND) <= 2.0:
                if prev is not None:
                    worst = max(worst, abs(tip.x - prev))
                prev = tip.x
            else:
                prev = None
    assert worst <= 0.8, f"planted foot drifts {worst:.2f}px/frame in the baked output"


# --------------------------------------------------------------------- budget
def test_walk_bakes_within_the_key_budget():
    """v1 spent (frames+1) × bones × 2 ≈ 2,500 keys on this. v2 must stay ~10× under."""
    from glaxnimate_ai.engine.bake import Scene, bake_rig
    from glaxnimate_ai.engine.session import SessionStore

    SessionStore()  # shared environment
    body = biped()
    gait = make_gait(body, "walk", cycle_frames=24)
    scene = Scene.create(960, 540, frames=FRAMES)
    stats = {}
    bake_rig(scene, body,
             lambda t: pose_at(body.rig, gait, t, ground_y=GROUND, body_x0=90),
             frames=FRAMES, stats=stats)
    assert stats["keyframes"] <= 260, f"budget blown: {stats['keyframes']} keys"
    assert stats["keyframes"] >= 30, "suspiciously few keys - is anything animated?"


# ------------------------------------------------- when a clip starts matters
def test_a_clip_bakes_on_the_frames_its_samples_claim():
    """The reducer counts keys from 0 because it only sees a list. Every
    `motion.*` generator starts at frame 0, so index and frame coincide and the
    lie stays hidden — until a clip starts partway through a film, when the
    whole thing silently plays at the top of the timeline instead. Measured
    before the fix: samples labelled 100..150 animated over frames 0..50.
    """
    from glaxnimate_ai.cartoon.motion import Sample
    from glaxnimate_ai.engine.session import SessionStore

    s = SessionStore().create(width=300, height=200, frames=200)
    smp = [Sample(f, Vec2(50 + (f - 100) * 2, 100)) for f in range(100, 151)]
    lay = s._add_moving_prop("filing", smp, name="probe")
    pos = list(lay.shapes)[0].transform.position

    assert pos.value_at_time(0.0).x == pytest.approx(50.0), "clip started early"
    assert pos.value_at_time(50.0).x == pytest.approx(50.0), "clip started early"
    assert pos.value_at_time(125.0).x == pytest.approx(100.0, abs=1.0)
    assert pos.value_at_time(150.0).x == pytest.approx(150.0, abs=1.0)
    assert pos.value_at_time(200.0).x == pytest.approx(150.0), "clip should hold"


def test_a_clip_that_starts_at_zero_is_unchanged():
    """The fix must not move anything that was already right."""
    from glaxnimate_ai.cartoon.motion import Sample
    from glaxnimate_ai.engine.session import SessionStore

    s = SessionStore().create(width=300, height=200, frames=50)
    smp = [Sample(f, Vec2(float(f) * 2, 100)) for f in range(51)]
    lay = s._add_moving_prop("filing", smp, name="probe")
    pos = list(lay.shapes)[0].transform.position
    assert pos.value_at_time(0.0).x == pytest.approx(0.0, abs=1.0)
    assert pos.value_at_time(50.0).x == pytest.approx(100.0, abs=1.0)


def test_non_contiguous_samples_are_refused_not_mistimed():
    from glaxnimate_ai.cartoon.motion import Sample
    from glaxnimate_ai.engine.session import SessionStore

    s = SessionStore().create(width=300, height=200, frames=200)
    gappy = [Sample(f, Vec2(float(f), 10)) for f in range(0, 100, 5)]  # every 5th
    with pytest.raises(ValueError, match="contiguous"):
        s._add_moving_prop("filing", gappy, name="gappy")


def test_props_scale_per_axis_at_draw_time():
    """Cutting a 240x72 bar into two 120x72 halves needs a per-axis scale, and
    `transform.scale` cannot deliver one: it is unwritable through these
    bindings (silently, for every type — see docs/glaxnimate-api.md). Draw-time
    scaling multiplies the coordinates before Qt sees them, so it works. This
    pins the route we actually depend on.
    """
    from glaxnimate_ai.cartoon.motion import Sample
    from glaxnimate_ai.engine.session import SessionStore
    from glaxnimate_ai.feedback.render import render_frame

    def bar_bbox(scale):
        s = SessionStore().create(width=600, height=300, frames=4, ground_y=290)
        smp = [Sample(f, Vec2(300, 150)) for f in range(5)]
        s._add_moving_prop("bar_magnet", smp, name="bar", scale=scale)
        im = render_frame(s.scene, 2).convert("RGB")
        px = im.load()
        xs, ys = [], []
        for y in range(im.height):
            for x in range(im.width):
                r, g, b = px[x, y]
                if r > 140 and b < 110 and g < 110:      # the red N half only
                    xs.append(x)
                    ys.append(y)
        return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1

    w1, h1 = bar_bbox(1.0)
    w2, h2 = bar_bbox((1.0, 2.0))       # twice as tall, same width
    assert w2 == pytest.approx(w1, abs=2), "per-axis scale changed the width"
    assert h2 == pytest.approx(h1 * 2, abs=3), "per-axis scale did not stretch y"

    w3, h3 = bar_bbox(2.0)              # uniform still works
    assert w3 == pytest.approx(w1 * 2, abs=3)
    assert h3 == pytest.approx(h1 * 2, abs=3)


def test_shot_refuses_to_silently_clobber_a_narrower_gate():
    """A broad prefix re-gating layers that already have their own shot is
    invisible when it happens and baffling afterwards: every mark that should
    appear on its own frame turns on at once. Caught twice by hand; now it is
    an error."""
    from glaxnimate_ai.engine.session import SessionStore

    s = SessionStore().create(width=200, height=200, frames=100)
    s._add_prop({"version": 1, "kind": "prop",
                 "shapes": [{"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5}]},
                x=50, layer_name="b1.tick")
    s._shot("b1.tick", 40, 100)
    with pytest.raises(ValueError, match="overwrite the gate"):
        s._shot("b1", 0, 100)
