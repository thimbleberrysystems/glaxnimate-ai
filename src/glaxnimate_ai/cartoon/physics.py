"""A real ragdoll: Verlet point-masses + rigid bone constraints, gravity, ground.

The other actions are hand-shaped pose functions; a ragdoll is *simulated*. Each
joint becomes a point mass, each bone a rigid distance constraint between two of
them; gravity pulls, the ground stops them, constraint relaxation keeps the bones
their true length. The floppy, tumbling, settling motion that a scripted arc can
only fake falls straight out of the maths — a figure thrown, hit, or dragged by a
cursor flails correctly because nothing is keyframed.

Pure Python (no Qt): positions in, a `pose_fn(t) -> Pose` out, so a ragdoll drops
into `bake_rig`, the linter and the timeline exactly like any other action.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .geometry import Vec2
from .presets import Body
from .rig import Pose

__all__ = ["ragdoll"]

PoseFn = Callable[[float], Pose]


def _skeleton(body: Body, pose0: Pose):
    """Point set + bone constraints derived from an initial solved pose.

    A point per joint tip plus the root origin ('@root'); a constraint per bone
    with its true length. `attach[name]` is the point each bone hangs from, which is
    also what the angle-recovery reads to turn simulated positions back into a Pose.
    """
    rig = body.rig
    root = rig.root_name
    frames = rig.solve(pose0)
    pts: dict[str, Vec2] = {"@root": frames[root].origin}
    cons: list[tuple[str, str, float]] = []
    attach: dict[str, str] = {}
    for name, j in rig.joints.items():
        if name == root or j.length == 0:
            continue
        pts[name] = frames[name].tip
        # a bone hangs from the parent's tip, except joints that branch off the root
        # (spine, thighs, tail) which pivot at the root origin.
        aid = "@root" if j.parent == root else j.parent
        attach[name] = aid
        cons.append((aid, name, float(j.length)))
    return pts, cons, attach


def _to_pose(rig, cur: dict[str, Vec2], attach: dict[str, str]) -> Pose:
    """Turn simulated world positions back into a Pose FK can reproduce."""
    root = rig.root_name
    rp = cur["@root"]
    pose = Pose(root=Vec2(rp.x, rp.y), root_angle=0.0)
    world: dict[str, float] = {}
    for name in rig._order:
        j = rig.joints[name]
        if name == root or j.length == 0:
            continue
        a, b = cur[attach[name]], cur[name]
        wa = math.degrees(math.atan2(b.y - a.y, b.x - a.x))
        world[name] = wa
        parent_world = 0.0 if j.parent == root else world.get(j.parent, 0.0)
        pose.angles[name] = wa - parent_world - j.rest_angle
    return pose


def ragdoll(body: Body, pose0: Pose, *, ground_y: float, frames: int,
            launch: tuple[float, float] = (0.0, 0.0), spin: float = 0.0,
            pin: str | None = None, pin_path: Callable[[int], Vec2] | None = None,
            gravity: float = 1.5, damping: float = 0.99, friction: float = 0.6,
            iters: int = 12) -> PoseFn:
    """Simulate a ragdoll and return its `pose_fn`.

    `pose0` is the pose at the moment control is lost; `launch` gives it an initial
    velocity (a throw or a hit), `spin` an angular one. `pin` + `pin_path` nails one
    point (a joint name, or '@root') to a moving target each frame — a figure held
    and dragged by a cursor, dangling from the grab. Gravity, ground collision and
    friction do the rest.
    """
    rig = body.rig
    pts, cons, attach = _skeleton(body, pose0)
    ids = list(pts)
    cur = {k: Vec2(v.x, v.y) for k, v in pts.items()}
    root0 = cur["@root"]
    per = 1.0 / max(frames, 1)
    prev: dict[str, Vec2] = {}
    for k, v in cur.items():   # prev = pos - per-step velocity (launch + spin about root)
        rx, ry = v.x - root0.x, v.y - root0.y
        sx, sy = -ry * math.radians(spin) * per, rx * math.radians(spin) * per
        prev[k] = Vec2(v.x - launch[0] * per - sx, v.y - launch[1] * per - sy)

    poses: list[Pose] = []
    for f in range(frames + 1):
        for k in ids:                              # Verlet integration
            if pin is not None and k == pin and pin_path is not None:
                p = pin_path(f)
                cur[k] = Vec2(p.x, p.y)
                prev[k] = Vec2(p.x, p.y)
                continue
            x, xp = cur[k], prev[k]
            nx = Vec2(x.x + (x.x - xp.x) * damping,
                      x.y + (x.y - xp.y) * damping + gravity)
            prev[k] = x
            cur[k] = nx
        for _ in range(iters):                     # satisfy bone + ground constraints
            for a, b, L in cons:
                pa, pb = cur[a], cur[b]
                dx, dy = pb.x - pa.x, pb.y - pa.y
                dist = math.hypot(dx, dy) or 1e-6
                diff = (dist - L) / dist
                pa_pin = pin is not None and a == pin
                pb_pin = pin is not None and b == pin
                if pa_pin and pb_pin:
                    continue
                if pa_pin:
                    cur[b] = Vec2(pb.x - dx * diff, pb.y - dy * diff)
                elif pb_pin:
                    cur[a] = Vec2(pa.x + dx * diff, pa.y + dy * diff)
                else:
                    cur[a] = Vec2(pa.x + dx * diff * 0.5, pa.y + dy * diff * 0.5)
                    cur[b] = Vec2(pb.x - dx * diff * 0.5, pb.y - dy * diff * 0.5)
            for k in ids:                          # ground plane
                if cur[k].y > ground_y:
                    vx = cur[k].x - prev[k].x
                    cur[k] = Vec2(cur[k].x - vx * friction, ground_y)
        poses.append(_to_pose(rig, cur, attach))

    def pose_fn(t: float) -> Pose:
        return poses[min(max(int(t), 0), len(poses) - 1)]

    return pose_fn
