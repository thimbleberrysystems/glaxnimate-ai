"""Motion events: the moments a sound designer would spot on the exposure sheet.

This is why audio is not a bolt-on here. The Timeline IR already computes, for
the linter, exactly the physical facts a foley artist works from — when a foot
plants, when a ball meets the ground, when a body leaves it. Extracting those as
events means sound *placement* is derived from the animation, not guessed by the
model: the same walk that lints clean gets its footsteps for free, on the right
frames, panned to where the character actually is.

All pure data in, pure data out — no Qt, no audio libraries.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cartoon.timeline import Timeline

__all__ = ["MotionEvent", "plant_onsets", "object_hits", "airborne_spans",
           "expression_stings"]


@dataclass(slots=True)
class MotionEvent:
    frame: int
    kind: str      # "plant" | "hit" | "launch" | "land" | "expression"
    #: horizontal position of the event, for panning
    x: float
    detail: str = ""


def plant_onsets(tl: Timeline, *, ground_y: float, tol: float = 1.5) -> list[MotionEvent]:
    """A foot striking the ground: airborne on frame f-1, planted on frame f.

    Same proximity rule as the linter's slip check, so the sounds land exactly
    where the linter says the feet do. Frame 0 is never an onset — a character
    that *starts* standing did not just stomp.
    """
    out: list[MotionEvent] = []
    for name in tl.contacts:
        t = tl.nodes[name]
        if not t.check_slip:
            continue  # a ball's contacts are hits, handled below
        prev_planted = True  # suppresses a spurious onset at frame 0
        for f, p in enumerate(t.tip):
            planted = abs(p.y - ground_y) <= tol
            if planted and not prev_planted:
                out.append(MotionEvent(f, "plant", p.x, name))
            prev_planted = planted
    return sorted(out, key=lambda e: e.frame)


def object_hits(samples, *, radius: float, ground_y: float,
                name: str = "object") -> list[MotionEvent]:
    """A free object (ball) striking the ground.

    Not a simple proximity threshold, for two measured reasons: the squash at
    impact compresses scale_y, lifting the sampled bottom *off* the ground line
    by design; and the true contact instant falls between integer frames, so the
    sampled minimum height never quite reaches zero. A hit is therefore a
    **local minimum of height above ground, close to it** — descent reverses to
    ascent within an impact zone scaled by the object's own size.
    """
    rows = list(samples)
    if len(rows) < 3:
        return []
    heights = [ground_y - (s.pos.y + radius * s.scale.y) for s in rows]
    zone = max(radius * 0.75, 4.0)

    out: list[MotionEvent] = []
    for i in range(1, len(rows) - 1):
        local_min = heights[i] <= heights[i - 1] and heights[i] < heights[i + 1]
        if local_min and heights[i] < zone:
            out.append(MotionEvent(rows[i].frame, "hit", rows[i].pos.x, name))
    # coming to rest on the final frame is a landing too
    if heights[-1] < zone and heights[-2] > heights[-1]:
        out.append(MotionEvent(rows[-1].frame, "hit", rows[-1].pos.x, name))
    return out


def airborne_spans(tl: Timeline, *, ground_y: float, tol: float = 1.5,
                   min_frames: int = 6) -> list[MotionEvent]:
    """Launch and landing of a genuinely airborne body (a jump).

    `min_frames` is the discriminator: a run's flight phase lasts a couple of
    frames per cycle and whooshing every stride would be noise, while a jump
    hangs for dozens. Only spans at least this long earn a launch and a land.
    """
    slip_contacts = [n for n in tl.contacts if tl.nodes[n].check_slip]
    if not slip_contacts:
        return []

    n_frames = min(len(tl.nodes[c].tip) for c in slip_contacts)
    grounded = [
        any(abs(tl.nodes[c].tip[f].y - ground_y) <= tol for c in slip_contacts)
        for f in range(n_frames)
    ]

    out: list[MotionEvent] = []
    start = None
    for f in range(1, n_frames):
        if grounded[f - 1] and not grounded[f]:
            start = f
        elif not grounded[f - 1] and grounded[f] and start is not None:
            if f - start >= min_frames:
                root = tl.nodes.get(tl.root)
                x0 = root.origin[start].x if root else 0.0
                x1 = root.origin[f].x if root else 0.0
                out.append(MotionEvent(start, "launch", x0))
                out.append(MotionEvent(f, "land", x1))
            start = None
    return out


def expression_stings(doc: dict) -> list[MotionEvent]:
    """One sting per expression swap — the classic 'boing-eyes' pop. Frame-0
    defaults are the character's starting face, not a moment."""
    out: list[MotionEvent] = []
    for ch in doc.get("characters", []):
        x = 0.0
        poses = ch.get("poses") or []
        for frame, att in ch.get("expressions", []):
            if frame <= 0:
                continue
            f = min(int(frame), len(poses) - 1)
            if poses:
                x = poses[f]["root"][0]
            out.append(MotionEvent(int(frame), "expression", x, att))
    return sorted(out, key=lambda e: e.frame)
