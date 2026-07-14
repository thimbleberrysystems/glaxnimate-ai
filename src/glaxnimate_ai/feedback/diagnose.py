"""Tier 1 of the critic stack: *is it good?*

The linter says whether the animation is broken. This says whether it is any
good — and it does so with arithmetic, not pixels.

That is the central bet of this project. Animators do not judge motion by staring
at it; they use instruments, and every one of those instruments is a number:

* **Spacing chart** — the gaps between successive positions. Tight = slow, wide =
  fast. That *is* ease-in and ease-out. Perfectly even spacing is the unmistakable
  signature of linear interpolation, i.e. of an animation nobody timed.
* **Arc trace** — living things move in arcs; only machines move in straight lines
  and only broken rigs zigzag. Counting direction reversals turns "that hand looks
  wobbly" into "the hand reverses direction 4 times between frames 8 and 14".
* **Balance** — a figure whose centre of mass is outside its support would fall
  over. That is not an aesthetic judgement, it is statics.
* **Silhouette** — animators check the pose reads as a black shape. So can we:
  measure how much of the limbs is hidden inside the torso.

A full report costs ~500 tokens and names the frame and the magnitude. An image
costs ~1,400 and says "looks a bit off". Reach for this first.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from ..cartoon.geometry import Vec2
from ..cartoon.presets import Body
from ..cartoon.rig import Pose

__all__ = ["Finding", "Diagnosis", "spacing_chart", "arc_quality", "diagnose_rig"]


@dataclass(slots=True)
class Finding:
    metric: str
    value: float
    verdict: str  # "good" | "poor"
    detail: str

    def __str__(self) -> str:
        mark = "ok  " if self.verdict == "good" else "POOR"
        return f"  {mark} {self.metric:<18} {self.value:>7.3f}  {self.detail}"


@dataclass(slots=True)
class Diagnosis:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f.verdict == "good" for f in self.findings)

    def add(self, *a, **kw) -> None:
        self.findings.append(Finding(*a, **kw))

    def format(self) -> str:
        poor = [f for f in self.findings if f.verdict == "poor"]
        head = "all metrics in range" if not poor else f"{len(poor)} metric(s) out of range"
        return head + "\n" + "\n".join(str(f) for f in self.findings)


# ------------------------------------------------------------------- timing
def spacing_chart(points: list[Vec2]) -> dict[str, float]:
    """Per-frame step lengths — the animator's spacing chart, as numbers.

    `variation` is the coefficient of variation of the steps. Near zero means
    every step is the same size: constant velocity, no ease, dead-linear
    interpolation. It is the single most reliable tell of un-timed animation.
    """
    steps = [b.distance_to(a) for a, b in zip(points, points[1:], strict=False)]
    moving = [s for s in steps if s > 1e-9]
    if len(moving) < 2:
        return {"variation": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    mean = statistics.fmean(moving)
    return {
        "variation": statistics.pstdev(moving) / mean if mean else 0.0,
        "mean": mean,
        "min": min(moving),
        "max": max(moving),
    }


# --------------------------------------------------------------------- arcs
def arc_quality(points: list[Vec2]) -> dict[str, float]:
    """Direction reversals and turning consistency along a path.

    A clean arc turns steadily the same way. A zigzag flips direction repeatedly —
    that is what `reversals` counts, and it is exactly the artefact a hand-authored
    limb path picks up when the keys are not on an arc.
    """
    vels = [b - a for a, b in zip(points, points[1:], strict=False)]
    vels = [v for v in vels if v.length() > 1e-6]
    if len(vels) < 3:
        return {"reversals": 0.0, "turn_variation": 0.0}

    turns: list[float] = []
    for a, b in zip(vels, vels[1:], strict=False):
        cross = a.x * b.y - a.y * b.x
        dot = a.x * b.x + a.y * b.y
        turns.append(math.degrees(math.atan2(cross, dot)))

    reversals = sum(
        1 for a, b in zip(turns, turns[1:], strict=False)
        if a * b < 0 and abs(a) > 3.0 and abs(b) > 3.0
    )
    return {
        "reversals": float(reversals),
        "turn_variation": statistics.pstdev(turns) if len(turns) > 1 else 0.0,
    }


# ------------------------------------------------------------------ posture
def _centre_of_mass(frame, rig) -> Vec2:
    """Mass-weighted centroid.

    Uses `Joint.mass` where the rig declares it, falling back to bone length. The
    fallback is a poor model on purpose-built rigs: length-weighting gives the legs
    ~57% of a humanoid's mass and anchors the centre of mass so firmly that a figure
    bent double still measures as balanced. Declaring masses moves the legs to ~30%,
    which is roughly true of people, and the check starts working.
    """
    total, acc = 0.0, Vec2()
    for name, jf in frame.items():
        j = rig.joints[name]
        w = j.mass if j.mass > 0 else max(j.length, 1.0)
        mid = (jf.origin + jf.tip) * 0.5
        acc = acc + mid * w
        total += w
    return acc / total if total else Vec2()


def diagnose_rig(
    body: Body,
    pose_fn: Callable[[float], Pose],
    *,
    frames: int,
    ground_y: float,
    track: str | None = None,
    contact_tol: float = 1.5,
) -> Diagnosis:
    """Measure the craft: timing, arcs, balance, silhouette.

    `track` names the joint whose path is checked for arc quality — a hand, the
    head, whatever carries the eye. Defaults to the first non-contact tip.
    """
    rig = body.rig
    world = [rig.solve(pose_fn(float(f))) for f in range(frames + 1)]
    d = Diagnosis()

    # --- Spacing: does the body ease, or does it slide at constant speed?
    root_path = [w[rig.root_name].origin for w in world]
    sp = spacing_chart(root_path)
    # A constant-speed walk is correct (the body really does advance evenly), so
    # this is reported, not condemned. It matters for *gestures*, not locomotion.
    d.add(
        "body_spacing", sp["variation"], "good",
        f"step {sp['min']:.1f}-{sp['max']:.1f}px"
        + ("; constant speed (correct for locomotion)" if sp["variation"] < 0.02 else "; eased"),
    )

    # --- Arcs on a tracked extremity.
    if track is None:
        tips = [n for n in rig.joints if not rig.joints[n].contact and rig.joints[n].length > 0]
        track = tips[0] if tips else rig.root_name
    path = [w[track].tip for w in world]
    aq = arc_quality(path)
    d.add(
        "arc_reversals", aq["reversals"],
        "good" if aq["reversals"] <= 4 else "poor",
        f"{track} path; >4 means it zigzags instead of arcing",
    )

    # --- Balance. Statics, so it only means anything when the figure is *still*.
    #
    # Walking is controlled falling: on every step the centre of mass legitimately
    # travels out beyond the planted foot, and the next foot catches it. Applied to
    # a walk, this check fires on healthy motion — measured on our own clean walk it
    # reads 0.199 leg-lengths of overhang, against 0.297 for a figure bent double.
    # It barely separates them, so no threshold rescues it there.
    #
    # Held poses are different: a standing character with its mass outside its feet
    # would simply fall over. So evaluate only near-stationary frames, and say so
    # plainly when there are none rather than inventing a verdict.
    still_overhangs: list[float] = []
    # Start at 1: frame 0 has no predecessor, so its speed is unknowable. Counting
    # it as "stationary" would make a walking figure report a balance score under
    # the heading "while standing" — a number that is not wrong so much as about
    # something else entirely.
    for f in range(1, len(world)):
        w, prev = world[f], world[f - 1]
        moving = w[rig.root_name].origin.distance_to(prev[rig.root_name].origin)
        if moving > 0.75:
            continue  # locomoting: not a statics problem
        planted = [
            w[c].tip.x for c in rig.contacts if abs(w[c].tip.y - ground_y) <= contact_tol
        ]
        if not planted:
            continue  # airborne: nothing to balance on
        com = _centre_of_mass(w, rig)
        lo, hi = min(planted), max(planted)
        past = max(lo - com.x, com.x - hi, 0.0)
        still_overhangs.append(past / body.leg_length)

    if still_overhangs:
        worst = max(still_overhangs)
        # Calibrated, not guessed: a standing figure measures ~0.05, the same figure
        # bent 80 degrees at the waist measures ~0.22. 0.15 sits between them.
        d.add(
            "off_balance", worst, "good" if worst < 0.15 else "poor",
            "centre of mass overhangs the feet by this many leg-lengths while standing",
        )
    else:
        d.add(
            "off_balance", 0.0, "good",
            "n/a - the figure is in motion throughout; balance is a check for held poses",
        )

    # --- Silhouette: do the limbs read, or are they buried in the torso?
    hidden = 0
    for w in world:
        xs = [w[c].tip.x for c in rig.contacts]
        root_x = w[rig.root_name].origin.x
        spread = max(abs(x - root_x) for x in xs) if xs else 0.0
        if spread < body.leg_length * 0.12:
            hidden += 1
    frac = hidden / len(world)
    d.add(
        "silhouette_flat", frac, "good" if frac < 0.35 else "poor",
        "fraction of frames where the limbs collapse onto the body axis",
    )

    return d
