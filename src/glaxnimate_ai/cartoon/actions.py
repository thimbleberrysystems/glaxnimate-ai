"""Actions: the principles composed into things a character *does*.

Gaits handle locomotion. This handles the rest of the acting vocabulary — a jump,
an idle, a wave, a loose cape trailing behind — by composing the animation
principles the library already has.

Two of the twelve principles had the maths but nothing that used them, which is
the same as not having them:

* **Anticipation** — you wind up before you go. `jump` crouches before it launches.
* **Follow-through / overlapping action** — loose parts (a tail, a cape, an ear,
  hair) keep moving after the body stops, and lag behind while it moves. `trail`
  drives a chain from a delayed, damped copy of its own base, so it swings and
  settles instead of moving rigidly with the body.

Everything here returns a `pose_fn(t) -> Pose`, the same currency `bake_rig` and
the linter already speak, so an action drops into the pipeline exactly like a gait.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .gait import Gait, pose_at
from .geometry import Vec2, clamp
from .presets import Body
from .principles import anticipate, ease_in, ease_in_out, ease_out, overshoot
from .rig import Pose, solve_two_bone

__all__ = [
    "jump", "idle", "wave", "trail", "sequence", "locomote",
    # combat / stunt beats — a stick figure that fights, not just walks
    "punch", "kick", "dash", "flip", "swing", "block", "knockback", "land",
    "line_of_action",
    # everyday acting verbs
    "celebrate", "fall", "sit", "tap", "pushup", "pedal", "fly",
    # snap-timing toolkit — the difference between snappy and floaty
    "hold", "hitstop", "retime",
]

PoseFn = Callable[[float], Pose]

#: The near-side and far-side arm chains of a biped, for strikes and guards.
_NEAR_ARM = ("arm_upper", "arm_lower")
_FAR_ARM = ("arm_upper_far", "arm_lower_far")


def _ik_foot_to(body: Body, pose: Pose, frames_solved, upper_name: str,
                lower_name: str, target: Vec2) -> None:
    """Place one two-bone leg's foot on an arbitrary world `target`."""
    upper = body.rig.joints[upper_name]
    lower = body.rig.joints[lower_name]
    hip = frames_solved[upper_name].origin
    wu, ll = solve_two_bone(hip, target, upper.length, lower.length, bend_positive=True)
    parent = frames_solved[upper.parent].angle if upper.parent else pose.root_angle
    pose.angles[upper_name] = wu - parent - upper.rest_angle
    pose.angles[lower_name] = ll - lower.rest_angle


def _ground_one_leg(body: Body, pose: Pose, frames_solved, upper_name: str,
                    lower_name: str, foot_y: float) -> None:
    """IK a single two-bone leg down to a fixed ground height, foot under the hip."""
    hip = frames_solved[upper_name].origin
    _ik_foot_to(body, pose, frames_solved, upper_name, lower_name,
                Vec2(hip.x, foot_y))


def _lift_legs(body: Body, pose: Pose, ground_y: float, *, lift: float = 26.0,
               spread: float = 0.0) -> None:
    """Pose both contact legs with the feet a safe margin *above* the ground — for a
    beat's airborne / translating phase.

    Feet clearly off the ground are neither planted (so a moving body can't skate)
    nor below it (so nothing penetrates): the two faults the linter polices during a
    dash or a knockback. `spread` splits the legs — the lead foot forward, the trail
    foot back — which reads as a running lunge rather than a symmetric tuck.
    """
    frames = body.rig.solve(pose)
    for i, (upper_name, lower_name) in enumerate(_leg_pairs(body)):
        hip = frames[upper_name].origin
        dx = spread if i == 0 else -spread
        _ik_foot_to(body, pose, frames, upper_name, lower_name,
                    Vec2(hip.x + dx, ground_y - lift))


def _ground_legs(body: Body, pose: Pose, frames_solved, foot_y: float) -> None:
    """IK both legs of a biped down to a fixed ground height. Used by standing actions."""
    for upper_name, lower_name in _leg_pairs(body):
        _ground_one_leg(body, pose, frames_solved, upper_name, lower_name, foot_y)


def _leg_pairs(body: Body) -> list[tuple[str, str]]:
    """(upper, lower) for each contact leg — works for any rig, not just the biped."""
    pairs = []
    for name, j in body.rig.joints.items():
        if j.contact and j.parent is not None:
            pairs.append((j.parent, name))
    return pairs


def jump(
    body: Body,
    *,
    ground_y: float,
    x: float = 0.0,
    height: float = 140.0,
    distance: float = 0.0,
    frames: int = 36,
    anticip: float = 0.22,
    land: float = 0.16,
) -> PoseFn:
    """A jump: anticipation, launch, an arc through the air, a squash landing.

    This is the canonical demonstration of the principles working together — three
    of the twelve in one move (anticipation, arcs, squash-and-stretch) — and the
    single most-requested thing a stick figure should be able to do.

    The squash is done honestly for a cut-out rig: the *legs bend*. A deep crouch
    reads as a squash and full extension at launch reads as a stretch, which is how
    real cut-out animation fakes volume without a deformable mesh. `anticip` and
    `land` are the fractions of the cycle spent winding up and absorbing impact.
    """
    hip = body.hip_height
    launch = anticip
    touchdown = 1.0 - land

    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)

        if p < launch:
            # Anticipation: sink into a crouch. `anticipate` dips below zero first,
            # which is the tiny counter-settle before the spring.
            s = anticipate(p / launch)
            rise = -0.28 * hip * s
            gx = x
        elif p < touchdown:
            # Airborne: a parabolic arc. Peak height at the midpoint of the flight.
            s = (p - launch) / (touchdown - launch)
            rise = height * 4.0 * s * (1.0 - s) + 0.02 * hip
            gx = x + distance * ease_in_out(s)
        else:
            # Landing: absorb into a crouch, then ease back up to standing.
            s = (p - touchdown) / (1.0 - touchdown)
            rise = -0.24 * hip * math.sin(math.pi * s) * (1.0 - ease_out(s) * 0.4)
            gx = x + distance

        pose = Pose(root=Vec2(gx, ground_y - hip - rise), root_angle=0.0)

        # Secondary: arms lift through the launch and reach up at the apex — a real
        # jump throws its arms up.
        arm_phase = clamp((p - launch * 0.5) / max(touchdown - launch * 0.5, 1e-6), 0, 1)
        arm = -60.0 * math.sin(math.pi * arm_phase)
        for j in ("arm_upper", "arm_upper_far"):
            if j in body.rig.joints:
                pose.angles[j] = arm

        frames_solved = body.rig.solve(pose)

        airborne = launch <= p < touchdown and rise > 0.12 * hip
        if airborne:
            # Tuck the feet up under the body.
            tuck = min((rise) / max(height, 1e-6), 1.0)
            for upper_name, lower_name in _leg_pairs(body):
                pose.angles[upper_name] = -40.0 * tuck
                pose.angles[lower_name] = 80.0 * tuck
        else:
            _ground_legs(body, pose, frames_solved, ground_y)

        return pose

    return pose_fn


def idle(body: Body, *, ground_y: float, x: float = 0.0, cycle_frames: float = 48.0) -> PoseFn:
    """A living hold: a slow breathing rise and fall, feet planted.

    A character that is perfectly still reads as dead. The smallest amount of life —
    a breath — is the difference between a pause and a freeze.
    """
    hip = body.hip_height
    amp = 0.02 * hip

    def pose_fn(t: float) -> Pose:
        rise = amp * math.sin(2.0 * math.pi * t / cycle_frames)
        pose = Pose(root=Vec2(x, ground_y - hip - rise), root_angle=0.0)
        if "spine" in body.rig.joints:
            pose.angles["spine"] = 1.5 * math.sin(2.0 * math.pi * t / cycle_frames)
        frames_solved = body.rig.solve(pose)
        _ground_legs(body, pose, frames_solved, ground_y)
        return pose

    return pose_fn


def wave(
    body: Body,
    *,
    ground_y: float,
    x: float = 0.0,
    arm: str = "arm_upper",
    forearm: str = "arm_lower",
    cycles: float = 3.0,
    frames: int = 48,
) -> PoseFn:
    """Stand and wave: the near arm raises and the forearm oscillates.

    Anticipation on the way up (the arm swings back slightly first), then a
    sinusoidal wave — a gesture, i.e. a secondary action the character does on
    purpose, as distinct from an involuntary follow-through.
    """
    base = idle(body, ground_y=ground_y, x=x)
    have_arm = arm in body.rig.joints and forearm in body.rig.joints

    def pose_fn(t: float) -> Pose:
        pose = base(t)
        if have_arm:
            p = clamp(t / frames, 0.0, 1.0)
            up = clamp(p / 0.25, 0, 1)  # arm reaches full height in the first quarter
            raise_amt = ease_out(up) - 0.12 * math.sin(math.pi * up)  # tiny anticipation dip
            pose.angles[arm] = -150.0 * raise_amt
            if p > 0.2:
                pose.angles[forearm] = 30.0 * math.sin(2.0 * math.pi * cycles * (p - 0.2))
        return pose

    return pose_fn


def trail(
    pose_fn: PoseFn,
    body: Body,
    chain: list[str],
    *,
    lag: float = 2.5,
    damping: float = 0.7,
    swing: float = 26.0,
) -> PoseFn:
    """Wrap a pose function so a loose chain (tail, cape, ear, hair) follows through.

    Follow-through and overlapping action: when the body moves, the tip of a loose
    chain lags; when the body stops, the chain keeps swinging and settles. Each
    joint down the chain lags a little more than the one above it (that stagger is
    the *overlapping* part), and the swing is driven by how fast the base is moving,
    so it is silent when the character is still and whips when it darts.

    `lag` is how many frames the motion trails; `damping` how quickly it settles;
    `swing` the maximum deflection in degrees.
    """
    def wrapped(t: float) -> Pose:
        pose = pose_fn(t)
        base_now = pose.root.x
        base_past = pose_fn(max(t - lag, 0.0)).root.x
        velocity = base_now - base_past  # how fast the body moved over the lag window

        for depth, joint in enumerate(chain):
            if joint not in body.rig.joints:
                continue
            # Each link trails the one above it: an extra half-frame of lag per link.
            local_lag = lag * (1.0 + depth * 0.5)
            past = pose_fn(max(t - local_lag, 0.0)).root.x
            v = base_now - past
            # Deflect opposite to travel (the tail streams *behind* the motion),
            # damped by depth so the tip moves most.
            deflect = clamp(-v * 0.4, -swing, swing) * (damping ** depth) * (1.0 + depth * 0.25)
            pose.angles[joint] = pose.angles.get(joint, 0.0) + deflect
            _ = velocity  # base velocity kept for callers that want the read
        return pose

    return wrapped


# =============================================================== combat / stunt
# Viral stick figures fight and stunt; they do not merely walk. These beats are
# the acting vocabulary for it — each an anticipation -> fast strike -> settle,
# composed from the same principles as `jump`, and each a `pose_fn(t)` that drops
# into `bake_rig`, the linter and the event system unchanged. `facing` is +1 for a
# figure facing right (+x), -1 for left; every strike reads in the facing direction.


def _set(pose: Pose, body: Body, joint: str, angle: float) -> None:
    """Set a joint angle only if the rig actually has that joint (stays generic)."""
    if joint in body.rig.joints:
        pose.angles[joint] = angle


def _stance(body: Body, ground_y: float, x: float, *, lean: float = 0.0,
            crouch: float = 0.0, lift: float = 0.0, root_angle: float = 0.0,
            plant: bool = True) -> Pose:
    """A grounded fighting stance: the base every combat beat poses on top of.

    `lean` tilts the spine (the line of action of a committed strike), `crouch`
    lowers the hips as a fraction of hip height (a coil before a launch), `lift`
    raises the whole body off the ground in pixels (an airborne beat). Feet re-plant
    to the ground unless the beat is airborne (`plant=False`), which keeps a punch's
    stance honest for the linter.
    """
    hip = body.hip_height
    pose = Pose(root=Vec2(x, ground_y - hip * (1.0 - crouch) - lift),
                root_angle=root_angle)
    _set(pose, body, "spine", lean)
    _set(pose, body, "head", -0.3 * lean)  # the head counters the lean a little
    if plant:
        frames = body.rig.solve(pose)
        _ground_legs(body, pose, frames, ground_y)
    return pose


def line_of_action(pose_fn: PoseFn, body: Body, *, curve: float = 1.0) -> PoseFn:
    """Bias a beat along a single curved line of action — the craft fundamental that
    makes a pose *read* as one shape instead of a bag of limbs.

    It amplifies the spine lean and bends the head and near arm to follow the same
    C-curve, so a lunge becomes a clean arc from planted foot to striking fist.
    `curve` scales the exaggeration; 0 leaves the beat untouched.
    """
    def wrapped(t: float) -> Pose:
        pose = pose_fn(t)
        lean = pose.angles.get("spine", 0.0)
        if "spine" in body.rig.joints:
            pose.angles["spine"] = lean * (1.0 + 0.35 * curve)
        if "head" in body.rig.joints:
            pose.angles["head"] = pose.angles.get("head", 0.0) - 0.15 * curve * lean
        return pose
    return wrapped


def _strike_env(p: float, wind: float, hit: float) -> tuple[str, float]:
    """The universal strike envelope, as (phase, s in [0,1]).

    Three beats: wind up slowly (anticipation), snap through fast (the strike lands
    at `hit`), then settle with recoil. `wind` and `hit` are cycle fractions.
    """
    if p < wind:
        return "wind", ease_out(p / max(wind, 1e-6))
    if p < hit:
        return "strike", ease_in((p - wind) / max(hit - wind, 1e-6), power=2.2)
    return "settle", overshoot((p - hit) / max(1.0 - hit, 1e-6))


def punch(body: Body, *, ground_y: float, x: float = 0.0, facing: float = 1.0,
          frames: int = 16, arm: tuple[str, str] = _NEAR_ARM,
          far: tuple[str, str] = _FAR_ARM) -> PoseFn:
    """A straight punch: cock the fist, snap it to full extension, recoil.

    The near arm drives out to horizontal while the far arm and the spine counter-
    rotate into it — the whole body behind the fist, which is what gives a stick
    line weight. The fist reaches full reach right at the `hit` frame so an effects
    or sound cue placed there lands on contact.
    """
    up, lo = arm
    fup, flo = far

    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        phase, s = _strike_env(p, wind=0.34, hit=0.60)
        if phase == "wind":
            reach, lean = -0.35 * s, -8.0 * s
        elif phase == "strike":
            reach, lean = -0.35 + 1.35 * s, -8.0 + 26.0 * s
        else:
            reach, lean = 1.0 - 0.22 * s, 18.0 - 6.0 * s

        pose = _stance(body, ground_y, x, lean=lean * facing)
        ext = clamp(reach, 0.0, 1.0)
        wound = clamp(-reach, 0.0, 1.0)
        # near arm: -95deg local is straight forward (horizontal); cock it back and
        # bend the elbow when wound, straighten to a jab at full reach.
        _set(pose, body, up, facing * (-58.0 - 37.0 * reach))
        _set(pose, body, lo, facing * (75.0 * wound + 8.0 * ext))
        # far arm pulls back as a counterweight, elbow bent by the chamber.
        _set(pose, body, fup, facing * (-35.0 + 15.0 * reach))
        _set(pose, body, flo, facing * 55.0)
        return pose

    return pose_fn


def kick(body: Body, *, ground_y: float, x: float = 0.0, facing: float = 1.0,
         frames: int = 18, leg: tuple[str, str] | None = None) -> PoseFn:
    """A front kick: chamber the near leg, snap it out horizontal, retract.

    The support leg stays planted (the linter still polices it); the kicking leg
    leaves the ground on purpose, so it is exempt from the ground solve. Arms throw
    back for balance — a real kick is a whole-body counter-rotation.
    """
    pairs = _leg_pairs(body)
    kick_pair = leg or (pairs[0] if pairs else ("thigh_l", "shin_l"))
    kthigh, kshin = kick_pair
    support = [pr for pr in pairs if pr != kick_pair]
    ll = body.rig.joints[kthigh].length + body.rig.joints[kshin].length

    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        phase, s = _strike_env(p, wind=0.30, hit=0.58)
        if phase == "wind":
            reach, lean = 0.0, 5.0 * s
        elif phase == "strike":
            reach, lean = s, 5.0 - 16.0 * s
        else:
            reach, lean = 1.0 - s, -11.0 + 6.0 * s
        # `air` lifts the kicking foot away from the ground in the middle of the beat
        # and sets it back down at both ends, so it never plants-and-slides nor digs
        # below the floor — it is grounded only while the body is still.
        air = math.sin(math.pi * clamp(p, 0.0, 1.0))

        pose = Pose(root=Vec2(x, ground_y - body.hip_height), root_angle=0.0)
        _set(pose, body, "spine", lean * facing)
        frames_solved = body.rig.solve(pose)
        for up_n, lo_n in support:  # the support leg is always planted
            _ground_one_leg(body, pose, frames_solved, up_n, lo_n, ground_y)

        hip = frames_solved[kthigh].origin
        # the kicking foot: chambered (up, near the body) -> struck (forward, hip
        # height), lifted by `air`, blended down to a plant at the ends.
        tx = hip.x + ll * facing * (0.12 + 0.80 * reach)
        ty = ground_y - ll * (0.55 - 0.08 * reach)
        target = Vec2(hip.x + (tx - hip.x) * air, ground_y + (ty - ground_y) * air)
        _ik_foot_to(body, pose, frames_solved, kthigh, kshin, target)

        _set(pose, body, "arm_upper", facing * (40.0 * reach))
        _set(pose, body, "arm_upper_far", facing * (30.0 * reach))
        return pose

    return pose_fn


def dash(body: Body, *, ground_y: float, x0: float = 0.0, x1: float = 160.0,
         facing: float | None = None, frames: int = 14) -> PoseFn:
    """A ground dash: explode from a coil into a low, stretched forward lunge.

    The signature stick-fight entrance. Deep anticipation coil, then a fast
    ease-in translation with the body raked forward over the lead foot and arms
    swept back — a strong single line of action. Pairs with speed-line fx (WS2).
    """
    face = facing if facing is not None else (1.0 if x1 >= x0 else -1.0)

    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        if p < 0.24:                                   # coil — planted, gathering
            s = ease_out(p / 0.24)
            pose = _stance(body, ground_y, x0, lean=8.0 * s * face, crouch=0.28 * s)
        elif p < 0.82:                                 # drive — a low airborne lunge
            s = ease_in((p - 0.24) / 0.58, power=1.7)
            gx = x0 + (x1 - x0) * s
            pose = _stance(body, ground_y, gx, lean=(8.0 - 34.0 * s) * face,
                           crouch=0.12, plant=False)
            # feet leave the ground during the drive: a dash is a slide THROUGH the
            # air, not a skate — planted feet dragged across the floor is the exact
            # contact-slip the linter flags.
            _lift_legs(body, pose, ground_y, lift=22.0, spread=36.0 * face)
        else:                                          # arrive — plant and absorb
            s = ease_out((p - 0.82) / 0.18)
            pose = _stance(body, ground_y, x1, lean=(-26.0 * (1.0 - s)) * face,
                           crouch=0.20 * (1.0 - s))
        _set(pose, body, "arm_upper", face * -45.0)
        _set(pose, body, "arm_upper_far", face * -55.0)
        return pose

    return pose_fn


def flip(body: Body, *, ground_y: float, x: float = 0.0, distance: float = 90.0,
         height: float = 150.0, facing: float = 1.0, frames: int = 26) -> PoseFn:
    """A tucked backflip/frontflip: launch, a full airborne rotation in a tuck, land.

    Rotation is the root turning a full 360 * facing; the tuck pulls the limbs in so
    the spin reads. Anticipation crouch and a squashing landing bracket the spin.
    """
    hip = body.hip_height
    launch, touchdown = 0.2, 0.82

    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        if p < launch:
            s = anticipate(p / launch)
            rise, gx, spin = -0.3 * hip * s, x, 0.0
        elif p < touchdown:
            s = (p - launch) / (touchdown - launch)
            rise = height * 4.0 * s * (1.0 - s)
            gx = x + distance * ease_in_out(s)
            spin = -360.0 * facing * ease_in_out(s)
        else:
            s = (p - touchdown) / (1.0 - touchdown)
            rise = -0.26 * hip * math.sin(math.pi * s)
            gx, spin = x + distance, 0.0

        airborne = launch <= p < touchdown
        pose = _stance(body, ground_y, gx, lift=rise, root_angle=spin,
                       crouch=0.3 if (p < launch or p >= touchdown) else 0.0,
                       plant=not airborne)
        if airborne:  # tuck: knees and elbows pulled in so the rotation reads
            for up_n, lo_n in _leg_pairs(body):
                _set(pose, body, up_n, -55.0)
                _set(pose, body, lo_n, 105.0)
            _set(pose, body, "arm_upper", -120.0)
            _set(pose, body, "arm_upper_far", -120.0)
        return pose

    return pose_fn


def swing(body: Body, *, ground_y: float, x: float = 0.0, facing: float = 1.0,
          frames: int = 18, arm: tuple[str, str] = _NEAR_ARM) -> PoseFn:
    """An overhead swing (a sword/club arc): raise high behind, sweep down and across.

    The arm travels a big arc from up-and-back to down-and-front; the spine whips
    over the top. The blade/prop is a held prop parented to the hand (WS4); this is
    the body that carries it.
    """
    up, lo = arm

    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        phase, s = _strike_env(p, wind=0.36, hit=0.62)
        # arc_pos: 0 = raised overhead & back, 1 = swept down & to the front.
        if phase == "wind":
            arc_pos, lean = 0.15 * (1.0 - s), -12.0 * s     # raise up and back
        elif phase == "strike":
            arc_pos, lean = s, -12.0 + 34.0 * s             # sweep over the top
        else:
            arc_pos, lean = 1.0 - 0.15 * s, 22.0 - 6.0 * s
        pose = _stance(body, ground_y, x, lean=lean * facing)
        # arm_upper sweeps from ~ -160deg (overhead, slightly back) to ~ -25deg
        # (down and forward); the elbow trails, straightening through the strike.
        _set(pose, body, up, facing * (-160.0 + 135.0 * arc_pos))
        _set(pose, body, lo, facing * (50.0 * (1.0 - arc_pos)))
        return pose

    return pose_fn


def block(body: Body, *, ground_y: float, x: float = 0.0, facing: float = 1.0,
          frames: int = 12) -> PoseFn:
    """A guard: snap both forearms up across the body and brace, with a small flinch.

    A defensive hold rather than a strike — quick raise, then a braced settle. Used
    as the reaction that is *not* a knockback: the blow is caught, not taken.
    """
    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        raise_amt = ease_out(clamp(p / 0.4, 0.0, 1.0))
        flinch = math.sin(math.pi * clamp((p - 0.4) / 0.6, 0.0, 1.0)) * 0.15
        pose = _stance(body, ground_y, x, lean=-4.0 * facing * raise_amt,
                       crouch=0.06 * raise_amt)
        _set(pose, body, "arm_upper", facing * (-70.0 * raise_amt))
        _set(pose, body, "arm_lower", facing * (110.0 * raise_amt - 40.0 * flinch))
        _set(pose, body, "arm_upper_far", facing * (-60.0 * raise_amt))
        _set(pose, body, "arm_lower_far", facing * (120.0 * raise_amt))
        return pose

    return pose_fn


def knockback(body: Body, *, ground_y: float, x: float = 0.0, facing: float = 1.0,
              distance: float = 70.0, frames: int = 16) -> PoseFn:
    """Take a hit: snap back off the blow, stagger, recover.

    The reaction half of a two-figure exchange (WS4). The torso jackknifes away from
    the strike, the figure slides back `distance` (against `facing`, i.e. away from
    the attacker), then straightens. Feet stay planted — a stagger, not a fall.
    """
    settle = x - facing * distance

    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        if p < 0.4:                                    # impact: flung back off the feet
            s = ease_out(p / 0.4)
            gx, lean = x + (settle - x) * s, 26.0 * s
            pose = _stance(body, ground_y, gx, lean=lean * facing, crouch=0.08,
                           plant=False)
            # knocked off the feet: they leave the ground for the fling, so the
            # backward slide never drags a planted foot across the floor.
            _lift_legs(body, pose, ground_y, lift=16.0, spread=-22.0 * facing)
        else:                                          # recover: planted, straighten up
            s = ease_in_out((p - 0.4) / 0.6)
            pose = _stance(body, ground_y, settle, lean=26.0 * (1.0 - s) * facing,
                           crouch=0.05 * (1.0 - s))
        back = 1.0 if p < 0.4 else (1.0 - s)
        _set(pose, body, "arm_upper", facing * (-30.0 * back))
        _set(pose, body, "arm_upper_far", facing * (-45.0 * back))
        return pose

    return pose_fn


def land(body: Body, *, ground_y: float, x: float = 0.0, frames: int = 14,
         impact: float = 0.32) -> PoseFn:
    """A hero landing: hit in a deep crouch (one hand down), absorb, rise to stand.

    The settle beat that ends a `flip` or `jump` with weight. Deepest crouch on the
    contact frame (`impact`), then ease up — the squash that reads as force absorbed.
    """
    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        if p < impact:
            s = ease_in(p / max(impact, 1e-6))
            crouch = 0.42 * s
        else:
            s = ease_out((p - impact) / max(1.0 - impact, 1e-6))
            crouch = 0.42 * (1.0 - s)
        pose = _stance(body, ground_y, x, crouch=crouch, lean=6.0 * crouch)
        # near arm reaches down toward the ground on the deep crouch (the hero pose)
        _set(pose, body, "arm_upper", -25.0 - 40.0 * crouch)
        _set(pose, body, "arm_lower", 30.0 * crouch)
        return pose

    return pose_fn


# ================================================================ everyday acting
# The non-combat verbs a scene needs constantly: cheer, take a pratfall, sit down,
# tap out a rhythm. Same pose_fn contract, same lint-clean discipline.


def celebrate(body: Body, *, ground_y: float, x: float = 0.0, frames: int = 30,
              pumps: int = 2) -> PoseFn:
    """Both arms thrown up and pumping, with a small bounce — joy, victory, a goal.

    Stays on the spot (no skating), the feet planted; the bounce is a crouch that
    springs. The single most-asked-for reaction after a walk."""
    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        pump = abs(math.sin(math.pi * pumps * p))
        pose = _stance(body, ground_y, x, crouch=0.14 * (1.0 - pump))
        for a in ("arm_upper", "arm_upper_far"):
            _set(pose, body, a, -150.0 + 22.0 * pump)   # up, pumping
        for a in ("arm_lower", "arm_lower_far"):
            _set(pose, body, a, 15.0 + 10.0 * pump)
        _set(pose, body, "spine", 2.0 * math.sin(math.pi * pumps * 2 * p))
        return pose

    return pose_fn


def fall(body: Body, *, ground_y: float, x: float = 0.0, facing: float = 1.0,
         frames: int = 24) -> PoseFn:
    """A pratfall: the feet slip out, the body rotates back and drops, then settles
    sitting on the ground — the banana-peel classic.

    Airborne through the fall (feet up, so nothing skates), landing hips-down. The
    root rotation is the whole gag; the legs sprawl forward."""
    hip = body.hip_height
    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        if p < 0.2:                                   # the slip — a quick wobble
            s = ease_in(p / 0.2)
            rot, drop = -10.0 * s, 0.0
        elif p < 0.55:                                # fall back and down onto the seat
            s = ease_in((p - 0.2) / 0.35, power=1.5)
            rot, drop = (-10.0 - 28.0 * s), 0.62 * hip * s
        else:                                         # landed sitting, small bounce
            s = ease_out((p - 0.55) / 0.45)
            rot = -38.0
            drop = 0.62 * hip + 0.03 * hip * math.sin(math.pi * s)
        pose = _stance(body, ground_y, x, root_angle=rot * facing, lift=-drop,
                       plant=False)
        # legs stick out forward: airborne while falling, feet resting on the floor
        # once seated — IK keeps them from ever digging below the ground line.
        landed = clamp((p - 0.4) / 0.3, 0.0, 1.0)
        frames_solved = body.rig.solve(pose)
        for up_n, lo_n in _leg_pairs(body):
            hipj = frames_solved[up_n].origin
            fy = ground_y - 0.42 * hip * (1.0 - landed)
            _ik_foot_to(body, pose, frames_solved, up_n, lo_n,
                        Vec2(hipj.x + 0.52 * hip * facing, fy))
        for a in ("arm_upper", "arm_upper_far"):
            _set(pose, body, a, -34.0 * facing)
        return pose

    return pose_fn


def sit(body: Body, *, ground_y: float, x: float = 0.0, seat: float | None = None,
        frames: int = 18) -> PoseFn:
    """Lower into a seated pose — hips drop, feet forward on the floor, knees bent.

    For a desk, a bench, a campfire. `seat` is the hip height when seated (defaults
    to about half standing height). Feet stay planted at a fixed spot, so it lints
    clean; pair with `tap` for typing or a held prop for a mug by the fire."""
    hip = body.hip_height
    seat_h = seat if seat is not None else hip * 0.5
    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        cur = hip + (seat_h - hip) * ease_in_out(p)   # standing -> seated
        pose = Pose(root=Vec2(x, ground_y - cur), root_angle=0.0)
        frames_solved = body.rig.solve(pose)
        for up_n, lo_n in _leg_pairs(body):           # feet planted, forward
            hipj = frames_solved[up_n].origin
            _ik_foot_to(body, pose, frames_solved, up_n, lo_n,
                        Vec2(hipj.x + 0.28 * hip, ground_y))
        return pose

    return pose_fn


def tap(body: Body, *, ground_y: float, x: float = 0.0, facing: float = 1.0,
        hits: int = 4, frames: int = 24) -> PoseFn:
    """Hands strike downward in alternation — drumming, typing, hammering a keyboard.

    The two hands trade off (one up while the other comes down), the rhythm set by
    `hits`. Put an sfx or an effect on each downbeat. Combine with `sit` for typing
    at a desk, or leave standing for a drummer."""
    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        strike = abs(math.sin(math.pi * hits * p))
        pose = _stance(body, ground_y, x, lean=5.0 * facing)
        _set(pose, body, "arm_upper", facing * (-38.0 - 24.0 * strike))
        _set(pose, body, "arm_lower", facing * (34.0 + 42.0 * strike))
        _set(pose, body, "arm_upper_far", facing * (-38.0 - 24.0 * (1.0 - strike)))
        _set(pose, body, "arm_lower_far", facing * (34.0 + 42.0 * (1.0 - strike)))
        return pose

    return pose_fn


def pushup(body: Body, *, ground_y: float, x: float = 0.0, facing: float = 1.0,
           reps: int = 3, frames: int = 36) -> PoseFn:
    """Push-ups: a horizontal plank, hands and feet on the ground, bobbing up and down.

    The body tips to horizontal (root rotated) with the hands planted forward and the
    feet planted back; the whole plank rises and dips `reps` times. Both contacts IK
    to the ground so nothing skates or sinks."""
    hip = body.hip_height
    plank_y = ground_y - 0.42 * hip           # how high the hips ride at the top
    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        dip = 0.5 - 0.5 * math.cos(2.0 * math.pi * reps * p)   # 0 up .. 1 down
        rise = (0.42 - 0.16 * dip) * hip
        # torso horizontal: rotate the whole rig so the spine lies flat, head forward
        pose = _stance(body, ground_y, x, root_angle=88.0 * facing,
                       lift=rise - hip, plant=False)
        frames_solved = body.rig.solve(pose)
        # hands plant forward, feet plant back — a straight, supported line
        for a_up, a_lo, fwd in (("arm_upper", "arm_lower", 0.42),
                                ("arm_upper_far", "arm_lower_far", 0.42)):
            if a_up in body.rig.joints:
                sh = frames_solved[a_up].origin
                _ik_foot_to(body, pose, frames_solved, a_up, a_lo,
                            Vec2(sh.x + fwd * hip * facing, ground_y))
        for up_n, lo_n in _leg_pairs(body):
            hipj = frames_solved[up_n].origin
            _ik_foot_to(body, pose, frames_solved, up_n, lo_n,
                        Vec2(hipj.x - 0.5 * hip * facing, ground_y))
        return pose

    return pose_fn


def pedal(body: Body, *, ground_y: float, x0: float = 0.0, x1: float = 200.0,
          seat: float | None = None, revolutions: float = 3.0, frames: int = 30) -> PoseFn:
    """Ride: seated and travelling x0->x1 while the legs pump a pedal circle.

    For a bike, a trike, a pedal-boat. The feet trace circles above the ground (never
    planted, so they cannot skate), the hips sit at `seat` height and glide forward.
    Add the vehicle itself as a prop on the same path (add_moving_prop / a rolling
    wheel)."""
    hip = body.hip_height
    seat_h = seat if seat is not None else hip * 0.62
    pairs = _leg_pairs(body)
    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        gx = x0 + (x1 - x0) * p
        pose = Pose(root=Vec2(gx, ground_y - seat_h), root_angle=0.0)
        _set(pose, body, "spine", 14.0)                    # lean forward over the bars
        frames_solved = body.rig.solve(pose)
        r = 0.16 * hip                                     # pedal-crank radius
        cx = gx + 0.14 * hip                               # crank centre, ahead of hips
        cy = ground_y - 0.20 * hip
        for i, (up_n, lo_n) in enumerate(pairs):           # feet 180deg apart on the crank
            ang = 2.0 * math.pi * revolutions * p + (0.0 if i == 0 else math.pi)
            _ik_foot_to(body, pose, frames_solved, up_n, lo_n,
                        Vec2(cx + r * math.cos(ang), cy + r * math.sin(ang)))
        for a in ("arm_upper", "arm_upper_far"):           # arms reach to the bars
            _set(pose, body, a, -70.0)
        return pose

    return pose_fn


def fly(body: Body, *, ground_y: float, x0: float = 0.0, x1: float = 240.0,
        height: float = 120.0, flaps: int = 4, frames: int = 30) -> PoseFn:
    """Flight: airborne and travelling x0->x1, wings (or arms) flapping and the body
    bobbing on each downstroke.

    Feet tuck up, so nothing ever touches the ground (no skating to police). Flaps any
    joints named 'wing*' if the body has them, otherwise the arms — so a bird flaps
    wings and a stick figure flaps its arms like a cartoon superhero. `height` is how
    high above the ground it cruises."""
    hip = body.hip_height
    wings = [j for j in body.rig.joints if "wing" in j.lower()] or \
            [j for j in ("arm_upper", "arm_upper_far") if j in body.rig.joints]
    def pose_fn(t: float) -> Pose:
        p = clamp(t / frames, 0.0, 1.0)
        flap = math.sin(2.0 * math.pi * flaps * p)          # -1 up .. +1 down
        gx = x0 + (x1 - x0) * p
        lift = height + 0.12 * height * (-flap)             # rises on the downstroke
        pose = _stance(body, ground_y, gx, lift=lift, plant=False)
        for w in wings:
            _set(pose, body, w, -95.0 + 55.0 * flap)        # sweep around horizontal
        for up_n, lo_n in _leg_pairs(body):                 # legs trail, tucked up
            _set(pose, body, up_n, -18.0)
            _set(pose, body, lo_n, 30.0)
        return pose

    return pose_fn


def locomote(body: Body, gait: Gait, *, ground_y: float, x0: float = 0.0) -> PoseFn:
    """Adapt a gait into a `pose_fn` so it composes in `sequence` with actions.

    Locomotion is a gait; the acting verbs are pose functions — and `sequence` speaks
    pose functions. This is the one-line bridge: `sequence((locomote(body, walk,
    ground_y=g, x0=90), 14), (fall(body, ground_y=g, x=230), 20))` walks in, then
    falls."""
    return lambda t: pose_at(body.rig, gait, t, ground_y=ground_y, body_x0=x0)


# ============================================================ snap-timing toolkit
# "Snappy vs floaty" is the timing tell that separates viral motion from a moving
# diagram. `diagnose.py` *detects* dead-linear spacing; these *author* the fix.


def hold(pose_fn: PoseFn, at: float, frames: float) -> PoseFn:
    """Freeze a beat on frame `at` for `frames` extra frames, then resume.

    A moving hold: the action stops dead, sits, then continues from exactly where it
    paused. The beat gets `frames` longer — bake over the original length PLUS
    `frames`. This is the time-remap that hitstop is a preset of.
    """
    def wrapped(t: float) -> Pose:
        if t < at:
            return pose_fn(t)
        if t < at + frames:
            return pose_fn(at)
        return pose_fn(t - frames)
    return wrapped


def hitstop(pose_fn: PoseFn, at: float, *, freeze: float = 3.0) -> PoseFn:
    """A brief freeze on impact — the cheapest way to give a hit weight.

    On the contact frame `at` the action stops for `freeze` frames (2-4 reads best),
    the shock of the blow, then continues. Place it on a strike's contact frame (the
    beats land ~60% through) and pair it with an impact flash and a hit sfx on the
    same frame. Adds `freeze` frames to the beat's length.
    """
    return hold(pose_fn, at, freeze)


def retime(pose_fn: PoseFn, span: float, ease: Callable[[float], float]) -> PoseFn:
    """Re-time a beat through an easing curve without changing its length.

    Snappier (`principles.ease_in` or `hold_snap`) sits then bursts; floatier
    (`ease_out`) leads then eases to a stop. `span` is the beat's frame count, `ease`
    maps [0,1]->[0,1]. This is per-beat timing control the LLM can dial in after the
    fact, the authoring twin of the spacing chart the critic reads.
    """
    def wrapped(t: float) -> Pose:
        p = clamp(t / span, 0.0, 1.0) if span > 0 else 0.0
        return pose_fn(ease(p) * span)
    return wrapped


def _lerp_pose(a: Pose, b: Pose, w: float) -> Pose:
    """Blend two poses: lerp the root, its angle, and every joint angle."""
    root = Vec2(a.root.x + (b.root.x - a.root.x) * w,
                a.root.y + (b.root.y - a.root.y) * w)
    out = Pose(root=root,
               root_angle=a.root_angle + (b.root_angle - a.root_angle) * w)
    for k in set(a.angles) | set(b.angles):
        av, bv = a.angles.get(k, 0.0), b.angles.get(k, 0.0)
        out.angles[k] = av + (bv - av) * w
    return out


def sequence(*segments: tuple[PoseFn, int], blend: float = 0.0) -> PoseFn:
    """Play pose functions back to back: [(action, frames), ...].

    Cartoon acting is one beat after another — crouch, then jump, then wave. This
    stitches actions into a timeline, each running in its own local frame count.

    `blend` cross-fades the last `blend` frames of each beat into the start of the
    next, easing the join — it removes the visible pop when the torso or arms jump
    between beats (dash->punch, jump->celebrate). Total length is unchanged. Note it
    smooths *poses*, not footwork: cross-fading two grounded beats whose feet sit at
    different x slides a planted foot (still slip) — there, match the foot positions
    or step, rather than lean on blend. Default 0 keeps the old hard cut.
    """
    if not segments:
        raise ValueError("sequence needs at least one (pose_fn, frames)")
    bounds = []
    acc = 0
    for fn, n in segments:
        bounds.append((acc, acc + n, fn))
        acc += n

    def pose_fn(t: float) -> Pose:
        for i, (start, end, fn) in enumerate(bounds):
            if t < end or end == acc:
                pose = fn(t - start)
                if blend > 0 and i + 1 < len(bounds) and t > end - blend:
                    nxt = bounds[i + 1][2]
                    w = ease_in_out(clamp((t - (end - blend)) / blend, 0.0, 1.0))
                    pose = _lerp_pose(pose, nxt(t - end), w)  # next beat at its start
                return pose
        return bounds[-1][2](t - bounds[-1][0])

    return pose_fn
