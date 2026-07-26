"""Rhythm: the tempo grid, so cuts and hits can land on the beat.

`auto_sfx` places sound where the motion already is; this is the inverse — it hands
back the frames the *music* lands on, so a cut, a punch or an effect can be snapped
to a downbeat. Beat-synced action is most of what makes a short-form montage feel
deliberate instead of arbitrary. Pure arithmetic on `bpm` and `fps`, no audio.
"""

from __future__ import annotations

__all__ = ["frames_per_beat", "beat_frames", "snap_to_beat"]


def frames_per_beat(bpm: float, fps: float) -> float:
    """How many frames one beat lasts. 120 bpm at 24 fps is 12 frames a beat."""
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    return fps * 60.0 / bpm


def beat_frames(bpm: float, fps: float, frames: int, *, division: int = 1) -> list[int]:
    """The frame of every beat (or `division` subdivision) up to `frames`.

    `division=1` is quarter-note beats; 2 is eighths, 4 is sixteenths — finer grids
    for busier cutting. Frames are rounded to the integer timeline the baker uses.
    """
    step = frames_per_beat(bpm, fps) / max(division, 1)
    out: list[int] = []
    i = 0
    while round(i * step) <= frames:
        out.append(round(i * step))
        i += 1
    return out


def snap_to_beat(frame: float, bpm: float, fps: float, *, division: int = 1) -> int:
    """Round `frame` to the nearest beat (or subdivision) — put a cut on the grid."""
    step = frames_per_beat(bpm, fps) / max(division, 1)
    return round(round(frame / step) * step)
