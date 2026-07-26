"""Lip-sync: a talking mouth derived from the dialogue audio, not guessed.

The model cannot hear its own dialogue, but it does not need to — the mouth is
arithmetic over the WAV that `say()` already renders and caches. A short-window RMS
envelope, quantised to three openness levels (closed / mid / open), becomes a
sequence of mouth-expression swaps on the exact frames the voice rises and falls.
Same numbers-first spirit as the rest of the feedback stack: no vision, no hearing,
just the envelope.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LIPSYNC_SLOTS", "mouth_levels"]

#: The three speech mouth attachments a face needs for lip-sync, closed -> open.
#: A face without all three simply is not lip-synced (say() falls back to silence).
LIPSYNC_SLOTS = ("say_closed", "say_mid", "say_open")


def mouth_levels(samples: np.ndarray, *, fps: float, start_frame: float,
                 sr: int = 44100, quiet: float = 0.15, loud: float = 0.5
                 ) -> list[tuple[int, int]]:
    """Per-frame mouth openness (0 closed, 1 mid, 2 open) from an audio envelope.

    Returns `(frame, level)` pairs only where the level *changes* — a hold-keyed
    swap costs nothing on the frames between — and always closes the mouth on the
    frame after the line ends. `quiet`/`loud` are fractions of the line's own peak,
    so a soft line still articulates instead of sitting shut.
    """
    if samples is None or len(samples) == 0:
        return [(int(start_frame), 0)]

    spf = max(sr / float(fps), 1.0)
    n_frames = int(np.ceil(len(samples) / spf))
    env = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = samples[int(i * spf):int((i + 1) * spf)]
        env[i] = float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.0

    peak = float(env.max()) or 1.0
    out: list[tuple[int, int]] = []
    prev: int | None = None
    for i in range(n_frames):
        v = env[i] / peak
        level = 0 if v < quiet else (1 if v < loud else 2)
        if level != prev:
            out.append((int(start_frame + i), level))
            prev = level
    if prev != 0:  # shut the mouth once the line is over
        out.append((int(start_frame + n_frames), 0))
    return out
