"""A seeded chiptune underscore. Simple by design, mutable by data.

This is not trying to be a composer. It is a background bed in the register the
visuals already live in — square-wave melody, triangle bass, the vocabulary of
early cartoons and game consoles — generated deterministically from a seed so
`{"seed": 7, "bpm": 100}` in the scene doc always renders the same bar of music.

Two cheap tricks carry the musicality:

* the melody walks a **major pentatonic** scale — no interval in it can clash
  with the I–V–vi–IV progression underneath, so a random walk cannot play a
  wrong note, only a boring one;
* the bass plays roots and fifths on the strong beats, which is what tells the
  ear where the chords are.

If it sounds naff, the model changes the seed and re-renders: the music is a
data field like a body's leg length, and taste stays in the iteration loop.
"""

from __future__ import annotations

import numpy as np

from .synth import SAMPLE_RATE

__all__ = ["render_music", "music_validate"]

#: I–V–vi–IV in scale degrees (0-indexed on the major scale).
_PROGRESSIONS = [
    [0, 4, 5, 3],   # I V vi IV — the four-chord song
    [0, 3, 4, 3],   # I IV V IV
    [5, 3, 0, 4],   # vi IV I V — the moodier rotation
]
_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_PENTATONIC = [0, 2, 4, 7, 9]


def music_validate(spec: dict) -> dict:
    bpm = float(spec.get("bpm", 96))
    if not 40 <= bpm <= 220:
        raise ValueError(f"bpm {bpm} is outside 40..220")
    gain = float(spec.get("gain", 0.25))
    if not 0 < gain <= 1.0:
        raise ValueError(f"music gain {gain} must be in (0, 1]")
    return {"seed": int(spec.get("seed", 0)), "bpm": bpm, "gain": gain}


def _tone(freq: float, dur_s: float, sr: int, *, osc: str, decay: float,
          gain: float) -> np.ndarray:
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    phase = freq * t
    if osc == "square":
        sig = np.sign(np.sin(2 * np.pi * phase)) * 0.5
    else:  # triangle
        sig = 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    env = np.exp(-decay * t)
    k = max(int(0.004 * sr), 1)
    env[:k] *= np.linspace(0, 1, k)
    if n > k:
        env[-k:] *= np.linspace(1, 0, k)
    return sig * env * gain


def _hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def render_music(spec: dict, *, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render the underscore to exactly `duration_s`, looping bars as needed."""
    spec = music_validate(spec)
    rng = np.random.default_rng(spec["seed"])

    key_root = 57 + int(rng.integers(0, 12))          # A3 .. G#4
    progression = _PROGRESSIONS[int(rng.integers(0, len(_PROGRESSIONS)))]
    beat = 60.0 / spec["bpm"]
    bar = 4 * beat

    total = int(duration_s * sr)
    out = np.zeros(total + sr, dtype=np.float64)

    def place(start_s: float, tone: np.ndarray) -> None:
        i0 = int(start_s * sr)
        if i0 < len(out):
            seg = tone[: len(out) - i0]
            out[i0:i0 + len(seg)] += seg

    # melody state persists across bars so phrases connect
    scale = [key_root + 12 + o + d for o in (0, 12) for d in _PENTATONIC]
    mel_i = int(rng.integers(0, len(scale)))

    t = 0.0
    bar_no = 0
    while t < duration_s:
        degree = progression[bar_no % len(progression)]
        root = key_root + _MAJOR[degree % 7] + (0 if degree < 7 else 12)

        # bass: root on 1, fifth on 3
        place(t, _tone(_hz(root - 12), beat * 0.9, sr, osc="triangle", decay=3, gain=0.5))
        place(t + 2 * beat,
              _tone(_hz(root - 12 + 7), beat * 0.9, sr, osc="triangle", decay=3, gain=0.4))

        # melody: eighth-note random walk on the pentatonic, with rests
        for e in range(8):
            if rng.random() < 0.3:
                continue  # a rest — silence is a note too
            mel_i = int(np.clip(mel_i + rng.integers(-2, 3), 0, len(scale) - 1))
            place(t + e * beat / 2,
                  _tone(_hz(scale[mel_i]), beat * 0.45, sr,
                        osc="square", decay=6, gain=0.22))

        t += bar
        bar_no += 1

    return np.clip(out[:total], -1.0, 1.0).astype(np.float32)
