"""Cartoon sound effects, synthesized from data patches.

The classic cartoon SFX vocabulary — boings, thuds, pops, slide whistles — was
physically synthesized in the first place (springs, whistles, drums), which makes
procedural synthesis period-correct rather than a cheap substitute. More
importantly for this project, it makes a sound **a document**: an `sfx` asset is
a JSON patch of oscillator layers, authorable and iterable by the model exactly
like a body or a face. No sample libraries, no licensing, no downloads.

A patch is a sum of layers:

    {"version": 1, "kind": "sfx",
     "layers": [{"osc": "sine",         # sine | square | triangle | noise
                 "f0": 520, "f1": 110,  # start/end Hz (exponential sweep)
                 "dur": 0.45,           # seconds
                 "attack": 0.003,       # fade-in, kills the onset click
                 "decay": 7.0,          # exponential amplitude decay, 1/s
                 "gain": 0.9,
                 "lp": 0}]}             # one-pole lowpass for noise, Hz (0=off)

Pitch sweeps integrate the instantaneous frequency, so a swept sine is phase-
continuous. Everything is deterministic — noise uses a fixed-seed generator — so
renders are reproducible and testable byte-for-byte.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SAMPLE_RATE", "sfx_validate", "render_patch", "BUILTIN"]

SAMPLE_RATE = 44_100
_OSCS = ("sine", "square", "triangle", "noise")


def sfx_validate(data: dict) -> dict:
    if data.get("kind") != "sfx" or data.get("version") != 1:
        raise ValueError("an sfx document needs kind='sfx' and version=1")
    layers = data.get("layers")
    if not layers:
        raise ValueError("an sfx patch needs a non-empty 'layers' list")
    for i, la in enumerate(layers):
        osc = la.get("osc")
        if osc not in _OSCS:
            raise ValueError(f"layers[{i}].osc is {osc!r}; must be one of {_OSCS}")
        if osc != "noise" and not la.get("f0"):
            raise ValueError(f"layers[{i}] ({osc}) needs a start frequency 'f0'")
        if float(la.get("dur", 0)) <= 0:
            raise ValueError(f"layers[{i}] needs a positive 'dur' in seconds")
    return data


def _wave(osc: str, phase: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    if osc == "sine":
        return np.sin(2 * np.pi * phase)
    if osc == "square":
        return np.sign(np.sin(2 * np.pi * phase)) * 0.7  # squares read loud; tame them
    if osc == "triangle":
        return 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    return rng.uniform(-1.0, 1.0, n)  # noise


def render_patch(patch: dict, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render a patch to mono float32 in [-1, 1]. Deterministic."""
    sfx_validate(patch)
    rng = np.random.default_rng(0)  # fixed seed: same patch, same samples, always

    total = max(int(sr * (la.get("dur", 0.2))) for la in patch["layers"])
    out = np.zeros(total, dtype=np.float64)

    for la in patch["layers"]:
        n = int(sr * la["dur"])
        t = np.arange(n) / sr
        osc = la["osc"]

        if osc == "noise":
            sig = _wave(osc, t, rng, n)
            lp = float(la.get("lp", 0))
            if lp > 0:
                # one-pole lowpass: y[i] = y[i-1] + a (x[i] - y[i-1])
                a = 1.0 - np.exp(-2.0 * np.pi * lp / sr)
                sig = _lowpass(sig, a)
        else:
            f0 = float(la["f0"])
            f1 = float(la.get("f1", f0))
            # exponential sweep; integrate frequency for continuous phase
            freq = f0 * (f1 / f0) ** (t / la["dur"]) if f1 != f0 else np.full(n, f0)
            phase = np.cumsum(freq) / sr
            sig = _wave(osc, phase, rng, n)

        env = np.exp(-float(la.get("decay", 6.0)) * t)
        attack = float(la.get("attack", 0.003))
        if attack > 0:
            k = min(int(attack * sr), n)
            if k > 0:
                env[:k] *= np.linspace(0.0, 1.0, k)
        out[:n] += sig * env * float(la.get("gain", 0.8))

    # a tiny fade-out so no patch ends on a click
    k = min(int(0.004 * sr), total)
    if k > 0:
        out[-k:] *= np.linspace(1.0, 0.0, k)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _lowpass(x: np.ndarray, a: float) -> np.ndarray:
    # scipy-free one-pole filter via the standard lfilter recurrence
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


def _patch(*layers: dict) -> dict:
    return {"version": 1, "kind": "sfx", "layers": list(layers)}


#: The shipped vocabulary. Every value here also gets written to assets/sfx/ so
#: the model can read a real example before authoring its own.
BUILTIN: dict[str, dict] = {
    "boing": _patch(
        {"osc": "sine", "f0": 520, "f1": 110, "dur": 0.45, "decay": 7, "gain": 0.85},
        {"osc": "sine", "f0": 1040, "f1": 220, "dur": 0.3, "decay": 10, "gain": 0.2},
    ),
    "thud": _patch(
        {"osc": "noise", "lp": 300, "dur": 0.12, "decay": 30, "gain": 0.7},
        {"osc": "sine", "f0": 75, "f1": 48, "dur": 0.18, "decay": 16, "gain": 0.8},
    ),
    "step": _patch(
        {"osc": "noise", "lp": 900, "dur": 0.07, "decay": 55, "gain": 0.5},
        {"osc": "sine", "f0": 170, "f1": 110, "dur": 0.05, "decay": 45, "gain": 0.3},
    ),
    "pop": _patch(
        {"osc": "sine", "f0": 900, "f1": 280, "dur": 0.06, "decay": 35, "gain": 0.8},
    ),
    "whoosh": _patch(
        {"osc": "noise", "lp": 1400, "dur": 0.35, "attack": 0.12, "decay": 9, "gain": 0.6},
    ),
    "slide_up": _patch(
        {"osc": "triangle", "f0": 320, "f1": 950, "dur": 0.5, "decay": 3.2, "gain": 0.6},
    ),
    "slide_down": _patch(
        {"osc": "triangle", "f0": 950, "f1": 320, "dur": 0.5, "decay": 3.2, "gain": 0.6},
    ),
    "splat": _patch(
        {"osc": "noise", "lp": 500, "dur": 0.15, "decay": 18, "gain": 0.7},
        {"osc": "sine", "f0": 220, "f1": 55, "dur": 0.2, "decay": 14, "gain": 0.5},
    ),
    "ding": _patch(
        {"osc": "sine", "f0": 1319, "dur": 0.8, "decay": 4, "gain": 0.5},
        {"osc": "sine", "f0": 2637, "dur": 0.5, "decay": 7, "gain": 0.12},
    ),
}
