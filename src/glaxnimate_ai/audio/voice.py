"""Dialogue: local neural TTS via piper, with a stub for model-less environments.

Voices are single ONNX files under `assets/voices/` (one ~60 MB download per
voice, network needed once):

    .venv/bin/python -m piper.download_voices en_US-lessac-medium --data-dir assets/voices

Everything downstream is offline. Synthesized lines are cached as WAVs inside
the project directory, so a scene *replays its dialogue without piper installed
at all* — the same persist-the-samples rule the scene document uses for poses.

`GLAXNIMATE_AI_TTS_STUB=1` swaps synthesis for a deterministic beep pattern whose
duration scales with the text. That is what the test suite uses: the contract
under test is caching, mixing, panning and persistence — not piper's acoustics,
which are not ours to test.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np

from .synth import SAMPLE_RATE

__all__ = ["synthesize", "voices_dir", "DEFAULT_VOICE"]

DEFAULT_VOICE = "en_US-lessac-medium"
_loaded: dict[str, object] = {}


def voices_dir() -> Path:
    from ..cartoon.assets import assets_root

    return assets_root() / "voices"


def _stub(text: str, sr: int) -> np.ndarray:
    """Beeps standing in for speech: ~0.05 s per character, alternating pitch."""
    dur = max(0.3, 0.05 * len(text))
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = np.where((t * 4).astype(int) % 2 == 0, 420.0, 520.0)
    sig = 0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr)
    k = int(0.01 * sr)
    sig[:k] *= np.linspace(0, 1, k)
    sig[-k:] *= np.linspace(1, 0, k)
    return sig.astype(np.float32)


def synthesize(text: str, voice: str = DEFAULT_VOICE,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """Text → mono float32 at `sr`. Raises a teaching error if the voice model
    is absent (with the exact command that fixes it)."""
    if os.environ.get("GLAXNIMATE_AI_TTS_STUB"):
        return _stub(text, sr)

    model = voices_dir() / f"{voice}.onnx"
    if not model.exists():
        have = sorted(p.stem for p in voices_dir().glob("*.onnx"))
        raise FileNotFoundError(
            f"voice model {voice!r} is not downloaded (have: {have or 'none'}). "
            f"Run: .venv/bin/python -m piper.download_voices {voice} "
            f"--data-dir {voices_dir()}"
        )

    if voice not in _loaded:
        from piper import PiperVoice

        _loaded[voice] = PiperVoice.load(str(model))
    pv = _loaded[voice]

    chunks = []
    native_sr = sr
    for chunk in pv.synthesize(text):
        native_sr = chunk.sample_rate
        arr = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        chunks.append(arr.astype(np.float32) / 32768.0)
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    mono = np.concatenate(chunks)

    if native_sr != sr:  # piper voices are typically 22050; the bus is 44100
        n_out = int(len(mono) * sr / native_sr)
        mono = np.interp(
            np.linspace(0, len(mono) - 1, n_out),
            np.arange(len(mono)), mono,
        ).astype(np.float32)
    return mono


def save_line(samples: np.ndarray, path: Path, sr: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1, 1) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def load_line(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sr
