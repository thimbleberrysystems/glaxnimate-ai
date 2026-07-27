"""Dialogue: local neural TTS via piper, with a stub for model-less environments.

piper is an optional extra (`pip install 'glaxnimate-ai[tts]'`) — it pulls
onnxruntime, and it is needed to *author* a line, not to replay one. Voices are
single ONNX files under `assets/voices/` (one ~60 MB download per voice, network
needed once):

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
    """Beeps standing in for speech: ~0.05 s per character, alternating pitch.

    A slow amplitude envelope fakes syllables so lip-sync has something to bite on
    (a flat tone would hold the mouth open the whole line); it dips toward silence
    at word gaps, so the mouth actually flaps. Deterministic in the text either way.
    """
    dur = max(0.3, 0.05 * len(text))
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = np.where((t * 4).astype(int) % 2 == 0, 420.0, 520.0)
    # ~4 syllables/sec, gapped at spaces so words separate; never fully silent mid-word
    syl = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 4.0 * t))
    words = max(text.count(" ") + 1, 1)
    gap = 0.5 + 0.5 * (np.sin(2 * np.pi * words * t / max(dur, 1e-6)) > -0.4)
    sig = 0.3 * syl * gap * np.sin(2 * np.pi * np.cumsum(f) / sr)
    k = int(0.01 * sr)
    sig[:k] *= np.linspace(0, 1, k)
    sig[-k:] *= np.linspace(1, 0, k)
    return sig.astype(np.float32)


def _gtts_synthesize(text: str, lang: str, sr: int) -> np.ndarray:
    """A line in any Google-TTS language (Tamil, Hindi, ...), decoded to mono float32.

    Piper has no Tamil voice, so a bare language code routes here instead. Needs the
    network at synthesis time; the rendered WAV is cached in the project, so replay
    stays offline like every other line.
    """
    try:
        from gtts import gTTS
    except ImportError as e:
        raise ImportError(
            f"speaking {lang!r} needs gTTS (piper has no voice for it). Run: "
            "uv pip install --python .venv/bin/python gtts"
        ) from e
    import io

    import av

    buf = io.BytesIO()
    gTTS(text, lang=lang).write_to_fp(buf)     # mp3
    buf.seek(0)
    chunks, native = [], sr
    with av.open(buf) as c:
        for fr in c.decode(audio=0):
            native = fr.sample_rate
            x = fr.to_ndarray()
            chunks.append(x.mean(axis=0) if x.ndim == 2 else x.reshape(-1))
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    mono = np.concatenate(chunks).astype(np.float32)
    if float(np.max(np.abs(mono))) > 1.5:      # int16 payload -> normalise
        mono /= 32768.0
    if native != sr:
        n = int(len(mono) * sr / native)
        mono = np.interp(np.linspace(0, len(mono) - 1, n),
                         np.arange(len(mono)), mono).astype(np.float32)
    return mono


def synthesize(text: str, voice: str = DEFAULT_VOICE,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """Text → mono float32 at `sr`. Raises a teaching error if the voice model
    is absent (with the exact command that fixes it).

    `voice` is a piper model name (``en_US-lessac-medium``) OR a Google-TTS language
    code for languages piper lacks: a bare code like ``ta`` (Tamil) / ``hi``, or an
    explicit ``gtts:ta``.
    """
    if os.environ.get("GLAXNIMATE_AI_TTS_STUB"):
        return _stub(text, sr)

    if voice.startswith("gtts:"):
        return _gtts_synthesize(text, voice.split(":", 1)[1], sr)
    if "-" not in voice and "_" not in voice:   # a bare language code -> gTTS
        return _gtts_synthesize(text, voice, sr)

    # Order matters: check the package before the model, because the command
    # that fetches the model IS the package. Reporting a missing download to
    # someone who has no piper sends them in a circle.
    try:
        from piper import PiperVoice
    except ImportError as e:
        raise ImportError(
            "dialogue needs piper-tts, which is an optional extra. "
            "Run: uv pip install --python .venv/bin/python 'glaxnimate-ai[tts]'"
            " -- then download a voice (the next error tells you how). "
            "Scenes with already-cached lines replay without it."
        ) from e

    model = voices_dir() / f"{voice}.onnx"
    if not model.exists():
        have = sorted(p.stem for p in voices_dir().glob("*.onnx"))
        raise FileNotFoundError(
            f"voice model {voice!r} is not downloaded (have: {have or 'none'}). "
            f"Run: .venv/bin/python -m piper.download_voices {voice} "
            f"--data-dir {voices_dir()}"
        )

    if voice not in _loaded:
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
