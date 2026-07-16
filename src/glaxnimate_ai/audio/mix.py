"""The mixer: a cue sheet in, a stereo buffer and a numeric report out.

The report is the point. The model cannot hear, and the project's whole critic
discipline is *numbers before senses*: before anyone listens, the mix answers
"how many cues, how loud, did anything clip, do cues pile up?" — the same way
the animation linter answers "do the feet slip?" without rendering a pixel.

The master bus is a tanh soft-limiter, so the output is bounded by construction;
"clipped samples: 0" in the report is a theorem, not a hope. What tanh *does*
audibly do when driven hard is squash — so the report also states the pre-limit
peak, and flags it when the input ran hot.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .synth import BUILTIN, SAMPLE_RATE, render_patch, sfx_validate

__all__ = ["Cue", "resolve_patch", "render_cues", "mix_report", "write_wav"]


@dataclass(slots=True)
class Cue:
    frame: float
    sfx: str | dict         # a builtin/asset name, or an inline patch
    gain: float = 1.0
    pan: float = 0.0        # -1 left .. +1 right


def resolve_patch(sfx: str | dict) -> dict:
    """Name → patch. Inline dicts win, then the asset library, then builtins."""
    if isinstance(sfx, dict):
        return sfx_validate(sfx)
    from ..cartoon import assets as A

    try:
        return sfx_validate(A.load_asset("sfx", sfx))
    except (FileNotFoundError, ValueError):
        pass
    if sfx in BUILTIN:
        return BUILTIN[sfx]
    have = sorted(set(BUILTIN) | set(A.list_assets().get("sfx", [])))
    raise ValueError(f"unknown sfx {sfx!r}; have {have}")


@dataclass(slots=True)
class MixResult:
    buffer: np.ndarray          # (n, 2) float32 in [-1, 1]
    sr: int
    pre_peak: float             # before the limiter, linear
    spans: list = field(default_factory=list)  # (start_s, end_s) per cue


def render_cues(
    cues: list[Cue],
    *,
    frames: int,
    fps: float,
    sr: int = SAMPLE_RATE,
    extra: list[tuple[float, np.ndarray, float, float]] | None = None,
) -> MixResult:
    """Place every cue on a stereo timeline and soft-limit the sum.

    `extra` carries pre-rendered mono material (music beds, TTS lines) as
    (start_seconds, samples, gain, pan) — same bus, same limiter, same report.
    """
    duration = frames / fps
    n = int(math.ceil(duration * sr)) + sr // 10  # tail room for a final decay
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    spans: list[tuple[float, float]] = []

    def place(start_s: float, mono: np.ndarray, gain: float, pan: float) -> None:
        i0 = int(start_s * sr)
        if i0 >= n or gain == 0.0:
            return
        seg = mono[: n - i0].astype(np.float64) * gain
        # equal-power pan
        theta = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi / 4.0
        left[i0:i0 + len(seg)] += seg * math.cos(theta)
        right[i0:i0 + len(seg)] += seg * math.sin(theta)
        spans.append((start_s, start_s + len(seg) / sr))

    for cue in cues:
        place(cue.frame / fps, render_patch(resolve_patch(cue.sfx), sr), cue.gain, cue.pan)
    for start_s, samples, gain, pan in (extra or []):
        place(start_s, samples, gain, pan)

    stereo = np.stack([left, right], axis=1)
    pre_peak = float(np.max(np.abs(stereo))) if len(cues) or extra else 0.0
    limited = np.tanh(stereo).astype(np.float32)  # bounded output, by construction
    return MixResult(limited, sr, pre_peak, spans)


def mix_report(cues: list[Cue], result: MixResult, *, frames: int, fps: float) -> str:
    """The audio linter: everything a model can check without ears."""
    if not cues and not result.spans:
        return "no audio: the cue sheet is empty (auto_sfx places cues from motion)"

    post_peak = float(np.max(np.abs(result.buffer))) if len(result.buffer) else 0.0
    db = (20 * math.log10(post_peak)) if post_peak > 0 else float("-inf")

    # pile-ups: how many spans overlap at each cue start
    worst_overlap = 0
    for s0, _ in result.spans:
        worst_overlap = max(
            worst_overlap, sum(1 for a, b in result.spans if a <= s0 < b)
        )

    lines = [
        f"{len(cues)} cue(s) over {frames / fps:.1f}s "
        f"| peak {db:.1f} dBFS | clipped samples: 0 (soft limiter)",
    ]
    if result.pre_peak > 1.5:
        lines.append(
            f"  WARNING: the bus ran hot before limiting (pre-peak {result.pre_peak:.2f}) "
            f"- audible squash; lower cue gains"
        )
    if worst_overlap > 4:
        lines.append(
            f"  WARNING: {worst_overlap} sounds overlap at once - it will read as mush; "
            f"thin the cues"
        )
    by_frame = sorted(cues, key=lambda c: c.frame)
    shown = ", ".join(
        f"f{c.frame:g}:{c.sfx if isinstance(c.sfx, str) else 'inline'}"
        for c in by_frame[:14]
    )
    more = f" (+{len(by_frame) - 14} more)" if len(by_frame) > 14 else ""
    lines.append(f"  cues: {shown}{more}")
    return "\n".join(lines)


def write_wav(result: MixResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (result.buffer * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(result.sr)
        w.writeframes(pcm.tobytes())
    return path
