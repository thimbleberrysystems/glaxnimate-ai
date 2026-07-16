"""Mux: combine Glaxnimate's silent MP4 with the rendered mix, via PyAV.

Glaxnimate's video exporter writes video only. PyAV bundles its own FFmpeg, so
this needs no system ffmpeg binary and no sudo — and it cannot conflict with the
Glaxnimate build's libav, because it never loads into the same ABI surface (it
is a self-contained wheel).

The video stream is *remuxed* (packets copied bit-for-bit, no re-encode, no
generation loss); the audio buffer is encoded to AAC. Frame-accurate, fast.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from .mix import MixResult

__all__ = ["mux_audio", "has_audio_stream"]


def mux_audio(video_path: str | Path, mix: MixResult, out_path: str | Path) -> Path:
    """video.mp4 + mixed buffer -> out.mp4 with an AAC track."""
    video_path, out_path = Path(video_path), Path(out_path)
    tmp = out_path.with_suffix(".muxing.mp4")

    with av.open(str(video_path)) as src, av.open(str(tmp), "w", format="mp4") as dst:
        in_v = src.streams.video[0]
        out_v = dst.add_stream_from_template(in_v)
        out_a = dst.add_stream("aac", rate=mix.sr)
        out_a.layout = "stereo"

        # --- audio: encode the float buffer
        # PyAV wants planar float (fltp) frames shaped (channels, samples)
        samples = mix.buffer.T.astype(np.float32)  # (2, n)
        chunk = 1024
        pts = 0
        for i in range(0, samples.shape[1], chunk):
            block = np.ascontiguousarray(samples[:, i:i + chunk])
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
            frame.sample_rate = mix.sr
            frame.pts = pts
            pts += block.shape[1]
            for pkt in out_a.encode(frame):
                dst.mux(pkt)
        for pkt in out_a.encode(None):  # flush
            dst.mux(pkt)

        # --- video: straight remux, zero re-encode
        for pkt in src.demux(in_v):
            if pkt.dts is None:
                continue
            pkt.stream = out_v
            dst.mux(pkt)

    tmp.replace(out_path)
    return out_path


def has_audio_stream(path: str | Path) -> bool:
    """Probe helper for tests and sanity checks."""
    with av.open(str(path)) as f:
        return len(f.streams.audio) > 0


def audio_duration(path: str | Path) -> float:
    with av.open(str(path)) as f:
        if not f.streams.audio:
            return 0.0
        a = f.streams.audio[0]
        if a.duration is None:
            return 0.0
        return float(a.duration * a.time_base) if a.time_base else 0.0


