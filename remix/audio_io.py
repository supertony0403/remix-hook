"""Audio decoding/encoding helpers (ffmpeg for decoding, soundfile for WAV)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def decode(path: Path, sr: int, channels: int = 2) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 array of shape (n, channels)."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vn", "-ac", str(channels), "-ar", str(sr),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).copy()


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def write_wav(path: Path, data: np.ndarray, sr: int, subtype: str = "FLOAT") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, subtype=subtype)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x
