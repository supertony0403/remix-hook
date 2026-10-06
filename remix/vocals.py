"""Time-stretch / pitch-shift with the rubberband CLI (R3 engine, formant preservation)
and chroma-based key checks."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import librosa
import numpy as np

from . import config
from .audio_io import read_wav, to_mono, write_wav

PITCH_NAMES = ["C", "Cis", "D", "Dis", "E", "F", "Fis", "G", "Gis", "A", "B", "H"]  # German (B = Bb, H = B)

# Krumhansl-Kessler key profiles
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def rubberband(x: np.ndarray, time_ratio: float, semitones: float = 0.0,
               formant: bool = True, crisp: int | None = None) -> np.ndarray:
    """Offline rubberband R3. `time_ratio` = output duration / input duration. Cached in work/."""
    key = hashlib.sha1(
        np.ascontiguousarray(x, np.float32).tobytes()
        + f"{x.shape}|{time_ratio:.9f}|{semitones}|{formant}|{crisp}|-3 -q".encode()
    ).hexdigest()[:16]
    cache = config.WORK / "rb" / f"{key}.wav"
    if cache.exists():
        y, _ = read_wav(cache)
        return y
    src = config.WORK / "rb" / f"{key}_in.wav"
    write_wav(src, x, config.SR)
    cmd = ["rubberband", "-q", "-3", "-t", f"{time_ratio:.9f}", "-p", f"{semitones}"]
    if formant:
        cmd.append("-F")
    if crisp is not None:
        cmd += ["-c", str(crisp)]
    subprocess.run(cmd + [str(src), str(cache)], check=True, capture_output=True)
    src.unlink(missing_ok=True)
    y, _ = read_wav(cache)
    return y


def chroma_mean(y: np.ndarray, sr: int, active_db: float = -35.0) -> np.ndarray:
    """Mean CQT chroma over frames that carry signal (silence would flatten the profile)."""
    m = to_mono(y).astype(np.float32)
    hop = 1024
    c = librosa.feature.chroma_cqt(y=m, sr=sr, hop_length=hop, bins_per_octave=36)
    rms = librosa.feature.rms(y=m, hop_length=hop)[0][: c.shape[1]]
    ref = np.percentile(rms[rms > 0], 95) if np.any(rms > 0) else 1.0
    keep = 20 * np.log10(rms + 1e-12) > 20 * np.log10(ref) + active_db
    c = c[:, : len(keep)][:, keep] if keep.any() else c
    v = c.mean(axis=1)
    return v / (v.sum() + 1e-12)


def key_scores(chroma: np.ndarray) -> list[tuple[str, float]]:
    """Pearson correlation with all 24 Krumhansl-Kessler profiles, best first."""
    out = []
    for i in range(12):
        out.append((f"{PITCH_NAMES[i]}-Dur", float(np.corrcoef(chroma, np.roll(KK_MAJOR, i))[0, 1])))
        out.append((f"{PITCH_NAMES[i]}-Moll", float(np.corrcoef(chroma, np.roll(KK_MINOR, i))[0, 1])))
    return sorted(out, key=lambda kv: -kv[1])


def transposition_scores(chroma_a: np.ndarray, chroma_b: np.ndarray) -> dict[int, float]:
    """Correlation of A's chroma shifted by k semitones (k in -6..+5) with B's chroma."""
    return {k: float(np.corrcoef(np.roll(chroma_a, k), chroma_b)[0, 1]) for k in range(-6, 6)}


def note_of(y: np.ndarray, sr: int) -> tuple[str, float]:
    """Median pYIN pitch of a short sung segment -> (note name, Hz)."""
    m = to_mono(y).astype(np.float32)
    f0, voiced, _ = librosa.pyin(m, fmin=80, fmax=900, sr=sr, frame_length=2048)
    f = f0[voiced & ~np.isnan(f0)] if voiced is not None else np.array([])
    if len(f) == 0:
        return ("?", 0.0)
    hz = float(np.median(f))
    return (librosa.hz_to_note(hz, unicode=False), hz)


def load_or(path: Path) -> np.ndarray | None:
    return read_wav(path)[0] if path.exists() else None
