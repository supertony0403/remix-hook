"""Tempo and bar-grid measurement.

librosa's tempo estimate is quantised to its hop lag (512 samples @ 22.05 kHz), which is why
the raw analysis reported 64.6 (= half of 129.2) for A and 152.0 for B. Here the tempo is
measured with a comb score on a high-resolution onset envelope (2.7 ms hop), which resolves
fractions of a BPM, and the bar phase (downbeat) is chosen from kick energy + harmonic change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .audio_io import to_mono

HOP = 128


@dataclass(frozen=True)
class Grid:
    """A constant-tempo 4/4 grid: downbeat of bar 0 at `downbeat0` seconds."""

    bpm: float
    downbeat0: float
    beats_per_bar: int = 4

    @property
    def beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar(self) -> float:
        return self.beat * self.beats_per_bar

    def bar_time(self, n: float) -> float:
        return self.downbeat0 + n * self.bar

    def beat_time(self, n: float) -> float:
        return self.downbeat0 + n * self.beat

    def nearest_bar(self, t: float) -> int:
        return int(round((t - self.downbeat0) / self.bar))

    def to_json(self) -> dict:
        return asdict(self)


def onset_envelope(y: np.ndarray, sr: int, band: tuple[float, float] | None = None) -> np.ndarray:
    """Onset strength (hop 128) with a local-mean subtraction; optional band-pass first."""
    y = to_mono(y)
    if band is not None:
        lo, hi = band
        if lo <= 0:
            sos = butter(4, hi, btype="lowpass", fs=sr, output="sos")
        else:
            sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
        y = sosfiltfilt(sos, y)
    agg = np.median if band is None else np.mean  # band-limited: most mel bands are empty
    o = librosa.onset.onset_strength(y=y.astype(np.float32), sr=sr, hop_length=HOP,
                                     n_fft=2048, aggregate=agg)
    o = o - np.convolve(o, np.ones(64) / 64, "same")
    return np.maximum(o, 0.0)


def comb_score(env: np.ndarray, sr: int, period: float, phase: float,
               t0: float = 0.0, t1: float | None = None) -> float:
    t1 = t1 if t1 is not None else len(env) * HOP / sr
    first = phase + np.ceil((t0 - phase) / period) * period
    ts = np.arange(first, t1, period)
    if len(ts) == 0:
        return 0.0
    return float(np.interp(ts * sr / HOP, np.arange(len(env)), env).mean())


def fit_tempo(env: np.ndarray, sr: int, lo: float, hi: float,
              t0: float = 0.0, t1: float | None = None) -> tuple[float, float, float]:
    """Return (bpm, beat_phase_seconds, score) maximising the comb score; coarse then fine."""

    def search(bpms: np.ndarray, phase_step: float) -> tuple[float, float, float]:
        best = (-1.0, 0.0, 0.0)
        for bpm in bpms:
            p = 60.0 / bpm
            for ph in np.arange(0.0, p, phase_step):
                s = comb_score(env, sr, p, ph, t0, t1)
                if s > best[0]:
                    best = (s, float(bpm), float(ph))
        return best

    s, bpm, ph = search(np.arange(lo, hi, 0.05), 0.004)
    s, bpm, ph = search(np.arange(bpm - 0.06, bpm + 0.06, 0.005), 0.001)
    return bpm, ph, s


def local_tempi(env: np.ndarray, sr: int, lo: float, hi: float, win: float = 30.0) -> list[float]:
    dur = len(env) * HOP / sr
    out = []
    for w0 in np.arange(0.0, dur - win + 1e-9, win):
        out.append(fit_tempo(env, sr, lo, hi, w0, w0 + win)[0])
    return out


def entry_times(x: np.ndarray, sr: int, floor_db: float = -45.0, rise_db: float = 20.0,
                min_silence: float = 0.6) -> list[float]:
    """Times where a stem enters after >= `min_silence` s below `floor_db` (5 ms frames)."""
    m = to_mono(x)
    f = int(0.005 * sr)
    n = len(m) // f
    rms = np.sqrt(np.mean(m[: n * f].reshape(n, f) ** 2, axis=1))
    lv = 20 * np.log10(rms + 1e-9)
    need = int(min_silence / 0.005)
    out, quiet = [], 0
    for i, v in enumerate(lv):
        if v < floor_db:
            quiet += 1
            continue
        if quiet >= need and v > floor_db + rise_db * 0.5:
            out.append(i * 0.005)
        quiet = 0
    return out


def downbeat_index(y: np.ndarray, sr: int, bpm: float, beat_phase: float,
                   drums: np.ndarray | None = None, harmonic: np.ndarray | None = None,
                   bass: np.ndarray | None = None) -> dict:
    """Which of the 4 beat positions is the bar's '1'.

    Decisive evidence: where drums/bass re-enter after silence (sections start on a '1').
    Tie-breaker: z(kick energy at beats k mod 4) + z(chroma change at beats k mod 4).
    Kick alone is ambiguous for half-time grooves (kick on 1 and 3, or a strong kick on 4).
    """
    beat = 60.0 / bpm
    dur = len(y) / sr
    beats = np.arange(beat_phase, dur - beat, beat)

    kick_env = onset_envelope(drums if drums is not None else y, sr, band=(0, 120))
    kick = np.interp(beats * sr / HOP, np.arange(len(kick_env)), kick_env)

    h = to_mono(harmonic if harmonic is not None else y).astype(np.float32)
    chroma = librosa.feature.chroma_cqt(y=h, sr=sr, hop_length=512)
    frames = np.clip((beats * sr / 512).astype(int), 0, chroma.shape[1] - 1)
    sync = librosa.util.sync(chroma, frames, aggregate=np.median)
    nov = np.r_[0.0, np.linalg.norm(np.diff(sync, axis=1), axis=0)]
    nov = nov[: len(beats)]
    if len(nov) < len(beats):
        nov = np.r_[nov, np.zeros(len(beats) - len(nov))]

    kick_s = np.array([kick[k::4].mean() for k in range(4)])
    nov_s = np.array([nov[k::4].mean() for k in range(4)])

    entries: list[float] = []
    for stem in (drums, bass):
        if stem is not None:
            entries += entry_times(stem, sr)
    votes = np.zeros(4)
    for t in entries:
        bi = (t - beat_phase) / beat
        if abs(bi - round(bi)) < 0.15:  # entry must sit on a beat
            votes[int(round(bi)) % 4] += 1

    def z(v: np.ndarray) -> np.ndarray:
        return (v - v.mean()) / (v.std() + 1e-9)

    total = z(kick_s) + z(nov_s) + (3.0 * votes / votes.sum() * 4 if votes.sum() else 0.0)
    k = int(np.argmax(total))
    return {
        "index": k,
        "downbeat0": float(beat_phase + k * beat),
        "kick_by_pos": kick_s.round(4).tolist(),
        "chroma_change_by_pos": nov_s.round(4).tolist(),
        "entry_votes_by_pos": votes.astype(int).tolist(),
        "entries_s": [round(t, 3) for t in entries],
    }


def save_grids(path: Path, grids: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(grids, indent=2))


def load_grid(path: Path, song: str) -> Grid:
    d = json.loads(path.read_text())[song]
    return Grid(bpm=d["bpm"], downbeat0=d["downbeat0"])
