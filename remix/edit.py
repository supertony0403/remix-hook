"""Sample-accurate cutting and placing: cuts snap to nearby zero crossings, 5-10 ms fades."""

from __future__ import annotations

import numpy as np

from . import config
from .fx import fade

SR = config.SR


def zero_cross_near(x: np.ndarray, idx: int, search_ms: float = 2.0) -> int:
    """Nearest sample to `idx` (within +-search_ms) where the mono signal changes sign."""
    m = x.mean(axis=1) if x.ndim == 2 else x
    r = int(search_ms / 1000 * SR)
    lo, hi = max(1, idx - r), min(len(m) - 1, idx + r)
    if hi <= lo:
        return int(np.clip(idx, 0, len(m)))
    seg = m[lo - 1:hi]
    zc = np.nonzero(np.signbit(seg[:-1]) != np.signbit(seg[1:]))[0] + lo
    if len(zc) == 0:
        return int(idx)
    return int(zc[np.argmin(np.abs(zc - idx))])


def cut(x: np.ndarray, t0: float, t1: float, fade_in_ms: float = 6.0, fade_out_ms: float = 8.0,
        snap: bool = True) -> tuple[np.ndarray, int]:
    """Return (segment, start_shift_samples). Bounds snap to zero crossings near t0/t1.

    `start_shift_samples` is how far the snapped start moved relative to round(t0*SR), so the
    caller can keep the musical anchor exact (shift is at most a couple of ms).
    """
    s0 = int(round(t0 * SR))
    s1 = int(round(t1 * SR))
    s0c = max(0, s0)
    if snap:
        s0c = zero_cross_near(x, s0c)
        s1 = zero_cross_near(x, min(s1, len(x) - 1))
    s1 = min(s1, len(x))
    seg = x[s0c:s1].copy()
    if len(seg) == 0:
        return np.zeros((0, x.shape[1]), np.float32), 0
    seg = fade(seg, int(fade_in_ms / 1000 * SR), int(fade_out_ms / 1000 * SR))
    return seg.astype(np.float32), s0c - s0


def add_at(dst: np.ndarray, seg: np.ndarray, start: int, gain: float = 1.0) -> None:
    """Mix `seg` into `dst` at sample `start` (clipped to dst bounds)."""
    if len(seg) == 0:
        return
    a = max(0, start)
    b = min(len(dst), start + len(seg))
    if b <= a:
        return
    dst[a:b] += seg[a - start:b - start] * gain


def gain_ramp(n: int, points: list[tuple[float, float]]) -> np.ndarray:
    """Piecewise-linear gain (in dB) through (time_s, dB) points, returned as linear gain."""
    t = np.arange(n) / SR
    ts, ds = zip(*points)
    return (10 ** (np.interp(t, ts, ds) / 20)).astype(np.float32)
