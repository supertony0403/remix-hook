"""Deterministic FX: synthesised one-shots plus processing helpers.

Everything is seeded or purely functional, so a rebuild produces bit-identical audio.
All functions return float32 arrays of shape (n, 2) unless noted.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from . import config

SR = config.SR
B1 = 61.735  # Hz, B1 (the "h" an octave above the lowest piano B)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], axis=1).astype(np.float32) if x.ndim == 1 else x.astype(np.float32)


def db(v: float) -> float:
    return float(10 ** (v / 20))


def fade(x: np.ndarray, fin: int, fout: int) -> np.ndarray:
    """Linear-in-amplitude (raised-cosine) fades in samples."""
    y = x.copy()
    n = len(y)
    if fin > 0:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, min(fin, n)))
        y[: len(w)] *= w[:, None] if y.ndim == 2 else w
    if fout > 0:
        w = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, min(fout, n)))
        y[n - len(w):] *= w[:, None] if y.ndim == 2 else w
    return y


# --------------------------------------------------------------------------- filters

def time_varying_filter(x: np.ndarray, cutoff_hz: np.ndarray, btype: str = "lowpass",
                        block: int = 256, order: int = 2) -> np.ndarray:
    """Butterworth filter whose cutoff follows `cutoff_hz` (per sample), processed in blocks.

    The filter state is carried across blocks so the sweep is click-free. Two cascaded
    2nd-order sections give 24 dB/oct.
    """
    x2 = x if x.ndim == 2 else x[:, None]
    out = np.empty_like(x2)
    zi = None
    nyq = SR / 2
    for start in range(0, len(x2), block):
        stop = min(start + block, len(x2))
        fc = float(np.clip(np.median(cutoff_hz[start:stop]), 20.0, nyq * 0.95))
        sos = butter(order, fc, btype=btype, fs=SR, output="sos")
        sos = np.vstack([sos, sos])
        if zi is None:
            zi = np.stack([sosfilt_zi(sos) * x2[0, c] for c in range(x2.shape[1])], axis=-1)
        for c in range(x2.shape[1]):
            out[start:stop, c], zi[..., c] = sosfilt(sos, x2[start:stop, c], zi=zi[..., c])
    return out if x.ndim == 2 else out[:, 0]


def log_sweep(n: int, f0: float, f1: float, curve: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n) ** curve
    return f0 * (f1 / f0) ** t


def highpass(x: np.ndarray, fc: float, order: int = 4) -> np.ndarray:
    sos = butter(order, fc, btype="highpass", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0).astype(np.float32)


def lowpass(x: np.ndarray, fc: float, order: int = 4) -> np.ndarray:
    sos = butter(order, fc, btype="lowpass", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- one-shots

def riser(dur: float, seed: int = 11) -> np.ndarray:
    """Noise riser (band sweeping 400 Hz -> 14 kHz) + saw pitch sweep (B2 -> B5), exp swell."""
    n = int(dur * SR)
    rng = _rng(seed)
    noise = rng.standard_normal((n, 2)).astype(np.float32) * 0.3
    cut = log_sweep(n, 400.0, 14_000.0, curve=1.6)
    noise = time_varying_filter(noise, cut, "lowpass")
    noise = highpass(noise, 250.0, order=2)
    t = np.arange(n) / SR
    f = log_sweep(n, B1 * 2, B1 * 32, curve=1.4)
    phase = 2 * np.pi * np.cumsum(f) / SR
    saw = sum((-1) ** (k + 1) * np.sin(k * phase) / k for k in range(1, 9)) * (2 / np.pi)
    saw = saw * 0.12
    detune = np.sin(phase * 1.006) * 0.05
    tone = np.stack([saw + detune, saw - detune], axis=1)
    env = (t / dur) ** 2.2
    y = (noise + tone) * env[:, None]
    return fade(y, 64, int(0.01 * SR)).astype(np.float32)


def snare_hit(vel: float = 1.0, seed: int = 0) -> np.ndarray:
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    rng = _rng(1000 + seed)
    noise = rng.standard_normal(n)
    sos = butter(2, [1500, 9000], btype="bandpass", fs=SR, output="sos")
    noise = sosfilt(sos, noise) * np.exp(-t / 0.055)
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.03) * 0.6
    y = (noise * 0.9 + body) * vel * 0.5
    return stereo(fade(y, 16, 64))


def snare_roll(grid_beat: float, bars: int, seed: int = 21) -> np.ndarray:
    """Classic build: 1/4 -> 1/8 -> 1/16 -> 1/32 over `bars` bars with rising velocity."""
    bar = grid_beat * 4
    n = int(bars * bar * SR) + int(0.3 * SR)
    out = np.zeros((n, 2), np.float32)
    divisions = {4: [4, 8, 16, 32], 2: [8, 16], 1: [16]}.get(bars, [4] * (bars - 3) + [8, 16, 32])
    hits = [b * bar + k * bar / div for b, div in enumerate(divisions) for k in range(div)]
    total = bars * bar
    for i, t0 in enumerate(hits):
        vel = 0.25 + 0.75 * (t0 / total) ** 1.3
        h = snare_hit(vel, seed=i % 7)
        s = int(round(t0 * SR))
        out[s:s + len(h)] += h[: n - s]
    return out


def kick(vel: float = 1.0) -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 48 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.18)
    click = np.exp(-t / 0.002) * 0.4
    y = np.tanh(1.6 * (body + click)) * vel * 0.8
    return stereo(fade(y, 8, 256))


def kick_build(grid_beat: float, bars: int) -> np.ndarray:
    """Kick pattern accelerating 1/4 -> 1/8 -> 1/16 across `bars` (split in thirds-ish)."""
    bar = grid_beat * 4
    n = int(bars * bar * SR) + int(0.4 * SR)
    out = np.zeros((n, 2), np.float32)
    plan = ([4] * max(1, bars // 2) + [8] * max(1, bars // 4))
    plan += [16] * (bars - len(plan))
    for b, div in enumerate(plan[:bars]):
        for k in range(div):
            t0 = b * bar + k * bar / div
            vel = 0.55 + 0.45 * (t0 / (bars * bar))
            h = kick(vel)
            s = int(round(t0 * SR))
            out[s:s + len(h)] += h[: n - s]
    return out


def sub_drop(dur: float = 2.4, root: float = B1) -> np.ndarray:
    """808-style sub in B: glide from B2 down to B1, long decay, gentle saturation."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = root + root * np.exp(-t / 0.06)  # starts an octave up, lands on root fast
    f = f * (1 - 0.06 * (t / dur))  # slight downward drift at the tail
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.minimum(1.0, t / 0.004) * np.exp(-t / (dur / 3.2))
    y = np.tanh(2.2 * np.sin(ph)) / np.tanh(2.2) * env * 0.9
    return stereo(fade(y, 0, int(0.05 * SR)))


def impact(dur: float = 3.0, seed: int = 31) -> np.ndarray:
    """Boom + noise crack with a long dark reverb tail (pedalboard Reverb, deterministic)."""
    from pedalboard import Pedalboard, Reverb

    n = int(dur * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(40 + 80 * np.exp(-t / 0.05)) / SR) * np.exp(-t / 0.45)
    rng = _rng(seed)
    crack = rng.standard_normal(n) * np.exp(-t / 0.03) * 0.6
    crack = sosfilt(butter(2, [200, 7000], btype="bandpass", fs=SR, output="sos"), crack)
    dry = stereo(boom * 0.8 + crack)
    board = Pedalboard([Reverb(room_size=0.92, damping=0.6, wet_level=0.6, dry_level=0.8, width=1.0)])
    wet = board(dry.T.copy(), SR).T
    wet = lowpass(wet, 6000.0, order=2)
    return fade(wet, 0, int(0.4 * SR)).astype(np.float32)


# --------------------------------------------------------------------------- processing

def tape_stop(x: np.ndarray, dur: float, power: float = 1.6) -> np.ndarray:
    """Playback-rate ramp 1 -> 0 over `dur` seconds (pitch and speed fall together)."""
    n_out = int(dur * SR)
    t = np.arange(n_out) / n_out
    rate = (1.0 - t) ** power
    pos = np.cumsum(rate)
    pos = np.clip(pos, 0, len(x) - 2)
    i0 = pos.astype(int)
    frac = (pos - i0)[:, None]
    y = x[i0] * (1 - frac) + x[i0 + 1] * frac
    return fade(y, 0, int(0.02 * SR)).astype(np.float32)


def sidechain_env(n: int, beat: float, offset: float = 0.0, depth_db: float = -7.0,
                  attack: float = 0.006, release: float = 0.22) -> np.ndarray:
    """Gain envelope ducking on every beat (kick-style pump)."""
    t = np.arange(n) / SR
    ph = np.mod(t - offset, beat)
    dip = 1.0 - db(depth_db)
    g = np.where(ph < attack, 1.0 - dip * (ph / attack),
                 1.0 - dip * np.exp(-(ph - attack) / (release / 3.0)))
    return g.astype(np.float32)


def reverb_send(x: np.ndarray, room: float = 0.5, damping: float = 0.5,
                predelay_ms: float = 40.0, hp_hz: float = 250.0, lp_hz: float = 8000.0) -> np.ndarray:
    """100% wet reverb (pedalboard Reverb = deterministic Freeverb), pre-delay, band-limited."""
    from pedalboard import Pedalboard, Reverb

    pad = int(predelay_ms / 1000 * SR)
    xin = np.vstack([np.zeros((pad, 2), np.float32), x])
    board = Pedalboard([Reverb(room_size=room, damping=damping, wet_level=1.0, dry_level=0.0, width=1.0)])
    y = board(xin.T.copy(), SR).T[: len(x)]  # pre-delay shifts the wet signal later
    y = highpass(y, hp_hz, order=2)
    return lowpass(y, lp_hz, order=2)


def slap_send(x: np.ndarray, delay_s: float, hp_hz: float = 400.0, lp_hz: float = 6000.0) -> np.ndarray:
    """Short slapback: one tap left at `delay_s`, right 12 ms later (Haas width), band-limited."""
    m = x.mean(axis=1)
    out = np.zeros((len(x), 2), np.float32)
    dl, dr = int(delay_s * SR), int((delay_s + 0.012) * SR)
    out[dl:, 0] = m[: len(m) - dl]
    out[dr:, 1] = m[: len(m) - dr]
    out[dl + dl:, 0] += 0.3 * m[: len(m) - 2 * dl]  # faint second tap
    out = highpass(out, hp_hz, order=2)
    return lowpass(out, lp_hz, order=2)


def delay_send(x: np.ndarray, delay_s: float, feedback: float = 0.55, repeats: int = 10,
               pingpong: bool = True, lp_hz: float = 5000.0, width: float = 1.0,
               hp_hz: float = 200.0) -> np.ndarray:
    """Tempo-synced feedback delay, wet only; each repeat darker (one-pole LP per pass)."""
    d = int(round(delay_s * SR))
    n = len(x) + d * (repeats + 1)
    out = np.zeros((n, 2), np.float32)
    tap = x.copy()
    for r in range(1, repeats + 1):
        tap = lowpass(tap, max(1200.0, lp_hz * (0.85 ** r)), order=1) * feedback
        if pingpong:
            tap_st = np.zeros_like(tap)
            ch = (r - 1) % 2
            # width 1.0 = hard ping-pong, 0.0 = mono; "slightly wide" ~0.5
            hi, lo = 0.5 + 0.5 * width, 0.5 - 0.5 * width
            tap_st[:, ch] = tap.mean(axis=1) * hi * 1.2
            tap_st[:, 1 - ch] = tap.mean(axis=1) * max(lo, 0.0) * 1.2 + tap.mean(axis=1) * 0.1
        else:
            tap_st = tap
        s = r * d
        out[s:s + len(tap_st)] += tap_st
    return highpass(out, hp_hz, order=2)
