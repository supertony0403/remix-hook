"""Bus processing, loudness targeting and a true-peak lookahead limiter.

The limiter is computed as a gain envelope g(t) on the summed mix. The same master gain and the
same g(t) are applied to every track stem, so sum(stems) == master holds sample-exactly and
DaVinci Resolve's plain 0 dB summation of A1-A6 reproduces the master without its own limiter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

from . import config

SR = config.SR


def true_peak_env(x: np.ndarray, os: int = 4, chunk: int = 480_000) -> np.ndarray:
    """Per-sample true-peak envelope (max over channels of the 4x oversampled |signal|)."""
    pad = 256
    n = len(x)
    out = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        lo, hi = max(0, a - pad), min(n, b + pad)
        up = resample_poly(x[lo:hi].astype(np.float64), os, 1, axis=0)
        env = np.abs(up).max(axis=1)
        env = env[: (hi - lo) * os].reshape(hi - lo, os).max(axis=1)
        out[a:b] = env[a - lo:b - lo]
    return np.maximum(out, np.abs(x).max(axis=1))


def true_peak_db(x: np.ndarray) -> float:
    return float(20 * np.log10(true_peak_env(x).max() + 1e-12))


def limiter_gain(x: np.ndarray, ceiling_db: float, lookahead_ms: float = 5.0,
                 release_ms: float = 140.0, true_peak: bool = True, block: int = 16) -> np.ndarray:
    """Gain envelope g with true_peak(x * g) <= ceiling (lookahead min + box smoothing)."""
    ceiling = 10 ** (ceiling_db / 20)
    env = true_peak_env(x) if true_peak else np.abs(x).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(env, 1e-9)).astype(np.float64)
    L = max(1, int(lookahead_ms / 1000 * SR))
    # g1[i] = min(need[i .. i+L])  (forward-looking hold)
    g1 = minimum_filter1d(need, size=L + 1, origin=-(L // 2), mode="nearest")
    # release: block-rate recursion g[k] = min(g1[k], g[k-1] + (1-g[k-1]) * a)
    nb = int(np.ceil(len(g1) / block))
    padded = np.r_[g1, np.ones(nb * block - len(g1))]
    gb = padded.reshape(nb, block).min(axis=1)
    a = 1.0 - np.exp(-block / (release_ms / 1000 * SR))
    rel = np.empty(nb)
    prev = 1.0
    for k in range(nb):
        v = prev + (1.0 - prev) * a
        prev = gb[k] if gb[k] < v else v
        rel[k] = prev
    g1 = np.minimum(g1, np.repeat(rel, block)[: len(g1)])
    # backward box average of length L+1: every sample of the window <= need at the peak
    g2 = uniform_filter1d(g1, size=L + 1, origin=L // 2, mode="nearest")
    return np.minimum(g2, 1.0).astype(np.float32)


def integrated_lufs(x: np.ndarray) -> float:
    return float(pyln.Meter(SR).integrated_loudness(x))


def bus_peak_control(x: np.ndarray, percentile: float, release_ms: float = 60.0) -> np.ndarray:
    """Tame isolated spikes of a stem (B's source has overs up to +6 dBFS): ceiling at the given
    percentile of |x|, sample-peak based, fast release. Returns the processed stem."""
    a = np.abs(x).max(axis=1)
    active = a[a > 1e-4]
    if len(active) == 0:
        return x
    ceil_db = 20 * np.log10(np.percentile(active, percentile))
    g = limiter_gain(x, ceil_db, lookahead_ms=2.0, release_ms=release_ms, true_peak=False)
    return (x * g[:, None]).astype(np.float32)


@dataclass
class MasterResult:
    gain_db: float  # static master gain applied to every stem
    envelope: np.ndarray  # limiter gain envelope applied to every stem
    master: np.ndarray
    lufs: float
    dbtp: float
    gr_mean_db: float  # mean gain reduction over non-silent samples
    gr_mean_loud_db: float  # mean GR where the limiter is active (>0.1 dB)
    gr_max_db: float


def master(mix: np.ndarray, target_lufs: float = config.TARGET_LUFS,
           ceiling_dbtp: float = config.TARGET_DBTP - 0.25, iters: int = 8) -> MasterResult:
    gain_db = target_lufs - integrated_lufs(mix)
    for _ in range(iters):
        y = mix * 10 ** (gain_db / 20)
        g = limiter_gain(y, ceiling_dbtp)
        out = y * g[:, None]
        lufs = integrated_lufs(out)
        err = target_lufs - lufs
        if abs(err) < 0.03:
            break
        gain_db += err * 1.15
    dbtp = true_peak_db(out)
    gr = -20 * np.log10(np.maximum(g, 1e-9))
    lvl = 20 * np.log10(np.abs(y).max(axis=1) + 1e-9)
    loud = lvl > -50.0
    active = gr > 0.1
    return MasterResult(
        gain_db=float(gain_db), envelope=g, master=out.astype(np.float32), lufs=lufs,
        dbtp=dbtp, gr_mean_db=float(gr[loud].mean()),
        gr_mean_loud_db=float(gr[active].mean()) if active.any() else 0.0,
        gr_max_db=float(gr.max()),
    )


# --------------------------------------------------------------------------- v2 master chain

def _smooth_gr(gr_db: np.ndarray, attack_ms: float, release_ms: float, block: int = 16) -> np.ndarray:
    """Attack/release smoothing of a gain-reduction curve (dB, >= 0) at block rate."""
    nb = int(np.ceil(len(gr_db) / block))
    padded = np.r_[gr_db, np.zeros(nb * block - len(gr_db))]
    gb = padded.reshape(nb, block).max(axis=1)
    aa = 1.0 - np.exp(-block / (attack_ms / 1000 * SR))
    ar = 1.0 - np.exp(-block / (release_ms / 1000 * SR))
    out = np.empty(nb)
    prev = 0.0
    for k in range(nb):
        a = aa if gb[k] > prev else ar
        prev += a * (gb[k] - prev)
        out[k] = prev
    sm = np.repeat(out, block)[: len(gr_db)]
    return uniform_filter1d(sm, size=block * 2)


def glue_comp_gain(x: np.ndarray, threshold_db: float, ratio: float = 1.6, attack_ms: float = 20.0,
                   release_ms: float = 200.0, window_ms: float = 50.0) -> np.ndarray:
    """Stereo-linked RMS "glue" bus compressor (50 ms detector) as a gain envelope."""
    p = uniform_filter1d((x.astype(np.float64) ** 2).mean(axis=1), size=int(window_ms / 1000 * SR))
    lvl = 10 * np.log10(p + 1e-12)
    gr = np.maximum(0.0, lvl - threshold_db) * (1 - 1 / ratio)
    gr = _smooth_gr(gr, attack_ms, release_ms)
    return (10 ** (-gr / 20)).astype(np.float32)


def soft_clip_gain(x: np.ndarray, threshold_db: float, ceiling_db: float) -> np.ndarray:
    """Per-sample, per-channel gain that realises a tanh soft clipper (identity below
    `threshold_db`, asymptote at `ceiling_db`). Returned as gain so it can be applied to stems."""
    t = 10 ** (threshold_db / 20)
    c = 10 ** (ceiling_db / 20)
    a = np.abs(x)
    over = a > t
    y = np.where(over, t + (c - t) * np.tanh((a - t) / (c - t)), a)
    g = np.ones_like(x, dtype=np.float32)
    g[over] = (y[over] / a[over]).astype(np.float32)
    return g


@dataclass
class ChainResult:
    pre_gain_db: float
    gain: np.ndarray  # (n, 2) total per-sample gain applied to every stem
    master: np.ndarray
    lufs: float
    dbtp: float
    comp_gr_mean_db: float
    comp_gr_max_db: float
    clip_gr_max_db: float
    clip_share: float  # share of samples touched by the clipper
    lim_gr_mean_db: float
    lim_gr_max_db: float


def master_chain(mix: np.ndarray, target_lufs: float = config.TARGET_LUFS,
                 ceiling_dbtp: float = config.TARGET_DBTP - 0.25, comp_threshold_db: float = -10.0,
                 clip_threshold_db: float = -4.0, clip_ceiling_db: float = 0.0,
                 iters: int = 8) -> ChainResult:
    """Glue compressor (1.6:1) -> soft clip saturation -> true-peak limiter, loudness-targeted."""
    pre = target_lufs - integrated_lufs(mix)
    for _ in range(iters):
        y0 = mix * 10 ** (pre / 20)
        gc = glue_comp_gain(y0, comp_threshold_db)
        y1 = y0 * gc[:, None]
        gs = soft_clip_gain(y1, clip_threshold_db, clip_ceiling_db)
        y2 = y1 * gs
        gl = limiter_gain(y2, ceiling_dbtp, release_ms=120.0)
        out = y2 * gl[:, None]
        lufs = integrated_lufs(out)
        err = target_lufs - lufs
        if abs(err) < 0.03:
            break
        pre += err * 1.1
    gain = (10 ** (pre / 20)) * gc[:, None] * gs * gl[:, None]
    loud = 20 * np.log10(np.abs(y0).max(axis=1) + 1e-9) > -50
    comp_gr = -20 * np.log10(gc)
    lim_gr = -20 * np.log10(np.maximum(gl, 1e-9))
    clip_gr = -20 * np.log10(np.maximum(gs.min(axis=1), 1e-9))
    return ChainResult(
        pre_gain_db=float(pre), gain=gain.astype(np.float32), master=out.astype(np.float32),
        lufs=float(lufs), dbtp=true_peak_db(out),
        comp_gr_mean_db=float(comp_gr[loud].mean()), comp_gr_max_db=float(comp_gr.max()),
        clip_gr_max_db=float(clip_gr.max()), clip_share=float(np.mean(clip_gr > 0.01)),
        lim_gr_mean_db=float(lim_gr[loud].mean()), lim_gr_max_db=float(lim_gr.max()),
    )
