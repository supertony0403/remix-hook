"""Vocal chain shared by B's rap/refrain vocals (A2) and A's hook (A3/A4).

HPF 100 Hz -> expander/gate between phrases -> de-esser (split band 5.5 kHz+) -> fast 4:1
compressor -> parallel saturation -> presence EQ (+2.5 dB @ 4 kHz) and optional notches ->
mono (vocals sit dead centre; width comes from the slap/delay returns).
All stages are deterministic (pedalboard DSP or numpy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pedalboard import (Compressor, Distortion, HighpassFilter, HighShelfFilter, NoiseGate,
                        Pedalboard, PeakFilter)
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from . import config

SR = config.SR


@dataclass(frozen=True)
class ChainSettings:
    hpf_hz: float = 100.0
    gate_threshold_db: float = -42.0
    gate_ratio: float = 3.0  # downward expander
    gate_release_ms: float = 90.0
    deess_hz: float = 5500.0
    deess_threshold_db: float = -30.0  # sibilant band level where reduction starts
    deess_max_db: float = 7.0
    comp_threshold_db: float = -22.0
    comp_ratio: float = 4.0
    comp_attack_ms: float = 3.0
    comp_release_ms: float = 60.0
    sat_drive_db: float = 10.0
    sat_mix: float = 0.22
    presence_hz: float = 4000.0
    presence_db: float = 2.5
    air_db: float = 1.0
    notches: tuple[tuple[float, float, float], ...] = ()  # (hz, q, gain_db)
    level_offset_db: float = 1.0  # output loudness relative to input (vocals a touch forward)


B_VOCALS = ChainSettings()
A_HOOK = ChainSettings(gate_threshold_db=-40.0, gate_ratio=4.0, sat_drive_db=12.0, sat_mix=0.28,
                       presence_db=3.0)
A_CHOPS = ChainSettings(gate_threshold_db=-90.0, gate_ratio=1.0, sat_drive_db=12.0, sat_mix=0.25,
                        presence_db=3.0, level_offset_db=0.0)


def _env_db(x: np.ndarray, win_ms: float) -> np.ndarray:
    p = uniform_filter1d(x.astype(np.float64) ** 2, size=max(1, int(win_ms / 1000 * SR)))
    return 10 * np.log10(p + 1e-12)


def deess(x: np.ndarray, hz: float, threshold_db: float, max_db: float) -> np.ndarray:
    """Split-band de-esser: only the band above `hz` is attenuated, by how far it exceeds the
    threshold (2:1 above threshold), smoothed 1 ms attack / 40 ms release."""
    sos = butter(4, hz, btype="highpass", fs=SR, output="sos")
    hi = sosfiltfilt(sos, x, axis=0)
    lo = x - hi  # zero-phase split -> exact complement
    lvl = _env_db(hi.mean(axis=1), 2.0)
    red = np.clip((lvl - threshold_db) * 0.5, 0.0, max_db)
    # hold each reduction for 40 ms (backward-looking max), then smooth 2 ms
    L = int(0.04 * SR)
    red = maximum_filter1d(red, size=L + 1, origin=L // 2, mode="nearest")
    red = uniform_filter1d(red, size=int(0.002 * SR))
    g = 10 ** (-red / 20)
    return (lo + hi * g[:, None]).astype(np.float32)


def process(x: np.ndarray, s: ChainSettings) -> np.ndarray:
    """Run the chain on a full track buffer (stereo in, centred stereo out)."""
    if not np.any(x):
        return x
    mono = x.mean(axis=1, keepdims=True).astype(np.float32)
    pre = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=s.hpf_hz),
        NoiseGate(threshold_db=s.gate_threshold_db, ratio=s.gate_ratio, attack_ms=1.5,
                  release_ms=s.gate_release_ms),
    ])
    y = pre(mono.T.copy(), SR).T
    y = deess(y, s.deess_hz, s.deess_threshold_db, s.deess_max_db)
    comp = Pedalboard([Compressor(threshold_db=s.comp_threshold_db, ratio=s.comp_ratio,
                                  attack_ms=s.comp_attack_ms, release_ms=s.comp_release_ms)])
    y = comp(y.T.copy(), SR).T
    sat = Pedalboard([Distortion(drive_db=s.sat_drive_db)])(y.T.copy(), SR).T
    # level-match the saturated branch before blending (Distortion adds gain)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)) + 1e-12)  # noqa: E731
    sat *= rms(y) / rms(sat)
    y = (1 - s.sat_mix) * y + s.sat_mix * sat
    eq = [PeakFilter(cutoff_frequency_hz=s.presence_hz, gain_db=s.presence_db, q=0.8),
          HighShelfFilter(cutoff_frequency_hz=10_000.0, gain_db=s.air_db, q=0.7)]
    eq += [PeakFilter(cutoff_frequency_hz=f, gain_db=g, q=q) for f, q, g in s.notches]
    y = Pedalboard(eq)(y.T.copy(), SR).T
    # loudness-preserving chain: auto make-up to the input loudness (+ offset)
    import pyloudnorm as pyln

    meter = pyln.Meter(SR)
    l_in = meter.integrated_loudness(np.repeat(mono, 2, axis=1))
    l_out = meter.integrated_loudness(np.repeat(y, 2, axis=1))
    if np.isfinite(l_in) and np.isfinite(l_out):
        y = y * 10 ** ((l_in - l_out + s.level_offset_db) / 20)
    return np.repeat(y.astype(np.float32), 2, axis=1)


def gap_to_phrase_ratio_db(x: np.ndarray, frame_ms: float = 50.0, gap_pct: float = 20.0,
                           phrase_pct: float = 90.0) -> float:
    """Direct-vs-room proxy: level in the phrase pauses relative to the phrase level.

    Frames of the active span are ranked by level; the `gap_pct` percentile (pauses, where
    only reverb/tails/bleed remain) minus the `phrase_pct` percentile (sung phrases).
    More negative = drier.
    """
    m = x.mean(axis=1)
    f = int(frame_ms / 1000 * SR)
    n = len(m) // f
    lv = 10 * np.log10(np.mean(m[: n * f].reshape(n, f) ** 2, axis=1) + 1e-12)
    active = lv > lv.max() - 60
    if not active.any():
        return float("nan")
    v = lv[active]
    return float(np.percentile(v, gap_pct) - np.percentile(v, phrase_pct))
