"""Autotune: snap voiced pitch to the B-minor scale, resynthesise with rubberband R3.

Analysis: pyworld Harvest f0 (5 ms frames) on the mono vocal at 16 kHz, voiced = f0 > 0.
Correction: nearest scale note with hysteresis, smoothed by a one-pole "retune speed",
scaled by `strength` (humanize keeps a little of the natural movement). Unvoiced frames get no
correction. Resynthesis: rubberband -3 -F --freqmap (time-varying pitch with formant
preservation), which keeps consonants and breaths natural and avoids vocoder buzz.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass

import numpy as np
import soxr

from . import config
from .audio_io import read_wav, write_wav

SR = config.SR
FRAME_MS = 5.0
# rubberband R3 applies a --freqmap value ~76 ms before the frame it is written for (measured
# with a sine step at 1.0/2.0/2.7 s: -77.7/-73.4/-77.6 ms); the map is shifted to compensate.
RB_FREQMAP_LEAD_S = 0.076
B_MINOR_PCS = (11, 1, 2, 4, 6, 7, 9)  # B C# D E F# G A


@dataclass(frozen=True)
class TuneStyle:
    retune_ms: float  # time to (~95 %) reach the target note, like an autotune "retune speed"
    strength: float  # 1.0 = hard snap, <1 keeps part of the natural deviation
    hysteresis: float = 0.30  # semitones beyond the midpoint before switching notes


RAP = TuneStyle(retune_ms=25.0, strength=0.85)
REFRAIN = TuneStyle(retune_ms=60.0, strength=0.70)
MAX_CORRECTION_ST = 1.2  # larger "corrections" come from f0 errors and only cause warble
APERIODICITY_MAX = 0.35  # D4C aperiodicity (300-1000 Hz) above this = breathy/transition: leave it
OUTLIER_ST = 2.5  # deviation from the 50 ms median pitch -> f0 error (octave jump), leave it


def f0_track(mono: np.ndarray, sr: int, with_ap: bool = False):
    """(times_s, f0_hz) with 0 for unvoiced frames; optionally the D4C aperiodicity
    (mean over 300-1000 Hz, linear 0..1) as a voicing-strength measure."""
    import pyworld as pw

    y16 = soxr.resample(mono.astype(np.float64), sr, 16_000)
    f0, t = pw.harvest(y16, 16_000, f0_floor=70.0, f0_ceil=900.0, frame_period=FRAME_MS)
    if not with_ap:
        return t, f0
    ap = pw.d4c(y16, f0, t, 16_000)
    bins = np.fft.rfftfreq((ap.shape[1] - 1) * 2, 1 / 16_000)
    band = (bins >= 300) & (bins <= 1000)
    return t, f0, ap[:, band].mean(axis=1)


def reliable_voicing(f0: np.ndarray, ap: np.ndarray) -> np.ndarray:
    """Voiced frames that are clearly periodic and not pitch outliers."""
    from scipy.ndimage import median_filter

    v = f0 > 0
    midi = np.where(v, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(sliding_window_view(np.pad(midi, 5, mode="edge"), 11), axis=1)
    ok = v & (ap < APERIODICITY_MAX) & (np.abs(np.nan_to_num(midi - med, nan=99.0)) < OUTLIER_ST)
    # drop voiced fragments shorter than 4 frames (20 ms): too short to tune, only blips
    out = ok.copy()
    i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]:
                j += 1
            if j - i < 4:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def scale_snap(midi: float, current: float | None, style: TuneStyle,
               pcs: tuple[int, ...] = B_MINOR_PCS) -> float:
    cands = np.array([o * 12 + pc for o in range(1, 9) for pc in pcs], float)
    best = float(cands[np.argmin(np.abs(cands - midi))])
    if current is not None and best != current:
        # stay on the current note until we are clearly past the midpoint to the new one
        if abs(midi - current) < abs(midi - best) + 2 * style.hysteresis:
            return current
    return best


def correction_curve(f0: np.ndarray, styles: np.ndarray, style_map: dict[int, TuneStyle]) -> np.ndarray:
    """Per-frame correction in semitones. `styles` holds a style id per frame (0 = untouched).

    Like a hardware autotune: the OUTPUT pitch glides towards the held scale note with the
    retune time constant (so a held note sits exactly on pitch regardless of input wobble),
    and humanize (1 - strength) lets part of the natural deviation through. Unvoiced frames
    get no correction; the correction releases to 0 over ~10 ms instead of jumping.
    """
    n = len(f0)
    voiced = f0 > 0
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), 0.0)
    # note decisions on a 25 ms median over VOICED frames only (zeros of unvoiced frames would
    # drag the median to MIDI 0 on short runs and snap to a nonsense note)
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(np.where(voiced, midi, np.nan), 2, mode="edge")
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(sliding_window_view(padded, 5), axis=1)
    midi_dec = np.where(voiced, np.where(np.isnan(med), midi, med), 0.0)
    corr = np.zeros(n)
    cur: float | None = None
    out_pitch = 0.0
    rel = 1.0 - np.exp(-FRAME_MS / 10.0)
    for i in range(n):
        sid = int(styles[i])
        if not voiced[i] or sid == 0:
            cur = None
            prev = corr[i - 1] if i else 0.0
            corr[i] = prev * (1.0 - rel)
            continue
        st = style_map[sid]
        new_note = scale_snap(midi_dec[i], cur, st)
        if cur is None:  # voicing onset: start on the note, no audible scoop
            out_pitch = new_note
        cur = new_note
        a = 1.0 - np.exp(-FRAME_MS / (st.retune_ms / 3.0))
        out_pitch += a * (cur - out_pitch)
        desired = out_pitch + (1.0 - st.strength) * (midi[i] - cur)
        corr[i] = float(np.clip(desired - midi[i], -MAX_CORRECTION_ST, MAX_CORRECTION_ST))
    return corr


def in_scale_fraction(f0: np.ndarray, cents: float = 30.0, pcs: tuple[int, ...] = B_MINOR_PCS) -> float:
    """Share of voiced frames within +-cents of a scale tone."""
    v = f0[f0 > 0]
    if len(v) == 0:
        return float("nan")
    midi = 69 + 12 * np.log2(v / 440.0)
    cands = np.array([o * 12 + pc for o in range(0, 10) for pc in pcs], float)
    dev = np.min(np.abs(midi[:, None] - cands[None, :]), axis=1) * 100
    return float(np.mean(dev <= cents))


def autotune(x: np.ndarray, regions: list[tuple[float, float, TuneStyle]]) -> tuple[np.ndarray, dict]:
    """Tune stereo vocal `x` (48 kHz) inside `regions` (seconds, style). Cached in work/."""
    mono = x.mean(axis=1)
    t, f0_all, ap = f0_track(mono, SR, with_ap=True)
    f0 = np.where(reliable_voicing(f0_all, ap), f0_all, 0.0)
    style_map: dict[int, TuneStyle] = {}
    styles = np.zeros(len(t), int)
    for k, (a, b, st) in enumerate(regions, start=1):
        style_map[k] = st
        styles[(t >= a) & (t < b)] = k
    corr = correction_curve(f0, styles, style_map)
    key = hashlib.sha1(np.ascontiguousarray(x, np.float32).tobytes() + np.round(corr, 5).tobytes()
                       + f"{RB_FREQMAP_LEAD_S}|-q -3 -F|{x.shape}".encode()).hexdigest()[:16]
    out_path = config.WORK / "tune" / f"{key}.wav"
    if not out_path.exists():
        src = config.WORK / "tune" / f"{key}_in.wav"
        fmap = config.WORK / "tune" / f"{key}.freqmap"
        write_wav(src, x, SR)
        frames = np.round((t + RB_FREQMAP_LEAD_S) * SR).astype(int)
        frames = np.r_[0, frames]
        mult = 2 ** (np.r_[0.0, corr] / 12)
        fmap.write_text("".join(f"{fr} {m:.6f}\n" for fr, m in zip(frames, mult)))
        subprocess.run(["rubberband", "-q", "-3", "-F", "--freqmap", str(fmap), "-t", "1.0",
                        str(src), str(out_path)], check=True, capture_output=True)
        src.unlink(missing_ok=True)
    y, _ = read_wav(out_path)
    if len(y) < len(x):
        y = np.vstack([y, np.zeros((len(x) - len(y), y.shape[1]), np.float32)])
    y = y[: len(x)]
    in_reg = styles > 0
    stats = {
        "voiced_frames": int(np.sum((f0_all > 0) & in_reg)),
        "tuned_frames": int(np.sum((f0 > 0) & in_reg)),
        "in_scale_before": in_scale_fraction(f0[in_reg]),
        "mean_abs_correction_st": float(np.mean(np.abs(corr[(f0 > 0) & in_reg]))),
    }
    return y, stats
