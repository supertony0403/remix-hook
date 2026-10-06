"""Render the arrangement into six continuous, equally long track buffers (48 kHz stereo float)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pyloudnorm as pyln

from . import config, fx as F
from .arrangement import (A_BAR, A_DOWNBEAT0, ALL, BAR, BEAT, B_DOWNBEAT0, HOOK_LEVEL_REF,
                          REFRAIN_LEVEL_REF, STRETCH, TRANSIENT_LEAD_S, Arrangement, Clip)
from .audio_io import read_wav, write_wav
from .edit import add_at, cut
from .stems import load_stems
from .tune import RAP, REFRAIN, autotune
from .vocalchain import A_CHOPS, A_HOOK, B_VOCALS
from .vocalchain import process as vocal_chain
from .vocals import rubberband

log = logging.getLogger(__name__)
SR = config.SR

A_RB_PATH = config.WORK / "a_vocals_rb.wav"
A_VOCAL_TARGET_OFFSET_DB = -1.0  # A vocals sit 1 dB under B's refrain-3 vocal loudness
DELAY_S = 0.5 * BEAT  # 1/8 note (0.2 s)
LONG_DELAY_S = 0.75 * BEAT  # dotted 1/8 for the outro feedback throw
SLAP_S = 0.25 * BEAT  # 1/16 note = 100 ms slap
REVERB_ROOM = 0.5  # RT60 ~1.0 s (v1: 0.86 -> ~2.9 s)
REVERB_PREDELAY_MS = 40.0
CHAINS = {"A2_b_gesang": B_VOCALS, "A3_a_hook": A_HOOK, "A4_chops": A_CHOPS}
TUNE_STYLES = {"rap": RAP, "refrain": REFRAIN}


@dataclass
class Sources:
    b: dict[str, np.ndarray]
    a_vocals: np.ndarray
    fx: dict[str, np.ndarray] = field(default_factory=dict)
    tune_stats: dict = field(default_factory=dict)


def a_rb_vocals() -> np.ndarray:
    """A vocals stretched 128 -> 150 BPM and shifted -2 semitones (formants kept), cached."""
    y, _ = read_wav(config.STEMS / "a" / "vocals.wav")
    z = rubberband(y, STRETCH, config.SEMITONES_A, formant=True)  # keyed cache in work/rb
    write_wav(A_RB_PATH, z, SR)  # convenience copy for listening
    return z


# loudness (LUFS over the one-shot's own duration) each synthesised FX is normalised to
FX_LUFS = {"riser": -20.0, "snare_roll": -21.0, "kick_build": -19.0, "sub_drop": -21.0,
           "impact": -21.0}


def _synth_fx(name: str) -> np.ndarray:
    if name == "riser":
        return F.riser(3.75 * BAR)
    if name.startswith("snare_roll"):
        return F.snare_roll(BEAT, int(name.removeprefix("snare_roll")))
    if name.startswith("kick_build"):
        return F.kick_build(BEAT, int(name.removeprefix("kick_build")))
    if name == "sub_drop":
        return F.sub_drop(2.4)
    if name == "impact":
        return F.impact(4.0)
    raise KeyError(name)


def make_fx(name: str) -> np.ndarray:
    x = _synth_fx(name)
    family = next(k for k in FX_LUFS if name.startswith(k))
    return (x * 10 ** ((FX_LUFS[family] - loudness(x)) / 20)).astype(np.float32)


def tune_regions(arr: Arrangement) -> list[tuple[float, float, object]]:
    """B-time regions (seconds) to autotune, from the B vocal clips' `tune` fx."""
    regions = []
    for c in arr.clips:
        st = next((f for f in c.fx if f.kind == "tune"), None)
        if c.source == "b:vocals" and st is not None:
            regions.append((source_time(c.source, c.start) - 0.05, source_time(c.source, c.end) + 0.05,
                            TUNE_STYLES[str(st.get("style"))]))
    return sorted(regions, key=lambda r: r[0])


def load_sources(arr: Arrangement, tuned: bool = True) -> Sources:
    src = Sources(b=load_stems("b"), a_vocals=a_rb_vocals())
    if tuned:
        src.b["vocals_raw"] = src.b["vocals"]
        src.b["vocals"], src.tune_stats = autotune(src.b["vocals"], tune_regions(arr))
    for c in arr.clips:
        if c.source.startswith("fx:"):
            name = c.source[3:]
            if name not in src.fx:
                src.fx[name] = make_fx(name)
    return src


# --------------------------------------------------------------------------- time bases

def source_time(source: str, bar: float) -> float:
    if source.startswith("b:"):
        return B_DOWNBEAT0 + bar * BAR
    if source == "a:vocals":
        return (A_DOWNBEAT0 + bar * A_BAR) * STRETCH
    if source.startswith("fx:"):
        return bar * BAR
    raise KeyError(source)


def source_array(src: Sources, source: str) -> np.ndarray:
    if source.startswith("b:"):
        return src.b[source[2:]]
    if source == "a:vocals":
        return src.a_vocals
    return src.fx[source[3:]]


def remix_samples(bars: float) -> int:
    return int(round(bars * BAR * SR))


# --------------------------------------------------------------------------- levels

def loudness(x: np.ndarray) -> float:
    meter = pyln.Meter(SR)
    return float(meter.integrated_loudness(x))


def level_trims(src: Sources) -> dict[str, float]:
    """Gain (dB) that brings A's hook / refrain to B's refrain vocal loudness + offset."""
    bv = src.b["vocals"]
    t0, t1 = source_time("b:vocals", 72), source_time("b:vocals", 88)
    target = loudness(bv[int(t0 * SR):int(t1 * SR)]) + A_VOCAL_TARGET_OFFSET_DB
    out = {}
    for name, (b0, b1) in (("hook", HOOK_LEVEL_REF), ("refrain", REFRAIN_LEVEL_REF)):
        s0, s1 = source_time("a:vocals", b0), source_time("a:vocals", b1)
        out[name] = target - loudness(src.a_vocals[int(s0 * SR):int(s1 * SR)])
    out["target_lufs"] = target
    out["b_ref_lufs"] = target - A_VOCAL_TARGET_OFFSET_DB
    return out


B_RIDE_MAX_DB = 4.0


def b_ride_trim(seg: np.ndarray, ref_lufs: float, offset_db: float) -> float:
    """Gentle vocal ride for B: bring a section to the reference (+offset), clamped."""
    return float(np.clip(ref_lufs + offset_db - loudness(seg), -B_RIDE_MAX_DB, B_RIDE_MAX_DB))


# --------------------------------------------------------------------------- rendering

@dataclass
class Render:
    tracks: dict[str, np.ndarray]
    trims: dict[str, float]
    n: int
    tune: dict = field(default_factory=dict)


def _clip_audio(c: Clip, src: Sources, trims: dict[str, float]) -> tuple[np.ndarray, int]:
    """Cut, process and level one clip; returns (audio, remix start sample)."""
    x = source_array(src, c.source)
    t0, t1 = source_time(c.source, c.start), source_time(c.source, c.end)
    pitch = next((f for f in c.fx if f.kind == "pitch"), None)
    if pitch is not None:
        pad = 0.08
        a0 = max(0, int((t0 - pad) * SR))
        a1 = min(len(x), int((t1 + pad) * SR))
        shifted = rubberband(x[a0:a1], 1.0, float(pitch.get("semitones")), formant=True)
        x, t0, t1 = shifted, t0 - a0 / SR, t1 - a0 / SR
    seg, shift = cut(x, t0, t1)
    at = remix_samples(c.remix_start) + shift
    if c.source.startswith("fx:"):
        at -= int(round(TRANSIENT_LEAD_S * SR))
    g = 10 ** (c.gain_db / 20)
    for f in c.fx:
        if f.kind == "level":
            g *= 10 ** (trims[str(f.get("ref"))] / 20)
        elif f.kind == "ride":
            g *= 10 ** (b_ride_trim(seg, trims["b_ref_lufs"], float(f.get("offset_db", 0.0))) / 20)
        elif f.kind == "hpf":
            seg = F.highpass(seg, float(f.get("hz")), order=2)
        elif f.kind == "lpf":
            seg = F.lowpass(seg, float(f.get("hz")), order=2)
        elif f.kind == "hpf_sweep":
            cutoff = F.log_sweep(len(seg), float(f.get("f0")), float(f.get("f1")),
                                 float(f.get("curve", 1.0)))
            seg = F.time_varying_filter(seg, cutoff, "highpass")
    return (seg * g).astype(np.float32), at


def _filter_sweep(x: np.ndarray, s0: int, s1: int, f0: float, f1: float, curve: float,
                  btype: str) -> None:
    """In-place cutoff sweep over [s0, s1) with 6 ms crossfades in and out of the filter."""
    xf = int(0.006 * SR)
    pre = int(0.05 * SR)
    s0, s1 = max(0, min(s0, len(x))), max(0, min(s1, len(x)))
    if s1 <= s0:
        return
    lo, hi = max(0, s0 - pre), min(len(x), s1 + xf)
    sweep = F.log_sweep(s1 - s0, f0, f1, curve)
    cutoff = np.r_[np.full(s0 - lo, f0), sweep, np.full(hi - s1, f1)]
    filt = F.time_varying_filter(x[lo:hi], cutoff, btype)
    w = np.zeros(hi - lo, np.float32)
    a = max(lo, s0 - xf)
    w[a - lo:s0 - lo] = np.linspace(0.0, 1.0, s0 - a, dtype=np.float32)
    w[s0 - lo:s1 - lo] = 1.0
    w[s1 - lo:hi - lo] = np.linspace(1.0, 0.0, hi - s1, dtype=np.float32)
    x[lo:hi] = x[lo:hi] * (1 - w[:, None]) + filt * w[:, None]


def _mute(x: np.ndarray, s0: int, s1: int) -> None:
    """Silence [s0, s1'): 6 ms fade-out before s0; the silence ends 8 ms before the following
    downbeat's physical transient (TRANSIENT_LEAD_S before the bar line) with a 2 ms fade-in,
    so the drop's attack is never cut."""
    fo, fi = int(0.006 * SR), int(0.002 * SR)
    s1 = s1 - int(round((TRANSIENT_LEAD_S + 0.008) * SR))
    s0, s1 = max(0, min(s0, len(x))), max(0, min(s1, len(x)))
    if s1 - s0 <= fi:
        return
    a = max(0, s0 - fo)
    x[a:s0] *= np.linspace(1.0, 0.0, s0 - a, dtype=np.float32)[:, None]
    x[s0:s1 - fi] = 0.0
    x[s1 - fi:s1] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]


def _apply_automation(tracks: dict[str, np.ndarray], arr: Arrangement) -> None:
    for a in arr.automation:
        s0, s1 = remix_samples(a.start), remix_samples(a.end)
        for name in a.tracks:
            x = tracks[name]
            if a.kind == "lpf":
                _filter_sweep(x, s0, s1, float(a.get("f0")), float(a.get("f1")),
                              float(a.get("curve", 1.0)), "lowpass")
            elif a.kind == "sidechain":
                env = F.sidechain_env(s1 - s0, BEAT, 0.0, float(a.get("depth_db")))
                x[s0:s1] *= env[:, None]
            elif a.kind == "tape_stop":
                stopped = F.tape_stop(x[s0:s1].copy(), s1 - s0, float(a.get("power", 1.6)))
                x[s0:s1] = stopped
            elif a.kind == "mute":
                _mute(x, s0, s1)
            else:
                raise KeyError(a.kind)


def _returns(buses: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros_like(next(iter(buses.values())))
    if np.any(buses["reverb"]):
        out += F.reverb_send(buses["reverb"], room=REVERB_ROOM, damping=0.5,
                             predelay_ms=REVERB_PREDELAY_MS, hp_hz=250.0)
    if np.any(buses["slap"]):
        out += F.slap_send(buses["slap"], SLAP_S)
    if np.any(buses["delay"]):
        out += F.delay_send(buses["delay"], DELAY_S, feedback=0.35, repeats=5, width=0.5,
                            hp_hz=350.0)[: len(out)]
    if np.any(buses["delay_long"]):
        d = F.delay_send(buses["delay_long"], LONG_DELAY_S, feedback=0.58, repeats=12,
                         lp_hz=4500.0, width=0.6, hp_hz=300.0)[: len(out)]
        out += d + 0.35 * F.reverb_send(d, room=0.55, damping=0.5, predelay_ms=REVERB_PREDELAY_MS,
                                        hp_hz=250.0)
    return out.astype(np.float32)


@dataclass
class Send:
    track: str
    bus: str
    start: int
    stop: int
    gain: float
    fade_out: int


def render(arr: Arrangement, src: Sources | None = None) -> Render:
    src = src or load_sources(arr)
    trims = level_trims(src)
    n = remix_samples(arr.total_bars + arr.tail_bars)
    tracks = {name: np.zeros((n, 2), np.float32) for name in ALL}
    buses = {k: np.zeros((n, 2), np.float32) for k in ("reverb", "slap", "delay", "delay_long")}
    sends: list[Send] = []

    for c in arr.clips:
        seg, at = _clip_audio(c, src, trims)
        add_at(tracks[c.track], seg, at)
        for f in c.fx:
            if f.kind == "send":
                sends.append(Send(c.track, str(f.get("bus")), at, at + len(seg),
                                  10 ** (float(f.get("db")) / 20), int(0.01 * SR)))
            elif f.kind == "throw":
                base = remix_samples(c.remix_start)  # musical start, independent of the snap shift
                r0 = int(round((float(f.get("start")) - c.start) * BAR * SR))
                r1 = int(round((float(f.get("end")) - c.start) * BAR * SR))
                sends.append(Send(c.track, str(f.get("bus")), max(at, base + r0), base + r1,
                                  10 ** (float(f.get("db")) / 20), int(0.03 * SR)))

    # vocal chains run on the whole track (continuous compressor/gate state)
    for name, settings in CHAINS.items():
        tracks[name] = vocal_chain(tracks[name], settings)
    # sends are taken post-chain so the room hears the processed voice
    for sd in sends:
        part = F.fade(tracks[sd.track][max(0, sd.start):sd.stop], int(0.005 * SR), sd.fade_out)
        add_at(buses[sd.bus], part, max(0, sd.start), sd.gain)

    _apply_automation(tracks, arr)
    # the bar-15 silence and the pre-drop silence also hold for the returns
    tracks[ALL[5]] = _returns(buses)
    for a in arr.automation:
        if a.kind == "mute" and set(a.tracks) == set(ALL):
            _mute(tracks[ALL[5]], remix_samples(a.start), remix_samples(a.end))
    log.info("trims %s", {k: round(v, 2) for k, v in trims.items()})
    return Render(tracks=tracks, trims=trims, n=n, tune=src.tune_stats)
