"""Track bus processing, master, and export (track WAVs, master WAV/MP3, report data)."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .audio_io import write_wav
from pedalboard import LowShelfFilter, Pedalboard
from scipy.signal import butter, sosfiltfilt

from .master import ChainResult, bus_peak_control, master_chain, true_peak_db

log = logging.getLogger(__name__)
SR = config.SR

# spike control per track (percentile of |x| over active samples used as the ceiling)
BUS_PEAK = {
    "A1_b_instrumental": 99.98,  # only the very top drum transients (kick must still knock)
    "A2_b_gesang": 99.95,  # B's vocal stem carries overs up to +5.6 dBFS (chain handles most)
    "A3_a_hook": 99.97,
}
LOW_SHELF_DB = 1.5  # kick/bass weight on the instrumental
DUCK_DB = 3.0  # instrumental 1-4 kHz ducked under the vocals
DUCK_BAND = (1000.0, 4000.0)
VOCAL_TRACKS = ("A2_b_gesang", "A3_a_hook", "A4_chops")
STEM_PEAK_MAX_DBFS = -0.3
# static track trims (dB) after the arrangement's own clip gains
TRACK_TRIM_DB = {
    "A1_b_instrumental": 0.0,
    "A2_b_gesang": 0.0,
    "A3_a_hook": 0.0,
    "A4_chops": -1.0,
    "A5_fx": 0.0,
    "A6_returns": 0.0,
}

TITLE = "Don't You Feel (Remix)"


@dataclass
class MixResult:
    tracks: dict[str, np.ndarray]  # final, mastered stems (sum == master)
    result: ChainResult
    duck_max_db: float


def vocal_duck(inst: np.ndarray, key: np.ndarray, depth_db: float = DUCK_DB,
               band: tuple[float, float] = DUCK_BAND) -> tuple[np.ndarray, float]:
    """Dynamic EQ: the instrumental's 1-4 kHz band dips by up to `depth_db` while vocals sound.

    Zero-phase band split (band + rest == input exactly), key = vocal RMS (20 ms), mapped
    -45 dBFS -> 0 dB ... -25 dBFS -> full depth, 10 ms attack / 180 ms release.
    """
    from .master import _smooth_gr
    from scipy.ndimage import uniform_filter1d

    sos = butter(2, band, btype="bandpass", fs=SR, output="sos")
    mid = sosfiltfilt(sos, inst, axis=0).astype(np.float32)
    rest = inst - mid
    p = uniform_filter1d((key.astype(np.float64) ** 2).mean(axis=1), size=int(0.02 * SR))
    lvl = 10 * np.log10(p + 1e-12)
    dip = np.clip((lvl + 45.0) / 20.0, 0.0, 1.0) * depth_db
    dip = _smooth_gr(dip, 10.0, 180.0)
    g = (10 ** (-dip / 20)).astype(np.float32)
    return (rest + mid * g[:, None]).astype(np.float32), float(dip.max())


def process_tracks(tracks: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    out = {}
    for name, x in tracks.items():
        y = x * 10 ** (TRACK_TRIM_DB[name] / 20)
        if name == "A1_b_instrumental":
            y = Pedalboard([LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=LOW_SHELF_DB, q=0.7)])(
                y.T.copy(), SR).T
        if name in BUS_PEAK:
            y = bus_peak_control(y, BUS_PEAK[name])
        out[name] = y.astype(np.float32)
    key = np.sum([out[k] for k in VOCAL_TRACKS], axis=0)
    out["A1_b_instrumental"], duck_max = vocal_duck(out["A1_b_instrumental"], key)
    return out, duck_max


def mixdown(tracks: dict[str, np.ndarray]) -> MixResult:
    """Master chain on the sum; the same per-sample gain is applied to every stem.

    Single stems can peak above the sum (samples of different stems cancel), so a final
    sample-peak stem limiter keeps every stem <= STEM_PEAK_MAX_DBFS. It only touches those
    rare samples; the master is then by definition the sum of the final stems and is
    re-checked (true peak, loudness) in the export/QA step.
    """
    pre, duck_max = process_tracks(tracks)
    mix = np.sum([pre[k] for k in config.TRACK_NAMES], axis=0)
    res = master_chain(mix)
    final = {}
    from .master import limiter_gain
    for k in config.TRACK_NAMES:
        f = (pre[k] * res.gain).astype(np.float32)
        pk = 20 * np.log10(np.abs(f).max() + 1e-12)
        if pk > STEM_PEAK_MAX_DBFS:
            g = limiter_gain(f, STEM_PEAK_MAX_DBFS - 0.1, lookahead_ms=1.0, release_ms=30.0,
                             true_peak=False)
            log.info("stem %s peak %.2f dBFS -> stem limiter (max %.2f dB)", k, pk,
                     float(-20 * np.log10(g.min())))
            f = (f * g[:, None]).astype(np.float32)
        final[k] = f
    return MixResult(tracks=final, result=res, duck_max_db=duck_max)


def track_path(name: str) -> Path:
    return config.OUT / "spuren" / f"{name}.wav"


def master_wav() -> Path:
    return config.OUT / "remix-hook.wav"


def master_mp3() -> Path:
    return config.OUT / "remix-hook.mp3"


def export(mr: MixResult, extra: dict | None = None) -> dict:
    for name, x in mr.tracks.items():
        write_wav(track_path(name), x, SR, subtype="FLOAT")
    master_sum = np.sum([mr.tracks[k] for k in config.TRACK_NAMES], axis=0)
    write_wav(master_wav(), master_sum, SR, subtype="PCM_24")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(master_wav()), "-codec:a", "libmp3lame",
         "-b:a", "320k", "-id3v2_version", "3", "-metadata", f"title={TITLE}",
         "-metadata", "artist=Remix Hook", str(master_mp3())],
        check=True,
    )
    r = mr.result
    info = {
        "lufs": round(r.lufs, 2),
        "dbtp": round(r.dbtp, 2),
        "master_pre_gain_db": round(r.pre_gain_db, 2),
        "comp_gr_mean_db": round(r.comp_gr_mean_db, 2),
        "comp_gr_max_db": round(r.comp_gr_max_db, 2),
        "clip_gr_max_db": round(r.clip_gr_max_db, 2),
        "clip_share": round(r.clip_share, 5),
        "gr_mean_db": round(r.lim_gr_mean_db, 2),
        "gr_max_db": round(r.lim_gr_max_db, 2),
        "vocal_duck_max_db": round(mr.duck_max_db, 2),
        "length_s": round(len(master_sum) / SR, 3),
        "track_peaks_dbfs": {k: round(float(20 * np.log10(np.abs(v).max() + 1e-12)), 2)
                             for k, v in mr.tracks.items()},
        "master_wav_dbtp": round(true_peak_db(master_sum), 2),
    }
    if extra:
        info.update(extra)
    (config.OUT / "report.json").write_text(json.dumps(info, indent=2))
    return info
