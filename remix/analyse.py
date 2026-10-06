"""Measure tempo/downbeat of A and B (high resolution) and the key relation; write docs/grid.json."""

from __future__ import annotations

import json

import numpy as np

from . import config
from . import grid as G
from .audio_io import read_wav
from .vocals import chroma_mean, key_scores, transposition_scores

GRID_JSON = config.DOCS / "grid.json"


def analyse() -> dict:
    sr = config.SR
    out: dict = {}
    librosa_bpm = {"a": json.loads((config.DOCS / "a_analyse.json").read_text())["tempo"],
                   "b": json.loads((config.DOCS / "b_analyse.json").read_text())["tempo"]}
    for song, lo, hi in (("a", 126.0, 131.0), ("b", 148.0, 154.0)):
        mix, _ = read_wav(config.WORK / f"{song}_mix.wav")
        st = {s: read_wav(config.STEMS / song / f"{s}.wav")[0] for s in ("drums", "bass", "other")}
        env = G.onset_envelope(st["drums"], sr)
        bpm, phase, score = G.fit_tempo(env, sr, lo, hi)
        local = G.local_tempi(env, sr, lo, hi, 30.0)
        db = G.downbeat_index(mix, sr, bpm, phase, drums=st["drums"],
                              harmonic=st["bass"] + st["other"], bass=st["bass"])
        # beat-list check: slope of the librosa beats (robust to its hop quantisation)
        beats = np.array(json.loads((config.DOCS / f"{song}_analyse.json").read_text())["beats"])
        slope = float(np.polyfit(np.arange(len(beats)), beats, 1)[0])
        beat_list_bpm = 60.0 / slope
        if song == "a":
            beat_list_bpm *= 2  # librosa tracked half tempo
        out[song] = {
            "bpm": round(bpm, 3), "beat_phase": round(phase, 4), "downbeat0": round(db["downbeat0"], 4),
            "downbeat_index": db["index"], "entry_votes_by_pos": db["entry_votes_by_pos"],
            "kick_by_pos": db["kick_by_pos"], "local_bpm_30s": [round(v, 2) for v in local],
            "librosa_tempo": round(librosa_bpm[song], 3),
            "beat_list_fit_bpm": round(beat_list_bpm, 3),
            "median_beat_interval_bpm": round(60.0 / float(np.median(np.diff(beats))) * (2 if song == "a" else 1), 3),
        }
    # key relation: A vocals (original) vs B harmonic stems
    av, _ = read_wav(config.STEMS / "a" / "vocals.wav")
    bh = read_wav(config.STEMS / "b" / "bass.wav")[0] + read_wav(config.STEMS / "b" / "other.wav")[0]
    ca, cb = chroma_mean(av, sr), chroma_mean(bh, sr)
    trans = transposition_scores(ca, cb)
    out["key"] = {
        "a_vocals_top": [(k, round(v, 3)) for k, v in key_scores(ca)[:3]],
        "b_harmonic_top": [(k, round(v, 3)) for k, v in key_scores(cb)[:3]],
        "best_transposition_a_to_b": max(trans, key=trans.get),
        "transposition_scores": {str(k): round(v, 3) for k, v in sorted(trans.items())},
    }
    GRID_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out
