"""Central paths and musical constants."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUELLE = ROOT / "quelle"
WORK = ROOT / "work"
STEMS = ROOT / "stems"
OUT = Path(os.environ.get("REMIX_OUT", str(ROOT / "out")))  # override for side-by-side renders
DOCS = ROOT / "docs"

SRC_A = QUELLE / "a_hook.mp4"
SRC_B = QUELLE / "b_song.mp3"

DATA_ROOT = Path("/mnt/steam-library/remix-hook")
os.environ.setdefault("TORCH_HOME", str(DATA_ROOT / "torch"))

SR = 48_000  # working sample rate
BPM_B = 150.0  # measured (librosa's 152 is hop-lag quantisation)
BEAT_B = 60.0 / BPM_B
BAR_B = 4 * BEAT_B  # 1.5789 s

SEMITONES_A = -2.0  # E major -> D major (relative major of B minor)

STEM_NAMES = ("drums", "bass", "other", "vocals")

TRACK_NAMES = (
    "A1_b_instrumental",
    "A2_b_gesang",
    "A3_a_hook",
    "A4_chops",
    "A5_fx",
    "A6_returns",
)

TARGET_LUFS = -9.5
TARGET_DBTP = -1.0
