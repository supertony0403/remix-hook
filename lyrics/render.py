"""Render a lyric timeline and make the delivery MP4.

    .venv/bin/python -m lyrics.render master 16x9   # Resolve: ProRes 422 HQ into ~/Videos/remix-hook
    .venv/bin/python -m lyrics.render mp4 16x9      # ffmpeg: master video + out/remix-hook-v2.2.wav
    .venv/bin/python -m lyrics.render check 16x9    # lengths, loudness, audio grid, contact sheets
    .venv/bin/python -m lyrics.render chunks 16x9 [0,3]  # section-wise render (+ concat) instead of master

The sound of the MP4 comes from `out/remix-hook-v2.2.wav`, not from the Resolve master: the remix
WAV was overwritten in place while the first render ran (23:32), so the master's audio may mix
two versions. The WAV starts at timeline frame 0 (no offset).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import resolve_io

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "lyrics"
VIDEOS = Path.home() / "Videos"
AUDIO = ROOT / "out" / "remix-hook-v2.2.wav"
FPS = 60


def master_path(fmt: str) -> Path:
    return resolve_io.VIDEOS / f"lyrics-remix-{fmt}-master.mov"


def mp4_path(fmt: str) -> Path:
    return OUT / f"lyrics-remix-{fmt}.mp4"


def probe(path: Path) -> dict:
    res = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def loudness(path: Path) -> dict:
    err = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-map", "0:a:0",
                          "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    tail = err[err.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", tail)
    tp = re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail)
    return {"lufs": float(i.group(1)) if i else None, "dbtp": float(tp.group(1)) if tp else None}


def pcm(path: Path, seconds: float, sr: int = 8000) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-t", str(seconds),
                          "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def best_lag_ms(a: np.ndarray, b: np.ndarray, sr: int = 8000, max_ms: int = 200) -> tuple[float, float]:
    """Lag (ms) maximising the normalised correlation of a against b, and that correlation."""
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    m = int(sr * max_ms / 1000)
    best = (0.0, -1.0)
    for lag in range(-m, m + 1, 4):
        x = a[max(0, lag): n + min(0, lag)]
        y = b[max(0, -lag): n - max(0, lag)]
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        if c > best[1]:
            best = (lag / sr * 1000, c)
    return best


def contact_sheet(mp4: Path, out: Path, start: float, length: float, every_s: float = 2.0) -> Path:
    vertical = "9x16" in mp4.name
    w, h = (240, 135) if not vertical else (135, 240)
    cols = 8 if not vertical else 11
    rows = int(np.ceil(length / every_s / cols))
    vf = (f"fps=1/{every_s},scale={w}:{h},drawtext=text='%{{pts\\:hms}}':x=4:y=4:fontsize=12:"
          f"fontcolor=white:box=1:boxcolor=black@0.6,tile={cols}x{rows}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(length), "-i", str(mp4),
                    "-vf", vf, "-frames:v", "1", "-copyts", str(out)], check=True, timeout=1800)
    return out


def do_master(fmt: str) -> None:
    t0 = time.time()
    resolve = resolve_io.connect()
    p = resolve_io.project(resolve)
    resolve_io.ensure_fonts(resolve)
    try:
        master = resolve_io.render_master(resolve, p, fmt, log=lambda m: print(m, flush=True))
    finally:
        if not p.IsRenderingInProgress():
            p.SetCurrentTimeline(resolve_io.find_timeline(p, resolve_io.TIMELINES["16x9"]))
    print(json.dumps({"fmt": fmt, "master": str(master), "render_min": round((time.time() - t0) / 60, 1)}),
          flush=True)


def section_spans(fmt: str) -> list[tuple[str, int, int]]:
    data = json.loads((resolve_io.WORK / "lyrics.json").read_text())
    return [(s["name"], int(round(s["comp_start"] * FPS)), int(round(s["comp_end"] * FPS)))
            for s in data["sections"]]


def do_chunks(fmt: str, only: str = "") -> None:
    """Section-wise ProRes render + lossless concat into the master path."""
    t0 = time.time()
    resolve = resolve_io.connect()
    p = resolve_io.project(resolve)
    resolve_io.ensure_fonts(resolve)
    spans = section_spans(fmt)
    sel = {int(x) for x in only.split(",") if x} or None
    try:
        resolve_io.render_chunks(resolve, p, fmt, spans, log=lambda m: print(m, flush=True), only=sel)
    finally:
        if not p.IsRenderingInProgress():
            p.SetCurrentTimeline(resolve_io.find_timeline(p, resolve_io.TIMELINES["16x9"]))
    parts = [resolve_io.VIDEOS / f"parts-{fmt}" / f"part{i:02d}.mov" for i in range(len(spans))]
    resolve_io.concat_parts(parts, master_path(fmt), [b - a for _, a, b in spans])
    print(json.dumps({"fmt": fmt, "master": str(master_path(fmt)), "render_min": round((time.time() - t0) / 60, 1)}),
          flush=True)


def video_path(fmt: str) -> Path:
    return resolve_io.WORK / f"lyrics-remix-{fmt}-video.mp4"


def do_mp4(fmt: str) -> None:
    """Encode the picture once, mux the v2.2 sound; if the AAC true peak ends above -1.0 dBTP,
    re-mux with the WAV lowered by 0.3 dB (video copied, no new encode)."""
    t0 = time.time()
    video = resolve_io.encode_video(master_path(fmt), video_path(fmt))
    mp4 = resolve_io.mux(video, AUDIO, mp4_path(fmt))
    loud = loudness(mp4)
    gain = 0.0
    if loud["dbtp"] is not None and loud["dbtp"] > -1.0:
        gain = -0.3
        mp4 = resolve_io.mux(video, AUDIO, mp4_path(fmt), gain_db=gain)
        loud = loudness(mp4)
    shutil.copy2(mp4, VIDEOS / mp4.name)
    print(json.dumps({"mp4": str(mp4), "copy": str(VIDEOS / mp4.name), "gain_db": gain, "loudness": loud,
                      "encode_min": round((time.time() - t0) / 60, 1)}), flush=True)


def do_check(fmt: str) -> dict:
    mp4 = mp4_path(fmt)
    info = probe(mp4)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    lag, corr = best_lag_ms(pcm(master_path(fmt), 20.0), pcm(AUDIO, 20.0))
    rep = {
        "mp4": str(mp4), "size_mb": round(mp4.stat().st_size / 1e6, 1),
        "video": {"codec": v["codec_name"], "w": v["width"], "h": v["height"], "fps": v["r_frame_rate"],
                  "frames": int(v.get("nb_frames", 0)), "duration": float(v["duration"])},
        "audio": {"codec": a["codec_name"], "rate": a["sample_rate"], "bitrate": a.get("bit_rate"),
                  "duration": float(a["duration"])},
        "av_diff_frames": round(abs(float(v["duration"]) - float(a["duration"])) * FPS, 2),
        "loudness": loudness(mp4),
        "intro_grid": {"lag_ms": lag, "corr": round(corr, 4)},
        "sheets": [str(contact_sheet(mp4, OUT / f"kontaktbogen-{fmt}-{i + 1}.png", s, 82.0))
                   for i, s in enumerate((0.0, 82.0))],
    }
    (OUT / f"check-{fmt}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    return rep


if __name__ == "__main__":
    cmd, fmt = sys.argv[1], sys.argv[2]
    if cmd == "chunks":
        do_chunks(fmt, sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        {"master": do_master, "mp4": do_mp4, "check": do_check}[cmd](fmt)
