"""Fill the existing DaVinci Resolve project "Remix Hook" / timeline "Remix" and render it.

The project and the timeline already exist (created by the coordinator). This module only
* refreshes the media pool clips of the track WAVs and the master via ReplaceClip,
* lays every track WAV at the timeline start on A1-A6 (A7 = muted master reference) with the
  full length (endFrame is exclusive: frames, not frames-1),
* makes sure the section markers exist,
* renders "Audio Only" WAV 24 bit into ~/Videos/remix-hook (Media Storage!), and
* compares the Resolve render with out/remix-hook.wav.
Other timelines ("Lyrics ...") and bins ("04 Lyrics") are never touched.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .arrangement import ARRANGEMENT, BAR
from .audio_io import decode, read_wav
from .mix import master_wav, track_path

log = logging.getLogger(__name__)

PROJECT = "Remix Hook"
TIMELINE = "Remix"
FPS = 60
RENDER_DIR = Path.home() / "Videos" / "remix-hook"
RENDER_NAME = "remix-hook-resolve"
TRACK_LABELS = {  # Resolve track index -> (file, bin)
    1: ("A1_b_instrumental", "02 Spuren"),
    2: ("A2_b_gesang", "02 Spuren"),
    3: ("A3_a_hook", "02 Spuren"),
    4: ("A4_chops", "02 Spuren"),
    5: ("A5_fx", "02 Spuren"),
    6: ("A6_returns", "02 Spuren"),
    7: ("remix-hook", "03 Master"),
}
MARKER_COLORS = ["Blue", "Red", "Green", "Yellow", "Green", "Pink", "Purple", "Red", "Blue"]


def connect():
    os.environ.setdefault("RESOLVE_SCRIPT_API", "/opt/resolve/Developer/Scripting")
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", "/opt/resolve/libs/Fusion/fusionscript.so")
    mod = "/opt/resolve/Developer/Scripting/Modules"
    if mod not in sys.path:
        sys.path.append(mod)
    import DaVinciResolveScript as dvr  # type: ignore

    r = dvr.scriptapp("Resolve")
    if r is None:
        raise RuntimeError("Resolve scripting not reachable (is Resolve running?)")
    return r


def _file_for(index: int) -> Path:
    name, _ = TRACK_LABELS[index]
    return master_wav() if index == 7 else track_path(name)


def _tc_frames(tc: str, fps: int = FPS) -> int:
    h, m, s, f = (int(v) for v in tc.split(":"))
    return ((h * 60 + m) * 60 + s) * fps + f


def _bin(root, name: str):
    for f in root.GetSubFolderList():
        if f.GetName() == name:
            return f
    raise RuntimeError(f"bin {name!r} missing")


def _find_item(folder, path: Path):
    for c in folder.GetClipList() or []:
        if Path(c.GetClipProperty("File Path") or "").resolve() == path.resolve():
            return c
    return None


def fill_project(render: bool = True) -> dict:
    r = connect()
    p = r.GetProjectManager().GetCurrentProject()
    if p is None or p.GetName() != PROJECT:
        raise RuntimeError(f"current project is {p.GetName() if p else None!r}, expected {PROJECT!r}")
    tl = next((p.GetTimelineByIndex(i) for i in range(1, p.GetTimelineCount() + 1)
               if p.GetTimelineByIndex(i).GetName() == TIMELINE), None)
    if tl is None:
        raise RuntimeError(f"timeline {TIMELINE!r} missing")
    p.SetCurrentTimeline(tl)
    mp = p.GetMediaPool()
    root = mp.GetRootFolder()
    expected_frames = int(round(len(read_wav(master_wav())[0]) / config.SR * FPS))

    # 1) media pool: refresh (ReplaceClip) or import
    items = {}
    for idx, (_, bin_name) in TRACK_LABELS.items():
        path = _file_for(idx)
        folder = _bin(root, bin_name)
        item = _find_item(folder, path)
        if item is None:
            mp.SetCurrentFolder(folder)
            imported = mp.ImportMedia([str(path)])
            item = imported[0] if imported else None
            log.info("imported %s -> %s", path.name, bin_name)
        else:
            ok = item.ReplaceClip(str(path))
            log.info("ReplaceClip %s: %s", path.name, ok)
        if item is None:
            raise RuntimeError(f"could not get media pool item for {path}")
        items[idx] = item

    # 2) timeline: one full-length clip per track at the timeline start
    start = tl.GetStartFrame()
    for idx, item in items.items():
        old = tl.GetItemListInTrack("audio", idx) or []
        if old:
            tl.DeleteClips(old, False)
        frames = _tc_frames(item.GetClipProperty("Duration"))  # 'Frames' is empty for audio
        res = mp.AppendToTimeline([{"mediaPoolItem": item, "startFrame": 0, "endFrame": frames,
                                    "mediaType": 2, "trackIndex": idx, "recordFrame": start}])
        placed = tl.GetItemListInTrack("audio", idx) or []
        dur = placed[0].GetDuration() if placed else None
        log.info("track %d %s: frames %s placed %s (append ok %s)", idx, item.GetName(), frames, dur, bool(res))
    tl.SetTrackEnable("audio", 7, False)

    # 3) markers (keep existing, add missing)
    have = tl.GetMarkers() or {}
    for s, color in zip(ARRANGEMENT.sections, MARKER_COLORS):
        f = int(round(s.start * BAR * FPS))
        if f not in have:
            tl.AddMarker(f, color, s.name, "", int(round(s.bars * BAR * FPS)), "")

    state = {
        "project": p.GetName(), "timeline": tl.GetName(), "expected_frames": expected_frames,
        "tracks": {i: [(it.GetName(), it.GetStart() - start, it.GetDuration())
                       for it in (tl.GetItemListInTrack("audio", i) or [])]
                   for i in range(1, tl.GetTrackCount("audio") + 1)},
        "track_names": {i: tl.GetTrackName("audio", i) for i in range(1, tl.GetTrackCount("audio") + 1)},
        "a7_enabled": tl.GetIsTrackEnabled("audio", 7),
        "markers": {k: v["name"] for k, v in (tl.GetMarkers() or {}).items()},
    }
    if render:
        state["render"] = render_and_compare(p)
    return state


def render_and_compare(p) -> dict:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    if p.IsRenderingInProgress():
        raise RuntimeError("another render is in progress")
    p.LoadRenderPreset("Audio Only")
    p.SetCurrentRenderFormatAndCodec("wav", "")
    target = RENDER_DIR / f"{RENDER_NAME}.wav"
    if target.exists():
        target.unlink()
    ok = p.SetRenderSettings({
        "SelectAllFrames": True, "TargetDir": str(RENDER_DIR), "CustomName": RENDER_NAME,
        "ExportVideo": False, "ExportAudio": True, "AudioBitDepth": 24, "AudioSampleRate": 48000,
    })
    job = p.AddRenderJob()
    if not job:
        raise RuntimeError("AddRenderJob returned '' (render dir outside Media Storage?)")
    log.info("render settings ok=%s job=%s format=%s", ok, job, p.GetCurrentRenderFormatAndCodec())
    p.StartRendering(job)
    t0 = time.time()
    while p.IsRenderingInProgress():
        if time.time() - t0 > 900:
            raise RuntimeError("render timeout")
        time.sleep(1.0)
    status = p.GetRenderJobStatus(job)
    p.DeleteRenderJob(job)
    files = sorted(RENDER_DIR.glob(f"{RENDER_NAME}*"))
    if not files:
        raise RuntimeError(f"render produced no file in {RENDER_DIR}: {status}")
    return {"status": status, "file": str(files[0]), **compare(files[0])}


def compare(render_file: Path) -> dict:
    """Level, length and correlation of the Resolve render against the master."""
    sr = config.SR
    a = read_wav(master_wav())[0]
    b = decode(render_file, sr)
    # align by cross-correlating the first 20 s (Resolve may add/drop a few samples)
    seg = int(20 * sr)
    ma, mb = a[:seg].mean(axis=1), b[:seg].mean(axis=1)
    n = 1 << int(np.ceil(np.log2(2 * seg)))
    xc = np.fft.irfft(np.fft.rfft(ma, n) * np.conj(np.fft.rfft(mb, n)), n)
    lag = int(np.argmax(xc))
    lag = lag - n if lag > n // 2 else lag  # positive: master lags render
    if lag > 0:
        a2, b2 = a[lag:], b
    else:
        a2, b2 = a, b[-lag:]
    m = min(len(a2), len(b2))
    a2, b2 = a2[:m], b2[:m]
    corr = float(np.corrcoef(a2.ravel(), b2.ravel())[0, 1])
    rms = lambda x: float(10 * np.log10(np.mean(x ** 2) + 1e-20))  # noqa: E731
    return {
        "len_master_s": round(len(a) / sr, 4), "len_render_s": round(len(b) / sr, 4),
        "lag_samples": lag, "correlation": round(corr, 6),
        "level_diff_db": round(rms(b2) - rms(a2), 3),
        "peak_master_dbfs": round(float(20 * np.log10(np.abs(a).max())), 2),
        "peak_render_dbfs": round(float(20 * np.log10(np.abs(b).max() + 1e-12)), 2),
        "ok": corr > 0.99 and abs(rms(b2) - rms(a2)) < 0.2 and abs(len(a) - len(b)) / sr < 0.05,
    }
