"""DaVinci Resolve side of the lyric video: carriers, comp placement, control frames, render.

Connection, dialog guard and font registration are copied from
~/Documents/Programmierung/nomissuccess-spot/resolve/resolve_api.py; placement and render follow
resolve/bauen.py and resolve/rendern.py there (carrier clip + ImportFusionComp, ProRes 422 HQ into
the Media Storage, ffmpeg to H.264). Rules kept: never LoadProject, never restart Resolve, no key
presses or screenshots, SaveProject() before every ImportFusionComp, check IsRenderingInProgress().
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = "Remix Hook"
TIMELINES = {"16x9": "Lyrics 16x9", "9x16": "Lyrics 9x16"}
BIN = "04 Lyrics"
TRACKS = {"background": 1, "lyrics": 2, "overlay": 3}
FPS = 60
CARRIER_FRAMES = 9960  # 166 s; the remix is 9840 frames
ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work" / "lyrics"
VIDEOS = Path.home() / "Videos" / "remix-hook"   # Media Storage (symlink to the big disk)


class ResolveError(RuntimeError):
    pass


def connect() -> Any:
    os.environ.setdefault("RESOLVE_SCRIPT_API", "/opt/resolve/Developer/Scripting")
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", "/opt/resolve/libs/Fusion/fusionscript.so")
    mod = os.path.join(os.environ["RESOLVE_SCRIPT_API"], "Modules")
    if mod not in sys.path:
        sys.path.append(mod)
    dvr = importlib.import_module("DaVinciResolveScript")
    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise ResolveError("Resolve not reachable")
    return resolve


def project(resolve: Any) -> Any:
    p = resolve.GetProjectManager().GetCurrentProject()
    if p is None or p.GetName() != PROJECT:
        raise ResolveError(f"open project is {p and p.GetName()!r}, expected {PROJECT!r} — not switching")
    return p


def find_timeline(p: Any, name: str) -> Any:
    for i in range(1, int(p.GetTimelineCount() or 0) + 1):
        tl = p.GetTimelineByIndex(i)
        if tl is not None and tl.GetName() == name:
            return tl
    raise ResolveError(f"timeline {name!r} not found")


def open_dialogs() -> list[dict]:
    try:
        out = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, timeout=5).stdout
        clients = json.loads(out or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return [{"title": c["title"]} for c in clients
            if c.get("class") == "resolve" and c.get("mapped") and not c["title"].startswith("DaVinci Resolve")]


def guard(step: str) -> None:
    d = open_dialogs()
    if d:
        raise ResolveError(f"after {step}: Resolve shows a dialog {d} — stopping")


def ensure_fonts(resolve: Any, families: tuple[str, ...] = ("Anton", "Bebas Neue", "Space Grotesk")
                 ) -> dict[str, bool]:
    """Fonts installed after Resolve started are invisible to Text+ until AddFont (session only).
    Raises if a font stays missing: Text+ would silently render a stand-in."""
    fm = resolve.Fusion().FontManager
    have = fm.GetFontList() or {}
    font_dir = Path.home() / ".local/share/fonts/lyrics"
    if not all(f in have for f in families):
        for path in sorted(font_dir.glob("*.[ot]tf")):
            fm.AddFont(str(path))
        have = fm.GetFontList() or {}
    state = {f: f in have for f in families}
    if not all(state.values()):
        raise ResolveError(f"fonts missing in Fusion: {state}")
    return state


def not_rendering(p: Any, what: str) -> None:
    if p.IsRenderingInProgress():
        raise ResolveError(f"Resolve is rendering — not {what}")


# ----------------------------------------------------------------------------- carriers
def carrier_path(fmt: str) -> Path:
    return WORK / f"traeger_{fmt}.mov"


def make_carrier(fmt: str, size: tuple[int, int]) -> Path:
    path = carrier_path(fmt)
    if path.exists():
        return path
    tmp = path.with_name(f".{path.name}")
    w, h = size
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"color=c=black:s={w}x{h}:r={FPS}", "-frames:v", str(CARRIER_FRAMES),
                    "-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le", "-f", "mov", str(tmp)],
                   check=True, timeout=1800)
    tmp.replace(path)
    return path


def bin_folder(pool: Any) -> Any:
    for f in pool.GetRootFolder().GetSubFolderList() or []:
        if f.GetName() == BIN:
            return f
    f = pool.AddSubFolder(pool.GetRootFolder(), BIN)
    if f is None:
        raise ResolveError(f"cannot create bin {BIN}")
    return f


def carrier_item(p: Any, fmt: str) -> Any:
    path = str(carrier_path(fmt).resolve())
    pool = p.GetMediaPool()
    folder = bin_folder(pool)
    for clip in folder.GetClipList() or []:
        if clip.GetClipProperty("File Path") == path:
            return clip
    pool.SetCurrentFolder(folder)
    items = pool.ImportMedia([path])
    guard("ImportMedia")
    if not items:
        raise ResolveError(f"ImportMedia failed for {path}")
    return items[0]


# ----------------------------------------------------------------------------- placement
class Placer:
    """Puts carrier clips with imported Fusion comps on a lyric timeline."""

    def __init__(self, resolve: Any, p: Any, fmt: str, log=print):
        self.resolve, self.p, self.fmt, self.log = resolve, p, fmt, log
        self.pool = p.GetMediaPool()
        self.tl = find_timeline(p, TIMELINES[fmt])
        self.carrier = carrier_item(p, fmt)
        self.t0 = int(self.tl.GetStartFrame())
        self.end_inclusive: bool | None = None

    def _ours(self, item: Any) -> bool:
        mpi = item.GetMediaPoolItem()
        return mpi is not None and mpi.GetClipProperty("File Path") == str(carrier_path(self.fmt).resolve())

    def clear(self, track: int, start: int, length: int) -> None:
        """Remove our carrier clips that overlap [start, start+length) on `track`."""
        doomed = []
        for item in self.tl.GetItemListInTrack("video", track) or []:
            a = int(item.GetStart()) - self.t0
            b = a + int(item.GetDuration())
            if a < start + length and b > start and self._ours(item):
                doomed.append(item)
        if doomed:
            self.tl.DeleteClips(doomed, False)

    def _append(self, track: int, start: int, length: int) -> Any:
        end = length - 1 if self.end_inclusive else length
        info = {"mediaPoolItem": self.carrier, "startFrame": 0, "endFrame": end,
                "trackIndex": track, "mediaType": 1, "recordFrame": self.t0 + start}
        items = self.pool.AppendToTimeline([info]) or []
        guard("AppendToTimeline")
        if not items:
            raise ResolveError(f"AppendToTimeline failed (track {track}, start {start})")
        return items[0]

    def place(self, track: int, start: int, length: int, comp: Path, name: str) -> Any:
        not_rendering(self.p, "placing clips")
        if self.p.GetCurrentTimeline().GetName() != self.tl.GetName():
            self.p.SetCurrentTimeline(self.tl)
        self.clear(track, start, length)
        item = self._append(track, start, length)
        if self.end_inclusive is None:
            dur = int(item.GetDuration())
            if dur == length + 1:
                self.end_inclusive = True
                self.tl.DeleteClips([item], False)
                item = self._append(track, start, length)
            else:
                self.end_inclusive = False
        got_start, got_len = int(item.GetStart()) - self.t0, int(item.GetDuration())
        if got_start != start or got_len != length:
            self.tl.DeleteClips([item], False)
            raise ResolveError(f"{name}: placed at {got_start}+{got_len}, expected {start}+{length}")
        self.resolve.GetProjectManager().SaveProject()
        comp_obj = item.ImportFusionComp(str(Path(comp).resolve()))
        guard(f"ImportFusionComp {name}")
        if comp_obj is None:
            raise ResolveError(f"ImportFusionComp failed for {comp}")
        names = item.GetFusionCompNameList() or []
        if not names:
            raise ResolveError(f"{name}: no Fusion comp on the clip after import")
        item.LoadFusionCompByName(names[-1])
        item.SetClipColor("Pink" if track == 2 else "Teal" if track == 1 else "Purple")
        item.SetName(name) if hasattr(item, "SetName") else None
        self.resolve.GetProjectManager().SaveProject()
        self.log(f"[{self.fmt}] V{track} {name}: frames {start}-{start + length}")
        return item


# ----------------------------------------------------------------------------- control frames
def render_frames(resolve: Any, p: Any, fmt: str, frames: list[int], tag: str,
                  out_dir: Path) -> list[Path]:
    """Render single frames of a lyric timeline to PNG (Deliver page, Media Storage), move them
    to `out_dir`. Restores page and current timeline afterwards."""
    not_rendering(p, "rendering control frames")
    tl = find_timeline(p, TIMELINES[fmt])
    page = resolve.GetCurrentPage()
    prev = p.GetCurrentTimeline()
    p.SetCurrentTimeline(tl)
    t0 = int(tl.GetStartFrame())
    stage = VIDEOS / "zz-frames" / tag
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    w, h = (1920, 1080) if fmt == "16x9" else (1080, 1920)
    try:
        p.SetCurrentRenderFormatAndCodec("png", "RGB8")
        p.SetCurrentRenderMode(1)
        for f in sorted(set(frames)):
            ok = p.SetRenderSettings({"SelectAllFrames": False, "MarkIn": t0 + f, "MarkOut": t0 + f,
                                      "TargetDir": str(stage), "CustomName": f"{tag}_{f:05d}_",
                                      "ExportVideo": True, "ExportAudio": False,
                                      "FormatWidth": w, "FormatHeight": h})
            job = p.AddRenderJob() if ok else ""
            guard("AddRenderJob")
            if not job:
                raise ResolveError(f"AddRenderJob refused frame {f}")
            jobs.append(job)
        if not p.StartRendering(jobs, False):
            raise ResolveError("StartRendering returned False")
        deadline = time.time() + 120 + 30 * len(jobs)
        while p.IsRenderingInProgress() and time.time() < deadline:
            time.sleep(0.5)
        if p.IsRenderingInProgress():
            p.StopRendering()
            raise ResolveError("control render timed out")
    finally:
        if not p.IsRenderingInProgress():
            for j in jobs:
                p.DeleteRenderJob(j)
        if prev is not None:
            p.SetCurrentTimeline(prev)
        if page and resolve.GetCurrentPage() != page:
            resolve.OpenPage(page)
    written = []
    for f in sorted(set(frames)):
        hits = sorted(stage.glob(f"{tag}_{f:05d}_*.png"))
        if hits:
            dest = out_dir / f"{tag}_{f:05d}.png"
            shutil.move(str(hits[0]), dest)
            written.append(dest)
    shutil.rmtree(stage, ignore_errors=True)
    return written


# ----------------------------------------------------------------------------- final render
def prores_hq(p: Any) -> str:
    codecs = p.GetRenderCodecs("mov") or {}
    for label, codec in codecs.items():
        if "422 HQ" in label:
            return codec
    raise ResolveError(f"no ProRes 422 HQ in {codecs}")


def render_master(resolve: Any, p: Any, fmt: str, log=print, timeout_s: int = 3 * 3600) -> Path:
    not_rendering(p, "starting a master render")
    tl = find_timeline(p, TIMELINES[fmt])
    VIDEOS.mkdir(parents=True, exist_ok=True)
    name = f"lyrics-remix-{fmt}-master"
    for job in p.GetRenderJobList() or []:  # only our own earlier jobs
        if str(job.get("OutputFilename", "")).startswith(name):
            p.DeleteRenderJob(job["JobId"])
    (VIDEOS / f"{name}.mov").unlink(missing_ok=True)
    page = resolve.GetCurrentPage()
    prev = p.GetCurrentTimeline()
    job = ""
    started = time.time()
    try:
        p.SetCurrentTimeline(tl)
        w, h = (1920, 1080) if fmt == "16x9" else (1080, 1920)
        if not p.SetCurrentRenderFormatAndCodec("mov", prores_hq(p)):
            raise ResolveError("SetCurrentRenderFormatAndCodec(mov, ProRes 422 HQ) failed")
        p.SetCurrentRenderMode(1)
        base = {"SelectAllFrames": True, "TargetDir": str(VIDEOS), "CustomName": name,
                "UniqueFilenameStyle": 0, "ExportVideo": True, "ExportAudio": True,
                "FormatWidth": w, "FormatHeight": h}
        ok = p.SetRenderSettings({**base, "AudioCodec": "lpcm", "AudioBitDepth": 24, "AudioSampleRate": 48000})
        if not ok:
            ok = p.SetRenderSettings(base)
        job = p.AddRenderJob() if ok else ""
        guard("AddRenderJob master")
        if not job:
            raise ResolveError(f"AddRenderJob refused ({fmt})")
        if not p.StartRendering([job], False):
            raise ResolveError("StartRendering returned False")
        last = 0.0
        while p.IsRenderingInProgress():
            if time.time() - started > timeout_s:
                p.StopRendering()
                raise ResolveError("render timed out")
            if time.time() - last > 60:
                st = p.GetRenderJobStatus(job)
                pct = float(st.get("CompletionPercentage") or 0)
                mins = (time.time() - started) / 60
                log(f"{time.strftime('%H:%M:%S')} {fmt}: {pct:.0f} % after {mins:.1f} min "
                    f"(~{pct / 100 * 164.0 / max(mins, 1e-6):.1f} s video/min)")
                last = time.time()
            time.sleep(2)
        st = p.GetRenderJobStatus(job)
        if st.get("JobStatus") != "Complete":
            raise ResolveError(f"render not complete: {st}")
    finally:
        if not p.IsRenderingInProgress():
            if job:
                p.DeleteRenderJob(job)
            if prev is not None:
                p.SetCurrentTimeline(prev)
        if page and resolve.GetCurrentPage() != page:
            resolve.OpenPage(page)
    hits = sorted(VIDEOS.glob(f"{name}*.mov"), key=lambda q: q.stat().st_mtime)
    if not hits:
        raise ResolveError(f"no master file {name}*.mov in {VIDEOS}")
    log(f"{fmt}: master {hits[-1]} in {(time.time() - started) / 60:.1f} min")
    return hits[-1]


def encode_video(master: Path, video: Path) -> Path:
    """H.264 (crf 17, 60 fps, yuv420p) of the master's picture only, kept for re-muxing."""
    video.parent.mkdir(parents=True, exist_ok=True)
    tmp = video.with_name(f".{video.name}")
    subprocess.run(["nice", "-n", "12", "ffmpeg", "-v", "error", "-y", "-i", str(master), "-map", "0:v:0",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-threads", "6",
                    "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", "-f", "mp4", str(tmp)],
                   check=True, timeout=4 * 3600)
    tmp.replace(video)
    return video


def mux(video: Path, audio: Path, mp4: Path, gain_db: float = 0.0) -> Path:
    """Video stream copied, `audio` (starts at timeline frame 0, no offset) as AAC 320k, faststart."""
    tmp = mp4.with_name(f".{mp4.name}")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy"]
    if gain_db:
        cmd += ["-af", f"volume={gain_db}dB"]
    cmd += ["-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", "-f", "mp4", str(tmp)]
    subprocess.run(cmd, check=True, timeout=1800)
    tmp.replace(mp4)
    return mp4
