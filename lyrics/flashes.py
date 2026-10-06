"""Flash planning with a photosensitivity limit.

Rule (brief): at most 3 bright full-frame flashes in any 1 s window, each at most 4 frames.
Hits beyond the budget are kept as *dim* pulses (opacity <= DIM_MAX), which are not flashes in
the WCAG sense (a small luminance change on a near-black frame).
"""

from __future__ import annotations

from dataclasses import dataclass

FPS = 60
MAX_BRIGHT_PER_S = 3
MAX_FRAMES = 4
BRIGHT = 0.55     # peak opacity of a bright white flash
DIM_MAX = 0.10    # a dim pulse stays below this


@dataclass(frozen=True)
class Flash:
    frame: int
    frames: int
    peak: float
    color: str = "#ffffff"

    @property
    def bright(self) -> bool:
        return self.peak > DIM_MAX


def plan(times_s: list[float], bright: list[bool] | None = None, frames: int = 3,
         color: str = "#ffffff", fps: int = FPS) -> list[Flash]:
    """Turn hit times into flashes, demoting bright hits that would break the limit."""
    if frames > MAX_FRAMES:
        raise ValueError(f"a flash may last at most {MAX_FRAMES} frames")
    want = bright if bright is not None else [True] * len(times_s)
    out: list[Flash] = []
    bright_frames: list[int] = []
    last_frame = -10**9
    for t, b in sorted(zip(times_s, want)):
        f = int(round(t * fps))
        if f <= last_frame + frames:  # never overlap two flashes
            continue
        recent = [x for x in bright_frames if f - x < fps]
        if b and len(recent) < MAX_BRIGHT_PER_S:
            out.append(Flash(f, frames, BRIGHT, color))
            bright_frames.append(f)
        else:
            out.append(Flash(f, frames, DIM_MAX * 0.8, color))
        last_frame = f
    return out


def merge_plans(*plans: list[Flash], fps: int = FPS) -> list[Flash]:
    """Combine plans from several sections and re-apply the global limit."""
    allf = sorted((f for p in plans for f in p), key=lambda f: f.frame)
    out: list[Flash] = []
    bright_frames: list[int] = []
    for fl in allf:
        if out and fl.frame <= out[-1].frame + out[-1].frames:
            continue
        if fl.bright:
            recent = [x for x in bright_frames if fl.frame - x < fps]
            if len(recent) >= MAX_BRIGHT_PER_S:
                fl = Flash(fl.frame, fl.frames, DIM_MAX * 0.8, fl.color)
            else:
                bright_frames.append(fl.frame)
        out.append(fl)
    return out


def max_bright_in_any_second(flashes: list[Flash], fps: int = FPS) -> int:
    frames = sorted(f.frame for f in flashes if f.bright)
    best = 0
    j = 0
    for i, f in enumerate(frames):
        while frames[j] <= f - fps:
            j += 1
        best = max(best, i - j + 1)
    return best
