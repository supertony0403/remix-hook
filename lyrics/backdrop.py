"""Full-length comps for V1 (background) and V3 (FX overlay: grain, flashes, light leaks).

Background: near-black ground with three drifting neon fields (pink, cyan, violet radial
gradients) whose weights follow the section's colour world; every kick lifts the fields' level
and scale by a few percent and the lift decays in 120 ms (7 frames).
Overlay: film grain, flash frames planned by `lyrics.flashes` (<= 3 bright per second, <= 4
frames), warm light leaks in the breakdown. Both go black for the "1 beat of black" windows.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from . import comp_writer as cw
from . import flashes as fl
from .style import CYAN, INK, PINK, SECTION_FIELDS, VIOLET, Fmt

KICK_DECAY_F = 7   # 120 ms at 60 fps


def _frames(t: float, fps: int = 60) -> int:
    return int(round(t * fps))


def black_windows(data: dict[str, Any]) -> list[tuple[float, float]]:
    """(start, end) seconds of the full-black beats: after the intro tape stop, before the final drop."""
    bar, beat = data["meta"]["bar_s"], data["meta"]["beat_s"]
    secs = {s["name"]: s for s in data["sections"]}
    out = [(15.75 * bar, 16 * bar)]
    fd = secs["Finaler Drop"]["start"]
    out.append((fd - beat, fd))
    return [(round(a, 3), round(b, 3)) for a, b in out]


def drum_kicks(data: dict[str, Any]) -> list[float]:
    """Kicks inside sections that have drums (the intro and the breakdown start without)."""
    secs = {s["name"]: s for s in data["sections"]}
    quiet = [(secs["Intro"]["start"], secs["Intro"]["end"] - 4 * data["meta"]["bar_s"]),
             (secs["Breakdown"]["start"], secs["Breakdown"]["start"] + 8 * data["meta"]["bar_s"])]
    quiet += black_windows(data)
    return [k for k in data["kicks"] if not any(a <= k < b for a, b in quiet)]


def _blob(c: cw.Comp, name: str, color: str, radius: float) -> str:
    """Radial neon field on a canvas twice the frame size, so its border never shows when it
    drifts (a merged larger image keeps its pixel size; `radius` is relative to the canvas)."""
    r, g, b = cw.hex_rgb(color)
    c.tool(name, "Background", {
        "Width": 2 * c.width, "Height": 2 * c.height,
        "Type": cw.FuID("Gradient"), "GradientType": cw.FuID("Radial"),
        "Start": (0.5, 0.5), "End": (0.5 + radius, 0.5),
        "Gradient": cw.Ctor("Gradient", {"Colors": {
            0: (r, g, b, 1.0),
            0.45: (r * 0.45, g * 0.45, b * 0.45, 0.45),
            1: (0.0, 0.0, 0.0, 0.0)}}),
    })
    return name


def background(data: dict[str, Any], f: Fmt) -> cw.Comp:
    n = _frames(data["meta"]["duration_s"])
    c = cw.Comp(f.width, f.height, n)
    base = c.background("Ground", INK, alpha=1.0)
    rng = random.Random(7)
    layers = []
    secs = data["sections"]
    for k, (name, color, rad) in enumerate((("Pink", PINK, 0.22), ("Cyan", CYAN, 0.21),
                                            ("Violet", VIOLET, 0.28))):
        blob = _blob(c, f"{name}Field", color, rad)
        tr = c.transform(f"{name}Drift", blob, motion_blur=False)
        # slow drift: a new target every 4 bars, eased
        keys = {}
        step = 4 * data["meta"]["bar_s"]
        t = 0.0
        phase = rng.random() * math.tau
        while t <= data["meta"]["duration_s"] + step:
            ang = phase + t * (0.11 + 0.03 * k)
            keys[_frames(min(t, data["meta"]["duration_s"]))] = (
                0.5 + 0.16 * math.cos(ang + k * 2.1), 0.5 + 0.13 * math.sin(1.3 * ang + k))
            t += step
        c.keyframes(tr, "Center", keys, ease="in_out_cubic")
        # weight per section, cross-faded over 0.25 s at every boundary
        wkeys: dict[int, float] = {}
        for s in secs:
            w = SECTION_FIELDS[s["name"]][k]
            a = _frames(s["start"])
            wkeys[a] = w
            wkeys[max(a + 1, _frames(s["end"]) - 15)] = w
        for a, b in black_windows(data):
            fa, fb = _frames(a), _frames(b)
            w_fb = _w_at(wkeys, fb)  # the drop starts at fb: its weight must survive
            for ff in list(wkeys):
                if fa <= ff <= fb:
                    del wkeys[ff]
            wkeys[fb] = w_fb
            wkeys[fa] = _w_at(wkeys, fa)
            wkeys[fa + 1] = 0.0
            wkeys[fb - 1] = 0.0
        wkeys = dict(sorted(wkeys.items()))
        m = c.merge(f"{name}Mix", base if not layers else layers[-1], tr, apply_mode="Screen")
        c.keyframes(m, "Blend", wkeys, ease="linear")
        layers.append(m)
    fields = layers[-1]
    # kick pulse: scale +5 % and level +25 % of the (dim) fields, decaying in 120 ms
    punch = c.transform("KickPunch", fields, motion_blur=False)
    lvl = c.merge("KickLevel", base, punch)
    skeys: dict[int, float] = {0: 1.0}
    bkeys: dict[int, float] = {0: 0.85}
    for k_t in drum_kicks(data):
        f0 = _frames(k_t)
        if f0 - 1 in skeys or f0 in skeys:
            continue
        skeys[f0 - 1] = skeys.get(f0 - 1, 1.0)
        skeys[f0] = 1.05
        skeys[f0 + KICK_DECAY_F] = 1.0
        bkeys[f0 - 1] = 0.85
        bkeys[f0] = 1.0
        bkeys[f0 + KICK_DECAY_F] = 0.85
    c.keyframes(punch, "Size", dict(sorted(skeys.items())), ease="out_cubic")
    c.keyframes(lvl, "Blend", dict(sorted(bkeys.items())), ease="out_cubic")
    c.output(lvl)
    return c


def _w_at(keys: dict[int, float], frame: int) -> float:
    before = [k for k in keys if k <= frame]
    return keys[max(before)] if before else 0.0


def flash_plan(data: dict[str, Any]) -> list[fl.Flash]:
    bar = data["meta"]["bar_s"]
    secs = {s["name"]: s for s in data["sections"]}
    plans = []
    # drops: bright flash on every bar one
    for name in ("Drop 1", "Finaler Drop"):
        s = secs[name]
        ones = [s["start"] + i * bar for i in range(s["bars"])]
        plans.append(fl.plan(ones, frames=3))
    # breakdown build: 1/4 (bars 68-70), 1/8 (70-72), 1/16 (72-75); the limiter keeps <= 3 bright/s
    hits: list[float] = []
    for b0, b1, div in ((68, 70, 4), (70, 72, 8), (72, 75, 16)):
        step = bar / div
        t = b0 * bar
        while t < b1 * bar - 1e-6:
            hits.append(round(t, 4))
            t += step
    plans.append(fl.plan(hits, frames=2))
    # post-chorus: soft pulses on the hook words
    pc = [w["start"] for ln in data["lines"] if ln["section"] == "Post-Chorus" for w in ln["words"]]
    plans.append(fl.plan(pc, bright=[False] * len(pc), frames=3))
    return fl.merge_plans(*plans)


def overlay(data: dict[str, Any], f: Fmt) -> cw.Comp:
    n = _frames(data["meta"]["duration_s"])
    c = cw.Comp(f.width, f.height, n)
    clear = c.background("Clear", (0, 0, 0), alpha=0.0)
    noise = c.fast_noise("Grain", scale=f.width / 3.0, detail=2.0, contrast=1.6, seethe_rate=4.0)
    grain = c.merge("GrainMix", clear, noise)
    gkeys = {0: 0.07}
    for a, b in black_windows(data):
        fa, fb = _frames(a), _frames(b)
        gkeys.update({fa: 0.07, fa + 1: 0.0, fb - 1: 0.0, fb: 0.07})
    gkeys[n - 1] = 0.07
    c.keyframes(grain, "Blend", dict(sorted(gkeys.items())), ease="linear")
    # light leaks in the breakdown (switched off elsewhere)
    secs = {s["name"]: s for s in data["sections"]}
    bd = secs["Breakdown"]
    r, g, b = cw.hex_rgb("#ff7a3d")
    c.tool("Leak", "Background", {
        "Width": 2 * f.width, "Height": 2 * f.height, "Type": cw.FuID("Gradient"),
        "GradientType": cw.FuID("Radial"), "Start": (0.5, 0.5), "End": (0.72, 0.5),
        "Gradient": cw.Ctor("Gradient", {"Colors": {0: (r, g, b, 1.0), 0.5: (r * 0.3, g * 0.2, b * 0.4, 0.3),
                                                    1: (0.0, 0.0, 0.0, 0.0)}})})
    lt = c.transform("LeakDrift", "Leak", motion_blur=False)
    fa, fb = _frames(bd["start"]), _frames(bd["start"] + 8 * data["meta"]["bar_s"])
    c.keyframes(lt, "Center", {fa: (0.2, 0.62), (fa + fb) // 2: (0.52, 0.55), fb: (0.8, 0.66)},
                ease="in_out_cubic")
    leak_on = c.merge("LeakMix", grain, lt, apply_mode="Screen")
    c.keyframes(leak_on, "Blend", {fa: 0.0, fa + 60: 0.32, fb - 90: 0.32, fb: 0.0}, ease="in_out_cubic")
    leak = c.dissolve("LeakSwitch", grain, leak_on)
    c.keyframes(leak, "Mix", {0: 0.0, fa: 1.0, fb: 0.0}, ease="step")
    # flashes
    flash_src = c.background("FlashWhite", "#ffffff", alpha=1.0)
    out = c.merge("FlashMix", leak, flash_src)
    fkeys: dict[int, float] = {0: 0.0}
    for fx in flash_plan(data):
        fkeys[fx.frame - 1] = 0.0
        fkeys[fx.frame] = fx.peak
        if fx.frames > 2:
            fkeys[fx.frame + fx.frames - 2] = fx.peak * 0.45
        fkeys[fx.frame + fx.frames - 1] = 0.0
    fkeys[n - 1] = 0.0
    c.keyframes(out, "Blend", dict(sorted(fkeys.items())), ease="linear")
    c.output(out)
    return c


def save_all(data: dict[str, Any], f: Fmt, out_dir: Path) -> dict[str, Path]:
    return {"background": background(data, f).save(out_dir / "background.comp"),
            "overlay": overlay(data, f).save(out_dir / "overlay.comp")}
