"""Build the lyric comps and put them into Resolve, section by section.

    .venv/bin/python -m lyrics.build comps                    # write all .comp files (both formats)
    .venv/bin/python -m lyrics.build place 16x9 background overlay Intro "Drop 1" ...
    .venv/bin/python -m lyrics.build frames 16x9 tag 300 900  # control PNGs from the timeline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import backdrop, resolve_io, style

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work" / "lyrics"
COMPS = WORK / "comps"
FPS = 60


def load_data() -> dict[str, Any]:
    full = WORK / "lyrics.json"
    if full.exists():
        return json.loads(full.read_text())
    return json.loads((WORK / "rhythm_interim.json").read_text())


def comp_dir(fmt: str) -> Path:
    return COMPS / fmt


def section_span_frames(data: dict[str, Any], name: str) -> tuple[int, int]:
    for s in data["sections"]:
        if s["name"] == name:
            a = s.get("comp_start", s["start"])
            b = s.get("comp_end", s["end"])
            return int(round(a * FPS)), int(round(b * FPS))
    raise KeyError(name)


def build_comps(data: dict[str, Any], fmts: list[str], only: list[str] | None = None) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for fmt in fmts:
        f = style.fmt(fmt)
        d = comp_dir(fmt)
        res: dict[str, Path] = {}
        if not only or "background" in only:
            res["background"] = backdrop.background(data, f).save(d / "background.comp")
        if not only or "overlay" in only:
            res["overlay"] = backdrop.overlay(data, f).save(d / "overlay.comp")
        if data.get("lines"):
            from . import scenes

            for s in data["sections"]:
                if only and s["name"] not in only:
                    continue
                comp = scenes.build_section(data, f, s["name"])
                res[s["name"]] = comp.save(d / f"{scenes.slug(s['name'])}.comp")
        out[fmt] = res
    return out


def place(fmt: str, what: list[str], data: dict[str, Any]) -> None:
    resolve = resolve_io.connect()
    p = resolve_io.project(resolve)
    resolve_io.ensure_fonts(resolve)
    resolve_io.make_carrier(fmt, style.FORMATS[fmt])
    placer = resolve_io.Placer(resolve, p, fmt)
    n = int(round(data["meta"]["duration_s"] * FPS))
    d = comp_dir(fmt)
    for w in what:
        if w in ("background", "overlay"):
            track = resolve_io.TRACKS[w]
            placer.place(track, 0, n, d / f"{w}.comp", f"{w} {fmt}")
        else:
            from . import scenes

            a, b = section_span_frames(data, w)
            placer.place(resolve_io.TRACKS["lyrics"], a, b - a, d / f"{scenes.slug(w)}.comp", f"{w} {fmt}")
    # leave the 16x9 lyric timeline current, as Anthony had it
    p.SetCurrentTimeline(resolve_io.find_timeline(p, resolve_io.TIMELINES["16x9"]))


def frames(fmt: str, tag: str, nums: list[int]) -> list[Path]:
    resolve = resolve_io.connect()
    p = resolve_io.project(resolve)
    out = resolve_io.render_frames(resolve, p, fmt, nums, tag, WORK / "frames")
    p.SetCurrentTimeline(resolve_io.find_timeline(p, resolve_io.TIMELINES["16x9"]))
    return out


def main(argv: list[str]) -> None:
    cmd = argv[0]
    data = load_data()
    if cmd == "comps":
        fmts = [a for a in argv[1:] if a in style.FORMATS] or list(style.FORMATS)
        only = [a for a in argv[1:] if a not in style.FORMATS] or None
        for fmt, res in build_comps(data, fmts, only).items():
            for k, v in res.items():
                print(fmt, k, v, v.stat().st_size)
    elif cmd == "place":
        place(argv[1], argv[2:], data)
    elif cmd == "frames":
        for pth in frames(argv[1], argv[2], [int(x) for x in argv[3:]]):
            print(pth)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
