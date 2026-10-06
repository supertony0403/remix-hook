"""CLI: build the remix end to end.

    .venv/bin/python -m remix.bauen alles        # stems -> analyse -> render/master -> checks
    .venv/bin/python -m remix.bauen render       # only render + master + export
    .venv/bin/python -m remix.bauen pruefen      # QA report (out/pruefbericht.json)
    .venv/bin/python -m remix.bauen resolve      # fill the DaVinci Resolve project + render
    REMIX_OUT=/path .venv/bin/python -m remix.bauen render   # render side by side, then
    .venv/bin/python -m remix.bauen uebernehmen --von /path  # promote atomically into out/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python remix/bauen.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "remix"

from . import config  # noqa: E402

log = logging.getLogger("remix")


def step_stems(model: str = "htdemucs_ft") -> None:
    from . import stems

    for song, src in (("a", config.SRC_A), ("b", config.SRC_B)):
        if stems.stems_exist(song) and (config.WORK / f"{song}_mix.wav").exists():
            log.info("stems %s: vorhanden", song)
            continue
        stems.separate(song, src, model, shifts=2)


def step_analyse() -> dict:
    from .analyse import analyse

    res = analyse()
    log.info("analyse: %s", json.dumps({k: res[k] for k in ("a", "b")}, indent=None))
    return res


def step_render() -> dict:
    from .arrangement import ARRANGEMENT
    from .mix import export, mixdown
    from .render import render

    t = time.time()
    r = render(ARRANGEMENT)
    mr = mixdown(r.tracks)
    info = export(mr, extra={"a_vocal_trims_db": {k: round(v, 2) for k, v in r.trims.items()},
                             "autotune": {k: round(float(v), 3) for k, v in r.tune.items()}})
    log.info("render %.1f s: %s", time.time() - t, json.dumps(info))
    return info


def step_pruefen() -> dict:
    from .checks import run_all

    rep = run_all()
    (config.OUT / "pruefbericht.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    for k, v in rep["checks"].items():
        log.info("%-28s %s  %s", k, "OK " if v["ok"] else "FAIL", v.get("detail", ""))
    return rep


def step_uebernehmen(src_dir: Path) -> list[str]:
    """Promote a side-by-side render (REMIX_OUT=...) into out/ atomically: copy to a temp file in
    the target directory, then os.replace, so readers (Resolve) never see a half-written file."""
    import os
    import shutil

    moved = []
    names = [f"spuren/{k}.wav" for k in config.TRACK_NAMES] + [
        "remix-hook.wav", "remix-hook.mp3", "report.json", "pruefbericht.json"]
    for name in names:
        src = src_dir / name
        if not src.exists():
            continue
        dst = config.OUT / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        moved.append(name)
    log.info("uebernommen nach %s: %s", config.OUT, moved)
    return moved


def step_resolve() -> dict:
    from .resolve_project import fill_project

    return fill_project()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="remix.bauen")
    ap.add_argument("schritt", choices=["stems", "analyse", "render", "pruefen", "resolve", "uebernehmen", "alles"])
    ap.add_argument("--von", type=Path, help="uebernehmen: Quellordner eines REMIX_OUT-Renders")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.schritt in ("stems", "alles"):
        step_stems()
    if args.schritt in ("analyse", "alles"):
        step_analyse()
    if args.schritt in ("render", "alles"):
        step_render()
    if args.schritt in ("pruefen", "alles"):
        step_pruefen()
    if args.schritt == "uebernehmen":
        if not args.von:
            ap.error("uebernehmen braucht --von <ordner>")
        step_uebernehmen(args.von)
    if args.schritt == "resolve":
        state = step_resolve()
        log.info("resolve: %s", json.dumps(state, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
