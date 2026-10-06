"""Fusion comps of the lyric video: rules from the brief, checked on the generated .comp text."""

from __future__ import annotations

import re

import pytest

from lyrics import backdrop, flashes, scenes, style, timing
from lyrics.comp_writer import MB_QUALITY_MAX


@pytest.fixture(scope="module")
def data() -> dict:
    if not timing.OUT_JSON.exists():
        pytest.skip("work/lyrics/lyrics.json missing — run `python -m lyrics.timing`")
    return timing.load()


@pytest.fixture(scope="module", params=list(style.FORMATS))
def comps(request, data: dict) -> dict[str, str]:
    f = style.fmt(request.param)
    out = {"background": backdrop.background(data, f).to_text(),
           "overlay": backdrop.overlay(data, f).to_text()}
    for s in data["sections"]:
        out[s["name"]] = scenes.build_section(data, f, s["name"]).to_text()
    return out


# ----------------------------------------------------------------------------- photosensitivity
def test_flash_limiter_unit() -> None:
    hits = [i * 0.1 for i in range(40)]  # 10 per second
    plan = flashes.plan(hits, frames=2)
    assert flashes.max_bright_in_any_second(plan) <= flashes.MAX_BRIGHT_PER_S
    assert all(f.frames <= flashes.MAX_FRAMES for f in plan)
    with pytest.raises(ValueError):
        flashes.plan([0.0], frames=5)


def test_flash_plan_of_the_video(data: dict) -> None:
    plan = backdrop.flash_plan(data)
    assert plan, "drops and build need flashes"
    assert flashes.max_bright_in_any_second(plan) <= 3
    assert all(f.frames <= 4 for f in plan)
    dim = [f for f in plan if not f.bright]
    assert all(f.peak <= flashes.DIM_MAX for f in dim)
    # the build gets denser: more hits in the last bars of the breakdown than in its first build bar
    bar = data["meta"]["bar_s"]
    n68 = sum(68 * bar <= f.frame / 60 < 69 * bar for f in plan)
    n74 = sum(74 * bar <= f.frame / 60 < 75 * bar for f in plan)
    assert n74 > n68


# ----------------------------------------------------------------------------- render performance
def test_motion_blur_quality_capped_and_keyed(comps: dict[str, str]) -> None:
    for name, text in comps.items():
        for q in re.findall(r"Quality = Input \{ Value = (\d+)", text):
            assert int(q) <= MB_QUALITY_MAX, name
        # every tool with motion blur on has Quality keyed (1 outside its motion windows)
        for block in re.findall(r"= Transform \{.*?\n\t\t\},", text, flags=re.S):
            if "MotionBlur = Input { Value = 1" in block:
                assert "Quality = Input {\n" in block or "Quality = Input { SourceOp" in block \
                    or re.search(r"Quality = Input \{\s*SourceOp", block), name


def test_one_comp_per_section_spans_exactly_the_section(data: dict) -> None:
    f = style.fmt("16x9")
    total = 0
    for s in data["sections"]:
        c = scenes.build_section(data, f, s["name"])
        a, b = scenes.span(data, s["name"])
        assert c.frames == int(round(b * 60)) - int(round(a * 60))
        total += c.frames
    assert total == int(round(data["meta"]["duration_s"] * 60))


# ----------------------------------------------------------------------------- readability
def test_every_line_is_shown_until_its_end_plus_hold(data: dict) -> None:
    cov = scenes.coverage(data)
    for ln in data["lines"]:
        wins = sorted(cov.get(ln["id"], []))
        assert wins, f"{ln['id']} {ln['text']!r} is in no comp"
        t = ln["start"]
        need = min(ln["end"] + scenes.HOLD, data["meta"]["duration_s"])
        for a, b in wins:
            if a <= t + 1 / 60:
                t = max(t, b)
        assert t >= need - 1 / 60, (ln["id"], ln["text"], wins, need)


def _spline_keys(text: str, spline: str) -> dict[int, float]:
    m = re.search(rf"\t\t{spline} = BezierSpline \{{(.*?)\n\t\t\}},", text, flags=re.S)
    assert m, spline
    return {int(f): float(v) for f, v in re.findall(r"\[(-?\d+)\] = \{ (-?[\d.e-]+)", m.group(1))}


def _value_at(keys: dict[int, float], frame: int) -> float:
    """Piecewise-linear bound of the eased curve (in_out_cubic stays between its keys)."""
    fs = sorted(keys)
    if frame <= fs[0]:
        return keys[fs[0]]
    for a, b in zip(fs, fs[1:]):
        if a <= frame <= b:
            return min(keys[a], keys[b])  # conservative: the lower of the two neighbours
    return keys[fs[-1]]


def test_lines_are_fully_visible_at_end_plus_hold(data: dict) -> None:
    """The actual fade curves: at last word + HOLD every line is still at >= 90 % (16:9)."""
    f = style.fmt("16x9")
    names = [s["name"] for s in data["sections"]]
    checked = 0
    for name in names:
        a, b = scenes.span(data, name)
        text = scenes.build_section(data, f, name).to_text()
        f0 = int(round(a * 60))
        fades: dict[str, list[str]] = {}
        for tool, spline in re.findall(r"\t\t(L\d{3}\w*F) = Dissolve \{.*?Mix = Input \{\s*SourceOp = \"(\w+)\"",
                                       text, flags=re.S):
            fades.setdefault(tool[:4], []).append(spline)
        for ln in data["lines"]:
            t = ln["end"] + scenes.HOLD
            if not (a <= t < b) or ln["id"] not in fades:
                continue
            frame = int(round(t * 60)) - f0
            best = max(_value_at(_spline_keys(text, sp), frame) for sp in fades[ln["id"]])
            assert best >= 0.9, (name, ln["id"], ln["text"], best)
            checked += 1
    assert checked >= len(data["lines"]) * 0.8, checked


def test_every_chop_is_rendered(data: dict) -> None:
    """Each chop lies in a comp that renders chops (FEEL flicker or the intro grid)."""
    f = style.fmt("16x9")
    for name in [s["name"] for s in data["sections"]]:
        a, b = scenes.span(data, name)
        chops = [ch for ch in data["chops"] if a <= ch["t"] < b]
        if not chops:
            continue
        text = scenes.build_section(data, f, name).to_text()
        if any(ch["kind"] == "feel" for ch in chops):
            assert '"FEEL"' in text and "FeelPos" in text, name
        if any(ch["kind"] == "dont_you" for ch in chops):
            assert "Grid0" in text, name


def test_all_sung_words_are_in_the_comps(data: dict, comps: dict[str, str]) -> None:
    texts = "\n".join(comps.values())
    shown = set(re.findall(r'StyledText = Input \{ Value = "([^"]*)"', texts))
    for ln in data["lines"]:
        for w in ln["words"]:
            assert w["w"].upper() in shown or w["w"] in shown, (ln["id"], w["w"])


def test_comps_have_output_and_no_dangling_links(comps: dict[str, str]) -> None:
    for name, text in comps.items():
        assert "MediaOut1 = MediaOut" in text
        assert re.search(r"MediaOut1 = MediaOut \{\s*Inputs = \{\s*Index[^\n]*\n\s*Input = Input \{ SourceOp", text), name
        tools = set(re.findall(r"^\t\t(\w+) = \w+ \{", text, flags=re.M))
        for op in re.findall(r'SourceOp = "(\w+)"', text):
            assert op in tools, (name, op)


def test_fonts_are_the_brief_fonts(comps: dict[str, str]) -> None:
    fonts = set()
    for text in comps.values():
        fonts |= set(re.findall(r'Font = Input \{ Value = "([^"]+)"', text))
    assert fonts <= {"Anton", "Bebas Neue", "Space Grotesk"}, fonts
