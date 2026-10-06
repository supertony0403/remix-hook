"""Timing data of the lyric video (work/lyrics/lyrics.json from lyrics.timing)."""

from __future__ import annotations

import re

import pytest

from lyrics import timing
from lyrics.timing import W

DATA_PATH = timing.OUT_JSON


@pytest.fixture(scope="module")
def data() -> dict:
    if not DATA_PATH.exists():
        pytest.skip("work/lyrics/lyrics.json missing — run `python -m lyrics.timing`")
    return timing.load()


def _sections(data: dict) -> dict:
    return {s["name"]: s for s in data["sections"]}


def test_every_word_lies_in_its_section(data: dict) -> None:
    secs = _sections(data)
    bar = data["meta"]["bar_s"]
    for ln in data["lines"]:
        s = secs[ln["section"]]
        for w in ln["words"]:
            # a pickup (anacrusis) may start up to one bar before its section
            assert s["start"] - timing.PICKUP_BARS * bar - 1e-6 <= w["start"] < s["end"], (ln["id"], w)
            assert w["end"] <= s["end"] + 1.0, (ln["id"], w)
            assert w["start"] < w["end"], (ln["id"], w)


def test_lines_do_not_overlap_and_are_sorted(data: dict) -> None:
    lines = data["lines"]
    for a, b in zip(lines, lines[1:]):
        assert a["end"] <= b["start"] + 1e-6, (a["id"], a["text"], b["id"], b["text"])


def test_line_rules(data: dict) -> None:
    for ln in data["lines"]:
        assert 1 <= len(ln["words"]) <= timing.MAX_WORDS, ln
        assert ln["singer"] in timing.SINGERS
        ws = ln["words"]
        for a, b in zip(ws, ws[1:]):
            assert b["start"] - a["end"] <= timing.PAUSE_S + 1e-6, (ln["id"], a, b)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z' ]", "", text.lower().replace("’", "'"))


def _found(data: dict, phrase: str) -> set[str]:
    """Sections in which `phrase` occurs in the word stream (across line breaks)."""
    words = [(w["w"], ln["section"]) for ln in data["lines"] for w in ln["words"]]
    target = _norm(phrase).split()
    hits = set()
    for i in range(len(words) - len(target) + 1):
        if [_norm(w) for w, _ in words[i:i + len(target)]] == target:
            hits.add(words[i][1])
    return hits


def test_hook_and_refrain_found_where_expected(data: dict) -> None:
    hook = _found(data, "don't you feel")
    assert {"Intro", "Post-Chorus", "Finaler Drop", "Outro"} <= hook, hook
    fall = _found(data, "fall asleep")
    assert {"B Refrain 1", "Finaler Drop"} <= fall, fall


def test_section_comps_cover_the_whole_song(data: dict) -> None:
    secs = data["sections"]
    assert secs[0]["comp_start"] == 0
    assert secs[-1]["comp_end"] == pytest.approx(data["meta"]["duration_s"])
    for a, b in zip(secs, secs[1:]):
        assert a["comp_end"] == pytest.approx(b["comp_start"])
        assert b["comp_start"] <= b["start"] + 1e-6


def test_rhythm_grid_and_chops(data: dict) -> None:
    beat = data["meta"]["beat_s"]
    assert data["meta"]["bpm"] == 150.0
    assert data["beats"][1] - data["beats"][0] == pytest.approx(beat)
    assert len(data["bars"]) == 101
    assert len(data["kicks"]) > 100
    kicks_intro = [k for k in data["kicks"] if k < 25.6]
    assert not kicks_intro, "the intro has no drums"
    kinds = {c["kind"] for c in data["chops"]}
    assert kinds == {"feel", "dont_you"}
    assert sum(c["kind"] == "feel" for c in data["chops"]) >= 30


def test_timing_check_is_tight(data: dict) -> None:
    tc = data["meta"]["timing_check"]
    assert tc["n"] > 100
    assert tc["median_ms"] < 60


# ----------------------------------------------------------------------------- pure functions
def _w(text: str, t: float, src: str, p: float = 0.9) -> W:
    return W(text, t, t + 0.3, p, src)


def test_consensus_replaces_only_with_agreement() -> None:
    cands = {
        "O_de": [_w("Ball", 0.0, "O_de"), _w("asleep", 0.4, "O_de"), _w("waiting", 0.8, "O_de")],
        "O_en": [_w("Fall", 0.0, "O_en"), _w("asleep", 0.4, "O_en")],
        "S_en": [_w("Fall", 0.0, "S_en"), _w("asleep", 0.4, "S_en")],
        "T": [_w("asleep", 0.4, "T"), _w("waiting", 0.8, "T")],
    }
    out, log = timing.consensus(cands)
    assert [w.text for w in out] == ["Fall", "asleep", "waiting"]
    assert any(e["op"] == "replace" and e["from"] == "Ball" and e["to"] == "Fall" for e in log), log


def test_zero_length_words_keep_their_order() -> None:
    """Regression (verse 1a, 55.99 s): zero-length 'ich' must not duplicate or swap words."""
    z = lambda t, a, src: W(t, a, a, 0.9, src)  # noqa: E731
    cands = {
        "S_de": [_w("Gott", 55.7, "S_de"), z("ich", 55.99, "S_de"), z("bin", 55.99, "S_de"),
                 _w("im", 56.21, "S_de")],
        "S_en": [_w("Gott", 55.7, "S_en"), _w("ich", 55.95, "S_en"), _w("bin", 56.05, "S_en"),
                 _w("im", 56.21, "S_en")],
        "O_de": [_w("Gott", 55.7, "O_de"), _w("ich", 55.97, "O_de"), _w("bin", 56.06, "O_de"),
                 _w("im", 56.22, "O_de")],
    }
    out, _ = timing.consensus(cands)
    assert [w.text for w in out] == ["Gott", "ich", "bin", "im"]
    assert all(w.end > w.start for w in out)


def test_consensus_keeps_unsupported_word_without_majority() -> None:
    cands = {"O_de": [_w("weh", 0.0, "O_de", 0.8)], "S_en": [_w("way", 0.0, "S_en")],
             "T": [_w("we", 0.0, "T")]}
    out, _ = timing.consensus(cands)
    assert len(out) == 1  # one of the candidates, nothing made up
    assert out[0].text in {"weh", "way", "we"}


def test_split_long_breaks_at_the_biggest_gap() -> None:
    ws = [(W(t, s, s + 0.2, 0.9, "x"), "B-Refrain") for t, s in
          [("a", 0.0), ("b", 0.25), ("c", 0.5), ("d", 0.75), ("e", 1.3), ("f", 1.55), ("g", 1.8), ("h", 2.05)]]
    parts = timing._split_long(ws)
    assert [[w.text for w, _ in p] for p in parts] == [["a", "b", "c", "d"], ["e", "f", "g", "h"]]


def test_clip_time_map_roundtrip() -> None:
    clip = next(c for c in timing.vocal_clips() if c.label == "B refrain 1")
    s0, _ = timing.clip_range_src(clip)
    r0, _ = timing.clip_range_remix(clip)
    assert timing.to_remix(clip, s0) == pytest.approx(r0)
