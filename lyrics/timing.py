"""Lyrics and rhythm timing for the lyric video -> work/lyrics/lyrics.json.

Three transcriptions per vocal clip of the arrangement (remix/arrangement.py, read only):

* S  faster-whisper on the remix stem (out/spuren/A2_b_gesang.wav, A3_a_hook.wav), per clip,
     in remix time — the base, as the brief asks.
* O  faster-whisper on the *un-tuned* source vocal stem (stems/{a,b}/vocals.wav) over the clip's
     source range, mapped into remix time through the clip (B plays at native tempo, A is stretched
     linearly by STRETCH), an independent second opinion.
* T  the original transcripts work/a_words.json / work/b_words.json, mapped the same way.

S is aligned against T and O (difflib on normalised tokens). A block of S is only replaced when
T and O agree on the replacement (the original transcript supports it); otherwise S stays. Nothing
is invented: every word in the output comes from one of the three transcriptions, except the A hook,
whose text is the brief's "don't you feel" (Whisper hears "without you here" there, see report)
and whose syllable times come from the arrangement's measured onsets.

Run:  .venv/bin/python -m lyrics.timing            (uses cached whisper output if present)
      .venv/bin/python -m lyrics.timing --whisper  (re-run faster-whisper, ~5 min CPU)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from remix import arrangement as arr

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work" / "lyrics"
OUT_JSON = WORK / "lyrics.json"
SPUREN = ROOT / "out" / "spuren"
WHISPER_PY = Path.home() / "Documents/Programmierung/nomissuccess-spot/.venv/bin/python"
HF_HOME = "/mnt/steam-library/remix-hook/hf"
MODEL = "medium"

FPS = 60
DURATION_S = 164.0
BAR = arr.BAR
BEAT = arr.BEAT

PAUSE_S = 0.35        # a gap longer than this ends a line
MAX_WORDS = 6
HOLD_S = 0.25         # every line stays at least this long after its last word
PICKUP_BARS = 1.0     # a line may start up to one bar before its section (anacrusis)

SINGERS = ("B-Rap", "B-Refrain", "A-Hook", "A-Refrain")
HOOK_WORDS = ("don't", "you", "feel")


# ----------------------------------------------------------------------------- clip time maps
def src_time(source: str, src_bar: float) -> float:
    """Seconds in the *original* source file of absolute source bar `src_bar`."""
    if source.startswith("b:"):
        return arr.B_DOWNBEAT0 + src_bar * BAR
    if source.startswith("a:"):
        return arr.A_DOWNBEAT0 + src_bar * arr.A_BAR
    raise ValueError(f"no time map for {source}")


def src_bar_of(source: str, t: float) -> float:
    if source.startswith("b:"):
        return (t - arr.B_DOWNBEAT0) / BAR
    if source.startswith("a:"):
        return (t - arr.A_DOWNBEAT0) / arr.A_BAR
    raise ValueError(f"no time map for {source}")


def to_remix(clip: arr.Clip, t_src: float) -> float:
    """Original-source seconds -> remix seconds through `clip` (linear in bars)."""
    return (clip.at + (src_bar_of(clip.source, t_src) - clip.src)) * BAR


def clip_range_remix(clip: arr.Clip) -> tuple[float, float]:
    return clip.remix_start * BAR, clip.remix_end * BAR


def clip_range_src(clip: arr.Clip) -> tuple[float, float]:
    return src_time(clip.source, clip.start), src_time(clip.source, clip.end)


def singer_of(clip: arr.Clip) -> str:
    lab = clip.label.lower()
    if clip.track == arr.T2:
        return "B-Rap" if "verse" in lab else "B-Refrain"
    return "A-Refrain" if "refrain" in lab else "A-Hook"


def vocal_clips() -> list[arr.Clip]:
    return [c for c in arr.ARRANGEMENT.clips if c.track in (arr.T2, arr.T3)]


def is_hook_clip(clip: arr.Clip) -> bool:
    return clip.track == arr.T3 and singer_of(clip) == "A-Hook"


# ----------------------------------------------------------------------------- whisper jobs
def whisper_jobs() -> list[dict]:
    jobs = []
    for i, c in enumerate(vocal_clips()):
        r0, r1 = clip_range_remix(c)
        stem = SPUREN / ("A2_b_gesang.wav" if c.track == arr.T2 else "A3_a_hook.wav")
        jobs.append({"id": f"S{i}", "wav": str(stem), "t0": r0 - 0.25, "t1": r1 + 0.35})
        s0, s1 = clip_range_src(c)
        orig = ROOT / "stems" / c.source[0] / "vocals.wav"
        jobs.append({"id": f"O{i}", "wav": str(orig), "t0": s0 - 0.25, "t1": s1 + 0.35})
    return jobs


def run_whisper(model: str = MODEL, language: str = "en") -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    jobs_path = WORK / "whisper_jobs.json"
    jobs_path.write_text(json.dumps(whisper_jobs(), indent=1))
    out = WORK / f"whisper_clips_{model}{'' if language == 'en' else '_' + language}.json"
    env = {"HF_HOME": HF_HOME, "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())}
    subprocess.run([str(WHISPER_PY), str(ROOT / "lyrics" / "whisper_run.py"), str(jobs_path), model,
                    str(out), language], check=True, env=env)
    return out


# ----------------------------------------------------------------------------- tokens
def norm(word: str) -> str:
    w = word.lower().replace("’", "'").replace("in'", "ing")
    return "".join(ch for ch in w if ch.isalnum())


def clean(word: str) -> str:
    """Display form: strip surrounding punctuation except apostrophes inside the word."""
    return word.strip().strip(".,!?;:\"“”…-").strip()


def ends_sentence(word: str) -> bool:
    return word.rstrip().endswith((".", "!", "?"))


@dataclass
class W:
    text: str        # as written by the transcriber (punctuation kept for line breaks)
    start: float     # remix seconds
    end: float
    p: float
    src: str         # S | O | T | arrangement

    @property
    def key(self) -> str:
        return norm(self.text)


def mapped_original(clip: arr.Clip, words: list[dict], src: str) -> list[W]:
    s0, s1 = clip_range_src(clip)
    out = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        if s0 - 0.05 <= mid <= s1 + 0.05:
            out.append(W(w["word"], to_remix(clip, w["start"]), to_remix(clip, w["end"]),
                          float(w.get("p", 0.5)), src))
    return out


PRIORITY = ("O_de", "O_en", "S_de", "S_en", "T")  # tie-break: un-tuned original stem first
SLOT_S = 0.45  # two words "say the same" if their keys match and their midpoints are this close


def _mid(w: W) -> float:
    return (w.start + w.end) / 2


def _support(w: W, own: str, cands: dict[str, list[W]]) -> set[str]:
    return {name for name, ws in cands.items() if name != own
            and any(x.key == w.key and abs(_mid(x) - _mid(w)) < SLOT_S for x in ws)}


def consensus(cands: dict[str, list[W]]) -> tuple[list[W], list[dict]]:
    """Pick the transcription the others agree with most, then fix single words by majority.

    Candidates: S_en/S_de = remix stem, O_en/O_de = un-tuned source stem, T = original transcript
    (all in remix time). A word is replaced only if >= 2 *other* transcriptions agree on a
    different word in the same slot; a word missing from the pick is inserted only if >= 3 agree.
    Nothing is invented: every output word comes from one of the candidates."""
    log: list[dict] = []
    cands = {k: _min_length(v) for k, v in cands.items() if v}
    if not cands:
        return [], log
    score = {k: sum(len(_support(w, k, cands)) for w in ws) for k, ws in cands.items()}
    best_name = max(cands, key=lambda k: (score[k], -PRIORITY.index(k)))
    best = [W(w.text, w.start, w.end, w.p, best_name) for w in cands[best_name]]
    log.append({"op": "pick", "from": "", "to": best_name, "scores": score})
    out: list[W] = []
    last_used: W | None = None
    for w in best:
        own = _support(w, best_name, cands)
        votes: dict[str, list[tuple[str, W]]] = {}
        for name, ws in cands.items():
            if name == best_name:
                continue
            for x in ws:
                if abs(_mid(x) - _mid(w)) < max(0.25, (w.end - w.start) / 2):
                    votes.setdefault(x.key, []).append((name, x))
        # an unsupported word is replaced if >= 2 other transcriptions agree on its slot
        agreed = [] if own else [(k, v) for k, v in votes.items()
                                 if k != w.key and len({n for n, _ in v}) >= 2]
        if agreed:
            k, v = max(agreed, key=lambda kv: len({n for n, _ in kv[1]}))
            name, x = min(v, key=lambda nv: PRIORITY.index(nv[0]))
            if last_used is not None and last_used is x:
                out[-1].end = max(out[-1].end, w.end)  # two picked words map to one ("I sleep" -> "asleep")
                continue
            log.append({"op": "replace", "t": round(w.start, 2), "from": w.text, "to": x.text,
                        "by": sorted({n for n, _ in v})})
            out.append(W(x.text, w.start, w.end, x.p, name))
            last_used = x
        elif not own and w.p < 0.15:
            log.append({"op": "drop", "t": round(w.start, 2), "from": w.text, "to": "", "p": w.p})
            last_used = None  # nobody else heard it and Whisper itself is unsure: a bleed, not a word
        else:
            out.append(w)
            last_used = None
    # words >= 3 transcriptions agree on that the pick dropped
    seen: list[W] = []
    for name in PRIORITY:
        for x in cands.get(name, []):
            if name == best_name or any(y.key == x.key and abs(_mid(y) - _mid(x)) < SLOT_S for y in seen):
                continue
            sup = _support(x, name, cands) - {best_name}
            if len(sup) + 1 >= 3 and not any(o.start < x.end and o.end > x.start for o in out) \
                    and not any(o.key == x.key and abs(_mid(o) - _mid(x)) < SLOT_S for o in out):
                log.append({"op": "insert", "t": round(x.start, 2), "from": "", "to": x.text,
                            "by": sorted(sup | {name})})
                out.append(W(x.text, x.start, x.end, x.p, name))
                seen.append(x)
    out.sort(key=lambda w: w.start)
    return [_borrow_punctuation(w, cands) for w in out], log


def _min_length(words: list[W], min_s: float = 0.06) -> list[W]:
    """Whisper sometimes returns zero-length words (often several at the same start): spread them
    so every word has a length and order is kept."""
    out = [W(w.text, w.start, w.end, w.p, w.src) for w in words]
    for i, w in enumerate(out):
        if i and w.start < out[i - 1].start + 0.02:
            w.start = out[i - 1].start + 0.02
        if w.end - w.start < min_s:
            nxt = out[i + 1].start if i + 1 < len(out) else w.start + min_s
            w.end = max(w.start + 0.02, min(w.start + min_s, max(nxt, w.start + 0.02)))
    return out


def _borrow_punctuation(w: W, cands: dict[str, list[W]]) -> W:
    """Phrase ends (',' '.' '?' '!') from any transcription that heard the same word in the same
    slot; only used to break lines, the displayed word is stripped again."""
    if w.text.rstrip()[-1:] in ",.?!":
        return w
    for name in PRIORITY:
        for x in cands.get(name, []):
            tail = x.text.rstrip()[-1:]
            if x.key == w.key and tail in ",.?!" and abs(_mid(x) - _mid(w)) < 0.25:
                return W(w.text + tail, w.start, w.end, w.p, w.src)
    return w


def snap_to_stem(words: list[W], stem: list[W], dev: list[float]) -> list[W]:
    """Use the remix stem's own word times where it heard the same word (brief: timing from the
    remix stems); otherwise keep the mapped source time. Collects |delta| for the report."""
    out = []
    for w in words:
        m = [x for x in stem if x.key == w.key and abs(x.start - w.start) < 0.35]
        if m:
            x = min(m, key=lambda x: abs(x.start - w.start))
            dev.append(abs(x.start - w.start))
            out.append(W(w.text, x.start, x.end, w.p, w.src))
        else:
            out.append(w)
    return out


def hook_words(clip: arr.Clip) -> list[W]:
    """'don't you feel' (or 'don't you') at the measured syllable onsets of the A hook."""
    k_off = clip.src - 1.0  # phrase k sits 2k A bars later; the onsets are given for phrase 0
    onsets = [arr.DONT + k_off, arr.YOU + k_off, arr.FEEL + k_off]
    ends = [arr.YOU + k_off, arr.FEEL + k_off, clip.end]
    out = []
    for word, a, b in zip(HOOK_WORDS, onsets, ends):
        if a >= clip.end - 0.02:
            break
        start = (clip.at + a - clip.src) * BAR
        end = (clip.at + min(b, clip.end) - clip.src) * BAR
        out.append(W(word, start, end, 1.0, "arrangement"))
    return out


# ----------------------------------------------------------------------------- sections, lines
@dataclass
class Sec:
    name: str
    start: float
    end: float
    start_bar: int
    bars: int


def sections() -> list[Sec]:
    out = [Sec(s.name, s.start * BAR, s.end * BAR, s.start, s.bars) for s in arr.ARRANGEMENT.sections]
    out[-1].end = DURATION_S  # the outro comp runs to the end of the audio (tail + end card)
    return out


def section_of(t: float, secs: list[Sec]) -> Sec:
    for s in secs:
        if s.start <= t < s.end:
            return s
    return secs[-1] if t >= secs[-1].start else secs[0]


def build_lines(words: list[tuple[W, str]], secs: list[Sec]) -> list[dict]:
    """Group (word, singer) into lines: pause > PAUSE_S, sentence end, MAX_WORDS, singer change.
    Each line belongs to the section of its first word, or the next one if it is a pickup."""
    words = sorted(words, key=lambda x: x[0].start)
    groups: list[list[tuple[W, str]]] = []
    for w, sg in words:
        if groups:
            prev, psg = groups[-1][-1]
            new = (w.start - prev.end > PAUSE_S or ends_sentence(prev.text) or psg != sg
                   or section_of(w.start, secs).name != section_of(prev.start, secs).name
                   and not _pickup(groups[-1], w, secs))
            if not new:
                groups[-1].append((w, sg))
                continue
        groups.append([(w, sg)])
    # words that open a pause-delimited phrase somewhere are good places to break long lines
    starts = {g[0][0].key for g in groups if 2 <= len(g) <= MAX_WORDS}
    groups = [piece for g in groups for piece in _split_long(g, starts)]
    lines = []
    for g in groups:
        first, last = g[0][0], g[-1][0]
        sec = section_of(first.start, secs)
        nxt = section_of(last.end - 1e-3, secs)
        if nxt.name != sec.name:  # pickup: the line belongs to where it is mostly sung
            mid = (first.start + last.end) / 2
            sec = section_of(mid, secs)
        lines.append({
            "section": sec.name, "singer": g[0][1],
            "text": " ".join(clean(w.text) for w, _ in g),
            "start": round(first.start, 3), "end": round(last.end, 3),
            "words": [{"w": clean(w.text), "start": round(w.start, 3), "end": round(w.end, 3),
                       "p": round(w.p, 3), "src": w.src, "key": _is_key(w.text)} for w, _ in g],
        })
    for i, ln in enumerate(lines):
        ln["id"] = f"L{i:03d}"
    return lines


def _split_long(g: list[tuple[W, str]], starts: set[str] = frozenset()) -> list[list[tuple[W, str]]]:
    """Split a group longer than MAX_WORDS at its largest internal gap (a capitalised word other
    than "I" counts as a phrase start and gets a small bonus), recursively."""
    if len(g) <= MAX_WORDS:
        return [g]
    best_i, best_v = 0, -1.0
    for i in range(2, len(g) - 1):  # keep at least two words on each side
        a, b = g[i - 1][0], g[i][0]
        word = clean(b.text)
        v = (b.start - a.end) + (0.08 if word[:1].isupper() and word != "I" else 0.0) \
            + (0.12 if a.text.rstrip().endswith((",", ";", ":")) else 0.0) \
            + (0.20 if b.key in starts and b.key not in {"i", "you", "to", "the", "a"} else 0.0) \
            - 0.02 * abs(i - len(g) / 2)  # prefer balanced lines
        if v > best_v:
            best_i, best_v = i, v
    return _split_long(g[:best_i], starts) + _split_long(g[best_i:], starts)


def _pickup(group: list[tuple[W, str]], w: W, secs: list[Sec]) -> bool:
    """True if the running group is a pickup (anacrusis) into w's section."""
    sec = section_of(w.start, secs)
    return all(sec.start - PICKUP_BARS * BAR <= x.start < sec.start for x, _ in group)


KEYWORDS = {"love", "drug", "drugs", "chaos", "storm", "heart", "feel", "fight", "fighting",
            "fightin", "blood", "boiling", "bleeding", "dreams", "die", "hell", "heaven", "health",
            "romance", "need", "stay", "fail", "price", "game", "noise", "mad", "true", "asleep",
            "crazy", "respect", "wrong", "born"}


def _is_key(word: str) -> bool:
    return norm(word) in KEYWORDS


def comp_spans(lines: list[dict], secs: list[Sec]) -> dict[str, tuple[float, float]]:
    """Time span of each section's comp: the section, pulled earlier for a pickup line."""
    spans = {s.name: [s.start, s.end] for s in secs}
    names = [s.name for s in secs]
    for ln in lines:
        k = names.index(ln["section"])
        if ln["start"] < spans[ln["section"]][0] and k > 0:
            cut = round(ln["start"] - 0.08, 3)
            spans[ln["section"]][0] = cut
            spans[names[k - 1]][1] = cut
    return {k: (round(a, 3), round(b, 3)) for k, (a, b) in spans.items()}


# ----------------------------------------------------------------------------- rhythm + chops
def _load_mono(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    import soundfile as sf
    import soxr

    y, fs = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return soxr.resample(y, fs, sr).astype(np.float32), sr


def _band_onsets(y: np.ndarray, sr: int, lo: float | None, hi: float | None, min_gap: float,
                 rel: float) -> list[float]:
    from scipy.signal import butter, find_peaks, sosfiltfilt

    if lo and hi:
        sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    elif hi:
        sos = butter(4, hi, btype="lowpass", fs=sr, output="sos")
    else:
        sos = butter(4, lo, btype="highpass", fs=sr, output="sos")
    x = sosfiltfilt(sos, y)
    hop = int(sr * 0.005)
    n = len(x) // hop
    env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    env_db = 20 * np.log10(env + 1e-9)
    flux = np.maximum(np.diff(env_db, prepend=env_db[0]), 0.0)
    flux = np.convolve(flux, np.ones(4), mode="same")  # 20 ms of rise
    loud = env_db > np.percentile(env_db, 60) - 6
    thr = rel * np.percentile(flux[loud], 99) if loud.any() else np.inf
    peaks, _ = find_peaks(flux * loud, height=thr, distance=max(1, int(min_gap / 0.005)))
    return [round(float(p * 0.005 - 0.01), 3) for p in peaks]


def rhythm() -> dict[str, Any]:
    """Kick and snare onsets in remix time.

    Detected on B's isolated drum stem (stems/b/drums.wav) and mapped through every drum clip of
    the arrangement on A1 — onsets on the summed instrumental also catch the 808/bass notes.
    Kick = band < 150 Hz, snare = 180-2500 Hz on beats 2, 3 or 4 (trap half-time snare on 3)."""
    y, sr = _load_mono(ROOT / "stems" / "b" / "drums.wav")
    k_src = _band_onsets(y, sr, None, 150.0, 0.18, 0.35)
    s_src = _band_onsets(y, sr, 180.0, 2500.0, 0.15, 0.35)
    kicks: list[float] = []
    snare_cand: list[float] = []
    for c in arr.ARRANGEMENT.clips:
        if c.track != arr.T1 or c.source != "b:drums" or any(x.kind == "hpf_sweep" for x in c.fx):
            continue
        s0, s1 = clip_range_src(c)
        kicks += [round(to_remix(c, t), 3) for t in k_src if s0 <= t < s1]
        snare_cand += [round(to_remix(c, t), 3) for t in s_src if s0 <= t < s1]
    # the breakdown build's synthetic kicks (A5 FX stem, 1/4 -> 1/8 -> 1/16)
    yfx, srf = _load_mono(SPUREN / "A5_fx.wav")
    fx_k = _band_onsets(yfx, srf, None, 120.0, 0.08, 0.25)
    for c in arr.ARRANGEMENT.clips:
        if c.source.startswith("fx:kick_build"):
            a, b = clip_range_remix(c)
            kicks += [t for t in fx_k if a - 0.02 <= t < b]
    kicks = sorted(set(kicks))
    snares = []
    for t in sorted(set(snare_cand)):  # snares sit on the backbeat of the 150 BPM grid
        beat = t / BEAT
        if abs(beat - round(beat)) * BEAT < 0.05 and (round(beat) % 2 == 1 or round(beat) % 4 == 2) \
                and not any(abs(t - k) < 0.03 for k in kicks):
            snares.append(t)
    beats = [round(i * BEAT, 3) for i in range(int(DURATION_S / BEAT) + 1)]
    bars = [round(i * BAR, 3) for i in range(arr.TOTAL_BARS + 1)]
    return {"kicks": kicks, "snares": snares, "beats": beats, "bars": bars}


def chops() -> list[dict]:
    """Onsets in the chop stem, labelled with the arrangement clip they belong to."""
    y, sr = _load_mono(SPUREN / "A4_chops.wav")
    onsets = _band_onsets(y, sr, 150.0, 6000.0, 0.05, 0.25)
    clips = [c for c in arr.ARRANGEMENT.clips if c.track == arr.T4]
    out = []
    for t in onsets:
        hit = None
        for c in clips:
            a, b = clip_range_remix(c)
            if a - 0.06 <= t <= b:
                hit = c
                break
        if hit is None:
            continue
        a, b = clip_range_remix(hit)
        if any(abs(e["clip_start"] - round(a, 3)) < 1e-6 for e in out):
            continue  # one event per chop clip (the tape-stop smears into several onsets)
        lab = hit.label.lower()
        kind = "dont_you" if "don't you" in lab else "feel"
        out.append({"t": round(t, 3), "end": round(b, 3), "kind": kind,
                    "text": "DON'T YOU" if kind == "dont_you" else "FEEL",
                    "label": hit.label, "clip_start": round(a, 3)})
    return out


# ----------------------------------------------------------------------------- build
def load_whisper(model: str = MODEL) -> dict[str, dict[str, list[dict]]]:
    """{"en": jobs, "de": jobs}; runs faster-whisper if a cache file is missing."""
    jobs_path = WORK / "whisper_jobs.json"
    if jobs_path.exists() and json.loads(jobs_path.read_text()) != json.loads(json.dumps(whisper_jobs())):
        raise RuntimeError("remix/arrangement.py changed since the whisper run (job list differs): "
                           "run `python -m lyrics.timing --whisper`")
    out = {}
    for lang, suffix in (("en", ""), ("de", "_de")):
        path = WORK / f"whisper_clips_{model}{suffix}.json"
        if not path.exists():
            run_whisper(model, lang)
        out[lang] = json.loads(path.read_text())["jobs"]
    return out


def build(model: str = MODEL) -> dict[str, Any]:
    secs = sections()
    wj = load_whisper(model)
    orig = {"a": json.loads((ROOT / "work" / "a_words.json").read_text()),
            "b": json.loads((ROOT / "work" / "b_words.json").read_text())}
    all_words: list[tuple[W, str]] = []
    changes: list[dict] = []
    timing_dev: list[float] = []
    for i, c in enumerate(vocal_clips()):
        r0, r1 = clip_range_remix(c)

        def stem(lang: str, i: int = i, r0: float = r0, r1: float = r1) -> list[W]:
            ws = [W(w["word"], w["start"], w["end"], w["p"], f"S_{lang}") for w in wj[lang].get(f"S{i}", [])]
            return [w for w in ws if r0 - 0.03 <= _mid(w) <= r1 + 0.1 and w.key]

        cands = {"S_en": stem("en"), "S_de": stem("de"),
                 "O_en": mapped_original(c, wj["en"].get(f"O{i}", []), "O_en"),
                 "O_de": mapped_original(c, wj["de"].get(f"O{i}", []), "O_de"),
                 "T": mapped_original(c, orig[c.source[0]], "T")}
        if is_hook_clip(c):
            merged = hook_words(c)
            heard = {k: " ".join(w.text for w in v) for k, v in cands.items() if v}
            changes.append({"clip": c.label, "op": "hook", "t": round(r0, 2), "from": heard,
                            "to": " ".join(w.text for w in merged)})
            for w in merged:  # onset check against what the stems heard ("don't"/"you")
                m = [x for x in cands["S_en"] + cands["S_de"] + cands["O_de"]
                     if x.key == w.key and abs(x.start - w.start) < 0.3]
                if m:
                    timing_dev.append(min(abs(x.start - w.start) for x in m))
        else:
            merged, log = consensus(cands)
            for e in log:
                e["clip"] = c.label
            changes.extend(log)
            merged = snap_to_stem(merged, cands["S_de"] + cands["S_en"], [])
            for x in cands["S_de"] + cands["S_en"]:  # remix stem vs source stem/transcript
                m = [y for y in cands["O_de"] + cands["O_en"] + cands["T"]
                     if y.key == x.key and abs(y.start - x.start) < 0.35]
                if m:
                    timing_dev.append(min(abs(y.start - x.start) for y in m))
        all_words.extend((w, singer_of(c)) for w in merged)

    all_words = _dedupe(all_words)
    lines = build_lines(all_words, secs)
    spans = comp_spans(lines, secs)
    rh = rhythm()
    data = {
        "meta": {"fps": FPS, "bpm": arr.BPM, "bar_s": BAR, "beat_s": BEAT, "duration_s": DURATION_S,
                 "whisper_model": model, "pause_s": PAUSE_S, "max_words": MAX_WORDS, "hold_s": HOLD_S,
                 "audio": "out/remix-hook.wav",
                 "timing_check": _stats(timing_dev)},
        "sections": [{"name": s.name, "start": round(s.start, 3), "end": round(s.end, 3),
                      "start_bar": s.start_bar, "bars": s.bars,
                      "comp_start": spans[s.name][0], "comp_end": spans[s.name][1]} for s in secs],
        "lines": lines,
        "chops": chops(),
        **rh,
        "changes": changes,
    }
    data["meta"]["words_total"] = sum(len(ln["words"]) for ln in lines)
    data["meta"]["lines_total"] = len(lines)
    return data


def _dedupe(words: list[tuple[W, str]]) -> list[tuple[W, str]]:
    """Drop a word that overlaps an earlier one by more than half (two clips' tails) and clamp
    small overlaps so word times are monotonic."""
    words = sorted(words, key=lambda x: x[0].start)  # stable: equal starts keep their order
    out: list[tuple[W, str]] = []
    for w, sg in words:
        if out:
            prev = out[-1][0]
            ov = prev.end - w.start
            if w.key == prev.key and (ov > 0.5 * (w.end - w.start) or abs(w.start - prev.start) < 0.1):
                continue
            if ov > 0:
                prev.end = max(prev.start + 0.02, w.start)
        out.append((w, sg))
    # Whisper sometimes gives a word zero length: give it up to 120 ms, never past the next word
    for i, (w, _sg) in enumerate(out):
        if w.end - w.start < 0.04:
            nxt = out[i + 1][0].start if i + 1 < len(out) else w.start + 0.12
            w.end = max(w.start + 0.02, min(w.start + 0.12, nxt))
    return out


def _stats(dev: list[float]) -> dict[str, float]:
    if not dev:
        return {"n": 0}
    d = np.array(dev)
    return {"n": int(d.size), "median_ms": round(float(np.median(d)) * 1000, 1),
            "p90_ms": round(float(np.percentile(d, 90)) * 1000, 1),
            "max_ms": round(float(d.max()) * 1000, 1)}


def save(data: dict[str, Any], path: Path = OUT_JSON) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return path


def load(path: Path = OUT_JSON) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", action="store_true", help="re-run faster-whisper")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    if args.whisper:
        run_whisper(args.model, "en")
        run_whisper(args.model, "de")
    data = build(args.model)
    p = save(data)
    print(f"{p}: {data['meta']['words_total']} words, {data['meta']['lines_total']} lines, "
          f"{len(data['chops'])} chops, {len(data['kicks'])} kicks, timing {data['meta']['timing_check']}")
    for ln in data["lines"]:
        print(f"{ln['start']:7.2f}-{ln['end']:7.2f} {ln['section']:13s} {ln['singer']:9s} {ln['text']}")
    for ch in data["changes"]:
        print("change", ch)


if __name__ == "__main__":
    main()
