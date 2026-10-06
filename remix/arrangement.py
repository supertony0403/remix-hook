"""The remix arrangement as data (bars, sources, effects), following docs/ARRANGEMENT.md.

Time bases
----------
* Remix: bar m starts at m * BAR seconds (bar 0 at t = 0). Tempo = B's measured 150.00 BPM.
* Source "b:<stem>": B bar n starts at B_DOWNBEAT0 + n * BAR (B is played at native tempo).
* Source "a:vocals": A vocals stretched 128 -> 150 BPM and shifted -2 semitones; A bar n starts
  at (A_DOWNBEAT0 + n * A_BAR) * STRETCH in the stretched file, i.e. one A bar == one remix bar.
* Source "fx:<name>": synthesised one-shot, bar 0 == start of the generated audio.

A clip places the source region [start, end) (absolute source bars) so that source bar `src`
lands on remix bar `at`. Cuts snap to zero crossings and get 6/8 ms fades. Instrumental splices
sit SPLICE bars (~13 ms) before the bar line so the downbeat transient is never faded.

Measured facts that shaped the plan (see docs/ARRANGEMENT.md, "Umsetzung"):
* B is 150.00 BPM (not 152; librosa's 152 is hop-lag quantisation), downbeat at 0.011 s.
* A is 128.00 BPM (not 129.2), downbeat at 0.061 s. Hook "don't you feel" = A bars 1.30-2.42,
  repeated every 2 bars (3x in A bars 0-6). The "but" is barely present in the stem.
* B's refrain is a dense rap: no gap >= 0.7 s inside refrain 3+4, so the full A phrase cannot sit
  "between" B lines. Refrain 3 stays original (one "feel" answer in its only gap); in refrain 4
  B's 2nd line is replaced by A "don't you feel" (true call-and-response), and A answers B's
  last line in the outro.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config

BPM = 150.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT  # 1.6 s

A_BPM = 128.0
A_BAR = 4 * 60.0 / A_BPM  # 1.875 s
A_DOWNBEAT0 = 0.061
B_DOWNBEAT0 = 0.011
STRETCH = A_BPM / BPM  # output/input duration ratio for A (0.85333)
SEMITONES_A = config.SEMITONES_A

SPLICE = 0.008  # bars (12.8 ms) before the bar line for instrumental splices

TOTAL_BARS = 100
TAIL_BARS = 2.5  # reverb/delay tail after the final downbeat


@dataclass(frozen=True)
class Fx:
    kind: str
    params: tuple[tuple[str, object], ...] = ()

    def get(self, key: str, default: object = None) -> object:
        return dict(self.params).get(key, default)


def fx(kind: str, **params: object) -> Fx:
    return Fx(kind, tuple(sorted(params.items())))


@dataclass(frozen=True)
class Clip:
    track: str
    source: str
    at: float
    src: float
    start: float
    end: float
    gain_db: float = 0.0
    fx: tuple[Fx, ...] = ()
    label: str = ""

    @property
    def remix_start(self) -> float:
        return self.at + (self.start - self.src)

    @property
    def remix_end(self) -> float:
        return self.at + (self.end - self.src)


@dataclass(frozen=True)
class Automation:
    tracks: tuple[str, ...]
    kind: str  # lpf | sidechain | tape_stop | mute
    start: float  # remix bars
    end: float
    params: tuple[tuple[str, object], ...] = ()

    def get(self, key: str, default: object = None) -> object:
        return dict(self.params).get(key, default)


def auto(tracks: str | tuple[str, ...], kind: str, start: float, end: float, **params: object) -> Automation:
    tr = (tracks,) if isinstance(tracks, str) else tracks
    return Automation(tr, kind, start, end, tuple(sorted(params.items())))


@dataclass(frozen=True)
class Section:
    name: str
    start: int
    bars: int
    note: str

    @property
    def end(self) -> int:
        return self.start + self.bars


@dataclass
class Arrangement:
    sections: list[Section]
    clips: list[Clip] = field(default_factory=list)
    automation: list[Automation] = field(default_factory=list)
    total_bars: int = TOTAL_BARS
    tail_bars: float = TAIL_BARS


T1, T2, T3, T4, T5, T6 = config.TRACK_NAMES
ALL = config.TRACK_NAMES

# --------------------------------------------------------------------------- A vocal landmarks
# absolute A bars (A grid: downbeat 0.061 s, 1.875 s/bar), measured on the A vocal stem
HOOK = (1.30, 2.42)  # "don't you feel" (phrase 1; +2 and +4 bars for phrases 2/3)
DONT = 1.349  # syllable onsets inside phrase 1
YOU = 1.477
FEEL = 1.706
DONT_YOU = (1.32, 1.70)
REFRAIN_A = (23.35, 31.30)  # "And it's time to close what we had ... without you here"
HOOK_LEVEL_REF = (1.30, 6.42)  # loudness reference regions for auto level
REFRAIN_LEVEL_REF = (24.5, 31.3)

# reverb throw = send of the phrase's last word (bars relative to the phrase's source bars)
THROW_FEEL = (2.05, 2.42)

# send levels (dB). v1 used -16 dB into a ~2.9 s room; v2 is drier and closer:
# a quiet 1/16 slap carries the space, the ~1.0 s room sits ~10 dB lower than in v1.
SLAP_DB = -17.0
ROOM_DB = -26.0
B_SLAP_DB = -19.0
B_ROOM_DB = -27.0
THROW_DB = -8.0

SECTIONS = [
    Section("Intro", 0, 16, "B intro (bass+pads) through LPF 250 Hz -> open; A hook x3 dry with "
            "reverb throws; riser + snare roll; bar 15 stutter + tape stop + 1 beat silence"),
    Section("Drop 1", 16, 8, "B refrain-3 instrumental, sub drop + impact, pitched 'feel' chops "
            "on the offbeats, sidechain pump"),
    Section("B Refrain 1", 24, 8, "B bars 16-24 original"),
    Section("B Strophe 1", 32, 16, "B bars 24-32 + 40-48 (verse shortened to 16 bars)"),
    Section("B Refrain 2", 48, 8, "B bars 48-56 original"),
    Section("Post-Chorus", 56, 4, "A 'don't you feel' x2 over B drums + bass"),
    Section("Breakdown", 60, 16, "B bass+pads filtered, A refrain; from bar 68 build (B drums "
            "HPF, kick 1/4->1/8->1/16, riser); bar 75 A 'don't you...' + silence"),
    Section("Finaler Drop", 76, 16, "B refrain 3+4 full; refrain 4 = call-and-response B/A; "
            "extra sub, sidechain"),
    Section("Outro", 92, 8, "B outro into LPF, A 'don't you feel' with delay feedback and "
            "reverb tail, final downbeat at bar 100"),
]


def _b(track: str, stems: tuple[str, ...], at: float, src: float, start: float, end: float,
       gain_db: float = 0.0, fxs: tuple[Fx, ...] = (), label: str = "") -> list[Clip]:
    return [Clip(track, f"b:{s}", at, src, start, end, gain_db, fxs, label or f"B {s}")
            for s in stems]


def _bvox(at: float, src: float, start: float, end: float, offset_db: float, label: str,
          style: str = "refrain", throw_end: float | None = None) -> Clip:
    """B vocals: autotune (`style` rap = hard 20 ms, refrain = subtle 60 ms), a gentle section
    ride (B's own mix varies ~6 dB between refrains), slap + small room like the A hook, and an
    optional reverb throw on the section's last phrase ending at source bar `throw_end`."""
    fxs = [fx("tune", style=style), fx("ride", offset_db=offset_db),
           fx("send", bus="slap", db=B_SLAP_DB), fx("send", bus="reverb", db=B_ROOM_DB)]
    if throw_end is not None:
        fxs.append(fx("throw", bus="reverb", start=throw_end - 0.30, end=throw_end, db=THROW_DB))
    return Clip(T2, "b:vocals", at, src, start, end, 0.0, tuple(fxs), label)


def _hook(track: str, at: float, src: float, region: tuple[float, float], gain_db: float = 0.0,
          fxs: tuple[Fx, ...] = (), label: str = "") -> Clip:
    return Clip(track, "a:vocals", at, src, region[0], region[1], gain_db, fxs, label)


def _phrase(at: float, k: int = 0, throw: str | None = "reverb", label: str = "",
            gain_db: float = 0.0, extra: tuple[Fx, ...] = ()) -> Clip:
    """A hook phrase k (0..2) with source bar 1+2k aligned to remix bar `at`."""
    off = 2 * k
    fxs: list[Fx] = [fx("level", ref="hook"), fx("send", bus="slap", db=SLAP_DB),
                     fx("send", bus="reverb", db=ROOM_DB)]
    if throw:
        fxs.append(fx("throw", bus=throw, start=THROW_FEEL[0] + off, end=THROW_FEEL[1] + off,
                      db=THROW_DB))
    fxs.extend(extra)
    return _hook(T3, at, 1.0 + off, (HOOK[0] + off, HOOK[1] + off), gain_db, tuple(fxs),
                 label or f"A hook {k + 1}")


def _chop(at: float, onset: float, length: float, semitones: float = 0.0, gain_db: float = 0.0,
          label: str = "chop", send_db: float = -24.0) -> Clip:
    fxs = [fx("level", ref="hook"), fx("hpf", hz=180.0), fx("send", bus="reverb", db=send_db)]
    if semitones:
        fxs.insert(0, fx("pitch", semitones=semitones))
    return Clip(T4, "a:vocals", at, onset, onset - 0.004, onset + length, gain_db, tuple(fxs), label)


def build() -> Arrangement:
    s = SPLICE
    clips: list[Clip] = []
    autos: list[Automation] = []

    # ------------------------------------------------------------------ 1 Intro (0-16)
    clips += _b(T1, ("bass", "other"), 0, 0, 0, 16 - s, label="B intro")
    for k in range(3):  # "don't you feel" lands on remix bars 4.3, 6.3, 8.3
        clips.append(_phrase(4 + 2 * k, k, throw="reverb" if k == 2 else None))
    autos.append(auto(T1, "lpf", 0, 15.5, f0=250.0, f1=18000.0, curve=2.2))
    clips.append(Clip(T5, "fx:riser", 12, 0, 0, 3.75, -3.0, (), "riser"))
    clips.append(Clip(T5, "fx:snare_roll4", 12, 0, 0, 3.5, -4.0, (), "snare roll"))
    # bar 15 stutter: don't-you at 1/8, then 1/16, then tape stop on beat 3, silence on beat 4
    for i, (pos, ln) in enumerate([(15.0, 0.125), (15.125, 0.125), (15.25, 0.0625),
                                   (15.3125, 0.0625), (15.375, 0.0625), (15.4375, 0.0625)]):
        onset = DONT if i % 2 == 0 else YOU
        clips.append(_chop(pos, onset, ln - 0.008, gain_db=-1.0, label="stutter don't you"))
    clips.append(Clip(T4, "a:vocals", 15.5, DONT, DONT_YOU[0], DONT_YOU[1], -1.0,
                      (fx("level", ref="hook"),), "tape-stop don't you"))
    autos.append(auto((T1, T4), "tape_stop", 15.5, 15.75, power=1.4))
    autos.append(auto(ALL, "mute", 15.75, 16.0))

    # ------------------------------------------------------------------ 2 Drop 1 (16-24)
    clips += _b(T1, ("drums", "bass", "other"), 16, 72, 72 - s, 80 - s, label="B refrain3 instr")
    clips.append(Clip(T5, "fx:sub_drop", 16, 0, 0, 1.6, -3.0, (), "sub drop"))
    clips.append(Clip(T5, "fx:impact", 16, 0, 0, 2.0, -6.0, (), "impact"))
    clips.append(Clip(T5, "fx:sub_drop", 20, 0, 0, 1.6, -6.0, (), "sub drop"))
    pattern = [0, 0, 3, 0, 5, 3, 0, -2] * 3 + [0, 3, 5, 8, 5, 3, 0]
    for i, semi in enumerate(pattern):  # offbeats (the "and" of every beat), last one omitted
        pos = 16 + i * 0.25 + 0.125
        clips.append(_chop(pos, FEEL, 0.115, semitones=semi, gain_db=-2.0, label=f"feel {semi:+d}"))
    autos.append(auto(T1, "sidechain", 16, 24, depth_db=-6.0))

    # ------------------------------------------------------------------ 3/4 Refrain 1 + Strophe 1
    clips += _b(T1, ("drums", "bass", "other"), 24, 16, 16 - s, 32 - s, label="B 16-32")
    clips.append(_bvox(24, 16, 15.94, 23.60, 0.0, "B refrain 1"))
    clips.append(_bvox(24, 16, 23.60, 31.75, -1.5, "B verse 1a", style="rap", throw_end=31.62))
    clips.append(Clip(T4, "a:vocals", 31.5, FEEL, FEEL - 0.004, FEEL + 0.30, -3.0,
                     (fx("level", ref="hook"), fx("hpf", hz=180.0),
                      fx("send", bus="slap", db=SLAP_DB),
                      fx("throw", bus="delay", start=FEEL + 0.05, end=FEEL + 0.30, db=-10.0)),
                     "feel answer"))
    clips += _b(T1, ("drums", "bass"), 40, 40, 40 - s, 60 - s, label="B 40-60")
    clips += _b(T1, ("other",), 40, 40, 40 - s, 56 - s, label="B 40-56")
    clips.append(_bvox(40, 40, 39.80, 47.80, -1.5, "B verse 1b", style="rap"))
    clips.append(_bvox(40, 40, 47.80, 55.60, 0.0, "B refrain 2", throw_end=55.50))

    # ------------------------------------------------------------------ 5 Post-Chorus (56-60)
    clips.append(_phrase(56, 0, throw=None))
    clips.append(_phrase(58, 1, throw="delay"))

    # ------------------------------------------------------------------ 6 Breakdown (60-76)
    clips += _b(T1, ("bass", "other"), 60, 56, 56 - s, 71, label="B 56-71 bass+pads")
    autos.append(auto(T1, "lpf", 60, 75, f0=650.0, f1=4000.0, curve=2.5))
    clips.append(Clip(T3, "a:vocals", 61, 24, REFRAIN_A[0], REFRAIN_A[1], 0.0,
                      (fx("level", ref="refrain"), fx("hpf", hz=160.0),
                       fx("send", bus="slap", db=SLAP_DB), fx("send", bus="reverb", db=ROOM_DB + 2),
                       fx("throw", bus="reverb", start=30.9, end=31.3, db=THROW_DB)),
                      "A refrain"))
    clips += _b(T1, ("drums",), 68, 64, 64, 71, gain_db=-1.0,
                fxs=(fx("hpf_sweep", f0=2500.0, f1=120.0, curve=1.0),), label="B drums build")
    clips.append(Clip(T5, "fx:kick_build7", 68, 0, 0, 7.0, -4.0, (), "kick build"))
    clips.append(Clip(T5, "fx:riser", 71.25, 0, 0, 3.75, -3.0, (), "riser"))
    clips.append(Clip(T5, "fx:snare_roll2", 73, 0, 0, 2.0, -7.0, (), "snare roll"))
    clips.append(Clip(T3, "a:vocals", 75, 1.0, DONT_YOU[0], DONT_YOU[1], 0.0,
                      (fx("level", ref="hook"), fx("send", bus="slap", db=SLAP_DB),
                       fx("throw", bus="reverb", start=YOU, end=DONT_YOU[1], db=THROW_DB - 2)),
                      "A don't you..."))
    autos.append(auto((T1, T4, T5), "mute", 75.0, 76.0))

    # ------------------------------------------------------------------ 7 Finaler Drop (76-92)
    clips += _b(T1, ("drums", "bass", "other"), 76, 72, 72 - s, 88 - s, label="B refrain3+4 instr")
    clips.append(_bvox(76, 72, 71.88, 81.86, 0.0, "B refrain 3 + refrain 4 line 1"))
    clips.append(_bvox(76, 72, 83.47, 87.60, 0.0, "B refrain 4 lines 3+4", throw_end=87.47))
    clips.append(Clip(T5, "fx:sub_drop", 76, 0, 0, 1.6, -2.5, (), "sub drop"))
    clips.append(Clip(T5, "fx:impact", 76, 0, 0, 2.0, -5.0, (), "impact"))
    clips.append(Clip(T5, "fx:sub_drop", 84, 0, 0, 1.6, -4.5, (), "sub drop"))
    clips.append(Clip(T4, "a:vocals", 83.5, FEEL, FEEL - 0.004, FEEL + 0.40, -2.0,
                      (fx("level", ref="hook"), fx("hpf", hz=180.0),
                       fx("throw", bus="delay", start=FEEL + 0.05, end=FEEL + 0.40, db=-10.0)),
                      "feel answer"))
    clips.append(_phrase(86, 0, throw=None, label="A answers (replaces B line 2)"))
    autos.append(auto(T1, "sidechain", 76, 92, depth_db=-5.0))

    # ------------------------------------------------------------------ 8 Outro (92-100)
    clips += _b(T1, ("bass", "other"), 92, 88, 88 - s, 96, label="B outro")
    autos.append(auto(T1, "lpf", 92, 100, f0=18000.0, f1=280.0, curve=1.0))
    clips.append(_phrase(92, 0, throw="delay", label="A answers B's last line"))
    clips.append(_phrase(96, 1, throw="delay_long", label="A final + delay feedback",
                         extra=(fx("send", bus="reverb", db=ROOM_DB + 8),)))
    clips.append(Clip(T5, "fx:sub_drop", 100, 0, 0, 1.6, -5.0, (), "final sub"))
    clips.append(Clip(T5, "fx:impact", 100, 0, 0, 2.5, -9.0, (), "final impact"))

    return Arrangement(SECTIONS, clips, autos)


ARRANGEMENT = build()
