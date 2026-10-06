"""One Fusion comp per section of the remix (V2 "Lyrics").

Every line becomes a *layer* (its words as Text+ tools, animated per word) that is switched on
only while it is visible: layers are chained through Dissolve tools keyed 0/1 with step keys,
so Fusion renders just the lines on screen (long sections stay fast). Each style first makes a
*schedule* (line -> on/off seconds, off >= last word + HOLD), then renders it; a line whose hold
reaches past its comp's end is rendered again, in its own style, by the next comp ("tail carry"),
so the readability rule holds across comp boundaries.

Styles (brief, section by section): intro (title from blur to sharp with the filter, single giant
hook words with glow, "DON'T YOU" grid on the stutter, everything falls on the tape stop, one beat
of black), drop (word slams 1.6 -> 1.0 in 4 frames with RGB split, camera shake, glitch slices,
flickering FEEL chops; the final drop as call-and-response B left / A right), refrain (centred line,
words pop with overshoot, previous line fades upwards), rap (word by word, jumping positions,
+-4 degrees, mask wipe, key words acid yellow and bigger), post-chorus (giant BUT DON'T YOU FEEL with
outline echoes zooming out), breakdown (big outline type, slow drift, soft fades) and outro (delay
echoes, end card to the end).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from . import comp_writer as cw
from . import timing
from .style import (ACID, CYAN, HOOK, PALE_CYAN, PALE_PINK, PINK, RAP, REFRAIN, WHITE, Fmt,
                    metrics, size_for_cap)

FPS = 60
HOLD = timing.HOLD_S
SLAM_F = 4          # 1.6 -> 1.0 in 4 frames
POP_F = 9           # pop with overshoot
FADE_F = 10


def slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


# ----------------------------------------------------------------------------- context
@dataclass
class Ctx:
    c: cw.Comp
    f: Fmt
    data: dict[str, Any]
    f0: int                      # absolute frame of comp frame 0
    n: int                       # comp length in frames
    rng: random.Random = field(default_factory=lambda: random.Random(11))

    def fr(self, t: float) -> int:
        return int(round(t * FPS)) - self.f0

    def name(self, base: str) -> str:
        return self.c._unique(base)

    def px(self, x: float, y: float) -> tuple[float, float]:
        return self.c.px(x, y)

    def u(self, v: float) -> float:
        return self.f.u(v)


@dataclass
class Item:
    line: dict[str, Any]
    on: float
    off: float
    slot: Any = None           # style-specific placement


@dataclass
class Layer:
    tool: str
    on: int                    # comp frames
    off: int
    lines: tuple[str, ...] = ()


# ----------------------------------------------------------------------------- generic helpers
def upper(text: str) -> str:
    return text.upper()


def layout_row(words: list[str], font: tuple[str, str], cap: float, width: int, max_w: float,
               min_cap: float, gap_em: float = 0.30, scales: list[float] | None = None
               ) -> tuple[float, list[list[tuple[int, float]]]]:
    """Fit words into one row (shrinking down to `min_cap`), else two rows. `scales` are per-word
    size factors (key words are bigger). Returns the cap height and rows of
    (word index, x offset of the word centre from the row centre)."""
    m = metrics(*font)
    sc = scales or [1.0] * len(words)

    def widths(c: float) -> list[float]:
        return [m.width_px(w, size_for_cap(font, c * k, width), width) for w, k in zip(words, sc)]

    def row(idx: list[int], ws: list[float], c: float) -> tuple[float, list[tuple[int, float]]]:
        gap = gap_em * c  # ~0.3 cap heights between words
        total = sum(ws[i] for i in idx) + gap * (len(idx) - 1)
        x = -total / 2
        out = []
        for i in idx:
            out.append((i, x + ws[i] / 2))
            x += ws[i] + gap
        return total, out

    c = cap
    while True:
        ws = widths(c)
        total, r = row(list(range(len(words))), ws, c)
        if total <= max_w:
            return c, [r]
        if c * 0.92 < min_cap:
            break
        c *= 0.92
    # two rows, split where the rows are most balanced
    c = cap
    while True:
        ws = widths(c)
        best = None
        for k in range(1, len(words)):
            t1, r1 = row(list(range(k)), ws, c)
            t2, r2 = row(list(range(k, len(words))), ws, c)
            score = max(t1, t2)
            if best is None or score < best[0]:
                best = (score, [r1, r2])
        if best[0] <= max_w or c * 0.92 < min_cap * 0.6:
            return c, best[1]
        c *= 0.92


def word_text(ctx: Ctx, base: str, text: str, font: tuple[str, str], cap: float, color: str,
              center_px: tuple[float, float], outline: float = 0.0, fill: float = 1.0) -> str:
    size = size_for_cap(font, cap, ctx.f.width)
    name = ctx.name(base)
    ctx.c.text(name, text, font=font[0], style=font[1], size=size, color=color,
               center=ctx.px(*center_px), justify="center", anchor_v="center")
    if outline:
        r, g, b = cw.hex_rgb(color)
        ctx.c.set_input(name, "Opacity1", fill)
        ctx.c.set_input(name, "Enabled2", 1)
        ctx.c.set_input(name, "Thickness2", outline)
        for k, v in (("Red2", r), ("Green2", g), ("Blue2", b), ("Opacity2", 1.0)):
            ctx.c.set_input(name, k, v)
    return name


def glow(ctx: Ctx, tool: str, size: float, amount: float = 0.9) -> str:
    bl = ctx.c.blur(ctx.name(f"{tool}Glow"), tool, size)
    return ctx.c.merge(ctx.name(f"{tool}G"), tool, bl, apply_mode="Screen", blend=amount)


def switch_chain(ctx: Ctx, layers: list[Layer], base: str | None = None, tag: str = "L") -> str:
    """Stack layers; each one is only evaluated between its on and off frame."""
    acc = base or ctx.c.background(ctx.name(f"{tag}Clear"), (0, 0, 0), alpha=0.0)
    for lay in sorted(layers, key=lambda l: l.on):
        on, off = max(0, lay.on), min(ctx.n, lay.off)
        if off <= 0 or on >= ctx.n or off <= on:
            continue
        m = ctx.c.merge(ctx.name(f"{tag}M"), acc, lay.tool)
        d = ctx.c.dissolve(ctx.name(f"{tag}S"), acc, m)
        keys: dict[int, float] = {}
        keys[0] = 1.0 if on == 0 else 0.0
        if on > 0:
            keys[on] = 1.0
        if off < ctx.n:
            keys[off] = 0.0
        if len(keys) >= 2:
            ctx.c.keyframes(d, "Mix", keys, ease="step")
        else:
            ctx.c.set_input(d, "Mix", keys[0])
        acc = d
    return acc


def words_layer(ctx: Ctx, item: Item, font: tuple[str, str], cap: float, center: tuple[float, float],
                colors: Callable[[dict], str], max_w: float, *, min_cap: float | None = None,
                anim: str = "pop", key_scale: float = 1.15, outline: float = 0.0, fill: float = 1.0,
                rgb: bool = False, glow_size: float = 0.0, angle: float = 0.0,
                text_fn: Callable[[str], str] = upper) -> str:
    """One line: its words laid out around `center` (px), each appearing at its onset."""
    ln = item.line
    texts = [text_fn(w["w"]) for w in ln["words"]]
    scales = [key_scale if w.get("key") else 1.0 for w in ln["words"]]
    cap_fit, rows = layout_row(texts, font, cap, ctx.f.width, max_w, min_cap or cap * 0.55, scales=scales)
    row_gap = cap_fit * 1.45
    acc = ctx.c.background(ctx.name(f"{ln['id']}Clear"), (0, 0, 0), alpha=0.0)
    y0 = center[1] - row_gap * (len(rows) - 1) / 2
    for ri, row in enumerate(rows):
        for wi, dx in row:
            w = ln["words"][wi]
            is_key = bool(w.get("key"))
            wc = cap_fit * (key_scale if is_key else 1.0)
            pos = (center[0] + dx, y0 + ri * row_gap)
            color = colors(w)
            base = f"{ln['id']}W{wi}"
            tool = word_text(ctx, base, texts[wi], font, wc, color, pos, outline, fill)
            if rgb:
                tool = rgb_split(ctx, tool, texts[wi], font, wc, pos, ctx.fr(w["start"]), outline, fill, color)
            if glow_size:
                tool = glow(ctx, tool, glow_size)
            t_on = ctx.fr(w["start"])
            piv = ctx.px(*pos)
            if anim == "slam":
                tr = ctx.c.transform(ctx.name(f"{base}T"), tool, pivot=piv, quality=4)
                ctx.c.keyframes(tr, "Size", {t_on: 1.6, t_on + SLAM_F: 1.0}, ease="out_cubic")
            elif anim == "pop":
                tr = ctx.c.transform(ctx.name(f"{base}T"), tool, pivot=piv, motion_blur=False)
                ctx.c.keyframes(tr, "Size", {t_on: 0.35, t_on + POP_F: 1.0}, ease="out_back")
            elif anim == "fade":
                tr = tool
            else:  # "cut": appears instantly with a 3-frame settle
                tr = ctx.c.transform(ctx.name(f"{base}T"), tool, pivot=piv, motion_blur=False)
                ctx.c.keyframes(tr, "Size", {t_on: 1.12, t_on + 3: 1.0}, ease="out_cubic")
            m = ctx.c.merge(ctx.name(f"{base}M"), acc, tr)
            if anim == "fade":
                if t_on > 0:
                    ctx.c.keyframes(m, "Blend", {t_on - 1: 0.0, t_on + 8: 1.0}, ease="out_cubic")
            elif t_on > 0:
                ctx.c.keyframes(m, "Blend", {0: 0.0, t_on: 1.0}, ease="step")
            acc = m
    if angle:
        acc = ctx.c.transform(ctx.name(f"{ln['id']}Rot"), acc, angle=angle, pivot=ctx.px(*center),
                              motion_blur=False)
    return acc


def rgb_split(ctx: Ctx, tool: str, text: str, font: tuple[str, str], cap: float, pos: tuple[float, float],
              t_on: int | list[int], outline: float, fill: float, main_color: str = WHITE) -> str:
    """Two colour copies, offset sideways and screened together under the word; the split is wide
    on every slam in `t_on` and settles to a few pixels."""
    onsets = [t_on] if isinstance(t_on, int) else list(t_on)
    copies = []
    fringe = (CYAN, ACID) if main_color == PINK else (PINK, CYAN)
    for col, sign in zip(fringe, (-1, 1)):
        cp = word_text(ctx, f"{tool}{'P' if sign < 0 else 'C'}", text, font, cap, col, pos, outline, fill)
        tr = ctx.c.transform(ctx.name(f"{cp}T"), cp, motion_blur=False)
        d0 = ctx.u(26) / ctx.f.width
        d1 = ctx.u(5) / ctx.f.width
        keys: dict[int, tuple[float, float]] = {}
        for k in onsets:
            keys[k] = (0.5 + sign * d0, 0.5)
            keys[k + 6] = (0.5 + sign * d1, 0.5)
            keys[k - 1] = keys.get(k - 1, (0.5 + sign * d1, 0.5))
        keys = {k: v for k, v in keys.items() if k >= 0} or {onsets[0]: (0.5 + sign * d0, 0.5)}
        if len(keys) >= 2:
            ctx.c.keyframes(tr, "Center", dict(sorted(keys.items())), ease="out_cubic")
        else:
            ctx.c.set_input(tr, "Center", next(iter(keys.values())))
        copies.append(tr)
    split = ctx.c.merge(ctx.name(f"{tool}RGB"), copies[0], copies[1], apply_mode="Screen")
    return ctx.c.merge(ctx.name(f"{tool}RGBM"), split, tool)  # the word itself stays on top


def line_fade(ctx: Ctx, tool: str, on: int, off: int, fade_in: int = 0, fade_out: int = FADE_F,
              low: float = 0.0, dim: tuple[int, float] | None = None) -> tuple[str, int]:
    """Fade a line in a Dissolve against clear. Fully visible from `on` (after `fade_in`) until
    `off` (which is >= last word + HOLD); the fade-out starts at `off`. `dim` = (frame, level)
    dims the line earlier (refrain: the previous line). Returns the tool and the frame after the
    fade (the layer's switch-off frame)."""
    clear = ctx.c.background(ctx.name(f"{tool}FC"), (0, 0, 0), alpha=0.0)
    d = ctx.c.dissolve(ctx.name(f"{tool}F"), clear, tool, mix=1.0)
    keys: dict[int, float] = {}
    if fade_in:
        keys[on] = 0.0
        keys[on + fade_in] = 1.0
    else:
        keys[on] = 1.0
    level = 1.0
    if dim is not None and dim[0] < off:
        d0 = max(dim[0], max(keys) + 1)
        keys[d0] = 1.0
        keys[min(d0 + 12, off)] = dim[1]
        level = dim[1]
    o = max(off, max(keys) + 1)
    keys[o] = level
    keys[o + fade_out] = low
    ctx.c.keyframes(d, "Mix", dict(sorted(keys.items())), ease="in_out_cubic")
    return d, o + fade_out + 1


def hold_off(line: dict[str, Any], *cands: float) -> float:
    return max([line["end"] + HOLD, *cands])


# ----------------------------------------------------------------------------- schedules
def sched_sequential(lines: list[dict], min_show: float = 0.0, until_next: bool = True,
                     extra: float = 0.0) -> list[Item]:
    """on = first word, off = max(end + HOLD, next line start) (+ extra)."""
    out = []
    for i, ln in enumerate(lines):
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else ln["end"] + 0.6
        off = hold_off(ln, nxt if until_next else 0.0, ln["start"] + min_show) + extra
        out.append(Item(ln, ln["start"], off))
    return out


# ----------------------------------------------------------------------------- styles
def style_refrain(ctx: Ctx, items: list[Item], *, singer_color=(PALE_PINK, PINK)) -> list[Layer]:
    """Centred line, words pop in (overshoot); when the next line starts this one moves up and
    dims, and leaves when the one after arrives."""
    f = ctx.f
    layers = []
    cap = ctx.u(118)
    for k, it in enumerate(items):
        ln = it.line
        on, off = ctx.fr(it.on), ctx.fr(it.off)
        tool = words_layer(ctx, it, REFRAIN, cap, (f.cx, f.cy), lambda w: ACID if w.get("key") else singer_color[0],
                           f.safe_w * 0.94, anim="pop")
        nxt = items[k + 1].on if k + 1 < len(items) else None
        mv = ctx.c.transform(ctx.name(f"{ln['id']}Up"), tool, motion_blur=False)
        if nxt is not None and ctx.fr(nxt) < off + 1:
            fn = ctx.fr(nxt)
            ctx.c.keyframes(mv, "Center", {fn: (0.5, 0.5), fn + 12: (0.5, 0.5 + ctx.u(150) / f.height)},
                            ease="out_cubic")
            ctx.c.keyframes(mv, "Size", {fn: 1.0, fn + 12: 0.72}, ease="out_cubic")
        dim = None
        if nxt is not None and ctx.fr(nxt) < off:
            dim = (max(ctx.fr(nxt), ctx.fr(ln["end"] + HOLD)), 0.38)
        fd, end_f = line_fade(ctx, mv, on, off, fade_out=FADE_F, dim=dim)
        layers.append(Layer(fd, on, end_f, (ln["id"],)))
    return layers


def sched_refrain(lines: list[dict]) -> list[Item]:
    out = []
    for i, ln in enumerate(lines):
        nn = lines[i + 2]["start"] if i + 2 < len(lines) else None
        n1 = lines[i + 1]["start"] if i + 1 < len(lines) else None
        off = hold_off(ln, (nn if nn is not None else (n1 + 1.0 if n1 is not None else ln["end"] + 0.8)))
        out.append(Item(ln, ln["start"], off))
    return out


RAP_SLOTS_16x9 = [(0.30, 0.36, -3.5), (0.70, 0.62, 3.0), (0.50, 0.22, -2.0), (0.32, 0.72, 4.0),
                  (0.68, 0.30, -4.0), (0.50, 0.55, 2.5)]
RAP_SLOTS_9x16 = [(0.5, 0.30, -3.5), (0.5, 0.62, 3.0), (0.5, 0.44, -2.0), (0.5, 0.74, 4.0),
                  (0.5, 0.22, -4.0), (0.5, 0.52, 2.5)]


def style_rap(ctx: Ctx, items: list[Item]) -> list[Layer]:
    f = ctx.f
    slots = RAP_SLOTS_9x16 if f.vertical else RAP_SLOTS_16x9
    left, top, right, bottom = f.safe
    layers = []
    for k, it in enumerate(items):
        ln = it.line
        on, off = ctx.fr(it.on), ctx.fr(it.off)
        sx, sy, ang = slots[(it.slot if it.slot is not None else k) % len(slots)]
        cx = left + sx * (right - left)
        cy = top + sy * (bottom - top)
        max_w = (right - left) * (0.62 if not f.vertical else 0.96)
        tool = words_layer(ctx, it, RAP, ctx.u(112), (cx, cy),
                           lambda w: ACID if w.get("key") else WHITE, max_w, anim="cut",
                           key_scale=1.35, angle=ang, text_fn=lambda s: s)
        # mask wipe: a rectangle sweeps open across the line in 6 frames
        mask = ctx.c.rect_mask(ctx.name(f"{ln['id']}Wipe"), center=(0.5, 0.5), width=1.0, height=1.0,
                               soft_edge=0.02)
        ctx.c.keyframes(mask, "Center", {on: (-0.5, 0.5), on + 7: (0.5, 0.5)}, ease="out_cubic")
        clear = ctx.c.background(ctx.name(f"{ln['id']}WC"), (0, 0, 0), alpha=0.0)
        wiped = ctx.c.merge(ctx.name(f"{ln['id']}Wiped"), clear, tool)
        ctx.c.apply_mask(wiped, mask)
        fd, end_f = line_fade(ctx, wiped, on, off, fade_out=6)
        layers.append(Layer(fd, on, end_f, (ln["id"],)))
    return layers


def sched_rap(lines: list[dict]) -> list[Item]:
    return sched_sequential(lines, min_show=0.0, until_next=True, extra=0.05)


def style_hook_words(ctx: Ctx, items: list[Item], cap: float, color: str = PALE_CYAN,
                     glow_size: float = 18.0, y: float | None = None) -> list[Layer]:
    """Each hook word alone, giant, with glow; the last word holds to line end + HOLD."""
    f = ctx.f
    layers = []
    for it in items:
        ln = it.line
        ws = ln["words"]
        for i, w in enumerate(ws):
            on = ctx.fr(w["start"])
            off = ctx.fr(ws[i + 1]["start"]) if i + 1 < len(ws) else ctx.fr(it.off)
            single = Item({**ln, "id": f"{ln['id']}h{i}", "words": [w]}, w["start"], it.off)
            tool = words_layer(ctx, single, HOOK, cap, (f.cx, y if y is not None else f.cy), lambda _w: color,
                               f.safe_w, anim="slam", glow_size=glow_size, key_scale=1.0)
            if i + 1 == len(ws):
                tool, off = line_fade(ctx, tool, on, off, fade_out=FADE_F)
            layers.append(Layer(tool, on, off, (ln["id"],)))
    return layers


def style_breakdown(ctx: Ctx, items: list[Item]) -> list[Layer]:
    f = ctx.f
    layers = []
    for k, it in enumerate(items):
        ln = it.line
        on, off = ctx.fr(it.on), ctx.fr(it.off)
        if ln["singer"] == "A-Hook":
            layers += style_hook_words(ctx, [it], ctx.u(230), color=PALE_CYAN, glow_size=22)
            continue
        par = (it.slot if it.slot is not None else k) % 2
        cy = f.cy + (-1 if par == 0 else 1) * ctx.u(60)
        tool = words_layer(ctx, it, REFRAIN, ctx.u(150), (f.cx, cy), lambda w: CYAN if not w.get("key") else ACID,
                           f.safe_w * 0.95, anim="fade", outline=0.012, fill=0.12)
        drift = ctx.c.transform(ctx.name(f"{ln['id']}Drift"), tool, motion_blur=False)
        dx = ctx.u(70) / f.width * (1 if par == 0 else -1)
        ctx.c.keyframes(drift, "Center", {on: (0.5 - dx / 2, 0.5), off: (0.5 + dx / 2, 0.5 + ctx.u(20) / f.height)},
                        ease="linear")
        fd, end_f = line_fade(ctx, drift, on, off, fade_in=14, fade_out=24)
        layers.append(Layer(fd, on, end_f, (ln["id"],)))
    return layers


def sched_breakdown(lines: list[dict]) -> list[Item]:
    out = []
    for i, ln in enumerate(lines):
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else ln["end"] + 1.2
        off = hold_off(ln, min(nxt + 0.3, ln["end"] + 1.4))
        out.append(Item(ln, ln["start"], off))
    return out


def style_postchorus(ctx: Ctx, items: list[Item]) -> list[Layer]:
    """Giant BUT DON'T YOU FEEL; outline echoes zoom outward from every word."""
    f = ctx.f
    layers = []
    for it in items:
        ln = it.line
        on, off = ctx.fr(it.on), ctx.fr(it.off)
        cap = ctx.u(190)
        tool = words_layer(ctx, it, HOOK, cap, (f.cx, f.cy + ctx.u(30)), lambda w: PALE_CYAN,
                           f.safe_w * 0.98, anim="slam", glow_size=16)
        but = word_text(ctx, f"{ln['id']}But", "BUT", HOOK, ctx.u(70), CYAN, (f.cx, f.cy - ctx.u(150)))
        bm = ctx.c.merge(ctx.name(f"{ln['id']}BM"), tool, but)
        ctx.c.keyframes(bm, "Blend", {on - 1 if on > 0 else 0: 0.0, max(on, 1) + 4: 1.0}, ease="out_cubic")
        acc = bm
        # outline echoes: the whole phrase as outline, zooming out and fading, on each word onset
        echo_txt = " ".join(upper(w["w"]) for w in ln["words"])
        for i, w in enumerate(ln["words"]):
            t = ctx.fr(w["start"])
            e = word_text(ctx, f"{ln['id']}E{i}", echo_txt, HOOK, cap, CYAN, (f.cx, f.cy + ctx.u(30)),
                          outline=0.006, fill=0.0)
            et = ctx.c.transform(ctx.name(f"{e}Z"), e, pivot=ctx.px(f.cx, f.cy), motion_blur=False)
            ctx.c.keyframes(et, "Size", {t: 1.0, t + 40: 2.1}, ease="out_cubic")
            em = ctx.c.merge(ctx.name(f"{e}M"), et, acc)  # echo behind the text
            # fade the echo itself: blend of the echo layer via a dissolve against the base
            ed = ctx.c.dissolve(ctx.name(f"{e}D"), acc, em)
            ctx.c.keyframes(ed, "Mix", {max(t - 1, 0): 0.0, t: 0.75, t + 40: 0.0}, ease="out_cubic")
            acc = ed
        fd, end_f = line_fade(ctx, acc, on, off, fade_out=FADE_F)
        layers.append(Layer(fd, on, end_f, (ln["id"],)))
    return layers


def style_drop_lines(ctx: Ctx, items: list[Item]) -> list[Layer]:
    """Final drop: call and response. B lines left (16:9) / top (9:16) in pink with slams + RGB
    split; A answers right / bottom in cyan."""
    f = ctx.f
    left, top, right, bottom = f.safe
    layers = []
    for it in items:
        ln = it.line
        on, off = ctx.fr(it.on), ctx.fr(it.off)
        is_a = ln["singer"].startswith("A")
        row = ((it.slot or 0) % 2) * 2 - 1  # consecutive lines alternate up/down: no overprint
        if f.vertical:
            center = (f.cx, top + (bottom - top) * (0.74 if is_a else 0.34) + row * ctx.u(70))
            max_w = f.safe_w
        else:
            center = (left + (right - left) * (0.78 if is_a else 0.285),
                      f.cy + (ctx.u(70) if is_a else -ctx.u(40)) + row * ctx.u(95))
            max_w = (right - left) * (0.40 if is_a else 0.54)
        if is_a:
            tool = words_layer(ctx, it, HOOK, ctx.u(150), center, lambda w: CYAN, max_w, anim="slam",
                               glow_size=14)
        else:
            tool = words_layer(ctx, it, HOOK, ctx.u(112), center, lambda w: ACID if w.get("key") else PINK,
                               max_w, anim="slam", rgb=True, key_scale=1.18)
        fd, end_f = line_fade(ctx, tool, on, off, fade_out=6)
        layers.append(Layer(fd, on, end_f, (ln["id"],)))
    return layers


def sched_drop(lines: list[dict]) -> list[Item]:
    out = []
    for i, ln in enumerate(lines):
        same = [x for x in lines[i + 1:] if x["singer"][0] == ln["singer"][0]]
        nxt = same[0]["start"] if same else ln["end"] + 0.8
        out.append(Item(ln, ln["start"], hold_off(ln, nxt)))
    return out


def style_outro(ctx: Ctx, items: list[Item]) -> list[Layer]:
    """Delay echoes: the phrase repeats every 1/8 note as trailing, fading copies."""
    f = ctx.f
    layers = []
    eighth = ctx.data["meta"]["beat_s"] / 2
    for it in items:
        ln = it.line
        on, off = ctx.fr(it.on), ctx.fr(it.off)
        main = words_layer(ctx, it, HOOK, ctx.u(170), (f.cx, f.cy - ctx.u(40)), lambda w: PALE_CYAN,
                           f.safe_w, anim="slam", glow_size=14)
        acc = main
        for k in range(1, 5):
            dt = ctx.fr(it.on + k * eighth) - on
            tr = ctx.c.transform(ctx.name(f"{ln['id']}Echo{k}"), main, motion_blur=False,
                                 center=(0.5 + k * ctx.u(16) / f.width, 0.5 - k * ctx.u(22) / f.height),
                                 size=1.0 - 0.03 * k)
            dd = ctx.c.merge(ctx.name(f"{ln['id']}EM{k}"), tr, acc)
            # echo k appears 1/8 note later per step and fades with the feedback
            ed = ctx.c.dissolve(ctx.name(f"{ln['id']}ED{k}"), acc, dd)
            a0 = on + dt
            ctx.c.keyframes(ed, "Mix", {max(a0 - 1, 0): 0.0, a0: 0.55 ** k * 1.4, a0 + 70: 0.0}, ease="out_cubic")
            acc = ed
        fd, end_f = line_fade(ctx, acc, on, off, fade_out=24)
        layers.append(Layer(fd, on, end_f, (ln["id"],)))
    return layers


def sched_outro(lines: list[dict]) -> list[Item]:
    return [Item(ln, ln["start"], hold_off(ln, ln["end"] + 1.6)) for ln in lines]


# ----------------------------------------------------------------------------- section-level fx
def shake(ctx: Ctx, tool: str, kicks: list[float], amp_px: float = 14.0) -> str:
    tr = ctx.c.transform(ctx.name("Shake"), tool, motion_blur=False)
    keys: dict[int, tuple[float, float]] = {0: (0.5, 0.5)}
    akeys: dict[int, float] = {0: 0.0}
    for t in kicks:
        k = ctx.fr(t)
        if k < 1 or k + 7 >= ctx.n or any(k - 1 <= x <= k + 6 for x in keys if x):
            continue
        dx = ctx.rng.uniform(-1, 1) * ctx.u(amp_px) / ctx.f.width
        dy = ctx.rng.uniform(-1, 1) * ctx.u(amp_px) / ctx.f.height
        keys[k - 1] = (0.5, 0.5)
        keys[k] = (0.5 + dx, 0.5 + dy)
        keys[k + 2] = (0.5 - dx * 0.5, 0.5 - dy * 0.5)
        keys[k + 6] = (0.5, 0.5)
        ang = ctx.rng.uniform(-0.8, 0.8)
        akeys[k - 1], akeys[k], akeys[k + 6] = 0.0, ang, 0.0
    if len(keys) > 1:
        ctx.c.keyframes(tr, "Center", dict(sorted(keys.items())), ease="out_cubic")
        ctx.c.keyframes(tr, "Angle", dict(sorted(akeys.items())), ease="out_cubic")
    return tr


def kick_punch(ctx: Ctx, tool: str, kicks: list[float], amount: float = 0.045) -> str:
    tr = ctx.c.transform(ctx.name("Punch"), tool, motion_blur=False)
    keys: dict[int, float] = {0: 1.0}
    for t in kicks:
        k = ctx.fr(t)
        if k < 1 or k + 7 >= ctx.n or any(k - 1 <= x <= k + 7 for x in keys if x):
            continue
        keys[k - 1], keys[k], keys[k + 7] = 1.0, 1.0 + amount, 1.0
    if len(keys) > 1:
        ctx.c.keyframes(tr, "Size", dict(sorted(keys.items())), ease="out_cubic")
    return tr


def glitch(ctx: Ctx, tool: str, times: list[float], n_slices: int = 4) -> str:
    """Horizontal slices shifted sideways for 3 frames at `times` (switched off otherwise)."""
    acc = tool
    for s in range(n_slices):
        h = ctx.rng.uniform(0.04, 0.12)
        y = ctx.rng.uniform(0.2, 0.8)
        mask = ctx.c.rect_mask(ctx.name(f"Slice{s}"), center=(0.5, y), width=1.2, height=h)
        tr = ctx.c.transform(ctx.name(f"SliceT{s}"), acc, motion_blur=False)
        keys: dict[int, tuple[float, float]] = {0: (0.5, 0.5)}
        for t in times:
            k = ctx.fr(t)
            if k < 1 or k + 4 >= ctx.n:
                continue
            dx = ctx.rng.choice((-1, 1)) * ctx.rng.uniform(25, 70) / ctx.f.width
            keys[k - 1], keys[k], keys[k + 2], keys[k + 3] = (0.5, 0.5), (0.5 + dx, 0.5), (0.5 - dx * 0.4, 0.5), (0.5, 0.5)
        if len(keys) > 1:
            ctx.c.keyframes(tr, "Center", dict(sorted(keys.items())), ease="step")
        ctx.c.apply_mask(tr, mask)
        acc = tr
    sw = ctx.c.dissolve(ctx.name("GlitchSw"), tool, acc)
    keys2: dict[int, float] = {0: 0.0}
    for t in times:
        k = ctx.fr(t)
        if 1 <= k and k + 4 < ctx.n:
            keys2[k - 1], keys2[k], keys2[k + 4] = 0.0, 1.0, 0.0
    if len(keys2) > 1:
        ctx.c.keyframes(sw, "Mix", dict(sorted(keys2.items())), ease="step")
    return sw


def feel_chops(ctx: Ctx, chops: list[dict], cap: float, low_band: bool = False) -> list[Layer]:
    """FEEL chops flicker at changing positions (one Text+ moved per chop). `low_band` keeps
    them in the lower edge of the frame, clear of the lyric lines (outside the drops)."""
    if not chops:
        return []
    f = ctx.f
    left, top, right, bottom = f.safe
    if low_band:
        top = bottom - (bottom - top) * 0.16
    txt = word_text(ctx, "Feel", "FEEL", HOOK, cap, ACID, (f.width / 2, f.height / 2))
    tr = ctx.c.transform(ctx.name("FeelPos"), txt, motion_blur=False)
    keys: dict[int, tuple[float, float]] = {}
    bkeys: dict[int, float] = {0: 0.0}
    colors = []
    for ch in chops:
        k = ctx.fr(ch["t"])
        e = max(k + 6, ctx.fr(ch["end"]))
        x = ctx.rng.uniform(left + cap, right - cap)
        y = ctx.rng.uniform(top + cap, bottom - cap)
        keys[k] = (x / f.width, 1 - y / f.height)
        bkeys[k] = 1.0
        bkeys[k + 3] = 0.55        # flicker
        bkeys[k + 5] = 1.0
        bkeys[e] = 0.0
        colors.append(k)
    if len(keys) >= 2:
        ctx.c.keyframes(tr, "Center", {k: (v[0], v[1]) for k, v in sorted(keys.items())}, ease="step")
    else:
        ctx.c.set_input(tr, "Center", next(iter(keys.values())))
    clear = ctx.c.background(ctx.name("FeelClear"), (0, 0, 0), alpha=0.0)
    m = ctx.c.merge(ctx.name("FeelM"), clear, tr)
    ctx.c.keyframes(m, "Blend", dict(sorted(bkeys.items())), ease="step")
    on = max(0, ctx.fr(chops[0]["t"]) - 1)
    off = min(ctx.n, ctx.fr(chops[-1]["end"]) + 8)
    return [Layer(m, on, off)]


# ----------------------------------------------------------------------------- sections
SCHED: dict[str, Callable[[list[dict]], list[Item]]] = {
    "B Refrain 1": sched_refrain, "B Refrain 2": sched_refrain, "B Strophe 1": sched_rap,
    "Post-Chorus": lambda ls: [Item(l, l["start"], hold_off(l, l["end"] + 0.5)) for l in ls],
    "Breakdown": sched_breakdown, "Finaler Drop": sched_drop, "Outro": sched_outro,
    "Intro": lambda ls: [Item(l, l["start"], hold_off(l)) for l in ls],
    "Drop 1": lambda ls: [Item(l, l["start"], hold_off(l)) for l in ls],
}


def render_items(ctx: Ctx, section: str, items: list[Item]) -> list[Layer]:
    if not items:
        return []
    if section in ("B Refrain 1", "B Refrain 2"):
        return style_refrain(ctx, items)
    if section == "B Strophe 1":
        # a refrain-singer word inside the verse ("Fuck") is just a line of the verse
        return style_rap(ctx, items)
    if section == "Post-Chorus":
        return style_postchorus(ctx, items)
    if section == "Breakdown":
        return style_breakdown(ctx, items)
    if section == "Finaler Drop":
        return style_drop_lines(ctx, items)
    if section == "Outro":
        return style_outro(ctx, items)
    if section == "Intro":
        return style_hook_words(ctx, items, ctx.u(360))
    return style_drop_lines(ctx, items)


def section_lines(data: dict[str, Any], name: str) -> list[dict]:
    return [ln for ln in data["lines"] if ln["section"] == name]


def schedule(data: dict[str, Any], name: str) -> list[Item]:
    items = SCHED[name](section_lines(data, name))
    for k, it in enumerate(items):
        it.slot = k  # position index in the section, kept when the next comp carries the line
    return items


def span(data: dict[str, Any], name: str) -> tuple[float, float]:
    for s in data["sections"]:
        if s["name"] == name:
            return s.get("comp_start", s["start"]), s.get("comp_end", s["end"])
    raise KeyError(name)


def coverage(data: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    """line id -> time windows in which some comp renders it (own comp + tail carry)."""
    cov: dict[str, list[tuple[float, float]]] = {}
    names = [s["name"] for s in data["sections"]]
    for i, name in enumerate(names):
        a, b = span(data, name)
        for it in schedule(data, name):
            cov.setdefault(it.line["id"], []).append((max(a, it.on), min(b, it.off)))
        if i > 0:
            for it in tail_items(data, names[i - 1], a):
                cov.setdefault(it.line["id"], []).append((a, min(b, it.off)))
    return cov


def tail_items(data: dict[str, Any], prev: str, start: float) -> list[Item]:
    return [it for it in schedule(data, prev) if it.off > start + 1 / FPS]


# ----------------------------------------------------------------------------- section builders
def build_section(data: dict[str, Any], f: Fmt, name: str) -> cw.Comp:
    a, b = span(data, name)
    f0 = int(round(a * FPS))
    n = int(round(b * FPS)) - f0
    c = cw.Comp(f.width, f.height, n)
    ctx = Ctx(c, f, data, f0, n)
    names = [s["name"] for s in data["sections"]]
    i = names.index(name)
    layers: list[Layer] = []
    if i > 0:
        layers += render_items(ctx, names[i - 1], tail_items(data, names[i - 1], a))
    items = schedule(data, name)
    special = {"Intro": _intro, "Drop 1": _drop1, "Finaler Drop": _final_drop, "Outro": _outro}
    if name in special:
        out = special[name](ctx, items, layers)
    else:
        layers += render_items(ctx, name, items)
        layers += feel_chops(ctx, _chops_in(ctx, "feel"), ctx.u(80), low_band=True)
        out = switch_chain(ctx, layers)
    c.output(out)
    return c


def _chops_in(ctx: Ctx, kind: str | None = None) -> list[dict]:
    a = ctx.f0 / FPS
    b = (ctx.f0 + ctx.n) / FPS
    return [ch for ch in ctx.data["chops"] if a <= ch["t"] < b and (kind is None or ch["kind"] == kind)]


def _kicks_in(ctx: Ctx) -> list[float]:
    a = ctx.f0 / FPS
    b = (ctx.f0 + ctx.n) / FPS
    return [k for k in ctx.data["kicks"] if a <= k < b]


def _intro(ctx: Ctx, items: list[Item], layers: list[Layer]) -> str:
    f, c = ctx.f, ctx.c
    bar = ctx.data["meta"]["bar_s"]
    t_stop, t_black = 15.5 * bar, 15.75 * bar
    # title: from blur to sharp with the opening low-pass (cutoff progress x**2.2, bars 0-15.5)
    title = word_text(ctx, "Title", "DON'T YOU FEEL", HOOK, ctx.u(150), WHITE, (f.cx, f.cy))
    tb = c.blur("TitleBlur", title, 60.0)
    keys = {}
    for k in range(13):
        x = k / 12
        keys[ctx.fr(x * t_stop)] = round(60.0 * (1 - x ** 2.2), 3)
    c.keyframes(tb, "XBlurSize", keys, ease="linear")
    tg = glow(ctx, tb, 30.0, 0.6)
    clear = c.background("TitleClear", (0, 0, 0), alpha=0.0)
    tm = c.merge("TitleM", clear, tg)
    hooks = [it for it in items if it.line["singer"] == "A-Hook"]
    tkeys = {0: 0.0, ctx.fr(0.6): 0.95}
    if hooks:
        h0, h1 = ctx.fr(hooks[0].on), ctx.fr(hooks[-1].off)
        tkeys.update({h0 - 6: 0.95, h0: 0.22, h1: 0.22, h1 + 20: 0.95})
    c.keyframes(tm, "Blend", dict(sorted(tkeys.items())), ease="in_out_cubic")
    words = style_hook_words(ctx, items, ctx.u(360))
    base = switch_chain(ctx, layers + words, base=tm, tag="I")
    # stutter: DON'T YOU copies multiply into a grid on every chop
    stut = sorted(_chops_in(ctx, "dont_you"), key=lambda ch: ch["t"])
    if stut:
        cols, rows = (4, 4) if not f.vertical else (2, 8)
        left, top, right, bottom = f.safe
        cells = [(left + (ci + 0.5) * (right - left) / cols, top + (ri + 0.5) * (bottom - top) / rows)
                 for ri in range(rows) for ci in range(cols)]
        cells.sort(key=lambda p: (p[0] - f.cx) ** 2 + (p[1] - f.cy) ** 2)
        counts = [1, 2, 4, 6, 9, 12, 16][: len(stut)]
        counts[-1] = len(cells)
        acc = base
        cap = ctx.u(64) if not f.vertical else ctx.u(58)
        for j, (x, y) in enumerate(cells):
            first = next((k for k, n in enumerate(counts) if n > j), None)
            if first is None:
                continue
            on = ctx.fr(stut[first]["t"])
            cell = word_text(ctx, f"Grid{j}", "DON'T YOU", HOOK, cap, CYAN if j % 3 else ACID, (x, y))
            m = c.merge(ctx.name("GridM"), acc, cell)
            sw = c.dissolve(ctx.name("GridS"), acc, m)
            c.keyframes(sw, "Mix", {0: 0.0, on: 1.0}, ease="step")
            acc = sw
        base = acc
    # tape stop: everything falls away with motion blur, then one beat of black
    fall = c.transform("TapeStopFall", base, quality=4, shutter=270.0)
    s0, s1 = ctx.fr(t_stop), ctx.fr(t_black)
    c.keyframes(fall, "Center", {s0: (0.5, 0.5), s1: (0.5, -0.9)}, ease="in_expo")
    black = c.background("Black", (0, 0, 0), alpha=0.0)
    out = c.dissolve("BlackBeat", fall, black)
    c.keyframes(out, "Mix", {0: 0.0, s1: 1.0}, ease="step")
    return out


def _drop_common(ctx: Ctx, content: str) -> str:
    kicks = _kicks_in(ctx)
    snares = [s for s in ctx.data["snares"] if ctx.f0 / FPS <= s < (ctx.f0 + ctx.n) / FPS]
    g_times = snares[::2] if snares else [k for i, k in enumerate(kicks) if i % 4 == 2]
    out = glitch(ctx, content, g_times)
    out = kick_punch(ctx, out, kicks)
    return shake(ctx, out, kicks)


def _drop1(ctx: Ctx, items: list[Item], layers: list[Layer]) -> str:
    f, c = ctx.f, ctx.c
    bar = ctx.data["meta"]["bar_s"]
    sec = next(s for s in ctx.data["sections"] if s["name"] == "Drop 1")
    ones = [sec["start"] + k * bar for k in range(sec["bars"])]
    # FEEL slams on every bar one: one word, re-slammed (1.6 -> 1.0 in 4 frames) with RGB split
    t0 = ctx.fr(ones[0])
    big = word_text(ctx, "Slam", "FEEL", HOOK, ctx.u(300), WHITE, (f.cx, f.cy))
    big = rgb_split(ctx, big, "FEEL", HOOK, ctx.u(300), (f.cx, f.cy), [ctx.fr(t) for t in ones], 0.0, 1.0)
    tr = c.transform("SlamT", big, pivot=(0.5, 0.5), quality=4)
    skeys: dict[int, float] = {}
    for t in ones:
        k = ctx.fr(t)
        if k > 0:
            skeys[k - 1] = 1.0  # hold between slams (no drift back to 1.6, blur only on the slam)
        skeys[k] = 1.6
        skeys[k + SLAM_F] = 1.0
    c.keyframes(tr, "Size", skeys, ease="out_cubic")
    clear = c.background("SlamClear", (0, 0, 0), alpha=0.0)
    sm = c.merge("SlamM", clear, tr)
    bkeys = {0: 0.0, t0: 1.0}
    for t in ones:
        k = ctx.fr(t)
        bkeys[k] = 1.0
        bkeys[k + 40] = 1.0
        bkeys[k + 80] = 0.8
    c.keyframes(sm, "Blend", dict(sorted(bkeys.items())), ease="out_cubic")
    lay = [Layer(sm, max(0, t0 - 1), ctx.n)]
    lay += feel_chops(ctx, _chops_in(ctx, "feel"), ctx.u(84))
    content = switch_chain(ctx, layers + lay, tag="D")
    return _drop_common(ctx, content)


def _final_drop(ctx: Ctx, items: list[Item], layers: list[Layer]) -> str:
    lay = style_drop_lines(ctx, items)
    lay += feel_chops(ctx, _chops_in(ctx, "feel"), ctx.u(90))
    content = switch_chain(ctx, layers + lay, tag="F")
    return _drop_common(ctx, content)


def _outro(ctx: Ctx, items: list[Item], layers: list[Layer]) -> str:
    f, c = ctx.f, ctx.c
    lay = layers + style_outro(ctx, items)
    # end card from after the last phrase to the very end
    last = max((it.off for it in items), default=ctx.f0 / FPS + 1.0)
    on = ctx.fr(last) + 6
    title = word_text(ctx, "EndTitle", "DON'T YOU FEEL", HOOK, ctx.u(150), WHITE, (f.cx, f.cy - ctx.u(30)))
    tg = glow(ctx, title, 24.0, 0.7)
    sub = word_text(ctx, "EndSub", "(REMIX)", REFRAIN, ctx.u(70), PINK, (f.cx, f.cy + ctx.u(110)))
    clear = c.background("EndClear", (0, 0, 0), alpha=0.0)
    m1 = c.merge("EndM1", clear, tg)
    m2 = c.merge("EndM2", m1, sub)
    c.keyframes(m2, "Blend", {on: 0.0, on + 40: 1.0}, ease="out_cubic")
    c.keyframes(m1, "Blend", {on: 0.0, on + 30: 1.0}, ease="out_cubic")
    zoom = c.transform("EndZoom", m2, motion_blur=False)
    c.keyframes(zoom, "Size", {on: 0.94, ctx.n - 1: 1.04}, ease="linear")
    lay.append(Layer(zoom, on, ctx.n + 1))
    return switch_chain(ctx, lay, tag="O")


def end_card_on(data: dict[str, Any]) -> float:
    items = schedule(data, "Outro")
    return max(it.off for it in items) + 0.1 if items else data["meta"]["duration_s"] - 6

