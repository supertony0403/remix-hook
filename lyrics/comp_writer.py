"""Write DaVinci Resolve 21 / Fusion ``.comp`` files from Python.

Copied from ~/Documents/Programmierung/nomissuccess-spot/fusion/comp_writer.py (06.10.2026,
measured on Resolve 21.1.1) for the remix lyric video. Changes: the nomissuccess timeline
contract at the end was dropped; ``dissolve``/``blur``/``fast_noise`` helpers were added
(regids checked against /opt/resolve/Developer/Fusion Templates); the cap-height table knows
Anton/Bebas Neue/Space Grotesk via FontMetrics.

A ``Comp`` collects tools (Text+, Transform, Merge, Background, masks, Loader …),
their links and their animation, and serialises them to the Lua-table text that
Fusion reads (``Composition { Tools = { … } }``). The structure follows comps saved by
Resolve 21.1.1 itself (see ``work/fusion/referenz/*.comp``):

* static input:      ``Size = Input { Value = 0.05, },``
* link:              ``Input = Input { SourceOp = "Text1", Source = "Output", },``
* animation:         a ``BezierSpline`` tool whose ``KeyFrames`` carry absolute
  ``RH``/``LH`` handles; point inputs go through an ``XYPath`` with an X and a Y spline
* character styling: a ``StyledTextFollower`` modifier drives ``StyledText``

Coordinates are Fusion's: normalised 0..1, origin bottom-left. ``Comp.px()`` converts
top-left pixel coordinates. Every generator gets an explicit ``Width``/``Height`` so a
comp renders the same no matter which timeline it is imported into; the background
is transparent unless asked otherwise, so comps can sit above the Blender shots.

Measured on Resolve 21.1.1 (05.10.2026):
* Text+ ``Size`` scales with the *image width* (cap height = ratio × Size × width).
* Rectangle masks: ``Width`` is relative to image width, ``Height`` to height.
* Ellipse masks: *both* sizes are relative to image width in Fusion (measured 06.10.2026,
  a ``Height`` of 54/1080 drew a 96 px tall cap); ``ellipse_mask`` converts, so callers
  keep passing ``Height`` relative to height like for rectangles.
* Text+ ``CharacterSpacing`` s adds (s − 1) × Size × image width after each glyph
  (measured 06.10.2026 on JetBrains Mono and Manrope, ±1 px); it is not a multiplier.
* Background colours are not premultiplied by ``TopLeftAlpha``: Merge adds the full
  colour. ``background`` premultiplies, so alpha < 1 really is translucent.
* Text+ anchor: ``HorizontalLeftCenterRight`` −1/0/1 = left/centre/right of ``Center``;
  ``VerticalTopCenterBottom`` −1 = the text hangs below ``Center`` (top anchor).
* ``ImportFusionComp`` turns Saver tools into MediaOut, so control renders attach a
  Saver through the API after import (see ``resolve/resolve_api.py``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESOLVE_VERSION = "DaVinci Resolve Studio 21.1.1.0010"

# cubic-bezier(x1, y1, x2, y2) in normalised segment space; None = hold ("step")
EASES: dict[str, tuple[float, float, float, float] | None] = {
    "linear": (1 / 3, 1 / 3, 2 / 3, 2 / 3),
    "in_cubic": (0.32, 0.0, 0.67, 0.0),
    "out_cubic": (0.33, 1.0, 0.68, 1.0),
    "in_out_cubic": (0.65, 0.0, 0.35, 1.0),
    "out_expo": (0.16, 1.0, 0.3, 1.0),
    "out_quart": (0.25, 1.0, 0.5, 1.0),
    "in_out_expo": (0.87, 0.0, 0.13, 1.0),
    "in_expo": (0.7, 0.0, 0.84, 0.0),
    "out_back": (0.34, 1.56, 0.64, 1.0),
    "step": None,
}

# StyledTextFollower "Order" value. Verified by render on Resolve 21.1.1 (delay 4, opacity
# step, frame 9); the German combo labels are shifted and must not be trusted.
FOLLOWER_ORDER: dict[str, int] = {
    "left_to_right": 0,
    "right_to_left": 1,
    "inside_out": 2,
    "outside_in": 3,
    "random_one_by_one": 4,
    "random": 5,
    "manual": 6,
}
FOLLOWER_DELAY_TYPE: dict[str, int] = {"none": 0, "between_each": 1, "first_to_last": 2}

# cap height in px = FONT_CAP_PER_SIZE[font] × Size × image width (measured in Resolve)
FONT_CAP_PER_SIZE: dict[str, float] = {
    "JetBrains Mono": 0.4375,
    "Adwaita Sans": 0.4792,   # stand-in while Manrope is not visible to Fusion
}

# Fusion scales a font so that its full height (hhea ascent + descent) equals
# FUSION_FONT_HEIGHT × Size × image width. Measured on JetBrains Mono (0.791) and
# Adwaita Sans (0.797); lets FontMetrics predict caps and widths for any font file.
FUSION_FONT_HEIGHT = 0.794

JUSTIFY = {"left": (-1, 0), "center": (0, 1), "right": (1, 2)}
ANCHOR_V = {"top": -1, "center": 0, "bottom": 1}

GRID_16x9 = (120, 96)          # left/right and top/bottom margin of the text grid, px

# Motion-blur samples per frame. Fusion renders a tool ``Quality`` times per frame while
# MotionBlur is on, moving or not; 8–24 made the master render ~3 h. Cap at 4 (6 for
# the fast rolling-counter drums, ``fast=True``); ``Comp.to_text`` additionally drops
# blur on tools that never move and keys Quality to 1 outside their motion windows.
MB_QUALITY_MAX = 4
MB_QUALITY_FAST = 6
SAFE_9x16 = (90, 215, 405)      # sides, top, bottom of the 9:16 safe area, px

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED = {"MediaOut1"}
_MODIFIER_TYPES = {"BezierSpline", "XYPath", "StyledTextFollower"}


# --------------------------------------------------------------------------------------
# Lua value types
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FuID:
    """``FuID { "name" }`` — Fusion's enum identifiers (blend modes, gradient type …)."""

    name: str


@dataclass(frozen=True)
class Ctor:
    """A typed Lua constructor such as ``StyledText { Value = "…" }``."""

    type_name: str
    body: Mapping[Any, Any] | tuple


@dataclass(frozen=True)
class Link:
    op: str
    source: str = "Output"


@dataclass(frozen=True)
class Expr:
    text: str


@dataclass(frozen=True)
class Box:
    """Pixel rectangle, top-left origin."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass
class FollowerSpec:
    """Character-level animation through a ``StyledTextFollower`` modifier.

    ``keys`` animate follower inputs over comp time (each character replays the curve
    ``delay`` frames after its predecessor). Useful inputs: ``CharacterOffset`` (point,
    in frame units), ``CharacterSizeX/Y``, ``CharacterAngleZ``, ``Opacity1`` (fill),
    ``Red1/Green1/Blue1``, ``SoftnessX1/Y1`` (per-character blur).
    """

    delay: float = 2.0
    order: str = "left_to_right"
    keys: dict[str, dict[int, Any]] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    ease: str = "out_cubic"
    per_key: dict[str, dict[int, str]] = field(default_factory=dict)
    delay_type: str = "between_each"

    def __post_init__(self) -> None:
        if self.order not in FOLLOWER_ORDER:
            raise ValueError(f"unknown follower order {self.order!r}; use one of {sorted(FOLLOWER_ORDER)}")
        if self.delay_type not in FOLLOWER_DELAY_TYPE:
            raise ValueError(f"unknown delay_type {self.delay_type!r}")
        _check_ease(self.ease)


@dataclass
class _Tool:
    name: str
    regid: str
    inputs: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    with_inputs: bool = True


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def hex_rgb(value: str) -> tuple[float, float, float]:
    """``#145fe4`` → (r, g, b) in 0..1."""
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", value.strip())
    if not m:
        raise ValueError(f"not a #rrggbb colour: {value!r}")
    h = m.group(1)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgba(color: Any, alpha: float | None = None) -> tuple[float, float, float, float]:
    if isinstance(color, str):
        r, g, b = hex_rgb(color)
        a = 1.0
    else:
        vals = tuple(float(c) for c in color)
        if len(vals) == 3:
            (r, g, b), a = vals, 1.0
        elif len(vals) == 4:
            r, g, b, a = vals
        else:
            raise ValueError(f"colour needs 3 or 4 components: {color!r}")
    if alpha is not None:
        a = float(alpha)
    for c in (r, g, b, a):
        _finite(c)
    return r, g, b, a


def _finite(x: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"not a finite number: {x!r}")
    return x


def _check_ease(name: str) -> None:
    if name not in EASES:
        raise ValueError(f"unknown ease {name!r}; use one of {sorted(EASES)}")


def _bezier(p1: float, p2: float, t: float) -> float:
    u = 1 - t
    return 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3


def sample_ease(name: str, steps: int = 30) -> list[float]:
    """Progress values (0..1) of an ease at ``steps+1`` evenly spaced times — for tests
    and for previewing what Fusion will interpolate between two keys."""
    _check_ease(name)
    curve = EASES[name]
    if curve is None:
        return [0.0] * steps + [1.0]
    x1, y1, x2, y2 = curve
    out = []
    for i in range(steps + 1):
        x = i / steps
        lo, hi = 0.0, 1.0
        for _ in range(60):  # bisection on x(t) = x
            mid = (lo + hi) / 2
            if _bezier(x1, x2, mid) < x:
                lo = mid
            else:
                hi = mid
        out.append(_bezier(y1, y2, (lo + hi) / 2))
    return out


def _num(x: float) -> str:
    _finite(x)
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return "0"
    text = format(x, ".10g")
    return text


def _quote(s: str) -> str:
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{out}"'


def _key(k: Any) -> str:
    if isinstance(k, str):
        return k if _NAME.match(k) else f"[{_quote(k)}]"
    return f"[{_num(k)}]"


def _lua(v: Any, ind: int) -> str:
    pad, pad1 = "\t" * ind, "\t" * (ind + 1)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _num(v)
    if isinstance(v, str):
        return _quote(v)
    if isinstance(v, Path):
        return _quote(str(v))
    if isinstance(v, FuID):
        return f"FuID {{ {_quote(v.name)} }}"
    if isinstance(v, Ctor):
        return f"{v.type_name} {_lua(v.body, ind)}"
    if isinstance(v, (tuple, list)):
        if all(isinstance(x, (int, float, str, bool)) for x in v):
            return "{ " + ", ".join(_lua(x, ind) for x in v) + " }"
        inner = ",\n".join(pad1 + _lua(x, ind + 1) for x in v)
        return "{\n" + inner + "\n" + pad + "}"
    if isinstance(v, Mapping):
        if not v:
            return "{ }"
        # Lua array part first ({ 1, RH = … } like Resolve writes keyframes), then named keys
        n = 0
        while (n + 1) in v and not isinstance(n + 1, bool):
            n += 1
        parts = [_lua(v[i], ind + 1) for i in range(1, n + 1)]
        named = [(k, val) for k, val in v.items() if not (isinstance(k, int) and 1 <= k <= n)]
        if all(isinstance(val, (int, float, str, bool, tuple, FuID)) for _, val in named) and len(v) <= 4:
            parts += [f"{_key(k)} = {_lua(val, ind + 1)}" for k, val in named]
            return "{ " + ", ".join(parts) + " }"
        inner = [pad1 + p for p in parts] + [f"{pad1}{_key(k)} = {_lua(val, ind + 1)}" for k, val in named]
        return "{\n" + ",\n".join(inner) + "\n" + pad + "}"
    raise TypeError(f"cannot write {type(v).__name__} to a comp: {v!r}")


def _input(v: Any, ind: int) -> str:
    if isinstance(v, Link):
        return f"Input {{ SourceOp = {_quote(v.op)}, Source = {_quote(v.source)}, }}"
    if isinstance(v, Expr):
        return f"Input {{ Expression = {_quote(v.text)}, }}"
    return f"Input {{ Value = {_lua(v, ind)}, }}"


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", s)


@dataclass(frozen=True)
class FontMetrics:
    """Metrics of an installed font file (via fc-match + fontTools), scaled the way
    Fusion's Text+ scales it. Used to place glyph-relative elements (the bell after
    "jemanden", digit cells of the rolling counter) without a render round trip."""

    family: str
    style: str
    path: str
    upm: int
    height: float            # (ascent + descent) / upm
    cap: float               # cap height / upm
    advances: tuple          # ((codepoint, advance units), …)

    @classmethod
    def find(cls, family: str, style: str = "Regular") -> FontMetrics:
        import subprocess

        try:
            out = subprocess.run(["fc-match", "-f", "%{family}|%{style}|%{file}", f"{family}:style={style}"],
                                 capture_output=True, text=True, timeout=10, check=True).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise LookupError(f"fc-match failed for {family} {style}: {exc}") from exc
        fams, _, path = out.split("|", 2)
        if family.lower() not in [f.strip().lower() for f in fams.split(",")]:
            raise LookupError(f"font {family!r} is not installed (fc-match offers {fams!r})")
        return cls.load(path, family, style)

    @classmethod
    def load(cls, path: str, family: str = "", style: str = "") -> FontMetrics:
        from fontTools.pens.boundsPen import BoundsPen
        from fontTools.ttLib import TTFont

        font = TTFont(path, fontNumber=0, lazy=True)
        upm = font["head"].unitsPerEm
        hhea = font["hhea"]
        cmap = font.getBestCmap()
        glyphs = font.getGlyphSet()
        pen = BoundsPen(glyphs)
        glyphs[cmap[ord("H")]].draw(pen)
        cap = pen.bounds[3] - pen.bounds[1]
        hmtx = font["hmtx"].metrics
        advances = tuple(sorted((cp, hmtx[g][0]) for cp, g in cmap.items() if g in hmtx))
        return cls(family, style, path, upm, (hhea.ascent - hhea.descent) / upm, cap / upm, advances)

    def em_px(self, size: float, image_width: int) -> float:
        return FUSION_FONT_HEIGHT * size * image_width / self.height

    def cap_px(self, size: float, image_width: int) -> float:
        return self.cap * self.em_px(size, image_width)

    def cap_per_size(self) -> float:
        return FUSION_FONT_HEIGHT * self.cap / self.height

    def width_px(self, text: str, size: float, image_width: int, tracking: float = 1.0) -> float:
        """Advance width of ``text`` (no kerning). ``tracking`` is Text+ CharacterSpacing,
        which adds (tracking − 1) × Size × image width between glyphs (measured)."""
        adv = dict(self.advances)
        units = sum(adv.get(ord(ch), self.upm // 2) for ch in text)
        extra = (tracking - 1.0) * size * image_width * max(0, len(text) - 1)
        return units / self.upm * self.em_px(size, image_width) + extra


# --------------------------------------------------------------------------------------
# the composition
# --------------------------------------------------------------------------------------


class Comp:
    """A Fusion composition of ``frames`` frames at ``width``×``height`` and ``fps``."""

    def __init__(self, width: int, height: int, frames: int, fps: int = 60):
        for n, v in (("width", width), ("height", height), ("frames", frames), ("fps", fps)):
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{n} must be a positive int, got {v!r}")
        self.width, self.height, self.frames, self.fps = width, height, frames, fps
        self._tools: dict[str, _Tool] = {}
        self._output: Link | None = None

    # ---------------------------------------------------------------- geometry helpers
    @property
    def vertical(self) -> bool:
        return self.height > self.width

    def px(self, x: float, y: float) -> tuple[float, float]:
        """Top-left pixel position → Fusion's normalised point (origin bottom-left)."""
        return (_finite(x) / self.width, 1 - _finite(y) / self.height)

    def px_w(self, dx: float) -> float:
        """Horizontal pixel distance → normalised (also mask ``Width``)."""
        return _finite(dx) / self.width

    def px_h(self, dy: float) -> float:
        """Vertical pixel distance → normalised (also mask ``Height``)."""
        return _finite(dy) / self.height

    def size_for_cap(self, cap_px: float, font: str = "JetBrains Mono", style: str = "Regular") -> float:
        """Text+ ``Size`` that gives a cap height of ``cap_px`` pixels for ``font``.
        Uses a measured ratio if known, else the installed font file."""
        ratio = FONT_CAP_PER_SIZE.get(font)
        if ratio is None:
            ratio = FontMetrics.find(font, style).cap_per_size()
        return _finite(cap_px) / (ratio * self.width)

    def safe_box(self) -> Box:
        """Layout box in pixels: the 9:16 safe area, or the 16:9 text grid."""
        if self.vertical:
            side, top, bottom = SAFE_9x16
            return Box(side, top, self.width - side, self.height - bottom)
        mx, my = GRID_16x9
        return Box(mx, my, self.width - mx, self.height - my)

    # ---------------------------------------------------------------- tool registry
    def _add(self, name: str, regid: str, inputs: dict | None = None, *, with_inputs: bool = True,
             **fields: Any) -> _Tool:
        if not isinstance(name, str) or not _NAME.match(name) or not name.isascii():
            raise ValueError(f"invalid tool name {name!r} (ASCII letters, digits, _; no leading digit)")
        if name in _RESERVED or name in self._tools:
            raise ValueError(f"tool name {name!r} is already used")
        tool = _Tool(name, regid, dict(inputs or {}), dict(fields), with_inputs)
        self._tools[name] = tool
        return tool

    def _get(self, name: str) -> _Tool:
        if name not in self._tools:
            raise KeyError(f"no tool called {name!r}")
        return self._tools[name]

    def _unique(self, base: str) -> str:
        base = _safe(base) or "Tool"
        if base[0].isdigit():
            base = "T" + base
        name, i = base, 2
        while name in self._tools or name in _RESERVED:
            name, i = f"{base}{i}", i + 1
        return name

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def tool(self, name: str, regid: str, inputs: Mapping[str, Any] | None = None, **fields: Any) -> str:
        """Generic tool (escape hatch for Glow, Blur, EllipseMask …). Values in ``inputs``
        may be numbers, tuples, strings, ``FuID``, ``Ctor`` or ``Link``."""
        self._add(name, regid, dict(inputs or {}), **fields)
        return name

    def set_input(self, tool: str, inp: str, value: Any) -> None:
        self._get(tool).inputs[inp] = value

    def link(self, tool: str, inp: str, source_tool: str, source: str = "Output") -> None:
        self._get(tool).inputs[inp] = Link(source_tool, source)

    def expression(self, tool: str, inp: str, expr: str) -> None:
        """Drive an input with a Fusion simple expression (``time`` = current frame)."""
        self._get(tool).inputs[inp] = Expr(expr)

    def output(self, tool: str) -> None:
        """Connect ``tool`` to MediaOut1 — what Resolve shows on the timeline."""
        self._output = Link(tool)

    def _frame_size(self) -> dict[str, Any]:
        return {"Width": self.width, "Height": self.height}

    # ---------------------------------------------------------------- tools
    def text(
        self,
        name: str,
        text: str,
        font: str = "Manrope",
        style: str = "Bold",
        size: float = 0.05,
        color: Any = (1.0, 1.0, 1.0, 1.0),
        center: tuple[float, float] = (0.5, 0.5),
        tracking: float = 1.0,
        justify: str = "left",
        follower: FollowerSpec | None = None,
        *,
        anchor_v: str = "top",
        line_spacing: float = 1.0,
        motion_blur: bool = False,
        quality: int = MB_QUALITY_MAX,
        shutter: float = 180.0,
    ) -> str:
        """Text+ tool. ``tracking`` is Fusion's CharacterSpacing (1.0 = font default).
        ``color`` is (r, g, b[, a]) or ``#rrggbb``; alpha goes to ``Opacity1``."""
        if justify not in JUSTIFY:
            raise ValueError(f"justify must be one of {sorted(JUSTIFY)}")
        if anchor_v not in ANCHOR_V:
            raise ValueError(f"anchor_v must be one of {sorted(ANCHOR_V)}")
        r, g, b, a = _rgba(color)
        anchor, just = JUSTIFY[justify]
        inputs: dict[str, Any] = {
            **self._frame_size(),
            "Font": font,
            "Style": style,
            "Size": _finite(size),
            "Center": (_finite(center[0]), _finite(center[1])),
            "CharacterSpacing": _finite(tracking),
            "LineSpacing": _finite(line_spacing),
            "HorizontalLeftCenterRight": anchor,
            "HorizontalJustificationNew": just,
            "VerticalTopCenterBottom": ANCHOR_V[anchor_v],
            "Red1": r, "Green1": g, "Blue1": b, "Opacity1": a,
        }
        if motion_blur:
            inputs.update(MotionBlur=1, Quality=min(int(quality), MB_QUALITY_MAX), ShutterAngle=_finite(shutter))
        tool = self._add(name, "TextPlus", inputs)
        if follower is None:
            tool.inputs["StyledText"] = text
        else:
            fname = self._unique(f"{name}Follower")
            finputs: dict[str, Any] = {
                "Text": Ctor("StyledText", {"Value": text}),
                "Delay": _finite(follower.delay),
                "Order": FOLLOWER_ORDER[follower.order],
                "DelayType": FOLLOWER_DELAY_TYPE[follower.delay_type],
            }
            finputs.update(follower.values)
            self._add(fname, "StyledTextFollower", finputs)
            tool.inputs["StyledText"] = Link(fname, "StyledText")
            for inp, keys in follower.keys.items():
                self.keyframes(fname, inp, keys, ease=follower.ease, per_key=follower.per_key.get(inp))
        return name

    def transform(
        self,
        name: str,
        input: str,
        center: tuple[float, float] = (0.5, 0.5),
        size: float = 1.0,
        angle: float = 0.0,
        motion_blur: bool = True,
        quality: int = MB_QUALITY_MAX,
        shutter: float = 180.0,
        *,
        pivot: tuple[float, float] | None = None,
        fast: bool = False,
    ) -> str:
        """Transform with real motion blur (``Quality`` samples over ``shutter`` degrees,
        capped at MB_QUALITY_MAX, or MB_QUALITY_FAST for ``fast`` counter drums)."""
        inputs: dict[str, Any] = {
            "Input": Link(input),
            "Center": (_finite(center[0]), _finite(center[1])),
            "Size": _finite(size),
            "Angle": _finite(angle),
        }
        if pivot is not None:
            inputs["Pivot"] = (_finite(pivot[0]), _finite(pivot[1]))
        if motion_blur:
            cap = MB_QUALITY_FAST if fast else MB_QUALITY_MAX
            inputs.update(MotionBlur=1, Quality=min(int(quality), cap), ShutterAngle=_finite(shutter))
        self._add(name, "Transform", inputs)
        return name

    def merge(self, name: str, bg: str, fg: str, blend: float = 1.0, apply_mode: str = "Normal", *,
              center: tuple[float, float] | None = None, size: float | None = None) -> str:
        """Composite ``fg`` over ``bg``. ``center``/``size`` place and scale the foreground
        (a smaller image such as an icon keeps its pixel size at ``size`` 1)."""
        inputs: dict[str, Any] = {"Background": Link(bg), "Foreground": Link(fg),
                                  "PerformDepthMerge": 0}
        if center is not None:
            inputs["Center"] = (_finite(center[0]), _finite(center[1]))
        if size is not None:
            inputs["Size"] = _finite(size)
        if blend != 1.0:
            inputs["Blend"] = _finite(blend)
        if apply_mode != "Normal":
            inputs["ApplyMode"] = FuID(apply_mode)
        self._add(name, "Merge", inputs)
        return name

    def merge_all(self, name: str, layers: Iterable[str], base: str | None = None) -> str:
        """Stack ``layers`` bottom→top over ``base`` (or over a transparent background)."""
        layers = list(layers)
        if not layers:
            raise ValueError("merge_all needs at least one layer")
        if base is None:
            base = self.background(self._unique(f"{name}Leer"), alpha=0.0)
        current = base
        for i, layer in enumerate(layers):
            last = i == len(layers) - 1
            current = self.merge(name if last else self._unique(f"{name}{i + 1}"), current, layer)
        return current

    def background(self, name: str, color: Any = (0.0, 0.0, 0.0), alpha: float = 0.0) -> str:
        r, g, b, a = _rgba(color, alpha)
        # Fusion does not premultiply a Background's colour by its alpha (measured)
        self._add(name, "Background", {**self._frame_size(), "TopLeftRed": r * a, "TopLeftGreen": g * a,
                                       "TopLeftBlue": b * a, "TopLeftAlpha": a})
        return name

    def gradient(self, name: str, stops: Iterable[tuple[float, Any]],
                 start: tuple[float, float] = (0.0, 0.5), end: tuple[float, float] = (1.0, 0.5)) -> str:
        """Linear gradient background; ``stops`` = [(position 0..1, colour), …]."""
        colors: dict[float, tuple] = {}
        for pos, col in stops:
            if not 0.0 <= pos <= 1.0:
                raise ValueError(f"gradient stop {pos} outside 0..1")
            colors[float(pos) if pos not in (0, 1) else int(pos)] = _rgba(col)
        if len(colors) < 2:
            raise ValueError("a gradient needs at least two stops")
        self._add(name, "Background", {
            **self._frame_size(),
            "Type": FuID("Gradient"),
            "Start": (_finite(start[0]), _finite(start[1])),
            "End": (_finite(end[0]), _finite(end[1])),
            "Gradient": Ctor("Gradient", {"Colors": dict(sorted(colors.items()))}),
        })
        return name

    def _mask(self, regid: str, name: str, center, width, height, soft_edge, invert, combine_with,
              level, extra: dict) -> str:
        inputs: dict[str, Any] = {
            "MaskWidth": self.width, "MaskHeight": self.height,
            "ClippingMode": FuID("None"),
            "Center": (_finite(center[0]), _finite(center[1])),
            "Width": _finite(width), "Height": _finite(height),
            "SoftEdge": _finite(soft_edge),
            **extra,
        }
        if invert:
            inputs["Invert"] = 1
        if level != 1.0:
            inputs["Level"] = _finite(level)
        if combine_with is not None:
            inputs["EffectMask"] = Link(combine_with, "Mask")
        self._add(name, regid, inputs)
        return name

    def rect_mask(self, name: str, center: tuple[float, float], width: float, height: float,
                  soft_edge: float = 0.0, corner_radius: float = 0.0, angle: float = 0.0,
                  invert: bool = False, combine_with: str | None = None, level: float = 1.0) -> str:
        """Rectangle mask. ``width`` relative to image width, ``height`` to image height."""
        extra: dict[str, Any] = {}
        if corner_radius:
            extra["CornerRadius"] = _finite(corner_radius)
        if angle:
            extra["Angle"] = _finite(angle)
        return self._mask("RectangleMask", name, center, width, height, soft_edge, invert,
                          combine_with, level, extra)

    def ellipse_mask(self, name: str, center: tuple[float, float], width: float, height: float,
                     soft_edge: float = 0.0, invert: bool = False, combine_with: str | None = None,
                     level: float = 1.0) -> str:
        # Fusion measures an ellipse's Height in image widths too (measured); convert
        return self._mask("EllipseMask", name, center, width, _finite(height) * self.height / self.width,
                          soft_edge, invert, combine_with, level, {})

    def dissolve(self, name: str, bg: str, fg: str, mix: float = 0.0) -> str:
        """Dissolve between two inputs. Keyed 0/1 with ``step`` keys it acts as a switch:
        at Mix 0 or 1 Fusion only requests the visible input, so layers that are off cost
        (almost) nothing — the lyric comps use this to keep long sections fast."""
        self._add(name, "Dissolve", {"Background": Link(bg), "Foreground": Link(fg),
                                     "Mix": _finite(mix)})
        return name

    def blur(self, name: str, input: str, size: float) -> str:
        self._add(name, "Blur", {"Input": Link(input), "XBlurSize": _finite(size)})
        return name

    def fast_noise(self, name: str, scale: float = 40.0, detail: float = 4.0, contrast: float = 1.0,
                   brightness: float = 0.0, seethe_rate: float = 0.0,
                   color1: Any = (0.0, 0.0, 0.0, 1.0), color2: Any = (1.0, 1.0, 1.0, 1.0)) -> str:
        """FastNoise generator (regid as in Resolve's 'Noise Dissolve' template)."""
        r1, g1, b1, a1 = _rgba(color1)
        r2, g2, b2, a2 = _rgba(color2)
        self._add(name, "FastNoise", {
            **self._frame_size(), "Detail": _finite(detail), "Contrast": _finite(contrast),
            "Brightness": _finite(brightness), "XScale": _finite(scale),
            "SeetheRate": _finite(seethe_rate),
            "Color1Red": r1, "Color1Green": g1, "Color1Blue": b1, "Color1Alpha": a1,
            "Color2Red": r2, "Color2Green": g2, "Color2Blue": b2, "Color2Alpha": a2,
        })
        return name

    def apply_mask(self, tool: str, mask: str) -> None:
        self._get(tool).inputs["EffectMask"] = Link(mask, "Mask")

    def image(self, name: str, path: str | Path) -> str:
        """Loader for a still image (PNG/JPEG); the frame is held for the whole comp."""
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        fmt = {".png": "PNGFormat", ".jpg": "JpegFormat", ".jpeg": "JpegFormat"}.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(f"unsupported image type {path.suffix}")
        clip = Ctor("Clip", {
            "ID": "Clip1", "Filename": str(path), "FormatID": fmt, "StartFrame": -1,
            "LengthSetManually": True, "TrimIn": 0, "TrimOut": 0, "ExtendFirst": 0, "ExtendLast": 0,
            "Loop": 0, "AspectMode": 0, "Depth": 0, "TimeCode": 0, "GlobalStart": 0, "GlobalEnd": 0,
        })
        inputs = {f"Clip1.{fmt}.PostMultiply": 1} if fmt == "PNGFormat" else {}
        self._add(name, "Loader", inputs, Clips=(clip,))
        return name

    # ---------------------------------------------------------------- animation
    def keyframes(self, tool: str, inp: str, keys: Mapping[int, Any], ease: str = "out_cubic",
                  per_key: Mapping[int, str] | None = None) -> str:
        """Animate ``tool.inp`` through ``keys`` = {frame: value}. Numbers get a
        BezierSpline, (x, y) points an XYPath. ``ease`` shapes every segment (real Bezier
        handles); ``per_key`` overrides the ease of the segment that *starts* at a key."""
        target = self._get(tool)
        _check_ease(ease)
        per_key = dict(per_key or {})
        for e in per_key.values():
            _check_ease(e)
        if len(keys) < 2:
            raise ValueError(f"{tool}.{inp}: keyframes need at least two keys")
        for f in keys:
            if isinstance(f, bool) or not isinstance(f, int):
                raise ValueError(f"{tool}.{inp}: frame {f!r} is not an int")
        frames = sorted(keys)
        first = keys[frames[0]]
        if isinstance(first, (tuple, list)):
            base = self._unique(f"{tool}{inp}XY")
            xs = {f: _finite(keys[f][0]) for f in frames}
            ys = {f: _finite(keys[f][1]) for f in frames}
            xname = self._spline(f"{base}X", xs, ease, per_key)
            yname = self._spline(f"{base}Y", ys, ease, per_key)
            self._add(base, "XYPath", {"X": Link(xname, "Value"), "Y": Link(yname, "Value")},
                      ShowKeyPoints=False, DrawMode="ModifyOnly")
            target.inputs[inp] = Link(base, "Value")
            return base
        values = {f: _finite(keys[f]) for f in frames}
        name = self._spline(f"{tool}{inp}", values, ease, per_key)
        target.inputs[inp] = Link(name, "Value")
        return name

    def _spline(self, base: str, values: dict[int, float], ease: str,
                per_key: Mapping[int, str]) -> str:
        frames = sorted(values)
        # expand "step" segments into a flat segment plus a one-frame jump
        pts: list[tuple[int, float, str]] = []  # (frame, value, ease of segment starting here)
        for i, f in enumerate(frames):
            seg = per_key.get(f, ease)
            if i + 1 < len(frames) and EASES[seg] is None:
                nxt = frames[i + 1]
                pts.append((f, values[f], "linear"))
                if nxt - f > 1:
                    pts.append((nxt - 1, values[f], "linear"))
            else:
                pts.append((f, values[f], seg))
        kf: dict[int, dict] = {}
        for i, (f, v, seg) in enumerate(pts):
            entry: dict[Any, Any] = {1: v}
            lin_in = lin_out = True
            if i > 0:
                pf, pv, pseg = pts[i - 1]
                _, _, x2, y2 = EASES[pseg]  # type: ignore[misc]
                entry["LH"] = (pf + x2 * (f - pf), pv + y2 * (v - pv))
                lin_in = pseg == "linear"
            if i + 1 < len(pts):
                nf, nv, _ = pts[i + 1]
                x1, y1, _, _ = EASES[seg]  # type: ignore[misc]
                entry["RH"] = (f + x1 * (nf - f), v + y1 * (nv - v))
                lin_out = seg == "linear"
            if lin_in and lin_out:
                entry["Flags"] = {"Linear": True}
            kf[f] = entry
        name = self._unique(base)
        self._add(name, "BezierSpline", with_inputs=False, SplineColor={"Red": 228, "Green": 75, "Blue": 141},
                  NameSet=True, KeyFrames=kf)
        return name

    # ---------------------------------------------------------------- output
    def _validate(self) -> None:
        names = set(self._tools) | {"MediaOut1"}
        for tool in self._tools.values():
            for inp, v in tool.inputs.items():
                if isinstance(v, Link) and v.op not in names:
                    raise ValueError(f"{tool.name}.{inp} links to unknown tool {v.op!r}")
        if self._output is not None and self._output.op not in names:
            raise ValueError(f"MediaOut1 links to unknown tool {self._output.op!r}")

    def _layout(self) -> dict[str, tuple[float, float]]:
        """Readable node positions for Fusion's node view (left → right by depth)."""
        flow = {n: t for n, t in self._tools.items() if t.regid not in _MODIFIER_TYPES}
        depth: dict[str, int] = {}

        def d(n: str, seen: frozenset = frozenset()) -> int:
            if n in depth:
                return depth[n]
            if n in seen:
                return 0
            ups = [v.op for v in flow[n].inputs.values() if isinstance(v, Link) and v.op in flow]
            depth[n] = 0 if not ups else 1 + max(d(u, seen | {n}) for u in ups)
            return depth[n]

        rows: dict[int, int] = {}
        pos = {}
        for n in flow:
            col = d(n)
            pos[n] = (110.0 * col, 49.5 * rows.get(col, 0))
            rows[col] = rows.get(col, 0) + 1
        out_col = (max(depth.values()) + 1) if depth else 0
        pos["MediaOut1"] = (110.0 * out_col, 0.0)
        return pos

    def _spline_keys(self, inp: Any) -> list[dict[int, float]]:
        """Keyframe dicts behind an input link (an XYPath gives its X and Y splines)."""
        if not isinstance(inp, Link) or inp.op not in self._tools:
            return []
        src = self._tools[inp.op]
        if src.regid == "BezierSpline":
            return [{f: kf[1] for f, kf in src.fields["KeyFrames"].items()}]
        if src.regid == "XYPath":
            return [k for axis in ("X", "Y") for k in self._spline_keys(src.inputs.get(axis))]
        return []

    def _motion_windows(self, tool: _Tool) -> list[tuple[int, int]]:
        """Frame ranges in which ``tool`` (or its character follower) moves."""
        splines = []
        extend = 0.0
        for inp in ("Center", "Size", "Angle", "Pivot", "CharacterSpacing"):
            splines += self._spline_keys(tool.inputs.get(inp))
        st = tool.inputs.get("StyledText")
        if isinstance(st, Link) and st.op in self._tools:
            fol = self._tools[st.op]
            text = fol.inputs.get("Text")
            n = len(text.body["Value"]) if isinstance(text, Ctor) else 1
            extend = float(fol.inputs.get("Delay", 0.0)) * n
            for k, v in fol.inputs.items():
                if k.startswith("Character"):
                    splines += self._spline_keys(v)
        wins = []
        for keys in splines:
            frames = sorted(keys)
            for a, b in zip(frames, frames[1:]):
                if keys[a] != keys[b]:
                    wins.append((a - 1, int(math.ceil(b + extend)) + 1))
        wins.sort()
        merged: list[list[int]] = []
        for a, b in wins:
            if merged and a <= merged[-1][1] + 2:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(a, b) for a, b in merged]

    def _optimise_motion_blur(self) -> None:
        """No blur on tools that never move; Quality 1 outside their motion windows."""
        for tool in list(self._tools.values()):
            if tool.inputs.get("MotionBlur") != 1 or isinstance(tool.inputs.get("Quality"), Link):
                continue
            wins = self._motion_windows(tool)
            if not wins:
                tool.inputs["MotionBlur"] = 0
                continue
            q = tool.inputs["Quality"]
            keys: dict[int, float] = {}
            for a, b in wins:
                keys[a] = q
                keys[b] = 1
            if min(keys) > 0:
                keys[0] = 1
            self.keyframes(tool.name, "Quality", dict(sorted(keys.items())), ease="step")

    def to_text(self) -> str:
        if not getattr(self, "_mb_done", False):
            self._mb_done = True
            self._optimise_motion_blur()
        self._validate()
        last = self.frames - 1
        pos = self._layout()
        lines = [
            "Composition {",
            "\tCurrentTime = 0,",
            f"\tRenderRange = {{ 0, {last} }},",
            f"\tGlobalRange = {{ 0, {last} }},",
            "\tHiQ = true,",
            f"\tVersion = {_quote(RESOLVE_VERSION)},",
            "\tTools = {",
        ]
        blocks = []
        for tool in self._tools.values():
            body = []
            for k, v in tool.fields.items():
                body.append(f"\t\t\t{_key(k)} = {_lua(v, 3)},")
            if tool.with_inputs:
                body.append("\t\t\tInputs = {")
                for k, v in tool.inputs.items():
                    body.append(f"\t\t\t\t{_key(k)} = {_input(v, 4)},")
                body.append("\t\t\t},")
            if tool.name in pos:
                x, y = pos[tool.name]
                body.append(f"\t\t\tViewInfo = OperatorInfo {{ Pos = {{ {_num(x)}, {_num(y)} }} }},")
            blocks.append(f"\t\t{_key(tool.name)} = {tool.regid} {{\n" + "\n".join(body) + "\n\t\t},")
        mo = ["\t\tMediaOut1 = MediaOut {", "\t\t\tInputs = {", '\t\t\t\tIndex = Input { Value = "0", },']
        if self._output is not None:
            mo.append(f"\t\t\t\tInput = {_input(self._output, 4)},")
        x, y = pos["MediaOut1"]
        mo += ["\t\t\t},", f"\t\t\tViewInfo = OperatorInfo {{ Pos = {{ {_num(x)}, {_num(y)} }} }},", "\t\t},"]
        blocks.append("\n".join(mo))
        lines += blocks
        lines += [
            "\t},",
            "\tPrefs = {",
            "\t\tComp = {",
            "\t\t\tFrameFormat = {",
            f"\t\t\t\tWidth = {self.width},",
            f"\t\t\t\tHeight = {self.height},",
            f"\t\t\t\tRate = {self.fps},",
            "\t\t\t},",
            "\t\t},",
            "\t},",
            "}",
        ]
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_text(), encoding="utf-8")
        return path
