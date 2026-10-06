"""Look of the lyric video: palette, fonts, formats, per-section colour worlds."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .comp_writer import FontMetrics

FORMATS: dict[str, tuple[int, int]] = {"16x9": (1920, 1080), "9x16": (1080, 1920)}
FPS = 60

INK = "#07060b"          # near-black ground
PINK = "#ff2d75"         # hot pink  -> B world
CYAN = "#00e5ff"         # cyan      -> A world
VIOLET = "#8a2be2"       # violet    -> shared / breakdown
ACID = "#e6ff00"         # acid yellow accent for key words
WHITE = "#ffffff"
PALE_PINK = "#ffd6e5"
PALE_CYAN = "#d6fbff"

# fonts (family, style) — Anton for hooks and slams, Bebas Neue for refrains, Space Grotesk for rap
HOOK = ("Anton", "Regular")
REFRAIN = ("Bebas Neue", "Regular")
RAP = ("Space Grotesk", "Bold")

# colour world per singer: (main, secondary)
SINGER_COLORS: dict[str, tuple[str, str]] = {
    "B-Refrain": (PALE_PINK, PINK),
    "B-Rap": (WHITE, PINK),
    "A-Hook": (PALE_CYAN, CYAN),
    "A-Refrain": (PALE_CYAN, CYAN),
}

# background neon field weights per section: (pink, cyan, violet)
SECTION_FIELDS: dict[str, tuple[float, float, float]] = {
    "Intro": (0.15, 0.55, 0.45),
    "Drop 1": (0.55, 0.65, 0.55),
    "B Refrain 1": (0.65, 0.10, 0.50),
    "B Strophe 1": (0.55, 0.05, 0.45),
    "B Refrain 2": (0.65, 0.10, 0.50),
    "Post-Chorus": (0.15, 0.75, 0.40),
    "Breakdown": (0.10, 0.35, 0.60),
    "Finaler Drop": (0.70, 0.70, 0.55),
    "Outro": (0.20, 0.45, 0.40),
}


@dataclass(frozen=True)
class Fmt:
    name: str
    width: int
    height: int

    @property
    def vertical(self) -> bool:
        return self.height > self.width

    @property
    def safe(self) -> tuple[int, int, int, int]:
        """left, top, right, bottom of the text area in px (9:16 keeps clear of UI overlays)."""
        if self.vertical:
            return 80, 230, self.width - 80, self.height - 420
        return 110, 90, self.width - 110, self.height - 90

    @property
    def safe_w(self) -> int:
        left, _, right, _ = self.safe
        return right - left

    @property
    def cx(self) -> float:
        left, _, right, _ = self.safe
        return (left + right) / 2

    @property
    def cy(self) -> float:
        _, top, _, bottom = self.safe
        return (top + bottom) / 2

    def u(self, px_16x9: float) -> float:
        """Scale a size designed for 1920x1080 to this format (9:16 uses the narrower width)."""
        return px_16x9 * (self.width / 1920 if not self.vertical else 1080 / 1920 * 1.55)


def fmt(name: str) -> Fmt:
    w, h = FORMATS[name]
    return Fmt(name, w, h)


@lru_cache(maxsize=None)
def metrics(family: str, style: str) -> FontMetrics:
    return FontMetrics.find(family, style)


def size_for_cap(font: tuple[str, str], cap_px: float, width: int) -> float:
    return cap_px / (metrics(*font).cap_per_size() * width)


def text_width(font: tuple[str, str], text: str, cap_px: float, width: int) -> float:
    size = size_for_cap(font, cap_px, width)
    return metrics(*font).width_px(text, size, width)
