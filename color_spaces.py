"""Color space utilities for OKLCH conversions and gamut enforcement."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

RgbTuple = Tuple[int, int, int]
OklchColor = Tuple[float, float, float]
SrgbFloat = Tuple[float, float, float]


# --- Basic RGB helpers -----------------------------------------------------

def hex_to_rgb(hex_color: str) -> RgbTuple:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


def rgb_to_hex(rgb: RgbTuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _srgb_byte_to_float(value: int) -> float:
    return max(0.0, min(1.0, value / 255.0))


def _srgb_float_to_byte(value: float) -> int:
    return max(0, min(255, int(round(value * 255.0))))


def _srgb_to_linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(channel: float) -> float:
    if channel <= 0.0031308:
        return channel * 12.92
    return 1.055 * (channel ** (1 / 2.4)) - 0.055


# --- OKLab/OKLCH conversions -----------------------------------------------

def _srgb_to_oklab(rgb: SrgbFloat) -> Tuple[float, float, float]:
    r_l, g_l, b_l = (_srgb_to_linear(c) for c in rgb)

    l = 0.4122214708 * r_l + 0.5363325363 * g_l + 0.0514459929 * b_l
    m = 0.2119034982 * r_l + 0.6806995451 * g_l + 0.1073969566 * b_l
    s = 0.0883024619 * r_l + 0.2817188376 * g_l + 0.6299787005 * b_l

    l_ = math.copysign(abs(l) ** (1 / 3), l)
    m_ = math.copysign(abs(m) ** (1 / 3), m)
    s_ = math.copysign(abs(s) ** (1 / 3), s)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return (L, a, b)


def _oklab_to_srgb(oklab: Tuple[float, float, float]) -> SrgbFloat:
    L, a, b = oklab
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    r_l = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_l = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_l = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    r = _linear_to_srgb(r_l)
    g = _linear_to_srgb(g_l)
    b = _linear_to_srgb(b_l)
    return (r, g, b)


def rgb_to_oklch(rgb: RgbTuple) -> OklchColor:
    srgb = tuple(_srgb_byte_to_float(c) for c in rgb)
    L, a, b = _srgb_to_oklab(srgb)
    C = math.sqrt(a * a + b * b)
    h = math.degrees(math.atan2(b, a)) % 360
    return (L, C, h)


def oklch_to_rgb(oklch: OklchColor) -> RgbTuple:
    L, C, h = oklch
    a = C * math.cos(math.radians(h))
    b = C * math.sin(math.radians(h))
    srgb = _oklab_to_srgb((L, a, b))
    return tuple(_srgb_float_to_byte(c) for c in srgb)


def hex_to_oklch(hex_color: str) -> OklchColor:
    return rgb_to_oklch(hex_to_rgb(hex_color))


def oklch_to_hex(oklch: OklchColor) -> str:
    return rgb_to_hex(oklch_to_rgb(oklch))


def normalize_hue(hue: float) -> float:
    return hue % 360.0


# --- Toe remap helpers -----------------------------------------------------

K1 = 0.206
K2 = 0.03
K3 = (1 + K1) / (1 + K2)


def toe(x: float) -> float:
    return 0.5 * (K3 * x - K1 + math.sqrt((K3 * x - K1) ** 2 + 4 * K2 * K3 * x))


def toe_inv(x: float) -> float:
    return (x * x + K1 * x) / (K3 * (x + K2))


# --- Gamut enforcement -----------------------------------------------------


def _is_in_gamut(srgb: SrgbFloat) -> bool:
    return all(0.0 <= channel <= 1.0 for channel in srgb)


@dataclass
class GamutResult:
    hex_color: str
    adjusted_oklch: OklchColor
    clipped: bool
    lightness_before: float
    lightness_after: float
    chroma_before: float
    chroma_after: float


def _oklch_to_srgb_float(oklch: OklchColor) -> SrgbFloat:
    L, C, h = oklch
    a = C * math.cos(math.radians(h))
    b = C * math.sin(math.radians(h))
    return _oklab_to_srgb((L, a, b))


def enforce_gamut(
    oklch_color: OklchColor,
    *,
    min_lightness: float,
    max_lightness: float,
    min_chroma: float,
    max_chroma: float,
    enforce_minimums: bool = False,
    chroma_epsilon: float = 1e-4,
    max_iterations: int = 20,
) -> GamutResult:
    l_raw, c_raw, h_raw = oklch_color
    h = normalize_hue(h_raw)
    lightness_clamped = min(l_raw, max_lightness)
    if enforce_minimums and lightness_clamped < min_lightness:
        lightness_clamped = min_lightness

    chroma_target = min(c_raw, max_chroma)
    chroma_target = max(chroma_target, 0.0)
    if enforce_minimums and chroma_target < min_chroma:
        chroma_target = min_chroma

    candidate: OklchColor = (lightness_clamped, chroma_target, h)
    srgb = _oklch_to_srgb_float(candidate)
    chroma_solution = chroma_target
    clipped = False

    if not _is_in_gamut(srgb):
        clipped = True
        low = 0.0
        high = chroma_target
        srgb_in_gamut = (_srgb_float_to_byte(0) / 255.0,) * 3  # placeholder
        for _ in range(max_iterations):
            mid = (low + high) / 2
            srgb_candidate = _oklch_to_srgb_float((lightness_clamped, mid, h))
            if _is_in_gamut(srgb_candidate):
                low = mid
                srgb_in_gamut = srgb_candidate
            else:
                high = mid
            if high - low < chroma_epsilon:
                break
        chroma_solution = low
        candidate = (lightness_clamped, chroma_solution, h)
        srgb = srgb_in_gamut

    srgb_clamped = tuple(min(1.0, max(0.0, c)) for c in srgb)
    rgb_bytes = tuple(_srgb_float_to_byte(c) for c in srgb_clamped)

    return GamutResult(
        hex_color=rgb_to_hex(rgb_bytes),
        adjusted_oklch=candidate,
        clipped=clipped or srgb != srgb_clamped,
        lightness_before=l_raw,
        lightness_after=lightness_clamped,
        chroma_before=c_raw,
        chroma_after=chroma_solution,
    )
