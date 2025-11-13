"""Centralized application default values for easier review and tweaks."""

from typing import Tuple

from color_spaces import OklchColor, oklch_to_hex

# Anchor defaults
DEFAULT_START_COLOR_OKLCH: OklchColor = (0.985, 0, 0)
DEFAULT_MIDDLE_COLOR_OKLCH: OklchColor = (0.556, 0, 0)
DEFAULT_END_COLOR_OKLCH: OklchColor = (0.145, 0, 0)
DEFAULT_MIDDLE_POSITION = 0.5

# Interpolation / curve defaults
DEFAULT_STEEPNESS = 1.0

# Gamut constraint defaults
DEFAULT_LIGHTNESS_BOUNDS: Tuple[float, float] = (0.05, 0.98)
DEFAULT_CHROMA_BOUNDS: Tuple[float, float] = (0.05, 0.37)
DEFAULT_ENFORCE_MINIMUMS = False
DEFAULT_UI_ENFORCE_MINIMUMS = False

# Export defaults
DEFAULT_EXPORT_KEY_PREFIX = ""
DEFAULT_EXPORT_LINE_TERMINATOR = ";"
DEFAULT_EXPORT_WRAP_QUOTES = False
# Stored as the PaletteFormat enum value name for cycle-free import.
DEFAULT_EXPORT_FORMAT = "oklch"
