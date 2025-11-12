import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from palette_generator import (
    AUTO_DERIVATION_RATIO,
    TAILWIND_SHADES,
    PaletteParams,
    clamp_channel,
    generate_palette,
    hex_to_rgb,
    palette_to_typescript_color_array_str,
    rgb_to_hex,
)


def shade_position(shade: int) -> float:
    idx = TAILWIND_SHADES.index(shade)
    return idx / (len(TAILWIND_SHADES) - 1)


def test_generate_palette_two_point_defaults():
    params = PaletteParams(start_color="#111111", end_color="#eeeeee", steepness=1.0)
    generated = generate_palette(params)

    assert generated.middle_shade is None
    assert generated.colors[TAILWIND_SHADES[0]] == "#111111"
    assert generated.colors[TAILWIND_SHADES[-1]] == "#eeeeee"
    assert list(generated.colors.keys()) == TAILWIND_SHADES
    assert generated.auto_labels == {}


def test_generate_palette_three_point_explicit_middle():
    params = PaletteParams(
        start_color="#111111",
        end_color="#eeeeee",
        steepness=1.0,
        middle_color="#777777",
        middle_position=shade_position(500),
    )
    generated = generate_palette(params)

    assert generated.middle_shade == 500
    assert generated.colors[500] == "#777777"
    assert generated.colors[TAILWIND_SHADES[0]] == "#111111"
    assert generated.colors[TAILWIND_SHADES[-1]] == "#eeeeee"


def test_generate_palette_middle_with_start_derives_end():
    start = "#102030"
    middle = "#506070"
    params = PaletteParams(
        start_color=start,
        end_color=None,
        steepness=1.0,
        middle_color=middle,
        middle_position=shade_position(500),
    )
    generated = generate_palette(params)

    start_rgb = hex_to_rgb(start)
    middle_rgb = hex_to_rgb(middle)
    expected_end = rgb_to_hex(
        tuple(clamp_channel(2 * middle_rgb[i] - start_rgb[i]) for i in range(3))
    )

    assert generated.colors[TAILWIND_SHADES[-1]] == expected_end
    assert generated.auto_labels.get(TAILWIND_SHADES[-1]) == "auto"


def test_generate_palette_middle_only_derives_both_endpoints():
    middle = "#808080"
    params = PaletteParams(
        start_color=None,
        end_color=None,
        steepness=1.0,
        middle_color=middle,
        middle_position=shade_position(500),
    )
    generated = generate_palette(params)

    middle_rgb = hex_to_rgb(middle)
    expected_start = rgb_to_hex(
        tuple(
            clamp_channel(middle_rgb[i] + (255 - middle_rgb[i]) * AUTO_DERIVATION_RATIO)
            for i in range(3)
        )
    )
    expected_end = rgb_to_hex(
        tuple(
            clamp_channel(middle_rgb[i] + (0 - middle_rgb[i]) * AUTO_DERIVATION_RATIO)
            for i in range(3)
        )
    )

    assert generated.colors[TAILWIND_SHADES[0]] == expected_start
    assert generated.colors[TAILWIND_SHADES[-1]] == expected_end
    assert generated.auto_labels.get(TAILWIND_SHADES[0]) == "auto"
    assert generated.auto_labels.get(TAILWIND_SHADES[-1]) == "auto"


def test_typescript_export_preserves_shade_order():
    params = PaletteParams(start_color="#000000", end_color="#ffffff", steepness=1.0)
    generated = generate_palette(params)
    ts_str = palette_to_typescript_color_array_str(generated.colors)

    last_idx = -1
    for shade in TAILWIND_SHADES:
        marker = f" {shade}:"
        idx = ts_str.find(marker)
        assert idx != -1, f"Missing shade {shade} in export"
        assert idx > last_idx, "Shades are out of order in export"
        last_idx = idx

    assert ts_str.strip().startswith("{")
    assert ts_str.strip().endswith("}")
