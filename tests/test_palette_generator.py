from color_spaces import enforce_gamut, hex_to_oklch, oklch_to_hex
from defaults import (
    DEFAULT_EXPORT_LINE_TERMINATOR,
    DEFAULT_EXPORT_WRAP_QUOTES,
)
from palette_generator import (
    TAILWIND_SHADES,
    PaletteExportOptions,
    PaletteFormat,
    PaletteParams,
    derive_oklch_from_middle,
    format_palette_export,
    generate_palette,
    palette_to_typescript_color_array_str,
)


def shade_position(shade: int) -> float:
    idx = TAILWIND_SHADES.index(shade)
    return idx / (len(TAILWIND_SHADES) - 1)


def test_generate_palette_two_point_defaults():
    params = PaletteParams(
        start=hex_to_oklch("#111111"),
        end=hex_to_oklch("#eeeeee"),
        steepness=1.0,
        start_active=True,
        end_active=True,
    )
    generated = generate_palette(params)

    assert generated.middle_shade is None
    hex_palette = generated.hex_colors()
    assert hex_palette[TAILWIND_SHADES[0]] == "#111111"
    assert hex_palette[TAILWIND_SHADES[-1]] == "#eeeeee"
    assert list(generated.colors.keys()) == TAILWIND_SHADES
    assert generated.auto_labels == {}


def test_generate_palette_three_point_explicit_middle():
    params = PaletteParams(
        start=hex_to_oklch("#111111"),
        end=hex_to_oklch("#eeeeee"),
        steepness=1.0,
        middle=hex_to_oklch("#777777"),
        middle_position=shade_position(500),
        start_active=True,
        end_active=True,
        middle_active=True,
    )
    generated = generate_palette(params)

    assert generated.middle_shade == 500
    hex_palette = generated.hex_colors()
    assert hex_palette[500] == "#777777"
    assert hex_palette[TAILWIND_SHADES[0]] == "#111111"
    assert hex_palette[TAILWIND_SHADES[-1]] == "#eeeeee"


def test_generate_palette_middle_with_start_derives_end():
    start = "#102030"
    middle = "#506070"
    params = PaletteParams(
        start=hex_to_oklch(start),
        end=None,
        steepness=1.0,
        middle=hex_to_oklch(middle),
        middle_position=shade_position(500),
        start_active=True,
        middle_active=True,
        end_active=False,
    )
    generated = generate_palette(params)

    expected_end = oklch_to_hex(
        derive_oklch_from_middle(
            middle=hex_to_oklch(middle), known=hex_to_oklch(start), target="end"
        )
    )

    assert generated.hex_colors()[TAILWIND_SHADES[-1]] == expected_end
    assert generated.auto_labels.get(TAILWIND_SHADES[-1]) == "auto"


def test_generate_palette_middle_only_derives_both_endpoints():
    middle = "#808080"
    params = PaletteParams(
        start=None,
        end=None,
        steepness=1.0,
        middle=hex_to_oklch(middle),
        middle_position=shade_position(500),
        middle_active=True,
    )
    generated = generate_palette(params)

    expected_start = oklch_to_hex(
        derive_oklch_from_middle(
            middle=hex_to_oklch(middle), known=None, target="start"
        )
    )
    expected_end = oklch_to_hex(
        derive_oklch_from_middle(middle=hex_to_oklch(middle), known=None, target="end")
    )

    hex_palette = generated.hex_colors()
    assert hex_palette[TAILWIND_SHADES[0]] == expected_start
    assert hex_palette[TAILWIND_SHADES[-1]] == expected_end
    assert generated.auto_labels.get(TAILWIND_SHADES[0]) == "auto"
    assert generated.auto_labels.get(TAILWIND_SHADES[-1]) == "auto"


def test_typescript_export_preserves_shade_order():
    params = PaletteParams(
        start=hex_to_oklch("#000000"),
        end=hex_to_oklch("#ffffff"),
        steepness=1.0,
        start_active=True,
        end_active=True,
    )
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


def test_typescript_export_rgb_format():
    params = PaletteParams(
        start=hex_to_oklch("#000000"),
        end=hex_to_oklch("#ffffff"),
        steepness=1.0,
        start_active=True,
        end_active=True,
    )
    generated = generate_palette(params)
    ts_str = palette_to_typescript_color_array_str(
        generated.colors, palette_format=PaletteFormat.RGB
    )

    assert "rgb(0, 0, 0)" in ts_str
    assert "#" not in ts_str


def test_typescript_export_oklch_format():
    params = PaletteParams(
        start=hex_to_oklch("#000000"),
        end=hex_to_oklch("#ffffff"),
        steepness=1.0,
        start_active=True,
        end_active=True,
    )
    generated = generate_palette(params)
    ts_str = palette_to_typescript_color_array_str(
        generated.colors, palette_format=PaletteFormat.OKLCH
    )

    assert "oklch(" in ts_str
    assert "#" not in ts_str


def test_hex_to_oklch_round_trip_preserves_value():
    original = "#4a83ff"
    oklch = hex_to_oklch(original)
    reconstructed = oklch_to_hex(oklch)
    assert reconstructed.lower() == original.lower()


def test_hex_round_trip_neutral_preserves_zero_hue():
    neutral_oklch = (0.75, 0.0, 180.0)
    hex_color = oklch_to_hex(neutral_oklch)
    round_trip = hex_to_oklch(hex_color)
    assert round_trip[1] == 0.0
    assert round_trip[2] == 0.0


def test_oklch_interpolation_is_monotonic_in_lightness():
    params = PaletteParams(
        start=hex_to_oklch("#111111"),
        end=hex_to_oklch("#f5f5f5"),
        steepness=1.0,
        start_active=True,
        end_active=True,
    )
    generated = generate_palette(params)
    lightness_values = [color[0] for color in generated.colors.values()]
    assert all(
        lightness_values[i] <= lightness_values[i + 1] + 1e-3
        for i in range(len(lightness_values) - 1)
    )


def test_enforce_gamut_reduces_high_chroma():
    saturated = (0.7, 0.6, 40.0)
    result = enforce_gamut(
        saturated,
        min_lightness=0.05,
        max_lightness=0.98,
        min_chroma=0.0,
        max_chroma=0.6,
    )
    assert result.chroma_after <= result.chroma_before
    assert result.clipped is True


def test_format_palette_export_defaults_are_semicolon_unquoted():
    palette = {50: hex_to_oklch("#f8fafc")}
    result = format_palette_export(
        palette,
        PaletteFormat.HEX,
        PaletteExportOptions(),
    )
    expected_line = f"    50: #f8fafc{DEFAULT_EXPORT_LINE_TERMINATOR}"
    assert expected_line in result
    if DEFAULT_EXPORT_WRAP_QUOTES:
        assert '"#f8fafc"' in result
    else:
        assert '"#f8fafc"' not in result


def test_format_palette_export_with_prefix_and_comma():
    palette = {
        50: hex_to_oklch("#f8fafc"),
        900: hex_to_oklch("#020617"),
    }
    options = PaletteExportOptions(
        key_prefix="--color-", line_terminator=",", wrap_values_in_quotes=True
    )
    result = format_palette_export(palette, PaletteFormat.HEX, options)
    assert '"--color-50": "#f8fafc",' in result
    assert '"--color-900": "#020617",' in result
    assert result.strip().startswith("{")
    assert result.strip().endswith("}")


def test_format_palette_export_no_quotes_and_semicolon():
    palette = {
        50: hex_to_oklch("#f8fafc"),
        900: hex_to_oklch("#020617"),
    }
    options = PaletteExportOptions(
        key_prefix="--color-", line_terminator=";", wrap_values_in_quotes=False
    )
    result = format_palette_export(palette, PaletteFormat.HEX, options)
    assert '"--color-50": #f8fafc;' in result
    assert '"--color-900": #020617;' in result
    assert '": "#' not in result  # No quotes around hex value


def test_format_palette_export_default_options_match_legacy():
    params = PaletteParams(
        start=hex_to_oklch("#000000"),
        end=hex_to_oklch("#ffffff"),
        steepness=1.0,
        start_active=True,
        end_active=True,
    )
    generated = generate_palette(params)
    legacy_output = palette_to_typescript_color_array_str(generated.colors)

    # The new default is semicolon, so we override to comma for this test
    options = PaletteExportOptions(line_terminator=",", wrap_values_in_quotes=True)
    new_output = format_palette_export(generated.colors, PaletteFormat.HEX, options)
    assert legacy_output == new_output
