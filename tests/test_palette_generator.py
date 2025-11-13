from color_spaces import enforce_gamut, hex_to_oklch, oklch_to_hex
from defaults import (
    DEFAULT_EXPORT_LINE_TERMINATOR,
    DEFAULT_EXPORT_WRAP_QUOTES,
)
from palette_generator import (
    AUTO_DERIVATION_RATIO,
    TAILWIND_SHADES,
    PaletteExportOptions,
    PaletteFormat,
    PaletteParams,
    clamp_channel,
    format_palette_export,
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


def test_typescript_export_rgb_format():
    params = PaletteParams(start_color="#000000", end_color="#ffffff", steepness=1.0)
    generated = generate_palette(params)
    ts_str = palette_to_typescript_color_array_str(
        generated.colors, palette_format=PaletteFormat.RGB
    )

    assert "rgb(0, 0, 0)" in ts_str
    assert "#" not in ts_str


def test_typescript_export_oklch_format():
    params = PaletteParams(start_color="#000000", end_color="#ffffff", steepness=1.0)
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


def test_oklch_interpolation_is_monotonic_in_lightness():
    params = PaletteParams(start_color="#111111", end_color="#f5f5f5", steepness=1.0)
    generated = generate_palette(params)
    lightness_values = [hex_to_oklch(color)[0] for color in generated.colors.values()]
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
    palette = {50: "#f8fafc"}
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
    palette = {50: "#f8fafc", 900: "#020617"}
    options = PaletteExportOptions(
        key_prefix="--color-", line_terminator=",", wrap_values_in_quotes=True
    )
    result = format_palette_export(palette, PaletteFormat.HEX, options)
    assert '"--color-50": "#f8fafc",' in result
    assert '"--color-900": "#020617",' in result
    assert result.strip().startswith("{")
    assert result.strip().endswith("}")


def test_format_palette_export_no_quotes_and_semicolon():
    palette = {50: "#f8fafc", 900: "#020617"}
    options = PaletteExportOptions(
        key_prefix="--color-", line_terminator=";", wrap_values_in_quotes=False
    )
    result = format_palette_export(palette, PaletteFormat.HEX, options)
    assert '"--color-50": #f8fafc;' in result
    assert '"--color-900": #020617;' in result
    assert '": "#' not in result  # No quotes around hex value


def test_format_palette_export_default_options_match_legacy():
    params = PaletteParams(start_color="#000000", end_color="#ffffff", steepness=1.0)
    generated = generate_palette(params)
    legacy_output = palette_to_typescript_color_array_str(generated.colors)

    # The new default is semicolon, so we override to comma for this test
    options = PaletteExportOptions(line_terminator=",", wrap_values_in_quotes=True)
    new_output = format_palette_export(generated.colors, PaletteFormat.HEX, options)
    assert legacy_output == new_output
