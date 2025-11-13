import importlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from color_spaces import (
    GamutResult,
    OklchColor,
    enforce_gamut,
    hex_to_oklch,
    hex_to_rgb,
    normalize_hue,
    oklch_to_hex,
    rgb_to_hex,
    toe,
    toe_inv,
)

pyperclip: Optional[Any]
try:
    pyperclip = importlib.import_module("pyperclip")
except ImportError:  # pragma: no cover - optional dependency
    pyperclip = None

TAILWIND_SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]
AUTO_DERIVATION_RATIO = 0.65


@dataclass
class PaletteParams:
    start_color: Optional[str] = None
    end_color: Optional[str] = None
    steepness: float = 1.0
    middle_color: Optional[str] = None
    middle_position: float = 0.5
    start_oklch: Optional[OklchColor] = None
    end_oklch: Optional[OklchColor] = None
    middle_oklch: Optional[OklchColor] = None
    min_lightness: float = 0.05
    max_lightness: float = 0.98
    min_chroma: float = 0.05
    max_chroma: float = 0.37
    gamut_space: str = "srgb"
    enforce_minimums: bool = False
    start_active: bool = False
    middle_active: bool = False
    end_active: bool = False


@dataclass
class PaletteExportOptions:
    key_prefix: str = ""
    line_terminator: str = ";"
    wrap_values_in_quotes: bool = False


@dataclass
class GeneratedPalette:
    colors: Dict[int, str]
    auto_labels: Dict[int, str]
    warnings: List[str] = field(default_factory=list)
    middle_shade: Optional[int] = None
    gamut_notes: Dict[int, str] = field(default_factory=dict)


class PaletteFormat(str, Enum):
    HEX = "hex"
    RGB = "rgb"
    OKLCH = "oklch"


PALETTE_FORMAT_LABELS = {
    PaletteFormat.HEX: "Hex (#RRGGBB)",
    PaletteFormat.RGB: "RGB (r, g, b)",
    PaletteFormat.OKLCH: "OKLCH (L, C, H)",
}


def format_hex_color(hex_color: str, palette_format: PaletteFormat) -> str:
    if palette_format == PaletteFormat.RGB:
        r, g, b = hex_to_rgb(hex_color)
        return f"rgb({r}, {g}, {b})"
    if palette_format == PaletteFormat.OKLCH:
        lightness, chroma, hue = hex_to_oklch(hex_color)
        return f"oklch({lightness:.4f}, {chroma:.4f}, {hue:.2f})"
    return hex_color


def clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def mix_toward(
    rgb: Tuple[int, int, int], target: Tuple[int, int, int], ratio: float
) -> Tuple[int, int, int]:
    r = clamp_channel(rgb[0] + (target[0] - rgb[0]) * ratio)
    g = clamp_channel(rgb[1] + (target[1] - rgb[1]) * ratio)
    b = clamp_channel(rgb[2] + (target[2] - rgb[2]) * ratio)
    return (r, g, b)


def derive_endpoint_from_middle(
    middle_color: str, known_color: Optional[str], target: str
) -> str:
    middle_rgb = hex_to_rgb(hex_color=middle_color)
    if known_color:
        known_rgb = hex_to_rgb(hex_color=known_color)
        r = clamp_channel(2 * middle_rgb[0] - known_rgb[0])
        g = clamp_channel(2 * middle_rgb[1] - known_rgb[1])
        b = clamp_channel(2 * middle_rgb[2] - known_rgb[2])
        return rgb_to_hex(rgb=(r, g, b))

    bound_rgb = (255, 255, 255) if target == "start" else (0, 0, 0)
    derived = mix_toward(rgb=middle_rgb, target=bound_rgb, ratio=AUTO_DERIVATION_RATIO)
    return rgb_to_hex(rgb=derived)


def linspace(start: float, stop: float, num: int) -> List[float]:
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def custom_sigmoid(x: float, steepness: float) -> float:
    return 1 / (1 + math.exp(-steepness * (x - 0.5)))


def normalize_sigmoid(x: float, steepness: float) -> float:
    sig_0 = custom_sigmoid(x=0, steepness=steepness)
    sig_1 = custom_sigmoid(x=1, steepness=steepness)
    return (custom_sigmoid(x=x, steepness=steepness) - sig_0) / (sig_1 - sig_0)


def _sync_oklch_editor_state(
    color_key: str,
    hex_value: str,
    *,
    force: bool = False,
) -> None:
    base_oklch = hex_to_oklch(hex_value)
    snapshot_key = f"{color_key}_hex_snapshot"
    st.session_state[snapshot_key] = hex_value

    l_state_key = f"{color_key}_oklch_l_state"
    c_state_key = f"{color_key}_oklch_c_state"
    h_state_key = f"{color_key}_oklch_h_state"
    l_snapshot_key = f"{color_key}_oklch_l_snapshot"
    c_snapshot_key = f"{color_key}_oklch_c_snapshot"
    h_snapshot_key = f"{color_key}_oklch_h_snapshot"

    if force or l_state_key not in st.session_state:
        st.session_state[l_state_key] = float(toe(base_oklch[0]))
    st.session_state[l_snapshot_key] = float(st.session_state[l_state_key])
    if force or c_state_key not in st.session_state:
        st.session_state[c_state_key] = float(base_oklch[1])
    st.session_state[c_snapshot_key] = float(st.session_state[c_state_key])
    if force or h_state_key not in st.session_state:
        st.session_state[h_state_key] = float(base_oklch[2])
    st.session_state[h_snapshot_key] = float(st.session_state[h_state_key])


def _sync_color_picker_widget(
    color_key: str, hex_value: str, *, force: bool = False
) -> None:
    picker_state_key = f"{color_key}_picker"
    if force or picker_state_key not in st.session_state:
        st.session_state[picker_state_key] = hex_value


def _request_streamlit_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()


def interpolate_oklch(
    color1: OklchColor, color2: OklchColor, factor: float, steepness: float
) -> OklchColor:
    adjusted = normalize_sigmoid(x=factor, steepness=steepness)
    lightness = color1[0] + adjusted * (color2[0] - color1[0])
    c = color1[1] + adjusted * (color2[1] - color1[1])
    delta_h = ((color2[2] - color1[2] + 180) % 360) - 180
    h = normalize_hue(color1[2] + adjusted * delta_h)
    return (lightness, c, h)


def apply_gamut_constraints(
    *,
    shade: int,
    oklch_color: OklchColor,
    params: PaletteParams,
    warnings: List[str],
    gamut_notes: Dict[int, str],
    record_warnings: bool,
) -> Tuple[str, OklchColor]:
    result: GamutResult = enforce_gamut(
        oklch_color,
        min_lightness=params.min_lightness,
        max_lightness=params.max_lightness,
        min_chroma=params.min_chroma,
        max_chroma=params.max_chroma,
        enforce_minimums=params.enforce_minimums,
    )

    deltas: List[str] = []
    if abs(result.lightness_after - result.lightness_before) > 1e-3:
        deltas.append(f"L {result.lightness_before:.2f}→{result.lightness_after:.2f}")
    if abs(result.chroma_after - result.chroma_before) > 1e-3:
        deltas.append(f"C {result.chroma_before:.2f}→{result.chroma_after:.2f}")
    if result.clipped:
        deltas.append("clipped to sRGB")

    if record_warnings and deltas:
        message = "; ".join(deltas)
        warnings.append(f"Shade {shade}: {message} to stay display-safe.")
        gamut_notes[shade] = message

    return result.hex_color, result.adjusted_oklch


def nearest_shade_index(position: float) -> Tuple[int, float]:
    if not TAILWIND_SHADES:
        raise ValueError("TAILWIND_SHADES cannot be empty")

    normalized_positions = [
        i / (len(TAILWIND_SHADES) - 1) for i in range(len(TAILWIND_SHADES))
    ]
    nearest = min(
        range(len(TAILWIND_SHADES)),
        key=lambda idx: abs(position - normalized_positions[idx]),
    )
    diff = abs(position - normalized_positions[nearest])
    return nearest, diff


def generate_palette(params: PaletteParams) -> GeneratedPalette:
    warnings: List[str] = []
    auto_labels: Dict[int, str] = {}
    gamut_notes: Dict[int, str] = {}

    start_color = params.start_color
    end_color = params.end_color
    middle_color = params.middle_color
    start_active = params.start_active
    end_active = params.end_active
    record_gamut_warnings = (
        params.start_active or params.middle_active or params.end_active
    )
    derived_start = False
    derived_end = False
    middle_shade: Optional[int] = None

    if middle_color:
        if not start_color and not end_color:
            start_color = derive_endpoint_from_middle(
                middle_color=middle_color, known_color=None, target="start"
            )
            end_color = derive_endpoint_from_middle(
                middle_color=middle_color, known_color=None, target="end"
            )
            derived_start = True
            derived_end = True
        elif not start_color:
            start_color = derive_endpoint_from_middle(
                middle_color=middle_color,
                known_color=params.end_color,
                target="start",
            )
            derived_start = True
        elif not end_color:
            end_color = derive_endpoint_from_middle(
                middle_color=middle_color,
                known_color=params.start_color,
                target="end",
            )
            derived_end = True
    else:
        if not start_color:
            start_color = "#ffffff"
            if start_active:
                warnings.append(
                    "Start color fell back to #ffffff because it was cleared."
                )
        if not end_color:
            end_color = "#000000"
            if end_active:
                warnings.append(
                    "End color fell back to #000000 because it was cleared."
                )

    if start_color is None or end_color is None:
        raise ValueError(
            "Start and end colors must be resolved before generating a palette."
        )

    if derived_start:
        auto_labels[TAILWIND_SHADES[0]] = "auto"
    if derived_end:
        auto_labels[TAILWIND_SHADES[-1]] = "auto"

    palette: Dict[int, str] = {}

    start_oklch = params.start_oklch or hex_to_oklch(start_color)
    end_oklch = params.end_oklch or hex_to_oklch(end_color)
    middle_oklch = None
    if middle_color:
        middle_oklch = params.middle_oklch or hex_to_oklch(middle_color)

    if middle_color and middle_oklch:
        idx, diff = nearest_shade_index(position=params.middle_position)
        middle_shade = TAILWIND_SHADES[idx]
        if diff > 1e-6:
            nearest_value = idx / (len(TAILWIND_SHADES) - 1)
            warnings.append(
                f"Middle position snapped to Tailwind shade {middle_shade} (normalized {nearest_value:.2f})."
            )

        lower_shades = [shade for shade in TAILWIND_SHADES if shade <= middle_shade]
        upper_shades = [shade for shade in TAILWIND_SHADES if shade >= middle_shade]

        lower_denominator = max(1, len(lower_shades) - 1)
        upper_denominator = max(1, len(upper_shades) - 1)

        for i, shade in enumerate(lower_shades):
            if shade == middle_shade:
                palette[shade] = middle_color
                continue
            factor = i / lower_denominator
            oklch_value = interpolate_oklch(
                color1=start_oklch,
                color2=middle_oklch,
                factor=factor,
                steepness=params.steepness,
            )
            hex_value, _ = apply_gamut_constraints(
                shade=shade,
                oklch_color=oklch_value,
                params=params,
                warnings=warnings,
                gamut_notes=gamut_notes,
                record_warnings=record_gamut_warnings,
            )
            palette[shade] = hex_value

        for i, shade in enumerate(upper_shades):
            if shade == middle_shade:
                if shade not in palette:
                    palette[shade] = middle_color
                continue
            factor = i / upper_denominator
            oklch_value = interpolate_oklch(
                color1=middle_oklch,
                color2=end_oklch,
                factor=factor,
                steepness=params.steepness,
            )
            hex_value, _ = apply_gamut_constraints(
                shade=shade,
                oklch_color=oklch_value,
                params=params,
                warnings=warnings,
                gamut_notes=gamut_notes,
                record_warnings=record_gamut_warnings,
            )
            palette[shade] = hex_value
    else:
        for i, shade in enumerate(TAILWIND_SHADES):
            factor = i / (len(TAILWIND_SHADES) - 1)
            oklch_value = interpolate_oklch(
                color1=start_oklch,
                color2=end_oklch,
                factor=factor,
                steepness=params.steepness,
            )
            hex_value, _ = apply_gamut_constraints(
                shade=shade,
                oklch_color=oklch_value,
                params=params,
                warnings=warnings,
                gamut_notes=gamut_notes,
                record_warnings=record_gamut_warnings,
            )
            palette[shade] = hex_value

    return GeneratedPalette(
        colors=palette,
        auto_labels=auto_labels,
        warnings=warnings,
        middle_shade=middle_shade,
        gamut_notes=gamut_notes,
    )


def format_palette_export(
    palette: Dict[int, str],
    palette_format: PaletteFormat,
    options: PaletteExportOptions,
) -> str:
    lines: List[str] = []
    for shade, color in palette.items():
        formatted_value = format_hex_color(color, palette_format)
        if options.wrap_values_in_quotes:
            value_str = f'"{formatted_value}"'
        else:
            value_str = formatted_value

        key_str = f"{options.key_prefix}{shade}"
        if " " in key_str or "-" in key_str:  # Basic check if key needs quotes
            key_str = f'"{key_str}"'

        lines.append(f"    {key_str}: {value_str}{options.line_terminator}")

    return "{\n" + "\n".join(lines) + "\n}"


def palette_to_typescript_color_array_str(
    palette: Dict[int, str], palette_format: PaletteFormat = PaletteFormat.HEX
) -> str:
    # Maintain backward compatibility for tests.
    return format_palette_export(
        palette,
        palette_format,
        PaletteExportOptions(line_terminator=",", wrap_values_in_quotes=True),
    )


def perceptual_color_editor(
    *,
    label: str,
    color_key: str,
    color_placeholder: Any,
) -> Tuple[str, Optional[OklchColor]]:
    pending_key = f"{color_key}_pending_hex"
    if pending_key in st.session_state:
        st.session_state[color_key] = st.session_state[pending_key]
        del st.session_state[pending_key]

    base_hex = st.session_state[color_key]
    snapshot_key = f"{color_key}_hex_snapshot"
    l_state_key = f"{color_key}_oklch_l_state"
    c_state_key = f"{color_key}_oklch_c_state"
    h_state_key = f"{color_key}_oklch_h_state"
    l_snapshot_key = f"{color_key}_oklch_l_snapshot"
    c_snapshot_key = f"{color_key}_oklch_c_snapshot"
    h_snapshot_key = f"{color_key}_oklch_h_snapshot"
    picker_key = f"{color_key}_picker"

    if l_state_key not in st.session_state:
        _sync_oklch_editor_state(color_key, base_hex)
    elif st.session_state.get(snapshot_key) != base_hex:
        _sync_oklch_editor_state(color_key, base_hex, force=True)

    slider_keys_ready = all(
        key in st.session_state for key in (l_state_key, c_state_key, h_state_key)
    )
    slider_snapshots_ready = all(
        key in st.session_state
        for key in (l_snapshot_key, c_snapshot_key, h_snapshot_key)
    )

    def _slider_dirty(state_key: str, snapshot_key: str) -> bool:
        if not slider_snapshots_ready:
            return False
        return not math.isclose(
            float(st.session_state[state_key]),
            float(st.session_state[snapshot_key]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )

    slider_changed = slider_keys_ready and (
        _slider_dirty(l_state_key, l_snapshot_key)
        or _slider_dirty(c_state_key, c_snapshot_key)
        or _slider_dirty(h_state_key, h_snapshot_key)
    )

    if slider_changed:
        slider_oklch = (
            max(0.0, min(1.0, toe_inv(float(st.session_state[l_state_key])))),
            float(st.session_state[c_state_key]),
            normalize_hue(float(st.session_state[h_state_key])),
        )
        slider_hex = oklch_to_hex(slider_oklch)
        st.session_state[l_snapshot_key] = float(st.session_state[l_state_key])
        st.session_state[c_snapshot_key] = float(st.session_state[c_state_key])
        st.session_state[h_snapshot_key] = float(st.session_state[h_state_key])
        if slider_hex != base_hex:
            base_hex = slider_hex
            st.session_state[color_key] = slider_hex
            st.session_state[snapshot_key] = slider_hex
            _sync_color_picker_widget(color_key, slider_hex, force=True)

    if picker_key not in st.session_state:
        st.session_state[picker_key] = base_hex

    picker_value = color_placeholder.color_picker(
        label="Color",
        key=picker_key,
        label_visibility="collapsed",
        value=st.session_state.get(picker_key, base_hex),
    )

    if picker_value != st.session_state.get(color_key):
        st.session_state[color_key] = picker_value
        base_hex = picker_value
        _sync_oklch_editor_state(color_key, base_hex, force=True)
        _request_streamlit_rerun()
    elif st.session_state.get(snapshot_key) != base_hex:
        _sync_oklch_editor_state(color_key, base_hex, force=True)

    base_oklch = hex_to_oklch(base_hex)

    l_display = st.slider(
        label="Lightness (perceptual)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get(l_state_key, toe(base_oklch[0]))),
        key=l_state_key,
    )
    chroma_value = st.slider(
        label="Chroma",
        min_value=0.0,
        max_value=0.45,
        step=0.005,
        value=float(st.session_state.get(c_state_key, base_oklch[1])),
        key=c_state_key,
    )
    hue_value = st.slider(
        label="Hue",
        min_value=0.0,
        max_value=360.0,
        step=1.0,
        value=float(st.session_state.get(h_state_key, base_oklch[2])),
        key=h_state_key,
    )

    actual_lightness = max(0.0, min(1.0, toe_inv(l_display)))
    oklch_tuple = (
        actual_lightness,
        chroma_value,
        normalize_hue(hue_value),
    )
    perceptual_hex = oklch_to_hex(oklch_tuple)

    if slider_changed and perceptual_hex != base_hex:
        st.session_state[color_key] = perceptual_hex
        st.session_state[snapshot_key] = perceptual_hex
        st.session_state[pending_key] = perceptual_hex
        _sync_color_picker_widget(color_key, perceptual_hex, force=True)
        base_hex = perceptual_hex
        _request_streamlit_rerun()

    return base_hex, oklch_tuple


def render_anchor(
    label: str, color_key: str, default_hex: str
) -> Tuple[Optional[str], Optional[OklchColor], bool]:
    if color_key not in st.session_state:
        st.session_state[color_key] = default_hex
    if f"{color_key}_active" not in st.session_state:
        st.session_state[f"{color_key}_active"] = False

    st.markdown(f"**{label}**")
    active = st.checkbox("Active", key=f"{color_key}_active")
    if not active:
        return None, None, False

    color_placeholder = st.empty()
    color_value, oklch = perceptual_color_editor(
        label=label,
        color_key=color_key,
        color_placeholder=color_placeholder,
    )
    return color_value, oklch, True


def palette_parameter_component() -> PaletteParams:
    st.header("Parameters")
    st.subheader("Anchors")
    start_color_value, start_oklch, start_active = render_anchor(
        "Start", "start_color", "#fafafa"
    )
    middle_color_value, middle_oklch, middle_active = render_anchor(
        "Middle", "middle_color", "#737373"
    )

    middle_position = 0.5

    end_color_value, end_oklch, end_active = render_anchor(
        "End", "end_color", "#0a0a0a"
    )

    st.divider()
    st.subheader("Interpolation Curve")

    steepness = st.slider(
        label="Steepness",
        min_value=1.0,
        max_value=16.0,
        value=st.session_state.get("steepness", 1.0),
        step=0.5,
        key="steepness",
    )

    x = linspace(start=0, stop=1, num=100)
    y = [normalize_sigmoid(x=i, steepness=steepness) for i in x]
    st.line_chart(data={"Curve": y})

    st.divider()
    st.subheader("Gamut Constraints")
    lightness_bounds = st.slider(
        label="Allowed Lightness",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("oklch_lightness_bounds", (0.05, 0.98)),
        step=0.01,
        key="oklch_lightness_bounds",
    )
    chroma_bounds = st.slider(
        label="Allowed Chroma",
        min_value=0.0,
        max_value=0.45,
        value=st.session_state.get("oklch_chroma_bounds", (0.05, 0.37)),
        step=0.01,
        key="oklch_chroma_bounds",
    )

    enforce_minimums = st.checkbox(
        label="Enforce safe-range floors",
        value=st.session_state.get("oklch_enforce_minimums", True),
        key="oklch_enforce_minimums",
        help="Raise shades that drift below the lightness/chroma bounds to avoid muddy ramps.",
    )

    min_lightness, max_lightness = lightness_bounds
    min_chroma, max_chroma = chroma_bounds

    return PaletteParams(
        start_color=start_color_value,
        end_color=end_color_value,
        steepness=steepness,
        middle_color=middle_color_value,
        middle_position=middle_position,
        start_oklch=start_oklch,
        end_oklch=end_oklch,
        middle_oklch=middle_oklch,
        min_lightness=min_lightness,
        max_lightness=max_lightness,
        min_chroma=min_chroma,
        max_chroma=max_chroma,
        enforce_minimums=enforce_minimums,
        start_active=start_active,
        middle_active=middle_active,
        end_active=end_active,
    )


def palette_component(generated_palette: GeneratedPalette) -> None:
    st.header("Palette")

    palette = generated_palette.colors

    signature = tuple(palette.items())
    if st.session_state.get("palette_signature") != signature:
        st.session_state["palette_signature"] = signature
        for shade, color in palette.items():
            st.session_state[f"color_{shade}"] = color

    with st.expander("Export Options", expanded=False):
        if "palette_format" not in st.session_state:
            st.session_state["palette_format"] = PaletteFormat.OKLCH
        if "export_line_terminator" not in st.session_state:
            st.session_state["export_line_terminator"] = ";"
        if "export_wrap_quotes" not in st.session_state:
            st.session_state["export_wrap_quotes"] = False
        if "export_key_prefix" not in st.session_state:
            st.session_state["export_key_prefix"] = ""

        key_prefix = st.text_input(
            "Key Prefix",
            key="export_key_prefix",
            help="Prepended before each shade number.",
        )
        line_terminator = st.text_input(
            "Line Terminator",
            key="export_line_terminator",
            help="Appended after each color (e.g., ;, ,, blank).",
        )
        wrap_values_in_quotes = st.checkbox(
            "Wrap Values In Quotes",
            key="export_wrap_quotes",
        )
        palette_format = st.selectbox(
            label="Format",
            options=list(PaletteFormat),
            format_func=lambda opt: PALETTE_FORMAT_LABELS[opt],
            key="palette_format",
        )

    cols = st.columns(len(palette))
    for (shade, color), col in zip(palette.items(), cols):
        col.color_picker(label=f"{shade}", key=f"color_{shade}")
        note = generated_palette.gamut_notes.get(shade)
        if note:
            col.caption("⚠️")

    export_options = PaletteExportOptions(
        key_prefix=key_prefix,
        line_terminator=line_terminator,
        wrap_values_in_quotes=wrap_values_in_quotes,
    )

    ts_color_array_str = format_palette_export(
        palette=palette, palette_format=palette_format, options=export_options
    )
    st.code(body=ts_color_array_str, language="typescript")

    if st.button(label="Copy"):
        if pyperclip:
            pyperclip.copy(ts_color_array_str)
            st.success("Copied to clipboard!")
        else:
            st.warning("pyperclip is not installed in this environment.")


def main() -> None:
    st.title("Tailwind Color Palette Generator")

    params = palette_parameter_component()

    palette: Optional[GeneratedPalette] = None
    with st.spinner("Generating palette..."):
        palette = generate_palette(params=params)

    if palette:
        palette_component(generated_palette=palette)


if __name__ == "__main__":
    main()
