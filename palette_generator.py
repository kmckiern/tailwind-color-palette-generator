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
    normalize_hue,
    oklch_to_hex,
    oklch_to_rgb,
    toe,
    toe_inv,
)
from defaults import (
    DEFAULT_CHROMA_BOUNDS,
    DEFAULT_END_COLOR_OKLCH,
    DEFAULT_ENFORCE_MINIMUMS,
    DEFAULT_EXPORT_FORMAT,
    DEFAULT_EXPORT_KEY_PREFIX,
    DEFAULT_EXPORT_LINE_TERMINATOR,
    DEFAULT_EXPORT_WRAP_QUOTES,
    DEFAULT_LIGHTNESS_BOUNDS,
    DEFAULT_MIDDLE_COLOR_OKLCH,
    DEFAULT_START_COLOR_OKLCH,
    DEFAULT_STEEPNESS,
    DEFAULT_UI_ENFORCE_MINIMUMS,
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
    start: Optional[OklchColor] = None
    end: Optional[OklchColor] = None
    steepness: float = DEFAULT_STEEPNESS
    middle: Optional[OklchColor] = None
    min_lightness: float = DEFAULT_LIGHTNESS_BOUNDS[0]
    max_lightness: float = DEFAULT_LIGHTNESS_BOUNDS[1]
    min_chroma: float = DEFAULT_CHROMA_BOUNDS[0]
    max_chroma: float = DEFAULT_CHROMA_BOUNDS[1]
    gamut_space: str = "srgb"
    enforce_minimums: bool = DEFAULT_ENFORCE_MINIMUMS
    start_active: bool = False
    middle_active: bool = False
    end_active: bool = False


@dataclass
class PaletteExportOptions:
    key_prefix: str = DEFAULT_EXPORT_KEY_PREFIX
    line_terminator: str = DEFAULT_EXPORT_LINE_TERMINATOR
    wrap_values_in_quotes: bool = DEFAULT_EXPORT_WRAP_QUOTES


@dataclass
class GeneratedPalette:
    colors: Dict[int, OklchColor]
    auto_labels: Dict[int, str]
    warnings: List[str] = field(default_factory=list)
    middle_shade: Optional[int] = None
    gamut_notes: Dict[int, str] = field(default_factory=dict)
    _hex_cache: Dict[int, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def hex_colors(self) -> Dict[int, str]:
        if not self._hex_cache:
            self._hex_cache = {
                shade: oklch_to_hex(color) for shade, color in self.colors.items()
            }
        return self._hex_cache


class PaletteFormat(str, Enum):
    HEX = "hex"
    RGB = "rgb"
    OKLCH = "oklch"


PALETTE_FORMAT_LABELS = {
    PaletteFormat.HEX: "Hex (#RRGGBB)",
    PaletteFormat.RGB: "RGB (r, g, b)",
    PaletteFormat.OKLCH: "OKLCH (L, C, H)",
}


def format_oklch_color(oklch_color: OklchColor, palette_format: PaletteFormat) -> str:
    if palette_format == PaletteFormat.RGB:
        r, g, b = oklch_to_rgb(oklch_color)
        return f"rgb({r}, {g}, {b})"
    if palette_format == PaletteFormat.OKLCH:
        lightness, chroma, hue = oklch_color
        return f"oklch({lightness:.4f}, {chroma:.4f}, {hue:.2f})"
    return oklch_to_hex(oklch_color)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def mirror_oklch(middle: OklchColor, known: OklchColor) -> OklchColor:
    delta_lightness = middle[0] - known[0]
    delta_chroma = middle[1] - known[1]
    hue_delta = ((middle[2] - known[2] + 180.0) % 360.0) - 180.0
    mirrored_lightness = _clamp01(middle[0] + delta_lightness)
    mirrored_chroma = max(0.0, middle[1] + delta_chroma)
    mirrored_hue = normalize_hue(middle[2] + hue_delta)
    return (mirrored_lightness, mirrored_chroma, mirrored_hue)


def mix_oklch_toward_extreme(
    color: OklchColor, *, target_lightness: float
) -> OklchColor:
    ratio = AUTO_DERIVATION_RATIO
    target = (target_lightness, 0.0, color[2])
    lightness = _clamp01(color[0] + (target[0] - color[0]) * ratio)
    chroma = max(0.0, color[1] + (target[1] - color[1]) * ratio)
    # Maintain hue when targeting neutrals (chroma = 0).
    hue = color[2]
    return (lightness, chroma, hue)


def derive_oklch_from_middle(
    *,
    middle: OklchColor,
    known: Optional[OklchColor],
    target: str,
) -> OklchColor:
    if known is not None:
        return mirror_oklch(middle, known)
    if target == "start":
        return mix_oklch_toward_extreme(middle, target_lightness=1.0)
    return mix_oklch_toward_extreme(middle, target_lightness=0.0)


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
    oklch: OklchColor,
    *,
    force: bool = False,
) -> None:
    snapshot_key = f"{color_key}_hex_snapshot"
    st.session_state[snapshot_key] = oklch_to_hex(oklch)

    l_state_key = f"{color_key}_oklch_l_state"
    c_state_key = f"{color_key}_oklch_c_state"
    h_state_key = f"{color_key}_oklch_h_state"
    l_snapshot_key = f"{color_key}_oklch_l_snapshot"
    c_snapshot_key = f"{color_key}_oklch_c_snapshot"
    h_snapshot_key = f"{color_key}_oklch_h_snapshot"

    if force or l_state_key not in st.session_state:
        st.session_state[l_state_key] = float(toe(oklch[0]))
    st.session_state[l_snapshot_key] = float(st.session_state[l_state_key])
    if force or c_state_key not in st.session_state:
        st.session_state[c_state_key] = float(oklch[1])
    st.session_state[c_snapshot_key] = float(st.session_state[c_state_key])
    if force or h_state_key not in st.session_state:
        st.session_state[h_state_key] = float(oklch[2])
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

    start_active = params.start_active
    middle_active = params.middle_active
    end_active = params.end_active
    start = params.start if start_active else None
    end = params.end if end_active else None
    middle = params.middle if middle_active else None
    record_gamut_warnings = start_active or middle_active or end_active
    derived_start = False
    derived_end = False
    middle_shade: Optional[int] = None

    if middle:
        if start is None and end is None:
            start = derive_oklch_from_middle(middle=middle, known=None, target="start")
            end = derive_oklch_from_middle(middle=middle, known=None, target="end")
            derived_start = True
            derived_end = True
        elif start is None:
            start = derive_oklch_from_middle(middle=middle, known=end, target="start")
            derived_start = True
        elif end is None:
            end = derive_oklch_from_middle(middle=middle, known=start, target="end")
            derived_end = True

    if start is None:
        start = DEFAULT_START_COLOR_OKLCH
        if start_active:
            warnings.append(
                f"Start color fell back to {oklch_to_hex(start)} because it was cleared."
            )
    if end is None:
        end = DEFAULT_END_COLOR_OKLCH
        if end_active:
            warnings.append(
                f"End color fell back to {oklch_to_hex(end)} because it was cleared."
            )

    if derived_start:
        auto_labels[TAILWIND_SHADES[0]] = "auto"
    if derived_end:
        auto_labels[TAILWIND_SHADES[-1]] = "auto"

    palette: Dict[int, OklchColor] = {}

    if middle:
        # Middle anchor always maps to shade 500
        middle_shade = 500

        lower_shades = [shade for shade in TAILWIND_SHADES if shade <= middle_shade]
        upper_shades = [shade for shade in TAILWIND_SHADES if shade >= middle_shade]

        lower_denominator = max(1, len(lower_shades) - 1)
        upper_denominator = max(1, len(upper_shades) - 1)

        for i, shade in enumerate(lower_shades):
            if shade == middle_shade:
                palette[shade] = middle
                continue
            factor = i / lower_denominator
            oklch_value = interpolate_oklch(
                color1=start,
                color2=middle,
                factor=factor,
                steepness=params.steepness,
            )
            _, adjusted_oklch = apply_gamut_constraints(
                shade=shade,
                oklch_color=oklch_value,
                params=params,
                warnings=warnings,
                gamut_notes=gamut_notes,
                record_warnings=record_gamut_warnings,
            )
            palette[shade] = adjusted_oklch

        for i, shade in enumerate(upper_shades):
            if shade == middle_shade:
                if shade not in palette:
                    palette[shade] = middle
                continue
            factor = i / upper_denominator
            oklch_value = interpolate_oklch(
                color1=middle,
                color2=end,
                factor=factor,
                steepness=params.steepness,
            )
            _, adjusted_oklch = apply_gamut_constraints(
                shade=shade,
                oklch_color=oklch_value,
                params=params,
                warnings=warnings,
                gamut_notes=gamut_notes,
                record_warnings=record_gamut_warnings,
            )
            palette[shade] = adjusted_oklch
    else:
        for i, shade in enumerate(TAILWIND_SHADES):
            factor = i / (len(TAILWIND_SHADES) - 1)
            oklch_value = interpolate_oklch(
                color1=start,
                color2=end,
                factor=factor,
                steepness=params.steepness,
            )
            _, adjusted_oklch = apply_gamut_constraints(
                shade=shade,
                oklch_color=oklch_value,
                params=params,
                warnings=warnings,
                gamut_notes=gamut_notes,
                record_warnings=record_gamut_warnings,
            )
            palette[shade] = adjusted_oklch

    return GeneratedPalette(
        colors=palette,
        auto_labels=auto_labels,
        warnings=warnings,
        middle_shade=middle_shade,
        gamut_notes=gamut_notes,
    )


def format_palette_export(
    palette: Dict[int, OklchColor],
    palette_format: PaletteFormat,
    options: PaletteExportOptions,
) -> str:
    lines: List[str] = []
    for shade, color in palette.items():
        formatted_value = format_oklch_color(color, palette_format)
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
    palette: Dict[int, OklchColor], palette_format: PaletteFormat = PaletteFormat.HEX
) -> str:
    # Maintain backward compatibility for tests.
    return format_palette_export(
        palette,
        palette_format,
        PaletteExportOptions(line_terminator=",", wrap_values_in_quotes=True),
    )


def _ensure_oklch_state(color_key: str, default: OklchColor) -> None:
    hex_key = f"{color_key}_hex"
    if hex_key not in st.session_state:
        st.session_state[hex_key] = oklch_to_hex(default)
        _sync_oklch_editor_state(color_key, default, force=True)


def perceptual_color_editor(
    *,
    label: str,
    color_key: str,
    color_placeholder: Any,
    default: OklchColor,
    container: Any = st,
) -> Optional[OklchColor]:
    _ensure_oklch_state(color_key, default)

    hex_key = f"{color_key}_hex"
    base_hex = st.session_state[hex_key]
    l_state_key = f"{color_key}_oklch_l_state"
    c_state_key = f"{color_key}_oklch_c_state"
    h_state_key = f"{color_key}_oklch_h_state"
    l_snapshot_key = f"{color_key}_oklch_l_snapshot"
    c_snapshot_key = f"{color_key}_oklch_c_snapshot"
    h_snapshot_key = f"{color_key}_oklch_h_snapshot"
    picker_key = f"{color_key}_picker"

    if l_state_key not in st.session_state:
        _sync_oklch_editor_state(color_key, hex_to_oklch(base_hex))

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
        slider_value = (
            max(0.0, min(1.0, toe_inv(float(st.session_state[l_state_key])))),
            float(st.session_state[c_state_key]),
            normalize_hue(float(st.session_state[h_state_key])),
        )
        slider_hex = oklch_to_hex(slider_value)
        st.session_state[l_snapshot_key] = float(st.session_state[l_state_key])
        st.session_state[c_snapshot_key] = float(st.session_state[c_state_key])
        st.session_state[h_snapshot_key] = float(st.session_state[h_state_key])
        if slider_hex != base_hex:
            base_hex = slider_hex
            st.session_state[hex_key] = slider_hex
            _sync_color_picker_widget(color_key, slider_hex, force=True)

    if picker_key not in st.session_state:
        st.session_state[picker_key] = base_hex

    picker_value = color_placeholder.color_picker(
        label="Color",
        key=picker_key,
        label_visibility="collapsed",
        value=st.session_state.get(picker_key, base_hex),
    )

    if picker_value != base_hex:
        base_hex = picker_value
        st.session_state[hex_key] = picker_value
        _sync_oklch_editor_state(color_key, hex_to_oklch(base_hex), force=True)
        _request_streamlit_rerun()

    base = hex_to_oklch(base_hex)

    l_display = container.slider(
        label="Lightness (perceptual)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get(l_state_key, toe(base[0]))),
        key=l_state_key,
    )
    chroma_value = container.slider(
        label="Chroma",
        min_value=0.0,
        max_value=0.45,
        step=0.005,
        value=float(st.session_state.get(c_state_key, base[1])),
        key=c_state_key,
    )
    hue_value = container.slider(
        label="Hue",
        min_value=0.0,
        max_value=360.0,
        step=1.0,
        value=float(st.session_state.get(h_state_key, base[2])),
        key=h_state_key,
    )

    actual_lightness = max(0.0, min(1.0, toe_inv(l_display)))
    oklch_tuple = (
        actual_lightness,
        chroma_value,
        normalize_hue(hue_value),
    )
    if slider_changed:
        new_hex = oklch_to_hex(oklch_tuple)
        if new_hex != base_hex:
            st.session_state[hex_key] = new_hex
            _sync_color_picker_widget(color_key, new_hex, force=True)
            _request_streamlit_rerun()

    return oklch_tuple


def render_anchor(
    label: str, color_key: str, default_oklch: OklchColor, container: Any
) -> Tuple[Optional[OklchColor], bool]:
    _ensure_oklch_state(color_key, default_oklch)
    if f"{color_key}_active" not in st.session_state:
        st.session_state[f"{color_key}_active"] = True

    container.markdown(f"**{label}**")
    active = container.checkbox("Active", key=f"{color_key}_active")
    if not active:
        return None, False

    color_placeholder = container.empty()
    oklch = perceptual_color_editor(
        label=label,
        color_key=color_key,
        color_placeholder=color_placeholder,
        default=default_oklch,
        container=container,
    )
    return oklch, True


def palette_parameter_component() -> PaletteParams:
    st.header("Parameters")

    st.subheader("Anchors")

    # Create 1x3 grid for anchors
    col_start, col_middle, col_end = st.columns(3)

    start, start_active = render_anchor(
        "Start", "start_color", DEFAULT_START_COLOR_OKLCH, col_start
    )
    middle, middle_active = render_anchor(
        "Middle", "middle_color", DEFAULT_MIDDLE_COLOR_OKLCH, col_middle
    )
    end, end_active = render_anchor(
        "End", "end_color", DEFAULT_END_COLOR_OKLCH, col_end
    )

    # Initialize expander states
    if "show_interpolation" not in st.session_state:
        st.session_state["show_interpolation"] = False
    if "show_gamut" not in st.session_state:
        st.session_state["show_gamut"] = False

    with st.expander(
        "Interpolation Curve", expanded=st.session_state["show_interpolation"]
    ):
        st.caption(
            "Adjust how colors transition between anchors—higher steepness creates more contrast in the middle shades."
        )

        steepness = st.slider(
            label="Steepness",
            min_value=1.0,
            max_value=16.0,
            value=st.session_state.get("steepness", DEFAULT_STEEPNESS),
            step=0.5,
            key="steepness",
        )

        x = linspace(start=0, stop=1, num=100)
        y = [normalize_sigmoid(x=i, steepness=steepness) for i in x]
        st.line_chart(data={"Curve": y})

        # Track expander state
        st.session_state["show_interpolation"] = True

    with st.expander("Gamut Constraints", expanded=st.session_state["show_gamut"]):
        st.caption(
            "Set display-safe bounds for lightness and chroma to prevent muddy or clipped colors."
        )

        lightness_bounds = st.slider(
            label="Allowed Lightness",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get(
                "oklch_lightness_bounds", DEFAULT_LIGHTNESS_BOUNDS
            ),
            step=0.01,
            key="oklch_lightness_bounds",
        )
        chroma_bounds = st.slider(
            label="Allowed Chroma",
            min_value=0.0,
            max_value=0.45,
            value=st.session_state.get("oklch_chroma_bounds", DEFAULT_CHROMA_BOUNDS),
            step=0.01,
            key="oklch_chroma_bounds",
        )

        enforce_minimums = st.checkbox(
            label="Enforce safe-range floors",
            value=st.session_state.get(
                "oklch_enforce_minimums", DEFAULT_UI_ENFORCE_MINIMUMS
            ),
            key="oklch_enforce_minimums",
            help="Raise shades that drift below the lightness/chroma bounds to avoid muddy ramps.",
        )

        # Track expander state
        st.session_state["show_gamut"] = True

    min_lightness, max_lightness = lightness_bounds
    min_chroma, max_chroma = chroma_bounds

    return PaletteParams(
        start=start if start_active else None,
        end=end if end_active else None,
        steepness=steepness,
        middle=middle if middle_active else None,
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

    palette_oklch = generated_palette.colors
    hex_palette = generated_palette.hex_colors()

    signature = tuple(hex_palette.items())
    if st.session_state.get("palette_signature") != signature:
        st.session_state["palette_signature"] = signature
        for shade, color in hex_palette.items():
            st.session_state[f"color_{shade}"] = color

    with st.expander("Export Options", expanded=False):
        if "palette_format" not in st.session_state:
            st.session_state["palette_format"] = PaletteFormat(DEFAULT_EXPORT_FORMAT)
        if "export_line_terminator" not in st.session_state:
            st.session_state["export_line_terminator"] = DEFAULT_EXPORT_LINE_TERMINATOR
        if "export_wrap_quotes" not in st.session_state:
            st.session_state["export_wrap_quotes"] = DEFAULT_EXPORT_WRAP_QUOTES
        if "export_key_prefix" not in st.session_state:
            st.session_state["export_key_prefix"] = DEFAULT_EXPORT_KEY_PREFIX

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

    cols = st.columns(len(hex_palette))
    for (shade, color), col in zip(hex_palette.items(), cols):
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
        palette=palette_oklch,
        palette_format=palette_format,
        options=export_options,
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
