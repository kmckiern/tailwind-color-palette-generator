import importlib
import math
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

pyperclip: Optional[Any]
try:
    pyperclip = importlib.import_module("pyperclip")
except ImportError:  # pragma: no cover - optional dependency
    pyperclip = None

import streamlit as st

TAILWIND_SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]
AUTO_DERIVATION_RATIO = 0.65


class PaletteParams(NamedTuple):
    start_color: Optional[str]
    end_color: Optional[str]
    steepness: float
    middle_color: Optional[str] = None
    middle_position: float = 0.5


class GeneratedPalette(NamedTuple):
    colors: Dict[int, str]
    auto_labels: Dict[int, str]
    warnings: List[str]
    middle_shade: Optional[int]


class PaletteFormat(str, Enum):
    HEX = "hex"
    RGB = "rgb"


PALETTE_FORMAT_LABELS = {
    PaletteFormat.HEX: "Hex (#RRGGBB)",
    PaletteFormat.RGB: "RGB (r, g, b)",
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def format_hex_color(hex_color: str, palette_format: PaletteFormat) -> str:
    if palette_format == PaletteFormat.RGB:
        r, g, b = hex_to_rgb(hex_color)
        return f"rgb({r}, {g}, {b})"
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


def interpolate_color(color1: str, color2: str, factor: float, steepness: float) -> str:
    rgb1 = hex_to_rgb(hex_color=color1)
    rgb2 = hex_to_rgb(hex_color=color2)

    adjusted_factor = normalize_sigmoid(x=factor, steepness=steepness)

    r = int(rgb1[0] + adjusted_factor * (rgb2[0] - rgb1[0]))
    g = int(rgb1[1] + adjusted_factor * (rgb2[1] - rgb1[1]))
    b = int(rgb1[2] + adjusted_factor * (rgb2[2] - rgb1[2]))

    return rgb_to_hex((r, g, b))


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

    start_color = params.start_color
    end_color = params.end_color
    middle_color = params.middle_color
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
            warnings.append("Start color fell back to #ffffff because it was cleared.")
        if not end_color:
            end_color = "#000000"
            warnings.append("End color fell back to #000000 because it was cleared.")

    if start_color is None or end_color is None:
        raise ValueError(
            "Start and end colors must be resolved before generating a palette."
        )

    if derived_start:
        auto_labels[TAILWIND_SHADES[0]] = "auto"
    if derived_end:
        auto_labels[TAILWIND_SHADES[-1]] = "auto"

    palette: Dict[int, str] = {}

    if middle_color:
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
            palette[shade] = interpolate_color(
                color1=start_color,
                color2=middle_color,
                factor=factor,
                steepness=params.steepness,
            )

        for i, shade in enumerate(upper_shades):
            if shade == middle_shade:
                if shade not in palette:
                    palette[shade] = middle_color
                continue
            factor = i / upper_denominator
            palette[shade] = interpolate_color(
                color1=middle_color,
                color2=end_color,
                factor=factor,
                steepness=params.steepness,
            )
    else:
        for i, shade in enumerate(TAILWIND_SHADES):
            factor = i / (len(TAILWIND_SHADES) - 1)
            palette[shade] = interpolate_color(
                color1=start_color,
                color2=end_color,
                factor=factor,
                steepness=params.steepness,
            )

    return GeneratedPalette(
        colors=palette,
        auto_labels=auto_labels,
        warnings=warnings,
        middle_shade=middle_shade,
    )


def palette_to_typescript_color_array_str(
    palette: Dict[int, str], palette_format: PaletteFormat = PaletteFormat.HEX
) -> str:
    ts_color_array_str = "{\n"
    for shade, color in palette.items():
        formatted = format_hex_color(color, palette_format)
        ts_color_array_str += f'    {shade}: "{formatted}",\n'
    ts_color_array_str += "}"
    return ts_color_array_str


def palette_parameter_component() -> PaletteParams:
    st.subheader("Parameters")

    if "start_color" not in st.session_state:
        st.session_state.start_color = "#FFFFFF"
    if "middle_color" not in st.session_state:
        st.session_state.middle_color = "#9333EA"
    if "end_color" not in st.session_state:
        st.session_state.end_color = "#000000"
    if "specify_edges" not in st.session_state:
        st.session_state.specify_edges = True
    if "specify_middle" not in st.session_state:
        st.session_state.specify_middle = True

    def swap_colors() -> None:
        st.session_state.start_color, st.session_state.end_color = (
            st.session_state.end_color,
            st.session_state.start_color,
        )

    toggle_start_end, toggle_middle = st.columns(2)
    specify_edges = toggle_start_end.checkbox(
        label="Specify start and end colors",
        key="specify_edges",
    )
    specify_middle = toggle_middle.checkbox(
        label="Specify middle color",
        key="specify_middle",
    )

    start_color_value = st.color_picker(
        label="Start color", key="start_color", disabled=not specify_edges
    )
    middle_color_value = st.color_picker(
        label="Middle color", key="middle_color", disabled=not specify_middle
    )
    end_color_value = st.color_picker(
        label="End color", key="end_color", disabled=not specify_edges
    )

    st.button(label="⬇️⬆️", on_click=swap_colors)

    steepness = st.slider(
        label="Interpolation Curve Steepness",
        min_value=1.0,
        max_value=16.0,
        value=st.session_state.get("steepness", 1.0),
        step=0.5,
        key="steepness",
    )

    x = linspace(start=0, stop=1, num=100)
    y = [normalize_sigmoid(x=i, steepness=steepness) for i in x]
    st.line_chart(data={"Curve": y})

    return PaletteParams(
        start_color=start_color_value if specify_edges else None,
        end_color=end_color_value if specify_edges else None,
        steepness=steepness,
        middle_color=middle_color_value if specify_middle else None,
        middle_position=0.5,
    )


def palette_component(generated_palette: GeneratedPalette) -> None:
    st.subheader("Palette")

    for warning in generated_palette.warnings:
        st.warning(warning)

    palette = generated_palette.colors

    signature = tuple(palette.items())
    if st.session_state.get("palette_signature") != signature:
        st.session_state["palette_signature"] = signature
        for shade, color in palette.items():
            st.session_state[f"color_{shade}"] = color

    cols = st.columns(len(palette))
    for (shade, color), col in zip(palette.items(), cols):
        col.color_picker(label=f"{shade}", value=color, key=f"color_{shade}")

    palette_format = st.selectbox(
        label="Format",
        options=list(PaletteFormat),
        format_func=lambda opt: PALETTE_FORMAT_LABELS[opt],
        key="palette_format",
    )

    ts_color_array_str = palette_to_typescript_color_array_str(
        palette=palette, palette_format=palette_format
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
