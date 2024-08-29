import math
from typing import Dict, List, NamedTuple, Tuple

import pyperclip
import streamlit as st

TAILWIND_SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]


class PaletteParams(NamedTuple):
    start_color: str
    end_color: str
    steepness: float


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


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


def generate_palette(params: PaletteParams) -> Dict[int, str]:
    palette = {}
    for i, shade in enumerate(TAILWIND_SHADES):
        factor = i / (len(TAILWIND_SHADES) - 1)
        palette[shade] = interpolate_color(
            color1=params.start_color,
            color2=params.end_color,
            factor=factor,
            steepness=params.steepness,
        )
    return palette


def palette_to_typescript_color_array_str(palette: Dict[int, str]) -> str:
    ts_color_array_str = "{\n"
    for shade, color in palette.items():
        ts_color_array_str += f'    {shade}: "{color}",\n'
    ts_color_array_str += "}"
    return ts_color_array_str


def palette_parameter_component() -> PaletteParams:
    st.subheader("Parameters")

    if "start_color" not in st.session_state:
        st.session_state.start_color = "#FFFFFF"
    if "end_color" not in st.session_state:
        st.session_state.end_color = "#000000"

    def swap_colors():
        st.session_state.start_color, st.session_state.end_color = (
            st.session_state.end_color,
            st.session_state.start_color,
        )

    start_color = st.color_picker(label="Start color", key="start_color")
    st.button(label="⬇️⬆️", on_click=swap_colors)
    end_color = st.color_picker(label="End color", key="end_color")

    steepness = st.slider(
        label="Interpolation Curve Steepness",
        min_value=1.0,
        max_value=16.0,
        value=1.0,
        step=0.5,
        key="steepness",
    )

    x = linspace(start=0, stop=1, num=100)
    y = [normalize_sigmoid(x=i, steepness=steepness) for i in x]
    st.line_chart(data={"Curve": y})

    return PaletteParams(
        start_color=start_color, end_color=end_color, steepness=steepness
    )


def palette_component(palette: Dict[int, str]) -> None:
    st.subheader("Palette")

    cols = st.columns(len(palette))
    for (shade, color), col in zip(palette.items(), cols):
        col.color_picker(label=f"{shade}", value=color, key=f"color_{shade}")

    ts_color_array_str = palette_to_typescript_color_array_str(palette=palette)
    st.code(body=ts_color_array_str, language="typescript")

    if st.button(label="Copy"):
        pyperclip.copy(ts_color_array_str)
        st.success("Copied to clipboard!")


def main() -> None:
    st.title("Tailwind Color Palette Generator")

    params = palette_parameter_component()

    palette = None
    with st.spinner("Generating palette..."):
        palette = generate_palette(params=params)

    if palette:
        palette_component(palette=palette)


if __name__ == "__main__":
    main()
