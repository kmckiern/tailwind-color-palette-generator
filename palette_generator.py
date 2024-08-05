import colorsys
from typing import Dict, Tuple

import numpy as np
import pyperclip
import streamlit as st

TAILWIND_SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def custom_sigmoid(x: float, steepness: float) -> float:
    return 1 / (1 + np.exp(-steepness * (x - 0.5)))


def normalize_sigmoid(x: float, steepness: float) -> float:
    sig_0 = custom_sigmoid(x=0, steepness=steepness)
    sig_1 = custom_sigmoid(x=1, steepness=steepness)
    return (custom_sigmoid(x=x, steepness=steepness) - sig_0) / (sig_1 - sig_0)


def interpolate_color(color1: str, color2: str, factor: float, steepness: float) -> str:
    rgb1 = hex_to_rgb(hex_color=color1)
    rgb2 = hex_to_rgb(hex_color=color2)
    hsv1 = colorsys.rgb_to_hsv(*(x / 255 for x in rgb1))
    hsv2 = colorsys.rgb_to_hsv(*(x / 255 for x in rgb2))

    adjusted_factor = normalize_sigmoid(x=factor, steepness=steepness)

    h = hsv1[0] + adjusted_factor * (hsv2[0] - hsv1[0])
    s = hsv1[1] + adjusted_factor * (hsv2[1] - hsv1[1])
    v = hsv1[2] + adjusted_factor * (hsv2[2] - hsv1[2])

    rgb = colorsys.hsv_to_rgb(h, s, v)
    return rgb_to_hex(rgb=tuple(int(x * 255) for x in rgb))


def generate_palette(
    start_color: str, end_color: str, steepness: float
) -> Dict[int, str]:
    palette = {}
    for i, shade in enumerate(TAILWIND_SHADES):
        factor = i / (len(TAILWIND_SHADES) - 1)
        palette[shade] = interpolate_color(
            color1=start_color, color2=end_color, factor=factor, steepness=steepness
        )
    return palette


def palette_to_typescript(palette: Dict[int, str]) -> str:
    ts_object = "{\n"
    for shade, color in palette.items():
        ts_object += f'    {shade}: "{color}",\n'
    ts_object += "}"
    return ts_object


def main():
    st.title("Tailwind Color Palette Generator")

    st.subheader("Parameters")
    start_color = st.color_picker(
        label="Start color", value="#FFFFFF", key="start_color"
    )
    end_color = st.color_picker(label="End color", value="#000000", key="end_color")

    steepness = st.slider(
        label="Interpolation Curve Steepness",
        min_value=1.0,
        max_value=16.0,
        value=1.0,
        step=0.5,
        key="steepness",
    )

    x = np.linspace(start=0, stop=1, num=100)
    y = [normalize_sigmoid(x=i, steepness=steepness) for i in x]
    st.line_chart(data={"Curve": y})

    palette = None
    with st.spinner("Generating palette..."):
        palette = generate_palette(
            start_color=start_color, end_color=end_color, steepness=steepness
        )

    if palette:
        st.subheader("Palette")
        cols = st.columns(len(palette))
        for (shade, color), col in zip(palette.items(), cols):
            col.color_picker(
                label=f"{shade}", value=color, key=f"color_{shade}", disabled=True
            )

        ts_object = palette_to_typescript(palette=palette)
        st.code(body=ts_object, language="typescript")

        if st.button(label="Copy"):
            pyperclip.copy(ts_object)
            st.success("Copied to clipboard!")


if __name__ == "__main__":
    main()
