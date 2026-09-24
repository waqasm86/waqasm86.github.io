#!/usr/bin/env python3
"""Generate a deterministic 1200×630 PNG social card."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 630


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = [Path("/usr/share/fonts/truetype/dejavu") / name, Path("/usr/share/fonts/dejavu") / name]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render(title: str, series: str, output: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0b1020")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, HEIGHT), fill="#42b883")
    draw.ellipse((930, -180, 1370, 260), fill="#152c43")
    draw.ellipse((990, 390, 1280, 680), fill="#123a35")

    draw.text((80, 62), series.upper(), font=font(25, bold=True), fill="#42b883")
    wrapped = textwrap.wrap(title, width=35)[:4]
    y = 145
    title_font = font(54, bold=True)
    for line in wrapped:
        draw.text((80, y), line, font=title_font, fill="#f4f7fb")
        y += 68
    draw.line((80, 500, 720, 500), fill="#314158", width=2)
    draw.text((80, 526), "Mohammad Waqas", font=font(28, bold=True), fill="#f4f7fb")
    draw.text((80, 568), "AI Inference & GPU Systems", font=font(23), fill="#a9b4c7")
    draw.text((860, 550), "waqasm86.github.io", font=font(20), fill="#a9b4c7")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--series", default="AI Inference & GPU Systems")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.title, args.series, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
