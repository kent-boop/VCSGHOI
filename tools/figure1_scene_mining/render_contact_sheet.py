#!/usr/bin/env python3
"""Render scenario-specific contact sheets for manual EvidenceHOI review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs/raw_candidates.csv"
DEFAULT_OUTPUT = ROOT / "outputs/contact_sheets"

RED = (220, 40, 40)
PURPLE = (145, 60, 190)
GREEN = (30, 170, 80)
GREY = (110, 110, 110)
BLUE = (40, 110, 220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--per-page", type=int, default=20)
    parser.add_argument("--columns", type=int, default=5)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for item in candidates:
        try:
            return ImageFont.truetype(item, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = font(14)
FONT_SMALL = font(11)
FONT_BOLD = font(15, True)


def loads(value: Any) -> list[Any]:
    if isinstance(value, str) and value:
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def box(value: Any) -> list[float]:
    values = loads(value)
    return [float(x) for x in values[:4]]


def draw_box(draw: ImageDraw.ImageDraw, coords: list[float], color: tuple[int, int, int], width: int = 4) -> None:
    for offset in range(width):
        draw.rectangle([coords[0] - offset, coords[1] - offset, coords[2] + offset, coords[3] + offset], outline=color)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    bounds = draw.textbbox(xy, text, font=FONT_SMALL)
    draw.rectangle((bounds[0] - 2, bounds[1] - 2, bounds[2] + 2, bounds[3] + 2), fill=color)
    draw.text(xy, text, fill="white", font=FONT_SMALL)


def scaled(coords: list[float], scale: float) -> list[float]:
    return [value * scale for value in coords]


def render_card(row: pd.Series, card_w: int = 360, image_h: int = 250, text_h: int = 116) -> Image.Image:
    source = Image.open(str(row["image_path"])).convert("RGB")
    fitted = ImageOps.contain(source, (card_w - 12, image_h - 12), Image.Resampling.LANCZOS)
    scale = min((card_w - 12) / source.width, (image_h - 12) / source.height)
    canvas = Image.new("RGB", (card_w, image_h + text_h), "white")
    x0 = (card_w - fitted.width) // 2
    y0 = (image_h - fitted.height) // 2
    canvas.paste(fitted, (x0, y0))
    draw = ImageDraw.Draw(canvas)

    hbox = [v + (x0 if i % 2 == 0 else y0) for i, v in enumerate(scaled(box(row["human_bbox"]), scale))]
    obox = [v + (x0 if i % 2 == 0 else y0) for i, v in enumerate(scaled(box(row["object_bbox"]), scale))]
    draw_box(draw, hbox, RED)
    draw_box(draw, obox, PURPLE)
    label(draw, (int(hbox[0]) + 2, max(2, int(hbox[1]) + 2)), "H", RED)
    label(draw, (int(obox[0]) + 2, max(2, int(obox[1]) + 2)), "O", PURPLE)

    decisive = loads(row["decisive_cue_candidates"])
    distractors = loads(row["distractor_candidates"])
    for idx, item in enumerate(decisive[:3]):
        coords = [v + (x0 if i % 2 == 0 else y0) for i, v in enumerate(scaled(item["bbox"], scale))]
        draw_box(draw, coords, GREEN, 3)
        label(draw, (int(coords[0]) + 2, max(2, int(coords[1]) + 2)), f"D{idx+1}", GREEN)
    for idx, item in enumerate(distractors[:4]):
        coords = [v + (x0 if i % 2 == 0 else y0) for i, v in enumerate(scaled(item["bbox"], scale))]
        draw_box(draw, coords, GREY, 2)
        label(draw, (int(coords[0]) + 2, max(2, int(coords[1]) + 2)), f"C{idx+1}", GREY)

    y = image_h + 4
    lines = [
        f"#{int(row['scenario_rank']):02d} {row['scenario']} | auto {float(row['auto_priority_score']):.1f}",
        f"{row['split']} / {row['file_name']} | {row['target_object']}",
        f"GT: {', '.join(loads(row['annotated_actions']))}",
        f"people/entities/context: {int(row['number_of_people'])}/{int(row['number_of_entities'])}/{int(row['number_of_context_objects'])}",
        f"D: {', '.join(x['category'] for x in decisive) or '-'}",
        f"C: {', '.join(x['category'] for x in distractors[:4]) or '-'}",
    ]
    for index, line in enumerate(lines):
        draw.text((8, y), line[:62], fill=(15, 15, 15), font=FONT_BOLD if index == 0 else FONT_SMALL)
        y += 18
    return canvas


def write_pages(frame: pd.DataFrame, scenario: str, output_dir: Path, per_page: int, columns: int) -> None:
    cards = [(row, render_card(row)) for _, row in frame.iterrows()]
    card_w, card_h = 360, 366
    rows_per_page = math.ceil(per_page / columns)
    for page_index in range(math.ceil(len(cards) / per_page)):
        page_cards = cards[page_index * per_page:(page_index + 1) * per_page]
        sheet = Image.new("RGB", (columns * card_w, 44 + rows_per_page * card_h), (232, 235, 240))
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 9), f"EvidenceHOI Figure 1 candidates | {scenario} | page {page_index + 1}", fill=(10, 20, 35), font=font(22, True))
        for index, (_, card) in enumerate(page_cards):
            col, row_index = index % columns, index // columns
            sheet.paste(card, (col * card_w, 44 + row_index * card_h))
        out = output_dir / scenario / f"{scenario}_page_{page_index + 1:02d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=92)


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    scenarios = sorted(frame["scenario"].unique()) if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        subset = frame[frame["scenario"] == scenario].sort_values("scenario_rank").head(args.top_k)
        write_pages(subset, scenario, args.output_dir.resolve(), args.per_page, args.columns)
    print(f"Rendered {len(scenarios)} scenario groups to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
