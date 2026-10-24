#!/usr/bin/env python3
"""Render contact sheets with active/passive pair assignments for Round 3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs/dual_ambiguity_round3/tennis_dual_candidates.csv"
DEFAULT_OUTPUT = ROOT / "outputs/dual_ambiguity_round3/contact_sheets"
RED, PURPLE, BLUE, CYAN, GREEN = (220, 40, 40), (145, 55, 190), (35, 95, 220), (20, 165, 180), (20, 170, 75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--per-page", type=int, default=20)
    return parser.parse_args()


def loads(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []
    return []


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


F11, F14, F16 = font(11), font(14), font(16, True)


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: tuple[int, int, int], width: int = 3) -> None:
    for offset in range(width):
        draw.rectangle([box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset], outline=color)


def tag(draw: ImageDraw.ImageDraw, box: list[float], text: str, color: tuple[int, int, int]) -> None:
    x, y = int(box[0]) + 2, max(2, int(box[1]) + 2)
    bounds = draw.textbbox((x, y), text, font=F11)
    draw.rectangle((bounds[0] - 2, bounds[1] - 2, bounds[2] + 2, bounds[3] + 2), fill=color)
    draw.text((x, y), text, fill="white", font=F11)


def render(row: pd.Series, card_w: int = 420, image_h: int = 280, text_h: int = 110) -> Image.Image:
    source = Image.open(row["image_path"]).convert("RGB")
    fitted = ImageOps.contain(source, (card_w - 12, image_h - 12), Image.Resampling.LANCZOS)
    scale = min((card_w - 12) / source.width, (image_h - 12) / source.height)
    x0, y0 = (card_w - fitted.width) // 2, (image_h - fitted.height) // 2
    canvas = Image.new("RGB", (card_w, image_h + text_h), "white")
    canvas.paste(fitted, (x0, y0))
    draw = ImageDraw.Draw(canvas)

    def transform(value: Any) -> list[float]:
        box = loads(value)
        return [box[0] * scale + x0, box[1] * scale + y0, box[2] * scale + x0, box[3] * scale + y0]

    boxes = [
        (transform(row["active_human_bbox"]), RED, "H1 active"),
        (transform(row["active_racket_bbox"]), PURPLE, "R1"),
        (transform(row["passive_human_bbox"]), BLUE, "H2 passive"),
        (transform(row["passive_racket_bbox"]), CYAN, "R2"),
    ]
    for box, color, text in boxes:
        draw_box(draw, box, color); tag(draw, box, text, color)
    for index, ball in enumerate(loads(row["sports_ball_bboxes"])):
        box = [ball[0] * scale + x0, ball[1] * scale + y0, ball[2] * scale + x0, ball[3] * scale + y0]
        draw_box(draw, box, GREEN); tag(draw, box, f"ball {index + 1}", GREEN)

    y = image_h + 3
    lines = [
        f"#{int(row['rank']):02d} score {float(row['dual_score']):.1f} | {row['file_name']}",
        f"active: {', '.join(loads(row['active_actions']))} | passive: {', '.join(loads(row['passive_actions']))}",
        f"ball={int(row['annotated_ball_count'])} | min human height={float(row['min_human_height_ratio']):.2f} | balance={float(row['human_size_balance']):.2f}",
        f"ball margin={float(row['ball_assignment_margin']):.2f} | pair separation={float(row['person_separation']):.2f}",
    ]
    for index, line in enumerate(lines):
        draw.text((7, y), line[:72], fill=(20, 25, 35), font=F16 if index == 0 else F11)
        y += 22
    return canvas


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input).head(args.top_k)
    cards = [render(row) for _, row in frame.iterrows()]
    cols, card_w, card_h = 4, 420, 390
    rows_per_page = math.ceil(args.per_page / cols)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for page in range(math.ceil(len(cards) / args.per_page)):
        subset = cards[page * args.per_page:(page + 1) * args.per_page]
        sheet = Image.new("RGB", (cols * card_w, 42 + rows_per_page * card_h), (232, 235, 240))
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), f"Dual-ambiguity tennis candidates | page {page + 1}", fill=(15, 25, 40), font=font(22, True))
        for index, card in enumerate(subset):
            sheet.paste(card, ((index % cols) * card_w, 42 + (index // cols) * card_h))
        sheet.save(output / f"dual_tennis_page_{page + 1:02d}.jpg", quality=93)
    print(f"rendered={len(cards)} pages={math.ceil(len(cards) / args.per_page)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
