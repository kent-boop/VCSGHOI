#!/usr/bin/env python3
"""Render finalist overlays and low-fidelity panel mockups."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
DEFAULT_SHORTLIST = ROOT / "outputs/shortlist.csv"
DEFAULT_OUTPUT = ROOT / "outputs/finalists"
RED, PURPLE, GREEN, GREY, BLUE, ORANGE = (220, 40, 40), (145, 55, 190), (25, 165, 75), (100, 100, 100), (35, 105, 220), (235, 135, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


F14, F18, F22, F28 = font(14), font(18, True), font(22, True), font(28, True)


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


def rect(draw: ImageDraw.ImageDraw, box: list[float], color: tuple[int, int, int], width: int = 4) -> None:
    for offset in range(width):
        draw.rectangle([box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset], outline=color)


def tag(draw: ImageDraw.ImageDraw, box: list[float], text: str, color: tuple[int, int, int]) -> None:
    x, y = int(box[0]) + 3, max(3, int(box[1]) + 3)
    bounds = draw.textbbox((x, y), text, font=F14)
    draw.rectangle((bounds[0] - 2, bounds[1] - 2, bounds[2] + 2, bounds[3] + 2), fill=color)
    draw.text((x, y), text, fill="white", font=F14)


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def anchor(hbox: list[float], obox: list[float]) -> tuple[float, float]:
    hc, oc = center(hbox), center(obox)
    return ((hc[0] + oc[0]) / 2, (hc[1] + oc[1]) / 2)


def overlay(row: pd.Series, roles: bool) -> Image.Image:
    image = Image.open(row["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    hbox, obox = loads(row["human_bbox"]), loads(row["object_bbox"])
    db = loads(row["manual_decisive_bboxes"])
    cb = loads(row["manual_distractor_bboxes"])
    rect(draw, hbox, RED); rect(draw, obox, PURPLE)
    tag(draw, hbox, "H" if roles else "target human", RED)
    tag(draw, obox, "O" if roles else "target object", PURPLE)
    a = anchor(hbox, obox)
    draw.line([center(hbox), center(obox)], fill=ORANGE, width=5)
    draw.ellipse((a[0] - 9, a[1] - 9, a[0] + 9, a[1] + 9), fill=ORANGE, outline="white", width=2)
    if roles:
        draw.text((a[0] + 12, a[1] - 12), "A", fill=ORANGE, font=F18)
    for index, box in enumerate(db):
        rect(draw, box, GREEN, 3); tag(draw, box, f"D{index + 1}" if roles else "decisive cue", GREEN)
    for index, box in enumerate(cb):
        rect(draw, box, GREY, 2); tag(draw, box, f"C{index + 1}" if roles else "context", GREY)
    if roles:
        draw.text((12, image.height - 34), f"S: {row['scene_evidence']}", fill=BLUE, font=F18, stroke_width=2, stroke_fill="white")
    return image


def fit_background(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(source, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (244, 245, 247))
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def panel_a(row: pd.Series) -> Image.Image:
    base = fit_background(Image.open(row["image_path"]).convert("RGB"), (900, 600))
    draw = ImageDraw.Draw(base, "RGBA")
    hbox = loads(row["human_bbox"]); obox = loads(row["object_bbox"])
    sx, sy = base.width / int(row["image_width"]), base.height / int(row["image_height"])
    scale = lambda b: [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
    a = anchor(scale(hbox), scale(obox))
    boxes = [(scale(b), GREEN, "useful?") for b in loads(row["manual_decisive_bboxes"])] + [(scale(b), GREY, "context?") for b in loads(row["manual_distractor_bboxes"])]
    draw.rounded_rectangle((10, 10, 890, 74), 12, fill=(255, 255, 255, 225), outline=(50, 60, 70, 255), width=2)
    draw.text((28, 22), "(a) Generic Context Aggregation", fill=(20, 30, 45), font=F28)
    draw.ellipse((a[0] - 15, a[1] - 15, a[0] + 15, a[1] + 15), fill=RED + (230,))
    for b, color, text in boxes:
        c = center(b)
        draw.line([c, a], fill=color + (210,), width=5)
        draw.ellipse((c[0] - 11, c[1] - 11, c[0] + 11, c[1] + 11), fill=color + (235,))
        draw.text((c[0] + 13, c[1] - 9), text, fill=color + (255,), font=F14, stroke_width=2, stroke_fill=(255, 255, 255, 230))
    draw.rounded_rectangle((18, 520, 882, 586), 10, fill=(255, 255, 255, 225))
    draw.text((34, 533), "Useful evidence and incidental context are mixed before updating the HO anchor.", fill=(30, 35, 45), font=F18)
    return base


def node(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label_text: str, color: tuple[int, int, int], radius: int = 30) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=3)
    bounds = draw.textbbox((0, 0), label_text, font=F18)
    draw.text((x - (bounds[2] - bounds[0]) / 2, y - 11), label_text, fill="white", font=F18)


def panel_b(row: pd.Series) -> Image.Image:
    canvas = Image.new("RGB", (1200, 650), (246, 248, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "(b) EvidenceHOI: Topology Matching", fill=(20, 30, 45), font=F28)
    photo = fit_background(Image.open(row["image_path"]).convert("RGB"), (530, 520))
    canvas.paste(photo, (20, 88))
    draw.rounded_rectangle((570, 88, 1180, 608), 18, fill=(31, 38, 55), outline=(80, 95, 120), width=2)
    draw.text((596, 108), "Visual Evidence Topology", fill="white", font=F22)
    v = {"H": (640, 220), "A": (760, 300), "O": (640, 390), "D": (895, 205), "S": (895, 330), "C": (895, 455)}
    for left, right in [("H", "A"), ("O", "A"), ("D", "A"), ("S", "A"), ("C", "A")]:
        color = GREEN if left == "D" else (BLUE if left == "S" else GREY if left == "C" else ORANGE)
        draw.line([v[left], v[right]], fill=color, width=6)
    node(draw, v["H"], "H", RED); node(draw, v["A"], "A", ORANGE); node(draw, v["O"], "O", PURPLE)
    node(draw, v["D"], "D", GREEN); node(draw, v["S"], "S", BLUE); node(draw, v["C"], "C", GREY)
    draw.text((1005, 108), "Language Prototype", fill="white", font=F22)
    proto = [(1040, 200, "Human State"), (1040, 285, "Anchor Relation"), (1040, 370, "Object State"), (1040, 455, "Discriminative Cue")]
    for x, y, text in proto:
        draw.rounded_rectangle((958, y - 27, 1158, y + 27), 10, fill=(67, 76, 100), outline=(150, 165, 195), width=2)
        bounds = draw.textbbox((0, 0), text, font=F14)
        draw.text((1058 - (bounds[2] - bounds[0]) / 2, y - 9), text, fill="white", font=F14)
    draw.line([(925, 205), (958, 455)], fill=GREEN, width=5)
    draw.text((824, 510), "D strengthened", fill=GREEN, font=F18)
    draw.text((824, 542), "C suppressed", fill=(185, 185, 185), font=F18)
    draw.text((824, 574), "verified evidence -> refined HO relation", fill=ORANGE, font=F14)
    return canvas


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.shortlist).sort_values("rank").head(args.top_k)
    output_dir = args.output_dir.resolve()
    for _, row in frame.iterrows():
        folder = output_dir / f"{int(row['rank']):02d}_{row['dataset'].lower().replace('-', '')}_{row['image_id']}"
        folder.mkdir(parents=True, exist_ok=True)
        original = Image.open(row["image_path"]).convert("RGB")
        original.save(folder / "original.jpg", quality=96)
        overlay(row, roles=False).save(folder / "target_pair_overlay.png")
        overlay(row, roles=True).save(folder / "evidence_role_overlay.png")
        panel_a(row).save(folder / "panel_a_mockup.png")
        panel_b(row).save(folder / "panel_b_mockup.png")
        metadata = {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}
        (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
