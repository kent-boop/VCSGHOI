#!/usr/bin/env python3
"""Render correct-GT single-pair context candidates for manual inspection."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

from render_evidence_shortlist import (
    draw_labeled_box,
    load_name_map,
    load_trace,
    resolve_image,
    scale_box,
)


WHITE = (255, 255, 255)
INK = (28, 32, 38)
MUTED = (91, 98, 108)
LINE = (205, 210, 216)
HUMAN = (190, 30, 45)
OBJECT = (35, 91, 190)
INFO = (22, 145, 86)
EXCLUDED = (190, 125, 30)
SOFT_INFO = (232, 247, 239)
SOFT_WARN = (253, 246, 227)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    value: str,
    size: int,
    *,
    bold: bool = False,
    fill: Tuple[int, int, int] = INK,
) -> None:
    draw.text(xy, value, font=font(size, bold=bold), fill=fill)


def fit_box(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    ratio = min(max_width / image.width, max_height / image.height)
    width = max(1, round(image.width * ratio))
    height = max(1, round(image.height * ratio))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def details_from_row(row: Mapping[str, str]) -> List[Dict[str, Any]]:
    raw = row.get("context_details_json", "")
    data = json.loads(raw) if raw else []
    return [dict(item) for item in data]


def annotated_image(
    record: Mapping[str, Any],
    source: Image.Image,
    row: Mapping[str, str],
    details: Sequence[Mapping[str, Any]],
) -> Image.Image:
    image = source.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    boxes = record["nodes"]["boxes_xyxy_norm"].float()
    width, height = image.size
    h_idx = int(row["human_node_idx"])
    o_idx = int(row["object_node_idx"])
    draw_labeled_box(draw, scale_box(boxes[h_idx], width, height), f"H{h_idx}", HUMAN, 5)
    draw_labeled_box(draw, scale_box(boxes[o_idx], width, height), f"O{o_idx}", OBJECT, 5)
    for item in details:
        node_idx = int(item["node_idx"])
        informative = bool(item["informative"])
        label = f"C{item['slot']} {item['name']}"
        if informative:
            label += " INFO"
        draw_labeled_box(
            draw,
            scale_box(boxes[node_idx], width, height),
            label,
            INFO if informative else EXCLUDED,
            5 if informative else 2,
        )
    return image


def context_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    item: Mapping[str, Any],
) -> int:
    informative = bool(item["informative"])
    fill = SOFT_INFO if informative else SOFT_WARN
    outline = INFO if informative else EXCLUDED
    height = 95
    draw.rounded_rectangle((x, y, x + width, y + height), radius=7, fill=fill, outline=outline, width=2)
    status = "INFORMATIVE" if informative else "EXCLUDED"
    text(draw, (x + 10, y + 8), f"C{item['slot']} {item['name']} | {status}", 15, bold=True, fill=outline)
    text(
        draw,
        (x + 10, y + 32),
        (
            f"det={float(item['detector_score']):.3f}  area={float(item['box_area']):.3f}  "
            f"IoU(H/O max)={float(item['max_endpoint_iou']):.3f}"
        ),
        13,
    )
    text(
        draw,
        (x + 10, y + 53),
        (
            f"final={float(item['final_action_weight_percent']):.5f}%  "
            f"H-C={float(item['hk_strength']):.2f}  C-O={float(item['ko_strength']):.2f}  "
            f"{'two-hop' if item['two_hop'] else 'one-hop'}"
        ),
        13,
    )
    reasons = ", ".join(str(value) for value in item.get("exclusion_reasons", [])) or "passes all filters"
    text(draw, (x + 10, y + 74), reasons, 12, fill=MUTED)
    return height


def build_panel(
    record: Mapping[str, Any],
    source: Image.Image,
    row: Mapping[str, str],
    rank: int,
) -> Image.Image:
    details = details_from_row(row)
    original = fit_box(source.convert("RGB"), 560, 520)
    overlay = fit_box(annotated_image(record, source, row, details), 560, 520)
    view_height = max(original.height, overlay.height)
    header = 102
    footer = 42
    gap = 12
    image_width = original.width + overlay.width + gap
    card_width = 550
    width = image_width + gap + card_width
    content_height = max(view_height, 40 + len(details) * 103)
    height = header + content_height + footer
    panel = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(panel)

    text(draw, (16, 10), f"#{rank:02d}  {row.get('file_name')}", 24, bold=True)
    text(
        draw,
        (16, 43),
        (
            f"Correct GT: {row.get('top_action_name')} {row.get('object_name')}  "
            f"p={float(row.get('top_action_probability', 0)):.3f}  "
            f"paper-score={float(row.get('paper_score', 0)):.3f}"
        ),
        17,
        bold=True,
        fill=INFO,
    )
    text(
        draw,
        (16, 70),
        (
            f"context total={float(row.get('context_weight_percent', 0)):.5f}%  |  "
            f"informative={float(row.get('informative_context_weight_percent', 0)):.5f}%  |  "
            f"visual={float(row.get('visual_quality', 0)):.3f}"
        ),
        15,
        fill=MUTED,
    )

    image_y = header
    panel.paste(original, (0, image_y))
    panel.paste(overlay, (original.width + gap, image_y))
    text(draw, (8, image_y + 8), "ORIGINAL", 14, bold=True)
    text(draw, (original.width + gap + 8, image_y + 8), "TRACE OVERLAY", 14, bold=True)

    card_x = image_width + gap
    text(draw, (card_x, header), "Context audit", 18, bold=True)
    y = header + 30
    for item in details:
        context_card(draw, card_x, y, card_width, item)
        y += 103

    text(
        draw,
        (16, height - 29),
        "Green contexts pass semantic, detector-score, and endpoint-IoU filters; orange contexts remain visible for audit.",
        13,
        fill=MUTED,
    )
    return panel


def make_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    if not paths:
        return
    thumb_width = 820
    columns = 2
    cells: List[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        height = max(1, round(image.height * thumb_width / image.width))
        cells.append(image.resize((thumb_width, height), Image.Resampling.LANCZOS))
    row_heights = [
        max(cell.height for cell in cells[start:start + columns])
        for start in range(0, len(cells), columns)
    ]
    sheet = Image.new("RGB", (thumb_width * columns, sum(row_heights)), (232, 234, 237))
    for index, cell in enumerate(cells):
        row, col = divmod(index, columns)
        y = sum(row_heights[:row])
        framed = ImageOps.expand(cell, border=2, fill=(75, 78, 82))
        sheet.paste(framed.resize((thumb_width, row_heights[row])), (col * thumb_width, y))
    sheet.save(output, quality=91)


def write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--image-pattern")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.ranking_csv).open("r", encoding="utf-8")))
    if args.limit > 0:
        rows = rows[: args.limit]
    outputs: List[Path] = []
    rendered: List[Dict[str, str]] = []
    failures: List[str] = []
    for rank, row in enumerate(rows, start=1):
        try:
            record = load_trace(Path(row["record_path"]))
            image_path = resolve_image(record, Path(args.image_root), args.image_pattern)
            if image_path is None:
                raise FileNotFoundError(f"image not found for {row.get('record_path')}")
            source = Image.open(image_path)
            panel = build_panel(record, source, row, rank)
            output = output_dir / f"{rank:03d}_{Path(row['file_name']).stem}_pair{row['pair_idx']}.png"
            panel.save(output)
            outputs.append(output)
            rendered.append(dict(row))
        except (OSError, KeyError, ValueError, IndexError, RuntimeError, json.JSONDecodeError) as exc:
            failures.append(f"{row.get('file_name')}: {exc}")
    write_rows(output_dir / "rendered_candidates.csv", rendered)
    if failures:
        (output_dir / "render_failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
    make_contact_sheet(outputs, output_dir / "correct_context_contact.jpg")
    print(f"Rendered {len(outputs)} correct-context panels; failures={len(failures)}")


if __name__ == "__main__":
    main()
