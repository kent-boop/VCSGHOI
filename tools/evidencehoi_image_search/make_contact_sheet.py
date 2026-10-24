#!/usr/bin/env python3
"""Create contact sheets and annotated previews for scored HOI candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PERSON_COLOR = (37, 99, 235)
OBJECT_COLOR = (245, 158, 11)
CONTEXT_COLOR = (16, 185, 129)
PRIMARY_COLOR = (220, 38, 38)
SECONDARY_COLOR = (147, 51, 234)
TEXT_BG = (255, 255, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (107, 114, 128)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = BLACK,
    bg: tuple[int, int, int] = TEXT_BG,
) -> None:
    x, y = xy
    w, h = text_size(draw, text, font)
    pad = 3
    draw.rectangle((x, y, x + w + pad * 2, y + h + pad * 2), fill=bg, outline=fill)
    draw.text((x + pad, y + pad), text, font=font, fill=fill)


def load_candidates(path: Path, bucket: str, top_k: int) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not path.exists():
        fallback = path.parent / "candidates_hard_filtered.jsonl"
        if fallback.exists():
            warnings.append(f"Input `{path}` not found; using fallback `{fallback}`.")
            path = fallback
        else:
            warnings.append(f"Input `{path}` not found; no fallback candidates_hard_filtered.jsonl exists.")
            return [], warnings
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                warnings.append(f"Skipped malformed JSONL line {line_no}: {exc}")
                continue
            if bucket != "all" and row.get("bucket_suggestion") != bucket:
                continue
            rows.append(row)
    rows.sort(key=lambda item: float(item.get("total_score", item.get("score", 0.0)) or 0.0), reverse=True)
    return rows[:top_k], warnings


def resolve_image_path(row: dict[str, Any], image_root: Path | None) -> Path:
    image_path = Path(str(row.get("image_path") or ""))
    if image_path.exists():
        return image_path
    file_name = str(row.get("file_name") or "")
    if image_root is not None:
        candidate = image_root / file_name
        if candidate.exists():
            return candidate
    return image_path


def scale_box(box: list[float] | tuple[float, float, float, float], scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    return (float(box[0]) * scale_x, float(box[1]) * scale_y, float(box[2]) * scale_x, float(box[3]) * scale_y)


def union_box(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int],
    width: int = 2,
    dash: int = 10,
) -> None:
    x1, y1, x2, y2 = box
    segments = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for start, end in segments:
        sx, sy = start
        ex, ey = end
        length = math.dist(start, end)
        if length <= 0:
            continue
        steps = int(length // dash) + 1
        for idx in range(steps):
            if idx % 2:
                continue
            t0 = idx / steps
            t1 = min((idx + 1) / steps, 1.0)
            p0 = (sx + (ex - sx) * t0, sy + (ey - sy) * t0)
            p1 = (sx + (ex - sx) * t1, sy + (ey - sy) * t1)
            draw.line((p0, p1), fill=fill, width=width)


def get_pair_boxes(pair: dict[str, Any]) -> tuple[tuple[float, float, float, float] | None, tuple[float, float, float, float] | None]:
    human_box = pair.get("human_bbox_xyxy")
    object_box = pair.get("object_bbox_xyxy")
    if human_box and object_box:
        return tuple(human_box), tuple(object_box)
    return None, None


def annotate_image(
    image: Image.Image,
    row: dict[str, Any],
    output_path: Path,
    max_size: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> tuple[Image.Image, list[str]]:
    warnings: list[str] = []
    original_w, original_h = image.size
    scale = min(max_size / max(original_w, original_h), 1.0)
    canvas = image.copy().convert("RGB")
    if scale < 1.0:
        canvas = canvas.resize((int(original_w * scale), int(original_h * scale)), Image.Resampling.LANCZOS)
    sx, sy = canvas.size[0] / original_w, canvas.size[1] / original_h
    draw = ImageDraw.Draw(canvas)

    top_pairs = row.get("top_3_pairs") or row.get("top_pairs", [])[:3]
    primary = row.get("selected_primary_pair") or (top_pairs[0] if top_pairs else None)
    secondary = row.get("selected_secondary_pairs") or top_pairs[1:3]

    context_ids: set[int] = set()
    for pair in top_pairs[:3]:
        context_ids.update(int(item) for item in pair.get("context_ids", []))
        for ctx in pair.get("context_set", []):
            if "instance_id" in ctx:
                context_ids.add(int(ctx["instance_id"]))

    # Draw context boxes first so primary boxes stay visible.
    context_label_idx = 1
    drawn_context: set[int] = set()
    for pair in top_pairs[:3]:
        for ctx in pair.get("context_set", [])[:3]:
            ctx_id = int(ctx.get("instance_id", -1))
            if ctx_id in drawn_context or "bbox_xyxy" not in ctx:
                continue
            box = scale_box(ctx["bbox_xyxy"], sx, sy)
            draw.rectangle(box, outline=CONTEXT_COLOR, width=2)
            draw_label(draw, (box[0], max(0, box[1] - 18)), f"C{context_label_idx}:{ctx.get('category_name', 'ctx')}", small_font, CONTEXT_COLOR)
            drawn_context.add(ctx_id)
            context_label_idx += 1

    def draw_pair(pair: dict[str, Any], label: str, color: tuple[int, int, int], width: int) -> None:
        h_box_raw = pair.get("human_bbox_xyxy")
        o_box_raw = pair.get("object_bbox_xyxy")
        if not h_box_raw or not o_box_raw:
            warnings.append(f"Pair {label} lacks human/object bbox fields; only text label is available.")
            return
        h_box = scale_box(h_box_raw, sx, sy)
        o_box = scale_box(o_box_raw, sx, sy)
        draw.rectangle(h_box, outline=color, width=width)
        draw.rectangle(o_box, outline=color, width=width)
        u_box = union_box(h_box, o_box)
        draw_dashed_rectangle(draw, u_box, color, width=max(1, width - 1))
        h_center = center(h_box)
        o_center = center(o_box)
        draw.line((h_center, o_center), fill=color, width=max(1, width - 1))
        object_name = pair.get("object_category_name", "object")
        actions = ",".join(pair.get("actions", [])[:3])
        draw_label(draw, (h_box[0], max(0, h_box[1] - 22)), f"{label}:P{pair.get('subject_id')}", small_font, color)
        draw_label(draw, (o_box[0], max(0, o_box[1] - 22)), f"O{pair.get('object_id')}:{object_name}", small_font, color)
        draw_label(draw, (u_box[0], min(canvas.size[1] - 24, u_box[3] + 2)), f"{label}: {object_name} / {actions}", small_font, color)

    if primary:
        draw_pair(primary, "q1", PRIMARY_COLOR, 5)
    for idx, pair in enumerate(secondary[:2], start=2):
        draw_pair(pair, f"q{idx}", SECONDARY_COLOR, 3)

    # Evidence paths for q1/q2 only, top 2-3 context nodes total.
    path_count = 0
    for pair_idx, pair in enumerate(top_pairs[:2], start=1):
        h_box_raw = pair.get("human_bbox_xyxy")
        o_box_raw = pair.get("object_bbox_xyxy")
        if not h_box_raw or not o_box_raw:
            continue
        h_center = center(scale_box(h_box_raw, sx, sy))
        o_center = center(scale_box(o_box_raw, sx, sy))
        for ctx in pair.get("context_set", [])[:2]:
            if path_count >= 3 or "bbox_xyxy" not in ctx:
                break
            c_center = center(scale_box(ctx["bbox_xyxy"], sx, sy))
            color = PRIMARY_COLOR if pair_idx == 1 else SECONDARY_COLOR
            draw.line((h_center, c_center), fill=color, width=2)
            draw.line((c_center, o_center), fill=color, width=2)
            draw.ellipse((c_center[0] - 4, c_center[1] - 4, c_center[0] + 4, c_center[1] + 4), fill=color)
            path_count += 1

    header = f"{row.get('image_id')} | score={row.get('total_score', row.get('score'))} | bucket={row.get('bucket_suggestion')}"
    draw_label(draw, (8, 8), header, font, BLACK, WHITE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return canvas, warnings


def make_crop_sheet(
    image: Image.Image,
    row: dict[str, Any],
    output_path: Path,
    font: ImageFont.ImageFont,
    crop_size: int = 160,
) -> None:
    crops: list[tuple[str, Image.Image]] = []
    primary = row.get("selected_primary_pair")
    secondary = row.get("selected_secondary_pairs", [])
    if primary:
        items = [("q1 human", primary.get("human_bbox_xyxy")), (f"q1 {primary.get('object_category_name', 'object')}", primary.get("object_bbox_xyxy"))]
        for ctx in primary.get("context_set", [])[:4]:
            items.append((f"ctx {ctx.get('category_name', 'context')}", ctx.get("bbox_xyxy")))
        for idx, pair in enumerate(secondary[:2], start=2):
            items.append((f"q{idx} {pair.get('object_category_name', 'object')}", pair.get("object_bbox_xyxy")))
    else:
        items = []
    for label, box in items:
        if not box:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(image.width - 1, x1))
        y1 = max(0, min(image.height - 1, y1))
        x2 = max(x1 + 1, min(image.width, x2))
        y2 = max(y1 + 1, min(image.height, y2))
        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        crop.thumbnail((crop_size, crop_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (crop_size, crop_size + 28), WHITE)
        tile.paste(crop, ((crop_size - crop.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((4, crop_size + 6), label[:24], font=font, fill=BLACK)
        crops.append((label, tile))
    if not crops:
        return
    cols = min(4, len(crops))
    rows = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * crop_size, rows * (crop_size + 28)), WHITE)
    for idx, (_, tile) in enumerate(crops):
        x = (idx % cols) * crop_size
        y = (idx // cols) * (crop_size + 28)
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_tile(annotated: Image.Image, row: dict[str, Any], tile_w: int, tile_h: int, font: ImageFont.ImageFont, small_font: ImageFont.ImageFont) -> Image.Image:
    meta_h = 150
    image_h = tile_h - meta_h
    img = annotated.copy()
    img.thumbnail((tile_w, image_h), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_w, tile_h), WHITE)
    tile.paste(img, ((tile_w - img.width) // 2, 0))
    draw = ImageDraw.Draw(tile)
    y = image_h + 4
    counts = row.get("counts", {})
    warnings = set(row.get("warnings", []))
    flags = []
    if row.get("small_object_relaxed") or any("small_object_relaxed" in w for w in warnings):
        flags.append("small_object_relaxed")
    if row.get("reject_flags", {}).get("reject_severe_sports") or "high_homogeneity_penalty" in warnings:
        flags.append("sports_dominated")
    if row.get("component_scores", {}).get("CrowdingPenalty", 0) > 0:
        flags.append("crowding_warning")
    if any("primary_pair_near_edge" in w for w in warnings):
        flags.append("edge_layout_warning")
    lines = [
        f"id={row.get('image_id')} score={row.get('total_score', row.get('score'))} bucket={row.get('bucket_suggestion')}",
        f"P={counts.get('num_persons')} pairs={counts.get('num_valid_hoi_pairs')} act={counts.get('num_unique_action_categories')} obj={counts.get('num_unique_object_categories')}",
        f"primary ctx={counts.get('primary_pair_context_count')} flags={','.join(flags) if flags else 'none'}",
    ]
    for pair in (row.get("top_3_pairs") or row.get("top_pairs", []))[:2]:
        lines.append(f"h{pair.get('subject_id')}-o{pair.get('object_id')} {pair.get('object_category_name')} {','.join(pair.get('actions', [])[:2])}")
    for line in lines:
        draw.text((8, y), line[:80], font=small_font, fill=BLACK)
        y += 18
    return tile


def save_raw_reference(image_path: Path, output_dir: Path, rank: int, image_id: str, copy_raw: bool) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix or ".jpg"
    target = output_dir / f"top_{rank:04d}_{image_id}{suffix}"
    if copy_raw and image_path.exists():
        shutil.copy2(image_path, target)
        return str(target)
    index_path = output_dir / "raw_selected_images_index.txt"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(f"top_{rank:04d}_{image_id}\t{image_path}\n")
    return str(image_path)


def write_review_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "image_id",
        "file_name",
        "image_path",
        "total_score",
        "bucket_suggestion",
        "num_persons",
        "num_valid_pairs",
        "num_unique_actions",
        "num_unique_objects",
        "primary_pair",
        "primary_context_count",
        "evidence_paths",
        "warnings",
        "review_decision",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            counts = row.get("counts", {})
            primary = row.get("selected_primary_pair") or {}
            primary_text = ""
            paths = []
            if primary:
                primary_text = f"h{primary.get('subject_id')}-o{primary.get('object_id')} {primary.get('object_category_name')} {'/'.join(primary.get('actions', []))}"
                for ctx in primary.get("context_set", [])[:3]:
                    paths.append(f"P{primary.get('subject_id')} -> C{ctx.get('instance_id')}({ctx.get('category_name')}) -> O{primary.get('object_id')}")
            writer.writerow(
                {
                    "rank": idx,
                    "image_id": row.get("image_id"),
                    "file_name": row.get("file_name"),
                    "image_path": row.get("image_path"),
                    "total_score": row.get("total_score", row.get("score")),
                    "bucket_suggestion": row.get("bucket_suggestion"),
                    "num_persons": counts.get("num_persons"),
                    "num_valid_pairs": counts.get("num_valid_hoi_pairs"),
                    "num_unique_actions": counts.get("num_unique_action_categories"),
                    "num_unique_objects": counts.get("num_unique_object_categories"),
                    "primary_pair": primary_text,
                    "primary_context_count": counts.get("primary_pair_context_count"),
                    "evidence_paths": " | ".join(paths),
                    "warnings": ";".join(row.get("warnings", [])),
                    "review_decision": "",
                    "notes": "",
                }
            )


def write_notes_template(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = ["# EvidenceHOI Candidate Review Notes", ""]
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    if not rows:
        lines.append("No candidates were available for review with the current input/bucket filter.")
        lines.append("")
    for idx, row in enumerate(rows[:20], start=1):
        primary = row.get("selected_primary_pair") or {}
        paths = []
        if primary:
            for ctx in primary.get("context_set", [])[:3]:
                paths.append(f"P{primary.get('subject_id')} -> C{ctx.get('instance_id')}({ctx.get('category_name')}) -> O{primary.get('object_id')}")
        lines.append(f"## {idx}. {row.get('file_name')} [{row.get('image_id')}]")
        lines.append("")
        lines.append(f"- Score: `{row.get('total_score', row.get('score'))}` Bucket: `{row.get('bucket_suggestion')}`")
        lines.append(f"- Primary pair: `{primary.get('subject_id')} -> {primary.get('object_id')} {primary.get('object_category_name')} {primary.get('actions')}`")
        lines.append(f"- Evidence paths: {paths if paths else 'none'}")
        lines.append("- Decision: keep / maybe / reject")
        lines.append("- Notes:")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_top_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Top Image Evidence Path Summary", ""]
    for idx, row in enumerate(rows[:20], start=1):
        primary = row.get("selected_primary_pair") or {}
        lines.append(f"## {idx}. image_id={row.get('image_id')} file={row.get('file_name')}")
        lines.append(f"- score={row.get('total_score', row.get('score'))} bucket={row.get('bucket_suggestion')}")
        if primary:
            lines.append(
                f"- primary_pair: h{primary.get('subject_id')} -> o{primary.get('object_id')} "
                f"{primary.get('object_category_name')} actions={primary.get('actions')} ctx={primary.get('context_count')}"
            )
            for ctx in primary.get("context_set", [])[:3]:
                lines.append(f"- path: P{primary.get('subject_id')} -> C{ctx.get('instance_id')}({ctx.get('category_name')}) -> O{primary.get('object_id')}")
        else:
            lines.append("- primary_pair: none")
        lines.append("")
    if not rows:
        lines.append("No top images available.")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheet and annotated previews for scored HOI candidates.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--bucket", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tile_width", type=int, default=420)
    parser.add_argument("--tile_height", type=int, default=430)
    parser.add_argument("--annotated_max_size", type=int, default=1200)
    parser.add_argument("--copy_raw", action="store_true", help="Copy raw images instead of writing an index file.")
    parser.add_argument("--make_pdf", action="store_true", help="Also save a PDF contact sheet when candidates are available.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    annotated_dir = output_dir / "annotated_images"
    crop_dir = output_dir / "crop_sheets"
    raw_dir = output_dir / "raw_selected_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates, warnings = load_candidates(args.candidates, args.bucket, args.top_k)
    font = load_font(16)
    small_font = load_font(12)
    tile_font = load_font(13)
    processed: list[dict[str, Any]] = []
    tiles: list[Image.Image] = []

    for rank, row in enumerate(candidates, start=1):
        image_path = resolve_image_path(row, args.image_root)
        if not image_path.exists():
            warnings.append(f"Skipped image_id={row.get('image_id')}: image path does not exist: {image_path}")
            continue
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            warnings.append(f"Skipped image_id={row.get('image_id')}: failed to open image `{image_path}`: {exc}")
            continue
        image_id = str(row.get("image_id"))
        annotated_path = annotated_dir / f"top_{rank:04d}_{image_id}.jpg"
        annotated, ann_warnings = annotate_image(image, row, annotated_path, args.annotated_max_size, font, small_font)
        warnings.extend(f"image_id={image_id}: {warning}" for warning in ann_warnings)
        crop_path = crop_dir / f"top_{rank:04d}_{image_id}_crops.jpg"
        make_crop_sheet(image, row, crop_path, small_font)
        raw_ref = save_raw_reference(image_path, raw_dir, rank, image_id, args.copy_raw)
        row["annotated_image_path"] = str(annotated_path)
        row["crop_sheet_path"] = str(crop_path) if crop_path.exists() else ""
        row["raw_image_reference"] = raw_ref
        processed.append(row)
        tiles.append(make_tile(annotated, row, args.tile_width, args.tile_height, font, tile_font))

    if tiles:
        cols = min(4, max(1, math.ceil(math.sqrt(len(tiles)))))
        rows_n = math.ceil(len(tiles) / cols)
        sheet = Image.new("RGB", (cols * args.tile_width, rows_n * args.tile_height), WHITE)
        for idx, tile in enumerate(tiles):
            x = (idx % cols) * args.tile_width
            y = (idx // cols) * args.tile_height
            sheet.paste(tile, (x, y))
        contact_path = output_dir / f"contact_sheet_top{args.top_k}.jpg"
        sheet.save(contact_path, quality=92)
        pdf_path = output_dir / f"contact_sheet_top{args.top_k}.pdf"
        if args.make_pdf:
            sheet.save(pdf_path, "PDF", resolution=120.0)
    else:
        contact_path = output_dir / f"contact_sheet_top{args.top_k}.jpg"
        placeholder = Image.new("RGB", (900, 240), WHITE)
        draw = ImageDraw.Draw(placeholder)
        draw.text((24, 24), "No candidates available for contact sheet.", font=font, fill=BLACK)
        draw.text((24, 58), f"Input: {args.candidates}", font=small_font, fill=GRAY)
        draw.text((24, 82), f"Bucket: {args.bucket}", font=small_font, fill=GRAY)
        for i, warning in enumerate(warnings[:5]):
            draw.text((24, 116 + i * 20), warning[:120], font=small_font, fill=GRAY)
        placeholder.save(contact_path, quality=92)
        pdf_path = None

    review_csv = output_dir / "review_table.csv"
    notes_path = output_dir / "review_notes_template.md"
    summary_path = output_dir / "top20_image_summary.md"
    warnings_path = output_dir / "warnings.json"
    write_review_table(review_csv, processed)
    write_notes_template(notes_path, processed, warnings)
    write_top_summary(summary_path, processed)
    warnings_path.write_text(json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"contact_sheet: {contact_path}")
    if args.make_pdf and pdf_path is not None:
        print(f"contact_sheet_pdf: {pdf_path}")
    print(f"annotated_images_dir: {annotated_dir}")
    print(f"review_table: {review_csv}")
    print("top_20_image_ids:", [row.get("image_id") for row in processed[:20]])
    print("top_20_summary_path:", summary_path)
    for idx, row in enumerate(processed[:20], start=1):
        primary = row.get("selected_primary_pair") or {}
        paths = []
        for ctx in primary.get("context_set", [])[:3]:
            paths.append(f"P{primary.get('subject_id')} -> C{ctx.get('instance_id')}({ctx.get('category_name')}) -> O{primary.get('object_id')}")
        print(
            f"{idx}. image_id={row.get('image_id')} primary=h{primary.get('subject_id')}->o{primary.get('object_id')} "
            f"{primary.get('object_category_name')} actions={primary.get('actions')} paths={paths}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
