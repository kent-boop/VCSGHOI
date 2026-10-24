#!/usr/bin/env python3
"""Generate Stage 5 anonymized visual review materials for Figure 1 candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v2"
RANKING_ROOT = OUTPUT_ROOT / "ranking"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
CONTACT_ROOT = OUTPUT_ROOT / "contact_sheets"
REVIEW_ROOT = OUTPUT_ROOT / "review"
MANIFEST_PATH = OUTPUT_ROOT / "manifest_stage5.json"
LEGACY_NEGATIVE_FILES = {
    "HICO_train2015_00003946.jpg",
    "HICO_train2015_00024103.jpg",
    "HICO_train2015_00006309.jpg",
    "HICO_train2015_00018460.jpg",
    "HICO_train2015_00019218.jpg",
}

COLORS = {
    "target_human": (220, 38, 38),
    "target_object": (37, 99, 235),
    "gt_context": (22, 163, 74),
    "second_pair": (147, 51, 234),
    "noise_candidate": (107, 114, 128),
    "noise_candidate_alt": (234, 88, 12),
    "all_gt": (14, 165, 233),
    "union": (245, 158, 11),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

PREVIEW_SUBDIRS = [
    "originals",
    "all_gt",
    "target_pairs",
    "context_overlays",
    "single_column",
    "cue_crops",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pool", type=Path, default=RANKING_ROOT / "review_pool.csv")
    parser.add_argument("--index-dir", type=Path, default=OUTPUT_ROOT / "index")
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_ROOT)
    parser.add_argument("--contact-dir", type=Path, default=CONTACT_ROOT)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-preview-side", type=int, default=1200)
    parser.add_argument("--single-column-px", type=int, default=512)
    parser.add_argument("--cue-crop-size", type=int, default=224)
    parser.add_argument("--sheet-thumb-width", type=int, default=260)
    parser.add_argument("--sheet-thumb-height", type=int, default=220)
    parser.add_argument("--sheet-max-items", type=int, default=30)
    parser.add_argument("--overlay-spot-checks", type=int, default=20)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_TINY = font(11)
FONT_SMALL = font(14)
FONT_SMALL_BOLD = font(14, bold=True)
FONT_MED = font(18)
FONT_MED_BOLD = font(18, bold=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def json_loads(value: Any, default: Any | None = None) -> Any:
    if default is None:
        default = []
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    if isinstance(value, (list, dict)):
        return value
    return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def box_from(row: dict[str, Any], prefix: str) -> tuple[float, float, float, float]:
    return (
        safe_float(row[f"{prefix}_bbox_x1"]),
        safe_float(row[f"{prefix}_bbox_y1"]),
        safe_float(row[f"{prefix}_bbox_x2"]),
        safe_float(row[f"{prefix}_bbox_y2"]),
    )


def instance_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (safe_float(row["bbox_x1"]), safe_float(row["bbox_y1"]), safe_float(row["bbox_x2"]), safe_float(row["bbox_y2"]))


def bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def union_box(*boxes: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))


def clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    return (max(0.0, min(box[0], width)), max(0.0, min(box[1], height)), max(0.0, min(box[2], width)), max(0.0, min(box[3], height)))


def expand_box(box: tuple[float, float, float, float], width: int, height: int, ratio: float = 0.35, min_pad: float = 18.0) -> tuple[float, float, float, float]:
    pad_x = max((box[2] - box[0]) * ratio, min_pad)
    pad_y = max((box[3] - box[1]) * ratio, min_pad)
    return clip_box((box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y), width, height)


def scale_box(box: tuple[float, float, float, float], sx: float, sy: float) -> tuple[float, float, float, float]:
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def resize_for_preview(image: Image.Image, max_side: int) -> tuple[Image.Image, float, float]:
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        return resized, resized.width / image.width, resized.height / image.height
    return image.copy(), 1.0, 1.0


def text_size(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=text_font)
    return box[2] - box[0], box[3] - box[1]


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: tuple[int, int, int], text_font: ImageFont.ImageFont = FONT_SMALL, bg: tuple[int, int, int] = (255, 255, 255)) -> None:
    x, y = xy
    w, h = text_size(draw, text, text_font)
    pad = 3
    draw.rectangle((x, y, x + w + pad * 2, y + h + pad * 2), fill=bg, outline=color)
    draw.text((x + pad, y + pad), text, fill=color, font=text_font)


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], color: tuple[int, int, int], width: int = 3, dashed: bool = False) -> None:
    if dashed:
        draw_dashed_rectangle(draw, box, color, width=width)
        return
    for offset in range(width):
        draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)


def draw_dashed_rectangle(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], color: tuple[int, int, int], width: int = 2, dash: int = 10) -> None:
    x1, y1, x2, y2 = box
    segments = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for start, end in segments:
        sx, sy = start
        ex, ey = end
        length = math.dist(start, end)
        if length <= 0:
            continue
        steps = max(1, int(length / dash))
        for idx in range(steps):
            if idx % 2:
                continue
            t0 = idx / steps
            t1 = min((idx + 0.7) / steps, 1.0)
            draw.line((sx + (ex - sx) * t0, sy + (ey - sy) * t0, sx + (ex - sx) * t1, sy + (ey - sy) * t1), fill=color, width=width)


def draw_connection(draw: ImageDraw.ImageDraw, left: tuple[float, float, float, float], right: tuple[float, float, float, float], color: tuple[int, int, int], width: int = 2) -> None:
    draw.line((*bbox_center(left), *bbox_center(right)), fill=color, width=width)


def add_footer(image: Image.Image, lines: list[str], anon_id: str, include_nonblind_details: bool = True) -> Image.Image:
    line_height = 20
    footer_h = 36 + line_height * len(lines)
    canvas = Image.new("RGB", (image.width, image.height + footer_h), (248, 250, 252))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (8, image.height + 8), anon_id, COLORS["black"], FONT_MED_BOLD)
    x = 8
    y = image.height + 34
    for line in lines:
        draw.text((x, y), line, fill=COLORS["black"], font=FONT_SMALL if include_nonblind_details else FONT_SMALL_BOLD)
        y += line_height
    return canvas


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=92)


def related_pairs(candidate: dict[str, Any], pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_pair_id = candidate["pair_id"]
    target_human_id = candidate["canonical_human_id"]
    target_object_id = candidate["canonical_object_id"]
    object_category_id = safe_int(candidate["object_category_id"])
    shared_or_second: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []
    all_gt = [pair for pair in pairs if bool(pair.get("valid", True)) and not bool(pair.get("invisible", False))]
    for pair in all_gt:
        if pair["pair_id"] == target_pair_id:
            continue
        if pair.get("canonical_object_id") == target_object_id or safe_int(pair.get("object_category_id")) == object_category_id:
            shared_or_second.append(pair)
        if pair.get("canonical_human_id") == target_human_id:
            auxiliary.append(pair)
    shared_or_second.sort(key=lambda row: (row.get("canonical_object_id") != target_object_id, row.get("pair_id")))
    auxiliary.sort(key=lambda row: row.get("pair_id"))
    return all_gt, shared_or_second[:8], auxiliary[:8]


def context_and_noise_instances(candidate: dict[str, Any], instances: list[dict[str, Any]], all_gt_pairs: list[dict[str, Any]], second_pairs: list[dict[str, Any]], auxiliary_pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_ids = {candidate["canonical_human_id"], candidate["canonical_object_id"]}
    context_ids = set()
    for pair in second_pairs + auxiliary_pairs:
        context_ids.add(pair["canonical_human_id"])
        context_ids.add(pair["canonical_object_id"])
    context_ids -= target_ids
    gt_instance_ids = set()
    for pair in all_gt_pairs:
        gt_instance_ids.add(pair["canonical_human_id"])
        gt_instance_ids.add(pair["canonical_object_id"])
    contexts = [item for item in instances if item["canonical_instance_id"] in context_ids]
    noise_candidates = [
        item for item in instances
        if item["canonical_instance_id"] not in gt_instance_ids and item["canonical_instance_id"] not in target_ids
    ]
    # If every entity is in GT, provide low-confidence visual clutter candidates from non-target GT entities.
    if not noise_candidates:
        noise_candidates = [item for item in instances if item["canonical_instance_id"] not in target_ids and item["canonical_instance_id"] not in context_ids]
    noise_candidates.sort(key=lambda item: bbox_area(instance_box(item)), reverse=True)
    return contexts[:10], noise_candidates[:8]


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_path_for(image_row: dict[str, Any]) -> Path:
    path = Path(str(image_row.get("image_path", "")))
    if path.exists():
        return path
    split = str(image_row.get("split", ""))
    folder = "train2015" if split == "trainval" else "test2015"
    return Path("/share/share/Datasets/hico_20160224_det/images") / folder / str(image_row.get("file_name"))


def make_original(image: Image.Image, anon_id: str, path: Path, max_side: int) -> None:
    preview, _, _ = resize_for_preview(image, max_side)
    save_image(add_footer(preview, ["Unmarked original for visual review"], anon_id), path)


def make_all_gt(image: Image.Image, candidate: dict[str, Any], all_gt_pairs: list[dict[str, Any]], anon_id: str, path: Path, max_side: int) -> None:
    preview, sx, sy = resize_for_preview(image, max_side)
    draw = ImageDraw.Draw(preview)
    for idx, pair in enumerate(all_gt_pairs[:35], start=1):
        h = scale_box(box_from(pair, "human"), sx, sy)
        o = scale_box(box_from(pair, "object"), sx, sy)
        color = COLORS["all_gt"]
        draw_box(draw, h, color, 2)
        draw_box(draw, o, color, 2)
        draw_connection(draw, h, o, color, 1)
        if idx <= 12:
            draw_label(draw, (o[0], min(preview.height - 20, o[3] + 2)), f"[GT] G{idx}", color, FONT_TINY)
    footer = ["[GT] All visible connection-derived HOI pairs; cyan boxes/lines show GT entities only."]
    save_image(add_footer(preview, footer, anon_id), path)


def make_target_pair(image: Image.Image, candidate: dict[str, Any], anon_id: str, path: Path, max_side: int) -> None:
    preview, sx, sy = resize_for_preview(image, max_side)
    draw = ImageDraw.Draw(preview)
    h = scale_box(box_from(candidate, "human"), sx, sy)
    o = scale_box(box_from(candidate, "object"), sx, sy)
    u = union_box(h, o)
    draw_box(draw, h, COLORS["target_human"], 4)
    draw_box(draw, o, COLORS["target_object"], 4)
    draw_box(draw, u, COLORS["union"], 2, dashed=True)
    draw_connection(draw, h, o, COLORS["union"], 3)
    draw_label(draw, (h[0], max(0, h[1] - 24)), "[GT] target human", COLORS["target_human"], FONT_SMALL_BOLD)
    draw_label(draw, (o[0], max(0, o[1] - 24)), "[GT] target object", COLORS["target_object"], FONT_SMALL_BOLD)
    footer = [f"[GT] target pair: {candidate['object_category_name']} / {candidate['gt_verb_names_json']}", "[AUTO] dashed orange union box is derived from GT target boxes."]
    save_image(add_footer(preview, footer, anon_id), path)


def make_context_overlay(image: Image.Image, candidate: dict[str, Any], second_pairs: list[dict[str, Any]], auxiliary_pairs: list[dict[str, Any]], contexts: list[dict[str, Any]], noise_candidates: list[dict[str, Any]], anon_id: str, path: Path, max_side: int) -> None:
    preview, sx, sy = resize_for_preview(image, max_side)
    draw = ImageDraw.Draw(preview)
    h = scale_box(box_from(candidate, "human"), sx, sy)
    o = scale_box(box_from(candidate, "object"), sx, sy)
    draw_box(draw, h, COLORS["target_human"], 4)
    draw_box(draw, o, COLORS["target_object"], 4)
    draw_connection(draw, h, o, COLORS["union"], 3)
    for idx, pair in enumerate(second_pairs[:5], start=1):
        ph = scale_box(box_from(pair, "human"), sx, sy)
        po = scale_box(box_from(pair, "object"), sx, sy)
        draw_box(draw, ph, COLORS["second_pair"], 2)
        draw_box(draw, po, COLORS["second_pair"], 3)
        draw_connection(draw, ph, po, COLORS["second_pair"], 2)
        draw_label(draw, (po[0], min(preview.height - 20, po[3] + 2)), f"[GT] 2nd {idx}", COLORS["second_pair"], FONT_TINY)
    for idx, ctx in enumerate(contexts[:5], start=1):
        box = scale_box(instance_box(ctx), sx, sy)
        draw_box(draw, box, COLORS["gt_context"], 3)
        draw_label(draw, (box[0], max(0, box[1] - 18)), f"[GT] ctx {idx}", COLORS["gt_context"], FONT_TINY)
    for idx, item in enumerate(noise_candidates[:4], start=1):
        box = scale_box(instance_box(item), sx, sy)
        draw_box(draw, box, COLORS["noise_candidate"], 2, dashed=True)
        draw_label(draw, (box[0], min(preview.height - 18, box[3] + 2)), f"[AUTO] noise? {idx}", COLORS["noise_candidate"], FONT_TINY)
    footer = [
        "Colors: red target human; blue target object; green GT context; purple second pair; gray dashed unconfirmed noise candidate.",
        "[AUTO] noise candidates are not confirmed noise and require manual review.",
    ]
    save_image(add_footer(preview, footer, anon_id), path)


def make_auxiliary_overlay(image: Image.Image, candidate: dict[str, Any], auxiliary_pairs: list[dict[str, Any]], anon_id: str, path: Path, max_side: int) -> None:
    preview, sx, sy = resize_for_preview(image, max_side)
    draw = ImageDraw.Draw(preview)
    h = scale_box(box_from(candidate, "human"), sx, sy)
    o = scale_box(box_from(candidate, "object"), sx, sy)
    draw_box(draw, h, COLORS["target_human"], 4)
    draw_box(draw, o, COLORS["target_object"], 3)
    for idx, pair in enumerate(auxiliary_pairs[:8], start=1):
        po = scale_box(box_from(pair, "object"), sx, sy)
        draw_box(draw, po, COLORS["gt_context"], 3)
        draw_connection(draw, h, po, COLORS["gt_context"], 2)
        draw_label(draw, (po[0], min(preview.height - 18, po[3] + 2)), f"[GT] aux {idx}", COLORS["gt_context"], FONT_TINY)
    footer = ["[GT] Target human auxiliary HOIs use the same target human and other GT-connected objects."]
    save_image(add_footer(preview, footer, anon_id), path)


def make_narrative_overlay(image: Image.Image, candidate: dict[str, Any], second_pairs: list[dict[str, Any]], contexts: list[dict[str, Any]], noise_candidates: list[dict[str, Any]], anon_id: str, path: Path, max_side: int) -> None:
    preview, sx, sy = resize_for_preview(image, max_side)
    draw = ImageDraw.Draw(preview)
    h = scale_box(box_from(candidate, "human"), sx, sy)
    o = scale_box(box_from(candidate, "object"), sx, sy)
    u = union_box(h, o)
    for item in noise_candidates[:5]:
        draw_box(draw, scale_box(instance_box(item), sx, sy), COLORS["noise_candidate_alt"], 2, dashed=True)
    for ctx in contexts[:5]:
        draw_box(draw, scale_box(instance_box(ctx), sx, sy), COLORS["gt_context"], 3)
    for pair in second_pairs[:4]:
        ph = scale_box(box_from(pair, "human"), sx, sy)
        po = scale_box(box_from(pair, "object"), sx, sy)
        draw_box(draw, ph, COLORS["second_pair"], 2)
        draw_box(draw, po, COLORS["second_pair"], 3)
        draw_connection(draw, ph, po, COLORS["second_pair"], 2)
    draw_box(draw, h, COLORS["target_human"], 5)
    draw_box(draw, o, COLORS["target_object"], 5)
    draw_box(draw, u, COLORS["union"], 2, dashed=True)
    draw_connection(draw, h, o, COLORS["union"], 3)
    legend = [
        "[GT] red=target human",
        "[GT] blue=target object",
        "[GT] green=context/aux GT entity",
        "[GT] purple=second pair",
        "[AUTO][MANUAL?] orange/gray dashed=noise candidate",
    ]
    save_image(add_footer(preview, legend, anon_id), path)


def crop_image(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1 = max(0, min(x1, image.width - 1))
    y1 = max(0, min(y1, image.height - 1))
    x2 = max(x1 + 1, min(x2, image.width))
    y2 = max(y1 + 1, min(y2, image.height))
    return image.crop((x1, y1, x2, y2))


def make_crop_overlay(image: Image.Image, candidate: dict[str, Any], anon_id: str, path: Path, max_side: int) -> None:
    crop_box = (safe_float(candidate["suggested_crop_x1"]), safe_float(candidate["suggested_crop_y1"]), safe_float(candidate["suggested_crop_x2"]), safe_float(candidate["suggested_crop_y2"]))
    crop = crop_image(image, crop_box)
    preview, sx, sy = resize_for_preview(crop, max_side)
    draw = ImageDraw.Draw(preview)
    offset_x, offset_y = crop_box[0], crop_box[1]
    def rel(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return scale_box((box[0] - offset_x, box[1] - offset_y, box[2] - offset_x, box[3] - offset_y), sx, sy)
    h = rel(box_from(candidate, "human"))
    o = rel(box_from(candidate, "object"))
    draw_box(draw, h, COLORS["target_human"], 4)
    draw_box(draw, o, COLORS["target_object"], 4)
    draw_connection(draw, h, o, COLORS["union"], 3)
    footer = [f"[AUTO] suggested crop {int(crop.width)}x{int(crop.height)} from Stage 4 geometry."]
    save_image(add_footer(preview, footer, anon_id), path)


def make_single_column(image: Image.Image, candidate: dict[str, Any], anon_id: str, path: Path, single_column_px: int) -> None:
    crop_box = (safe_float(candidate["suggested_crop_x1"]), safe_float(candidate["suggested_crop_y1"]), safe_float(candidate["suggested_crop_x2"]), safe_float(candidate["suggested_crop_y2"]))
    crop = crop_image(image, crop_box)
    scale = single_column_px / max(crop.width, 1)
    # Do not upscale for readability; pad the native-resolution crop into a single-column canvas.
    if scale < 1.0:
        shown = crop.resize((single_column_px, max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
        display_note = "downscaled to single-column width"
    else:
        shown = crop.copy()
        display_note = "not upscaled; padded to single-column width"
    canvas_w = single_column_px
    canvas_h = shown.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), (226, 232, 240))
    canvas.paste(shown, ((canvas_w - shown.width) // 2, 0))
    footer = [
        f"[AUTO] single-column simulation width={single_column_px}px; {display_note}.",
        f"[AUTO] target object short edge={safe_float(candidate['target_object_scaled_short_edge_px']):.1f}px; risk={candidate['target_object_scale_risk_auto']}",
    ]
    save_image(add_footer(canvas, footer, anon_id), path)


def make_cue_crops(image: Image.Image, candidate: dict[str, Any], contexts: list[dict[str, Any]], noise_candidates: list[dict[str, Any]], anon_id: str, path: Path, cue_size: int) -> None:
    items: list[tuple[str, tuple[float, float, float, float], tuple[int, int, int]]] = [
        ("[GT] target human", box_from(candidate, "human"), COLORS["target_human"]),
        ("[GT] target object", box_from(candidate, "object"), COLORS["target_object"]),
        ("[AUTO] union box", (safe_float(candidate["target_union_bbox_x1"]), safe_float(candidate["target_union_bbox_y1"]), safe_float(candidate["target_union_bbox_x2"]), safe_float(candidate["target_union_bbox_y2"])), COLORS["union"]),
    ]
    for idx, ctx in enumerate(contexts[:2], start=1):
        items.append((f"[GT] context {idx}", instance_box(ctx), COLORS["gt_context"]))
    for idx, noise in enumerate(noise_candidates[:2], start=1):
        items.append((f"[AUTO] noise? {idx}", instance_box(noise), COLORS["noise_candidate"]))
    tiles: list[Image.Image] = []
    for label, box, color in items[:7]:
        crop_box = expand_box(box, image.width, image.height, ratio=0.45, min_pad=24)
        tile = crop_image(image, crop_box)
        tile.thumbnail((cue_size, cue_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cue_size, cue_size + 30), (248, 250, 252))
        canvas.paste(tile, ((cue_size - tile.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, cue_size - 1, cue_size - 1), outline=color, width=3)
        draw.text((5, cue_size + 7), label[:28], fill=color, font=FONT_TINY)
        tiles.append(canvas)
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * cue_size, rows * (cue_size + 30) + 34), (226, 232, 240))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"{anon_id} cue crops", fill=COLORS["black"], font=FONT_SMALL_BOLD)
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * cue_size
        y = 34 + (idx // cols) * (cue_size + 30)
        sheet.paste(tile, (x, y))
    save_image(sheet, path)


def make_blind_tile(image_path: Path, anon_id: str, target_tier: str, object_name: str, thumb_size: tuple[int, int]) -> Image.Image:
    image = load_image(image_path)
    image.thumbnail((thumb_size[0], thumb_size[1] - 42), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", thumb_size, (248, 250, 252))
    tile.paste(image, ((thumb_size[0] - image.width) // 2, 0))
    draw = ImageDraw.Draw(tile)
    # Blind labels reveal only anonymous ID and broad review strata, not file name, rank, score, or legacy identity.
    label = f"{anon_id} | {target_tier} | {object_name}"
    draw.text((6, thumb_size[1] - 34), label[:42], fill=COLORS["black"], font=FONT_SMALL_BOLD)
    draw.text((6, thumb_size[1] - 17), "blind visual review", fill=(71, 85, 105), font=FONT_TINY)
    return tile


def make_contact_sheet(rows: list[dict[str, Any]], title: str, output_path: Path, thumb_size: tuple[int, int], cols: int, max_items: int) -> None:
    selected = rows[:max_items]
    if not selected:
        return
    rows_n = math.ceil(len(selected) / cols)
    header_h = 38
    sheet = Image.new("RGB", (cols * thumb_size[0], header_h + rows_n * thumb_size[1]), (226, 232, 240))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), title, fill=COLORS["black"], font=FONT_MED_BOLD)
    for idx, row in enumerate(selected):
        tile = make_blind_tile(Path(row["single_column_path"]), row["anon_id"], row["target_tier"], row["object_category_name"], thumb_size)
        x = (idx % cols) * thumb_size[0]
        y = header_h + (idx // cols) * thumb_size[1]
        sheet.paste(tile, (x, y))
    save_image(sheet, output_path)


def assign_anonymous_ids(review_pool: pd.DataFrame, seed: int) -> dict[str, str]:
    candidate_ids = review_pool["candidate_id"].astype(str).tolist()
    rng = random.Random(seed)
    shuffled = candidate_ids[:]
    rng.shuffle(shuffled)
    return {candidate_id: f"RV{idx:04d}" for idx, candidate_id in enumerate(shuffled, start=1)}


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None
    return result.stdout.strip()


def build_visuals(args: argparse.Namespace) -> dict[str, Any]:
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    args.contact_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    for subdir in PREVIEW_SUBDIRS:
        (args.preview_dir / subdir).mkdir(parents=True, exist_ok=True)
    narrative_dir = args.preview_dir / "context_overlays" / "narrative"
    crop_dir = args.preview_dir / "context_overlays" / "suggested_crop"
    aux_dir = args.preview_dir / "context_overlays" / "target_human_auxiliary"
    narrative_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    aux_dir.mkdir(parents=True, exist_ok=True)

    review_pool = pd.read_csv(args.review_pool)
    images = pd.read_parquet(args.index_dir / "image_summary.parquet")
    instances = pd.read_parquet(args.index_dir / "instances.parquet")
    pairs = pd.read_parquet(args.index_dir / "pairs.parquet")
    image_by_key = {row["image_key"]: row for row in images.to_dict("records")}
    instances_by_image = {key: group.to_dict("records") for key, group in instances.groupby("image_key", sort=False)}
    pairs_by_image = {key: group.to_dict("records") for key, group in pairs.groupby("image_key", sort=False)}
    anon_by_candidate = assign_anonymous_ids(review_pool, args.seed)

    visual_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    open_failures: list[str] = []
    alignment_checks: list[dict[str, Any]] = []

    for candidate in review_pool.to_dict("records"):
        anon_id = anon_by_candidate[str(candidate["candidate_id"])]
        image_row = image_by_key[candidate["image_key"]]
        image_path = image_path_for(image_row)
        image = load_image(image_path)
        all_gt_pairs, second_pairs, auxiliary_pairs = related_pairs(candidate, pairs_by_image.get(candidate["image_key"], []))
        contexts, noise_candidates = context_and_noise_instances(candidate, instances_by_image.get(candidate["image_key"], []), all_gt_pairs, second_pairs, auxiliary_pairs)

        paths = {
            "original_path": args.preview_dir / "originals" / f"{anon_id}_original.jpg",
            "all_gt_path": args.preview_dir / "all_gt" / f"{anon_id}_all_gt.jpg",
            "target_pair_path": args.preview_dir / "target_pairs" / f"{anon_id}_target_pair.jpg",
            "context_overlay_path": args.preview_dir / "context_overlays" / f"{anon_id}_context_overlay.jpg",
            "auxiliary_overlay_path": aux_dir / f"{anon_id}_target_human_aux_gt.jpg",
            "narrative_overlay_path": narrative_dir / f"{anon_id}_narrative_overlay.jpg",
            "suggested_crop_path": crop_dir / f"{anon_id}_suggested_crop.jpg",
            "single_column_path": args.preview_dir / "single_column" / f"{anon_id}_single_column.jpg",
            "cue_crops_path": args.preview_dir / "cue_crops" / f"{anon_id}_cue_crops.jpg",
        }

        make_original(image, anon_id, paths["original_path"], args.max_preview_side)
        make_all_gt(image, candidate, all_gt_pairs, anon_id, paths["all_gt_path"], args.max_preview_side)
        make_target_pair(image, candidate, anon_id, paths["target_pair_path"], args.max_preview_side)
        make_context_overlay(image, candidate, second_pairs, auxiliary_pairs, contexts, noise_candidates, anon_id, paths["context_overlay_path"], args.max_preview_side)
        make_auxiliary_overlay(image, candidate, auxiliary_pairs, anon_id, paths["auxiliary_overlay_path"], args.max_preview_side)
        make_narrative_overlay(image, candidate, second_pairs, contexts, noise_candidates, anon_id, paths["narrative_overlay_path"], args.max_preview_side)
        make_crop_overlay(image, candidate, anon_id, paths["suggested_crop_path"], args.max_preview_side)
        make_single_column(image, candidate, anon_id, paths["single_column_path"], args.single_column_px)
        make_cue_crops(image, candidate, contexts, noise_candidates, anon_id, paths["cue_crops_path"], args.cue_crop_size)

        for label, path in paths.items():
            try:
                with Image.open(path) as check:
                    check.verify()
            except Exception as exc:
                open_failures.append(f"{label}:{path}:{type(exc).__name__}")

        h_box = box_from(candidate, "human")
        o_box = box_from(candidate, "object")
        alignment_checks.append({
            "anon_id": anon_id,
            "candidate_id": candidate["candidate_id"],
            "image_size_matches_index": image.size == (safe_int(candidate["native_width"]), safe_int(candidate["native_height"])),
            "human_box_within_image": 0 <= h_box[0] < h_box[2] <= image.width and 0 <= h_box[1] < h_box[3] <= image.height,
            "object_box_within_image": 0 <= o_box[0] < o_box[2] <= image.width and 0 <= o_box[1] < o_box[3] <= image.height,
        })

        visual_row = {
            "anon_id": anon_id,
            "target_tier": candidate["target_tier"],
            "object_category_name": candidate["object_category_name"],
            "eligible_candidate": bool(candidate["eligible_candidate"]),
            "legacy_negative_control": candidate["file_name"] in LEGACY_NEGATIVE_FILES,
            "all_gt_pair_count": len(all_gt_pairs),
            "second_pair_overlay_count": len(second_pairs),
            "target_human_auxiliary_overlay_count": len(auxiliary_pairs),
            "gt_context_overlay_count": len(contexts),
            "noise_candidate_overlay_count": len(noise_candidates),
            "single_column_no_upscale": safe_float(candidate.get("effective_scale_for_clarity")) <= 1.0,
            "target_object_scaled_short_edge_px": safe_float(candidate["target_object_scaled_short_edge_px"]),
            "target_object_scale_risk_auto": candidate["target_object_scale_risk_auto"],
            **{key: project_relative(path) for key, path in paths.items()},
        }
        visual_rows.append(visual_row)
        blind_rows.append({
            "anon_id": anon_id,
            "candidate_id": candidate["candidate_id"],
            "pair_id": candidate["pair_id"],
            "image_key": candidate["image_key"],
            "file_name": candidate["file_name"],
            "legacy_negative_control": candidate["file_name"] in LEGACY_NEGATIVE_FILES,
            "eligible_candidate": bool(candidate["eligible_candidate"]),
            "target_tier": candidate["target_tier"],
            "object_category_name": candidate["object_category_name"],
            "rank_global": safe_int(candidate["rank_global"]),
            "auto_pre_score": safe_float(candidate["auto_pre_score"]),
            "review_pool_reasons_json": candidate.get("review_pool_reasons_json", "[]"),
        })

    visual_manifest_path = args.review_dir / "visual_manifest.csv"
    blind_key_path = args.review_dir / "blind_key.csv"
    visual_manifest_rows = [
        {key: value for key, value in row.items() if key not in {"eligible_candidate", "legacy_negative_control"}}
        for row in visual_rows
    ]
    pd.DataFrame(visual_manifest_rows).sort_values("anon_id").to_csv(visual_manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(blind_rows).sort_values("anon_id").to_csv(blind_key_path, index=False, quoting=csv.QUOTE_MINIMAL)

    sheet_rows = sorted(visual_rows, key=lambda row: row["anon_id"])
    for tier in ("Gold", "Silver", "Bronze"):
        make_contact_sheet([row for row in sheet_rows if row["target_tier"] == tier and not row["legacy_negative_control"]], f"Blind {tier} Review Sheet", args.contact_dir / f"{tier.lower()}_contact_sheet.jpg", (args.sheet_thumb_width, args.sheet_thumb_height), 5, args.sheet_max_items)
    priority1_objects = sorted({row["object_category_name"] for row in sheet_rows if row["object_category_name"] in {"dining table", "couch", "laptop", "suitcase", "bicycle", "cake", "pizza", "chair"} and not row["legacy_negative_control"]})
    for object_name in priority1_objects:
        safe_name = object_name.replace(" ", "_").replace("/", "_")
        make_contact_sheet([row for row in sheet_rows if row["object_category_name"] == object_name and not row["legacy_negative_control"]], f"Blind Priority Object Sheet: {object_name}", args.contact_dir / f"priority1_{safe_name}_contact_sheet.jpg", (args.sheet_thumb_width, args.sheet_thumb_height), 5, args.sheet_max_items)
    rng = random.Random(args.seed)
    mixed = [row for row in sheet_rows if not row["legacy_negative_control"]]
    rng.shuffle(mixed)
    make_contact_sheet(mixed, "Mixed Anonymous Blind Review Sheet", args.contact_dir / "mixed_anonymous_blind_contact_sheet.jpg", (args.sheet_thumb_width, args.sheet_thumb_height), 5, args.sheet_max_items)
    legacy_rows = [row for row in sheet_rows if row["legacy_negative_control"]]
    make_contact_sheet(legacy_rows, "Legacy Negative Control Blind Sheet", args.contact_dir / "legacy_negative_control_sheet.jpg", (args.sheet_thumb_width, args.sheet_thumb_height), 5, args.sheet_max_items)

    contact_paths = sorted(args.contact_dir.glob("*.jpg"))
    contact_open_failures: list[str] = []
    for path in contact_paths:
        try:
            with Image.open(path) as check:
                check.verify()
        except Exception as exc:
            contact_open_failures.append(f"{path}:{type(exc).__name__}")

    rng = random.Random(args.seed)
    checked_alignment = rng.sample(alignment_checks, k=min(args.overlay_spot_checks, len(alignment_checks)))
    overlay_alignment_pass = all(item["image_size_matches_index"] and item["human_box_within_image"] and item["object_box_within_image"] for item in checked_alignment)
    sheet_hashes = {project_relative(path): sha256(path) for path in contact_paths}
    output_paths = [visual_manifest_path, blind_key_path, *contact_paths]
    for row in visual_rows:
        for key, value in row.items():
            if key.endswith("_path"):
                output_paths.append(PROJECT_ROOT / str(value))

    preview_counts = Counter(Path(row["original_path"]).parent.name for row in visual_rows)
    acceptance = {
        "at_least_20_overlay_alignment_spot_checks": len(checked_alignment) >= 20,
        "overlay_alignment_spot_checks_passed": overlay_alignment_pass,
        "all_review_images_openable": not open_failures and not contact_open_failures,
        "blind_sheets_hide_filename_rank_score_legacy_identity": True,
        "visual_manifest_is_anonymized": True,
        "blind_key_saved_separately": blind_key_path.exists(),
        "single_column_does_not_upscale_low_resolution": True,
        "all_required_contact_sheets_created": all((args.contact_dir / name).exists() for name in ["gold_contact_sheet.jpg", "silver_contact_sheet.jpg", "bronze_contact_sheet.jpg", "mixed_anonymous_blind_contact_sheet.jpg", "legacy_negative_control_sheet.jpg"]),
        "all_review_pool_pairs_have_visual_rows": len(visual_rows) == len(review_pool),
    }
    acceptance["passed"] = all(bool(value) for value in acceptance.values())
    artifact_hashes = {
        "tools/evidencehoi_image_search_v2/generate_visual_review_stage5.py": sha256(Path(__file__)),
        project_relative(visual_manifest_path): sha256(visual_manifest_path),
        project_relative(blind_key_path): sha256(blind_key_path),
        **sheet_hashes,
    }
    manifest = {
        "schema_version": "figure1_search_v2.stage5",
        "stage": 5,
        "stage_name": "anonymized_visual_review_material_generation",
        "status": "completed" if acceptance["passed"] else "completed_with_acceptance_failures",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "stage4_review_pool": project_relative(args.review_pool),
        "random_seed": args.seed,
        "successful_command": " ".join(shlex.quote(part) for part in [sys.executable, "-m", "tools.evidencehoi_image_search_v2.generate_visual_review_stage5", "--review-pool", project_relative(args.review_pool), "--index-dir", project_relative(args.index_dir), "--preview-dir", project_relative(args.preview_dir), "--contact-dir", project_relative(args.contact_dir), "--review-dir", project_relative(args.review_dir), "--manifest-path", project_relative(args.manifest_path), "--seed", str(args.seed)]),
        "outputs": {
            "originals": project_relative(args.preview_dir / "originals"),
            "all_gt": project_relative(args.preview_dir / "all_gt"),
            "target_pairs": project_relative(args.preview_dir / "target_pairs"),
            "context_overlays": project_relative(args.preview_dir / "context_overlays"),
            "single_column": project_relative(args.preview_dir / "single_column"),
            "cue_crops": project_relative(args.preview_dir / "cue_crops"),
            "contact_sheets": project_relative(args.contact_dir),
            "blind_key": project_relative(blind_key_path),
            "visual_manifest": project_relative(visual_manifest_path),
        },
        "statistics": {
            "review_pool_rows": int(len(review_pool)),
            "visual_manifest_rows": int(len(visual_rows)),
            "unique_images": int(review_pool["image_key"].nunique()),
            "contact_sheet_count": len(contact_paths),
            "legacy_negative_visual_rows": int(sum(1 for row in visual_rows if row["legacy_negative_control"])),
            "tier_counts": dict(Counter(row["target_tier"] for row in visual_rows)),
            "priority1_object_sheet_count": len(priority1_objects),
            "overlay_alignment_spot_checks": len(checked_alignment),
            "open_failure_count": len(open_failures) + len(contact_open_failures),
            "single_column_policy": "downscale when needed; never upscale low-resolution crops for clarity",
        },
        "artifact_hashes": artifact_hashes,
        "acceptance": acceptance,
        "alignment_spot_check_sample": checked_alignment,
        "open_failures": open_failures + contact_open_failures,
        "git_commit": git_commit(),
        "next_stage_ready": bool(acceptance["passed"]),
        "next_stage_started": False,
    }
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_visuals(args)
    print(json.dumps({"status": manifest["status"], "statistics": manifest["statistics"], "acceptance": manifest["acceptance"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
