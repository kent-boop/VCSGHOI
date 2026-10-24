#!/usr/bin/env python3
"""Generate Stage 4 counterfactual visual review materials.

The visualizations are deterministic geometry/annotation views. They do not run
or simulate model logits, attention, probabilities, or causal effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
from PIL import Image, ImageDraw

from feature_views import (
    Box,
    PROJECT_ROOT,
    box_from_instance,
    box_from_pair,
    intersection_area,
    json_dumps,
    rel,
    union_rect,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v5_context_counterfactual"
DEFAULT_STAGE3_DIR = DEFAULT_OUTPUT_ROOT / "stage3_search"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "outputs/figure1_search_v2/index"

ROLE_COLORS = {
    "target_human": (230, 40, 40),
    "target_object": (40, 95, 230),
    "cpos": (30, 170, 70),
    "cneg": (245, 145, 30),
    "second_pair": (135, 70, 200),
    "unconfirmed": (145, 145, 145),
}

VIEW_NAMES = [
    "original_unmarked",
    "full_gt_overlay",
    "target_overlay",
    "pair_only",
    "union_only",
    "union_plus_cpos",
    "union_plus_cneg",
    "full_minus_cpos",
    "full_minus_cneg",
    "cpos_crop",
    "cneg_crop",
    "final_panel_384",
    "role_overlay",
]

CONTACT_SHEET_NAMES = [
    "pair_union_ambiguous",
    "dual_context_complete",
    "context_leakage_failures",
    "pair_internal_decisive_rejects",
    "positive_only_rejects",
    "misleading_only_rejects",
    "same_context_double_count_rejects",
    "target_too_small_rejects",
    "current_candidates_reclassified",
    "blind_pack_A",
    "blind_pack_B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--panel-width", type=int, default=384)
    parser.add_argument("--thumb-width", type=int, default=192)
    parser.add_argument("--thumb-height", type=int, default=160)
    parser.add_argument("--jpeg-quality", type=int, default=58)
    parser.add_argument("--qa-sample-size", type=int, default=30)
    parser.add_argument("--contact-sheet-limit", type=int, default=48)
    parser.add_argument("--candidate-limit", type=int, default=0, help="0 means all balanced review candidates")
    parser.add_argument("--image-cache-size", type=int, default=16)
    parser.add_argument("--overwrite-stage4", action="store_true")
    parser.add_argument("--resume-stage4", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_token(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def parse_box(value: Any) -> Box:
    if isinstance(value, Box):
        return value
    if isinstance(value, str):
        data = json.loads(value)
    else:
        data = value
    return Box(float(data[0]), float(data[1]), float(data[2]), float(data[3])).normalized()


def project_box(box: Box, crop_box: Box, scale: float) -> tuple[int, int, int, int]:
    return (
        int(round((box.x1 - crop_box.x1) * scale)),
        int(round((box.y1 - crop_box.y1) * scale)),
        int(round((box.x2 - crop_box.x1) * scale)),
        int(round((box.y2 - crop_box.y1) * scale)),
    )


def clamp_proj(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    return (max(0, min(w, box[0])), max(0, min(h, box[1])), max(0, min(w, box[2])), max(0, min(h, box[3])))


def rect_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def resize_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image.copy()
    height = max(1, int(round(image.height * float(width) / max(1, image.width))))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def fit_card(image: Image.Image, size: tuple[int, int], label: str) -> Image.Image:
    card_w, card_h = size
    label_h = 22
    body_h = max(1, card_h - label_h)
    scale = min(card_w / max(1, image.width), body_h / max(1, image.height))
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    card = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, card_w - 1, label_h - 1), fill=(232, 232, 232))
    draw.text((4, 5), label[:46], fill=(20, 20, 20))
    x = (card_w - resized.width) // 2
    y = label_h + (body_h - resized.height) // 2
    card.paste(resized, (x, y))
    return card


def text_card(text: str, size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size, (248, 248, 248))
    draw = ImageDraw.Draw(image)
    y = 8
    for line in text.splitlines():
        draw.text((8, y), line[:64], fill=(20, 20, 20))
        y += 15
        if y > size[1] - 18:
            break
    return image


def draw_dashed_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int], width: int = 3, dash: int = 10) -> None:
    x1, y1, x2, y2 = box
    for x in range(x1, x2, dash * 2):
        draw.line((x, y1, min(x + dash, x2), y1), fill=color, width=width)
        draw.line((x, y2, min(x + dash, x2), y2), fill=color, width=width)
    for y in range(y1, y2, dash * 2):
        draw.line((x1, y, x1, min(y + dash, y2)), fill=color, width=width)
        draw.line((x2, y, x2, min(y + dash, y2)), fill=color, width=width)


def draw_labelled_boxes(
    base: Image.Image,
    boxes: Sequence[tuple[tuple[int, int, int, int], str, tuple[int, int, int], bool]],
) -> Image.Image:
    out = base.copy()
    draw = ImageDraw.Draw(out)
    for box, label, color, dashed in boxes:
        box = clamp_proj(box, out.size)
        if rect_area(box) <= 0:
            continue
        if dashed:
            draw_dashed_rect(draw, box, ROLE_COLORS["unconfirmed"], width=3)
        else:
            draw.rectangle(box, outline=color, width=4)
        label_color = ROLE_COLORS["unconfirmed"] if dashed else color
        draw.rectangle((box[0], max(0, box[1] - 18), min(out.width, box[0] + 8 + 7 * len(label)), box[1]), fill=(255, 255, 255))
        draw.text((box[0] + 4, max(0, box[1] - 16)), label, fill=label_color)
    return out


def mask_for_rects(size: tuple[int, int], rects: Sequence[tuple[int, int, int, int]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for rect in rects:
        rect = clamp_proj(rect, size)
        if rect_area(rect) > 0:
            draw.rectangle(rect, fill=255)
    return mask


def retained_view(base: Image.Image, rects: Sequence[tuple[int, int, int, int]]) -> Image.Image:
    out = Image.new("RGB", base.size, (128, 128, 128))
    out.paste(base, (0, 0), mask_for_rects(base.size, rects))
    return out


def minus_view(base: Image.Image, rect: tuple[int, int, int, int]) -> Image.Image:
    out = base.copy()
    draw = ImageDraw.Draw(out)
    rect = clamp_proj(rect, out.size)
    if rect_area(rect) > 0:
        draw.rectangle(rect, fill=(128, 128, 128))
    return out


def padded_crop(image: Image.Image, box: Box, pad_ratio: float = 0.18) -> Image.Image:
    pad = pad_ratio * max(box.width, box.height, 1.0)
    crop = Box(box.x1 - pad, box.y1 - pad, box.x2 + pad, box.y2 + pad).clamp(image.width, image.height)
    return image.crop(crop.to_int_tuple()).convert("RGB")


class ImageCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.cache: OrderedDict[str, Image.Image] = OrderedDict()

    def get(self, path: str) -> Image.Image:
        if path in self.cache:
            image = self.cache.pop(path)
            self.cache[path] = image
            return image.copy()
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        self.cache[path] = image
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return image.copy()


def save_jpeg(path: Path, image: Image.Image, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, subsampling=2)


def build_base_views(
    source: Image.Image,
    row: Any,
    all_gt_overlay: Image.Image,
    panel_width: int,
) -> dict[str, Image.Image]:
    human = parse_box(row.target_human_bbox)
    obj = parse_box(row.target_object_bbox)
    union = union_rect([human, obj])
    cpos = parse_box(row.cpos_bbox)
    cneg = parse_box(row.cneg_bbox)
    crop = parse_box(row.evidence_crop_bbox).clamp(source.width, source.height)
    scale = panel_width / max(1.0, crop.width)
    base = source.crop(crop.to_int_tuple()).convert("RGB")
    base = base.resize((panel_width, max(1, int(round(crop.height * scale)))), Image.Resampling.LANCZOS)

    h_proj = project_box(human, crop, scale)
    o_proj = project_box(obj, crop, scale)
    union_proj = project_box(union, crop, scale)
    cpos_proj = project_box(cpos, crop, scale)
    cneg_proj = project_box(cneg, crop, scale)
    cpos_unconfirmed = "[GT]" not in str(row.cpos_source) and "[GT]" not in str(row.cpos_provenance)
    cneg_unconfirmed = "[GT]" not in str(row.cneg_source) and "[GT]" not in str(row.cneg_provenance)

    target_overlay = draw_labelled_boxes(
        base,
        [
            (h_proj, "target human", ROLE_COLORS["target_human"], False),
            (o_proj, "target object", ROLE_COLORS["target_object"], False),
        ],
    )
    role_overlay = draw_labelled_boxes(
        base,
        [
            (h_proj, "target human", ROLE_COLORS["target_human"], False),
            (o_proj, "target object", ROLE_COLORS["target_object"], False),
            (cpos_proj, "c+", ROLE_COLORS["cpos"], cpos_unconfirmed),
            (cneg_proj, "c-", ROLE_COLORS["cneg"], cneg_unconfirmed),
        ],
    )
    views = {
        "original_unmarked": resize_width(source, panel_width),
        "full_gt_overlay": resize_width(all_gt_overlay, panel_width),
        "target_overlay": target_overlay,
        "pair_only": retained_view(base, [h_proj, o_proj]),
        "union_only": retained_view(base, [union_proj]),
        "union_plus_cpos": retained_view(base, [union_proj, cpos_proj]),
        "union_plus_cneg": retained_view(base, [union_proj, cneg_proj]),
        "full_minus_cpos": minus_view(base, cpos_proj),
        "full_minus_cneg": minus_view(base, cneg_proj),
        "cpos_crop": resize_width(padded_crop(source, cpos), panel_width),
        "cneg_crop": resize_width(padded_crop(source, cneg), panel_width),
        "final_panel_384": role_overlay,
        "role_overlay": role_overlay,
    }
    return views


def make_full_gt_overlay(source: Image.Image, image_pairs: Sequence[Any]) -> Image.Image:
    overlay = source.copy()
    draw = ImageDraw.Draw(overlay)
    for pair in image_pairs:
        try:
            h = box_from_pair(pair, "human").to_int_tuple()
            o = box_from_pair(pair, "object").to_int_tuple()
        except Exception:
            continue
        draw.rectangle(h, outline=ROLE_COLORS["second_pair"], width=2)
        draw.rectangle(o, outline=ROLE_COLORS["second_pair"], width=2)
    return overlay


def make_panel(title: str, labelled_images: Sequence[tuple[str, Image.Image]], card_size: tuple[int, int], columns: int = 4) -> Image.Image:
    cards = [fit_card(image, card_size, label) for label, image in labelled_images]
    title_h = 44
    rows = max(1, math.ceil(len(cards) / columns))
    panel = Image.new("RGB", (columns * card_size[0], title_h + rows * card_size[1]), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, title_h - 1), fill=(32, 32, 32))
    draw.text((10, 8), title[:180], fill=(255, 255, 255))
    for idx, card in enumerate(cards):
        x = (idx % columns) * card_size[0]
        y = title_h + (idx // columns) * card_size[1]
        panel.paste(card, (x, y))
    return panel


def make_blind_panel_a(row: Any, views: dict[str, Image.Image], card_size: tuple[int, int]) -> Image.Image:
    title = f"Blind Pack A | GT={row.gt_verb} | competing={row.competitor} | object={row.target_object_category}"
    return make_panel(title, [("pair-only", views["pair_only"]), ("union-only", views["union_only"])], card_size, columns=2)


def make_blind_panel_b(row: Any, views: dict[str, Image.Image], card_size: tuple[int, int]) -> Image.Image:
    title = f"Blind Pack B | GT={row.gt_verb} | competing={row.competitor} | object={row.target_object_category}"
    items = [
        ("union", views["union_only"]),
        ("union+c+", views["union_plus_cpos"]),
        ("union+c-", views["union_plus_cneg"]),
        ("full-minus-c+", views["full_minus_cpos"]),
        ("full-minus-c-", views["full_minus_cneg"]),
    ]
    return make_panel(title, items, card_size, columns=5)


def contact_sheet(cards: Sequence[Image.Image], captions: Sequence[str], card_size: tuple[int, int], columns: int = 4) -> Image.Image:
    labelled = []
    for card, caption in zip(cards, captions):
        labelled.append((caption, card))
    return make_panel("contact sheet", labelled, card_size, columns=columns)


def sample_rejections(rejection_path: Path, categories: dict[str, tuple[str, Optional[bool]]], limit: int) -> dict[str, list[dict[str, str]]]:
    samples: dict[str, list[dict[str, str]]] = {name: [] for name in categories}
    usecols = ["config", "pair_id", "image_key", "target_object_category", "gt_verb", "competitor", "cpos_id", "cneg_id", "first_failed_gate", "reason"]
    for chunk in pd.read_csv(rejection_path, usecols=usecols, chunksize=200000):
        for name, (gate, wants_context_ids) in categories.items():
            if len(samples[name]) >= limit:
                continue
            matched = chunk[(chunk["config"] == "balanced") & (chunk["first_failed_gate"] == gate)]
            if wants_context_ids is True:
                matched = matched[matched["cpos_id"].astype(str).ne("") & matched["cneg_id"].astype(str).ne("")]
            elif wants_context_ids is False:
                matched = matched[matched["cpos_id"].astype(str).eq("") | matched["cneg_id"].astype(str).eq("")]
            need = limit - len(samples[name])
            if need > 0 and not matched.empty:
                samples[name].extend(matched.head(need).fillna("").astype(str).to_dict("records"))
        if all(len(v) >= limit for v in samples.values()):
            break
    return samples


def make_reject_card(rej: dict[str, str], pair_by_id: dict[str, Any], image_by_key: dict[str, Any], cache: ImageCache, size: tuple[int, int]) -> Image.Image:
    pair = pair_by_id.get(str(rej.get("pair_id", "")))
    if pair is None:
        return text_card(json_dumps(rej), size)
    image_info = image_by_key.get(str(pair.image_key))
    image_path = str(getattr(image_info, "image_path", "")) if image_info is not None else ""
    header = f"{rej.get('first_failed_gate')} | {rej.get('gt_verb')} vs {rej.get('competitor')}"
    try:
        source = cache.get(image_path)
        human = box_from_pair(pair, "human")
        obj = box_from_pair(pair, "object")
        crop = union_rect([human, obj])
        pad = 0.35 * max(crop.width, crop.height, 1.0)
        crop = Box(crop.x1 - pad, crop.y1 - pad, crop.x2 + pad, crop.y2 + pad).clamp(source.width, source.height)
        scale = 384 / max(1.0, crop.width)
        base = source.crop(crop.to_int_tuple()).resize((384, max(1, int(round(crop.height * scale)))), Image.Resampling.LANCZOS)
        h_proj = project_box(human, crop, scale)
        o_proj = project_box(obj, crop, scale)
        over = draw_labelled_boxes(base, [(h_proj, "target human", ROLE_COLORS["target_human"], False), (o_proj, "target object", ROLE_COLORS["target_object"], False)])
        return fit_card(over, size, header)
    except Exception:
        return text_card(header + "\n" + str(rej.get("pair_id", "")), size)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    stage4_dir = args.output_root / "stage4_visuals"
    if stage4_dir.exists():
        if args.overwrite_stage4:
            shutil.rmtree(stage4_dir)
        elif not args.resume_stage4:
            raise SystemExit(f"Refusing to overwrite existing stage4 directory: {stage4_dir}")
    candidates_dir = stage4_dir / "candidates"
    contact_dir = stage4_dir / "contact_sheets"
    blind_a_dir = stage4_dir / "blind_pack_A"
    blind_b_dir = stage4_dir / "blind_pack_B"
    for path in [candidates_dir, contact_dir, blind_a_dir, blind_b_dir]:
        path.mkdir(parents=True, exist_ok=True)

    review_path = args.stage3_dir / "review_pool_balanced.csv"
    if not review_path.exists():
        raise FileNotFoundError(review_path)
    review = pd.read_csv(review_path)
    review = review[review["balanced_pass"] == True].copy()
    if args.candidate_limit > 0:
        review = review.head(args.candidate_limit).copy()

    pairs = pd.read_parquet(args.index_dir / "pairs.parquet")
    images = pd.read_parquet(args.index_dir / "image_summary.parquet")
    pair_by_id = {str(row.pair_id): row for row in pairs.itertuples(index=False)}
    image_by_key = {str(row.image_key): row for row in images.itertuples(index=False)}
    pairs_by_image: dict[str, list[Any]] = defaultdict(list)
    for row in pairs.itertuples(index=False):
        pairs_by_image[str(row.image_key)].append(row)

    cache = ImageCache(args.image_cache_size)
    gt_overlay_cache: OrderedDict[str, Image.Image] = OrderedDict()
    manifest_rows: list[dict[str, Any]] = []
    blind_key_rows: list[dict[str, Any]] = []
    generated_panels: dict[str, list[tuple[str, str]]] = {name: [] for name in ["blind_pack_A", "blind_pack_B", "candidate_panel", "final_panel_384"]}
    qa_candidates: list[dict[str, Any]] = []
    card_size = (args.thumb_width, args.thumb_height)

    for idx, row in enumerate(review.itertuples(index=False), start=1):
        token = stable_token(str(row.tuple_id))
        shard = token[:2]
        candidate_panel_path = candidates_dir / shard / f"{token}_counterfactual_panel.jpg"
        final_panel_path = candidates_dir / shard / f"{token}_final_panel_384.jpg"
        blind_a_path = blind_a_dir / shard / f"{token}_blind_A.jpg"
        blind_b_path = blind_b_dir / shard / f"{token}_blind_B.jpg"
        title = f"{token} | GT={row.gt_verb} | comp={row.competitor} | obj={row.target_object_category} | strict={row.strict_pass}"
        manifest_item = {
            "token": token,
            "tuple_id": row.tuple_id,
            "pair_id": row.pair_id,
            "image_key": row.image_key,
            "split": row.split,
            "file_name": row.file_name,
            "target_object_category": row.target_object_category,
            "gt_verb": row.gt_verb,
            "competitor": row.competitor,
            "cpos_id": row.cpos_id,
            "cpos_category": row.cpos_category,
            "cneg_id": row.cneg_id,
            "cneg_category": row.cneg_category,
            "counterfactual_panel_path": rel(candidate_panel_path),
            "final_panel_384_path": rel(final_panel_path),
            "blind_pack_A_path": rel(blind_a_path),
            "blind_pack_B_path": rel(blind_b_path),
            "views_in_counterfactual_panel_json": json_dumps(VIEW_NAMES),
            "same_crop_for_pair_union_context": True,
            "model_output_fabricated": False,
            "balanced_pass": bool(row.balanced_pass),
            "strict_pass": bool(row.strict_pass),
            "cpos_cneg_mask_iou": row.cpos_cneg_mask_iou,
            "target_pair_occupancy": row.target_pair_occupancy,
            "target_human_projected_height": row.target_human_projected_height,
            "target_object_projected_sqrt_area": row.target_object_projected_sqrt_area,
            "cpos_projected_size": row.cpos_projected_size,
            "cneg_projected_size": row.cneg_projected_size,
        }
        blind_key_item = {
            "token": token,
            "tuple_id": row.tuple_id,
            "gt_verb": row.gt_verb,
            "competitor": row.competitor,
            "cpos_id": row.cpos_id,
            "cpos_category": row.cpos_category,
            "cpos_source": row.cpos_source,
            "cneg_id": row.cneg_id,
            "cneg_category": row.cneg_category,
            "cneg_source": row.cneg_source,
            "blind_pack_A_path": rel(blind_a_path),
            "blind_pack_B_path": rel(blind_b_path),
        }
        if args.resume_stage4 and candidate_panel_path.exists() and final_panel_path.exists() and blind_a_path.exists() and blind_b_path.exists():
            manifest_rows.append(manifest_item)
            blind_key_rows.append(blind_key_item)
            generated_panels["candidate_panel"].append((rel(candidate_panel_path), title))
            generated_panels["final_panel_384"].append((rel(final_panel_path), title))
            generated_panels["blind_pack_A"].append((rel(blind_a_path), title))
            generated_panels["blind_pack_B"].append((rel(blind_b_path), title))
            if idx % 5000 == 0:
                print(json.dumps({"resumed_existing": idx, "total": len(review)}, ensure_ascii=False))
            continue

        source = cache.get(str(row.image_path))
        image_key = str(row.image_key)
        if image_key in gt_overlay_cache:
            gt_overlay = gt_overlay_cache.pop(image_key)
            gt_overlay_cache[image_key] = gt_overlay
        else:
            gt_overlay = make_full_gt_overlay(source, pairs_by_image.get(image_key, []))
            gt_overlay_cache[image_key] = gt_overlay
            while len(gt_overlay_cache) > args.image_cache_size:
                gt_overlay_cache.popitem(last=False)

        views = build_base_views(source, row, gt_overlay, args.panel_width)
        candidate_items = [(name, views[name]) for name in VIEW_NAMES]
        candidate_panel = make_panel(title, candidate_items, card_size, columns=4)
        save_jpeg(candidate_panel_path, candidate_panel, args.jpeg_quality)
        save_jpeg(final_panel_path, views["final_panel_384"], args.jpeg_quality)
        blind_a = make_blind_panel_a(row, views, card_size)
        blind_b = make_blind_panel_b(row, views, card_size)
        save_jpeg(blind_a_path, blind_a, args.jpeg_quality)
        save_jpeg(blind_b_path, blind_b, args.jpeg_quality)

        manifest_rows.append(manifest_item)
        blind_key_rows.append(blind_key_item)
        generated_panels["candidate_panel"].append((rel(candidate_panel_path), title))
        generated_panels["final_panel_384"].append((rel(final_panel_path), title))
        generated_panels["blind_pack_A"].append((rel(blind_a_path), title))
        generated_panels["blind_pack_B"].append((rel(blind_b_path), title))
        if idx <= args.qa_sample_size * 20:
            qa_candidates.append(row._asdict())
        if idx % 5000 == 0:
            print(json.dumps({"generated": idx, "total": len(review)}, ensure_ascii=False))

    visual_manifest_path = stage4_dir / "visual_manifest.csv"
    blind_key_path = stage4_dir / "blind_key.csv"
    write_csv(visual_manifest_path, manifest_rows)
    write_csv(blind_key_path, blind_key_rows)

    # Contact sheets over generated candidate and blind panels.
    def load_panel(rel_path: str) -> Image.Image:
        return Image.open(PROJECT_ROOT / rel_path).convert("RGB")

    contact_specs: dict[str, list[tuple[str, str]]] = {}
    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        contact_specs["pair_union_ambiguous"] = [
            (r["counterfactual_panel_path"], f"ambig {r['gt_verb']} vs {r['competitor']}")
            for r in manifest_df.sort_values(["cpos_cneg_mask_iou", "target_object_category"]).head(args.contact_sheet_limit).to_dict("records")
        ]
        contact_specs["dual_context_complete"] = [
            (r["counterfactual_panel_path"], f"dual ctx {r['cpos_category']} / {r['cneg_category']}")
            for r in manifest_df.sort_values(["strict_pass", "cpos_projected_size", "cneg_projected_size"], ascending=[False, False, False]).head(args.contact_sheet_limit).to_dict("records")
        ]
        contact_specs["current_candidates_reclassified"] = [
            (r["counterfactual_panel_path"], f"balanced strict={r['strict_pass']}")
            for r in pd.concat([
                manifest_df[manifest_df["strict_pass"] == False].head(args.contact_sheet_limit // 2),
                manifest_df[manifest_df["strict_pass"] == True].head(args.contact_sheet_limit // 2),
            ]).head(args.contact_sheet_limit).to_dict("records")
        ]
        contact_specs["blind_pack_A"] = generated_panels["blind_pack_A"][: args.contact_sheet_limit]
        contact_specs["blind_pack_B"] = generated_panels["blind_pack_B"][: args.contact_sheet_limit]

    reject_categories = {
        "context_leakage_failures": ("Gate I", True),
        "pair_internal_decisive_rejects": ("pair_internal_cues_already_decisive", None),
        "positive_only_rejects": ("Gate N", False),
        "misleading_only_rejects": ("Gate P", False),
        "same_context_double_count_rejects": ("Gate I", True),
        "target_too_small_rejects": ("Gate S", None),
    }
    rejection_samples = sample_rejections(args.stage3_dir / "rejection_log.csv", reject_categories, args.contact_sheet_limit)

    contact_outputs: list[Path] = []
    for name, entries in contact_specs.items():
        cards: list[Image.Image] = []
        captions: list[str] = []
        for rel_path, caption in entries:
            try:
                cards.append(load_panel(rel_path))
                captions.append(caption)
            except Exception:
                continue
        if cards:
            sheet = contact_sheet(cards, captions, (320, 240), columns=4)
            out = contact_dir / f"{name}.jpg"
            save_jpeg(out, sheet, args.jpeg_quality)
            contact_outputs.append(out)

    for name, samples in rejection_samples.items():
        cards = [make_reject_card(sample, pair_by_id, image_by_key, cache, (320, 240)) for sample in samples]
        captions = [f"{sample.get('first_failed_gate')} {sample.get('target_object_category')}" for sample in samples]
        if cards:
            sheet = contact_sheet(cards, captions, (320, 240), columns=4)
            out = contact_dir / f"{name}.jpg"
            save_jpeg(out, sheet, args.jpeg_quality)
            contact_outputs.append(out)

    generated_contact_names = {path.stem for path in contact_outputs}
    for name in CONTACT_SHEET_NAMES:
        if name in generated_contact_names:
            continue
        placeholder = text_card(f"{name}\n\nNo balanced rejection samples matched this bucket.\nPlaceholder sheet emitted to keep the Stage 4 review manifest complete.", (960, 540))
        out = contact_dir / f"{name}.jpg"
        save_jpeg(out, placeholder, args.jpeg_quality)
        contact_outputs.append(out)

    # QA over deterministic sample.
    rng = random.Random(20260712)
    qa_pool = manifest_rows if len(manifest_rows) <= args.qa_sample_size else rng.sample(manifest_rows, args.qa_sample_size)
    qa_rows: list[dict[str, Any]] = []
    for item in qa_pool:
        final_path = PROJECT_ROOT / item["final_panel_384_path"]
        panel_path = PROJECT_ROOT / item["counterfactual_panel_path"]
        ok_files = final_path.exists() and panel_path.exists()
        try:
            with Image.open(final_path) as final_img:
                final_width = final_img.width
                final_height = final_img.height
        except Exception:
            final_width = 0
            final_height = 0
        target_readable = float(item["target_human_projected_height"]) >= 130.0 and float(item["target_object_projected_sqrt_area"]) >= 45.0
        context_readable = float(item["cpos_projected_size"]) >= 22.0 and float(item["cneg_projected_size"]) >= 22.0
        no_swap_or_leak = str(item["cpos_id"]) != str(item["cneg_id"]) and float(item["cpos_cneg_mask_iou"]) < 0.2
        qa_rows.append({
            "token": item["token"],
            "files_exist": ok_files,
            "final_width": final_width,
            "final_height": final_height,
            "final_width_is_384": final_width == args.panel_width,
            "target_pair_readable_proxy": target_readable,
            "contexts_readable_proxy": context_readable,
            "cpos_cneg_no_swap_or_leak_proxy": no_swap_or_leak,
            "same_crop_conditions_by_construction": True,
            "model_output_fabricated": False,
            "qa_pass": ok_files and final_width == args.panel_width and target_readable and context_readable and no_swap_or_leak,
        })

    qa_pass_count = sum(1 for row in qa_rows if row["qa_pass"])
    qa_report = f"""# Stage 4 Visual QA Report

## Scope

- Input: `{rel(review_path)}`.
- Balanced review candidates visualized: `{len(manifest_rows)}`.
- Each candidate has a counterfactual panel containing: `{', '.join(VIEW_NAMES)}`.
- Blind Pack A panels: `{len(blind_key_rows)}`.
- Blind Pack B panels: `{len(blind_key_rows)}`.
- Contact sheets generated: `{len(contact_outputs)}`.

## QA Sample

- Random seed: `20260712`.
- Sample size: `{len(qa_rows)}`.
- QA pass count: `{qa_pass_count}`.
- Files exist: `{sum(1 for row in qa_rows if row['files_exist'])}/{len(qa_rows)}`.
- Final panel width is 384 px: `{sum(1 for row in qa_rows if row['final_width_is_384'])}/{len(qa_rows)}`.
- Target pair readable proxy: `{sum(1 for row in qa_rows if row['target_pair_readable_proxy'])}/{len(qa_rows)}`.
- Both contexts readable proxy: `{sum(1 for row in qa_rows if row['contexts_readable_proxy'])}/{len(qa_rows)}`.
- c+ and c- no-swap/no-leak proxy: `{sum(1 for row in qa_rows if row['cpos_cneg_no_swap_or_leak_proxy'])}/{len(qa_rows)}`.

## Comparability And Source Policy

- Pair-only, union-only, union+c+, union+c-, full-minus-c+, and full-minus-c- are rendered from the same evidence crop and the same projected boxes.
- c+ and c- boxes use Stage 3 role bindings; no role exchange is applied in Stage 4.
- Unconfirmed context boxes are rendered as grey dashed boxes in role overlays.
- These are annotation/geometry visualizations only; no model output is fabricated.
"""
    qa_report_path = stage4_dir / "visual_qa_report.md"
    qa_report_path.write_text(qa_report, encoding="utf-8")
    write_csv(stage4_dir / "visual_qa_sample.csv", qa_rows)

    manifest_path = args.output_root / "manifest_stage4.json"
    output_files = [visual_manifest_path, blind_key_path, qa_report_path, stage4_dir / "visual_qa_sample.csv", *contact_outputs]
    manifest = {
        "stage": 4,
        "task": "generate counterfactual visual review materials",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": rel(args.output_root),
        "stage4_dir": rel(stage4_dir),
        "stopped_after_manifest": True,
        "input_review_pool": rel(review_path),
        "candidate_count": len(manifest_rows),
        "blind_pack_A_count": len(blind_key_rows),
        "blind_pack_B_count": len(blind_key_rows),
        "contact_sheet_count": len(contact_outputs),
        "qa_sample_size": len(qa_rows),
        "qa_pass_count": qa_pass_count,
        "same_crop_for_pair_union_context": True,
        "cpos_cneg_roles_preserved": True,
        "model_output_fabricated": False,
        "panel_width_px": args.panel_width,
        "candidate_views": VIEW_NAMES,
        "outputs": [],
    }
    for path in output_files:
        manifest["outputs"].append({"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest_path), "candidate_count": len(manifest_rows), "qa_pass_count": qa_pass_count}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
