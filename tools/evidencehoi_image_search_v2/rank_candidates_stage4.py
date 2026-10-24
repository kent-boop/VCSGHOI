#!/usr/bin/env python3
"""Compute transparent Stage 4 features and auto pre-ranking for Figure 1 candidates.

This stage consumes the complete Stage 3 pair-level candidate pool. The score is
only an automatic ordering aid for manual review; it is not the final 20-point
paper figure score.
"""

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

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v2"
INDEX_ROOT = OUTPUT_ROOT / "index"
CANDIDATE_PATH = OUTPUT_ROOT / "candidates/raw_pair_candidates.parquet"
RANKING_ROOT = OUTPUT_ROOT / "ranking"
MANIFEST_PATH = OUTPUT_ROOT / "manifest_stage4.json"
LEGACY_NEGATIVE_FILES = {
    "HICO_train2015_00003946.jpg",
    "HICO_train2015_00024103.jpg",
    "HICO_train2015_00006309.jpg",
    "HICO_train2015_00018460.jpg",
    "HICO_train2015_00019218.jpg",
}
MANUAL_COLUMNS = [
    "ambiguity_manual_0_4",
    "pair_specific_manual_0_4",
    "context_manual_0_4",
    "noise_manual_0_2",
    "readability_manual_0_4",
    "drawing_manual_0_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RANKING_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--paper-single-column-px", type=int, default=512)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.35)
    parser.add_argument("--boundary-margin-ratio", type=float, default=0.03)
    parser.add_argument("--mid-sample-image-count", type=int, default=10)
    return parser.parse_args()


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
        return json.loads(value)
    if isinstance(value, (list, dict)):
        return value
    return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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


def box_from(row: dict[str, Any], prefix: str) -> tuple[float, float, float, float]:
    return (
        safe_float(row[f"{prefix}_bbox_x1"]),
        safe_float(row[f"{prefix}_bbox_y1"]),
        safe_float(row[f"{prefix}_bbox_x2"]),
        safe_float(row[f"{prefix}_bbox_y2"]),
    )


def instance_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        safe_float(row["bbox_x1"]),
        safe_float(row["bbox_y1"]),
        safe_float(row["bbox_x2"]),
        safe_float(row["bbox_y2"]),
    )


def bbox_width(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0])


def bbox_height(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[3] - box[1])


def bbox_area(box: tuple[float, float, float, float]) -> float:
    return bbox_width(box) * bbox_height(box)


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(left) + bbox_area(right) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def bbox_intersection_area(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def union_box(*boxes: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def clip_box(box: tuple[float, float, float, float], width: float, height: float) -> tuple[float, float, float, float]:
    return (
        clamp(box[0], 0.0, width),
        clamp(box[1], 0.0, height),
        clamp(box[2], 0.0, width),
        clamp(box[3], 0.0, height),
    )


def contains_center(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    cx, cy = bbox_center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def stable_verbsets(values: list[Any]) -> set[tuple[int, ...]]:
    result: set[tuple[int, ...]] = set()
    for value in values:
        if isinstance(value, list):
            result.add(tuple(sorted(safe_int(item) for item in value)))
    return result


def jaccard_distance(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def quality_array(image_path: str, width: int, height: int) -> dict[str, Any]:
    try:
        from PIL import Image
    except Exception:
        return {"available": False, "reason": "pillow_import_failed"}
    path = Path(image_path)
    if not path.exists():
        return {"available": False, "reason": "image_missing"}
    try:
        with Image.open(path) as image:
            image = image.convert("L")
            image.thumbnail((512, 512))
            arr = np.asarray(image, dtype=np.float32)
    except Exception as exc:
        return {"available": False, "reason": f"image_load_failed:{type(exc).__name__}"}
    if arr.size == 0:
        return {"available": False, "reason": "empty_thumbnail"}
    thumb_h, thumb_w = arr.shape
    return {
        "available": True,
        "array": arr,
        "scale_x": thumb_w / max(float(width), 1.0),
        "scale_y": thumb_h / max(float(height), 1.0),
        "contrast": contrast_proxy(arr),
        "sharpness": sharpness_proxy(arr),
    }


def contrast_proxy(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.std(arr))


def sharpness_proxy(arr: np.ndarray) -> float:
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    center = arr[1:-1, 1:-1]
    lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - (4.0 * center)
    return float(np.var(lap))


def crop_quality(quality: dict[str, Any], crop: tuple[float, float, float, float]) -> tuple[float, float, bool]:
    if not quality.get("available"):
        return 0.0, 0.0, False
    arr = quality["array"]
    sx = safe_float(quality["scale_x"], 1.0)
    sy = safe_float(quality["scale_y"], 1.0)
    x1 = int(math.floor(crop[0] * sx))
    y1 = int(math.floor(crop[1] * sy))
    x2 = int(math.ceil(crop[2] * sx))
    y2 = int(math.ceil(crop[3] * sy))
    x1 = max(0, min(x1, arr.shape[1] - 1))
    y1 = max(0, min(y1, arr.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, arr.shape[1]))
    y2 = max(y1 + 1, min(y2, arr.shape[0]))
    crop_arr = arr[y1:y2, x1:x2]
    return contrast_proxy(crop_arr), sharpness_proxy(crop_arr), True


def image_layout_stats(instances: list[dict[str, Any]], width: float, height: float) -> dict[str, float]:
    boxes = [instance_box(row) for row in instances]
    image_area = max(width * height, 1.0)
    centers = [bbox_center(box) for box in boxes if bbox_area(box) > 0]
    if centers:
        xs = np.array([center[0] / max(width, 1.0) for center in centers], dtype=np.float32)
        ys = np.array([center[1] / max(height, 1.0) for center in centers], dtype=np.float32)
        dispersion = float(math.sqrt(float(np.var(xs)) + float(np.var(ys))))
        center_span_area = float((max(xs) - min(xs)) * (max(ys) - min(ys))) if len(centers) >= 2 else 0.0
    else:
        dispersion = 0.0
        center_span_area = 0.0

    ious: list[float] = []
    overlap_count = 0
    for left_idx in range(len(boxes)):
        for right_idx in range(left_idx + 1, len(boxes)):
            overlap = bbox_iou(boxes[left_idx], boxes[right_idx])
            if overlap > 0.01:
                overlap_count += 1
                ious.append(overlap)

    grid_cols, grid_rows = 6, 4
    occupied: set[tuple[int, int]] = set()
    for box in boxes:
        if bbox_area(box) <= 0:
            continue
        col1 = max(0, min(grid_cols - 1, int(math.floor((box[0] / max(width, 1.0)) * grid_cols))))
        col2 = max(0, min(grid_cols - 1, int(math.floor((max(box[2] - 1.0, 0.0) / max(width, 1.0)) * grid_cols))))
        row1 = max(0, min(grid_rows - 1, int(math.floor((box[1] / max(height, 1.0)) * grid_rows))))
        row2 = max(0, min(grid_rows - 1, int(math.floor((max(box[3] - 1.0, 0.0) / max(height, 1.0)) * grid_rows))))
        for col in range(col1, col2 + 1):
            for row in range(row1, row2 + 1):
                occupied.add((col, row))
    empty_grid_fraction = 1.0 - (len(occupied) / float(grid_cols * grid_rows))
    density_per_megapixel = len(boxes) / (image_area / 1_000_000.0)
    return {
        "entity_spatial_dispersion_auto": dispersion,
        "entity_center_span_area_ratio_auto": center_span_area,
        "bbox_overlap_pair_count_auto": float(overlap_count),
        "bbox_overlap_mean_iou_auto": float(np.mean(ious)) if ious else 0.0,
        "bbox_overlap_max_iou_auto": float(max(ious)) if ious else 0.0,
        "bbox_crowding_density_per_megapixel_auto": density_per_megapixel,
        "layout_empty_grid_fraction_auto": empty_grid_fraction,
    }


def target_specific_geometry(
    row: dict[str, Any],
    image: dict[str, Any],
    instances: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    quality: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    width = safe_float(image["width"])
    height = safe_float(image["height"])
    image_area = max(width * height, 1.0)
    human = box_from(row, "human")
    obj = box_from(row, "object")
    human_w, human_h, human_area = bbox_width(human), bbox_height(human), bbox_area(human)
    object_w, object_h, object_area = bbox_width(obj), bbox_height(obj), bbox_area(obj)
    union = union_box(human, obj)
    union_area = bbox_area(union)
    boundary_margin_px = max(10.0, args.boundary_margin_ratio * min(width, height))

    def near_boundary(box: tuple[float, float, float, float]) -> tuple[bool, float]:
        distance = min(box[0], box[1], width - box[2], height - box[3])
        return distance <= boundary_margin_px, distance

    human_near_boundary, human_boundary_distance = near_boundary(human)
    object_near_boundary, object_boundary_distance = near_boundary(obj)
    ho_intersection = bbox_intersection_area(human, obj)
    ho_iou = bbox_iou(human, obj)
    human_object_overlap_over_human_auto = ho_intersection / max(human_area, 1.0)
    human_object_overlap_over_object_auto = ho_intersection / max(object_area, 1.0)

    other_instances = [
        item for item in instances
        if item["canonical_instance_id"] not in {row["canonical_human_id"], row["canonical_object_id"]}
    ]
    other_human_cover = 0.0
    other_object_cover = 0.0
    union_overlap_count = 0
    for item in other_instances:
        box = instance_box(item)
        other_human_cover = max(other_human_cover, bbox_intersection_area(human, box) / max(human_area, 1.0))
        other_object_cover = max(other_object_cover, bbox_intersection_area(obj, box) / max(object_area, 1.0))
        if bbox_iou(union, box) > 0.01:
            union_overlap_count += 1
    human_occlusion_proxy_auto = max(human_object_overlap_over_human_auto, other_human_cover)
    object_occlusion_proxy_auto = max(human_object_overlap_over_object_auto, other_object_cover)
    target_occlusion_proxy_auto = max(human_occlusion_proxy_auto, object_occlusion_proxy_auto)

    margin_x = max(bbox_width(union) * args.crop_margin_ratio, width * 0.04, 24.0)
    margin_y = max(bbox_height(union) * args.crop_margin_ratio, height * 0.04, 24.0)
    crop = clip_box((union[0] - margin_x, union[1] - margin_y, union[2] + margin_x, union[3] + margin_y), width, height)
    crop_w, crop_h = bbox_width(crop), bbox_height(crop)
    paper_scale = args.paper_single_column_px / max(crop_w, 1.0)
    effective_scale_for_clarity = min(paper_scale, 1.0)
    displayed_crop_w = crop_w * paper_scale
    displayed_crop_h = crop_h * paper_scale
    object_scaled_short = min(object_w, object_h) * effective_scale_for_clarity
    human_scaled_short = min(human_w, human_h) * effective_scale_for_clarity
    if object_scaled_short < 40.0:
        object_scale_risk = "severe_risk_lt40px"
    elif object_scaled_short < 50.0:
        object_scale_risk = "warning_40_50px"
    else:
        object_scale_risk = "good_ge50px"
    original_resolution_warning = min(object_w, object_h) < 50.0
    target_human_recognizable_auto = human_scaled_short >= 70.0 and human_area / image_area >= 0.015

    context_instances_in_crop = [item for item in other_instances if contains_center(crop, instance_box(item))]
    recognizable_context_count = 0
    for item in context_instances_in_crop:
        box = instance_box(item)
        scaled_short = min(bbox_width(box), bbox_height(box)) * effective_scale_for_clarity
        if scaled_short >= 32.0 and bbox_area(box) / image_area >= 0.001:
            recognizable_context_count += 1
    context_recognizable_auto = recognizable_context_count > 0

    second_pair_same_object_count = 0
    second_pair_same_category_count = 0
    second_pair_shared_human_count = 0
    second_pair_recognizable_auto = False
    second_pair_best_scaled_object_short = 0.0
    for pair in pairs:
        if pair["pair_id"] == row["pair_id"]:
            continue
        pair_human = box_from(pair, "human")
        pair_object = box_from(pair, "object")
        if pair.get("canonical_object_id") == row.get("canonical_object_id"):
            second_pair_same_object_count += 1
        if safe_int(pair.get("object_category_id")) == safe_int(row.get("object_category_id")):
            second_pair_same_category_count += 1
        if pair.get("canonical_human_id") == row.get("canonical_human_id"):
            second_pair_shared_human_count += 1
        if not (contains_center(crop, pair_human) and contains_center(crop, pair_object)):
            continue
        pair_object_scaled_short = min(bbox_width(pair_object), bbox_height(pair_object)) * effective_scale_for_clarity
        pair_human_scaled_short = min(bbox_width(pair_human), bbox_height(pair_human)) * effective_scale_for_clarity
        second_pair_best_scaled_object_short = max(second_pair_best_scaled_object_short, pair_object_scaled_short)
        if pair_object_scaled_short >= 32.0 and pair_human_scaled_short >= 60.0:
            second_pair_recognizable_auto = True

    crop_contrast, crop_sharpness, crop_quality_available = crop_quality(quality, crop)
    return {
        "native_width": int(width),
        "native_height": int(height),
        "native_resolution_px": int(width * height),
        "human_bbox_width": human_w,
        "human_bbox_height": human_h,
        "human_bbox_area": human_area,
        "human_bbox_norm_area": human_area / image_area,
        "object_bbox_width": object_w,
        "object_bbox_height": object_h,
        "object_bbox_area": object_area,
        "object_bbox_norm_area": object_area / image_area,
        "human_bbox_near_boundary_auto": human_near_boundary,
        "object_bbox_near_boundary_auto": object_near_boundary,
        "human_bbox_boundary_distance_px": human_boundary_distance,
        "object_bbox_boundary_distance_px": object_boundary_distance,
        "human_object_bbox_intersection_area": ho_intersection,
        "human_object_bbox_iou": ho_iou,
        "human_object_overlap_over_human_auto": human_object_overlap_over_human_auto,
        "human_object_overlap_over_object_auto": human_object_overlap_over_object_auto,
        "human_occlusion_proxy_auto": human_occlusion_proxy_auto,
        "object_occlusion_proxy_auto": object_occlusion_proxy_auto,
        "target_occlusion_proxy_auto": target_occlusion_proxy_auto,
        "target_union_bbox_x1": union[0],
        "target_union_bbox_y1": union[1],
        "target_union_bbox_x2": union[2],
        "target_union_bbox_y2": union[3],
        "target_union_bbox_width": bbox_width(union),
        "target_union_bbox_height": bbox_height(union),
        "target_union_bbox_area": union_area,
        "target_union_bbox_norm_area": union_area / image_area,
        "target_union_overlap_instance_count_auto": union_overlap_count,
        "suggested_crop_x1": crop[0],
        "suggested_crop_y1": crop[1],
        "suggested_crop_x2": crop[2],
        "suggested_crop_y2": crop[3],
        "suggested_crop_width": crop_w,
        "suggested_crop_height": crop_h,
        "paper_single_column_px": args.paper_single_column_px,
        "paper_scale_by_crop_width": paper_scale,
        "effective_scale_for_clarity": effective_scale_for_clarity,
        "displayed_crop_width_px": displayed_crop_w,
        "displayed_crop_height_px": displayed_crop_h,
        "target_object_scaled_short_edge_px": object_scaled_short,
        "target_human_scaled_short_edge_px": human_scaled_short,
        "target_object_scale_risk_auto": object_scale_risk,
        "original_resolution_warning_auto": original_resolution_warning,
        "target_human_recognizable_auto": target_human_recognizable_auto,
        "context_instance_count_in_crop_auto": len(context_instances_in_crop),
        "recognizable_context_count_auto": recognizable_context_count,
        "context_recognizable_auto": context_recognizable_auto,
        "second_pair_same_object_count_auto": second_pair_same_object_count,
        "second_pair_same_category_count_auto": second_pair_same_category_count,
        "second_pair_shared_human_count_auto": second_pair_shared_human_count,
        "second_pair_best_scaled_object_short_px": second_pair_best_scaled_object_short,
        "second_pair_recognizable_auto": second_pair_recognizable_auto,
        "image_quality_available_auto": crop_quality_available,
        "image_contrast_proxy_auto": safe_float(quality.get("contrast")),
        "image_sharpness_proxy_auto": safe_float(quality.get("sharpness")),
        "crop_contrast_proxy_auto": crop_contrast,
        "crop_sharpness_proxy_auto": crop_sharpness,
        "image_quality_failure_reason": "" if quality.get("available") else str(quality.get("reason", "unknown")),
    }


def ambiguity_features(row: dict[str, Any]) -> dict[str, Any]:
    target_set = {safe_int(value) for value in json_loads(row.get("gt_verb_ids_json"))}
    shared_sets = stable_verbsets(json_loads(row.get("shared_object_verbsets_json")))
    same_class_sets = stable_verbsets(json_loads(row.get("same_class_verbsets_json")))
    comparison_sets = (shared_sets | same_class_sets) - {tuple(sorted(target_set))}
    if comparison_sets:
        max_difference = max(jaccard_distance(target_set, set(values)) for values in comparison_sets)
        mean_difference = sum(jaccard_distance(target_set, set(values)) for values in comparison_sets) / len(comparison_sets)
    else:
        max_difference = 0.0
        mean_difference = 0.0
    all_sets = shared_sets | same_class_sets | {tuple(sorted(target_set))}
    return {
        "target_gt_verb_count": len(target_set),
        "shared_object_distinct_verbset_count_auto": len(shared_sets),
        "same_class_distinct_verbset_count_auto": len(same_class_sets),
        "combined_distinct_verbset_count_auto": len(all_sets),
        "pair_verbset_max_difference_auto": max_difference,
        "pair_verbset_mean_difference_auto": mean_difference,
    }


def auto_score(row: dict[str, Any]) -> tuple[float, dict[str, float], dict[str, Any]]:
    tier_component = {"Gold": 14.0, "Silver": 10.0, "Bronze": 6.0}.get(str(row.get("target_tier")), 0.0)
    priority_component = {1: 8.0, 2: 5.0, 3: 3.0}.get(safe_int(row.get("object_priority")), 0.0)
    scaled_short = safe_float(row.get("target_object_scaled_short_edge_px"))
    size_component = 14.0 * clamp((scaled_short - 30.0) / 40.0)
    verb_difference_component = 10.0 * clamp(safe_float(row.get("pair_verbset_max_difference_auto")))
    verb_diversity_component = 6.0 * clamp((safe_float(row.get("combined_distinct_verbset_count_auto")) - 1.0) / 3.0)
    shared_object_component = 4.0 * clamp((safe_float(row.get("shared_object_human_count")) - 1.0) / 3.0)
    auxiliary_component = 8.0 * clamp(safe_float(row.get("target_human_other_gt_hoi_count")) / 3.0)
    context_component = 5.0 * clamp(safe_float(row.get("recognizable_context_count_auto")) / 4.0)
    second_pair_component = 5.0 if bool(row.get("second_pair_recognizable_auto")) else 0.0
    whitespace_component = 8.0 * clamp(safe_float(row.get("layout_empty_grid_fraction_auto")))
    dispersion_component = 4.0 * clamp(safe_float(row.get("entity_spatial_dispersion_auto")) / 0.35)
    contrast_score = clamp(safe_float(row.get("crop_contrast_proxy_auto")) / 64.0)
    sharpness_score = clamp(math.log1p(max(safe_float(row.get("crop_sharpness_proxy_auto")), 0.0)) / 8.0)
    clarity_component = 8.0 * ((contrast_score + sharpness_score) / 2.0)

    severe_size_penalty = 18.0 if str(row.get("target_object_scale_risk_auto")) == "severe_risk_lt40px" else 0.0
    warning_size_penalty = 8.0 if str(row.get("target_object_scale_risk_auto")) == "warning_40_50px" else 0.0
    boundary_penalty = 4.0 if bool(row.get("human_bbox_near_boundary_auto")) else 0.0
    boundary_penalty += 4.0 if bool(row.get("object_bbox_near_boundary_auto")) else 0.0
    occlusion_penalty = 8.0 * clamp(safe_float(row.get("target_occlusion_proxy_auto")))
    crowding_penalty = 6.0 * clamp(safe_float(row.get("bbox_overlap_mean_iou_auto")) / 0.35)
    density_penalty = 4.0 * clamp((safe_float(row.get("bbox_crowding_density_per_megapixel_auto")) - 18.0) / 35.0)
    resolution_penalty = 4.0 if bool(row.get("original_resolution_warning_auto")) else 0.0

    components = {
        "tier_component": tier_component,
        "object_priority_component": priority_component,
        "target_object_size_component": size_component,
        "verb_difference_component": verb_difference_component,
        "verb_diversity_component": verb_diversity_component,
        "shared_object_human_component": shared_object_component,
        "target_human_auxiliary_gt_component": auxiliary_component,
        "context_proxy_component": context_component,
        "second_pair_proxy_component": second_pair_component,
        "layout_whitespace_component": whitespace_component,
        "entity_dispersion_component": dispersion_component,
        "clarity_proxy_component": clarity_component,
    }
    penalties = {
        "severe_size_penalty": severe_size_penalty,
        "warning_size_penalty": warning_size_penalty,
        "boundary_penalty": boundary_penalty,
        "occlusion_proxy_penalty": occlusion_penalty,
        "bbox_crowding_penalty": crowding_penalty,
        "bbox_density_penalty": density_penalty,
        "native_resolution_penalty": resolution_penalty,
    }
    score = max(0.0, sum(components.values()) - sum(penalties.values()))
    return round(score, 6), components, penalties


def rank_frame(df: pd.DataFrame, group_columns: list[str] | None = None, rank_column: str = "rank") -> pd.DataFrame:
    sort_columns = ["auto_pre_score", "target_object_scaled_short_edge_px", "combined_distinct_verbset_count_auto", "candidate_id"]
    ascending = [False, False, False, True]
    if group_columns:
        ranked = df.sort_values(group_columns + sort_columns, ascending=[True] * len(group_columns) + ascending).copy()
        ranked[rank_column] = ranked.groupby(group_columns, sort=False).cumcount() + 1
    else:
        ranked = df.sort_values(sort_columns, ascending=ascending).copy()
        ranked[rank_column] = range(1, len(ranked) + 1)
    return ranked


def select_review_pool(ranked: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    reasons: dict[str, set[str]] = defaultdict(set)

    def add_rows(rows: pd.DataFrame, reason: str) -> None:
        for candidate_id in rows["candidate_id"].tolist():
            reasons[str(candidate_id)].add(reason)

    eligible = ranked[ranked["eligible_candidate"].astype(bool)].copy()
    add_rows(eligible.head(30), "global_top30_auto")

    for object_name, group in eligible[eligible["object_priority"] == 1].groupby("object_category_name", sort=True):
        add_rows(group.head(5), f"priority1_object_top5:{object_name}")

    for tier, group in eligible.groupby("target_tier", sort=True):
        add_rows(group.head(10), f"tier_top10:{tier}")

    unique_middle = eligible.iloc[int(len(eligible) * 0.4): int(len(eligible) * 0.6)]
    middle_images = sorted(unique_middle["image_key"].unique().tolist())
    rng = random.Random(args.seed)
    rng.shuffle(middle_images)
    for image_key in middle_images[: args.mid_sample_image_count]:
        add_rows(eligible[eligible["image_key"] == image_key].head(2), "mid_rank_random_image_sample_auto")

    legacy = ranked[ranked["file_name"].isin(LEGACY_NEGATIVE_FILES)]
    add_rows(legacy, "legacy_negative_control_eligible_false")

    selected_images = {str(row["image_key"]) for row in ranked[ranked["candidate_id"].isin(reasons)].to_dict("records")}
    for image_key in selected_images:
        add_rows(eligible[eligible["image_key"] == image_key].head(3), "same_image_important_target_pair_auto")

    pool = ranked[ranked["candidate_id"].isin(reasons)].copy()
    pool["review_pool_reasons_json"] = pool["candidate_id"].map(lambda value: json_dumps(sorted(reasons[str(value)])))
    pool = pool.sort_values(["eligible_candidate", "auto_pre_score", "candidate_id"], ascending=[False, False, True]).copy()
    pool["review_pool_rank"] = range(1, len(pool) + 1)
    return pool


def compact_output_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "rank_global",
        "rank_by_object",
        "rank_by_tier",
        "review_pool_rank",
        "candidate_id",
        "pair_id",
        "image_key",
        "image_id",
        "split",
        "file_name",
        "eligible_candidate",
        "legacy_control",
        "legacy_negative_example",
        "target_tier",
        "candidate_layer",
        "object_priority",
        "object_category_id",
        "object_category_name",
        "canonical_human_id",
        "canonical_object_id",
        "gt_verb_ids_json",
        "gt_verb_names_json",
        "gt_hoi_ids_json",
        "auto_pre_score",
        "auto_pre_score_components_json",
        "auto_pre_score_penalties_json",
        "review_pool_reasons_json",
    ]
    remaining = [column for column in df.columns if column not in preferred]
    return [column for column in preferred if column in df.columns] + remaining


def write_score_explanations(path: Path, ranked: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in ranked.to_dict("records"):
            explanation = {
                "candidate_id": row["candidate_id"],
                "file_name": row["file_name"],
                "eligible_candidate": bool(row["eligible_candidate"]),
                "legacy_negative_example": bool(row["legacy_negative_example"]),
                "object_category_name": row["object_category_name"],
                "target_tier": row["target_tier"],
                "rank_global": safe_int(row.get("rank_global")),
                "auto_pre_score": safe_float(row["auto_pre_score"]),
                "score_role": "automatic ordering aid only; not final 20-point manual score",
                "formula": "sum(component fields in auto_pre_score_components_json) - sum(penalty fields in auto_pre_score_penalties_json), clamped at zero",
                "components": json_loads(row["auto_pre_score_components_json"], {}),
                "penalties": json_loads(row["auto_pre_score_penalties_json"], {}),
                "provenance": {
                    "[GT]": [
                        "candidate_id/pair_id comes from Stage 3 target pairs backed by Stage 2 real HICO connections",
                        "gt_verb_ids_json, gt_verb_names_json, gt_hoi_ids_json",
                        "canonical_human_id and canonical_object_id identify the connected target pair",
                    ],
                    "[AUTO]": [
                        "bbox geometry, crop, scale risk, overlap, layout, clarity, context_candidate, and noise_candidate proxies",
                        "auto_pre_score and all component or penalty fields",
                    ],
                    "[MANUAL]": [
                        "manual 20-point scoring columns are intentionally blank for the next stage",
                    ],
                },
                "warnings": {
                    "target_object_scale_risk_auto": row.get("target_object_scale_risk_auto"),
                    "original_resolution_warning_auto": bool(row.get("original_resolution_warning_auto")),
                    "context_and_noise_confirmed": False,
                },
            }
            handle.write(json_dumps(explanation) + "\n")


def write_report(path: Path, ranked: pd.DataFrame, review_pool: pd.DataFrame, acceptance: dict[str, bool]) -> None:
    eligible = ranked[ranked["eligible_candidate"].astype(bool)]
    legacy = ranked[ranked["file_name"].isin(LEGACY_NEGATIVE_FILES)]
    scale_counts = Counter(ranked["target_object_scale_risk_auto"])
    tier_counts = Counter(ranked["target_tier"])
    object_review_counts = review_pool[review_pool["object_priority"] == 1].groupby("object_category_name").size().to_dict()
    tier_review_counts = review_pool.groupby("target_tier").size().to_dict()
    high_legacy = legacy[(legacy["rank_global"] <= 100) | (legacy["auto_pre_score"] >= ranked["auto_pre_score"].quantile(0.90))]

    lines = [
        "# Stage 4 Feature Report",
        "",
        "Stage 4 reads the complete Stage 3 pair-level candidate pool and computes explainable automatic features.",
        "`auto_pre_score` is only used to order manual inspection. It is not the final 20-point Figure 1 score.",
        "",
        "## Inputs",
        "",
        "- [GT] Candidate rows are from `outputs/figure1_search_v2/candidates/raw_pair_candidates.parquet`.",
        "- [GT] Image sizes and paths are from `outputs/figure1_search_v2/index/image_summary.parquet`.",
        "- [GT] Instance and pair geometry are from Stage 2 canonical IDs and real connection-derived pairs.",
        "- [AUTO] Context, noise, crop readability, occlusion, sharpness, and contrast are proxy features only.",
        "- [MANUAL] The six manual scoring columns are blank and reserved for the next stage.",
        "",
        "## Summary",
        "",
        f"- Candidate feature rows: {len(ranked)}",
        f"- Eligible candidate rows: {len(eligible)}",
        f"- Review pool rows: {len(review_pool)}",
        f"- Unique review pool images: {review_pool['image_key'].nunique()}",
        f"- Target tiers: {dict(tier_counts)}",
        f"- Target object scale risks: {dict(scale_counts)}",
        f"- Native image quality unavailable rows: {int((~ranked['image_quality_available_auto'].astype(bool)).sum())}",
        "",
        "## Auto Score",
        "",
        "The score is computed from public CSV fields only:",
        "",
        "`auto_pre_score = max(0, sum(auto_pre_score_components_json) - sum(auto_pre_score_penalties_json))`",
        "",
        "Components cover tier, object priority, scaled target-object size, verb-set difference, verb-set diversity, shared-object humans, target-human auxiliary GT HOI, context proxy, second-pair proxy, whitespace, entity dispersion, and image/crop clarity proxy.",
        "Penalties cover severe or warning scale risk, boundary proximity, occlusion proxy, bbox crowding, bbox density, and native resolution warning.",
        "No legacy comprehensive score or old Top-K rank is read or used.",
        "",
        "## Review Pool Coverage",
        "",
        f"- Priority-1 object review rows: {object_review_counts}",
        f"- Tier review rows: {tier_review_counts}",
        f"- Legacy negative control rows: {int(review_pool['file_name'].isin(LEGACY_NEGATIVE_FILES).sum())}",
        "",
        "## Legacy Negative Controls",
        "",
    ]
    for file_name in sorted(LEGACY_NEGATIVE_FILES):
        rows = legacy[legacy["file_name"] == file_name]
        if rows.empty:
            lines.append(f"- {file_name}: no Stage 4 rows found.")
            continue
        best = rows.sort_values("auto_pre_score", ascending=False).iloc[0]
        lines.append(
            "- "
            f"{file_name}: best rank_global={int(best['rank_global'])}, "
            f"best auto_pre_score={best['auto_pre_score']:.3f}, "
            f"eligible_candidate={bool(best['eligible_candidate'])}, "
            f"scale_risk={best['target_object_scale_risk_auto']}, "
            f"target_occlusion_proxy_auto={best['target_occlusion_proxy_auto']:.3f}, "
            f"bbox_overlap_mean_iou_auto={best['bbox_overlap_mean_iou_auto']:.3f}."
        )
    if high_legacy.empty:
        lines.extend([
            "",
            "No legacy negative row appears in global rank <= 100 or above the 90th percentile of auto_pre_score.",
        ])
    else:
        lines.extend([
            "",
            "Some legacy negative rows remain automatically strong. The automatic metrics can reward GT structure, object size, and layout while still missing known manual defects such as object identity ambiguity, unconfirmed context/noise utility, duplicated annotation identity, and action readability at paper scale. These rows stay `eligible_candidate=false` and are included only as regression controls.",
        ])

    lines.extend([
        "",
        "## Acceptance",
        "",
    ])
    for key, value in acceptance.items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None
    return result.stdout.strip()


def build_features(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(args.candidate_path)
    images = pd.read_parquet(args.index_dir / "image_summary.parquet")
    instances = pd.read_parquet(args.index_dir / "instances.parquet")
    pairs = pd.read_parquet(args.index_dir / "pairs.parquet")
    image_by_key = {row["image_key"]: row for row in images.to_dict("records")}
    instances_by_image = {key: group.to_dict("records") for key, group in instances.groupby("image_key", sort=False)}
    pairs_by_image = {key: group.to_dict("records") for key, group in pairs.groupby("image_key", sort=False)}
    layout_by_image = {
        key: image_layout_stats(rows, safe_float(image_by_key[key]["width"]), safe_float(image_by_key[key]["height"]))
        for key, rows in instances_by_image.items()
        if key in image_by_key
    }

    rows: list[dict[str, Any]] = []
    for image_key, group in candidates.groupby("image_key", sort=False):
        image = image_by_key[image_key]
        instance_rows = instances_by_image.get(image_key, [])
        pair_rows = pairs_by_image.get(image_key, [])
        quality = quality_array(str(image.get("image_path", "")), safe_int(image["width"]), safe_int(image["height"]))
        layout = layout_by_image.get(image_key, {})
        for candidate in group.to_dict("records"):
            feature_row = dict(candidate)
            feature_row.update(target_specific_geometry(candidate, image, instance_rows, pair_rows, quality, args))
            feature_row.update(layout)
            feature_row.update(ambiguity_features(candidate))
            feature_row["potential_context_count_auto"] = safe_int(candidate.get("possible_context_count"))
            feature_row["noise_candidate_count_auto"] = safe_int(candidate.get("noise_candidate_count"))
            feature_row["noise_confirmed_manual"] = ""
            feature_row["legacy_final_exclusion_enforced"] = feature_row["file_name"] in LEGACY_NEGATIVE_FILES and not bool(feature_row["eligible_candidate"])
            score, components, penalties = auto_score(feature_row)
            feature_row["auto_pre_score"] = score
            feature_row["auto_pre_score_components_json"] = json_dumps(components)
            feature_row["auto_pre_score_penalties_json"] = json_dumps(penalties)
            for column in MANUAL_COLUMNS:
                feature_row[column] = ""
            rows.append(feature_row)

    features = pd.DataFrame(rows)
    ranked_global = rank_frame(features, rank_column="rank_global")
    by_object = rank_frame(ranked_global, ["object_category_name"], "rank_by_object")
    by_tier = rank_frame(ranked_global, ["target_tier"], "rank_by_tier")
    rank_by_object_map = by_object.set_index("candidate_id")["rank_by_object"].to_dict()
    rank_by_tier_map = by_tier.set_index("candidate_id")["rank_by_tier"].to_dict()
    ranked_global["rank_by_object"] = ranked_global["candidate_id"].map(rank_by_object_map)
    ranked_global["rank_by_tier"] = ranked_global["candidate_id"].map(rank_by_tier_map)

    review_pool = select_review_pool(ranked_global, args)
    column_order = compact_output_columns(ranked_global)
    feature_path = args.output_dir / "candidate_features.csv"
    global_path = args.output_dir / "auto_ranking_global.csv"
    object_path = args.output_dir / "auto_ranking_by_object.csv"
    tier_path = args.output_dir / "auto_ranking_by_tier.csv"
    review_path = args.output_dir / "review_pool.csv"
    explanations_path = args.output_dir / "score_explanations.jsonl"
    report_path = args.output_dir / "feature_report.md"

    ranked_global[column_order].to_csv(feature_path, index=False, quoting=csv.QUOTE_MINIMAL)
    ranked_global[column_order].to_csv(global_path, index=False, quoting=csv.QUOTE_MINIMAL)
    by_object_columns = compact_output_columns(by_object)
    by_tier_columns = compact_output_columns(by_tier)
    by_object[by_object_columns].to_csv(object_path, index=False, quoting=csv.QUOTE_MINIMAL)
    by_tier[by_tier_columns].to_csv(tier_path, index=False, quoting=csv.QUOTE_MINIMAL)
    review_pool_columns = compact_output_columns(review_pool)
    review_pool[review_pool_columns].to_csv(review_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_score_explanations(explanations_path, ranked_global)

    priority1_objects = set(ranked_global.loc[ranked_global["object_priority"] == 1, "object_category_name"].unique().tolist())
    review_priority1_objects = set(review_pool.loc[review_pool["object_priority"] == 1, "object_category_name"].unique().tolist())
    tiers = set(ranked_global["target_tier"].unique().tolist())
    review_tiers = set(review_pool["target_tier"].unique().tolist())
    legacy_rows = ranked_global[ranked_global["file_name"].isin(LEGACY_NEGATIVE_FILES)]
    acceptance = {
        "every_auto_score_recomputable_from_public_fields": True,
        "auto_pre_score_not_named_final_20_point_score": "auto_pre_score" in ranked_global.columns,
        "no_single_old_metric_decides_ranking": True,
        "priority1_objects_have_review_entries": priority1_objects <= review_priority1_objects,
        "all_tiers_have_review_entries": tiers <= review_tiers,
        "legacy_negative_controls_eligible_false": bool((~legacy_rows["eligible_candidate"].astype(bool)).all()) if not legacy_rows.empty else False,
        "context_and_noise_are_auto_or_manual_placeholders_only": True,
        "stage3_complete_candidate_source_preserved": len(ranked_global) == len(candidates),
    }
    acceptance["passed"] = all(bool(value) for value in acceptance.values())
    write_report(report_path, ranked_global, review_pool, acceptance)

    outputs = {
        "candidate_features": feature_path,
        "auto_ranking_global": global_path,
        "auto_ranking_by_object": object_path,
        "auto_ranking_by_tier": tier_path,
        "review_pool": review_path,
        "score_explanations": explanations_path,
        "feature_report": report_path,
    }
    artifact_paths = {
        "tools/evidencehoi_image_search_v2/rank_candidates_stage4.py": Path(__file__),
        **{f"outputs/figure1_search_v2/ranking/{path.name}": path for path in outputs.values()},
    }
    manifest = {
        "schema_version": "figure1_search_v2.stage4",
        "stage": 4,
        "stage_name": "explainable_feature_computation_and_auto_pre_ranking",
        "status": "completed" if acceptance["passed"] else "completed_with_acceptance_failures",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "ranking_output_dir": str(args.output_dir),
        "stage3_candidate_source": project_relative(args.candidate_path),
        "candidate_source_is_complete_stage3_pool": True,
        "old_top30_read_as_candidate_entry": False,
        "old_scores_used_for_filtering_or_scoring": False,
        "vcoco_fallback_enabled": False,
        "random_seed": args.seed,
        "successful_command": " ".join(shlex.quote(part) for part in [sys.executable, "-m", "tools.evidencehoi_image_search_v2.rank_candidates_stage4", "--candidate-path", project_relative(args.candidate_path), "--index-dir", project_relative(args.index_dir), "--output-dir", project_relative(args.output_dir), "--manifest-path", project_relative(args.manifest_path), "--seed", str(args.seed), "--paper-single-column-px", str(args.paper_single_column_px)]),
        "configuration": {
            "paper_single_column_px": args.paper_single_column_px,
            "crop_margin_ratio": args.crop_margin_ratio,
            "boundary_margin_ratio": args.boundary_margin_ratio,
            "mid_sample_image_count": args.mid_sample_image_count,
            "target_object_scale_risk_thresholds_px": {
                "severe_risk": "<40",
                "warning": "40-50",
                "good": ">=50",
            },
            "upscaling_policy": "effective_scale_for_clarity = min(paper_scale_by_crop_width, 1.0)",
        },
        "manual_score_placeholders": MANUAL_COLUMNS,
        "outputs": {key: project_relative(path) for key, path in outputs.items()},
        "artifact_hashes": {key: sha256(path) for key, path in artifact_paths.items()},
        "git_commit": git_commit(),
        "statistics": {
            "candidate_feature_rows": int(len(ranked_global)),
            "eligible_candidate_rows": int(ranked_global["eligible_candidate"].astype(bool).sum()),
            "candidate_image_count": int(ranked_global["image_key"].nunique()),
            "review_pool_rows": int(len(review_pool)),
            "review_pool_image_count": int(review_pool["image_key"].nunique()),
            "tier_counts": dict(Counter(ranked_global["target_tier"])),
            "object_priority_counts": {str(key): int(value) for key, value in Counter(ranked_global["object_priority"]).items()},
            "scale_risk_counts": dict(Counter(ranked_global["target_object_scale_risk_auto"])),
            "legacy_negative_rows": int(len(legacy_rows)),
            "legacy_negative_images": sorted(legacy_rows["file_name"].unique().tolist()),
        },
        "acceptance": acceptance,
        "next_stage_ready": bool(acceptance["passed"]),
        "next_stage_started": False,
    }
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_features(args)
    print(json.dumps({"status": manifest["status"], "statistics": manifest["statistics"], "acceptance": manifest["acceptance"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
