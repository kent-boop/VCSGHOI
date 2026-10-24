#!/usr/bin/env python3
"""Parse HICO-style HOI annotations and compute per-image statistics.

The script is intended as a reusable, read-only preprocessing step for finding
Figure 1 candidates. It does not score or visualize images.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PERSON_CATEGORY_ID = 1

CSV_COLUMNS = [
    "image_id",
    "filename",
    "image_path",
    "width",
    "height",
    "human_count",
    "object_count",
    "hoi_pair_count",
    "verb_count",
    "object_category_count",
    "multi_person",
    "one_person_multiple_objects",
    "multiple_people_same_object_category",
    "same_object_near_multiple_people",
    "shared_action",
    "shared_object_category",
    "short_pair_summary",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze HICO-style HOI annotations for Figure 1 candidate search."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory, e.g. hico_20160224_det.")
    parser.add_argument("--annotation", required=True, help="Path to annotation JSON, absolute or relative to dataset root.")
    parser.add_argument("--image-root", required=True, help="Image root directory, absolute or relative to dataset root.")
    parser.add_argument("--output-dir", required=True, help="Directory for image_stats.csv/json.")
    parser.add_argument("--split", default=None, help="Optional split label stored in JSON output metadata.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args(argv)


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base / path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_action_names(dataset_root: Path) -> Tuple[Dict[int, str], List[str]]:
    warnings: List[str] = []
    candidates = [dataset_root / "list_action.txt", dataset_root / "hico_list_vb.txt"]
    action_path = next((p for p in candidates if p.exists()), None)
    if action_path is None:
        warnings.append("Action/verb mapping not found: expected list_action.txt or hico_list_vb.txt under dataset root.")
        return {}, warnings

    mapping: Dict[int, str] = {}
    with action_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if not parts[0].isdigit():
                continue
            mapping[int(parts[0])] = parts[1]

    if not mapping:
        warnings.append(f"Action/verb mapping file was found but no entries were parsed: {action_path}")
    return mapping, warnings


def load_object_names() -> Tuple[Dict[int, str], List[str]]:
    warnings: List[str] = []
    repo_root = Path(__file__).resolve().parents[1]
    meta_path = repo_root / "src" / "data" / "datasets" / "builtin_meta.py"
    if not meta_path.exists():
        warnings.append(f"Object name mapping missing: {meta_path}")
        return {}, warnings

    spec = importlib.util.spec_from_file_location("_figure1_builtin_meta", meta_path)
    if spec is None or spec.loader is None:
        warnings.append(f"Could not import object metadata from: {meta_path}")
        return {}, warnings
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    categories = getattr(module, "COCO_CATEGORIES", None)
    if not isinstance(categories, list):
        warnings.append(f"COCO_CATEGORIES not found in: {meta_path}")
        return {}, warnings

    mapping: Dict[int, str] = {}
    for item in categories:
        if not isinstance(item, dict):
            continue
        cat_id = item.get("id")
        name = item.get("name")
        if isinstance(cat_id, int) and isinstance(name, str) and name != "N/A":
            mapping[cat_id] = name
    if not mapping:
        warnings.append(f"Object metadata imported but no category names were parsed: {meta_path}")
    return mapping, warnings


def try_read_image_size(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not path.exists():
        return None, None
    try:
        from PIL import Image
    except Exception:
        return None, None
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None, None


def normalize_bbox(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    out: List[float] = []
    for value in bbox:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        out.append(float(value))
    return out


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def bbox_diag(bbox: Sequence[float]) -> float:
    width = max(0.0, float(bbox[2] - bbox[0]))
    height = max(0.0, float(bbox[3] - bbox[1]))
    return math.hypot(width, height)


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def is_near(human_bbox: Optional[Sequence[float]], object_bbox: Optional[Sequence[float]]) -> bool:
    if human_bbox is None or object_bbox is None:
        return False
    scale = max(bbox_diag(human_bbox), bbox_diag(object_bbox), 1.0)
    return center_distance(human_bbox, object_bbox) <= 1.5 * scale


def annotation_image_id(sample: Dict[str, Any], fallback_index: int) -> Any:
    for key in ["image_id", "img_id", "id"]:
        if key in sample:
            return sample[key]
    filename = sample.get("file_name") or sample.get("filename")
    if isinstance(filename, str):
        stem = Path(filename).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            return int(digits)
    return fallback_index


def build_instances(sample: Dict[str, Any], object_names: Dict[int, str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_instances = sample.get("annotations", [])
    if not isinstance(raw_instances, list):
        raw_instances = []

    human_boxes: List[Dict[str, Any]] = []
    object_boxes: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_instances):
        if not isinstance(raw, dict):
            continue
        category_id = raw.get("category_id")
        bbox = normalize_bbox(raw.get("bbox"))
        instance = {
            "index": idx,
            "bbox": bbox,
            "category_id": category_id,
            "category_name": object_names.get(category_id) if isinstance(category_id, int) else None,
        }
        if category_id == PERSON_CATEGORY_ID:
            human_boxes.append(instance)
        else:
            object_boxes.append(instance)
    return human_boxes, object_boxes


def build_pair_summary(
    pairs: Sequence[Dict[str, Any]],
    max_items: int = 6,
) -> str:
    pieces: List[str] = []
    for pair in pairs[:max_items]:
        verb = pair.get("verb_name") or f"verb:{pair.get('verb_id')}"
        obj = pair.get("object_category_name") or f"obj:{pair.get('object_category_id')}"
        pieces.append(f"h{pair.get('human_index')}-o{pair.get('object_index')}:{verb}/{obj}")
    if len(pairs) > max_items:
        pieces.append(f"+{len(pairs) - max_items} more")
    return "; ".join(pieces)


def analyze_sample(
    sample: Dict[str, Any],
    sample_index: int,
    image_root: Path,
    object_names: Dict[int, str],
    action_names: Dict[int, str],
) -> Dict[str, Any]:
    filename = sample.get("file_name") or sample.get("filename") or sample.get("image")
    if not isinstance(filename, str):
        filename = f"<missing_filename_{sample_index}>"
    image_path = image_root / filename
    width = sample.get("width")
    height = sample.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        width, height = try_read_image_size(image_path)

    human_boxes, object_boxes = build_instances(sample, object_names)
    instance_by_index: Dict[int, Dict[str, Any]] = {
        inst["index"]: inst for inst in [*human_boxes, *object_boxes]
    }

    raw_hois = sample.get("hoi_annotation", [])
    if not isinstance(raw_hois, list):
        raw_hois = []

    pairs: List[Dict[str, Any]] = []
    for pair_idx, raw in enumerate(raw_hois):
        if not isinstance(raw, dict):
            continue
        human_index = raw.get("subject_id", raw.get("human_id"))
        object_index = raw.get("object_id", raw.get("obj_id"))
        verb_id = raw.get("category_id", raw.get("verb_id", raw.get("hoi_category_id")))
        human_inst = instance_by_index.get(human_index) if isinstance(human_index, int) else None
        object_inst = instance_by_index.get(object_index) if isinstance(object_index, int) else None
        object_category_id = object_inst.get("category_id") if object_inst else None
        object_category_name = (
            object_names.get(object_category_id) if isinstance(object_category_id, int) else None
        )
        pairs.append(
            {
                "pair_index": pair_idx,
                "human_index": human_index,
                "object_index": object_index,
                "verb_id": verb_id,
                "verb_name": action_names.get(verb_id) if isinstance(verb_id, int) else None,
                "object_category_id": object_category_id,
                "object_category_name": object_category_name,
                "human_bbox": human_inst.get("bbox") if human_inst else None,
                "object_bbox": object_inst.get("bbox") if object_inst else None,
                "raw": raw,
            }
        )

    unique_humans_in_pairs = {p["human_index"] for p in pairs if isinstance(p.get("human_index"), int)}
    unique_objects_in_pairs = {p["object_index"] for p in pairs if isinstance(p.get("object_index"), int)}
    verbs = [p["verb_id"] for p in pairs if isinstance(p.get("verb_id"), int)]
    object_categories = [
        p["object_category_id"] for p in pairs if isinstance(p.get("object_category_id"), int)
    ]

    human_to_objects: Dict[int, set[int]] = defaultdict(set)
    object_to_humans: Dict[int, set[int]] = defaultdict(set)
    category_to_humans: Dict[int, set[int]] = defaultdict(set)
    near_object_to_humans: Dict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        human_index = pair.get("human_index")
        object_index = pair.get("object_index")
        object_category_id = pair.get("object_category_id")
        if isinstance(human_index, int) and isinstance(object_index, int):
            human_to_objects[human_index].add(object_index)
            object_to_humans[object_index].add(human_index)
            if is_near(pair.get("human_bbox"), pair.get("object_bbox")):
                near_object_to_humans[object_index].add(human_index)
        if isinstance(human_index, int) and isinstance(object_category_id, int):
            category_to_humans[object_category_id].add(human_index)

    verb_counts = Counter(verbs)
    object_category_counts = Counter(object_categories)
    image_id = annotation_image_id(sample, sample_index)

    return {
        "image_id": image_id,
        "filename": filename,
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "human_boxes": human_boxes,
        "object_boxes": object_boxes,
        "hoi_pairs": pairs,
        "human_count": len(human_boxes),
        "object_count": len(object_boxes),
        "hoi_pair_count": len(pairs),
        "verb_count": len(set(verbs)),
        "object_category_count": len(set(object_categories)),
        "multi_person": len(human_boxes) >= 2,
        "multiple_different_people_in_pairs": len(unique_humans_in_pairs) >= 2,
        "one_person_multiple_objects": any(len(objects) >= 2 for objects in human_to_objects.values()),
        "same_object_near_multiple_people": any(len(humans) >= 2 for humans in near_object_to_humans.values()),
        "multiple_people_same_object_category": any(len(humans) >= 2 for humans in category_to_humans.values()),
        "shared_action": any(count >= 2 for count in verb_counts.values()),
        "shared_object_category": any(count >= 2 for count in object_category_counts.values()),
        "unique_humans_in_pairs": sorted(unique_humans_in_pairs),
        "unique_objects_in_pairs": sorted(unique_objects_in_pairs),
        "verb_ids": sorted(set(verbs)),
        "verb_names": [action_names.get(v) for v in sorted(set(verbs))],
        "object_category_ids": sorted(set(object_categories)),
        "object_category_names": [object_names.get(c) for c in sorted(set(object_categories))],
        "verb_histogram": {str(k): v for k, v in sorted(verb_counts.items())},
        "object_category_histogram": {str(k): v for k, v in sorted(object_category_counts.items())},
        "short_pair_summary": build_pair_summary(pairs),
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ""
    return value


def write_csv(path: Path, image_stats: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for stat in image_stats:
            writer.writerow({col: csv_value(stat.get(col)) for col in CSV_COLUMNS})


def write_json(
    path: Path,
    metadata: Dict[str, Any],
    image_stats: Sequence[Dict[str, Any]],
    warnings: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "warnings": list(warnings),
        "images": list(image_stats),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_summary(image_stats: Sequence[Dict[str, Any]], warnings: Sequence[str]) -> None:
    total = len(image_stats)
    at_least_2_people = sum(1 for x in image_stats if x["human_count"] >= 2)
    at_least_3_pairs = sum(1 for x in image_stats if x["hoi_pair_count"] >= 3)
    at_least_4_objects = sum(1 for x in image_stats if x["object_count"] >= 4)
    all_conditions = sum(
        1
        for x in image_stats
        if x["human_count"] >= 2 and x["hoi_pair_count"] >= 3 and x["object_count"] >= 4
    )

    print("Dataset analysis summary")
    print(f"- total_images: {total}")
    print(f"- images_with_at_least_2_people: {at_least_2_people}")
    print(f"- images_with_at_least_3_hoi_pairs: {at_least_3_pairs}")
    print(f"- images_with_at_least_4_objects: {at_least_4_objects}")
    print(f"- images_satisfying_all_three_conditions: {all_conditions}")
    if warnings:
        print("Warnings")
        for warning in warnings:
            print(f"- {warning}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    annotation_path = resolve_path(args.annotation, dataset_root).resolve()
    image_root = resolve_path(args.image_root, dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    annotations = load_json(annotation_path)
    if not isinstance(annotations, list):
        raise ValueError(f"Expected top-level annotation JSON to be a list, got {type(annotations).__name__}")
    if args.max_images is not None:
        annotations = annotations[: max(0, args.max_images)]

    action_names, action_warnings = load_action_names(dataset_root)
    object_names, object_warnings = load_object_names()
    warnings = [*action_warnings, *object_warnings]
    if not action_names:
        warnings.append("Action/verb names unavailable; output keeps verb ids only.")
    if not object_names:
        warnings.append("Object category names unavailable; output keeps object category ids only.")

    image_stats: List[Dict[str, Any]] = []
    missing_image_count = 0
    for idx, sample in enumerate(annotations):
        if not isinstance(sample, dict):
            warnings.append(f"Skipped non-dict sample at index {idx}: {type(sample).__name__}")
            continue
        stat = analyze_sample(sample, idx, image_root, object_names, action_names)
        if not Path(stat["image_path"]).exists():
            missing_image_count += 1
        image_stats.append(stat)

    if missing_image_count:
        warnings.append(f"Image files missing for {missing_image_count} parsed annotations under {image_root}")

    metadata = {
        "dataset_root": str(dataset_root),
        "annotation": str(annotation_path),
        "image_root": str(image_root),
        "split": args.split,
        "max_images": args.max_images,
        "action_name_count": len(action_names),
        "object_name_count": len(object_names),
    }

    csv_path = output_dir / "image_stats.csv"
    json_path = output_dir / "image_stats.json"
    write_csv(csv_path, image_stats)
    write_json(json_path, metadata, image_stats, warnings)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print_summary(image_stats, warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
