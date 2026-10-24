#!/usr/bin/env python3
"""Load HOI annotations into normalized ImageRecord JSONL.

This is an adapter layer only. It does not rank, filter, or visualize images.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from hoi_schema import HOIRelationRecord, ImageRecord, InstanceRecord, record_to_dict
except ImportError:  # pragma: no cover - supports import from outside this directory.
    sys.path.append(str(Path(__file__).resolve().parent))
    from hoi_schema import HOIRelationRecord, ImageRecord, InstanceRecord, record_to_dict


COCO_CATEGORY_NAMES = {
    0: "N/A",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "N/A",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "N/A",
    27: "backpack",
    28: "umbrella",
    29: "N/A",
    30: "N/A",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    45: "N/A",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    66: "N/A",
    67: "dining table",
    68: "N/A",
    69: "N/A",
    70: "toilet",
    71: "N/A",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "N/A",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

COCO_CATEGORY_TO_CONTIGUOUS_ID = {
    category_id: contiguous_id
    for contiguous_id, category_id in enumerate(
        category_id
        for category_id, name in sorted(COCO_CATEGORY_NAMES.items())
        if category_id != 0 and name != "N/A"
    )
}

VCOCO_BODY_ONLY_ACTIONS = {"walk", "run", "smile", "stand"}
VCOCO_PREFERRED_OBJECT_ACTIONS = {
    "hold",
    "sit",
    "look",
    "talk_on_phone",
    "eat",
    "drink",
    "carry",
    "ride",
    "work_on_computer",
}


class LoaderStats:
    def __init__(self) -> None:
        self.skipped_relations: Counter[str] = Counter()
        self.skipped_instances: Counter[str] = Counter()
        self.warnings: list[str] = []
        self.vcoco_non_preferred_kept: Counter[str] = Counter()

    def warn(self, message: str) -> None:
        if len(self.warnings) < 200:
            self.warnings.append(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_verb_names(paths: list[Path]) -> dict[int, str]:
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        names: dict[int, str] = {}
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("-") or stripped.startswith("id"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                try:
                    names[int(parts[0])] = parts[1]
                except ValueError:
                    continue
        if names:
            return names
    return {}


def load_hico_text_labels(paths: list[Path]) -> dict[tuple[int, int], str]:
    for path in paths:
        if not path.exists() or not path.is_file() or path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not node.targets:
                continue
            target_name = getattr(node.targets[0], "id", "")
            if target_name != "hico_text_label":
                continue
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(value, dict):
                return {tuple(k): str(v) for k, v in value.items() if isinstance(k, tuple) and len(k) == 2}
    return {}


def default_hico_metadata_paths(ann_path: Path) -> tuple[list[Path], list[Path]]:
    dataset_root = ann_path.parent.parent if ann_path.parent.name == "annotations" else ann_path.parent
    verb_paths = [dataset_root / "list_action.txt", dataset_root / "hico_list_vb.txt"]
    text_paths = [Path.cwd() / "hico_text_label.py"]
    return verb_paths, text_paths


def category_name(category_id: int, names: dict[int, str] | None = None) -> str:
    if names and category_id in names:
        return names[category_id]
    return COCO_CATEGORY_NAMES.get(category_id, f"object_{category_id}")


def clamp_bbox_xyxy(raw_bbox: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def coco_xywh_to_xyxy(raw_bbox: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x, y, w, h = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    except (TypeError, ValueError):
        return None
    return clamp_bbox_xyxy((x, y, x + max(0.0, w), y + max(0.0, h)), width, height)


def resolve_image_path(file_name: str, image_root: Path, split: str) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    candidates = [
        image_root / file_name,
        image_root / split / file_name,
        image_root / "images" / split / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def image_size(record: dict[str, Any], image_path: Path, stats: LoaderStats) -> tuple[int, int] | None:
    width = record.get("width") or record.get("w")
    height = record.get("height") or record.get("h")
    if width is not None and height is not None:
        try:
            return int(width), int(height)
        except (TypeError, ValueError):
            pass
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as exc:
        stats.warn(f"Failed to read image size for {image_path}: {exc}")
        return None


def hico_hoi_name(
    hoi_id: int | None,
    verb_id: int,
    verb_name: str,
    object_name: str,
    object_category_id: int,
    text_labels: dict[tuple[int, int], str],
) -> str | None:
    object_contiguous_id = COCO_CATEGORY_TO_CONTIGUOUS_ID.get(object_category_id)
    zero_based_key = None if object_contiguous_id is None else (verb_id - 1, object_contiguous_id)
    one_based_key = (verb_id, object_category_id)
    if zero_based_key is not None and zero_based_key in text_labels:
        return text_labels[zero_based_key]
    if one_based_key in text_labels:
        return text_labels[one_based_key]
    if hoi_id is None:
        return f"person {verb_name} {object_name}"
    return f"hoi_{hoi_id}: person {verb_name} {object_name}"


def adapt_hico_records(
    ann_path: Path,
    image_root: Path,
    split: str,
    stats: LoaderStats,
    max_images: int | None,
    verb_name_path: Path | None,
    hico_text_label_path: Path | None,
) -> list[ImageRecord]:
    raw = load_json(ann_path)
    if not isinstance(raw, list):
        raise ValueError(f"HICO annotation must be a list, got {type(raw).__name__}")

    default_verb_paths, default_text_paths = default_hico_metadata_paths(ann_path)
    verb_paths = [verb_name_path] if verb_name_path else default_verb_paths
    text_paths = [hico_text_label_path] if hico_text_label_path else default_text_paths
    verb_names = load_verb_names([p for p in verb_paths if p is not None])
    text_labels = load_hico_text_labels([p for p in text_paths if p is not None])

    records: list[ImageRecord] = []
    for img_index, item in enumerate(raw):
        if max_images is not None and len(records) >= max_images:
            break
        if not isinstance(item, dict):
            stats.warn(f"Skipped non-dict HICO image record at index {img_index}")
            continue
        file_name = str(item.get("file_name") or item.get("filename") or "")
        if not file_name:
            stats.warn(f"Skipped HICO image record without file_name at index {img_index}")
            continue
        path = resolve_image_path(file_name, image_root, split)
        size = image_size(item, path, stats)
        if size is None:
            continue
        width, height = size

        instances: list[InstanceRecord] = []
        valid_instance_ids: set[int] = set()
        annotations = item.get("annotations", [])
        if not isinstance(annotations, list):
            annotations = []
            stats.warn(f"HICO image {file_name} has non-list annotations")
        for instance_id, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                stats.skipped_instances["non_dict_annotation"] += 1
                continue
            try:
                cat_id = int(ann.get("category_id"))
            except (TypeError, ValueError):
                stats.skipped_instances["missing_or_invalid_category_id"] += 1
                continue
            bbox = clamp_bbox_xyxy(ann.get("bbox"), width, height)
            if bbox is None:
                stats.skipped_instances["invalid_bbox"] += 1
                continue
            name = category_name(cat_id)
            instances.append(
                InstanceRecord(
                    instance_id=instance_id,
                    bbox_xyxy=bbox,
                    category_id=cat_id,
                    category_name=name,
                    is_person=cat_id == 1,
                )
            )
            valid_instance_ids.add(instance_id)

        relations: list[HOIRelationRecord] = []
        hois = item.get("hoi_annotation", [])
        if not isinstance(hois, list):
            hois = []
            stats.warn(f"HICO image {file_name} has non-list hoi_annotation")
        ann_by_id = {inst.instance_id: inst for inst in instances}
        for hoi in hois:
            if not isinstance(hoi, dict):
                stats.skipped_relations["non_dict_relation"] += 1
                continue
            try:
                subject_id = int(hoi.get("subject_id"))
                object_id = int(hoi.get("object_id"))
            except (TypeError, ValueError):
                stats.skipped_relations["missing_subject_or_object_id"] += 1
                continue
            if object_id < 0:
                stats.skipped_relations["missing_object_role"] += 1
                continue
            if subject_id not in valid_instance_ids or object_id not in valid_instance_ids:
                stats.skipped_relations["subject_or_object_instance_invalid"] += 1
                continue
            try:
                verb_id = int(hoi.get("category_id", hoi.get("verb_id")))
            except (TypeError, ValueError):
                stats.skipped_relations["missing_or_invalid_verb_id"] += 1
                continue
            hoi_id_raw = hoi.get("hoi_category_id")
            try:
                hoi_id = None if hoi_id_raw is None else int(hoi_id_raw)
            except (TypeError, ValueError):
                hoi_id = None
            verb_name = verb_names.get(verb_id, f"verb_{verb_id}")
            obj = ann_by_id[object_id]
            relations.append(
                HOIRelationRecord(
                    subject_id=subject_id,
                    object_id=object_id,
                    verb_id=verb_id,
                    verb_name=verb_name,
                    hoi_id=hoi_id,
                    hoi_name=hico_hoi_name(hoi_id, verb_id, verb_name, obj.category_name, obj.category_id, text_labels),
                )
            )

        image_id = str(item.get("img_id") or Path(file_name).stem)
        records.append(
            ImageRecord(
                image_id=image_id,
                file_name=file_name,
                image_path=str(path),
                width=width,
                height=height,
                split=split,
                dataset="hico",
                instances=instances,
                relations=relations,
            )
        )
    return records


def infer_coco_instances_path(ann_path: Path, split: str) -> Path | None:
    parent = ann_path.parent
    candidates = [
        parent / f"instances_{split}.json",
        parent / f"instances_{split}2014.json",
        parent / "instances_trainval2014.json",
        parent / "instances_vcoco_all_2014.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_coco_instances(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]], dict[int, str]]:
    data = load_json(path)
    images = {int(item["id"]): item for item in data.get("images", []) if "id" in item}
    anns_by_id: dict[int, dict[str, Any]] = {}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if "id" not in ann or "image_id" not in ann:
            continue
        ann_id = int(ann["id"])
        image_id = int(ann["image_id"])
        anns_by_id[ann_id] = ann
        anns_by_image[image_id].append(ann)
    names = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", []) if "id" in cat and "name" in cat}
    return images, anns_by_image, anns_by_id, names


def is_vcoco_official_action_schema(data: Any) -> bool:
    return isinstance(data, list) and bool(data) and isinstance(data[0], dict) and "action_name" in data[0] and "role_object_id" in data[0]


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def reshape_role_object_ids(flat: list[Any], role_count: int, row_count: int) -> list[list[int]]:
    values: list[int] = []
    for value in flat:
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            values.append(0)
    if role_count <= 0:
        return []
    rows: list[list[int]] = []
    for idx in range(row_count):
        start = idx * role_count
        rows.append(values[start : start + role_count])
    return rows


def adapt_vcoco_records(
    ann_path: Path,
    image_root: Path,
    split: str,
    stats: LoaderStats,
    max_images: int | None,
    coco_instances_path: Path | None,
) -> list[ImageRecord]:
    vcoco_data = load_json(ann_path)
    if not is_vcoco_official_action_schema(vcoco_data):
        stats.warn("V-COCO adapter expected official action-role schema; no records produced.")
        return []
    if coco_instances_path is None:
        coco_instances_path = infer_coco_instances_path(ann_path, split)
    if coco_instances_path is None or not coco_instances_path.exists():
        stats.warn("V-COCO requires a COCO instances JSON to recover boxes; pass --coco_instances_path.")
        return []

    images, anns_by_image, anns_by_id, coco_names = load_coco_instances(coco_instances_path)
    relations_by_image: dict[int, list[HOIRelationRecord]] = defaultdict(list)

    action_to_id: dict[str, int] = {}
    for action_index, action in enumerate(vcoco_data):
        action_name = str(action.get("action_name", f"action_{action_index}"))
        action_to_id.setdefault(action_name, action_index)
        role_names = [str(x) for x in as_list(action.get("role_name"))]
        if action_name in VCOCO_BODY_ONLY_ACTIONS or len(role_names) <= 1:
            positives = sum(1 for label in as_list(action.get("label")) if int(label) == 1)
            stats.skipped_relations["vcoco_body_only_action"] += positives
            continue
        if action_name not in VCOCO_PREFERRED_OBJECT_ACTIONS:
            stats.vcoco_non_preferred_kept[action_name] += 1

        ann_ids = as_list(action.get("ann_id"))
        labels = as_list(action.get("label"))
        image_ids = as_list(action.get("image_id"))
        role_rows = reshape_role_object_ids(as_list(action.get("role_object_id")), len(role_names), len(ann_ids))

        for row_idx, ann_id_raw in enumerate(ann_ids):
            try:
                subject_ann_id = int(ann_id_raw)
                label = int(labels[row_idx])
                image_id = int(image_ids[row_idx])
            except (TypeError, ValueError, IndexError):
                stats.skipped_relations["vcoco_malformed_action_row"] += 1
                continue
            if label != 1:
                continue
            if subject_ann_id not in anns_by_id:
                stats.skipped_relations["vcoco_subject_ann_missing"] += 1
                continue
            role_row = role_rows[row_idx] if row_idx < len(role_rows) else []
            if len(role_row) <= 1:
                stats.skipped_relations["vcoco_role_object_missing"] += 1
                continue
            for role_idx, object_ann_id in enumerate(role_row[1:], start=1):
                if object_ann_id <= 0:
                    stats.skipped_relations["vcoco_role_object_missing"] += 1
                    continue
                if object_ann_id not in anns_by_id:
                    stats.skipped_relations["vcoco_object_ann_missing"] += 1
                    continue
                role_name = role_names[role_idx] if role_idx < len(role_names) else "obj"
                verb_id = action_to_id[action_name]
                verb_name = f"{action_name}_{role_name}"
                relations_by_image[image_id].append(
                    HOIRelationRecord(
                        subject_id=subject_ann_id,
                        object_id=object_ann_id,
                        verb_id=verb_id,
                        verb_name=verb_name,
                        hoi_id=None,
                        hoi_name=f"person {verb_name}",
                    )
                )

    records: list[ImageRecord] = []
    for image_id in sorted(relations_by_image):
        if max_images is not None and len(records) >= max_images:
            break
        info = images.get(image_id)
        if not info:
            stats.warn(f"V-COCO image id {image_id} missing from COCO instances file")
            continue
        file_name = str(info.get("file_name") or f"{image_id}.jpg")
        path = resolve_image_path(file_name, image_root, split)
        size = image_size(info, path, stats)
        if size is None:
            continue
        width, height = size
        instances: list[InstanceRecord] = []
        valid_ids: set[int] = set()
        for ann in anns_by_image.get(image_id, []):
            try:
                instance_id = int(ann["id"])
                cat_id = int(ann.get("category_id"))
            except (TypeError, ValueError, KeyError):
                stats.skipped_instances["vcoco_missing_id_or_category"] += 1
                continue
            bbox = coco_xywh_to_xyxy(ann.get("bbox"), width, height)
            if bbox is None:
                stats.skipped_instances["vcoco_invalid_bbox"] += 1
                continue
            name = category_name(cat_id, coco_names)
            instances.append(
                InstanceRecord(
                    instance_id=instance_id,
                    bbox_xyxy=bbox,
                    category_id=cat_id,
                    category_name=name,
                    is_person=cat_id == 1,
                )
            )
            valid_ids.add(instance_id)
        relations = []
        for rel in relations_by_image[image_id]:
            if rel.subject_id not in valid_ids or rel.object_id not in valid_ids:
                stats.skipped_relations["vcoco_subject_or_object_instance_invalid"] += 1
                continue
            relations.append(rel)
        records.append(
            ImageRecord(
                image_id=str(image_id),
                file_name=file_name,
                image_path=str(path),
                width=width,
                height=height,
                split=split,
                dataset="vcoco",
                instances=instances,
                relations=relations,
            )
        )
    return records


def write_jsonl(records: list[ImageRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record_to_dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def top_names(records: list[ImageRecord]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    objects: Counter[str] = Counter()
    verbs: Counter[str] = Counter()
    for record in records:
        for instance in record.instances:
            objects[instance.category_name] += 1
        for relation in record.relations:
            verbs[relation.verb_name] += 1
    return objects.most_common(20), verbs.most_common(20)


def write_sanity_report(records: list[ImageRecord], stats: LoaderStats, output_path: Path, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_count = len(records)
    avg_instances = sum(len(record.instances) for record in records) / image_count if image_count else 0.0
    avg_relations = sum(len(record.relations) for record in records) / image_count if image_count else 0.0
    object_top, verb_top = top_names(records)

    lines: list[str] = []
    lines.append("# HOI Loader Sanity Report")
    lines.append("")
    lines.append("This report summarizes normalized ImageRecord output only. No image scoring or candidate filtering is performed.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Dataset: `{args.dataset}`")
    lines.append(f"- Annotation path: `{args.ann_path}`")
    lines.append(f"- Image root: `{args.image_root}`")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Output cache: `{args.output_cache}`")
    if getattr(args, "coco_instances_path", None):
        lines.append(f"- COCO instances path: `{args.coco_instances_path}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Number of images: `{image_count}`")
    lines.append(f"- Average instances per image: `{avg_instances:.3f}`")
    lines.append(f"- Average relations per image: `{avg_relations:.3f}`")
    lines.append(f"- Total skipped instances: `{sum(stats.skipped_instances.values())}`")
    lines.append(f"- Total skipped relations: `{sum(stats.skipped_relations.values())}`")
    lines.append("")
    lines.append("## Skipped Instances")
    lines.append("")
    if stats.skipped_instances:
        for reason, count in stats.skipped_instances.most_common():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Skipped Relations")
    lines.append("")
    if stats.skipped_relations:
        for reason, count in stats.skipped_relations.most_common():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    if stats.vcoco_non_preferred_kept:
        lines.append("")
        lines.append("## V-COCO Non-Preferred Object Actions Kept")
        lines.append("")
        for action, count in stats.vcoco_non_preferred_kept.most_common():
            lines.append(f"- `{action}`: {count}")
    lines.append("")
    lines.append("## Top 20 Object Names")
    lines.append("")
    if object_top:
        for name, count in object_top:
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Top 20 Verb Names")
    lines.append("")
    if verb_top:
        for name, count in verb_top:
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Example ImageRecords")
    lines.append("")
    for idx, record in enumerate(records[:5], start=1):
        preview = asdict(record)
        preview["instances"] = preview["instances"][:5]
        preview["relations"] = preview["relations"][:5]
        lines.append(f"### Example {idx}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(preview, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    if stats.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in stats.warnings:
            lines.append(f"- {warning}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize HOI annotations into ImageRecord JSONL.")
    parser.add_argument("--dataset", choices=("hico", "vcoco"), required=True)
    parser.add_argument("--ann_path", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_cache", type=Path, required=True)
    parser.add_argument("--sanity_report", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=None, help="Optional debug limit; omit for full conversion.")
    parser.add_argument("--verb_name_path", type=Path, default=None, help="Optional HICO verb name txt file.")
    parser.add_argument("--hico_text_label_path", type=Path, default=None, help="Optional hico_text_label.py path.")
    parser.add_argument("--coco_instances_path", type=Path, default=None, help="Required for official V-COCO action-role annotations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = LoaderStats()
    ann_path = args.ann_path.resolve()
    image_root = args.image_root.resolve()
    output_cache = args.output_cache.resolve()
    sanity_report = args.sanity_report.resolve()

    if args.dataset == "hico":
        records = adapt_hico_records(
            ann_path=ann_path,
            image_root=image_root,
            split=args.split,
            stats=stats,
            max_images=args.max_images,
            verb_name_path=args.verb_name_path,
            hico_text_label_path=args.hico_text_label_path,
        )
    else:
        records = adapt_vcoco_records(
            ann_path=ann_path,
            image_root=image_root,
            split=args.split,
            stats=stats,
            max_images=args.max_images,
            coco_instances_path=args.coco_instances_path,
        )

    write_jsonl(records, output_cache)
    write_sanity_report(records, stats, sanity_report, args)
    print(f"Wrote {len(records)} ImageRecord rows to {output_cache}")
    print(f"Wrote sanity report to {sanity_report}")
    print(f"Skipped relations: {sum(stats.skipped_relations.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
