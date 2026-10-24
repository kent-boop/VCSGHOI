#!/usr/bin/env python3
"""Stage 2 hard filtering for EvidenceHOI figure candidate images.

Input is normalized ImageRecord JSONL from Stage 1. This script only applies
hard structural filters over GT boxes and HOI relations; it does not run models,
load pixels for scoring, or create visualizations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from hoi_schema import ImageRecord, InstanceRecord, HOIRelationRecord, record_from_dict
except ImportError:  # pragma: no cover - supports execution from repo root.
    sys.path.append(str(Path(__file__).resolve().parent))
    from hoi_schema import ImageRecord, InstanceRecord, HOIRelationRecord, record_from_dict


ACTION_GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sports_ball": (
        "kick",
        "hit",
        "catch",
        "throw",
        "dribble",
        "serve",
        "swing",
        "play baseball",
        "play tennis",
        "skateboard",
        "ski",
        "snowboard",
    ),
    "manipulation": ("hold", "carry", "use", "open", "close", "cut", "repair", "wash", "clean"),
    "posture": ("sit", "sit_on", "sit on", "stand", "lie", "lie_on", "lean"),
    "perception": ("look", "look_at", "look at", "watch", "read"),
    "communication": ("talk_on_phone", "talk on phone", "talk", "listen"),
    "locomotion": ("ride", "walk", "run", "pull", "push"),
    "consumption": ("eat", "drink", "sip"),
    "work_study": ("work_on", "work on", "type_on", "type on", "use laptop", "use computer", "write"),
}

SPORTS_OBJECTS = {
    "sports ball",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "skis",
    "snowboard",
    "skateboard",
    "surfboard",
}

SMALL_OBJECTS = {
    "cup",
    "cell phone",
    "phone",
    "remote",
    "mouse",
    "fork",
    "knife",
    "spoon",
    "bottle",
    "wine glass",
    "book",
    "toothbrush",
    "scissors",
}

AMBIGUOUS_CONTEXT_OBJECTS = {"N/A"}


def bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = bbox_area((x1, y1, x2, y2))
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def union_box(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def expand_box(
    box: tuple[float, float, float, float],
    scale: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    cx, cy = bbox_center(box)
    w = (box[2] - box[0]) * scale
    h = (box[3] - box[1]) * scale
    return (
        max(0.0, cx - w / 2.0),
        max(0.0, cy - h / 2.0),
        min(float(width), cx + w / 2.0),
        min(float(height), cy + h / 2.0),
    )


def point_inside(box: tuple[float, float, float, float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def area_ratio(instance: InstanceRecord, image_area: float) -> float:
    return bbox_area(instance.bbox_xyxy) / image_area if image_area > 0 else 0.0


def action_group(verb_name: str) -> str:
    normalized = verb_name.lower().replace("-", "_")
    spaced = normalized.replace("_", " ")
    for group, keywords in ACTION_GROUP_KEYWORDS.items():
        for keyword in keywords:
            key_norm = keyword.lower().replace("-", "_")
            key_spaced = key_norm.replace("_", " ")
            if key_norm in normalized or key_spaced in spaced:
                return group
    return "other"


def load_records(path: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(record_from_dict(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"Failed to parse ImageRecord JSONL at line {line_no}: {exc}") from exc
    return records


def ensure_size(record: ImageRecord, image_root: Path | None) -> tuple[int, int]:
    if record.width > 0 and record.height > 0:
        return record.width, record.height
    candidates = [Path(record.image_path)]
    if image_root is not None:
        candidates.append(image_root / record.file_name)
    for path in candidates:
        if path.exists():
            with Image.open(path) as img:
                return img.size
    return max(record.width, 1), max(record.height, 1)


def pair_sort_key(pair_info: dict[str, Any]) -> tuple[float, int, int]:
    return (
        float(pair_info.get("context_count", 0)),
        int(pair_info.get("num_actions", 0)),
        int(pair_info.get("object_area_rank", 0)),
    )


def compute_max_matching(pairs: list[tuple[int, int]]) -> int:
    humans = sorted({h for h, _ in pairs})
    objects = sorted({o for _, o in pairs})
    adjacency = {h: [o for hh, o in pairs if hh == h] for h in humans}
    matched_object: dict[int, int] = {}

    def augment(human_id: int, seen: set[int]) -> bool:
        for object_id in adjacency.get(human_id, []):
            if object_id in seen:
                continue
            seen.add(object_id)
            if object_id not in matched_object or augment(matched_object[object_id], seen):
                matched_object[object_id] = human_id
                return True
        return False

    result = 0
    for human_id in humans:
        if augment(human_id, set()):
            result += 1
    return result


def dominant_ratio(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    return counts.most_common(1)[0][1] / len(values)


def entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def context_jaccard_mean(pair_infos: list[dict[str, Any]]) -> float:
    sets = [set(item["instance_id"] for item in pair.get("context_set", [])) for pair in pair_infos]
    scores: list[float] = []
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            union = a | b
            if not union:
                scores.append(0.0)
            else:
                scores.append(len(a & b) / len(union))
    return mean(scores)


def average_pair_union_overlap(
    pair_infos: list[dict[str, Any]],
    instances_by_id: dict[int, InstanceRecord],
) -> float:
    union_boxes: list[tuple[float, float, float, float]] = []
    for pair in pair_infos:
        human = instances_by_id.get(int(pair["subject_id"]))
        obj = instances_by_id.get(int(pair["object_id"]))
        if human is None or obj is None:
            continue
        union_boxes.append(union_box(human.bbox_xyxy, obj.bbox_xyxy))
    overlaps: list[float] = []
    for i, a in enumerate(union_boxes):
        for b in union_boxes[i + 1 :]:
            overlaps.append(bbox_iou(a, b))
    return mean(overlaps)


def jaccard_penalty(value: float) -> float:
    if value == 0.0:
        return 0.1
    if value < 0.2:
        return min(0.15, (0.2 - value) * 0.5)
    if value > 0.7:
        return min(0.2, (value - 0.7) * 0.5)
    return 0.0


def candidate_contexts(
    record: ImageRecord,
    pair: tuple[int, int],
    instances_by_id: dict[int, InstanceRecord],
    args: argparse.Namespace,
    image_area: float,
) -> list[dict[str, Any]]:
    h_id, o_id = pair
    human = instances_by_id[h_id]
    obj = instances_by_id[o_id]
    expanded = expand_box(union_box(human.bbox_xyxy, obj.bbox_xyxy), args.context_expand_scale, record.width, record.height)
    contexts: list[dict[str, Any]] = []
    for inst in record.instances:
        if inst.instance_id in {h_id, o_id}:
            continue
        if inst.category_name in AMBIGUOUS_CONTEXT_OBJECTS:
            continue
        inst_area = area_ratio(inst, image_area)
        if inst_area < args.context_object_min_area:
            continue
        cx, cy = bbox_center(inst.bbox_xyxy)
        if not point_inside(expanded, cx, cy):
            continue
        contexts.append(
            {
                "instance_id": inst.instance_id,
                "category_id": inst.category_id,
                "category_name": inst.category_name,
                "bbox_xyxy": list(inst.bbox_xyxy),
                "area_ratio": round(inst_area, 6),
                "path": f"h{h_id} -> k{inst.instance_id} -> o{o_id}",
            }
        )
    return contexts


def compute_soft_scores(
    counts: dict[str, Any],
    pair_infos: list[dict[str, Any]],
    action_names: list[str],
    pair_action_groups: list[str],
    object_categories: list[str],
    instances_by_id: dict[int, InstanceRecord],
    image_width: int,
    image_height: int,
    reject_severe_sports: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], str]:
    selected_pairs = pair_infos[:3]
    warnings: list[str] = []

    pair_diversity = 0.5 * min(counts["num_valid_hoi_pairs"] / 5.0, 1.0) + 0.5 * min(
        counts["max_unique_person_object_matching"] / 3.0, 1.0
    )
    action_diversity = min(counts["num_unique_action_groups"] / 3.0, 1.0)
    object_diversity = min(counts["num_unique_object_categories"] / 5.0, 1.0)

    pair_context_scores = [min(float(pair["context_count"]) / 3.0, 1.0) for pair in selected_pairs]
    evidence_context_score = mean(pair_context_scores)
    jaccard_mean = context_jaccard_mean(selected_pairs)
    jaccard_pen = jaccard_penalty(jaccard_mean) if len(selected_pairs) >= 2 else 0.0
    evidence_context_score = clamp01(evidence_context_score - jaccard_pen)
    if len(selected_pairs) >= 2 and (jaccard_mean < 0.2 or jaccard_mean > 0.7):
        warnings.append(f"context_jaccard_outside_ideal:{jaccard_mean:.3f}")

    clarity_values: list[float] = []
    object_too_small_penalty = 0.0
    for pair in selected_pairs:
        person_area_score = min(float(pair["human_area"]) / 0.04, 1.0)
        object_area = float(pair["object_area"])
        object_area_score = min(object_area / 0.01, 1.0)
        pair_clarity = 0.6 * person_area_score + 0.4 * object_area_score
        if object_area < 0.001:
            pair_clarity -= 0.3
            object_too_small_penalty = max(object_too_small_penalty, 0.3)
            warnings.append(f"very_small_object:h{pair['subject_id']}-o{pair['object_id']}")
        if pair.get("small_object_relaxed"):
            warnings.append(f"small_object_relaxed:h{pair['subject_id']}-o{pair['object_id']}")
        clarity_values.append(clamp01(pair_clarity))
    overlap_penalty = 0.5 if counts["max_human_human_iou"] > 0.5 else 0.0
    visual_clarity = clamp01(mean(clarity_values) - overlap_penalty)

    if selected_pairs:
        primary = selected_pairs[0]
        cx = float(primary["center_x"])
        cy = float(primary["center_y"])
        center_distance = math.dist((cx, cy), (0.5, 0.5))
        image_diag = math.sqrt(0.5)
        center_score = clamp01(1.0 - center_distance / image_diag)
        if not (0.2 < cx < 0.8 and 0.2 < cy < 0.8):
            center_score = max(0.0, center_score - 0.25)
            warnings.append(f"primary_pair_near_edge:center=({cx:.3f},{cy:.3f})")
        # Slightly favor center-left without forcing layout.
        if 0.25 <= cx <= 0.55 and 0.25 <= cy <= 0.75:
            center_score = min(1.0, center_score + 0.05)
        if len(selected_pairs) >= 2:
            x_spread = max(float(pair["center_x"]) for pair in selected_pairs) - min(float(pair["center_x"]) for pair in selected_pairs)
            y_spread = max(float(pair["center_y"]) for pair in selected_pairs) - min(float(pair["center_y"]) for pair in selected_pairs)
            spread_score = clamp01((x_spread + y_spread) / 0.8)
        else:
            spread_score = 0.0
        layout_score = clamp01(0.75 * center_score + 0.25 * spread_score)
    else:
        layout_score = 0.0

    homogeneity_penalty = 0.0
    if counts["dominant_action_group_ratio"] > 0.75:
        homogeneity_penalty += 1.0
    if counts["dominant_object_category_ratio"] > 0.65:
        homogeneity_penalty += 1.0
    if reject_severe_sports:
        homogeneity_penalty += 1.5
    if counts["num_unique_action_groups"] == 1 and counts["num_valid_hoi_pairs"] >= 3:
        homogeneity_penalty += 1.0

    avg_pair_union_overlap = average_pair_union_overlap(pair_infos, instances_by_id)
    crowding_penalty = 0.0
    if counts["total_gt_boxes"] > 18:
        crowding_penalty += 1.0
    if counts["num_valid_hoi_pairs"] > 10:
        crowding_penalty += 1.0
    if counts["num_persons"] > 6:
        crowding_penalty += 1.0
    if avg_pair_union_overlap > 0.7:
        crowding_penalty += 1.0
    elif avg_pair_union_overlap > 0.45:
        crowding_penalty += 0.5

    total_score = (
        2.0 * pair_diversity
        + 1.5 * action_diversity
        + 1.5 * object_diversity
        + 2.0 * evidence_context_score
        + 1.5 * visual_clarity
        + 1.0 * layout_score
        - 2.0 * homogeneity_penalty
        - 1.5 * crowding_penalty
    )

    component_scores = {
        "PairDiversity": round(pair_diversity, 6),
        "ActionDiversity": round(action_diversity, 6),
        "ObjectDiversity": round(object_diversity, 6),
        "EvidenceContextScore": round(evidence_context_score, 6),
        "VisualClarity": round(visual_clarity, 6),
        "LayoutScore": round(layout_score, 6),
        "HomogeneityPenalty": round(homogeneity_penalty, 6),
        "CrowdingPenalty": round(crowding_penalty, 6),
        "action_entropy": round(entropy(action_names), 6),
        "action_group_entropy": round(entropy(pair_action_groups), 6),
        "context_jaccard_mean": round(jaccard_mean, 6),
        "context_jaccard_penalty": round(jaccard_pen, 6),
        "average_pair_union_overlap": round(avg_pair_union_overlap, 6),
        "object_too_small_penalty": round(object_too_small_penalty, 6),
    }
    total_score = round(total_score, 6)

    if homogeneity_penalty >= 2.0:
        warnings.append("high_homogeneity_penalty")
    if crowding_penalty >= 1.5:
        warnings.append("high_crowding_penalty")

    if total_score >= 5.5 and counts["num_valid_hoi_pairs"] >= 3 and counts["num_unique_action_groups"] >= 3:
        bucket = "A"
    elif total_score >= 4.0 and counts["visible_pair_candidates"] >= 2:
        bucket = "B"
    else:
        bucket = "C"

    for idx, pair in enumerate(selected_pairs, start=1):
        pair["rank"] = idx
        pair["context_object_names"] = [item["category_name"] for item in pair.get("context_set", [])]
        pair["context_ids"] = [item["instance_id"] for item in pair.get("context_set", [])]

    return {"total_score": total_score, "components": component_scores}, selected_pairs, sorted(set(warnings)), bucket


def mode_thresholds(args: argparse.Namespace) -> dict[str, Any]:
    min_actions = args.min_unique_action_categories
    min_pairs = args.min_valid_pairs
    max_pairs = args.max_valid_pairs
    min_context = args.min_primary_pair_context_count
    min_mean_context = args.min_mean_pair_context_count_top3
    min_matching = args.min_matching

    if min_actions is None:
        if args.mode == "figure1":
            min_actions = 3
        elif args.mode == "figure2":
            min_actions = 2
        elif args.mode in {"relaxed", "qualitative"}:
            min_actions = 2
        else:
            min_actions = 3
    if args.mode == "figure2":
        min_pairs = min_pairs if args.min_valid_pairs_user_set else 2
        max_pairs = max_pairs if args.max_valid_pairs_user_set else 8
        min_context = min_context if args.min_primary_context_user_set else 1
    if args.mode == "qualitative":
        min_pairs = min_pairs if args.min_valid_pairs_user_set else 2
        max_pairs = max_pairs if args.max_valid_pairs_user_set else 12
        min_context = min_context if args.min_primary_context_user_set else 1
        min_mean_context = min_mean_context if args.min_mean_context_user_set else 0.5
        min_matching = min_matching if args.min_matching_user_set else 1
    if args.mode == "relaxed":
        min_pairs = min_pairs if args.min_valid_pairs_user_set else 2
        max_pairs = max_pairs if args.max_valid_pairs_user_set else 10
        min_context = min_context if args.min_primary_context_user_set else 1
        min_matching = min_matching if args.min_matching_user_set else 2
    return {
        "min_actions": min_actions,
        "min_pairs": min_pairs,
        "max_pairs": max_pairs,
        "min_context": min_context,
        "min_mean_context": min_mean_context,
        "min_matching": min_matching,
    }


def analyze_record(record: ImageRecord, args: argparse.Namespace, thresholds: dict[str, Any]) -> dict[str, Any]:
    width, height = ensure_size(record, args.image_root)
    record.width = width
    record.height = height
    image_area = float(max(width * height, 1))
    instances_by_id = {inst.instance_id: inst for inst in record.instances}
    persons = [inst for inst in record.instances if inst.is_person]

    pair_actions: dict[tuple[int, int], set[str]] = defaultdict(set)
    pair_action_ids: dict[tuple[int, int], set[int]] = defaultdict(set)
    pair_relations: dict[tuple[int, int], list[HOIRelationRecord]] = defaultdict(list)
    for rel in record.relations:
        if rel.subject_id == rel.object_id:
            continue
        if rel.subject_id not in instances_by_id or rel.object_id not in instances_by_id:
            continue
        subject = instances_by_id[rel.subject_id]
        obj = instances_by_id[rel.object_id]
        if not subject.is_person:
            continue
        if obj.is_person:
            continue
        pair = (rel.subject_id, rel.object_id)
        pair_actions[pair].add(rel.verb_name)
        pair_action_ids[pair].add(rel.verb_id)
        pair_relations[pair].append(rel)

    valid_pairs = sorted(pair_actions.keys())
    objects_in_pairs = [instances_by_id[o] for _, o in valid_pairs]
    humans_in_pairs = [instances_by_id[h] for h, _ in valid_pairs]
    object_categories = [obj.category_name for obj in objects_in_pairs]
    action_names = sorted({name for names in pair_actions.values() for name in names})
    action_groups = [action_group(name) for name in action_names]
    pair_action_groups = [action_group(name) for pair in valid_pairs for name in pair_actions[pair]]

    human_ious = []
    for i, a in enumerate(persons):
        for b in persons[i + 1 :]:
            human_ious.append(bbox_iou(a.bbox_xyxy, b.bbox_xyxy))
    key_object_ious = []
    for i, a in enumerate(objects_in_pairs):
        for b in objects_in_pairs[i + 1 :]:
            key_object_ious.append(bbox_iou(a.bbox_xyxy, b.bbox_xyxy))

    pair_infos: list[dict[str, Any]] = []
    visible_pair_candidates = 0
    small_object_relaxed = False
    for pair in valid_pairs:
        h_id, o_id = pair
        human = instances_by_id[h_id]
        obj = instances_by_id[o_id]
        human_area = area_ratio(human, image_area)
        object_area = area_ratio(obj, image_area)
        object_min_area_ok = object_area >= args.main_object_min_area
        object_small_relaxed = False
        if not object_min_area_ok and obj.category_name in SMALL_OBJECTS and object_area >= args.small_object_min_area:
            object_min_area_ok = True
            object_small_relaxed = True
            small_object_relaxed = True
        context = candidate_contexts(record, pair, instances_by_id, args, image_area)
        pair_center = bbox_center(union_box(human.bbox_xyxy, obj.bbox_xyxy))
        if human_area >= args.secondary_person_min_area and object_min_area_ok:
            visible_pair_candidates += 1
        pair_infos.append(
            {
                "subject_id": h_id,
                "object_id": o_id,
                "human_bbox_xyxy": list(human.bbox_xyxy),
                "object_bbox_xyxy": list(obj.bbox_xyxy),
                "object_category_id": obj.category_id,
                "object_category_name": obj.category_name,
                "actions": sorted(pair_actions[pair]),
                "action_ids": sorted(pair_action_ids[pair]),
                "action_groups": sorted({action_group(name) for name in pair_actions[pair]}),
                "num_actions": len(pair_actions[pair]),
                "human_area": round(human_area, 6),
                "object_area": round(object_area, 6),
                "small_object_relaxed": object_small_relaxed,
                "context_count": len(context),
                "context_set": context,
                "candidate_paths": [item["path"] for item in context],
                "center_x": round(pair_center[0] / max(width, 1), 6),
                "center_y": round(pair_center[1] / max(height, 1), 6),
                "object_area_rank": int(object_area * 1_000_000),
            }
        )
    pair_infos.sort(key=pair_sort_key, reverse=True)

    primary = pair_infos[0] if pair_infos else None
    top3 = pair_infos[:3]
    mean_context_top3 = sum(float(p["context_count"]) for p in top3) / len(top3) if top3 else 0.0
    secondary_pair_context_count = int(top3[1]["context_count"]) if len(top3) >= 2 else 0
    main_pair_center_x = float(primary["center_x"]) if primary else None
    main_pair_center_y = float(primary["center_y"]) if primary else None
    main_pair_area_ok = False
    secondary_pair_area_ok = False
    if pair_infos:
        main_pair_area_ok = bool(
            float(pair_infos[0]["human_area"]) >= args.main_person_min_area
            and (
                float(pair_infos[0]["object_area"]) >= args.main_object_min_area
                or bool(pair_infos[0]["small_object_relaxed"])
            )
        )
    if len(pair_infos) >= 2:
        secondary_pair_area_ok = any(float(p["human_area"]) >= args.secondary_person_min_area for p in pair_infos[1:])
    elif pair_infos:
        secondary_pair_area_ok = float(pair_infos[0]["human_area"]) >= args.secondary_person_min_area

    dominant_action_group_ratio = dominant_ratio(pair_action_groups)
    dominant_object_category_ratio = dominant_ratio(object_categories)
    sports_group_count = sum(1 for group in pair_action_groups if group == "sports_ball")
    sports_object_count = sum(1 for name in object_categories if name in SPORTS_OBJECTS)
    sports_dominates = bool(pair_action_groups) and sports_group_count / len(pair_action_groups) > args.sports_dominate_ratio
    sports_objects_dominate = bool(object_categories) and sports_object_count / len(object_categories) > args.sports_object_dominate_ratio
    reject_severe_sports = bool((sports_dominates or sports_objects_dominate) and len(set(object_categories)) <= 2)

    counts = {
        "num_persons": len(persons),
        "num_instances": len(record.instances),
        "num_gt_boxes": len(record.instances),
        "total_gt_boxes": len(record.instances),
        "num_valid_hoi_pairs": len(valid_pairs),
        "num_unique_humans_in_pairs": len({h for h, _ in valid_pairs}),
        "num_unique_objects_in_pairs": len({o for _, o in valid_pairs}),
        "num_unique_object_categories": len(set(object_categories)),
        "num_unique_action_categories": len(action_names),
        "num_unique_action_groups": len(set(action_groups)),
        "dominant_action_group_ratio": round(dominant_action_group_ratio, 6),
        "dominant_object_category_ratio": round(dominant_object_category_ratio, 6),
        "max_human_human_iou": round(max(human_ious) if human_ious else 0.0, 6),
        "max_key_object_iou": round(max(key_object_ious) if key_object_ious else 0.0, 6),
        "max_unique_person_object_matching": compute_max_matching(valid_pairs),
        "primary_pair_context_count": int(primary["context_count"]) if primary else 0,
        "secondary_pair_context_count": secondary_pair_context_count,
        "mean_pair_context_count_top3": round(mean_context_top3, 6),
        "main_pair_center_x": main_pair_center_x,
        "main_pair_center_y": main_pair_center_y,
        "visible_pair_candidates": visible_pair_candidates,
    }

    reject_flags: dict[str, bool] = {
        "reject_num_persons": not (args.min_persons <= counts["num_persons"] <= args.max_persons),
        "reject_num_valid_hoi_pairs": not (thresholds["min_pairs"] <= counts["num_valid_hoi_pairs"] <= thresholds["max_pairs"]),
        "reject_total_gt_boxes": not (args.min_gt_boxes <= counts["total_gt_boxes"] <= args.max_gt_boxes),
        "reject_unique_object_categories": counts["num_unique_object_categories"] < args.min_unique_object_categories,
        "reject_unique_action_categories": counts["num_unique_action_categories"] < thresholds["min_actions"],
        "reject_matching": counts["max_unique_person_object_matching"] < thresholds["min_matching"],
        "reject_main_person_area": not main_pair_area_ok,
        "reject_secondary_person_area": not secondary_pair_area_ok,
        "reject_primary_pair_context": counts["primary_pair_context_count"] < thresholds["min_context"],
        "reject_mean_context_top3": counts["mean_pair_context_count_top3"] < thresholds["min_mean_context"],
        "reject_human_overlap": counts["max_human_human_iou"] > args.max_human_human_iou,
        "reject_action_homogeneity": counts["dominant_action_group_ratio"] > args.max_dominant_action_group_ratio,
        "reject_object_homogeneity": counts["dominant_object_category_ratio"] > args.max_dominant_object_category_ratio,
        "reject_severe_sports": reject_severe_sports,
    }

    score_info, selected_top3, warnings, bucket = compute_soft_scores(
        counts=counts,
        pair_infos=pair_infos,
        action_names=action_names,
        pair_action_groups=pair_action_groups,
        object_categories=object_categories,
        instances_by_id=instances_by_id,
        image_width=width,
        image_height=height,
        reject_severe_sports=reject_severe_sports,
    )

    result = {
        "image_id": record.image_id,
        "file_name": record.file_name,
        "image_path": record.image_path,
        "score": score_info["total_score"],
        "total_score": score_info["total_score"],
        "component_scores": score_info["components"],
        "counts": counts,
        "top_pairs": pair_infos[: min(args.max_top_pairs, len(pair_infos))],
        "top_3_pairs": selected_top3,
        "selected_primary_pair": selected_top3[0] if selected_top3 else None,
        "selected_secondary_pairs": selected_top3[1:] if len(selected_top3) > 1 else [],
        "context_sets": {
            f"{p['subject_id']}->{p['object_id']}": p["context_set"] for p in pair_infos[: min(args.max_top_pairs, len(pair_infos))]
        },
        "reject_flags": reject_flags,
        "small_object_relaxed": small_object_relaxed,
        "warnings": warnings,
        "bucket_suggestion": bucket,
        "split": record.split,
        "dataset": record.dataset,
    }
    result["passed"] = not any(reject_flags.values())
    result["reject_reasons"] = [name for name, value in reject_flags.items() if value]
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image_id",
        "file_name",
        "image_path",
        "score",
        "total_score",
        "dataset",
        "split",
        "bucket_suggestion",
        "num_persons",
        "num_instances",
        "num_gt_boxes",
        "total_gt_boxes",
        "num_valid_hoi_pairs",
        "num_unique_humans_in_pairs",
        "num_unique_objects_in_pairs",
        "num_unique_object_categories",
        "num_unique_action_categories",
        "num_unique_action_groups",
        "dominant_action_group_ratio",
        "dominant_object_category_ratio",
        "max_human_human_iou",
        "max_key_object_iou",
        "max_unique_person_object_matching",
        "primary_pair_context_count",
        "secondary_pair_context_count",
        "mean_pair_context_count_top3",
        "main_pair_center_x",
        "main_pair_center_y",
        "visible_pair_candidates",
        "PairDiversity",
        "ActionDiversity",
        "ObjectDiversity",
        "EvidenceContextScore",
        "VisualClarity",
        "LayoutScore",
        "HomogeneityPenalty",
        "CrowdingPenalty",
        "action_entropy",
        "action_group_entropy",
        "context_jaccard_mean",
        "context_jaccard_penalty",
        "average_pair_union_overlap",
        "small_object_relaxed",
        "selected_primary_pair",
        "selected_secondary_pairs",
        "warnings",
        "top_pairs_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            counts = row["counts"]
            components = row.get("component_scores", {})
            out = {field: row.get(field) for field in fields}
            out.update({key: counts.get(key) for key in counts})
            out.update({key: components.get(key) for key in components if key in fields})
            summaries = []
            for pair in row.get("top_pairs", [])[:3]:
                summaries.append(
                    f"h{pair['subject_id']}-o{pair['object_id']}:{pair['object_category_name']}:{'/'.join(pair['actions'])}:ctx{pair['context_count']}"
                )
            out["top_pairs_summary"] = " | ".join(summaries)
            primary = row.get("selected_primary_pair") or {}
            if primary:
                out["selected_primary_pair"] = (
                    f"h{primary['subject_id']}-o{primary['object_id']}:{primary['object_category_name']}:"
                    f"{'/'.join(primary['actions'])}:ctx{primary['context_count']}"
                )
            secondary = []
            for pair in row.get("selected_secondary_pairs", []):
                secondary.append(f"h{pair['subject_id']}-o{pair['object_id']}:{pair['object_category_name']}:ctx{pair['context_count']}")
            out["selected_secondary_pairs"] = " | ".join(secondary)
            out["warnings"] = ";".join(row.get("warnings", []))
            writer.writerow(out)


def write_rejected_summary(path: Path, analyzed: list[dict[str, Any]]) -> None:
    reason_counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()
    for row in analyzed:
        reasons = row.get("reject_reasons", [])
        for reason in reasons:
            reason_counts[reason] += 1
        combo_counts[",".join(reasons) if reasons else "passed"] += 1
    payload = {
        "total_images": len(analyzed),
        "passed_images": sum(1 for row in analyzed if row.get("passed")),
        "rejected_images": sum(1 for row in analyzed if not row.get("passed")),
        "reject_reason_counts": dict(reason_counts.most_common()),
        "reject_reason_combo_counts_top50": dict(combo_counts.most_common(50)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def threshold_rule_labels() -> dict[str, str]:
    return {
        "reject_num_persons": "num_persons outside range",
        "reject_num_valid_hoi_pairs": "num_valid_hoi_pairs outside range",
        "reject_total_gt_boxes": "total_gt_boxes outside range",
        "reject_unique_object_categories": "num_unique_object_categories too low",
        "reject_unique_action_categories": "num_unique_action_categories too low",
        "reject_matching": "max_unique_person_object_matching too low",
        "reject_main_person_area": "primary pair visibility too low",
        "reject_secondary_person_area": "secondary pair visibility too low",
        "reject_primary_pair_context": "primary_pair_context_count too low",
        "reject_mean_context_top3": "mean_pair_context_count_top3 too low",
        "reject_human_overlap": "human-human overlap too high",
        "reject_action_homogeneity": "dominant_action_group_ratio too high",
        "reject_object_homogeneity": "dominant_object_category_ratio too high",
        "reject_severe_sports": "sports scene dominance too high",
    }


def format_command(tokens: list[str]) -> str:
    return " \\\n+  ".join(tokens)


def strict_figure1_thresholds_for_report(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = mode_thresholds(args)
    thresholds.update(
        {
            "min_actions": 3,
            "min_matching": 3,
            "min_context": 2,
        }
    )
    return thresholds


def pattern_counts(rows: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str]]:
    objects: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    object_actions: Counter[str] = Counter()
    for row in rows:
        for pair in row.get("top_pairs", [])[: min(5, len(row.get("top_pairs", [])))]:
            obj = str(pair.get("object_category_name", "unknown"))
            objects[obj] += 1
            pair_actions = [str(action) for action in pair.get("actions", [])]
            for action in pair_actions:
                actions[action] += 1
                object_actions[f"{obj}:{action}"] += 1
            for group in pair.get("action_groups", []):
                groups[str(group)] += 1
    return objects, actions, groups, object_actions


def sports_scene_summary(analyzed: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> tuple[str, list[str]]:
    severe_sports = sum(1 for row in analyzed if row.get("reject_flags", {}).get("reject_severe_sports"))
    retained_sports = 0
    retained_total = max(len(candidates), 1)
    examples: list[str] = []
    for row in candidates:
        has_sports = False
        for pair in row.get("top_pairs", [])[:5]:
            obj_name = str(pair.get("object_category_name", ""))
            pair_groups = {str(group) for group in pair.get("action_groups", [])}
            if obj_name in SPORTS_OBJECTS or "sports_ball" in pair_groups:
                has_sports = True
                if len(examples) < 5:
                    examples.append(f"{row.get('image_id')}:{obj_name}:{'/'.join(str(a) for a in pair.get('actions', []))}")
                break
        if has_sports:
            retained_sports += 1
    retained_ratio = retained_sports / retained_total
    severe_ratio = severe_sports / max(len(analyzed), 1)
    if retained_ratio >= 0.25 or severe_ratio >= 0.1:
        status = "sports scenes appear frequent enough to monitor"
    else:
        status = "sports scenes are not dominant under the current thresholds"
    summary = (
        f"{status}; severe_sports_rejects={severe_sports}/{len(analyzed)} "
        f"({severe_ratio:.1%}), retained_sports={retained_sports}/{len(candidates)} ({retained_ratio:.1%})"
    )
    return summary, examples


def recommended_commands(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> tuple[str, list[str]]:
    base = [
        "python",
        "tools/evidencehoi_image_search/filter_candidates.py",
        "--records",
        str(args.records),
    ]
    if args.image_root is not None:
        base.extend(["--image_root", str(args.image_root)])
    base.extend(["--output_dir", str(args.output_dir)])

    if len(candidates) < args.target_min_candidates:
        command = base + [
            "--mode",
            "relaxed",
            "--min_unique_action_categories",
            "2",
            "--min_matching",
            "2",
            "--min_primary_pair_context_count",
            "1",
            "--context_expand_scale",
            "2.0",
            "--auto_threshold_report",
        ]
        notes = [
            "Candidate count is below target_min_candidates; relax action diversity, matching, and primary context.",
            "Try context expansion scale in the 1.5 to 2.0 range; the command below uses 2.0 for recall.",
        ]
        return format_command(command), notes

    if len(candidates) > args.target_max_candidates:
        keep_percent = 100.0 * args.target_max_candidates / max(len(candidates), 1)
        command = base + [
            "--mode",
            "figure1",
            "--min_unique_action_categories",
            "3",
            "--min_matching",
            "3",
            "--min_primary_pair_context_count",
            "3",
            "--max_dominant_object_category_ratio",
            "0.55",
            "--auto_threshold_report",
        ]
        notes = [
            "Candidate count is above target_max_candidates; tighten context and object homogeneity.",
            f"After scoring, keep only the top {keep_percent:.1f}% by total_score to target about {args.target_max_candidates} images.",
            "Also require num_unique_action_groups >= 3 during manual review or in a follow-up filter if candidate count remains too high.",
        ]
        return format_command(command), notes

    command = base + ["--mode", "figure1", "--auto_threshold_report"]
    notes = ["Candidate count is within the requested range; keep current strict thresholds and inspect the top-ranked candidates."]
    return format_command(command), notes


def threshold_report_markdown(args: argparse.Namespace, analyzed: list[dict[str, Any]], candidates: list[dict[str, Any]], thresholds: dict[str, Any]) -> str:
    reason_counts: Counter[str] = Counter()
    exclusive_counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()
    for row in analyzed:
        reasons = row.get("reject_reasons", [])
        for reason in reasons:
            reason_counts[reason] += 1
        if len(reasons) == 1:
            exclusive_counts[reasons[0]] += 1
        combo_counts[",".join(reasons) if reasons else "passed"] += 1

    objects, actions, groups, object_actions = pattern_counts(candidates)
    sports_summary, sports_examples = sports_scene_summary(analyzed, candidates)
    command, command_notes = recommended_commands(args, candidates)
    if len(candidates) < args.target_min_candidates:
        status = "too_few_candidates"
    elif len(candidates) > args.target_max_candidates:
        status = "too_many_candidates"
    else:
        status = "within_target_range"

    lines = [
        "# EvidenceHOI Threshold Diagnostics",
        "",
        "## Run Summary",
        "",
        f"- records: `{args.records}`",
        f"- image_root: `{args.image_root}`",
        f"- output_dir: `{args.output_dir}`",
        f"- total_images: `{len(analyzed)}`",
        f"- strict_figure1_retained: `{len(candidates)}`",
        f"- target_min_candidates: `{args.target_min_candidates}`",
        f"- target_max_candidates: `{args.target_max_candidates}`",
        f"- status: `{status}`",
        "",
        "## Strict Figure 1 Thresholds Used",
        "",
        "- num_unique_action_categories >= `3`",
        "- max_unique_person_object_matching >= `3`",
        "- primary_pair_context_count >= `2`",
        f"- min_valid_pairs >= `{thresholds['min_pairs']}`",
        f"- max_valid_pairs <= `{thresholds['max_pairs']}`",
        f"- min_mean_pair_context_count_top3 >= `{thresholds['min_mean_context']}`",
        f"- context_expand_scale = `{args.context_expand_scale}`",
        "",
        "## Per-Rule Filtering Counts",
        "",
        "| Rule | Images Failing Rule | Only This Rule Failed |",
        "| --- | ---: | ---: |",
    ]
    labels = threshold_rule_labels()
    for reason, label in labels.items():
        lines.append(f"| `{label}` (`{reason}`) | {reason_counts.get(reason, 0)} | {exclusive_counts.get(reason, 0)} |")

    lines.extend(["", "## Most Common Reject Reasons", ""])
    if reason_counts:
        for reason, count in reason_counts.most_common(15):
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- No reject reasons; every image passed.")

    lines.extend(["", "## Most Common Reject Reason Combinations", ""])
    for combo, count in combo_counts.most_common(15):
        lines.append(f"- `{combo}`: `{count}`")

    lines.extend(["", "## Top Retained Object And Action Patterns", ""])
    if candidates:
        lines.append("### Object Categories")
        lines.append("")
        for item, count in objects.most_common(15):
            lines.append(f"- `{item}`: `{count}`")
        lines.append("")
        lines.append("### Verb Categories")
        lines.append("")
        for item, count in actions.most_common(15):
            lines.append(f"- `{item}`: `{count}`")
        lines.append("")
        lines.append("### Action Groups")
        lines.append("")
        for item, count in groups.most_common(15):
            lines.append(f"- `{item}`: `{count}`")
        lines.append("")
        lines.append("### Object:Verb Patterns")
        lines.append("")
        for item, count in object_actions.most_common(20):
            lines.append(f"- `{item}`: `{count}`")
    else:
        lines.append("- No retained candidates; object/action patterns are unavailable.")

    lines.extend(["", "## Sports Scene Check", "", f"- {sports_summary}"])
    if sports_examples:
        lines.append(f"- examples: `{', '.join(sports_examples)}`")

    lines.extend(["", "## Recommendation", ""])
    for note in command_notes:
        lines.append(f"- {note}")
    lines.extend(["", "## Recommended Next Command", "", "```bash", command, "```", ""])

    if len(candidates) < args.target_min_candidates:
        lines.extend(
            [
                "## Relaxed Settings To Try",
                "",
                "- num_unique_action_categories >= `2`",
                "- max_unique_person_object_matching >= `2`",
                "- primary_pair_context_count >= `1`",
                "- context_expand_scale from `1.5` to `2.0`",
                "",
            ]
        )
    elif len(candidates) > args.target_max_candidates:
        lines.extend(
            [
                "## Stricter Settings To Try",
                "",
                "- num_unique_action_groups >= `3`",
                "- max_unique_person_object_matching >= `3`",
                "- primary_pair_context_count >= `3`",
                "- dominant_object_category_ratio <= `0.55`",
                "- keep only top score percentile after soft scoring",
                "",
            ]
        )
    return "\n".join(lines)


def write_threshold_report(path: Path, args: argparse.Namespace, analyzed: list[dict[str, Any]], candidates: list[dict[str, Any]], thresholds: dict[str, Any]) -> None:
    path.write_text(threshold_report_markdown(args, analyzed, candidates, thresholds), encoding="utf-8")


def format_command(tokens: list[str]) -> str:
    return (" " + "\\" + "\n  ").join(tokens)


def strict_figure1_thresholds_for_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_actions": 3,
        "min_pairs": args.min_valid_pairs,
        "max_pairs": args.max_valid_pairs,
        "min_context": 2,
        "min_mean_context": args.min_mean_pair_context_count_top3,
        "min_matching": 3,
    }


def readable_summary(rows: list[dict[str, Any]], limit: int) -> str:
    lines = ["# Top Candidate Summary", ""]
    if not rows:
        lines.append("No candidates passed the current hard-filter thresholds.")
        lines.append("")
        return "\n".join(lines)
    for idx, row in enumerate(rows[:limit], start=1):
        counts = row["counts"]
        lines.append(f"## {idx}. {row['file_name']}")
        lines.append("")
        lines.append(f"- image_id: `{row['image_id']}`")
        lines.append(f"- image_path: `{row['image_path']}`")
        lines.append(f"- total_score: `{row.get('total_score')}` bucket=`{row.get('bucket_suggestion')}`")
        components = row.get("component_scores", {})
        lines.append(
            "- component scores: "
            f"PairDiversity={components.get('PairDiversity')}, "
            f"ActionDiversity={components.get('ActionDiversity')}, "
            f"ObjectDiversity={components.get('ObjectDiversity')}, "
            f"EvidenceContextScore={components.get('EvidenceContextScore')}, "
            f"VisualClarity={components.get('VisualClarity')}, "
            f"LayoutScore={components.get('LayoutScore')}, "
            f"HomogeneityPenalty={components.get('HomogeneityPenalty')}, "
            f"CrowdingPenalty={components.get('CrowdingPenalty')}"
        )
        lines.append(
            "- counts: "
            f"persons={counts['num_persons']}, boxes={counts['num_gt_boxes']}, "
            f"pairs={counts['num_valid_hoi_pairs']}, objects={counts['num_unique_object_categories']}, "
            f"actions={counts['num_unique_action_categories']}, groups={counts['num_unique_action_groups']}"
        )
        lines.append(
            "- ratios: "
            f"dominant_action_group={counts['dominant_action_group_ratio']}, "
            f"dominant_object={counts['dominant_object_category_ratio']}, "
            f"max_human_iou={counts['max_human_human_iou']}, "
            f"context_jaccard={components.get('context_jaccard_mean')}"
        )
        if row.get("warnings"):
            lines.append(f"- warnings: {row['warnings']}")
        for pair_idx, pair in enumerate(row.get("top_pairs", [])[:3], start=1):
            context_names = [item["category_name"] for item in pair.get("context_set", [])]
            lines.append(
                f"- q{pair_idx}: h{pair['subject_id']} -> o{pair['object_id']} "
                f"({pair['object_category_name']}), actions={pair['actions']}, "
                f"groups={pair['action_groups']}, context_count={pair['context_count']}, contexts={context_names}"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-filter normalized HOI ImageRecord JSONL candidates.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--mode", choices=("figure1", "figure2", "qualitative", "relaxed"), default="figure1")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_top_pairs", type=int, default=8)
    parser.add_argument("--summary_limit", type=int, default=20)

    parser.add_argument("--min_persons", type=int, default=2)
    parser.add_argument("--max_persons", type=int, default=5)
    parser.add_argument("--min_valid_pairs", type=int, default=3)
    parser.add_argument("--max_valid_pairs", type=int, default=8)
    parser.add_argument("--min_gt_boxes", type=int, default=6)
    parser.add_argument("--max_gt_boxes", type=int, default=18)
    parser.add_argument("--min_unique_object_categories", type=int, default=3)
    parser.add_argument("--min_unique_action_categories", type=int, default=None)
    parser.add_argument("--min_matching", type=int, default=2)

    parser.add_argument("--main_person_min_area", type=float, default=0.02)
    parser.add_argument("--secondary_person_min_area", type=float, default=0.01)
    parser.add_argument("--main_object_min_area", type=float, default=0.004)
    parser.add_argument("--context_object_min_area", type=float, default=0.002)
    parser.add_argument("--small_object_min_area", type=float, default=0.0015)
    parser.add_argument("--min_primary_pair_context_count", type=int, default=2)
    parser.add_argument("--min_mean_pair_context_count_top3", type=float, default=1.0)
    parser.add_argument("--max_human_human_iou", type=float, default=0.5)
    parser.add_argument("--max_dominant_action_group_ratio", type=float, default=0.75)
    parser.add_argument("--max_dominant_object_category_ratio", type=float, default=0.65)
    parser.add_argument("--context_expand_scale", type=float, default=1.75)
    parser.add_argument("--sports_dominate_ratio", type=float, default=0.6)
    parser.add_argument("--sports_object_dominate_ratio", type=float, default=0.6)
    parser.add_argument("--target_min_candidates", type=int, default=100)
    parser.add_argument("--target_max_candidates", type=int, default=1000)
    parser.add_argument(
        "--auto_threshold_report",
        action="store_true",
        help="Write threshold_report.md with tuning advice without overwriting candidate result files.",
    )

    args = parser.parse_args()
    supplied = set()
    for idx, token in enumerate(sys.argv):
        if token.startswith("--"):
            supplied.add(token)
    args.min_valid_pairs_user_set = "--min_valid_pairs" in supplied
    args.max_valid_pairs_user_set = "--max_valid_pairs" in supplied
    args.min_primary_context_user_set = "--min_primary_pair_context_count" in supplied
    args.min_mean_context_user_set = "--min_mean_pair_context_count_top3" in supplied
    args.min_matching_user_set = "--min_matching" in supplied
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = strict_figure1_thresholds_for_report(args) if args.auto_threshold_report else mode_thresholds(args)
    records = load_records(args.records)
    analyzed = [analyze_record(record, args, thresholds) for record in records]
    candidates = [row for row in analyzed if row["passed"]]
    candidates.sort(key=lambda row: float(row.get("total_score", 0.0)), reverse=True)

    if args.auto_threshold_report:
        report_path = args.output_dir / "threshold_report.md"
        write_threshold_report(report_path, args, analyzed, candidates, thresholds)
        print(f"Loaded records: {len(records)}")
        print(f"Strict Figure 1 retained: {len(candidates)}")
        print(f"Wrote {report_path}")
        print("Auto threshold report mode did not write or overwrite candidate result files.")
        return 0

    jsonl_path = args.output_dir / "candidates_hard_filtered.jsonl"
    csv_path = args.output_dir / "candidates_hard_filtered.csv"
    rejected_path = args.output_dir / "rejected_summary.json"
    summary_path = args.output_dir / "top20_readable_summary.md"

    write_jsonl(jsonl_path, candidates)
    write_csv(csv_path, candidates)
    write_rejected_summary(rejected_path, analyzed)
    summary_text = readable_summary(candidates, args.summary_limit)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"Loaded records: {len(records)}")
    print(f"Passed hard filter: {len(candidates)}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {rejected_path}")
    print(f"Wrote {summary_path}")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
