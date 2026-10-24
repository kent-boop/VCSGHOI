#!/usr/bin/env python3
"""Rank HICO images for EvidenceHOI Figure 1 suitability.

The ranking is heuristic and intentionally transparent: each image receives
separate scores for multiplicity, diversity, pair-specific evidence potential,
ambiguity/contrast, visual clarity, and penalties. This script does not select a
final image and does not generate visualizations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CSV_COLUMNS = [
    "rank",
    "figure1_score",
    "filename",
    "image_id",
    "image_path",
    "human_count",
    "object_count",
    "hoi_pair_count",
    "verb_count",
    "object_category_count",
    "multiplicity_score",
    "diversity_score",
    "evidence_potential_score",
    "ambiguity_score",
    "visual_clarity_score",
    "penalty_score",
    "final_reason",
    "short_pair_summary",
]

CONTEXT_OBJECT_NAMES = {
    "dining table",
    "chair",
    "cup",
    "bottle",
    "sports ball",
    "bicycle",
    "motorcycle",
    "laptop",
    "keyboard",
    "mouse",
    "cell phone",
    "remote",
    "backpack",
    "handbag",
    "suitcase",
    "bench",
    "frisbee",
    "skis",
    "snowboard",
    "skateboard",
    "surfboard",
    "tennis racket",
    "baseball bat",
    "baseball glove",
    "kite",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "book",
}

FOOD_NAMES = {
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
}

SPORTS_EQUIPMENT_NAMES = {
    "sports ball",
    "frisbee",
    "skis",
    "snowboard",
    "skateboard",
    "surfboard",
    "tennis racket",
    "baseball bat",
    "baseball glove",
    "kite",
}

BAG_NAMES = {"backpack", "handbag", "suitcase"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank EvidenceHOI Figure 1 candidate images.")
    parser.add_argument(
        "--stats-json",
        default="outputs/figure1_candidate_search/image_stats.json",
        help="Input image_stats.json from analyze_hoi_dataset_for_figure1.py.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Original image root. If omitted, use metadata.image_root from stats JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figure1_candidate_search",
        help="Output directory for ranked_candidates.csv/json.",
    )
    parser.add_argument("--top-csv", type=int, default=100, help="Number of rows to write to CSV.")
    parser.add_argument("--print-top", type=int, default=20, help="Number of top candidates to print.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional debug limit.")
    parser.add_argument(
        "--skip-brightness",
        action="store_true",
        help="Skip image brightness estimation. Geometry-only clarity scoring still runs.",
    )
    return parser.parse_args(argv)


def load_stats(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("images"), list):
        return payload.get("metadata", {}), payload["images"]
    if isinstance(payload, list):
        return {}, payload
    raise ValueError(f"Unsupported stats JSON schema: {path}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bbox_area(bbox: Optional[Sequence[float]]) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_center(bbox: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    if not bbox or len(bbox) != 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def bbox_diag(bbox: Optional[Sequence[float]]) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    return math.hypot(max(0.0, float(bbox[2]) - float(bbox[0])), max(0.0, float(bbox[3]) - float(bbox[1])))


def bbox_iou(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    ac = bbox_center(a)
    bc = bbox_center(b)
    if ac is None or bc is None:
        return float("inf")
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])


def is_near(a: Optional[Sequence[float]], b: Optional[Sequence[float]], factor: float = 1.5) -> bool:
    if not a or not b:
        return False
    return center_distance(a, b) <= factor * max(bbox_diag(a), bbox_diag(b), 1.0)


def category_name(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    return None


def pair_object_name(pair: Dict[str, Any]) -> Optional[str]:
    return category_name(pair.get("object_category_name"))


def pair_verb_name(pair: Dict[str, Any]) -> Optional[str]:
    return category_name(pair.get("verb_name"))


def object_name(obj: Dict[str, Any]) -> Optional[str]:
    return category_name(obj.get("category_name"))


def image_area(stat: Dict[str, Any]) -> Optional[float]:
    width = stat.get("width")
    height = stat.get("height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
        return float(width) * float(height)
    return None


def estimate_brightness(image_path: Path, skip: bool) -> Optional[float]:
    if skip or not image_path.exists():
        return None
    try:
        from PIL import Image, ImageStat
    except Exception:
        return None
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            gray.thumbnail((48, 48))
            return float(ImageStat.Stat(gray).mean[0])
    except Exception:
        return None


def score_multiplicity(stat: Dict[str, Any], reasons: List[str]) -> float:
    human_count = int(stat.get("human_count") or 0)
    object_count = int(stat.get("object_count") or 0)
    pair_count = int(stat.get("hoi_pair_count") or 0)
    score = 0.0

    if human_count >= 2:
        score += 6.0
        reasons.append(f"有 {human_count} 个人，可展示多个人相关候选对")
    if 2 <= human_count <= 4:
        score += 3.0
    elif human_count > 6:
        score -= 1.5

    if object_count >= 4:
        score += 6.0
        reasons.append(f"有 {object_count} 个非人目标，场景上下文较丰富")
    elif object_count >= 2:
        score += 3.0

    if pair_count >= 3:
        score += 7.0
        reasons.append(f"包含 {pair_count} 个 HOI pair，足以比较 pair-specific evidence")
    elif pair_count == 2:
        score += 3.0

    if 3 <= pair_count <= 8:
        score += 5.0
        reasons.append("HOI pair 数量处在可解释范围内")
    elif pair_count == 2:
        score += 2.0
    elif pair_count > 12:
        score -= min(4.0, (pair_count - 12) * 0.5)

    return clamp(score, 0.0, 25.0)


def score_diversity(stat: Dict[str, Any], reasons: List[str]) -> float:
    verb_count = int(stat.get("verb_count") or 0)
    object_category_count = int(stat.get("object_category_count") or 0)
    pairs = stat.get("hoi_pairs") or []
    score = 0.0

    score += min(8.0, max(0, verb_count - 1) * 3.0)
    score += min(8.0, max(0, object_category_count - 1) * 2.5)
    if verb_count >= 2:
        reasons.append(f"有 {verb_count} 类动作，能体现不同 evidence chain")
    if object_category_count >= 2:
        reasons.append(f"涉及 {object_category_count} 类物体，交互语义有差异")

    verb_object_pairs = {
        (pair.get("verb_id"), pair.get("object_category_id"))
        for pair in pairs
        if pair.get("verb_id") is not None and pair.get("object_category_id") is not None
    }
    if len(verb_object_pairs) >= 3:
        score += 4.0
        reasons.append("同图包含多种 verb-object 组合")

    return clamp(score, 0.0, 20.0)


def score_evidence_potential(stat: Dict[str, Any], reasons: List[str]) -> float:
    pairs = stat.get("hoi_pairs") or []
    object_boxes = stat.get("object_boxes") or []
    score = 0.0

    if stat.get("one_person_multiple_objects"):
        score += 5.0
        reasons.append("存在同一人关联多个物体，适合展示候选对证据差异")

    human_to_objects: Dict[Any, set[Any]] = defaultdict(set)
    human_to_categories: Dict[Any, set[Any]] = defaultdict(set)
    for pair in pairs:
        h = pair.get("human_index")
        o = pair.get("object_index")
        c = pair.get("object_category_id")
        if h is not None and o is not None:
            human_to_objects[h].add(o)
        if h is not None and c is not None:
            human_to_categories[h].add(c)

    multi_people_with_objects = sum(1 for objects in human_to_objects.values() if objects)
    if multi_people_with_objects >= 2:
        score += 5.0
        reasons.append("多个人分别参与交互，可比较不同人的 pair evidence")

    if any(len(cats) >= 2 for cats in human_to_categories.values()):
        score += 3.0

    if stat.get("multiple_people_same_object_category"):
        score += 4.0
        reasons.append("多个人接近或关联同类物体，具有局部组内对比潜力")

    if stat.get("same_object_near_multiple_people"):
        score += 4.0
        reasons.append("同一物体附近有多个人，容易形成 pair-specific 判断场景")

    context_names = sorted({object_name(obj) for obj in object_boxes if object_name(obj) in CONTEXT_OBJECT_NAMES})
    context_names = [x for x in context_names if x]
    if context_names:
        score += min(5.0, 1.2 * len(context_names))
        reasons.append("包含可作为上下文证据的物体：" + "、".join(context_names[:5]))

    if any(name in FOOD_NAMES for name in context_names):
        score += 1.0
    if any(name in SPORTS_EQUIPMENT_NAMES for name in context_names):
        score += 1.0
    if any(name in BAG_NAMES for name in context_names):
        score += 0.8

    return clamp(score, 0.0, 25.0)


def score_ambiguity(stat: Dict[str, Any], reasons: List[str]) -> float:
    pairs = stat.get("hoi_pairs") or []
    object_boxes = stat.get("object_boxes") or []
    score = 0.0

    verb_to_categories: Dict[Any, set[Any]] = defaultdict(set)
    category_to_verbs: Dict[Any, set[Any]] = defaultdict(set)
    paired_object_indices = set()
    for pair in pairs:
        verb = pair.get("verb_id")
        cat = pair.get("object_category_id")
        obj_idx = pair.get("object_index")
        if verb is not None and cat is not None:
            verb_to_categories[verb].add(cat)
            category_to_verbs[cat].add(verb)
        if obj_idx is not None:
            paired_object_indices.add(obj_idx)

    same_action_diff_object = [v for v, cats in verb_to_categories.items() if len(cats) >= 2]
    same_object_diff_action = [c for c, verbs in category_to_verbs.items() if len(verbs) >= 2]
    if same_action_diff_object:
        score += 5.0
        reasons.append("存在相同动作对应不同物体，适合做 contrast")
    if same_object_diff_action:
        score += 5.0
        reasons.append("存在同类物体对应不同动作，能体现证据链差异")

    humans = stat.get("human_boxes") or []
    distractor_count = 0
    semantic_far_context = 0
    for obj in object_boxes:
        obj_idx = obj.get("index")
        obj_bbox = obj.get("bbox")
        obj_name_value = object_name(obj)
        if obj_idx in paired_object_indices:
            continue
        near_any_human = any(is_near(obj_bbox, human.get("bbox"), factor=1.2) for human in humans)
        if near_any_human:
            distractor_count += 1
        if obj_name_value in CONTEXT_OBJECT_NAMES and not near_any_human:
            semantic_far_context += 1

    if distractor_count:
        score += min(4.0, distractor_count * 1.5)
        reasons.append(f"有 {distractor_count} 个靠近人物但未成对的潜在干扰物")
    if semantic_far_context:
        score += min(3.0, semantic_far_context * 1.0)
        reasons.append("存在稍远但语义相关的 context object")

    if stat.get("shared_action"):
        score += 2.0
    if stat.get("shared_object_category"):
        score += 2.0

    return clamp(score, 0.0, 20.0)


def score_visual_clarity_and_penalty(
    stat: Dict[str, Any],
    brightness: Optional[float],
    reasons: List[str],
    penalty_reasons: List[str],
) -> Tuple[float, float]:
    width = stat.get("width")
    height = stat.get("height")
    area = image_area(stat)
    pair_count = int(stat.get("hoi_pair_count") or 0)
    human_boxes = stat.get("human_boxes") or []
    object_boxes = stat.get("object_boxes") or []
    paired_boxes: List[Sequence[float]] = []
    for pair in stat.get("hoi_pairs") or []:
        if pair.get("human_bbox"):
            paired_boxes.append(pair["human_bbox"])
        if pair.get("object_bbox"):
            paired_boxes.append(pair["object_bbox"])

    clarity = 12.0
    penalty = 0.0

    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        min_side = min(float(width), float(height))
        if min_side >= 420:
            clarity += 3.0
        elif min_side < 300:
            penalty += 5.0
            penalty_reasons.append("图像分辨率偏低")
    else:
        penalty += 2.0
        penalty_reasons.append("无法读取图像尺寸")

    if area:
        small_boxes = 0
        edge_boxes = 0
        usable_boxes = [box for box in paired_boxes if box]
        for box in usable_boxes:
            ratio = bbox_area(box) / area
            if ratio < 0.003:
                small_boxes += 1
            margin_x = min(float(box[0]), max(0.0, float(width) - float(box[2]))) if width else 0.0
            margin_y = min(float(box[1]), max(0.0, float(height) - float(box[3]))) if height else 0.0
            if width and height and (margin_x < 0.015 * float(width) or margin_y < 0.015 * float(height)):
                edge_boxes += 1
        if small_boxes:
            penalty += min(7.0, small_boxes * 1.2)
            penalty_reasons.append(f"有 {small_boxes} 个关键 bbox 偏小")
        if edge_boxes:
            penalty += min(5.0, edge_boxes * 0.8)
            penalty_reasons.append(f"有 {edge_boxes} 个关键 bbox 靠近图像边缘")

        medium_or_large = sum(1 for box in paired_boxes if box and bbox_area(box) / area >= 0.01)
        if medium_or_large >= min(4, len(paired_boxes)):
            clarity += 2.0

    all_boxes = [box.get("bbox") for box in [*human_boxes, *object_boxes] if box.get("bbox")]
    high_overlap = 0
    for i, box_a in enumerate(all_boxes):
        for box_b in all_boxes[i + 1 :]:
            if bbox_iou(box_a, box_b) > 0.65:
                high_overlap += 1
    if high_overlap:
        penalty += min(7.0, high_overlap * 0.7)
        penalty_reasons.append(f"bbox 高重叠较多：{high_overlap} 对")

    if pair_count > 10:
        penalty += min(8.0, (pair_count - 10) * 0.9)
        penalty_reasons.append("HOI pair 过多，概念图可能不可读")
    elif 3 <= pair_count <= 8:
        clarity += 2.0

    if brightness is not None:
        if brightness < 45:
            penalty += 6.0
            penalty_reasons.append("图像整体偏暗")
        elif brightness < 65:
            penalty += 2.0
            penalty_reasons.append("图像亮度略低")
        elif 75 <= brightness <= 205:
            clarity += 1.0

    if penalty <= 2.0:
        reasons.append("基础视觉清晰度较好，适合画概念图")

    return clamp(clarity, 0.0, 20.0), clamp(penalty, 0.0, 30.0)


def resolve_image_path(stat: Dict[str, Any], image_root: Path) -> Path:
    filename = stat.get("filename")
    if isinstance(filename, str):
        candidate = image_root / filename
        if candidate.exists():
            return candidate
    path = stat.get("image_path")
    if isinstance(path, str):
        return Path(path)
    return image_root / str(filename or "")


def score_image(stat: Dict[str, Any], image_root: Path, skip_brightness: bool) -> Dict[str, Any]:
    reasons: List[str] = []
    penalty_reasons: List[str] = []
    image_path = resolve_image_path(stat, image_root)

    multiplicity = score_multiplicity(stat, reasons)
    diversity = score_diversity(stat, reasons)
    evidence = score_evidence_potential(stat, reasons)
    ambiguity = score_ambiguity(stat, reasons)
    brightness = estimate_brightness(image_path, skip_brightness)
    clarity, penalty = score_visual_clarity_and_penalty(stat, brightness, reasons, penalty_reasons)

    figure1_score = multiplicity + diversity + evidence + ambiguity + clarity - penalty
    figure1_score = round(figure1_score, 4)

    if penalty_reasons:
        final_reason = "；".join(reasons[:5]) + "。扣分：" + "、".join(penalty_reasons[:4])
    else:
        final_reason = "；".join(reasons[:6])
    if not final_reason:
        final_reason = "候选对数量或证据差异不足，作为 Figure 1 示例的解释空间有限"

    out = dict(stat)
    out.update(
        {
            "image_path": str(image_path),
            "figure1_score": figure1_score,
            "multiplicity_score": round(multiplicity, 4),
            "diversity_score": round(diversity, 4),
            "evidence_potential_score": round(evidence, 4),
            "ambiguity_score": round(ambiguity, 4),
            "visual_clarity_score": round(clarity, 4),
            "penalty_score": round(penalty, 4),
            "brightness": brightness,
            "reason_items": reasons,
            "penalty_reason_items": penalty_reasons,
            "final_reason": final_reason,
        }
    )
    return out


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_csv(path: Path, ranked: Sequence[Dict[str, Any]], top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rank, item in enumerate(ranked[:top_k], start=1):
            row = dict(item)
            row["rank"] = rank
            writer.writerow({col: csv_value(row.get(col)) for col in CSV_COLUMNS})


def write_json(path: Path, metadata: Dict[str, Any], ranked: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "ranked_candidates": [dict(item, rank=i) for i, item in enumerate(ranked, start=1)],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_top(ranked: Sequence[Dict[str, Any]], top_k: int) -> None:
    print(f"Top {min(top_k, len(ranked))} Figure 1 candidates")
    for rank, item in enumerate(ranked[:top_k], start=1):
        print(
            f"{rank:02d}. {item.get('filename')} | score={item.get('figure1_score'):.2f} | "
            f"H={item.get('human_count')} O={item.get('object_count')} P={item.get('hoi_pair_count')} "
            f"V={item.get('verb_count')} C={item.get('object_category_count')}"
        )
        print(f"    原因：{item.get('final_reason')}")
        pair_summary = item.get("short_pair_summary")
        if pair_summary:
            print(f"    pairs：{pair_summary}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stats_path = Path(args.stats_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata, images = load_stats(stats_path)
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]

    image_root_text = args.image_root or metadata.get("image_root")
    if not image_root_text:
        raise ValueError("image root missing: pass --image-root or include metadata.image_root in stats JSON")
    image_root = Path(image_root_text).resolve()

    ranked = [score_image(stat, image_root, args.skip_brightness) for stat in images]
    ranked.sort(
        key=lambda x: (
            x["figure1_score"],
            x.get("hoi_pair_count") or 0,
            x.get("verb_count") or 0,
            x.get("object_category_count") or 0,
        ),
        reverse=True,
    )

    out_metadata = {
        "stats_json": str(stats_path),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "scored_image_count": len(ranked),
        "csv_top_k": args.top_csv,
        "skip_brightness": bool(args.skip_brightness),
        "score_definition": "multiplicity + diversity + evidence_potential + ambiguity + visual_clarity - penalty",
        "input_metadata": metadata,
    }
    csv_path = output_dir / "ranked_candidates.csv"
    json_path = output_dir / "ranked_candidates.json"
    write_csv(csv_path, ranked, args.top_csv)
    write_json(json_path, out_metadata, ranked)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print_top(ranked, args.print_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
