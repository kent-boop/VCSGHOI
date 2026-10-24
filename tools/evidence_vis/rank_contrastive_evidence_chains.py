#!/usr/bin/env python3
"""Rank pair-specific traces for semantically contrastive evidence-chain figures."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from render_evidence_shortlist import load_name_map, load_trace
from render_pair_specific_shortlist import box_iou


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def split_values(value: Any) -> List[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def jaccard_distance(first: Iterable[Any], second: Iterable[Any]) -> float:
    first_set = set(first)
    second_set = set(second)
    union = first_set | second_set
    if not union:
        return 0.0
    return 1.0 - len(first_set & second_set) / len(union)


def selected_context_categories(
    record: Mapping[str, Any],
    pair_idx: int,
    node_names: Mapping[int, str],
) -> Dict[str, Any]:
    graph = record["graph"]
    nodes = record["nodes"]
    mask = graph["selected_context_mask"][pair_idx].bool()
    indices = graph["selected_context_indices"][pair_idx][mask].tolist()
    labels = [int(nodes["labels"][node_idx].item()) for node_idx in indices]
    names = [node_names.get(label, str(label)) for label in labels]
    return {
        "indices": [int(index) for index in indices],
        "labels": labels,
        "label_set": set(labels),
        "names": names,
        "name_set": set(names),
    }


def correctness_tier(first_correct: bool, second_correct: bool) -> str:
    correct_count = int(first_correct) + int(second_correct)
    if correct_count == 2:
        return "A_both_correct"
    if correct_count == 1:
        return "B_one_correct"
    return "C_diagnostic"


def endpoint_overlap_metrics(
    record: Mapping[str, Any], row: Mapping[str, Any]
) -> Dict[str, float]:
    boxes = record["nodes"]["boxes_xyxy_norm"].float()
    pairs = record["pairs"]["indices"].long()
    first_idx = int(row["pair_idx_1"])
    second_idx = int(row["pair_idx_2"])
    human1, object1 = [int(value) for value in pairs[first_idx].tolist()]
    human2, object2 = [int(value) for value in pairs[second_idx].tolist()]
    human_iou = box_iou(boxes[human1], boxes[human2])
    object_iou = box_iou(boxes[object1], boxes[object2])
    control_type = str(row.get("control_type", "uncontrolled"))
    if control_type == "same_object_node":
        controlled_iou = human_iou
    elif control_type == "same_human_node":
        controlled_iou = object_iou
    else:
        controlled_iou = max(human_iou, object_iou)
    return {
        "human_endpoint_iou": human_iou,
        "object_endpoint_iou": object_iou,
        "controlled_endpoint_iou": controlled_iou,
    }


def contrast_score(row: Mapping[str, Any]) -> float:
    correct_fraction = float(row["correct_pair_count"]) / 2.0
    control_bonus = {
        "same_human_node": 1.0,
        "same_object_node": 0.90,
        "same_object_class": 0.65,
        "uncontrolled": 0.30,
    }.get(str(row["control_type"]), 0.30)
    different_object_bonus = 1.0 if row["object_name_1"] != row["object_name_2"] else 0.0
    return (
        2.30 * float(row["gt_action_distance"])
        + 1.90 * float(row["context_category_distance"])
        + 0.45 * float(row["context_set_distance"])
        + 0.85 * correct_fraction
        + 0.35 * float(row["predicted_action_distance"])
        + 0.70 * float(row["visual_quality"])
        + 0.30 * float(row["path_quality"])
        + 0.40 * control_bonus
        + 0.30 * (1.0 - float(row["controlled_endpoint_iou"]))
        + 0.20 * different_object_bonus
    )


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-csv", required=True)
    parser.add_argument("--node-names-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-gt-action-distance", type=float, default=1.0)
    parser.add_argument("--min-context-category-distance", type=float, default=0.25)
    parser.add_argument("--max-controlled-endpoint-iou", type=float, default=0.40)
    parser.add_argument("--max-per-image", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--allow-no-interaction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_names = load_name_map(args.node_names_json)
    source_rows = list(csv.DictReader(Path(args.ranking_csv).open("r", encoding="utf-8")))

    candidates: List[Dict[str, Any]] = []
    skip_counts: Dict[str, int] = defaultdict(int)
    seen: Set[tuple] = set()
    for source_row in source_rows:
        action_names_1 = split_values(source_row.get("pair_1_gt_action_names"))
        action_names_2 = split_values(source_row.get("pair_2_gt_action_names"))
        if not action_names_1 or not action_names_2:
            skip_counts["missing_gt_action"] += 1
            continue
        if not args.allow_no_interaction and (
            "no_interaction" in action_names_1 or "no_interaction" in action_names_2
        ):
            skip_counts["no_interaction"] += 1
            continue
        gt_action_distance = jaccard_distance(action_names_1, action_names_2)
        if gt_action_distance + 1e-9 < args.min_gt_action_distance:
            skip_counts["gt_action_too_similar"] += 1
            continue

        record = load_trace(Path(source_row["record_path"]))
        first_idx = int(source_row["pair_idx_1"])
        second_idx = int(source_row["pair_idx_2"])
        canonical_key = (
            str(source_row["record_path"]),
            min(first_idx, second_idx),
            max(first_idx, second_idx),
        )
        if canonical_key in seen:
            skip_counts["duplicate_pair"] += 1
            continue
        seen.add(canonical_key)

        first_context = selected_context_categories(record, first_idx, node_names)
        second_context = selected_context_categories(record, second_idx, node_names)
        category_distance = jaccard_distance(
            first_context["label_set"], second_context["label_set"]
        )
        if category_distance + 1e-9 < args.min_context_category_distance:
            skip_counts["context_categories_too_similar"] += 1
            continue
        endpoint_metrics = endpoint_overlap_metrics(record, source_row)
        endpoint_iou = endpoint_metrics["controlled_endpoint_iou"]
        if endpoint_iou > args.max_controlled_endpoint_iou:
            skip_counts["controlled_endpoint_overlap"] += 1
            continue

        first_correct = bool_value(source_row.get("pair_1_final_correct"))
        second_correct = bool_value(source_row.get("pair_2_final_correct"))
        predicted_action_distance = float(
            source_row.get("pair_1_top_action_name")
            != source_row.get("pair_2_top_action_name")
        )
        row: Dict[str, Any] = dict(source_row)
        row.update({
            "contrast_tier": correctness_tier(first_correct, second_correct),
            "correct_pair_count": int(first_correct) + int(second_correct),
            "gt_action_distance": f"{gt_action_distance:.6f}",
            "predicted_action_distance": f"{predicted_action_distance:.6f}",
            "context_category_distance": f"{category_distance:.6f}",
            "context_category_ids_1": ";".join(map(str, first_context["labels"])),
            "context_category_names_1": ";".join(first_context["names"]),
            "context_category_ids_2": ";".join(map(str, second_context["labels"])),
            "context_category_names_2": ";".join(second_context["names"]),
            "human_endpoint_iou": f"{endpoint_metrics['human_endpoint_iou']:.6f}",
            "object_endpoint_iou": f"{endpoint_metrics['object_endpoint_iou']:.6f}",
            "controlled_endpoint_iou": f"{endpoint_iou:.6f}",
        })
        row["contrast_score"] = f"{contrast_score(row):.6f}"
        candidates.append(row)

    tier_order = {"A_both_correct": 0, "B_one_correct": 1, "C_diagnostic": 2}
    candidates.sort(
        key=lambda row: (
            tier_order[row["contrast_tier"]],
            -float(row["contrast_score"]),
            int(row["image_id"]),
        )
    )
    if args.max_per_image > 0:
        image_counts: Dict[str, int] = defaultdict(int)
        limited: List[Dict[str, Any]] = []
        for row in candidates:
            image_key = str(row["image_id"])
            if image_counts[image_key] >= args.max_per_image:
                continue
            image_counts[image_key] += 1
            limited.append(row)
        candidates = limited
    if args.top_n > 0:
        candidates = candidates[: args.top_n]

    write_rows(output_dir / "contrastive_candidates.csv", candidates)
    for tier in tier_order:
        write_rows(
            output_dir / f"{tier}.csv",
            [row for row in candidates if row["contrast_tier"] == tier],
        )
    summary = [
        f"source_rows={len(source_rows)}",
        f"candidate_rows={len(candidates)}",
        *(f"{tier}={sum(row['contrast_tier'] == tier for row in candidates)}" for tier in tier_order),
        *(f"skipped_{key}={value}" for key, value in sorted(skip_counts.items())),
    ]
    (output_dir / "contrastive_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))


if __name__ == "__main__":
    main()
