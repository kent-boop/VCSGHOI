#!/usr/bin/env python3
"""Stream-rank a correct GT HOI anchor against a semantically different partner."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from rank_evidence_examples import (
    clamp01,
    selected_contexts,
    set_jaccard_distance,
    visual_quality,
    weighted_jaccard_distance,
)
from render_evidence_shortlist import load_name_map, load_trace
from render_pair_specific_shortlist import box_iou


def name_of(mapping: Mapping[int, str], index: int) -> str:
    return mapping.get(int(index), str(int(index)))


def multiset_jaccard_distance(first: Sequence[str], second: Sequence[str]) -> float:
    first_counts = Counter(first)
    second_counts = Counter(second)
    keys = set(first_counts) | set(second_counts)
    if not keys:
        return 0.0
    intersection = sum(min(first_counts[key], second_counts[key]) for key in keys)
    union = sum(max(first_counts[key], second_counts[key]) for key in keys)
    return 1.0 - intersection / max(union, 1)


def context_info(
    record: Mapping[str, Any],
    pair_idx: int,
    node_names: Mapping[int, str],
) -> Dict[str, Any]:
    ids, weights, two_hop_count = selected_contexts(record["graph"], pair_idx)
    nodes = record["nodes"]
    labels = [int(nodes["labels"][node_idx].item()) for node_idx in ids]
    names = [name_of(node_names, label) for label in labels]
    return {
        "ids": ids,
        "weights": weights,
        "two_hop_count": two_hop_count,
        "labels": labels,
        "names": names,
    }


def action_info(
    gt_targets: torch.Tensor,
    action_probs: torch.Tensor,
    pair_idx: int,
    action_names: Mapping[int, str],
) -> Dict[str, Any]:
    gt_ids = torch.nonzero(gt_targets[pair_idx], as_tuple=False).flatten().tolist()
    gt_ids = [int(value) for value in gt_ids]
    top_action_id = int(action_probs[pair_idx].argmax().item())
    top_probability = float(action_probs[pair_idx, top_action_id].item())
    positive = bool(gt_ids)
    correct = positive and top_action_id in gt_ids
    return {
        "positive": positive,
        "correct": correct,
        "gt_ids": gt_ids,
        "gt_names": [name_of(action_names, value) for value in gt_ids],
        "top_action_id": top_action_id,
        "top_action_name": name_of(action_names, top_action_id),
        "top_probability": top_probability,
    }


def endpoint_metrics(
    record: Mapping[str, Any],
    first_idx: int,
    second_idx: int,
    object_label_1: int,
    object_label_2: int,
) -> Dict[str, Any]:
    pairs = record["pairs"]["indices"].long()
    boxes = record["nodes"]["boxes_xyxy_norm"].float()
    human1, object1 = [int(value) for value in pairs[first_idx].tolist()]
    human2, object2 = [int(value) for value in pairs[second_idx].tolist()]
    human_iou = box_iou(boxes[human1], boxes[human2])
    object_iou = box_iou(boxes[object1], boxes[object2])
    if object1 == object2:
        control_type = "same_object_node"
        controlled_iou = human_iou
    elif human1 == human2:
        control_type = "same_human_node"
        controlled_iou = object_iou
    elif object_label_1 == object_label_2:
        control_type = "same_object_class"
        controlled_iou = max(human_iou, object_iou)
    else:
        control_type = "uncontrolled"
        controlled_iou = max(human_iou, object_iou)
    return {
        "human1": human1,
        "object1": object1,
        "human2": human2,
        "object2": object2,
        "human_iou": human_iou,
        "object_iou": object_iou,
        "controlled_iou": controlled_iou,
        "control_type": control_type,
    }


def tier_name(partner: Mapping[str, Any]) -> str:
    if partner["positive"] and partner["correct"]:
        return "A_both_positive_correct"
    if partner["positive"]:
        return "B_positive_partner_wrong"
    return "C_false_positive_partner"


def score_candidate(row: Mapping[str, Any]) -> float:
    control_bonus = {
        "same_human_node": 1.0,
        "same_object_node": 0.8,
        "same_object_class": 0.45,
        "uncontrolled": 0.25,
    }.get(str(row["control_type"]), 0.25)
    partner_gt_bonus = 1.0 if int(row["partner_gt_positive"]) else 0.0
    partner_correct_bonus = 1.0 if str(row["pair_2_final_correct"]).lower() == "true" else 0.0
    informative_bonus = min(int(row["informative_context_difference_count"]) / 2.0, 1.0)
    return (
        1.55 * float(row["context_multiset_distance"])
        + 1.10 * float(row["context_category_distance"])
        + 0.65 * float(row["context_set_distance"])
        + 0.75 * informative_bonus
        + 0.60 * control_bonus
        + 0.55 * float(row["visual_quality"])
        + 0.30 * float(row["path_quality"])
        + 0.35 * float(row["pair_2_top_action_probability"])
        + 0.20 * float(row["pair_1_top_action_probability"])
        + 0.25 * (1.0 - float(row["controlled_endpoint_iou"]))
        + 0.20 * partner_gt_bonus
        + 0.35 * partner_correct_bonus
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
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-names-json", required=True)
    parser.add_argument("--object-names-json", required=True)
    parser.add_argument("--node-names-json", required=True)
    parser.add_argument("--min-context-count", type=int, default=2)
    parser.add_argument("--min-context-category-distance", type=float, default=0.25)
    parser.add_argument("--min-context-multiset-distance", type=float, default=0.25)
    parser.add_argument("--min-partner-probability", type=float, default=0.15)
    parser.add_argument("--max-controlled-endpoint-iou", type=float, default=0.65)
    parser.add_argument("--max-per-image", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=120)
    parser.add_argument("--allow-anchor-wrong", action="store_true")
    parser.add_argument("--allow-no-interaction-prediction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    records_dir = trace_dir / "records" if (trace_dir / "records").is_dir() else trace_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_names = load_name_map(args.action_names_json)
    object_names = load_name_map(args.object_names_json)
    node_names = load_name_map(args.node_names_json)

    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = defaultdict(int)
    record_count = 0
    for trace_path in sorted(records_dir.glob("*.pt")):
        record_count += 1
        record = load_trace(trace_path)
        pairs = record.get("pairs") or {}
        graph = record.get("graph") or {}
        pair_indices = pairs.get("indices")
        object_labels = pairs.get("object_labels")
        gt_targets = pairs.get("gt_action_targets")
        action_probs = pairs.get("action_probs")
        pair_valid = graph.get("pair_valid_mask")
        if any(value is None for value in (pair_indices, object_labels, gt_targets, action_probs, pair_valid)):
            skipped["missing_trace_fields"] += 1
            continue
        pair_indices = pair_indices.long()
        object_labels = object_labels.long()
        gt_targets = gt_targets.bool()
        action_probs = action_probs.float()
        pair_valid = pair_valid.bool()

        pair_cache: Dict[int, Dict[str, Any]] = {}
        for pair_idx in range(int(pair_indices.shape[0])):
            if not bool(pair_valid[pair_idx]):
                continue
            context = context_info(record, pair_idx, node_names)
            if len(context["ids"]) < args.min_context_count:
                continue
            action = action_info(gt_targets, action_probs, pair_idx, action_names)
            quality, _ = visual_quality(record, pair_idx, context["ids"])
            pair_cache[pair_idx] = {"context": context, "action": action, "quality": quality}

        image_rows: List[Dict[str, Any]] = []
        seen_pairs = set()
        for anchor_idx, anchor_data in pair_cache.items():
            anchor = anchor_data["action"]
            if not anchor["positive"]:
                continue
            if not args.allow_anchor_wrong and not anchor["correct"]:
                continue
            for partner_idx, partner_data in pair_cache.items():
                if partner_idx == anchor_idx:
                    continue
                canonical = tuple(sorted((anchor_idx, partner_idx)))
                if canonical in seen_pairs and partner_data["action"]["positive"]:
                    continue
                partner = partner_data["action"]
                if partner["top_probability"] < args.min_partner_probability:
                    continue
                if partner["top_action_name"] == anchor["top_action_name"]:
                    continue
                if not args.allow_no_interaction_prediction and (
                    partner["top_action_name"] == "no_interaction"
                    or anchor["top_action_name"] == "no_interaction"
                ):
                    continue

                object_label_1 = int(object_labels[anchor_idx].item())
                object_label_2 = int(object_labels[partner_idx].item())
                object_name_1 = name_of(object_names, object_label_1)
                object_name_2 = name_of(object_names, object_label_2)
                if object_name_1 == object_name_2:
                    continue
                endpoints = endpoint_metrics(
                    record, anchor_idx, partner_idx, object_label_1, object_label_2
                )
                if endpoints["controlled_iou"] > args.max_controlled_endpoint_iou:
                    continue

                first_context = anchor_data["context"]
                second_context = partner_data["context"]
                category_distance = set_jaccard_distance(
                    first_context["labels"], second_context["labels"]
                )
                multiset_distance = multiset_jaccard_distance(
                    first_context["names"], second_context["names"]
                )
                if category_distance < args.min_context_category_distance:
                    continue
                if multiset_distance < args.min_context_multiset_distance:
                    continue
                first_weight_map = dict(zip(first_context["ids"], first_context["weights"]))
                second_weight_map = dict(zip(second_context["ids"], second_context["weights"]))
                node_set_distance = set_jaccard_distance(
                    first_context["ids"], second_context["ids"]
                )
                weighted_distance = weighted_jaccard_distance(
                    first_weight_map, second_weight_map
                )
                informative_names = (
                    set(first_context["names"]) ^ set(second_context["names"])
                ) - {"person", object_name_1, object_name_2}
                path_quality = min(
                    first_context["two_hop_count"] / max(len(first_context["ids"]), 1),
                    second_context["two_hop_count"] / max(len(second_context["ids"]), 1),
                )
                pair_specific_score = clamp01(
                    0.38 * node_set_distance
                    + 0.27 * weighted_distance
                    + 0.20 * category_distance
                    + 0.15 * multiset_distance
                )
                row: Dict[str, Any] = {
                    "record_path": str(trace_path),
                    "image_id": record.get("image", {}).get("image_id"),
                    "file_name": record.get("image", {}).get("file_name"),
                    "pair_idx_1": anchor_idx,
                    "pair_idx_2": partner_idx,
                    "human_node_idx_1": endpoints["human1"],
                    "object_node_idx_1": endpoints["object1"],
                    "human_node_idx_2": endpoints["human2"],
                    "object_node_idx_2": endpoints["object2"],
                    "object_label_1": object_label_1,
                    "object_name_1": object_name_1,
                    "object_label_2": object_label_2,
                    "object_name_2": object_name_2,
                    "control_type": endpoints["control_type"],
                    "context_ids_1": ";".join(map(str, first_context["ids"])),
                    "context_ids_2": ";".join(map(str, second_context["ids"])),
                    "context_set_distance": f"{node_set_distance:.6f}",
                    "context_weighted_distance": f"{weighted_distance:.6f}",
                    "context_category_distance": f"{category_distance:.6f}",
                    "context_multiset_distance": f"{multiset_distance:.6f}",
                    "context_category_names_1": ";".join(first_context["names"]),
                    "context_category_names_2": ";".join(second_context["names"]),
                    "informative_context_difference": ";".join(sorted(informative_names)),
                    "informative_context_difference_count": len(informative_names),
                    "two_hop_count_1": first_context["two_hop_count"],
                    "two_hop_count_2": second_context["two_hop_count"],
                    "visual_quality": f"{min(anchor_data['quality'], partner_data['quality']):.6f}",
                    "path_quality": f"{path_quality:.6f}",
                    "pair_specific_score": f"{pair_specific_score:.6f}",
                    "human_endpoint_iou": f"{endpoints['human_iou']:.6f}",
                    "object_endpoint_iou": f"{endpoints['object_iou']:.6f}",
                    "controlled_endpoint_iou": f"{endpoints['controlled_iou']:.6f}",
                    "pair_1_gt_matched": True,
                    "pair_1_gt_action_ids": ";".join(map(str, anchor["gt_ids"])),
                    "pair_1_gt_action_names": ";".join(anchor["gt_names"]),
                    "pair_1_top_action_id": anchor["top_action_id"],
                    "pair_1_top_action_name": anchor["top_action_name"],
                    "pair_1_top_action_probability": f"{anchor['top_probability']:.6f}",
                    "pair_1_final_correct": anchor["correct"],
                    "pair_2_gt_matched": partner["positive"],
                    "pair_2_gt_action_ids": ";".join(map(str, partner["gt_ids"])),
                    "pair_2_gt_action_names": ";".join(partner["gt_names"]),
                    "pair_2_top_action_id": partner["top_action_id"],
                    "pair_2_top_action_name": partner["top_action_name"],
                    "pair_2_top_action_probability": f"{partner['top_probability']:.6f}",
                    "pair_2_final_correct": partner["correct"],
                    "partner_gt_positive": int(partner["positive"]),
                    "contrast_tier": tier_name(partner),
                }
                row["anchor_contrast_score"] = f"{score_candidate(row):.6f}"
                row["selection_note"] = (
                    "Correct GT anchor vs positive partner"
                    if partner["positive"]
                    else "Correct GT anchor vs false-positive diagnostic partner"
                )
                image_rows.append(row)
                seen_pairs.add(canonical)

        image_rows.sort(key=lambda item: float(item["anchor_contrast_score"]), reverse=True)
        rows.extend(image_rows[: args.max_per_image] if args.max_per_image > 0 else image_rows)

    rows.sort(key=lambda item: float(item["anchor_contrast_score"]), reverse=True)
    if args.top_n > 0:
        rows = rows[: args.top_n]
    write_rows(output_dir / "anchor_partner_candidates.csv", rows)
    tier_names = [
        "A_both_positive_correct",
        "B_positive_partner_wrong",
        "C_false_positive_partner",
    ]
    for tier in tier_names:
        write_rows(output_dir / f"{tier}.csv", [row for row in rows if row["contrast_tier"] == tier])
    summary = [
        f"record_count={record_count}",
        f"candidate_count={len(rows)}",
        *(f"{tier}={sum(row['contrast_tier'] == tier for row in rows)}" for tier in tier_names),
        *(f"skipped_{key}={value}" for key, value in sorted(skipped.items())),
    ]
    (output_dir / "anchor_partner_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
