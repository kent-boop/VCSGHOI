#!/usr/bin/env python3
"""Rank correct-GT single-pair EvidenceHOI traces for paper visualization.

The ranker is intentionally trace-only: it does not rerun the model.  For each
GT-positive pair whose final top-1 action is correct, it finds the matching
retrieved action prototype and measures the final A/H/O/C1-C4 allocation from
``sync_evidence_weights``.  Context nodes are marked informative when they are
not the human/person endpoint class, not the target-object detector class, and
do not substantially overlap either endpoint.

Outputs retain both the complete audit set and the filtered paper candidates.
The score is a mining heuristic; source-image clarity still requires manual
inspection.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from rank_evidence_examples import box_area, clamp01, visual_quality
from render_evidence_shortlist import load_name_map, load_trace
from render_pair_specific_shortlist import box_iou


EPS = 1e-12


def name_of(mapping: Mapping[int, str], index: int) -> str:
    return mapping.get(int(index), str(int(index)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().cpu().flatten()[0].item())
    return float(value)


def semicolon(values: Iterable[Any]) -> str:
    return ";".join(str(value) for value in values)


def prototype_row_for_action(
    action_ids: torch.Tensor,
    action_mask: torch.Tensor,
    pair_idx: int,
    action_id: int,
) -> Optional[int]:
    matches = action_mask[pair_idx].bool() & (action_ids[pair_idx].long() == int(action_id))
    rows = torch.nonzero(matches, as_tuple=False).flatten()
    return int(rows[0].item()) if rows.numel() else None


def quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p99": None, "max": None}
    data = torch.tensor(list(values), dtype=torch.float64)
    return {
        "count": int(data.numel()),
        "p50": float(torch.quantile(data, 0.50).item()),
        "p90": float(torch.quantile(data, 0.90).item()),
        "p99": float(torch.quantile(data, 0.99).item()),
        "max": float(data.max().item()),
    }


def percentile_lookup(values: Sequence[float]):
    ordered = sorted(float(value) for value in values)

    def lookup(value: float) -> float:
        if not ordered:
            return 0.0
        return bisect.bisect_right(ordered, float(value)) / len(ordered)

    return lookup


def context_rows(
    record: Mapping[str, Any],
    pair_idx: int,
    prototype_row: int,
    node_names: Mapping[int, str],
    max_endpoint_iou: float,
    min_informative_score: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    nodes = record["nodes"]
    pairs = record["pairs"]
    graph = record["graph"]

    boxes = nodes["boxes_xyxy_norm"].float()
    labels = nodes["labels"].long()
    scores = nodes["scores"].float()
    pair_nodes = pairs["indices"].long()
    human_idx, object_idx = [int(value) for value in pair_nodes[pair_idx].tolist()]
    human_detector_label = int(labels[human_idx].item())
    object_detector_label = int(labels[object_idx].item())

    selected_indices = graph["selected_context_indices"][pair_idx].long()
    selected_mask = graph["selected_context_mask"][pair_idx].bool()
    two_hop = graph["selected_context_two_hop_mask"][pair_idx].bool()
    candidate_weights = graph["candidate_evidence_weights"][pair_idx].float()
    action_weights = graph["sync_evidence_weights"][pair_idx, prototype_row].float()
    evidence_mask = graph["candidate_evidence_mask"][pair_idx].bool()
    hk_scores = graph.get("selected_context_edge_hk_score")
    ko_scores = graph.get("selected_context_edge_ko_score")
    hk_masks = graph.get("selected_context_edge_hk_mask")
    ko_masks = graph.get("selected_context_edge_ko_mask")
    areas = box_area(boxes)

    total_action_weight = float(action_weights[evidence_mask].sum().item())
    details: List[Dict[str, Any]] = []
    for slot in range(int(selected_indices.shape[0])):
        if not bool(selected_mask[slot]):
            continue
        node_idx = int(selected_indices[slot].item())
        label = int(labels[node_idx].item())
        node_name = name_of(node_names, label)
        iou_h = box_iou(boxes[node_idx], boxes[human_idx])
        iou_o = box_iou(boxes[node_idx], boxes[object_idx])
        max_iou = max(iou_h, iou_o)
        reasons: List[str] = []
        if node_idx in {human_idx, object_idx}:
            reasons.append("endpoint_node")
        if label == human_detector_label or node_name.strip().lower() == "person":
            reasons.append("person")
        if label == object_detector_label:
            reasons.append("target_object_class")
        if max_iou >= max_endpoint_iou:
            reasons.append("endpoint_overlap")
        node_score = float(scores[node_idx].item())
        if node_score < min_informative_score:
            reasons.append("low_detector_score")

        hk_raw = scalar(hk_scores[pair_idx, slot]) if isinstance(hk_scores, torch.Tensor) else 0.0
        ko_raw = scalar(ko_scores[pair_idx, slot]) if isinstance(ko_scores, torch.Tensor) else 0.0
        hk_valid = bool(hk_masks[pair_idx, slot]) if isinstance(hk_masks, torch.Tensor) else False
        ko_valid = bool(ko_masks[pair_idx, slot]) if isinstance(ko_masks, torch.Tensor) else False
        action_weight = float(action_weights[3 + slot].item())
        details.append({
            "slot": slot + 1,
            "node_idx": node_idx,
            "detector_label": label,
            "name": node_name,
            "detector_score": node_score,
            "box_area": float(areas[node_idx].item()),
            "iou_human": iou_h,
            "iou_object": iou_o,
            "max_endpoint_iou": max_iou,
            "two_hop": bool(two_hop[slot]),
            "hk_valid": hk_valid,
            "ko_valid": ko_valid,
            "hk_strength": sigmoid(hk_raw),
            "ko_strength": sigmoid(ko_raw),
            "candidate_selection_weight": float(candidate_weights[3 + slot].item()),
            "final_action_weight": action_weight,
            "final_action_weight_percent": 100.0 * action_weight / max(total_action_weight, EPS),
            "informative": not reasons,
            "exclusion_reasons": reasons,
        })

    valid_slots = [item for item in details]
    informative = [item for item in details if item["informative"]]
    context_weight_sum = sum(float(item["final_action_weight"]) for item in valid_slots)
    informative_weight_sum = sum(float(item["final_action_weight"]) for item in informative)
    candidate_context_weight_sum = sum(float(item["candidate_selection_weight"]) for item in valid_slots)
    informative_scores = [float(item["detector_score"]) for item in informative]
    informative_areas = [float(item["box_area"]) for item in informative]
    informative_two_hop = sum(bool(item["two_hop"]) for item in informative)
    unique_labels = {int(item["detector_label"]) for item in informative}
    metrics = {
        "total_action_weight": total_action_weight,
        "context_weight_sum": context_weight_sum,
        "context_weight_percent": 100.0 * context_weight_sum / max(total_action_weight, EPS),
        "candidate_context_weight_sum": candidate_context_weight_sum,
        "informative_context_count": len(informative),
        "informative_context_weight_sum": informative_weight_sum,
        "informative_context_weight_percent": 100.0 * informative_weight_sum / max(total_action_weight, EPS),
        "informative_context_share": informative_weight_sum / max(context_weight_sum, EPS),
        "informative_context_mean_score": sum(informative_scores) / max(len(informative_scores), 1),
        "informative_context_min_score": min(informative_scores) if informative_scores else 0.0,
        "informative_context_median_area": (
            float(torch.tensor(informative_areas).median().item()) if informative_areas else 0.0
        ),
        "informative_two_hop_count": informative_two_hop,
        "informative_two_hop_fraction": informative_two_hop / max(len(informative), 1),
        "informative_label_diversity": len(unique_labels) / max(len(informative), 1),
        "human_detector_label": human_detector_label,
        "human_detector_name": name_of(node_names, human_detector_label),
        "object_detector_label": object_detector_label,
        "object_detector_name": name_of(node_names, object_detector_label),
    }
    return details, metrics


def scan_record(
    record: Mapping[str, Any],
    path: Path,
    args: argparse.Namespace,
    action_names: Mapping[int, str],
    object_names: Mapping[int, str],
    node_names: Mapping[int, str],
    skip_counts: Counter,
) -> List[Dict[str, Any]]:
    if not record.get("graph"):
        skip_counts["missing_graph"] += 1
        return []
    pairs = record["pairs"]
    graph = record["graph"]
    required_pair = ["indices", "action_probs", "gt_action_targets", "gt_pair_matched"]
    required_graph = [
        "pair_valid_mask", "selected_context_indices", "selected_context_mask",
        "candidate_evidence_mask", "candidate_evidence_weights",
        "selected_context_two_hop_mask", "selected_hoi_action_ids",
        "selected_hoi_mask", "sync_evidence_weights",
    ]
    if any(key not in pairs for key in required_pair) or any(key not in graph for key in required_graph):
        skip_counts["missing_required_fields"] += 1
        return []

    action_probs = pairs["action_probs"].float()
    gt_targets = pairs["gt_action_targets"].bool()
    gt_matched = pairs["gt_pair_matched"].bool()
    pair_valid = graph["pair_valid_mask"].bool()
    selected_actions = graph["selected_hoi_action_ids"].long()
    selected_action_mask = graph["selected_hoi_mask"].bool()
    pair_indices = pairs["indices"].long()
    pair_object_labels = pairs.get("object_labels")
    pair_object_scores = pairs.get("object_scores")
    proposal_probs = pairs.get("proposal_probs")

    rows: List[Dict[str, Any]] = []
    for pair_idx in range(int(action_probs.shape[0])):
        if not bool(pair_valid[pair_idx]):
            skip_counts["invalid_pair"] += 1
            continue
        gt_ids = torch.nonzero(gt_targets[pair_idx], as_tuple=False).flatten().tolist()
        gt_ids = [int(value) for value in gt_ids]
        if not gt_ids:
            skip_counts["no_positive_gt"] += 1
            continue
        if args.require_gt_pair_matched and not bool(gt_matched[pair_idx]):
            skip_counts["gt_not_matched"] += 1
            continue
        top_action_id = int(action_probs[pair_idx].argmax().item())
        top_probability = float(action_probs[pair_idx, top_action_id].item())
        if top_action_id not in gt_ids:
            skip_counts["top1_incorrect"] += 1
            continue
        prototype_row = prototype_row_for_action(
            selected_actions, selected_action_mask, pair_idx, top_action_id
        )
        if prototype_row is None:
            skip_counts["top1_prototype_missing"] += 1
            continue

        context_details, context_metrics = context_rows(
            record,
            pair_idx,
            prototype_row,
            node_names,
            args.max_endpoint_iou,
            args.min_informative_context_score,
        )
        context_ids = [int(item["node_idx"]) for item in context_details]
        quality, quality_parts = visual_quality(record, pair_idx, context_ids)
        h_idx, o_idx = [int(value) for value in pair_indices[pair_idx].tolist()]
        object_label = int(pair_object_labels[pair_idx].item()) if pair_object_labels is not None else -1
        object_score = scalar(pair_object_scores[pair_idx]) if pair_object_scores is not None else 0.0
        proposal_probability = scalar(proposal_probs[pair_idx]) if proposal_probs is not None else 0.0
        informative = [item for item in context_details if item["informative"]]
        noninformative = [item for item in context_details if not item["informative"]]

        row: Dict[str, Any] = {
            "record_path": str(path),
            "image_id": record.get("image", {}).get("image_id"),
            "file_name": record.get("image", {}).get("file_name"),
            "orig_height": (record.get("image", {}).get("orig_size_hw") or [0, 0])[0],
            "orig_width": (record.get("image", {}).get("orig_size_hw") or [0, 0])[1],
            "pair_idx": pair_idx,
            "human_node_idx": h_idx,
            "object_node_idx": o_idx,
            "object_label": object_label,
            "object_name": name_of(object_names, object_label),
            "object_score": object_score,
            "proposal_probability": proposal_probability,
            "gt_pair_matched": bool(gt_matched[pair_idx]),
            "gt_action_ids": semicolon(gt_ids),
            "gt_action_names": semicolon(name_of(action_names, value) for value in gt_ids),
            "top_action_id": top_action_id,
            "top_action_name": name_of(action_names, top_action_id),
            "top_action_probability": top_probability,
            "top_action_prototype_row": prototype_row,
            "final_correct": True,
            "context_count": len(context_details),
            "context_node_ids": semicolon(item["node_idx"] for item in context_details),
            "context_names": semicolon(item["name"] for item in context_details),
            "context_detector_scores": semicolon(f"{item['detector_score']:.6f}" for item in context_details),
            "context_final_weights": semicolon(f"{item['final_action_weight']:.10f}" for item in context_details),
            "context_two_hop": semicolon(int(item["two_hop"]) for item in context_details),
            "context_endpoint_ious": semicolon(f"{item['max_endpoint_iou']:.6f}" for item in context_details),
            "informative_context_node_ids": semicolon(item["node_idx"] for item in informative),
            "informative_context_names": semicolon(item["name"] for item in informative),
            "informative_context_detector_scores": semicolon(f"{item['detector_score']:.6f}" for item in informative),
            "informative_context_final_weights": semicolon(f"{item['final_action_weight']:.10f}" for item in informative),
            "informative_context_two_hop": semicolon(int(item["two_hop"]) for item in informative),
            "informative_context_endpoint_ious": semicolon(f"{item['max_endpoint_iou']:.6f}" for item in informative),
            "noninformative_context_names": semicolon(item["name"] for item in noninformative),
            "noninformative_reasons": semicolon(
                f"C{item['slot']}:{','.join(item['exclusion_reasons'])}" for item in noninformative
            ),
            "context_details_json": json.dumps(context_details, ensure_ascii=False, separators=(",", ":")),
            "visual_quality": quality,
            **context_metrics,
            **quality_parts,
        }
        row["eligible"] = (
            context_metrics["informative_context_count"] >= args.min_informative_context_count
            and context_metrics["informative_context_weight_sum"] >= args.min_informative_context_weight
            and (not args.require_informative_two_hop or context_metrics["informative_two_hop_count"] > 0)
        )
        rows.append(row)
    return rows


def add_ranking_scores(rows: List[Dict[str, Any]]) -> None:
    eligible = [row for row in rows if row["eligible"]]
    info_weight_pct = percentile_lookup([row["informative_context_weight_sum"] for row in eligible])
    context_weight_pct = percentile_lookup([row["context_weight_sum"] for row in eligible])
    pair_prob_pct = percentile_lookup([row["top_action_probability"] for row in eligible])
    visual_pct = percentile_lookup([row["visual_quality"] for row in eligible])
    detector_pct = percentile_lookup([row["informative_context_mean_score"] for row in eligible])
    area_pct = percentile_lookup([row["informative_context_median_area"] for row in eligible])

    for row in rows:
        row["informative_weight_percentile"] = info_weight_pct(row["informative_context_weight_sum"])
        row["context_weight_percentile"] = context_weight_pct(row["context_weight_sum"])
        row["pair_probability_percentile"] = pair_prob_pct(row["top_action_probability"])
        row["visual_quality_percentile"] = visual_pct(row["visual_quality"])
        row["informative_detector_percentile"] = detector_pct(row["informative_context_mean_score"])
        row["informative_area_percentile"] = area_pct(row["informative_context_median_area"])
        path_quality = clamp01(
            0.70 * row["informative_two_hop_fraction"]
            + 0.30 * row["informative_label_diversity"]
        )
        row["informative_path_quality"] = path_quality
        row["paper_score"] = (
            0.30 * row["informative_weight_percentile"]
            + 0.08 * row["context_weight_percentile"]
            + 0.14 * row["pair_probability_percentile"]
            + 0.14 * row["visual_quality_percentile"]
            + 0.10 * row["informative_detector_percentile"]
            + 0.08 * row["informative_area_percentile"]
            + 0.10 * path_quality
            + 0.04 * clamp01(row["object_score"])
            + 0.02 * clamp01(row["proposal_probability"])
        ) if row["eligible"] else 0.0


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def diverse_action_rows(rows: Sequence[Mapping[str, Any]], per_action: int) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = defaultdict(int)
    selected: List[Dict[str, Any]] = []
    for row in rows:
        action = str(row["top_action_name"])
        if counts[action] >= per_action:
            continue
        selected.append(dict(row))
        counts[action] += 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, help="Directory containing records/*.pt")
    parser.add_argument("--records-glob", default="records/*.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-names-json", required=True)
    parser.add_argument("--object-names-json", required=True)
    parser.add_argument("--node-names-json", required=True)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--per-action", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-endpoint-iou", type=float, default=0.65)
    parser.add_argument("--min-informative-context-count", type=int, default=1)
    parser.add_argument("--min-informative-context-score", type=float, default=0.15)
    parser.add_argument("--min-informative-context-weight", type=float, default=0.0)
    parser.add_argument("--require-informative-two-hop", action="store_true")
    parser.add_argument("--no-require-gt-pair-matched", dest="require_gt_pair_matched", action="store_false")
    parser.set_defaults(require_gt_pair_matched=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_names = load_name_map(args.action_names_json)
    object_names = load_name_map(args.object_names_json)
    node_names = load_name_map(args.node_names_json)
    record_paths = sorted(trace_dir.glob(args.records_glob))
    if args.max_records > 0:
        record_paths = record_paths[: args.max_records]

    skip_counts: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    for index, path in enumerate(record_paths, start=1):
        record = load_trace(path)
        rows.extend(
            scan_record(
                record, path, args, action_names, object_names, node_names, skip_counts
            )
        )
        if index % 100 == 0 or index == len(record_paths):
            print(f"Processed {index}/{len(record_paths)} records; correct pairs={len(rows)}")

    add_ranking_scores(rows)
    all_correct = sorted(rows, key=lambda row: float(row["paper_score"]), reverse=True)
    eligible = [row for row in all_correct if row["eligible"]]
    top_rows = eligible[: args.top_n] if args.top_n > 0 else eligible
    per_action_rows = diverse_action_rows(eligible, max(args.per_action, 1))

    write_csv(output_dir / "correct_pairs_all.csv", all_correct)
    write_csv(output_dir / "correct_context_candidates.csv", eligible)
    write_csv(output_dir / "top_candidates.csv", top_rows)
    write_csv(output_dir / "top_per_action.csv", per_action_rows)
    write_jsonl(output_dir / "correct_context_candidates.jsonl", eligible)

    action_counts = Counter(str(row["top_action_name"]) for row in eligible)
    summary = {
        "record_count": len(record_paths),
        "correct_pair_count": len(all_correct),
        "eligible_correct_context_count": len(eligible),
        "top_candidate_count": len(top_rows),
        "action_count": len(action_counts),
        "top_actions": dict(action_counts.most_common()),
        "filters": {
            "require_gt_pair_matched": args.require_gt_pair_matched,
            "max_endpoint_iou": args.max_endpoint_iou,
            "min_informative_context_count": args.min_informative_context_count,
            "min_informative_context_score": args.min_informative_context_score,
            "min_informative_context_weight": args.min_informative_context_weight,
            "require_informative_two_hop": args.require_informative_two_hop,
        },
        "quantiles_all_correct": {
            "context_weight_sum": quantiles([row["context_weight_sum"] for row in all_correct]),
            "context_weight_percent": quantiles([row["context_weight_percent"] for row in all_correct]),
            "informative_context_weight_sum": quantiles(
                [row["informative_context_weight_sum"] for row in all_correct]
            ),
            "informative_context_weight_percent": quantiles(
                [row["informative_context_weight_percent"] for row in all_correct]
            ),
        },
        "quantiles_eligible": {
            "context_weight_sum": quantiles([row["context_weight_sum"] for row in eligible]),
            "context_weight_percent": quantiles([row["context_weight_percent"] for row in eligible]),
            "informative_context_weight_sum": quantiles(
                [row["informative_context_weight_sum"] for row in eligible]
            ),
            "informative_context_weight_percent": quantiles(
                [row["informative_context_weight_percent"] for row in eligible]
            ),
            "paper_score": quantiles([row["paper_score"] for row in eligible]),
        },
        "skip_counts": dict(sorted(skip_counts.items())),
        "score_definition": {
            "informative_weight_percentile": 0.30,
            "context_weight_percentile": 0.08,
            "pair_probability_percentile": 0.14,
            "visual_quality_percentile": 0.14,
            "informative_detector_percentile": 0.10,
            "informative_area_percentile": 0.08,
            "informative_path_quality": 0.10,
            "object_score": 0.04,
            "proposal_probability": 0.02,
        },
    }
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
