#!/usr/bin/env python3
"""Rank EvidenceHOI test images for compact qualitative figures.

The script produces three complementary leaderboards:

1. ``hypothesis_sensitive_candidates.csv``: one pair, GT action vs. a
   plausible non-GT prototype with different evidence matching.
2. ``pair_specific_candidates.csv``: two pairs in one image whose VET context
   selections differ, preferably under a controlled shared human/object setup.
3. ``combined_hero_candidates.csv``: focal pairs satisfying both stories.

The score is only a mining heuristic. Final paper examples still require manual
inspection and, for faithfulness claims, a second-pass evidence-deletion run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

EPS = 1e-8


def load_trace(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older PyTorch.
        return torch.load(path, map_location="cpu")


def tensor(value: Any, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu()
    else:
        out = torch.as_tensor(value)
    return out.to(dtype=dtype) if dtype is not None else out


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.float()
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0.0)
    return wh[..., 0] * wh[..., 1]


def normalized_entropy(weights: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    w = weights.float().flatten()
    if mask is not None:
        m = mask.to(dtype=torch.bool).flatten()
        w = w[m]
    w = w.clamp(min=0.0)
    if w.numel() <= 1 or float(w.sum()) <= EPS:
        return 0.0
    w = w / w.sum().clamp(min=EPS)
    entropy = -(w * w.clamp(min=EPS).log()).sum()
    return float((entropy / math.log(w.numel())).item())


def js_divergence(p: torch.Tensor, q: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    p = p.float().flatten().clamp(min=0.0)
    q = q.float().flatten().clamp(min=0.0)
    if mask is not None:
        m = mask.to(dtype=torch.bool).flatten()
        p, q = p[m], q[m]
    if p.numel() == 0 or q.numel() == 0:
        return 0.0
    p = p / p.sum().clamp(min=EPS)
    q = q / q.sum().clamp(min=EPS)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp(min=EPS).log() - m.clamp(min=EPS).log())).sum()
    kl_qm = (q * (q.clamp(min=EPS).log() - m.clamp(min=EPS).log())).sum()
    # Normalize by log(2), so the result lies approximately in [0, 1].
    return float((0.5 * (kl_pm + kl_qm) / math.log(2.0)).clamp(min=0.0, max=1.0).item())


def attention_l1_distance(a: torch.Tensor, b: torch.Tensor, slot_mask: Optional[torch.Tensor] = None) -> float:
    """Average total-variation distance across prototype slots."""
    a = a.float()
    b = b.float()
    if a.shape != b.shape or a.dim() != 2:
        return 0.0
    per_slot = 0.5 * (a - b).abs().sum(dim=-1)
    if slot_mask is not None:
        valid = slot_mask.to(dtype=torch.bool)
        if not bool(valid.any()):
            return 0.0
        per_slot = per_slot[valid]
    return float(per_slot.mean().clamp(min=0.0, max=1.0).item())


def set_jaccard_distance(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - len(sa & sb) / len(union)


def weighted_jaccard_distance(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    numerator = sum(min(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    denominator = sum(max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) for k in keys)
    if denominator <= EPS:
        return 0.0
    return 1.0 - numerator / denominator


def sigmoid_scalar(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def action_index(action_id: int, num_logits: int, valid_action_ids: Sequence[int]) -> Optional[int]:
    """Map a text-bank action id to the model's action-logit index."""
    if 0 <= action_id < num_logits:
        return int(action_id)
    if len(valid_action_ids) == num_logits:
        try:
            return list(valid_action_ids).index(int(action_id))
        except ValueError:
            return None
    return None


def load_name_map(path: Optional[str]) -> Mapping[int, str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {idx: str(name) for idx, name in enumerate(data)}
    if isinstance(data, dict):
        return {int(key): str(value) for key, value in data.items()}
    raise ValueError(f"Name map must be a list or dict: {path}")


def name_of(mapping: Mapping[int, str], idx: int) -> str:
    return mapping.get(int(idx), str(int(idx)))


def selected_contexts(graph: Mapping[str, Any], pair_idx: int) -> Tuple[List[int], List[float], int]:
    indices = tensor(graph.get("selected_context_indices"), torch.long)
    mask = tensor(graph.get("selected_context_mask"), torch.bool)
    weights = tensor(graph.get("candidate_evidence_weights"), torch.float32)
    two_hop = tensor(graph.get("selected_context_two_hop_mask"), torch.bool)
    if indices is None or mask is None or pair_idx >= indices.shape[0]:
        return [], [], 0
    valid = mask[pair_idx]
    ids = indices[pair_idx][valid].tolist()
    if weights is not None and weights.shape[-1] >= 3 + mask.shape[-1]:
        context_weights = weights[pair_idx, 3 : 3 + mask.shape[-1]][valid].tolist()
    else:
        context_weights = [1.0 for _ in ids]
    two_hop_count = int(two_hop[pair_idx][valid].sum().item()) if two_hop is not None else 0
    return [int(x) for x in ids], [float(x) for x in context_weights], two_hop_count


def visual_quality(
    record: Mapping[str, Any],
    pair_idx: int,
    context_ids: Sequence[int],
) -> Tuple[float, Dict[str, float]]:
    nodes = record["nodes"]
    pairs = record["pairs"]
    boxes = tensor(nodes["boxes_xyxy_norm"], torch.float32)
    scores = tensor(nodes["scores"], torch.float32)
    valid_mask = tensor(nodes["valid_mask"], torch.bool)
    pair_indices = tensor(pairs["indices"], torch.long)
    assert boxes is not None and scores is not None and valid_mask is not None and pair_indices is not None

    h_idx, o_idx = [int(x) for x in pair_indices[pair_idx].tolist()]
    areas = box_area(boxes)
    endpoint_min_area = float(min(areas[h_idx], areas[o_idx]).item())
    context_area = float(areas[list(context_ids)].median().item()) if context_ids else 0.0
    context_score = float(scores[list(context_ids)].mean().item()) if context_ids else 0.0
    valid_node_count = int(valid_mask.sum().item())

    orig_size = record.get("image", {}).get("orig_size_hw")
    aspect_quality = 0.5
    if isinstance(orig_size, (list, tuple)) and len(orig_size) == 2 and float(orig_size[0]) > 0:
        ratio = float(orig_size[1]) / float(orig_size[0])
        aspect_quality = math.exp(-abs(math.log(max(ratio, EPS) / 1.35)))

    components = {
        "endpoint_area_quality": clamp01(endpoint_min_area / 0.025),
        "context_area_quality": clamp01(context_area / 0.012),
        "context_score_quality": clamp01((context_score - 0.10) / 0.55),
        "clutter_quality": clamp01(1.0 - max(valid_node_count - 12, 0) / 20.0),
        "aspect_quality": clamp01(aspect_quality),
        "endpoint_min_area": endpoint_min_area,
        "context_median_area": context_area,
        "context_mean_score": context_score,
        "valid_node_count": float(valid_node_count),
    }
    score = (
        0.28 * components["endpoint_area_quality"]
        + 0.24 * components["context_area_quality"]
        + 0.22 * components["context_score_quality"]
        + 0.14 * components["clutter_quality"]
        + 0.12 * components["aspect_quality"]
    )
    return clamp01(score), components


def prototype_row_for_action(
    action_ids: torch.Tensor,
    prototype_mask: torch.Tensor,
    pair_idx: int,
    action_id: int,
) -> Optional[int]:
    matches = (action_ids[pair_idx] == int(action_id)) & prototype_mask[pair_idx]
    rows = matches.nonzero(as_tuple=False).flatten()
    return int(rows[0].item()) if rows.numel() else None


def hypothesis_rows(
    record: Mapping[str, Any],
    path: Path,
    args: argparse.Namespace,
    action_names: Mapping[int, str],
    object_names: Mapping[int, str],
) -> List[Dict[str, Any]]:
    graph = record.get("graph") or {}
    pairs = record.get("pairs") or {}
    if not graph or "sync_evidence_weights" not in graph or "selected_hoi_action_ids" not in graph:
        return []

    pair_indices = tensor(pairs.get("indices"), torch.long)
    action_logits = tensor(pairs.get("action_logits"), torch.float32)
    action_probs = tensor(pairs.get("action_probs"), torch.float32)
    coarse_logits = tensor(pairs.get("coarse_action_logits"), torch.float32)
    gt_targets = tensor(pairs.get("gt_action_targets"), torch.bool)
    gt_pair_matched = tensor(pairs.get("gt_pair_matched"), torch.bool)
    pair_obj_labels = tensor(pairs.get("object_labels"), torch.long)
    pair_obj_scores = tensor(pairs.get("object_scores"), torch.float32)
    pair_valid = tensor(graph.get("pair_valid_mask"), torch.bool)
    selected_action_ids = tensor(graph.get("selected_hoi_action_ids"), torch.long)
    prototype_mask = tensor(graph.get("selected_hoi_mask"), torch.bool)
    evidence_weights = tensor(graph.get("sync_evidence_weights"), torch.float32)
    evidence_mask = tensor(graph.get("candidate_evidence_mask"), torch.bool)
    text_to_visual = tensor(graph.get("sync_text_to_visual_attn"), torch.float32)
    coverage = tensor(graph.get("sync_prototype_coverage"), torch.float32)
    slot_mask = tensor(graph.get("prototype_slot_mask"), torch.bool)
    if any(value is None for value in (
        pair_indices,
        action_logits,
        action_probs,
        coarse_logits,
        gt_targets,
        gt_pair_matched,
        pair_obj_labels,
        pair_obj_scores,
        pair_valid,
        selected_action_ids,
        prototype_mask,
        evidence_weights,
        evidence_mask,
        text_to_visual,
        coverage,
        slot_mask,
    )):
        return []

    valid_action_ids = record.get("metadata", {}).get("valid_action_ids", [])
    rows: List[Dict[str, Any]] = []
    pair_count = int(pair_indices.shape[0])
    for pair_idx in range(pair_count):
        if not bool(pair_valid[pair_idx]) or not bool(gt_pair_matched[pair_idx]):
            continue
        context_ids, context_weights, two_hop_count = selected_contexts(graph, pair_idx)
        if len(context_ids) < args.min_context_count or two_hop_count < args.min_two_hop_count:
            continue

        quality, quality_parts = visual_quality(record, pair_idx, context_ids)
        if quality_parts["context_mean_score"] < args.min_context_score:
            continue
        if quality_parts["context_median_area"] < args.min_context_area:
            continue

        gt_action_ids = gt_targets[pair_idx].nonzero(as_tuple=False).flatten().tolist()
        gt_action_ids = [int(x) for x in gt_action_ids]
        if not gt_action_ids:
            continue

        # Evaluate every GT prototype and keep every valid competitor; the CSV
        # sorter will surface the strongest story.
        for gt_action_id in gt_action_ids:
            gt_proto_row = prototype_row_for_action(
                selected_action_ids, prototype_mask, pair_idx, gt_action_id
            )
            if gt_proto_row is None and args.require_gt_prototype:
                continue
            if gt_proto_row is None:
                continue
            gt_logit_idx = action_index(gt_action_id, action_logits.shape[-1], valid_action_ids)
            if gt_logit_idx is None:
                continue

            competitor_candidates: List[Tuple[float, int, int]] = []
            for proto_row in range(selected_action_ids.shape[1]):
                if not bool(prototype_mask[pair_idx, proto_row]):
                    continue
                competitor_action_id = int(selected_action_ids[pair_idx, proto_row].item())
                if competitor_action_id < 0 or competitor_action_id in gt_action_ids:
                    continue
                competitor_logit_idx = action_index(
                    competitor_action_id, action_logits.shape[-1], valid_action_ids
                )
                if competitor_logit_idx is None:
                    continue
                competitor_candidates.append((
                    float(action_probs[pair_idx, competitor_logit_idx].item()),
                    competitor_action_id,
                    proto_row,
                ))
            if not competitor_candidates:
                continue
            competitor_prob, competitor_action_id, competitor_proto_row = max(
                competitor_candidates, key=lambda item: item[0]
            )
            competitor_logit_idx = action_index(
                competitor_action_id, action_logits.shape[-1], valid_action_ids
            )
            assert competitor_logit_idx is not None

            gt_prob = float(action_probs[pair_idx, gt_logit_idx].item())
            gt_logit = float(action_logits[pair_idx, gt_logit_idx].item())
            competitor_logit = float(action_logits[pair_idx, competitor_logit_idx].item())
            coarse_gt_logit = float(coarse_logits[pair_idx, gt_logit_idx].item())
            coarse_comp_logit = float(coarse_logits[pair_idx, competitor_logit_idx].item())
            final_margin = gt_logit - competitor_logit
            coarse_margin = coarse_gt_logit - coarse_comp_logit
            refinement_gain = final_margin - coarse_margin
            final_correct = gt_prob > competitor_prob
            if args.require_final_correct and not final_correct:
                continue

            valid_visual_mask = evidence_mask[pair_idx]
            gt_w = evidence_weights[pair_idx, gt_proto_row]
            comp_w = evidence_weights[pair_idx, competitor_proto_row]
            hypothesis_js = js_divergence(gt_w, comp_w, valid_visual_mask)
            common_slot_mask = slot_mask[pair_idx, gt_proto_row] & slot_mask[pair_idx, competitor_proto_row]
            heatmap_distance = attention_l1_distance(
                text_to_visual[pair_idx, gt_proto_row],
                text_to_visual[pair_idx, competitor_proto_row],
                common_slot_mask,
            )
            concentration = 1.0 - normalized_entropy(gt_w, valid_visual_mask)
            gt_coverage = float(
                coverage[pair_idx, gt_proto_row][slot_mask[pair_idx, gt_proto_row]].mean().item()
            ) if bool(slot_mask[pair_idx, gt_proto_row].any()) else 0.0

            context_slice = slice(3, int(valid_visual_mask.shape[0]))
            gt_context_w = gt_w[context_slice]
            comp_context_w = comp_w[context_slice]
            top_context_switch = 0.0
            gt_top_context_rank = -1
            comp_top_context_rank = -1
            if gt_context_w.numel() > 0 and len(context_ids) > 0:
                gt_top_context_rank = int(gt_context_w[: len(context_ids)].argmax().item())
                comp_top_context_rank = int(comp_context_w[: len(context_ids)].argmax().item())
                top_context_switch = float(gt_top_context_rank != comp_top_context_rank)

            ambiguity = (
                math.exp(-4.0 * abs(gt_prob - competitor_prob))
                * clamp01(2.0 * min(gt_prob, competitor_prob))
            )
            refinement_gain_norm = sigmoid_scalar(refinement_gain)
            path_quality = clamp01(0.55 * min(len(context_ids) / 3.0, 1.0) + 0.45 * (two_hop_count / max(len(context_ids), 1)))
            hypothesis_score = clamp01(
                0.26 * hypothesis_js
                + 0.18 * heatmap_distance
                + 0.16 * ambiguity
                + 0.12 * concentration
                + 0.10 * top_context_switch
                + 0.10 * clamp01(gt_prob)
                + 0.08 * refinement_gain_norm
            )

            h_idx, o_idx = [int(x) for x in pair_indices[pair_idx].tolist()]
            object_label = int(pair_obj_labels[pair_idx].item())
            row = {
                "record_path": str(path),
                "image_id": record.get("image", {}).get("image_id"),
                "file_name": record.get("image", {}).get("file_name"),
                "pair_idx": pair_idx,
                "human_node_idx": h_idx,
                "object_node_idx": o_idx,
                "object_label": object_label,
                "object_name": name_of(object_names, object_label),
                "object_score": float(pair_obj_scores[pair_idx].item()),
                "gt_action_id": gt_action_id,
                "gt_action_name": name_of(action_names, gt_action_id),
                "competitor_action_id": competitor_action_id,
                "competitor_action_name": name_of(action_names, competitor_action_id),
                "gt_prototype_row": gt_proto_row,
                "competitor_prototype_row": competitor_proto_row,
                "gt_probability": gt_prob,
                "competitor_probability": competitor_prob,
                "final_margin": final_margin,
                "coarse_margin": coarse_margin,
                "refinement_gain": refinement_gain,
                "refinement_gain_norm": refinement_gain_norm,
                "final_correct": int(final_correct),
                "ambiguity": ambiguity,
                "hypothesis_js": hypothesis_js,
                "heatmap_distance": heatmap_distance,
                "evidence_concentration": concentration,
                "prototype_coverage": gt_coverage,
                "top_context_switch": top_context_switch,
                "gt_top_context_rank": gt_top_context_rank,
                "competitor_top_context_rank": comp_top_context_rank,
                "context_ids": ";".join(str(x) for x in context_ids),
                "context_weights": ";".join(f"{x:.5f}" for x in context_weights),
                "context_count": len(context_ids),
                "two_hop_count": two_hop_count,
                "path_quality": path_quality,
                "visual_quality": quality,
                "hypothesis_score": hypothesis_score,
                **quality_parts,
            }
            rows.append(row)
    return rows


def pair_specific_rows(
    record: Mapping[str, Any],
    path: Path,
    args: argparse.Namespace,
    object_names: Mapping[int, str],
) -> List[Dict[str, Any]]:
    graph = record.get("graph") or {}
    pairs = record.get("pairs") or {}
    pair_indices = tensor(pairs.get("indices"), torch.long)
    pair_obj_labels = tensor(pairs.get("object_labels"), torch.long)
    pair_valid = tensor(graph.get("pair_valid_mask"), torch.bool)
    candidate_weights = tensor(graph.get("candidate_evidence_weights"), torch.float32)
    if any(value is None for value in (pair_indices, pair_obj_labels, pair_valid)):
        return []

    rows: List[Dict[str, Any]] = []
    pair_count = int(pair_indices.shape[0])
    for first in range(pair_count):
        if not bool(pair_valid[first]):
            continue
        first_ids, first_weights, first_two_hop = selected_contexts(graph, first)
        if len(first_ids) < args.min_context_count or first_two_hop < args.min_two_hop_count:
            continue
        first_quality, first_quality_parts = visual_quality(record, first, first_ids)
        if first_quality_parts["context_mean_score"] < args.min_context_score:
            continue
        first_weight_map = {node_id: weight for node_id, weight in zip(first_ids, first_weights)}
        for second in range(first + 1, pair_count):
            if not bool(pair_valid[second]):
                continue
            second_ids, second_weights, second_two_hop = selected_contexts(graph, second)
            if len(second_ids) < args.min_context_count or second_two_hop < args.min_two_hop_count:
                continue
            second_quality, second_quality_parts = visual_quality(record, second, second_ids)
            second_weight_map = {node_id: weight for node_id, weight in zip(second_ids, second_weights)}

            h1, o1 = [int(x) for x in pair_indices[first].tolist()]
            h2, o2 = [int(x) for x in pair_indices[second].tolist()]
            obj1 = int(pair_obj_labels[first].item())
            obj2 = int(pair_obj_labels[second].item())
            same_object_node = o1 == o2
            same_human_node = h1 == h2
            same_object_class = obj1 == obj2
            if same_object_node:
                control_type, control_bonus = "same_object_node", 1.0
            elif same_human_node:
                control_type, control_bonus = "same_human_node", 0.90
            elif same_object_class:
                control_type, control_bonus = "same_object_class", 0.78
            else:
                control_type, control_bonus = "uncontrolled", 0.45
            if args.require_controlled_pair and control_type == "uncontrolled":
                continue

            set_distance = set_jaccard_distance(first_ids, second_ids)
            weighted_distance = weighted_jaccard_distance(first_weight_map, second_weight_map)
            specificity = clamp01(0.52 * set_distance + 0.48 * weighted_distance)
            quality = min(first_quality, second_quality)
            path_quality = min(
                clamp01(first_two_hop / max(len(first_ids), 1)),
                clamp01(second_two_hop / max(len(second_ids), 1)),
            )
            pair_specific_score = clamp01(
                0.55 * specificity + 0.22 * control_bonus + 0.13 * quality + 0.10 * path_quality
            )
            rows.append({
                "record_path": str(path),
                "image_id": record.get("image", {}).get("image_id"),
                "file_name": record.get("image", {}).get("file_name"),
                "pair_idx_1": first,
                "pair_idx_2": second,
                "human_node_idx_1": h1,
                "object_node_idx_1": o1,
                "human_node_idx_2": h2,
                "object_node_idx_2": o2,
                "object_label_1": obj1,
                "object_name_1": name_of(object_names, obj1),
                "object_label_2": obj2,
                "object_name_2": name_of(object_names, obj2),
                "control_type": control_type,
                "control_bonus": control_bonus,
                "context_ids_1": ";".join(str(x) for x in first_ids),
                "context_ids_2": ";".join(str(x) for x in second_ids),
                "context_set_distance": set_distance,
                "context_weighted_distance": weighted_distance,
                "pair_specificity": specificity,
                "two_hop_count_1": first_two_hop,
                "two_hop_count_2": second_two_hop,
                "visual_quality": quality,
                "path_quality": path_quality,
                "pair_specific_score": pair_specific_score,
            })
    return rows


def combined_rows(
    hypothesis: Sequence[Dict[str, Any]],
    pair_specific: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_record: Dict[str, List[Dict[str, Any]]] = {}
    for row in pair_specific:
        by_record.setdefault(str(row["record_path"]), []).append(row)

    combined: List[Dict[str, Any]] = []
    for hrow in hypothesis:
        candidates = by_record.get(str(hrow["record_path"]), [])
        focal = int(hrow["pair_idx"])
        matching = [
            row for row in candidates
            if focal in (int(row["pair_idx_1"]), int(row["pair_idx_2"]))
        ]
        if not matching:
            continue
        partner = max(matching, key=lambda row: float(row["pair_specific_score"]))
        partner_idx = int(partner["pair_idx_2"]) if int(partner["pair_idx_1"]) == focal else int(partner["pair_idx_1"])
        combined_score = clamp01(
            0.38 * float(hrow["hypothesis_score"])
            + 0.30 * float(partner["pair_specific_score"])
            + 0.14 * float(hrow["refinement_gain_norm"])
            + 0.10 * float(hrow["visual_quality"])
            + 0.08 * float(hrow["path_quality"])
        )
        row = dict(hrow)
        row.update({
            "partner_pair_idx": partner_idx,
            "pair_control_type": partner["control_type"],
            "pair_specificity": partner["pair_specificity"],
            "pair_specific_score": partner["pair_specific_score"],
            "partner_context_ids": (
                partner["context_ids_2"]
                if int(partner["pair_idx_1"]) == focal
                else partner["context_ids_1"]
            ),
            "combined_score": combined_score,
        })
        combined.append(row)
    return combined


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_and_limit(rows: List[Dict[str, Any]], key: str, top_n: int) -> List[Dict[str, Any]]:
    rows.sort(key=lambda row: float(row.get(key, 0.0)), reverse=True)
    return rows[:top_n] if top_n > 0 else rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, help="Directory containing records/*.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records-glob", default="records/*.pt")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--action-names-json")
    parser.add_argument("--object-names-json")
    parser.add_argument("--min-context-count", type=int, default=2)
    parser.add_argument("--min-two-hop-count", type=int, default=1)
    parser.add_argument("--min-context-score", type=float, default=0.15)
    parser.add_argument("--min-context-area", type=float, default=0.002)
    parser.add_argument("--require-gt-prototype", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-final-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-controlled-pair", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_names = load_name_map(args.action_names_json)
    object_names = load_name_map(args.object_names_json)

    hypothesis: List[Dict[str, Any]] = []
    pair_specific: List[Dict[str, Any]] = []
    record_paths = sorted(trace_dir.glob(args.records_glob))
    for index, path in enumerate(record_paths, start=1):
        record = load_trace(path)
        hypothesis.extend(hypothesis_rows(record, path, args, action_names, object_names))
        pair_specific.extend(pair_specific_rows(record, path, args, object_names))
        if index % 500 == 0:
            print(f"Processed {index}/{len(record_paths)} records")

    hypothesis = sort_and_limit(hypothesis, "hypothesis_score", args.top_n)
    pair_specific = sort_and_limit(pair_specific, "pair_specific_score", args.top_n)
    combined = sort_and_limit(combined_rows(hypothesis, pair_specific), "combined_score", args.top_n)

    write_csv(output_dir / "hypothesis_sensitive_candidates.csv", hypothesis)
    write_csv(output_dir / "pair_specific_candidates.csv", pair_specific)
    write_csv(output_dir / "combined_hero_candidates.csv", combined)
    summary = {
        "record_count": len(record_paths),
        "hypothesis_candidate_count": len(hypothesis),
        "pair_specific_candidate_count": len(pair_specific),
        "combined_candidate_count": len(combined),
        "filters": {
            "min_context_count": args.min_context_count,
            "min_two_hop_count": args.min_two_hop_count,
            "min_context_score": args.min_context_score,
            "min_context_area": args.min_context_area,
            "require_gt_prototype": args.require_gt_prototype,
            "require_final_correct": args.require_final_correct,
            "require_controlled_pair": args.require_controlled_pair,
        },
    }
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
