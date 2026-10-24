#!/usr/bin/env python3
"""Export the fixed Top-20 EvidenceHOI mechanism validation package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch


VARIANTS = (
    "full",
    "no_anchor_writeback",
    "no_prototype_state_writeback",
    "no_setm",
    "endpoint_only_setm",
)
VISUAL_ROLES = ("E_ho", "X_h", "X_o", "P_1", "P_2", "P_3", "P_4")
TEXT_ROLES = ("H", "O", "A", "D", "S", "C")
STAGE_NAMES = (
    "base_query_decoder",
    "after_layer_0_decoder",
    "after_layer_1_decoder",
    "normal_final_decoder",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--top20-csv", type=Path, required=True)
    parser.add_argument("--test-annotation", type=Path, required=True)
    parser.add_argument("--train-annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--action-names", type=Path, required=True)
    parser.add_argument("--object-names", type=Path, required=True)
    parser.add_argument("--node-names", type=Path, required=True)
    parser.add_argument("--prompt-metadata", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_names(path: Path) -> Dict[int, str]:
    value = load_json(path)
    if isinstance(value, list):
        return {index: str(item) for index, item in enumerate(value)}
    return {int(key): str(item) for key, item in value.items()}


def scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else default
    if isinstance(value, np.generic):
        return value.item()
    return default if value is None else value


def tensor_item(value: Any, *indices: int, default: Any = None) -> Any:
    if not isinstance(value, torch.Tensor):
        return default
    try:
        return scalar(value[indices], default)
    except (IndexError, TypeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: OrderedDict[str, None] = OrderedDict()
    for row in rows:
        for key in row:
            fields.setdefault(str(key), None)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            serialized = {
                key: ("true" if value else "false") if isinstance(value, bool) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)
    os.replace(temp, path)


def trace_record_map(trace_dir: Path) -> Dict[int, Path]:
    rows = [
        json.loads(line)
        for line in (trace_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {}
    for row in rows:
        path = Path(row["record_path"])
        result[int(row["image_id"])] = path if path.is_absolute() else trace_dir / path
    return result


def mapping_rows(path: Path) -> Dict[str, Dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return {str(row["candidate_id"]): row for row in rows}


def tensor_hash(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
    return str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()


def proposal_fingerprint(record: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for section, key in (
        ("pairs", "indices"),
        ("pairs", "proposal_logits"),
        ("nodes", "boxes_xyxy_norm"),
        ("nodes", "labels"),
        ("nodes", "scores"),
    ):
        digest.update(section.encode())
        digest.update(key.encode())
        digest.update(tensor_hash(record[section][key]))
    return digest.hexdigest()


def retrieval_fingerprint(record: Mapping[str, Any], pair_idx: int) -> str:
    digest = hashlib.sha256()
    for layer in record["graph_layers"]:
        for key in ("selected_hoi_indices", "selected_hoi_mask", "selected_hoi_action_ids", "selected_hoi_object_ids"):
            value = layer.get(key)
            if isinstance(value, torch.Tensor):
                digest.update(key.encode())
                digest.update(tensor_hash(value[pair_idx]))
    return digest.hexdigest()


def l2(tensor: torch.Tensor) -> float:
    return float(tensor.float().norm().item())


def rms(tensor: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(tensor.float().square())).item()) if tensor.numel() else 0.0


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right).item() / denominator.item()) if denominator.item() > 0 else 0.0


def update_stats(pre: torch.Tensor, post: torch.Tensor) -> Dict[str, float]:
    delta = post.float() - pre.float()
    pre_norm = l2(pre)
    post_norm = l2(post)
    delta_norm = l2(delta)
    return {
        "pre_norm_l2": pre_norm,
        "post_norm_l2": post_norm,
        "delta_l2": delta_norm,
        "delta_rms": rms(delta),
        "relative_delta": delta_norm / max(pre_norm, 1e-12),
        "cosine_pre_post": cosine(pre, post),
    }


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    left_area = max(float(left[2]) - float(left[0]), 0.0) * max(float(left[3]) - float(left[1]), 0.0)
    right_area = max(float(right[2]) - float(right[0]), 0.0) * max(float(right[3]) - float(right[1]), 0.0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def norm_to_pixel(box: torch.Tensor, height: int, width: int) -> List[float]:
    return [
        float(box[0]) * width,
        float(box[1]) * height,
        float(box[2]) * width,
        float(box[3]) * height,
    ]


def json_list(value: Sequence[Any]) -> str:
    return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))


def build_hoi_id_map(train_annotations: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, int], int]:
    mapping: Dict[Tuple[int, int], Set[int]] = {}
    for image in train_annotations:
        boxes = image["annotations"]
        for hoi in image["hoi_annotation"]:
            key = (int(hoi["category_id"]), int(boxes[int(hoi["object_id"])]["category_id"]))
            mapping.setdefault(key, set()).add(int(hoi["hoi_category_id"]))
    ambiguous = {key: value for key, value in mapping.items() if len(value) != 1}
    if ambiguous or len(mapping) != 600:
        raise RuntimeError(f"Invalid HOI ID mapping: keys={len(mapping)}, ambiguous={len(ambiguous)}")
    return {key: next(iter(value)) for key, value in mapping.items()}


def selected_pair(record: Mapping[str, Any], pair_idx: int) -> Tuple[int, int]:
    pair = record["pairs"]["indices"][pair_idx]
    return int(pair[0].item()), int(pair[1].item())


def build_gt_rows(
    cases: Sequence[Mapping[str, str]],
    full_records: Mapping[int, Mapping[str, Any]],
    full_mapping: Mapping[str, Mapping[str, str]],
    test_by_file: Mapping[str, Mapping[str, Any]],
    hoi_id_map: Mapping[Tuple[int, int], int],
    action_names: Mapping[int, str],
    object_names: Mapping[int, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Set[int]], Dict[str, bool]]:
    rows: List[Dict[str, Any]] = []
    valid_actions: Dict[str, Set[int]] = {}
    mapping_valid_by_candidate: Dict[str, bool] = {}
    for case in cases:
        candidate_id = str(case["candidate_id"])
        rank = int(case["mechanism_rank"])
        image_id = int(case["image_id"])
        file_name = str(case["file_name"])
        mapping = full_mapping[candidate_id]
        pair_idx = int(mapping["pair_idx"])
        record = full_records[image_id]
        human_node, object_node = selected_pair(record, pair_idx)
        height, width = [int(value) for value in record["image"]["orig_size_hw"]]
        human_norm = record["nodes"]["boxes_xyxy_norm"][human_node]
        object_norm = record["nodes"]["boxes_xyxy_norm"][object_node]
        human_pixel = norm_to_pixel(human_norm, height, width)
        object_pixel = norm_to_pixel(object_norm, height, width)
        pred_object_category = int(record["nodes"]["labels"][object_node].item())
        valid_obj_ids = [int(value) for value in record["metadata"]["valid_object_ids"]]
        annotation = test_by_file[file_name]
        gt_boxes = annotation["annotations"]
        candidates = []
        for hoi in annotation["hoi_annotation"]:
            gt_h = gt_boxes[int(hoi["subject_id"])]
            gt_o = gt_boxes[int(hoi["object_id"])]
            human_iou = box_iou(human_pixel, gt_h["bbox"])
            object_iou = box_iou(object_pixel, gt_o["bbox"])
            object_compatible = int(gt_o["category_id"]) == pred_object_category
            candidates.append((min(human_iou, object_iou), object_compatible, hoi, gt_h, gt_o, human_iou, object_iou))
        valid = [
            item for item in candidates
            if item[1] and item[5] >= 0.5 and item[6] >= 0.5
        ]
        selected = valid if valid else sorted(candidates, key=lambda item: (item[1], item[0]), reverse=True)[:1]
        valid_actions[candidate_id] = set()
        mapping_valid_by_candidate[candidate_id] = bool(valid)
        if not selected:
            selected = [(0.0, False, None, None, None, 0.0, 0.0)]
        for _, object_compatible, hoi, gt_h, gt_o, human_iou, object_iou in selected:
            gt_available = hoi is not None
            action_raw = int(hoi["category_id"]) if gt_available else -1
            action_model = action_raw - 1 if action_raw > 0 else -1
            object_raw = int(gt_o["category_id"]) if gt_available else -1
            object_model = valid_obj_ids.index(object_raw) if object_raw in valid_obj_ids else -1
            mapping_valid = bool(
                gt_available and object_compatible and human_iou >= 0.5 and object_iou >= 0.5
            )
            if mapping_valid:
                valid_actions[candidate_id].add(action_model)
            pair_warning = str(mapping.get("pair_mapping_warning", "False")).lower() == "true"
            warning = bool(not mapping_valid or min(human_iou, object_iou) < 0.8 or pair_warning)
            reasons = []
            if not gt_available:
                reasons.append("no_gt_interaction")
            if gt_available and not object_compatible:
                reasons.append("object_category_mismatch")
            if human_iou < 0.5:
                reasons.append("low_human_iou")
            if object_iou < 0.5:
                reasons.append("low_object_iou")
            if pair_warning:
                reasons.append("original_to_validation_pair_warning")
            hoi_id = hoi_id_map.get((action_raw, object_raw), -1) if gt_available else -1
            rows.append({
                "mechanism_rank": rank,
                "candidate_id": candidate_id,
                "image_id": image_id,
                "file_name": file_name,
                "original_pair_idx": int(mapping["original_pair_idx"]),
                "selected_pair_idx": pair_idx,
                "pred_human_box_xyxy_norm": json_list(human_norm.tolist()),
                "pred_object_box_xyxy_norm": json_list(object_norm.tolist()),
                "pred_human_box_xyxy_pixel": json_list(human_pixel),
                "pred_object_box_xyxy_pixel": json_list(object_pixel),
                "gt_human_box_xyxy_pixel": "" if gt_h is None else json_list(gt_h["bbox"]),
                "gt_object_box_xyxy_pixel": "" if gt_o is None else json_list(gt_o["bbox"]),
                "human_iou": human_iou,
                "object_iou": object_iou,
                "min_pair_iou": min(human_iou, object_iou),
                "gt_hoi_id_raw_1based": hoi_id,
                "gt_hoi_id_zero_based": hoi_id - 1 if hoi_id > 0 else -1,
                "gt_action_category_id_raw": action_raw,
                "gt_action_id_model": action_model,
                "gt_action_name": action_names.get(action_model, ""),
                "gt_object_category_id_raw": object_raw,
                "gt_object_id_model": object_model,
                "gt_object_name": object_names.get(object_model, ""),
                "gt_available": gt_available,
                "mapping_valid": mapping_valid,
                "mapping_warning": warning,
                "mapping_method": "box_iou+object_category",
                "failure_reason": ";".join(reasons),
            })
    return rows, valid_actions, mapping_valid_by_candidate


def copy_and_npz(
    run_root: Path,
    cases: Sequence[Mapping[str, str]],
    records_by_variant: Mapping[str, Mapping[int, Path]],
    mappings: Mapping[str, Mapping[str, Mapping[str, str]]],
    image_root: Path,
) -> None:
    arrays_dir = run_root / "arrays"
    images_dir = run_root / "source_images"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    for variant, record_paths in records_by_variant.items():
        raw_dir = run_root / "raw_records" / variant
        raw_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            candidate_id = str(case["candidate_id"])
            image_id = int(case["image_id"])
            rank = int(case["mechanism_rank"])
            source = record_paths[image_id]
            destination = raw_dir / f"{rank:02d}_{source.name}"
            source_stat = source.stat()
            destination_stat = destination.stat() if destination.is_file() else None
            if destination_stat is None or (destination_stat.st_size, destination_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
                shutil.copy2(source, destination)
    for case in cases:
        candidate_id = str(case["candidate_id"])
        image_id = int(case["image_id"])
        rank = int(case["mechanism_rank"])
        file_name = str(case["file_name"])
        record = torch.load(records_by_variant["full"][image_id], map_location="cpu")
        pair_idx = int(mappings["full"][candidate_id]["pair_idx"])
        mechanism = record["mechanism"]
        arrays: Dict[str, np.ndarray] = {}
        for key, value in mechanism.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 0 and value.shape[0] == 32:
                    value = value[pair_idx]
                elif value.dim() > 1 and value.shape[1] == 32:
                    value = value[:, pair_idx]
                arrays[key] = value.detach().cpu().numpy()
        for layer_idx, layer in enumerate(record["graph_layers"]):
            for key, value in layer.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() > 0 and value.shape[0] == 32:
                        value = value[pair_idx]
                    arrays[f"layer_{layer_idx}__{key}"] = value.detach().cpu().numpy()

        graph_after_layers = mechanism["graph_pair_features_after_layers"]
        arrays["graph_pair_features_after_layer_0"] = graph_after_layers[0, pair_idx].detach().cpu().numpy()
        arrays["graph_pair_features_after_layer_1"] = graph_after_layers[1, pair_idx].detach().cpu().numpy()
        arrays["probe_decoder_hidden"] = mechanism["probe_decoder_hidden_final"][:, pair_idx].detach().cpu().numpy()
        canonical_layer_keys = {
            "pair_anchor_pre_sync": "pair_anchor_pre_sync",
            "sync_anchor_delta_proposed": "sync_anchor_delta_proposed",
            "sync_anchor_delta_applied": "sync_anchor_delta_applied",
            "pair_anchor_post_sync_actual": "pair_anchor_post_sync_actual",
            "candidate_evidence_tokens": "candidate_evidence_tokens",
            "candidate_evidence_mask": "candidate_evidence_mask",
            "candidate_evidence_weights": "candidate_evidence_weights",
            "prototype_slots_pre_sync": "prototype_slots_pre_sync",
            "updated_prototype_slots_proposed": "updated_prototype_slots_proposed",
            "prototype_slots_post_writeback_actual": "prototype_slots_post_writeback_actual",
            "prototype_slot_mask": "prototype_slot_mask",
            "match_logits": "sync_match_logits",
            "visual_to_text_probability": "sync_visual_to_text_attn",
            "text_to_visual_probability": "sync_text_to_visual_attn",
            "alpha": "sync_evidence_weights",
            "pi": "sync_prototype_weights",
            "role_coverage": "sync_prototype_coverage",
            "selected_context_indices": "selected_context_indices",
            "selected_hoi_indices": "selected_hoi_indices",
        }
        for canonical_name, trace_name in canonical_layer_keys.items():
            values = [layer[trace_name][pair_idx] for layer in record["graph_layers"]]
            arrays[canonical_name] = torch.stack(values, dim=0).detach().cpu().numpy()
        np.savez_compressed(arrays_dir / f"{rank:02d}_{candidate_id}.npz", **arrays)
        source_image = image_root / file_name
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        destination = images_dir / file_name
        if not destination.exists():
            shutil.copy2(source_image, destination)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    run_id = run_root.name
    cases = list(csv.DictReader(args.top20_csv.open("r", encoding="utf-8")))
    if len(cases) != 20:
        raise ValueError(f"Expected 20 cases, got {len(cases)}")
    action_names = load_names(args.action_names)
    object_names = load_names(args.object_names)
    node_names = load_names(args.node_names)
    test_annotations = load_json(args.test_annotation)
    train_annotations = load_json(args.train_annotation)
    test_by_file = {str(item["file_name"]): item for item in test_annotations}
    hoi_id_map = build_hoi_id_map(train_annotations)

    records_by_variant: Dict[str, Dict[int, Path]] = {}
    mappings: Dict[str, Dict[str, Dict[str, str]]] = {}
    loaded_records: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for variant in VARIANTS:
        trace_dir = run_root / "variants" / variant / "traces"
        records_by_variant[variant] = trace_record_map(trace_dir)
        mappings[variant] = mapping_rows(run_root / "manifests" / f"{variant}_pair_mapping.csv")
        loaded_records[variant] = {
            image_id: torch.load(path, map_location="cpu")
            for image_id, path in records_by_variant[variant].items()
        }
        if len(loaded_records[variant]) != 20:
            raise ValueError(f"{variant}: expected 20 records, got {len(loaded_records[variant])}")

    full_records = loaded_records["full"]
    sample_record = next(iter(full_records.values()))
    valid_object_ids = [int(value) for value in sample_record["metadata"]["valid_object_ids"]]
    write_json(run_root / "labels/action_id_mapping.json", {
        "schema_version": 1,
        "model_id_basis": "zero_based",
        "raw_category_id_basis": "one_based",
        "items": [
            {
                "action_id_model": action_id,
                "action_category_id_raw": action_id + 1,
                "action_name": action_names.get(action_id, ""),
            }
            for action_id in range(117)
        ],
    })
    write_json(run_root / "labels/object_id_mapping.json", {
        "schema_version": 1,
        "model_id_basis": "zero_based",
        "raw_category_id_basis": "COCO_category_id",
        "items": [
            {
                "object_id_model": object_id,
                "object_category_id_raw": raw_id,
                "object_name": object_names.get(object_id, ""),
            }
            for object_id, raw_id in enumerate(valid_object_ids)
        ],
    })
    gt_rows, valid_gt_actions, gt_mapping_valid = build_gt_rows(
        cases,
        full_records,
        mappings["full"],
        test_by_file,
        hoi_id_map,
        action_names,
        object_names,
    )
    write_csv(run_root / "tables/gt_pair_mapping.csv", gt_rows)

    query_rows: List[Dict[str, Any]] = []
    layer_score_rows: List[Dict[str, Any]] = []
    visual_rows: List[Dict[str, Any]] = []
    prototype_rows: List[Dict[str, Any]] = []
    anchor_rows: List[Dict[str, Any]] = []
    context_rows: List[Dict[str, Any]] = []

    for case in cases:
        candidate_id = str(case["candidate_id"])
        rank = int(case["mechanism_rank"])
        image_id = int(case["image_id"])
        file_name = str(case["file_name"])
        mapping = mappings["full"][candidate_id]
        pair_idx = int(mapping["pair_idx"])
        record = full_records[image_id]
        mechanism = record["mechanism"]
        base = mechanism["base_hoi_query"][pair_idx]
        graph_final = mechanism["graph_pair_features_final"][pair_idx]
        delta = mechanism["gated_query_delta"][pair_idx]
        corrected = mechanism["corrected_hoi_query"][pair_idx]
        residual_error = float((corrected.float() - base.float() - delta.float()).abs().max().item())
        query_rows.append({
            "run_id": run_id,
            "variant": "full",
            "mechanism_rank": rank,
            "candidate_id": candidate_id,
            "image_id": image_id,
            "file_name": file_name,
            "original_pair_idx": int(mapping["original_pair_idx"]),
            "selected_pair_idx": pair_idx,
            "pair_idx": pair_idx,
            "base_query_l2": l2(base),
            "graph_pair_features_l2": l2(graph_final),
            "query_delta_l2": l2(delta),
            "corrected_query_l2": l2(corrected),
            "cosine_base_vs_graph_pair": cosine(base, graph_final),
            "cosine_base_vs_delta": cosine(base, delta),
            "cosine_base_vs_corrected": cosine(base, corrected),
            "graph_gate_raw": float(mechanism["graph_gate_raw"]),
            "graph_gate_tanh": float(mechanism["graph_gate_tanh"]),
            "residual_max_abs_error": residual_error,
        })

        probe_logits = mechanism["probe_action_logits"][:, pair_idx]
        probe_probs = mechanism["probe_action_probabilities"][:, pair_idx]
        normal_probs = probe_probs[3]
        normal_top1 = int(normal_probs.argmax().item())
        normal_competitor = int(
            torch.topk(normal_probs, k=2).indices[1].item()
            if normal_top1 == int(torch.topk(normal_probs, k=2).indices[0].item())
            else torch.topk(normal_probs, k=2).indices[0].item()
        )
        for stage_idx, stage in enumerate(STAGE_NAMES):
            stage_top1 = int(probe_probs[stage_idx].argmax().item())
            stage_competitor = int(torch.topk(probe_probs[stage_idx], k=2).indices[1].item())
            for action_id in range(117):
                layer_score_rows.append({
                    "run_id": run_id,
                    "variant": "full",
                    "seed": 3407,
                    "mechanism_rank": rank,
                    "candidate_id": candidate_id,
                    "image_id": image_id,
                    "file_name": file_name,
                    "original_pair_idx": int(mapping["original_pair_idx"]),
                    "selected_pair_idx": pair_idx,
                    "pair_idx": pair_idx,
                    "proposal_fingerprint": proposal_fingerprint(record),
                    "stage": stage,
                    "layer_idx": [-1, 0, 1, 1][stage_idx],
                    "action_id_model": action_id,
                    "action_category_id_raw": action_id + 1,
                    "action_name": action_names.get(action_id, ""),
                    "action_logit": float(probe_logits[stage_idx, action_id]),
                    "action_probability": float(probe_probs[stage_idx, action_id]),
                    "is_gt_action": action_id in valid_gt_actions[candidate_id],
                    "is_normal_final_top1": action_id == normal_top1,
                    "top1_action_id": stage_top1,
                    "top1_action_name": action_names.get(stage_top1, ""),
                    "top1_probability": float(probe_probs[stage_idx, stage_top1]),
                    "strongest_competitor_action_id": stage_competitor,
                    "strongest_competitor_action_name": action_names.get(stage_competitor, ""),
                    "strongest_competitor_probability": float(probe_probs[stage_idx, stage_competitor]),
                })

        for layer_idx, layer in enumerate(record["graph_layers"]):
            selected_ids = layer["selected_hoi_action_ids"][pair_idx]
            selected_bank = layer["selected_hoi_indices"][pair_idx]
            selected_mask = layer["selected_hoi_mask"][pair_idx].bool()
            evidence_pre = layer["candidate_evidence_tokens"][pair_idx]
            evidence_post = layer["updated_visual_tokens_proposed"][pair_idx]
            evidence_mask = layer["candidate_evidence_mask"][pair_idx].bool()
            alpha = layer["sync_evidence_weights"][pair_idx]
            prototype_pre = layer["prototype_slots_pre_sync"][pair_idx]
            prototype_post = layer["updated_prototype_slots_proposed"][pair_idx]
            prototype_mask = layer["prototype_slot_mask"][pair_idx].bool()
            coverage = layer["sync_prototype_coverage"][pair_idx]
            pi = layer["sync_prototype_weights"][pair_idx]
            for prototype_idx in range(5):
                action_id = int(selected_ids[prototype_idx].item()) if selected_mask[prototype_idx] else -1
                for visual_idx, visual_role in enumerate(VISUAL_ROLES):
                    valid = bool(selected_mask[prototype_idx] and evidence_mask[visual_idx])
                    row = {
                        "run_id": run_id,
                        "variant": "full",
                        "mechanism_rank": rank,
                        "candidate_id": candidate_id,
                        "selected_pair_idx": pair_idx,
                        "layer_idx": layer_idx,
                        "prototype_idx": prototype_idx,
                        "prototype_action_id": action_id,
                        "prototype_action_name": action_names.get(action_id, ""),
                        "prototype_valid": bool(selected_mask[prototype_idx]),
                        "visual_idx": visual_idx,
                        "visual_role": visual_role,
                        "visual_valid": valid,
                        "setm_alpha_weight": float(alpha[prototype_idx, visual_idx]) if valid else "",
                    }
                    if valid:
                        row.update(update_stats(evidence_pre[visual_idx], evidence_post[prototype_idx, visual_idx]))
                    else:
                        row.update({key: "" for key in (
                            "pre_norm_l2", "post_norm_l2", "delta_l2", "delta_rms", "relative_delta", "cosine_pre_post"
                        )})
                    visual_rows.append(row)
                for role_idx, role in enumerate(TEXT_ROLES):
                    valid = bool(selected_mask[prototype_idx] and prototype_mask[prototype_idx, role_idx])
                    row = {
                        "run_id": run_id,
                        "variant": "full",
                        "mechanism_rank": rank,
                        "candidate_id": candidate_id,
                        "selected_pair_idx": pair_idx,
                        "layer_idx": layer_idx,
                        "prototype_idx": prototype_idx,
                        "prototype_action_id": action_id,
                        "prototype_action_name": action_names.get(action_id, ""),
                        "prototype_valid": bool(selected_mask[prototype_idx]),
                        "text_bank_index": int(selected_bank[prototype_idx]) if selected_mask[prototype_idx] else -1,
                        "text_role_idx": role_idx,
                        "text_role": role,
                        "text_role_raw": layer["prototype_slot_names"][role_idx],
                        "text_role_valid": valid,
                        "prototype_role_coverage": float(coverage[prototype_idx, role_idx]) if valid else "",
                        "setm_pi_weight": float(pi[prototype_idx]) if selected_mask[prototype_idx] else "",
                    }
                    if valid:
                        row.update(update_stats(prototype_pre[prototype_idx, role_idx], prototype_post[prototype_idx, role_idx]))
                    else:
                        row.update({key: "" for key in (
                            "pre_norm_l2", "post_norm_l2", "delta_l2", "delta_rms", "relative_delta", "cosine_pre_post"
                        )})
                    prototype_rows.append(row)

            anchor_pre = layer["pair_anchor_pre_sync"][pair_idx]
            anchor_proposed = layer["sync_anchor_delta_proposed"][pair_idx]
            anchor_applied = layer["sync_anchor_delta_applied"][pair_idx]
            anchor_post = layer["pair_anchor_post_sync_actual"][pair_idx]
            candidate_proposed = layer["candidate_anchor_delta_proposed"][pair_idx]
            candidate_applied = layer["candidate_anchor_delta_applied"][pair_idx]
            anchor_rows.append({
                "run_id": run_id,
                "variant": "full",
                "mechanism_rank": rank,
                "candidate_id": candidate_id,
                "selected_pair_idx": pair_idx,
                "layer_idx": layer_idx,
                "pair_valid": bool(layer["pair_valid_mask"][pair_idx]),
                "anchor_pre_l2": l2(anchor_pre),
                "anchor_delta_proposed_l2": l2(anchor_proposed),
                "anchor_delta_applied_l2": l2(anchor_applied),
                "anchor_post_actual_l2": l2(anchor_post),
                "anchor_relative_delta": l2(anchor_applied) / max(l2(anchor_pre), 1e-12),
                "cosine_pre_post": cosine(anchor_pre, anchor_post),
                "anchor_gate_raw": float(mechanism["setm_anchor_gate_raw"]),
                "anchor_gate_tanh": float(mechanism["setm_anchor_gate_tanh"]),
                "candidate_anchor_delta_proposed_l2": l2(candidate_proposed),
                "candidate_anchor_delta_applied_l2": l2(candidate_applied),
                "setm_anchor_delta_proposed_l2": l2(anchor_proposed),
                "setm_anchor_delta_applied_l2": l2(anchor_applied),
                "visual_writeback_applied": bool(layer["visual_writeback_applied"]),
            })

        layer0, layer1 = record["graph_layers"]
        for slot_idx in range(4):
            node0 = int(layer0["selected_context_indices"][pair_idx, slot_idx])
            node1 = int(layer1["selected_context_indices"][pair_idx, slot_idx])
            label0 = int(record["nodes"]["labels"][node0]) if node0 >= 0 else -1
            label1 = int(record["nodes"]["labels"][node1]) if node1 >= 0 else -1
            context_rows.append({
                "run_id": run_id,
                "mechanism_rank": rank,
                "candidate_id": candidate_id,
                "selected_pair_idx": pair_idx,
                "slot_idx": slot_idx,
                "layer_0_node_idx": node0,
                "layer_0_node_name": node_names.get(label0, ""),
                "layer_1_node_idx": node1,
                "layer_1_node_name": node_names.get(label1, ""),
                "changed": node0 != node1,
                "layer_0_candidate_weight": float(layer0["candidate_evidence_weights"][pair_idx, 3 + slot_idx]),
                "layer_1_candidate_weight": float(layer1["candidate_evidence_weights"][pair_idx, 3 + slot_idx]),
                "layer_0_detector_score": float(record["nodes"]["scores"][node0]) if node0 >= 0 else "",
                "layer_1_detector_score": float(record["nodes"]["scores"][node1]) if node1 >= 0 else "",
                "layer_0_hk_valid": bool(layer0["selected_context_edge_hk_mask"][pair_idx, slot_idx]),
                "layer_0_ko_valid": bool(layer0["selected_context_edge_ko_mask"][pair_idx, slot_idx]),
                "layer_1_hk_valid": bool(layer1["selected_context_edge_hk_mask"][pair_idx, slot_idx]),
                "layer_1_ko_valid": bool(layer1["selected_context_edge_ko_mask"][pair_idx, slot_idx]),
            })

    write_csv(run_root / "tables/query_residual_summary.csv", query_rows)
    write_csv(run_root / "tables/layer_prediction_scores.csv", layer_score_rows)
    write_csv(run_root / "tables/visual_unit_updates.csv", visual_rows)
    write_csv(run_root / "tables/prototype_role_updates.csv", prototype_rows)
    write_csv(run_root / "tables/anchor_writeback.csv", anchor_rows)
    write_csv(run_root / "tables/context_transitions.csv", context_rows)

    ablation_rows: List[Dict[str, Any]] = []
    ablation_summary: List[Dict[str, Any]] = []
    proposal_manifest: List[Dict[str, Any]] = []
    retrieval_manifest: List[Dict[str, Any]] = []
    variant_diffs: Dict[str, Any] = {}
    for case in cases:
        candidate_id = str(case["candidate_id"])
        rank = int(case["mechanism_rank"])
        image_id = int(case["image_id"])
        full_record = loaded_records["full"][image_id]
        full_pair_idx = int(mappings["full"][candidate_id]["pair_idx"])
        full_logits = full_record["pairs"]["action_logits"][full_pair_idx].float()
        full_probs = full_logits.sigmoid()
        baseline_top1 = int(full_probs.argmax())
        analysis_target = int(case["top_action_id"])
        competitor = int(torch.topk(full_probs, k=2).indices[1])
        full_proposal_fp = proposal_fingerprint(full_record)
        full_retrieval_fp = retrieval_fingerprint(full_record, full_pair_idx)
        for variant in VARIANTS:
            record = loaded_records[variant][image_id]
            pair_idx = int(mappings[variant][candidate_id]["pair_idx"])
            proposal_fp = proposal_fingerprint(record)
            retrieval_fp = retrieval_fingerprint(record, pair_idx)
            proposal_equal = proposal_fp == full_proposal_fp
            retrieval_equal = retrieval_fp == full_retrieval_fp
            if not proposal_equal or not retrieval_equal:
                raise RuntimeError(
                    f"{variant}/{candidate_id}: fingerprint mismatch proposal={proposal_equal}, retrieval={retrieval_equal}"
                )
            logits = record["pairs"]["action_logits"][pair_idx].float()
            probs = logits.sigmoid()
            top1 = int(probs.argmax())
            layers = record["graph_layers"]
            setm_calls = sum(int(bool(layer.get("setm_called", False))) for layer in layers)
            proposal_manifest.append({
                "variant": variant,
                "image_id": image_id,
                "candidate_id": candidate_id,
                "proposal_fingerprint": proposal_fp,
                "matches_full": proposal_equal,
            })
            retrieval_manifest.append({
                "variant": variant,
                "image_id": image_id,
                "candidate_id": candidate_id,
                "retrieval_fingerprint": retrieval_fp,
                "matches_full": retrieval_equal,
            })
            for action_id in range(117):
                ablation_rows.append({
                    "run_id": run_id,
                    "variant": variant,
                    "seed": 3407,
                    "mechanism_rank": rank,
                    "candidate_id": candidate_id,
                    "image_id": image_id,
                    "file_name": case["file_name"],
                    "original_pair_idx": int(mappings[variant][candidate_id]["original_pair_idx"]),
                    "selected_pair_idx": pair_idx,
                    "pair_idx": pair_idx,
                    "proposal_fingerprint": proposal_fp,
                    "retrieval_fingerprint": retrieval_fp,
                    "action_id_model": action_id,
                    "action_category_id_raw": action_id + 1,
                    "action_name": action_names.get(action_id, ""),
                    "action_logit": float(logits[action_id]),
                    "action_probability": float(probs[action_id]),
                    "full_action_logit": float(full_logits[action_id]),
                    "full_action_probability": float(full_probs[action_id]),
                    "action_logit_delta_vs_full": float(logits[action_id] - full_logits[action_id]),
                    "action_probability_delta_vs_full": float(probs[action_id] - full_probs[action_id]),
                    "is_gt_action": action_id in valid_gt_actions[candidate_id],
                    "is_baseline_top1": action_id == baseline_top1,
                    "baseline_top1_action_id": baseline_top1,
                    "variant_top1_action_id": top1,
                    "variant_top1_action_name": action_names.get(top1, ""),
                    "prediction_changed": top1 != baseline_top1,
                    "analysis_target_action_id": analysis_target,
                    "analysis_target_source": "original_mechanism_rank_action",
                    "target_probability_delta_vs_full": float(probs[analysis_target] - full_probs[analysis_target]),
                    "strongest_competitor_action_id": competitor,
                    "competitor_probability_delta_vs_full": float(probs[competitor] - full_probs[competitor]),
                    "setm_called": setm_calls > 0,
                    "visual_writeback_applied": any(bool(layer.get("visual_writeback_applied", False)) for layer in layers),
                    "text_writeback_applied": any(bool(layer.get("text_writeback_applied", False)) for layer in layers),
                    "skipped": False,
                    "skipped_reason": "",
                })
            ablation_summary.append({
                "run_id": run_id,
                "variant": variant,
                "mechanism_rank": rank,
                "candidate_id": candidate_id,
                "image_id": image_id,
                "selected_pair_idx": pair_idx,
                "gt_mapping_valid": gt_mapping_valid[candidate_id],
                "baseline_top1_action_id": baseline_top1,
                "variant_top1_action_id": top1,
                "prediction_changed": top1 != baseline_top1,
                "analysis_target_action_id": analysis_target,
                "full_target_probability": float(full_probs[analysis_target]),
                "variant_target_probability": float(probs[analysis_target]),
                "target_probability_delta_vs_full": float(probs[analysis_target] - full_probs[analysis_target]),
                "proposal_fingerprint_equal": proposal_equal,
                "retrieval_fingerprint_equal": retrieval_equal,
                "setm_call_count": setm_calls,
            })

    write_csv(run_root / "tables/ablation_scores.csv", ablation_rows)
    write_csv(run_root / "tables/ablation_case_summary.csv", ablation_summary)
    write_csv(run_root / "manifests/proposal_fingerprints.csv", proposal_manifest)
    write_csv(run_root / "manifests/retrieval_fingerprints.csv", retrieval_manifest)

    variant_diffs = {
        "full": {"write_visual": True, "write_text": True, "setm": True, "setm_visual_units": [0, 1, 2, 3, 4, 5, 6]},
        "no_anchor_writeback": {"write_visual": False, "write_text": True, "setm": True, "setm_visual_units": [0, 1, 2, 3, 4, 5, 6]},
        "no_prototype_state_writeback": {"write_visual": True, "write_text": False, "setm": True, "setm_visual_units": [0, 1, 2, 3, 4, 5, 6]},
        "no_setm": {"write_visual": False, "write_text": False, "setm": False, "setm_visual_units": []},
        "endpoint_only_setm": {"write_visual": True, "write_text": True, "setm": True, "setm_visual_units": [0, 1, 2]},
    }
    write_json(run_root / "config/variant_config_diff.json", variant_diffs)

    copy_and_npz(run_root, cases, records_by_variant, mappings, args.image_root)
    shutil.copy2(run_root / "manifests/full_pair_mapping.csv", run_root / "provenance/original_to_validation_pair_mapping.csv")
    full_log = run_root / "logs/variants/full.log"
    if full_log.is_file():
        shutil.copy2(full_log, run_root / "logs/baseline.log")

    counts = {
        "top20_cases": len(cases),
        "gt_pair_mapping_rows": len(gt_rows),
        "gt_mapping_valid_candidates": sum(gt_mapping_valid.values()),
        "query_residual_rows": len(query_rows),
        "layer_prediction_rows": len(layer_score_rows),
        "visual_unit_update_rows": len(visual_rows),
        "prototype_role_update_rows": len(prototype_rows),
        "anchor_writeback_rows": len(anchor_rows),
        "context_transition_rows": len(context_rows),
        "ablation_score_rows": len(ablation_rows),
        "ablation_case_summary_rows": len(ablation_summary),
    }
    write_json(run_root / "reports/export_counts.json", counts)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
