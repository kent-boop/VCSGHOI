#!/usr/bin/env python3
"""Strict raw-trace validation for one EvidenceHOI mechanism variant."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch


VARIANTS = {
    "full",
    "no_anchor_writeback",
    "no_prototype_state_writeback",
    "no_setm",
    "endpoint_only_setm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--expected-records", type=int, default=20)
    parser.add_argument("--expected-pairs", type=int, default=32)
    parser.add_argument("--expected-layers", type=int, default=2)
    parser.add_argument("--expected-m", type=int, default=4)
    parser.add_argument("--expected-k", type=int, default=5)
    parser.add_argument("--expected-roles", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def iter_tensors(value: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_tensors(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            yield from iter_tensors(item, child)


def load_manifest(trace_dir: Path) -> List[Dict[str, Any]]:
    path = trace_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_record(trace_dir: Path, row: Mapping[str, Any]) -> Path:
    path = Path(str(row["record_path"]))
    return path if path.is_absolute() else trace_dir / path


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return math.inf
    if left.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def validate_record(
    record: Mapping[str, Any],
    variant: str,
    expected_pairs: int,
    expected_layers: int,
    expected_m: int,
    expected_k: int,
    expected_roles: int,
    tolerance: float,
) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    image_id = record.get("image", {}).get("image_id")
    pairs = record.get("pairs", {})
    mechanism = record.get("mechanism")
    layers = record.get("graph_layers", [])

    pair_indices = pairs.get("indices")
    if not isinstance(pair_indices, torch.Tensor) or pair_indices.shape != (expected_pairs, 2):
        errors.append(f"pair indices shape: {None if not isinstance(pair_indices, torch.Tensor) else tuple(pair_indices.shape)}")
    if len(layers) != expected_layers:
        errors.append(f"layer count: {len(layers)}")
    if not isinstance(mechanism, Mapping):
        errors.append("mechanism payload missing")
        mechanism = {}
    if mechanism.get("variant") != variant:
        errors.append(f"mechanism variant mismatch: {mechanism.get('variant')!r}")

    nonfinite = []
    for name, tensor in iter_tensors(record):
        if torch.is_floating_point(tensor) and tensor.numel() and not torch.isfinite(tensor).all().item():
            nonfinite.append(name)
    if nonfinite:
        errors.append("non-finite tensors: " + ", ".join(nonfinite[:20]))

    setm_calls = 0
    retrieval_reference = None
    layer_feature_tensors = []
    for layer_index, layer in enumerate(layers):
        if int(layer.get("layer_idx", -1)) != layer_index:
            errors.append(f"layer index mismatch at {layer_index}")
        evidence = layer.get("candidate_evidence_tokens")
        prototypes = layer.get("prototype_slots_pre_sync")
        if not isinstance(evidence, torch.Tensor) or evidence.shape != (expected_pairs, 3 + expected_m, 256):
            errors.append(f"layer {layer_index} evidence shape: {getattr(evidence, 'shape', None)}")
        if not isinstance(prototypes, torch.Tensor):
            errors.append(f"layer {layer_index} prototype slots missing")
        else:
            if prototypes.shape[0] != expected_pairs or prototypes.shape[1] != expected_k:
                errors.append(f"layer {layer_index} prototype K shape: {tuple(prototypes.shape)}")
            if prototypes.shape[-2] != expected_roles or prototypes.shape[-1] != 256:
                errors.append(f"layer {layer_index} prototype role shape: {tuple(prototypes.shape)}")

        retrieval = layer.get("selected_hoi_indices")
        if isinstance(retrieval, torch.Tensor):
            if retrieval_reference is None:
                retrieval_reference = retrieval
            elif not torch.equal(retrieval_reference, retrieval):
                errors.append(f"retrieval IDs changed at layer {layer_index}")
        else:
            errors.append(f"layer {layer_index} retrieval IDs missing")

        setm_called = bool(layer.get("setm_called", False))
        setm_calls += int(setm_called)
        applied = layer.get("sync_anchor_delta_applied")
        if not isinstance(applied, torch.Tensor):
            errors.append(f"layer {layer_index} applied anchor delta missing")
        feature = layer.get("graph_pair_features_after_layer")
        if isinstance(feature, torch.Tensor):
            layer_feature_tensors.append(feature)
        else:
            errors.append(f"layer {layer_index} pair feature missing")

        if variant == "full":
            if not setm_called or not layer.get("visual_writeback_applied") or not layer.get("text_writeback_applied"):
                errors.append(f"full write-back flags invalid at layer {layer_index}")
        elif variant == "no_anchor_writeback":
            if not setm_called or layer.get("visual_writeback_applied") or not layer.get("text_writeback_applied"):
                errors.append(f"no_anchor flags invalid at layer {layer_index}")
            if isinstance(applied, torch.Tensor) and torch.count_nonzero(applied).item() != 0:
                errors.append(f"no_anchor applied delta nonzero at layer {layer_index}")
        elif variant == "no_prototype_state_writeback":
            if not setm_called or not layer.get("visual_writeback_applied") or layer.get("text_writeback_applied"):
                errors.append(f"no_prototype flags invalid at layer {layer_index}")
            pre = layer.get("prototype_slots_pre_sync")
            post = layer.get("prototype_slots_post_writeback_actual")
            if isinstance(pre, torch.Tensor) and isinstance(post, torch.Tensor) and not torch.equal(pre, post):
                errors.append(f"no_prototype actual text state changed at layer {layer_index}")
        elif variant == "no_setm":
            if setm_called or layer.get("visual_writeback_applied") or layer.get("text_writeback_applied"):
                errors.append(f"no_setm flags invalid at layer {layer_index}")
            if "sync_match_logits" in layer:
                errors.append(f"no_setm has match logits at layer {layer_index}")
            if isinstance(applied, torch.Tensor) and torch.count_nonzero(applied).item() != 0:
                errors.append(f"no_setm applied delta nonzero at layer {layer_index}")
        elif variant == "endpoint_only_setm":
            mask = layer.get("sync_visual_mask")
            if not setm_called:
                errors.append(f"endpoint_only did not call SETM at layer {layer_index}")
            if not isinstance(mask, torch.Tensor) or torch.count_nonzero(mask[..., 3:]).item() != 0:
                errors.append(f"endpoint_only context mask invalid at layer {layer_index}")
            if layer.get("requested_masked_visual_indices") != [3, 4, 5, 6]:
                errors.append(f"endpoint_only requested indices invalid at layer {layer_index}")

    expected_setm_calls = 0 if variant == "no_setm" else expected_layers
    if setm_calls != expected_setm_calls:
        errors.append(f"SETM call count {setm_calls}, expected {expected_setm_calls}")

    graph_layers = mechanism.get("graph_pair_features_after_layers")
    graph_final = mechanism.get("graph_pair_features_final")
    if isinstance(graph_layers, torch.Tensor):
        if tuple(graph_layers.shape) != (expected_layers, expected_pairs, 256):
            errors.append(f"mechanism layer features shape: {tuple(graph_layers.shape)}")
        if isinstance(graph_final, torch.Tensor):
            diff = max_abs(graph_layers[-1], graph_final)
            if diff > tolerance:
                errors.append(f"layer1/final graph pair diff {diff}")
    else:
        errors.append("mechanism layer feature stack missing")

    residual_error = float(mechanism.get("residual_max_abs_error", math.inf))
    if residual_error > tolerance:
        errors.append(f"residual error {residual_error}")
    normal_logits = mechanism.get("normal_final_action_logits")
    pair_logits = pairs.get("action_logits")
    if isinstance(normal_logits, torch.Tensor) and isinstance(pair_logits, torch.Tensor):
        diff = max_abs(normal_logits, pair_logits)
        if diff > tolerance:
            errors.append(f"normal mechanism/pair logits diff {diff}")

    layer1_diff = None
    if variant == "full":
        probe_logits = mechanism.get("probe_action_logits")
        if not isinstance(probe_logits, torch.Tensor) or tuple(probe_logits.shape) != (4, expected_pairs, 117):
            errors.append(f"full probe logits shape: {getattr(probe_logits, 'shape', None)}")
        stage_names = mechanism.get("probe_stage_names")
        if stage_names != [
            "base_query_decoder",
            "after_layer_0_decoder",
            "after_layer_1_decoder",
            "normal_final_decoder",
        ]:
            errors.append(f"probe stage names: {stage_names!r}")
        layer1_diff = float(mechanism.get("after_layer_1_vs_normal_final_max_abs_diff", math.inf))
        if layer1_diff > tolerance:
            errors.append(f"layer1 probe/final diff {layer1_diff}")

    return {
        "image_id": image_id,
        "pair_count": 0 if not isinstance(pair_indices, torch.Tensor) else int(pair_indices.shape[0]),
        "layer_count": len(layers),
        "setm_call_count": setm_calls,
        "residual_max_abs_error": residual_error,
        "layer1_vs_final_max_abs_diff": layer1_diff,
        "passed": not errors,
        "errors": errors,
    }, errors


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir.resolve()
    manifest = load_manifest(trace_dir)
    global_errors: List[str] = []
    if len(manifest) != args.expected_records:
        global_errors.append(f"record count {len(manifest)}, expected {args.expected_records}")
    image_ids = [int(row["image_id"]) for row in manifest]
    if len(set(image_ids)) != len(image_ids):
        global_errors.append("duplicate image IDs in merged manifest")

    rows = []
    for manifest_row in manifest:
        record_path = resolve_record(trace_dir, manifest_row)
        if not record_path.is_file():
            global_errors.append(f"missing record: {record_path}")
            continue
        record = torch.load(record_path, map_location="cpu")
        summary, errors = validate_record(
            record,
            variant=args.variant,
            expected_pairs=args.expected_pairs,
            expected_layers=args.expected_layers,
            expected_m=args.expected_m,
            expected_k=args.expected_k,
            expected_roles=args.expected_roles,
            tolerance=args.tolerance,
        )
        summary["record_path"] = str(record_path)
        rows.append(summary)
        global_errors.extend(f"image {summary['image_id']}: {error}" for error in errors)

    report = {
        "trace_dir": str(trace_dir),
        "variant": args.variant,
        "expected_records": args.expected_records,
        "record_count": len(manifest),
        "passed": not global_errors,
        "max_residual_error": max((row["residual_max_abs_error"] for row in rows), default=math.inf),
        "max_layer1_vs_final_diff": max(
            (row["layer1_vs_final_max_abs_diff"] for row in rows if row["layer1_vs_final_max_abs_diff"] is not None),
            default=None,
        ),
        "errors": global_errors,
        "records": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, args.output_json)
    print(json.dumps({key: report[key] for key in ("variant", "record_count", "passed", "max_residual_error", "max_layer1_vs_final_diff")}, indent=2))
    if global_errors:
        for error in global_errors[:50]:
            print("ERROR:", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
