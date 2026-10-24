#!/usr/bin/env python3
"""Validate an EvidenceHOI trace directory without loading the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch


REQUIRED_GRAPH_FIELDS = (
    "selected_context_indices",
    "selected_context_mask",
    "selected_context_two_hop_mask",
    "selected_hoi_action_ids",
)

SETM_GRAPH_FIELDS = (
    "sync_text_to_visual_attn",
    "sync_evidence_weights",
)


def load_record(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def validate_record(path: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    record = load_record(path)
    if not isinstance(record, Mapping):
        return False, [f"{path}: record is not a mapping"]

    for key in ("image", "nodes", "pairs", "metadata"):
        if key not in record:
            errors.append(f"{path}: missing top-level field '{key}'")

    graph = record.get("graph")
    graph_available = isinstance(graph, Mapping)
    if graph_available:
        variant = str(graph.get("mechanism_validation_variant", "full"))
        required_graph_fields = REQUIRED_GRAPH_FIELDS
        if variant != "no_setm":
            required_graph_fields += SETM_GRAPH_FIELDS
        for key in required_graph_fields:
            if key not in graph:
                errors.append(f"{path}: graph missing '{key}'")

    metadata_available = bool(record.get("metadata", {}).get("graph_trace_available", False))
    if graph_available != metadata_available:
        errors.append(
            f"{path}: graph presence ({graph_available}) disagrees with "
            f"metadata.graph_trace_available ({metadata_available})"
        )

    for tensor in iter_tensors(record):
        if tensor.device.type != "cpu":
            errors.append(f"{path}: trace tensor remained on {tensor.device}")
            break
        if tensor.requires_grad:
            errors.append(f"{path}: trace tensor still requires gradients")
            break

    pair_indices = record.get("pairs", {}).get("indices")
    if not isinstance(pair_indices, torch.Tensor) or pair_indices.ndim != 2 or pair_indices.shape[-1] != 2:
        errors.append(f"{path}: pairs.indices must have shape [K, 2]")

    return graph_available, errors


def validate_trace_directory(
    trace_dir: Path,
    *,
    expected_records: int | None = None,
    require_graph: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    records_dir = trace_dir / "records"
    record_paths = sorted(records_dir.glob("*.pt")) if records_dir.is_dir() else []
    temp_paths = sorted(trace_dir.rglob("*.tmp")) if trace_dir.exists() else []
    errors: List[str] = []

    if expected_records is not None and len(record_paths) != expected_records:
        errors.append(
            f"expected {expected_records} records, found {len(record_paths)}"
        )
    if not record_paths:
        errors.append("no trace records found")

    graph_count = 0
    for path in record_paths:
        graph_available, record_errors = validate_record(path)
        graph_count += int(graph_available)
        errors.extend(record_errors)

    manifest_path = trace_dir / "manifest.jsonl"
    manifest_rows = 0
    if manifest_path.is_file():
        for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{manifest_path}:{line_no}: invalid JSON: {exc}")
            manifest_rows += 1
    else:
        errors.append("manifest.jsonl is missing")

    if manifest_rows != len(record_paths):
        errors.append(
            f"manifest has {manifest_rows} rows, but records/ contains {len(record_paths)} files"
        )
    if require_graph and graph_count != len(record_paths):
        errors.append(
            f"graph trace is available for {graph_count}/{len(record_paths)} records"
        )
    if temp_paths:
        errors.append(f"found {len(temp_paths)} unfinished temporary files")

    summary = {
        "trace_dir": str(trace_dir),
        "record_count": len(record_paths),
        "manifest_rows": manifest_rows,
        "graph_record_count": graph_count,
        "graph_available_for_all": bool(record_paths) and graph_count == len(record_paths),
        "temporary_file_count": len(temp_paths),
        "passed": not errors,
        "errors": errors,
    }
    return summary, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--require-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, errors = validate_trace_directory(
        args.trace_dir,
        expected_records=args.expected_records,
        require_graph=args.require_graph,
    )
    summary_path = args.trace_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"record count: {summary['record_count']}")
    print(
        "graph available: "
        f"{summary['graph_available_for_all']} "
        f"({summary['graph_record_count']}/{summary['record_count']})"
    )
    print(f"manifest rows: {summary['manifest_rows']}")
    if errors:
        for error in errors:
            print(f"[Trace validation error] {error}")
        raise SystemExit(1)
    print("TRACE VALIDATION PASSED")


if __name__ == "__main__":
    main()
