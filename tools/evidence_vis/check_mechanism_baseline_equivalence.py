#!/usr/bin/env python3
"""Compare a new full mechanism trace with the pre-modification Top-20 reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-trace-dir", type=Path, required=True)
    parser.add_argument("--old-trace-dir", type=Path, required=True)
    parser.add_argument("--new-mapping-csv", type=Path, required=True)
    parser.add_argument("--old-mapping-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def record_map(trace_dir: Path) -> Dict[int, Path]:
    rows = [
        json.loads(line)
        for line in (trace_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = {}
    for row in rows:
        path = Path(row["record_path"])
        out[int(row["image_id"])] = path if path.is_absolute() else trace_dir / path
    return out


def mapping_rows(path: Path) -> Dict[int, Mapping[str, str]]:
    return {int(row["image_id"]): row for row in csv.DictReader(path.open("r", encoding="utf-8"))}


def tensor_bytes(tensor: torch.Tensor) -> bytes:
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
        tensor = record[section][key]
        digest.update(section.encode())
        digest.update(key.encode())
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return math.inf
    return float((left.float() - right.float()).abs().max().item()) if left.numel() else 0.0


def main() -> None:
    args = parse_args()
    new_records = record_map(args.new_trace_dir)
    old_records = record_map(args.old_trace_dir)
    new_mapping = mapping_rows(args.new_mapping_csv)
    old_mapping = mapping_rows(args.old_mapping_csv)
    image_ids = sorted(new_mapping)
    errors = []
    rows = []
    for image_id in image_ids:
        new_record = torch.load(new_records[image_id], map_location="cpu")
        old_record = torch.load(old_records[image_id], map_location="cpu")
        new_fp = proposal_fingerprint(new_record)
        old_fp = proposal_fingerprint(old_record)
        fingerprint_equal = new_fp == old_fp
        new_pair_idx = int(new_mapping[image_id]["pair_idx"])
        old_pair_idx = int(old_mapping[image_id]["pair_idx"])
        new_logits = new_record["pairs"]["action_logits"][new_pair_idx]
        old_logits = old_record["pairs"]["action_logits"][old_pair_idx]
        target_diff = max_abs(new_logits, old_logits)
        new_top1 = int(new_logits.argmax().item())
        old_top1 = int(old_logits.argmax().item())
        entire_pair_logits_diff = (
            max_abs(new_record["pairs"]["action_logits"], old_record["pairs"]["action_logits"])
            if fingerprint_equal else None
        )
        passed = (
            fingerprint_equal
            and target_diff <= args.tolerance
            and entire_pair_logits_diff is not None
            and entire_pair_logits_diff <= args.tolerance
            and new_top1 == old_top1
        )
        if not passed:
            errors.append(
                f"image {image_id}: fingerprint_equal={fingerprint_equal}, "
                f"target_diff={target_diff}, all_diff={entire_pair_logits_diff}, top1={old_top1}->{new_top1}"
            )
        rows.append(
            {
                "image_id": image_id,
                "new_pair_idx": new_pair_idx,
                "old_pair_idx": old_pair_idx,
                "new_proposal_fingerprint": new_fp,
                "old_proposal_fingerprint": old_fp,
                "proposal_fingerprint_equal": fingerprint_equal,
                "target_logits_max_abs_diff": target_diff,
                "all_pair_logits_max_abs_diff": entire_pair_logits_diff,
                "old_top1": old_top1,
                "new_top1": new_top1,
                "passed": passed,
            }
        )

    report = {
        "new_trace_dir": str(args.new_trace_dir.resolve()),
        "old_trace_dir": str(args.old_trace_dir.resolve()),
        "tolerance": args.tolerance,
        "candidate_count": len(rows),
        "proposal_fingerprint_equal_count": sum(row["proposal_fingerprint_equal"] for row in rows),
        "max_target_logits_abs_diff": max((row["target_logits_max_abs_diff"] for row in rows), default=math.inf),
        "max_all_pair_logits_abs_diff": max(
            (row["all_pair_logits_max_abs_diff"] for row in rows if row["all_pair_logits_max_abs_diff"] is not None),
            default=None,
        ),
        "top1_equal_count": sum(row["old_top1"] == row["new_top1"] for row in rows),
        "passed": not errors and len(rows) == 20,
        "errors": errors,
        "records": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, args.output_json)
    print(json.dumps({key: report[key] for key in (
        "candidate_count",
        "proposal_fingerprint_equal_count",
        "max_target_logits_abs_diff",
        "max_all_pair_logits_abs_diff",
        "top1_equal_count",
        "passed",
    )}, indent=2))
    if not report["passed"]:
        for error in errors:
            print("ERROR:", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
