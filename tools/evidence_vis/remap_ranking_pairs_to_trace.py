#!/usr/bin/env python3
"""Map selected pairs from a source ranking to a new trace by H/O box IoU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch


def box_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    x1 = torch.maximum(a[0], b[0])
    y1 = torch.maximum(a[1], b[1])
    x2 = torch.minimum(a[2], b[2])
    y2 = torch.minimum(a[3], b[3])
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[2] - a[0]).clamp(min=0) * (a[3] - a[1]).clamp(min=0)
    area_b = (b[2] - b[0]).clamp(min=0) * (b[3] - b[1]).clamp(min=0)
    return float((intersection / (area_a + area_b - intersection).clamp(min=1e-12)).item())


def trace_map(trace_dir: Path) -> Dict[int, Path]:
    output: Dict[int, Path] = {}
    for line in (trace_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = row["image_id"]
        if isinstance(image_id, list):
            image_id = image_id[0]
        path = Path(row["record_path"])
        output[int(image_id)] = path if path.is_absolute() else trace_dir / path
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-csv", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--warning-iou", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.ranking_csv.open("r", encoding="utf-8")))
    records = trace_map(args.trace_dir)
    mapped_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for row in rows:
        image_id = int(row["image_id"])
        old_record = torch.load(row["record_path"], map_location="cpu")
        new_path = records[image_id]
        new_record = torch.load(new_path, map_location="cpu")

        old_pair_idx = int(row["pair_idx"])
        old_h, old_o = [int(value) for value in old_record["pairs"]["indices"][old_pair_idx].tolist()]
        old_boxes = old_record["nodes"]["boxes_xyxy_norm"].float()
        old_labels = old_record["nodes"]["labels"].long()
        new_boxes = new_record["nodes"]["boxes_xyxy_norm"].float()
        new_labels = new_record["nodes"]["labels"].long()

        candidates: List[Tuple[int, float, float, int, int, int, float, float]] = []
        for new_pair_idx, pair in enumerate(new_record["pairs"]["indices"].tolist()):
            new_h, new_o = [int(value) for value in pair]
            human_iou = box_iou(old_boxes[old_h], new_boxes[new_h])
            object_iou = box_iou(old_boxes[old_o], new_boxes[new_o])
            labels_match = int(
                int(old_labels[old_h]) == int(new_labels[new_h])
                and int(old_labels[old_o]) == int(new_labels[new_o])
            )
            candidates.append((
                labels_match,
                min(human_iou, object_iou),
                0.5 * (human_iou + object_iou),
                new_pair_idx,
                new_h,
                new_o,
                human_iou,
                object_iou,
            ))
        best = max(candidates)
        _, min_iou, mean_iou, new_pair_idx, new_h, new_o, human_iou, object_iou = best

        old_probs = old_record["pairs"]["action_probs"][old_pair_idx].float()
        new_probs = new_record["pairs"]["action_probs"][new_pair_idx].float()
        old_top_action = int(old_probs.argmax().item())
        new_top_action = int(new_probs.argmax().item())

        mapped = dict(row)
        mapped.update({
            "record_path": str(new_path),
            "original_record_path": row["record_path"],
            "original_pair_idx": old_pair_idx,
            "pair_idx": new_pair_idx,
            "original_human_node_idx": old_h,
            "original_object_node_idx": old_o,
            "human_node_idx": new_h,
            "object_node_idx": new_o,
            "pair_mapping_human_iou": human_iou,
            "pair_mapping_object_iou": object_iou,
            "pair_mapping_min_iou": min_iou,
            "pair_mapping_mean_iou": mean_iou,
            "pair_mapping_warning": min_iou < args.warning_iou,
            "original_top_action_id": old_top_action,
            "detailed_top_action_id": new_top_action,
            "top_action_stable": old_top_action == new_top_action,
            "original_top_action_probability": float(old_probs[old_top_action].item()),
            "detailed_same_action_probability": float(new_probs[old_top_action].item()),
        })
        mapped_rows.append(mapped)
        report_rows.append({
            "mechanism_rank": int(row.get("mechanism_rank", len(report_rows) + 1)),
            "image_id": image_id,
            "file_name": row.get("file_name"),
            "original_pair_idx": old_pair_idx,
            "detailed_pair_idx": new_pair_idx,
            "human_iou": human_iou,
            "object_iou": object_iou,
            "min_iou": min_iou,
            "warning": min_iou < args.warning_iou,
            "original_top_action_id": old_top_action,
            "detailed_top_action_id": new_top_action,
            "top_action_stable": old_top_action == new_top_action,
            "same_action_probability_delta": float(new_probs[old_top_action].item() - old_probs[old_top_action].item()),
        })

    fieldnames: List[str] = []
    for row in mapped_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapped_rows)

    warnings = [row for row in report_rows if row["warning"]]
    stable_count = sum(bool(row["top_action_stable"]) for row in report_rows)
    report = {
        "image_count": len(report_rows),
        "warning_iou": args.warning_iou,
        "mapping_warning_count": len(warnings),
        "top_action_stable_count": stable_count,
        "min_human_iou": min(row["human_iou"] for row in report_rows),
        "min_object_iou": min(row["object_iou"] for row in report_rows),
        "mean_abs_same_action_probability_delta": sum(abs(row["same_action_probability_delta"]) for row in report_rows) / len(report_rows),
        "max_abs_same_action_probability_delta": max(abs(row["same_action_probability_delta"]) for row in report_rows),
        "records": report_rows,
    }
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    print(f"Wrote remapped ranking: {args.output_csv}")


if __name__ == "__main__":
    main()
