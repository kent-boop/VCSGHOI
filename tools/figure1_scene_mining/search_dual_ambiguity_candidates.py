#!/usr/bin/env python3
"""Search HICO-DET for two-person, two-racket, pair-specific tennis evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = PROJECT_ROOT / "outputs/figure1_search_v2/index"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs/dual_ambiguity_round3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-iou", type=float, default=0.45)
    return parser.parse_args()


def loads(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []
    return []


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pair_box(row: pd.Series, prefix: str) -> list[float]:
    return [float(row[f"{prefix}_bbox_x1"]), float(row[f"{prefix}_bbox_y1"]), float(row[f"{prefix}_bbox_x2"]), float(row[f"{prefix}_bbox_y2"])]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes: list[list[float]]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def cluster_boxes(boxes: list[list[float]], threshold: float) -> tuple[list[list[float]], list[int]]:
    clusters: list[list[list[float]]] = []
    assignments: list[int] = []
    for box in boxes:
        best, best_iou = -1, 0.0
        for index, members in enumerate(clusters):
            score = max(iou(box, member) for member in members)
            if score > best_iou:
                best, best_iou = index, score
        if best >= 0 and best_iou >= threshold:
            clusters[best].append(box)
            assignments.append(best)
        else:
            clusters.append([box])
            assignments.append(len(clusters) - 1)
    return [union_box(members) for members in clusters], assignments


def instance_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "category": str(row["category_name"]),
        "bbox": [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])],
        "instance_id": str(row["canonical_instance_id"]),
    }


def main() -> int:
    args = parse_args()
    pairs = pd.read_parquet(args.index_dir / "pairs.parquet")
    instances = pd.read_parquet(args.index_dir / "instances.parquet")
    images = pd.read_parquet(args.index_dir / "image_summary.parquet").set_index("image_key", drop=False)
    rackets = pairs[(pairs["object_category_name"] == "tennis racket") & (pairs["valid"] == True) & (pairs["invisible"] == False)].copy()
    instances_by_image = {key: frame for key, frame in instances.groupby("image_key", sort=False)}

    active_actions = {"swing"}
    passive_actions = {"carry", "hold", "inspect"}
    rows: list[dict[str, Any]] = []

    for image_key, frame in rackets.groupby("image_key", sort=False):
        if len(frame) < 2 or image_key not in images.index:
            continue
        image = images.loc[image_key]
        if isinstance(image, pd.DataFrame):
            image = image.iloc[0]
        width, height = float(image["width"]), float(image["height"])
        image_area = max(1.0, width * height)

        hboxes = [pair_box(row, "human") for _, row in frame.iterrows()]
        rboxes = [pair_box(row, "object") for _, row in frame.iterrows()]
        hclusters, hassign = cluster_boxes(hboxes, args.cluster_iou)
        rclusters, rassign = cluster_boxes(rboxes, args.cluster_iou)
        if len(hclusters) < 2 or len(rclusters) < 2:
            continue

        pair_records: list[dict[str, Any]] = []
        for position, (_, row) in enumerate(frame.iterrows()):
            pair_records.append({
                "candidate_id": str(row["pair_id"]),
                "human_cluster": hassign[position],
                "racket_cluster": rassign[position],
                "human_bbox": hboxes[position],
                "racket_bbox": rboxes[position],
                "actions": loads(row["verb_names_json"]),
            })

        active = [p for p in pair_records if active_actions.intersection(p["actions"])]
        passive = [p for p in pair_records if passive_actions.intersection(p["actions"]) and not active_actions.intersection(p["actions"])]
        combinations = [(a, b) for a in active for b in passive if a["human_cluster"] != b["human_cluster"] and a["racket_cluster"] != b["racket_cluster"]]
        if not combinations:
            continue

        image_instances = instances_by_image.get(image_key, pd.DataFrame())
        payloads = [instance_payload(row) for _, row in image_instances.iterrows()]
        ball_boxes = [item["bbox"] for item in payloads if item["category"] == "sports ball"]
        non_target_context = [item for item in payloads if item["category"] not in {"person", "tennis racket", "sports ball"}]

        best: dict[str, Any] | None = None
        for active_pair, passive_pair in combinations:
            ah, ph = active_pair["human_bbox"], passive_pair["human_bbox"]
            ar, pr = active_pair["racket_bbox"], passive_pair["racket_bbox"]
            ah_ratio, ph_ratio = area(ah) / image_area, area(ph) / image_area
            ar_ratio, pr_ratio = area(ar) / image_area, area(pr) / image_area
            min_h_height = min((ah[3] - ah[1]) / height, (ph[3] - ph[1]) / height)
            human_balance = min(area(ah), area(ph)) / max(1.0, max(area(ah), area(ph)))
            person_separation = math.dist(center(ah), center(ph)) / max(1.0, math.hypot(width, height))
            racket_separation = math.dist(center(ar), center(pr)) / max(1.0, math.hypot(width, height))
            ball_size = max((area(box) / image_area for box in ball_boxes), default=0.0)
            ball_to_active = min((math.dist(center(box), center(ar)) for box in ball_boxes), default=1e9) / max(1.0, math.hypot(width, height))
            ball_to_passive = min((math.dist(center(box), center(pr)) for box in ball_boxes), default=1e9) / max(1.0, math.hypot(width, height))
            ball_assignment_margin = (ball_to_passive - ball_to_active) if ball_boxes else -0.25

            score = (
                20.0
                + 16 * min(min_h_height / 0.28, 1.0)
                + 10 * human_balance
                + 8 * min(person_separation / 0.28, 1.0)
                + 8 * min(racket_separation / 0.28, 1.0)
                + 8 * min(min(ar_ratio, pr_ratio) / 0.018, 1.0)
                + (13 if ball_boxes else 0)
                + 8 * min(ball_size / 0.002, 1.0)
                + 10 * max(-0.5, min(1.0, ball_assignment_margin / 0.20))
                + 2 * min(len(non_target_context), 3)
                + 2 * (width >= height)
            )
            record = {
                "dual_score": round(score, 4),
                "image_key": image_key,
                "image_id": str(image["image_id"]),
                "split": str(image["split"]),
                "file_name": str(image["file_name"]),
                "image_path": str(image["image_path"]),
                "image_width": int(image["width"]),
                "image_height": int(image["height"]),
                "active_candidate_id": active_pair["candidate_id"],
                "passive_candidate_id": passive_pair["candidate_id"],
                "active_human_bbox": dumps(ah),
                "active_racket_bbox": dumps(ar),
                "active_actions": dumps(active_pair["actions"]),
                "passive_human_bbox": dumps(ph),
                "passive_racket_bbox": dumps(pr),
                "passive_actions": dumps(passive_pair["actions"]),
                "sports_ball_bboxes": dumps(ball_boxes),
                "all_context_instances": dumps(payloads),
                "human_size_balance": round(human_balance, 5),
                "min_human_height_ratio": round(min_h_height, 5),
                "person_separation": round(person_separation, 5),
                "racket_separation": round(racket_separation, 5),
                "ball_assignment_margin": round(ball_assignment_margin, 5),
                "ball_to_active_racket": round(ball_to_active, 5),
                "ball_to_passive_racket": round(ball_to_passive, 5),
                "annotated_ball_count": len(ball_boxes),
                "non_target_context_count": len(non_target_context),
                "candidate_story": "active swing pair versus passive hold/carry pair in the same scene",
            }
            if best is None or record["dual_score"] > best["dual_score"]:
                best = record
        if best:
            rows.append(best)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["dual_score", "file_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    result.to_csv(output / "tennis_dual_candidates.csv", index=False)
    (output / "search_summary.json").write_text(json.dumps({
        "candidate_images": len(result),
        "with_annotated_ball": int((result["annotated_ball_count"] > 0).sum()) if not result.empty else 0,
        "criteria": "two clustered humans, two clustered rackets, active swing pair, distinct passive carry/hold/inspect pair",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.head(50)[["rank", "dual_score", "file_name", "active_actions", "passive_actions", "annotated_ball_count", "min_human_height_ratio", "human_size_balance"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
