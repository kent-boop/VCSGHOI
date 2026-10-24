#!/usr/bin/env python3
"""Generate high-recall pair-level Figure 1 candidates from the v2 HICO index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v2"
INDEX_ROOT = OUTPUT_ROOT / "index"
CANDIDATE_ROOT = OUTPUT_ROOT / "candidates"

PRIORITY_OBJECTS: dict[int, list[str]] = {
    1: [
        "dining table", "couch", "laptop", "suitcase", "bicycle",
        "cake", "pizza", "chair",
    ],
    2: [
        "book", "bottle", "cup", "backpack", "handbag", "skateboard",
        "surfboard", "skis", "snowboard", "motorcycle", "horse",
    ],
    3: ["cell phone", "sports ball", "knife", "fork", "spoon", "remote"],
}

PRIORITY_BY_OBJECT = {
    name: priority
    for priority, names in PRIORITY_OBJECTS.items()
    for name in names
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--min-object-area-ratio", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_loads(value: str) -> Any:
    return json.loads(value) if isinstance(value, str) and value else []


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def area(row: pd.Series, prefix: str) -> float:
    return max(0.0, float(row[f"{prefix}_bbox_x2"]) - float(row[f"{prefix}_bbox_x1"])) * max(
        0.0, float(row[f"{prefix}_bbox_y2"]) - float(row[f"{prefix}_bbox_y1"])
    )


def bbox_valid(row: pd.Series) -> bool:
    return (
        float(row["human_bbox_x1"]) < float(row["human_bbox_x2"])
        and float(row["human_bbox_y1"]) < float(row["human_bbox_y2"])
        and float(row["object_bbox_x1"]) < float(row["object_bbox_x2"])
        and float(row["object_bbox_y1"]) < float(row["object_bbox_y2"])
    )


def candidate_tier(row: pd.Series, has_action_ambiguity: bool, has_auxiliary: bool, rich_entities: bool) -> str | None:
    if bool(row["shared_object_status"]) and len(json_loads(row["shared_object_verbsets_json"])) >= 2:
        return "Gold"
    if bool(row["same_class_multi_instance_status"]) and len(json_loads(row["same_class_verbsets_json"])) >= 2:
        return "Silver"
    if has_action_ambiguity and (has_auxiliary or rich_entities):
        return "Bronze"
    return None


def rejection_reason(row: pd.Series, image_area: float, min_object_area_ratio: float, legacy_control: bool) -> str | None:
    if row["object_category_name"] not in PRIORITY_BY_OBJECT and not legacy_control:
        return "object_not_in_search_priority_groups"
    if not bool(row["valid"]) or bool(row["invisible"]):
        return "invalid_or_invisible_pair"
    if not bbox_valid(row):
        return "invalid_bbox"
    if int(row["raw_relation_count"]) <= 0 or not json_loads(row["raw_annotation_provenance_json"]):
        return "missing_gt_provenance"
    if "no_interaction" in json_loads(row["verb_names_json"]):
        return "target_pair_has_no_interaction_gt"
    if image_area <= 0:
        return "invalid_image_area"
    if area(row, "object") / image_area < min_object_area_ratio and not legacy_control:
        return "target_object_extremely_small"
    return None


def build_candidates(index_dir: Path, output_dir: Path, min_object_area_ratio: float, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(index_dir / "pairs.parquet")
    images = pd.read_parquet(index_dir / "image_summary.parquet")
    image_by_key = images.set_index("image_key")
    all_object_names = sorted(pairs["object_category_name"].unique())

    candidate_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    image_candidate_accumulator: dict[str, dict[str, Any]] = {}
    object_split_tier_counter: Counter[tuple[str, str, str]] = Counter()
    object_priority_counter: Counter[tuple[int, str, str, str]] = Counter()
    rejection_counter: Counter[tuple[str, str]] = Counter()

    for row in pairs.itertuples(index=False):
        pair = pd.Series(row._asdict())
        image = image_by_key.loc[pair["image_key"]]
        image_area = float(image["width"]) * float(image["height"])
        object_name = str(pair["object_category_name"])
        legacy_control = not bool(pair["eligible_for_final"])
        reason = rejection_reason(pair, image_area, min_object_area_ratio, legacy_control)
        priority = PRIORITY_BY_OBJECT.get(object_name, 0 if legacy_control else None)
        base_rejection = {
            "pair_id": pair["pair_id"],
            "image_key": pair["image_key"],
            "split": pair["split"],
            "file_name": pair["file_name"],
            "object_category_id": int(pair["object_category_id"]),
            "object_category_name": object_name,
            "object_priority": priority if priority is not None else 0,
            "pair_structural_tier": pair["pair_structural_tier"],
            "eligible_for_final": bool(pair["eligible_for_final"]),
            "legacy_negative_example": bool(pair["legacy_negative_example"]),
        }
        if reason is not None:
            rejection_counter[(object_name, reason)] += 1
            rejection_rows.append({**base_rejection, "rejection_reason": reason})
            continue

        verb_ids = json_loads(pair["verb_ids_json"])
        verb_names = json_loads(pair["verb_names_json"])
        hoi_ids = json_loads(pair["hoi_ids_json"])
        other_gt_hoi = json_loads(pair["target_human_other_gt_hoi_json"])
        shared_verbsets = json_loads(pair["shared_object_verbsets_json"])
        same_class_verbsets = json_loads(pair["same_class_verbsets_json"])
        has_action_ambiguity = len(verb_ids) >= 2 or len(shared_verbsets) >= 2 or len(same_class_verbsets) >= 2
        has_auxiliary = len(other_gt_hoi) > 0
        rich_entities = int(image["canonical_instance_count"]) >= 6 or int(image["valid_pair_count"]) >= 3
        tier = candidate_tier(pair, has_action_ambiguity, has_auxiliary, rich_entities)
        if tier is None and legacy_control:
            tier = str(pair["pair_structural_tier"])
        if tier is None:
            rejection_counter[(object_name, "bronze_requirements_not_met")] += 1
            rejection_rows.append({**base_rejection, "rejection_reason": "bronze_requirements_not_met"})
            continue

        final_pool = bool(pair["eligible_for_final"])
        candidate_layer = "legacy_control" if not final_pool else tier
        object_area_ratio = area(pair, "object") / image_area
        human_area_ratio = area(pair, "human") / image_area
        possible_context_count = max(0, int(image["canonical_object_count"]) - 1)
        noise_candidate_count = possible_context_count
        context_priority = "other_gt_hoi_for_target_human" if has_auxiliary else "possible_context_only"
        candidate = {
            "candidate_id": pair["pair_id"],
            "pair_id": pair["pair_id"],
            "image_key": pair["image_key"],
            "image_id": pair["image_id"],
            "split": pair["split"],
            "file_name": pair["file_name"],
            "eligible_candidate": final_pool,
            "candidate_layer": candidate_layer,
            "legacy_control": not final_pool,
            "legacy_negative_example": bool(pair["legacy_negative_example"]),
            "target_tier": tier,
            "object_priority": int(priority),
            "object_category_id": int(pair["object_category_id"]),
            "object_category_name": object_name,
            "canonical_human_id": pair["canonical_human_id"],
            "canonical_object_id": pair["canonical_object_id"],
            "human_bbox_x1": float(pair["human_bbox_x1"]),
            "human_bbox_y1": float(pair["human_bbox_y1"]),
            "human_bbox_x2": float(pair["human_bbox_x2"]),
            "human_bbox_y2": float(pair["human_bbox_y2"]),
            "object_bbox_x1": float(pair["object_bbox_x1"]),
            "object_bbox_y1": float(pair["object_bbox_y1"]),
            "object_bbox_x2": float(pair["object_bbox_x2"]),
            "object_bbox_y2": float(pair["object_bbox_y2"]),
            "human_area_ratio": human_area_ratio,
            "object_area_ratio": object_area_ratio,
            "gt_verb_ids_json": json_dumps(verb_ids),
            "gt_verb_names_json": json_dumps(verb_names),
            "gt_hoi_ids_json": json_dumps(hoi_ids),
            "target_action_family_hit": True,
            "target_action_family": "non_no_interaction_gt",
            "raw_relation_count": int(pair["raw_relation_count"]),
            "raw_annotation_provenance_json": pair["raw_annotation_provenance_json"],
            "shared_object_status": bool(pair["shared_object_status"]),
            "shared_object_human_count": int(pair["shared_object_human_count"]),
            "shared_object_humans_json": pair["shared_object_humans_json"],
            "shared_object_verbsets_json": pair["shared_object_verbsets_json"],
            "same_class_multi_instance_status": bool(pair["same_class_multi_instance_status"]),
            "same_class_object_count": int(pair["same_class_object_count"]),
            "same_class_objects_json": pair["same_class_objects_json"],
            "same_class_verbsets_json": pair["same_class_verbsets_json"],
            "target_human_other_gt_hoi_count": int(pair["target_human_other_gt_hoi_count"]),
            "target_human_other_gt_hoi_json": pair["target_human_other_gt_hoi_json"],
            "context_priority_source": context_priority,
            "possible_context_count": possible_context_count,
            "noise_candidate_count": noise_candidate_count,
            "image_human_count": int(image["canonical_human_count"]),
            "image_object_count": int(image["canonical_object_count"]),
            "image_pair_count": int(image["valid_pair_count"]),
            "image_instance_count": int(image["canonical_instance_count"]),
            "has_action_prototype_ambiguity_auto": has_action_ambiguity,
            "has_auxiliary_gt_hoi": has_auxiliary,
            "rich_entity_features": rich_entities,
            "valid": bool(pair["valid"]),
            "invisible": bool(pair["invisible"]),
        }
        candidate_rows.append(candidate)
        object_split_tier_counter[(object_name, pair["split"], tier)] += 1
        object_priority_counter[(int(priority), object_name, pair["split"], tier)] += 1
        summary = image_candidate_accumulator.setdefault(pair["image_key"], {
            "image_key": pair["image_key"],
            "image_id": pair["image_id"],
            "split": pair["split"],
            "file_name": pair["file_name"],
            "eligible_for_final": bool(image["eligible_for_final"]),
            "legacy_control": not bool(image["eligible_for_final"]),
            "candidate_pair_count": 0,
            "eligible_candidate_pair_count": 0,
            "gold_count": 0,
            "silver_count": 0,
            "bronze_count": 0,
            "priority1_count": 0,
            "priority2_count": 0,
            "priority3_count": 0,
            "legacy_control_count": 0,
            "object_categories": set(),
        })
        summary["candidate_pair_count"] += 1
        summary["eligible_candidate_pair_count"] += 1 if final_pool else 0
        summary[f"{tier.lower()}_count"] += 1
        if int(priority) in {1, 2, 3}:
            summary[f"priority{priority}_count"] += 1
        else:
            summary["legacy_control_count"] += 1
        summary["object_categories"].add(object_name)

    image_rows = []
    for value in image_candidate_accumulator.values():
        value = dict(value)
        value["object_categories_json"] = json_dumps(sorted(value.pop("object_categories")))
        image_rows.append(value)
    image_rows.sort(key=lambda row: (row["split"], row["file_name"]))

    coverage_rows = []
    for priority, names in PRIORITY_OBJECTS.items():
        for object_name in names:
            for split in ("trainval", "test"):
                base = {
                    "object_priority": priority,
                    "object_category_name": object_name,
                    "split": split,
                }
                total = 0
                for tier in ("Gold", "Silver", "Bronze"):
                    count = object_split_tier_counter[(object_name, split, tier)]
                    base[f"{tier.lower()}_candidate_count"] = count
                    total += count
                base["total_candidate_count"] = total
                if total == 0:
                    has_mapping = object_name in PRIORITY_BY_OBJECT
                    rejected = sum(count for (obj, _reason), count in rejection_counter.items() if obj == object_name)
                    if not has_mapping:
                        zero_reason = "mapping_missing"
                    elif rejected > 0:
                        zero_reason = "filtering_or_no_eligible_actions"
                    else:
                        zero_reason = "data_distribution_no_pairs"
                    base["zero_count_reason"] = zero_reason
                else:
                    base["zero_count_reason"] = "nonzero"
                coverage_rows.append(base)

    pd.DataFrame(candidate_rows).to_parquet(output_dir / "raw_pair_candidates.parquet", index=False)
    with (output_dir / "raw_image_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]) if image_rows else [])
        writer.writeheader()
        writer.writerows(image_rows)
    with (output_dir / "rejection_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rejection_rows[0]) if rejection_rows else [])
        writer.writeheader()
        writer.writerows(rejection_rows)
    with (output_dir / "coverage_by_object.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    report_path = output_dir / "coverage_report.md"
    write_report(report_path, candidate_rows, rejection_rows, coverage_rows, all_object_names, min_object_area_ratio)

    return {
        "schema_version": "figure1_search_v2.stage3.candidates",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "formal_candidate_ranking_executed": False,
        "candidate_source": "complete_stage2_hico_index",
        "input_pair_count": int(len(pairs)),
        "candidate_pair_count": int(len(candidate_rows)),
        "eligible_candidate_pair_count": int(sum(1 for row in candidate_rows if row["eligible_candidate"])),
        "legacy_control_pair_count": int(sum(1 for row in candidate_rows if row["legacy_control"])),
        "candidate_image_count": int(len(image_rows)),
        "rejection_count": int(len(rejection_rows)),
        "tier_counts": dict(Counter(row["target_tier"] for row in candidate_rows)),
        "priority_counts": dict(Counter(str(row["object_priority"]) for row in candidate_rows)),
        "split_counts": dict(Counter(row["split"] for row in candidate_rows)),
        "legacy_controls": sorted({row["file_name"] for row in candidate_rows if row["legacy_control"]}),
        "legacy_controls_eligible_false": all(not row["eligible_candidate"] for row in candidate_rows if row["legacy_control"]),
        "min_object_area_ratio": min_object_area_ratio,
        "outputs": {
            "raw_pair_candidates": str(output_dir / "raw_pair_candidates.parquet"),
            "raw_image_candidates": str(output_dir / "raw_image_candidates.csv"),
            "rejection_log": str(output_dir / "rejection_log.csv"),
            "coverage_by_object": str(output_dir / "coverage_by_object.csv"),
            "coverage_report": str(report_path),
        },
    }


def write_report(
    path: Path,
    candidates: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    all_object_names: list[str],
    min_object_area_ratio: float,
) -> None:
    tier_counts = Counter(row["target_tier"] for row in candidates)
    priority_counts = Counter(row["object_priority"] for row in candidates)
    split_counts = Counter(row["split"] for row in candidates)
    rejection_counts = Counter(row["rejection_reason"] for row in rejections)
    lines = [
        "# Stage 3 High-Recall Candidate Coverage",
        "",
        "Candidate source: complete Stage 2 HICO-DET structured index.",
        "Old Top-30, old recommendations, and old aggregate scores were not read as candidate input.",
        "",
        "## Hard Filters",
        "",
        "- Target object must be in priority group 1, 2, or 3.",
        "- Target pair must contain at least one non-`no_interaction` GT verb.",
        "- Human/object bboxes must be valid and visible.",
        "- Target pair must carry raw GT provenance from Stage 2.",
        f"- Target object area ratio must be at least `{min_object_area_ratio}`.",
        "",
        "Human/entity counts, auxiliary context, possible context, noise candidates, and layout are recorded only as features.",
        "",
        "## Summary",
        "",
        f"- Pair candidates: {len(candidates)}",
        f"- Rejections: {len(rejections)}",
        f"- Candidate splits: {dict(split_counts)}",
        f"- Candidate tiers: {dict(tier_counts)}",
        f"- Candidate object priority counts: {dict(priority_counts)}",
        f"- Rejection reasons: {dict(rejection_counts)}",
        "",
        "## Coverage By Object",
        "",
        "| Priority | Object | Split | Gold | Silver | Bronze | Total | Zero reason |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['object_priority']} | {row['object_category_name']} | {row['split']} "
            f"| {row['gold_candidate_count']} | {row['silver_candidate_count']} "
            f"| {row['bronze_candidate_count']} | {row['total_candidate_count']} | {row['zero_count_reason']} |"
        )
    lines.extend([
        "",
        "## Objects Not In Retrieval Categories",
        "",
        "The full index contains additional object categories, but they are rejected only because they are outside Stage 0 priority groups.",
        "",
        ", ".join(name for name in all_object_names if name not in PRIORITY_BY_OBJECT),
        "",
        "## Provenance Notes",
        "",
        "- [GT] Each candidate row has a specific target pair and raw connection provenance inherited from Stage 2.",
        "- [AUTO] `possible_context_count` and `noise_candidate_count` are unconfirmed features, not manual evidence claims.",
        "- [AUTO] `target_human_other_gt_hoi_json` is the preferred context source when present.",
        "- [MANUAL] Visual evidence utility, action ambiguity, and true noise status remain unresolved.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    stats = build_candidates(args.index_dir, args.output_dir, args.min_object_area_ratio, args.seed)
    (args.output_dir / "candidate_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
