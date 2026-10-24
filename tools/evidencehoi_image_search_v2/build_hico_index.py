#!/usr/bin/env python3
"""Build the v2 structured HICO-DET index from audited GT connections."""

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

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml

try:
    from .canonicalize import SourceInstance, canonicalize_instances
    from .geometry import BBox, area, mat_to_half_open
    from .mappings import MappingBundle
except ImportError:  # pragma: no cover
    from canonicalize import SourceInstance, canonicalize_instances
    from geometry import BBox, area, mat_to_half_open
    from mappings import MappingBundle


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
DEFAULT_CONFIG = PROJECT_ROOT / "outputs/figure1_search_v2/configs/search_config_v2.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/figure1_search_v2/index")
    parser.add_argument("--high-iou-threshold", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def matlab_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.reshape(-1))
    return [value]


def matlab_connections(value: Any) -> list[tuple[int, int]]:
    array = np.asarray(value)
    if array.size == 0:
        return []
    array = array.astype(int)
    if array.ndim == 1:
        if array.size != 2:
            return []
        array = array.reshape(1, 2)
    return [(int(row[0]), int(row[1])) for row in array]


def mat_box(box: Any) -> tuple[float, float, float, float]:
    return float(box.x1), float(box.y1), float(box.x2), float(box.y2)


def mat_size(record: Any) -> tuple[int, int, int]:
    size = record.size
    return int(size.width), int(size.height), int(size.depth)


def image_id_from_name(file_name: str, json_by_name: dict[str, dict[str, Any]]) -> str:
    item = json_by_name.get(file_name)
    if item and item.get("img_id") is not None:
        return str(item["img_id"])
    return Path(file_name).stem


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_member_payload(member: SourceInstance) -> dict[str, Any]:
    return {
        "source_instance_id": member.source_instance_id,
        "role": member.role,
        "category_id": member.category_id,
        "category_name": member.category_name,
        "bbox_xyxy_half_open": list(member.bbox),
        "provenance": member.provenance,
    }


def select_tier_samples(pair_rows: list[dict[str, Any]], tier: str, limit: int = 10) -> list[dict[str, Any]]:
    selected = [row for row in pair_rows if row["pair_structural_tier"] == tier]
    selected.sort(key=lambda row: (
        0 if row["eligible_for_final"] else 1,
        -row["verb_count"],
        row["split"],
        row["file_name"],
        row["pair_id"],
    ))
    return selected[:limit]


def build_index(config: dict[str, Any], output_dir: Path, high_iou_threshold: float, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hico = config["hico_det"]
    mappings = config["mappings"]
    mat_path = Path(hico["official_bbox_annotation"])
    train_json_path = Path(hico["train_annotation"])
    test_json_path = Path(hico["test_annotation"])
    train_root = Path(hico["train_images"])
    test_root = Path(hico["test_images"])
    exclusion_path = PROJECT_ROOT / config["legacy_exclusions"]["path"]
    exclusions = {
        line.strip() for line in exclusion_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    bundle = MappingBundle(
        Path(mappings["object_categories"]),
        Path(mappings["actions"]),
        Path(str(mappings["hoi_text"]).split("::", 1)[0]),
        Path(mappings["verb_object_compatibility"]),
        mat_path,
    )
    train_json_by_name = {item["file_name"]: item for item in load_json(train_json_path)}
    test_json_by_name = {item["file_name"]: item for item in load_json(test_json_path)}
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    instances: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    canonical_audit: list[dict[str, Any]] = []

    relation_rows: list[dict[str, Any]] = []
    pair_accumulator: dict[tuple[str, str], dict[str, Any]] = {}
    source_instance_counter = 0
    relation_counter = 0
    source_to_canonical_global: dict[int, str] = {}
    source_to_bbox_global: dict[int, BBox] = {}
    canonical_by_id_global: dict[str, dict[str, Any]] = {}

    split_specs = [
        ("trainval", mat["bbox_train"], train_root, train_json_by_name),
        ("test", mat["bbox_test"], test_root, test_json_by_name),
    ]

    for split, mat_records, image_root, json_by_name in split_specs:
        for image_index, record in enumerate(mat_records):
            file_name = str(record.filename)
            image_key = f"{split}:{Path(file_name).stem}"
            image_id = image_id_from_name(file_name, json_by_name)
            width, height, depth = mat_size(record)
            image_path = image_root / file_name
            eligible_for_final = file_name not in exclusions
            source_instances: list[SourceInstance] = []
            local_relation_payloads: list[dict[str, Any]] = []
            invisible_hoi_count = 0
            visible_hoi_count = 0
            source_key_to_id: dict[tuple[str, int, int], int] = {}

            for hoi_local_index, hoi in enumerate(matlab_items(record.hoi)):
                hoi_id = int(hoi.id)
                hoi_mapping = bundle.for_hoi_id(hoi_id)
                invisible = int(hoi.invis) == 1
                if invisible:
                    invisible_hoi_count += 1
                    continue
                visible_hoi_count += 1
                humans = matlab_items(hoi.bboxhuman)
                objects = matlab_items(hoi.bboxobject)

                for local_human_index, human_box in enumerate(humans, start=1):
                    key = ("human", hoi_id, local_human_index)
                    source_id = source_instance_counter
                    source_instance_counter += 1
                    source_key_to_id[key] = source_id
                    bbox = mat_to_half_open(mat_box(human_box))
                    source_instances.append(
                        SourceInstance(
                            source_instance_id=source_id,
                            role="human",
                            category_id=1,
                            category_name="person",
                            bbox=bbox,
                            provenance={
                                "split": split,
                                "file_name": file_name,
                                "image_index_0based": image_index,
                                "hoi_local_index_0based": hoi_local_index,
                                "hoi_id_1based": hoi_id,
                                "mat_human_index_1based": local_human_index,
                                "mat_bbox_1based_inclusive": list(mat_box(human_box)),
                            },
                        )
                    )
                for local_object_index, object_box in enumerate(objects, start=1):
                    key = ("object", hoi_id, local_object_index)
                    source_id = source_instance_counter
                    source_instance_counter += 1
                    source_key_to_id[key] = source_id
                    bbox = mat_to_half_open(mat_box(object_box))
                    source_instances.append(
                        SourceInstance(
                            source_instance_id=source_id,
                            role="object",
                            category_id=hoi_mapping.object_sparse_id,
                            category_name=hoi_mapping.object_name,
                            bbox=bbox,
                            provenance={
                                "split": split,
                                "file_name": file_name,
                                "image_index_0based": image_index,
                                "hoi_local_index_0based": hoi_local_index,
                                "hoi_id_1based": hoi_id,
                                "mat_object_index_1based": local_object_index,
                                "mat_bbox_1based_inclusive": list(mat_box(object_box)),
                            },
                        )
                    )
                for connection_index, (human_index, object_index) in enumerate(matlab_connections(hoi.connection)):
                    human_source_id = source_key_to_id[("human", hoi_id, human_index)]
                    object_source_id = source_key_to_id[("object", hoi_id, object_index)]
                    relation_id = f"{image_key}:r{relation_counter}"
                    relation_counter += 1
                    local_relation_payloads.append({
                        "relation_id": relation_id,
                        "split": split,
                        "image_key": image_key,
                        "image_id": image_id,
                        "file_name": file_name,
                        "hoi_id": hoi_id,
                        "verb_id": hoi_mapping.verb_id,
                        "verb_name": hoi_mapping.verb_name,
                        "hoi_text": hoi_mapping.hoi_text,
                        "object_category_id": hoi_mapping.object_sparse_id,
                        "object_category_name": hoi_mapping.object_name,
                        "human_source_instance_id": human_source_id,
                        "object_source_instance_id": object_source_id,
                        "mat_connection_index_0based": connection_index,
                        "mat_connection_1based": [human_index, object_index],
                        "hoi_local_index_0based": hoi_local_index,
                        "invisible": False,
                        "valid": True,
                    })

            canonical_instances, source_to_canonical = canonicalize_instances(
                image_key, source_instances, high_iou_threshold=high_iou_threshold
            )
            source_to_canonical_global.update(source_to_canonical)
            for source in source_instances:
                source_to_bbox_global[source.source_instance_id] = source.bbox

            for canonical in canonical_instances:
                row = {
                    "image_key": image_key,
                    "image_id": image_id,
                    "split": split,
                    "file_name": file_name,
                    "eligible_for_final": eligible_for_final,
                    "legacy_negative_example": not eligible_for_final,
                    "canonical_instance_id": canonical.canonical_id,
                    "role": canonical.role,
                    "category_id": canonical.category_id,
                    "category_name": canonical.category_name,
                    "bbox_x1": canonical.bbox[0],
                    "bbox_y1": canonical.bbox[1],
                    "bbox_x2": canonical.bbox[2],
                    "bbox_y2": canonical.bbox[3],
                    "bbox_area": area(canonical.bbox),
                    "source_instance_count": len(canonical.members),
                    "source_instance_ids_json": json_dumps([m.source_instance_id for m in canonical.members]),
                    "source_members_json": json_dumps([source_member_payload(m) for m in canonical.members]),
                    "merge_method": canonical.merge_method,
                    "merge_confidence": canonical.merge_confidence,
                    "min_pairwise_iou": canonical.min_pairwise_iou,
                }
                instances.append(row)
                canonical_by_id_global[canonical.canonical_id] = row
                canonical_audit.append({
                    "image_key": image_key,
                    "file_name": file_name,
                    "split": split,
                    "canonical_instance_id": canonical.canonical_id,
                    "role": canonical.role,
                    "category_id": canonical.category_id,
                    "category_name": canonical.category_name,
                    "merge_method": canonical.merge_method,
                    "merge_confidence": canonical.merge_confidence,
                    "min_pairwise_iou": canonical.min_pairwise_iou,
                    "source_instance_count": len(canonical.members),
                    "source_instance_ids": ";".join(str(m.source_instance_id) for m in canonical.members),
                    "source_provenance_json": json_dumps([m.provenance for m in canonical.members]),
                })

            for relation in local_relation_payloads:
                human_id = source_to_canonical[relation["human_source_instance_id"]]
                object_id = source_to_canonical[relation["object_source_instance_id"]]
                relation["canonical_human_id"] = human_id
                relation["canonical_object_id"] = object_id
                relation_rows.append(relation)
                pair_key = (human_id, object_id)
                payload = pair_accumulator.setdefault(pair_key, {
                    "image_key": image_key,
                    "image_id": image_id,
                    "split": split,
                    "file_name": file_name,
                    "eligible_for_final": eligible_for_final,
                    "legacy_negative_example": not eligible_for_final,
                    "canonical_human_id": human_id,
                    "canonical_object_id": object_id,
                    "object_category_id": relation["object_category_id"],
                    "object_category_name": relation["object_category_name"],
                    "relations": [],
                    "verb_ids": set(),
                    "verb_names": set(),
                    "hoi_ids": set(),
                })
                payload["relations"].append(relation)
                payload["verb_ids"].add(relation["verb_id"])
                payload["verb_names"].add(relation["verb_name"])
                payload["hoi_ids"].add(relation["hoi_id"])

            image_summaries.append({
                "image_key": image_key,
                "image_id": image_id,
                "split": split,
                "file_name": file_name,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "depth": depth,
                "eligible_for_final": eligible_for_final,
                "legacy_negative_example": not eligible_for_final,
                "visible_hoi_entries": visible_hoi_count,
                "invisible_hoi_entries": invisible_hoi_count,
                "source_instance_count": len(source_instances),
                "canonical_instance_count": len(canonical_instances),
                "canonical_human_count": sum(1 for item in canonical_instances if item.role == "human"),
                "canonical_object_count": sum(1 for item in canonical_instances if item.role == "object"),
                "raw_connection_count": len(local_relation_payloads),
                "valid_pair_count": 0,  # filled after pair finalization
                "gold_pair_count": 0,
                "silver_pair_count": 0,
                "bronze_pair_count": 0,
                "image_best_tier": "None",
            })

    # Build pair-level structural indexes per image.
    pairs_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in pair_accumulator.values():
        pairs_by_image[payload["image_key"]].append(payload)

    finalized_pairs: list[dict[str, Any]] = []
    for image_key, payloads in pairs_by_image.items():
        object_to_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        class_to_pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
        human_to_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for payload in payloads:
            object_to_pairs[payload["canonical_object_id"]].append(payload)
            class_to_pairs[payload["object_category_id"]].append(payload)
            human_to_pairs[payload["canonical_human_id"]].append(payload)

        for payload in payloads:
            object_peer_pairs = object_to_pairs[payload["canonical_object_id"]]
            object_peer_humans = sorted({p["canonical_human_id"] for p in object_peer_pairs})
            object_peer_verbsets = {tuple(sorted(p["verb_ids"])) for p in object_peer_pairs}
            class_peer_pairs = class_to_pairs[payload["object_category_id"]]
            class_peer_objects = sorted({p["canonical_object_id"] for p in class_peer_pairs})
            class_peer_verbsets = {tuple(sorted(p["verb_ids"])) for p in class_peer_pairs}
            if len(object_peer_humans) >= 2 and len(object_peer_verbsets) >= 2:
                tier = "Gold"
            elif len(class_peer_objects) >= 2 and len(class_peer_verbsets) >= 2:
                tier = "Silver"
            else:
                tier = "Bronze"

            human_instance = canonical_by_id_global[payload["canonical_human_id"]]
            object_instance = canonical_by_id_global[payload["canonical_object_id"]]
            target_human_other = []
            for peer in human_to_pairs[payload["canonical_human_id"]]:
                if peer is payload:
                    continue
                target_human_other.append({
                    "canonical_object_id": peer["canonical_object_id"],
                    "object_category_id": peer["object_category_id"],
                    "object_category_name": peer["object_category_name"],
                    "verb_ids": sorted(peer["verb_ids"]),
                    "verb_names": sorted(peer["verb_names"]),
                })
            relation_provenance = []
            for relation in payload["relations"]:
                relation_provenance.append({
                    "relation_id": relation["relation_id"],
                    "hoi_id": relation["hoi_id"],
                    "verb_id": relation["verb_id"],
                    "verb_name": relation["verb_name"],
                    "mat_connection_1based": relation["mat_connection_1based"],
                    "hoi_local_index_0based": relation["hoi_local_index_0based"],
                    "human_source_instance_id": relation["human_source_instance_id"],
                    "object_source_instance_id": relation["object_source_instance_id"],
                })
            pair_id = f"{payload['image_key']}:p{len(finalized_pairs)}"
            finalized_pairs.append({
                "pair_id": pair_id,
                "image_key": payload["image_key"],
                "image_id": payload["image_id"],
                "split": payload["split"],
                "file_name": payload["file_name"],
                "eligible_for_final": payload["eligible_for_final"],
                "legacy_negative_example": payload["legacy_negative_example"],
                "canonical_human_id": payload["canonical_human_id"],
                "canonical_object_id": payload["canonical_object_id"],
                "human_bbox_x1": human_instance["bbox_x1"],
                "human_bbox_y1": human_instance["bbox_y1"],
                "human_bbox_x2": human_instance["bbox_x2"],
                "human_bbox_y2": human_instance["bbox_y2"],
                "object_bbox_x1": object_instance["bbox_x1"],
                "object_bbox_y1": object_instance["bbox_y1"],
                "object_bbox_x2": object_instance["bbox_x2"],
                "object_bbox_y2": object_instance["bbox_y2"],
                "object_category_id": payload["object_category_id"],
                "object_category_name": payload["object_category_name"],
                "verb_ids_json": json_dumps(sorted(payload["verb_ids"])),
                "verb_names_json": json_dumps(sorted(payload["verb_names"])),
                "hoi_ids_json": json_dumps(sorted(payload["hoi_ids"])),
                "verb_count": len(payload["verb_ids"]),
                "raw_relation_count": len(payload["relations"]),
                "raw_annotation_provenance_json": json_dumps(relation_provenance),
                "shared_object_status": len(object_peer_humans) >= 2,
                "shared_object_human_count": len(object_peer_humans),
                "shared_object_humans_json": json_dumps(object_peer_humans),
                "shared_object_verbsets_json": json_dumps([list(v) for v in sorted(object_peer_verbsets)]),
                "same_class_multi_instance_status": len(class_peer_objects) >= 2,
                "same_class_object_count": len(class_peer_objects),
                "same_class_objects_json": json_dumps(class_peer_objects),
                "same_class_verbsets_json": json_dumps([list(v) for v in sorted(class_peer_verbsets)]),
                "target_human_other_gt_hoi_json": json_dumps(target_human_other),
                "target_human_other_gt_hoi_count": len(target_human_other),
                "pair_structural_tier": tier,
                "gold_shared_instance_confidence": object_instance["merge_confidence"] if tier == "Gold" else 0.0,
                "object_merge_method": object_instance["merge_method"],
                "human_merge_method": human_instance["merge_method"],
                "invisible": False,
                "valid": True,
            })

    pairs = finalized_pairs

    pair_counts_by_image = Counter(row["image_key"] for row in pairs)
    tier_counts_by_image: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pairs:
        tier_counts_by_image[row["image_key"]][row["pair_structural_tier"]] += 1
    tier_order = {"Gold": 3, "Silver": 2, "Bronze": 1, "None": 0}
    for summary in image_summaries:
        image_key = summary["image_key"]
        summary["valid_pair_count"] = pair_counts_by_image[image_key]
        for tier in ("Gold", "Silver", "Bronze"):
            summary[f"{tier.lower()}_pair_count"] = tier_counts_by_image[image_key][tier]
        present = [tier for tier in ("Gold", "Silver", "Bronze") if tier_counts_by_image[image_key][tier] > 0]
        summary["image_best_tier"] = max(present, key=lambda item: tier_order[item]) if present else "None"

    pd.DataFrame(instances).to_parquet(output_dir / "instances.parquet", index=False)
    pd.DataFrame(pairs).to_parquet(output_dir / "pairs.parquet", index=False)
    pd.DataFrame(image_summaries).to_parquet(output_dir / "image_summary.parquet", index=False)

    with (output_dir / "canonicalization_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(canonical_audit[0]))
        writer.writeheader()
        writer.writerows(canonical_audit)

    tier_samples = {tier: select_tier_samples(pairs, tier, 10) for tier in ("Gold", "Silver", "Bronze")}
    manual_spot_check = write_manual_spot_check(output_dir / "manual_spot_check.md", tier_samples)

    stats = build_statistics(
        instances, pairs, image_summaries, canonical_audit, relation_rows, exclusions,
        output_dir, high_iou_threshold, seed, manual_spot_check,
    )
    (output_dir / "index_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def write_manual_spot_check(path: Path, samples: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    lines = [
        "# Stage 2 Structural Spot Check",
        "",
        "This is an index-level [GT]/[AUTO] spot check, not visual manual review.",
        "Every listed pair is backed by real official MAT connection provenance.",
        "",
    ]
    summary: dict[str, Any] = {}
    for tier, rows in samples.items():
        summary[tier] = len(rows)
        lines.extend([f"## {tier}", ""])
        if not rows:
            lines.extend(["No samples available.", ""])
            continue
        lines.append("| Split | Image | Pair | Object | Verbs | Provenance | Eligible |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in rows:
            verbs = ", ".join(json.loads(row["verb_names_json"]))
            hois = ", ".join(str(x) for x in json.loads(row["hoi_ids_json"]))
            lines.append(
                f"| {row['split']} | {row['file_name']} | `{row['canonical_human_id']} -> {row['canonical_object_id']}` "
                f"| {row['object_category_name']} | {verbs} | HOI {hois} | {row['eligible_for_final']} |"
            )
        lines.append("")
    lines.extend([
        "## Audit Conclusion",
        "",
        "- [GT] All sampled rows originate from official connection rows; no human-object Cartesian product is present.",
        "- [AUTO] Gold samples are shared canonical object instances with distinct human links and different verb sets; object merge confidence is recorded per row.",
        "- [AUTO] Silver samples use same object category with multiple canonical object instances and different verb sets.",
        "- [AUTO] Bronze samples have valid GT pairs but do not satisfy Gold or Silver structure in the current index.",
        "- [MANUAL] Visual action ambiguity, evidence utility, and noise status remain unconfirmed.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_statistics(
    instances: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    images: list[dict[str, Any]],
    canonical_audit: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    exclusions: set[str],
    output_dir: Path,
    high_iou_threshold: float,
    seed: int,
    spot_check_counts: dict[str, int],
) -> dict[str, Any]:
    tiers = Counter(row["pair_structural_tier"] for row in pairs)
    split_pairs = Counter(row["split"] for row in pairs)
    split_images = Counter(row["split"] for row in images)
    merge_methods = Counter(row["merge_method"] for row in instances)
    merged_instances = [row for row in instances if row["source_instance_count"] > 1]
    provenance_relation_count = sum(
        len(json.loads(row["raw_annotation_provenance_json"])) for row in pairs
    )
    old_exclusion_rows = [row for row in images if row["file_name"] in exclusions]
    return {
        "schema_version": "figure1_search_v2.stage2.index_statistics",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "random_seed": seed,
        "high_iou_threshold": high_iou_threshold,
        "formal_candidate_ranking_executed": False,
        "image_count": len(images),
        "instance_count": len(instances),
        "pair_count": len(pairs),
        "raw_relation_count": len(relation_rows),
        "provenance_relation_count_in_pairs": provenance_relation_count,
        "all_pairs_trace_to_real_connection": provenance_relation_count == len(relation_rows) == sum(row["raw_relation_count"] for row in pairs),
        "split_image_counts": dict(split_images),
        "split_pair_counts": dict(split_pairs),
        "tier_pair_counts": dict(tiers),
        "canonical_merge_methods": dict(merge_methods),
        "merged_instance_count": len(merged_instances),
        "exact_merged_instance_count": sum(1 for row in merged_instances if row["merge_method"] == "exact_bbox"),
        "high_iou_merged_instance_count": sum(1 for row in merged_instances if row["merge_method"] == "high_iou_complete_link"),
        "legacy_exclusion_images_in_index": sorted(row["file_name"] for row in old_exclusion_rows),
        "legacy_exclusion_eligible_false": all(not row["eligible_for_final"] for row in old_exclusion_rows),
        "spot_check_counts": spot_check_counts,
        "outputs": {
            "instances": str(output_dir / "instances.parquet"),
            "pairs": str(output_dir / "pairs.parquet"),
            "image_summary": str(output_dir / "image_summary.parquet"),
            "canonicalization_audit": str(output_dir / "canonicalization_audit.csv"),
            "manual_spot_check": str(output_dir / "manual_spot_check.md"),
        },
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stats = build_index(config, args.output_dir, args.high_iou_threshold, args.seed)
    print(json.dumps({
        "images": stats["image_count"],
        "instances": stats["instance_count"],
        "pairs": stats["pair_count"],
        "tiers": stats["tier_pair_counts"],
        "all_pairs_trace_to_real_connection": stats["all_pairs_trace_to_real_connection"],
    }, indent=2, ensure_ascii=False))
    return 0 if stats["all_pairs_trace_to_real_connection"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
