#!/usr/bin/env python3
"""Validate the final self-contained EvidenceHOI Top-20 mechanism package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


VARIANTS = (
    "full",
    "no_anchor_writeback",
    "no_prototype_state_writeback",
    "no_setm",
    "endpoint_only_setm",
)

REQUIRED_NPZ_KEYS = (
    "base_hoi_query",
    "graph_pair_features_after_layer_0",
    "graph_pair_features_after_layer_1",
    "graph_pair_features_final",
    "gated_query_delta",
    "corrected_hoi_query",
    "probe_queries",
    "probe_decoder_hidden",
    "probe_action_logits",
    "probe_action_probabilities",
    "pair_anchor_pre_sync",
    "sync_anchor_delta_proposed",
    "sync_anchor_delta_applied",
    "pair_anchor_post_sync_actual",
    "candidate_evidence_tokens",
    "candidate_evidence_mask",
    "candidate_evidence_weights",
    "prototype_slots_pre_sync",
    "updated_prototype_slots_proposed",
    "prototype_slots_post_writeback_actual",
    "prototype_slot_mask",
    "match_logits",
    "visual_to_text_probability",
    "text_to_visual_probability",
    "alpha",
    "pi",
    "role_coverage",
    "selected_context_indices",
    "selected_hoi_indices",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prototype-bank", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


class Checks:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        name: str,
        passed: bool,
        expected: Any,
        actual: Any,
        message: str = "",
        tolerance: Optional[float] = None,
        affected: Optional[Sequence[Any]] = None,
    ) -> None:
        self.rows.append({
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "expected": expected,
            "actual": actual,
            "tolerance": tolerance,
            "affected_candidates": list(affected or []),
            "message": message,
        })

    @property
    def passed(self) -> bool:
        return all(row["status"] == "PASS" for row in self.rows)


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    checks = Checks()

    required = [
        "run_meta.json",
        "README.md",
        "reports/field_dictionary.md",
        "reports/result_summary.md",
        "reports/warnings.md",
        "config/full_args.json",
        "config/environment.json",
        "config/file_hashes.json",
        "config/variant_config_diff.json",
        "manifests/top20_cases.csv",
        "manifests/source_asset_manifest.csv",
        "manifests/proposal_fingerprints.csv",
        "manifests/retrieval_fingerprints.csv",
        "labels/action_names.json",
        "labels/object_names.json",
        "labels/action_id_mapping.json",
        "labels/object_id_mapping.json",
        "labels/prototype_prompt_metadata.json",
        "logs/baseline.log",
        "provenance/checkpoint_load_report.json",
        "provenance/prototype_bank_integrity.json",
        "provenance/commands.txt",
        "tables/layer_prediction_scores.csv",
        "tables/query_residual_summary.csv",
        "tables/visual_unit_updates.csv",
        "tables/prototype_role_updates.csv",
        "tables/anchor_writeback.csv",
        "tables/context_transitions.csv",
        "tables/gt_pair_mapping.csv",
        "tables/ablation_scores.csv",
        "tables/ablation_case_summary.csv",
        "provenance/baseline_equivalence_report.json",
    ]
    missing = [relative for relative in required if not (root / relative).is_file()]
    checks.add("required_files", not missing, "all required files", missing or "all present", affected=missing)

    expected_rows = {
        "tables/layer_prediction_scores.csv": 9360,
        "tables/query_residual_summary.csv": 20,
        "tables/visual_unit_updates.csv": 1400,
        "tables/prototype_role_updates.csv": 1200,
        "tables/anchor_writeback.csv": 40,
        "tables/context_transitions.csv": 80,
        "tables/ablation_scores.csv": 11700,
        "tables/ablation_case_summary.csv": 100,
    }
    table_cache: Dict[str, List[Dict[str, str]]] = {}
    for relative, expected in expected_rows.items():

        rows = csv_rows(root / relative)
        table_cache[relative] = rows
        checks.add(f"row_count:{relative}", len(rows) == expected, expected, len(rows))

    cases = csv_rows(root / "manifests/top20_cases.csv")
    checks.add(
        "fixed_top20",
        len(cases) == 20 and len({row["image_id"] for row in cases}) == 20,
        "20 rows and 20 unique images",
        {"rows": len(cases), "unique_images": len({row["image_id"] for row in cases})},
    )
    original_cases = csv_rows(root / "manifests/top20_source_original.csv")
    case_identity = {
        (row["candidate_id"], row["image_id"])
        for row in cases
    }
    original_identity = {
        (row["candidate_id"], row["image_id"])
        for row in original_cases
    }
    checks.add(
        "top20_identity_matches_source",
        len(original_cases) == 20 and case_identity == original_identity,
        "exact 20 candidate_id/image_id pairs",
        {"source_rows": len(original_cases), "equal": case_identity == original_identity},
    )

    full_args = json.loads((root / "config/full_args.json").read_text(encoding="utf-8"))
    effective = {
        "M": int(full_args.get("graph_evidence_topk", -1)),
        "K": int(full_args.get("graph_text_topk", -1)),
        "layers": int(full_args.get("graph_num_layers", -1)),
        "batch_size": int(full_args.get("batch_size", -1)),
        "seed": int(full_args.get("seed", -1)),
    }
    checks.add(
        "effective_configuration",
        effective == {"M": 4, "K": 5, "layers": 2, "batch_size": 1, "seed": 3407},
        {"M": 4, "K": 5, "layers": 2, "batch_size": 1, "seed": 3407},
        effective,
    )
    file_hashes = json.loads((root / "config/file_hashes.json").read_text(encoding="utf-8"))
    mandatory_hashes = ("checkpoint", "detr_weights", "blip_pretrained", "dataset_annotation", "top20_mapping", "prototype_bank", "prototype_mask", "prototype_prompt_metadata", "code_files")
    missing_hashes = [key for key in mandatory_hashes if key not in file_hashes or not file_hashes[key]]
    checks.add("mandatory_asset_and_code_hashes", not missing_hashes, list(mandatory_hashes), missing_hashes or "all present")

    source_images = list((root / "source_images").glob("*.jpg"))
    arrays = list((root / "arrays").glob("*.npz"))
    checks.add("source_images", len(source_images) == 20, 20, len(source_images))
    checks.add("npz_cases", len(arrays) == 20, 20, len(arrays))
    npz_missing: Dict[str, List[str]] = {}
    npz_shape_errors: List[str] = []
    npz_nonfinite: List[str] = []
    retrieval_layer_mismatch: List[str] = []
    expected_npz_shapes = {
        "graph_pair_features_after_layer_0": (256,),
        "graph_pair_features_after_layer_1": (256,),
        "graph_pair_features_final": (256,),
        "probe_action_logits": (4, 117),
        "probe_action_probabilities": (4, 117),
        "candidate_evidence_tokens": (2, 7, 256),
        "candidate_evidence_mask": (2, 7),
        "candidate_evidence_weights": (2, 7),
        "prototype_slots_pre_sync": (2, 5, 6, 256),
        "updated_prototype_slots_proposed": (2, 5, 6, 256),
        "prototype_slots_post_writeback_actual": (2, 5, 6, 256),
        "prototype_slot_mask": (2, 5, 6),
        "match_logits": (2, 5, 7, 6),
        "visual_to_text_probability": (2, 5, 7, 6),
        "text_to_visual_probability": (2, 5, 6, 7),
        "alpha": (2, 5, 7),
        "pi": (2, 5),
        "role_coverage": (2, 5, 6),
        "selected_context_indices": (2, 4),
        "selected_hoi_indices": (2, 5),
    }
    for path in arrays:
        with np.load(path, allow_pickle=False) as archive:
            missing_keys = [key for key in REQUIRED_NPZ_KEYS if key not in archive]
            if missing_keys:
                npz_missing[path.name] = missing_keys
                continue
            for key, expected_shape in expected_npz_shapes.items():
                if tuple(archive[key].shape) != expected_shape:
                    npz_shape_errors.append(f"{path.name}:{key}:{tuple(archive[key].shape)}")
            for key in archive.files:
                value = archive[key]
                if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                    npz_nonfinite.append(f"{path.name}:{key}")
            same_ids = np.array_equal(archive["selected_hoi_indices"][0], archive["selected_hoi_indices"][1])
            same_mask = np.array_equal(archive["prototype_slot_mask"][0].any(axis=-1), archive["prototype_slot_mask"][1].any(axis=-1))
            if not (same_ids and same_mask):
                retrieval_layer_mismatch.append(path.name)
    checks.add("npz_required_keys", not npz_missing, list(REQUIRED_NPZ_KEYS), npz_missing or "all present", affected=list(npz_missing))
    checks.add("npz_required_shapes", not npz_shape_errors, expected_npz_shapes, npz_shape_errors or "all exact", affected=npz_shape_errors)
    checks.add("npz_finite_float_tensors", not npz_nonfinite, 0, len(npz_nonfinite), affected=npz_nonfinite)
    checks.add("full_retrieval_stable_between_layers", not retrieval_layer_mismatch, "20/20 identical", 20 - len(retrieval_layer_mismatch), affected=retrieval_layer_mismatch)
    for variant in VARIANTS:
        raw = list((root / "raw_records" / variant).glob("*.pt"))
        checks.add(f"raw_records:{variant}", len(raw) == 20, 20, len(raw))
        raw_report_path = root / "provenance" / f"{variant}_raw_validation.json"
        report = json.loads(raw_report_path.read_text(encoding="utf-8")) if raw_report_path.is_file() else {}
        checks.add(
            f"raw_validation:{variant}",
            bool(report.get("passed", False)),
            True,
            report.get("passed"),
            message="; ".join(report.get("errors", [])[:3]),
        )
        variant_artifacts = [
            root / "config/variants" / f"{variant}_effective_args.json",
            root / "logs/variants" / f"{variant}.log",
        ]
        missing_variant_artifacts = [str(path.relative_to(root)) for path in variant_artifacts if not path.is_file()]
        checks.add(
            f"variant_artifacts:{variant}",
            not missing_variant_artifacts,
            "effective args and log",
            missing_variant_artifacts or "both present",
            affected=missing_variant_artifacts,
        )
        expected_setm_calls = 0 if variant == "no_setm" else 2
        bad_record_schema = [
            item.get("image_id")
            for item in report.get("records", [])
            if int(item.get("pair_count", -1)) != 32
            or int(item.get("layer_count", -1)) != 2
            or int(item.get("setm_call_count", -1)) != expected_setm_calls
            or not bool(item.get("passed", False))
        ]
        checks.add(
            f"record_schema_and_setm_calls:{variant}",
            len(report.get("records", [])) == 20 and not bad_record_schema,
            {"records": 20, "pairs": 32, "layers": 2, "setm_calls": expected_setm_calls},
            {"records": len(report.get("records", [])), "bad": bad_record_schema},
            affected=bad_record_schema,
        )

    residual_rows = table_cache.get("tables/query_residual_summary.csv", [])
    residual_errors = [float(row["residual_max_abs_error"]) for row in residual_rows]
    max_residual = max(residual_errors, default=math.inf)
    checks.add("query_residual_identity", max_residual <= 1e-5, "<=1e-5", max_residual, tolerance=1e-5)

    full_raw = json.loads((root / "provenance/full_raw_validation.json").read_text(encoding="utf-8"))
    layer1_diff = full_raw.get("max_layer1_vs_final_diff")
    checks.add(
        "layer1_probe_vs_normal_final",
        layer1_diff is not None and float(layer1_diff) <= 1e-5,
        "<=1e-5",
        layer1_diff,
        tolerance=1e-5,
    )

    equivalence = json.loads((root / "provenance/baseline_equivalence_report.json").read_text(encoding="utf-8"))
    checks.add(
        "baseline_equivalence",
        bool(equivalence.get("passed", False)),
        True,
        equivalence.get("passed"),
        message="; ".join(equivalence.get("errors", [])[:3]),
    )

    proposal_rows = csv_rows(root / "manifests/proposal_fingerprints.csv")
    retrieval_rows = csv_rows(root / "manifests/retrieval_fingerprints.csv")
    bad_proposals = [row["candidate_id"] + ":" + row["variant"] for row in proposal_rows if not bool_value(row["matches_full"])]
    bad_retrieval = [row["candidate_id"] + ":" + row["variant"] for row in retrieval_rows if not bool_value(row["matches_full"])]
    checks.add("proposal_fingerprints", len(proposal_rows) == 100 and not bad_proposals, "100 equal", {"rows": len(proposal_rows), "bad": bad_proposals}, affected=bad_proposals)
    checks.add("retrieval_fingerprints", len(retrieval_rows) == 100 and not bad_retrieval, "100 equal", {"rows": len(retrieval_rows), "bad": bad_retrieval}, affected=bad_retrieval)

    anchor_rows = table_cache.get("tables/anchor_writeback.csv", [])
    candidate_applied_bad = [
        row["candidate_id"] for row in anchor_rows
        if abs(float(row["candidate_anchor_delta_applied_l2"])) > 1e-12
    ]
    setm_nonpositive = [
        row["candidate_id"] for row in anchor_rows
        if bool_value(row["pair_valid"]) and float(row["setm_anchor_delta_applied_l2"]) <= 0
    ]
    checks.add("candidate_anchor_not_applied", not candidate_applied_bad, 0, len(candidate_applied_bad), affected=candidate_applied_bad)
    checks.add("setm_anchor_applied", not setm_nonpositive, "positive for valid targets", len(setm_nonpositive), affected=setm_nonpositive)

    gt_rows = csv_rows(root / "tables/gt_pair_mapping.csv")
    gt_candidates = {row["candidate_id"] for row in gt_rows}
    valid_gt_candidates = {row["candidate_id"] for row in gt_rows if bool_value(row["mapping_valid"])}
    checks.add("gt_candidate_coverage", len(gt_candidates) == 20, 20, len(gt_candidates))
    rank_warning = {
        int(row["mechanism_rank"])
        for row in gt_rows
        if bool_value(row["mapping_warning"])
    }
    checks.add("rank_6_warning_retained", 6 in rank_warning, True, 6 in rank_warning)
    checks.add("rank_19_warning_retained", 19 in rank_warning, True, 19 in rank_warning)
    formerly_empty = {1, 8, 11, 13, 14, 19, 20}
    explained = {
        int(row["mechanism_rank"])
        for row in gt_rows
        if int(row["mechanism_rank"]) in formerly_empty
        and (row["gt_available"] or row["failure_reason"])
    }
    checks.add("formerly_empty_gt_explained", explained == formerly_empty, sorted(formerly_empty), sorted(explained))

    ablation_rows = table_cache.get("tables/ablation_scores.csv", [])
    variant_counts = {variant: 0 for variant in VARIANTS}
    for row in ablation_rows:
        variant_counts[row["variant"]] += 1
    checks.add(
        "ablation_variant_counts",
        all(count == 2340 for count in variant_counts.values()),
        {variant: 2340 for variant in VARIANTS},
        variant_counts,
    )
    unique_specs = {
        "tables/layer_prediction_scores.csv": ("candidate_id", "stage", "action_id_model"),
        "tables/query_residual_summary.csv": ("candidate_id",),
        "tables/visual_unit_updates.csv": ("candidate_id", "layer_idx", "prototype_idx", "visual_idx"),
        "tables/prototype_role_updates.csv": ("candidate_id", "layer_idx", "prototype_idx", "text_role_idx"),
        "tables/anchor_writeback.csv": ("candidate_id", "layer_idx"),
        "tables/context_transitions.csv": ("candidate_id", "slot_idx"),
        "tables/ablation_scores.csv": ("variant", "candidate_id", "action_id_model"),
        "tables/ablation_case_summary.csv": ("variant", "candidate_id"),
    }
    duplicate_tables: Dict[str, int] = {}
    noncanonical_boolean_cells: List[str] = []
    nonfinite_text_cells: List[str] = []
    for relative, key_fields in unique_specs.items():
        rows = table_cache.get(relative, [])
        keys = [tuple(row[field] for field in key_fields) for row in rows]
        if len(keys) != len(set(keys)):
            duplicate_tables[relative] = len(keys) - len(set(keys))
        for row_idx, row in enumerate(rows, start=2):
            for field, value in row.items():
                normalized = value.strip()
                if normalized in {"True", "False"}:
                    noncanonical_boolean_cells.append(f"{relative}:{row_idx}:{field}")
                if normalized.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                    nonfinite_text_cells.append(f"{relative}:{row_idx}:{field}")
    checks.add("table_unique_keys", not duplicate_tables, "no duplicate composite keys", duplicate_tables or "all unique", affected=list(duplicate_tables))
    checks.add("csv_boolean_lowercase", not noncanonical_boolean_cells, "true/false", len(noncanonical_boolean_cells), affected=noncanonical_boolean_cells[:50])
    checks.add("csv_no_nan_or_inf_text", not nonfinite_text_cells, 0, len(nonfinite_text_cells), affected=nonfinite_text_cells[:50])

    expected_variant_fields = {
        "full": (True, True, True),
        "no_anchor_writeback": (False, True, True),
        "no_prototype_state_writeback": (True, False, True),
        "no_setm": (False, False, False),
        "endpoint_only_setm": (True, True, True),
    }
    bad_variant_fields: List[str] = []
    for row in ablation_rows:
        observed = (
            bool_value(row["visual_writeback_applied"]),
            bool_value(row["text_writeback_applied"]),
            bool_value(row["setm_called"]),
        )
        if observed != expected_variant_fields[row["variant"]]:
            bad_variant_fields.append(f"{row['variant']}:{row['candidate_id']}:{row['action_id_model']}")
    checks.add("variant_causal_field_semantics", not bad_variant_fields, expected_variant_fields, len(bad_variant_fields), affected=bad_variant_fields[:50])

    mapping_rows = csv_rows(root / "manifests/full_pair_mapping.csv")
    incomplete_mapping = [
        row.get("candidate_id", "")
        for row in mapping_rows
        if row.get("pair_idx", "") == ""
    ]
    checks.add("target_pair_mapping_complete_or_warned", len(mapping_rows) == 20 and not incomplete_mapping, 20, {"rows": len(mapping_rows), "incomplete": incomplete_mapping}, affected=incomplete_mapping)

    prompt_rows = json.loads((root / "labels/prototype_prompt_metadata.json").read_text(encoding="utf-8"))
    required_prompt_fields = ("human_node", "object_node", "ho_edge", "discriminative", "scene_context")
    incomplete_prompts = [
        int(item.get("index", -1))
        for item in prompt_rows
        if not bool(item.get("missing", True))
        and (
            not isinstance(item.get("field_prompts"), Mapping)
            or any(not str(item["field_prompts"].get(field, "")).strip() for field in required_prompt_fields)
        )
    ]
    checks.add("valid_prototype_prompts_complete", len(prompt_rows) == 520 and not incomplete_prompts, {"rows": 520, "required_fields": list(required_prompt_fields)}, {"rows": len(prompt_rows), "incomplete": incomplete_prompts}, affected=incomplete_prompts)

    action_mapping = json.loads((root / "labels/action_id_mapping.json").read_text(encoding="utf-8"))
    object_mapping = json.loads((root / "labels/object_id_mapping.json").read_text(encoding="utf-8"))
    checks.add("label_id_mapping_sizes", len(action_mapping.get("items", [])) == 117 and len(object_mapping.get("items", [])) == 80, {"actions": 117, "objects": 80}, {"actions": len(action_mapping.get("items", [])), "objects": len(object_mapping.get("items", []))})

    before_hash = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))["prototype_bank_sha256_before"]
    after_hash = sha256_file(args.prototype_bank)
    integrity = {
        "path": str(args.prototype_bank.resolve()),
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "unchanged": before_hash == after_hash,
    }
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    (root / "provenance/prototype_bank_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checks.add("prototype_bank_unchanged", integrity["unchanged"], before_hash, after_hash)

    report = {
        "run_root": str(root),
        "passed": checks.passed,
        "check_count": len(checks.rows),
        "failed_check_count": sum(row["status"] == "FAIL" for row in checks.rows),
        "gt_mapping_valid_candidates": len(valid_gt_candidates),
        "max_residual_error": max_residual,
        "max_layer1_vs_final_diff": layer1_diff,
        "checks": checks.rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, args.output_json)
    print(json.dumps({key: report[key] for key in (
        "passed",
        "check_count",
        "failed_check_count",
        "gt_mapping_valid_candidates",
        "max_residual_error",
        "max_layer1_vs_final_diff",
    )}, indent=2))
    if not report["passed"]:
        for row in checks.rows:
            if row["status"] == "FAIL":
                print("FAIL:", row["name"], row["actual"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
