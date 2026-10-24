#!/usr/bin/env python3
"""Emit Stage 5 review artifacts when visual manual review is unavailable.

This script intentionally does not fabricate human visual scores. It preserves
candidate-tuple granularity, Blind Pack A/B paths, and first-failed-gate records
so that a real reviewer can fill the review later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from feature_views import PROJECT_ROOT, rel


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v5_context_counterfactual"
DEFAULT_STAGE4_DIR = DEFAULT_OUTPUT_ROOT / "stage4_visuals"

ROUND_A_COLUMNS = [
    "token",
    "tuple_id",
    "pair_id",
    "target_object_category",
    "gt_verb",
    "competitor",
    "blind_pack_A_path",
    "gt_reasonable_0_1",
    "competitor_reasonable_0_1",
    "pair_only_gt_support_0_5",
    "pair_only_competitor_support_0_5",
    "union_only_gt_support_0_5",
    "union_only_competitor_support_0_5",
    "pair_ambiguity_0_4",
    "union_ambiguity_0_4",
    "round_A_pass",
    "round_A_first_failed_gate",
    "round_A_notes",
]

ROUND_BC_COLUMNS = [
    "token",
    "tuple_id",
    "pair_id",
    "target_object_category",
    "gt_verb",
    "competitor",
    "cpos_id",
    "cpos_category",
    "cneg_id",
    "cneg_category",
    "blind_pack_B_path",
    "cpos_outside_pair_0_1",
    "cpos_gt_support_explanation",
    "cpos_maskout_weakens_gt_0_1",
    "cpos_common_cooccurrence_only_0_1",
    "positive_gain_0_6",
    "round_B_pass",
    "cneg_competitor_target",
    "cneg_scene_or_neighborhood_explanation",
    "cneg_addition_makes_competitor_more_reasonable_0_1",
    "cneg_maskout_weakens_misleading_prior_0_1",
    "misleading_plausibility_0_4",
    "round_C_pass",
    "round_BC_first_failed_gate",
    "round_BC_notes",
]

ADJ_COLUMNS = [
    "token",
    "tuple_id",
    "pair_id",
    "target_object_category",
    "gt_verb",
    "competitor",
    "cpos_category",
    "cneg_category",
    "target_pair_subjectivity_0_4",
    "pair_union_ambiguity_0_4",
    "cpos_spatial_externality_0_3",
    "cpos_counterfactual_gain_0_6",
    "cneg_misleading_plausibility_0_4",
    "selectivity_contrast_0_5",
    "drawing_feasibility_0_2",
    "final_score_0_28",
    "pair_ambiguity_pass",
    "union_ambiguity_pass",
    "positive_gain_pass",
    "misleading_plausibility_pass",
    "selectivity_pass",
    "final_score_pass",
    "all_minima_pass",
    "passed_all_gates",
    "first_failed_gate",
    "required_sentence",
    "adjudication_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage4-dir", type=Path, default=DEFAULT_STAGE4_DIR)
    parser.add_argument("--overwrite-stage5", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_file():
        raise RuntimeError(f"Output path is not a file: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    stage5_dir = args.output_root / "stage5_review"
    if stage5_dir.exists() and not args.overwrite_stage5:
        raise SystemExit(f"Refusing to overwrite existing stage5 directory: {stage5_dir}")
    stage5_dir.mkdir(parents=True, exist_ok=True)

    visual_manifest_path = args.stage4_dir / "visual_manifest.csv"
    blind_key_path = args.stage4_dir / "blind_key.csv"
    if not visual_manifest_path.exists():
        raise FileNotFoundError(visual_manifest_path)
    visual = pd.read_csv(visual_manifest_path)
    blind_key = pd.read_csv(blind_key_path)
    merged = visual.merge(
        blind_key[["token", "cpos_source", "cneg_source"]],
        on="token",
        how="left",
    )

    blocker = "manual_visual_review_unavailable"
    blocker_note = (
        "The current model/tool environment cannot inspect image contents. "
        "A tool image read returned 'Cannot read image (this model does not support image input)'. "
        "Stage 5 requires human visual judgments from Blind Pack A/B, so scores are not fabricated."
    )

    round_a_rows: list[dict[str, Any]] = []
    round_bc_rows: list[dict[str, Any]] = []
    adj_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        base = {
            "token": row.token,
            "tuple_id": row.tuple_id,
            "pair_id": row.pair_id,
            "target_object_category": row.target_object_category,
            "gt_verb": row.gt_verb,
            "competitor": row.competitor,
            "cpos_id": row.cpos_id,
            "cpos_category": row.cpos_category,
            "cneg_id": row.cneg_id,
            "cneg_category": row.cneg_category,
            "blind_pack_A_path": row.blind_pack_A_path,
            "blind_pack_B_path": row.blind_pack_B_path,
        }
        round_a_rows.append({
            **base,
            "gt_reasonable_0_1": "",
            "competitor_reasonable_0_1": "",
            "pair_only_gt_support_0_5": "",
            "pair_only_competitor_support_0_5": "",
            "union_only_gt_support_0_5": "",
            "union_only_competitor_support_0_5": "",
            "pair_ambiguity_0_4": "",
            "union_ambiguity_0_4": "",
            "round_A_pass": False,
            "round_A_first_failed_gate": blocker,
            "round_A_notes": blocker_note,
        })
        round_bc_rows.append({
            **base,
            "cpos_outside_pair_0_1": "",
            "cpos_gt_support_explanation": "",
            "cpos_maskout_weakens_gt_0_1": "",
            "cpos_common_cooccurrence_only_0_1": "",
            "positive_gain_0_6": "",
            "round_B_pass": False,
            "cneg_competitor_target": row.competitor,
            "cneg_scene_or_neighborhood_explanation": "",
            "cneg_addition_makes_competitor_more_reasonable_0_1": "",
            "cneg_maskout_weakens_misleading_prior_0_1": "",
            "misleading_plausibility_0_4": "",
            "round_C_pass": False,
            "round_BC_first_failed_gate": blocker,
            "round_BC_notes": blocker_note,
        })
        adj = {
            **base,
            "target_pair_subjectivity_0_4": "",
            "pair_union_ambiguity_0_4": "",
            "cpos_spatial_externality_0_3": "",
            "cpos_counterfactual_gain_0_6": "",
            "cneg_misleading_plausibility_0_4": "",
            "selectivity_contrast_0_5": "",
            "drawing_feasibility_0_2": "",
            "final_score_0_28": "",
            "pair_ambiguity_pass": False,
            "union_ambiguity_pass": False,
            "positive_gain_pass": False,
            "misleading_plausibility_pass": False,
            "selectivity_pass": False,
            "final_score_pass": False,
            "all_minima_pass": False,
            "passed_all_gates": False,
            "first_failed_gate": blocker,
            "required_sentence": "",
            "adjudication_notes": blocker_note,
        }
        adj_rows.append(adj)
        reject_rows.append({
            **adj,
            "rejection_reason": blocker_note,
        })

    write_csv(stage5_dir / "blind_round_A.csv", round_a_rows, ROUND_A_COLUMNS)
    write_csv(stage5_dir / "blind_round_BC.csv", round_bc_rows, ROUND_BC_COLUMNS)
    write_csv(stage5_dir / "adjudication.csv", adj_rows, ADJ_COLUMNS)
    write_csv(stage5_dir / "passed_all_gates.csv", [], ADJ_COLUMNS)
    write_csv(stage5_dir / "rejected_with_first_failed_gate.csv", reject_rows, ADJ_COLUMNS + ["rejection_reason"])
    write_csv(stage5_dir / "shortlist_pre_top10.csv", [], ADJ_COLUMNS)

    report = f"""# Stage 5 Manual Counterfactual Review Report

## Status

Stage 5 is blocked in this environment because the current model cannot inspect image contents. Attempting to read Blind/visual panels returned: `Cannot read image (this model does not support image input)`.

No Round A-D visual scores were fabricated. All candidate tuples are recorded with `first_failed_gate={blocker}`.

## Inputs

- Visual manifest: `{rel(visual_manifest_path)}`
- Blind key: `{rel(blind_key_path)}`
- Candidate tuple count: `{len(merged)}`

## Required Manual Criteria Not Executed

- Round A: Pair/union ambiguity visual judgment.
- Round B: Positive context counterfactual gain.
- Round C: Misleading context plausibility.
- Round D: Selectivity and 384 px readability.

## Outputs

- `blind_round_A.csv`: template rows with Blind Pack A paths and blocked status.
- `blind_round_BC.csv`: template rows with Blind Pack B paths and blocked status.
- `adjudication.csv`: candidate-level blocked adjudication rows.
- `passed_all_gates.csv`: empty, because no real manual review was possible.
- `rejected_with_first_failed_gate.csv`: all candidates with `first_failed_gate={blocker}`.
- `shortlist_pre_top10.csv`: empty.

## Integrity Statement

This stage does not use proxy scores, geometry scores, or total-score compensation to replace human visual review. The required sentence for pass candidates is intentionally blank because there are no pass candidates.
"""
    report_path = stage5_dir / "manual_review_report.md"
    report_path.write_text(report, encoding="utf-8")

    outputs = [
        stage5_dir / "blind_round_A.csv",
        stage5_dir / "blind_round_BC.csv",
        stage5_dir / "adjudication.csv",
        stage5_dir / "passed_all_gates.csv",
        stage5_dir / "rejected_with_first_failed_gate.csv",
        stage5_dir / "shortlist_pre_top10.csv",
        report_path,
        Path(__file__),
    ]
    manifest = {
        "stage": 5,
        "task": "manual counterfactual review using Blind Pack A/B",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": rel(args.output_root),
        "stage5_dir": rel(stage5_dir),
        "stopped_after_manifest": True,
        "manual_visual_review_completed": False,
        "blocker": blocker,
        "blocker_detail": blocker_note,
        "candidate_tuple_count": len(merged),
        "passed_all_gates_count": 0,
        "rejected_count": len(merged),
        "scores_fabricated": False,
        "proxy_scores_used_for_manual_review": False,
        "all_rejections_have_first_failed_gate": True,
        "outputs": [],
    }
    for path in outputs:
        manifest["outputs"].append({"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest_path = args.output_root / "manifest_stage5.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest_path), "candidate_tuple_count": len(merged), "passed_all_gates_count": 0, "blocker": blocker}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
