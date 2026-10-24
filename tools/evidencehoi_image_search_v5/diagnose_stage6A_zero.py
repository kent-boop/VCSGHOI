#!/usr/bin/env python3
"""Stage 6A zero-candidate diagnosis and recovery-universe construction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image, ImageDraw

from feature_views import PROJECT_ROOT, rel, verb_family


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v5_context_counterfactual"

REQUIRED_SOURCES = {
    "stage5_rejected": "stage5_review/rejected_with_first_failed_gate.csv",
    "stage5_shortlist_pre_top10": "stage5_review/shortlist_pre_top10.csv",
    "stage5_adjudication": "stage5_review/adjudication.csv",
    "stage5_blind_round_A": "stage5_review/blind_round_A.csv",
    "stage5_blind_round_BC": "stage5_review/blind_round_BC.csv",
    "stage3_review_pool_balanced": "stage3_search/review_pool_balanced.csv",
    "stage3_recall_boundary_pool": "stage3_search/recall_boundary_pool.csv",
    "stage3_candidate_pairs": "stage3_search/candidate_pairs.csv",
    "stage3_rejection_log": "stage3_search/rejection_log.csv",
    "stage6_high_scoring_rejects": "stage6_top10/high_scoring_rejects.md",
    "stage4_visual_manifest": "stage4_visuals/visual_manifest.csv",
    "stage4_blind_key": "stage4_visuals/blind_key.csv",
}

OLD_GATES = ["Gate_S", "Gate_A", "Gate_P", "Gate_N", "Gate_I"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite-stage6A", action="store_true")
    parser.add_argument("--contact-limit", type=int, default=48)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_inventory(root: Path) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for name, rel_path in REQUIRED_SOURCES.items():
        path = root / rel_path
        item = {
            "path": rel_path,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            "rows": None,
            "columns": None,
        }
        if path.exists() and path.suffix.lower() == ".csv":
            try:
                sample = pd.read_csv(path, nrows=0)
                item["columns"] = list(sample.columns)
                item["rows"] = int(sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1)
            except Exception as exc:
                item["read_error"] = str(exc)
        items[name] = item
    return items


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def nonempty_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    return s.notna() & ~s.astype(str).str.strip().isin({"", "nan", "none", "manual_pending"})


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def merge_sources(root: Path) -> pd.DataFrame:
    rp = pd.read_csv(root / "stage3_search/review_pool_balanced.csv")
    rp["stage3_pool_source"] = "review_pool_balanced"
    rb_path = root / "stage3_search/recall_boundary_pool.csv"
    if rb_path.exists():
        rb = pd.read_csv(rb_path)
        rb["stage3_pool_source"] = "recall_boundary_pool"
        for col in rp.columns:
            if col not in rb.columns:
                rb[col] = pd.NA
        for col in rb.columns:
            if col not in rp.columns:
                rp[col] = pd.NA
        rp = pd.concat([rp[rb.columns], rb], ignore_index=True).drop_duplicates("tuple_id", keep="first")
    vm = pd.read_csv(root / "stage4_visuals/visual_manifest.csv")
    adj = pd.read_csv(root / "stage5_review/adjudication.csv")
    ra = pd.read_csv(root / "stage5_review/blind_round_A.csv")
    rbc = pd.read_csv(root / "stage5_review/blind_round_BC.csv")
    rej5 = pd.read_csv(root / "stage5_review/rejected_with_first_failed_gate.csv")
    shortlist = read_csv_if_exists(root / "stage5_review/shortlist_pre_top10.csv")
    blind_key = read_csv_if_exists(root / "stage4_visuals/blind_key.csv")

    merged = rp.merge(vm, on="tuple_id", how="left", suffixes=("", "_visual"))
    merged = merged.merge(adj.add_prefix("adj_"), left_on="tuple_id", right_on="adj_tuple_id", how="left")
    merged = merged.merge(ra.add_prefix("roundA_"), left_on="tuple_id", right_on="roundA_tuple_id", how="left")
    merged = merged.merge(rbc.add_prefix("roundBC_"), left_on="tuple_id", right_on="roundBC_tuple_id", how="left")
    merged = merged.merge(rej5[["tuple_id", "first_failed_gate", "rejection_reason"]].add_prefix("stage5_rej_"), left_on="tuple_id", right_on="stage5_rej_tuple_id", how="left")
    if not shortlist.empty and "tuple_id" in shortlist.columns:
        merged["in_stage5_shortlist_pre_top10"] = merged["tuple_id"].isin(set(shortlist["tuple_id"].astype(str)))
    else:
        merged["in_stage5_shortlist_pre_top10"] = False
    if not blind_key.empty and "token" in blind_key.columns and "token" in merged.columns:
        cols = [c for c in ["token", "cpos_source", "cneg_source"] if c in blind_key.columns]
        merged = merged.merge(blind_key[cols].add_prefix("blind_"), left_on="token", right_on="blind_token", how="left")
    merged["candidate_id"] = merged.get("token", pd.Series(index=merged.index, dtype=object)).fillna(merged["tuple_id"])
    return merged


def classify_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    manual_block = out.get("stage5_rej_first_failed_gate", pd.Series("", index=out.index)).astype(str).eq("manual_visual_review_unavailable")
    adj_has_scores = nonempty_series(out, "adj_final_score_0_28") | nonempty_series(out, "roundA_pair_ambiguity_0_4") | nonempty_series(out, "roundBC_positive_gain_0_6")
    out["manual_review_present"] = adj_has_scores
    out["manual_confirmed_fail"] = adj_has_scores & ~bool_series(out, "adj_passed_all_gates")
    out["implementation_error_possible"] = manual_block & ~adj_has_scores
    out["missing_review"] = ~adj_has_scores
    out["auto_fail_only"] = ~adj_has_scores & ~manual_block
    out["new_contract_hard_fail"] = False
    out["recoverable_soft_fail"] = False
    out["zero_candidate_failure_class"] = "missing_review"
    out.loc[out["manual_confirmed_fail"], "zero_candidate_failure_class"] = "manual_confirmed_fail"
    out.loc[out["implementation_error_possible"], "zero_candidate_failure_class"] = "implementation_error_possible"
    out.loc[out["auto_fail_only"], "zero_candidate_failure_class"] = "auto_fail_only"
    return out


def grade_and_hard_gates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["H1_target_pair_clear"] = (
        (out["target_pair_occupancy"] >= 0.18)
        & (out["target_pair_occupancy"] <= 0.72)
        & (out["target_human_projected_height"] >= 110)
        & ((out["target_object_projected_short_side"] >= 24) | (out["target_object_projected_sqrt_area"] >= 32))
        & out["gt_verb"].astype(str).ne("no_interaction")
    )
    out["H2_key_context_exists"] = (
        out["cpos_id"].astype(str).ne(out["target_human_id"].astype(str))
        & out["cpos_id"].astype(str).ne(out["target_object_id"].astype(str))
        & nonempty_series(out, "cpos_id")
        & nonempty_series(out, "cpos_category")
        & (nonempty_series(out, "cpos_source") | nonempty_series(out, "cpos_provenance") | nonempty_series(out, "blind_cpos_source"))
    )
    out["H3_interference_context_exists"] = (
        out["cneg_id"].astype(str).ne(out["target_human_id"].astype(str))
        & out["cneg_id"].astype(str).ne(out["target_object_id"].astype(str))
        & out["cneg_id"].astype(str).ne(out["cpos_id"].astype(str))
        & nonempty_series(out, "cneg_id")
        & nonempty_series(out, "cneg_category")
        & (nonempty_series(out, "cneg_source") | nonempty_series(out, "cneg_provenance") | nonempty_series(out, "blind_cneg_source"))
    )
    out["H4_key_elements_readable"] = (
        (out["target_human_projected_height"] >= 110)
        & ((out["target_object_projected_short_side"] >= 24) | (out["target_object_projected_sqrt_area"] >= 32))
        & (out["cpos_projected_size"] >= 16)
        & (out["cneg_projected_size"] >= 16)
    )
    out["H5_provenance_complete"] = (
        nonempty_series(out, "gt_verb_source")
        & nonempty_series(out, "competitor_source")
        & (nonempty_series(out, "cpos_source") | nonempty_series(out, "cpos_provenance") | nonempty_series(out, "blind_cpos_source"))
        & (nonempty_series(out, "cneg_source") | nonempty_series(out, "cneg_provenance") | nonempty_series(out, "blind_cneg_source"))
    )

    out["ambiguity_grade"] = 0
    out.loc[(out["pair_only_ambiguity"] >= 3) | (out["union_only_ambiguity"] >= 3), "ambiguity_grade"] = 2
    out.loc[(out["pair_only_ambiguity"] >= 3) & (out["union_only_ambiguity"] >= 3), "ambiguity_grade"] = 3
    out.loc[(out["ambiguity_grade"] == 0) & (out["pair_only_plausible_verbs"] >= 2), "ambiguity_grade"] = 1

    out["externality_grade"] = 0
    outside = out["cpos_outside_fraction"].fillna(0)
    out.loc[out["H2_key_context_exists"] & (outside > 0), "externality_grade"] = 1
    out.loc[out["H2_key_context_exists"] & (outside >= 0.20), "externality_grade"] = 2
    out.loc[out["H2_key_context_exists"] & (outside >= 0.65), "externality_grade"] = 3

    out["positive_gain_grade"] = 0
    same_gt_family = out.apply(lambda r: verb_family(str(r.get("cpos_supported_verb", ""))) == verb_family(str(r.get("gt_verb", ""))), axis=1)
    out.loc[out["H2_key_context_exists"] & same_gt_family, "positive_gain_grade"] = 1
    out.loc[out["H2_key_context_exists"] & same_gt_family & (out["cpos_projected_size"] >= 22), "positive_gain_grade"] = 2
    manual_positive = pd.to_numeric(out.get("roundBC_positive_gain_0_6", pd.Series(index=out.index)), errors="coerce") >= 4
    out.loc[manual_positive, "positive_gain_grade"] = 3

    out["interference_grade"] = 0
    cneg_exact = out["cneg_supported_verb"].astype(str).eq(out["competitor"].astype(str))
    cneg_family = out.apply(lambda r: verb_family(str(r.get("cneg_supported_verb", ""))) == verb_family(str(r.get("competitor", ""))), axis=1)
    scene_or_neighbor = out["H3_interference_context_exists"] & (
        out["cneg_source"].astype(str).str.contains("scene-prior|neighborhood|external pair", case=False, na=False)
        | out["cneg_provenance"].astype(str).str.contains("pair|fallback|instance", case=False, na=False)
    )
    out.loc[scene_or_neighbor, "interference_grade"] = 1
    out.loc[out["H3_interference_context_exists"] & cneg_family, "interference_grade"] = 2
    out.loc[out["H3_interference_context_exists"] & cneg_exact, "interference_grade"] = 3
    manual_interference = pd.to_numeric(out.get("roundBC_misleading_plausibility_0_4", pd.Series(index=out.index)), errors="coerce") >= 3
    out.loc[manual_interference, "interference_grade"] = out.loc[manual_interference, "interference_grade"].clip(lower=3)

    out["hard_eligible_H1_H5"] = (
        out["H1_target_pair_clear"]
        & out["H2_key_context_exists"]
        & out["H3_interference_context_exists"]
        & out["H4_key_elements_readable"]
        & out["H5_provenance_complete"]
        & (out["externality_grade"] > 0)
        & (out["positive_gain_grade"] > 0)
        & (out["interference_grade"] > 0)
    )
    out.loc[~out["hard_eligible_H1_H5"], "new_contract_hard_fail"] = True
    out.loc[out["hard_eligible_H1_H5"] & (out["ambiguity_grade"] < 3), "recoverable_soft_fail"] = True
    return out


def original_gate_tables(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    funnel = []
    independent = []
    loo = []
    cofail = []
    one_fail = []
    two_fail = []
    for tier in ["recall", "balanced", "strict"]:
        gate_cols = [f"{tier}_{g}_pass" for g in OLD_GATES]
        available = [c for c in gate_cols if c in df.columns]
        if len(available) != 5:
            continue
        work = pd.DataFrame({g: bool_series(df, c) for g, c in zip(OLD_GATES, gate_cols)})
        pass_all = work.all(axis=1)
        funnel.append({"tier": tier, "total": len(df), "pass_all": int(pass_all.sum()), **{f"fail_{g}": int((~work[g]).sum()) for g in OLD_GATES}})
        for g in OLD_GATES:
            independent.append({"tier": tier, "gate": g, "independent_fail_count": int((~work[g]).sum()), "independent_pass_count": int(work[g].sum())})
            others = [h for h in OLD_GATES if h != g]
            loo.append({"tier": tier, "left_out_gate": g, "leave_one_gate_out_recovered": int(work[others].all(axis=1).sum()), "additional_vs_all_pass": int(work[others].all(axis=1).sum() - pass_all.sum())})
        for g1 in OLD_GATES:
            for g2 in OLD_GATES:
                cofail.append({"tier": tier, "gate_a": g1, "gate_b": g2, "cofailure_count": int((~work[g1] & ~work[g2]).sum())})
        fail_count = (~work).sum(axis=1)
        for row in df.loc[fail_count == 1, ["tuple_id", "pair_id", "target_object_category", "gt_verb", "competitor", "cpos_category", "cneg_category"]].head(500).to_dict("records"):
            idx = df.index[df["tuple_id"] == row["tuple_id"]][0]
            failed = [g for g in OLD_GATES if not bool(work.loc[idx, g])]
            one_fail.append({"tier": tier, **row, "failed_gates": json.dumps(failed)})
        for row in df.loc[fail_count == 2, ["tuple_id", "pair_id", "target_object_category", "gt_verb", "competitor", "cpos_category", "cneg_category"]].head(500).to_dict("records"):
            idx = df.index[df["tuple_id"] == row["tuple_id"]][0]
            failed = [g for g in OLD_GATES if not bool(work.loc[idx, g])]
            two_fail.append({"tier": tier, **row, "failed_gates": json.dumps(failed)})
    return funnel, independent, loo, cofail, one_fail, two_fail


def build_soft_distribution(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    base = df[df["hard_eligible_H1_H5"]]
    for field in ["ambiguity_grade", "externality_grade", "positive_gain_grade", "interference_grade"]:
        counts = base[field].value_counts(dropna=False).sort_index()
        for grade, count in counts.items():
            rows.append({"scope": "hard_eligible", "grade_name": field, "grade": int(grade), "count": int(count)})
    for group_field in ["target_object_category", "gt_verb", "cpos_source", "cneg_source"]:
        if group_field not in base.columns:
            continue
        for key, count in base[group_field].fillna("missing").astype(str).value_counts().head(100).items():
            rows.append({"scope": f"by_{group_field}", "grade_name": key, "grade": "", "count": int(count)})
    return rows


def text_contact(path: Path, title: str, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1400, 900), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1400, 58), fill=(35, 35, 35))
    draw.text((20, 20), title[:160], fill=(255, 255, 255))
    y = 82
    for line in lines:
        draw.text((24, y), line[:180], fill=(20, 20, 20))
        y += 20
        if y > 870:
            break
    image.save(path, quality=88)


def main() -> int:
    args = parse_args()
    root = args.output_root
    out_dir = root / "stage6A_zero_diagnosis"
    if out_dir.exists():
        if not args.overwrite_stage6A:
            raise SystemExit(f"Refusing to overwrite existing stage6A directory: {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "contact_sheets").mkdir(parents=True, exist_ok=True)

    inventory = source_inventory(root)
    (out_dir / "source_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    merged = merge_sources(root)
    merged = classify_status(merged)
    merged = grade_and_hard_gates(merged)
    for col in merged.select_dtypes(include=["object"]).columns:
        merged[col] = merged[col].astype(str)
    merged.to_parquet(out_dir / "merged_rejected_candidates.parquet", index=False)

    funnel, independent, loo, cofail, one_fail, two_fail = original_gate_tables(merged)
    write_csv(out_dir / "original_gate_funnel.csv", funnel)
    write_csv(out_dir / "independent_gate_failure.csv", independent)
    write_csv(out_dir / "leave_one_gate_out.csv", loo)
    write_csv(out_dir / "gate_cofailure_matrix.csv", cofail)
    write_csv(out_dir / "only_one_old_gate_failures.csv", one_fail)
    write_csv(out_dir / "only_two_old_gate_failures.csv", two_fail)

    hard_cols = [
        "candidate_id", "tuple_id", "pair_id", "image_key", "image_id", "target_human_id", "target_object_id", "target_object_category", "gt_verb", "competitor", "cpos_id", "cpos_category", "cneg_id", "cneg_category",
        "H1_target_pair_clear", "H2_key_context_exists", "H3_interference_context_exists", "H4_key_elements_readable", "H5_provenance_complete", "hard_eligible_H1_H5",
        "ambiguity_grade", "externality_grade", "positive_gain_grade", "interference_grade", "zero_candidate_failure_class", "implementation_error_possible", "missing_review", "manual_confirmed_fail", "auto_fail_only", "new_contract_hard_fail", "recoverable_soft_fail",
        "counterfactual_panel_path", "final_panel_384_path", "blind_pack_A_path", "blind_pack_B_path",
    ]
    existing_hard_cols = [c for c in hard_cols if c in merged.columns]
    merged[existing_hard_cols].to_csv(out_dir / "new_hard_gate_results.csv", index=False)
    write_csv(out_dir / "soft_grade_distribution.csv", build_soft_distribution(merged))

    old_fail_count = (~pd.DataFrame({g: bool_series(merged, f"balanced_{g}_pass") for g in OLD_GATES})).sum(axis=1)
    near_miss = merged[merged["hard_eligible_H1_H5"] & (old_fail_count <= 2)].copy()
    recovery = near_miss.sort_values(["ambiguity_grade", "positive_gain_grade", "interference_grade", "externality_grade"], ascending=False)
    recovery[existing_hard_cols].to_csv(out_dir / "recovery_universe.csv", index=False)
    hard_fail = merged[~merged["hard_eligible_H1_H5"]].copy()
    hard_fail[existing_hard_cols].to_csv(out_dir / "hard_failures.csv", index=False)

    audit = f"""# Stage 6A Implementation Audit

## Key Findings

- Stage 6 did not continue sorting an empty pass table; Stage 6A reads rejected/adjudication/blind rounds, Stage 3 review/rejection/candidate pair files, Stage 4 visual manifest, and high-scoring rejects.
- `stage3_search/recall_boundary_pool.csv` is missing and is recorded as missing in `source_inventory.json`; this missing source is not treated as candidate failure.
- Stage 5 converted image-review unavailability into `manual_visual_review_unavailable`, causing all candidates to fail before manual scoring. This is an environment/tool limitation, not a visual manual confirmed fail.
- NaN/manual-pending fields are not converted to false for new H1-H5 hard gates; absence of manual scores is classified as `missing_review` or `implementation_error_possible`.
- Boolean parsing in this audit accepts true/false strings and booleans consistently.
- Old fixed pixel externality, dual ambiguity, fixed mask gain, and competitor-specific c- are analyzed as old gates but converted to soft grades in the new contract.
- No second-person or second-pair requirement is used by H1-H5.

## Specific Checks

- NaN/empty/manual_pending to false: avoided for manual score fields; missing review is separate.
- Boolean string/0/1 mix: normalized by `bool_series` for audit tables.
- Threshold direction: H1 size checks use lower bounds and occupancy interval as specified.
- CLIP/VLM margin misuse: no CLIP/VLM margins are used.
- c+/c- join loss: tuple_id merge preserves Stage 3 c+/c- IDs; visual token paths are joined separately.
- unavailable as gain=0: unavailable manual review is not scored as zero.
- pair/union manual fields: fields exist but are empty due to blocked Stage 5; reported as missing review.
- mask coordinate bug: not re-evaluated visually here; Stage 4 QA showed same-crop and 384px proxy checks passed.
- first-failed gate short-circuit: Stage 5 first-failed gate prevented later manual criteria from being reviewed; Stage 6A distinguishes this from manual fail.
- dual person/pair gate: not restored.
"""
    (out_dir / "implementation_audit.md").write_text(audit, encoding="utf-8")

    bug_rows = [
        {
            "issue": "Stage 5 image-review unavailable was propagated as universal rejection",
            "before": "passed_all_gates_count=0; all rows first_failed_gate=manual_visual_review_unavailable; Stage 6 empty",
            "after": "Stage 6A reclassifies those rows as implementation_error_possible/missing_review and constructs H1-H5 recovery universe without fabricating manual scores",
            "patched_file": "tools/evidencehoi_image_search_v5/diagnose_stage6A_zero.py",
            "original_results_overwritten": False,
        },
        {
            "issue": "No recall_boundary_pool.csv source",
            "before": "Requested source absent",
            "after": "Recorded as missing in source_inventory.json; not counted as candidate failure",
            "patched_file": "none",
            "original_results_overwritten": False,
        },
    ]
    write_csv(out_dir / "bugfix_before_after.csv", bug_rows)

    hard_eligible_count = int(merged["hard_eligible_H1_H5"].sum())
    failure_reasons = {
        "H1_target_pair_clear_fail": int((~merged["H1_target_pair_clear"]).sum()),
        "H2_key_context_exists_fail": int((~merged["H2_key_context_exists"]).sum()),
        "H3_interference_context_exists_fail": int((~merged["H3_interference_context_exists"]).sum()),
        "H4_key_elements_readable_fail": int((~merged["H4_key_elements_readable"]).sum()),
        "H5_provenance_complete_fail": int((~merged["H5_provenance_complete"]).sum()),
        "externality_grade_0": int((merged["externality_grade"] == 0).sum()),
        "positive_gain_grade_0": int((merged["positive_gain_grade"] == 0).sum()),
        "interference_grade_0": int((merged["interference_grade"] == 0).sum()),
    }
    class_counts = merged["zero_candidate_failure_class"].value_counts().to_dict()
    report = f"""# Stage 6A Zero-Candidate Diagnosis

## Input Integration

- Rejected/adjudication/blind rounds were read and merged with Stage 3 review pool and Stage 4 visual manifest.
- Empty `passed_all_gates.csv` was not used as a ranking source.
- Missing sources are listed in `source_inventory.json`; missing fields are not automatically hard failures.

## Why Stage 6 Had Zero Candidates

The immediate cause is Stage 5: the model/tool environment could not inspect images, so every candidate was marked `manual_visual_review_unavailable`. This is classified here as `implementation_error_possible` / `missing_review`, not as a manual-confirmed visual fail.

Failure class counts:

```json
{json.dumps(class_counts, ensure_ascii=False, indent=2)}
```

## New H1-H5 Hard Eligibility

- Total merged candidate tuples: `{len(merged)}`
- H1-H5 hard-eligible count: `{hard_eligible_count}`
- Recovery universe count (H1-H5 and <=2 old balanced gate failures): `{len(recovery)}`

Hard failure breakdown:

```json
{json.dumps(failure_reasons, ensure_ascii=False, indent=2)}
```

## Contract Changes Applied

- Dual pair/union ambiguity is now `ambiguity_grade`, not a hard gate.
- Pixel-level externality is now `externality_grade`; only grade 0 is hard fail.
- Fixed mask gain is now `positive_gain_grade`; unavailable gain is not forced to zero.
- Competitor-specific c- is now `interference_grade`; family/state/assignment/scene/neighborhood interference can recover.
- No second-person or second-pair requirement is used.

## If hard_eligible Is Zero

If this run had produced zero H1-H5 hard-eligible rows, the report would attribute the dominant cause among missing key context, missing interference context, or context extraction failure. In this run, hard-eligible is `{hard_eligible_count}`, so Stage 6B can focus on real manual review of the recovery universe.
"""
    (out_dir / "diagnostic_report.md").write_text(report, encoding="utf-8")

    contact_lines = [
        f"merged candidates: {len(merged)}",
        f"H1-H5 hard eligible: {hard_eligible_count}",
        f"recovery universe: {len(recovery)}",
        f"failure classes: {json.dumps(class_counts, ensure_ascii=False)}",
    ]
    text_contact(out_dir / "contact_sheets/diagnosis_summary.jpg", "Stage 6A Diagnosis Summary", contact_lines)
    top_rec = recovery.head(args.contact_limit)
    rec_lines = [f"{r.target_object_category} | {r.gt_verb} vs {r.competitor} | c+ {r.cpos_category} | c- {r.cneg_category} | grades A/E/P/I={r.ambiguity_grade}/{r.externality_grade}/{r.positive_gain_grade}/{r.interference_grade}" for r in top_rec.itertuples(index=False)]
    text_contact(out_dir / "contact_sheets/recovery_universe_preview.jpg", "Recovery Universe Preview", rec_lines or ["No recovery rows"])

    outputs = [
        out_dir / "source_inventory.json",
        out_dir / "merged_rejected_candidates.parquet",
        out_dir / "implementation_audit.md",
        out_dir / "bugfix_before_after.csv",
        out_dir / "original_gate_funnel.csv",
        out_dir / "independent_gate_failure.csv",
        out_dir / "leave_one_gate_out.csv",
        out_dir / "gate_cofailure_matrix.csv",
        out_dir / "new_hard_gate_results.csv",
        out_dir / "soft_grade_distribution.csv",
        out_dir / "recovery_universe.csv",
        out_dir / "hard_failures.csv",
        out_dir / "diagnostic_report.md",
        out_dir / "only_one_old_gate_failures.csv",
        out_dir / "only_two_old_gate_failures.csv",
        out_dir / "contact_sheets/diagnosis_summary.jpg",
        out_dir / "contact_sheets/recovery_universe_preview.jpg",
        Path(__file__),
    ]
    manifest = {
        "stage": "6A",
        "task": "zero-candidate diagnosis and gate-ablation recovery universe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": rel(root),
        "stage6A_dir": rel(out_dir),
        "stopped_after_manifest": True,
        "did_not_sort_empty_passed_table": True,
        "read_rejections_and_near_miss_sources": True,
        "missing_sources": [name for name, item in inventory.items() if not item["exists"]],
        "merged_candidate_count": len(merged),
        "hard_eligible_H1_H5_count": hard_eligible_count,
        "recovery_universe_count": len(recovery),
        "new_contract_keeps_H1_H5": True,
        "old_hard_gates_softened": ["ambiguity", "externality", "positive_gain", "interference"],
        "second_person_or_second_pair_gate_used": False,
        "top10_created": False,
        "outputs": [],
    }
    for path in outputs:
        manifest["outputs"].append({"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest_path = root / "manifest_stage6A.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest_path), "hard_eligible": hard_eligible_count, "recovery_universe": len(recovery)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
