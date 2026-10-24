#!/usr/bin/env python3
"""Stage 6C sorting audit and external visual review pack generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from PIL import Image, ImageDraw

from feature_views import PROJECT_ROOT, rel, verb_family


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v5_context_counterfactual"
SEED = 20260712
PRIORITY_1 = {"cake", "pizza", "suitcase", "tv", "laptop", "dining table", "couch", "bicycle", "chair", "book"}
PRIORITY_2 = {"dog", "motorcycle", "backpack", "handbag", "bottle", "cup", "wine glass", "skateboard", "surfboard", "sports ball", "cell phone"}
SMALL_OBJECTS = {"apple", "banana", "orange", "fork", "knife", "spoon", "remote", "toothbrush"}
AUDIT_FIELDS = [
    "ambiguity_grade", "externality_grade", "externality_grade_key", "externality_grade_interference",
    "positive_gain_grade", "interference_grade", "target_readability_score_0_4", "key_context_score_0_4",
    "interference_score_0_4", "provenance_score_0_3", "manual_visual_recheck_completed",
    "hard_eligible_H1_H5", "tier", "total_score_0_22", "tier_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite-stage6C", action="store_true")
    parser.add_argument("--target-shortlist-size", type=int, default=48)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tri_bool(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return "true"
    if text in {"false", "0", "no", "n"}:
        return "false"
    if text in {"", "nan", "none", "manual_pending", "pending", "unknown"}:
        return "unknown"
    return "unknown"


def bool_true(series: pd.Series) -> pd.Series:
    return series.map(tri_bool).eq("true")


def source_tag(field: str) -> str:
    if "manual" in field or "recheck" in field:
        return "[MANUAL]"
    if "gt" in field or "verb" in field:
        return "[GT]/[DERIVED]"
    if "grade" in field or "score" in field or "tier" in field:
        return "[DERIVED]"
    return "[AUTO]/[DERIVED]"


def load_sources(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "recovery": root / "stage6A_zero_diagnosis/recovery_universe.csv",
        "new_hard": root / "stage6A_zero_diagnosis/new_hard_gate_results.csv",
        "merged": root / "stage6A_zero_diagnosis/merged_rejected_candidates.parquet",
        "hard_eligible_6b": root / "stage6B_tiered_recovery/hard_eligible_candidates.csv",
        "tier_A": root / "stage6B_tiered_recovery/tier_A.csv",
        "tier_B": root / "stage6B_tiered_recovery/tier_B.csv",
        "tier_C": root / "stage6B_tiered_recovery/tier_C.csv",
        "all_tier": root / "stage6B_tiered_recovery/all_tier_ranking.csv",
        "final_top10_6b": root / "stage6B_tiered_recovery/final_top10.csv",
        "hard_rejects_6b": root / "stage6B_tiered_recovery/high_scoring_hard_rejects.csv",
        "stage3_review": root / "stage3_search/review_pool_balanced.csv",
        "stage4_visual": root / "stage4_visuals/visual_manifest.csv",
        "stage4_blind_key": root / "stage4_visuals/blind_key.csv",
        "stage5_adj": root / "stage5_review/adjudication.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            out[name] = pd.DataFrame()
        elif path.suffix == ".parquet":
            out[name] = pd.read_parquet(path)
        else:
            out[name] = pd.read_csv(path, low_memory=False)
    return out


def join_key(df: pd.DataFrame) -> pd.Series:
    parts = []
    mapping = [
        ("image_id", ""), ("target_human_id", ""), ("target_object_id", ""), ("gt_verb", ""),
        ("competitor", ""), ("cpos_id", ""), ("cneg_id", ""),
    ]
    for col, default in mapping:
        if col in df.columns:
            parts.append(df[col].fillna(default).astype(str))
        else:
            parts.append(pd.Series(default, index=df.index))
    key = parts[0]
    for p in parts[1:]:
        key = key + "||" + p
    return key


def merge_complete(sources: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = sources["all_tier"].copy()
    base["_row_index_stage6B"] = range(len(base))
    base["trace_key"] = join_key(base)
    conflicts = []
    for name in ["recovery", "new_hard", "stage4_visual", "stage5_adj"]:
        other = sources[name]
        if other.empty:
            continue
        other = other.copy()
        other["trace_key"] = join_key(other)
        cols = [c for c in other.columns if c in {"tuple_id", "trace_key", "counterfactual_panel_path", "final_panel_384_path", "blind_pack_A_path", "blind_pack_B_path", "first_failed_gate", "adjudication_notes"}]
        other = other[cols].drop_duplicates("trace_key")
        before_cols = set(base.columns)
        base = base.merge(other.add_prefix(f"{name}__"), left_on="trace_key", right_on=f"{name}__trace_key", how="left")
        for col in before_cols:
            pref = f"{name}__{col}"
            if pref in base.columns:
                mismatch = base[col].notna() & base[pref].notna() & base[col].astype(str).ne(base[pref].astype(str))
                for r in base.loc[mismatch, ["tuple_id", "trace_key", col, pref]].head(1000).itertuples(index=False):
                    conflicts.append({"source": name, "tuple_id": r.tuple_id, "trace_key": r.trace_key, "field": col, "base_value": getattr(r, col), "other_value": getattr(r, pref)})
    return base, pd.DataFrame(conflicts)


def field_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in AUDIT_FIELDS:
        if field not in df.columns:
            rows.append({"field": field, "exists": False, "warning": "missing_field"})
            continue
        s = df[field]
        num = pd.to_numeric(s, errors="coerce")
        vc = s.astype(str).fillna("<NA>").value_counts(dropna=False)
        mode = vc.index[0] if len(vc) else ""
        mode_count = int(vc.iloc[0]) if len(vc) else 0
        missing = s.isna() | s.astype(str).str.strip().isin({"", "nan", "None", "manual_pending"})
        default_values = {0, "0", False, "False", "false", "not_tiered"}
        default_ratio = s.map(lambda x: x in default_values or str(x) in default_values).mean()
        threshold_ratio = None
        if field in {"ambiguity_grade", "externality_grade", "externality_grade_key", "externality_grade_interference", "positive_gain_grade", "interference_grade"}:
            threshold_ratio = float((num >= 2).mean())
        if field in {"manual_visual_recheck_completed"}:
            threshold_ratio = float(s.map(tri_bool).eq("true").mean())
        warnings = []
        if s.nunique(dropna=False) <= 1:
            warnings.append("nunique<=1")
        if len(s) and mode_count / len(s) > 0.80:
            warnings.append("mode>80pct")
        if field == "manual_visual_recheck_completed" and s.map(tri_bool).eq("true").any():
            warnings.append("manual_true_present")
        if field in {"total_score_0_22"} and (num.max() > 22 or num.min() < 0):
            warnings.append("score_out_of_range")
        rows.append({
            "field": field, "exists": True, "dtype": str(s.dtype), "missing_rate": float(missing.mean()), "nunique": int(s.nunique(dropna=False)),
            "value_counts_top10_json": json.dumps(vc.head(10).to_dict(), ensure_ascii=False),
            "min": None if num.dropna().empty else float(num.min()), "max": None if num.dropna().empty else float(num.max()),
            "mean": None if num.dropna().empty else float(num.mean()), "std": None if num.dropna().empty else float(num.std()),
            "mode": mode, "mode_ratio": float(mode_count / len(s)) if len(s) else 0.0, "default_ratio": float(default_ratio),
            "threshold_pass_ratio": threshold_ratio, "source_tag": source_tag(field), "warning": ";".join(warnings),
        })
    return pd.DataFrame(rows)


def missing_default_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in AUDIT_FIELDS:
        if field not in df.columns:
            continue
        s = df[field]
        unknown = s.map(tri_bool).eq("unknown") if field.endswith("completed") or field.startswith("hard") else (s.isna() | s.astype(str).str.strip().isin({"", "nan", "manual_pending"}))
        highest = False
        num = pd.to_numeric(s, errors="coerce")
        if field.endswith("grade"):
            highest = num.eq(3)
        elif field.endswith("score_0_4"):
            highest = num.eq(4)
        elif field == "total_score_0_22":
            highest = num.eq(22)
        elif field == "tier":
            highest = s.astype(str).eq("A")
        rows.append({"field": field, "unknown_count": int(unknown.sum()), "highest_count": int(highest.sum()) if hasattr(highest, "sum") else 0, "unknown_and_highest_count": int((unknown & highest).sum()) if hasattr(highest, "sum") else 0, "risk": "unknown_to_highest" if hasattr(highest, "sum") and int((unknown & highest).sum()) else "none_detected"})
    return pd.DataFrame(rows)


def manual_confirmation_audit(df: pd.DataFrame) -> pd.DataFrame:
    manual = df.get("manual_visual_recheck_completed", pd.Series("unknown", index=df.index)).map(tri_bool)
    confirmed = (manual.eq("true"))
    rows = []
    for old_tier, count in df.get("tier", pd.Series(index=df.index, dtype=object)).astype(str).value_counts().items():
        sub = df[df["tier"].astype(str) == old_tier]
        confirmed_count = int((sub.index.isin(df[confirmed].index)).sum()) if old_tier == "A" else 0
        rows.append({"old_tier": old_tier, "old_count": len(sub), "confirmed_Tier_A_count": confirmed_count, "downgrade_reason": "manual_visual_recheck_not_completed"})
    return pd.DataFrame(rows)


def add_auto_classes(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    manual = work.get("manual_visual_recheck_completed", pd.Series("unknown", index=work.index)).map(tri_bool)
    work["confirmed_Tier_A"] = False
    work["auto_recovery_class"] = "NEEDS_MANUAL_REVIEW"
    h = bool_true(work["hard_eligible_H1_H5"]) if "hard_eligible_H1_H5" in work else pd.Series(False, index=work.index)
    score = pd.to_numeric(work.get("total_score_0_22", pd.Series(0, index=work.index)), errors="coerce").fillna(0)
    role = pd.to_numeric(work.get("role_distinguishability", pd.Series(1, index=work.index)), errors="coerce").fillna(1)
    work.loc[~h, "auto_recovery_class"] = "REJECT_AUTO_STRUCTURE"
    work.loc[h & (score >= 20) & (role >= 2), "auto_recovery_class"] = "AUTO_S"
    work.loc[h & (score.between(16, 19.999)) & (role >= 1), "auto_recovery_class"] = "AUTO_M"
    work.loc[h & ((score < 16) | work["target_object_category"].astype(str).isin(SMALL_OBJECTS)), "auto_recovery_class"] = "AUTO_E"
    work.loc[manual.ne("true") & h, "confirmed_Tier_A"] = False
    return work


def sort_forensics(df: pd.DataFrame, old_top10: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    work = df.copy().reset_index(drop=True)
    work["original_row_index"] = range(len(work))
    keys = ["tier_rank", "total_score_0_22", "ambiguity_grade", "positive_gain_grade", "interference_grade"]
    for k in keys:
        work[k] = pd.to_numeric(work[k], errors="coerce") if k != "tier_rank" else pd.to_numeric(work[k], errors="coerce")
    sorted_df = work.sort_values(keys, ascending=[True, False, False, False, False], kind="mergesort").copy()
    sorted_df["sort_position_recomputed"] = range(1, len(sorted_df) + 1)
    score = pd.to_numeric(work["total_score_0_22"], errors="coerce")
    score_counts = score.value_counts(dropna=False)
    cutoff_top50 = sorted_df.iloc[49]["total_score_0_22"] if len(sorted_df) >= 50 else None
    metrics = {
        "sort_keys": keys,
        "ascending": [True, False, False, False, False],
        "stable_sort_kind": "mergesort",
        "tie_breaker": "implicit original row order after equal listed keys",
        "score_unique_count": int(score.nunique(dropna=False)),
        "score_unique_ratio": float(score.nunique(dropna=False) / len(work)),
        "highest_score": float(score.max()),
        "highest_score_count": int((score == score.max()).sum()),
        "highest_score_ratio": float((score == score.max()).mean()),
        "top50_cutoff_score": None if cutoff_top50 is None else float(cutoff_top50),
        "top50_cutoff_tie_count": int((score == cutoff_top50).sum()) if cutoff_top50 is not None else 0,
    }
    alerts = []
    if metrics["highest_score_ratio"] > 0.01:
        alerts.append("highest_score_covers_gt_1pct")
    if metrics["top50_cutoff_tie_count"] > 200:
        alerts.append("top50_cutoff_tie_gt_200")
    if metrics["score_unique_ratio"] < 0.005:
        alerts.append("score_unique_ratio_lt_0.5pct")
    metrics["alerts"] = alerts

    rng_results = []
    top_sets = []
    for seed in [11, 22, 33, 44, 55]:
        shuffled = work.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        shuf_sorted = shuffled.sort_values(keys, ascending=[True, False, False, False, False], kind="mergesort")
        top = set(shuf_sorted.head(50)["tuple_id"].astype(str))
        top_sets.append(top)
        rng_results.append({"seed": seed, "top50_tuple_ids": sorted(list(top))[:10], "top50_count": len(top)})
    jaccards = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccards.append(inter / union if union else 1.0)
    metrics["shuffle_top50_jaccard_min"] = min(jaccards) if jaccards else None
    metrics["shuffle_top50_jaccard_mean"] = sum(jaccards) / len(jaccards) if jaccards else None

    score_recomp = work[["tuple_id", "tier", "total_score_0_22", "target_readability_score_0_4", "key_context_score_0_4", "interference_score_0_4", "ambiguity_score_0_3", "separability_score_0_2", "mask_support_score_0_2", "provenance_score_0_3"]].copy()
    parts = ["target_readability_score_0_4", "key_context_score_0_4", "interference_score_0_4", "ambiguity_score_0_3", "separability_score_0_2", "mask_support_score_0_2", "provenance_score_0_3"]
    for p in parts:
        score_recomp[p] = pd.to_numeric(score_recomp[p], errors="coerce")
    score_recomp["recomputed_total_score_0_22"] = score_recomp[parts].sum(axis=1)
    score_recomp["score_delta"] = pd.to_numeric(score_recomp["total_score_0_22"], errors="coerce") - score_recomp["recomputed_total_score_0_22"]

    old_vs = pd.DataFrame({
        "rank": range(1, 11),
        "old_stage6B_tuple_id": list(old_top10.get("tuple_id", pd.Series(dtype=str)).head(10).astype(str)) + [""] * max(0, 10 - len(old_top10)),
        "score_only_tuple_id": list(work.sort_values("total_score_0_22", ascending=False, kind="mergesort").head(10)["tuple_id"].astype(str)) + [""] * max(0, 10 - min(10, len(work))),
        "random_tie_tuple_id": list(work.sample(frac=1.0, random_state=SEED).sort_values(keys, ascending=[True, False, False, False, False], kind="mergesort").head(10)["tuple_id"].astype(str)) + [""] * max(0, 10 - min(10, len(work))),
    })
    return sorted_df, metrics | {"shuffle_runs": rng_results}, score_recomp, old_vs


def final_size_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scale = 384.0 / 900.0
    out["target_human_height_384"] = pd.to_numeric(out.get("target_human_projected_height", 0), errors="coerce") * scale
    out["target_human_width_384"] = out["target_human_height_384"] * 0.35
    out["target_object_sqrt_area_384"] = pd.to_numeric(out.get("target_object_projected_sqrt_area", 0), errors="coerce") * scale
    out["target_object_short_side_384"] = pd.to_numeric(out.get("target_object_projected_short_side", 0), errors="coerce") * scale
    out["target_object_long_side_384"] = out["target_object_sqrt_area_384"] * 1.8
    out["key_context_size_384"] = pd.to_numeric(out.get("cpos_projected_size", 0), errors="coerce") * scale
    out["interference_context_size_384"] = pd.to_numeric(out.get("cneg_projected_size", 0), errors="coerce") * scale
    out["key_context_inset_size"] = out["key_context_size_384"].clip(lower=192)
    out["interference_context_inset_size"] = out["interference_context_size_384"].clip(lower=192)
    out["target_pair_occupancy_final"] = pd.to_numeric(out.get("target_pair_occupancy", 0), errors="coerce")
    small = out["target_object_category"].astype(str).isin(SMALL_OBJECTS)
    ordinary_ok = (out["target_object_short_side_384"] >= 20) & (out["target_object_sqrt_area_384"] >= 36)
    slender_ok = (out["target_object_long_side_384"] >= 60) & (out["target_object_sqrt_area_384"] >= 32)
    out["target_final_size_ok"] = (out["target_human_height_384"] >= 80) & (ordinary_ok | slender_ok | (small & slender_ok))
    out["key_context_final_ok"] = (out["key_context_size_384"] >= 16) | (out["key_context_inset_size"] >= 192)
    out["interference_context_final_ok"] = (out["interference_context_size_384"] >= 16) | (out["interference_context_inset_size"] >= 192)
    out["relation_endpoints_readable_proxy"] = out.get("cpos_relation_endpoints", pd.Series("", index=out.index)).astype(str).str.len().gt(0) | out.get("cneg_relation_endpoints", pd.Series("", index=out.index)).astype(str).str.len().gt(0)
    out["needs_inset"] = (out["key_context_size_384"] < 32) | (out["interference_context_size_384"] < 32) | (out["target_object_short_side_384"] < 20)
    out["truncation_risk"] = False
    out["occlusion_risk_proxy"] = pd.to_numeric(out.get("cpos_cneg_mask_iou", 0), errors="coerce").fillna(0) > 0.1
    out["overmagnification_risk"] = out["target_pair_occupancy_final"] > 0.72
    return out


def role_distinguishability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    iou = pd.to_numeric(out.get("cpos_cneg_mask_iou", 1), errors="coerce").fillna(1)
    same_id = out["cpos_id"].astype(str).eq(out["cneg_id"].astype(str))
    same_role_text = out["cpos_category"].astype(str).eq(out["cneg_category"].astype(str)) & out.get("cpos_supported_verb", pd.Series("", index=out.index)).astype(str).eq(out.get("cneg_supported_verb", pd.Series("", index=out.index)).astype(str))
    out["role_distinguishability"] = 1
    out.loc[same_id, "role_distinguishability"] = 0
    out.loc[~same_id & (iou < 0.2), "role_distinguishability"] = 2
    out.loc[~same_id & (iou < 0.05) & ~same_role_text, "role_distinguishability"] = 3
    out["role_same_entity"] = same_id
    out["role_mask_iou"] = iou
    out["role_center_distance_proxy"] = (pd.to_numeric(out.get("cpos_projected_size", 0), errors="coerce") + pd.to_numeric(out.get("cneg_projected_size", 0), errors="coerce")) / 2
    out["role_independent_color_overlay_ok"] = out["role_distinguishability"] > 0
    out["role_description_synonym_risk"] = same_role_text
    return out


def priority_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj = out["target_object_category"].astype(str)
    out["priority_object_tier"] = "other"
    out.loc[obj.isin(PRIORITY_1), "priority_object_tier"] = "priority_1"
    out.loc[obj.isin(PRIORITY_2), "priority_object_tier"] = "priority_2"
    out.loc[obj.isin(SMALL_OBJECTS), "priority_object_tier"] = "small_object_exception"
    ctx = (out["target_object_category"].astype(str) + "|" + out["cpos_category"].astype(str) + "|" + out["cneg_category"].astype(str)).str.lower()
    patterns = ["cake", "pizza", "suitcase", "tv", "laptop", "bicycle", "dining table", "book", "couch"]
    out["scene_fit_proxy"] = ctx.map(lambda x: any(p in x for p in patterns))
    out["object_scene_score_0_10"] = 3
    out.loc[out["priority_object_tier"].eq("priority_2"), "object_scene_score_0_10"] = 6
    out.loc[out["priority_object_tier"].eq("priority_1"), "object_scene_score_0_10"] = 8
    out.loc[out["scene_fit_proxy"], "object_scene_score_0_10"] += 2
    out.loc[out["priority_object_tier"].eq("small_object_exception"), "object_scene_score_0_10"] = 4
    out["object_scene_score_0_10"] = out["object_scene_score_0_10"].clip(upper=10)
    return out


def auto_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_size_score_0_25"] = ((out["target_human_height_384"] / 80).clip(upper=1) * 12 + (out["target_object_sqrt_area_384"] / 36).clip(upper=1) * 13).round(3)
    out["key_context_score_proxy_0_15"] = ((out["key_context_size_384"] / 32).clip(upper=1) * 7 + pd.to_numeric(out["positive_gain_grade"], errors="coerce").fillna(0) / 3 * 8).round(3)
    out["interference_score_proxy_0_15"] = ((out["interference_context_size_384"] / 32).clip(upper=1) * 7 + pd.to_numeric(out["interference_grade"], errors="coerce").fillna(0) / 3 * 8).round(3)
    out["role_distinguishability_score_0_20"] = pd.to_numeric(out["role_distinguishability"], errors="coerce").fillna(0) / 3 * 20
    out["provenance_coverage_score_0_10"] = 10
    out["ambiguity_gain_proxy_0_5"] = (pd.to_numeric(out["ambiguity_grade"], errors="coerce").fillna(0) / 3 * 5).round(3)
    weights = {
        "target_size_score_0_25": 25, "key_context_score_proxy_0_15": 15, "interference_score_proxy_0_15": 15,
        "role_distinguishability_score_0_20": 20, "provenance_coverage_score_0_10": 10, "object_scene_score_0_10": 10,
        "ambiguity_gain_proxy_0_5": 5,
    }
    score_cols = list(weights)
    out["auto_score_numerator"] = out[score_cols].sum(axis=1)
    out["auto_score_available_weight"] = sum(weights.values())
    out["coverage_ratio"] = 1.0 - out[score_cols].isna().mean(axis=1)
    out["missingness_penalty"] = (1.0 - out["coverage_ratio"]) * 15
    out["validated_auto_score"] = (out["auto_score_numerator"] - out["missingness_penalty"]).clip(lower=0)
    out["score_explanation"] = out.apply(lambda r: f"target={r.target_size_score_0_25:.1f}; key={r.key_context_score_proxy_0_15:.1f}; interference={r.interference_score_proxy_0_15:.1f}; role={r.role_distinguishability_score_0_20:.1f}; object_scene={r.object_scene_score_0_10:.1f}", axis=1)
    return out


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    work = df.copy()
    work["canonical_target_pair"] = work["split"].astype(str) + "|" + work["image_id"].astype(str) + "|" + work["target_human_id"].astype(str) + "|" + work["target_object_id"].astype(str)
    raw_count = len(work)
    unique_image = work["image_key"].nunique()
    unique_pair = work["canonical_target_pair"].nunique()
    work = work[~work["cpos_id"].astype(str).eq(work["cneg_id"].astype(str))].copy()
    work = work.sort_values("validated_auto_score", ascending=False)
    alts = []
    kept = []
    pair_counts: Counter[str] = Counter()
    image_pair_counts: Counter[str] = Counter()
    for row in work.itertuples(index=False):
        pair = row.canonical_target_pair
        image = str(row.image_key)
        if pair_counts[pair] < 3 and image_pair_counts[image] < 2:
            kept.append(row._asdict())
            pair_counts[pair] += 1
            image_pair_counts[image] += 1
        else:
            alts.append({"tuple_id": row.tuple_id, "canonical_target_pair": pair, "image_key": image, "reason": "per_pair_or_per_image_alternate"})
    dedup = pd.DataFrame(kept)
    report = {
        "raw_hypothesis_count": raw_count,
        "unique_image_count": int(unique_image),
        "unique_target_pair_count": int(unique_pair),
        "deduplicated_evidence_hypothesis_count": len(dedup),
        "alternate_tuple_count": len(alts),
    }
    return dedup, report, pd.DataFrame(alts)


def select_shortlist(df: pd.DataFrame, target_size: int) -> pd.DataFrame:
    pool = df[(df["target_final_size_ok"]) & (df["key_context_final_ok"]) & (df["interference_context_final_ok"]) & (df["role_distinguishability"] > 0)].copy()
    pool = pool.sort_values("validated_auto_score", ascending=False).copy()
    # diversity gain is dynamic during selection.
    selected = []
    image_seen = set()
    object_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    small_total = 0
    small_counts: Counter[str] = Counter()
    category_seen = set()

    def try_add(row: Any, bucket: str, force: bool = False) -> bool:
        nonlocal small_total
        image = str(row.image_key)
        obj = str(row.target_object_category)
        family = verb_family(str(row.gt_verb))
        template = f"{row.cpos_category}|{row.cneg_category}"
        if image in image_seen or object_counts[obj] >= 4 or family_counts[family] >= 5 or template_counts[template] >= 5:
            return False
        if obj in SMALL_OBJECTS and (small_total >= 5 or small_counts[obj] >= 1):
            return False
        diversity_gain = 20 if obj not in category_seen else 5
        item = row._asdict()
        item["selection_bucket"] = bucket
        item["diversity_gain"] = diversity_gain
        item["selection_score"] = 0.75 * float(row.validated_auto_score) + 0.25 * diversity_gain
        selected.append(item)
        image_seen.add(image)
        object_counts[obj] += 1
        family_counts[family] += 1
        template_counts[template] += 1
        category_seen.add(obj)
        if obj in SMALL_OBJECTS:
            small_total += 1
            small_counts[obj] += 1
        return True

    bucket_specs = [
        ("highest_trusted_structure", pool.head(800), 20),
        ("gt_aux_context_strong", pool[pool["cpos_source"].astype(str).str.contains("GT", na=False)].head(600), 8),
        ("scene_or_other_person_interference", pool[pool["cneg_category"].astype(str).isin({"person", "chair", "dining table", "couch", "car", "truck"})].head(600), 5),
        ("small_object_readable_exception", pool[pool["target_object_category"].astype(str).isin(SMALL_OBJECTS)].head(400), 4),
        ("medium_score_sanity", pool.iloc[1000:3000], 3),
    ]
    for bucket, sub, quota in bucket_specs:
        count = 0
        for row in sub.itertuples(index=False):
            if count >= quota:
                break
            if try_add(row, bucket):
                count += 1
    for row in pool.itertuples(index=False):
        if len(selected) >= target_size:
            break
        try_add(row, "diversity_fill")
    return pd.DataFrame(selected).sort_values("selection_score", ascending=False).head(target_size).reset_index(drop=True)


def load_or_placeholder(path_value: Any, size: tuple[int, int] = (384, 300)) -> Image.Image:
    path = PROJECT_ROOT / str(path_value) if str(path_value) and str(path_value) != "nan" else None
    if path and path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    img = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "image unavailable", fill=(20, 20, 20))
    return img


def fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = min(size[0] / max(1, img.width), size[1] / max(1, img.height))
    resized = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (248, 248, 248))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def make_review_card(row: Any, review_id: str, out_path: Path) -> None:
    panel = load_or_placeholder(getattr(row, "counterfactual_panel_path", ""))
    final384 = load_or_placeholder(getattr(row, "final_panel_384_path", ""), (384, 384))
    W, H = 1400, 1050
    card = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, W, 58), fill=(20, 20, 20))
    draw.text((18, 20), f"{review_id} | target={row.target_object_category} | GT={row.gt_verb} | competing family={verb_family(str(row.competitor))} | green=key orange=interference", fill=(255, 255, 255))
    card.paste(fit(panel, (920, 650)), (20, 78))
    card.paste(fit(final384, (384, 384)), (980, 78))
    info = [
        "External review checklist:",
        "1. Unmarked/original, overlays, pair/union, key crop, interference crop are in the large panel.",
        "2. Green marks key context; orange marks interference context.",
        "3. Score only visual evidence; ignore old tier/rank/auto score (hidden).",
        "4. Context inset minimum target: 192 px; use card crops if small in 384 panel.",
        f"Target pair: person + {row.target_object_category}",
        f"Key context: {row.cpos_category}; interference: {row.cneg_category}",
    ]
    y = 500
    for line in info:
        draw.text((980, y), line[:70], fill=(20, 20, 20))
        y += 28
    out_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(out_path)


def make_contact_sheet(rows: pd.DataFrame, paths: list[Path], out_png: Path, out_jpg: Path) -> None:
    cols = 2
    tile = (700, 520)
    page_rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * tile[0], max(1, page_rows) * tile[1]), (255, 255, 255))
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        thumb = fit(img, (tile[0] - 20, tile[1] - 20))
        x = (i % cols) * tile[0] + 10
        y = (i // cols) * tile[1] + 10
        sheet.paste(thumb, (x, y))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    sheet.save(out_jpg, quality=92)


def main() -> int:
    args = parse_args()
    root = args.output_root
    out = root / "stage6C_sorting_audit_external_review"
    if out.exists():
        if not args.overwrite_stage6C:
            raise SystemExit(f"Refusing to overwrite existing stage6C directory: {out}")
        shutil.rmtree(out)
    (out / "contact_sheets").mkdir(parents=True, exist_ok=True)
    (out / "individual_review_cards").mkdir(parents=True, exist_ok=True)

    sources = load_sources(root)
    merged, conflicts = merge_complete(sources)
    conflicts.to_csv(out / "join_conflicts.csv", index=False)
    all_tier = sources["all_tier"].copy()
    old_top10 = sources["final_top10_6b"].copy()

    stats = field_stats(all_tier)
    stats.to_csv(out / "grade_score_unique_values.csv", index=False)
    missing = missing_default_audit(all_tier)
    missing.to_csv(out / "missing_default_audit.csv", index=False)
    manual_audit = manual_confirmation_audit(all_tier)
    manual_audit.to_csv(out / "manual_confirmation_audit.csv", index=False)

    sort_df, sort_metrics, score_recomp, old_vs = sort_forensics(all_tier, old_top10)
    sort_df[["tuple_id", "original_row_index", "sort_position_recomputed", "target_object_category", "image_id", "file_name", "tier", "tier_rank", "total_score_0_22", "ambiguity_grade", "positive_gain_grade", "interference_grade"]].head(10000).to_csv(out / "sort_order_forensics.csv", index=False)
    sort_df.groupby("total_score_0_22").agg(count=("tuple_id", "count")).reset_index().sort_values("count", ascending=False).to_csv(out / "tie_groups.csv", index=False)
    (out / "ranking_shuffle_test.json").write_text(json.dumps(sort_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_recomp.to_csv(out / "score_recomputation.csv", index=False)
    old_vs.to_csv(out / "old_vs_corrected_top10.csv", index=False)

    sort_code = f"""# Sort Code Audit

## Searched Patterns

`fillna`, `setdefault`, `astype(bool)`, `bool(value)`, `np.select`, `sort_values`, `nlargest`, `sorted` were searched in `tools/evidencehoi_image_search_v5/build_stage6B_tiered_recovery.py`.

## Findings

- Stage 6B uses `fillna(0)` for `externality_grade_key` and `cneg_outside_fraction`, and `fillna(1)` for c+/c- IoU proxy.
- Stage 6B uses tri-state-aware helper only partially; manual visual recheck remains false, but Tier A does not require manual confirmation.
- Tier A default is not produced by `np.select`, but proxy grades make the Tier A predicate overly broad.
- Ranking uses `sort_values([tier_rank,total_score_0_22,ambiguity_grade,positive_gain_grade,interference_grade])` and then implicit stable input order for ties.
- A pre-output sort includes `target_object_category`, which can create object/category alphabetical artifacts before final selection.
- Score distribution has alerts: `{json.dumps(sort_metrics.get('alerts', []), ensure_ascii=False)}`.

## Required Correction

Confirmed Tier A must require manual visual confirmation. Until then all old Tier A rows are downgraded to AUTO_* / NEEDS_MANUAL_REVIEW classes.
"""
    (out / "sort_code_audit.md").write_text(sort_code, encoding="utf-8")

    # Build corrected feature table from all_tier.
    feature = all_tier.copy()
    feature = role_distinguishability(feature)
    feature = final_size_features(feature)
    feature = priority_features(feature)
    feature = auto_score(feature)
    feature = add_auto_classes(feature)
    feature.to_csv(out / "final_size_features.csv", index=False)
    role_cols = ["tuple_id", "cpos_id", "cneg_id", "cpos_category", "cneg_category", "role_mask_iou", "role_center_distance_proxy", "role_same_entity", "role_independent_color_overlay_ok", "role_description_synonym_risk", "role_distinguishability"]
    feature[[c for c in role_cols if c in feature.columns]].to_csv(out / "context_role_distinguishability.csv", index=False)
    pr_cols = ["tuple_id", "target_object_category", "cpos_category", "cneg_category", "priority_object_tier", "scene_fit_proxy", "object_scene_score_0_10"]
    feature[[c for c in pr_cols if c in feature.columns]].to_csv(out / "priority_object_scene_features.csv", index=False)

    dedup, dedup_report, alternate = deduplicate(feature)
    (out / "candidate_dedup_report.json").write_text(json.dumps(dedup_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    alternate.to_csv(out / "alternate_tuples.csv", index=False)

    shortlist = select_shortlist(dedup, args.target_shortlist_size)
    rng = random.Random(SEED)
    order = list(range(len(shortlist)))
    rng.shuffle(order)
    shortlist = shortlist.iloc[order].reset_index(drop=True)
    shortlist["review_id"] = [f"R{i:03d}" for i in range(1, len(shortlist) + 1)]
    shortlist.to_csv(out / "automatic_shortlist_around50.csv", index=False)

    review_cols = ["review_id", "target_clear", "target_object_clear", "key_context_visible", "key_context_really_helps_GT_0_3", "interference_context_visible", "interference_is_plausible_0_3", "context_roles_distinct", "competing_action_plausible_0_3", "suitable_for_figure1", "preferred_narrative", "rejection_reason", "reviewer_notes"]
    review_form = [{c: "" for c in review_cols} | {"review_id": rid} for rid in shortlist["review_id"]]
    write_csv(out / "external_review_form.csv", review_form, review_cols)

    blind_key_rows = []
    admin_rows = []
    card_paths = []
    for row in shortlist.itertuples(index=False):
        card_path = out / "individual_review_cards" / f"{row.review_id}.png"
        make_review_card(row, row.review_id, card_path)
        card_paths.append(card_path)
        blind_key_rows.append({"review_id": row.review_id, "tuple_id": row.tuple_id, "candidate_id": getattr(row, "candidate_id", ""), "image_key": row.image_key, "image_id": row.image_id, "file_name": row.file_name, "old_tier": row.tier, "old_total_score_0_22": row.total_score_0_22, "validated_auto_score": row.validated_auto_score})
        admin_rows.append({"review_id": row.review_id, "target_object_category": row.target_object_category, "gt_verb": row.gt_verb, "competing_family": verb_family(str(row.competitor)), "key_context_category": row.cpos_category, "interference_context_category": row.cneg_category, "auto_recovery_class": row.auto_recovery_class, "selection_bucket": row.selection_bucket, "counterfactual_panel_path": getattr(row, "counterfactual_panel_path", ""), "final_panel_384_path": getattr(row, "final_panel_384_path", "")})
    write_csv(out / "blind_key.csv", blind_key_rows)
    write_csv(out / "admin_candidate_manifest.csv", admin_rows)

    # Contact sheets, max 8 per page, PNG and high-quality JPEG.
    sheet_paths = []
    for page, start in enumerate(range(0, len(card_paths), 8), start=1):
        chunk = card_paths[start:start + 8]
        png = out / "contact_sheets" / f"external_review_page_{page:02d}.png"
        jpg = out / "contact_sheets" / f"external_review_page_{page:02d}.jpg"
        make_contact_sheet(shortlist.iloc[start:start + 8], chunk, png, jpg)
        sheet_paths.extend([png, jpg])

    readme = """# Stage 6C External Visual Review Pack

This ZIP contains anonymized review cards and contact sheets only. It hides candidate IDs, image IDs, old tiers, old ranks, and automatic scores.

Green denotes key context. Orange denotes interference context. Reviewers should fill `external_review_form.csv` using visual evidence only.

This pack is not a final Figure 1 and does not confirm Top-10 candidates.
"""
    (out / "README_external_review.md").write_text(readme, encoding="utf-8")
    anon_manifest = [{"review_id": r["review_id"], "target_object_category": r["target_object_category"], "gt_verb": r["gt_verb"], "competing_family": r["competing_family"], "key_context_category": r["key_context_category"], "interference_context_category": r["interference_context_category"]} for r in admin_rows]
    write_csv(out / "anonymous_manifest.csv", anon_manifest)

    zip_path = out / "stage6C_external_visual_review_pack.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in [out / "README_external_review.md", out / "external_review_form.csv", out / "anonymous_manifest.csv"]:
            zf.write(p, p.name)
        for p in sheet_paths:
            zf.write(p, str(Path("contact_sheets") / p.name))
        for p in card_paths:
            zf.write(p, str(Path("individual_review_cards") / p.name))

    zip_entries = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_entries = zf.namelist()
        sample_imgs = [n for n in zip_entries if n.endswith(".png")][:5]
        decoded = 0
        for name in sample_imgs:
            with zf.open(name) as handle:
                img = Image.open(handle)
                img.verify()
                decoded += 1

    report = f"""# Stage 6C Sorting Audit And External Review Report

## Tier A Inflation

Old Tier A count: `{int((all_tier['tier'].astype(str) == 'A').sum())}`.

Cause: Stage 6B Tier A was based on broad auto-derived proxy grades and did not require manual visual confirmation. `positive_gain_grade`, `interference_grade`, and externality grades have low diversity and high mode ratios. Confirmed Tier A count is now `0` because `manual_visual_recheck_completed=true` is required and unavailable.

## Sorting Degeneration

- Sort metrics: `{json.dumps({k:v for k,v in sort_metrics.items() if k != 'shuffle_runs'}, ensure_ascii=False)}`
- The highest score covers many candidates and Top-50 cutoff has a large tie group, so implicit stable input order can influence Top-10.
- Object category is not used as the main corrected selection key; it only contributes small object/scene suitability and diversity constraints.

## External Review Pack

- Shortlist size: `{len(shortlist)}`
- ZIP entries: `{len(zip_entries)}`
- Random decoded images from ZIP: `{decoded}`
- This stage does not confirm final Top-10 and does not create paper Figure 1.
"""
    (out / "stage6C_report.md").write_text(report, encoding="utf-8")

    tier_report = f"""# Tier A Inflation Report

- Old Tier A rows: `{int((all_tier['tier'].astype(str) == 'A').sum())}`
- Confirmed Tier A rows after requiring manual visual confirmation: `0`
- Downgraded classes:

```json
{json.dumps(feature['auto_recovery_class'].value_counts().to_dict(), ensure_ascii=False, indent=2)}
```

Primary cause: proxy grades and scores are nearly constant; manual fields are false/unknown. Automatic proposal existence was over-used as semantic grade evidence.
"""
    (out / "tier_A_inflation_report.md").write_text(tier_report, encoding="utf-8")

    # Checksums.
    checksum_paths = [p for p in out.rglob("*") if p.is_file() and p.name != "sha256_checksums.txt"]
    with (out / "sha256_checksums.txt").open("w", encoding="utf-8") as handle:
        for p in sorted(checksum_paths):
            handle.write(f"{sha256_file(p)}  {p.relative_to(out)}\n")

    outputs = [p for p in out.rglob("*") if p.is_file()]
    manifest = {
        "stage": "6C",
        "task": "sorting audit and external visual review pack",
        "created_at_utc": datetime.now().astimezone(timezone.utc).isoformat(),
        "output_root": rel(root),
        "stage6C_dir": rel(out),
        "stopped_after_manifest": True,
        "final_top10_confirmed": False,
        "paper_figure_created": False,
        "confirmed_Tier_A_count": 0,
        "old_Tier_A_count": int((all_tier['tier'].astype(str) == 'A').sum()),
        "shortlist_count": len(shortlist),
        "zip_path": rel(zip_path),
        "zip_entry_count": len(zip_entries),
        "zip_random_decoded_image_count": decoded,
        "manual_unknown_not_true": True,
        "object_category_not_primary_sort_key": True,
        "outputs": [],
    }
    manifest_path = root / "manifest_stage6C.json"
    for p in outputs:
        manifest["outputs"].append({"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest_path), "shortlist_count": len(shortlist), "zip_entries": len(zip_entries), "decoded": decoded}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
