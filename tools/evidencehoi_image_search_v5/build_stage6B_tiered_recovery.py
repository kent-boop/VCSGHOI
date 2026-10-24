#!/usr/bin/env python3
"""Stage 6B tiered recovery from Stage 6A recovery universe.

This produces a provisional tiered recovery Top-10 from H1-H5 hard-eligible
rows. It does not fabricate visual recheck results: manual visual recheck remains
explicitly pending because the current model cannot inspect images.
"""

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

CARD_COLUMNS = [
    "rank", "tier", "narrative_type", "candidate_id", "tuple_id", "pair_id", "image_key", "image_id", "target_object_category", "gt_verb", "competitor",
    "target_readability_score_0_4", "key_context_score_0_4", "interference_score_0_4", "ambiguity_score_0_3", "separability_score_0_2", "mask_support_score_0_2", "provenance_score_0_3", "total_score_0_22",
    "ambiguity_grade", "externality_grade_key", "externality_grade_interference", "positive_gain_grade", "interference_grade",
    "cpos_id", "cpos_category", "cpos_source", "cpos_support_mechanism", "cneg_id", "cneg_category", "cneg_source", "cneg_misselection_mechanism",
    "second_pair_exists", "second_pair_role", "paircentric_narrative", "scene_level_narrative", "neighborhood_narrative", "evidencehoi_narrative", "must_not_claim", "residual_risk", "required_sentence", "manual_visual_recheck_completed", "source_visual_panel_path", "final_panel_384_path", "evidence_card_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite-stage6B", action="store_true")
    parser.add_argument("--candidate-cap", type=int, default=2000, help="cap for detailed tier scoring after deterministic prefilter")
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


def text_image(path: Path, title: str, lines: Sequence[str], size: tuple[int, int] = (1600, 1000)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0], 60), fill=(35, 35, 35))
    draw.text((18, 20), title[:190], fill=(255, 255, 255))
    y = 84
    for line in lines:
        text = str(line)
        while len(text) > 155:
            draw.text((24, y), text[:155], fill=(20, 20, 20))
            y += 18
            text = text[155:]
        draw.text((24, y), text, fill=(20, 20, 20))
        y += 22
        if y > size[1] - 35:
            break
    image.save(path, quality=88)


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["ambiguity_grade", "externality_grade", "positive_gain_grade", "interference_grade", "target_human_projected_height", "target_object_projected_sqrt_area", "target_object_projected_short_side", "cpos_projected_size", "cneg_projected_size", "cpos_cneg_mask_iou", "target_pair_occupancy"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work["externality_grade_key"] = work["externality_grade"].fillna(0).astype(int)
    cneg_out = pd.to_numeric(work.get("cneg_outside_fraction", pd.Series(0, index=work.index)), errors="coerce").fillna(0)
    work["externality_grade_interference"] = 0
    work.loc[cneg_out > 0, "externality_grade_interference"] = 1
    work.loc[cneg_out >= 0.20, "externality_grade_interference"] = 2
    work.loc[cneg_out >= 0.65, "externality_grade_interference"] = 3

    work["context_visually_separable_proxy"] = work["cpos_id"].astype(str).ne(work["cneg_id"].astype(str)) & (pd.to_numeric(work.get("cpos_cneg_mask_iou", 0), errors="coerce").fillna(1) < 0.2)
    work["tier"] = "not_tiered"
    tier_a = (
        as_bool(work["hard_eligible_H1_H5"])
        & (work["ambiguity_grade"] == 3)
        & (work["positive_gain_grade"] >= 2)
        & (work["interference_grade"] >= 2)
        & (work["externality_grade_key"] >= 2)
        & (work["externality_grade_interference"] >= 1)
    )
    tier_b = (
        as_bool(work["hard_eligible_H1_H5"])
        & ~tier_a
        & (work["ambiguity_grade"] >= 1)
        & (work["positive_gain_grade"] >= 2)
        & (work["interference_grade"] >= 1)
        & work["context_visually_separable_proxy"]
    )
    tier_c = (
        as_bool(work["hard_eligible_H1_H5"])
        & ~tier_a
        & ~tier_b
        & (work["positive_gain_grade"] >= 1)
        & (work["interference_grade"] >= 1)
        & work["context_visually_separable_proxy"]
    )
    work.loc[tier_a, "tier"] = "A"
    work.loc[tier_b, "tier"] = "B"
    work.loc[tier_c, "tier"] = "C"
    work["narrative_type"] = "not_eligible"
    work.loc[tier_a, "narrative_type"] = "strong_external_evidence"
    work.loc[tier_b, "narrative_type"] = "entity_level_context"
    work.loc[tier_c, "narrative_type"] = "selective_context_not_strong_ambiguity"

    work["target_readability_score_0_4"] = 3
    work.loc[(work["target_human_projected_height"] >= 150) & ((work["target_object_projected_short_side"] >= 40) | (work["target_object_projected_sqrt_area"] >= 55)), "target_readability_score_0_4"] = 4
    work["key_context_score_0_4"] = work["positive_gain_grade"].clip(0, 3) + 1
    work.loc[work["key_context_score_0_4"] > 4, "key_context_score_0_4"] = 4
    work["interference_score_0_4"] = work["interference_grade"].clip(0, 3) + 1
    work.loc[work["interference_score_0_4"] > 4, "interference_score_0_4"] = 4
    work["ambiguity_score_0_3"] = work["ambiguity_grade"].clip(0, 3)
    work["separability_score_0_2"] = 1
    work.loc[(work["externality_grade_key"] >= 2) & (work["externality_grade_interference"] >= 1), "separability_score_0_2"] = 2
    work["mask_support_score_0_2"] = 1
    work.loc[(work["positive_gain_grade"] >= 2) & (work["interference_grade"] >= 2), "mask_support_score_0_2"] = 2
    work["provenance_score_0_3"] = 2
    work.loc[work["gt_verb_source"].astype(str).str.contains("GT", na=False) & work["cpos_source"].astype(str).str.len().gt(0) & work["cneg_source"].astype(str).str.len().gt(0), "provenance_score_0_3"] = 3
    work["total_score_0_22"] = work[["target_readability_score_0_4", "key_context_score_0_4", "interference_score_0_4", "ambiguity_score_0_3", "separability_score_0_2", "mask_support_score_0_2", "provenance_score_0_3"]].sum(axis=1)
    work["score_minima_pass"] = (work["target_readability_score_0_4"] >= 3) & (work["key_context_score_0_4"] >= 2) & (work["interference_score_0_4"] >= 2) & (work["provenance_score_0_3"] >= 2)
    return work


def mechanism_text(row: Any) -> tuple[str, str]:
    cpos_src = str(getattr(row, "cpos_source", ""))
    cneg_src = str(getattr(row, "cneg_source", ""))
    key = f"{row.cpos_category} is a separate context entity/relation associated with {row.gt_verb}; provenance={cpos_src[:80]}"
    if row.interference_grade >= 2:
        mis = f"{row.cneg_category} can support competitor family/action {row.competitor} or an alternative pair/state; provenance={cneg_src[:80]}"
    else:
        mis = f"{row.cneg_category} can be selected by scene-level or neighborhood aggregation as evidence pollution; provenance={cneg_src[:80]}"
    return key, mis


def required_sentence(row: Any) -> str:
    return (
        f"目标 pair 为 {row.target_object_category}+person。关键 context {row.cpos_category} 通过 independent entity/relation evidence 帮助解释 GT {row.gt_verb}；"
        f"独立干扰 context {row.cneg_category} 可能因 scene-level/neighborhood aggregation 被错误聚合。"
        f"该图的歧义等级为 {row.ambiguity_grade}，适合用于 {row.narrative_type} 型叙事。"
    )


def rank_select(work: pd.DataFrame) -> pd.DataFrame:
    candidates = work[(work["tier"].isin(["A", "B", "C"])) & as_bool(work["hard_eligible_H1_H5"]) & work["score_minima_pass"]].copy()
    candidates["tier_rank"] = candidates["tier"].map({"A": 0, "B": 1, "C": 2})
    candidates = candidates.sort_values(["tier_rank", "total_score_0_22", "ambiguity_grade", "positive_gain_grade", "interference_grade"], ascending=[True, False, False, False, False])
    chosen = []
    image_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    for row in candidates.itertuples(index=False):
        image = str(row.image_key)
        obj = str(row.target_object_category)
        action = f"{row.gt_verb}|{row.competitor}"
        ctx = f"{row.cpos_category}|{row.cneg_category}"
        if image_counts[image] >= 1:
            continue
        if object_counts[obj] >= 2:
            continue
        if action_counts[action] >= 2:
            continue
        if context_counts[ctx] >= 2:
            continue
        chosen.append(row._asdict())
        image_counts[image] += 1
        object_counts[obj] += 1
        action_counts[action] += 1
        context_counts[ctx] += 1
        if len(chosen) >= 10:
            break
    return pd.DataFrame(chosen)


def make_card(row: Any, rank: int, out_dir: Path) -> tuple[dict[str, Any], Path]:
    key_mech, mis_mech = mechanism_text(row)
    sent = required_sentence(row)
    tier_desc = {"A": "strong external evidence", "B": "entity-level context", "C": "selective context, not strong ambiguity"}.get(row.tier, "unknown")
    second_pair_exists = "[GT]" in str(row.cpos_provenance) or "[GT]" in str(row.cneg_provenance)
    second_pair_role = "context provenance only; not a qualification gate" if second_pair_exists else "none required"
    paircentric = "Pair-centric lacks explicit organization and verification of the independent context entity/relation."
    if row.externality_grade_key >= 2:
        paircentric = "Pair-centric information is insufficient; external key context completes the action interpretation."
    scene = f"Scene-level aggregation may over-weight {row.cneg_category} as a competing scene/entity cue."
    neigh = f"Neighborhood aggregation may pull in nearby/same-class {row.cneg_category} rather than the key {row.cpos_category}."
    evhoi = f"EvidenceHOI should retain {row.cpos_category} and reject/downweight {row.cneg_category}."
    must_not = "Do not claim GT attention, causal model effect, or completed human visual recheck."
    risk = "Provisional recovery ranking depends on Stage 6A proxies; manual visual recheck remains pending."
    card = {
        "rank": rank,
        "tier": row.tier,
        "narrative_type": row.narrative_type,
        "candidate_id": row.candidate_id,
        "tuple_id": row.tuple_id,
        "pair_id": row.pair_id,
        "image_key": row.image_key,
        "image_id": row.image_id,
        "target_object_category": row.target_object_category,
        "gt_verb": row.gt_verb,
        "competitor": row.competitor if row.tier in {"A", "B"} else f"{verb_family(str(row.competitor))} / evidence pollution",
        "target_readability_score_0_4": row.target_readability_score_0_4,
        "key_context_score_0_4": row.key_context_score_0_4,
        "interference_score_0_4": row.interference_score_0_4,
        "ambiguity_score_0_3": row.ambiguity_score_0_3,
        "separability_score_0_2": row.separability_score_0_2,
        "mask_support_score_0_2": row.mask_support_score_0_2,
        "provenance_score_0_3": row.provenance_score_0_3,
        "total_score_0_22": row.total_score_0_22,
        "ambiguity_grade": row.ambiguity_grade,
        "externality_grade_key": row.externality_grade_key,
        "externality_grade_interference": row.externality_grade_interference,
        "positive_gain_grade": row.positive_gain_grade,
        "interference_grade": row.interference_grade,
        "cpos_id": row.cpos_id,
        "cpos_category": row.cpos_category,
        "cpos_source": row.cpos_source,
        "cpos_support_mechanism": key_mech,
        "cneg_id": row.cneg_id,
        "cneg_category": row.cneg_category,
        "cneg_source": row.cneg_source,
        "cneg_misselection_mechanism": mis_mech,
        "second_pair_exists": second_pair_exists,
        "second_pair_role": second_pair_role,
        "paircentric_narrative": paircentric,
        "scene_level_narrative": scene,
        "neighborhood_narrative": neigh,
        "evidencehoi_narrative": evhoi,
        "must_not_claim": must_not,
        "residual_risk": risk,
        "required_sentence": sent,
        "manual_visual_recheck_completed": False,
        "source_visual_panel_path": getattr(row, "counterfactual_panel_path", ""),
        "final_panel_384_path": getattr(row, "final_panel_384_path", ""),
        "evidence_card_path": "",
    }
    path = out_dir / "evidence_cards" / f"rank_{rank:02d}_{row.candidate_id}.md"
    body = "# Evidence Card\n\n" + "\n".join([f"- {k}: `{v}`" for k, v in card.items() if k != "evidence_card_path"])
    path.write_text(body + "\n", encoding="utf-8")
    card["evidence_card_path"] = rel(path)
    return card, path


def contact_from_rows(path: Path, title: str, rows: pd.DataFrame, limit: int = 30) -> None:
    lines = []
    for r in rows.head(limit).itertuples(index=False):
        lines.append(f"{getattr(r,'tier','?')} | {r.target_object_category} | {r.gt_verb} vs {r.competitor} | c+ {r.cpos_category} | c- {r.cneg_category} | score {getattr(r,'total_score_0_22','')}")
    if not lines:
        lines = ["No rows"]
    text_image(path, title, lines)


def main() -> int:
    args = parse_args()
    root = args.output_root
    out_dir = root / "stage6B_tiered_recovery"
    if out_dir.exists():
        if not args.overwrite_stage6B:
            raise SystemExit(f"Refusing to overwrite existing stage6B directory: {out_dir}")
        shutil.rmtree(out_dir)
    for sub in [out_dir, out_dir / "evidence_cards", out_dir / "contact_sheets"]:
        sub.mkdir(parents=True, exist_ok=True)

    rec_path = root / "stage6A_zero_diagnosis/recovery_universe.csv"
    hard_path = root / "stage6A_zero_diagnosis/new_hard_gate_results.csv"
    merged_path = root / "stage6A_zero_diagnosis/merged_rejected_candidates.parquet"
    visual_path = root / "stage4_visuals/visual_manifest.csv"
    high_reject_md = root / "stage6_top10/high_scoring_rejects.md"

    recovery = pd.read_csv(rec_path)
    hard = pd.read_csv(hard_path)
    merged = pd.read_parquet(merged_path)
    visual = pd.read_csv(visual_path)
    # Use recovery as primary universe, then add fields missing from the compact CSV.
    keep_cols = [c for c in merged.columns if c not in recovery.columns or c in {"tuple_id"}]
    if len(keep_cols) > 1:
        recovery = recovery.merge(merged[keep_cols], on="tuple_id", how="left", suffixes=("", "_merged"))
    if "counterfactual_panel_path" not in recovery.columns and "counterfactual_panel_path" in visual.columns:
        recovery = recovery.merge(visual[["tuple_id", "counterfactual_panel_path", "final_panel_384_path"]], on="tuple_id", how="left")

    scored = score_rows(recovery)
    # Deterministic cap after score/tier sort only to keep files readable; final selection still uses all scored rows if under cap is not needed.
    scored["tier_rank"] = scored["tier"].map({"A": 0, "B": 1, "C": 2, "not_tiered": 9}).fillna(9)
    scored = scored.sort_values(["tier_rank", "total_score_0_22", "target_object_category"], ascending=[True, False, True])
    detailed = scored.head(args.candidate_cap).copy()
    final = rank_select(scored)

    manual_rows = []
    for r in detailed.itertuples(index=False):
        manual_rows.append({
            "candidate_id": r.candidate_id,
            "tuple_id": r.tuple_id,
            "tier": r.tier,
            "needs_visual_recheck": True,
            "manual_visual_recheck_completed": False,
            "recheck_items": "original,target_overlay,384px,pair_union,key_context_crop,interference_context_crop,mask_reveal",
            "reason": "Current model cannot inspect images; Stage 6B ranking is provisional and requires human visual confirmation.",
            "counterfactual_panel_path": getattr(r, "counterfactual_panel_path", ""),
            "final_panel_384_path": getattr(r, "final_panel_384_path", ""),
        })
    write_csv(out_dir / "manual_recheck.csv", manual_rows)

    hard_eligible = scored[as_bool(scored["hard_eligible_H1_H5"])].copy()
    hard_eligible.to_csv(out_dir / "hard_eligible_candidates.csv", index=False)
    scored[scored["tier"] == "A"].to_csv(out_dir / "tier_A.csv", index=False)
    scored[scored["tier"] == "B"].to_csv(out_dir / "tier_B.csv", index=False)
    scored[scored["tier"] == "C"].to_csv(out_dir / "tier_C.csv", index=False)
    scored.to_csv(out_dir / "all_tier_ranking.csv", index=False)

    cards = []
    card_paths = []
    for i, row in enumerate(final.itertuples(index=False), start=1):
        card, path = make_card(row, i, out_dir)
        cards.append(card)
        card_paths.append(path)
    write_csv(out_dir / "final_top10.csv", cards, CARD_COLUMNS)

    high_hard = scored[(~as_bool(scored["hard_eligible_H1_H5"]))].copy()
    high_hard = high_hard.sort_values("total_score_0_22", ascending=False).head(200)
    high_hard.to_csv(out_dir / "high_scoring_hard_rejects.csv", index=False)
    high_md = "# High Scoring Hard Rejects\n\nNo H1-H5 hard-failing candidate is mixed into final_top10.\n\n"
    high_md += f"Rows exported: `{len(high_hard)}`. Source high-scoring reject note: `{rel(high_reject_md)}`.\n"
    (out_dir / "high_scoring_hard_rejects.md").write_text(high_md, encoding="utf-8")

    contact_from_rows(out_dir / "contact_sheets/tier_A.jpg", "Tier A Preview", scored[scored["tier"] == "A"])
    contact_from_rows(out_dir / "contact_sheets/tier_B.jpg", "Tier B Preview", scored[scored["tier"] == "B"])
    contact_from_rows(out_dir / "contact_sheets/tier_C.jpg", "Tier C Preview", scored[scored["tier"] == "C"])
    contact_from_rows(out_dir / "contact_sheets/final_top10_originals.jpg", "Final Top-10 Originals Placeholder", final)
    contact_from_rows(out_dir / "contact_sheets/final_top10_context_map.jpg", "Final Top-10 Context Map", final)
    contact_from_rows(out_dir / "contact_sheets/final_top10_384px.jpg", "Final Top-10 384px Placeholder", final)

    tier_counts = scored["tier"].value_counts().to_dict()
    final_count = len(cards)
    gap = max(0, 10 - final_count)
    analysis = f"""# Stage 6B Final Top-10 Analysis

## Status

This is a provisional tiered recovery Top-10 generated from `stage6A_zero_diagnosis/recovery_universe.csv`, not from empty `passed_all_gates.csv`.

- Recovery universe rows: `{len(recovery)}`
- H1-H5 hard-eligible rows: `{len(hard_eligible)}`
- Tier counts: `{json.dumps(tier_counts, ensure_ascii=False)}`
- Final provisional count: `{final_count}`
- Manual visual recheck completed: `false`

Because this environment cannot inspect images, `manual_recheck.csv` marks candidates as needing visual recheck. The ranking uses Stage 6A hard gates and soft grades only; it must not be used as a paper Figure 1 final until humans verify the listed panels.

## Selection Rules

- Tier A before Tier B before Tier C.
- Same image at most one tuple.
- Same target object at most two.
- Same action template at most two.
- Same key/interference category template at most two.
- No second-person or second-pair requirement.
- No H1-H5 hard reject can enter final_top10.

## Tier C Caveat

Tier C rows use `narrative_type=selective_context_not_strong_ambiguity` and do not claim strong action ambiguity.
"""
    (out_dir / "final_top10_analysis.md").write_text(analysis, encoding="utf-8")

    gap_report = f"""# Remaining Gap Report

- Final provisional rows: `{final_count}`
- Gap to 10: `{gap}`
- H1-H5 was not relaxed.
- If manual recheck rejects final rows,补搜 should target genuinely visible key context plus independent interference context rather than pair-internal cues.
"""
    (out_dir / "remaining_gap_report.md").write_text(gap_report, encoding="utf-8")

    repro = {
        "stage": "6B",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "recovery_universe": rel(rec_path),
            "new_hard_gate_results": rel(hard_path),
            "merged_rejected_candidates": rel(merged_path),
            "visual_manifest": rel(visual_path),
            "high_scoring_rejects": rel(high_reject_md),
        },
        "did_not_read_empty_passed_for_ranking": True,
        "manual_visual_recheck_completed": False,
        "top10_created_as_provisional_recovery": True,
        "paper_figure_created": False,
        "h1_h5_relaxed": False,
        "second_person_or_second_pair_gate_used": False,
        "tier_counts": tier_counts,
        "final_count": final_count,
    }
    repro_path = out_dir / "reproducibility_manifest.json"
    repro_path.write_text(json.dumps(repro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    outputs = [
        out_dir / "manual_recheck.csv",
        out_dir / "hard_eligible_candidates.csv",
        out_dir / "tier_A.csv",
        out_dir / "tier_B.csv",
        out_dir / "tier_C.csv",
        out_dir / "all_tier_ranking.csv",
        out_dir / "final_top10.csv",
        out_dir / "final_top10_analysis.md",
        *card_paths,
        out_dir / "contact_sheets/tier_A.jpg",
        out_dir / "contact_sheets/tier_B.jpg",
        out_dir / "contact_sheets/tier_C.jpg",
        out_dir / "contact_sheets/final_top10_originals.jpg",
        out_dir / "contact_sheets/final_top10_context_map.jpg",
        out_dir / "contact_sheets/final_top10_384px.jpg",
        out_dir / "high_scoring_hard_rejects.csv",
        out_dir / "high_scoring_hard_rejects.md",
        out_dir / "remaining_gap_report.md",
        repro_path,
        Path(__file__),
    ]
    manifest = {
        "stage": "6B",
        "task": "tiered recovery and provisional Top-10 from Stage 6A recovery universe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": rel(root),
        "stage6B_dir": rel(out_dir),
        "stopped_after_manifest": True,
        "did_not_sort_empty_passed_table": True,
        "manual_visual_recheck_completed": False,
        "final_top10_is_provisional": True,
        "paper_figure_created": False,
        "h1_h5_required": True,
        "h1_h5_relaxed": False,
        "second_person_or_second_pair_gate_used": False,
        "tier_C_strong_ambiguity_claim_allowed": False,
        "recovery_universe_count": len(recovery),
        "hard_eligible_count": len(hard_eligible),
        "tier_counts": tier_counts,
        "final_top10_count": final_count,
        "shortage": gap,
        "outputs": [],
    }
    for path in outputs:
        if path.exists() and path.is_file():
            manifest["outputs"].append({"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest_path = root / "manifest_stage6B.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest_path), "final_top10_count": final_count, "tier_counts": tier_counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
