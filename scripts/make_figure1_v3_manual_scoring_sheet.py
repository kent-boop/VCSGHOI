#!/usr/bin/env python3
"""Create a 20-point manual scoring sheet for v3 Figure 1 candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
DEFAULT_PRIORITY = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive/priority_candidates.csv"
DEFAULT_COUNTERFACTUAL = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive/counterfactual_crops/counterfactual_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive/manual_scoring"
DEFAULT_TRACKS_DIR = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive/tracks"


SCORING_COLUMNS = [
    "shortlist_rank",
    "candidate_id",
    "pair_id",
    "image_key",
    "file_name",
    "priority_group",
    "track_id",
    "eligible_for_figure1_shortlist",
    "shortlist_gate_reason",
    "score_cross_human_pair_heterogeneity_0_5",
    "score_action_prototype_ambiguity_0_4",
    "score_context_discriminative_gain_0_5",
    "score_scene_neighborhood_interference_0_3",
    "score_readability_drawing_friendliness_0_3",
    "manual_total_0_20",
    "q1_target_positive_gt",
    "q2_pair_only_confusing_action",
    "q3_decisive_context_object_part_relation",
    "q4_why_context_changes_action_ranking_not_cooccurrence",
    "q5_second_person_different_activity",
    "q6_why_second_pair_evidence_may_mislead_target",
    "q7_scene_level_wrongly_mixes",
    "q8_neighborhood_based_wrongly_keeps",
    "q9_evidencehoi_should_keep_evidence_path",
    "q10_context_masked_more_ambiguous",
    "counterfactual_pair_only_crop",
    "counterfactual_pair_plus_context_crop",
    "counterfactual_context_masked_crop",
    "counterfactual_second_evidence_crop",
    "counterfactual_correct_evidence_crop",
    "manual_review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority", type=Path, default=DEFAULT_PRIORITY)
    parser.add_argument("--counterfactual", type=Path, default=DEFAULT_COUNTERFACTUAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracks-dir", type=Path, default=DEFAULT_TRACKS_DIR)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--track-limit", type=int, default=120)
    return parser.parse_args()


def load_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(str(value))
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORING_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def text_join(items: list[Any], limit: int = 4) -> str:
    values = [str(item) for item in items if str(item)]
    if len(values) > limit:
        return "; ".join(values[:limit]) + f"; +{len(values) - limit} more"
    return "; ".join(values)


def first_context(row: dict[str, str]) -> tuple[str, str, str]:
    contexts = load_json_list(row.get("decisive_context_candidates", "[]"))
    for item in contexts:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "track_c_hico_context_cue_candidates":
            candidates = item.get("candidates")
            if isinstance(candidates, list):
                for cand in candidates:
                    if isinstance(cand, dict):
                        category = str(cand.get("category", ""))
                        inst_id = str(cand.get("instance_id", ""))
                        source = str(cand.get("source", ""))
                        if category and inst_id:
                            return category, inst_id, source
        if item.get("type") == "target_human_auxiliary_pair":
            pair_types = item.get("pair_types") if isinstance(item.get("pair_types"), list) else []
            return text_join(pair_types, 2), str(item.get("pair_id", "")), str(item.get("source", ""))
    return "", "", ""


def first_second_activity(row: dict[str, str]) -> tuple[str, str]:
    second_pair = row.get("second_pair_id", "")
    second_type = row.get("second_pair_type", "")
    if second_pair or second_type:
        return second_pair, second_type
    neighbors = load_json_list(row.get("misleading_neighbor", "[]"))
    for item in neighbors:
        if isinstance(item, dict):
            families = item.get("verb_families") if isinstance(item.get("verb_families"), list) else []
            second_type = f"{item.get('object_category','')}/{','.join(str(x) for x in families)}"
            return str(item.get("pair_id", "")), second_type
    return "", ""


def counterfactual_by_pair(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out.setdefault(row.get("pair_id", ""), row)
    return out


def has_counterfactual(cf: Optional[dict[str, str]]) -> bool:
    if not cf:
        return False
    required = [
        "pair_only_crop",
        "pair_plus_single_context_crop",
        "context_masked_crop",
        "target_pair_plus_second_person_evidence_crop",
        "target_pair_plus_correct_evidence_crop",
    ]
    return all(cf.get(key) for key in required)


def make_row(priority_row: dict[str, str], cf_row: Optional[dict[str, str]]) -> dict[str, Any]:
    target_verbs = [str(v) for v in load_json_list(priority_row.get("target_verbs", "[]")) if str(v) != "no_interaction"]
    target_types = [str(v) for v in load_json_list(priority_row.get("target_pair_type", "[]"))]
    confusion = [str(v) for v in load_json_list(priority_row.get("confusion_verb_set", "[]")) if str(v) not in target_verbs and str(v) != "no_interaction"]
    context_category, context_id, context_source = first_context(priority_row)
    second_pair, second_type = first_second_activity(priority_row)
    misleading_neighbors = load_json_list(priority_row.get("misleading_neighbor", "[]"))
    scene_mix = priority_row.get("contrast_pair_type", "")
    neighborhood = ""
    if misleading_neighbors:
        pieces = []
        for item in misleading_neighbors[:3]:
            if isinstance(item, dict):
                pieces.append(f"{item.get('object_category','')} from {item.get('pair_id','')}")
        neighborhood = "; ".join(pieces)

    q1 = f"GT verb(s): {text_join(target_verbs)}; target pair type: {text_join(target_types)}"
    q2 = f"Pair-only may be confused with: {text_join(confusion[:6])}" if confusion else ""
    q3 = f"{context_category} ({context_id}); source={context_source}" if context_category else ""
    q4 = (
        f"Review whether {context_category} changes support toward {text_join(target_verbs)} rather than merely co-occurring; compare pair-only/context/masked crops."
        if context_category else ""
    )
    q5 = f"Second pair/activity: {second_pair} / {second_type}" if (second_pair or second_type) else ""
    q6 = priority_row.get("borrowed_evidence_failure_mode", "") or (
        f"Borrowing second activity evidence ({second_type}) could shift target interpretation away from {text_join(target_verbs)}."
        if second_type else ""
    )
    q7 = f"Scene-level may mix in dominant/contrast activity evidence: {scene_mix}" if scene_mix else ""
    q8 = f"Neighborhood-based method may retain nearby conflicting evidence: {neighborhood}" if neighborhood else ""
    q9 = f"Keep target pair path plus correct context ({context_category}); do not copy second evidence ({second_type})." if context_category else ""
    q10 = "[MANUAL_REQUIRED] Use context_masked_crop vs pair_plus_single_context_crop to decide; expected: masked crop is more ambiguous." if has_counterfactual(cf_row) else ""

    gate_missing = []
    if not q2:
        gate_missing.append("missing_q2_pair_only_confusing_action")
    if not q3:
        gate_missing.append("missing_q3_decisive_context")
    if not (q6 or q7):
        gate_missing.append("missing_q6_or_q7_interference_answer")
    if not has_counterfactual(cf_row):
        gate_missing.append("missing_counterfactual_crops_for_q10")
    eligible = not gate_missing

    return {
        "shortlist_rank": "",
        "candidate_id": priority_row.get("candidate_id", ""),
        "pair_id": priority_row.get("pair_id", ""),
        "image_key": priority_row.get("image_key", ""),
        "file_name": priority_row.get("file_name", ""),
        "priority_group": priority_row.get("priority_group", ""),
        "track_id": priority_row.get("track_id", ""),
        "eligible_for_figure1_shortlist": eligible,
        "shortlist_gate_reason": "pass" if eligible else ";".join(gate_missing),
        "score_cross_human_pair_heterogeneity_0_5": "",
        "score_action_prototype_ambiguity_0_4": "",
        "score_context_discriminative_gain_0_5": "",
        "score_scene_neighborhood_interference_0_3": "",
        "score_readability_drawing_friendliness_0_3": "",
        "manual_total_0_20": "",
        "q1_target_positive_gt": q1,
        "q2_pair_only_confusing_action": q2,
        "q3_decisive_context_object_part_relation": q3,
        "q4_why_context_changes_action_ranking_not_cooccurrence": q4,
        "q5_second_person_different_activity": q5,
        "q6_why_second_pair_evidence_may_mislead_target": q6,
        "q7_scene_level_wrongly_mixes": q7,
        "q8_neighborhood_based_wrongly_keeps": q8,
        "q9_evidencehoi_should_keep_evidence_path": q9,
        "q10_context_masked_more_ambiguous": q10,
        "counterfactual_pair_only_crop": cf_row.get("pair_only_crop", "") if cf_row else "",
        "counterfactual_pair_plus_context_crop": cf_row.get("pair_plus_single_context_crop", "") if cf_row else "",
        "counterfactual_context_masked_crop": cf_row.get("context_masked_crop", "") if cf_row else "",
        "counterfactual_second_evidence_crop": cf_row.get("target_pair_plus_second_person_evidence_crop", "") if cf_row else "",
        "counterfactual_correct_evidence_crop": cf_row.get("target_pair_plus_correct_evidence_crop", "") if cf_row else "",
        "manual_review_notes": "Scores are manual only. Do not add points for duplicate instances, human count, object count, or raw HOI count.",
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    priority_rows = read_csv(args.priority)[: args.limit]
    merged_by_pair: dict[str, dict[str, str]] = {}
    for row in priority_rows:
        merged_by_pair.setdefault(row.get("pair_id", ""), row)
    for track_id in ["D", "A", "B", "C"]:
        for row in read_csv(args.tracks_dir / f"track_{track_id}.csv")[: args.track_limit]:
            pair_id = row.get("pair_id", "")
            if pair_id and pair_id not in merged_by_pair:
                row = dict(row)
                row.setdefault("priority_group", f"track_{track_id}_directed_pool")
                merged_by_pair[pair_id] = row
    priority_rows = list(merged_by_pair.values())
    cf_rows = counterfactual_by_pair(read_csv(args.counterfactual))
    scored = [make_row(row, cf_rows.get(row.get("pair_id", ""))) for row in priority_rows]
    shortlist = [row for row in scored if row["eligible_for_figure1_shortlist"]]
    excluded = [row for row in scored if not row["eligible_for_figure1_shortlist"]]
    for rank, row in enumerate(shortlist, start=1):
        row["shortlist_rank"] = rank

    write_csv(args.output_dir / "manual_scoring_candidates.csv", scored)
    write_csv(args.output_dir / "figure1_shortlist_manual_scoring.csv", shortlist)
    write_csv(args.output_dir / "manual_scoring_excluded.csv", excluded)

    summary = {
        "schema_version": "figure1_search_v3.manual_scoring_20pt",
        "score_rubric": {
            "cross_human_pair_heterogeneity": "0-5",
            "action_prototype_ambiguity": "0-4",
            "context_discriminative_gain": "0-5",
            "scene_neighborhood_interference": "0-3",
            "readability_drawing_friendliness": "0-3",
            "total": "0-20",
        },
        "no_direct_count_bonus": ["duplicate_instance_count", "human_count", "object_count", "raw_hoi_count"],
        "input_priority": str(args.priority.relative_to(PROJECT_ROOT)),
        "input_counterfactual_manifest": str(args.counterfactual.relative_to(PROJECT_ROOT)),
        "candidate_rows": len(scored),
        "candidate_pool_note": "priority candidates plus directed per-track A/B/C/D candidates; manual score fields remain blank",
        "shortlist_rows": len(shortlist),
        "excluded_rows": len(excluded),
        "gate_rule": "exclude if q2 pair-only confusion, q3 decisive context, or q6/q7 interference answer is missing; counterfactual crops required for q10 review",
        "outputs": {
            "manual_scoring_candidates": str((args.output_dir / "manual_scoring_candidates.csv").relative_to(PROJECT_ROOT)),
            "figure1_shortlist_manual_scoring": str((args.output_dir / "figure1_shortlist_manual_scoring.csv").relative_to(PROJECT_ROOT)),
            "manual_scoring_excluded": str((args.output_dir / "manual_scoring_excluded.csv").relative_to(PROJECT_ROOT)),
        },
    }
    (args.output_dir / "manual_scoring_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
