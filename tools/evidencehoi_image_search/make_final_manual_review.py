#!/usr/bin/env python3
"""Generate final manual review checklist from scored buckets and contact sheets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("tools/evidencehoi_image_search/results")
OUTPUT_PATH = RESULTS_ROOT / "final_manual_review.md"


SPECS = [
    {
        "label": "HICO trainval A",
        "dataset_dir": "hico_trainval",
        "bucket": "A",
        "bucket_stem": "bucket_A_figure1",
        "jsonl": RESULTS_ROOT / "hico_trainval" / "bucket_A_figure1.jsonl",
        "contact_dir": RESULTS_ROOT / "hico_trainval" / "contact_sheets" / "bucket_A_figure1_top50",
        "use": "Figure 1",
    },
    {
        "label": "HICO trainval B",
        "dataset_dir": "hico_trainval",
        "bucket": "B",
        "bucket_stem": "bucket_B_figure2",
        "jsonl": RESULTS_ROOT / "hico_trainval" / "bucket_B_figure2.jsonl",
        "contact_dir": RESULTS_ROOT / "hico_trainval" / "contact_sheets" / "bucket_B_figure2_top50",
        "use": "Figure 2",
    },
    {
        "label": "HICO test C",
        "dataset_dir": "hico_test",
        "bucket": "C",
        "bucket_stem": "bucket_C_diagnostic",
        "jsonl": RESULTS_ROOT / "hico_test" / "bucket_C_diagnostic.jsonl",
        "contact_dir": RESULTS_ROOT / "hico_test" / "contact_sheets" / "bucket_C_diagnostic_top50",
        "use": "diagnostic",
    },
]


CHECKLIST = [
    "主 pair 是否一眼能看懂？",
    "至少 3 个 HO pairs 是否清楚？",
    "不同 pair 的动作是否有 evidence requirement 差异？",
    "是否能画出 2-3 条 evidence paths？",
    "是否能支持 Figure 1 的三栏对比： (a) Generic Context Aggregation (b) Local Grouping (c) EvidenceHOI",
    "是否容易裁剪圆形节点？",
    "是否太像 GroupHOI / local grouping 场景？",
    "是否存在严重遮挡或过度拥挤？",
    "主 pair 是否位于画面中心区域？",
    "上下文物体是否可见、可解释？",
    "是否可作为 Figure 2 pipeline 的 clean example？",
    "是否适合 qualitative / diagnostic figure？",
]


SPORTS_OBJECTS = {
    "sports ball",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "skis",
    "snowboard",
    "skateboard",
    "surfboard",
}


SMALL_OBJECTS = {"cell phone", "cup", "fork", "knife", "spoon", "mouse", "remote", "book", "bottle", "wine glass"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def load_review_rank(contact_dir: Path) -> dict[str, int]:
    table = contact_dir / "review_table.csv"
    ranks: dict[str, int] = {}
    if not table.exists():
        return ranks
    with table.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                ranks[str(row.get("image_id"))] = int(str(row.get("rank")))
            except (TypeError, ValueError):
                continue
    return ranks


def visual_path(row: dict[str, Any], spec: dict[str, Any], fallback_rank: int) -> str:
    image_id = str(row.get("image_id"))
    rank = spec["review_ranks"].get(image_id, fallback_rank)
    path = spec["contact_dir"] / "annotated_images" / f"top_{rank:04d}_{image_id}.jpg"
    return str(path)


def crop_path(row: dict[str, Any], spec: dict[str, Any], fallback_rank: int) -> str:
    image_id = str(row.get("image_id"))
    rank = spec["review_ranks"].get(image_id, fallback_rank)
    path = spec["contact_dir"] / "crop_sheets" / f"top_{rank:04d}_{image_id}_crops.jpg"
    return str(path)


def pair_summary(pair: dict[str, Any] | None) -> str:
    if not pair:
        return "none"
    actions = "/".join(str(action) for action in pair.get("actions", []))
    return f"h{pair.get('subject_id')}-o{pair.get('object_id')} {pair.get('object_category_name')} {actions} ctx={pair.get('context_count')}"


def evidence_paths(row: dict[str, Any]) -> list[str]:
    paths = row.get("evidence_deletion_candidate_paths") or []
    if paths:
        return [str(path) for path in paths[:3]]
    primary = row.get("selected_primary_pair") or {}
    result: list[str] = []
    for ctx in primary.get("context_set", [])[:3]:
        result.append(f"P{primary.get('subject_id')} -> C{ctx.get('instance_id')}({ctx.get('category_name')}) -> O{primary.get('object_id')}")
    return result


def has_sports(row: dict[str, Any]) -> bool:
    for pair in row.get("top_pairs", [])[:5]:
        obj = str(pair.get("object_category_name", ""))
        groups = {str(group) for group in pair.get("action_groups", [])}
        if obj in SPORTS_OBJECTS or "sports_ball" in groups:
            return True
    return False


def smallest_object_pair(row: dict[str, Any]) -> dict[str, Any] | None:
    pairs = row.get("top_3_pairs") or row.get("top_pairs", [])[:3]
    if not pairs:
        return None
    return min(pairs, key=lambda pair: float(pair.get("object_area", 1.0)))


def risk_sentence(row: dict[str, Any]) -> str:
    counts = row.get("counts", {})
    comps = row.get("component_scores", {})
    primary = row.get("selected_primary_pair") or {}
    smallest = smallest_object_pair(row)
    if smallest and (float(smallest.get("object_area", 1.0)) < 0.004 or smallest.get("object_category_name") in SMALL_OBJECTS):
        return f"风险：{smallest.get('object_category_name')} object 太小，圆形节点和箭头标注可能需要放大裁剪。"
    if float(comps.get("ActionDiversity", 0.0)) < 0.8 or int(counts.get("num_unique_action_groups", 0)) < 3:
        return "风险：动作差异不够明显，Figure 1 的 evidence requirement 对比可能需要文字辅助。"
    if int(counts.get("num_valid_hoi_pairs", 0)) >= 7 or float(comps.get("CrowdingPenalty", 0.0)) >= 1.0:
        return "风险：线太多 Figure 1 会乱，更适合作为 diagnostic 或局部裁剪示例。"
    if has_sports(row):
        return "风险：太像 sports/group scene，可能被理解成 GroupHOI 或 local grouping 场景。"
    if "primary_pair_near_edge" in ";".join(str(w) for w in row.get("warnings", [])):
        return "风险：主 pair 不在画面中心区域，版面裁剪和三栏对比可能不够稳定。"
    if int(primary.get("context_count", 0)) < 3:
        return "风险：可解释上下文偏少，evidence paths 可能需要从 secondary pairs 补充。"
    return "风险：需要人工确认上下文物体是否真的支撑动作，而不是仅由空间接近造成。"


def recommendation_sentence(row: dict[str, Any], use: str) -> str:
    counts = row.get("counts", {})
    primary = row.get("selected_primary_pair") or {}
    object_name = primary.get("object_category_name", "object")
    actions = "/".join(str(action) for action in primary.get("actions", []))
    paths = evidence_paths(row)
    if use == "Figure 1":
        return (
            f"推荐作为 Figure 1，因为它有 {counts.get('num_valid_hoi_pairs')} 个 HO pairs、"
            f"{counts.get('num_unique_action_groups')} 类 action groups，主 pair `{object_name}:{actions}` 可与 {len(paths)} 条 evidence paths 形成三栏对比。"
        )
    if use == "Figure 2":
        return (
            f"推荐作为 Figure 2，因为主 pair `{object_name}:{actions}`、secondary pairs 和 context nodes 都较完整，"
            "适合展示 human/object/context 到 evidence path 的 pipeline。"
        )
    return (
        f"推荐作为 diagnostic，因为它包含 ambiguous tags 或 test split 场景，可用于展示 `{object_name}:{actions}` 的混淆、删除 evidence 或失败分析。"
    )


def manual_score(row: dict[str, Any], use: str) -> float:
    counts = row.get("counts", {})
    comps = row.get("component_scores", {})
    scores = row.get("bucket_scores", {})
    base = float(row.get("total_score", row.get("score", 0.0)))
    if use == "Figure 1":
        bucket_score = float(scores.get("A_score", 0.0))
        return (
            1.2 * bucket_score
            + 0.8 * min(int(counts.get("num_valid_hoi_pairs", 0)), 6)
            + 1.2 * min(int(counts.get("num_unique_action_groups", 0)), 4)
            + 0.6 * float(comps.get("EvidenceContextScore", 0.0))
            - 1.2 * float(comps.get("CrowdingPenalty", 0.0))
            - (1.0 if has_sports(row) else 0.0)
        )
    if use == "Figure 2":
        bucket_score = float(scores.get("B_score", 0.0))
        return (
            1.5 * bucket_score
            + 1.0 * float(comps.get("VisualClarity", 0.0))
            + 1.0 * float(comps.get("LayoutScore", 0.0))
            + 0.5 * min(int(counts.get("primary_pair_context_count", 0)), 5)
            - 0.8 * float(comps.get("CrowdingPenalty", 0.0))
        )
    bucket_score = float(scores.get("C_score", 0.0))
    return (
        1.5 * bucket_score
        + 0.6 * len(row.get("ambiguous_action_tags", []))
        + 0.4 * len(row.get("possible_prototype_shuffle_pairs", []))
        + (1.0 if row.get("whether_test_split") else 0.0)
        + 0.2 * base
    )


def collect_candidates() -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for spec in SPECS:
        spec["review_ranks"] = load_review_rank(spec["contact_dir"])
        all_rows = load_jsonl(spec["jsonl"])
        rows = [row for row in all_rows if str(row.get("image_id")) in spec["review_ranks"]]
        rows.sort(key=lambda row: spec["review_ranks"].get(str(row.get("image_id")), 10_000))
        for idx, row in enumerate(rows, start=1):
            key = f"{row.get('dataset')}:{row.get('split')}:{row.get('image_id')}"
            entry = merged.setdefault(key, {"row": row, "uses": [], "visuals": [], "scores": {}})
            use = str(spec["use"])
            entry["uses"].append(use)
            entry["visuals"].append(
                {
                    "use": use,
                    "source": spec["label"],
                    "visual_path": visual_path(row, spec, idx),
                    "crop_path": crop_path(row, spec, idx),
                    "source_rank": idx,
                }
            )
            entry["scores"][use] = manual_score(row, use)
    return list(merged.values())


def best_entries(entries: list[dict[str, Any]], use: str, limit: int = 2) -> list[dict[str, Any]]:
    candidates = [entry for entry in entries if use in entry["scores"]]
    candidates.sort(key=lambda entry: float(entry["scores"].get(use, 0.0)), reverse=True)
    return candidates[:limit]


def preferred_visual(entry: dict[str, Any], use: str) -> dict[str, Any]:
    for visual in entry["visuals"]:
        if visual["use"] == use:
            return visual
    return entry["visuals"][0]


def write_markdown(entries: list[dict[str, Any]]) -> None:
    lines = ["# EvidenceHOI Final Manual Review", ""]
    lines.append("Generated from scored candidates and existing contact sheets.")
    lines.append("")

    ranking_specs = [
        ("Best Figure 1 candidate", "Figure 1", 0),
        ("Backup Figure 1 candidate", "Figure 1", 1),
        ("Best Figure 2 candidate", "Figure 2", 0),
        ("Backup Figure 2 candidate", "Figure 2", 1),
        ("Best diagnostic candidate", "diagnostic", 0),
        ("Backup diagnostic candidate", "diagnostic", 1),
    ]
    lines.append("## Final Ranking")
    lines.append("")
    for idx, (title, use, rank_idx) in enumerate(ranking_specs, start=1):
        ranked = best_entries(entries, use, rank_idx + 1)
        if len(ranked) <= rank_idx:
            lines.append(f"{idx}. {title}: not available")
            continue
        entry = ranked[rank_idx]
        row = entry["row"]
        visual = preferred_visual(entry, use)
        lines.append(f"{idx}. {title}: `{row.get('image_id')}` ({row.get('file_name')})")
        lines.append(f"   visual: `{visual['visual_path']}`")
    lines.append("")

    entries.sort(key=lambda entry: max(float(score) for score in entry["scores"].values()), reverse=True)
    lines.append("## Candidate Checklist")
    lines.append("")
    for idx, entry in enumerate(entries, start=1):
        row = entry["row"]
        counts = row.get("counts", {})
        comps = row.get("component_scores", {})
        uses = ", ".join(dict.fromkeys(entry["uses"]))
        lines.append(f"### {idx}. {row.get('image_id')} ({row.get('file_name')})")
        lines.append(f"- intended_use: `{uses}`")
        lines.append(f"- image_path: `{row.get('image_path')}`")
        for visual in entry["visuals"]:
            lines.append(f"- {visual['use']} visual_path: `{visual['visual_path']}`")
            lines.append(f"- {visual['use']} crop_path: `{visual['crop_path']}`")
        lines.append(
            "- metrics: "
            f"pairs={counts.get('num_valid_hoi_pairs')}, verbs={counts.get('num_unique_action_categories')}, "
            f"objects={counts.get('num_unique_object_categories')}, action_groups={counts.get('num_unique_action_groups')}, "
            f"primary_context={counts.get('primary_pair_context_count')}, visible_pairs={counts.get('visible_pair_candidates')}"
        )
        lines.append(
            "- scores: "
            f"total={row.get('total_score', row.get('score'))}, A={row.get('bucket_scores', {}).get('A_score')}, "
            f"B={row.get('bucket_scores', {}).get('B_score')}, C={row.get('bucket_scores', {}).get('C_score')}, "
            f"visual_clarity={comps.get('VisualClarity')}, layout={comps.get('LayoutScore')}, crowding={comps.get('CrowdingPenalty')}"
        )
        lines.append(f"- primary_pair: {pair_summary(row.get('selected_primary_pair'))}")
        secondary = [pair_summary(pair) for pair in row.get("selected_secondary_pairs", [])[:2]]
        lines.append(f"- secondary_pairs: {secondary if secondary else 'none'}")
        paths = evidence_paths(row)
        lines.append(f"- evidence_paths: {paths if paths else 'none'}")
        use_for_sentence = entry["uses"][0]
        lines.append(f"- recommendation: {recommendation_sentence(row, use_for_sentence)}")
        lines.append(f"- risk: {risk_sentence(row)}")
        lines.append("- manual_checks:")
        for item in CHECKLIST:
            lines.append(f"  - [ ] {item}")
        lines.append("")
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    entries = collect_candidates()
    write_markdown(entries)
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
