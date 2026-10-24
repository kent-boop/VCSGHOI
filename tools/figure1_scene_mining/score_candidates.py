#!/usr/bin/env python3
"""Merge manual review with mined candidates and write shortlist/report outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT / "outputs/raw_candidates.csv"
DEFAULT_REVIEW = ROOT / "manual_review.yaml"
DEFAULT_OUTPUT = ROOT / "outputs"
SCORE_KEYS = ["ambiguity", "decisive_evidence", "distractor", "topology", "cropability", "composition", "long_range", "source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def report(frame: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = ["# EvidenceHOI Figure 1 候选底图最终报告", ""]
    lines += ["## 1. 数据资源检查", "",
              "- 使用数据集：HICO-DET trainval 与 test。",
              "- 图片根目录：`/share/share/Datasets/hico_20160224_det/images/{train2015,test2015}`。",
              "- 标注与 pair/instance 索引：复用 `outputs/figure1_search_v2/index/` 中经过连接关系与坐标系审计的 parquet 索引。",
              "- V-COCO：项目本地目录为空，本轮未使用。",
              "- 网络图片：未使用；因此不存在来源/许可不明的网页图片。", ""]
    lines += ["## 2. 筛选统计", "",
              f"- 扫描目标 pair：`{sum(summary['scanned_target_pairs_by_scenario'].values())}`。",
              f"- 自动结构过滤后保留原始候选：`{summary['retained_raw_total']}`（六类各最多 250）。",
              "- 人工查看：六类各前 40 张 contact-sheet 候选，并对边界候选追加查看原图。",
              f"- 最终人工 shortlist：`{len(frame)}`。",
              "- 自动分数只用于检索排序，以下 100 分制结果来自原图人工视觉复核。", ""]
    lines += ["## 3. Top 10 候选", "", "| Rank | Score | Scene | Image | GT action | Decisive evidence | Main risk |", "|---:|---:|---|---|---|---|---|"]
    for _, row in frame.iterrows():
        lines.append(f"| {int(row['rank'])} | {int(row['total_score'])} | {row['scenario']} | `{row['split']}:{row['file_name']}` | `{row['annotated_actions']}` | {row['decisive_evidence']} | {row['reviewer_notes']} |")
    lines += ["", "## 4. Top 3 最终候选", ""]
    for _, row in frame.head(3).iterrows():
        lines += [f"### Candidate {int(row['rank'])}: `{row['file_name']}` ({row['scenario']})", "",
                  f"- Dataset / split / image ID：HICO-DET / `{row['split']}` / `{row['image_id']}`",
                  f"- GT action：`{row['annotated_actions']}`",
                  f"- Alternative plausible actions：`{row['alternative_plausible_actions']}`",
                  f"- H：{row['figure_node_H']}", f"- A：{row['figure_node_A']}", f"- O：{row['figure_node_O']}",
                  f"- S：{row['figure_node_S']}", f"- D：{row['figure_node_D']}", f"- C：{row['figure_node_C']}", "",
                  f"Panel (a)：{row['generic_aggregation_story']}", "",
                  f"Panel (b)：{row['evidencehoi_story']}", "",
                  f"风险：{row['reviewer_notes']}", ""]
    first = frame.iloc[0]
    lines += ["## 5. 最终推荐", "",
              f"明确推荐使用 **`{first['file_name']}`**（HICO-DET `{first['split']}`，GT `{first['annotated_actions']}`）。",
              "",
              "它优于其余候选的核心原因是：目标人–刀 pair 保留 `hold`/`cut_with` 的解释空间，苹果和切开食材提供清晰的外部决定性证据，同时陶罐、篮筐、木桶、长凳和额外苹果形成自然但非决定性的上下文；人物数量为一，画面右侧还保留了拓扑与文字的绘制空间。",
              "",
              "最大风险是刀身较小。后续裁剪必须保留双手、刀刃、苹果/切开食材和右侧陶罐；不建议只裁人物上半身。Figure 1 的动作文字必须使用 HICO-DET 的真实名称 `cut_with`，可在解释性文字中对比同属数据集的 `hold`，不要写成数据集中不存在的自由动作名称。", ""]
    lines += ["## 6. 可复现命令", "", "```bash",
              "python tools/figure1_scene_mining/mine_candidates.py --max-per-scenario 250",
              "python tools/figure1_scene_mining/render_contact_sheet.py --top-k 40 --per-page 20",
              "python tools/figure1_scene_mining/score_candidates.py",
              "python tools/figure1_scene_mining/annotate_candidates.py --top-k 3",
              "```", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    raw = pd.read_csv(args.raw)
    review_doc = yaml.safe_load(args.review.read_text(encoding="utf-8"))
    review_rows = review_doc["candidates"]
    review = pd.DataFrame(review_rows)
    unknown = sorted(set(review["candidate_id"]) - set(raw["candidate_id"]))
    if unknown:
        raise ValueError(f"Manual review contains unknown candidate IDs: {unknown}")
    merged = review.merge(raw, on="candidate_id", how="left", validate="one_to_one")

    for key in SCORE_KEYS:
        merged[f"{key}_score"] = merged["scores"].map(lambda item: int(item[key]))
    merged["total_score"] = merged[[f"{key}_score" for key in SCORE_KEYS]].sum(axis=1)
    merged = merged.sort_values(["total_score", "candidate_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))
    merged["alternative_plausible_actions"] = merged["alternative_plausible_actions"].map(dumps)
    merged["manual_decisive_bboxes"] = merged["manual_decisive_bboxes"].map(dumps)
    merged["manual_distractor_bboxes"] = merged["manual_distractor_bboxes"].map(dumps)
    merged["human_bbox"] = merged["human_bbox"].astype(str)
    merged["object_bbox"] = merged["object_bbox"].astype(str)
    merged = merged.drop(columns=["scores"])

    preferred = [
        "rank", "total_score", "dataset", "split", "image_id", "image_path", "image_width", "image_height",
        "human_bbox", "object_bbox", "target_object", "annotated_actions", "alternative_plausible_actions",
        "decisive_evidence", "scene_evidence", "potential_distractors", "long_range_relation",
        "ambiguity_score", "decisive_evidence_score", "distractor_score", "topology_score", "cropability_score",
        "composition_score", "long_range_score", "source_score", "figure_node_H", "figure_node_A", "figure_node_O",
        "figure_node_S", "figure_node_D", "figure_node_C", "generic_aggregation_story", "evidencehoi_story",
        "license_or_source_note", "reviewer_notes", "scenario", "candidate_id", "file_name", "manual_decisive_bboxes",
        "manual_distractor_bboxes", "number_of_people", "number_of_entities", "number_of_context_objects",
    ]
    remaining = [col for col in merged.columns if col not in preferred]
    shortlist = merged[[col for col in preferred if col in merged.columns] + remaining]
    shortlist.to_csv(output_dir / "shortlist.csv", index=False)

    summary = json.loads((output_dir / "mining_summary.json").read_text(encoding="utf-8"))
    (output_dir / "final_report.md").write_text(report(shortlist, summary), encoding="utf-8")
    (output_dir / "shortlist_summary.json").write_text(json.dumps({
        "manual_review_count": len(shortlist), "top3_candidate_ids": shortlist.head(3)["candidate_id"].tolist(),
        "final_recommendation": shortlist.iloc[0]["candidate_id"], "scores_are_manual": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(shortlist[["rank", "total_score", "scenario", "file_name", "annotated_actions"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
