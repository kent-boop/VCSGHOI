#!/usr/bin/env python3
"""Validate and score manually reviewed dual-ambiguity Figure 1 candidates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_REVIEW = ROOT / "manual_dual_ambiguity_review.yaml"
DEFAULT_INDEX = PROJECT_ROOT / "outputs/figure1_search_v2/index"
DEFAULT_OUTPUT = ROOT / "outputs/dual_ambiguity_round3/final"
SCORE_KEYS = ["pair_assignment", "interaction_ambiguity", "decisive_relation", "chain_contrast", "single_column_clarity", "topology_completeness", "source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sheet(frame: pd.DataFrame, path: Path) -> None:
    card_w, card_h, image_h, cols = 410, 340, 250, 3
    rows = (len(frame) + cols - 1) // cols
    canvas = Image.new("RGB", (card_w * cols, card_h * rows), (235, 238, 242))
    draw = ImageDraw.Draw(canvas)
    regular = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    for index, (_, row) in enumerate(frame.iterrows()):
        image = Image.open(row["image_path"]).convert("RGB")
        image = ImageOps.contain(image, (card_w - 12, image_h - 12), Image.Resampling.LANCZOS)
        col, grid_row = index % cols, index // cols
        x = col * card_w + (card_w - image.width) // 2
        y = grid_row * card_h + 6 + (image_h - image.height) // 2
        canvas.paste(image, (x, y))
        tx, ty = col * card_w + 8, grid_row * card_h + image_h + 3
        draw.text((tx, ty), f"#{int(row['rank'])} {int(row['total_score'])}/100 {row['target_role']} vs {row['competitor_role']}", fill=(20, 25, 35), font=bold)
        draw.text((tx, ty + 22), row["file_name"], fill=(40, 45, 55), font=regular)
        draw.text((tx, ty + 42), f"GT {row['target_actions']} vs {row['competitor_actions']}", fill=(55, 60, 70), font=regular)
        draw.text((tx, ty + 62), row["decisive_relation"][:56], fill=(25, 110, 60), font=regular)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)


def report(frame: pd.DataFrame) -> str:
    lines = ["# EvidenceHOI Figure 1 Dual-Ambiguity Round 3", "",
             "## Selection contract", "",
             "A candidate must simultaneously provide pair-assignment ambiguity, interaction-hypothesis ambiguity, a decisive third-party relation, a competing mismatched relation, and single-column readability.", "",
             "## Ranked candidates", "",
             "| Rank | Score | Image | Target GT | Competing GT | Decisive chain | Main risk |",
             "|---:|---:|---|---|---|---|---|"]
    for _, row in frame.iterrows():
        lines.append(f"| {int(row['rank'])} | {int(row['total_score'])} | `{row['file_name']}` | `{row['target_actions']}` | `{row['competitor_actions']}` | {row['decisive_relation']} | {row['risk']} |")
    lines += ["", "## Top 3", ""]
    for _, row in frame.head(3).iterrows():
        lines += [f"### {int(row['rank'])}. `{row['file_name']}`", "",
                  f"- Pair assignment: {row['pair_assignment_story']}",
                  f"- Interaction ambiguity: {row['interaction_hypotheses']}",
                  f"- Decisive relation: `{row['decisive_relation']}`",
                  f"- Critical mismatch: {row['mismatch_relation']}",
                  f"- Shared cues: {row['shared_cues']}",
                  f"- Suppressible pair-specific cues: {row['suppressible_cues']}",
                  f"- Nodes: {row['figure_nodes']}",
                  f"- Risk: {row['risk']}", ""]
    first = frame.iloc[0]
    lines += ["## Final recommendation", "",
              f"Use **`{first['file_name']}`** as the new primary Figure 1 base image. It is the first local candidate that cleanly combines a shared ball, two meaningful human–ball pairs, different evidence chains, a large mediating object, and a visible mismatch condition without turning the figure into a crowded detection scene.", "",
              "Recommended target anchor: batter–sports ball with HICO-DET GT `hit`. Competing pair: wicketkeeper–sports ball with GT `catch`. The visual prototype comparison should be independent support, not a mutually exclusive softmax: bat-mediated relation is strengthened for hit, while the wicketkeeper catch chain is retained only for its own pair and suppressed for the batter pair.", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    review = yaml.safe_load(args.review.read_text(encoding="utf-8"))["candidates"]
    frame = pd.DataFrame(review)
    pairs = pd.read_parquet(args.index_dir / "pairs.parquet").set_index("pair_id", drop=False)
    images = pd.read_parquet(args.index_dir / "image_summary.parquet").drop_duplicates("file_name").set_index("file_name", drop=False)
    for _, row in frame.iterrows():
        for key in ["target_candidate_id", "competitor_candidate_id"]:
            if row[key] not in pairs.index:
                raise ValueError(f"Unknown pair ID: {row[key]}")
        target_gt = set(json.loads(pairs.loc[row["target_candidate_id"], "verb_names_json"]))
        competitor_gt = set(json.loads(pairs.loc[row["competitor_candidate_id"], "verb_names_json"]))
        if not set(row["target_actions"]).issubset(target_gt):
            raise ValueError(f"Target GT mismatch for {row['file_name']}")
        if not set(row["competitor_actions"]).issubset(competitor_gt):
            raise ValueError(f"Competitor GT mismatch for {row['file_name']}")
    frame["dataset"] = "HICO-DET"
    frame["split"] = frame["target_candidate_id"].map(lambda value: value.split(":", 1)[0])
    frame["image_path"] = frame["file_name"].map(lambda name: str(images.loc[name, "image_path"]))
    frame["image_id"] = frame["file_name"].map(lambda name: str(images.loc[name, "image_id"]))
    frame["image_width"] = frame["file_name"].map(lambda name: int(images.loc[name, "width"]))
    frame["image_height"] = frame["file_name"].map(lambda name: int(images.loc[name, "height"]))
    for key in SCORE_KEYS:
        frame[f"{key}_score"] = frame["scores"].map(lambda value: int(value[key]))
    frame["total_score"] = frame[[f"{key}_score" for key in SCORE_KEYS]].sum(axis=1)
    frame = frame.sort_values(["total_score", "file_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    for key in ["target_actions", "competitor_actions", "target_human_bbox", "competitor_human_bbox", "shared_object_bbox", "decisive_relation_bbox", "context_bboxes"]:
        frame[key] = frame[key].map(dumps)
    frame = frame.drop(columns=["scores"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "shortlist.csv", index=False)
    (output / "final_report.md").write_text(report(frame), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({"shortlist_size": len(frame), "top3": frame.head(3)["file_name"].tolist(), "final_recommendation": frame.iloc[0]["file_name"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    originals = output / "shortlist_originals"
    originals.mkdir(parents=True, exist_ok=True)
    for _, row in frame.iterrows():
        destination = originals / f"{int(row['rank']):02d}_{row['file_name']}"
        if not destination.exists():
            shutil.copy2(row["image_path"], destination)
    sheet(frame, output / "shortlist.jpg")
    print(frame[["rank", "total_score", "file_name", "target_actions", "competitor_actions"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
