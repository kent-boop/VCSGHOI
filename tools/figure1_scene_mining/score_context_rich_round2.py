#!/usr/bin/env python3
"""Build the manually reviewed context-rich Figure 1 shortlist."""

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
DEFAULT_RAW = ROOT / "outputs/context_rich_round2/context_rich_candidates.csv"
DEFAULT_REVIEW = ROOT / "manual_context_rich_review.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/context_rich_round2/final"
SCORE_KEYS = ["ambiguity", "decisive_evidence", "distractor", "topology", "cropability", "composition", "long_range", "source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_sheet(frame: pd.DataFrame, path: Path) -> None:
    card_w, card_h, image_h, cols = 340, 310, 230, 5
    rows = (len(frame) + cols - 1) // cols
    sheet = Image.new("RGB", (card_w * cols, card_h * rows), (235, 238, 242))
    draw = ImageDraw.Draw(sheet)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 13)
    bold = ImageFont.truetype(bold_path, 15)
    for n, (_, row) in enumerate(frame.iterrows()):
        image = Image.open(row["image_path"]).convert("RGB")
        image = ImageOps.contain(image, (card_w - 12, image_h - 12), Image.Resampling.LANCZOS)
        col, grid_row = n % cols, n // cols
        x = col * card_w + (card_w - image.width) // 2
        y = grid_row * card_h + 6 + (image_h - image.height) // 2
        sheet.paste(image, (x, y))
        tx, ty = col * card_w + 8, grid_row * card_h + image_h + 3
        draw.text((tx, ty), f"#{int(row['rank'])} {int(row['total_score'])}/100 {row['scenario']}", fill=(20, 25, 35), font=bold)
        draw.text((tx, ty + 21), row["file_name"], fill=(35, 40, 50), font=font)
        draw.text((tx, ty + 40), f"GT {row['annotated_actions']} | contexts {int(row['visible_context_count'])}", fill=(55, 60, 70), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=94)


def make_report(frame: pd.DataFrame, raw_count: int) -> str:
    lines = ["# EvidenceHOI Figure 1 Context-rich Round 2", "",
             "## Why a second round was necessary", "",
             "The first shortlist over-weighted target-pair clarity and often retained images containing only a person, the target object, and one cue. Round 2 treats context richness as an eligibility condition: every shortlisted image exposes at least three visually distinguishable context cues beyond the target pair.", "",
             "## Search result", "",
             f"- Structurally eligible context-rich image pool: `{raw_count}`.",
             f"- Pixel-reviewed shortlist: `{len(frame)}` unique images.",
             "- Network images used: `0`; every image is traceable to local HICO-DET.",
             "- Automatic scores were used only to order the review pool; the 100-point scores below are manual.", "",
             "## Revised Top 10", "",
             "| Rank | Score | Scenario | Image | GT | Visible context | Main risk |",
             "|---:|---:|---|---|---|---|---|"]
    for _, row in frame.iterrows():
        lines.append(f"| {int(row['rank'])} | {int(row['total_score'])} | {row['scenario']} | `{row['file_name']}` | `{row['annotated_actions']}` | {row['potential_distractors']} | {row['reviewer_notes']} |")
    lines += ["", "## Top 3 narrative mapping", ""]
    for _, row in frame.head(3).iterrows():
        lines += [f"### {int(row['rank'])}. `{row['file_name']}`", "",
                  f"- H: {row['figure_node_H']}", f"- A: {row['figure_node_A']}", f"- O: {row['figure_node_O']}",
                  f"- S: {row['figure_node_S']}", f"- D: {row['figure_node_D']}", f"- C: {row['figure_node_C']}",
                  f"- Panel (a): {row['generic_aggregation_story']}",
                  f"- Panel (b): {row['evidencehoi_story']}", ""]
    first = frame.iloc[0]
    lines += ["## Revised final recommendation", "",
              f"Use **`{first['file_name']}`** as the revised primary image. It preserves person–knife action ambiguity while providing a decisive banana/cut-piece cue and at least seven separable context cues: paper towels, fruit tray, bag, sink, cabinets, notes, and peel. Unlike the sparse first-round images, both useful evidence and suppressible context remain visible at the same time.", "",
              "Crop vertically around the person and counter, but retain the right-side paper towels, fruit tray, banana peel, and lower-right sink edge. The figure text must use HICO-DET action `cut_with`; `hold` may appear only as a dataset-valid alternative hypothesis.", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    raw = pd.read_csv(args.raw)
    review = pd.DataFrame(yaml.safe_load(args.review.read_text(encoding="utf-8"))["candidates"])
    unknown = sorted(set(review["candidate_id"]) - set(raw["candidate_id"]))
    if unknown:
        raise ValueError(f"Unknown candidate IDs: {unknown}")
    merged = review.merge(raw, on="candidate_id", how="left", validate="one_to_one")
    for key in SCORE_KEYS:
        merged[f"{key}_score"] = merged["scores"].map(lambda item: int(item[key]))
    merged["total_score"] = merged[[f"{key}_score" for key in SCORE_KEYS]].sum(axis=1)
    merged = merged.sort_values(["total_score", "candidate_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))
    merged["alternative_plausible_actions"] = merged["alternative_plausible_actions"].map(dumps)
    merged["manual_decisive_bboxes"] = merged["manual_decisive_bboxes"].map(dumps)
    merged["manual_distractor_bboxes"] = merged["manual_distractor_bboxes"].map(dumps)
    merged = merged.drop(columns=["scores"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "shortlist.csv", index=False)
    (output / "final_report.md").write_text(make_report(merged, len(raw)), encoding="utf-8")
    (output / "shortlist_summary.json").write_text(json.dumps({
        "pool_size": len(raw), "shortlist_size": len(merged), "minimum_visible_context": int(merged["visible_context_count"].min()),
        "top3": merged.head(3)["candidate_id"].tolist(), "final_recommendation": merged.iloc[0]["candidate_id"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    originals = output / "shortlist_originals"
    originals.mkdir(parents=True, exist_ok=True)
    for _, row in merged.iterrows():
        destination = originals / f"{int(row['rank']):02d}_{row['scenario']}_{row['file_name']}"
        if not destination.exists():
            shutil.copy2(row["image_path"], destination)
    make_sheet(merged, output / "shortlist_top10.jpg")
    print(merged[["rank", "total_score", "scenario", "file_name", "visible_context_count"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
