#!/usr/bin/env python3
"""Build Stage 6 final Top-10 recommendations from Stage 5 pass rows only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image, ImageDraw

from feature_views import PROJECT_ROOT, rel


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v5_context_counterfactual"
DEFAULT_STAGE5_DIR = DEFAULT_OUTPUT_ROOT / "stage5_review"
DEFAULT_STAGE4_DIR = DEFAULT_OUTPUT_ROOT / "stage4_visuals"
DEFAULT_STAGE3_DIR = DEFAULT_OUTPUT_ROOT / "stage3_search"

TOP_COLUMNS = [
    "rank",
    "token",
    "tuple_id",
    "pair_id",
    "target_object_category",
    "gt_verb",
    "competitor",
    "cpos_category",
    "cneg_category",
    "pair_union_ambiguity_0_4",
    "cpos_counterfactual_gain_0_6",
    "cneg_misleading_plausibility_0_4",
    "selectivity_contrast_0_5",
    "final_score_0_28",
    "required_sentence",
    "evidence_card_path",
    "contact_panel_path",
    "source_visual_panel_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage5-dir", type=Path, default=DEFAULT_STAGE5_DIR)
    parser.add_argument("--stage4-dir", type=Path, default=DEFAULT_STAGE4_DIR)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--overwrite-stage6", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def text_image(path: Path, title: str, body: str, size: tuple[int, int] = (1400, 900)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0], 56), fill=(35, 35, 35))
    draw.text((18, 18), title[:170], fill=(255, 255, 255))
    y = 80
    for raw_line in body.splitlines():
        line = raw_line
        while len(line) > 120:
            draw.text((24, y), line[:120], fill=(20, 20, 20))
            y += 18
            line = line[120:]
        draw.text((24, y), line, fill=(20, 20, 20))
        y += 22
        if y > size[1] - 36:
            break
    image.save(path, quality=90)


def diversity_select(passed: pd.DataFrame) -> pd.DataFrame:
    if passed.empty:
        return passed.copy()
    work = passed.copy()
    for col in ["final_score_0_28", "pair_union_ambiguity_0_4", "cpos_counterfactual_gain_0_6", "cneg_misleading_plausibility_0_4", "selectivity_contrast_0_5"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    work = work.sort_values(
        ["final_score_0_28", "selectivity_contrast_0_5", "cpos_counterfactual_gain_0_6", "cneg_misleading_plausibility_0_4"],
        ascending=False,
    )
    object_counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    selected = []
    for row in work.itertuples(index=False):
        obj = str(row.target_object_category)
        combo = f"{row.gt_verb}|{row.competitor}"
        template = f"{row.cpos_category}|{row.cneg_category}"
        if object_counts[obj] >= 2:
            continue
        if combo_counts[combo] >= 2:
            continue
        if template_counts[template] >= 2:
            continue
        selected.append(row._asdict())
        object_counts[obj] += 1
        combo_counts[combo] += 1
        template_counts[template] += 1
        if len(selected) >= 10:
            break
    return pd.DataFrame(selected)


def build_required_sentence(row: Any) -> str:
    existing = str(getattr(row, "required_sentence", "") or "").strip()
    if existing and existing.lower() != "nan":
        return existing
    return (
        f"Pair-only 和 union-only 下，{row.gt_verb} 与 {row.competitor} 仍然混淆；"
        f"外部 {row.cpos_category} 是 GT 动作 {row.gt_verb} 的决定性证据；"
        f"外部 {row.cneg_category} 则诱导竞争动作 {row.competitor}；"
        "EvidenceHOI 应保留前者并拒绝后者。"
    )


def main() -> int:
    args = parse_args()
    stage6_dir = args.output_root / "stage6_top10"
    if stage6_dir.exists() and not args.overwrite_stage6:
        raise SystemExit(f"Refusing to overwrite existing stage6 directory: {stage6_dir}")
    for sub in [stage6_dir, stage6_dir / "evidence_cards", stage6_dir / "contact_sheets"]:
        sub.mkdir(parents=True, exist_ok=True)

    passed_path = args.stage5_dir / "passed_all_gates.csv"
    rejected_path = args.stage5_dir / "rejected_with_first_failed_gate.csv"
    visual_manifest_path = args.stage4_dir / "visual_manifest.csv"
    stage3_review_path = args.stage3_dir / "review_pool_balanced.csv"
    if not passed_path.exists():
        raise FileNotFoundError(passed_path)

    passed = pd.read_csv(passed_path)
    rejected = pd.read_csv(rejected_path) if rejected_path.exists() else pd.DataFrame()
    visual = pd.read_csv(visual_manifest_path) if visual_manifest_path.exists() else pd.DataFrame()
    stage3 = pd.read_csv(stage3_review_path, usecols=["tuple_id", "strict_pass", "pair_internal_decisive_proxy"]) if stage3_review_path.exists() else pd.DataFrame()

    selected = diversity_select(passed)
    if not selected.empty and not visual.empty:
        selected = selected.merge(
            visual[["token", "counterfactual_panel_path", "final_panel_384_path"]],
            on="token",
            how="left",
        )
    top_rows: list[dict[str, Any]] = []
    contact_images: list[Path] = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        sentence = build_required_sentence(row)
        card_path = stage6_dir / "evidence_cards" / f"rank_{rank:02d}_{row.token}.md"
        contact_path = stage6_dir / "contact_sheets" / f"rank_{rank:02d}_{row.token}_card.jpg"
        source_panel = str(getattr(row, "counterfactual_panel_path", ""))
        body = f"""# Evidence Card {rank}

- Target pair and GT: `{row.target_object_category}` / `{row.gt_verb}`
- Competitor: `{row.competitor}`
- Pair/union ambiguity: `{row.pair_union_ambiguity_0_4}`
- c+: `{row.cpos_category}`; gain `{row.cpos_counterfactual_gain_0_6}`
- c-: `{row.cneg_category}`; misleading gain `{row.cneg_misleading_plausibility_0_4}`
- Scene-level misselection: see manual adjudication notes.
- Neighborhood misselection: see manual adjudication notes.
- EvidenceHOI should: retain c+ and reject c-.
- Max risk and source labels: see Stage 5 adjudication row and Stage 4 blind key.

Core sentence: {sentence}

Source visual panel: `{source_panel}`
"""
        card_path.write_text(body, encoding="utf-8")
        text_image(contact_path, f"Rank {rank}: {row.gt_verb} vs {row.competitor}", body)
        contact_images.append(contact_path)
        top_rows.append({
            "rank": rank,
            "token": row.token,
            "tuple_id": row.tuple_id,
            "pair_id": row.pair_id,
            "target_object_category": row.target_object_category,
            "gt_verb": row.gt_verb,
            "competitor": row.competitor,
            "cpos_category": row.cpos_category,
            "cneg_category": row.cneg_category,
            "pair_union_ambiguity_0_4": row.pair_union_ambiguity_0_4,
            "cpos_counterfactual_gain_0_6": row.cpos_counterfactual_gain_0_6,
            "cneg_misleading_plausibility_0_4": row.cneg_misleading_plausibility_0_4,
            "selectivity_contrast_0_5": row.selectivity_contrast_0_5,
            "final_score_0_28": row.final_score_0_28,
            "required_sentence": sentence,
            "evidence_card_path": rel(card_path),
            "contact_panel_path": rel(contact_path),
            "source_visual_panel_path": source_panel,
        })

    write_csv(stage6_dir / "final_top10.csv", top_rows, TOP_COLUMNS)

    actual_count = len(top_rows)
    shortage = max(0, 10 - actual_count)
    blocker_counts = rejected["first_failed_gate"].value_counts().to_dict() if "first_failed_gate" in rejected.columns else {}
    strict_available = int(stage3["strict_pass"].sum()) if "strict_pass" in stage3.columns else None
    pair_internal_decisive = int(stage3["pair_internal_decisive_proxy"].sum()) if "pair_internal_decisive_proxy" in stage3.columns else None

    analysis = f"""# Stage 6 Final Top-10 Analysis

## Source Rule

Final Top-10 is formed only from `stage5_review/passed_all_gates.csv`. Rejected rows and pair-internal cue candidates are not used to fill slots.

## Result

- Passed Stage 5 candidates available: `{len(passed)}`
- Final selected count: `{actual_count}`
- Shortage relative to Top-10: `{shortage}`
- Strict Stage 3 candidates available before manual review: `{strict_available}`
- Pair-internal decisive candidates are not eligible for filling: `{pair_internal_decisive}`

Because Stage 5 had no visually adjudicated pass rows, Stage 6 produces an empty Top-10. This is intentional and avoids using Figure-5-like pair-internal cue candidates or proxy scores to compensate for missing manual review.

## Required Gates

Every final candidate would need to satisfy all Stage 6 gates: readable target pair, real pair/union ambiguity, external c+, positive gain, c+ maskout loss, independent c-, c- support for a competitor, scene/neighborhood rationale, final-size readability, and complete provenance. No candidate can be certified for these gates without Stage 5 visual pass rows.

## Directed Follow-Up

1. Run real human review on `stage4_visuals/blind_pack_A/` and `stage4_visuals/blind_pack_B/`.
2. Fill `stage5_review/blind_round_A.csv`, `blind_round_BC.csv`, and `adjudication.csv` with actual visual scores.
3. Require all minima independently: pair ambiguity >= 3, union ambiguity >= 3, positive gain >= 4, misleading plausibility >= 3, selectivity >= 4, final score >= 21.
4. Re-run Stage 6 from the updated `passed_all_gates.csv`.
5. If fewer than 10 remain, prioritize new full-index search/review for objects underrepresented in Stage 3 Strict: suitcase, TV, book, and diverse external contexts such as remote, couch, car/trunk, bed, table, tools, helmet, road.
"""
    (stage6_dir / "final_top10_analysis.md").write_text(analysis, encoding="utf-8")

    primary = f"""# Primary And Backups

No primary candidate is recommended yet because `passed_all_gates.csv` is empty.

## Backups

No backups are promoted from rejected candidates. Stage 6 does not use total-score compensation, Stage 3 proxies, or pair-internal decisive cases to fill Top-10.

## Required Next Action

Complete actual Stage 5 human visual adjudication, then re-run Stage 6.
"""
    (stage6_dir / "primary_and_backups.md").write_text(primary, encoding="utf-8")

    high_rejects = f"""# High Scoring Rejects

No rejected candidate is promoted.

## Rejection Summary

`first_failed_gate` counts from Stage 5 rejected rows:

```json
{json.dumps(blocker_counts, ensure_ascii=False, indent=2)}
```

All available rejected rows failed before manual visual adjudication with `manual_visual_review_unavailable`. They may be high-quality Stage 3/4 candidates, but they are not eligible for Stage 6 without real Round A-D scores.
"""
    (stage6_dir / "high_scoring_rejects.md").write_text(high_rejects, encoding="utf-8")

    # Contact sheet placeholder if no cards exist.
    if not contact_images:
        text_image(
            stage6_dir / "contact_sheets" / "final_top10_empty.jpg",
            "Stage 6 Final Top-10 Empty",
            "No candidates passed Stage 5. Actual final Top-10 count is 0. Complete human review before creating Figure 1 materials.",
        )

    repro = {
        "stage": 6,
        "inputs": {
            "passed_all_gates": rel(passed_path),
            "rejected_with_first_failed_gate": rel(rejected_path),
            "visual_manifest": rel(visual_manifest_path),
            "stage3_review_pool": rel(stage3_review_path),
        },
        "selection_rule": "Stage 5 pass rows only; max 2 per object, max 2 per GT/competitor, max 2 per c+/c- template.",
        "selected_count": actual_count,
        "target_top10_shortage": shortage,
        "diversity_constraints_applied": True,
        "pair_internal_decisive_fill_allowed": False,
        "paper_figure_created": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    repro_path = stage6_dir / "reproducibility_manifest.json"
    repro_path.write_text(json.dumps(repro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    outputs = [
        stage6_dir / "final_top10.csv",
        stage6_dir / "final_top10_analysis.md",
        stage6_dir / "primary_and_backups.md",
        stage6_dir / "high_scoring_rejects.md",
        repro_path,
        *sorted((stage6_dir / "evidence_cards").glob("*")),
        *sorted((stage6_dir / "contact_sheets").glob("*")),
        Path(__file__),
    ]
    manifest = {
        "stage": 6,
        "task": "final Top-10 and primary recommendation from Stage 5 passes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": rel(args.output_root),
        "stage6_dir": rel(stage6_dir),
        "stopped_after_manifest": True,
        "paper_figure_created": False,
        "source_passed_all_gates_only": True,
        "passed_all_gates_count": len(passed),
        "final_top10_count": actual_count,
        "shortage": shortage,
        "pair_internal_decisive_used_to_fill": False,
        "requires_directed_followup": shortage > 0,
        "outputs": [],
    }
    for path in outputs:
        if path.exists() and path.is_file():
            manifest["outputs"].append({"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest_path = args.output_root / "manifest_stage6.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest_path), "final_top10_count": actual_count, "shortage": shortage}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
