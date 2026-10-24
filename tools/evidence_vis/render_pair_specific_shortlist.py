#!/usr/bin/env python3
"""Render pair-specific EvidenceHOI candidates for manual inspection.

Each panel shows the same source image twice. The left view overlays the first
human-object pair and its selected contexts; the right view overlays the
controlled partner pair and its selected contexts. This keeps dense boxes from
both pairs readable without changing the underlying Trace data.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from PIL import Image, ImageDraw

from render_evidence_shortlist import (
    context_ids,
    draw_labeled_box,
    load_trace,
    resolve_image,
    scale_box,
)


PAIR1_COLORS = {
    "human": (190, 30, 45),
    "object": (30, 90, 190),
    "context": (225, 125, 20),
}
PAIR2_COLORS = {
    "human": (115, 35, 150),
    "object": (15, 135, 100),
    "context": (20, 145, 160),
}


def box_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float()
    second = second.float()
    top_left = torch.maximum(first[:2], second[:2])
    bottom_right = torch.minimum(first[2:], second[2:])
    intersection = (bottom_right - top_left).clamp(min=0.0).prod()
    first_area = (first[2:] - first[:2]).clamp(min=0.0).prod()
    second_area = (second[2:] - second[:2]).clamp(min=0.0).prod()
    union = first_area + second_area - intersection
    return float((intersection / union.clamp(min=1e-8)).item())


def controlled_endpoint_iou(
    record: Mapping[str, Any],
    row: Mapping[str, str],
) -> float:
    boxes = record["nodes"]["boxes_xyxy_norm"].float()
    pairs = record["pairs"]["indices"].long()
    first_idx = int(row["pair_idx_1"])
    second_idx = int(row["pair_idx_2"])
    human1, object1 = [int(value) for value in pairs[first_idx].tolist()]
    human2, object2 = [int(value) for value in pairs[second_idx].tolist()]
    if row.get("control_type") == "same_object_node":
        return box_iou(boxes[human1], boxes[human2])
    return box_iou(boxes[object1], boxes[object2])


def draw_pair_view(
    record: Mapping[str, Any],
    source: Image.Image,
    pair_idx: int,
    pair_name: str,
    colors: Mapping[str, Tuple[int, int, int]],
) -> Tuple[Image.Image, List[int]]:
    image = source.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    boxes = record["nodes"]["boxes_xyxy_norm"].float()
    pair_indices = record["pairs"]["indices"].long()
    width, height = image.size

    human_idx, object_idx = [int(value) for value in pair_indices[pair_idx].tolist()]
    draw_labeled_box(
        draw,
        scale_box(boxes[human_idx], width, height),
        f"{pair_name}-H{human_idx}",
        colors["human"],
        5,
    )
    draw_labeled_box(
        draw,
        scale_box(boxes[object_idx], width, height),
        f"{pair_name}-O{object_idx}",
        colors["object"],
        5,
    )

    selected = context_ids(record.get("graph") or {}, pair_idx)
    for rank, node_idx in enumerate(selected, start=1):
        draw_labeled_box(
            draw,
            scale_box(boxes[node_idx], width, height),
            f"{pair_name}-C{rank}:{node_idx}",
            colors["context"],
            3,
        )
    return image, selected


def add_view_title(image: Image.Image, title: str) -> Image.Image:
    band_height = 36
    output = Image.new("RGB", (image.width, image.height + band_height), "white")
    ImageDraw.Draw(output).text((8, 10), title, fill="black")
    output.paste(image, (0, band_height))
    return output


def build_panel(
    record: Mapping[str, Any],
    source: Image.Image,
    row: Mapping[str, str],
) -> Image.Image:
    first_idx = int(row["pair_idx_1"])
    second_idx = int(row["pair_idx_2"])
    first, first_contexts = draw_pair_view(
        record,
        source,
        first_idx,
        "P1",
        PAIR1_COLORS,
    )
    second, second_contexts = draw_pair_view(
        record,
        source,
        second_idx,
        "P2",
        PAIR2_COLORS,
    )
    first = add_view_title(
        first,
        f"Pair {first_idx}: {row.get('object_name_1', row.get('object_label_1', '?'))}",
    )
    second = add_view_title(
        second,
        f"Pair {second_idx}: {row.get('object_name_2', row.get('object_label_2', '?'))}",
    )

    header_height = 48
    footer_height = 48
    gap = 8
    panel = Image.new(
        "RGB",
        (first.width + second.width + gap, header_height + first.height + footer_height),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (8, 8),
        (
            f"image={row.get('image_id')}  control={row.get('control_type')}  "
            f"score={float(row.get('pair_specific_score', 0.0)):.3f}  "
            f"set-distance={float(row.get('context_set_distance', 0.0)):.3f}  "
            f"endpoint-iou={float(row.get('controlled_endpoint_iou', 0.0)):.3f}"
        ),
        fill="black",
    )
    panel.paste(first, (0, header_height))
    panel.paste(second, (first.width + gap, header_height))
    draw.text(
        (8, header_height + first.height + 12),
        f"P1 contexts={first_contexts}    P2 contexts={second_contexts}",
        fill="black",
    )
    return panel


def write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-per-image", type=int, default=1)
    parser.add_argument("--max-controlled-endpoint-iou", type=float, default=1.0)
    parser.add_argument("--image-pattern")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.ranking_csv).open("r", encoding="utf-8")))
    image_counts: Dict[str, int] = defaultdict(int)
    rendered_rows: List[Dict[str, str]] = []
    failures: List[str] = []

    for row in rows:
        image_key = str(row.get("image_id", row.get("file_name", "unknown")))
        if args.max_per_image > 0 and image_counts[image_key] >= args.max_per_image:
            continue
        trace_path = Path(row["record_path"])
        try:
            record = load_trace(trace_path)
            candidate_row = dict(row)
            endpoint_iou = controlled_endpoint_iou(record, candidate_row)
            candidate_row["controlled_endpoint_iou"] = f"{endpoint_iou:.6f}"
            if endpoint_iou > args.max_controlled_endpoint_iou:
                continue
            image_path = resolve_image(
                record,
                Path(args.image_root),
                args.image_pattern,
            )
            if image_path is None:
                raise FileNotFoundError(f"image not found for {trace_path}")
            source = Image.open(image_path)
            panel = build_panel(record, source, candidate_row)
        except (OSError, KeyError, ValueError, IndexError, RuntimeError) as exc:
            failures.append(f"{image_key}: {exc}")
            continue

        rank = len(rendered_rows) + 1
        first_idx = int(row["pair_idx_1"])
        second_idx = int(row["pair_idx_2"])
        panel.save(
            output_dir / f"{rank:03d}_{image_key}_pairs{first_idx}_{second_idx}.png"
        )
        rendered_rows.append(candidate_row)
        image_counts[image_key] += 1
        if len(rendered_rows) >= args.limit:
            break

    write_rows(output_dir / "rendered_candidates.csv", rendered_rows)
    if failures:
        (output_dir / "render_failures.txt").write_text(
            "\n".join(failures) + "\n",
            encoding="utf-8",
        )
    print(f"Rendered {len(rendered_rows)} pair-specific panels; failures={len(failures)}")


if __name__ == "__main__":
    main()
