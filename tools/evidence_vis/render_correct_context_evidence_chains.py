#!/usr/bin/env python3
"""Render publication-oriented single-pair EvidenceHOI evidence chains."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from render_evidence_shortlist import (
    draw_labeled_box,
    load_name_map,
    load_trace,
    resolve_image,
    scale_box,
)


WHITE = (255, 255, 255)
INK = (28, 32, 38)
MUTED = (91, 98, 108)
LINE = (202, 208, 216)
HUMAN = (190, 39, 51)
OBJECT = (42, 94, 184)
CONTEXT = (218, 137, 29)
INFO = (27, 139, 82)
PURPLE = (114, 69, 157)
SOFT_BLUE = (233, 240, 252)
SOFT_RED = (252, 235, 238)
SOFT_GREEN = (232, 247, 239)
SOFT_GOLD = (253, 246, 226)
SOFT_PURPLE = (243, 236, 249)
SOFT_GRAY = (247, 248, 250)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    value: str,
    size: int,
    *,
    bold: bool = False,
    fill: Tuple[int, int, int] = INK,
) -> None:
    draw.text(xy, value, font=font(size, bold=bold), fill=fill)


def fit_box(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    ratio = min(max_width / image.width, max_height / image.height)
    width = max(1, round(image.width * ratio))
    height = max(1, round(image.height * ratio))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def details_from_row(row: Mapping[str, str]) -> List[Dict[str, Any]]:
    return [dict(item) for item in json.loads(row["context_details_json"])]


def name_of(mapping: Mapping[int, str], index: int) -> str:
    return mapping.get(int(index), str(int(index)))


def draw_overlay(
    record: Mapping[str, Any],
    source: Image.Image,
    row: Mapping[str, str],
    details: Sequence[Mapping[str, Any]],
) -> Image.Image:
    image = source.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    boxes = record["nodes"]["boxes_xyxy_norm"].float()
    width, height = image.size
    h_idx = int(row["human_node_idx"])
    o_idx = int(row["object_node_idx"])
    draw_labeled_box(draw, scale_box(boxes[h_idx], width, height), "H", HUMAN, 6)
    draw_labeled_box(draw, scale_box(boxes[o_idx], width, height), "O", OBJECT, 6)
    for item in details:
        node_idx = int(item["node_idx"])
        informative = bool(item["informative"])
        draw_labeled_box(
            draw,
            scale_box(boxes[node_idx], width, height),
            f"C{item['slot']}",
            INFO if informative else CONTEXT,
            5 if informative else 3,
        )
    return image


def prototype_details(
    record: Mapping[str, Any],
    pair_idx: int,
    action_names: Mapping[int, str],
) -> List[Dict[str, Any]]:
    graph = record["graph"]
    mask = graph["selected_hoi_mask"][pair_idx].bool()
    action_ids = graph["selected_hoi_action_ids"][pair_idx].long()
    scores = graph["sync_prototype_scores"][pair_idx].float()
    weights = graph["sync_prototype_weights"][pair_idx].float()
    rows: List[Dict[str, Any]] = []
    for prototype_row in range(int(action_ids.shape[0])):
        if not bool(mask[prototype_row]):
            continue
        action_id = int(action_ids[prototype_row].item())
        rows.append({
            "prototype_row": prototype_row,
            "action_id": action_id,
            "action_name": name_of(action_names, action_id),
            "score": float(scores[prototype_row].item()),
            "weight": float(weights[prototype_row].item()),
        })
    return rows


def evidence_details(
    record: Mapping[str, Any],
    pair_idx: int,
    prototype_row: int,
) -> Tuple[List[str], List[float], List[float]]:
    graph = record["graph"]
    values = graph["sync_evidence_weights"][pair_idx, prototype_row].float()
    mask = graph["candidate_evidence_mask"][pair_idx].bool()
    raw_roles = graph.get("visual_token_roles", ["anchor", "human", "object", "context_1", "context_2", "context_3", "context_4"])
    role_map = {"anchor": "A", "human": "H", "object": "O"}
    roles = [role_map.get(str(role), str(role).replace("context_", "C")) for role in raw_roles]
    total = float(values[mask].sum().item())
    numeric = [float(value) for value in values.tolist()]
    percentages = [100.0 * value / max(total, 1e-12) for value in numeric]
    return roles, numeric, percentages


def rounded(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int] = LINE,
    width: int = 2,
    radius: int = 8,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_pair_summary(
    draw: ImageDraw.ImageDraw,
    row: Mapping[str, str],
    x: int,
    y: int,
    width: int,
) -> int:
    height = 105
    rounded(draw, (x, y, x + width, y + height), SOFT_GREEN, outline=INFO, width=3)
    text(draw, (x + 16, y + 10), "Correct GT human-object pair", 19, bold=True, fill=INFO)
    text(
        draw,
        (x + 16, y + 40),
        f"GT: {row['gt_action_names']} | object: {row['object_name']}",
        16,
        bold=True,
    )
    text(
        draw,
        (x + 16, y + 67),
        (
            f"Prediction: {row['top_action_name']}  p={float(row['top_action_probability']):.3f}  "
            f"pair proposal={float(row['proposal_probability']):.3f}"
        ),
        15,
        fill=INK,
    )
    return height


def draw_context_cards(
    draw: ImageDraw.ImageDraw,
    details: Sequence[Mapping[str, Any]],
    x: int,
    y: int,
    width: int,
) -> int:
    text(draw, (x, y), "Selected context nodes and graph paths", 18, bold=True, fill=CONTEXT)
    y += 29
    card_height = 79
    for item in details:
        informative = bool(item["informative"])
        fill = SOFT_GREEN if informative else SOFT_GOLD
        outline = INFO if informative else CONTEXT
        rounded(draw, (x, y, x + width, y + card_height), fill, outline=outline)
        tag = "informative" if informative else "selected / audit"
        text(
            draw,
            (x + 12, y + 7),
            f"C{item['slot']}  {item['name']}  [{tag}]",
            15,
            bold=True,
            fill=outline,
        )
        text(
            draw,
            (x + 12, y + 31),
            (
                f"det={float(item['detector_score']):.3f}  selection-w={float(item['candidate_selection_weight']):.5f}  "
                f"IoU(H/O max)={float(item['max_endpoint_iou']):.3f}"
            ),
            13,
        )
        text(
            draw,
            (x + 12, y + 53),
            (
                f"H-C={float(item['hk_strength']):.2f}  C-O={float(item['ko_strength']):.2f}  "
                f"{'two-hop' if item['two_hop'] else 'one-hop'}  "
                f"final action evidence={float(item['final_action_weight_percent']):.5f}%"
            ),
            13,
        )
        y += card_height + 7
    return 29 + len(details) * (card_height + 7)


def draw_prototypes(
    draw: ImageDraw.ImageDraw,
    prototypes: Sequence[Mapping[str, Any]],
    predicted_row: int,
    x: int,
    y: int,
    width: int,
) -> int:
    text(draw, (x, y), "Retrieved action prototypes", 18, bold=True, fill=PURPLE)
    y += 30
    gap = 8
    card_width = (width - gap * 4) // 5
    for index in range(5):
        x1 = x + index * (card_width + gap)
        if index >= len(prototypes):
            rounded(draw, (x1, y, x1 + card_width, y + 72), SOFT_GRAY)
            continue
        item = prototypes[index]
        selected = int(item["prototype_row"]) == predicted_row
        fill = SOFT_GREEN if selected else SOFT_PURPLE
        outline = INFO if selected else PURPLE
        rounded(draw, (x1, y, x1 + card_width, y + 72), fill, outline=outline, width=3 if selected else 2)
        action_name = str(item["action_name"])
        if len(action_name) > 15:
            action_name = action_name[:14] + "…"
        text(draw, (x1 + 8, y + 8), action_name, 13, bold=selected, fill=outline)
        text(draw, (x1 + 8, y + 31), f"score={float(item['score']):.3f}", 12)
        text(draw, (x1 + 8, y + 50), f"weight={float(item['weight']):.3f}", 12, fill=MUTED)
    return 102


def draw_evidence_allocation(
    draw: ImageDraw.ImageDraw,
    roles: Sequence[str],
    values: Sequence[float],
    percentages: Sequence[float],
    x: int,
    y: int,
    width: int,
) -> int:
    context_sum = sum(values[3:])
    context_percent = sum(percentages[3:])
    text(draw, (x, y), "Final evidence allocation for predicted prototype", 18, bold=True, fill=OBJECT)
    text(
        draw,
        (x, y + 27),
        f"Context total={context_sum:.8f} ({context_percent:.5f}%) — exact Trace values, not a causal contribution claim",
        13,
        fill=MUTED,
    )
    y += 52
    gap = 7
    cell_width = (width - gap * 6) // 7
    role_colors = [PURPLE, HUMAN, OBJECT, CONTEXT, CONTEXT, CONTEXT, CONTEXT]
    role_fills = [SOFT_PURPLE, SOFT_RED, SOFT_BLUE, SOFT_GOLD, SOFT_GOLD, SOFT_GOLD, SOFT_GOLD]
    for index, role in enumerate(roles[:7]):
        x1 = x + index * (cell_width + gap)
        rounded(draw, (x1, y, x1 + cell_width, y + 78), role_fills[index], outline=role_colors[index])
        text(draw, (x1 + 8, y + 6), role, 15, bold=True, fill=role_colors[index])
        text(draw, (x1 + 8, y + 29), f"{values[index]:.8f}", 11)
        text(draw, (x1 + 8, y + 51), f"{percentages[index]:.5f}%", 11, fill=MUTED)

    bar_y = y + 91
    rounded(draw, (x, bar_y, x + width, bar_y + 22), SOFT_GRAY, outline=LINE)
    cursor = x
    for index, percentage in enumerate(percentages[:7]):
        segment = round(width * max(percentage, 0.0) / 100.0)
        if segment <= 0:
            continue
        draw.rectangle((cursor, bar_y, min(x + width, cursor + segment), bar_y + 22), fill=role_colors[index])
        cursor += segment
    return 122


def build_panel(
    record: Mapping[str, Any],
    source: Image.Image,
    row: Mapping[str, str],
    action_names: Mapping[int, str],
    figure_index: int,
) -> Tuple[Image.Image, Dict[str, Any], List[Dict[str, Any]]]:
    pair_idx = int(row["pair_idx"])
    prototype_row = int(row["top_action_prototype_row"])
    details = details_from_row(row)
    prototypes = prototype_details(record, pair_idx, action_names)
    roles, values, percentages = evidence_details(record, pair_idx, prototype_row)

    canvas_width = 1900
    header_height = 102
    footer_height = 42
    margin = 22
    gap = 24
    left_width = 820
    right_width = canvas_width - margin * 2 - gap - left_width
    content_height = 820
    canvas_height = header_height + content_height + footer_height
    panel = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    draw = ImageDraw.Draw(panel)

    text(draw, (margin, 12), f"EvidenceHOI trace #{figure_index}: {row['file_name']}", 27, bold=True)
    text(
        draw,
        (margin, 51),
        (
            f"Pair H{row['human_node_idx']}–O{row['object_node_idx']} | "
            f"{row['top_action_name']} {row['object_name']} | final top-1 correct"
        ),
        19,
        bold=True,
        fill=INFO,
    )
    text(
        draw,
        (margin, 78),
        "Red=human, blue=object, green=informative context, gold=other selected context",
        14,
        fill=MUTED,
    )

    overlay = draw_overlay(record, source, row, details)
    overlay = fit_box(overlay, left_width, 700)
    image_x = margin + (left_width - overlay.width) // 2
    image_y = header_height + 12
    framed = ImageOps.expand(overlay, border=3, fill=LINE)
    panel.paste(framed, (image_x - 3, image_y - 3))
    summary_y = image_y + overlay.height + 18
    rounded(draw, (margin, summary_y, margin + left_width, summary_y + 83), SOFT_GRAY)
    text(draw, (margin + 14, summary_y + 9), "Trace selection summary", 16, bold=True)
    text(
        draw,
        (margin + 14, summary_y + 34),
        (
            f"visual={float(row['visual_quality']):.3f} | paper score={float(row['paper_score']):.3f} | "
            f"context={float(row['context_weight_percent']):.5f}%"
        ),
        14,
    )
    text(
        draw,
        (margin + 14, summary_y + 57),
        f"informative context={float(row['informative_context_weight_percent']):.5f}%",
        14,
        fill=INFO,
    )

    right_x = margin + left_width + gap
    y = header_height + 4
    y += draw_pair_summary(draw, row, right_x, y, right_width) + 13
    y += draw_context_cards(draw, details, right_x, y, right_width) + 10
    y += draw_prototypes(draw, prototypes, prototype_row, right_x, y, right_width) + 8
    draw_evidence_allocation(draw, roles, values, percentages, right_x, y, right_width)

    text(
        draw,
        (margin, canvas_height - 29),
        "All boxes, paths, prototypes, probabilities, and A/H/O/C values are read directly from the saved Trace; no model rerun was performed.",
        13,
        fill=MUTED,
    )

    evidence_row: Dict[str, Any] = dict(row)
    evidence_row.update({
        "figure_index": figure_index,
        **{f"evidence_{role}_weight": value for role, value in zip(roles, values)},
        **{f"evidence_{role}_percent": value for role, value in zip(roles, percentages)},
        "rendered_context_weight_sum": sum(values[3:]),
        "rendered_context_weight_percent": sum(percentages[3:]),
    })
    context_rows: List[Dict[str, Any]] = []
    for item in details:
        context_rows.append({
            "figure_index": figure_index,
            "file_name": row["file_name"],
            "pair_idx": pair_idx,
            "top_action_name": row["top_action_name"],
            "object_name": row["object_name"],
            **item,
            "exclusion_reasons": ";".join(str(value) for value in item.get("exclusion_reasons", [])),
        })
    return panel, evidence_row, context_rows


def parse_selection(spec: str) -> Tuple[Path, int]:
    path_raw, rank_raw = spec.rsplit("#", 1)
    rank = int(rank_raw)
    if rank <= 0:
        raise ValueError(f"Selection rank must be one-based: {spec}")
    return Path(path_raw), rank


def load_selections(specs: Sequence[str]) -> List[Dict[str, str]]:
    selections: List[Dict[str, str]] = []
    for spec in specs:
        csv_path, rank = parse_selection(spec)
        rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
        if rank > len(rows):
            raise IndexError(f"Selection {spec} exceeds {len(rows)} rows")
        row = dict(rows[rank - 1])
        row["selection_source_csv"] = str(csv_path)
        row["selection_source_rank"] = str(rank)
        selections.append(row)
    return selections


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    if not paths:
        return
    thumb_width = 950
    columns = 2
    cells: List[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        height = max(1, round(image.height * thumb_width / image.width))
        cells.append(image.resize((thumb_width, height), Image.Resampling.LANCZOS))
    row_heights = [
        max(cell.height for cell in cells[start:start + columns])
        for start in range(0, len(cells), columns)
    ]
    sheet = Image.new("RGB", (thumb_width * columns, sum(row_heights)), (230, 232, 235))
    for index, cell in enumerate(cells):
        row, col = divmod(index, columns)
        y = sum(row_heights[:row])
        framed = ImageOps.expand(cell, border=2, fill=(70, 73, 78))
        sheet.paste(framed.resize((thumb_width, row_heights[row])), (col * thumb_width, y))
    sheet.save(output, quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        action="append",
        required=True,
        help="One-based CSV row as PATH#RANK; repeat for each final example",
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-names-json", required=True)
    parser.add_argument("--image-pattern")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_names = load_name_map(args.action_names_json)
    selections = load_selections(args.selection)
    output_paths: List[Path] = []
    evidence_rows: List[Dict[str, Any]] = []
    context_rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for figure_index, row in enumerate(selections, start=1):
        try:
            record = load_trace(Path(row["record_path"]))
            image_path = resolve_image(record, Path(args.image_root), args.image_pattern)
            if image_path is None:
                raise FileNotFoundError(f"image not found for {row['record_path']}")
            source = Image.open(image_path)
            panel, evidence_row, per_context = build_panel(
                record, source, row, action_names, figure_index
            )
            output = output_dir / f"{figure_index:02d}_{Path(row['file_name']).stem}_{row['top_action_name']}.png"
            panel.save(output)
            output_paths.append(output)
            evidence_rows.append(evidence_row)
            context_rows.extend(per_context)
        except (OSError, KeyError, ValueError, IndexError, RuntimeError, json.JSONDecodeError) as exc:
            failures.append(f"{row.get('file_name')}: {exc}")

    write_rows(output_dir / "selected_examples.csv", evidence_rows)
    write_rows(output_dir / "context_node_values.csv", context_rows)
    if failures:
        (output_dir / "render_failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
    make_contact_sheet(output_paths, output_dir / "evidence_chain_contact.jpg")
    manifest = {
        "rendered_count": len(output_paths),
        "failure_count": len(failures),
        "selections": [
            {
                "figure_index": index,
                "file_name": row.get("file_name"),
                "pair_idx": int(row["pair_idx"]),
                "top_action_name": row.get("top_action_name"),
                "object_name": row.get("object_name"),
                "selection_source_csv": row.get("selection_source_csv"),
                "selection_source_rank": int(row["selection_source_rank"]),
            }
            for index, row in enumerate(selections, start=1)
        ],
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
