#!/usr/bin/env python3
"""Render selected EvidenceHOI pair-specific examples as evidence-chain panels."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from render_evidence_shortlist import load_name_map, load_trace, resolve_image
from render_pair_specific_shortlist import (
    PAIR1_COLORS,
    PAIR2_COLORS,
    add_view_title,
    draw_pair_view,
)


WHITE = (255, 255, 255)
INK = (28, 32, 38)
MUTED = (92, 99, 108)
LINE = (205, 210, 216)
GREEN = (31, 132, 80)
RED = (183, 47, 58)
GOLD = (218, 143, 31)
BLUE = (36, 100, 190)
PURPLE = (118, 57, 156)
SOFT_GREEN = (231, 246, 237)
SOFT_RED = (252, 235, 237)
SOFT_GOLD = (253, 246, 226)
SOFT_BLUE = (232, 240, 252)
SOFT_PURPLE = (242, 234, 249)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def name_of(mapping: Mapping[int, str], index: int) -> str:
    return mapping.get(int(index), str(int(index)))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))



def format_weight(value: float) -> str:
    if abs(value) < 0.001:
        return f"{value:.1e}"
    return f"{value:.2f}"
def fit_width(image: Image.Image, width: int) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int] = LINE,
    radius: int = 6,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], value: str, size: int, *, bold: bool = False, fill: Tuple[int, int, int] = INK) -> None:
    draw.text(xy, value, font=font(size, bold=bold), fill=fill)


def context_details(
    record: Mapping[str, Any],
    pair_idx: int,
    node_names: Mapping[int, str],
) -> List[Dict[str, Any]]:
    graph = record["graph"]
    nodes = record["nodes"]
    mask = graph["selected_context_mask"][pair_idx].bool()
    indices = graph["selected_context_indices"][pair_idx][mask].tolist()
    candidate_weights = graph["candidate_evidence_weights"][pair_idx, 3:7][mask].float().tolist()
    two_hop = graph["selected_context_two_hop_mask"][pair_idx][mask].tolist()
    hk_scores = graph["selected_context_edge_hk_score"][pair_idx][mask].float().tolist()
    ko_scores = graph["selected_context_edge_ko_score"][pair_idx][mask].float().tolist()
    details: List[Dict[str, Any]] = []
    for rank, (node_idx, weight, is_two_hop, hk, ko) in enumerate(
        zip(indices, candidate_weights, two_hop, hk_scores, ko_scores),
        start=1,
    ):
        label_id = int(nodes["labels"][node_idx].item())
        details.append({
            "rank": rank,
            "node_idx": int(node_idx),
            "name": name_of(node_names, label_id),
            "score": float(nodes["scores"][node_idx].item()),
            "weight": float(weight),
            "two_hop": bool(is_two_hop),
            "hk": sigmoid(float(hk)),
            "ko": sigmoid(float(ko)),
        })
    return details


def prototype_details(
    record: Mapping[str, Any],
    pair_idx: int,
    action_names: Mapping[int, str],
) -> List[Dict[str, Any]]:
    graph = record["graph"]
    mask = graph["selected_hoi_mask"][pair_idx].bool()
    actions = graph["selected_hoi_action_ids"][pair_idx][mask].tolist()
    scores = graph["sync_prototype_scores"][pair_idx][mask].float().tolist()
    weights = graph["sync_prototype_weights"][pair_idx][mask].float().tolist()
    rows = [
        {
            "row": row,
            "action_id": int(action_id),
            "name": name_of(action_names, int(action_id)),
            "score": float(score),
            "weight": float(weight),
        }
        for row, (action_id, score, weight) in enumerate(zip(actions, scores, weights))
    ]
    return sorted(rows, key=lambda item: item["weight"], reverse=True)


def action_evidence(
    record: Mapping[str, Any],
    pair_idx: int,
    action_id: int,
) -> Tuple[List[float], str]:
    graph = record["graph"]
    mask = graph["selected_hoi_mask"][pair_idx].bool()
    selected_actions = graph["selected_hoi_action_ids"][pair_idx]
    valid_rows = torch.nonzero(mask & (selected_actions == int(action_id)), as_tuple=False).flatten()
    if valid_rows.numel() > 0:
        row = int(valid_rows[0].item())
        values = graph["sync_evidence_weights"][pair_idx, row].float().tolist()
        return [float(value) for value in values], "predicted prototype"
    values = graph["candidate_evidence_weights"][pair_idx].float().tolist()
    return [float(value) for value in values], "candidate evidence"


def endpoint_names(
    record: Mapping[str, Any],
    pair_idx: int,
    node_names: Mapping[int, str],
) -> Tuple[str, str]:
    pairs = record["pairs"]["indices"].long()
    nodes = record["nodes"]
    human_idx, object_idx = [int(value) for value in pairs[pair_idx].tolist()]
    human_label = int(nodes["labels"][human_idx].item())
    object_label = int(nodes["labels"][object_idx].item())
    return (
        f"H{human_idx}: {name_of(node_names, human_label)}",
        f"O{object_idx}: {name_of(node_names, object_label)}",
    )


def draw_chain_card(
    record: Mapping[str, Any],
    row: Mapping[str, str],
    prefix: str,
    pair_idx: int,
    action_names: Mapping[int, str],
    node_names: Mapping[int, str],
    accent: Tuple[int, int, int],
    soft_accent: Tuple[int, int, int],
    width: int = 720,
) -> Image.Image:
    height = 500
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    gt_actions = row.get(f"{prefix}_gt_action_names", "") or "none"
    predicted = row.get(f"{prefix}_top_action_name", "unknown")
    probability = float(row.get(f"{prefix}_top_action_probability", 0.0))
    correct = bool_value(row.get(f"{prefix}_final_correct", False))
    top_action_id = int(row.get(f"{prefix}_top_action_id", -1))
    status_color = GREEN if correct else RED
    status_fill = SOFT_GREEN if correct else SOFT_RED
    status_text = "CORRECT" if correct else "DIAGNOSTIC / WRONG"

    rounded_box(draw, (0, 0, width - 1, height - 1), WHITE, outline=accent, width=3)
    draw.rectangle((2, 2, width - 3, 60), fill=soft_accent)
    text(draw, (16, 10), f"Pair {pair_idx} evidence chain", 21, bold=True)
    rounded_box(draw, (width - 184, 12, width - 16, 48), status_fill, outline=status_color)
    text(draw, (width - 168, 20), status_text, 14, bold=True, fill=status_color)

    human_name, object_name = endpoint_names(record, pair_idx, node_names)
    rounded_box(draw, (18, 78, 205, 124), SOFT_BLUE, outline=BLUE)
    text(draw, (30, 91), human_name, 16, bold=True, fill=BLUE)
    text(draw, (215, 91), "+", 22, bold=True, fill=MUTED)
    rounded_box(draw, (245, 78, 432, 124), SOFT_PURPLE, outline=PURPLE)
    text(draw, (257, 91), object_name, 16, bold=True, fill=PURPLE)
    text(draw, (445, 91), "->", 22, bold=True, fill=MUTED)
    rounded_box(draw, (486, 70, width - 18, 132), status_fill, outline=status_color)
    text(draw, (500, 78), f"GT: {gt_actions}", 14, bold=True)
    text(draw, (500, 103), f"Pred: {predicted}  p={probability:.3f}", 14, bold=True, fill=status_color)

    contexts = context_details(record, pair_idx, node_names)
    text(draw, (18, 144), "Selected context nodes: H -> C -> O", 16, bold=True, fill=accent)
    gap = 10
    card_width = (width - 36 - gap * 3) // 4
    for index, context in enumerate(contexts[:4]):
        x1 = 18 + index * (card_width + gap)
        x2 = x1 + card_width
        rounded_box(draw, (x1, 172, x2, 280), SOFT_GOLD, outline=GOLD)
        text(draw, (x1 + 10, 181), f"C{context['rank']}  {context['name']}", 15, bold=True, fill=GOLD)
        text(draw, (x1 + 10, 205), f"node #{context['node_idx']}", 13, fill=MUTED)
        text(draw, (x1 + 10, 225), f"det={context['score']:.2f}  sel-w={context['weight']:.2f}", 12)
        text(draw, (x1 + 10, 247), f"H-C={context['hk']:.2f}  C-O={context['ko']:.2f}", 12)
        text(draw, (x1 + 10, 265), "two-hop" if context["two_hop"] else "one-hop", 12, bold=True, fill=GREEN if context["two_hop"] else RED)

    prototypes = prototype_details(record, pair_idx, action_names)
    text(draw, (18, 298), "Retrieved action prototypes", 16, bold=True, fill=accent)
    proto_gap = 8
    proto_width = (width - 36 - proto_gap * 4) // 5
    for index, prototype in enumerate(prototypes[:5]):
        x1 = 18 + index * (proto_width + proto_gap)
        x2 = x1 + proto_width
        highlight = prototype["action_id"] == top_action_id
        fill = status_fill if highlight else (245, 246, 248)
        outline = status_color if highlight else LINE
        rounded_box(draw, (x1, 325, x2, 377), fill, outline=outline)
        text(draw, (x1 + 8, 334), prototype["name"][:16], 13, bold=highlight, fill=status_color if highlight else INK)
        text(draw, (x1 + 8, 355), f"w={prototype['weight']:.2f}", 12, fill=MUTED)

    evidence, evidence_source = action_evidence(record, pair_idx, top_action_id)
    roles = ["A", "H", "O", "C1", "C2", "C3", "C4"]
    context_sum = sum(evidence[3:]) if len(evidence) >= 7 else 0.0
    text(
        draw,
        (18, 395),
        f"Evidence allocation for {predicted} ({evidence_source}) | context sum={context_sum:.2e} ({100.0 * context_sum:.3f}%)",
        14,
        bold=True,
        fill=accent,
    )
    bar_left = 18
    bar_top = 423
    bar_width = (width - 36 - 6 * 8) // 7
    maximum = max(evidence) if evidence else 1.0
    for index, role in enumerate(roles):
        x1 = bar_left + index * (bar_width + 8)
        value = evidence[index] if index < len(evidence) else 0.0
        rounded_box(draw, (x1, bar_top, x1 + bar_width, bar_top + 50), (247, 248, 250), outline=LINE)
        text(draw, (x1 + 7, bar_top + 5), role, 12, bold=True)
        fill_height = max(1 if value > 0 else 0, round(20 * value / max(maximum, 1e-8)))
        draw.rectangle((x1 + 30, bar_top + 25 - fill_height, x1 + bar_width - 8, bar_top + 25), fill=accent)
        text(draw, (x1 + 7, bar_top + 31), format_weight(value), 10, fill=MUTED)
    return image


def build_panel(
    record: Mapping[str, Any],
    source: Image.Image,
    row: Mapping[str, str],
    action_names: Mapping[int, str],
    node_names: Mapping[int, str],
) -> Image.Image:
    first_idx = int(row["pair_idx_1"])
    second_idx = int(row["pair_idx_2"])
    first_view, _ = draw_pair_view(record, source, first_idx, "P1", PAIR1_COLORS)
    second_view, _ = draw_pair_view(record, source, second_idx, "P2", PAIR2_COLORS)
    first_view = fit_width(first_view, 720)
    second_view = fit_width(second_view, 720)
    first_view = add_view_title(
        first_view,
        f"P1 | GT {row.get('pair_1_gt_action_names') or 'none'} | Pred {row.get('pair_1_top_action_name')}",
    )
    second_view = add_view_title(
        second_view,
        f"P2 | GT {row.get('pair_2_gt_action_names') or 'none'} | Pred {row.get('pair_2_top_action_name')}",
    )
    view_height = max(first_view.height, second_view.height)
    if first_view.height != view_height:
        first_view = first_view.resize((first_view.width, view_height), Image.Resampling.LANCZOS)
    if second_view.height != view_height:
        second_view = second_view.resize((second_view.width, view_height), Image.Resampling.LANCZOS)

    first_chain = draw_chain_card(
        record, row, "pair_1", first_idx, action_names, node_names,
        PAIR1_COLORS["human"], SOFT_RED,
    )
    second_chain = draw_chain_card(
        record, row, "pair_2", second_idx, action_names, node_names,
        PAIR2_COLORS["human"], SOFT_PURPLE,
    )
    header_height = 92
    footer_height = 42
    gap = 8
    width = first_view.width + second_view.width + gap
    height = header_height + view_height + gap + first_chain.height + footer_height
    panel = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(panel)
    text(draw, (18, 12), str(row.get("file_name")), 24, bold=True)
    text(
        draw,
        (18, 46),
        f"control={row.get('control_type')}  pair-specific score={float(row.get('pair_specific_score', 0.0)):.3f}  context-set distance={float(row.get('context_set_distance', 0.0)):.3f}",
        16,
        fill=MUTED,
    )
    text(draw, (18, 70), str(row.get("selection_note", "")), 14, fill=MUTED)
    panel.paste(first_view, (0, header_height))
    panel.paste(second_view, (first_view.width + gap, header_height))
    chain_y = header_height + view_height + gap
    panel.paste(first_chain, (0, chain_y))
    panel.paste(second_chain, (first_chain.width + gap, chain_y))
    text(
        draw,
        (18, height - 30),
        "Context cards report detector confidence, selection weight, sigmoid edge strengths, and two-hop status.",
        13,
        fill=MUTED,
    )
    return panel


def make_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    if not paths:
        return
    thumb_width = 720
    columns = 2
    cells: List[Image.Image] = []
    for path in paths:
        cells.append(fit_width(Image.open(path).convert("RGB"), thumb_width))
    row_heights = [
        max(cell.height for cell in cells[start:start + columns])
        for start in range(0, len(cells), columns)
    ]
    sheet = Image.new("RGB", (thumb_width * columns, sum(row_heights)), (235, 235, 235))
    for index, cell in enumerate(cells):
        row, col = divmod(index, columns)
        y = sum(row_heights[:row])
        framed = ImageOps.expand(cell, border=2, fill=(75, 75, 75))
        sheet.paste(framed.resize((thumb_width, row_heights[row])), (col * thumb_width, y))
    sheet.save(output, quality=90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-names-json", required=True)
    parser.add_argument("--node-names-json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.selection_csv).open("r", encoding="utf-8")))
    if args.limit > 0:
        rows = rows[: args.limit]
    action_names = load_name_map(args.action_names_json)
    node_names = load_name_map(args.node_names_json)
    outputs: List[Path] = []
    rendered: List[Dict[str, str]] = []
    failures: List[str] = []
    for rank, row in enumerate(rows, start=1):
        try:
            trace_path = Path(row["record_path"])
            record = load_trace(trace_path)
            image_path = resolve_image(record, Path(args.image_root), None)
            if image_path is None:
                raise FileNotFoundError(f"image not found for {trace_path}")
            source = Image.open(image_path)
            panel = build_panel(record, source, row, action_names, node_names)
            output = output_dir / f"{rank:02d}_{Path(row['file_name']).stem}_evidence_chain.png"
            panel.save(output)
            outputs.append(output)
            rendered.append(dict(row))
        except (OSError, KeyError, ValueError, IndexError, RuntimeError) as exc:
            failures.append(f"{row.get('file_name')}: {exc}")
    if rendered:
        fieldnames = list(rendered[0].keys())
        with (output_dir / "rendered_selection.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rendered)
    if failures:
        (output_dir / "render_failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
    make_contact_sheet(outputs, output_dir / "evidence_chain_contact.jpg")
    print(f"Rendered {len(outputs)} evidence-chain panels; failures={len(failures)}")


if __name__ == "__main__":
    main()
