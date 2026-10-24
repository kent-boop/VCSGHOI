#!/usr/bin/env python3
"""Render ranked EvidenceHOI candidates as compact inspection panels.

Each panel contains an image overlay plus GT/competitor SETM heatmaps. It is a
mining aid, not a publication-ready renderer; after selecting one or two hero
examples, produce the final figure with consistent typography and manual label
placement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


SLOT_LABELS = ["H", "O", "A", "D", "S", "C"]


def load_trace(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_name_map(path: Optional[str]) -> Mapping[int, str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {idx: str(name) for idx, name in enumerate(data)}
    if isinstance(data, dict):
        return {int(key): str(value) for key, value in data.items()}
    raise ValueError(f"Expected a list/dict name map: {path}")


def name_of(mapping: Mapping[int, str], idx: int) -> str:
    return mapping.get(int(idx), str(int(idx)))


def resolve_image(
    record: Mapping[str, Any],
    image_root: Path,
    image_pattern: Optional[str],
) -> Optional[Path]:
    file_name = record.get("image", {}).get("file_name")
    if file_name:
        candidate = Path(str(file_name))
        if candidate.is_file():
            return candidate
        candidate = image_root / candidate
        if candidate.is_file():
            return candidate
    if image_pattern:
        image_id = record.get("image", {}).get("image_id")
        try:
            formatted = image_pattern.format(image_id=int(image_id))
        except (TypeError, ValueError):
            formatted = image_pattern.format(image_id=image_id)
        candidate = image_root / formatted
        if candidate.is_file():
            return candidate
    return None


def scale_box(box: torch.Tensor, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    width: int = 4,
) -> None:
    draw.rectangle(box, outline=color, width=width)
    x1, y1, _, _ = box
    text_box = draw.textbbox((x1, y1), label)
    tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
    y_text = max(0, y1 - th - 4)
    draw.rectangle((x1, y_text, x1 + tw + 6, y_text + th + 4), fill=(255, 255, 255))
    draw.text((x1 + 3, y_text + 2), label, fill=color)


def context_ids(graph: Mapping[str, Any], pair_idx: int) -> List[int]:
    indices = graph.get("selected_context_indices")
    mask = graph.get("selected_context_mask")
    if not isinstance(indices, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return []
    valid = mask[pair_idx].to(dtype=torch.bool)
    return [int(x) for x in indices[pair_idx][valid].tolist()]


def overlay_image(record: Mapping[str, Any], source: Image.Image, row: Mapping[str, str]) -> Image.Image:
    image = source.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    boxes = record["nodes"]["boxes_xyxy_norm"].float()
    labels = record["nodes"]["labels"].long()
    width, height = image.size

    focal = int(row["pair_idx"])
    pair_indices = record["pairs"]["indices"].long()
    h_idx, o_idx = [int(x) for x in pair_indices[focal].tolist()]
    draw_labeled_box(draw, scale_box(boxes[h_idx], width, height), f"H{h_idx}", (190, 30, 45), 5)
    draw_labeled_box(draw, scale_box(boxes[o_idx], width, height), f"O{o_idx}", (30, 90, 190), 5)
    for rank, node_idx in enumerate(context_ids(record.get("graph") or {}, focal), start=1):
        draw_labeled_box(
            draw,
            scale_box(boxes[node_idx], width, height),
            f"C{rank}:{int(labels[node_idx])}",
            (225, 125, 20),
            4,
        )

    partner_raw = row.get("partner_pair_idx")
    if partner_raw not in (None, ""):
        partner = int(partner_raw)
        ph_idx, po_idx = [int(x) for x in pair_indices[partner].tolist()]
        draw_labeled_box(draw, scale_box(boxes[ph_idx], width, height), f"H'{ph_idx}", (115, 35, 150), 3)
        draw_labeled_box(draw, scale_box(boxes[po_idx], width, height), f"O'{po_idx}", (15, 135, 100), 3)
        for rank, node_idx in enumerate(context_ids(record.get("graph") or {}, partner), start=1):
            draw_labeled_box(
                draw,
                scale_box(boxes[node_idx], width, height),
                f"C'{rank}:{int(labels[node_idx])}",
                (20, 145, 95),
                2,
            )
    return image


def heat_color(value: float) -> Tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    return (255, int(255 * (1.0 - value)), int(255 * (1.0 - value)))


def draw_heatmap(
    attn: torch.Tensor,
    evidence_weights: torch.Tensor,
    title: str,
    token_labels: Sequence[str],
    cell: int = 34,
) -> Image.Image:
    attn = attn.float()
    rows, cols = attn.shape
    left = 38
    top = 42
    bottom = 54
    image = Image.new("RGB", (left + cols * cell + 8, top + rows * cell + bottom), "white")
    draw = ImageDraw.Draw(image)
    draw.text((6, 6), title, fill="black")
    for row in range(rows):
        draw.text((8, top + row * cell + 10), SLOT_LABELS[row] if row < len(SLOT_LABELS) else str(row), fill="black")
        for col in range(cols):
            value = float(attn[row, col].item())
            x1 = left + col * cell
            y1 = top + row * cell
            draw.rectangle((x1, y1, x1 + cell - 1, y1 + cell - 1), fill=heat_color(value), outline=(210, 210, 210))
            draw.text((x1 + 4, y1 + 10), f"{value:.2f}", fill="black")
    for col, label in enumerate(token_labels[:cols]):
        draw.text((left + col * cell + 4, top + rows * cell + 4), label, fill="black")
    y_bar = top + rows * cell + 28
    for col in range(cols):
        value = float(evidence_weights[col].item())
        x1 = left + col * cell
        draw.rectangle((x1, y_bar, x1 + cell - 1, y_bar + 14), fill=heat_color(value), outline=(180, 180, 180))
    draw.text((4, y_bar), "w", fill="black")
    return image


def fit_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    scale = height / image.height
    return image.resize((max(1, int(round(image.width * scale))), height), Image.Resampling.LANCZOS)


def build_panel(
    record: Mapping[str, Any],
    source: Image.Image,
    row: Mapping[str, str],
    action_names: Mapping[int, str],
) -> Image.Image:
    overlay = overlay_image(record, source, row)
    graph = record.get("graph") or {}
    pair_idx = int(row["pair_idx"])
    gt_row = int(row["gt_prototype_row"])
    comp_row = int(row["competitor_prototype_row"])
    attn = graph["sync_text_to_visual_attn"].float()
    weights = graph["sync_evidence_weights"].float()
    token_roles = graph.get("visual_token_roles", ["A", "H", "O", "C1", "C2", "C3", "C4"])
    compact_roles = [
        {"anchor": "A", "human": "H", "object": "O"}.get(str(role), str(role).replace("context_", "C"))
        for role in token_roles
    ]
    gt_action = int(row["gt_action_id"])
    comp_action = int(row["competitor_action_id"])
    gt_map = draw_heatmap(
        attn[pair_idx, gt_row],
        weights[pair_idx, gt_row],
        f"GT: {name_of(action_names, gt_action)}",
        compact_roles,
    )
    comp_map = draw_heatmap(
        attn[pair_idx, comp_row],
        weights[pair_idx, comp_row],
        f"Comp: {name_of(action_names, comp_action)}",
        compact_roles,
    )
    right_width = max(gt_map.width, comp_map.width)
    right = Image.new("RGB", (right_width, gt_map.height + comp_map.height + 8), "white")
    right.paste(gt_map, (0, 0))
    right.paste(comp_map, (0, gt_map.height + 8))
    overlay = fit_height(overlay, right.height)

    title_height = 44
    panel = Image.new("RGB", (overlay.width + right.width + 12, title_height + right.height), "white")
    title = (
        f"image={row.get('image_id')}  pair={pair_idx}  "
        f"score={float(row.get('combined_score', row.get('hypothesis_score', 0))):.3f}  "
        f"pair-control={row.get('pair_control_type', '-') }"
    )
    ImageDraw.Draw(panel).text((8, 10), title, fill="black")
    panel.paste(overlay, (0, title_height))
    panel.paste(right, (overlay.width + 12, title_height))
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--image-pattern", help="e.g. HICO_test2015_{image_id:08d}.jpg")
    parser.add_argument("--action-names-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_names = load_name_map(args.action_names_json)
    rows = list(csv.DictReader(Path(args.ranking_csv).open("r", encoding="utf-8")))
    failures: List[str] = []
    for rank, row in enumerate(rows[: args.limit], start=1):
        trace_path = Path(row["record_path"])
        record = load_trace(trace_path)
        image_path = resolve_image(record, Path(args.image_root), args.image_pattern)
        if image_path is None:
            failures.append(f"{rank}: image not found for {trace_path}")
            continue
        source = Image.open(image_path)
        panel = build_panel(record, source, row, action_names)
        panel.save(output_dir / f"{rank:03d}_{record['image']['image_id']}_pair{row['pair_idx']}.png")
    if failures:
        (output_dir / "missing_images.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
    print(f"Rendered {min(len(rows), args.limit) - len(failures)} panels; missing={len(failures)}")


if __name__ == "__main__":
    main()
