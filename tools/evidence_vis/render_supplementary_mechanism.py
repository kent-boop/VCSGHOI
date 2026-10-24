#!/usr/bin/env python3
"""Render publication-oriented EvidenceHOI mechanism figures from real Trace.

Each panel shows the target pair, selected visual evidence chain, retrieved
prototypes, learned (M+3) by 6 soft correspondence, alpha evidence weights,
pi prototype weights, six-role coverage, and coarse/final prediction scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from rank_supplementary_examples import (
    FINAL_ROLE_LABELS,
    profile_compatible,
    trace_profile,
)
from render_evidence_shortlist import (
    draw_labeled_box,
    load_name_map,
    load_trace,
    resolve_image,
    scale_box,
)


WHITE = (255, 255, 255)
INK = (29, 35, 43)
MUTED = (88, 98, 111)
LINE = (199, 207, 217)
HUMAN = (196, 47, 58)
OBJECT = (42, 91, 180)
CONTEXT = (222, 139, 36)
GREEN = (26, 139, 84)
PURPLE = (118, 76, 166)
BLUE = (48, 112, 188)
SOFT_GRAY = (247, 249, 251)
SOFT_GREEN = (233, 247, 239)
SOFT_PURPLE = (245, 239, 250)
SOFT_GOLD = (254, 247, 228)
SOFT_BLUE = (235, 242, 252)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def put_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    value: str,
    size: int,
    *,
    bold: bool = False,
    fill: Tuple[int, int, int] = INK,
) -> None:
    draw.text(xy, value, font=font(size, bold=bold), fill=fill)


def put_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    value: str,
    size: int,
    width_chars: int,
    *,
    fill: Tuple[int, int, int] = INK,
    bold: bool = False,
    spacing: int = 4,
) -> int:
    lines = textwrap.wrap(value, width=max(width_chars, 10)) or [""]
    draw.multiline_text(
        xy,
        "\n".join(lines),
        font=font(size, bold=bold),
        fill=fill,
        spacing=spacing,
    )
    return len(lines) * (size + spacing)


def rounded(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int],
    *,
    outline: Tuple[int, int, int] = LINE,
    width: int = 2,
    radius: int = 10,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_box(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    ratio = min(max_width / image.width, max_height / image.height)
    size = (
        max(1, int(round(image.width * ratio))),
        max(1, int(round(image.height * ratio))),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def center(box: torch.Tensor, width: int, height: int) -> Tuple[int, int]:
    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    return (
        int(round(0.5 * (x1 + x2) * width)),
        int(round(0.5 * (y1 + y2) * height)),
    )


def context_details(row: Mapping[str, str]) -> List[Dict[str, Any]]:
    return [dict(item) for item in json.loads(row["context_details_json"])]


def draw_overlay(
    record: Mapping[str, Any],
    source: Image.Image,
    row: Mapping[str, str],
) -> Image.Image:
    image = source.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    boxes = record["nodes"]["boxes_xyxy_norm"].float()
    width, height = image.size
    h_idx = int(row["human_node_idx"])
    o_idx = int(row["object_node_idx"])
    h_center = center(boxes[h_idx], width, height)
    o_center = center(boxes[o_idx], width, height)
    details = context_details(row)

    for item in details:
        c_idx = int(item["node_idx"])
        c_center = center(boxes[c_idx], width, height)
        if bool(item.get("hk_valid", False)):
            draw.line(
                (h_center, c_center),
                fill=GREEN if bool(item.get("informative", False)) else CONTEXT,
                width=max(2, int(round(2 + 4 * float(item.get("hk_strength", 0.0))))),
            )
        if bool(item.get("ko_valid", False)):
            draw.line(
                (c_center, o_center),
                fill=GREEN if bool(item.get("informative", False)) else CONTEXT,
                width=max(2, int(round(2 + 4 * float(item.get("ko_strength", 0.0))))),
            )

    draw_labeled_box(draw, scale_box(boxes[h_idx], width, height), "H", HUMAN, 7)
    draw_labeled_box(draw, scale_box(boxes[o_idx], width, height), "O", OBJECT, 7)
    for item in details:
        c_idx = int(item["node_idx"])
        color = GREEN if bool(item.get("informative", False)) else CONTEXT
        draw_labeled_box(
            draw,
            scale_box(boxes[c_idx], width, height),
            f"C{int(item['slot'])}",
            color,
            5 if bool(item.get("informative", False)) else 3,
        )
    return image


def visual_labels(row: Mapping[str, str], graph: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    raw_roles = graph.get(
        "visual_token_roles",
        ["anchor", "human", "object", "context_1", "context_2", "context_3", "context_4"],
    )
    short = []
    names = []
    detail_by_slot = {int(item["slot"]): item for item in context_details(row)}
    for index, role in enumerate(raw_roles):
        role = str(role)
        if role == "anchor":
            short.append("A")
            names.append(f"target HO: {row['top_action_name']} {row['object_name']}")
        elif role == "human":
            short.append("H")
            names.append("target human")
        elif role == "object":
            short.append("O")
            names.append(f"target object: {row['object_name']}")
        else:
            slot = index - 2
            short.append(f"C{slot}")
            item = detail_by_slot.get(slot, {})
            names.append(str(item.get("name", f"context {slot}")))
    return short, names


def heat_color(value: float) -> Tuple[int, int, int]:
    value = max(0.0, min(1.0, float(value)))
    return (
        int(round(246 - 178 * value)),
        int(round(249 - 137 * value)),
        int(round(252 - 58 * value)),
    )


def blend_color(
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    amount: float,
) -> Tuple[int, int, int]:
    amount = max(0.0, min(1.0, float(amount)))
    return tuple(
        int(round(start[index] + amount * (end[index] - start[index])))
        for index in range(3)
    )


def deviation_color(value: float, maximum: float) -> Tuple[int, int, int]:
    """Diverging color for a signed probability deviation from uniform."""
    neutral = (244, 246, 249)
    if maximum <= 0:
        return neutral
    strength = min(abs(float(value)) / float(maximum), 1.0)
    endpoint = CONTEXT if value >= 0 else BLUE
    return blend_color(neutral, endpoint, 0.18 + 0.72 * strength)


def draw_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    color: Tuple[int, int, int],
    *,
    maximum: float = 1.0,
) -> None:
    draw.rectangle((x, y, x + width, y + height), fill=SOFT_GRAY, outline=LINE, width=1)
    ratio = max(0.0, min(1.0, float(value) / max(maximum, 1e-12)))
    fill_width = int(round(width * ratio))
    if fill_width > 0:
        draw.rectangle(
            (x, y, x + fill_width, y + height),
            fill=color,
        )


def draw_chain(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    labels: Sequence[str],
    names: Sequence[str],
    alpha: torch.Tensor,
    mask: torch.Tensor,
    details: Sequence[Mapping[str, Any]],
) -> int:
    put_text(draw, (x, y), "Pair-specific visual evidence chain", 23, bold=True, fill=CONTEXT)
    put_text(draw, (x, y + 31), "Alpha values are learned soft aggregation weights.", 15, fill=MUTED)
    y += 59
    detail_by_slot = {int(item["slot"]): item for item in details}
    for index, (label, name) in enumerate(zip(labels, names)):
        valid = index < mask.numel() and bool(mask[index])
        value = float(alpha[index].item()) if index < alpha.numel() else 0.0
        fill = SOFT_GOLD if index >= 3 else (SOFT_PURPLE if index == 0 else SOFT_BLUE)
        outline = CONTEXT if index >= 3 else (PURPLE if index == 0 else BLUE)
        rounded(draw, (x, y, x + width, y + 58), fill, outline=outline if valid else LINE)
        put_text(draw, (x + 12, y + 8), label, 17, bold=True, fill=outline)
        put_text(draw, (x + 58, y + 8), str(name)[:35], 16, bold=True)
        put_text(draw, (x + width - 154, y + 8), f"alpha={value:.6f}", 14, fill=MUTED)
        if index >= 3:
            item = detail_by_slot.get(index - 2, {})
            path = (
                "H-C-O" if bool(item.get("two_hop", False))
                else ("H-C" if bool(item.get("hk_valid", False)) else "C-O")
            )
            put_text(
                draw,
                (x + 58, y + 33),
                f"{path}; detector={float(item.get('detector_score', 0.0)):.2f}",
                13,
                fill=MUTED,
            )
        else:
            put_text(draw, (x + 58, y + 33), "valid" if valid else "masked", 13, fill=MUTED)
        y += 64
    return y


def prototype_rows(
    record: Mapping[str, Any],
    pair_idx: int,
    action_names: Mapping[int, str],
) -> List[Dict[str, Any]]:
    graph = record["graph"]
    action_ids = graph["selected_hoi_action_ids"][pair_idx].long()
    selected_mask = graph["selected_hoi_mask"][pair_idx].bool()
    retrieval_scores = graph["selected_hoi_scores"][pair_idx].float()
    pi = graph["sync_prototype_weights"][pair_idx].float()
    coverage = graph["sync_prototype_coverage"][pair_idx].float()
    slot_mask = graph["prototype_slot_mask"][pair_idx].bool()
    rows: List[Dict[str, Any]] = []
    for index in range(int(action_ids.numel())):
        if not bool(selected_mask[index]):
            continue
        action_id = int(action_ids[index].item())
        valid_coverage = coverage[index][slot_mask[index]]
        rows.append({
            "row": index,
            "action_id": action_id,
            "action_name": action_names.get(action_id, str(action_id)),
            "retrieval_score": float(retrieval_scores[index].item()),
            "pi": float(pi[index].item()),
            "coverage_mean": (
                float(valid_coverage.mean().item()) if valid_coverage.numel() else 0.0
            ),
        })
    return rows


def draw_prototypes(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    prototypes: Sequence[Mapping[str, Any]],
    selected_row: int,
) -> int:
    put_text(draw, (x, y), "Retrieved Top-K knowledge prototypes", 23, bold=True, fill=PURPLE)
    put_text(draw, (x, y + 31), "Pi is the learned prototype aggregation weight.", 15, fill=MUTED)
    y += 59
    for item in prototypes:
        selected = int(item["row"]) == selected_row
        rounded(
            draw,
            (x, y, x + width, y + 68),
            SOFT_GREEN if selected else SOFT_PURPLE,
            outline=GREEN if selected else PURPLE,
            width=3 if selected else 2,
        )
        prefix = "selected" if selected else f"prototype {int(item['row']) + 1}"
        put_text(draw, (x + 12, y + 8), prefix, 14, bold=selected, fill=GREEN if selected else PURPLE)
        put_text(draw, (x + 116, y + 8), str(item["action_name"])[:28], 17, bold=True)
        put_text(
            draw,
            (x + 12, y + 38),
            (
                f"retrieval={float(item['retrieval_score']):.3f}    "
                f"pi={float(item['pi']):.3f}    coverage(mean)={float(item['coverage_mean']):.3f}"
            ),
            14,
            fill=MUTED,
        )
        y += 76
    return y


def draw_prediction_scores(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    row: Mapping[str, str],
) -> int:
    put_text(draw, (x, y), "Prediction refinement", 23, bold=True, fill=GREEN)
    y += 36
    items = [
        (
            f"GT: {row['top_action_name']}",
            float(row["best_gt_coarse_probability"]),
            float(row["best_gt_final_probability"]),
            GREEN,
        ),
        (
            f"Competing: {row['competitor_action_name']}",
            float(row["competitor_coarse_probability"]),
            float(row["competitor_final_probability"]),
            HUMAN,
        ),
    ]
    for label, coarse, final, color in items:
        put_text(draw, (x, y), label[:45], 15, bold=True)
        put_text(draw, (x + width - 168, y), f"{coarse:.3f} -> {final:.3f}", 14, fill=MUTED)
        y += 25
        draw_bar(draw, x, y, width, 14, coarse, (166, 174, 185))
        y += 20
        draw_bar(draw, x, y, width, 14, final, color)
        y += 28
    return y


def draw_correspondence(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    matrix: torch.Tensor,
    alpha: torch.Tensor,
    visual_mask: torch.Tensor,
    role_mask: torch.Tensor,
    visual_short: Sequence[str],
    coverage: torch.Tensor,
    correspondence_view: str,
) -> int:
    valid_role_count = int(role_mask.sum().item())
    uniform_baseline = 1.0 / max(valid_role_count, 1)
    valid_values = matrix[visual_mask][:, role_mask]
    max_abs_deviation = (
        float((valid_values - uniform_baseline).abs().max().item())
        if valid_values.numel() else 0.0
    )
    use_deviation = correspondence_view == "deviation"
    title = (
        "Path-role correspondence: deviation diagnostic"
        if use_deviation else "Learned path-role soft correspondence"
    )
    put_text(draw, (x, y), title, 25, bold=True, fill=BLUE)
    if use_deviation:
        subtitle = (
            f"Cells show [P(role | visual unit) - {uniform_baseline:.3f}] x 10^-3; "
            f"max |delta|={max_abs_deviation * 1e3:.2f} x 10^-3."
        )
    else:
        subtitle = (
            "Rows: visual units; columns: final KPT roles H/O/A/K/S/C. "
            "Values are P(role | visual unit)."
        )
    put_text(
        draw,
        (x, y + 34),
        subtitle,
        15,
        fill=MUTED,
    )
    y += 70
    left = 94
    alpha_width = 124
    gap = 8
    cell_width = (width - left - alpha_width - gap) // 6
    cell_height = 66
    put_text(draw, (x + 6, y + 8), "visual", 14, bold=True, fill=MUTED)
    for col, label in enumerate(FINAL_ROLE_LABELS):
        put_text(
            draw,
            (x + left + col * cell_width + cell_width // 2 - 9, y + 7),
            label,
            17,
            bold=True,
            fill=BLUE,
        )
    put_text(draw, (x + left + 6 * cell_width + gap + 16, y + 8), "alpha", 15, bold=True, fill=CONTEXT)
    y += 38

    for row_idx in range(int(matrix.shape[0])):
        valid_visual = row_idx < visual_mask.numel() and bool(visual_mask[row_idx])
        label = visual_short[row_idx] if row_idx < len(visual_short) else str(row_idx)
        put_text(
            draw,
            (x + 22, y + row_idx * cell_height + 20),
            label,
            17,
            bold=True,
            fill=INK if valid_visual else MUTED,
        )
        for col in range(6):
            value = float(matrix[row_idx, col].item()) if col < matrix.shape[1] else 0.0
            valid = valid_visual and col < role_mask.numel() and bool(role_mask[col])
            x1 = x + left + col * cell_width
            y1 = y + row_idx * cell_height
            deviation = value - uniform_baseline
            color = (
                deviation_color(deviation, max_abs_deviation)
                if valid and use_deviation
                else heat_color(value) if valid
                else (239, 241, 244)
            )
            draw.rectangle(
                (x1, y1, x1 + cell_width - 3, y1 + cell_height - 4),
                fill=color,
                outline=WHITE,
                width=2,
            )
            if valid and use_deviation:
                cell_text = f"{deviation * 1e3:+.2f}"
            elif valid:
                cell_text = f"{value:.4f}"
            else:
                cell_text = "--"
            put_text(
                draw,
                (x1 + 16, y1 + 20),
                cell_text,
                14,
                bold=valid and use_deviation and abs(deviation) >= 0.8 * max_abs_deviation,
                fill=INK,
            )
        alpha_value = float(alpha[row_idx].item()) if row_idx < alpha.numel() else 0.0
        ax = x + left + 6 * cell_width + gap
        ay = y + row_idx * cell_height + 18
        draw_bar(draw, ax, ay, alpha_width - 4, 13, alpha_value, CONTEXT)
        put_text(draw, (ax + 4, ay + 17), f"{alpha_value:.6f}", 12, fill=MUTED)

    y += int(matrix.shape[0]) * cell_height + 24
    put_text(draw, (x, y), "Six-role visual coverage", 19, bold=True, fill=PURPLE)
    y += 31
    cov_cell = width // 6
    for col, label in enumerate(FINAL_ROLE_LABELS):
        value = float(coverage[col].item()) if col < coverage.numel() else 0.0
        valid = col < role_mask.numel() and bool(role_mask[col])
        rounded(
            draw,
            (x + col * cov_cell, y, x + (col + 1) * cov_cell - 8, y + 61),
            SOFT_PURPLE if valid else SOFT_GRAY,
            outline=PURPLE if valid else LINE,
        )
        put_text(draw, (x + col * cov_cell + 12, y + 8), label, 16, bold=True, fill=PURPLE)
        put_text(draw, (x + col * cov_cell + 12, y + 33), f"{value:.3f}", 14, fill=INK)
    return y + 75


def build_figure(
    record: Mapping[str, Any],
    source: Image.Image,
    row: Mapping[str, str],
    action_names: Mapping[int, str],
    figure_index: int,
    *,
    expected_m: int,
    expected_k: int,
    expected_roles: int,
    expected_layers: int,
    strict_config: bool,
    correspondence_view: str = "probability",
) -> Tuple[Image.Image, Dict[str, Any], str]:
    graph = record["graph"]
    pair_idx = int(row["pair_idx"])
    prototype_row = int(row["top_action_prototype_row"])
    profile = trace_profile(record)
    compatible = profile_compatible(
        profile,
        expected_m=expected_m,
        expected_k=expected_k,
        expected_roles=expected_roles,
        expected_layers=expected_layers,
    )
    if strict_config and not compatible:
        raise ValueError(f"Trace profile mismatch: observed={profile}")

    matrix = graph["sync_visual_to_text_attn"][pair_idx, prototype_row].float()
    alpha = graph["sync_evidence_weights"][pair_idx, prototype_row].float()
    visual_mask = graph["candidate_evidence_mask"][pair_idx].bool()
    role_mask = graph["prototype_slot_mask"][pair_idx, prototype_row].bool()
    coverage = graph["sync_prototype_coverage"][pair_idx, prototype_row].float()
    visual_short, visual_names = visual_labels(row, graph)
    prototypes = prototype_rows(record, pair_idx, action_names)
    details = context_details(row)

    canvas_width = 2500
    canvas_height = 1500
    margin = 28
    header = 94
    footer = 54
    left_x, left_width = margin, 780
    middle_x, middle_width = 838, 630
    right_x, right_width = 1498, 974
    panel = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    draw = ImageDraw.Draw(panel)

    put_text(draw, (margin, 14), f"EvidenceHOI mechanism trace #{figure_index}", 31, bold=True)
    put_text(
        draw,
        (margin, 56),
        (
            f"{row['file_name']} | pair H{row['human_node_idx']}-O{row['object_node_idx']} | "
            f"GT/prediction: {row['top_action_name']} {row['object_name']}"
        ),
        19,
        bold=True,
        fill=GREEN,
    )
    profile_text = f"M={profile['m']}, K={profile['k']}, roles={profile['roles']}, L={profile['layers']}"
    put_text(
        draw,
        (canvas_width - 380, 25),
        profile_text,
        17,
        bold=True,
        fill=GREEN if compatible else HUMAN,
    )
    mechanism_score_raw = row.get("mechanism_score", "")
    if mechanism_score_raw not in (None, ""):
        put_text(
            draw,
            (canvas_width - 690, 55),
            (
                f"relative mechanism score={float(mechanism_score_raw):.6f} "
                "(formal threshold 0.05 not met)"
            ),
            14,
            bold=True,
            fill=HUMAN,
        )

    overlay = fit_box(draw_overlay(record, source, row), left_width, 810)
    image_y = header + 14
    image_x = left_x + (left_width - overlay.width) // 2
    panel.paste(ImageOps.expand(overlay, border=3, fill=LINE), (image_x - 3, image_y - 3))
    summary_y = image_y + overlay.height + 24
    rounded(draw, (left_x, summary_y, left_x + left_width, summary_y + 195), SOFT_GREEN, outline=GREEN)
    put_text(draw, (left_x + 16, summary_y + 12), "Target-pair and context audit", 21, bold=True, fill=GREEN)
    put_text(
        draw,
        (left_x + 16, summary_y + 48),
        (
            f"final p={float(row['top_action_probability']):.3f}; "
            f"pair proposal={float(row['proposal_probability']):.3f}; "
            f"GT actions={row['gt_action_names']}"
        ),
        16,
    )
    put_text(
        draw,
        (left_x + 16, summary_y + 80),
        (
            f"informative contexts={row['informative_context_count']}; "
            f"two-hop={row['informative_two_hop_count']}; "
            f"context alpha={float(row.get('alpha_context_percent', 0.0)):.5f}%"
        ),
        16,
    )
    put_wrapped(
        draw,
        (left_x + 16, summary_y + 113),
        "Red/blue boxes are the target human/object. Green or gold boxes are selected context nodes; lines show available H-C and C-O relations.",
        15,
        80,
        fill=MUTED,
    )

    y = header + 4
    y = draw_chain(
        draw,
        middle_x,
        y,
        middle_width,
        visual_short,
        visual_names,
        alpha,
        visual_mask,
        details,
    )
    y += 18
    y = draw_prototypes(draw, middle_x, y, middle_width, prototypes, prototype_row)
    y += 18
    draw_prediction_scores(draw, middle_x, y, middle_width, row)

    draw_correspondence(
        draw,
        right_x,
        header + 4,
        right_width,
        matrix,
        alpha,
        visual_mask,
        role_mask,
        visual_short,
        coverage,
        correspondence_view,
    )
    rounded(
        draw,
        (right_x, 1135, right_x + right_width, 1398),
        SOFT_BLUE,
        outline=BLUE,
    )
    put_text(draw, (right_x + 18, 1152), "Interpretation boundary", 21, bold=True, fill=BLUE)
    put_wrapped(
        draw,
        (right_x + 18, 1190),
        (
            "The matrix, alpha, pi, and coverage are learned soft model states. "
            "The deviation view is centered on the row-uniform probability and does not change the saved values. "
            "They visualize correspondence and refinement dynamics; they are not "
            "human evidence labels, causal attributions, or proof of correct evidence."
        ),
        17,
        85,
        fill=INK,
        spacing=6,
    )
    put_text(
        draw,
        (margin, canvas_height - footer + 17),
        "All boxes, paths, scores, prototypes, correspondences, alpha, pi, and coverage are read from the saved evaluation Trace.",
        15,
        fill=MUTED,
    )

    metadata = {
        "figure_index": figure_index,
        "candidate_id": row.get("candidate_id", ""),
        "record_path": row["record_path"],
        "file_name": row["file_name"],
        "pair_idx": pair_idx,
        "prototype_row": prototype_row,
        "action": row["top_action_name"],
        "object": row["object_name"],
        "trace_profile": profile,
        "profile_compatible": compatible,
        "final_role_labels": FINAL_ROLE_LABELS,
        "visual_labels": visual_short,
        "visual_names": visual_names,
        "correspondence_semantics": "visual_to_text_attn = P(role | visual unit)",
        "correspondence_view": correspondence_view,
        "correspondence_uniform_baseline": 1.0 / max(int(role_mask.sum().item()), 1),
        "correspondence_max_abs_deviation": float(
            (
                matrix[visual_mask][:, role_mask]
                - 1.0 / max(int(role_mask.sum().item()), 1)
            )
            .abs()
            .max()
            .item()
        ),
        "correspondence_matrix": matrix.tolist(),
        "alpha_evidence_weights": alpha.tolist(),
        "pi_prototype_weights": graph["sync_prototype_weights"][pair_idx].float().tolist(),
        "coverage": coverage.tolist(),
        "prototype_rows": prototypes,
        "interpretation_boundary": (
            "Learned soft support only; not ground-truth evidence or causal attribution."
        ),
    }
    prototype_names = ", ".join(str(item["action_name"]) for item in prototypes)
    caption = (
        f"Learned EvidenceHOI correspondence for {row['top_action_name']} "
        f"{row['object_name']}. The target human and object are shown in red and blue; "
        f"selected contextual entities are shown in green/gold. The model retrieves "
        f"the Top-K prototypes ({prototype_names}) and forms a soft correspondence "
        f"between the M+3 visual units and the six KPT roles H/O/A/K/S/C. "
        f"The deviation view is centered on its row-uniform baseline. Alpha and "
        f"pi denote learned evidence and prototype aggregation weights, respectively. "
        f"These quantities diagnose model behavior and are not ground-truth or causal evidence."
    )
    return panel, metadata, caption


def parse_ranks(spec: str, row_count: int) -> List[int]:
    ranks: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            ranks.extend(range(int(start_raw), int(end_raw) + 1))
        else:
            ranks.append(int(part))
    unique = []
    seen = set()
    for rank in ranks:
        if rank < 1 or rank > row_count:
            raise IndexError(f"Rank {rank} is outside 1..{row_count}")
        if rank not in seen:
            seen.add(rank)
            unique.append(rank)
    return unique


def make_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    if not paths:
        return
    thumb_width = 1250
    columns = 2
    thumbs: List[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        height = int(round(image.height * thumb_width / image.width))
        thumbs.append(image.resize((thumb_width, height), Image.Resampling.LANCZOS))
    rows = (len(thumbs) + columns - 1) // columns
    cell_height = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_height), (232, 235, 239))
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, columns)
        sheet.paste(thumb, (col * thumb_width, row * cell_height))
    sheet.save(output, quality=94)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-csv", required=True, type=Path)
    parser.add_argument("--ranks", default="1", help="One-based ranks, e.g. 1,3,5-7")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--image-pattern")
    parser.add_argument("--action-names-json", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-m", type=int, default=4)
    parser.add_argument("--expected-k", type=int, default=5)
    parser.add_argument("--expected-roles", type=int, default=6)
    parser.add_argument("--expected-layers", type=int, default=2)
    parser.add_argument("--strict-config", action="store_true")
    parser.add_argument(
        "--correspondence-view",
        choices=("probability", "deviation"),
        default="probability",
        help="Show absolute P(role|visual unit) or signed deviation from the row-uniform baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.ranking_csv.open("r", encoding="utf-8")))
    ranks = parse_ranks(args.ranks, len(rows))
    action_names = load_name_map(args.action_names_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []
    metadata_rows: List[Dict[str, Any]] = []
    captions: List[str] = []
    failures: List[str] = []

    for figure_index, rank in enumerate(ranks, start=1):
        row = rows[rank - 1]
        try:
            record = load_trace(Path(row["record_path"]))
            image_path = resolve_image(record, args.image_root, args.image_pattern)
            if image_path is None:
                raise FileNotFoundError(f"Cannot resolve source image for {row['record_path']}")
            with Image.open(image_path) as source:
                panel, metadata, caption = build_figure(
                    record,
                    source,
                    row,
                    action_names,
                    figure_index,
                    expected_m=args.expected_m,
                    expected_k=args.expected_k,
                    expected_roles=args.expected_roles,
                    expected_layers=args.expected_layers,
                    strict_config=args.strict_config,
                    correspondence_view=args.correspondence_view,
                )
            output = (
                args.output_dir
                / f"{figure_index:02d}_{Path(row['file_name']).stem}_p{row['pair_idx']}_{row['top_action_name']}.png"
            )
            panel.save(output)
            metadata["ranking_csv"] = str(args.ranking_csv)
            metadata["ranking_rank"] = rank
            metadata["source_image_path"] = str(image_path)
            metadata["rendered_file"] = str(output)
            output_paths.append(output)
            metadata_rows.append(metadata)
            captions.append(f"Figure S{figure_index}. {caption}")
        except (OSError, KeyError, ValueError, IndexError, RuntimeError, json.JSONDecodeError) as exc:
            failures.append(f"rank {rank}: {type(exc).__name__}: {exc}")

    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "caption_drafts.md").write_text(
        "\n\n".join(captions) + ("\n" if captions else ""),
        encoding="utf-8",
    )
    (args.output_dir / "render_report.json").write_text(
        json.dumps(
            {
                "requested_ranks": ranks,
                "rendered_count": len(output_paths),
                "failures": failures,
                "strict_config": args.strict_config,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    make_contact_sheet(output_paths, args.output_dir / "supplementary_mechanism_contact.jpg")
    print(f"Rendered {len(output_paths)}/{len(ranks)} figures to {args.output_dir}")
    for failure in failures:
        print(f"[render failure] {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
