#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = PROJECT_ROOT / "tools" / "evidence_vis"
if str(VIS_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_DIR))

import rank_supplementary_examples as ranker
import render_supplementary_mechanism as renderer


def make_record() -> dict:
    visual_to_text = torch.full((7, 6), 0.02)
    assignments = [2, 0, 1, 3, 4, 5, 3]
    for visual, role in enumerate(assignments):
        visual_to_text[visual, role] = 0.90
    visual_to_text = visual_to_text / visual_to_text.sum(dim=-1, keepdim=True)
    text_to_visual = visual_to_text.transpose(0, 1)
    text_to_visual = text_to_visual / text_to_visual.sum(dim=-1, keepdim=True)
    visual_to_text = visual_to_text.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    text_to_visual = text_to_visual.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    alpha = torch.tensor([0.12, 0.24, 0.20, 0.16, 0.12, 0.10, 0.06])
    alpha = alpha.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1)
    coverage = text_to_visual.max(dim=-1).values

    return {
        "image": {
            "image_id": 1,
            "file_name": "synthetic.jpg",
            "orig_size_hw": [480, 640],
        },
        "nodes": {
            "boxes_xyxy_norm": torch.tensor(
                [
                    [0.05, 0.08, 0.36, 0.94],
                    [0.35, 0.40, 0.70, 0.82],
                    [0.60, 0.05, 0.82, 0.34],
                    [0.74, 0.42, 0.96, 0.78],
                    [0.42, 0.08, 0.58, 0.28],
                    [0.12, 0.55, 0.30, 0.76],
                ]
            ),
            "scores": torch.tensor([0.99, 0.97, 0.92, 0.89, 0.86, 0.83]),
            "labels": torch.tensor([1, 2, 3, 4, 5, 6]),
            "types": torch.tensor([0, 1, 1, 1, 1, 1]),
            "valid_mask": torch.ones(6, dtype=torch.bool),
        },
        "pairs": {
            "indices": torch.tensor([[0, 1]]),
            "action_probs": torch.tensor([[0.10, 0.18, 0.82, 0.35]]),
            "coarse_action_probs": torch.tensor([[0.12, 0.20, 0.55, 0.42]]),
            "gt_action_targets": torch.tensor([[0, 0, 1, 0]], dtype=torch.uint8),
            "gt_pair_matched": torch.tensor([True]),
            "proposal_probs": torch.tensor([0.91]),
            "object_scores": torch.tensor([0.97]),
            "object_labels": torch.tensor([2]),
        },
        "graph": {
            "schema_version": 1,
            "stage": "layer",
            "layer_idx": 1,
            "visual_token_roles": [
                "anchor",
                "human",
                "object",
                "context_1",
                "context_2",
                "context_3",
                "context_4",
            ],
            "prototype_slot_names": [
                "human_state",
                "object_state",
                "action_cue",
                "key_cue",
                "scene_cue",
                "context_object",
            ],
            "pair_valid_mask": torch.tensor([True]),
            "selected_context_indices": torch.tensor([[2, 3, 4, 5]]),
            "selected_context_mask": torch.ones(1, 4, dtype=torch.bool),
            "selected_context_two_hop_mask": torch.tensor([[True, True, False, True]]),
            "selected_context_edge_hk_mask": torch.tensor([[True, True, True, True]]),
            "selected_context_edge_ko_mask": torch.tensor([[True, True, False, True]]),
            "candidate_evidence_mask": torch.ones(1, 7, dtype=torch.bool),
            "candidate_evidence_weights": alpha[:, 0],
            "selected_hoi_action_ids": torch.tensor([[0, 2, 3]]),
            "selected_hoi_mask": torch.ones(1, 3, dtype=torch.bool),
            "selected_hoi_scores": torch.tensor([[0.41, 0.83, 0.52]]),
            "prototype_slot_mask": torch.ones(1, 3, 6, dtype=torch.bool),
            "sync_visual_to_text_attn": visual_to_text,
            "sync_text_to_visual_attn": text_to_visual,
            "sync_evidence_weights": alpha,
            "sync_prototype_scores": torch.tensor([[0.1, 1.5, 0.3]]),
            "sync_prototype_weights": torch.tensor([[0.16, 0.68, 0.16]]),
            "sync_prototype_coverage": coverage,
            "sync_anchor_delta": torch.ones(1, 256) * 0.01,
        },
        "metadata": {"graph_trace_available": True},
    }


def make_row() -> dict:
    details = []
    names = ["helmet", "road", "traffic light", "car"]
    for slot, name in enumerate(names, start=1):
        details.append(
            {
                "slot": slot,
                "node_idx": slot + 1,
                "name": name,
                "detector_score": 0.90 - slot * 0.02,
                "informative": True,
                "two_hop": slot != 3,
                "hk_valid": True,
                "ko_valid": slot != 3,
                "hk_strength": 0.8,
                "ko_strength": 0.7,
            }
        )
    return {
        "candidate_id": "synthetic__p0__a2",
        "record_path": "synthetic.pt",
        "file_name": "synthetic.jpg",
        "pair_idx": "0",
        "human_node_idx": "0",
        "object_node_idx": "1",
        "top_action_id": "2",
        "top_action_name": "ride",
        "top_action_probability": "0.82",
        "top_action_prototype_row": "1",
        "object_name": "bicycle",
        "proposal_probability": "0.91",
        "gt_action_ids": "2",
        "gt_action_names": "ride",
        "informative_context_count": "4",
        "informative_two_hop_count": "3",
        "alpha_context_percent": "44.0",
        "best_gt_coarse_probability": "0.55",
        "best_gt_final_probability": "0.82",
        "competitor_action_name": "hold",
        "competitor_coarse_probability": "0.42",
        "competitor_final_probability": "0.35",
        "context_details_json": json.dumps(details),
    }


def test_profile_and_mechanism_metrics() -> None:
    record = make_record()
    profile = ranker.trace_profile(record)
    assert ranker.profile_compatible(
        profile,
        expected_m=4,
        expected_k=3,
        expected_roles=6,
        expected_layers=2,
    )
    metrics = ranker.mechanism_metrics(record, pair_idx=0, prototype_row=1)
    assert metrics["mechanism_valid"]
    assert metrics["endpoint_role_support"] > 0.5
    assert metrics["context_role_support"] > 0.2
    assert metrics["correspondence_concentration"] > 0.5


def test_renderer_builds_full_mechanism_panel() -> None:
    panel, metadata, caption = renderer.build_figure(
        make_record(),
        Image.new("RGB", (640, 480), (210, 220, 230)),
        make_row(),
        {0: "hold", 2: "ride", 3: "sit_on"},
        1,
        expected_m=4,
        expected_k=3,
        expected_roles=6,
        expected_layers=2,
        strict_config=True,
    )
    assert panel.size == (2500, 1500)
    assert metadata["profile_compatible"]
    assert len(metadata["correspondence_matrix"]) == 7
    assert len(metadata["correspondence_matrix"][0]) == 6
    assert "H/O/A/K/S/C" in caption
    assert "not ground-truth or causal evidence" in caption


def main() -> None:
    test_profile_and_mechanism_metrics()
    test_renderer_builds_full_mechanism_panel()
    print("supplementary_evidence_visualization: all tests passed")


if __name__ == "__main__":
    main()
