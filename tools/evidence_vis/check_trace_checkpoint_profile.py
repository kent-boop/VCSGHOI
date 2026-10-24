#!/usr/bin/env python3
"""Inspect architecture-defining EvidenceHOI checkpoint tensor shapes."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict

import torch


RANK_EMBEDDING_KEY = "visual_scene_graph.text_graph_reasoner.rank_embedding.weight"
ROLE_EMBEDDING_KEY = "visual_scene_graph.candidate_evidence_inducer.role_embedding.weight"
BLIP_POSITION_KEY = "blip.visual_encoder.pos_embed"


def inspect_checkpoint(path: Path) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint = torch.load(path, map_location="meta")
    state = checkpoint["model"]
    missing = [
        key for key in (RANK_EMBEDDING_KEY, ROLE_EMBEDDING_KEY, BLIP_POSITION_KEY)
        if key not in state
    ]
    if missing:
        raise KeyError(f"Checkpoint is missing profile keys: {missing}")

    saved_args = checkpoint.get("args")
    args_dict = vars(saved_args) if hasattr(saved_args, "__dict__") else {}
    role_count = int(state[ROLE_EMBEDDING_KEY].shape[0])
    return {
        "checkpoint": str(path),
        "backbone": args_dict.get("backbone"),
        "graph_text_topk": int(state[RANK_EMBEDDING_KEY].shape[0]),
        "graph_evidence_topk": role_count - 3,
        "candidate_role_count": role_count,
        "blip_width": int(state[BLIP_POSITION_KEY].shape[-1]),
        "epoch": checkpoint.get("epoch"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-evidence-topk", required=True, type=int)
    parser.add_argument("--expected-text-topk", required=True, type=int)
    parser.add_argument("--expected-backbone", default="resnet101")
    parser.add_argument("--expected-blip-width", default=1024, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = inspect_checkpoint(args.checkpoint)
    print(
        "Checkpoint profile: "
        f"backbone={profile['backbone']}, "
        f"M={profile['graph_evidence_topk']}, "
        f"R={profile['graph_text_topk']}, "
        f"BLIP-width={profile['blip_width']}, "
        f"epoch={profile['epoch']}"
    )

    errors = []
    expected = {
        "graph_evidence_topk": args.expected_evidence_topk,
        "graph_text_topk": args.expected_text_topk,
        "backbone": args.expected_backbone,
        "blip_width": args.expected_blip_width,
    }
    for key, expected_value in expected.items():
        actual_value = profile[key]
        if actual_value != expected_value:
            errors.append(
                f"{key}: checkpoint has {actual_value}, requested {expected_value}"
            )

    if errors:
        for error in errors:
            print(f"[Checkpoint profile error] {error}")
        raise SystemExit(1)
    print("CHECKPOINT PROFILE PASSED")


if __name__ == "__main__":
    main()
