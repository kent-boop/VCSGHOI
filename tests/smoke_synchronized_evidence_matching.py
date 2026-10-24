import json
import os
import sys
import tempfile

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.synchronized_evidence_topology_matcher import SynchronizedEvidenceTopologyMatcher
from src.models.visual_scene_graph_oov7 import VisualSceneGraphReasoner


def assert_no_nan(name, tensor):
    assert not torch.isnan(tensor).any().item(), name


def test_sync_matcher_standalone():
    torch.manual_seed(23)
    batch_size, num_pairs, topk, num_visual, num_slots, dim = 2, 3, 4, 7, 6, 32

    visual_tokens = torch.randn(batch_size, num_pairs, num_visual, dim)
    visual_mask = torch.ones(batch_size, num_pairs, num_visual, dtype=torch.bool)
    prototype_slots = torch.randn(batch_size, num_pairs, topk, num_slots, dim)
    prototype_slot_mask = torch.ones(batch_size, num_pairs, topk, num_slots, dtype=torch.bool)
    pair_valid_mask = torch.ones(batch_size, num_pairs, dtype=torch.bool)

    pair_valid_mask[0, 2] = False
    prototype_slot_mask[1, 1, 3, :] = False
    prototype_slot_mask[0, 0, 0, 4] = False
    visual_mask[1, 0, 5] = False

    matcher = SynchronizedEvidenceTopologyMatcher(
        d_model=dim,
        num_slots=num_slots,
        dropout=0.0,
        visual_gate_init=0.0,
        text_gate_init=0.0,
        anchor_gate_init=0.0,
        normalize_inputs=True,
        debug=True,
    )
    matcher.eval()

    with torch.no_grad():
        out = matcher(
            visual_tokens=visual_tokens,
            visual_mask=visual_mask,
            prototype_slots=prototype_slots,
            prototype_slot_mask=prototype_slot_mask,
            pair_valid_mask=pair_valid_mask,
        )

    assert out.match_logits.shape == (batch_size, num_pairs, topk, num_visual, num_slots)
    assert out.visual_to_text_attn.shape == (batch_size, num_pairs, topk, num_visual, num_slots)
    assert out.text_to_visual_attn.shape == (batch_size, num_pairs, topk, num_slots, num_visual)
    assert out.updated_visual_tokens.shape == (batch_size, num_pairs, topk, num_visual, dim)
    assert out.updated_prototype_slots.shape == (batch_size, num_pairs, topk, num_slots, dim)
    assert out.evidence_weights.shape == (batch_size, num_pairs, topk, num_visual)
    assert out.prototype_scores.shape == (batch_size, num_pairs, topk)
    assert out.prototype_weights.shape == (batch_size, num_pairs, topk)
    assert out.prototype_coverage.shape == (batch_size, num_pairs, topk, num_slots)
    assert out.matched_context.shape == (batch_size, num_pairs, dim)
    assert out.anchor_delta.shape == (batch_size, num_pairs, dim)

    for name, tensor in [
        ("match_logits", out.match_logits),
        ("visual_to_text_attn", out.visual_to_text_attn),
        ("text_to_visual_attn", out.text_to_visual_attn),
        ("updated_visual_tokens", out.updated_visual_tokens),
        ("updated_prototype_slots", out.updated_prototype_slots),
        ("evidence_weights", out.evidence_weights),
        ("prototype_scores", out.prototype_scores),
        ("prototype_weights", out.prototype_weights),
        ("prototype_coverage", out.prototype_coverage),
        ("matched_context", out.matched_context),
        ("anchor_delta", out.anchor_delta),
    ]:
        assert_no_nan(name, tensor)

    assert out.anchor_delta[0, 2].abs().sum().item() == 0
    assert out.matched_context[0, 2].abs().sum().item() == 0
    assert out.evidence_weights[0, 2].abs().sum().item() == 0
    assert out.anchor_delta.abs().max().item() < 1e-6


def make_text_assets(tmpdir, d_model=32, num_hoi=6):
    field_order = [
        "human_node",
        "object_node",
        "ho_edge",
        "discriminative",
        "scene_context",
        "oo_context",
    ]
    hoi_ids = [
        "0_0",
        "1_0",
        "2_1",
        "3_1",
        "4_2",
        "5_2",
    ]
    object_to_hoi_indices = {
        "0": [0, 1],
        "1": [2, 3],
        "2": [4, 5],
    }

    text_features = np.random.randn(num_hoi, 6, d_model).astype("float32")
    text_mask = np.ones((num_hoi, 6), dtype=bool)

    field_order_path = os.path.join(tmpdir, "field_order.json")
    hoi_ids_path = os.path.join(tmpdir, "hoi_ids.json")
    object_to_hoi_indices_path = os.path.join(tmpdir, "object_to_hoi_indices.json")
    text_features_path = os.path.join(tmpdir, "text_features.npy")
    text_mask_path = os.path.join(tmpdir, "text_mask.npy")

    with open(field_order_path, "w", encoding="utf-8") as f:
        json.dump(field_order, f)
    with open(hoi_ids_path, "w", encoding="utf-8") as f:
        json.dump(hoi_ids, f)
    with open(object_to_hoi_indices_path, "w", encoding="utf-8") as f:
        json.dump(object_to_hoi_indices, f)

    np.save(text_features_path, text_features)
    np.save(text_mask_path, text_mask)

    return {
        "text_features_path": text_features_path,
        "text_mask_path": text_mask_path,
        "hoi_ids_path": hoi_ids_path,
        "object_to_hoi_indices_path": object_to_hoi_indices_path,
        "field_order_path": field_order_path,
    }


def build_visual_inputs():
    torch.manual_seed(29)
    batch_size, num_nodes, num_pairs, dim = 1, 6, 3, 32
    node_feats = torch.randn(batch_size, num_nodes, dim)
    node_blip_feats = torch.randn(batch_size, num_nodes, dim)
    boxes = torch.tensor([[
        [0.05, 0.05, 0.25, 0.45],
        [0.30, 0.10, 0.50, 0.40],
        [0.55, 0.12, 0.75, 0.42],
        [0.15, 0.55, 0.35, 0.85],
        [0.60, 0.55, 0.85, 0.90],
        [0.38, 0.50, 0.52, 0.70],
    ]], dtype=torch.float32)
    labels = torch.tensor([[0, 0, 1, 2, 1, 2]], dtype=torch.long)
    node_types = torch.tensor([[0, 1, 1, 1, 0, 1]], dtype=torch.long)
    node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
    pair_indices = torch.tensor([[
        [0, 1],
        [0, 2],
        [4, 5],
    ]], dtype=torch.long)
    pair_scores = torch.randn(batch_size, num_pairs)
    pair_feats = torch.randn(batch_size, num_pairs, dim)
    union_feats = torch.randn(batch_size, num_pairs, dim)
    return {
        "node_feats": node_feats,
        "node_blip_feats": node_blip_feats,
        "boxes": boxes,
        "labels": labels,
        "node_types": node_types,
        "node_mask": node_mask,
        "pair_indices": pair_indices,
        "pair_scores": pair_scores,
        "pair_feats": pair_feats,
        "union_feats": union_feats,
        "global_feat": None,
    }


def test_oov7_full_forward_with_sync_matching():
    batch_size, num_nodes, num_pairs, dim = 1, 6, 3, 32
    with tempfile.TemporaryDirectory() as tmpdir:
        assets = make_text_assets(tmpdir, d_model=dim, num_hoi=6)
        model = VisualSceneGraphReasoner(
            d_model=dim,
            num_obj_classes=80,
            num_layers=1,
            num_heads=4,
            k_oo=2,
            enable_oo_edges=True,
            enable_text_graph=True,
            text_features_path=assets["text_features_path"],
            text_mask_path=assets["text_mask_path"],
            hoi_ids_path=assets["hoi_ids_path"],
            object_to_hoi_indices_path=assets["object_to_hoi_indices_path"],
            field_order_path=assets["field_order_path"],
            graph_text_topk=2,
            graph_text_pre_topk=3,
            graph_text_num_layers=1,
            graph_text_num_heads=4,
            graph_text_cross_num_heads=4,
            graph_text_sync_mode="posthoc",
            graph_text_prior_injection="none",
            graph_text_dense_injection=False,
            graph_text_debug=True,
            graph_enable_candidate_evidence=True,
            graph_evidence_topk=3,
            graph_evidence_gate_init=0.0,
            graph_evidence_debug=True,
            graph_enable_sync_evidence_matching=True,
            graph_sync_visual_gate_init=0.0,
            graph_sync_text_gate_init=0.0,
            graph_sync_anchor_gate_init=0.0,
            graph_sync_debug=True,
            graph_text_force_oov3=True,
            graph_text_prototype_slot_debug=True,
            dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            out = model(**build_visual_inputs())

    assert out.pair_feats.shape == (batch_size, num_pairs, dim)
    assert out.node_feats.shape == (batch_size, num_nodes, dim)
    assert_no_nan("oov7 pair_feats", out.pair_feats)
    assert_no_nan("oov7 node_feats", out.node_feats)

    dbg = out.debug_info
    assert dbg["candidate_evidence_enabled"] is True
    assert dbg["candidate_evidence_applied"] is True
    assert dbg["sync_evidence_enabled"] is True
    assert dbg["sync_evidence_applied"] is True
    assert "sync_match_logits_shape" in dbg
    assert dbg["sync_match_logits_shape"][0] == batch_size
    assert dbg["sync_match_logits_shape"][1] == num_pairs
    assert dbg["sync_match_logits_shape"][-1] == 6
    assert "text_prototype_slots_shape" in dbg
    assert dbg["text_prototype_slots_shape"] is not None
    assert "sync_has_nan" in dbg
    assert dbg["sync_has_nan"] is False


def test_oov7_default_off_path():
    batch_size, num_nodes, num_pairs, dim = 1, 6, 3, 32
    model_off = VisualSceneGraphReasoner(
        d_model=dim,
        num_obj_classes=80,
        num_layers=1,
        num_heads=4,
        k_oo=2,
        enable_oo_edges=True,
        enable_text_graph=False,
        graph_enable_candidate_evidence=False,
        graph_enable_sync_evidence_matching=False,
        dropout=0.0,
    )
    model_off.eval()

    with torch.no_grad():
        out_off = model_off(**build_visual_inputs())

    assert out_off.pair_feats.shape == (batch_size, num_pairs, dim)
    assert out_off.node_feats.shape == (batch_size, num_nodes, dim)
    assert_no_nan("oov7 off pair_feats", out_off.pair_feats)
    dbg_off = out_off.debug_info
    assert dbg_off.get("sync_evidence_enabled", False) is False
    assert dbg_off.get("sync_evidence_applied", False) is False


def test_oov7_invalid_configs():
    dim = 32
    try:
        VisualSceneGraphReasoner(
            d_model=dim,
            num_obj_classes=80,
            num_layers=1,
            num_heads=4,
            enable_text_graph=False,
            graph_enable_candidate_evidence=True,
            graph_enable_sync_evidence_matching=True,
        )
        raise AssertionError("Expected ValueError when sync matching is enabled without text graph")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        assets = make_text_assets(tmpdir, d_model=dim, num_hoi=6)
        try:
            VisualSceneGraphReasoner(
                d_model=dim,
                num_obj_classes=80,
                num_layers=1,
                num_heads=4,
                enable_text_graph=True,
                text_features_path=assets["text_features_path"],
                text_mask_path=assets["text_mask_path"],
                hoi_ids_path=assets["hoi_ids_path"],
                object_to_hoi_indices_path=assets["object_to_hoi_indices_path"],
                field_order_path=assets["field_order_path"],
                graph_enable_candidate_evidence=False,
                graph_enable_sync_evidence_matching=True,
            )
            raise AssertionError("Expected ValueError when sync matching is enabled without CETI")
        except ValueError:
            pass


def main():
    test_sync_matcher_standalone()
    test_oov7_full_forward_with_sync_matching()
    test_oov7_default_off_path()
    test_oov7_invalid_configs()
    print("Smoke test passed: synchronized evidence topology matching.")


if __name__ == "__main__":
    main()
