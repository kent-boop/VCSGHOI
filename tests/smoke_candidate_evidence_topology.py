import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.candidate_evidence_topology import CandidateEvidenceTopologyInducer
from src.models.visual_scene_graph_oov6 import VisualSceneGraphReasoner


def assert_no_nan(name, tensor):
    assert not torch.isnan(tensor).any().item(), f"{name} contains NaN"


def build_ceti_inputs():
    torch.manual_seed(7)
    bsz, num_nodes, num_pairs, dim = 2, 6, 4, 32
    node_feats = torch.randn(bsz, num_nodes, dim)
    edge_attr_dense = torch.randn(bsz, num_nodes, num_nodes, dim)
    edge_mask_dense = torch.zeros(bsz, num_nodes, num_nodes, dtype=torch.bool)
    node_mask = torch.ones(bsz, num_nodes, dtype=torch.bool)

    edge_mask_dense[:, 0, 1] = True
    edge_mask_dense[:, 0, 2] = True
    edge_mask_dense[:, 2, 1] = True
    edge_mask_dense[:, 3, 4] = True

    pair_indices = torch.tensor([
        [[0, 1], [0, 2], [3, 4], [-1, 2]],
        [[0, 1], [1, 2], [2, 3], [5, 0]],
    ], dtype=torch.long)
    pair_valid_mask = torch.ones(bsz, num_pairs, dtype=torch.bool)
    pair_valid_mask[0, 3] = False
    return node_feats, edge_attr_dense, edge_mask_dense, node_mask, pair_indices, pair_valid_mask


def test_candidate_evidence_inducer_standalone():
    bsz, num_nodes, num_pairs, dim = 2, 6, 4, 32
    node_feats, edge_attr_dense, edge_mask_dense, node_mask, pair_indices, pair_valid_mask = build_ceti_inputs()
    module = CandidateEvidenceTopologyInducer(
        d_model=dim,
        max_context_paths=3,
        dropout=0.0,
        gate_init=0.0,
        debug=True,
    )
    module.eval()

    with torch.no_grad():
        out = module(
            node_feats=node_feats,
            edge_attr_dense=edge_attr_dense,
            edge_mask_dense=edge_mask_dense,
            node_mask=node_mask,
            pair_indices=pair_indices,
            pair_valid_mask=pair_valid_mask,
        )

    assert out.evidence_tokens.shape == (bsz, num_pairs, 6, dim)
    assert out.evidence_mask.shape == (bsz, num_pairs, 6)
    assert out.evidence_weights.shape == (bsz, num_pairs, 6)
    assert out.evidence_context.shape == (bsz, num_pairs, dim)
    assert out.anchor_delta.shape == (bsz, num_pairs, dim)
    assert out.selected_context_indices.shape == (bsz, num_pairs, 3)
    assert out.selected_context_mask.shape == (bsz, num_pairs, 3)

    assert_no_nan("evidence_tokens", out.evidence_tokens)
    assert_no_nan("evidence_weights", out.evidence_weights)
    assert_no_nan("evidence_context", out.evidence_context)
    assert_no_nan("anchor_delta", out.anchor_delta)
    assert torch.allclose(out.anchor_delta[0, 3], torch.zeros_like(out.anchor_delta[0, 3]))
    assert out.anchor_delta.abs().max().item() <= 1e-7


def test_candidate_evidence_no_valid_context():
    bsz, _, num_pairs, dim = 2, 6, 4, 32
    node_feats, edge_attr_dense, edge_mask_dense, node_mask, pair_indices, pair_valid_mask = build_ceti_inputs()
    edge_mask_dense.zero_()
    module = CandidateEvidenceTopologyInducer(
        d_model=dim,
        max_context_paths=3,
        dropout=0.0,
        gate_init=0.0,
        allow_node_only_context=False,
        debug=True,
    )
    module.eval()

    with torch.no_grad():
        out = module(
            node_feats=node_feats,
            edge_attr_dense=edge_attr_dense,
            edge_mask_dense=edge_mask_dense,
            node_mask=node_mask,
            pair_indices=pair_indices,
            pair_valid_mask=pair_valid_mask,
        )

    assert out.evidence_tokens.shape == (bsz, num_pairs, 6, dim)
    assert out.evidence_mask.shape == (bsz, num_pairs, 6)
    assert out.evidence_context.shape == (bsz, num_pairs, dim)
    assert_no_nan("no_context evidence_tokens", out.evidence_tokens)
    assert_no_nan("no_context evidence_weights", out.evidence_weights)
    assert_no_nan("no_context evidence_context", out.evidence_context)
    assert_no_nan("no_context anchor_delta", out.anchor_delta)
    assert not out.selected_context_mask.any().item()
    assert out.evidence_mask[..., :3].eq(pair_valid_mask.unsqueeze(-1)).all().item()
    assert not out.evidence_mask[..., 3:].any().item()


def build_reasoner_inputs():
    torch.manual_seed(11)
    bsz, num_nodes, num_pairs, dim = 2, 6, 4, 32
    node_feats = torch.randn(bsz, num_nodes, dim)
    xy_min = torch.rand(bsz, num_nodes, 2) * 0.6
    wh = torch.rand(bsz, num_nodes, 2) * 0.3 + 0.05
    xy_max = (xy_min + wh).clamp(max=1.0)
    boxes = torch.cat([xy_min, xy_max], dim=-1)
    labels = torch.randint(0, 80, (bsz, num_nodes), dtype=torch.long)
    node_types = torch.ones(bsz, num_nodes, dtype=torch.long)
    node_types[:, 0] = 0
    node_types[:, 3] = 0
    node_mask = torch.ones(bsz, num_nodes, dtype=torch.bool)
    pair_indices = torch.tensor([
        [[0, 1], [0, 2], [3, 4], [3, 5]],
        [[0, 1], [0, 2], [3, 4], [3, 5]],
    ], dtype=torch.long)
    pair_scores = torch.randn(bsz, num_pairs)
    pair_feats = torch.randn(bsz, num_pairs, dim)
    union_feats = torch.randn(bsz, num_pairs, dim)
    return node_feats, boxes, labels, node_types, node_mask, pair_indices, pair_scores, pair_feats, union_feats


def test_visual_scene_graph_reasoner_candidate_evidence_on():
    bsz, num_nodes, num_pairs, dim = 2, 6, 4, 32
    model = VisualSceneGraphReasoner(
        d_model=dim,
        num_obj_classes=80,
        num_layers=1,
        num_heads=4,
        k_oo=2,
        enable_oo_edges=True,
        enable_text_graph=False,
        dropout=0.0,
        graph_enable_candidate_evidence=True,
        graph_evidence_topk=3,
        graph_evidence_gate_init=0.0,
        graph_evidence_debug=True,
    )
    model.eval()
    inputs = build_reasoner_inputs()

    with torch.no_grad():
        out = model(*inputs)

    assert out.pair_feats.shape == (bsz, num_pairs, dim)
    assert out.node_feats.shape == (bsz, num_nodes, dim)
    assert out.debug_info["candidate_evidence_enabled"] is True
    assert out.debug_info["candidate_evidence_applied"] is True
    assert "candidate_evidence_tokens_shape" in out.debug_info
    assert_no_nan("reasoner_on pair_feats", out.pair_feats)
    assert_no_nan("reasoner_on node_feats", out.node_feats)


def test_visual_scene_graph_reasoner_candidate_evidence_off():
    bsz, _, num_pairs, dim = 2, 6, 4, 32
    model = VisualSceneGraphReasoner(
        d_model=dim,
        num_obj_classes=80,
        num_layers=1,
        num_heads=4,
        k_oo=2,
        enable_oo_edges=True,
        enable_text_graph=False,
        dropout=0.0,
    )
    model.eval()
    inputs = build_reasoner_inputs()

    with torch.no_grad():
        out = model(*inputs)

    assert out.pair_feats.shape == (bsz, num_pairs, dim)
    assert out.debug_info["candidate_evidence_enabled"] is False
    assert out.debug_info["candidate_evidence_applied"] is False
    assert_no_nan("reasoner_off pair_feats", out.pair_feats)


def main():
    test_candidate_evidence_inducer_standalone()
    test_candidate_evidence_no_valid_context()
    test_visual_scene_graph_reasoner_candidate_evidence_on()
    test_visual_scene_graph_reasoner_candidate_evidence_off()
    print("smoke_candidate_evidence_topology: all tests passed")


if __name__ == "__main__":
    main()
