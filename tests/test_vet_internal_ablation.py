from types import SimpleNamespace

import pytest
import torch

from src.models.candidate_evidence_topology_vet_ablation import (
    CandidateEvidenceTopologyVETAblation,
    VET_ABLATION_SPECS,
)
from src.models.setm_ablation_matcher import SETMAblationMatcher
from src.models.visual_evidence_topology_vet_ablation_layerwise_setm import (
    VisualEvidenceTopologyVETAblationLayerwiseSETM,
)


def candidate_inputs(dim=16):
    torch.manual_seed(17)
    bsz, nodes, pairs = 1, 7, 2
    node_feats = torch.randn(bsz, nodes, dim)
    edge_attr = torch.randn(bsz, nodes, nodes, dim)
    edge_mask = torch.zeros(bsz, nodes, nodes, dtype=torch.bool)
    edge_scores = torch.zeros(bsz, nodes, nodes)
    # q0=(0,1): node 2 is the strongest two-hop path, node 4 a weaker
    # two-hop path, and node 3 is one-sided.
    for src, dst, score in [(0, 2, .9), (2, 1, .8), (0, 4, .5), (4, 1, .4), (0, 3, .95)]:
        edge_mask[0, src, dst] = True
        edge_scores[0, src, dst] = score
    node_mask = torch.ones(bsz, nodes, dtype=torch.bool)
    pair_indices = torch.tensor([[[0, 1], [-1, 2]]])
    pair_valid = torch.tensor([[True, False]])
    return dict(
        node_feats=node_feats,
        edge_attr_dense=edge_attr,
        edge_mask_dense=edge_mask,
        edge_score_dense=edge_scores,
        node_mask=node_mask,
        pair_indices=pair_indices,
        pair_valid_mask=pair_valid,
    )


@pytest.mark.parametrize("mode", list(VET_ABLATION_SPECS))
def test_fixed_length_masks_and_invalid_pair(mode):
    module = CandidateEvidenceTopologyVETAblation(
        d_model=16,
        max_context_paths=3,
        dropout=0.0,
        gate_init=1.0,
        ablation_mode=mode,
    )
    out = module(**candidate_inputs())
    assert out.evidence_tokens.shape == (1, 2, 6, 16)
    assert out.evidence_mask.shape == (1, 2, 6)
    assert out.evidence_weights.shape == (1, 2, 6)
    expected = {"anchor_raw": 1, "anchor_svrm": 1, "endpoints": 3}.get(mode, 6)
    assert out.evidence_mask[0, 0].sum().item() == expected
    assert not out.evidence_mask[0, 1].any()
    assert out.evidence_tokens[0, 1].abs().sum().item() == 0
    assert out.evidence_weights[0, 1].abs().sum().item() == 0
    assert torch.isfinite(out.evidence_tokens).all()
    assert torch.isfinite(out.evidence_weights).all()
    if mode in {"anchor_raw", "anchor_svrm", "endpoints"}:
        assert out.anchor_delta.abs().sum().item() == 0


def test_context_uniform_is_deterministic_and_prioritizes_connectivity():
    module = CandidateEvidenceTopologyVETAblation(
        d_model=16,
        max_context_paths=3,
        dropout=0.0,
        gate_init=1.0,
        ablation_mode="context_uniform",
    )
    first = module(**candidate_inputs())
    second = module(**candidate_inputs())
    assert first.selected_context_indices[0, 0].tolist() == [2, 4, 3]
    assert torch.equal(first.selected_context_indices, second.selected_context_indices)
    assert torch.allclose(first.evidence_weights[0, 0], torch.full((6,), 1 / 6))
    assert torch.allclose(first.context_path_weights[0, 0], torch.full((3,), 1 / 3))


def test_v4_v5_share_selection_but_not_weighting():
    v4 = CandidateEvidenceTopologyVETAblation(
        d_model=16, max_context_paths=3, dropout=0.0,
        gate_init=1.0, ablation_mode="path_selection",
    )
    v5 = CandidateEvidenceTopologyVETAblation(
        d_model=16, max_context_paths=3, dropout=0.0,
        gate_init=1.0, ablation_mode="path_reweight",
    )
    v5.load_state_dict(v4.state_dict())
    out4, out5 = v4(**candidate_inputs()), v5(**candidate_inputs())
    assert torch.equal(out4.selected_context_indices, out5.selected_context_indices)
    assert torch.equal(out4.selected_context_mask, out5.selected_context_mask)
    assert torch.allclose(out4.evidence_weights[0, 0], torch.full((6,), 1 / 6))
    assert torch.allclose(out5.evidence_weights[0, 0, :3], torch.full_like(out5.evidence_weights[..., :3], 1 / 6))
    assert torch.allclose(out5.evidence_weights[0, 0, 3:].sum(), torch.tensor(0.5))


def test_v4_selector_has_straight_through_gradient():
    module = CandidateEvidenceTopologyVETAblation(
        d_model=16, max_context_paths=2, dropout=0.0,
        gate_init=0.0, ablation_mode="path_selection",
    )
    out = module(**candidate_inputs())
    loss = out.evidence_tokens[..., 3:, :].square().mean()
    loss.backward()
    grad = sum(
        float(param.grad.abs().sum().item())
        for param in module.path_score_mlp.parameters()
        if param.grad is not None
    )
    assert grad > 0


def test_all_modes_have_identical_parameter_count():
    counts = []
    for mode in VET_ABLATION_SPECS:
        module = CandidateEvidenceTopologyVETAblation(
            d_model=16, max_context_paths=3, dropout=0.0, ablation_mode=mode
        )
        counts.append(sum(p.numel() for p in module.parameters()))
    assert len(set(counts)) == 1


def reasoner_inputs(dim=16):
    torch.manual_seed(31)
    boxes = torch.tensor([[
        [.05, .05, .25, .8], [.3, .1, .5, .5], [.55, .1, .75, .5],
        [.15, .55, .35, .9], [.6, .55, .85, .9], [.38, .5, .52, .7],
    ]])
    return dict(
        node_feats=torch.randn(1, 6, dim),
        node_blip_feats=torch.randn(1, 6, dim),
        boxes=boxes,
        labels=torch.tensor([[0, 1, 2, 3, 4, 5]]),
        node_types=torch.tensor([[0, 1, 1, 0, 1, 1]]),
        node_mask=torch.ones(1, 6, dtype=torch.bool),
        pair_indices=torch.tensor([[[0, 1], [3, 4]]]),
        pair_scores=torch.tensor([[.9, .8]]),
        pair_feats=torch.randn(1, 2, dim),
        union_feats=torch.randn(1, 2, dim),
        prototype_slots=torch.randn(1, 2, 2, 6, dim),
        prototype_slot_mask=torch.ones(1, 2, 2, 6, dtype=torch.bool),
    )


@pytest.mark.parametrize("mode", ["anchor_raw", "anchor_svrm", "endpoints", "context_uniform", "path_selection", "path_reweight"])
def test_layerwise_schedule_and_svrm_flag(mode):
    model = VisualEvidenceTopologyVETAblationLayerwiseSETM(
        d_model=16, num_obj_classes=80, num_layers=2, num_heads=4,
        k_oo=2, dropout=0.0, graph_oo_mode="hybrid",
        graph_evidence_topk=2, graph_evidence_dropout=0.0,
        graph_evidence_gate_init=1.0, vet_ablation_mode=mode,
    )
    matcher = SETMAblationMatcher(
        d_model=16, num_slots=6, dropout=0.0, stage=4,
        visual_gate_init=1.0, text_gate_init=1.0, anchor_gate_init=1.0,
    )
    out = model(setm_matcher=matcher, setm_stage=4, **reasoner_inputs())
    assert out.debug_info["actual_setm_calls"] == 2
    assert out.debug_info["expected_setm_calls"] == 2
    expected_svrm = mode != "anchor_raw"
    assert all(item["svrm_applied"] == expected_svrm for item in out.layerwise_setm_debug)
    assert out.evidence_tokens.shape == (1, 2, 5, 16)
    assert torch.isfinite(out.pair_feats).all()


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown graph_vet_ablation_mode"):
        CandidateEvidenceTopologyVETAblation(d_model=16, ablation_mode="not-a-mode")

