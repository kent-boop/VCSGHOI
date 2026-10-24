"""Dependency-free smoke checks for the isolated VET ablation.

Run with: ``python tests/smoke_vet_internal_ablation.py``.
"""

import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.candidate_evidence_topology_vet_ablation import (  # noqa: E402
    CandidateEvidenceTopologyVETAblation,
    VET_ABLATION_SPECS,
)
from src.models.setm_ablation_matcher import SETMAblationMatcher  # noqa: E402
from src.models.visual_evidence_topology_vet_ablation_layerwise_setm import (  # noqa: E402
    VisualEvidenceTopologyVETAblationLayerwiseSETM,
)


def candidate_inputs(dim=16):
    torch.manual_seed(17)
    node = torch.randn(1, 7, dim)
    edge = torch.randn(1, 7, 7, dim)
    mask = torch.zeros(1, 7, 7, dtype=torch.bool)
    score = torch.zeros(1, 7, 7)
    for src, dst, value in [(0, 2, .9), (2, 1, .8), (0, 4, .5), (4, 1, .4), (0, 3, .95)]:
        mask[0, src, dst] = True
        score[0, src, dst] = value
    return {
        "node_feats": node,
        "edge_attr_dense": edge,
        "edge_mask_dense": mask,
        "edge_score_dense": score,
        "node_mask": torch.ones(1, 7, dtype=torch.bool),
        "pair_indices": torch.tensor([[[0, 1], [-1, 2]]]),
        "pair_valid_mask": torch.tensor([[True, False]]),
    }


def check_candidate_modes():
    outputs = {}
    counts = []
    expected_counts = {"anchor_raw": 1, "anchor_svrm": 1, "endpoints": 3}
    for mode in VET_ABLATION_SPECS:
        module = CandidateEvidenceTopologyVETAblation(
            d_model=16,
            max_context_paths=3,
            dropout=0.0,
            gate_init=1.0,
            ablation_mode=mode,
        )
        out = module(**candidate_inputs())
        outputs[mode] = (module, out)
        counts.append(sum(param.numel() for param in module.parameters()))
        assert out.evidence_tokens.shape == (1, 2, 6, 16)
        assert out.evidence_mask[0, 0].sum().item() == expected_counts.get(mode, 6)
        assert not out.evidence_mask[0, 1].any()
        assert out.evidence_tokens[0, 1].abs().sum().item() == 0
        assert torch.isfinite(out.evidence_tokens).all()
        assert torch.isfinite(out.evidence_weights).all()
        if mode in expected_counts:
            assert out.anchor_delta.abs().sum().item() == 0
    assert len(set(counts)) == 1
    fixed = outputs["context_uniform"][1]
    assert fixed.selected_context_indices[0, 0].tolist() == [2, 4, 3]
    assert torch.allclose(fixed.evidence_weights[0, 0], torch.full((6,), 1 / 6))

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
    assert torch.allclose(out5.evidence_weights[0, 0, :3], torch.full((3,), 1 / 6))
    assert torch.allclose(out5.evidence_weights[0, 0, 3:].sum(), torch.tensor(.5))
    # With identical selected paths, learned relative path weights must be causal.
    assert not torch.allclose(out4.evidence_context[0, 0], out5.evidence_context[0, 0])
    assert not torch.allclose(out4.anchor_delta[0, 0], out5.anchor_delta[0, 0])

    loss = out4.evidence_tokens[..., 3:, :].square().mean()
    loss.backward()
    selector_grad = sum(
        float(param.grad.abs().sum().item())
        for param in v4.path_score_mlp.parameters()
        if param.grad is not None
    )
    assert selector_grad > 0, "V4 path selector is a no-op: zero gradient"
    return selector_grad


def reasoner_inputs(dim=16):
    torch.manual_seed(31)
    boxes = torch.tensor([[
        [.05, .05, .25, .8], [.3, .1, .5, .5], [.55, .1, .75, .5],
        [.15, .55, .35, .9], [.6, .55, .85, .9], [.38, .5, .52, .7],
    ]])
    return {
        "node_feats": torch.randn(1, 6, dim),
        "node_blip_feats": torch.randn(1, 6, dim),
        "boxes": boxes,
        "labels": torch.tensor([[0, 1, 2, 3, 4, 5]]),
        "node_types": torch.tensor([[0, 1, 1, 0, 1, 1]]),
        "node_mask": torch.ones(1, 6, dtype=torch.bool),
        "pair_indices": torch.tensor([[[0, 1], [3, 4]]]),
        "pair_scores": torch.tensor([[.9, .8]]),
        "pair_feats": torch.randn(1, 2, dim),
        "union_feats": torch.randn(1, 2, dim),
        "prototype_slots": torch.randn(1, 2, 2, 6, dim),
        "prototype_slot_mask": torch.ones(1, 2, 2, 6, dtype=torch.bool),
    }


def check_layerwise_schedule():
    counts = []
    for mode in VET_ABLATION_SPECS:
        model = VisualEvidenceTopologyVETAblationLayerwiseSETM(
            d_model=16,
            num_obj_classes=80,
            num_layers=2,
            num_heads=4,
            k_oo=2,
            dropout=0.0,
            graph_oo_mode="hybrid",
            graph_evidence_topk=2,
            graph_evidence_dropout=0.0,
            graph_evidence_gate_init=1.0,
            vet_ablation_mode=mode,
        )
        matcher = SETMAblationMatcher(
            d_model=16,
            num_slots=6,
            dropout=0.0,
            stage=4,
            visual_gate_init=1.0,
            text_gate_init=1.0,
            anchor_gate_init=1.0,
        )
        out = model(setm_matcher=matcher, setm_stage=4, **reasoner_inputs())
        counts.append(sum(param.numel() for param in model.parameters()))
        assert out.debug_info["actual_setm_calls"] == 2
        assert out.debug_info["expected_setm_calls"] == 2
        expected_svrm = mode != "anchor_raw"
        assert all(row["svrm_applied"] == expected_svrm for row in out.layerwise_setm_debug)
        assert torch.isfinite(out.pair_feats).all()
    assert len(set(counts)) == 1


def main():
    selector_grad = check_candidate_modes()
    check_layerwise_schedule()
    print(f"VET internal ablation smoke passed; V4 selector grad={selector_grad:.6f}")


if __name__ == "__main__":
    main()

