import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.text_prototype_slots import TextPrototypeSlotBuilder
from src.models.text_scene_graph_oov3 import ExplicitTextGraphState, TextSceneGraphReasoner


def build_inputs():
    torch.manual_seed(17)
    batch_size, num_pairs, topk, dim = 2, 3, 4, 16
    num_fields = 6
    num_text_nodes = 4

    node_feats = torch.randn(batch_size, num_pairs, topk, num_text_nodes, dim)
    node_mask = torch.ones(batch_size, num_pairs, topk, num_text_nodes, dtype=torch.bool)

    edge_attr = torch.randn(batch_size, num_pairs, topk, num_text_nodes, num_text_nodes, dim)
    edge_mask = torch.zeros(batch_size, num_pairs, topk, num_text_nodes, num_text_nodes, dtype=torch.bool)
    edge_mask[..., 0, 1] = True
    edge_mask[..., 1, 0] = True

    raw_field_feats = torch.randn(batch_size, num_pairs, topk, num_fields, dim)
    raw_field_mask = torch.ones(batch_size, num_pairs, topk, num_fields, dtype=torch.bool)

    selected_hoi_mask = torch.ones(batch_size, num_pairs, topk, dtype=torch.bool)
    pair_valid_mask = torch.ones(batch_size, num_pairs, dtype=torch.bool)
    selected_hoi_indices = torch.arange(topk).view(1, 1, topk).expand(batch_size, num_pairs, topk)

    pair_valid_mask[0, 2] = False
    selected_hoi_mask[1, 1, 3] = False
    raw_field_mask[0, 0, 0, 4] = False
    node_mask[0, 0, 0, 2] = False

    return (
        node_feats,
        node_mask,
        edge_attr,
        edge_mask,
        raw_field_feats,
        raw_field_mask,
        selected_hoi_mask,
        pair_valid_mask,
        selected_hoi_indices,
    )


def test_text_prototype_slot_builder():
    batch_size, num_pairs, topk, dim = 2, 3, 4, 16
    (
        node_feats,
        node_mask,
        edge_attr,
        edge_mask,
        raw_field_feats,
        raw_field_mask,
        selected_hoi_mask,
        pair_valid_mask,
        selected_hoi_indices,
    ) = build_inputs()

    builder = TextPrototypeSlotBuilder(
        d_model=dim,
        fallback_to_raw_fields=True,
        sanitize=True,
        debug=True,
    )
    builder.eval()

    with torch.no_grad():
        out = builder(
            node_feats=node_feats,
            node_mask=node_mask,
            edge_attr=edge_attr,
            edge_mask=edge_mask,
            raw_field_feats=raw_field_feats,
            raw_field_mask=raw_field_mask,
            selected_hoi_mask=selected_hoi_mask,
            pair_valid_mask=pair_valid_mask,
            selected_hoi_indices=selected_hoi_indices,
        )

    assert out.prototype_slots.shape == (batch_size, num_pairs, topk, 6, dim)
    assert out.prototype_slot_mask.shape == (batch_size, num_pairs, topk, 6)
    assert not torch.isnan(out.prototype_slots).any().item()
    assert not torch.isnan(out.prototype_slot_mask.float()).any().item()

    assert out.prototype_slots[0, 2].abs().sum().item() == 0
    assert out.prototype_slot_mask[0, 2].sum().item() == 0
    assert out.prototype_slots[1, 1, 3].abs().sum().item() == 0
    assert out.prototype_slot_mask[1, 1, 3].sum().item() == 0
    assert out.prototype_slot_mask[0, 0, 0, 4].item() is False

    node_mask2 = node_mask.clone()
    raw_field_mask2 = raw_field_mask.clone()
    node_mask2[0, 0, 1, 2] = False
    raw_field_mask2[0, 0, 1, 4] = True

    with torch.no_grad():
        out2 = builder(
            node_feats=node_feats,
            node_mask=node_mask2,
            edge_attr=edge_attr,
            edge_mask=edge_mask,
            raw_field_feats=raw_field_feats,
            raw_field_mask=raw_field_mask2,
            selected_hoi_mask=selected_hoi_mask,
            pair_valid_mask=pair_valid_mask,
            selected_hoi_indices=selected_hoi_indices,
        )

    assert out2.prototype_slot_mask[0, 0, 1, 4].item() is True


class FakeBank:
    DEFAULT_FIELD_ORDER = [
        "human_node",
        "object_node",
        "ho_edge",
        "discriminative",
        "scene_context",
        "oo_context",
    ]
    field_to_index = {
        "human_node": 0,
        "object_node": 1,
        "ho_edge": 2,
        "discriminative": 3,
        "scene_context": 4,
        "oo_context": 5,
    }


class FakeRetriever:
    def __init__(self):
        self.bank = FakeBank()

    def _preview_hoi_ids(self, selected_hoi_indices, selected_hoi_mask):
        return []


def test_text_scene_graph_export_layer_outputs():
    batch_size, num_pairs, topk, dim = 2, 3, 4, 16
    (
        node_feats,
        node_mask,
        edge_attr,
        edge_mask,
        raw_field_feats,
        raw_field_mask,
        selected_hoi_mask,
        pair_valid_mask,
        selected_hoi_indices,
    ) = build_inputs()

    reasoner = TextSceneGraphReasoner(
        retriever=FakeRetriever(),
        d_model=dim,
        num_layers=0,
        num_heads=4,
        dropout=0.0,
        use_scene_token=True,
        use_oo_context_token=True,
        max_topk=topk,
        debug=True,
    )
    reasoner.eval()

    state = ExplicitTextGraphState(
        node_feats=node_feats,
        node_mask=node_mask,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        selected_hoi_indices=selected_hoi_indices,
        selected_hoi_mask=selected_hoi_mask,
        selected_scores=torch.randn(batch_size, num_pairs, topk),
        pair_valid_mask=pair_valid_mask,
        raw_field_feats=raw_field_feats,
        raw_field_mask=raw_field_mask,
        debug_info={"smoke": True},
    )

    with torch.no_grad():
        text_out = reasoner.export_layer_outputs(state)

    assert text_out.text_tokens.shape[0] == batch_size
    assert text_out.text_tokens.shape[1] == num_pairs
    assert text_out.selected_hoi_indices.shape == (batch_size, num_pairs, topk)
    assert text_out.selected_hoi_mask.shape == (batch_size, num_pairs, topk)
    assert text_out.pooled_human_node.shape == (batch_size, num_pairs, dim)
    assert text_out.pooled_object_node.shape == (batch_size, num_pairs, dim)
    assert text_out.pooled_ho_edge.shape == (batch_size, num_pairs, dim)
    assert text_out.pooled_discriminative.shape == (batch_size, num_pairs, dim)
    assert text_out.pooled_scene_context.shape == (batch_size, num_pairs, dim)
    assert text_out.pooled_oo_context.shape == (batch_size, num_pairs, dim)
    assert text_out.pooled_all.shape == (batch_size, num_pairs, dim)

    assert text_out.prototype_slots.shape == (batch_size, num_pairs, topk, 6, dim)
    assert text_out.prototype_slot_mask.shape == (batch_size, num_pairs, topk, 6)
    assert not torch.isnan(text_out.prototype_slots).any().item()
    assert "prototype_slots_shape" in text_out.debug_info
    assert "prototype_slot_valid_ratio" in text_out.debug_info

    assert text_out.prototype_slots[0, 2].abs().sum().item() == 0
    assert text_out.prototype_slot_mask[0, 2].sum().item() == 0


def main():
    test_text_prototype_slot_builder()
    test_text_scene_graph_export_layer_outputs()
    print("Smoke test passed: text prototype slots.")


if __name__ == "__main__":
    main()
