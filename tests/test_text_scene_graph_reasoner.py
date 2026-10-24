import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_text_scene_graph_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "models" / "text_scene_graph.py"
    spec = importlib.util.spec_from_file_location("text_scene_graph_reasoner_direct", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TEXT_SCENE_GRAPH_MODULE = _load_text_scene_graph_module()
TextGraphFeatureBank = _TEXT_SCENE_GRAPH_MODULE.TextGraphFeatureBank
TextHOIRetriever = _TEXT_SCENE_GRAPH_MODULE.TextHOIRetriever
TextSceneGraphReasoner = _TEXT_SCENE_GRAPH_MODULE.TextSceneGraphReasoner


FIELD_ORDER = [
    "human_node",
    "object_node",
    "ho_edge",
    "discriminative",
    "scene_context",
    "oo_context",
]


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _unit(index, dim=8):
    value = np.zeros((dim,), dtype=np.float32)
    value[index] = 1.0
    return value


def _make_tiny_bank(tmp_path):
    num_hoi, num_fields, dim = 5, 6, 8
    features = np.zeros((num_hoi, num_fields, dim), dtype=np.float32)
    mask = np.ones((num_hoi, num_fields), dtype=np.int64)

    for hoi_idx in range(num_hoi):
        for field_idx in range(num_fields):
            features[hoi_idx, field_idx] = _unit((hoi_idx + field_idx) % dim, dim=dim)
    ho_edge_idx = FIELD_ORDER.index("ho_edge")
    discriminative_idx = FIELD_ORDER.index("discriminative")
    scene_idx = FIELD_ORDER.index("scene_context")
    oo_idx = FIELD_ORDER.index("oo_context")
    features[0, ho_edge_idx] = _unit(0)
    features[1, ho_edge_idx] = _unit(1)
    features[2, ho_edge_idx] = _unit(2)
    features[3, ho_edge_idx] = _unit(3)
    features[4, ho_edge_idx] = _unit(4)
    features[0, discriminative_idx] = _unit(1)
    features[1, discriminative_idx] = _unit(0)
    mask[0, scene_idx] = 0
    mask[1, oo_idx] = 0
    features[0, scene_idx] = 0.0
    features[1, oo_idx] = 0.0

    paths = {
        "features": tmp_path / "features.npy",
        "mask": tmp_path / "mask.npy",
        "hoi_ids": tmp_path / "hoi_ids.json",
        "object_to_hoi": tmp_path / "object_to_hoi.json",
        "field_order": tmp_path / "field_order.json",
    }
    np.save(paths["features"], features)
    np.save(paths["mask"], mask)
    _write_json(paths["hoi_ids"], ["10_1", "11_1", "12_1", "20_2", "21_2"])
    _write_json(paths["object_to_hoi"], {"1": [0, 1, 2], "2": [3, 4]})
    _write_json(paths["field_order"], FIELD_ORDER)

    return TextGraphFeatureBank(
        text_features_path=str(paths["features"]),
        text_mask_path=str(paths["mask"]),
        hoi_ids_path=str(paths["hoi_ids"]),
        object_to_hoi_indices_path=str(paths["object_to_hoi"]),
        field_order_path=str(paths["field_order"]),
        d_model=dim,
        normalize=True,
        strict_shape=False,
    )


def _make_reasoner(bank, debug=False):
    retriever = TextHOIRetriever(
        bank=bank,
        d_model=8,
        topk=3,
        pre_topk=3,
        temperature=1.0,
        rerank_weight=0.0,
        use_discriminative_rerank=True,
        query_proj=True,
        debug=debug,
    )
    return TextSceneGraphReasoner(
        retriever=retriever,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_topk=3,
        debug=debug,
    )


def test_text_scene_graph_reasoner_shapes_masks_and_pooling(tmp_path):
    bank = _make_tiny_bank(tmp_path)
    reasoner = _make_reasoner(bank, debug=True)
    visual_pair_feats = torch.zeros((2, 4, 8), dtype=torch.float32)
    visual_pair_feats[0, 0, 0] = 1.0
    visual_pair_feats[0, 1, 4] = 1.0
    visual_pair_feats[0, 2, 1] = 1.0
    visual_pair_feats[1, 0, 3] = 1.0
    visual_pair_feats[1, 1, 0] = 1.0
    visual_pair_feats[1, 2, 2] = 1.0
    visual_pair_feats[1, 3, 4] = 1.0
    pair_object_labels = torch.tensor([[1, 2, 99, 1], [2, 1, 1, 99]], dtype=torch.long)
    pair_valid_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    out = reasoner(visual_pair_feats, pair_object_labels, pair_valid_mask=pair_valid_mask)

    assert out.text_tokens.shape == (2, 4, 18, 8)
    assert out.text_token_mask.shape == (2, 4, 18)
    assert out.selected_hoi_indices.shape == (2, 4, 3)
    assert out.selected_hoi_mask.shape == (2, 4, 3)
    assert out.pooled_ho_edge.shape == (2, 4, 8)
    assert out.pooled_discriminative.shape == (2, 4, 8)
    assert out.pooled_all.shape == (2, 4, 8)

    scene_local_idx = FIELD_ORDER.index("scene_context")
    oo_local_idx = FIELD_ORDER.index("oo_context")
    token_mask_by_field = out.text_token_mask.reshape(2, 4, 3, 6)
    selected_first_pair = out.selected_hoi_indices[0, 0]
    first_pair_scene_mask = token_mask_by_field[0, 0, :, scene_local_idx]
    first_pair_oo_mask = token_mask_by_field[0, 0, :, oo_local_idx]
    assert first_pair_scene_mask[selected_first_pair == 0].sum().item() == 0
    assert first_pair_oo_mask[selected_first_pair == 1].sum().item() == 0

    assert out.text_token_mask[0, 2].any().item() is False
    assert torch.all(out.text_tokens[0, 2] == 0)
    assert torch.all(out.pooled_all[0, 2] == 0)
    assert out.text_token_mask[0, 3].any().item() is False
    assert torch.all(out.text_tokens[0, 3] == 0)
    assert not torch.isnan(out.text_tokens).any()
    assert not torch.isnan(out.pooled_ho_edge).any()
    assert not torch.isnan(out.pooled_discriminative).any()
    assert out.debug_info["text_tokens_shape"] == (2, 4, 18, 8)
    assert out.debug_info["text_has_nan"] is False
    assert "selected_hoi_ids_preview" in out.debug_info


def test_text_scene_graph_reasoner_backward_and_bank_buffers_static(tmp_path):
    bank = _make_tiny_bank(tmp_path)
    reasoner = _make_reasoner(bank)
    visual_pair_feats = torch.zeros((2, 4, 8), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        visual_pair_feats[0, 0, 0] = 1.0
        visual_pair_feats[0, 1, 4] = 1.0
        visual_pair_feats[1, 0, 3] = 1.0
        visual_pair_feats[1, 1, 1] = 1.0
    pair_object_labels = torch.tensor([[1, 2, 99, 1], [2, 1, 1, 99]], dtype=torch.long)
    pair_valid_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    out = reasoner(visual_pair_feats, pair_object_labels, pair_valid_mask=pair_valid_mask)
    loss = out.pooled_all.sum() + out.pooled_ho_edge.sum() + out.text_tokens[out.text_token_mask].sum()
    loss.backward()

    assert reasoner.retriever.query_proj.weight.grad is not None
    assert torch.isfinite(reasoner.retriever.query_proj.weight.grad).all()
    assert reasoner.field_type_embedding.weight.grad is not None
    assert reasoner.encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert reasoner.score_embedding[0].weight.grad is not None
    assert visual_pair_feats.grad is not None
    assert torch.isfinite(visual_pair_feats.grad).all()
    assert bank.text_features.requires_grad is False
    assert bank.text_features.grad is None
    assert bank.text_mask.requires_grad is False
    assert bank.hoi_action_ids.requires_grad is False
    assert out.debug_info["text_has_nan"] is False
