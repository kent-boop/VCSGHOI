import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_text_scene_graph_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "models" / "text_scene_graph.py"
    spec = importlib.util.spec_from_file_location("text_scene_graph_retriever_direct", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TEXT_SCENE_GRAPH_MODULE = _load_text_scene_graph_module()
TextGraphFeatureBank = _TEXT_SCENE_GRAPH_MODULE.TextGraphFeatureBank
TextHOIRetriever = _TEXT_SCENE_GRAPH_MODULE.TextHOIRetriever


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


def _unit(index, dim=4):
    value = np.zeros((dim,), dtype=np.float32)
    value[index] = 1.0
    return value


def _make_tiny_bank(tmp_path):
    num_hoi, num_fields, dim = 5, 6, 4
    features = np.zeros((num_hoi, num_fields, dim), dtype=np.float32)
    mask = np.ones((num_hoi, num_fields), dtype=np.int64)
    ho_edge_idx = FIELD_ORDER.index("ho_edge")
    discriminative_idx = FIELD_ORDER.index("discriminative")

    features[0, ho_edge_idx] = _unit(0)
    features[1, ho_edge_idx] = _unit(1)
    features[2, ho_edge_idx] = _unit(2)
    features[3, ho_edge_idx] = _unit(3)
    features[4, ho_edge_idx] = _unit(0)

    features[0, discriminative_idx] = _unit(1)
    features[1, discriminative_idx] = _unit(0)
    features[2, discriminative_idx] = _unit(2)
    features[3, discriminative_idx] = _unit(3)
    features[4, discriminative_idx] = _unit(0)

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

    bank = TextGraphFeatureBank(
        text_features_path=str(paths["features"]),
        text_mask_path=str(paths["mask"]),
        hoi_ids_path=str(paths["hoi_ids"]),
        object_to_hoi_indices_path=str(paths["object_to_hoi"]),
        field_order_path=str(paths["field_order"]),
        d_model=dim,
        normalize=True,
        strict_shape=False,
    )
    return bank


def test_text_hoi_retriever_object_conditioned_topk_and_padding(tmp_path):
    bank = _make_tiny_bank(tmp_path)
    retriever = TextHOIRetriever(
        bank=bank,
        d_model=4,
        topk=4,
        pre_topk=4,
        temperature=1.0,
        use_discriminative_rerank=False,
        query_proj=False,
        debug=True,
    )
    visual_pair_feats = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    pair_object_labels = torch.tensor([[1, 2]], dtype=torch.long)

    out = retriever(visual_pair_feats, pair_object_labels)

    assert out.selected_hoi_indices.shape == (1, 2, 4)
    assert out.selected_hoi_mask.tolist() == [[[True, True, True, False], [True, True, False, False]]]
    assert set(out.selected_hoi_indices[0, 0, out.selected_hoi_mask[0, 0]].tolist()).issubset({0, 1, 2})
    assert set(out.selected_hoi_indices[0, 1, out.selected_hoi_mask[0, 1]].tolist()).issubset({3, 4})
    assert out.selected_hoi_indices[0, 0, 0].item() == 1
    assert out.selected_hoi_indices[0, 1, 0].item() == 4
    assert out.selected_hoi_indices[0, 1, 2:].tolist() == [0, 0]
    assert torch.all(out.selected_scores[~out.selected_hoi_mask] == 0)
    assert not torch.isnan(out.selected_scores).any()
    assert out.debug_info["object_valid_pair_count"] == 2
    assert out.debug_info["no_candidate_pair_count"] == 0
    assert "selected_hoi_ids_preview" in out.debug_info


def test_text_hoi_retriever_discriminative_rerank_changes_order(tmp_path):
    bank = _make_tiny_bank(tmp_path)
    visual_pair_feats = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    pair_object_labels = torch.tensor([[1]], dtype=torch.long)

    coarse_only = TextHOIRetriever(
        bank=bank,
        d_model=4,
        topk=2,
        pre_topk=3,
        temperature=1.0,
        use_discriminative_rerank=False,
        query_proj=False,
    )
    reranked = TextHOIRetriever(
        bank=bank,
        d_model=4,
        topk=2,
        pre_topk=3,
        temperature=1.0,
        rerank_weight=2.0,
        use_discriminative_rerank=True,
        query_proj=False,
    )

    coarse_out = coarse_only(visual_pair_feats, pair_object_labels)
    rerank_out = reranked(visual_pair_feats, pair_object_labels)

    assert coarse_out.selected_hoi_indices[0, 0, 0].item() == 0
    assert rerank_out.selected_hoi_indices[0, 0, 0].item() == 1
    assert rerank_out.coarse_scores is not None
    assert rerank_out.rerank_scores is not None
    assert torch.isfinite(rerank_out.selected_scores[rerank_out.selected_hoi_mask]).all()


def test_text_hoi_retriever_unknown_and_pair_mask_return_empty(tmp_path):
    bank = _make_tiny_bank(tmp_path)
    retriever = TextHOIRetriever(bank=bank, d_model=4, topk=3, pre_topk=3, query_proj=False)
    visual_pair_feats = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    pair_object_labels = torch.tensor([[99, 1]], dtype=torch.long)
    pair_valid_mask = torch.tensor([[True, False]])

    out = retriever(visual_pair_feats, pair_object_labels, pair_valid_mask=pair_valid_mask)

    assert out.selected_hoi_mask.tolist() == [[[False, False, False], [False, False, False]]]
    assert torch.all(out.selected_scores == 0)
    assert out.pair_object_ids.tolist() == [[-1, 1]]
    assert out.pair_object_valid_mask.tolist() == [[False, True]]
    assert out.debug_info["object_valid_pair_count"] == 0
    assert out.debug_info["selected_mask_ratio"] == 0.0
    assert out.debug_info["has_nan"] is False


def test_text_hoi_retriever_backward_only_query_proj_has_grad(tmp_path):
    bank = _make_tiny_bank(tmp_path)
    retriever = TextHOIRetriever(
        bank=bank,
        d_model=4,
        topk=2,
        pre_topk=3,
        temperature=1.0,
        use_discriminative_rerank=True,
        query_proj=True,
    )
    visual_pair_feats = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32, requires_grad=True)
    pair_object_labels = torch.tensor([[1]], dtype=torch.long)

    out = retriever(visual_pair_feats, pair_object_labels)
    loss = out.selected_scores[out.selected_hoi_mask].sum()
    loss.backward()

    assert retriever.query_proj.weight.grad is not None
    assert torch.isfinite(retriever.query_proj.weight.grad).all()
    assert visual_pair_feats.grad is not None
    assert bank.text_features.requires_grad is False
    assert bank.text_features.grad is None
    assert bank.text_mask.requires_grad is False
    assert bank.hoi_action_ids.requires_grad is False
    assert not torch.isnan(out.selected_scores).any()
