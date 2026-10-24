import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_visual_scene_graph_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "models" / "visual_scene_graph.py"
    spec = importlib.util.spec_from_file_location("visual_scene_graph_text_direct", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VisualSceneGraphReasoner = _load_visual_scene_graph_module().VisualSceneGraphReasoner


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


def _make_text_files(tmp_path, all_text_mask_false=False):
    num_hoi, num_fields, dim = 5, 6, 8
    features = np.zeros((num_hoi, num_fields, dim), dtype=np.float32)
    mask = np.ones((num_hoi, num_fields), dtype=np.int64)
    for hoi_idx in range(num_hoi):
        for field_idx in range(num_fields):
            features[hoi_idx, field_idx] = _unit((hoi_idx + field_idx) % dim, dim=dim)
    if all_text_mask_false:
        mask[:] = 0
        features[:] = 0.0

    paths = {
        "features": tmp_path / "features.npy",
        "mask": tmp_path / "mask.npy",
        "hoi_ids": tmp_path / "hoi_ids.json",
        "object_to_hoi": tmp_path / "object_to_hoi.json",
        "field_order": tmp_path / "field_order.json",
        "oo_prior": tmp_path / "oo_prior.npy",
    }
    np.save(paths["features"], features)
    np.save(paths["mask"], mask)
    np.save(paths["oo_prior"], np.zeros((80, 80), dtype=np.float32))
    _write_json(paths["hoi_ids"], ["10_1", "11_1", "12_1", "20_2", "21_2"])
    _write_json(paths["object_to_hoi"], {"1": [0, 1, 2], "2": [3, 4]})
    _write_json(paths["field_order"], FIELD_ORDER)
    return paths


def _text_kwargs(paths, gate_init=0.0, debug=False):
    return {
        "enable_text_graph": True,
        "text_features_path": str(paths["features"]),
        "text_mask_path": str(paths["mask"]),
        "hoi_ids_path": str(paths["hoi_ids"]),
        "object_to_hoi_indices_path": str(paths["object_to_hoi"]),
        "field_order_path": str(paths["field_order"]),
        "oo_affordance_prior_path": str(paths["oo_prior"]),
        "graph_text_topk": 3,
        "graph_text_pre_topk": 3,
        "graph_text_temperature": 1.0,
        "graph_text_rerank_weight": 0.0,
        "graph_text_num_layers": 1,
        "graph_text_num_heads": 2,
        "graph_text_dropout": 0.0,
        "graph_text_cross_num_heads": 2,
        "graph_text_gate_init": gate_init,
        "graph_text_debug": debug,
    }


def _make_inputs(unknown_label=False):
    node_feats = torch.randn(1, 4, 8)
    boxes = torch.tensor([[[0.05, 0.05, 0.25, 0.25], [0.3, 0.3, 0.5, 0.5], [0.55, 0.55, 0.75, 0.75], [0.8, 0.8, 0.95, 0.95]]])
    labels = torch.tensor([[1, 1, 2, 99 if unknown_label else 2]], dtype=torch.long)
    node_types = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
    node_mask = torch.tensor([[True, True, True, True]])
    pair_indices = torch.tensor([[[0, 1], [0, 2], [0, 3]]], dtype=torch.long)
    pair_scores = torch.tensor([[0.2, 0.5, 0.4]])
    pair_feats = torch.randn(1, 3, 8)
    union_feats = torch.randn(1, 3, 8)
    return {
        "node_feats": node_feats,
        "boxes": boxes,
        "labels": labels,
        "node_types": node_types,
        "node_mask": node_mask,
        "pair_indices": pair_indices,
        "pair_scores": pair_scores,
        "pair_feats": pair_feats,
        "union_feats": union_feats,
    }


def _make_reasoner(**kwargs):
    return VisualSceneGraphReasoner(
        d_model=8,
        num_obj_classes=80,
        num_layers=1,
        num_heads=2,
        k_oo=1,
        dropout=0.0,
        graph_oo_mode="spatial",
        **kwargs,
    )


def test_visual_scene_graph_text_disabled_needs_no_paths_and_runs():
    torch.manual_seed(3)
    reasoner = _make_reasoner(enable_text_graph=False)
    out = reasoner(**_make_inputs())
    assert out.pair_feats.shape == (1, 3, 8)
    assert torch.isfinite(out.pair_feats).all()
    assert out.debug_info["text_graph_enabled"] is False
    assert out.debug_info["text_injection_norm"] == 0.0


def test_visual_scene_graph_text_gate_zero_and_positive(tmp_path):
    paths = _make_text_files(tmp_path)
    torch.manual_seed(7)
    reasoner_zero = _make_reasoner(**_text_kwargs(paths, gate_init=0.0, debug=True))
    inputs = _make_inputs()
    out_zero = reasoner_zero(**inputs)
    assert out_zero.pair_feats.shape == (1, 3, 8)
    assert torch.isfinite(out_zero.pair_feats).all()
    assert out_zero.debug_info["text_graph_enabled"] is True
    assert out_zero.debug_info["text_tokens_shape"] == (1, 3, 18, 8)
    assert abs(out_zero.debug_info["text_injection_norm"]) <= 1e-8
    assert "text_selected_hoi_ids_preview" in out_zero.debug_info

    torch.manual_seed(7)
    reasoner_positive = _make_reasoner(**_text_kwargs(paths, gate_init=0.1))
    out_positive = reasoner_positive(**inputs)
    assert torch.isfinite(out_positive.pair_feats).all()
    assert out_positive.debug_info["text_injection_norm"] > 0.0
    assert out_positive.debug_info["text_has_nan"] is False


def test_visual_scene_graph_text_unknown_label_and_all_false_text_mask(tmp_path):
    paths = _make_text_files(tmp_path)
    reasoner = _make_reasoner(**_text_kwargs(paths, gate_init=0.1))
    out = reasoner(**_make_inputs(unknown_label=True))
    assert torch.isfinite(out.pair_feats).all()
    assert out.debug_info["text_no_candidate_pair_count"] >= 1
    assert out.debug_info["text_has_nan"] is False

    false_dir = tmp_path / "false_mask"
    false_dir.mkdir()
    false_paths = _make_text_files(false_dir, all_text_mask_false=True)
    reasoner_false = _make_reasoner(**_text_kwargs(false_paths, gate_init=0.1))
    out_false = reasoner_false(**_make_inputs())
    assert torch.isfinite(out_false.pair_feats).all()
    assert out_false.debug_info["text_valid_token_count_mean"] == 0.0
    assert out_false.debug_info["text_injection_norm"] == 0.0
    assert out_false.debug_info["text_has_nan"] is False


def test_visual_scene_graph_text_missing_path_raises(tmp_path):
    paths = _make_text_files(tmp_path)
    paths["features"] = tmp_path / "missing.npy"
    try:
        _make_reasoner(**_text_kwargs(paths, gate_init=0.0))
    except FileNotFoundError as exc:
        assert "text_features file not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing text_features path")


def test_visual_scene_graph_text_backward(tmp_path):
    paths = _make_text_files(tmp_path)
    reasoner = _make_reasoner(**_text_kwargs(paths, gate_init=0.1))
    inputs = _make_inputs()
    inputs["node_feats"].requires_grad_(True)
    inputs["pair_feats"].requires_grad_(True)
    inputs["union_feats"].requires_grad_(True)

    out = reasoner(**inputs)
    loss = out.pair_feats.sum()
    loss.backward()

    assert reasoner.text_cross_attn.in_proj_weight.grad is not None
    assert torch.isfinite(reasoner.text_cross_attn.in_proj_weight.grad).all()
    assert reasoner.text_graph_reasoner.field_type_embedding.weight.grad is not None
    assert reasoner.text_graph_reasoner.encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert reasoner.text_graph_reasoner.retriever.query_proj.weight.grad is not None
    bank = reasoner.text_graph_reasoner.retriever.bank
    assert bank.text_features.requires_grad is False
    assert bank.text_features.grad is None
    assert bank.text_mask.requires_grad is False
    assert inputs["pair_feats"].grad is not None
    assert inputs["union_feats"].grad is not None
