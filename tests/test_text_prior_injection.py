import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_visual_scene_graph_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "models" / "visual_scene_graph.py"
    spec = importlib.util.spec_from_file_location("visual_scene_graph_prior_direct", module_path)
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


def _make_text_files(tmp_path, oo_context_all_false=False):
    num_hoi, num_fields, dim = 5, 6, 8
    features = np.zeros((num_hoi, num_fields, dim), dtype=np.float32)
    mask = np.ones((num_hoi, num_fields), dtype=np.int64)
    for hoi_idx in range(num_hoi):
        for field_idx in range(num_fields):
            features[hoi_idx, field_idx] = _unit((hoi_idx + field_idx) % dim, dim=dim)
    if oo_context_all_false:
        oo_idx = FIELD_ORDER.index("oo_context")
        mask[:, oo_idx] = 0
        features[:, oo_idx] = 0.0

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
    oo_prior = np.zeros((80, 80), dtype=np.float32)
    oo_prior[1, 2] = 0.9
    oo_prior[2, 1] = 0.7
    np.save(paths["oo_prior"], oo_prior)
    _write_json(paths["hoi_ids"], ["10_1", "11_1", "12_1", "20_2", "21_2"])
    _write_json(paths["object_to_hoi"], {"1": [0, 1, 2], "2": [3, 4]})
    _write_json(paths["field_order"], FIELD_ORDER)
    return paths


def _text_kwargs(paths, injection="none", node_gate=0.0, edge_gate=0.0, scene_gate=0.0, oo_gate=0.0, **extra):
    kwargs = {
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
        "graph_text_gate_init": 0.0,
        "graph_text_prior_injection": injection,
        "graph_text_node_prior_gate_init": node_gate,
        "graph_text_edge_prior_gate_init": edge_gate,
        "graph_text_scene_prior_gate_init": scene_gate,
        "graph_text_oo_prior_gate_init": oo_gate,
    }
    kwargs.update(extra)
    return kwargs


def _make_inputs(unknown_label=False):
    return {
        "node_feats": torch.randn(1, 4, 8),
        "boxes": torch.tensor([[[0.05, 0.05, 0.25, 0.25], [0.3, 0.3, 0.5, 0.5], [0.55, 0.55, 0.75, 0.75], [0.8, 0.8, 0.95, 0.95]]]),
        "labels": torch.tensor([[1, 1, 2, 99 if unknown_label else 2]], dtype=torch.long),
        "node_types": torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
        "node_mask": torch.tensor([[True, True, True, True]]),
        "pair_indices": torch.tensor([[[0, 1], [0, 2], [0, 3]]], dtype=torch.long),
        "pair_scores": torch.tensor([[0.2, 0.5, 0.4]]),
        "pair_feats": torch.randn(1, 3, 8),
        "union_feats": torch.randn(1, 3, 8),
    }


def _make_reasoner(**kwargs):
    kwargs.setdefault("graph_oo_mode", "spatial")
    return VisualSceneGraphReasoner(
        d_model=8,
        num_obj_classes=80,
        num_layers=1,
        num_heads=2,
        k_oo=1,
        dropout=0.0,
        **kwargs,
    )


def test_text_prior_injection_none_outputs_finite(tmp_path):
    paths = _make_text_files(tmp_path)
    reasoner = _make_reasoner(**_text_kwargs(paths, injection="none"))
    out = reasoner(**_make_inputs())
    assert torch.isfinite(out.pair_feats).all()
    assert out.debug_info["text_prior_injection_mode"] == "none"
    assert out.debug_info["text_node_prior_injection_norm"] == 0.0
    assert out.debug_info["text_edge_prior_injection_norm"] == 0.0


def test_text_prior_edge_only_gate_zero(tmp_path):
    paths = _make_text_files(tmp_path)
    reasoner = _make_reasoner(**_text_kwargs(paths, injection="edge_only", edge_gate=0.0))
    out = reasoner(**_make_inputs())
    assert torch.isfinite(out.pair_feats).all()
    assert out.debug_info["text_prior_injection_mode"] == "edge_only"
    assert out.debug_info["text_edge_prior_injection_norm"] == 0.0


def test_text_prior_node_edge_gate_positive_changes_priors(tmp_path):
    paths = _make_text_files(tmp_path)
    reasoner = _make_reasoner(**_text_kwargs(paths, injection="node_edge", node_gate=0.1, edge_gate=0.1))
    out = reasoner(**_make_inputs())
    assert torch.isfinite(out.pair_feats).all()
    assert out.debug_info["text_node_prior_injection_norm"] > 0.0
    assert out.debug_info["text_edge_prior_injection_norm"] > 0.0
    assert out.debug_info["text_prior_has_nan"] is False


def test_text_prior_scene_oo_defaults_and_masked_oo(tmp_path):
    paths = _make_text_files(tmp_path)
    reasoner_default = _make_reasoner(**_text_kwargs(paths, injection="all", node_gate=0.1, edge_gate=0.1, scene_gate=0.1, oo_gate=0.1))
    out_default = reasoner_default(**_make_inputs())
    assert out_default.debug_info["text_scene_prior_injection_norm"] == 0.0
    assert out_default.debug_info["text_oo_prior_injection_norm"] == 0.0

    false_dir = tmp_path / "oo_false"
    false_dir.mkdir()
    false_paths = _make_text_files(false_dir, oo_context_all_false=True)
    reasoner_oo = _make_reasoner(**_text_kwargs(
        false_paths,
        injection="all",
        oo_gate=0.1,
        graph_text_use_oo_prior=True,
    ))
    out_oo = reasoner_oo(**_make_inputs())
    assert torch.isfinite(out_oo.pair_feats).all()
    assert out_oo.debug_info["text_oo_prior_injection_norm"] == 0.0
    assert out_oo.debug_info["text_prior_has_nan"] is False


def test_text_prior_oo_text_hybrid_and_unknown_label(tmp_path):
    paths = _make_text_files(tmp_path)
    reasoner = _make_reasoner(
        **_text_kwargs(paths, injection="none"),
        graph_oo_mode="text_hybrid",
        graph_oo_prior_path=str(paths["oo_prior"]),
        graph_oo_lambda_text=1.0,
        graph_oo_lambda_geo=0.0,
        graph_oo_lambda_share=0.0,
        graph_oo_prior_symmetric=False,
    )
    out = reasoner(**_make_inputs())
    assert torch.isfinite(out.pair_feats).all()
    assert out.debug_info["oo_text_used_per_batch"] == [True]
    assert out.debug_info["oo_text_mean"] > 0.0
    assert out.debug_info["oo_score_mean"] > 0.0

    unknown_out = reasoner(**_make_inputs(unknown_label=True))
    assert torch.isfinite(unknown_out.pair_feats).all()
    assert unknown_out.debug_info["text_has_nan"] is False
    assert unknown_out.debug_info["has_nan"] is False
