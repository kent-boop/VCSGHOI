import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_visual_scene_graph_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "models" / "visual_scene_graph_oov0.py"
    spec = importlib.util.spec_from_file_location("visual_scene_graph_oov0_direct", module_path)
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


def _make_tiny_text_graph_files(tmp_path):
    features = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8) + 1.0
    features[0, 0, 0] = np.nan
    mask = np.ones((4, 6), dtype=np.int64)
    mask[2, 5] = 0
    paths = {
        "features": tmp_path / "features.npy",
        "mask": tmp_path / "mask.npy",
        "hoi_ids": tmp_path / "hoi_ids.json",
        "object_to_hoi": tmp_path / "object_to_hoi.json",
        "field_order": tmp_path / "field_order.json",
    }
    np.save(paths["features"], features)
    np.save(paths["mask"], mask)
    _write_json(paths["hoi_ids"], ["10_0", "11_0", "12_1", "13_1"])
    _write_json(paths["object_to_hoi"], {"0": [0, 1], "1": [2], "2": []})
    _write_json(paths["field_order"], FIELD_ORDER)
    return paths


def _graph_inputs():
    return {
        "node_feats": torch.randn(1, 4, 8),
        "boxes": torch.tensor([[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.5, 0.5], [0.5, 0.5, 0.7, 0.7], [0.8, 0.8, 0.9, 0.9]]]),
        "labels": torch.tensor([[1, 0, 1, 99]], dtype=torch.long),
        "node_types": torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
        "node_mask": torch.tensor([[True, True, True, True]]),
        "pair_indices": torch.tensor([[[0, 1], [0, 2], [0, 3]]], dtype=torch.long),
        "pair_scores": torch.tensor([[0.2, 0.3, 0.4]]),
        "pair_feats": torch.randn(1, 3, 8),
        "union_feats": torch.randn(1, 3, 8),
    }


def test_visual_scene_graph_text_prior_default_disabled():
    reasoner = VisualSceneGraphReasoner(d_model=8, num_heads=2, num_layers=1, dropout=0.0)
    assert reasoner.graph_enable_text_graph is False
    assert reasoner.text_graph_prior is None

    out = reasoner(**_graph_inputs())
    assert out.debug_info["graph_enable_text_graph"] is False
    assert out.debug_info["text_graph_debug"]["applied"] is False
    assert torch.isfinite(out.pair_feats).all()


def test_visual_scene_graph_text_prior_gate_zero_keeps_output(tmp_path):
    paths = _make_tiny_text_graph_files(tmp_path)
    torch.manual_seed(7)
    base = VisualSceneGraphReasoner(d_model=8, num_heads=2, num_layers=1, dropout=0.0)
    torch.manual_seed(7)
    with_text = VisualSceneGraphReasoner(
        d_model=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        graph_enable_text_graph=True,
        graph_text_features_path=str(paths["features"]),
        graph_text_mask_path=str(paths["mask"]),
        graph_text_hoi_ids_path=str(paths["hoi_ids"]),
        graph_text_object_to_hoi_indices_path=str(paths["object_to_hoi"]),
        graph_text_field_order_path=str(paths["field_order"]),
        graph_text_strict_shape=False,
        graph_text_gate_init=0.0,
        graph_text_max_candidates=1,
    )
    base.eval()
    with_text.eval()
    torch.manual_seed(11)
    inputs = _graph_inputs()

    out_base = base(**inputs)
    out_text = with_text(**inputs)

    assert torch.allclose(out_base.pair_feats, out_text.pair_feats, atol=1e-6)
    assert out_text.debug_info["graph_enable_text_graph"] is True
    assert out_text.debug_info["text_graph_debug"]["applied"] is True
    assert out_text.debug_info["text_graph_debug"]["has_nan"] is False

    all_false_inputs = dict(inputs)
    all_false_inputs["node_mask"] = torch.zeros_like(inputs["node_mask"])
    all_false = with_text(**all_false_inputs)
    assert torch.isfinite(all_false.pair_feats).all()
