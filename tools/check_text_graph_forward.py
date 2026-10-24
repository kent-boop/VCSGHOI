import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


FIELD_ORDER = [
    "human_node",
    "object_node",
    "ho_edge",
    "discriminative",
    "scene_context",
    "oo_context",
]


def _load_visual_scene_graph_reasoner():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "models" / "visual_scene_graph.py"
    spec = importlib.util.spec_from_file_location("visual_scene_graph_sanity_direct", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.VisualSceneGraphReasoner


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _unit(index, dim):
    value = np.zeros((dim,), dtype=np.float32)
    value[index] = 1.0
    return value


def _make_tiny_text_graph(tmp_dir, d_model=8):
    num_hoi = 5
    features = np.zeros((num_hoi, len(FIELD_ORDER), d_model), dtype=np.float32)
    mask = np.ones((num_hoi, len(FIELD_ORDER)), dtype=np.int64)
    for hoi_idx in range(num_hoi):
        for field_idx in range(len(FIELD_ORDER)):
            features[hoi_idx, field_idx] = _unit((hoi_idx + field_idx) % d_model, d_model)

    paths = {
        "features": tmp_dir / "features.npy",
        "mask": tmp_dir / "mask.npy",
        "hoi_ids": tmp_dir / "hoi_ids.json",
        "object_to_hoi": tmp_dir / "object_to_hoi.json",
        "field_order": tmp_dir / "field_order.json",
        "oo_prior": tmp_dir / "oo_prior.npy",
    }
    np.save(paths["features"], features)
    np.save(paths["mask"], mask)
    np.save(paths["oo_prior"], np.zeros((80, 80), dtype=np.float32))
    _write_json(paths["hoi_ids"], ["10_1", "11_1", "12_1", "20_2", "21_2"])
    _write_json(paths["object_to_hoi"], {"1": [0, 1, 2], "2": [3, 4]})
    _write_json(paths["field_order"], FIELD_ORDER)
    return paths


def main():
    torch.manual_seed(7)
    d_model = 8
    batch_size = 2
    num_nodes = 5
    num_pairs = 4
    VisualSceneGraphReasoner = _load_visual_scene_graph_reasoner()

    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_tiny_text_graph(Path(tmp), d_model=d_model)
        reasoner = VisualSceneGraphReasoner(
            d_model=d_model,
            num_obj_classes=80,
            num_layers=1,
            num_heads=2,
            k_oo=1,
            dropout=0.0,
            enable_text_graph=True,
            text_features_path=str(paths["features"]),
            text_mask_path=str(paths["mask"]),
            hoi_ids_path=str(paths["hoi_ids"]),
            object_to_hoi_indices_path=str(paths["object_to_hoi"]),
            field_order_path=str(paths["field_order"]),
            oo_affordance_prior_path=str(paths["oo_prior"]),
            graph_text_topk=3,
            graph_text_pre_topk=3,
            graph_text_temperature=1.0,
            graph_text_num_heads=2,
            graph_text_cross_num_heads=2,
            graph_text_gate_init=0.0,
            graph_text_prior_injection="none",
        )
        node_feats = torch.randn(batch_size, num_nodes, d_model)
        node_blip_feats = torch.randn(batch_size, num_nodes, d_model)
        boxes = torch.rand(batch_size, num_nodes, 4)
        boxes[..., 2:] = torch.maximum(boxes[..., :2] + 0.05, boxes[..., 2:]).clamp(max=1.0)
        labels = torch.tensor([[1, 1, 2, 2, 3], [1, 2, 1, 99, 2]], dtype=torch.long)
        node_types = torch.tensor([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)
        node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
        pair_indices = torch.tensor([[[0, 1], [0, 2], [0, 3], [0, 4]], [[0, 1], [0, 2], [0, 3], [0, 4]]], dtype=torch.long)
        pair_scores = torch.randn(batch_size, num_pairs)
        pair_feats = torch.randn(batch_size, num_pairs, d_model)
        union_feats = torch.randn(batch_size, num_pairs, d_model)
        global_feat = torch.randn(batch_size, d_model)

        out = reasoner(
            node_feats=node_feats,
            node_blip_feats=node_blip_feats,
            boxes=boxes,
            labels=labels,
            node_types=node_types,
            node_mask=node_mask,
            pair_indices=pair_indices,
            pair_scores=pair_scores,
            pair_feats=pair_feats,
            union_feats=union_feats,
            global_feat=global_feat,
        )
        info = out.debug_info
        print(f"pair_feats shape: {tuple(out.pair_feats.shape)}")
        print(f"text_graph_enabled: {info.get('text_graph_enabled')}")
        print(f"text_gate: {info.get('text_gate')}")
        print(f"text_injection_norm: {info.get('text_injection_norm')}")
        print(f"text_no_candidate_pair_count: {info.get('text_no_candidate_pair_count')}")
        print(f"has_nan: {info.get('has_nan') or info.get('text_has_nan')}")
        if bool(info.get("has_nan") or info.get("text_has_nan")):
            raise RuntimeError("Text graph sanity forward produced NaN")


if __name__ == "__main__":
    main()
