import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_text_scene_graph_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "models" / "text_scene_graph.py"
    spec = importlib.util.spec_from_file_location("text_scene_graph_direct", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TEXT_SCENE_GRAPH_MODULE = _load_text_scene_graph_module()
TextGraphFeatureBank = _TEXT_SCENE_GRAPH_MODULE.TextGraphFeatureBank
TextGraphPairPrior = _TEXT_SCENE_GRAPH_MODULE.TextGraphPairPrior


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


def _make_tiny_bank_files(tmp_path, object_to_hoi_indices):
    c, f, d = 4, 6, 8
    features = np.arange(c * f * d, dtype=np.float32).reshape(c, f, d) + 1.0
    mask = np.ones((c, f), dtype=np.int64)
    mask[1, 4] = 0
    mask[2, 5] = 0
    features[1, 4, :] = 0.0
    features[2, 5, :] = 0.0

    paths = {
        "features": tmp_path / "hico_hoi_graph_blip256_text_features_6x256.npy",
        "mask": tmp_path / "hico_hoi_graph_blip256_text_mask_6.npy",
        "hoi_ids": tmp_path / "hico_hoi_graph_blip256_hoi_ids.json",
        "object_to_hoi": tmp_path / "hico_hoi_graph_blip256_object_to_hoi_indices.json",
        "field_order": tmp_path / "hico_hoi_graph_blip256_field_order.json",
        "human_node": tmp_path / "hico_hoi_graph_blip256_human_node_features.npy",
        "oo_prior": tmp_path / "hico_hoi_conditioned_text_graph_gpt55_oo_affordance_prior.npy",
    }

    np.save(paths["features"], features)
    np.save(paths["mask"], mask)
    np.save(paths["human_node"], features[:, 0, :])
    oo_prior = np.eye(80, dtype=np.float32)
    np.save(paths["oo_prior"], oo_prior)
    _write_json(paths["hoi_ids"], ["10_0", "11_0", "12_1", "13_1"])
    _write_json(paths["object_to_hoi"], object_to_hoi_indices)
    _write_json(paths["field_order"], FIELD_ORDER)
    return paths, features, mask


def _build_bank(paths, label_ids=None):
    return TextGraphFeatureBank(
        text_features_path=str(paths["features"]),
        text_mask_path=str(paths["mask"]),
        hoi_ids_path=str(paths["hoi_ids"]),
        object_to_hoi_indices_path=str(paths["object_to_hoi"]),
        field_order_path=str(paths["field_order"]),
        d_model=8,
        normalize=True,
        strict_shape=False,
        human_node_features_path=str(paths["human_node"]),
        oo_affordance_prior_path=str(paths["oo_prior"]),
        label_ids=label_ids,
    )


def _build_pair_prior(paths, gate_init=0.0, label_ids=None, field_names=None, max_candidates=0):
    return TextGraphPairPrior(
        text_features_path=str(paths["features"]),
        text_mask_path=str(paths["mask"]),
        hoi_ids_path=str(paths["hoi_ids"]),
        object_to_hoi_indices_path=str(paths["object_to_hoi"]),
        field_order_path=str(paths["field_order"]),
        d_model=8,
        field_names=field_names,
        max_candidates=max_candidates,
        gate_init=gate_init,
        normalize=True,
        strict_shape=False,
        human_node_features_path=str(paths["human_node"]),
        oo_affordance_prior_path=str(paths["oo_prior"]),
        label_ids=label_ids,
    )


def _build_bank_with_strict_consistency(paths, human_node_features_path):
    return TextGraphFeatureBank(
        text_features_path=str(paths["features"]),
        text_mask_path=str(paths["mask"]),
        hoi_ids_path=str(paths["hoi_ids"]),
        object_to_hoi_indices_path=str(paths["object_to_hoi"]),
        field_order_path=str(paths["field_order"]),
        d_model=8,
        normalize=True,
        strict_shape=False,
        strict_consistency=True,
        human_node_features_path=str(human_node_features_path),
    )


def test_text_graph_feature_bank_with_int_candidate_lists(tmp_path):
    paths, _, _ = _make_tiny_bank_files(tmp_path, {"0": [0, 1], "1": [2, 3], "2": []})
    bank = _build_bank(paths)

    assert bank.text_features.shape == (4, 6, 8)
    assert bank.text_mask.shape == (4, 6)
    assert bank.human_node_features.shape == (4, 8)
    assert bank.object_node_features.shape == (4, 8)
    assert bank.ho_edge_features.shape == (4, 8)
    assert bank.discriminative_features.shape == (4, 8)
    assert bank.scene_context_features.shape == (4, 8)
    assert bank.oo_context_features.shape == (4, 8)
    assert bank.oo_affordance_prior.shape == (80, 80)
    assert bank.hoi_object_ids.tolist() == [0, 0, 1, 1]
    assert bank.hoi_action_ids.tolist() == [10, 11, 12, 13]
    assert bank.field_to_index == {name: idx for idx, name in enumerate(FIELD_ORDER)}
    assert bank.index_to_field == {idx: name for idx, name in enumerate(FIELD_ORDER)}
    assert bank.object_to_hoi_indices[0] == [0, 1]
    assert bank.object_to_hoi_indices[1] == [2, 3]
    assert bank.object_to_hoi_indices[2] == []

    labels = torch.tensor([[0, 1, 79, 80, -1]])
    mapped, valid = bank.map_labels_to_hico_object_ids(labels)
    assert mapped.tolist() == [[0, 1, 79, -1, -1]]
    assert valid.tolist() == [[True, True, True, False, False]]

    candidates, candidate_mask = bank.get_candidate_hoi_indices(torch.tensor([[0, 1, 2, 99]]))
    assert candidates.shape == (1, 4, 2)
    assert candidate_mask.shape == (1, 4, 2)
    assert candidates[0, 0].tolist() == [0, 1]
    assert candidates[0, 1].tolist() == [2, 3]
    assert candidate_mask[0, 2].tolist() == [False, False]
    assert candidate_mask[0, 3].tolist() == [False, False]

    field_features, field_masks = bank.gather_fields(candidates, ["human_node", "scene_context", "oo_context"])
    assert field_features.shape == (1, 4, 2, 3, 8)
    assert field_masks.shape == (1, 4, 2, 3)
    assert field_masks[0, 0, 0].tolist() == [True, True, True]
    assert field_masks[0, 0, 1].tolist() == [True, False, True]
    assert field_masks[0, 1, 0].tolist() == [True, True, False]
    assert torch.all(field_features[~field_masks] == 0)
    assert not torch.isnan(field_features).any()

    summary = bank.get_debug_summary()
    assert summary["num_hoi"] == 4
    assert summary["d_model"] == 8
    assert summary["field_order"] == FIELD_ORDER
    assert summary["object_candidate_count_min"] == 0
    assert summary["object_candidate_count_max"] == 2
    assert summary["oo_prior_loaded"] is True
    assert summary["has_nan"] is False


def test_text_graph_feature_bank_with_str_candidate_lists_and_label_mapping(tmp_path):
    paths, _, _ = _make_tiny_bank_files(tmp_path, {"0": ["10_0", "11_0"], "1": ["12_1", "13_1"]})
    bank = _build_bank(paths, label_ids=[1, 3])

    assert bank.object_to_hoi_indices[0] == [0, 1]
    assert bank.object_to_hoi_indices[1] == [2, 3]

    labels = torch.tensor([0, 1, 2, 3, 4, -1])
    mapped, valid = bank.map_labels_to_hico_object_ids(labels)
    assert mapped.tolist() == [-1, 0, -1, 1, -1, -1]
    assert valid.tolist() == [False, True, False, True, False, False]

    features, masks = bank.gather_fields(torch.tensor([0, 3, -1, 99]), ["object_node", "oo_context"])
    assert features.shape == (4, 2, 8)
    assert masks.tolist() == [[True, True], [True, True], [False, False], [False, False]]
    assert torch.all(features[2:] == 0)

    moved = bank.to(torch.device("cpu"))
    assert moved.text_features.device == moved.text_mask.device
    assert moved.hoi_object_ids.device == moved.text_features.device
    assert moved.oo_affordance_prior.device == moved.text_features.device


def test_strict_consistency_uses_raw_main_features_for_optional_fields(tmp_path):
    paths, features, _ = _make_tiny_bank_files(tmp_path, {"0": [0, 1], "1": [2, 3]})
    bank = _build_bank_with_strict_consistency(paths, paths["human_node"])
    assert bank.text_features.shape == (4, 6, 8)

    mismatched_path = tmp_path / "mismatched_human_node.npy"
    np.save(mismatched_path, features[:, 0, :] + 10.0)
    with pytest.raises(ValueError, match="differ from text_features"):
        _build_bank_with_strict_consistency(paths, mismatched_path)

    with pytest.raises(FileNotFoundError, match="Optional human_node feature file not found"):
        _build_bank_with_strict_consistency(paths, tmp_path / "missing_human_node.npy")


def test_text_graph_pair_prior_gate_zero_and_edge_cases(tmp_path):
    paths, features, _ = _make_tiny_bank_files(tmp_path, {"0": [0, 1], "1": [2], "2": []})
    features[0, 0, 0] = np.nan
    features[1, 1, 1] = np.inf
    np.save(paths["features"], features)
    prior = _build_pair_prior(paths, gate_init=0.0, field_names=["object_node", "ho_edge"], max_candidates=1)

    pair_indices = torch.tensor([[[0, 1], [0, 2], [0, 3], [-1, 1]]], dtype=torch.long)
    labels = torch.tensor([[1, 0, 2, 99]], dtype=torch.long)
    pair_mask = torch.tensor([[True, True, False, True]])
    delta, debug = prior(pair_indices, labels, pair_mask=pair_mask)

    assert delta.shape == (1, 4, 8)
    assert torch.all(delta == 0)
    assert debug["enabled"] is True
    assert debug["valid_pair_count_per_batch"] == [2]
    assert debug["valid_text_pair_count_per_batch"] == [1]
    assert debug["unknown_object_label_count"] == 1
    assert debug["candidate_count_max"] == 1
    assert debug["has_nan"] is False


def test_text_graph_pair_prior_nonzero_gate_is_finite_with_empty_masks(tmp_path):
    paths, _, _ = _make_tiny_bank_files(tmp_path, {"0": [0, 1], "1": [2, 3]})
    prior = _build_pair_prior(paths, gate_init=0.1, label_ids=[1, 3])

    empty_delta, empty_debug = prior(
        torch.zeros((1, 0, 2), dtype=torch.long),
        torch.tensor([[1, 3]], dtype=torch.long),
    )
    assert empty_delta.shape == (1, 0, 8)
    assert empty_debug["has_nan"] is False

    masked_delta, masked_debug = prior(
        torch.tensor([[[0, 1], [0, 0]]], dtype=torch.long),
        torch.tensor([[1, 3]], dtype=torch.long),
        pair_mask=torch.zeros((1, 2), dtype=torch.bool),
    )
    assert masked_delta.shape == (1, 2, 8)
    assert torch.all(masked_delta == 0)
    assert masked_debug["valid_pair_count_per_batch"] == [0]
    assert masked_debug["has_nan"] is False
