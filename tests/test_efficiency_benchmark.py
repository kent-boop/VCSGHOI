from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tools.benchmark_utils import (
    BenchmarkError,
    CheckpointError,
    compare_environments,
    count_parameters,
    dynamic_workload_rows,
    load_and_validate_manifest,
    load_checkpoint_strict,
    prepare_official_batch,
    select_stratified_sample_ids,
    validate_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "efficiency_variants.yaml"


def test_manifest_contains_required_paper_variants():
    manifest = load_and_validate_manifest(MANIFEST)
    required = {
        "baseline_r50_vitb",
        "vet_r50_vitb",
        "kpt_r50_vitb",
        "cross_attention_r50_vitb",
        "evidencehoi_r50_vitb",
    }
    assert required.issubset(manifest["variants"])
    assert manifest["baseline_variant"] == "baseline_r50_vitb"
    assert all(item["model_file"] == "src/models/stip.py" for item in manifest["variants"].values())


def test_full_variant_encodes_final_invariants():
    variant = load_and_validate_manifest(MANIFEST)["variants"]["evidencehoi_r50_vitb"]
    flags = variant["config_args"]
    assert variant["paper_components"] == {"VET": True, "KPT": True, "SETM": True}
    assert variant["expected"] == {"d": 256, "L": 2, "k_oo": 3, "M": 4, "K": 1, "coarse_pool": 20}
    assert flags["graph_enable_candidate_evidence"] is True
    assert flags["graph_enable_sync_evidence_matching"] is True
    assert flags["graph_enable_layerwise_sync_evidence_matching"] is True
    assert flags["graph_text_dynamic_retrieval"] is False
    assert flags["graph_layerwise_sync_final_matching"] is False


def test_large_variant_encodes_checkpoint_invariants():
    variant = load_and_validate_manifest(MANIFEST)["variants"]["evidencehoi_r101_vitl"]
    flags = variant["config_args"]
    assert variant["backbone"] == "resnet101"
    assert variant["visual_encoder"] == "vit_large"
    assert variant["expected"] == {"d": 256, "L": 2, "k_oo": 3, "M": 1, "K": 3, "coarse_pool": 20}
    assert flags["blip_vit"] == "large"
    assert flags["graph_evidence_topk"] == 1
    assert flags["graph_text_topk"] == 3
    assert flags["graph_text_pre_topk"] == 20
    assert flags["graph_text_num_layers"] is None


def test_unresolved_table_rows_are_blocked_for_formal_use():
    variants = load_and_validate_manifest(MANIFEST)["variants"]
    assert variants["kpt_r50_vitb"]["paper_configuration_verified"] is False
    assert variants["cross_attention_r50_vitb"]["paper_configuration_verified"] is False
    assert variants["baseline_r101_vitl"]["enabled"] is False
    assert variants["evidencehoi_r101_vitl"]["enabled"] is True
    assert variants["evidencehoi_r101_vitl"]["paper_configuration_verified"] is True


class _TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared = torch.nn.Parameter(torch.ones(2, 3))
        self.shared = shared
        self.alias = shared
        self.frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        self.register_buffer("not_a_parameter", torch.ones(5))


def test_parameter_counter_deduplicates_and_separates_frozen():
    summary, rows = count_parameters(_TiedModel())
    assert summary["total_params"] == 10
    assert summary["trainable_params"] == 6
    assert summary["frozen_params"] == 4
    assert summary["buffers"] == 5
    assert sum(row["parameters"] for row in rows) == 10


def test_strict_checkpoint_load_accepts_exact_and_rejects_mismatch(tmp_path):
    model = torch.nn.Linear(3, 2)
    exact = tmp_path / "exact.pth"
    torch.save({"model": model.state_dict(), "epoch": 7}, exact)
    report = load_checkpoint_strict(model, exact)
    assert report["strict"] is True
    assert report["epoch"] == 7
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []

    bad = tmp_path / "bad.pth"
    torch.save({"model": {"weight": model.weight.detach().clone()}}, bad)
    with pytest.raises(CheckpointError, match="Strict checkpoint validation failed"):
        load_checkpoint_strict(model, bad)


def test_prepare_batch_preserves_official_forward_order():
    from src.util.misc import NestedTensor

    samples = NestedTensor(torch.randn(1, 3, 4, 5), torch.zeros(1, 4, 5, dtype=torch.bool))
    targets = ({"orig_size": torch.tensor([4, 5]), "image_id": torch.tensor([9])},)
    images = (object(),)
    prepared = prepare_official_batch((samples, targets, images), torch.device("cpu"))
    assert prepared[0].tensors.device.type == "cpu"
    assert prepared[1] is images
    assert prepared[2][0]["image_id"].item() == 9


def _valid_output(value=0.0):
    tensor = torch.tensor([value], dtype=torch.float32)
    return {
        "pred_logits": tensor,
        "pred_boxes": tensor,
        "pred_rel_pairs": [torch.zeros(1, 2, dtype=torch.long)],
        "pred_actions": [tensor],
        "pred_action_exists": [tensor],
    }


def test_output_validation_rejects_nonfinite_and_missing_keys():
    validate_outputs(_valid_output())
    with pytest.raises(BenchmarkError, match="Non-finite"):
        validate_outputs(_valid_output(float("nan")))
    invalid = _valid_output()
    invalid.pop("pred_boxes")
    with pytest.raises(BenchmarkError, match="missing keys"):
        validate_outputs(invalid)


def test_environment_mismatch_is_rejected_unless_explicitly_allowed():
    environment = {
        "pytorch": "1",
        "torchvision": "1",
        "cuda_runtime": "1",
        "cudnn": 1,
        "gpu": {"name": "GPU-A", "driver_version": "1"},
        "precision": "fp32",
        "backend": {"cudnn_benchmark": False},
    }
    changed = deepcopy(environment)
    changed["gpu"]["name"] = "GPU-B"
    with pytest.raises(BenchmarkError, match="environments differ"):
        compare_environments([environment, changed])
    compare_environments([environment, changed], allow_mixed=True)


class _MetadataOnlyDataset:
    def __init__(self):
        self.ids = list(range(30))
        self.annotations = [
            {"annotations": [None] * (index % 10), "hoi_annotation": [None] * (index % 5)}
            for index in self.ids
        ]

    def __len__(self):
        return len(self.ids)


def test_fixed_subset_selection_is_deterministic_and_unique():
    dataset = _MetadataOnlyDataset()
    first = select_stratified_sample_ids(dataset, 12, seed=3407)
    second = select_stratified_sample_ids(dataset, 12, seed=3407)
    assert first == second
    assert len(first) == len(set(first)) == 12


def test_dynamic_workload_uses_layerwise_setm_shapes():
    outputs = {
        "graph_input_debug": [{"detected_object_count": 12, "proposal_count": 32}],
        "graph_reasoner_debug": [{
            "valid_pair_count_per_batch": [32],
            "text_layer_count": 0,
            "layerwise_sync_layer_count": 2,
            "layerwise_sync_debug": [
                {
                    "candidate_evidence_tokens_shape": [1, 32, 7, 256],
                    "prototype_slots_shape": [1, 32, 1, 6, 256],
                    "sync_match_logits_shape": [1, 32, 1, 7, 6],
                },
                {
                    "candidate_evidence_tokens_shape": [1, 32, 7, 256],
                    "prototype_slots_shape": [1, 32, 1, 6, 256],
                    "sync_match_logits_shape": [1, 32, 1, 7, 6],
                },
            ],
        }],
    }
    row = dynamic_workload_rows(outputs, "evidencehoi_r50_vitb", [7])[0]
    assert row["selected_paths_M"] == 4
    assert row["retrieved_prototypes_K"] == 1
    assert row["kpt_reasoning_layers"] == 2
    assert row["setm_reasoning_layers"] == 2
