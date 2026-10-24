import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_existing_helpers():
    path = REPO_ROOT / "tests" / "smoke_synchronized_evidence_matching.py"
    spec = importlib.util.spec_from_file_location("sync_smoke_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_existing_helpers()
from src.models.visual_scene_graph_oov8 import VisualSceneGraphReasoner


VARIANTS = (
    "full",
    "no_anchor_writeback",
    "no_prototype_state_writeback",
    "no_setm",
    "endpoint_only_setm",
)


def build_model(assets, variant="full", trace=True):
    torch.manual_seed(20260801)
    model = VisualSceneGraphReasoner(
        d_model=32,
        num_obj_classes=80,
        num_layers=2,
        num_heads=4,
        k_oo=2,
        enable_oo_edges=True,
        enable_hh_edges=False,
        enable_text_graph=True,
        text_features_path=assets["text_features_path"],
        text_mask_path=assets["text_mask_path"],
        hoi_ids_path=assets["hoi_ids_path"],
        object_to_hoi_indices_path=assets["object_to_hoi_indices_path"],
        field_order_path=assets["field_order_path"],
        graph_text_topk=5,
        graph_text_pre_topk=6,
        graph_text_num_layers=2,
        graph_text_num_heads=4,
        graph_text_cross_num_heads=4,
        graph_text_sync_mode="layerwise",
        graph_text_sync_query_source="base_pair",
        graph_text_dynamic_retrieval=False,
        graph_text_dense_injection=True,
        graph_text_prior_injection="all",
        graph_text_use_human_node_prior=True,
        graph_text_use_object_node_prior=True,
        graph_text_use_ho_edge_prior=True,
        graph_text_use_discriminative_prior=True,
        graph_text_use_scene_prior=True,
        graph_text_use_oo_prior=True,
        graph_enable_candidate_evidence=True,
        graph_evidence_topk=4,
        graph_evidence_gate_init=0.2,
        graph_enable_sync_evidence_matching=True,
        graph_sync_visual_gate_init=0.3,
        graph_sync_text_gate_init=0.4,
        graph_sync_anchor_gate_init=0.5,
        graph_enable_layerwise_sync_evidence_matching=True,
        graph_layerwise_sync_write_visual=True,
        graph_layerwise_sync_write_text=True,
        graph_layerwise_sync_final_matching=False,
        graph_layerwise_sync_disable_old_dense_injection=True,
        graph_layerwise_sync_write_reverse_edge=False,
        graph_layerwise_sync_detach_text_update=False,
        graph_export_evidence_trace=trace,
        graph_export_trace_each_layer=trace,
        graph_export_trace_to_cpu=True,
        graph_export_trace_float_dtype="float32",
        graph_export_trace_include_match_logits=trace,
        graph_export_trace_include_features=trace,
        graph_export_mechanism_validation_trace=trace,
        graph_mechanism_validation_variant=variant,
        dropout=0.0,
    )
    model.eval()
    return model


def run_model(model):
    with torch.no_grad():
        return model(**helpers.build_visual_inputs())


def assert_tensor_equal(left, right, name):
    assert left.shape == right.shape, (name, left.shape, right.shape)
    assert torch.equal(left, right), name


def assert_all_finite(value, name):
    assert torch.isfinite(value).all().item(), name


def test_default_path_equivalence(assets):
    no_trace = run_model(build_model(assets, variant="full", trace=False))
    trace = run_model(build_model(assets, variant="full", trace=True))
    assert_tensor_equal(no_trace.pair_feats, trace.pair_feats, "trace changed normal pair output")
    assert_tensor_equal(no_trace.node_feats, trace.node_feats, "trace changed normal node output")


def test_variants(assets):
    outputs = {variant: run_model(build_model(assets, variant=variant, trace=True)) for variant in VARIANTS}

    full = outputs["full"]
    assert full.debug_info["setm_call_count"] == 2
    assert len(full.debug_info["mechanism_layer_pair_features"]) == 2
    assert torch.equal(full.debug_info["mechanism_layer_pair_features"][-1], full.pair_feats)

    baseline_retrieval = None
    for variant, output in outputs.items():
        layers = output.debug_info["evidence_trace_layers"]
        assert len(layers) == 2, variant
        assert len(output.debug_info["mechanism_layer_pair_features"]) == 2, variant
        assert_all_finite(output.pair_feats, f"{variant}:pair_feats")

        retrieval = [layer["selected_hoi_indices"] for layer in layers]
        if baseline_retrieval is None:
            baseline_retrieval = retrieval
        else:
            for expected, actual in zip(baseline_retrieval, retrieval):
                assert_tensor_equal(expected, actual, f"{variant}:retrieval changed")

        for layer in layers:
            assert layer["candidate_evidence_tokens"].shape[-2] == 7
            assert layer["prototype_slots_pre_sync"].shape[-2] == 6
            assert layer["graph_pair_features_after_layer"].shape[-1] == 32
            assert_all_finite(layer["graph_pair_features_after_layer"], f"{variant}:layer pair")

    no_anchor = outputs["no_anchor_writeback"]
    assert no_anchor.debug_info["setm_call_count"] == 2
    for layer in no_anchor.debug_info["evidence_trace_layers"]:
        assert layer["setm_called"] is True
        assert layer["visual_writeback_applied"] is False
        assert layer["text_writeback_applied"] is True
        assert torch.count_nonzero(layer["sync_anchor_delta_applied"]).item() == 0
        assert_tensor_equal(
            layer["pair_anchor_pre_sync"],
            layer["pair_anchor_post_sync_actual"],
            "no_anchor_writeback changed actual anchor",
        )

    no_text = outputs["no_prototype_state_writeback"]
    assert no_text.debug_info["setm_call_count"] == 2
    for layer in no_text.debug_info["evidence_trace_layers"]:
        assert layer["setm_called"] is True
        assert layer["visual_writeback_applied"] is True
        assert layer["text_writeback_applied"] is False
        assert_tensor_equal(
            layer["prototype_slots_pre_sync"],
            layer["prototype_slots_post_writeback_actual"],
            "no_prototype_state_writeback changed actual text state",
        )

    no_setm = outputs["no_setm"]
    assert no_setm.debug_info["setm_call_count"] == 0
    assert no_setm.debug_info["sync_evidence_skip_reason"] == "mechanism_variant_no_setm"
    for layer in no_setm.debug_info["evidence_trace_layers"]:
        assert layer["setm_called"] is False
        assert layer["setm_correction_count"] == 0
        assert layer["visual_writeback_applied"] is False
        assert layer["text_writeback_applied"] is False
        assert "sync_match_logits" not in layer
        assert torch.count_nonzero(layer["sync_anchor_delta_applied"]).item() == 0
        assert_tensor_equal(
            layer["pair_anchor_pre_sync"],
            layer["pair_anchor_post_sync_actual"],
            "no_setm changed actual anchor",
        )

    endpoint = outputs["endpoint_only_setm"]
    assert endpoint.debug_info["setm_call_count"] == 2
    for layer in endpoint.debug_info["evidence_trace_layers"]:
        assert layer["setm_called"] is True
        assert layer["requested_masked_visual_indices"] == [3, 4, 5, 6]
        assert torch.count_nonzero(layer["sync_visual_mask"][..., 3:]).item() == 0
        assert_tensor_equal(
            layer["candidate_evidence_mask"][..., :3],
            layer["sync_visual_mask"][..., :3],
            "endpoint_only changed endpoint mask",
        )


def main():
    np.random.seed(20260801)
    with tempfile.TemporaryDirectory() as tmpdir:
        assets = helpers.make_text_assets(tmpdir, d_model=32, num_hoi=6)
        test_default_path_equivalence(assets)
        test_variants(assets)
    print("mechanism_validation_trace: all tests passed")


if __name__ == "__main__":
    main()
