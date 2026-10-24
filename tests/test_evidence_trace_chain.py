from __future__ import annotations

import ast
import copy
import importlib.util
import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_METHOD_NAMES = {
    "_trace_tensor",
    "_trace_python_value",
    "_trace_image_metadata",
    "_squeeze_trace_batch",
    "_build_pair_gt_targets",
    "_build_evidence_trace_record",
}


def load_file_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trace_helper_class():
    source_path = PROJECT_ROOT / "src/models/stip_evidence_trace.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    stip_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "STIP"
    )
    methods = [
        copy.deepcopy(node) for node in stip_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in TRACE_METHOD_NAMES
    ]
    assert {method.name for method in methods} == TRACE_METHOD_NAMES
    helper_class = ast.ClassDef(
        name="TraceSTIP",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    helper_module = ast.Module(body=[helper_class], type_ignores=[])
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "F": F,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "np": np,
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(helper_module), str(source_path), "exec"), namespace)
    trace_class = namespace["TraceSTIP"]
    namespace["STIP"] = trace_class
    return trace_class


trace_io = load_file_module("evidence_trace_io_test", "src/models/evidence_trace_io.py")
validator = load_file_module(
    "validate_evidence_trace_test",
    "tools/evidence_vis/validate_evidence_trace.py",
)
arg_parser_module = load_file_module("arg_parser_test", "src/engine/arg_parser.py")
ranker = load_file_module("rank_evidence_examples_test", "tools/evidence_vis/rank_evidence_examples.py")

EvidenceTraceWriter = trace_io.EvidenceTraceWriter
build_run_metadata = trace_io.build_run_metadata
merge_manifests = trace_io.merge_manifests
validate_trace_directory = validator.validate_trace_directory
get_args_parser = arg_parser_module.get_args_parser
TraceSTIP = load_trace_helper_class()


@contextmanager
def assert_raises(exception_type, pattern: str):
    try:
        yield
    except exception_type as exc:
        assert re.search(pattern, str(exc)), str(exc)
    else:
        raise AssertionError(f"Expected {exception_type.__name__}")


def make_trace_model():
    model = TraceSTIP()
    model.graph_export_trace_float_dtype = "float16"
    model.graph_export_trace_topk_actions = 3
    model.graph_export_trace_include_all_action_logits = True
    model.args = SimpleNamespace(
        dataset_file="hico-det",
        valid_obj_ids=[0, 1],
        valid_ids=[0, 1, 2, 3],
        pair_filter_threshold=0.0,
        use_pair_score_weighting=False,
    )
    return model


def make_graph_output() -> SimpleNamespace:
    graph = {
        "selected_context_indices": torch.tensor([[[2, 3], [0, 1]]]),
        "selected_context_mask": torch.tensor([[[True, True], [True, False]]]),
        "selected_context_two_hop_mask": torch.tensor([[[True, True], [True, False]]]),
        "candidate_evidence_mask": torch.ones(1, 2, 5, dtype=torch.bool),
        "candidate_evidence_weights": torch.full((1, 2, 5), 0.2),
        "pair_valid_mask": torch.tensor([[True, True]]),
        "selected_hoi_action_ids": torch.tensor([[[1, 2, 3], [0, 2, 3]]]),
        "selected_hoi_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "sync_text_to_visual_attn": torch.full((1, 2, 3, 6, 5), 0.2),
        "sync_evidence_weights": torch.full((1, 2, 3, 5), 0.2),
        "sync_prototype_coverage": torch.ones(1, 2, 3, 6),
        "prototype_slot_mask": torch.ones(1, 2, 3, 6, dtype=torch.bool),
    }
    return SimpleNamespace(debug_info={"evidence_trace": graph})


def make_record(image_id: int = 101) -> dict:
    model = make_trace_model()
    target = {
        "image_id": torch.tensor(image_id),
        "orig_size": torch.tensor([480, 640]),
        "file_name": f"HICO_test2015_{image_id:08d}.jpg",
        "hois": torch.tensor([[0, 1, 1], [0, 1, 2]]),
    }
    return model._build_evidence_trace_record(
        imgid=0,
        target=target,
        graph_out=make_graph_output(),
        sampled_rel_pairs=torch.tensor([[0, 1], [2, 3]]),
        sampled_rel_pred_exists=torch.tensor([2.0, -1.0]),
        sampled_actions=torch.tensor(
            [[-1.0, 1.0, 0.5, -0.5], [0.2, -0.3, 0.4, 0.1]]
        ),
        final_action_logits=torch.tensor(
            [[-0.5, 2.0, 1.5, -1.0], [0.1, -0.2, 0.3, 0.0]]
        ),
        node_boxes_xyxy=torch.tensor(
            [
                [0.05, 0.10, 0.30, 0.90],
                [0.35, 0.25, 0.60, 0.75],
                [0.62, 0.12, 0.82, 0.88],
                [0.70, 0.30, 0.95, 0.70],
            ]
        ),
        node_class_logits=torch.tensor(
            [
                [3.0, 0.2, -1.0, 0.0],
                [0.1, 2.5, -0.5, 0.0],
                [2.0, 0.1, -0.8, 0.0],
                [0.2, 2.0, -0.4, 0.0],
            ]
        ),
        node_scores=torch.tensor([0.95, 0.85, 0.90, 0.80]),
        node_labels=torch.tensor([0, 1, 0, 1]),
        node_types=torch.tensor([0, 1, 0, 1]),
        node_mask=torch.tensor([True, True, True, True]),
        trace_det2gt=(torch.tensor([0, 1]), torch.tensor([0, 1])),
    )


def test_trace_record_has_multilabel_gt_and_unmatched_pair() -> None:
    record = make_record()
    targets = record["pairs"]["gt_action_targets"]
    matched = record["pairs"]["gt_pair_matched"]

    assert targets.dtype == torch.uint8
    assert targets[0, 1].item() == 1
    assert targets[0, 2].item() == 1
    assert matched.tolist() == [True, False]
    assert record["graph"]["selected_context_indices"].shape == (2, 2)
    assert record["metadata"]["graph_trace_available"]
    assert record["image"]["file_name"] == "HICO_test2015_00000101.jpg"

    for section in (record["nodes"], record["pairs"], record["graph"]):
        for value in section.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
                assert not value.requires_grad


def test_writer_metadata_merge_and_validation(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    writer = EvidenceTraceWriter(trace_dir, rank=0)
    first = make_record(101)
    second = copy.deepcopy(first)
    second["image"]["image_id"] = 102

    paths = writer.write_outputs(
        {"evidence_traces": [first, second]},
        expected_count=2,
    )
    assert len(paths) == 2

    args = SimpleNamespace(
        resume="/checkpoints/best.pth",
        detr_weights="/checkpoints/detr_r101.pth",
        dataset_file="hico-det",
        data_path="/datasets/hico",
        model_variant=None,
        stip_model_variant="evidence_trace",
        backbone="resnet101",
        graph_evidence_topk=4,
        graph_text_topk=5,
        seed=3407,
        action_names=["hold", "ride"],
        valid_ids=[0, 1],
        valid_obj_ids=[1, 2],
        graph_text_hoi_ids_path="/features/hoi_ids.json",
        graph_text_object_to_hoi_indices_path="/features/object_to_hoi.json",
    )
    metadata_path = writer.write_run_metadata(
        build_run_metadata(args, dataset_split="test", rank=0, world_size=1)
    )
    merge_manifests(trace_dir)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model"]["variant"] == "evidence_trace"
    assert metadata["model"]["backbone"] == "resnet101"
    assert metadata["model"]["graph_evidence_topk"] == 4
    assert metadata["model"]["graph_text_topk"] == 5

    summary, errors = validate_trace_directory(
        trace_dir,
        expected_records=2,
        require_graph=True,
    )
    assert not errors
    assert summary["passed"]
    assert summary["record_count"] == 2


def test_ranker_consumes_trace_schema() -> None:
    record = make_record()
    args = SimpleNamespace(
        min_context_count=1,
        min_two_hop_count=1,
        min_context_score=0.0,
        min_context_area=0.0,
        require_gt_prototype=True,
        require_final_correct=False,
        require_controlled_pair=False,
    )
    path = Path("/tmp/synthetic_evidence_trace.pt")
    hypothesis = ranker.hypothesis_rows(record, path, args, {}, {})
    pair_specific = ranker.pair_specific_rows(record, path, args, {}, {})
    assert hypothesis
    assert isinstance(pair_specific, list)
    assert isinstance(ranker.combined_rows(hypothesis, pair_specific), list)

    strict_args = copy.deepcopy(args)
    strict_args.pair_require_gt_matched = True
    strict_args.pair_require_final_correct = False
    assert ranker.pair_specific_rows(record, path, strict_args, {}, {}) == []

def test_merge_manifests_deduplicates_distributed_padding(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    rank0_writer = EvidenceTraceWriter(trace_dir, rank=0)
    rank1_writer = EvidenceTraceWriter(trace_dir, rank=1)

    rank0_writer.write_records([make_record(201), make_record(202)])
    rank1_writer.write_records([make_record(201)])
    merge_manifests(trace_dir)

    manifest_rows = [
        json.loads(line)
        for line in (trace_dir / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    duplicate_rows = [
        json.loads(line)
        for line in (trace_dir / "manifest_duplicates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["image_id"] for row in manifest_rows] == [201, 202]
    assert len(duplicate_rows) == 1
    assert duplicate_rows[0]["image_id"] == 201
    assert len(list((trace_dir / "records").glob("*.pt"))) == 2
    assert len(list((trace_dir / "duplicate_records").glob("*.pt"))) == 1

    merge_summary = json.loads(
        (trace_dir / "manifest_merge_summary.json").read_text(encoding="utf-8")
    )
    assert merge_summary == {
        "raw_manifest_rows": 3,
        "unique_manifest_rows": 2,
        "duplicate_manifest_rows": 1,
        "archived_duplicate_records": 1,
    }

    summary, errors = validate_trace_directory(
        trace_dir,
        expected_records=2,
        require_graph=True,
    )
    assert not errors
    assert summary["passed"]


def test_combined_ranking_uses_all_pair_candidates_before_limit() -> None:
    hypothesis = [{
        "record_path": "records/target.pt",
        "pair_idx": 0,
        "hypothesis_score": 0.7,
        "refinement_gain_norm": 0.6,
        "visual_quality": 0.5,
        "path_quality": 0.8,
    }]
    pair_specific = [
        {
            "record_path": "records/unmatched.pt",
            "pair_idx_1": 1,
            "pair_idx_2": 2,
            "pair_specific_score": 0.99,
            "control_type": "same_object_node",
            "pair_specificity": 1.0,
            "context_ids_1": "1;2",
            "context_ids_2": "3;4",
        },
        {
            "record_path": "records/target.pt",
            "pair_idx_1": 0,
            "pair_idx_2": 3,
            "pair_specific_score": 0.10,
            "control_type": "same_human_node",
            "pair_specificity": 0.8,
            "context_ids_1": "5;6",
            "context_ids_2": "7;8",
        },
    ]

    _, limited_pairs, combined, raw_counts = ranker.prepare_rankings(
        hypothesis,
        pair_specific,
        top_n=1,
    )
    assert limited_pairs[0]["record_path"] == "records/unmatched.pt"
    assert raw_counts["combined"] == 1
    assert len(combined) == 1
    assert combined[0]["record_path"] == "records/target.pt"
    assert combined[0]["partner_pair_idx"] == 3


def test_writer_fails_fast_for_missing_or_wrong_trace_count(tmp_path: Path) -> None:
    writer = EvidenceTraceWriter(tmp_path / "traces", rank=0)
    with assert_raises(RuntimeError, r"no 'evidence_traces'"):
        writer.write_outputs({}, expected_count=1)
    with assert_raises(RuntimeError, r"Expected 2 evidence traces"):
        writer.write_outputs({"evidence_traces": [make_record()]}, expected_count=2)


def test_trace_cli_options_select_isolated_variant() -> None:
    parser = get_args_parser()
    args = parser.parse_args(
        ["--stip_model_variant", "evidence_trace", "--eval_max_samples", "2"]
    )
    assert args.stip_model_variant == "evidence_trace"
    assert args.eval_max_samples == 2


def main() -> None:
    test_trace_record_has_multilabel_gt_and_unmatched_pair()
    with tempfile.TemporaryDirectory() as first_tmp:
        test_writer_metadata_merge_and_validation(Path(first_tmp))
    test_ranker_consumes_trace_schema()
    with tempfile.TemporaryDirectory() as distributed_tmp:
        test_merge_manifests_deduplicates_distributed_padding(
            Path(distributed_tmp)
        )
    test_combined_ranking_uses_all_pair_candidates_before_limit()
    with tempfile.TemporaryDirectory() as second_tmp:
        test_writer_fails_fast_for_missing_or_wrong_trace_count(Path(second_tmp))
    test_trace_cli_options_select_isolated_variant()
    print("evidence_trace_chain: all tests passed")


if __name__ == "__main__":
    main()
