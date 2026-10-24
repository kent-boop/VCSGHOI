#!/usr/bin/env python3
"""FLOPs profiling helpers for the dynamic, official EvidenceHOI forward."""

from __future__ import annotations

import collections
import contextlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.benchmark_utils import (
    BenchmarkError,
    inference_context,
    precision_context,
    prepare_official_batch,
    sample_identifier,
    summarize_numbers,
    validate_outputs,
)


def _flatten_tensor_outputs(value: Any) -> Tuple[torch.Tensor, ...]:
    tensors: List[torch.Tensor] = []

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            if item.is_floating_point():
                tensors.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if not tensors:
        raise BenchmarkError("Cannot profile a forward with no floating tensor outputs")
    return tuple(tensors)


class _FvcoreOfficialForwardWrapper(torch.nn.Module):
    """Tensor-only tracer adapter; computation remains in the official model."""

    def __init__(self, model: torch.nn.Module, images: Sequence[Any], targets: Sequence[Mapping[str, Any]]):
        super().__init__()
        self.model = model
        self.images = images
        self.targets = targets

    def forward(self, tensors: torch.Tensor, mask: torch.Tensor):
        from src.util.misc import NestedTensor

        outputs = self.model(NestedTensor(tensors, mask), self.images, self.targets)
        return _flatten_tensor_outputs(outputs)


# fvcore uses the common complexity convention in which one
# multiply-accumulate (MAC) is one unit. Paper GFLOPs below use 1 MAC = 2
# FLOPs. Non-MAC elementwise, indexing, comparison, reduction, and activation
# operations are explicitly excluded and recorded in the output artifacts.
_FVCORE_IGNORED_NON_MAC_OPS = {
    "aten::__or__", "aten::add", "aten::add_", "aten::any", "aten::argmax",
    "aten::atan2", "aten::clamp_min", "aten::clone", "aten::cos", "aten::cumsum",
    "aten::div", "aten::embedding", "aten::exp", "aten::expand_as", "aten::eye",
    "aten::fill_", "aten::fill_diagonal_", "aten::gelu", "aten::index_add_",
    "aten::index_select", "aten::isfinite", "aten::lift_fresh",
    "aten::linalg_vector_norm", "aten::log", "aten::logical_and", "aten::lt",
    "aten::masked_fill_", "aten::max_pool2d", "aten::maximum", "aten::min",
    "aten::minimum", "aten::mul", "aten::nan_to_num", "aten::ne", "aten::neg",
    "aten::numel", "aten::pow", "aten::prod", "aten::repeat", "aten::rsqrt",
    "aten::scatter_reduce", "aten::sigmoid", "aten::sin", "aten::softmax",
    "aten::sort", "aten::sqrt", "aten::square", "aten::sub", "aten::sum",
    "aten::tanh", "aten::topk", "aten::where",
}


def _baddbmm_mac_jit(inputs, outputs) -> float:
    from fvcore.nn.jit_handles import get_shape

    left = get_shape(inputs[1])
    right = get_shape(inputs[2])
    if left is None or right is None or len(left) != 3 or len(right) != 3:
        raise BenchmarkError(f"Cannot infer baddbmm shapes: {left}, {right}")
    if left[0] != right[0] or left[2] != right[1]:
        raise BenchmarkError(f"Invalid baddbmm shapes: {left}, {right}")
    return float(left[0] * left[1] * left[2] * right[2])


def _cdist_mac_jit(inputs, outputs) -> float:
    from fvcore.nn.jit_handles import get_shape

    left = get_shape(inputs[0])
    output = get_shape(outputs[0])
    if left is None or output is None:
        raise BenchmarkError(f"Cannot infer cdist shapes: {left}, {output}")
    output_elements = 1
    for dim in output:
        output_elements *= dim
    return float(output_elements * left[-1])


def _roi_align_mac_jit(inputs, outputs) -> float:
    from fvcore.nn.jit_handles import get_shape

    output = get_shape(outputs[0])
    if output is None:
        raise BenchmarkError("Cannot infer ROI Align output shape")
    output_elements = 1
    for dim in output:
        output_elements *= dim
    sampling_ratio = inputs[5].toIValue()
    samples = int(sampling_ratio) ** 2 if isinstance(sampling_ratio, int) and sampling_ratio > 0 else 1
    # Each bilinear sample is a weighted sum of four neighboring pixels.
    return float(output_elements * samples * 4)


def _fvcore_one(model, prepared_batch) -> Dict[str, Any]:
    try:
        import fvcore
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise BenchmarkError("fvcore is not installed") from exc

    samples, images, targets = prepared_batch
    wrapper = _FvcoreOfficialForwardWrapper(model, images, targets)
    analysis = FlopCountAnalysis(wrapper, (samples.tensors, samples.mask))
    analysis.tracer_warnings("none")
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    for operator in sorted(_FVCORE_IGNORED_NON_MAC_OPS):
        analysis.set_op_handle(operator, None)
    analysis.set_op_handle("aten::baddbmm", _baddbmm_mac_jit)
    analysis.set_op_handle("aten::cdist", _cdist_mac_jit)
    analysis.set_op_handle("torchvision::roi_align", _roi_align_mac_jit)
    total_macs = float(analysis.total())
    total_flops = total_macs * 2.0
    unsupported = {str(key): int(value) for key, value in analysis.unsupported_ops().items()}
    uncalled = sorted(str(name) for name in analysis.uncalled_modules())
    by_module_macs = {
        str(name): float(value) for name, value in analysis.by_module().items()
        if value and (
            str(name).startswith("model.detr")
            or str(name).startswith("model.blip")
            or str(name).startswith("model.visual_scene_graph")
            or str(name).startswith("model.interaction_decoder")
        )
    }
    by_module_flops = {name: value * 2.0 for name, value in by_module_macs.items()}
    return {
        "backend": "fvcore.nn.FlopCountAnalysis+EvidenceHOI_MAC_policy_v1",
        "backend_version": getattr(fvcore, "__version__", None),
        "value_kind": "MAC-derived FLOPs; 1 MAC = 2 FLOPs; non-MAC ops explicitly excluded",
        "supported_macs": total_macs,
        "supported_flops": total_flops,
        "unsupported_ops": unsupported,
        "uncalled_modules": uncalled,
        "coverage_status": "complete_supported_ops" if not unsupported else "partial",
        "paper_macs": total_macs if not unsupported else None,
        "paper_flops": total_flops if not unsupported else None,
        "mac_to_flop_multiplier": 2.0,
        "ignored_non_mac_ops": sorted(_FVCORE_IGNORED_NON_MAC_OPS),
        "custom_mac_ops": {
            "aten::baddbmm": "B*M*K*N MACs",
            "aten::cdist": "output_elements*feature_dim MAC-equivalents",
            "torchvision::roi_align": "4 MACs per bilinear sample",
        },
        "module_macs": by_module_macs,
        "module_flops": by_module_flops,
    }


def _torch_profiler_one(model, prepared_batch, precision: str, device: torch.device) -> Dict[str, Any]:
    if not hasattr(torch, "profiler"):
        raise BenchmarkError("torch.profiler is unavailable in this PyTorch version")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=True) as profile:
        with inference_context(), precision_context(precision, device):
            outputs = model(*prepared_batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    validate_outputs(outputs)
    supported_flops = float(sum(float(getattr(event, "flops", 0) or 0) for event in profile.key_averages()))
    return {
        "backend": "torch.profiler(with_flops=True)",
        "backend_version": torch.__version__,
        "value_kind": "runtime FLOPs for operators supported by torch.profiler",
        "supported_macs": supported_flops / 2.0,
        "supported_flops": supported_flops,
        "unsupported_ops": {"coverage_unknown": 1},
        "uncalled_modules": [],
        "coverage_status": "unknown",
        "paper_macs": None,
        "paper_flops": None,
        "mac_to_flop_multiplier": 2.0,
        "ignored_non_mac_ops": [],
        "custom_mac_ops": {},
        "module_macs": {},
        "module_flops": {},
    }


def profile_flops_on_real_batches(
    model: torch.nn.Module,
    cached_batches: Sequence[Any],
    device: torch.device,
    precision: str,
    max_images: int = 50,
    backend_preference: str = "auto",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Profile real images and return per-image rows, summary, and diagnostics."""
    if backend_preference not in {"auto", "fvcore", "torch_profiler"}:
        raise BenchmarkError(f"Unknown FLOPs backend: {backend_preference}")
    rows: List[Dict[str, Any]] = []
    diagnostics: List[str] = []
    backend_counts = collections.Counter()
    paper_values: List[float] = []
    paper_mac_values: List[float] = []
    supported_values: List[float] = []
    supported_mac_values: List[float] = []
    module_values: Dict[str, List[float]] = collections.defaultdict(list)
    module_mac_values: Dict[str, List[float]] = collections.defaultdict(list)
    ignored_non_mac_ops = set()
    custom_mac_ops: Dict[str, str] = {}
    images_seen = 0

    for batch in cached_batches:
        if images_seen >= max_images:
            break
        prepared = prepare_official_batch(batch, device)
        batch_image_ids = [sample_identifier(target) for target in prepared[2]]
        # Main protocol is batch size 1.  Refuse to assign one dynamic count to
        # several images because that would be falsely precise.
        if len(batch_image_ids) != 1:
            raise BenchmarkError("FLOPs profiling currently requires batch_size=1")
        if backend_preference == "torch_profiler":
            result = _torch_profiler_one(model, prepared, precision, device)
            diagnostics.append(
                f"image_id={batch_image_ids[0]} torch_profiler forced after fvcore static tracing exceeded the time budget"
            )
        else:
            try:
                with inference_context(), precision_context(precision, device):
                    result = _fvcore_one(model, prepared)
            except Exception as exc:
                if backend_preference == "fvcore":
                    raise
                diagnostics.append(f"image_id={batch_image_ids[0]} fvcore failed: {type(exc).__name__}: {exc}")
                result = _torch_profiler_one(model, prepared, precision, device)
        backend_counts[result["backend"]] += 1
        supported_values.append(result["supported_flops"])
        supported_mac_values.append(result["supported_macs"])
        ignored_non_mac_ops.update(result.get("ignored_non_mac_ops", []))
        custom_mac_ops.update(result.get("custom_mac_ops", {}))
        for module_name, module_flops in result.get("module_flops", {}).items():
            module_values[module_name].append(float(module_flops))
        for module_name, module_macs in result.get("module_macs", {}).items():
            module_mac_values[module_name].append(float(module_macs))
        if result.get("unsupported_ops"):
            diagnostics.append(
                f"image_id={batch_image_ids[0]} unsupported_ops={result['unsupported_ops']} "
                f"coverage={result['coverage_status']}"
            )
        if result["paper_flops"] is not None:
            paper_values.append(result["paper_flops"])
        if result.get("paper_macs") is not None:
            paper_mac_values.append(result["paper_macs"])
        rows.append({
            "image_id": batch_image_ids[0],
            "backend": result["backend"],
            "value_kind": result["value_kind"],
            "supported_macs": result["supported_macs"],
            "supported_flops": result["supported_flops"],
            "paper_macs": result.get("paper_macs"),
            "paper_flops": result["paper_flops"],
            "coverage_status": result["coverage_status"],
            "unsupported_ops": repr(result["unsupported_ops"]),
            "uncalled_modules": repr(result["uncalled_modules"]),
            "module_macs": repr(result.get("module_macs", {})),
            "module_flops": repr(result.get("module_flops", {})),
        })
        images_seen += 1
        del prepared
        if device.type == "cuda":
            torch.cuda.empty_cache()

    paper_complete = len(paper_values) == len(rows) and bool(rows)
    paper_mac_complete = len(paper_mac_values) == len(rows) and bool(rows)
    summary = {
        "profiled_images": len(rows),
        "backend_counts": dict(backend_counts),
        "definition": "MAC-derived FLOPs; 1 MAC = 2 FLOPs; non-MAC elementwise/index/reduction operations are explicitly excluded",
        "mac_to_flop_multiplier": 2.0,
        "ignored_non_mac_ops": sorted(ignored_non_mac_ops),
        "custom_mac_ops": dict(sorted(custom_mac_ops.items())),
        "supported_macs": summarize_numbers(supported_mac_values),
        "supported_flops": summarize_numbers(supported_values),
        "paper_macs": summarize_numbers(paper_mac_values) if paper_mac_complete else None,
        "paper_flops": summarize_numbers(paper_values) if paper_complete else None,
        "paper_value_available": paper_complete,
        "module_macs": {name: summarize_numbers(values) for name, values in sorted(module_mac_values.items())},
        "module_flops": {name: summarize_numbers(values) for name, values in sorted(module_values.items())},
        "module_values_are_inclusive": True,
        "coverage_note": (
            "All fvcore-traced images reported no unsupported MAC-producing operators under the documented policy."
            if paper_complete
            else "N/A for the paper table because one or more images used a partial/unknown-coverage profiler."
        ),
    }
    return rows, summary, diagnostics
