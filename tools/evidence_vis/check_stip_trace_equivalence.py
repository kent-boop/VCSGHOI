#!/usr/bin/env python3
"""Audit whether a Trace-enabled STIP file preserves the base model structure.

This is a source-structure check, not a numerical inference test. It verifies
that trainable module assignments, the criterion, post-processing, and all
non-Trace helpers are unchanged. The only accepted shared-method differences
are opt-in Trace wiring in __init__ and forward.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Set


TRACE_HELPERS = {
    "_trace_tensor",
    "_trace_python_value",
    "_trace_image_metadata",
    "_squeeze_trace_batch",
    "_build_pair_gt_targets",
    "_build_evidence_trace_record",
}
ALLOWED_CHANGED_SHARED_METHODS = {"__init__", "forward"}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def classes(tree: ast.Module) -> Dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def methods(node: ast.ClassDef) -> Dict[str, ast.AST]:
    return {
        item.name: item
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def assigned_self_names(node: ast.AST) -> Iterable[str]:
    for item in ast.walk(node):
        if not isinstance(item, (ast.Assign, ast.AnnAssign)):
            continue
        targets: Sequence[ast.AST]
        value: ast.AST | None
        if isinstance(item, ast.Assign):
            targets = item.targets
            value = item.value
        else:
            targets = [item.target]
            value = item.value
        if value is None:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                yield target.attr


def trainable_assignment_names(node: ast.ClassDef) -> Set[str]:
    names: Set[str] = set()
    for item in ast.walk(node):
        if not isinstance(item, (ast.Assign, ast.AnnAssign)):
            continue
        value = item.value
        if value is None:
            continue
        calls = [child for child in ast.walk(value) if isinstance(child, ast.Call)]
        is_trainable = any(
            dotted_name(call.func).startswith(("nn.", "torch.nn."))
            or dotted_name(call.func) == "torch.tensor"
            for call in calls
        )
        if not is_trainable:
            continue
        names.update(assigned_self_names(item))
    return names


def audit(base_path: Path, trace_path: Path) -> Mapping[str, object]:
    base_classes = classes(parse(base_path))
    trace_classes = classes(parse(trace_path))
    required = {"STIP", "STIPCriterion", "STIPPostProcess", "RelationFeatureExtractor"}
    missing_base = sorted(required - set(base_classes))
    missing_trace = sorted(required - set(trace_classes))

    if missing_base or missing_trace:
        return {
            "passed": False,
            "base": str(base_path),
            "trace": str(trace_path),
            "missing_base_classes": missing_base,
            "missing_trace_classes": missing_trace,
        }

    base_methods = methods(base_classes["STIP"])
    trace_methods = methods(trace_classes["STIP"])
    shared = set(base_methods) & set(trace_methods)
    changed_shared = sorted(
        name for name in shared if dump(base_methods[name]) != dump(trace_methods[name])
    )
    only_trace = sorted(set(trace_methods) - set(base_methods))
    only_base = sorted(set(base_methods) - set(trace_methods))

    class_equal = {
        name: dump(base_classes[name]) == dump(trace_classes[name])
        for name in ("STIPCriterion", "STIPPostProcess", "RelationFeatureExtractor")
    }
    base_trainable = trainable_assignment_names(base_classes["STIP"])
    trace_trainable = trainable_assignment_names(trace_classes["STIP"])
    trainable_only_base = sorted(base_trainable - trace_trainable)
    trainable_only_trace = sorted(trace_trainable - base_trainable)
    passed = (
        not only_base
        and set(only_trace) == TRACE_HELPERS
        and set(changed_shared) == ALLOWED_CHANGED_SHARED_METHODS
        and all(class_equal.values())
        and not trainable_only_base
        and not trainable_only_trace
    )
    return {
        "passed": passed,
        "base": str(base_path),
        "trace": str(trace_path),
        "interpretation": (
            "Same parameterized model; Trace file only adds opt-in evaluation export."
            if passed else
            "Unexpected structural differences found; inspect before sharing checkpoints."
        ),
        "changed_shared_methods": changed_shared,
        "trace_only_methods": only_trace,
        "base_only_methods": only_base,
        "unchanged_auxiliary_classes": class_equal,
        "trainable_assignment_count": len(base_trainable),
        "trainable_assignments_only_base": trainable_only_base,
        "trainable_assignments_only_trace": trainable_only_trace,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("src/models/stip.py"))
    parser.add_argument("--trace", type=Path, default=Path("src/models/stip_graph6.py"))
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = dict(audit(args.base, args.trace))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
