import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OSSM_RUNTIME_FILES = (
    "src/models/stip_ossm_box.py",
    "src/models/ossm_blip_roi.py",
    "src/models/ossm_relation_features.py",
    "src/models/stip_b1_blip_detr.py",
    "src/models/text_selector.py",
    "src/models/fusion_module.py",
    "src/models/moe_fusion_box.py",
)


def test_ossm_runtime_dependency_chain_imports_no_sam_or_evidence_graph():
    forbidden = (
        "src.lm.SAM",
        "segment_anything",
        "visual_scene_graph",
        "text_scene_graph",
        "candidate_evidence",
        "synchronized_evidence",
    )
    for relative_path in OSSM_RUNTIME_FILES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        for module_name in imported:
            assert not any(token in module_name for token in forbidden), (
                relative_path,
                module_name,
            )


def test_ossm_entry_has_no_sam_parameter_or_constructor_call():
    source = (ROOT / "src/models/stip_ossm_box.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "sam"
        if isinstance(node, ast.Name):
            assert node.id not in {"SamPredictor", "sam_model_registry"}
