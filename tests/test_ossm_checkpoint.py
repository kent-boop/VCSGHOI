from pathlib import Path

import torch
import torch.nn as nn

from src.util.ossm_checkpoint import load_ossm_initialization


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.other = nn.Linear(2, 1)


def test_old_initialization_strips_module_and_filters_only_sam(tmp_path: Path):
    model = _TinyModel()
    expected_weight = torch.full_like(model.linear.weight, 3.0)
    checkpoint = {
        "model": {
            "module.linear.weight": expected_weight,
            "module.linear.bias": torch.full_like(model.linear.bias, 2.0),
            "module.sam.image_encoder.weight": torch.ones(1),
            "module.unknown.weight": torch.ones(1),
            "module.other.weight": torch.ones(7),
        },
        "epoch": 99,
        "optimizer": {"not": "used"},
    }
    path = tmp_path / "old.pth"
    torch.save(checkpoint, path)

    report = load_ossm_initialization(model, path)
    assert torch.equal(model.linear.weight, expected_weight)
    assert report.loaded_count == 2
    assert report.filtered_sam == ["module.sam.image_encoder.weight"]
    assert report.unexpected == ["module.unknown.weight"]
    assert report.shape_mismatches["other.weight"] == ((7,), (1, 2))
    assert "other.bias" in report.missing


def test_initialization_rejects_checkpoint_without_compatible_keys(tmp_path: Path):
    path = tmp_path / "bad.pth"
    torch.save({"model": {"sam.only": torch.ones(1)}}, path)
    try:
        load_ossm_initialization(_TinyModel(), path)
    except RuntimeError as error:
        assert "No compatible parameters" in str(error)
    else:
        raise AssertionError("Expected incompatible checkpoint to fail")

