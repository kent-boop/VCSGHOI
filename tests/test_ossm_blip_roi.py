import pytest
import torch
import torch.nn as nn

from src.models.ossm_blip_roi import (
    build_ordered_pair_features,
    build_blip_tensor_from_sample,
    enclosing_union_boxes,
    normalized_xyxy_boxes,
    pool_blip_patch_tokens,
)


def test_box_conversion_and_enclosing_union_are_normalized():
    boxes = torch.tensor(
        [
            [0.25, 0.40, 0.20, 0.40],
            [0.70, 0.50, 0.20, 0.20],
        ]
    )
    xyxy = normalized_xyxy_boxes(boxes)
    expected = torch.tensor(
        [[0.15, 0.20, 0.35, 0.60], [0.60, 0.40, 0.80, 0.60]]
    )
    assert torch.allclose(xyxy, expected)
    union = enclosing_union_boxes(xyxy[:1], xyxy[1:])
    assert torch.allclose(union, torch.tensor([[0.15, 0.20, 0.80, 0.60]]))


def test_ordered_pair_features_preserve_human_object_roles():
    human = torch.tensor([[1.0, 2.0]])
    obj = torch.tensor([[3.0, 5.0]])
    human_object = build_ordered_pair_features(human, obj)
    object_human = build_ordered_pair_features(obj, human)
    assert torch.equal(
        human_object, torch.tensor([[1.0, 2.0, 3.0, 5.0, 3.0, 10.0, -2.0, -3.0]])
    )
    assert not torch.equal(human_object, object_human)


def test_roi_align_preserves_count_and_handles_empty_boxes():
    patches = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
    pooled = pool_blip_patch_tokens(
        patches, boxes, nn.Identity(), output_size=1
    )
    assert pooled.shape == (2, 3)
    empty = pool_blip_patch_tokens(
        patches, boxes[:0], nn.Identity(), output_size=1
    )
    assert empty.shape == (0, 3)


def test_roi_align_rejects_non_square_patch_sequence():
    with pytest.raises(ValueError, match="square grid"):
        pool_blip_patch_tokens(
            torch.zeros(6, 3), torch.tensor([[0.0, 0.0, 1.0, 1.0]]), nn.Identity()
        )


def test_augmented_sample_preprocessing_discards_only_batch_padding():
    rgb = torch.full((3, 4, 6), 0.5)
    mean = rgb.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = rgb.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    sample = torch.zeros(3, 6, 8)
    sample[:, :4, :6] = (rgb - mean) / std
    mask = torch.ones(6, 8, dtype=torch.bool)
    mask[:4, :6] = False
    blip_image, valid_size = build_blip_tensor_from_sample(sample, mask, 32)
    assert valid_size == (4, 6)
    assert blip_image.shape == (3, 32, 32)
    assert torch.isfinite(blip_image).all()
