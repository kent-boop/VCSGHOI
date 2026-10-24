import torch

from src.models.ossm_box_regions import (
    BLIP_MEAN,
    BLIP_STD,
    DETR_MEAN,
    DETR_STD,
    build_box_region_batch,
)


def _detr_sample(rgb, padded_height=None, padded_width=None):
    height, width = rgb.shape[-2:]
    padded_height = padded_height or height
    padded_width = padded_width or width
    mean = rgb.new_tensor(DETR_MEAN).view(3, 1, 1)
    std = rgb.new_tensor(DETR_STD).view(3, 1, 1)
    sample = rgb.new_zeros((3, padded_height, padded_width))
    sample[:, :height, :width] = (rgb - mean) / std
    mask = torch.ones((padded_height, padded_width), dtype=torch.bool)
    mask[:height, :width] = False
    return sample, mask


def test_disjoint_pair_excludes_gap_but_union_includes_it_and_xy_are_not_swapped():
    sample, mask = _detr_sample(torch.full((3, 2, 4), 0.5), 4, 5)
    # On an 8x4 resized canvas these map to [0,0,2,2] and [6,2,8,4].
    boxes = torch.tensor(
        [[0.125, 0.25, 0.25, 0.50], [0.875, 0.75, 0.25, 0.50]]
    )
    result = build_box_region_batch(sample, mask, boxes, torch.tensor([[0, 1]]), 8)

    assert result.valid_size == (2, 4)
    assert result.resized_size == (4, 8)
    assert result.box_pair_masks.shape == (1, 8, 8)
    assert result.box_pair_masks.sum().item() == 8
    assert result.union_masks.sum().item() == 32
    assert not result.box_pair_masks[0, 1, 4]
    assert result.union_masks[0, 1, 4]
    assert result.box_pair_masks[0, 3, 7]
    assert not result.box_pair_masks[0, 7, 3]


def test_overlap_is_counted_once_and_identical_boxes_match_union():
    sample, mask = _detr_sample(torch.full((3, 4, 4), 0.5))
    boxes = torch.tensor(
        [
            [0.375, 0.375, 0.50, 0.50],
            [0.625, 0.625, 0.50, 0.50],
            [0.375, 0.375, 0.50, 0.50],
        ]
    )
    pairs = torch.tensor([[0, 1], [0, 2]])
    result = build_box_region_batch(sample, mask, boxes, pairs, 8)

    assert result.box_pair_masks[0].sum().item() == 28  # 16 + 16 - 4
    assert result.union_masks[0].sum().item() == 36
    assert torch.equal(result.box_pair_masks[1], result.union_masks[1])


def test_normalize_then_mask_gives_exact_zero_outside_region():
    rgb = torch.full((3, 4, 4), 0.5)
    sample, mask = _detr_sample(rgb)
    boxes = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    result = build_box_region_batch(sample, mask, boxes, torch.tensor([[0, 0]]), 8)

    region = result.box_pair_masks[0]
    assert torch.count_nonzero(result.box_pair_images[0, :, ~region]) == 0
    mean = rgb.new_tensor(BLIP_MEAN)
    std = rgb.new_tensor(BLIP_STD)
    expected = (rgb.new_tensor([0.5, 0.5, 0.5]) - mean) / std
    assert torch.allclose(result.box_pair_images[0, :, 1, 1], expected, atol=1e-5)


def test_boundary_tiny_boxes_receive_one_pixel_and_q_zero_is_supported():
    sample, mask = _detr_sample(torch.rand(3, 3, 5), 4, 7)
    boxes = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    one = build_box_region_batch(sample, mask, boxes, torch.tensor([[0, 1]]), 10)
    assert one.box_pair_masks.sum().item() == 2
    assert one.union_masks.sum().item() > 2

    empty = build_box_region_batch(
        sample, mask, boxes, torch.empty((0, 2), dtype=torch.long), 10
    )
    assert empty.box_pair_images.shape == (0, 3, 10, 10)
    assert empty.union_images.shape == (0, 3, 10, 10)
    assert empty.box_pair_masks.shape == (0, 10, 10)


def test_multiple_queries_preserve_order_and_shapes():
    sample, mask = _detr_sample(torch.rand(3, 6, 3))
    boxes = torch.tensor(
        [[0.25, 0.25, 0.3, 0.3], [0.75, 0.75, 0.3, 0.3], [0.5, 0.5, 0.2, 0.2]]
    )
    pairs = torch.tensor([[0, 1], [2, 1], [0, 2]])
    result = build_box_region_batch(sample, mask, boxes, pairs, 12)
    assert result.box_pair_images.shape == (3, 3, 12, 12)
    assert result.union_images.shape == (3, 3, 12, 12)
    assert not torch.equal(result.box_pair_masks[0], result.box_pair_masks[1])

