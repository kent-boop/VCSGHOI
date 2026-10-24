"""SAM-free full-canvas region construction for the OSSM box variant.

The old model fed BLIP a 384x384 canvas whose normalized pixels were masked by
SAM.  This module preserves that canvas/normalization contract while replacing
segmentation with DETR-predicted rectangular regions.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from src.util import box_ops


DETR_MEAN = (0.485, 0.456, 0.406)
DETR_STD = (0.229, 0.224, 0.225)
BLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP_STD = (0.26862854, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class BoxRegionBatch:
    """Region tensors and geometry for one image.

    All image tensors use the old BLIP-normalized-zero background semantics:
    normalization happens first, then pixels outside the region are multiplied
    by zero.
    """

    box_pair_images: torch.Tensor
    union_images: torch.Tensor
    box_pair_masks: torch.Tensor
    union_masks: torch.Tensor
    boxes_xyxy: torch.Tensor
    canvas: torch.Tensor
    resized_size: Tuple[int, int]
    valid_size: Tuple[int, int]


def _channel_constants(tensor, values):
    return tensor.new_tensor(values).view(3, 1, 1)


def valid_image_from_nested(sample_tensor, sample_mask):
    """Remove right/bottom NestedTensor padding and undo DETR normalization."""
    if sample_tensor.ndim != 3 or sample_tensor.shape[0] != 3:
        raise ValueError(f"sample_tensor must be [3,H,W], got {tuple(sample_tensor.shape)}")
    if sample_mask.ndim != 2 or tuple(sample_mask.shape) != tuple(sample_tensor.shape[-2:]):
        raise ValueError(
            "sample_mask must be [H,W] and match sample_tensor, got "
            f"{tuple(sample_mask.shape)} vs {tuple(sample_tensor.shape[-2:])}"
        )

    valid_pixels = (~sample_mask.bool()).nonzero(as_tuple=False)
    if valid_pixels.numel() == 0:
        valid_h, valid_w = sample_tensor.shape[-2:]
    else:
        valid_h, valid_w = (valid_pixels.max(dim=0).values + 1).tolist()

    image = sample_tensor[:, :valid_h, :valid_w]
    detr_mean = _channel_constants(image, DETR_MEAN)
    detr_std = _channel_constants(image, DETR_STD)
    return (image * detr_std + detr_mean).clamp(0, 1), (int(valid_h), int(valid_w))


def build_blip_canvas(sample_tensor, sample_mask, image_size=384):
    """Make the old aspect-preserving, top-left-padded BLIP canvas."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    image, valid_size = valid_image_from_nested(sample_tensor, sample_mask)
    valid_h, valid_w = valid_size
    scale = float(image_size) / float(max(valid_h, valid_w))
    # Match the old PIL implementation, which used int() (floor for positives).
    resized_h = max(1, min(image_size, int(valid_h * scale)))
    resized_w = max(1, min(image_size, int(valid_w * scale)))
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    rgb_canvas = image.new_zeros((3, image_size, image_size))
    rgb_canvas[:, :resized_h, :resized_w] = resized
    blip_mean = _channel_constants(rgb_canvas, BLIP_MEAN)
    blip_std = _channel_constants(rgb_canvas, BLIP_STD)
    normalized_canvas = (rgb_canvas - blip_mean) / blip_std
    return normalized_canvas, (resized_h, resized_w), valid_size


def scale_boxes_to_canvas(boxes_cxcywh, resized_size):
    """Convert normalized DETR boxes to integer half-open canvas xyxy boxes."""
    if boxes_cxcywh.ndim != 2 or boxes_cxcywh.shape[-1] != 4:
        raise ValueError(f"boxes_cxcywh must be [N,4], got {tuple(boxes_cxcywh.shape)}")
    resized_h, resized_w = resized_size
    boxes = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh.detach()).clamp(0, 1)
    scale = boxes.new_tensor([resized_w, resized_h, resized_w, resized_h])
    boxes = boxes * scale
    start = torch.floor(boxes[:, :2])
    end = torch.ceil(boxes[:, 2:])

    start[:, 0].clamp_(0, max(resized_w - 1, 0))
    start[:, 1].clamp_(0, max(resized_h - 1, 0))
    end[:, 0].clamp_(1, resized_w)
    end[:, 1].clamp_(1, resized_h)
    # Degenerate/tiny predictions still receive one valid canvas pixel.
    end[:, 0] = torch.maximum(end[:, 0], start[:, 0] + 1).clamp(max=resized_w)
    end[:, 1] = torch.maximum(end[:, 1], start[:, 1] + 1).clamp(max=resized_h)
    return torch.cat([start, end], dim=-1).to(dtype=torch.long)


def _rectangle_masks(boxes_xyxy, height, width):
    """Rasterize integer half-open xyxy boxes without Python pixel loops."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0, height, width), dtype=torch.bool, device=boxes_xyxy.device)
    y = torch.arange(height, device=boxes_xyxy.device).view(1, height, 1)
    x = torch.arange(width, device=boxes_xyxy.device).view(1, 1, width)
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    return (
        (x >= x1[:, None, None])
        & (x < x2[:, None, None])
        & (y >= y1[:, None, None])
        & (y < y2[:, None, None])
    )


def build_box_region_batch(
    sample_tensor,
    sample_mask,
    boxes_cxcywh,
    relation_pairs,
    image_size=384,
):
    """Build pair-of-boxes and enclosing-union BLIP inputs for one image.

    ``box_pair`` is the boolean OR of the human and object rectangles, so the
    gap between disjoint boxes is excluded and overlap is counted once.
    ``union`` is the enclosing rectangle and therefore includes that gap.
    """
    if relation_pairs.ndim != 2 or relation_pairs.shape[-1] != 2:
        raise ValueError(f"relation_pairs must be [Q,2], got {tuple(relation_pairs.shape)}")
    if relation_pairs.numel():
        if relation_pairs.min() < 0 or relation_pairs.max() >= boxes_cxcywh.shape[0]:
            raise IndexError("relation_pairs contains an index outside boxes_cxcywh")

    canvas, resized_size, valid_size = build_blip_canvas(
        sample_tensor, sample_mask, image_size=image_size
    )
    boxes_xyxy = scale_boxes_to_canvas(boxes_cxcywh, resized_size)
    q = relation_pairs.shape[0]
    if q == 0:
        empty_masks = torch.zeros(
            (0, image_size, image_size), dtype=torch.bool, device=sample_tensor.device
        )
        empty_images = sample_tensor.new_zeros((0, 3, image_size, image_size))
        return BoxRegionBatch(
            empty_images,
            empty_images.clone(),
            empty_masks,
            empty_masks.clone(),
            boxes_xyxy,
            canvas,
            resized_size,
            valid_size,
        )

    pairs = relation_pairs.to(device=boxes_xyxy.device, dtype=torch.long)
    human_boxes = boxes_xyxy[pairs[:, 0]]
    object_boxes = boxes_xyxy[pairs[:, 1]]
    human_masks = _rectangle_masks(human_boxes, image_size, image_size)
    object_masks = _rectangle_masks(object_boxes, image_size, image_size)
    box_pair_masks = human_masks | object_masks

    union_boxes = torch.cat(
        [
            torch.minimum(human_boxes[:, :2], object_boxes[:, :2]),
            torch.maximum(human_boxes[:, 2:], object_boxes[:, 2:]),
        ],
        dim=-1,
    )
    union_masks = _rectangle_masks(union_boxes, image_size, image_size)
    box_pair_images = canvas.unsqueeze(0) * box_pair_masks.unsqueeze(1).to(canvas.dtype)
    union_images = canvas.unsqueeze(0) * union_masks.unsqueeze(1).to(canvas.dtype)
    return BoxRegionBatch(
        box_pair_images,
        union_images,
        box_pair_masks,
        union_masks,
        boxes_xyxy,
        canvas,
        resized_size,
        valid_size,
    )
