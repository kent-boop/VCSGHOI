"""BLIP patch-map ROI helpers for the SAM-free OSSM variant."""

import math

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from src.util import box_ops


DETR_MEAN = (0.485, 0.456, 0.406)
DETR_STD = (0.229, 0.224, 0.225)
BLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP_STD = (0.26862854, 0.26130258, 0.27577711)


def build_blip_tensor_from_sample(sample_tensor, sample_mask, image_size=384):
    """Convert the valid DETR-augmented image to a square BLIP input."""
    if sample_tensor.ndim != 3 or sample_tensor.shape[0] != 3:
        raise ValueError(
            f"sample_tensor must be [3,H,W], got {tuple(sample_tensor.shape)}"
        )
    if sample_mask.ndim != 2 or sample_mask.shape != sample_tensor.shape[-2:]:
        raise ValueError(
            "sample_mask must be [H,W] and match sample_tensor, got "
            f"{tuple(sample_mask.shape)} vs {tuple(sample_tensor.shape[-2:])}"
        )
    if image_size <= 0 or image_size % 16:
        raise ValueError("BLIP image_size must be a positive multiple of patch size 16")

    valid_pixels = (~sample_mask).nonzero(as_tuple=False)
    if valid_pixels.numel() == 0:
        raise ValueError("sample_mask contains no valid image pixels")
    valid_h, valid_w = (valid_pixels.max(dim=0).values + 1).tolist()

    detr_mean = sample_tensor.new_tensor(DETR_MEAN).view(3, 1, 1)
    detr_std = sample_tensor.new_tensor(DETR_STD).view(3, 1, 1)
    blip_mean = sample_tensor.new_tensor(BLIP_MEAN).view(3, 1, 1)
    blip_std = sample_tensor.new_tensor(BLIP_STD).view(3, 1, 1)

    image = sample_tensor[:, :valid_h, :valid_w]
    image = (image * detr_std + detr_mean).clamp(0, 1)
    image = F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (image - blip_mean) / blip_std, (valid_h, valid_w)


def normalized_xyxy_boxes(boxes_cxcywh):
    """Convert normalized DETR cxcywh boxes to clipped normalized xyxy boxes."""
    if boxes_cxcywh.ndim != 2 or boxes_cxcywh.shape[-1] != 4:
        raise ValueError(
            f"boxes_cxcywh must be [N,4], got {tuple(boxes_cxcywh.shape)}"
        )
    return box_ops.box_cxcywh_to_xyxy(boxes_cxcywh).clamp(0, 1)


def enclosing_union_boxes(human_boxes_xyxy, object_boxes_xyxy):
    """Return the enclosing rectangle for each ordered human-object pair."""
    if human_boxes_xyxy.shape != object_boxes_xyxy.shape:
        raise ValueError(
            "human and object boxes must have the same shape, got "
            f"{tuple(human_boxes_xyxy.shape)} and {tuple(object_boxes_xyxy.shape)}"
        )
    return torch.cat(
        [
            torch.minimum(human_boxes_xyxy[:, :2], object_boxes_xyxy[:, :2]),
            torch.maximum(human_boxes_xyxy[:, 2:], object_boxes_xyxy[:, 2:]),
        ],
        dim=-1,
    )


def build_ordered_pair_features(human_features, object_features):
    """Build a role-aware pair descriptor before its learned projection."""
    if human_features.shape != object_features.shape:
        raise ValueError(
            "human and object features must have the same shape, got "
            f"{tuple(human_features.shape)} and {tuple(object_features.shape)}"
        )
    return torch.cat(
        [
            human_features,
            object_features,
            human_features * object_features,
            human_features - object_features,
        ],
        dim=-1,
    )


def pool_blip_patch_tokens(
    patch_tokens,
    boxes_xyxy,
    pooler,
    output_size=5,
):
    """ROIAlign normalized boxes on projected BLIP patch tokens."""
    if patch_tokens.ndim != 2:
        raise ValueError(
            f"patch_tokens must be [num_patches,C], got {tuple(patch_tokens.shape)}"
        )
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError(f"boxes_xyxy must be [N,4], got {tuple(boxes_xyxy.shape)}")
    if boxes_xyxy.shape[0] == 0:
        return patch_tokens.new_zeros((0, patch_tokens.shape[-1]))

    patch_count, channels = patch_tokens.shape
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError(
            f"BLIP patch-token count must form a square grid, got {patch_count}"
        )
    patch_map = patch_tokens.transpose(0, 1).reshape(
        1, channels, grid_size, grid_size
    )
    scale = boxes_xyxy.new_tensor(
        [grid_size, grid_size, grid_size, grid_size]
    )
    roi_boxes = torch.cat(
        [
            boxes_xyxy.new_zeros((boxes_xyxy.shape[0], 1)),
            boxes_xyxy.clamp(0, 1) * scale.unsqueeze(0),
        ],
        dim=-1,
    )
    roi_features = roi_align(
        patch_map,
        roi_boxes,
        output_size=output_size,
        sampling_ratio=2,
        aligned=True,
    )
    return pooler(roi_features.flatten(start_dim=1))


__all__ = [
    "BLIP_MEAN",
    "BLIP_STD",
    "build_ordered_pair_features",
    "build_blip_tensor_from_sample",
    "enclosing_union_boxes",
    "normalized_xyxy_boxes",
    "pool_blip_patch_tokens",
]
