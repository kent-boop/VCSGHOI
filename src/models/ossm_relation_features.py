"""Old OSSM relation proposal features with configurable local assets."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import roi_align

from src.util import box_ops


def make_fc(dim_in, dim_out, a=1):
    layer = nn.Linear(dim_in, dim_out)
    nn.init.kaiming_uniform_(layer.weight, a=a)
    nn.init.constant_(layer.bias, 0)
    return layer


class OSSMRelationFeatureExtractor(nn.Module):
    """Relation representation used by the last SAM-based OSSM checkpoint.

    The implementation intentionally retains the old object-action prior and
    its 20% train-time dropout.  Only the asset location is made explicit.
    """

    COCO_MAP_INDICES = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
    )

    def __init__(self, args, in_channels, resolution=5, out_dim=1024):
        super().__init__()
        self.args = args
        self.resolution = resolution
        fusion_dim = args.hidden_dim * 2

        if args.use_spatial_feature:
            self.spatial_proj = make_fc(8, 64)
            fusion_dim += 64
        if args.use_tail_semantic_feature:
            self.label_embedding = nn.Embedding(args.num_classes + 1, 300)
            fusion_dim += 300
        if args.use_union_feature:
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1), nn.ReLU(inplace=True)
            )
            self.visual_proj = make_fc(256 * (resolution ** 2), 256)
            fusion_dim += 256

        self.prior_feat_dim = 256
        self.register_buffer(
            "coco_map_indices", torch.tensor(self.COCO_MAP_INDICES, dtype=torch.long)
        )
        prior_path = Path(args.ossm_obj_act_prior)
        if not prior_path.is_file():
            raise FileNotFoundError(
                f"OSSM object-action prior not found: {prior_path}. "
                "Set --ossm_obj_act_prior to an 80 x num_actions .npy file."
            )
        prior = np.load(prior_path)
        expected_shape = (80, args.num_actions)
        if prior.shape != expected_shape:
            raise ValueError(
                f"OSSM object-action prior must have shape {expected_shape}, got {prior.shape}"
            )
        prior = prior.astype(np.float32, copy=False) * 0.8 + 0.1
        if prior.shape[1] > 57:
            prior[:, 57] = np.maximum(prior[:, 57], 0.15)
        self.register_buffer("obj_act_prior", torch.from_numpy(prior))
        self.prior_proj = nn.Linear(args.num_actions, self.prior_feat_dim)
        nn.init.normal_(self.prior_proj.weight, std=0.01)
        nn.init.constant_(self.prior_proj.bias, 0)
        fusion_dim += self.prior_feat_dim

        self.fusion_fc = nn.Sequential(
            make_fc(fusion_dim, out_dim),
            nn.ReLU(),
            make_fc(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, rel_pairs, features, boxes, inst_reprs, idx, obj_label_logits=None, **_):
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0, 1)
        head_boxes = xyxy_boxes[rel_pairs[:, 0]]
        tail_boxes = xyxy_boxes[rel_pairs[:, 1]]
        union_boxes = torch.cat(
            [
                torch.minimum(head_boxes[:, :2], tail_boxes[:, :2]),
                torch.maximum(head_boxes[:, 2:], tail_boxes[:, 2:]),
            ],
            dim=1,
        )

        head_feats = inst_reprs[rel_pairs[:, 0]]
        tail_feats = inst_reprs[rel_pairs[:, 1]].clone()
        tail_feats[rel_pairs[:, 0] == rel_pairs[:, 1]] = 0
        relation_feats = torch.cat([head_feats, tail_feats], dim=-1)

        if self.args.use_spatial_feature:
            layouts = self.extract_spatial_layout_feats(xyxy_boxes)
            spatial = self.spatial_proj(layouts[rel_pairs[:, 0], rel_pairs[:, 1]])
            relation_feats = torch.cat([relation_feats, spatial], dim=-1)

        if self.args.use_tail_semantic_feature:
            semantic = (obj_label_logits.softmax(-1) @ self.label_embedding.weight)[rel_pairs[:, 1]]
            relation_feats = torch.cat([relation_feats, semantic], dim=-1)

        if self.args.use_union_feature:
            valid_pixels = (~features.mask[idx]).nonzero(as_tuple=False)
            if valid_pixels.numel() == 0:
                h, w = features.mask[idx].shape
            else:
                h, w = (valid_pixels.max(dim=0).values + 1).tolist()
            projected = self.input_proj(features.tensors[idx : idx + 1])
            batch_ids = union_boxes.new_zeros((len(union_boxes), 1))
            scale = union_boxes.new_tensor([w, h, w, h]).unsqueeze(0)
            rois = torch.cat([batch_ids, union_boxes * scale], dim=-1)
            union_visual = roi_align(
                projected,
                rois,
                output_size=self.resolution,
                sampling_ratio=2,
            )
            union_visual = self.visual_proj(union_visual.flatten(start_dim=1))
            relation_feats = torch.cat([relation_feats, union_visual], dim=-1)

        prior_feats = relation_feats.new_zeros((len(rel_pairs), self.prior_feat_dim))
        use_prior = not (self.training and torch.rand(1).item() < 0.2)
        if use_prior and obj_label_logits is not None:
            object_probs = obj_label_logits.softmax(-1)
            if object_probs.shape[1] >= 91:
                object_probs = object_probs[..., self.coco_map_indices]
            elif object_probs.shape[1] >= 80:
                object_probs = object_probs[:, :80]
            else:
                raise ValueError(
                    "OSSM object-action prior requires at least 80 object logits, "
                    f"got {object_probs.shape[1]}"
                )
            action_priors = object_probs[rel_pairs[:, 1]] @ self.obj_act_prior
            prior_feats = self.prior_proj(action_priors)

        return self.fusion_fc(torch.cat([relation_feats, prior_feats], dim=-1))

    @staticmethod
    def extract_spatial_layout_feats(xyxy_boxes):
        centers = torch.stack(
            [
                (xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2,
                (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2,
            ],
            dim=1,
        )
        dxdy = centers.unsqueeze(1) - centers.unsqueeze(0)
        theta = (torch.atan2(dxdy[..., 1], dxdy[..., 0]) / np.pi).unsqueeze(-1)
        distance = dxdy.norm(dim=-1, keepdim=True)
        area = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(dim=1)
        intersection_lt = torch.maximum(xyxy_boxes[:, None, :2], xyxy_boxes[None, :, :2])
        intersection_rb = torch.minimum(xyxy_boxes[:, None, 2:], xyxy_boxes[None, :, 2:])
        overlap = (intersection_rb - intersection_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        union_lt = torch.minimum(xyxy_boxes[:, None, :2], xyxy_boxes[None, :, :2])
        union_rb = torch.maximum(xyxy_boxes[:, None, 2:], xyxy_boxes[None, :, 2:])
        union = (union_rb - union_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        return torch.cat(
            [
                dxdy,
                distance,
                theta,
                overlap,
                union,
                area[:, None, None].expand_as(union),
                area[None, :, None].expand_as(union),
            ],
            dim=-1,
        )
