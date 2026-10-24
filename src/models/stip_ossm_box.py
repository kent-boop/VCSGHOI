"""Independent SAM-free OSSM variant using BLIP patch-map ROIs.

This restores the last OSSM text/adaptation/MoE/query/memory path while using
human/object and enclosing-union box features in place of SAM masks.  It has no
dependency on the Evidence graph stack or ``src.lm.SAM``.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.util.misc import NestedTensor, nested_tensor_from_tensor_list

from .fusion_module import TextGuidedBlipOptimizer, VisualAdapter, VisualTextAdapter
from .moe_fusion_box import OSSMBoxMoEFusion
from .ossm_blip_roi import (
    BLIP_MEAN,
    BLIP_STD,
    build_ordered_pair_features,
    build_blip_tensor_from_sample,
    enclosing_union_boxes,
    normalized_xyxy_boxes,
    pool_blip_patch_tokens,
)
from .ossm_relation_features import OSSMRelationFeatureExtractor, make_fc
from .stip_b1_blip_detr import (
    STIP as _SAMFreeSTIPUtilities,
    STIPCriterion as _BaseSTIPCriterion,
    STIPPostProcess,
)
from .text_selector import HierarchicalSimilarityCalculator
from .transformer import TransformerDecoder, TransformerDecoderLayer


class STIP(_SAMFreeSTIPUtilities):
    """OSSM with global/instance/action BLIP experts and no SAM runtime."""

    def __init__(self, args, detr, blip, detr_matcher):
        # Do not call the B1 constructor: it would create a different set of
        # edge-fusion modules.  We inherit only its SAM-free utility methods.
        nn.Module.__init__(self)
        self.args = args
        self.detr_matcher = detr_matcher
        self.detr = detr
        self.blip = blip
        for parameter in self.blip.parameters():
            parameter.requires_grad_(False)
        if not args.train_detr:
            for parameter in self.detr.parameters():
                parameter.requires_grad_(False)

        if args.relation_feature_map_from == "backbone":
            relation_feature_map_dim = 1024
        elif args.relation_feature_map_from == "detr_encoder":
            relation_feature_map_dim = args.hidden_dim
        else:
            raise ValueError(
                "OSSM box supports relation_feature_map_from in "
                f"{{'backbone','detr_encoder'}}, got {args.relation_feature_map_from!r}"
            )

        rel_rep_dim = 1024
        self.coarse_relation_feature_extractor = OSSMRelationFeatureExtractor(
            args, in_channels=relation_feature_map_dim, out_dim=rel_rep_dim
        )
        self.relation_proposal_mlp = nn.Sequential(
            make_fc(rel_rep_dim, rel_rep_dim // 2),
            nn.ReLU(),
            make_fc(rel_rep_dim // 2, 1),
        )
        self.rel_query_pre_proj = make_fc(rel_rep_dim, args.hidden_dim)

        if args.no_interaction_decoder:
            args.hoi_aux_loss = False
        else:
            self.memory_input_proj = nn.Conv2d(
                relation_feature_map_dim, args.hidden_dim, kernel_size=1
            )
            if args.use_relation_dependency_encoding:
                self.relation_dependency_embeddings = nn.Embedding(6, args.hidden_dim)
                self.relation_dependency_content_aware_mapping = nn.Sequential(
                    make_fc(args.hidden_dim * 2, args.hidden_dim),
                    nn.ReLU(),
                    make_fc(args.hidden_dim, args.hidden_dim),
                )
            if args.use_query_fourier_encoding:
                self.fourier_feature_embedding = make_fc(1, args.hidden_dim // 2)
                self.fourier_mlp = nn.Sequential(
                    make_fc(args.hidden_dim, args.hidden_dim),
                    nn.ReLU(),
                    make_fc(args.hidden_dim, args.hidden_dim // 8),
                )
            decoder_layer = TransformerDecoderLayer(
                d_model=args.hidden_dim, nhead=args.hoi_nheads
            )
            decoder_norm = nn.LayerNorm(args.hidden_dim)
            self.interaction_decoder = TransformerDecoder(
                decoder_layer,
                args.hoi_dec_layers,
                decoder_norm,
                return_intermediate=True,
            )

        self.action_embed = nn.Linear(args.hidden_dim, args.num_actions)
        self.sampled_actions = nn.Sequential(
            nn.Linear(rel_rep_dim, 512), nn.ReLU(), nn.Linear(512, args.num_actions)
        )

        self.text_selector = HierarchicalSimilarityCalculator(
            text_lib_path=args.ossm_text_library
        )
        dataset_text_rows = {
            "hico-det": 520,
            "vcoco": 289,
        }
        if args.dataset_file not in dataset_text_rows:
            raise ValueError(
                "OSSM box text validation supports HICO-DET and V-COCO only, "
                f"got {args.dataset_file!r}"
            )
        expected_text_shape = (dataset_text_rows[args.dataset_file], 3, 256)
        if tuple(self.text_selector.text_library.shape) != expected_text_shape:
            raise ValueError(
                f"{args.dataset_file} OSSM text library must have shape "
                f"{expected_text_shape}, "
                f"got {tuple(self.text_selector.text_library.shape)}"
            )
        self.scene_optimizer = TextGuidedBlipOptimizer()
        self.instance_optimizer = TextGuidedBlipOptimizer()
        self.action_cross_attn = nn.MultiheadAttention(
            embed_dim=args.hidden_dim, num_heads=4
        )
        self.action_norm = nn.LayerNorm(args.hidden_dim)
        self.action_dropout = nn.Dropout(0.1)
        self.memory_text_attn = nn.MultiheadAttention(
            embed_dim=args.hidden_dim, num_heads=4, batch_first=False
        )
        self.memory_text_norm = nn.LayerNorm(args.hidden_dim)
        self.memory_text_dropout = nn.Dropout(0.1)

        if args.ossm_region_image_size <= 0 or args.ossm_region_image_size % 16:
            raise ValueError("--ossm_region_image_size must be a positive multiple of BLIP patch size 16")
        patch_count = (args.ossm_region_image_size // 16) ** 2
        self.blip_scene_pos_embed = nn.Embedding(patch_count, args.hidden_dim)
        self.blip_inst_pos_embed = nn.Embedding(patch_count, args.hidden_dim)
        nn.init.normal_(self.blip_scene_pos_embed.weight, std=0.02)
        nn.init.normal_(self.blip_inst_pos_embed.weight, std=0.02)

        self.blip_roi_output_size = 5
        roi_flat_dim = args.hidden_dim * self.blip_roi_output_size ** 2
        self.instance_roi_pooler = nn.Sequential(
            nn.LayerNorm(roi_flat_dim),
            nn.Linear(roi_flat_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )
        self.action_roi_pooler = nn.Sequential(
            nn.LayerNorm(roi_flat_dim),
            nn.Linear(roi_flat_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )
        # Human and object roles are ordered.  The difference term changes sign
        # when roles are swapped, unlike a symmetric mean/sum fusion.
        self.instance_pair_fusion = nn.Sequential(
            nn.LayerNorm(args.hidden_dim * 4),
            nn.Linear(args.hidden_dim * 4, args.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
        )

        visual_dim_dict = {
            "global": 256,
            "instance": 256,
            "action": 256,
            "rel": rel_rep_dim,
        }
        self.text_input_dim = 256
        self.moe_out_dim = rel_rep_dim
        self.moe_fusion_module = OSSMBoxMoEFusion(
            visual_dim_dict=visual_dim_dict,
            text_dim=self.text_input_dim,
            out_dim=self.moe_out_dim,
            hidden_dim=1024,
            use_visual_router=getattr(args, "use_moe_visual_router", True),
            use_aux_loss=getattr(args, "use_moe_aux_loss", True),
            noisy_gating=getattr(args, "use_moe_noisy_gating", True),
        )
        self.moe_loss_log_var = nn.Parameter(torch.tensor(0.0))
        self.scene_adapter = VisualAdapter(visual_dim=256, text_dim=256, bottleneck=64)
        self.instance_adapter = VisualAdapter(visual_dim=256, text_dim=256, bottleneck=64)
        self.action_adapter = VisualTextAdapter(
            visual_dim=rel_rep_dim, text_dim=256, bottleneck=128
        )

        self.region_image_size = args.ossm_region_image_size
        self.keep_region_debug = getattr(args, "ossm_debug_regions", False)
        self.last_box_regions = None
        print(
            "[OSSM-BOX] SAM-free OSSM enabled: "
            "experts=(global, instance[human/object ROI], action[union ROI]), "
            f"image={self.region_image_size}, roi={self.blip_roi_output_size}x"
            f"{self.blip_roi_output_size}"
        )

    def build_blip_tensor_from_image(self, image, image_size=384, device=None):
        """Match the old global BLIP preprocessing (square LANCZOS resize)."""
        if isinstance(image, Image.Image):
            resampling = getattr(Image, "Resampling", Image)
            image_array = np.asarray(
                image.resize((image_size, image_size), resampling.LANCZOS),
                dtype=np.float32,
            ) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        elif torch.is_tensor(image):
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"image tensor must be [3,H,W], got {tuple(image.shape)}")
            image_tensor = image.float()
            if image_tensor.max() > 1:
                image_tensor = image_tensor / 255.0
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            raise TypeError(f"Unsupported global BLIP image type: {type(image)}")
        if device is not None:
            image_tensor = image_tensor.to(device)
        image_tensor = image_tensor.float().clamp(0, 1)
        mean = image_tensor.new_tensor(BLIP_MEAN).view(3, 1, 1)
        std = image_tensor.new_tensor(BLIP_STD).view(3, 1, 1)
        return (image_tensor - mean) / std

    def _select_relations(
        self,
        imgid,
        rel_mat,
        gt_rel_pairs,
        relation_feature_map,
        boxes,
        inst_repr,
        object_logits,
    ):
        if self.training:
            ground_truth_pairs = gt_rel_pairs[imgid]
            bg_instance_ids = object_logits.softmax(-1)[:, -1] > 1
            if len(ground_truth_pairs):
                rel_mat[ground_truth_pairs[:, :1], ~bg_instance_ids] = 1
                rel_mat[ground_truth_pairs[:, 0], ground_truth_pairs[:, 1]] = 0
            negative_pairs = rel_mat.nonzero(as_tuple=False)

            if self.args.use_hard_mining_for_relation_discovery:
                all_pairs = torch.cat([ground_truth_pairs, negative_pairs], dim=0)
                all_reps = self.coarse_relation_feature_extractor(
                    all_pairs,
                    relation_feature_map,
                    boxes.detach(),
                    inst_repr,
                    obj_label_logits=object_logits,
                    idx=imgid,
                )
                all_exists = self.relation_proposal_mlp(all_reps).squeeze(1)
                gt_count = len(ground_truth_pairs)
                gt_indices = torch.arange(gt_count, device=all_exists.device)
                negative_order = all_exists[gt_count:].argsort(descending=True) + gt_count
                selected = torch.cat([gt_indices, negative_order])[: self.args.num_hoi_queries]
                return all_pairs[selected], all_reps[selected], all_exists[selected]

            negative_order = torch.randperm(len(negative_pairs))
            sampled_pairs = torch.cat(
                [ground_truth_pairs, negative_pairs[negative_order]], dim=0
            )[: self.args.num_hoi_queries]
            sampled_reps = self.coarse_relation_feature_extractor(
                sampled_pairs,
                relation_feature_map,
                boxes.detach(),
                inst_repr,
                obj_label_logits=object_logits,
                idx=imgid,
            )
            sampled_exists = self.relation_proposal_mlp(sampled_reps).squeeze(1)
            return sampled_pairs, sampled_reps, sampled_exists

        all_pairs = rel_mat.nonzero(as_tuple=False)
        all_reps = self.coarse_relation_feature_extractor(
            all_pairs,
            relation_feature_map,
            boxes.detach(),
            inst_repr,
            obj_label_logits=object_logits,
            idx=imgid,
        )
        all_exists = self.relation_proposal_mlp(all_reps).squeeze(1)
        selected = all_exists.argsort(descending=True)[: self.args.num_hoi_queries]
        return all_pairs[selected], all_reps[selected], all_exists[selected]

    def _fuse_roi_semantics(
        self,
        imgid,
        samples,
        boxes,
        sampled_pairs,
        sampled_reps,
        blip_action,
        action_text,
    ):
        roi_image, valid_size = build_blip_tensor_from_sample(
            samples.tensors[imgid],
            samples.mask[imgid],
            image_size=self.region_image_size,
        )
        roi_multiscale = self.blip(roi_image.unsqueeze(0), "None", "multilscale")
        if len(roi_multiscale) != 3:
            raise ValueError(
                "OSSM ROI extraction requires BLIP's last three layers, got "
                f"{len(roi_multiscale)} outputs"
            )

        boxes_xyxy = normalized_xyxy_boxes(boxes.detach())
        node_instance_features = F.normalize(
            pool_blip_patch_tokens(
                roi_multiscale[1][0, 1:, :],
                boxes_xyxy,
                self.instance_roi_pooler,
                output_size=self.blip_roi_output_size,
            ),
            dim=-1,
        )
        human_boxes = boxes_xyxy[sampled_pairs[:, 0]]
        object_boxes = boxes_xyxy[sampled_pairs[:, 1]]
        union_boxes = enclosing_union_boxes(human_boxes, object_boxes)
        human_features = node_instance_features[sampled_pairs[:, 0]]
        object_features = node_instance_features[sampled_pairs[:, 1]]
        instance_pair_input = build_ordered_pair_features(
            human_features, object_features
        )
        instance_features = F.normalize(
            self.instance_pair_fusion(instance_pair_input), dim=-1
        )
        action_features = F.normalize(
            pool_blip_patch_tokens(
                roi_multiscale[2][0, 1:, :],
                union_boxes,
                self.action_roi_pooler,
                output_size=self.blip_roi_output_size,
            ),
            dim=-1,
        )

        query_count = len(sampled_pairs)
        visual_global = blip_action.expand(query_count, -1)
        action_text_input = action_text.unsqueeze(0).expand(query_count, -1, -1)
        action_text_dynamic = self.action_adapter(action_text_input, sampled_reps)

        similarity = torch.bmm(
            action_features.unsqueeze(1), action_text_dynamic.transpose(1, 2)
        ).squeeze(1) / np.sqrt(self.text_input_dim)
        router_text = torch.sum(
            action_text_dynamic * similarity.softmax(dim=-1).unsqueeze(-1), dim=1
        )
        fused, moe_aux_loss = self.moe_fusion_module(
            rel_reps=sampled_reps.unsqueeze(0),
            visual_global=visual_global.unsqueeze(0),
            visual_instance=instance_features.unsqueeze(0),
            visual_action=action_features.unsqueeze(0),
            text_features=router_text.unsqueeze(0),
        )
        region_debug = None
        if self.keep_region_debug:
            region_debug = {
                "image_index": imgid,
                "roi_image": roi_image.detach().cpu(),
                "boxes_xyxy": boxes_xyxy.detach().cpu(),
                "human_boxes_xyxy": human_boxes.detach().cpu(),
                "object_boxes_xyxy": object_boxes.detach().cpu(),
                "union_boxes_xyxy": union_boxes.detach().cpu(),
                "valid_size": tuple(valid_size),
                "patch_grid_size": int(np.sqrt(roi_multiscale[1].shape[1] - 1)),
                "roi_output_size": self.blip_roi_output_size,
                "instance_layer": -2,
                "action_layer": -1,
            }
            self.last_box_regions = region_debug
        return fused.squeeze(0), action_text_dynamic, moe_aux_loss, region_debug

    def forward(self, samples: NestedTensor, images, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if targets is None:
            raise ValueError("OSSM box requires targets for original sizes and train matching")
        start_time = time.time()

        features, pos = self.detr.backbone(samples)
        src, mask = features[-1].decompose()
        if mask is None:
            raise ValueError("DETR backbone must return a padding mask")
        hs, detr_encoder_outs = self.detr.transformer(
            self.detr.input_proj(src), mask, self.detr.query_embed.weight, pos[-1]
        )
        inst_repr = hs[-1]
        num_nodes = inst_repr.shape[1]
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()

        det2gt_indices = None
        gt_rel_pairs = None
        if self.training:
            detr_outs = {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
            }
            det2gt_indices = self.detr_matcher(detr_outs, targets)
            gt_rel_pairs = []
            for (det_indices, gt_indices), target in zip(det2gt_indices, targets):
                gt_to_det = torch.zeros(
                    len(gt_indices), device=det_indices.device, dtype=det_indices.dtype
                )
                gt_to_det[gt_indices] = det_indices
                pairs = gt_to_det[target["relation_map"].sum(-1).nonzero(as_tuple=False)]
                order = torch.randperm(len(pairs))
                gt_rel_pairs.append(pairs[order])

        memory_input, memory_input_mask = features[-2].decompose()
        memory_pos = pos[-2]
        if self.args.relation_feature_map_from == "backbone":
            relation_feature_map = features[-2]
        else:
            relation_feature_map = NestedTensor(detr_encoder_outs, memory_input_mask)
            memory_input = detr_encoder_outs
        if not self.args.no_interaction_decoder:
            memory_input = self.memory_input_proj(memory_input)

        pred_rel_exists = []
        pred_rel_pairs = []
        pred_actions = []
        pred_sampled_actions = []
        moe_aux_losses = []
        debug_rows = []
        topk_rows = []

        for imgid in range(samples.tensors.shape[0]):
            probs = outputs_class[-1, imgid].softmax(-1)
            inst_scores, inst_labels = probs[:, :-1].max(-1)
            human_ids = torch.logical_and(inst_scores > 0.5, inst_labels == 1).nonzero(
                as_tuple=False
            )
            background_ids = probs[:, -1] > 1
            if self.args.apply_nms_on_detr and not self.training:
                suppressed = self.apply_nms(
                    inst_scores, inst_labels, outputs_coord[-1, imgid]
                )
                background_ids[suppressed] = True

            relation_matrix = probs.new_zeros((num_nodes, num_nodes))
            relation_matrix[human_ids, ~background_ids] = 1
            if self.args.dataset_file != "vcoco":
                relation_matrix.fill_diagonal_(0)
            if self.args.adaptive_relation_query_num:
                if relation_matrix.count_nonzero() == 0:
                    relation_matrix[0, min(1, num_nodes - 1)] = 1
            elif relation_matrix.count_nonzero() < self.args.num_hoi_queries:
                if len(human_ids):
                    row = np.random.choice(human_ids.squeeze(1).tolist())
                else:
                    row = 0
                relation_matrix[row] = 1
                if self.args.dataset_file != "vcoco":
                    relation_matrix.fill_diagonal_(0)

            global_image = self.build_blip_tensor_from_image(
                images[imgid],
                image_size=self.region_image_size,
                device=samples.tensors.device,
            )
            multiscale = self.blip(global_image.unsqueeze(0), "None", "multilscale")
            blip_scene = multiscale[0][:, 1:, :]
            blip_instance = multiscale[1][:, 1:, :]
            blip_action = F.normalize(multiscale[2][:, 0, :], dim=-1)
            text_features, selected_indices = self.text_selector(blip_action)
            if getattr(self.args, "ossm_topk_gt_stats", False) and not self.training:
                topk_rows.append({name: ids[0].detach().cpu().tolist()
                                  for name, ids in selected_indices.items()})
            action_text = text_features["action"][0]
            dynamic_scene_text = self.scene_adapter(
                text_features["scene"], blip_action
            )
            dynamic_instance_text = self.instance_adapter(
                text_features["instance"], blip_action
            )
            optimized_scene = self.scene_optimizer(blip_scene, dynamic_scene_text)
            optimized_instance = self.instance_optimizer(
                blip_instance, dynamic_instance_text
            )

            sampled_pairs, sampled_reps, sampled_exists = self._select_relations(
                imgid,
                relation_matrix,
                gt_rel_pairs,
                relation_feature_map,
                outputs_coord[-1, imgid],
                inst_repr[imgid],
                outputs_class[-1, imgid],
            )
            sampled_reps, dynamic_action_text, moe_aux_loss, regions = (
                self._fuse_roi_semantics(
                    imgid,
                    samples,
                    outputs_coord[-1, imgid],
                    sampled_pairs,
                    sampled_reps,
                    blip_action,
                    action_text,
                )
            )
            moe_aux_losses.append(moe_aux_loss)
            sampled_action_logits = self.sampled_actions(sampled_reps)

            query_reps = self.rel_query_pre_proj(sampled_reps).unsqueeze(1)
            query_for_attention = query_reps.permute(1, 0, 2)
            text_for_attention = dynamic_action_text.permute(1, 0, 2)
            attention_output, _ = self.action_cross_attn(
                query_for_attention, text_for_attention, text_for_attention
            )
            query_reps = self.action_norm(
                query_reps + self.action_dropout(attention_output.permute(1, 0, 2))
            )

            if self.args.no_interaction_decoder:
                decoder_outputs = query_reps.unsqueeze(0)
            else:
                query_pos = None
                dependency_encodings = None
                target_mask = None
                subj_mask, obj_mask, union_mask, _ = self.generate_layout_masks(
                    sampled_pairs,
                    memory_input_mask,
                    outputs_coord[-1, imgid],
                    idx=imgid,
                )
                if self.args.use_relation_tgt_mask:
                    target_mask = torch.diag(sampled_exists) != 0
                    attend_ids = sampled_exists.argsort(descending=True)[
                        : self.args.use_relation_tgt_mask_attend_topk
                    ]
                    target_mask[:, attend_ids] = True
                    target_mask = (
                        target_mask.float()
                        .masked_fill(~target_mask, float("-inf"))
                        .masked_fill(target_mask, 0.0)
                    )
                if self.args.use_query_fourier_encoding:
                    coordinates = self.fourier_feature_embedding(
                        outputs_coord[-1, imgid][sampled_pairs].reshape(
                            len(sampled_pairs), 8, 1
                        )
                    ) / np.sqrt(self.args.hidden_dim / 2)
                    query_pos = self.fourier_mlp(
                        torch.cat([torch.cos(coordinates), torch.sin(coordinates)], dim=-1)
                    ).reshape(len(sampled_pairs), -1).unsqueeze(1)
                if self.args.use_relation_dependency_encoding:
                    dependency_map = torch.zeros(
                        (len(sampled_pairs), len(sampled_pairs)),
                        device=sampled_reps.device,
                        dtype=torch.long,
                    )
                    dependency_map[
                        sampled_pairs[:, 0, None] == sampled_pairs[None, :, 0]
                    ] = 1
                    dependency_map[
                        sampled_pairs[:, 1, None] == sampled_pairs[None, :, 1]
                    ] = 2
                    dependency_map[
                        sampled_pairs[:, 0, None] == sampled_pairs[None, :, 1]
                    ] = 3
                    dependency_map[
                        sampled_pairs[:, 1, None] == sampled_pairs[None, :, 0]
                    ] = 4
                    dependency_map.fill_diagonal_(5)
                    dependency_encodings = self.relation_dependency_embeddings(
                        dependency_map
                    )
                    dependency_encodings = (
                        self.relation_dependency_content_aware_mapping(
                            torch.cat(
                                [
                                    query_reps.permute(1, 0, 2).expand_as(
                                        dependency_encodings
                                    ),
                                    dependency_encodings,
                                ],
                                dim=-1,
                            )
                        ).unsqueeze(2)
                    )

                detr_memory = memory_input[imgid : imgid + 1].flatten(2).permute(2, 0, 1)
                detr_pos = memory_pos[imgid : imgid + 1].flatten(2).permute(2, 0, 1)
                scene_memory = optimized_scene.permute(1, 0, 2)
                instance_memory = optimized_instance.permute(1, 0, 2)
                visual_memory = torch.cat(
                    [scene_memory, instance_memory, detr_memory], dim=0
                )
                text_memory = action_text.unsqueeze(1)
                memory_attention, _ = self.memory_text_attn(
                    visual_memory, text_memory, text_memory
                )
                enhanced_memory = self.memory_text_norm(
                    visual_memory + self.memory_text_dropout(memory_attention)
                )

                scene_ids = torch.arange(scene_memory.shape[0], device=scene_memory.device)
                instance_ids = torch.arange(
                    instance_memory.shape[0], device=instance_memory.device
                )
                if scene_memory.shape[0] > self.blip_scene_pos_embed.num_embeddings:
                    raise ValueError("BLIP scene tokens exceed OSSM positional embedding capacity")
                if instance_memory.shape[0] > self.blip_inst_pos_embed.num_embeddings:
                    raise ValueError("BLIP instance tokens exceed OSSM positional embedding capacity")
                all_pos = torch.cat(
                    [
                        self.blip_scene_pos_embed(scene_ids).unsqueeze(1),
                        self.blip_inst_pos_embed(instance_ids).unsqueeze(1),
                        detr_pos,
                    ],
                    dim=0,
                )
                detr_padding_mask = memory_input_mask[imgid : imgid + 1].flatten(1)
                blip_padding_mask = torch.zeros(
                    (1, scene_memory.shape[0] + instance_memory.shape[0]),
                    dtype=torch.bool,
                    device=detr_padding_mask.device,
                )
                memory_padding_mask = torch.cat(
                    [blip_padding_mask, detr_padding_mask], dim=1
                )
                decoder_outputs = self.interaction_decoder(
                    tgt=query_reps,
                    tgt_mask=target_mask,
                    query_pos=query_pos,
                    query_structure_encoding=dependency_encodings,
                    memory=enhanced_memory,
                    memory_key_padding_mask=memory_padding_mask,
                    pos=all_pos,
                    memory_role_embedding=None,
                    memory_value_embedding=None,
                )

            pred_sampled_actions.append(sampled_action_logits)
            pred_rel_pairs.append(sampled_pairs)
            pred_actions.append(self.action_embed(decoder_outputs))
            pred_rel_exists.append(sampled_exists)
            if self.keep_region_debug:
                debug_rows.append(
                    {
                        "image_index": imgid,
                        "proposal_count": int(len(sampled_pairs)),
                        "valid_size": list(regions["valid_size"]),
                        "patch_grid_size": regions["patch_grid_size"],
                        "roi_output_size": regions["roi_output_size"],
                        "instance_layer": regions["instance_layer"],
                        "action_layer": regions["action_layer"],
                        "sam_runtime": False,
                        "evidence_graph_runtime": False,
                    }
                )

        if moe_aux_losses:
            moe_aux_loss = torch.stack(moe_aux_losses).mean()
        else:
            moe_aux_loss = outputs_coord.new_zeros(())
        if self.training and getattr(self.args, "use_moe_aux_loss", False):
            moe_loss_weighted = 0.05 * moe_aux_loss
        else:
            moe_loss_weighted = moe_aux_loss.new_zeros(())

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_rel_pairs": pred_rel_pairs,
            "pred_actions": [prediction[-1].squeeze(1) for prediction in pred_actions],
            "pred_sampled_actions": pred_sampled_actions,
            "pred_action_exists": pred_rel_exists,
            "det2gt_indices": det2gt_indices,
            "hoi_recognition_time": time.time() - start_time,
            "moe_aux_loss": moe_aux_loss,
            "moe_loss_weighted": moe_loss_weighted,
        }
        if getattr(self.args, "ossm_topk_gt_stats", False) and not self.training:
            output["ossm_topk_indices"] = topk_rows
        if self.keep_region_debug:
            output["ossm_box_debug"] = debug_rows
        if self.args.hoi_aux_loss:
            output["hoi_aux_outputs"] = self._set_hoi_aux_loss(pred_actions)
        if self.args.train_detr and self.args.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return output


class STIPCriterion(_BaseSTIPCriterion):
    """Current stable criterion plus the old (currently zero) MoE loss key."""

    def __init__(self, args, matcher):
        super().__init__(args, matcher)
        self.weight_dict["loss_moe_balance"] = 1.0

    def forward(self, outputs, targets, log=False):
        losses = super().forward(outputs, targets, log=log)
        if "moe_loss_weighted" in outputs:
            losses["loss_moe_balance"] = outputs["moe_loss_weighted"]
        return losses


__all__ = ["STIP", "STIPCriterion", "STIPPostProcess"]
