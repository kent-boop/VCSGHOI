"""Synchronized Visual-Text Scene Graph Co-Reasoning version. Old files are preserved."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from PIL import Image
from src.util.misc import NestedTensor, nested_tensor_from_tensor_list
from torchvision.ops import roi_align
from .transformer import TransformerDecoderLayer, TransformerDecoder
from src.util import box_ops
from src.util.misc import accuracy, is_dist_avail_and_initialized, get_world_size, get_rank
from src.models.visual_scene_graph_oov8 import build_visual_scene_graph_reasoner

class STIP(nn.Module):
    def __init__(self, args, detr, blip, detr_matcher):
        super().__init__()
        self.args = args
        self.detr_matcher = detr_matcher
        # * Instance Transformer ---------------
        self.detr = detr
        self.blip = blip
        for p in self.blip.parameters():
            p.requires_grad_(False)
        if not args.train_detr:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        # --------------------------------------
        # relation feature map
        if self.args.relation_feature_map_from == 'backbone':
            # if args.use_high_resolution_relation_feature_map:
            #     relation_feature_map_dim = 1024
            # else:
            #     relation_feature_map_dim = 2048
            relation_feature_map_dim = 1024
        elif self.args.relation_feature_map_from == 'detr_encoder':
            relation_feature_map_dim = self.args.hidden_dim
        # relation proposal
        rel_rep_dim = 1024
        self.coarse_relation_feature_extractor = RelationFeatureExtractor(args, blip=blip,
                                                                          in_channels=relation_feature_map_dim,
                                                                          out_dim=rel_rep_dim)
        # self.union_box_feature_extractor = RelationFeatureExtractor(args, in_channels=relation_feature_map_dim, out_dim=rel_rep_dim)
        self.relation_proposal_mlp = nn.Sequential(
            make_fc(rel_rep_dim, rel_rep_dim // 2), nn.ReLU(),
            make_fc(rel_rep_dim // 2, 1)
        )
        # relation classification
        self.rel_query_pre_proj = make_fc(rel_rep_dim, self.args.hidden_dim)
        if self.args.no_interaction_decoder:#False
            self.args.hoi_aux_loss = False
        else:
            self.memory_input_proj = nn.Conv2d(relation_feature_map_dim, self.args.hidden_dim, kernel_size=1)
            # if args.use_memory_layout_encoding:
            #     self.layout_embeddings = nn.Embedding(6,
            #                                           self.args.hidden_dim)  # 0-pad, 1-image, 2-union, 3-subj, 4-obj, 5-intersection
            #     self.layout_content_aware_mapping = nn.Sequential(
            #         make_fc(self.args.hidden_dim * 2, self.args.hidden_dim), nn.ReLU(),
            #         make_fc(self.args.hidden_dim, self.args.hidden_dim)
            #     )
            if self.args.use_relation_dependency_encoding:#True
                self.relation_dependency_embeddings = nn.Embedding(6, self.args.hidden_dim)
                self.relation_dependency_content_aware_mapping = nn.Sequential(
                    make_fc(self.args.hidden_dim * 2, self.args.hidden_dim), nn.ReLU(),
                    make_fc(self.args.hidden_dim, self.args.hidden_dim)
                )
            if self.args.use_query_fourier_encoding:
                self.fourier_feature_embedding = make_fc(1, self.args.hidden_dim // 2)  # group=8, group_dim=1
                self.fourier_mlp = nn.Sequential(
                    make_fc(self.args.hidden_dim, self.args.hidden_dim), nn.ReLU(),
                    make_fc(self.args.hidden_dim, self.args.hidden_dim // 8)
                )
            decoder_layer = TransformerDecoderLayer(d_model=self.args.hidden_dim, nhead=self.args.hoi_nheads)
            decoder_norm = nn.LayerNorm(self.args.hidden_dim)
            self.interaction_decoder = TransformerDecoder(decoder_layer, self.args.hoi_dec_layers, decoder_norm,
                                                          return_intermediate=True)
        self.action_embed = nn.Linear(self.args.hidden_dim, self.args.num_actions)
        self.sampled_actions = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, self.args.num_actions))
        self.use_edge_blip_fusion = getattr(args, 'use_edge_blip_fusion', True)
        self.use_blip_global_memory = getattr(args, 'use_blip_global_memory', True)
        self.edge_blip_gate = nn.Sequential(
            nn.Linear(self.args.hidden_dim * 4, self.args.hidden_dim * 3),
            nn.Sigmoid(),
        )
        self.edge_blip_proj = nn.Sequential(
            nn.Linear(self.args.hidden_dim * 3, self.args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
        )
        self.edge_rel_proj = nn.Linear(rel_rep_dim + self.args.hidden_dim, rel_rep_dim)
        self.blip_memory_proj = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.blip_roi_output_size = 5
        self.blip_roi_pooler = nn.Sequential(
            nn.LayerNorm(self.args.hidden_dim * self.blip_roi_output_size * self.blip_roi_output_size),
            nn.Linear(self.args.hidden_dim * self.blip_roi_output_size * self.blip_roi_output_size, self.args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
        )
        self.use_visual_scene_graph = getattr(args, 'use_visual_scene_graph', False)
        self.graph_debug = getattr(args, 'graph_debug', False)
        enable_text_graph = getattr(args, "graph_enable_text_graph", getattr(args, "enable_text_graph", False))
        enable_candidate_evidence = getattr(args, "graph_enable_candidate_evidence", False)
        #context path
        graph_evidence_topk = getattr(args, "graph_evidence_topk", 4)
        graph_evidence_gate_init = getattr(args, "graph_evidence_gate_init", 0.0)
        graph_evidence_dropout = getattr(args, "graph_evidence_dropout", 0.1)
        graph_evidence_allow_node_only_context = getattr(args, "graph_evidence_allow_node_only_context", False)
        graph_evidence_debug = getattr(args, "graph_evidence_debug", False)
        enable_sync_matching = getattr(args, "graph_enable_sync_evidence_matching", False)
        graph_sync_visual_gate_init = getattr(args, "graph_sync_visual_gate_init", 0.0)
        graph_sync_text_gate_init = getattr(args, "graph_sync_text_gate_init", 0.0)
        graph_sync_anchor_gate_init = getattr(args, "graph_sync_anchor_gate_init", 0.0)
        graph_sync_dropout = getattr(args, "graph_sync_dropout", 0.1)
        graph_sync_debug = getattr(args, "graph_sync_debug", False)
        graph_text_force_oov3 = getattr(args, "graph_text_force_oov3", True)
        graph_text_prototype_slot_fallback_to_raw = getattr(args, "graph_text_prototype_slot_fallback_to_raw", True)
        graph_text_prototype_slot_debug = getattr(args, "graph_text_prototype_slot_debug", False)
        enable_layerwise_sync_matching = getattr(args, "graph_enable_layerwise_sync_evidence_matching", False)
        graph_layerwise_sync_write_visual = getattr(args, "graph_layerwise_sync_write_visual", True)
        graph_layerwise_sync_write_text = getattr(args, "graph_layerwise_sync_write_text", True)
        graph_layerwise_sync_final_matching = getattr(args, "graph_layerwise_sync_final_matching", False)
        graph_layerwise_sync_disable_old_dense_injection = getattr(args, "graph_layerwise_sync_disable_old_dense_injection", True)
        graph_layerwise_sync_write_reverse_edge = getattr(args, "graph_layerwise_sync_write_reverse_edge", False)
        graph_layerwise_sync_detach_text_update = getattr(args, "graph_layerwise_sync_detach_text_update", False)
        graph_layerwise_sync_debug = getattr(args, "graph_layerwise_sync_debug", False)
        graph_store_dense_debug_tensors = getattr(args, "graph_store_dense_debug_tensors", False)
        if enable_sync_matching and not enable_candidate_evidence:
            raise ValueError(
                "graph_enable_sync_evidence_matching=True requires "
                "graph_enable_candidate_evidence=True."
            )
        if enable_sync_matching and not enable_text_graph:
            raise ValueError(
                "graph_enable_sync_evidence_matching=True requires "
                "graph_enable_text_graph=True."
            )
        self.visual_scene_graph = build_visual_scene_graph_reasoner(args)
        self.graph_gate = nn.Parameter(torch.tensor(0.0))
        self._graph_debug_printed = False
        self._graph_debug_steps = 0
        self._graph_debug_prints = 0
        self.graph_debug_max_prints = getattr(args, 'graph_debug_max_prints', 20)
        self.graph_debug_interval = getattr(args, 'graph_debug_interval', 100)
        self.graph_node_score_thresh = getattr(args, 'graph_node_score_thresh', 0.05)
        self.graph_min_score_nodes = getattr(args, 'graph_min_score_nodes', 0)
        self.graph_max_extra_nodes = getattr(args, 'graph_max_extra_nodes', self.args.num_queries)
        def _short_path(path):
            return os.path.basename(path) if path else ""
        print(
            "[STIP-Graph6] Visual graph base: "
            f"use_visual_scene_graph={self.use_visual_scene_graph}, "
            f"graph_num_layers={getattr(args, 'graph_num_layers', 2)}, "
            f"graph_num_heads={getattr(args, 'graph_num_heads', 8)}, "
            f"graph_k_oo={getattr(args, 'graph_k_oo', 3)}, "
            f"graph_oo_mode={getattr(args, 'graph_oo_mode', 'topk')}, "
            f"graph_enable_oo_edges={getattr(args, 'graph_enable_oo_edges', True)}, "
            f"graph_enable_hh_edges={getattr(args, 'graph_enable_hh_edges', False)}"
        )
        print(
            "[STIP-Graph6] Text graph / prototype slots: "
            f"graph_enable_text_graph={enable_text_graph}, "
            f"graph_text_sync_mode={getattr(args, 'graph_text_sync_mode', 'layerwise')}, "
            f"graph_text_topk={getattr(args, 'graph_text_topk', 5)}, "
            f"graph_text_pre_topk={getattr(args, 'graph_text_pre_topk', 20)}, "
            f"graph_text_force_oov3={graph_text_force_oov3}, "
            f"graph_text_prototype_slot_fallback_to_raw={graph_text_prototype_slot_fallback_to_raw}, "
            f"graph_text_use_scene_token={getattr(args, 'graph_text_use_scene_token', True)}, "
            f"graph_text_use_oo_context_token={getattr(args, 'graph_text_use_oo_context_token', True)}, "
            f"features={_short_path(getattr(args, 'graph_text_features_path', ''))}, "
            f"mask={_short_path(getattr(args, 'graph_text_mask_path', ''))}, "
            f"prototype_debug={graph_text_prototype_slot_debug}"
        )
        print(
            "[STIP-Graph6] EvidenceHOI modules: "
            f"graph_enable_candidate_evidence={enable_candidate_evidence}, "
            f"graph_evidence_topk={graph_evidence_topk}, "
            f"graph_evidence_allow_node_only_context={graph_evidence_allow_node_only_context}, "
            f"graph_enable_sync_evidence_matching={enable_sync_matching}, "
            f"graph_sync_visual_gate_init={graph_sync_visual_gate_init}, "
            f"graph_sync_text_gate_init={graph_sync_text_gate_init}, "
            f"graph_sync_anchor_gate_init={graph_sync_anchor_gate_init}, "
            f"graph_evidence_gate_init={graph_evidence_gate_init}, "
            f"graph_evidence_dropout={graph_evidence_dropout}, "
            f"graph_sync_dropout={graph_sync_dropout}, "
            f"graph_evidence_debug={graph_evidence_debug}, "
            f"graph_sync_debug={graph_sync_debug}"
        )
        print(
            "[STIP-Graph6] EvidenceHOI Stage 4-standard: "
            f"graph_enable_layerwise_sync_evidence_matching={enable_layerwise_sync_matching}, "
            f"graph_layerwise_sync_write_visual={graph_layerwise_sync_write_visual}, "
            f"graph_layerwise_sync_write_text={graph_layerwise_sync_write_text}, "
            f"graph_layerwise_sync_final_matching={graph_layerwise_sync_final_matching}, "
            f"graph_layerwise_sync_disable_old_dense_injection={graph_layerwise_sync_disable_old_dense_injection}, "
            f"graph_layerwise_sync_write_reverse_edge={graph_layerwise_sync_write_reverse_edge}, "
            f"graph_layerwise_sync_detach_text_update={graph_layerwise_sync_detach_text_update}, "
            f"graph_layerwise_sync_debug={graph_layerwise_sync_debug}, "
            f"graph_store_dense_debug_tensors={graph_store_dense_debug_tensors}"
        )

    def forward(self, samples: NestedTensor, images, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        #trasnsfer for samples (input to detr)
        start_time = time.time()
        # >>>>>>>>>>>>  BACKBONE LAYERS  <<<<<<<<<<<<<<<
        features, pos = self.detr.backbone(samples)
        bs, h, w = features[-1].tensors.shape[0], features[-1].tensors.shape[2], features[-1].tensors.shape[3]
        src, mask = features[-1].decompose()

        assert mask is not None
        # ----------------------------------------------
        # >>>>>>>>>>>> OBJECT DETECTION LAYERS <<<<<<<<<<
        hs, detr_encoder_outs = self.detr.transformer(self.detr.input_proj(src), mask, self.detr.query_embed.weight,
                                                      pos[-1])
        inst_repr = hs[-1]
        num_nodes = inst_repr.shape[1]

        # Prediction Heads for Object Detection
        outputs_class_org = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()

        outputs_class = outputs_class_org

        det2gt_indices = None
        if self.training:
            detr_outs = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            det2gt_indices = self.detr_matcher(detr_outs, targets)
            gt_rel_pairs = []
            for (ds, gs), t in zip(det2gt_indices, targets):
                gt2det_map = torch.zeros(len(gs)).to(device=ds.device, dtype=ds.dtype)
                gt2det_map[gs] = ds
                gt_rels = gt2det_map[t['relation_map'].sum(-1).nonzero(as_tuple=False)]
                perm = torch.randperm(len(gt_rels))

                gt_rel_pairs.append(gt_rels[perm])
                # if len(gt_rel_pairs[-1]) > self.args.num_hoi_queries:
                #     print(f"imageid={t['image_id']}, gt_relation_count={len(gt_rel_pairs[-1])}")
                #     # check_annotation(samples, targets, rel_num=20, idx=0)

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        pred_rel_exists, pred_rel_pairs, pred_actions, pred_sampled_actions = [], [], [], []
        graph_inputs = []
        graph_input_debug = []
        graph_reasoner_debug = []
        graph_debug_stats = []
        memory_input, memory_input_mask = features[-2].decompose()
        memory_pos = pos[-2]
        # print(f"memory_pos Shape: {memory_pos.shape}")
        if self.args.relation_feature_map_from == 'backbone':
            relation_feature_map = features[-2]
        elif self.args.relation_feature_map_from == 'detr_encoder':
            relation_feature_map = NestedTensor(detr_encoder_outs, memory_input_mask)
            memory_input = detr_encoder_outs

        if not self.args.no_interaction_decoder:
            memory_input = self.memory_input_proj(memory_input)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        for imgid in range(bs):
            # >>>>>>>>>>>> relation proposal <<<<<<<<<<<<<<<
            probs = outputs_class[-1, imgid].softmax(-1)
            inst_scores, inst_labels = probs[:, :-1].max(-1)
            human_instance_ids = torch.logical_and(inst_scores > 0.5, inst_labels == 1).nonzero(as_tuple=False)
            bg_instance_ids = (probs[:, -1] > 1)
            img_tensor_blip, blip_valid_size = self.build_blip_tensor_from_sample(samples.tensors[imgid], samples.mask[imgid])
            if self.args.apply_nms_on_detr and not self.training:
                suppress_ids = self.apply_nms(inst_scores, inst_labels, outputs_coord[-1, imgid])
                bg_instance_ids[suppress_ids] = True

            rel_mat = torch.zeros((num_nodes, num_nodes)).to(probs.device)
            rel_mat[human_instance_ids, ~bg_instance_ids] = 1  # subj is human, obj is not background
            if self.args.dataset_file != 'vcoco': rel_mat.fill_diagonal_(0)
            if self.args.adaptive_relation_query_num:
                if len(rel_mat.nonzero(as_tuple=False)) == 0: rel_mat[0, 1] = 1
            else:  # ensure enough queries
                if len(rel_mat.nonzero(as_tuple=False)) < self.args.num_hoi_queries:
                    tmp_id = np.random.choice(human_instance_ids.squeeze(1).tolist()) if len(
                        human_instance_ids) > 0 else 0
                    rel_mat[tmp_id] = 1

            #blip visual encoder, last layer tokens only
            blip_last_layer = self.blip(img_tensor_blip.unsqueeze(0).to(samples.tensors.device), 'None', 'last_layer')
            blip_action = blip_last_layer[:, 0, :]#last layer cls token
            blip_patches = blip_last_layer[:, 1:, :]#last layer patch tokens
            blip_action = F.normalize(blip_action, dim=-1)
            #all image to BLIP
            img_tensor_scene = self.build_blip_tensor_from_image(images[imgid], device=samples.tensors.device)
            blip_scene = self.blip(img_tensor_scene.unsqueeze(0), 'None', 'last_layer')
            blip_scene_action = F.normalize(blip_scene[:, 0, :], dim=-1)
            blip_global_patches = blip_scene[:, 1:, :]
            blip_global_context = F.normalize(blip_action + blip_scene_action, dim=-1)

            if self.training:
                rel_mat[gt_rel_pairs[imgid][:, :1], ~bg_instance_ids] = 1
                rel_mat[gt_rel_pairs[imgid][:, 0], gt_rel_pairs[imgid][:, 1]] = 0
                rel_pairs = rel_mat.nonzero(as_tuple=False)  # neg pairs

                if self.args.use_hard_mining_for_relation_discovery:
                    # hard negative sampling
                    all_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs], dim=0)
                    gt_pair_count = len(gt_rel_pairs[imgid])
                    all_rel_reps = self.coarse_relation_feature_extractor(all_pairs, relation_feature_map,
                                                                          outputs_coord[-1, imgid].detach(),
                                                                          inst_repr[imgid],
                                                                          obj_label_logits=outputs_class[-1, imgid],
                                                                          image=images[imgid],
                                                                          idx=imgid)
                    p_relation_exist_logits = self.relation_proposal_mlp(all_rel_reps)

                    gt_inds = torch.arange(gt_pair_count).to(p_relation_exist_logits.device)
                    _, sort_rel_inds = p_relation_exist_logits[gt_pair_count:].squeeze(1).sort(descending=True)
                    # _, sort_rel_inds = torch.cat([inst_scores[all_pairs[:, 1:]], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1)[gt_pair_count:].sort(descending=True)
                    sampled_rel_inds = torch.cat([gt_inds, sort_rel_inds + gt_pair_count])[:self.args.num_hoi_queries]

                    sampled_rel_pairs = all_pairs[sampled_rel_inds]
                    sampled_rel_reps = all_rel_reps[sampled_rel_inds]


                    sampled_rel_reps = sampled_rel_reps.squeeze(0)
                    sampled_actions = self.sampled_actions(sampled_rel_reps)
                    sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]
                else:
                    # random sampling
                    sampled_neg_inds = torch.randperm(len(rel_pairs))
                    sampled_rel_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs[sampled_neg_inds]], dim=0)[
                                        :self.args.num_hoi_queries]
                    sampled_rel_reps = self.coarse_relation_feature_extractor(sampled_rel_pairs, relation_feature_map,
                                                                              outputs_coord[-1, imgid].detach(),
                                                                              inst_repr[imgid],
                                                                              obj_label_logits=outputs_class[-1, imgid],
                                                                              idx=imgid)
                    sampled_rel_pred_exists = self.relation_proposal_mlp(sampled_rel_reps).squeeze(1)
            else:
                rel_pairs = rel_mat.nonzero(as_tuple=False)
                rel_reps = self.coarse_relation_feature_extractor(rel_pairs, relation_feature_map,
                                                                  outputs_coord[-1, imgid].detach(), inst_repr[imgid],
                                                                  obj_label_logits=outputs_class[-1, imgid], idx=imgid)
                p_relation_exist_logits = self.relation_proposal_mlp(rel_reps)

                _, sort_rel_inds = p_relation_exist_logits.squeeze(1).sort(descending=True)
                # _, sort_rel_inds = torch.cat([inst_scores[rel_pairs[:, 1:]], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1).sort(descending=True)
                sampled_rel_inds = sort_rel_inds[:self.args.num_hoi_queries]

                sampled_rel_pairs = rel_pairs[sampled_rel_inds]
                sampled_rel_reps = rel_reps[sampled_rel_inds]

                sampled_rel_reps = sampled_rel_reps.squeeze(0)
                sampled_actions = self.sampled_actions(sampled_rel_reps)
                sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]
            #normlized boxes cxcywh
            pred_boxes_img = outputs_coord[-1, imgid]#normlized boxes cxcywh
            pred_boxes_xyxy_img = box_ops.box_cxcywh_to_xyxy(pred_boxes_img).clamp(0, 1)
            raw_node_mask = inst_scores > self.graph_node_score_thresh
            node_mask = self.build_graph_node_mask(inst_scores, sampled_rel_pairs, num_nodes)
            proposal_endpoint_mask = torch.zeros_like(node_mask)
            if sampled_rel_pairs.numel() > 0:
                proposal_endpoint_mask[sampled_rel_pairs.reshape(-1)] = True
            #detected object indices (exclude background)
            detected_obj_inds = node_mask.nonzero(as_tuple=False).squeeze(1)
            #human and object inds/boxes (find detr for proposals)
            proposal_subj_inds = sampled_rel_pairs[:, 0]
            proposal_obj_inds = sampled_rel_pairs[:, 1]
            proposal_human_boxes = pred_boxes_xyxy_img[proposal_subj_inds]
            proposal_object_boxes = pred_boxes_xyxy_img[proposal_obj_inds]
            proposal_union_boxes = torch.cat([
                torch.min(proposal_human_boxes[:, :2], proposal_object_boxes[:, :2]),
                torch.max(proposal_human_boxes[:, 2:], proposal_object_boxes[:, 2:]),
            ], dim=-1)
            node_detr_features = inst_repr[imgid]#this image query feature
            node_blip_roi_features = self.pool_blip_patch_tokens(blip_patches.squeeze(0), pred_boxes_xyxy_img)
            proposal_union_imgs = self.make_cropped_region_imgs_from_boxes(
                proposal_union_boxes,
                img_tensor_blip.to(device=proposal_union_boxes.device),
            )
            proposal_union_blip_features = self.blip(proposal_union_imgs, 'None', 'itc')
            proposal_human_blip_features = node_blip_roi_features[proposal_subj_inds]
            proposal_object_blip_features = node_blip_roi_features[proposal_obj_inds]
            if self.use_edge_blip_fusion:
                sampled_rel_reps = self.fuse_edge_blip_features(
                    sampled_rel_reps,
                    proposal_human_blip_features,
                    proposal_object_blip_features,
                    proposal_union_blip_features,
                    blip_global_context.expand(sampled_rel_reps.shape[0], -1),
                )
            sampled_actions = self.sampled_actions(sampled_rel_reps)
            #define node types and masks for graph construction
            node_types = torch.ones(num_nodes, dtype=torch.long, device=pred_boxes_xyxy_img.device)
            node_types[inst_labels == 1] = 0#0为human, 1为object
            graph_input = {
                "blip_action": blip_action.squeeze(0),#cls token as global feature
                "blip_scene_action": blip_scene_action.squeeze(0),#raw-image cls token as scene context
                "blip_global_context": blip_global_context.squeeze(0),#augmented sample + raw-image context
                "blip_global_patches": blip_global_patches.squeeze(0),#raw-image patch tokens as global scene patches
                "blip_patches": blip_patches.squeeze(0),#pathes tokens
                "node_detr_features": node_detr_features,#detr query features for all nodes
                "node_blip_roi_features": node_blip_roi_features,#blip roi features for all nodes
                "node_boxes": pred_boxes_xyxy_img,#normlized boxes in xyxy format for all nodes
                "node_classes": inst_labels,#predicted classes for all nodes
                "node_types": node_types,#human 0 or object 1
                "node_mask": node_mask,#mask for valid nodes(True or False )
                "proposal_pairs": sampled_rel_pairs,#(关系提议的主体和客体节点索引对)
                "proposal_scores": sampled_rel_pred_exists,#(关系提议的存在概率)
                "proposal_pair_features": sampled_rel_reps,#(关系提议的特征表示)
                #proposals class and box
                "proposal_human_boxes": proposal_human_boxes,
                "proposal_human_classes": inst_labels[proposal_subj_inds],
                "proposal_human_blip_features": proposal_human_blip_features,
                "proposal_object_boxes": proposal_object_boxes,
                "proposal_object_classes": inst_labels[proposal_obj_inds],
                "proposal_object_blip_features": proposal_object_blip_features,
                "proposal_union_boxes": proposal_union_boxes,
                "proposal_union_blip_features": proposal_union_blip_features,
                "detected_object_indices": detected_obj_inds,
                "detected_object_boxes": pred_boxes_xyxy_img[detected_obj_inds],
                "detected_object_classes": inst_labels[detected_obj_inds],
                "detected_object_detr_features": node_detr_features[detected_obj_inds],
                "detected_object_blip_roi_features": node_blip_roi_features[detected_obj_inds],
            }
            graph_inputs.append(graph_input)

            graph_input_debug.append({
                "image_index": imgid,
                "proposal_count": int(sampled_rel_pairs.shape[0]),
                "detected_object_count": int(detected_obj_inds.shape[0]),
                "blip_image_source": "samples_valid_region",
                "blip_scene_source": "raw_image",
                "blip_valid_size": blip_valid_size,
                "blip_input_shape": list(img_tensor_blip.shape),
                "blip_scene_input_shape": list(img_tensor_scene.shape),
                "graph_node_score_thresh": float(self.graph_node_score_thresh),
                "shape_summary": {
                    key: list(value.shape) for key, value in graph_input.items()
                },
                "consistency": {
                    "proposal_pairs_is_nx2": sampled_rel_pairs.dim() == 2 and sampled_rel_pairs.shape[1] == 2,
                    "proposal_human_boxes_matches_pairs": graph_input["proposal_human_boxes"].shape[0] == sampled_rel_pairs.shape[0],
                    "proposal_object_boxes_matches_pairs": graph_input["proposal_object_boxes"].shape[0] == sampled_rel_pairs.shape[0],
                    "proposal_human_classes_matches_pairs": graph_input["proposal_human_classes"].shape[0] == sampled_rel_pairs.shape[0],
                    "proposal_object_classes_matches_pairs": graph_input["proposal_object_classes"].shape[0] == sampled_rel_pairs.shape[0],
                    "proposal_pair_features_matches_pairs": graph_input["proposal_pair_features"].shape[0] == sampled_rel_pairs.shape[0],
                    "proposal_union_blip_features_matches_pairs": graph_input["proposal_union_blip_features"].shape[0] == sampled_rel_pairs.shape[0],
                    "detected_object_boxes_matches_indices": graph_input["detected_object_boxes"].shape[0] == detected_obj_inds.shape[0],
                    "detected_object_classes_matches_indices": graph_input["detected_object_classes"].shape[0] == detected_obj_inds.shape[0],
                    "blip_action_is_1d": graph_input["blip_action"].dim() == 1,
                    "blip_scene_action_is_1d": graph_input["blip_scene_action"].dim() == 1,
                    "blip_global_patches_is_2d": graph_input["blip_global_patches"].dim() == 2,
                    "blip_patches_is_2d": graph_input["blip_patches"].dim() == 2,
                    "node_detr_features_is_2d": graph_input["node_detr_features"].dim() == 2,
                    "node_blip_roi_features_is_2d": graph_input["node_blip_roi_features"].dim() == 2,
                }
            })

            # >>>>>>>>>>>> relation classification <<<<<<<<<<<<<<<
            query_reps = self.rel_query_pre_proj(sampled_rel_reps).unsqueeze(1)# (query, 1, dim)
            if self.use_visual_scene_graph:
                graph_out = self.visual_scene_graph(
                    node_feats=node_detr_features.unsqueeze(0),
                    node_blip_feats=node_blip_roi_features.unsqueeze(0),
                    boxes=pred_boxes_xyxy_img.unsqueeze(0),
                    labels=inst_labels.unsqueeze(0),
                    node_types=node_types.unsqueeze(0),
                    node_mask=node_mask.unsqueeze(0),
                    pair_indices=sampled_rel_pairs.unsqueeze(0),
                    pair_scores=sampled_rel_pred_exists.unsqueeze(0),
                    pair_feats=query_reps.squeeze(1).unsqueeze(0),
                    union_feats=proposal_union_blip_features.unsqueeze(0),
                    global_feat=blip_global_context,
                )
                graph_pair_feats = graph_out.pair_feats.squeeze(0)
                query_reps = query_reps + torch.tanh(self.graph_gate) * graph_pair_feats.unsqueeze(1)
                graph_reasoner_debug.append(graph_out.debug_info)
                if self.graph_debug:
                    graph_debug_stats.append({
                        "valid_nodes": int(node_mask.sum().item()),
                        "raw_score_nodes": int(raw_node_mask.sum().item()),
                        "proposal_endpoint_nodes": int(proposal_endpoint_mask.sum().item()),
                        "forced_nodes": int((node_mask & proposal_endpoint_mask & ~raw_node_mask).sum().item()),
                        "e_ho_edges": int(graph_out.debug_info.get('e_ho_edge_count', 0)),
                        "e_oo_edges": int(graph_out.debug_info.get('e_oo_edge_count', 0)),
                        "raw_nodes_t001": int((inst_scores > 0.01).sum().item()),
                        "raw_nodes_t003": int((inst_scores > 0.03).sum().item()),
                        "raw_nodes_t005": int((inst_scores > 0.05).sum().item()),
                        "raw_nodes_t007": int((inst_scores > 0.07).sum().item()),
                        "raw_nodes_t010": int((inst_scores > 0.10).sum().item()),
                        "pair_mean": float(graph_pair_feats.mean().item()),
                        "pair_std": float(graph_pair_feats.std().item()),
                        "text_sync_mode": graph_out.debug_info.get('text_sync_mode', 'disabled'),
                        "text_dense_total_edge_update_count": int(graph_out.debug_info.get('text_dense_total_edge_update_count', 0)),
                        "text_dense_total_node_update_count": int(graph_out.debug_info.get('text_dense_total_node_update_count', 0)),
                        "text_layer_count": int(graph_out.debug_info.get('text_layer_count', 0)),
                        "text_context_norm_mean": float(graph_out.debug_info.get('text_context_norm_mean', 0.0)),
                        "candidate_evidence_applied": int(bool(graph_out.debug_info.get("candidate_evidence_applied", False))),
                        "candidate_evidence_valid_pair_count": int(graph_out.debug_info.get("candidate_evidence_valid_pair_count", 0)),
                        "candidate_evidence_context_path_count_mean": float(graph_out.debug_info.get("candidate_evidence_context_path_count_mean", 0.0)),
                        "candidate_evidence_anchor_delta_norm_mean": float(graph_out.debug_info.get("candidate_evidence_anchor_delta_norm_mean", 0.0)),
                        "prototype_valid_count": int(graph_out.debug_info.get("prototype_valid_count", 0)),
                        "prototype_slot_valid_ratio": float(graph_out.debug_info.get("prototype_slot_valid_ratio", graph_out.debug_info.get("text_prototype_slot_valid_ratio", 0.0))),
                        "sync_evidence_applied": int(bool(graph_out.debug_info.get("sync_evidence_applied", False))),
                        "sync_valid_pair_count": int(graph_out.debug_info.get("sync_valid_pair_count", 0)),
                        "sync_valid_prototype_count": int(graph_out.debug_info.get("sync_valid_prototype_count", 0)),
                        "sync_anchor_delta_norm_mean": float(graph_out.debug_info.get("sync_anchor_delta_norm_mean", 0.0)),
                        "sync_has_nan": int(bool(graph_out.debug_info.get("sync_has_nan", False))),
                        "layerwise_sync_applied": int(bool(graph_out.debug_info.get("layerwise_sync_applied", False))),
                        "layerwise_sync_layer_count": int(graph_out.debug_info.get("layerwise_sync_layer_count", 0)),
                        "layerwise_sync_total_anchor_delta_norm": float(graph_out.debug_info.get("layerwise_sync_total_anchor_delta_norm", 0.0)),
                        "layerwise_sync_has_nan": int(bool(graph_out.debug_info.get("layerwise_sync_has_nan", False))),
                        "layerwise_sync_enabled": int(bool(graph_out.debug_info.get("layerwise_sync_enabled", False))),
                    })
            query_perm = query_reps.permute(1, 0, 2)#(1,q,dim)


            if self.args.no_interaction_decoder:
                outs = query_reps.unsqueeze(0)
            else:
                query_pos_encoding, relation_dependency_encodings, layout_encodings, memory_union_mask, tgt_mask = None, None, None, None, None
                subj_mask, obj_mask, union_mask, _ = self.generate_layout_masks(sampled_rel_pairs, memory_input_mask,
                                                                                outputs_coord[-1, imgid], idx=imgid)
                if self.args.use_relation_tgt_mask:#False
                    tgt_mask = (torch.diag(sampled_rel_pred_exists) != 0)
                    attend_ids = sampled_rel_pred_exists.sort(descending=True)[1][
                                 :self.args.use_relation_tgt_mask_attend_topk]
                    tgt_mask[:, attend_ids] = True
                    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1,
                                                                                                      float(0.0))
                if self.args.use_query_fourier_encoding:#False
                    query_coords = self.fourier_feature_embedding(
                        outputs_coord[-1, imgid][sampled_rel_pairs].view(len(sampled_rel_pairs), 8, 1)) / np.sqrt(
                        self.args.hidden_dim / 2)
                    query_pos_encoding = self.fourier_mlp(
                        torch.cat([torch.cos(query_coords), torch.sin(query_coords)], dim=-1)).view(
                        len(sampled_rel_pairs), -1).unsqueeze(1)
                if self.args.use_relation_dependency_encoding:#True
                    dependency_map = torch.zeros((len(sampled_rel_pairs), len(sampled_rel_pairs))).to(
                        sampled_rel_reps.device).long()  # independent: 0
                    dependency_map[sampled_rel_pairs[:, 0].unsqueeze(1) == sampled_rel_pairs[:, 0].unsqueeze(
                        0)] = 1  # same_subj: 1
                    dependency_map[
                        sampled_rel_pairs[:, 1].unsqueeze(1) == sampled_rel_pairs[:, 1].unsqueeze(0)] = 2  # same_obj: 2
                    dependency_map[
                        sampled_rel_pairs[:, 0].unsqueeze(1) == sampled_rel_pairs[:, 1].unsqueeze(0)] = 3  # subj=obj: 3
                    dependency_map[
                        sampled_rel_pairs[:, 1].unsqueeze(1) == sampled_rel_pairs[:, 0].unsqueeze(0)] = 4  # obj=subj: 4
                    dependency_map.fill_diagonal_(5)  # self: 5
                    relation_dependency_encodings = self.relation_dependency_embeddings(dependency_map)
                    relation_dependency_encodings = self.relation_dependency_content_aware_mapping(
                        torch.cat([query_reps.permute(1, 0, 2).expand(*relation_dependency_encodings.shape),
                                   relation_dependency_encodings], dim=-1)
                    ).unsqueeze(2)  # (#query, #query, batch size, dim)
                if self.args.use_memory_union_mask:#False
                    memory_union_mask = union_mask.flatten(1)
                # if self.args.use_memory_layout_encoding:
                #     layout_map = (~union_mask).long() + (~memory_input_mask[imgid:imgid + 1]).long() + (
                #         ~subj_mask).long() + (~obj_mask).long() * 2
                #     layout_encodings = self.layout_embeddings(layout_map)
                #     layout_encodings = self.layout_content_aware_mapping(
                #         torch.cat([memory_input[imgid:imgid + 1].permute(0, 2, 3, 1).expand(*layout_encodings.shape),
                #                    layout_encodings], dim=-1)
                #     ).flatten(start_dim=1, end_dim=2).unsqueeze(2)  # (#query, #memory, batch size, dim)
                # text encoder hico_lib
                # hicolib = hico_lib.hico_lib
                # text_lib = np.zeros((520,3,256))
                # num = 0
                # for data in hicolib:
                #     caption = hicolib[data]
                #     text_lib_embedding = self.blip(None, caption, 'text_encoding')
                #     text_lib[num] = text_lib_embedding.cpu().numpy()
                #     num = num+1
                # text_lib = text_lib.reshape((-1,256))
                # np.savetxt('hico_lib.txt',text_lib,delimiter=',')
                # cross text encoding and image encoding
                # text_embedding = self.text_encoder.to('cuda')
                # text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
                # text_embedding = self.text_encoder_proj(text_embedding.reshape(520, -1))
                detr_mem_flat = memory_input[imgid:imgid + 1].flatten(2).permute(2, 0, 1)  # (h*w, 1, 256)
                detr_pos_flat = memory_pos[imgid:imgid + 1].flatten(2).permute(2, 0, 1)# (h*w, 1, 256)
                memory_key_padding_mask = memory_input_mask[imgid:imgid + 1].flatten(1)
                if self.use_blip_global_memory:
                    blip_mem_flat = self.blip_memory_proj(blip_global_patches).permute(1, 0, 2)
                    blip_pos_flat = torch.zeros_like(blip_mem_flat)
                    blip_padding_mask = torch.zeros(
                        (1, blip_mem_flat.shape[0]),
                        dtype=torch.bool,
                        device=memory_key_padding_mask.device,
                    )
                    decoder_memory = torch.cat([blip_mem_flat, detr_mem_flat], dim=0)
                    decoder_pos = torch.cat([blip_pos_flat, detr_pos_flat], dim=0)
                    memory_key_padding_mask = torch.cat([blip_padding_mask, memory_key_padding_mask], dim=1)
                    if memory_union_mask is not None:
                        blip_memory_mask = torch.zeros(
                            (memory_union_mask.shape[0], blip_mem_flat.shape[0]),
                            dtype=memory_union_mask.dtype,
                            device=memory_union_mask.device,
                        )
                        memory_union_mask = torch.cat([blip_memory_mask, memory_union_mask], dim=1)
                else:
                    decoder_memory = detr_mem_flat
                    decoder_pos = detr_pos_flat


                outs = self.interaction_decoder(tgt=query_reps,
                                                tgt_mask=tgt_mask,
                                                 memory_mask=memory_union_mask,
                                                 query_pos=query_pos_encoding,
                                                 query_structure_encoding=relation_dependency_encodings,
                                                 # inter-ineraction semantic structure
                                                 memory=decoder_memory,
                                                 memory_key_padding_mask=memory_key_padding_mask,
                                                 pos=decoder_pos,
                                                 memory_role_embedding=None,  # intra-ineraction spatial structure
                                                 memory_value_embedding=None)

            action_logits = self.action_embed(outs)
            pred_sampled_actions.append(sampled_actions)
            pred_rel_pairs.append(sampled_rel_pairs)
            pred_actions.append(action_logits)
            pred_rel_exists.append(sampled_rel_pred_exists)

        hoi_recognition_time = time.time() - start_time
        self._maybe_print_graph_debug_summary(graph_debug_stats)
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "graph_inputs": graph_inputs,
            "graph_input_debug": graph_input_debug,
            "graph_reasoner_debug": graph_reasoner_debug,
            "pred_rel_pairs": pred_rel_pairs,
            "pred_actions": [p[-1].squeeze(1) for p in pred_actions],
            "pred_sampled_actions": pred_sampled_actions,
            "pred_action_exists": pred_rel_exists,
            "det2gt_indices": det2gt_indices,
            "hoi_recognition_time": hoi_recognition_time,
        }
        if self.args.hoi_aux_loss: out['hoi_aux_outputs'] = self._set_hoi_aux_loss(pred_actions)
        if self.args.train_detr and self.args.aux_loss: out['aux_outputs'] = self._set_aux_loss(outputs_class,
                                                                                                outputs_coord)

        return out

    def _maybe_print_graph_debug_summary(self, stats):
        if not self.graph_debug or len(stats) == 0 or get_rank() != 0:
            return
        if self._graph_debug_prints >= self.graph_debug_max_prints:
            return
        should_print = self._graph_debug_steps % max(int(self.graph_debug_interval), 1) == 0
        self._graph_debug_steps += 1
        if not should_print:
            return

        def summarize(key):
            values = [s[key] for s in stats]
            return min(values), sum(values) / len(values), max(values)

        def mean_key(key, default=0.0):
            vals = [float(s.get(key, default)) for s in stats]
            return sum(vals) / len(vals) if vals else 0.0

        valid_min, valid_mean, valid_max = summarize("valid_nodes")
        raw_min, raw_mean, raw_max = summarize("raw_score_nodes")
        endpoint_min, endpoint_mean, endpoint_max = summarize("proposal_endpoint_nodes")
        forced_min, forced_mean, forced_max = summarize("forced_nodes")
        ho_min, ho_mean, ho_max = summarize("e_ho_edges")
        oo_min, oo_mean, oo_max = summarize("e_oo_edges")
        text_sync_modes = sorted({str(s.get("text_sync_mode", "disabled")) for s in stats})
        text_edge_updates = sum(s.get("text_dense_total_edge_update_count", 0) for s in stats) / len(stats)
        text_node_updates = sum(s.get("text_dense_total_node_update_count", 0) for s in stats) / len(stats)
        text_layer_count = sum(s.get("text_layer_count", 0) for s in stats) / len(stats)
        text_context_norm = sum(s.get("text_context_norm_mean", 0.0) for s in stats) / len(stats)
        print(
            "[VisualSceneGraph Debug] "
            f"step={self._graph_debug_steps}, "
            f"thresh={float(self.graph_node_score_thresh):.3f}, "
            f"batch={len(stats)}, "
            f"valid_nodes(min/mean/max)={valid_min}/{valid_mean:.1f}/{valid_max}, "
            f"raw_score_nodes={raw_min}/{raw_mean:.1f}/{raw_max}, "
            f"proposal_endpoint_nodes={endpoint_min}/{endpoint_mean:.1f}/{endpoint_max}, "
            f"forced_nodes={forced_min}/{forced_mean:.1f}/{forced_max}, "
            f"e_ho_edges={ho_min}/{ho_mean:.1f}/{ho_max}, "
            f"e_oo_edges={oo_min}/{oo_mean:.1f}/{oo_max}, "
            f"raw@0.01={summarize('raw_nodes_t001')[1]:.1f}, "
            f"raw@0.03={summarize('raw_nodes_t003')[1]:.1f}, "
            f"raw@0.05={summarize('raw_nodes_t005')[1]:.1f}, "
            f"raw@0.07={summarize('raw_nodes_t007')[1]:.1f}, "
            f"raw@0.10={summarize('raw_nodes_t010')[1]:.1f}, "
            f"pair_mean={sum(s['pair_mean'] for s in stats) / len(stats):.6f}, "
            f"pair_std={sum(s['pair_std'] for s in stats) / len(stats):.6f}, "
            f"text_sync={','.join(text_sync_modes)}, "
            f"text_layers={text_layer_count:.1f}, "
            f"text_dense_edge_updates={text_edge_updates:.1f}, "
            f"text_dense_node_updates={text_node_updates:.1f}, "
            f"text_context_norm={text_context_norm:.6f}, "
            f"ceti_applied={mean_key('candidate_evidence_applied'):.2f}, "
            f"ceti_paths={mean_key('candidate_evidence_context_path_count_mean'):.2f}, "
            f"ceti_delta={mean_key('candidate_evidence_anchor_delta_norm_mean'):.6f}, "
            f"proto_valid={mean_key('prototype_valid_count'):.1f}, "
            f"proto_slot_ratio={mean_key('prototype_slot_valid_ratio'):.4f}, "
            f"sync_applied={mean_key('sync_evidence_applied'):.2f}, "
            f"sync_proto={mean_key('sync_valid_prototype_count'):.1f}, "
            f"sync_delta={mean_key('sync_anchor_delta_norm_mean'):.6f}, "
            f"sync_nan={mean_key('sync_has_nan'):.1f}, "
            f"layerwise_sync={mean_key('layerwise_sync_applied'):.2f}, "
            f"layerwise_layers={mean_key('layerwise_sync_layer_count'):.1f}, "
            f"layerwise_delta={mean_key('layerwise_sync_total_anchor_delta_norm'):.6f}, "
            f"layerwise_nan={mean_key('layerwise_sync_has_nan'):.1f}, "
            f"graph_gate={self.graph_gate.item():.6f}"
        )
        self._graph_debug_prints += 1

    @torch.jit.unused
    def _set_hoi_aux_loss(self, pred_actions):
        return [{'pred_actions': [p[l].squeeze(1) for p in pred_actions]} for l in range(self.args.hoi_dec_layers - 1)]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": l, "pred_boxes": b} for l, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def fuse_edge_blip_features(self, rel_reps, human_feats, object_feats, union_feats, global_feats):
        gate_input = torch.cat([human_feats, object_feats, union_feats, global_feats], dim=-1)
        human_gate, object_gate, union_gate = self.edge_blip_gate(gate_input).chunk(3, dim=-1)
        blip_edge_feats = self.edge_blip_proj(torch.cat([
            human_feats * human_gate,
            object_feats * object_gate,
            union_feats * union_gate,
        ], dim=-1))
        return self.edge_rel_proj(torch.cat([rel_reps, blip_edge_feats], dim=-1))

    # merge boxes (NMS)
    def apply_nms(self, inst_scores, inst_labels, cxcywh_boxes, threshold=0.7):
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(cxcywh_boxes)
        box_areas = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(-1)
        box_area_sum = box_areas.unsqueeze(1) + box_areas.unsqueeze(0)

        union_boxes = torch.cat([torch.min(xyxy_boxes.unsqueeze(1)[:, :, :2], xyxy_boxes.unsqueeze(0)[:, :, :2]),
                                 torch.max(xyxy_boxes.unsqueeze(1)[:, :, 2:], xyxy_boxes.unsqueeze(0)[:, :, 2:])],
                                dim=-1)
        union_area = (union_boxes[:, :, 2:] - union_boxes[:, :, :2]).prod(-1)
        iou = torch.clamp(box_area_sum - union_area, min=0) / union_area
        box_match_mat = torch.logical_and(iou > threshold, inst_labels.unsqueeze(1) == inst_labels.unsqueeze(0))

        suppress_ids = []
        for box_match in box_match_mat:
            group_ids = box_match.nonzero(as_tuple=False).squeeze(1)
            if len(group_ids) > 1:
                max_score_inst_id = group_ids[inst_scores[group_ids].argmax()]
                bg_ids = group_ids[group_ids != max_score_inst_id]
                suppress_ids.append(bg_ids)
                box_match_mat[:, bg_ids] = False
        if len(suppress_ids) > 0:
            suppress_ids = torch.cat(suppress_ids, dim=0)
        return suppress_ids

    def build_blip_tensor_from_sample(self, sample_tensor, sample_mask, image_size=384):
        """Build BLIP input from the same augmented image seen by DETR."""
        detr_mean = sample_tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        detr_std = sample_tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        blip_mean = sample_tensor.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        blip_std = sample_tensor.new_tensor([0.26862854, 0.26130258, 0.27577711]).view(3, 1, 1)

        valid_pixels = (~sample_mask).nonzero(as_tuple=False)
        if valid_pixels.numel() == 0:
            valid_h, valid_w = sample_tensor.shape[-2:]
        else:
            valid_h, valid_w = (valid_pixels.max(dim=0)[0] + 1).tolist()

        valid_img = sample_tensor[:, :valid_h, :valid_w]
        valid_img = (valid_img * detr_std + detr_mean).clamp(0, 1)
        valid_img = F.interpolate(
            valid_img.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        blip_img = (valid_img - blip_mean) / blip_std
        return blip_img, [valid_h, valid_w]

    def normalize_blip_image_tensor(self, image_tensor):
        blip_mean = image_tensor.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        blip_std = image_tensor.new_tensor([0.26862854, 0.26130258, 0.27577711]).view(3, 1, 1)
        return (image_tensor - blip_mean) / blip_std

    def build_blip_tensor_from_image(self, image, image_size=384, device=None):
        """Build BLIP input from the raw PIL image returned by the dataset."""
        if isinstance(image, Image.Image):
            image_np = np.asarray(image.resize((image_size, image_size), Image.BICUBIC), dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        elif torch.is_tensor(image):
            image_tensor = image
            if image_tensor.dim() != 3:
                raise ValueError(f"image tensor must be [3,H,W], got {tuple(image_tensor.shape)}")
            if image_tensor.max() > 1:
                image_tensor = image_tensor.float() / 255.0
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            raise TypeError(f"Unsupported image type for BLIP preprocessing: {type(image)}")
        if device is not None:
            image_tensor = image_tensor.to(device)
        image_tensor = image_tensor.to(dtype=torch.float32).clamp(0, 1)
        return self.normalize_blip_image_tensor(image_tensor)

    def pool_blip_patch_tokens(self, blip_patches, boxes_xyxy, output_size=None):
        """Pool BLIP whole-image patch tokens by normalized xyxy boxes."""
        if boxes_xyxy.numel() == 0:
            return blip_patches.new_zeros((0, blip_patches.shape[-1]))
        if output_size is None:
            output_size = self.blip_roi_output_size
        patch_count, channels = blip_patches.shape
        grid_size = int(np.sqrt(patch_count))
        if grid_size * grid_size != patch_count:
            raise ValueError(f"BLIP patch token count must form a square grid, got {patch_count}")
        patch_map = blip_patches.transpose(0, 1).reshape(1, channels, grid_size, grid_size)
        scaled_boxes = torch.cat([
            boxes_xyxy.new_zeros((boxes_xyxy.shape[0], 1)),
            boxes_xyxy.clamp(0, 1) * boxes_xyxy.new_tensor([grid_size, grid_size, grid_size, grid_size]).unsqueeze(0),
        ], dim=-1)
        roi_feats = roi_align(patch_map, scaled_boxes, output_size=output_size, sampling_ratio=2, aligned=True)
        return self.blip_roi_pooler(roi_feats.flatten(start_dim=1))

    def make_cropped_region_imgs_from_boxes(self, boxes_xyxy, scaled_img, image_size=384):
        """Crop normalized boxes from a BLIP-normalized image and resize each crop for BLIP."""
        if boxes_xyxy.numel() == 0:
            return scaled_img.new_zeros((0, *scaled_img.shape))
        blip_mean = scaled_img.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        blip_std = scaled_img.new_tensor([0.26862854, 0.26130258, 0.27577711]).view(3, 1, 1)
        image_01 = (scaled_img * blip_std + blip_mean).clamp(0, 1)
        _, height, width = scaled_img.shape
        scaled_boxes = boxes_xyxy.clamp(0, 1) * boxes_xyxy.new_tensor([width, height, width, height]).unsqueeze(0)
        x1y1 = torch.floor(scaled_boxes[:, :2]).long()
        x2y2 = torch.ceil(scaled_boxes[:, 2:]).long()
        x1y1[:, 0].clamp_(0, width - 1)
        x1y1[:, 1].clamp_(0, height - 1)
        x2y2[:, 0].clamp_(1, width)
        x2y2[:, 1].clamp_(1, height)
        crops = []
        min_crop = 1
        for (x1, y1), (x2, y2) in zip(x1y1.tolist(), x2y2.tolist()):
            x2 = max(x2, x1 + min_crop)
            y2 = max(y2, y1 + min_crop)
            x2 = min(x2, width)
            y2 = min(y2, height)
            crop = image_01[:, y1:y2, x1:x2]
            crop = F.interpolate(
                crop.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            crops.append(self.normalize_blip_image_tensor(crop))
        return torch.stack(crops, dim=0)

    def make_region_imgs_from_boxes(self, boxes_xyxy, scaled_img):
        return self.make_cropped_region_imgs_from_boxes(boxes_xyxy, scaled_img)

    def build_graph_node_mask(self, inst_scores, sampled_rel_pairs, num_nodes):
        node_mask = inst_scores > self.graph_node_score_thresh
        if self.graph_min_score_nodes > 0 and int(node_mask.sum().item()) < self.graph_min_score_nodes:
            keep_count = min(int(self.graph_min_score_nodes), num_nodes)
            keep = inst_scores.topk(k=keep_count).indices
            node_mask[keep] = True
        if self.graph_max_extra_nodes > 0 and int(node_mask.sum().item()) > self.graph_max_extra_nodes:
            keep = inst_scores.topk(k=self.graph_max_extra_nodes).indices
            limited_mask = torch.zeros_like(node_mask)
            limited_mask[keep] = True
            node_mask = limited_mask
        if sampled_rel_pairs.numel() > 0:
            node_mask[sampled_rel_pairs.reshape(-1)] = True
        if not bool(node_mask.any()):
            node_mask[inst_scores.argmax()] = True
        return node_mask

    def class_transform(self,output_class):

        _,bz ,num ,class_num = output_class.shape
        output_class_trans = torch.zeros([_,bz,num,92],dtype=torch.float32).to('cuda')
        output_class_trans[:,:,:,[1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90,91]] \
            = output_class[:,:,:,[0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80]]
        return output_class_trans

    def generate_layout_masks(self, rel_pairs, feature_masks, boxes, idx):
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0, 1)
        head_boxes = xyxy_boxes[rel_pairs[:, 0]]
        tail_boxes = xyxy_boxes[rel_pairs[:, 1]]
        union_boxes = torch.cat([
            torch.min(head_boxes[:, :2], tail_boxes[:, :2]),
            torch.max(head_boxes[:, 2:], tail_boxes[:, 2:])
        ], dim=1)

        h, w = (~feature_masks[idx]).nonzero(as_tuple=False).max(dim=0)[0] + 1  # mask: image area=False, pad area=True
        scaled_head_boxes = head_boxes * torch.tensor([w, h, w, h]).to(device=head_boxes.device,
                                                                       dtype=head_boxes.dtype).unsqueeze(0)
        scaled_tail_boxes = tail_boxes * torch.tensor([w, h, w, h]).to(device=tail_boxes.device,
                                                                       dtype=tail_boxes.dtype).unsqueeze(0)
        scaled_union_boxes = union_boxes * torch.tensor([w, h, w, h]).to(device=union_boxes.device,
                                                                         dtype=union_boxes.dtype).unsqueeze(0)
        bound_upper_inds = (torch.tensor([w, h, w, h]) - 1).unsqueeze(0).float().to(feature_masks.device)
        rounded_head_boxes = torch.min(scaled_head_boxes.round(), bound_upper_inds).int()
        rounded_tail_boxes = torch.min(scaled_tail_boxes.round(), bound_upper_inds).int()
        rounded_union_boxes = torch.min(scaled_union_boxes.round(), bound_upper_inds).int()

        role_embeddings = None
        # build masks: hit region=False, other region=True
        rel_head_mask = torch.ones_like(feature_masks[idx]).unsqueeze(0).repeat((len(rel_pairs), 1, 1))
        rel_tail_mask = torch.ones_like(feature_masks[idx]).unsqueeze(0).repeat((len(rel_pairs), 1, 1))
        rel_union_mask = torch.ones_like(feature_masks[idx]).unsqueeze(0).repeat((len(rel_pairs), 1, 1))
        for rid in range(len(rel_union_mask)):
            rel_head_mask[rid, rounded_head_boxes[rid, 1]:rounded_head_boxes[rid, 3] + 1,
            rounded_head_boxes[rid, 0]:rounded_head_boxes[rid, 2] + 1] = False
            rel_tail_mask[rid, rounded_tail_boxes[rid, 1]:rounded_tail_boxes[rid, 3] + 1,
            rounded_tail_boxes[rid, 0]:rounded_tail_boxes[rid, 2] + 1] = False
            rel_union_mask[rid, rounded_union_boxes[rid, 1]:rounded_union_boxes[rid, 3] + 1,
            rounded_union_boxes[rid, 0]:rounded_union_boxes[rid, 2] + 1] = False

        return rel_head_mask, rel_tail_mask, rel_union_mask, role_embeddings

class STIPCriterion(nn.Module):
    """ This class computes the loss for STIP.
    1. proposal loss
    2. relation classification loss
    """

    def __init__(self, args, matcher):
        super().__init__()
        self.args = args
        self.matcher = matcher
        self.weight_dict = {
            'loss_proposal': args.proposal_loss_coef,
            'loss_act': args.action_loss_coef,
            'loss_act_sampled': args.action_sampled_loss_coef,
            'loss_act_fore': args.action_fore_loss_coef,
            'loss_global_act_fore': args.global_action_fore_loss_coef,
            'loss_fea': args.action_fea_loss_coef,
        }
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        if args.hoi_aux_loss:
            for i in range(args.hoi_dec_layers - 1):
                self.weight_dict.update({f'loss_act_{i}': self.weight_dict['loss_act']})

        if args.dataset_file == 'vcoco':
            self.invalid_ids = args.invalid_ids
            self.valid_ids = args.valid_ids
        elif args.dataset_file == 'hico-det':
            self.invalid_ids = []
            self.valid_ids = list(range(self.args.num_actions))
            self.hico_valid_obj_ids = torch.tensor(self.args.valid_obj_ids)

        if args.train_detr:
            self.num_classes = args.num_classes
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = args.eos_coef
            self.register_buffer('empty_weight', empty_weight)

            self.detr_losses = ['labels', 'boxes', 'cardinality']
            det_weights = {'loss_ce': 1 * args.finetune_detr_weight,
                           'loss_bbox': args.bbox_loss_coef * args.finetune_detr_weight,
                           'loss_giou': args.giou_loss_coef * args.finetune_detr_weight}
            if args.aux_loss:
                aux_weights = {}
                for i in range(args.dec_layers - 1):
                    aux_weights.update({k + f'_{i}': v for k, v in det_weights.items()})
                det_weights.update(aux_weights)
            self.weight_dict.update(det_weights)

    #######################################################################################################################
    # * DETR Losses
    #######################################################################################################################
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, log=False):
        # instance matching
        if outputs['det2gt_indices'] is None:
            outputs_without_aux = {k: v for k, v in outputs.items() if (k != 'aux_outputs' and k != 'hoi_aux_outputs')}
            indices = self.matcher(outputs_without_aux, targets)
        else:
            indices = outputs['det2gt_indices']

        # generate relation targets
        all_rel_pair_targets, all_obj_act_targets, all_global_act_targets = [], [], []
        for imgid, (tgt, (det_idxs, gtbox_idxs)) in enumerate(zip(targets, indices)):
            det2gt_map = {int(d): int(g) for d, g in zip(det_idxs, gtbox_idxs)}
            gt_relation_map = tgt['relation_map']
            rel_pairs = outputs['pred_rel_pairs'][imgid]
            rel_pair_targets = torch.zeros((len(rel_pairs), gt_relation_map.shape[-1])).to(gt_relation_map.device)
            obj_act_targets = torch.zeros(100, gt_relation_map.shape[-1]).to(gt_relation_map.device)
            global_act_targets = torch.zeros(1, gt_relation_map.shape[-1]).to(gt_relation_map.device)

            for idx, rel in enumerate(rel_pairs):
                if (int(rel[0]) in det2gt_map) and (int(rel[1]) in det2gt_map):
                    rel_pair_targets[idx] = gt_relation_map[det2gt_map[int(rel[0])], det2gt_map[int(rel[1])]]
            all_rel_pair_targets.append(rel_pair_targets)
            global_act_mask = torch.nonzero(rel_pair_targets.sum(0))
            global_act_targets[0][global_act_mask] = 1
            # all_obj_act_targets.append(obj_act_targets)
            all_global_act_targets.append(global_act_targets)

            for idx in range(100):
                if idx in det2gt_map:
                    obj_act_targets[idx] = gt_relation_map[det2gt_map[int(idx)]].argmax(dim=0)
            all_obj_act_targets.append(obj_act_targets)

        all_rel_pair_targets = torch.cat(all_rel_pair_targets, dim=0)

        prior_verb_label_mask = None
        if self.args.dataset_file == 'hico-det':
            # no_interaction_id = self.args.action_names.index('no_interaction')
            # rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) - all_rel_pair_targets[..., no_interaction_id] > 0).float()
            rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) > 0).float()
            if self.args.use_prior_verb_label_mask:
                pred_obj_labels = outputs['pred_logits'][:, :, self.args.valid_obj_ids].argmax(-1)
                tail_obj_ids = [p[:, 1] for p in outputs['pred_rel_pairs']]
                tail_obj_labels = torch.cat([l[id] for l, id in zip(pred_obj_labels, tail_obj_ids)])
                prior_verb_label_mask = self.args.correct_mat.transpose(0, 1)[tail_obj_labels]
        else:
            rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) > 0).float()

        loss_proposal = self.proposal_loss(torch.cat(outputs['pred_action_exists'], dim=0), rel_proposal_targets)
        loss_action = self.action_loss(torch.cat(outputs['pred_actions'], dim=0)[..., self.valid_ids],
                                       all_rel_pair_targets[..., self.valid_ids], prior_verb_label_mask)
        loss_action_sampled = self.action_loss(torch.cat(outputs['pred_sampled_actions'], dim=0)[..., self.valid_ids],
                                               all_rel_pair_targets[..., self.valid_ids], prior_verb_label_mask)

        loss_dict = {'loss_proposal': loss_proposal, 'loss_act': loss_action, 'loss_act_sampled': loss_action_sampled, }

        if 'outs_features' in outputs:
            target_fea = torch.stack(outputs['outs_features'], 0)
            pre_fea = torch.stack(outputs['hs_features'], 0)
            loss_fea = self.l1_loss(pre_fea.reshape(len(pre_fea) * 32, 256).softmax(0).log(),
                                    target_fea.reshape(len(target_fea) * 32, 256).softmax(0))
            loss_dict.update({'loss_fea': loss_fea})

        if 'hoi_aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['hoi_aux_outputs']):
                aux_loss = {
                    f'loss_act_{i}': self.action_loss(
                        torch.cat(aux_outputs['pred_actions'], dim=0)[..., self.valid_ids],
                        all_rel_pair_targets[..., self.valid_ids], prior_verb_label_mask)
                }
                loss_dict.update(aux_loss)
        
        # jointly train objects and relation decoder
        if self.args.train_detr:
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.detr_losses:
                loss_dict.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.detr_losses:
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        loss_dict.update(l_dict)

        return loss_dict

    # def proposal_loss(self, inputs, targets):
    #     # loss = focal_loss(inputs, targets, gamma=self.args.proposal_focal_loss_gamma, alpha=self.args.proposal_focal_loss_alpha)

    #     ## conventional BCE
    #     # loss_bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    #     # loss = loss_bce[targets<0.5].mean() # neg loss
    #     # if targets.sum() > 0:
    #     #     loss += loss_bce[targets>0.5].mean() # pos loss

    #     # focal loss to balance positive/negative
    #     probs = inputs.sigmoid()
    #     pos_inds = targets.eq(1).float()
    #     neg_inds = targets.lt(1).float()
    #     pos_loss = torch.log(probs) * torch.pow(1 - probs, self.args.proposal_focal_loss_gamma) * pos_inds
    #     neg_loss = torch.log(1 - probs) * torch.pow(probs, self.args.proposal_focal_loss_gamma) * neg_inds
    #     pos_loss = pos_loss.sum()
    #     neg_loss = neg_loss.sum()

    #     # normalize
    #     num_pos = pos_inds.float().sum()
    #     if num_pos == 0:
    #         loss = -neg_loss
    #     else:
    #         loss = -(pos_loss + neg_loss) / num_pos
    #     return loss

    def proposal_loss(self, inputs, targets):
        # Numerically stable focal loss on logits. This avoids log(sigmoid(x))
        # producing inf and then inf * 0 producing NaN for saturated logits.
        gamma = self.args.proposal_focal_loss_gamma
        alpha = self.args.proposal_focal_loss_alpha

        targets = targets.to(dtype=inputs.dtype)
        pos_inds = targets.eq(1).to(dtype=inputs.dtype)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = inputs.sigmoid()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = torch.pow(1 - p_t, gamma)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = bce_loss * focal_weight * alpha_t

        num_pos = pos_inds.sum()
        if num_pos == 0:
            return loss.sum()
        return loss.sum() / num_pos

    def action_loss(self, inputs, targets, prior_verb_label_mask=None):
        # Numerically stable focal loss on logits. This preserves the original
        # positive/negative focal terms while avoiding log(0) for saturated logits.
        gamma = self.args.action_focal_loss_gamma
        targets = targets.to(dtype=inputs.dtype)
        pos_inds = targets.eq(1).to(dtype=inputs.dtype)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = inputs.sigmoid()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * torch.pow(1 - p_t, gamma)
        if prior_verb_label_mask is not None:  # mask invalid predictions
            loss = loss * prior_verb_label_mask.to(dtype=loss.dtype)

        num_pos = pos_inds.sum()
        if num_pos == 0:
            return loss.sum()
        return loss.sum() / num_pos


class STIPPostProcess(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, target_sizes, threshold=0, dataset='coco'):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # for relationship post-processing
        h_indices = [p[:, 0] for p in outputs['pred_rel_pairs']]
        o_indices = [p[:, 1] for p in outputs['pred_rel_pairs']]
        if dataset == 'vcoco':
            prob = F.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1)

            pair_actions = [a.sigmoid() for a in outputs['pred_actions']]
            # pair_actions = outputs['pred_actions'].sigmoid() * outputs['pred_action_exists'].sigmoid() # cls_score ＊　interactiveness score

            results = []
            for batch_idx, (s, l, b) in enumerate(zip(scores, labels, boxes)):
                h_inds = (l == 1) & (s > threshold)
                o_inds = (s > threshold)

                h_box, h_cat = b[h_inds], s[h_inds]
                o_box, o_cat = b[o_inds], s[o_inds]

                # for scenario 1 in v-coco dataset
                o_inds = torch.cat((o_inds, torch.ones(1).type(torch.bool).to(o_inds.device)))
                o_box = torch.cat(
                    (o_box, torch.Tensor([0, 0, 0, 0]).unsqueeze(0).to(o_box.device)))  ## add an empty box

                result_dict = {
                    'h_box': h_box, 'h_cat': h_cat,
                    'o_box': o_box, 'o_cat': o_cat,
                    'scores': s, 'labels': l, 'boxes': b
                }

                K = boxes.shape[1]
                n_act = pair_actions[batch_idx].shape[-1]
                score = torch.zeros((n_act, K, K + 1)).to(pair_actions[batch_idx].device)
                for h_idx, o_idx, pair_action in zip(h_indices[batch_idx], o_indices[batch_idx],
                                                     pair_actions[batch_idx]):
                    if h_idx == o_idx: o_idx = -1  ## special case: head=tail
                    score[:, h_idx, o_idx] = pair_action

                score = score[:, h_inds, :]
                score = score[:, :, o_inds]

                result_dict.update({
                    'pair_score': score,
                    'hoi_recognition_time': outputs['hoi_recognition_time'],
                })

                results.append(result_dict)
        elif dataset == 'hico-det':
            # tail classification score
            _valid_obj_ids = self.args.valid_obj_ids + [self.args.valid_obj_ids[-1] + 1]
            # _valid_obj_ids = self.args.valid_obj_ids
            out_obj_logits = outputs['pred_logits'][..., _valid_obj_ids]
            obj_scores, obj_labels = [], []
            for o_ids, lgts in zip(o_indices, out_obj_logits):
                img_obj_scores, img_obj_labels = F.softmax(lgts[o_ids], -1)[..., :-1].max(-1)
                # img_obj_scores, img_obj_labels = F.softmax(lgts[o_ids], -1)[..., :].max(-1)
                obj_scores.append(img_obj_scores)
                obj_labels.append(img_obj_labels)

            # actions
            out_verb_logits = outputs['pred_actions']
            verb_scores = [l.sigmoid() for l in out_verb_logits]
            # verb_scores = out_verb_logits.sigmoid() * outputs['pred_action_exists'].sigmoid().unsqueeze(-1) # interactiveness

            pair_exist_scores = [s.sigmoid() for s in outputs['pred_action_exists']]
            pair_filter_thr = getattr(self.args, 'pair_filter_threshold', 0.08)
            if pair_filter_thr > 0:
                for i in range(len(verb_scores)):
                    mask = pair_exist_scores[i] < pair_filter_thr
                    verb_scores[i] = verb_scores[i].clone()
                    verb_scores[i][mask] = 0.0

            if getattr(self.args, 'use_pair_score_weighting', False):
                for i in range(len(verb_scores)):
                    verb_scores[i] = verb_scores[i] * pair_exist_scores[i].unsqueeze(-1)

            # accumulate results (iterate through interaction queries)
            results = []
            for batch_idx, (os, ol, vs, box, h_idx, o_idx) in enumerate(
                    zip(obj_scores, obj_labels, verb_scores, boxes, h_indices, o_indices)):
                dedupe_iou_thr = float(getattr(self.args, 'pair_dedupe_iou_thr', 0.55))
                ignore_obj_label = bool(getattr(self.args, 'pair_dedupe_ignore_obj_label', False))
                if h_idx.numel() > 0 and dedupe_iou_thr > 0:
                    qn = h_idx.shape[0]
                    if qn > 1:
                        sb = box[h_idx, :]
                        ob = box[o_idx, :]
                        used = torch.zeros(qn, dtype=torch.bool, device=vs.device)
                        keep_idx = []

                        for i in range(qn):
                            if used[i]:
                                continue
                            group = [i]
                            used[i] = True
                            for j in range(i + 1, qn):
                                if used[j]:
                                    continue
                                if (not ignore_obj_label) and (not bool((ol[i] == ol[j]).item())):
                                    continue

                                iou_h = box_ops.box_iou(sb[i:i + 1], sb[j:j + 1])[0][0, 0]
                                iou_o = box_ops.box_iou(ob[i:i + 1], ob[j:j + 1])[0][0, 0]
                                if iou_h >= dedupe_iou_thr and iou_o >= dedupe_iou_thr:
                                    used[j] = True
                                    group.append(j)

                            if len(group) == 1:
                                keep_idx.append(i)
                            else:
                                group_t = torch.tensor(group, dtype=torch.long, device=vs.device)
                                best_local = group_t[os[group_t].argmax()]
                                vs[best_local] = vs[group_t].max(dim=0).values
                                keep_idx.append(int(best_local.item()))

                        keep_idx_t = torch.tensor(sorted(set(keep_idx)), dtype=torch.long, device=vs.device)
                        h_idx = h_idx[keep_idx_t]
                        o_idx = o_idx[keep_idx_t]
                        os = os[keep_idx_t]
                        ol = ol[keep_idx_t]
                        vs = vs[keep_idx_t]

                valid_obj_ids_tensor = torch.tensor(self.args.valid_obj_ids, device=ol.device)
                # label
                sl = torch.full_like(ol, 0)  # self.subject_category_id = 0 in HICO-DET
                # mapped_ol = valid_obj_ids_tensor[ol]
                l = torch.cat((sl, ol))
                # l = torch.cat((sl, mapped_ol))
                # boxes
                sb = box[h_idx, :]
                ob = box[o_idx, :]
                b = torch.cat((sb, ob))

                vs = vs * os.unsqueeze(1)
                ids = torch.arange(b.shape[0])
                res_dict = {
                    'labels': l.to('cpu'),
                    'boxes': b.to('cpu'),
                    'verb_scores': vs.to('cpu'),
                    'sub_ids': ids[:ids.shape[0] // 2],
                    'obj_ids': ids[ids.shape[0] // 2:],
                    'hoi_recognition_time': outputs['hoi_recognition_time'],
                    'orig_size': torch.tensor([img_h[batch_idx], img_w[batch_idx]])
                }
                results.append(res_dict)

        return results


class RelationFeatureExtractor(nn.Module):
    def __init__(self, args, in_channels, blip=None, resolution=5, out_dim=1024):
        super(RelationFeatureExtractor, self).__init__()
        self.args = args
        self.resolution = resolution

        # head & tail feature (base feature)
        instr_hidden_dim = self.args.hidden_dim
        fusion_dim = instr_hidden_dim * 2

        # spatial feature
        if args.use_spatial_feature:
            spatial_in_dim, spatial_out_dim = 8, 64
            self.spatial_proj = make_fc(spatial_in_dim, spatial_out_dim)
            fusion_dim += spatial_out_dim

        # tail semantic feature
        if args.use_tail_semantic_feature:
            semantic_dim = 300
            self.label_embedding = nn.Embedding(self.args.num_classes + 1, semantic_dim)
            fusion_dim += semantic_dim

        # union feature
        if args.use_union_feature:
            out_ch, union_out_dim = 256, 256
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_ch, kernel_size=1),
                nn.ReLU(inplace=True),
            )  # reduce channel size before pooling
            self.visual_proj = make_fc(out_ch * (resolution ** 2), union_out_dim)
            fusion_dim += union_out_dim

        #LLM prior knowledge integration
        self.prior_feat_dim = 256
        coco_indices = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
            43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 
            85, 86, 87, 88, 89, 90
        ]
        self.register_buffer('coco_map_indices', torch.tensor(coco_indices, dtype=torch.long))
        #load matrix
        try:
            prior_np = np.load("/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/obj_act_prior.npy")
            print(f"[Info] Successfully loaded GPT prior matrix: {prior_np.shape}")
        except FileNotFoundError:
            print("[Warning] 'obj_act_prior.npy' not found! Initializing with ZEROS.")
            prior_np = np.zeros((80, 117), dtype=np.float32)

        #softening
        prior_np = prior_np * 0.8 + 0.1
        #no interaction
        if prior_np.shape[1] > 57:
            prior_np[:, 57] = np.maximum(prior_np[:, 57], 0.15)
            print("[Info] Applied 'No-Interaction' smoothing (ID 57 floor set to 0.15).")
        
        #buffer
        self.register_buffer('obj_act_prior', torch.tensor(prior_np,dtype=torch.float32))
        matrix_action_dim = prior_np.shape[1] 
        self.prior_proj = nn.Linear(matrix_action_dim, self.prior_feat_dim)
        nn.init.normal_(self.prior_proj.weight, std=0.01)
        nn.init.constant_(self.prior_proj.bias, 0)

        fusion_dim += self.prior_feat_dim

        # fusion
        self.fusion_fc = nn.Sequential(
            make_fc(fusion_dim, out_dim), nn.ReLU(),
            make_fc(out_dim, out_dim), nn.ReLU()
        )

    def forward(self, rel_pairs, features, boxes, inst_reprs, idx, image=None, obj_label_logits=None,
                action_label_logits=None, global_action_label_logits=None):
        """pool feature for boxes on one image
            features: dxhxw
            boxes: Nx4 (cx_cy_wh, nomalized to 0-1)
            rel_pairs: Nx2
        """
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0, 1)
        head_boxes = xyxy_boxes[rel_pairs[:, 0]]
        tail_boxes = xyxy_boxes[rel_pairs[:, 1]]
        union_boxes = torch.cat([
            torch.min(head_boxes[:, :2], tail_boxes[:, :2]),
            torch.max(head_boxes[:, 2:], tail_boxes[:, 2:])
        ], dim=1)

        # head & tail features
        head_feats = inst_reprs[rel_pairs[:, 0]]
        tail_feats = inst_reprs[rel_pairs[:, 1]]
        tail_feats[rel_pairs[:, 0] == rel_pairs[:, 1]] = 0  # set to 0 when head==tail for VCOCO (i.e., tail overlapped)

        relation_feats = torch.cat([head_feats, tail_feats], dim=-1)

        # spatial layout feats
        if self.args.use_spatial_feature:
            box_layout_feats = self.extract_spatial_layout_feats(xyxy_boxes)
            rel_spatial_feats = self.spatial_proj(box_layout_feats[rel_pairs[:, 0], rel_pairs[:, 1]])
            relation_feats = torch.cat([relation_feats, rel_spatial_feats], dim=-1)

        # semantic feature
        if self.args.use_tail_semantic_feature:
            semantic_feats = (obj_label_logits.softmax(-1) @ self.label_embedding.weight)[rel_pairs[:, 1]]
            relation_feats = torch.cat([relation_feats, semantic_feats], dim=-1)

        # union feature
        if self.args.use_union_feature:
            # H, W = features.tensors.shape[-2:] # stacked image size
            h, w = (~features.mask[idx]).nonzero(as_tuple=False).max(dim=0)[
                       0] + 1  # mask: image area=False, pad area=True
            proj_feature = self.input_proj(features.tensors[idx:idx + 1])
            scaled_union_boxes = torch.cat(
                [
                    torch.zeros((len(union_boxes), 1)).to(device=union_boxes.device),
                    union_boxes * torch.tensor([w, h, w, h]).to(device=union_boxes.device,
                                                                dtype=union_boxes.dtype).unsqueeze(0),
                ], dim=-1
            )
            union_visual_feats = roi_align(proj_feature, scaled_union_boxes, output_size=self.resolution,
                                           sampling_ratio=2)
            union_visual_feats = self.visual_proj(union_visual_feats.flatten(start_dim=1))
            relation_feats = torch.cat([relation_feats, union_visual_feats], dim=-1)

        #LLM prior knowledge feature
        prior_feats = torch.zeros(relation_feats.shape[0], self.prior_feat_dim).to(relation_feats.device)
        use_prior = True
        if self.training and torch.rand(1).item() < 0.2:
            use_prior = False
        
        if use_prior and (obj_label_logits is not None):
            obj_probs_full = obj_label_logits.softmax(-1)
            
            if obj_probs_full.shape[1] >= 91:
                # [N, 92] -> [N, 80]
                obj_probs_80 = obj_probs_full[..., self.coco_map_indices]
            else:
                obj_probs_80 = obj_probs_full[:, :80]

            current_obj_probs = obj_probs_80[rel_pairs[:, 1]] # [Num_Pairs, 80]
            
            action_priors = torch.matmul(current_obj_probs, self.obj_act_prior)
            
            prior_feats = self.prior_proj(action_priors)

        relation_feats = torch.cat([relation_feats, prior_feats], dim=-1)
        x = self.fusion_fc(relation_feats)
        return x

    def extract_spatial_layout_feats(self, xyxy_boxes):
        box_center = torch.stack([(xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2, (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2],
                                 dim=1)
        dxdy = box_center.unsqueeze(1) - box_center.unsqueeze(0)  # distances
        theta = (torch.atan2(dxdy[..., 1], dxdy[..., 0]) / np.pi).unsqueeze(-1)
        dis = dxdy.norm(dim=-1, keepdim=True)

        box_area = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(dim=1)  # areas
        intersec_lt = torch.max(xyxy_boxes.unsqueeze(1)[..., :2], xyxy_boxes.unsqueeze(0)[..., :2])
        intersec_rb = torch.min(xyxy_boxes.unsqueeze(1)[..., 2:], xyxy_boxes.unsqueeze(0)[..., 2:])
        overlap = (intersec_rb - intersec_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        union_lt = torch.min(xyxy_boxes.unsqueeze(1)[..., :2], xyxy_boxes.unsqueeze(0)[..., :2])
        union_rb = torch.max(xyxy_boxes.unsqueeze(1)[..., 2:], xyxy_boxes.unsqueeze(0)[..., 2:])
        union = (union_rb - union_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        spatial_feats = torch.cat([
            dxdy, dis, theta,  # dx, dy, distance, theta
            overlap, union, box_area[:, None, None].expand(*union.shape), box_area[None, :, None].expand(*union.shape)
            # overlap, union, subj, obj
        ], dim=-1)
        return spatial_feats

# conventional focal loss to balance hard/easy
def focal_loss(blogits, target_classes, alpha=0.5, gamma=2, prior_verb_label_mask=None, class_weights=None):
    probs = blogits.sigmoid()  # prob(positive)
    loss_bce = F.binary_cross_entropy_with_logits(blogits, target_classes, reduction='none', weight=class_weights)
    p_t = probs * target_classes + (1 - probs) * (1 - target_classes)
    loss_bce = ((1 - p_t) ** gamma * loss_bce)

    alpha_t = alpha * target_classes + (1 - alpha) * (1 - target_classes)
    loss_focal = alpha_t * loss_bce

    if prior_verb_label_mask is not None:
        loss_focal = loss_focal * prior_verb_label_mask

    loss = loss_focal.sum() / max(target_classes.sum(), 1)
    return loss


def make_fc(dim_in, hidden_dim, a=1):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
        a: negative slope
    '''
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=a)
    nn.init.constant_(fc.bias, 0)
    return fc


def make_conv3x3(
        in_channels,
        out_channels,
        padding=1,
        dilation=1,
        stride=1,
        kaiming_init=True
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
        nn.init.constant_(conv.bias, 0)
    return conv
