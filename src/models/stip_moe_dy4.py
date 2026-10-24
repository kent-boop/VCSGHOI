import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from PIL import Image
from src.util.misc import NestedTensor, nested_tensor_from_tensor_list
from torchvision.ops import roi_align
from torchvision import transforms
from .transformer import TransformerDecoderLayer, TransformerDecoder
from src.util import box_ops
from src.util.misc import accuracy, is_dist_avail_and_initialized, get_world_size
from src.lm.SAM import SamPredictor
from src.models.text_selector import HierarchicalSimilarityCalculator
from src.models.fusion_module import TextGuidedBlipOptimizer
from src.models.MoEfusion3 import MoEFusion
from src.models.fusion_module import VisualAdapter, VisualTextAdapter

class STIP(nn.Module):
    def __init__(self, args, detr, sam, blip, detr_matcher):
        super().__init__()
        self.args = args
        self.detr_matcher = detr_matcher
        # * Instance Transformer ---------------
        self.detr = detr
        self.sam = sam
        self.blip = blip
        for p in self.sam.parameters():
            p.requires_grad_(False)
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
        
        #text sim_matrix
        self.text_selector = HierarchicalSimilarityCalculator(text_lib_path='hico_L2_embeds_256.npy')
        # fusion module
        self.scene_optimizer = TextGuidedBlipOptimizer()
        self.instance_optimizer = TextGuidedBlipOptimizer()
        #action query improvement
        self.action_cross_attn = nn.MultiheadAttention(embed_dim=self.args.hidden_dim, num_heads=4)
        self.action_norm = nn.LayerNorm(self.args.hidden_dim)
        self.action_dropout = nn.Dropout(0.1)
        #memory improvement
        self.memory_text_attn = nn.MultiheadAttention(embed_dim=self.args.hidden_dim, num_heads=4,batch_first=False)
        self.memory_text_norm = nn.LayerNorm(self.args.hidden_dim)
        self.memory_text_dropout = nn.Dropout(0.1)
        # cross attention
        # self.text_encoder_proj = nn.Linear(3 * 256, 256)
        self.blip_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862854, 0.26130258, 0.27577711))
        ])
        #learnable position encoding 384*383/16*16=576 hidden_dim=256
        self.blip_scene_pos_embed = nn.Embedding(num_embeddings=576,embedding_dim=self.args.hidden_dim)
        self.blip_inst_pos_embed = nn.Embedding(num_embeddings=576,embedding_dim=self.args.hidden_dim)
        #init pos embedding
        nn.init.normal_(self.blip_scene_pos_embed.weight, std=0.02)
        nn.init.normal_(self.blip_inst_pos_embed.weight, std=0.02)
        #MoE fusion module
        visual_dim_dict = {
            'sam': 256,
            'union': 256,
            'global': 256,
            'rel': 1024  # This matches rel_rep_dim
        }
        self.text_input_dim = 256
        self.moe_out_dim = 1024
        # ==========================================
        print(f"[STIP Model Init] Initializing MoE with:")
        print(f"  - use_visual_router: {getattr(args, 'use_moe_visual_router', 'Not Found')}")
        print(f"  - use_aux_loss:      {getattr(args, 'use_moe_aux_loss', 'Not Found')}")
        print(f"  - noisy_gating:      {getattr(args, 'use_moe_noisy_gating', 'Not Found')}")
        # ==========================================
        self.moe_fusion_module = MoEFusion(
            visual_dim_dict=visual_dim_dict,
            text_dim=self.text_input_dim,
            out_dim=self.moe_out_dim,
            hidden_dim=1024, # Hidden dim inside experts
            use_visual_router=getattr(args, 'use_moe_visual_router', True), # Default to True if not in args
            use_aux_loss=getattr(args, 'use_moe_aux_loss', True),
            noisy_gating=getattr(args, 'use_moe_noisy_gating', True)
        )
        #learnable 
        self.moe_loss_log_var = nn.Parameter(torch.tensor(0.0))
        #dynamic text adapter
        self.scene_adapter = VisualAdapter(visual_dim=256, text_dim=256, bottleneck=64)
        self.instance_adapter = VisualAdapter(visual_dim=256, text_dim=256, bottleneck=64)
        self.action_adapter = VisualTextAdapter(visual_dim=1024, text_dim=256, bottleneck=128)


    def forward(self, samples: NestedTensor, images, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        #trasnsfer for samples
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).to(samples.tensors.device).view(3, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).to(samples.tensors.device).view(3, 1, 1)
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

        # test on class transform
        # probs_org = outputs_class_org[-1, 0].softmax(-1)
        # inst_scores_org, inst_labels_org = probs_org[:, :-1].max(-1)
        # probs_ts = outputs_class[-1, 0].softmax(-1)
        # inst_scores_ts, inst_labels_ts = probs_ts[:, :-1].max(-1)

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
        pred_rel_exists, pred_rel_pairs, pred_actions, pred_sampled_actions, pred_actions_fore, pred_global_actions_fore, outs_features, hs_features = [], [], [], [], [], [], [], []
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
            #original images
            img_orig = images[imgid]
            img_orig_resized = img_orig.resize((384,384), Image.ANTIALIAS)
            img_tensor_blip = self.blip_transform(img_orig_resized)
            # restore samples
            # samples.tensors: [batch, 3, H_pad, W_pad]
            # samples.mask : [batch, H_pad, W_pad]
            img_tensor = samples.tensors[imgid]
            img_mask = samples.mask[imgid]
            #get valid_size image
            valid_h = (~img_mask).sum(dim=0).max()
            valid_w = (~img_mask).sum(dim=1).max()
            img_valid = img_tensor[:, :valid_h, :valid_w]
            img_restored = (img_valid * pixel_std + pixel_mean).clamp(0,1)*255
            img_restored_np = img_restored.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            image_pil_restored = Image.fromarray(img_restored_np)


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

            #text-guaided visual_feature fusion
            multilscale_blip = self.blip(img_tensor_blip.unsqueeze(0).to('cuda'), 'None', 'multilscale')#list of multi-scale features
            # blip_bg = multilscale_blip[0][:, 1:, :]      
            # blip_instance = multilscale_blip[1][:, 1:, :]
            blip_action = multilscale_blip[2][:, 0, :]#cls token
            blip_action = F.normalize(blip_action, dim=-1)

            # images_blip = self.blip(scaled_img.unsqueeze(0).to('cuda'), 'None', 'itc')#(1,576+1,dim)


            ################ action fusion ################ 
            optimized_action_feat = blip_action.repeat(self.args.num_hoi_queries,1)   
            # images_blip = images_blip.repeat(32,1)

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
                                                                          image=images[imgid], ins_masks=instance_masks,
                                                                          idx=imgid)
                    p_relation_exist_logits = self.relation_proposal_mlp(all_rel_reps)

                    gt_inds = torch.arange(gt_pair_count).to(p_relation_exist_logits.device)
                    _, sort_rel_inds = p_relation_exist_logits[gt_pair_count:].squeeze(1).sort(descending=True)
                    # _, sort_rel_inds = torch.cat([inst_scores[all_pairs[:, 1:]], p_relation_exist_logits.sigmoid()], dim=-1).prod(-1)[gt_pair_count:].sort(descending=True)
                    sampled_rel_inds = torch.cat([gt_inds, sort_rel_inds + gt_pair_count])[:self.args.num_hoi_queries]

                    sampled_rel_pairs = all_pairs[sampled_rel_inds]
                    sampled_rel_reps = all_rel_reps[sampled_rel_inds]

                    # if self.args.proj_blip:
                    #     if self.args.merge_blip:
                    #         blip_merge = self.merge_blip(torch.cat((blip_sam, blip_union,optimized_action_feat), dim=-1))
                    #         sampled_rel_reps = torch.cat((sampled_rel_reps, blip_merge), dim=-1)
                    #         sampled_rel_reps = self.proj_blip(sampled_rel_reps)
                    #     else:
                    #         sampled_rel_reps = torch.cat((sampled_rel_reps, blip_sam, blip_union,optimized_action_feat), dim=-1)
                    #         sampled_rel_reps = self.proj_blip(sampled_rel_reps)

                    sampled_rel_reps = sampled_rel_reps.squeeze(0)

                    # if self.args.only_blip:
                    #     blip_sem = self.proj_blip(torch.cat((blip_sam, blip_union,images_blip), dim=-1))
                    #     sampled_rel_reps = blip_sem

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


                # if self.args.proj_blip:
                #     if self.args.merge_blip:
                #         blip_merge = self.merge_blip(torch.cat((blip_sam, blip_union,optimized_action_feat), dim=-1))
                #         sampled_rel_reps = torch.cat((sampled_rel_reps, blip_merge), dim=-1)
                #         sampled_rel_reps = self.proj_blip(sampled_rel_reps)
                #     else:
                #         sampled_rel_reps = torch.cat((sampled_rel_reps, blip_sam, blip_union,optimized_action_feat), dim=-1)
                #         sampled_rel_reps = self.proj_blip(sampled_rel_reps)

                # if self.args.only_blip:
                #     blip_sem = self.proj_blip(torch.cat((blip_sam, blip_union,images_blip), dim=-1))
                #     sampled_rel_reps = blip_sem


                sampled_rel_reps = sampled_rel_reps.squeeze(0)

                sampled_actions = self.sampled_actions(sampled_rel_reps)
                sampled_rel_pred_exists = p_relation_exist_logits.squeeze(1)[sampled_rel_inds]

            # >>>>>>>>>>>> relation classification <<<<<<<<<<<<<<<
            query_reps = self.rel_query_pre_proj(sampled_rel_reps).unsqueeze(1)# (query, 1, dim)
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
                # images_encoding = self.blip(scaled_img.unsqueeze(0).to('cuda'), 'None', 'image_encoding')


                detr_mem_flat = memory_input[imgid:imgid + 1].flatten(2).permute(2, 0, 1)  # (h*w, 1, 256)
                detr_pos_flat = memory_pos[imgid:imgid + 1].flatten(2).permute(2, 0, 1)# (h*w, 1, 256)
                # blip_scene_mem = optimized_scene_feat.permute(1, 0, 2) 
                # blip_instance_mem = optimized_instance_feat.permute(1, 0, 2)
                # visual_memory = torch.concat((blip_scene_mem,blip_instance_mem,detr_mem_flat),dim=0)
                # #text-guided memory improvement
                # text_kv = action_text.unsqueeze(1)
                # mem_attn_out, _ = self.memory_text_attn(
                #     query=visual_memory, 
                #     key=text_kv, 
                #     value=text_kv
                # )
                # #residual connection
                # memory_enhanced = self.memory_text_norm(visual_memory + self.memory_text_dropout(mem_attn_out))
                # #positional encoding
                # # blip_scene_pos = torch.zeros_like(blip_scene_mem)
                # # blip_instance_pos = torch.zeros_like(blip_instance_mem)
                # L_scene = blip_scene_mem.shape[0]
                # scene_pos_ids = torch.arange(L_scene, device=blip_scene_mem.device)
                # blip_scene_pos = self.blip_scene_pos_embed(scene_pos_ids).unsqueeze(1)
                # L_inst = blip_instance_mem.shape[0]
                # inst_pos_ids = torch.arange(L_inst, device=blip_instance_mem.device)
                # blip_instance_pos = self.blip_inst_pos_embed(inst_pos_ids).unsqueeze(1)
                # pos_all = torch.concat((blip_scene_pos,blip_instance_pos,detr_pos_flat),dim=0)
                # #memory_mask
                # blip_len = blip_scene_mem.shape[0] + blip_instance_mem.shape[0]
                # detr_len = detr_mem_flat.shape[0]
                # detr_mask_flat = memory_input_mask[imgid:imgid + 1].flatten(1)
                # blip_mask = torch.zeros((1, blip_len), dtype=torch.bool,device=detr_mask_flat.device)
                # memory_key_padding_mask = torch.cat((blip_mask, detr_mask_flat), dim=1)
                # # memory_key_padding_mask=torch.tensor(torch.zeros(1, 1152 + memory_input[imgid:imgid + 1].shape[2] * memory_input[imgid:imgid + 1].shape[3]), dtype=torch.bool).to('cuda')


                outs = self.interaction_decoder(tgt=query_reps,
                                                tgt_mask=tgt_mask,
                                                query_pos=query_pos_encoding,
                                                query_structure_encoding=relation_dependency_encodings,
                                                # inter-ineraction semantic structure
                                                memory=memory_enhanced,
                                                memory_key_padding_mask=memory_key_padding_mask,
                                                pos=pos_all,
                                                memory_role_embedding=None,  # intra-ineraction spatial structure
                                                memory_value_embedding=None)

            action_logits = self.action_embed(outs)
            pred_sampled_actions.append(sampled_actions)
            pred_rel_pairs.append(sampled_rel_pairs)
            pred_actions.append(action_logits)
            pred_rel_exists.append(sampled_rel_pred_exists)

        # ==================== 新增监控代码开始 ====================
        if self.training:
            # 获取最后一张图产生的权重 (注意：因为在循环内，我们拿最后 imgid 的那份)
            if hasattr(self.moe_fusion_module, 'last_weights'):
                # 初始化一个步数计数器 (如果没有的话)
                if not hasattr(self, 'mon_step'): self.mon_step = 0
                self.mon_step += 1
                
                # 每 300 个 batch 打印一次，你可以调大这个数字
                if self.mon_step % 50 == 0:
                    # last_weights 形状: [1, N, 3]
                    avg_w = self.moe_fusion_module.last_weights.mean(dim=[0, 1])
                    print(f"\n" + "!"*50)
                    print(f"  [MoE MONITOR] Step: {self.mon_step}")
                    print(f"  SAM: {avg_w[0]:.4f} | Union: {avg_w[1]:.4f} | Global: {avg_w[2]:.4f}")
                    print("!"*50 + "\n")
        # ==================== 新增监控代码结束 ====================

        hoi_recognition_time = time.time() - start_time
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_rel_pairs": pred_rel_pairs,
            "pred_actions": [p[-1].squeeze(1) for p in pred_actions],
            "pred_sampled_actions": pred_sampled_actions,
            "pred_action_exists": pred_rel_exists,
            "det2gt_indices": det2gt_indices,
            "hoi_recognition_time": hoi_recognition_time,
            "moe_aux_loss":moe_aux_loss,
            "moe_loss_weighted": moe_loss_weighted
        }
        if self.args.hoi_aux_loss: out['hoi_aux_outputs'] = self._set_hoi_aux_loss(pred_actions)
        if self.args.train_detr and self.args.aux_loss: out['aux_outputs'] = self._set_aux_loss(outputs_class,
                                                                                                outputs_coord)

        return out

    @torch.jit.unused
    def _set_hoi_aux_loss(self, pred_actions):
        return [{'pred_actions': [p[l].squeeze(1) for p in pred_actions]} for l in range(self.args.hoi_dec_layers - 1)]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": l, "pred_boxes": b} for l, b in zip(outputs_class[:-1], outputs_coord[:-1])]

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

    def get_mask_from_sam(self, sam, image_restored, size, boxes):
        '''
        get mask from sam
        given image and boxes
        '''
        predictor = SamPredictor(sam)
        target_size = 384
        # resize image to target size
        w,h = image_restored.size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = image_restored.resize((new_w, new_h), Image.ANTIALIAS)
        #padding
        padded_image = Image.new("RGB", (target_size, target_size),(0,0,0))
        padded_image.paste(image_resized, ((0,0)))#in the top-left corner
        image_np = np.array(padded_image)
        #sam
        predictor.set_image(image_np)
        boxes_xy = box_ops.box_cxcywh_to_xyxy(boxes).cpu().numpy()
        scale_fct = np.array([new_w, new_h, new_w, new_h])
        boxes_xy = boxes_xy * scale_fct[None, None, :]
        boxes_xy[..., 0::2] = np.clip(boxes_xy[..., 0::2], 0, new_w) 
        boxes_xy[..., 1::2] = np.clip(boxes_xy[..., 1::2], 0, new_h) 
        # if train_detr:
        # boxes_xy = boxes_xy.detach().cpu().numpy() * scale_fct[None, None, :]
        instance_masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xy,
            multimask_output=False,
        )
        # print(f"[DEBUG] SAM Output mask shape (raw): {instance_masks.shape}")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862854, 0.26130258, 0.27577711))
        ])

        scaled_img_tensor = transform(padded_image)
        instance_masks = torch.tensor(instance_masks).float()
        if len(instance_masks.shape) == 3:
            instance_masks = instance_masks.squeeze(1)
        # print(f"[DEBUG] Final mask shape: {instance_masks.shape}")
        masked_instances = instance_masks * scaled_img_tensor
        # print(f"[DEBUG] Masked instance shape: {masked_instances.shape}")

        return masked_instances, boxes_xy, scaled_img_tensor

    def make_union_mask(self, ins_masks, rel_pairs):
        ins_masks = ins_masks.to(rel_pairs.device)
        human_masks = ins_masks[rel_pairs[:, 0]]
        object_masks = ins_masks[rel_pairs[:, 1]]
        union_masks = human_masks + object_masks
        return union_masks

    def make_union_img(self, scaled_boxes, sampled_rel_pairs, scaled_img):
        scaled_boxes = torch.tensor(scaled_boxes).to(sampled_rel_pairs.device).squeeze(0)
        head_boxes = scaled_boxes[sampled_rel_pairs[:, 0]]
        tail_boxes = scaled_boxes[sampled_rel_pairs[:, 1]]
        union_boxes = torch.cat([
            torch.min(head_boxes[:, :2], tail_boxes[:, :2]),
            torch.max(head_boxes[:, 2:], tail_boxes[:, 2:])
        ], dim=1)

        union_masks = torch.zeros(union_boxes.size()[0], scaled_img.size()[1], scaled_img.size()[2])
        union_boxes = union_boxes.to(torch.int).clamp_(min=0, max=union_masks.size()[1])
        union_boxes = union_boxes.view(union_boxes.size()[0], union_boxes.size()[1], 1, 1)

        y = torch.arange(0, union_masks.size()[1]).view(1, union_masks.size()[1], 1).float().to(union_boxes.device)
        x = torch.arange(0, union_masks.size()[1]).view(1, 1, union_masks.size()[1]).float().to(union_boxes.device)

        mask_y = (y >= union_boxes[:, 0:1, :, :]) & (y < union_boxes[:, 2:3, :, :])
        mask_x = (x >= union_boxes[:, 1:2, :, :]) & (x < union_boxes[:, 3:4, :, :])

        union_masks = mask_y & mask_x
        union_imgs = scaled_img.unsqueeze(0).to(union_masks.device) * union_masks

        return union_imgs


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
            'loss_moe_balance':1.0
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
        
        if 'moe_loss_weighted' in outputs:
            loss_dict['loss_moe_balance'] = outputs['moe_loss_weighted']

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

    def proposal_loss(self, inputs, targets):
        # loss = focal_loss(inputs, targets, gamma=self.args.proposal_focal_loss_gamma, alpha=self.args.proposal_focal_loss_alpha)

        ## conventional BCE
        # loss_bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss = loss_bce[targets<0.5].mean() # neg loss
        # if targets.sum() > 0:
        #     loss += loss_bce[targets>0.5].mean() # pos loss

        # focal loss to balance positive/negative
        probs = inputs.sigmoid()
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        pos_loss = torch.log(probs) * torch.pow(1 - probs, self.args.proposal_focal_loss_gamma) * pos_inds
        neg_loss = torch.log(1 - probs) * torch.pow(probs, self.args.proposal_focal_loss_gamma) * neg_inds
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # normalize
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

    def action_loss(self, inputs, targets, prior_verb_label_mask=None):
        # loss = focal_loss(inputs, targets, gamma=self.args.action_focal_loss_gamma, alpha=self.args.action_focal_loss_alpha, prior_verb_label_mask=prior_verb_label_mask)
        probs = inputs.sigmoid()

        # focal loss to balance positive/negative
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        pos_loss = torch.log(probs) * torch.pow(1 - probs, self.args.action_focal_loss_gamma) * pos_inds
        neg_loss = torch.log(1 - probs) * torch.pow(probs, self.args.action_focal_loss_gamma) * neg_inds
        if prior_verb_label_mask is not None:  # mask invalid predictions
            pos_loss = pos_loss * prior_verb_label_mask
            neg_loss = neg_loss * prior_verb_label_mask
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # normalize
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


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

            # accumulate results (iterate through interaction queries)
            results = []
            for batch_idx, (os, ol, vs, box, h_idx, o_idx) in enumerate(
                    zip(obj_scores, obj_labels, verb_scores, boxes, h_indices, o_indices)):
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

    def forward(self, rel_pairs, features, boxes, inst_reprs, idx, image=None, ins_masks=None, obj_label_logits=None,
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

    def make_union_mask(self, ins_masks, rel_pairs):
        ins_masks = ins_masks.to(rel_pairs.device)
        human_masks = ins_masks[rel_pairs[:, 0]]
        object_masks = ins_masks[rel_pairs[:, 1]]
        union_masks = human_masks + object_masks
        return union_masks


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
