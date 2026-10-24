# ------------------------------------------------------------------------
# HOTR official code : engine/arg_parser.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# Modified arguments are represented with *
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # DETR Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # DETR Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # DETR Transformer (= Encoder, Instance Decoder)
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss Option
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # Loss coefficients (DETR)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Matcher (DETR)
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * HOI Detection
    parser.add_argument('--HOIDet', action='store_true',
                        help="Train HOI Detection head if the flag is provided")
    parser.add_argument('--share_enc', action='store_true',
                        help="Share the Encoder in DETR for HOI Detection if the flag is provided")
    parser.add_argument('--pretrained_dec', action='store_true',
                        help="Use Pre-trained Decoder in DETR for Interaction Decoder if the flag is provided")                        
    parser.add_argument('--hoi_enc_layers', default=1, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--hoi_dec_layers', default=6, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--hoi_nheads', default=8, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--hoi_dim_feedforward', default=2048, type=int,
                        help="Number of decoding layers in HOI transformer")
    # parser.add_argument('--hoi_mode', type=str, default=None, help='[inst | pair | all]')
    parser.add_argument('--num_hoi_queries', default=32, type=int,
                        help="Number of Queries for Interaction Decoder")
    parser.add_argument('--hoi_aux_loss', action='store_true')
    parser.add_argument('--pair_filter_threshold', default=0.08, type=float,
                        help="Filter HOI pairs whose interactiveness score is below this threshold during post-processing")
    parser.add_argument('--use_pair_score_weighting', action='store_true', default=False,
                        help="Multiply verb scores by pair interactiveness scores during post-processing")
    parser.add_argument('--pair_dedupe_iou_thr', default=0.55, type=float,
                        help="Suppress duplicate HOI pair queries whose subject and object IoU are both above this threshold")
    parser.add_argument('--pair_dedupe_ignore_obj_label', action='store_true', default=False,
                        help="Ignore predicted object labels when suppressing duplicate HOI pair queries")


    # * HOTR Matcher
    parser.add_argument('--set_cost_idx', default=1, type=float,
                        help="IDX coefficient in the matching cost")
    parser.add_argument('--set_cost_act', default=1, type=float,
                        help="Action coefficient in the matching cost")
    parser.add_argument('--set_cost_tgt', default=1, type=float,
                        help="Target coefficient in the matching cost")

    # * HOTR Loss coefficients
    parser.add_argument('--temperature', default=0.05, type=float, help="temperature")
    parser.add_argument('--hoi_idx_loss_coef', default=1, type=float)
    parser.add_argument('--hoi_act_loss_coef', default=1, type=float)
    parser.add_argument('--hoi_tgt_loss_coef', default=1, type=float)
    parser.add_argument('--hoi_eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")

    # * dataset parameters
    parser.add_argument('--dataset_file', help='[coco | vcoco]')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--object_threshold', type=float, default=0, help='Threshold for object confidence')

    # machine parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--custom_path', default='',
                        help="Data path for custom inference. Only required for custom_main.py")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--init_from_ossm', default='', type=str,
                        help='Initialize model weights from an old OSSM checkpoint without restoring epoch/optimizer state.')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)

    # mode
    parser.add_argument('--eval', '--validation', action='store_true', help="Only evaluate results if the flag is provided")
    parser.add_argument('--validate', action='store_true', help="Validate after every epoch")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # * WanDB
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project_name', default='HOTR')
    parser.add_argument('--group_name', default='KakaoBrain')
    parser.add_argument('--run_name', default='run_000001')

    # STIP
    parser.add_argument('--STIP_relation_head', action='store_true', default=False)
    parser.add_argument('--stip_model_variant', default='default', choices=['default', 'evidence_trace', 'stip_evidence_trace', 'b1_blip_detr', 'b2_svrm', 'b3_vet', 'stip_b3_vet', 'b4_vet_lpt', 'stip_b4_vet_lpt', 'b5_setm_ablation', 'stip_b5_setm_ablation', 'vet_internal_ablation', 'ossm_box'],
                        help='STIP implementation variant. Use evidence_trace only for opt-in evaluation trace export; the default keeps the original stip.py entry.')
    parser.add_argument(
        '--blip_pretrained',
        default='/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_base_retrieval_coco.pth',
        type=str,
        help='BLIP retrieval checkpoint used by STIP. Kept explicit so visualization runs do not require source edits.',
    )
    parser.add_argument(
        '--blip_vit',
        default='base',
        choices=['base', 'large'],
        help='BLIP vision-transformer size; it must match --blip_pretrained and the STIP checkpoint.',
    )
    parser.add_argument('--blip_image_size', default=384, type=int)
    parser.add_argument('--model_variant', default=None, choices=['b3_vet', 'stip_b3_vet', 'b4_vet_lpt', 'stip_b4_vet_lpt', 'b5_setm_ablation', 'stip_b5_setm_ablation', 'vet_internal_ablation', 'ossm_box'],
                        help='Optional alias for selecting an ablation variant without changing the default STIP variant selection.')
    parser.add_argument('--ossm_blip_pretrained', default='', type=str,
                        help='BLIP retrieval checkpoint used only by the ossm_box variant; specify the old BLIP-L checkpoint for migration experiments.')
    parser.add_argument('--ossm_blip_vit', default='large', choices=['base', 'large'],
                        help='BLIP vision-transformer size used only by ossm_box.')
    parser.add_argument('--ossm_text_library', default='hico_L_embeds_256.npy', type=str,
                        help='OSSM hierarchical text library (.npy, [num_hoi,3,256]).')
    parser.add_argument('--ossm_obj_act_prior', default='obj_act_prior.npy', type=str,
                        help='OSSM object-action prior (.npy, [80,num_actions]).')
    parser.add_argument('--ossm_region_image_size', default=384, type=int,
                        help='Square BLIP input size for SAM-free patch-map ROI extraction.')
    parser.add_argument('--ossm_topk_gt_stats', action='store_true',
                        help='Export HICO GT text retrieval recall and per-image Top-k details (ossm_box evaluation only).')
    parser.add_argument('--ossm_debug_regions', action='store_true', default=False,
                        help='Retain the latest human/object/union ROI boxes on CPU for inspection.')
    parser.add_argument('--finetune_detr', action='store_true', default=False)
    parser.add_argument('--use_high_resolution_relation_feature_map', action='store_true', default=False)
    parser.add_argument('--use_visual_scene_graph', action='store_true', default=False)
    parser.add_argument('--graph_num_layers', default=2, type=int)
    parser.add_argument('--graph_num_heads', default=8, type=int)
    parser.add_argument('--graph_k_oo', default=3, type=int)
    parser.add_argument('--graph_oo_mode', default='spatial', choices=['spatial', 'share_human', 'hybrid', 'text_hybrid'])
    parser.add_argument('--graph_oo_lambda_geo', default=1.0, type=float)
    parser.add_argument('--graph_oo_lambda_share', default=0.5, type=float)
    parser.add_argument('--graph_oo_lambda_text', default=0.0, type=float)
    parser.add_argument('--graph_oo_min_score', default=0.0, type=float)
    parser.add_argument('--graph_oo_fallback_to_spatial', action='store_true', default=True)
    parser.add_argument('--no_graph_oo_fallback_to_spatial', dest='graph_oo_fallback_to_spatial', action='store_false')
    parser.add_argument('--graph_oo_share_eps', default=1e-6, type=float)
    parser.add_argument('--graph_oo_prior_path', default='', type=str)
    parser.add_argument('--graph_oo_prior_normalize', default='none', choices=['none', 'minmax', 'sigmoid'])
    parser.add_argument('--graph_oo_prior_symmetric', action='store_true', default=True)
    parser.add_argument('--no_graph_oo_prior_symmetric', dest='graph_oo_prior_symmetric', action='store_false')
    parser.add_argument('--graph_oo_prior_eps', default=1e-6, type=float)
    parser.add_argument('--graph_enable_oo_edges', action='store_true', default=True)
    parser.add_argument('--no_graph_oo_edges', dest='graph_enable_oo_edges', action='store_false', default=True)
    parser.add_argument('--graph_enable_hh_edges', action='store_true', default=False)
    parser.add_argument('--graph_enable_path_context', action='store_true', default=False)
    parser.add_argument('--graph_path_topk', default=3, type=int)
    parser.add_argument('--graph_path_dropout', default=0.1, type=float)
    parser.add_argument('--graph_path_gate_init', default=0.0, type=float)
    parser.add_argument('--graph_path_use_edge_scores', action='store_true', default=True)
    parser.add_argument('--no_graph_path_use_edge_scores', dest='graph_path_use_edge_scores', action='store_false')
    parser.add_argument('--graph_path_score_thresh', default=0.0, type=float)
    parser.add_argument('--graph_dropout', default=0.1, type=float)
    parser.add_argument('--graph_debug', action='store_true', default=False)
    parser.add_argument('--use_vet', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--use_lpt', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--vet_debug', action='store_true', default=False)
    parser.add_argument('--lpt_debug', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--vet_gate_init', default=0.0, type=float)
    parser.add_argument('--lpt_gate_init', default=0.0, type=float)
    parser.add_argument('--lpt_query_source', default='vet_pair', choices=['vet_pair', 'query_reps', 'base_pair', 'refined_edge'])
    parser.add_argument('--lpt_context_source', default='slots', choices=['slots', 'pooled_all', 'concat_slots_pooled'])
    parser.add_argument('--lpt_use_selected_score_weight', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--lpt_detach_query', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--lpt_topk', default=5, type=int)
    parser.add_argument('--lpt_pre_topk', default=20, type=int)
    parser.add_argument('--lpt_temperature', default=0.07, type=float)
    parser.add_argument('--lpt_rerank_weight', default=0.5, type=float)
    parser.add_argument('--lpt_num_layers', default=1, type=int)
    parser.add_argument('--lpt_num_heads', default=8, type=int)
    parser.add_argument('--lpt_dropout', default=0.1, type=float)
    parser.add_argument('--vet_node_score_thresh', default=0.05, type=float)
    parser.add_argument('--vet_min_score_nodes', default=0, type=int)
    parser.add_argument('--vet_max_extra_nodes', default=None, type=int)
    parser.add_argument('--vet_store_dense_debug_tensors', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_vet_ablation_mode', default='path_reweight', choices=['anchor_raw', 'anchor_svrm', 'endpoints', 'context_uniform', 'path_selection', 'path_reweight'])
    parser.add_argument('--graph_vet_selection_temperature', default=1.0, type=float)
    parser.add_argument('--vet_run_kind', default='smoke', choices=['smoke', 'formal'])
    parser.add_argument('--vet_allow_matching_resume', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--vet_train_max_samples', default=0, type=int)
    parser.add_argument('--vet_val_max_samples', default=0, type=int)
    parser.add_argument('--graph_node_score_thresh', default=0.05, type=float)
    parser.add_argument('--graph_min_score_nodes', default=0, type=int)
    parser.add_argument('--graph_max_extra_nodes', default=-1, type=int)
    parser.add_argument('--graph_debug_interval', default=100, type=int)
    parser.add_argument('--graph_debug_max_prints', default=20, type=int)
    parser.add_argument('--graph_enable_text_graph', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_text_features_path', default='', type=str)
    parser.add_argument('--graph_text_mask_path', default='', type=str)
    parser.add_argument('--graph_text_hoi_ids_path', default='', type=str)
    parser.add_argument('--graph_text_object_to_hoi_indices_path', default='', type=str)
    parser.add_argument('--graph_text_field_order_path', default='', type=str)
    parser.add_argument('--graph_text_strict_shape', nargs='?', const=True, default=True, type=str2bool,
                        help='Require the HICO-DET 520-class text graph shape. Set False for V-COCO 289-class assets.')
    parser.add_argument('--graph_text_human_node_features_path', default='', type=str)
    parser.add_argument('--graph_text_object_node_features_path', default='', type=str)
    parser.add_argument('--graph_text_ho_edge_features_path', default='', type=str)
    parser.add_argument('--graph_text_discriminative_features_path', default='', type=str)
    parser.add_argument('--graph_text_scene_context_features_path', default='', type=str)
    parser.add_argument('--graph_text_oo_context_features_path', default='', type=str)
    parser.add_argument('--graph_text_oo_affordance_prior_path', default='', type=str)
    parser.add_argument('--graph_text_topk', default=5, type=int)
    parser.add_argument('--graph_text_pre_topk', default=20, type=int)
    parser.add_argument('--graph_text_temperature', default=0.07, type=float)
    parser.add_argument('--graph_text_rerank_weight', default=0.5, type=float)
    parser.add_argument('--graph_text_use_discriminative_rerank', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_text_num_layers', default=None, type=int)
    parser.add_argument('--graph_text_num_heads', default=8, type=int)
    parser.add_argument('--graph_text_dropout', default=0.1, type=float)
    parser.add_argument('--graph_text_use_scene_token', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_text_use_oo_context_token', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_text_cross_num_heads', default=8, type=int)
    parser.add_argument('--graph_text_query_source', default=None, choices=['base_pair', 'refined_edge', 'union', 'refined_plus_union'])
    parser.add_argument('--graph_text_sync_mode', default='layerwise', type=str, choices=['posthoc', 'layerwise'],
                        help='Text graph fusion mode. posthoc keeps old behavior; layerwise performs synchronized visual-text graph reasoning.')
    parser.add_argument('--graph_text_dense_injection', nargs='?', const=True, default=True, type=str2bool,
                        help='Write text graph updates back to dense visual node/edge states.')
    parser.add_argument('--graph_text_final_pair_injection', nargs='?', const=True, default=False, type=str2bool,
                        help='Whether to additionally apply old pair-level final text injection after layerwise reasoning. Default False to avoid double injection.')
    parser.add_argument('--graph_text_dynamic_retrieval', nargs='?', const=True, default=False, type=str2bool,
                        help='Whether to rebuild text graph candidates at every graph layer. Default False.')
    parser.add_argument('--graph_text_force_oov3', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_text_prototype_slot_fallback_to_raw', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_text_prototype_slot_debug', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_enable_reciprocal_setm', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_reciprocal_use_path_context', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_reciprocal_write_text', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_reciprocal_write_visual', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_reciprocal_update_human', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_reciprocal_update_object', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_reciprocal_update_anchor', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_reciprocal_update_scene', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_reciprocal_update_context', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_reciprocal_update_discriminative', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_reciprocal_detach_text_update', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_reciprocal_debug', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_sync_visual_gate_init', default=0.0, type=float)
    parser.add_argument('--graph_sync_text_gate_init', default=0.0, type=float)
    parser.add_argument('--graph_sync_anchor_gate_init', default=0.0, type=float)
    parser.add_argument('--graph_sync_dropout', default=None, type=float)
    parser.add_argument('--graph_sync_debug', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--use_setm_ablation', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--setm_ablation_stage', default=5, type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--disable_simple_lpt_residual_for_setm', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--setm_dropout', default=0.1, type=float)
    parser.add_argument('--setm_visual_gate_init', default=0.0, type=float)
    parser.add_argument('--setm_text_gate_init', default=0.0, type=float)
    parser.add_argument('--setm_anchor_gate_init', default=0.0, type=float)
    parser.add_argument('--setm_debug', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument("--graph_enable_candidate_evidence", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--graph_evidence_topk", default=4, type=int)
    parser.add_argument("--graph_evidence_gate_init", default=0.0, type=float)
    parser.add_argument("--graph_evidence_dropout", default=0.1, type=float)
    parser.add_argument("--graph_evidence_allow_node_only_context", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--graph_evidence_debug", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--graph_enable_sync_evidence_matching", nargs="?", const=True, default=False, type=str2bool)

    parser.add_argument("--graph_enable_layerwise_sync_evidence_matching", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--graph_layerwise_sync_write_visual", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--graph_layerwise_sync_write_text", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--graph_layerwise_sync_final_matching", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--graph_layerwise_sync_disable_old_dense_injection", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--graph_layerwise_sync_write_reverse_edge", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--graph_layerwise_sync_detach_text_update", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--graph_layerwise_sync_debug", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--graph_store_dense_debug_tensors", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument('--graph_export_evidence_trace', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_export_trace_dir', default='', type=str)
    parser.add_argument('--graph_export_trace_each_layer', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_export_trace_to_cpu', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--no-graph_export_trace_to_cpu', dest='graph_export_trace_to_cpu', action='store_false')
    parser.add_argument('--graph_export_trace_float_dtype', default='float16', choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--graph_export_trace_include_match_logits', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_export_trace_include_features', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_export_trace_topk_actions', default=10, type=int)
    parser.add_argument('--graph_export_trace_include_all_action_logits', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--no-graph_export_trace_include_all_action_logits', dest='graph_export_trace_include_all_action_logits', action='store_false')
    parser.add_argument('--graph_export_trace_match_ground_truth', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--no-graph_export_trace_match_ground_truth', dest='graph_export_trace_match_ground_truth', action='store_false')
    parser.add_argument('--graph_export_trace_overwrite', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_export_mechanism_validation_trace', nargs='?', const=True, default=False, type=str2bool,
                        help='Evaluation-only export of query residuals, post-writeback layer pair states, and decoder probes.')
    parser.add_argument('--graph_mechanism_validation_variant', default='full', type=str,
                        choices=['full', 'no_anchor_writeback', 'no_prototype_state_writeback', 'no_setm', 'endpoint_only_setm'],
                        help='Deterministic evaluation-only EvidenceHOI mechanism intervention.')
    parser.add_argument('--eval_max_samples', default=0, type=int,
                        help='Evaluation-only sample limit for smoke tests; 0 evaluates the full split.')
    parser.add_argument('--eval_image_ids_file', default='', type=str,
                        help='Optional JSON list/dict or text file of evaluation dataset indices. '
                             'Cannot be combined with --eval_max_samples.')

    parser.add_argument('--graph_text_sync_query_source', default='base_pair', type=str, choices=['base_pair', 'refined_edge', 'union', 'refined_plus_union'],
                        help='Query feature source used for initial text graph retrieval in layerwise mode.')
    parser.add_argument('--graph_text_sync_layers_with_visual', nargs='?', const=True, default=True, type=str2bool,
                        help='Use graph_num_layers as graph_text_num_layers unless graph_text_num_layers is explicitly provided.')
    parser.add_argument('--graph_text_edge_inject_reverse', nargs='?', const=True, default=True, type=str2bool,
                        help='Also inject text HO edge update into reverse visual OH edge.')
    parser.add_argument('--graph_text_node_scatter_reduce', default='mean', type=str, choices=['mean', 'sum'],
                        help='How to aggregate pair-level text node priors onto dense visual nodes.')
    parser.add_argument('--graph_text_gate_init', default=0.0, type=float)
    parser.add_argument('--graph_text_prior_injection', default=None, choices=['none', 'edge_only', 'node_edge', 'all'])
    parser.add_argument('--graph_text_use_human_node_prior', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_text_use_object_node_prior', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_text_use_ho_edge_prior', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--graph_text_use_discriminative_prior', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_text_use_scene_prior', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_text_use_oo_prior', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--graph_text_node_prior_gate_init', default=0.0, type=float)
    parser.add_argument('--graph_text_edge_prior_gate_init', default=0.0, type=float)
    parser.add_argument('--graph_text_scene_prior_gate_init', default=0.0, type=float)
    parser.add_argument('--graph_text_oo_prior_gate_init', default=0.0, type=float)
    parser.add_argument('--graph_text_debug', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--use_edge_blip_fusion', dest='use_edge_blip_fusion', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--no_edge_blip_fusion', dest='use_edge_blip_fusion', action='store_false', default=True)
    parser.add_argument('--use_blip_global_memory', dest='use_blip_global_memory', nargs='?', const=True, default=True, type=str2bool)
    parser.add_argument('--no_blip_global_memory', dest='use_blip_global_memory', action='store_false', default=True)
    return parser
