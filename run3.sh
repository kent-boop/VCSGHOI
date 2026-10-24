#!/bin/bash

# =========================================================
#  STIP 训练通用启动脚本 (单卡/多卡自动适配)
# =========================================================

# 1. 显卡设置 (在此处修改)
# 单卡示例: GPU_IDS="0" 或 GPU_IDS="2"
# 多卡示例: GPU_IDS="0,1" 或 GPU_IDS="0,1,2,3"
GPU_IDS="0,1,2,3,4,6,7"  # 修改为你想使用的显卡 ID，逗号分隔

# 2. 基础配置
# 注意：如果是多卡，BATCH_SIZE 是每张卡的 batch size，总 batch size 会翻倍 #eval 64 train 48 train detr 12

BATCH_SIZE=28
NUM_WORKERS=4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="checkpoints/hico-det_sync_vt_graph8_large"
DATA_PATH="/share/share/Datasets/hico_20160224_det"
DETR_WEIGHTS="/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/detr_coco/checkpoint0120.pth"
# DETR_WEIGHTS="/share/share/HOI/pretrained_model/HICO/all/res101/checkpoint0105.pth"
RESUME_CHECKPOINT="/wzj/zijian/OSSM-HOI-main/checkpoints/EvidenceHOI-base/STIP_opensource/hico_run_graph6_base/best.pth"
OO_PRIOR_PATH="${SCRIPT_DIR}/oo_affordance_prior_80.npy"
TEXT_GRAPH_DIR="${SCRIPT_DIR}/hico_text_graph_blip256"
TEXT_FEATURES_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_features_6x256.npy"
TEXT_MASK_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_mask_6.npy"
TEXT_HOI_IDS_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_hoi_ids.json"
TEXT_OBJECT_TO_HOI_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_object_to_hoi_indices.json"
TEXT_FIELD_ORDER_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_field_order.json"

# =========================================================
#  自动计算逻辑 (无需修改)
# =========================================================
# ========== 环境变量 ==========
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=6
# 设置可见显卡
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
# 优化显存分配，防止碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 优化配置
export NCCL_ALGO=Ring
export NCCL_NSOCKS_PERTRACK=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_PROTO=LL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
# 分布式设置
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29506
# 计算显卡数量
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "========================================================="
echo "启动配置:"
echo "使用显卡 ID: ${GPU_IDS}"
echo "显卡数量 (nproc_per_node): ${NUM_GPUS}"
echo "单卡 Batch Size: ${BATCH_SIZE}"
echo "总 Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "========================================================="

# 分布式端口设置 (防止冲突，随机生成一个端口)
MASTER_PORT=$((29000 + $RANDOM % 1000))
mkdir -p "${OUTPUT_DIR}"
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_backend="c10d" \
    --master_port=${MASTER_PORT} \
    STIP_main.py \
    --output_dir "${OUTPUT_DIR}" \
    --group_name "STIP_opensource" \
    --run_name "hico_run_graph6_base" \
    --HOIDet \
    --validate \
    --no_aux_loss \
    --hoi_aux_loss \
    --dataset_file "hico-det" \
    --data_path "${DATA_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --reduce_lr_on_plateau_patience 2 \
    --lr 5e-4 \
    --resume "${RESUME_CHECKPOINT}" \
    --eval \
    --num_hoi_queries 32 \
    --backbone "resnet50" \
    --pair_dedupe_iou_thr 0.55 \
    --pair_filter_threshold 0 \
    --use_visual_scene_graph \
    --graph_debug \
    --graph_oo_mode text_hybrid \
    --graph_oo_lambda_geo 1.0 \
    --graph_oo_lambda_share 0.5 \
    --graph_oo_lambda_text 0.5 \
    --graph_oo_prior_path "${OO_PRIOR_PATH}" \
    --graph_oo_prior_normalize none \
    --graph_enable_text_graph True \
    --graph_text_features_path "${TEXT_FEATURES_PATH}" \
    --graph_text_mask_path "${TEXT_MASK_PATH}" \
    --graph_text_hoi_ids_path "${TEXT_HOI_IDS_PATH}" \
    --graph_text_object_to_hoi_indices_path "${TEXT_OBJECT_TO_HOI_PATH}" \
    --graph_text_field_order_path "${TEXT_FIELD_ORDER_PATH}" \
    --graph_text_topk 3 \
    --graph_text_pre_topk 20 \
    --graph_text_temperature 0.07 \
    --graph_text_rerank_weight 0.5 \
    --graph_text_sync_mode layerwise \
    --graph_text_dense_injection True \
    --graph_text_final_pair_injection False \
    --graph_enable_candidate_evidence True \
    --graph_evidence_topk 1 \
    --graph_enable_sync_evidence_matching True \
    --graph_enable_layerwise_sync_evidence_matching True \
    --graph_layerwise_sync_write_visual True \
    --graph_layerwise_sync_write_text True \
    --graph_layerwise_sync_final_matching False \
    --graph_layerwise_sync_disable_old_dense_injection True \
    --graph_layerwise_sync_write_reverse_edge False \
    --graph_layerwise_sync_detach_text_update False \
    --graph_layerwise_sync_debug False \
    --graph_num_layers 2 \
    --graph_store_dense_debug_tensors False \
    --graph_text_dynamic_retrieval False \
    --graph_text_sync_query_source base_pair \
    --graph_text_sync_layers_with_visual True \
    --graph_text_edge_inject_reverse True \
    --graph_text_node_scatter_reduce mean \
    --graph_text_prior_injection all \
    --graph_text_use_ho_edge_prior True \
    --graph_text_use_human_node_prior True \
    --graph_text_use_object_node_prior True \
    --graph_text_use_discriminative_prior True \
    --graph_text_use_scene_prior True \
    --graph_text_use_oo_prior True \
    --graph_text_gate_init 0.0 \
    --graph_text_edge_prior_gate_init 0.0 \
    --graph_text_node_prior_gate_init 0.0 \
    --graph_text_scene_prior_gate_init 0.0 \
    --graph_text_oo_prior_gate_init 0.0 \
    --graph_enable_path_context \
    --graph_path_topk 3 \
    --graph_path_dropout 0.1 \
    --graph_path_gate_init 0.0 \
    --graph_path_use_edge_scores \
    --graph_path_score_thresh 0.0 \
    --graph_node_score_thresh 0.05 \
    --graph_min_score_nodes 40 \
    --graph_debug_interval 20 \
    --graph_debug_max_prints 20 \
    --detr_weights "${DETR_WEIGHTS}" \
    --num_workers ${NUM_WORKERS} \
    2>&1 | tee "${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"
    #--eval \
    # --train_detr \
    # --lr_detr 1e-5 \
        # --backbone "resnet101" \     --resume "${RESUME_CHECKPOINT}" \
    #         --pair_filter_threshold 0.08 \
    # --use_pair_score_weighting \
    # --pair_dedupe_iou_thr 0.55 \
#     --graph_oo_mode text_hybrid \
# --graph_oo_lambda_geo 1.0 \
# --graph_oo_lambda_share 0.5 \
# --graph_oo_lambda_text 0.5 \
# --graph_oo_prior_path /path/to/oo_affordance_prior_80.npy \
# --graph_oo_prior_normalize none
