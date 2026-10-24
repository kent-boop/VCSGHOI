#!/bin/bash

# =========================================================
#  B2 SVRM-only 消融 baseline 启动脚本
#  B2 = B1 BLIP+DETR direct decoder
#       + Shared Visual Relation Memory only
#  不启用 VET / LPT / SETM / text graph / candidate evidence / sync evidence
# =========================================================

# 1. 显卡设置
# 单卡示例: GPU_IDS="0" 或 GPU_IDS="2"
# 多卡示例: GPU_IDS="0,1" 或 GPU_IDS="0,1,2,3"
GPU_IDS="0,1,2,3,4,5,6,7"

# 2. 基础配置
# 如果是多卡，BATCH_SIZE 是每张卡的 batch size，总 batch size = BATCH_SIZE * NUM_GPUS
BATCH_SIZE=18
NUM_WORKERS=4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="checkpoints/hico-det_b2_svrm_ablation"
DATA_PATH="/share/share/Datasets/hico_20160224_det"
DETR_WEIGHTS="/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/detr_coco/checkpoint0120.pth"

# 可选：如需从已有 B2 checkpoint 继续或评估，取消下一行注释并确认 checkpoint 与 B2 参数兼容。
RESUME_CHECKPOINT="/wzj/zijian/OSSM-HOI-main/checkpoints/hico-det_b2_svrm_ablation/STIP_opensource/hico_run_b2_svrm/checkpoint.pth"

# =========================================================
#  自动计算逻辑
# =========================================================
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ALGO=Ring
export NCCL_NSOCKS_PERTRACK=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_PROTO=LL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR="127.0.0.1"

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}
MASTER_PORT=$((29000 + RANDOM % 1000))

mkdir -p "${OUTPUT_DIR}"

echo "========================================================="
echo "B2 SVRM-only ablation 启动配置:"
echo "使用显卡 ID: ${GPU_IDS}"
echo "显卡数量 (nproc_per_node): ${NUM_GPUS}"
echo "单卡 Batch Size: ${BATCH_SIZE}"
echo "总 Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "输出目录: ${OUTPUT_DIR}"
echo "模型变体: b2_svrm"
echo "SVRM: enabled, visual-only"
echo "Text/Evidence/SETM modules: disabled"
echo "========================================================="

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_backend="c10d" \
    --master_port=${MASTER_PORT} \
    STIP_main.py \
    --output_dir "${OUTPUT_DIR}" \
    --group_name "STIP_opensource" \
    --run_name "hico_run_b2_svrm" \
    --stip_model_variant "b2_svrm" \
    --HOIDet \
    --validate \
    --no_aux_loss \
    --hoi_aux_loss \
    --dataset_file "hico-det" \
    --data_path "${DATA_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --resume "${RESUME_CHECKPOINT}" \
    --reduce_lr_on_plateau_patience 2 \
    --lr 5e-4 \
    --num_hoi_queries 32 \
    --backbone "resnet50" \
    --pair_dedupe_iou_thr 0.55 \
    --pair_filter_threshold 0 \
    --graph_num_layers 2 \
    --graph_num_heads 8 \
    --graph_k_oo 3 \
    --graph_oo_mode spatial \
    --graph_oo_lambda_geo 1.0 \
    --graph_oo_lambda_share 0.5 \
    --graph_enable_oo_edges \
    --detr_weights "${DETR_WEIGHTS}" \
    --num_workers ${NUM_WORKERS} \
    2>&1 | tee "${OUTPUT_DIR}/train_log_b2_svrm_$(date +%Y%m%d_%H%M%S).txt"

# 如需加载 B2 checkpoint，可在上方取消 RESUME_CHECKPOINT，并在 torchrun 参数中加入：
#     --resume "${RESUME_CHECKPOINT}" \
# 如需只评估，可加入：
#     --eval \
# 注意：B2 SVRM-only 不要启用以下参数：
#     --use_visual_scene_graph
#     --graph_enable_text_graph
#     --graph_enable_candidate_evidence
#     --graph_enable_sync_evidence_matching
#     --graph_enable_layerwise_sync_evidence_matching
#     --graph_oo_mode text_hybrid
#     --graph_oo_prior_path
#     --graph_text_*
