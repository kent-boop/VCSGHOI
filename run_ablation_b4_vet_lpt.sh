#!/bin/bash

# =========================================================
#  B4-VET-LPT 消融模型启动脚本
#  B4-VET-LPT = B3-VET
#               + Language Prototype Topology only
#  包含：B1 BLIP+DETR direct decoder、完整 visual-only VET
#        (SVRM + Candidate Evidence Chain Induction)、LPT prototype context residual
#  不启用 SETM/STEM / explicit evidence-prototype matching / layerwise synchronization
# =========================================================

# 1. 显卡设置
# 单卡示例: GPU_IDS="0" 或 GPU_IDS="2"
# 多卡示例: GPU_IDS="0,1" 或 GPU_IDS="0,1,2,3"
GPU_IDS="0,1,2,3,4,5,6,7"

# 2. 基础配置
# 如果是多卡，BATCH_SIZE 是每张卡的 batch size，总 batch size = BATCH_SIZE * NUM_GPUS
# B4 在 B3-VET 基础上增加 LPT 文本原型检索与文本图推理，显存略高于 B3。
BATCH_SIZE=16
NUM_WORKERS=4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="checkpoints/hico-det_b4_vet_lpt_ablation"
DATA_PATH="/share/share/Datasets/hico_20160224_det"
DETR_WEIGHTS="/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/detr_coco/checkpoint0120.pth"

# HICO LPT 文本图资源
TEXT_GRAPH_DIR="${SCRIPT_DIR}/hico_text_graph_blip256"
TEXT_FEATURES_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_features_6x256.npy"
TEXT_MASK_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_mask_6.npy"
TEXT_HOI_IDS_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_hoi_ids.json"
TEXT_OBJECT_TO_HOI_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_object_to_hoi_indices.json"
TEXT_FIELD_ORDER_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_field_order.json"

# 可选：如需从已有 B4 checkpoint 继续或评估，设置为 checkpoint 路径；留空则从 DETR 权重开始。
# RESUME_CHECKPOINT=""

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

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_ARGS=(--resume "${RESUME_CHECKPOINT}")
fi

echo "========================================================="
echo "B4-VET-LPT ablation 启动配置:"
echo "使用显卡 ID: ${GPU_IDS}"
echo "显卡数量 (nproc_per_node): ${NUM_GPUS}"
echo "单卡 Batch Size: ${BATCH_SIZE}"
echo "总 Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "输出目录: ${OUTPUT_DIR}"
echo "模型变体: b4_vet_lpt"
echo "VET: enabled (SVRM + candidate evidence)"
echo "LPT: enabled (prototype slots -> compact language context residual)"
echo "SETM/STEM/layerwise sync: disabled/ignored"
echo "========================================================="

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_backend="c10d" \
    --master_port=${MASTER_PORT} \
    STIP_main.py \
    --output_dir "${OUTPUT_DIR}" \
    --group_name "STIP_opensource" \
    --run_name "hico_run_b4_vet_lpt" \
    --stip_model_variant "b4_vet_lpt" \
    --HOIDet \
    --validate \
    --no_aux_loss \
    --hoi_aux_loss \
    --dataset_file "hico-det" \
    --data_path "${DATA_PATH}" \
    --batch_size ${BATCH_SIZE} \
    "${RESUME_ARGS[@]}" \
    --reduce_lr_on_plateau_patience 2 \
    --lr 5e-4 \
    --num_hoi_queries 32 \
    --backbone "resnet50" \
    --pair_dedupe_iou_thr 0.55 \
    --pair_filter_threshold 0 \
    --use_vet True \
    --use_lpt True \
    --vet_gate_init 0.0 \
    --lpt_gate_init 0.0 \
    --lpt_query_source vet_pair \
    --lpt_context_source slots \
    --lpt_use_selected_score_weight True \
    --vet_node_score_thresh 0.05 \
    --vet_min_score_nodes 0 \
    --vet_max_extra_nodes 100 \
    --graph_num_layers 2 \
    --graph_num_heads 8 \
    --graph_k_oo 3 \
    --graph_oo_mode hybrid \
    --graph_oo_lambda_geo 1.0 \
    --graph_oo_lambda_share 0.5 \
    --graph_oo_min_score 0.0 \
    --graph_enable_oo_edges \
    --graph_oo_fallback_to_spatial \
    --graph_evidence_topk 4 \
    --graph_evidence_gate_init 0.0 \
    --graph_evidence_dropout 0.1 \
    --graph_evidence_allow_node_only_context False \
    --graph_evidence_debug False \
    --graph_text_features_path "${TEXT_FEATURES_PATH}" \
    --graph_text_mask_path "${TEXT_MASK_PATH}" \
    --graph_text_hoi_ids_path "${TEXT_HOI_IDS_PATH}" \
    --graph_text_object_to_hoi_indices_path "${TEXT_OBJECT_TO_HOI_PATH}" \
    --graph_text_field_order_path "${TEXT_FIELD_ORDER_PATH}" \
    --graph_text_topk 5 \
    --graph_text_pre_topk 20 \
    --graph_text_temperature 0.07 \
    --graph_text_rerank_weight 0.5 \
    --graph_text_use_discriminative_rerank True \
    --graph_text_num_layers 1 \
    --graph_text_num_heads 8 \
    --graph_text_dropout 0.1 \
    --graph_text_use_scene_token True \
    --graph_text_use_oo_context_token True \
    --graph_text_prototype_slot_fallback_to_raw True \
    --graph_text_prototype_slot_debug False \
    --graph_enable_text_graph False \
    --graph_text_dynamic_retrieval False \
    --graph_enable_sync_evidence_matching False \
    --graph_enable_layerwise_sync_evidence_matching False \
    --detr_weights "${DETR_WEIGHTS}" \
    --num_workers ${NUM_WORKERS} \
    2>&1 | tee "${OUTPUT_DIR}/train_log_b4_vet_lpt_$(date +%Y%m%d_%H%M%S).txt"

# 如需加载 B4 checkpoint，可设置 RESUME_CHECKPOINT，例如：
# RESUME_CHECKPOINT="/path/to/checkpoint.pth"
# 如需只评估，可加入：
#     --eval \
# 注意：B4-VET-LPT 是 LPT-only after VET，不要启用以下参数：
#     --graph_enable_sync_evidence_matching True
#     --graph_enable_layerwise_sync_evidence_matching True
#     --graph_text_sync_mode layerwise
#     --graph_text_dense_injection True
#     --graph_text_final_pair_injection True
#     --graph_text_prior_injection all
#     --graph_enable_reciprocal_setm True
