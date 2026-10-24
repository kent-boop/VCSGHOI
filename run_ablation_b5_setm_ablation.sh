#!/bin/bash

# =========================================================
#  B5-SETM-Ablation 消融模型启动脚本
#
#  B5 = B4-VET-LPT + SETM 内部策略消融
#  注意：B5 不是在消融 VET 或 LPT。
#       B5 默认保留完整 VET：SVRM + Candidate Evidence Topology Induction
#       B5 默认保留完整 LPT：object-conditioned retrieval + structured prototype slots
#       本脚本只通过 SETM_ABLATION_STAGE 切换 SETM 内部策略。
#
#  五行表对应关系：
#    SETM_ABLATION_STAGE=1  One-way Match
#      - 显式 visual evidence token -> prototype slot matching
#      - 只更新 visual evidence token
#      - evidence/prototype 都使用 uniform weight
#      - 只在 final-stage 执行
#
#    SETM_ABLATION_STAGE=2  + Bi-dir. Update
#      - 在 Stage 1 基础上增加 text_to_visual_attn
#      - 更新 prototype slots
#      - 更新后的 prototype slots 会反向影响最终 anchor_delta
#      - evidence/prototype 仍为 uniform weight
#      - 只在 final-stage 执行
#
#    SETM_ABLATION_STAGE=3  + Evid. Reweight.
#      - 在 Stage 2 基础上启用 learned evidence reweighting
#      - prototype 仍为 uniform weight
#      - 只在 final-stage 执行
#
#    SETM_ABLATION_STAGE=4  + Proto. Reweight.
#      - 在 Stage 3 基础上启用 learned prototype reweighting
#      - 启用 prototype coverage
#      - full final-stage SETM
#
#    SETM_ABLATION_STAGE=5  + Layerwise Sync
#      - 使用 Stage 4 SETM 机制
#      - 每个 visual relation reasoning layer 后执行一次 SETM
#      - 写回 visual HO edge state，并携带临时 image-conditioned prototype slots
#      - 不修改离线 language prototype bank
#      - 必须保持 graph_text_dynamic_retrieval=False，避免重建 text state 抹掉 visual-to-text 更新
# =========================================================

# 1. 显卡设置
# 单卡示例: GPU_IDS="0" 或 GPU_IDS="2"
# 多卡示例: GPU_IDS="0,1" 或 GPU_IDS="0,1,2,3"
GPU_IDS="0,1,2,3,4,5,6,7"

# 2. SETM 内部消融行号，只改这里即可跑五行表
# 1 = One-way Match
# 2 = + Bi-dir. Update
# 3 = + Evid. Reweight.
# 4 = + Proto. Reweight.
# 5 = + Layerwise Sync
SETM_ABLATION_STAGE=4

# 3. 基础配置
# 如果是多卡，BATCH_SIZE 是每张卡的 batch size，总 batch size = BATCH_SIZE * NUM_GPUS。
# B5 Stage 5 比 B4 多 layerwise SETM，显存更高；显存不足时优先调小 BATCH_SIZE。
BATCH_SIZE=16
NUM_WORKERS=4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/share/share/Datasets/hico_20160224_det"
DETR_WEIGHTS="/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/detr_coco/checkpoint0120.pth"

# HICO LPT 文本图资源。B5 必须保留完整 LPT，因此这些路径必须有效。
TEXT_GRAPH_DIR="${SCRIPT_DIR}/hico_text_graph_blip256"
TEXT_FEATURES_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_features_6x256.npy"
TEXT_MASK_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_mask_6.npy"
TEXT_HOI_IDS_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_hoi_ids.json"
TEXT_OBJECT_TO_HOI_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_object_to_hoi_indices.json"
TEXT_FIELD_ORDER_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_field_order.json"

# 可选：如需从已有 B5 checkpoint 继续或评估，设置为 checkpoint 路径；留空则从 DETR 权重开始。
# 注意：B5 不建议直接 resume B4 checkpoint 做严格 ablation，除非你明确接受新 SETM 参数随机初始化。
RESUME_CHECKPOINT="/wzj/zijian/OSSM-HOI-main/checkpoints/hico-det_b5_setm_stage4_ablation/STIP_opensource/hico_run_b5_setm_stage4/checkpoint.pth"

# 4. 自动生成输出目录与 run name
OUTPUT_DIR="checkpoints/hico-det_b5_setm_stage${SETM_ABLATION_STAGE}_ablation"
RUN_NAME="hico_run_b5_setm_stage${SETM_ABLATION_STAGE}"

# =========================================================
#  根据 SETM_ABLATION_STAGE 自动设置 Stage 5 专属参数
# =========================================================
LAYERWISE_ARGS=()
if [[ "${SETM_ABLATION_STAGE}" == "5" ]]; then
    # Stage 5 = Layerwise Sync。
    # 下面这些参数只对第 5 行有意义。
    # graph_text_dynamic_retrieval=False:
    #   必须关闭。否则每层重建 text_state 会擦掉上一层 visual-to-text 更新。
    # graph_layerwise_sync_write_visual=True:
    #   将 CETI+SETM anchor delta 写回当前 HO edge state。
    # graph_layerwise_sync_write_text=True:
    #   将 updated_prototype_slots 作为临时 image-conditioned text state 传给下一层。
    # graph_layerwise_sync_disable_old_dense_injection=True:
    #   防止旧 dense injection 路径造成重复注入。
    # graph_layerwise_sync_final_matching=False:
    #   防止 layerwise SETM 后再额外做一次 final-stage SETM。
    LAYERWISE_ARGS=(
        --graph_text_dynamic_retrieval False
        --graph_layerwise_sync_write_visual True
        --graph_layerwise_sync_write_text True
        --graph_layerwise_sync_disable_old_dense_injection True
        --graph_layerwise_sync_final_matching False
        --graph_layerwise_sync_write_reverse_edge False
        --graph_layerwise_sync_detach_text_update False
        --graph_layerwise_sync_debug False
    )
else
    # Stage 1-4 = final-stage SETM。
    # 不启用 layerwise scheduling；SETM anchor_delta 是唯一替代 B4 simple LPT residual 的语言原型 refinement。
    LAYERWISE_ARGS=(
        --graph_text_dynamic_retrieval False
        --graph_layerwise_sync_write_visual True
        --graph_layerwise_sync_write_text True
        --graph_layerwise_sync_disable_old_dense_injection True
        --graph_layerwise_sync_final_matching False
    )
fi

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

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}
MASTER_PORT=$((29000 + RANDOM % 1000))

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_ARGS=(--resume "${RESUME_CHECKPOINT}")
fi

echo "========================================================="
echo "B5-SETM-Ablation 启动配置:"
echo "使用显卡 ID: ${GPU_IDS}"
echo "显卡数量 (nproc_per_node): ${NUM_GPUS}"
echo "单卡 Batch Size: ${BATCH_SIZE}"
echo "总 Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "输出目录: ${OUTPUT_DIR}"
echo "run_name: ${RUN_NAME}"
echo "模型变体: b5_setm_ablation"
echo "SETM_ABLATION_STAGE: ${SETM_ABLATION_STAGE}"
echo "VET: enabled, complete SVRM + CETI"
echo "LPT: enabled, complete retrieval + structured prototype slots"
echo "B4 simple LPT residual: disabled when SETM ablation is active"
echo "========================================================="

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_backend="c10d" \
    --master_port=${MASTER_PORT} \
    STIP_main.py \
    --output_dir "${OUTPUT_DIR}" \
    --group_name "STIP_opensource" \
    --run_name "${RUN_NAME}" \
    --model_variant "b5_setm_ablation" \
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
    --use_edge_blip_fusion True \
    --use_blip_global_memory True \
    --use_vet True \
    --use_lpt True \
    --use_setm_ablation True \
    --setm_ablation_stage ${SETM_ABLATION_STAGE} \
    --disable_simple_lpt_residual_for_setm True \
    --vet_gate_init 0.0 \
    --lpt_gate_init 0.0 \
    --setm_visual_gate_init 0.0 \
    --setm_text_gate_init 0.0 \
    --setm_anchor_gate_init 0.0 \
    --graph_sync_dropout 0.1 \
    --graph_sync_debug False \
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
    "${LAYERWISE_ARGS[@]}" \
    --detr_weights "${DETR_WEIGHTS}" \
    --num_workers ${NUM_WORKERS} \
    2>&1 | tee "${OUTPUT_DIR}/train_log_b5_setm_stage${SETM_ABLATION_STAGE}_$(date +%Y%m%d_%H%M%S).txt"

# =========================================================
#  快速修改说明
# =========================================================
# 只跑五行 SETM 表时，通常只需要修改：
#   SETM_ABLATION_STAGE=1/2/3/4/5
#
# 常用显存相关修改：
#   GPU_IDS       控制用哪几张卡
#   BATCH_SIZE    控制单卡 batch size
#
# VET 完整性参数，不建议为了 B5 SETM 内部消融关闭：
#   --use_vet True
#   --graph_num_layers 2
#   --graph_evidence_topk 4
#   graph_enable_hh_edges 默认就是 False；不要写 "--graph_enable_hh_edges False"，
#   因为该参数是 store_true 开关，不接收额外的 False 字符串。
#
# LPT 完整性参数，不建议为了 B5 SETM 内部消融关闭：
#   --use_lpt True
#   --graph_text_topk 5
#   --graph_text_pre_topk 20
#   --graph_text_num_layers 1
#
# SETM gate 初始化参数：
#   --setm_visual_gate_init 0.0   控制 visual evidence token update 初始门控
#   --setm_text_gate_init 0.0     控制 prototype slot update 初始门控
#   --setm_anchor_gate_init 0.0   控制 SETM anchor_delta 写回 query 的初始门控
#
# Stage 5 专属参数：
#   --graph_text_dynamic_retrieval False
#   --graph_layerwise_sync_write_visual True
#   --graph_layerwise_sync_write_text True
#   --graph_layerwise_sync_disable_old_dense_injection True
#   --graph_layerwise_sync_final_matching False
#
# 如需只评估，可加入：
#   --eval \
