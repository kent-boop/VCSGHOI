#!/bin/bash

set -euo pipefail

# =========================================================
#  STIP 后处理消融验证脚本
#  顺序验证：baseline -> dedupe only -> hard filter only -> weighting only
# =========================================================

# 可通过环境变量覆盖，例如：GPU_IDS="0,1" BATCH_SIZE=8 ./run_postprocess_ablation.sh
GPU_IDS=${GPU_IDS:-"0,2,3,4,5,7"}
BATCH_SIZE=${BATCH_SIZE:-24}
NUM_WORKERS=${NUM_WORKERS:-4}
OUTPUT_BASE=${OUTPUT_BASE:-"checkpoints/hico-det_graph2/postprocess_ablation"}
DATA_PATH=${DATA_PATH:-"/share/share/Datasets/hico_20160224_det"}
DETR_WEIGHTS=${DETR_WEIGHTS:-"/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/detr_coco/checkpoint0120.pth"}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-"/wzj/zijian/OSSM-HOI-main/checkpoints/hico-det_graph2/STIP_opensource/hico_run1/best.pth"}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_NSOCKS_PERTRACK=${NCCL_NSOCKS_PERTRACK:-4}
export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-2}
export NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY:-1}
export NCCL_PROTO=${NCCL_PROTO:-LL}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export PYTHONUNBUFFERED=1
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${OUTPUT_BASE}"

echo "========================================================="
echo "启动后处理消融验证"
echo "GPU_IDS: ${GPU_IDS}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "BATCH_SIZE per GPU: ${BATCH_SIZE}"
echo "OUTPUT_BASE: ${OUTPUT_BASE}"
echo "RESUME_CHECKPOINT: ${RESUME_CHECKPOINT}"
echo "========================================================="

run_eval() {
    local name="$1"
    local pair_filter_threshold="$2"
    local pair_dedupe_iou_thr="$3"
    local use_pair_weighting="$4"

    local output_dir="${OUTPUT_BASE}/${TIMESTAMP}_${name}"
    local log_file="${output_dir}/eval.log"
    local master_port=$((29000 + RANDOM % 1000))

    mkdir -p "${output_dir}"

    echo "=========================================================" | tee "${log_file}"
    echo "Running ablation: ${name}" | tee -a "${log_file}"
    echo "pair_filter_threshold=${pair_filter_threshold}" | tee -a "${log_file}"
    echo "pair_dedupe_iou_thr=${pair_dedupe_iou_thr}" | tee -a "${log_file}"
    echo "use_pair_score_weighting=${use_pair_weighting}" | tee -a "${log_file}"
    echo "output_dir=${output_dir}" | tee -a "${log_file}"
    echo "=========================================================" | tee -a "${log_file}"

    local extra_args=()
    if [[ "${use_pair_weighting}" == "1" ]]; then
        extra_args+=(--use_pair_score_weighting)
    fi
    if [[ -n "${RESUME_CHECKPOINT}" ]]; then
        extra_args+=(--resume "${RESUME_CHECKPOINT}")
    fi

    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --rdzv_endpoint="${MASTER_ADDR}:${master_port}" \
        --rdzv_backend="c10d" \
        --master_port="${master_port}" \
        STIP_main.py \
        --output_dir "${output_dir}" \
        --group_name "STIP_opensource" \
        --run_name "${name}" \
        --HOIDet \
        --validate \
        --no_aux_loss \
        --hoi_aux_loss \
        --dataset_file "hico-det" \
        --data_path "${DATA_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --reduce_lr_on_plateau_patience 2 \
        --lr 5e-4 \
        --pair_filter_threshold "${pair_filter_threshold}" \
        --pair_dedupe_iou_thr "${pair_dedupe_iou_thr}" \
        --num_hoi_queries 32 \
        --backbone "resnet50" \
        --eval \
        --use_visual_scene_graph \
        --graph_debug \
        --graph_node_score_thresh 0.05 \
        --graph_min_score_nodes 40 \
        --graph_debug_interval 20 \
        --graph_debug_max_prints 20 \
        --detr_weights "${DETR_WEIGHTS}" \
        --num_workers "${NUM_WORKERS}" \
        "${extra_args[@]}" \
        2>&1 | tee -a "${log_file}"
}

run_eval "baseline_no_postprocess" "0" "0" "0"
run_eval "dedupe_only_iou055" "0" "0.55" "0"
run_eval "hard_filter_only_thr008" "0.08" "0" "0"
run_eval "weighting_only" "0" "0" "1"

echo "========================================================="
echo "后处理消融验证完成。日志目录：${OUTPUT_BASE}/${TIMESTAMP}_*"
echo "========================================================="
