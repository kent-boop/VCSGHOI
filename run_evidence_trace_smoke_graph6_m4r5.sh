#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0}"

if [[ "${GPU_IDS}" == *,* ]]; then
    echo "[Error] The two-image Trace smoke test must use exactly one GPU." >&2
    exit 1
fi

export GPU_IDS
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export EVAL_MAX_SAMPLES=2
export TRACE_EXPECTED_RECORDS=2
export RUN_TAG
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/evidence_trace_smoke_${RUN_TAG}}"

exec bash "${SCRIPT_DIR}/run_evidence_trace_graph6_m4r5_v2.sh"
