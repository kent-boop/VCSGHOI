#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_ENV_BIN="${PYTHON_ENV_BIN:-/share/apps/anaconda3/envs/SPLHOI/bin}"
export PATH="${PYTHON_ENV_BIN}:${PATH}"

TOP20_CSV="${TOP20_CSV:-${SCRIPT_DIR}/outputs/evidence_trace_stip_final_20260731_190715/supplementary_top20_mechanism_relative/ranking/top20_mechanism_candidates.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/evidence_trace_top20_detailed_${RUN_TAG}}"
IMAGE_IDS_FILE="${IMAGE_IDS_FILE:-${OUTPUT_ROOT}/top20_image_ids.json}"
LOCAL_PACKAGE_DIR="${LOCAL_PACKAGE_DIR:-${OUTPUT_ROOT}/local_vet_setm_package_detailed}"
REMAPPED_CSV="${REMAPPED_CSV:-${OUTPUT_ROOT}/top20_detailed_pair_mapping.csv}"
PAIR_MAPPING_REPORT="${PAIR_MAPPING_REPORT:-${OUTPUT_ROOT}/top20_detailed_pair_mapping.json}"

ACTION_NAMES_JSON="${ACTION_NAMES_JSON:-${SCRIPT_DIR}/outputs/evidence_trace_graph6_m4r5_full/label_maps/action_names.json}"
OBJECT_NAMES_JSON="${OBJECT_NAMES_JSON:-${SCRIPT_DIR}/outputs/evidence_trace_graph6_m4r5_full/label_maps/object_names.json}"
NODE_NAMES_JSON="${NODE_NAMES_JSON:-${SCRIPT_DIR}/outputs/evidence_trace_graph6_m4r5_full/label_maps/coco_node_names.json}"
PROMPT_METADATA_JSON="${PROMPT_METADATA_JSON:-${SCRIPT_DIR}/hico_text_graph_blip256/hico_hoi_graph_blip256_prompt_metadata.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/share/share/Datasets/hico_20160224_det/images/test2015}"

mkdir -p "${OUTPUT_ROOT}"

python - "${TOP20_CSV}" "${IMAGE_IDS_FILE}" <<'PY'
import csv
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = list(csv.DictReader(source.open("r", encoding="utf-8")))
image_ids = list(dict.fromkeys(int(row["image_id"]) for row in rows))
if len(image_ids) != 20:
    raise SystemExit(f"Expected exactly 20 unique image IDs, got {len(image_ids)}")
output.write_text(json.dumps({"image_ids": image_ids}, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(image_ids)} image IDs to {output}")
PY

export RUN_TAG
export OUTPUT_ROOT
export EVAL_IMAGE_IDS_FILE="${IMAGE_IDS_FILE}"
export EVAL_MAX_SAMPLES=0
export TRACE_EXPECTED_RECORDS=20
export TRACE_EACH_LAYER=True
export TRACE_INCLUDE_MATCH_LOGITS=True
export TRACE_INCLUDE_FEATURES=True
export GRAPH_STORE_DENSE_DEBUG_TENSORS=True
export TRACE_FLOAT_DTYPE="${TRACE_FLOAT_DTYPE:-float32}"
export TRACE_OVERWRITE="${TRACE_OVERWRITE:-False}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export GPU_IDS="${GPU_IDS:-4,5,6,7}"

bash "${SCRIPT_DIR}/run_evidence_trace_stip_final.sh"

python "${SCRIPT_DIR}/tools/evidence_vis/validate_detailed_vet_setm_trace.py" \
    --trace-dir "${OUTPUT_ROOT}/traces" \
    --image-ids-file "${IMAGE_IDS_FILE}" \
    --expected-layers 2 \
    --expected-m 4 \
    --expected-k 5 \
    --expected-roles 6 \
    --require-features \
    --require-dense

python "${SCRIPT_DIR}/tools/evidence_vis/remap_ranking_pairs_to_trace.py" \
    --ranking-csv "${TOP20_CSV}" \
    --trace-dir "${OUTPUT_ROOT}/traces" \
    --output-csv "${REMAPPED_CSV}" \
    --report-json "${PAIR_MAPPING_REPORT}" \
    --warning-iou 0.8

python "${SCRIPT_DIR}/tools/evidence_vis/export_vet_setm_local_package.py" \
    --ranking-csv "${REMAPPED_CSV}" \
    --trace-dir "${OUTPUT_ROOT}/traces" \
    --output-dir "${LOCAL_PACKAGE_DIR}" \
    --action-names-json "${ACTION_NAMES_JSON}" \
    --object-names-json "${OBJECT_NAMES_JSON}" \
    --node-names-json "${NODE_NAMES_JSON}" \
    --prompt-metadata-json "${PROMPT_METADATA_JSON}" \
    --image-root "${IMAGE_ROOT}" \
    --copy-source-images \
    --copy-raw-records

mkdir -p "${LOCAL_PACKAGE_DIR}/provenance"
cp "${REMAPPED_CSV}" "${LOCAL_PACKAGE_DIR}/provenance/"
cp "${PAIR_MAPPING_REPORT}" "${LOCAL_PACKAGE_DIR}/provenance/"
cp "${OUTPUT_ROOT}/traces/detailed_vet_setm_validation.json" \
    "${LOCAL_PACKAGE_DIR}/provenance/"

echo "Detailed trace     : ${OUTPUT_ROOT}/traces"
echo "Local data package : ${LOCAL_PACKAGE_DIR}"
