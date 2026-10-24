#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-hico-det}"
RUN_KIND="${RUN_KIND:-formal}"
DRY_RUN="${DRY_RUN:-1}"
MODES=(anchor_raw anchor_svrm endpoints context_uniform path_selection path_reweight)
SEEDS=(3407 3408 3409)

if [[ "${DATASET}" == "hico-det" ]]; then
  LAUNCHER="${PROJECT_ROOT}/run_vet_internal_ablation_hico.sh"
elif [[ "${DATASET}" == "vcoco" ]]; then
  LAUNCHER="${PROJECT_ROOT}/run_vet_internal_ablation_vcoco.sh"
else
  echo "Invalid DATASET=${DATASET}" >&2; exit 2
fi

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    command=(env MODE="${mode}" SEED="${seed}" RUN_KIND="${RUN_KIND}" "${LAUNCHER}")
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%q ' "${command[@]}"; printf '\n'
    else
      "${command[@]}"
    fi
  done
done

