#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET=hico-det
exec "${SCRIPT_DIR}/scripts/run_vet_internal_ablation_common.sh" "$@"

