#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODE=path_reweight
export RUN_KIND="${RUN_KIND:-formal}"
exec "${SCRIPT_DIR}/run_vet_internal_ablation_hico.sh" "$@"

