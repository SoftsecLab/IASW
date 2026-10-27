#!/usr/bin/env bash
set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "======================================================================"
echo "IASW paper ablation reproduction"
echo "WARNING: some launchers may require GPUs."
echo "======================================================================"

LAUNCHERS=(
    "scripts/ablations/1_ablation_num_candidates.sh"   # Number of candidates (Fig.)
    "scripts/ablations/2_ablation_token_len.sh"        # Target token length (Fig.)
    "scripts/ablations/3_ablation_whitening_ranks.sh"  # Whitening rank sweep (Table)
)

for launcher in "${LAUNCHERS[@]}"; do
    echo
    echo "----------------------------------------------------------------------"
    echo "[RUN] ${launcher}"
    echo "----------------------------------------------------------------------"

    if [[ ! -f "${launcher}" ]]; then
        echo "ERROR: required launcher missing: ${launcher}" >&2
        exit 1
    fi

    bash "${launcher}"
done

echo
echo "Ablation wrapper finished."
echo "Recommended next:"
echo "  bash scripts/reproduce/50_analysis.sh"
