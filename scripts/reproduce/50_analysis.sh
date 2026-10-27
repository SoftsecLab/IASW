#!/usr/bin/env bash
set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "======================================================================"
echo "Reproduce offline/special analyses"
echo "======================================================================"

run_if_exists() {
    local label="$1"
    shift
    local script="$1"
    shift

    echo
    echo "----------------------------------------------------------------------"
    echo "${label}"
    echo "----------------------------------------------------------------------"

    if [[ ! -f "${script}" ]]; then
        echo "[SKIP] missing ${script}"
        return
    fi

    python "${script}" "$@"
}

# IAS-W alignment table used by the manuscript.
ALIGN_INPUT="${ALIGN_INPUT:-results/main_exp/c4/llama3.1-8b-instruct/records/iasw.jsonl}"
ALIGN_OUT="${ALIGN_OUT:-results/analysis/iasw/alignment_ablation_human}"

if [[ -f "${ALIGN_INPUT}" ]]; then
    run_if_exists \
        "IAS-W alignment ablation" \
        analysis/iasw_alignment/extract_iasw_alignment_ablation.py \
        --input_file "${ALIGN_INPUT}" \
        --output_dir "${ALIGN_OUT}" \
        --negative_sources human
else
    echo "[SKIP] IAS-W alignment: missing ${ALIGN_INPUT}"
fi

# Ordinary paper-side robustness/quality/efficiency are rebuilt by build_paper.
echo
echo "----------------------------------------------------------------------"
echo "Paper-side analysis aggregation"
echo "----------------------------------------------------------------------"
bash scripts/paper.sh

echo
echo "Offline analysis wrapper finished."
