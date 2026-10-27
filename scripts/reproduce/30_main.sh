#!/usr/bin/env bash
set -Eeuo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "======================================================================"
echo "IASW main experiment reproduction"
echo "WARNING: this may require substantial GPU time."
echo "======================================================================"

RUN_DOWNLOAD="${RUN_DOWNLOAD:-0}"
RUN_PRECALIB="${RUN_PRECALIB:-1}"
RUN_CALIB="${RUN_CALIB:-1}"
RUN_MAIN="${RUN_MAIN:-1}"

run_step() {
    local flag="$1"
    local script="$2"
    local label="$3"

    if [[ "${flag}" != "1" ]]; then
        echo "[SKIP] ${label}"
        return
    fi

    if [[ ! -f "${script}" ]]; then
        echo "ERROR: required script missing: ${script}" >&2
        exit 1
    fi

    echo
    echo "----------------------------------------------------------------------"
    echo "${label}"
    echo "----------------------------------------------------------------------"
    bash "${script}"
}

run_step "${RUN_DOWNLOAD}" \
    scripts/main/0_0_download_dataset.sh \
    "Step 0: download/prepare datasets"

run_step "${RUN_PRECALIB}" \
    scripts/main/0_1_run_pre_calibration_simmark_semstamp.sh \
    "Step 1: pre-calibration baselines"

run_step "${RUN_CALIB}" \
    scripts/main/0_2_build_calib_table.sh \
    "Step 2: build calibration tables"

run_step "${RUN_MAIN}" \
    scripts/main/1_run_main_pipeline.sh \
    "Step 3: main experiments"

echo
echo "Main experiment wrapper finished."
echo "Recommended next:"
echo "  bash scripts/reproduce/10_paper.sh"
