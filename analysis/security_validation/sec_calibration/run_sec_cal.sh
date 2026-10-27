#!/bin/bash
set -Eeuo pipefail


# ================================================================
# Project root
# ================================================================

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_DIR="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_DIR}"

export PROJECT_DIR
export PROJECT_ROOT="${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"


# ================================================================
# Environment
# ================================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export NLTK_DATA="${NLTK_DATA:-./models/nltk_data}"


# ================================================================
# Configuration
# ================================================================

DATASET="c4"

MODEL_PATH="./models/llama3.1-8b-instruct"

MODEL_NAME="$(basename "${MODEL_PATH}")"

EMBEDDER_PATH="./models/bge-small-en-v1.5"

TARGET_LEN=200


# ------------------------------------------------
# Independent Security Calibration
# ------------------------------------------------

SEC_SKIP_LINES=5000

SEC_NUM_SAMPLES=5000

SEC_BATCH_SIZE=16


# ------------------------------------------------
# Existing IAS-W statistics
# ------------------------------------------------

IAS_RANK=64

IAS_EPS_TAG="0p0001"

IAS_CALIB_SKIP_LINES=3000

IAS_CALIB_NUM_SAMPLES=1000

IAS_STATS_KEY="rank${IAS_RANK}_eps${IAS_EPS_TAG}_calibskip${IAS_CALIB_SKIP_LINES}_calibn${IAS_CALIB_NUM_SAMPLES}"

export IAS_STATS_KEY="${IAS_STATS_KEY}"

export IASW_STATS_KEY="${IAS_STATS_KEY}"

export IAS_ALIGNMENT_MODE="dp_local"

export IASW_ALIGNMENT_MODE="dp_local"

export IAS_TARGET_LEN="${TARGET_LEN}"


# ------------------------------------------------
# Security detector
# ------------------------------------------------

KGW_GATE=2.0

TARGET_FPR=0.01


# ================================================================
# Paths
# ================================================================

RAW_FILE="${PROJECT_DIR}/data/raw_${DATASET}.jsonl"


SEC_DATA_DIR="${PROJECT_DIR}/data/security_calibration/${DATASET}/${MODEL_NAME}"

SEC_RESULT_DIR="${PROJECT_DIR}/results/security_validation/sec_calibration/${DATASET}/${MODEL_NAME}"


mkdir -p \
    "${SEC_DATA_DIR}" \
    "${SEC_RESULT_DIR}"


SEC_FILE="${SEC_DATA_DIR}/sec_cal_len${TARGET_LEN}_skip${SEC_SKIP_LINES}_n${SEC_NUM_SAMPLES}.jsonl"


THRESHOLD_FILE="${SEC_RESULT_DIR}/gate_threshold.json"


# ================================================================
# Preflight
# ================================================================

echo
echo "======================================================================"
echo "INDEPENDENT SECURITY CALIBRATION"
echo "======================================================================"
echo "Dataset          : ${DATASET}"
echo "Model            : ${MODEL_NAME}"
echo "Target length    : ${TARGET_LEN}"
echo
echo "IAS-W calibration:"
echo "  skip           : ${IAS_CALIB_SKIP_LINES}"
echo "  n              : ${IAS_CALIB_NUM_SAMPLES}"
echo
echo "Security calibration:"
echo "  skip           : ${SEC_SKIP_LINES}"
echo "  n              : ${SEC_NUM_SAMPLES}"
echo
echo "KGW gate         : ${KGW_GATE}"
echo "Target FPR       : ${TARGET_FPR}"
echo "======================================================================"
echo


if [[ ! -f "${RAW_FILE}" ]]; then
    echo "ERROR: raw dataset not found:"
    echo "${RAW_FILE}"
    exit 1
fi


if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model not found:"
    echo "${MODEL_PATH}"
    exit 1
fi


if [[ ! -d "${EMBEDDER_PATH}" ]]; then
    echo "ERROR: embedder not found:"
    echo "${EMBEDDER_PATH}"
    exit 1
fi


# ================================================================
# Step 1
# Build independent Sec-Cal
# ================================================================

echo
echo "======================================================================"
echo "[STEP 1] Build independent Sec-Cal"
echo "======================================================================"
echo


python analysis/security_validation/sec_calibration/build_sec_cal.py \
    --raw_file "${RAW_FILE}" \
    --output_dir "${SEC_DATA_DIR}" \
    --dataset_type "${DATASET}" \
    --model_path "${MODEL_PATH}" \
    --target_len "${TARGET_LEN}" \
    --skip_lines "${SEC_SKIP_LINES}" \
    --num_samples "${SEC_NUM_SAMPLES}" \
    --batch_size "${SEC_BATCH_SIZE}"


# ================================================================
# Step 2
# Official KGW + IAS-W detection
# ================================================================

echo
echo "======================================================================"
echo "[STEP 2] Official KGW + IAS-W detection"
echo "======================================================================"
echo


python -m src.watermarks.runner \
    --task detect \
    --target_file "${SEC_FILE}" \
    --dataset_type "${DATASET}" \
    --method universal_dual \
    --micro_method kgw \
    --macro_method iasw \
    --model_path "${MODEL_PATH}" \
    --embedder_path "${EMBEDDER_PATH}" \
    --mode baseline_eval


# ================================================================
# Step 3
# Calibrate final gated Fisher threshold
# ================================================================

echo
echo "======================================================================"
echo "[STEP 3] Calibrate gated Fisher threshold"
echo "======================================================================"
echo


python analysis/security_validation/sec_calibration/calibrate_gate.py \
    --input_file "${SEC_FILE}" \
    --output_file "${THRESHOLD_FILE}" \
    --kgw_gate "${KGW_GATE}" \
    --target_fpr "${TARGET_FPR}"


# ================================================================
# Done
# ================================================================

echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"

echo
echo "Detected Sec-Cal:"
echo "${SEC_FILE}"

echo
echo "Frozen security threshold:"
echo "${THRESHOLD_FILE}"

