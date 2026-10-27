#!/bin/bash
set -e

cd ./IASW


# ============================================================
# Configuration
# ============================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DATASET="c4"

MODEL_PATH="./models/llama3.1-8b-instruct"

EMBEDDER_PATH="./models/bge-small-en-v1.5"

BASELINE_FILE="data/main_exp/baseline_triplets/c4/llama3.1-8b-instruct/baseline_len200.jsonl"

OUT_DIR="results/security_validation/spoofing/c4/llama3.1-8b-instruct"

ANCHOR_FILE="results/security_validation/anchor_recovery/c4/llama3.1-8b-instruct/estimated_anchors_wm_N200.pt"


# ============================================================
# IAS-W configuration
# ============================================================

export IASW_TARGET_LEN=200
export IASW_SCORING_PROFILE=dual

# If your main experiment explicitly used a stats key, keep it here.
export IASW_STATS_KEY="rank64_eps0p0001_calibskip3000_calibn1000"


mkdir -p "${OUT_DIR}"


# ============================================================
# Shared generation parameters
#
# IMPORTANT:
# These should match the current main experiment.
# ============================================================

NUM_SAMPLES=300
START_INDEX=200

K=32
MAX_TOTAL=200
MAX_SENT=40

TEMPERATURE=1.0
TOP_K=-1
TOP_P=-1

SEED=20260917


# ============================================================
# 1. Random-K32
# ============================================================

python -m analysis.security_validation.generate_spoofing \
    --baseline_file "${BASELINE_FILE}" \
    --output_file "${OUT_DIR}/random_k32.jsonl" \
    --model_path "${MODEL_PATH}" \
    --embedder_path "${EMBEDDER_PATH}" \
    --dataset_type "${DATASET}" \
    --mode random \
    --target_len 200 \
    --num_samples ${NUM_SAMPLES} \
    --start_index ${START_INDEX} \
    --num_candidates ${K} \
    --max_total_tokens ${MAX_TOTAL} \
    --max_sent_tokens ${MAX_SENT} \
    --temperature ${TEMPERATURE} \
    --top_k ${TOP_K} \
    --top_p ${TOP_P} \
    --seed ${SEED} \
    --device cuda:0


# ============================================================
# 2. Estimated-anchor-K32
# ============================================================

python -m analysis.security_validation.generate_spoofing \
    --baseline_file "${BASELINE_FILE}" \
    --output_file "${OUT_DIR}/estimated_n200_k32.jsonl" \
    --model_path "${MODEL_PATH}" \
    --embedder_path "${EMBEDDER_PATH}" \
    --dataset_type "${DATASET}" \
    --mode estimated \
    --estimated_anchor_file "${ANCHOR_FILE}" \
    --target_len 200 \
    --num_samples ${NUM_SAMPLES} \
    --start_index ${START_INDEX} \
    --num_candidates ${K} \
    --max_total_tokens ${MAX_TOTAL} \
    --max_sent_tokens ${MAX_SENT} \
    --temperature ${TEMPERATURE} \
    --top_k ${TOP_K} \
    --top_p ${TOP_P} \
    --seed ${SEED} \
    --device cuda:0


# ============================================================
# 3. True-anchor oracle upper bound
# ============================================================

python -m analysis.security_validation.generate_spoofing \
    --baseline_file "${BASELINE_FILE}" \
    --output_file "${OUT_DIR}/true_anchor_k32.jsonl" \
    --model_path "${MODEL_PATH}" \
    --embedder_path "${EMBEDDER_PATH}" \
    --dataset_type "${DATASET}" \
    --mode true \
    --target_len 200 \
    --num_samples ${NUM_SAMPLES} \
    --start_index ${START_INDEX} \
    --num_candidates ${K} \
    --max_total_tokens ${MAX_TOTAL} \
    --max_sent_tokens ${MAX_SENT} \
    --temperature ${TEMPERATURE} \
    --top_k ${TOP_K} \
    --top_p ${TOP_P} \
    --seed ${SEED} \
    --device cuda:0


# ============================================================
# Official KGW + IAS-W Fisher detection
# ============================================================

for FILE in \
    random_k32.jsonl \
    estimated_n200_k32.jsonl \
    true_anchor_k32.jsonl

do

    echo
    echo "============================================================"
    echo "DETECT: ${FILE}"
    echo "============================================================"

    python -m src.watermarks.runner \
        --task detect \
        --method universal_dual \
        --model_path "${MODEL_PATH}" \
        --embedder_path "${EMBEDDER_PATH}" \
        --target_file "${OUT_DIR}/${FILE}" \
        --dataset_type "${DATASET}" \
        --num_samples ${NUM_SAMPLES} \
        --mode baseline_eval \
        --micro_method kgw \
        --macro_method iasw \
        --secret_key DUAL_MACRO_2026 \
        --max_total_tokens 200 \
        --device cuda:0

done


# ============================================================
# Evaluate
# ============================================================

python -m analysis.security_validation.generate_spoofing \
    --files \
        "${OUT_DIR}/random_k32.jsonl" \
        "${OUT_DIR}/estimated_n200_k32.jsonl" \
        "${OUT_DIR}/true_anchor_k32.jsonl" \
    --labels \
        Random-K32 \
        Estimated-N200-K32 \
        True-Anchor-K32 \
    --output_dir "${OUT_DIR}/evaluation" \
    --fpr 0.01


echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo
echo "Main result:"
echo "  ${OUT_DIR}/evaluation/spoofing_summary.csv"