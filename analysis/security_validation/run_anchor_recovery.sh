#!/bin/bash
set -e

cd ./IASW


# ============================================================
# Config
# ============================================================

DATASET="c4"
MODEL="llama3.1-8b-instruct"

INPUT_FILE="results/main_exp/${DATASET}/${MODEL}/records/dual_kgw_iasw.jsonl"

EMBEDDER="./models/bge-small-en-v1.5"

# ------------------------------------------------------------
# IMPORTANT:
# 改成你正式 IAS-W 使用的 stats 文件
# 不要重新估计 whitening
# ------------------------------------------------------------

STATS_FILE="cache/iasw_stats/c4/llama3.1-8b-instruct/ias_stats_c4_llama3.1-8b-instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt"


OUT_DIR="results/security_validation/anchor_recovery/${DATASET}/${MODEL}"


# ============================================================
# Sanity check
# ============================================================

if [ ! -f "${INPUT_FILE}" ]; then
    echo "[ERROR] missing input:"
    echo "  ${INPUT_FILE}"
    exit 1
fi

if [ ! -f "${STATS_FILE}" ]; then
    echo "[ERROR] missing IAS-W stats:"
    echo "  ${STATS_FILE}"
    echo
    echo "Please replace STATS_FILE with the stats file used by the"
    echo "current formal IAS-W detector."
    exit 1
fi


mkdir -p "${OUT_DIR}"


# ============================================================
# Step 1
# Multi-document anchor estimation
# ============================================================

python analysis/security_validation/anchor_recovery.py \
    --input_file "${INPUT_FILE}" \
    --stats_file "${STATS_FILE}" \
    --embedder_path "${EMBEDDER}" \
    --output_dir "${OUT_DIR}" \
    --secret_key "DUAL_MACRO_2026" \
    --device cuda:0 \
    --batch_size 64 \
    --max_positions 12 \
    --n_values 10 25 50 100 200 400 \
    --sources wm no_wm human


# ============================================================
# Step 2
# Summary + figures
# ============================================================

python analysis/security_validation/evaluate_anchor_recovery.py \
    --input_csv "${OUT_DIR}/anchor_recovery_per_position.csv" \
    --output_dir "${OUT_DIR}"


echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo
echo "Results:"
echo "  ${OUT_DIR}/anchor_recovery_summary.csv"
echo "  ${OUT_DIR}/anchor_recovery_curve.pdf"
echo "  ${OUT_DIR}/anchor_recovery_by_position_N400.pdf"