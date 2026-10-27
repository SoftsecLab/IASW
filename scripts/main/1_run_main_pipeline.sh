#!/bin/bash
set -Eeuo pipefail

# ================================================================
# PROJECT ROOT BOOTSTRAP
# ================================================================

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_SCAN="${_SCRIPT_DIR}"

while [[ "${_PROJECT_SCAN}" != "/" && ! -d "${_PROJECT_SCAN}/src" ]]; do
    _PROJECT_SCAN="$(dirname "${_PROJECT_SCAN}")"
done
if [[ ! -d "${_PROJECT_SCAN}/src" ]]; then
    echo "ERROR: cannot locate project root containing src/" >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-${_PROJECT_SCAN}}"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_DIR}}"
SCRIPT_DIR="${SCRIPT_DIR:-${_SCRIPT_DIR}}"

cd "${PROJECT_DIR}"

export PROJECT_DIR
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"


# =================================================================
# 0. 环境与运行模式
# =================================================================

# 默认使用物理 GPU 0。
# 如需指定其他 GPU：
#
# CUDA_VISIBLE_DEVICES=1 ./xxx.sh
#
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NLTK_DATA="${NLTK_DATA:-./models/nltk_data}"


# ================================================================
# Test / Formal mode
# ================================================================

IS_TEST=0
NUM_SAMPLES=500

if [[ "${1:-}" == "--test" ]]; then
    IS_TEST=1
    NUM_SAMPLES=5
    echo "==================================================="
    echo "🟢 TEST 模式"
    echo "样本数: ${NUM_SAMPLES}"
    echo "==================================================="
fi

# =================================================================
# 1. 日志
# =================================================================

if [[ ${IS_TEST} -eq 0 ]]; then

    LOG_DIR="${PROJECT_DIR}/results/logs/main_exp"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "==================================================="
    echo "⚙️ 启动统一文本水印基准测试"
    echo "模式    : Formal"
    echo "样本数  : ${NUM_SAMPLES}"
    echo "日志    : ${LOG_FILE}"
    echo "项目目录: ${PROJECT_DIR}"
    echo "==================================================="

fi


# =================================================================
# 2. 全局实验参数
# =================================================================
# -------------------------------------------------
# Backbone
# -------------------------------------------------

LLM_PATH="./models/llama3.1-8b-instruct"
# 其他 backbone
LLM_PATH="./models/opt-1.3b-semstamp"
LLM_PATH="./models/Llama-3.2-3B-Instruct"
LLM_PATH="./models/qwen2.5-7b-instruct"
LLM_PATH="./models/Mistral-7B-Instruct"

MODEL_NAME="$(basename "${LLM_PATH}")"

# -------------------------------------------------
# IAS-W / quality sentence embedder
# -------------------------------------------------
EMBEDDER_PATH="./models/bge-small-en-v1.5"

# -------------------------------------------------
# Attack models
# -------------------------------------------------

PARAPHRASE_PATH="./models/tuner007_pegasus_paraphrase"
DIPPER_PATH="./models/dipper-paraphraser-xxl"

FWD_TRANS_PATH="./models/opus-mt-en-fr"
BWD_TRANS_PATH="./models/opus-mt-fr-en"

NLLB_PATH="./models/nllb-200-distilled-600M"


# -------------------------------------------------
# Dataset
DATASETS=(
    "c4"
    # "wiki"
    # "dolly"
)


# -------------------------------------------------
# Generation
# -------------------------------------------------
TARGET_LEN=200
# =================================================================
# 2.1 IAS-W default runtime config
# =================================================================

IAS_RANK=64
IAS_EPS_TAG="0p0001"
IAS_CAND=32
IAS_ALIGNMENT_MODE="dp_local"

IAS_CALIB_SKIP_LINES=3000
IAS_CALIB_NUM_SAMPLES=1000

IAS_STATS_KEY="rank${IAS_RANK}_eps${IAS_EPS_TAG}_calibskip${IAS_CALIB_SKIP_LINES}_calibn${IAS_CALIB_NUM_SAMPLES}"

export IAS_STATS_KEY="${IAS_STATS_KEY}"
export IASW_STATS_KEY="${IAS_STATS_KEY}"

export IAS_ALIGNMENT_MODE="${IAS_ALIGNMENT_MODE}"
export IASW_ALIGNMENT_MODE="${IAS_ALIGNMENT_MODE}"

export IAS_TARGET_LEN="${TARGET_LEN}"



# =================================================================
# 2.3 Preflight
# =================================================================

echo
echo "==================================================="
echo "GLOBAL CONFIG"
echo "==================================================="
echo "PROJECT_DIR   = ${PROJECT_DIR}"
echo "LLM_PATH      = ${LLM_PATH}"
echo "MODEL_NAME    = ${MODEL_NAME}"
echo "TARGET_LEN    = ${TARGET_LEN}"
echo "NUM_SAMPLES   = ${NUM_SAMPLES}"
echo
echo "==================================================="


if [[ ! -d "${LLM_PATH}" ]]; then
    echo "❌ Missing LLM: ${LLM_PATH}"
    exit 1
fi

if [[ ! -d "${EMBEDDER_PATH}" ]]; then
    echo "❌ Missing embedder: ${EMBEDDER_PATH}"
    exit 1
fi


# =================================================================
# 2.4 当前需要运行的方法
# =================================================================

ALL_METHODS=(
    "single:kgw"
    "single:sweet"
    "single:ias"
    "single:iasw"
    "single:dipmark"
    "single:sta"
    "single:semstamp"
    "single:ksemstamp"
    "single:simmark_cosine"
    "single:simmark_euclidean"
    "single:synguard"
    "single:dualmark"

    "dual:kgw:iasw"
    "dual:sweet:iasw"
    "dual:dipmark:iasw"
)


# =================================================================
# 2.5 Attack suite
# =================================================================

ATTACKS=(
    "pegasus"
    "dipper"
    "back_translation"
    "copy_paste"
    "del_ins"
    "drop"
    "swap"
    "typo"
    "sentence_reorder"
)




# =================================================================
# 3. 核心大循环
# =================================================================

for DS_TYPE in "${DATASETS[@]}"; do

    echo
    echo "==================================================="
    echo "📦 数据集: ${DS_TYPE^^}"
    echo "==================================================="

    RAW_INPUT_FILE="${PROJECT_DIR}/data/raw_${DS_TYPE}.jsonl"

    if [[ ! -f "${RAW_INPUT_FILE}" ]]; then
        echo "❌ 找不到原始数据文件:"
        echo "   ${RAW_INPUT_FILE}"
        echo "跳过 ${DS_TYPE}。"
        continue
    fi

    # =========================================================
    # Test / Formal directory
    # =========================================================

    if [[ ${IS_TEST} -eq 1 ]]; then
        BASELINE_DIR="${PROJECT_DIR}/data/tests/baseline_triplets/${DS_TYPE}/${MODEL_NAME}"
        RESULT_ROOT="${PROJECT_DIR}/results/tests/${DS_TYPE}/${MODEL_NAME}"
    else
        BASELINE_DIR="${PROJECT_DIR}/data/main_exp/baseline_triplets/${DS_TYPE}/${MODEL_NAME}"
        RESULT_ROOT="${PROJECT_DIR}/results/main_exp/${DS_TYPE}/${MODEL_NAME}"
    fi

    RECORDS_DIR="${RESULT_ROOT}/records"
    PLOTS_DIR="${RESULT_ROOT}/plots"
    SUMMARY_DIR="${RESULT_ROOT}/summaries"

    mkdir -p \
        "${BASELINE_DIR}" \
        "${RECORDS_DIR}" \
        "${PLOTS_DIR}" \
        "${SUMMARY_DIR}"

    BASELINE_FILE="${BASELINE_DIR}/baseline_len${TARGET_LEN}.jsonl"

    # =========================================================
    # Step 0
    # Baseline triplets
    # =========================================================

    echo
    echo "--------------------------------------------------"
    echo "[STEP 0] Baseline triplets"
    echo "--------------------------------------------------"


    if [[ ! -f "${BASELINE_FILE}" ]]; then

        echo "➡️ 生成 ${DS_TYPE^^} baseline"
        echo "目标样本数: ${NUM_SAMPLES}"

        python src/utils/build_baseline_dataset.py \
            --input_file "${RAW_INPUT_FILE}" \
            --output_file "${BASELINE_FILE}" \
            --dataset_type "${DS_TYPE}" \
            --model_path "${LLM_PATH}" \
            --target_len "${TARGET_LEN}" \
            --num_samples "${NUM_SAMPLES}" \
            --batch_size 16
    else
        echo "⚡ Baseline 已存在:"
        echo "   ${BASELINE_FILE}"
        echo "样本数:"
        wc -l "${BASELINE_FILE}"

    fi

    # =========================================================
    # Iterate methods
    # =========================================================

    for METHOD_ENTRY in "${ALL_METHODS[@]}"; do

        TYPE="$(echo "${METHOD_ENTRY}" | cut -d':' -f1)"

        # =====================================================
        # Parse method
        # =====================================================

        EXTRA_ARGS=()

        if [[ "${TYPE}" == "single" ]]; then

            METHOD_NAME="$(echo "${METHOD_ENTRY}" | cut -d':' -f2)"
            FILE_PREFIX="${METHOD_NAME}"

            # =================================================


        elif [[ "${TYPE}" == "dual" ]]; then

            MICRO="$(echo "${METHOD_ENTRY}" | cut -d':' -f2)"
            MACRO="$(echo "${METHOD_ENTRY}" | cut -d':' -f3)"

            METHOD_NAME="universal_dual"
            FILE_PREFIX="dual_${MICRO}_${MACRO}"

            EXTRA_ARGS=(
                --micro_method
                "${MICRO}"
                --macro_method
                "${MACRO}"
            )


        else
            echo "❌ Unknown method entry:"
            echo "   ${METHOD_ENTRY}"
            exit 1
        fi


        TARGET_JSONL="${RECORDS_DIR}/${FILE_PREFIX}.jsonl"


        echo
        echo "==================================================="
        echo "🚀 WATERMARK"
        echo "==================================================="
        echo "Dataset     : ${DS_TYPE}"
        echo "Entry       : ${METHOD_ENTRY}"
        echo "Method      : ${METHOD_NAME}"
        echo "File prefix : ${FILE_PREFIX}"
        echo "Output      : ${TARGET_JSONL}"
        echo "==================================================="




        # =========================================================
        # Step 1
        # Generation
        # =========================================================

        echo
        echo "--------------------------------------------------"
        echo "[STEP 1] Watermarked generation"
        echo "--------------------------------------------------"


        python -m src.watermarks.runner \
            --task generate \
            --baseline_file "${BASELINE_FILE}" \
            --target_file "${TARGET_JSONL}" \
            --dataset_type "${DS_TYPE}" \
            --method "${METHOD_NAME}" \
            "${EXTRA_ARGS[@]}" \
            --model_path "${LLM_PATH}" \
            --embedder_path "${EMBEDDER_PATH}" \
            --max_total_tokens "${TARGET_LEN}" \
            --max_sent_tokens 40 \
            --num_candidates "${IAS_CAND}" \
            --num_samples "${NUM_SAMPLES}"




        # =========================================================
        # Step 2
        # Baseline detection
        # =========================================================

        echo
        echo "--------------------------------------------------"
        echo "[STEP 2] Baseline detection"
        echo "--------------------------------------------------"


        python -m src.watermarks.runner \
            --task detect \
            --target_file "${TARGET_JSONL}" \
            --dataset_type "${DS_TYPE}" \
            --method "${METHOD_NAME}" \
            "${EXTRA_ARGS[@]}" \
            --model_path "${LLM_PATH}" \
            --embedder_path "${EMBEDDER_PATH}" \
            --mode baseline_eval




        # =========================================================
        # Step 3
        # Attacks
        # =========================================================

        echo
        echo "--------------------------------------------------"
        echo "[STEP 3] Attack robustness"
        echo "--------------------------------------------------"


        for ATTACK in "${ATTACKS[@]}"; do

            echo
            echo "==================================================="
            echo "🔥 ATTACK: ${ATTACK}"
            echo "==================================================="


            # -----------------------------------------------------
            # Generate attacked text
            # -----------------------------------------------------

            python src/attacks/run_attack.py \
                --target_file "${TARGET_JSONL}" \
                --attack_name "${ATTACK}" \
                --dipper_path "${DIPPER_PATH}" \
                --paraphrase_path "${PARAPHRASE_PATH}" \
                --fwd_trans_path "${FWD_TRANS_PATH}" \
                --bwd_trans_path "${BWD_TRANS_PATH}" \


            # -----------------------------------------------------
            # Detect attacked text
            # -----------------------------------------------------

            python -m src.watermarks.runner \
                --task detect \
                --target_file "${TARGET_JSONL}" \
                --dataset_type "${DS_TYPE}" \
                --method "${METHOD_NAME}" \
                "${EXTRA_ARGS[@]}" \
                --model_path "${LLM_PATH}" \
                --embedder_path "${EMBEDDER_PATH}" \
                --mode attack_eval \
                --attack_name "${ATTACK}"



        done


        # =========================================================
        # Step 4
        # Quality
        # =========================================================

        echo
        echo "--------------------------------------------------"
        echo "[STEP 4] Generation quality"
        echo "--------------------------------------------------"


        # python src/evaluation/evaluate_quality.py \
        #     --input_file "${TARGET_JSONL}" \
        #     --model_path "${LLM_PATH}" \
        #     --embedder_path "${EMBEDDER_PATH}"


        # =========================================================
        # Step 5
        # Figures / ROC
        # =========================================================

        echo
        echo "--------------------------------------------------"
        echo "[STEP 5] Evaluation plots"
        echo "--------------------------------------------------"


        # ---------------------------------------------------------
        # Standardized Z-score distribution
        # ---------------------------------------------------------

        echo "➡️ 生成 Z-Score distribution"

        python src/evaluation/plot_z_distribution.py \
            --input_file "${TARGET_JSONL}" \
            --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_z_dist.pdf" \
            --method "${FILE_PREFIX}"


        # ---------------------------------------------------------
        # ROC
        # ---------------------------------------------------------

        echo "➡️ 生成 ROC / fixed-FPR metrics"


        python src/evaluation/plot_roc.py \
            --input_file "${TARGET_JSONL}" \
            --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_roc_combined.pdf" \
            --method "${FILE_PREFIX}"


        # =========================================================
        # Step 5.1
        # IAS-W ablation figures
        # =========================================================

        if [[ "${TYPE}" == "single" && "${FILE_PREFIX}" == "iasw" ]]; then

            echo
            echo "--------------------------------------------------"
            echo "[STEP 5.1] IAS-W ablation plots"
            echo "--------------------------------------------------"


            python src/evaluation/plot_ablation_z_dist.py \
                --input_file "${TARGET_JSONL}" \
                --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_ablation_z_dist.pdf" \
                --method "${FILE_PREFIX}"


            python src/evaluation/plot_ablation_roc.py \
                --input_file "${TARGET_JSONL}" \
                --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_ablation_roc.pdf" \
                --method "${FILE_PREFIX}"

        fi


        fi


    done


    # =========================================================
    # Step 6
    # Dataset-level summary
    # =========================================================

    echo
    echo "==================================================="
    echo "📊 Dataset summary: ${DS_TYPE^^}"
    echo "==================================================="


    python src/evaluation/auto_summarize.py \
        --exp_data_dir "${RECORDS_DIR}" \
        --plot_dir "${PLOTS_DIR}" \
        --exp_res_dir "${SUMMARY_DIR}"


done


echo
echo "==================================================="
echo "🎉 所有实验阶段调度完毕"
echo "==================================================="
