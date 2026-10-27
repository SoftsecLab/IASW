#!/bin/bash
set -e

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
export PROJECT_DIR PROJECT_ROOT
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"


# =========================================================
# 基础环境
# =========================================================
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NLTK_DATA="${NLTK_DATA:-./models/nltk_data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_DIR="${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# =========================================================
# TEST 模式 / 样本数量
# =========================================================
IS_TEST=0
NUM_SAMPLES="${NUM_SAMPLES:-500}"

if [ "${1:-}" == "--test" ]; then
    IS_TEST=1
    NUM_SAMPLES="${TEST_NUM_SAMPLES:-5}"
    echo "🟢 启用 TEST 模式: NUM_SAMPLES=${NUM_SAMPLES}"
fi

# =========================================================
# 模型与数据路径
# =========================================================
LLM_PATH="${LLM_PATH:-./models/llama3.1-8b-instruct}"
MODEL_NAME=$(basename "${LLM_PATH}")
EMBEDDER_PATH="${EMBEDDER_PATH:-./models/bge-small-en-v1.5}"

DS_TYPE="${DS_TYPE:-c4}"
RAW_INPUT_FILE="${PROJECT_DIR}/data/raw_${DS_TYPE}.jsonl"

if [ ! -f "${RAW_INPUT_FILE}" ]; then
    echo "❌ 找不到原始数据文件: ${RAW_INPUT_FILE}"
    exit 1
fi

# =========================================================
# IAS-W 配置
# =========================================================
IAS_RANK="${IAS_RANK:-64}"
IAS_EPS="${IAS_EPS:-0.0001}"
IAS_EPS_TAG="${IAS_EPS_TAG:-0p0001}"
IAS_CAND="${IAS_CAND:-32}"
IAS_ALIGNMENT_MODE="${IAS_ALIGNMENT_MODE:-dp_local}"

IAS_CALIB_SKIP_LINES="${IAS_CALIB_SKIP_LINES:-3000}"
IAS_CALIB_NUM_SAMPLES="${IAS_CALIB_NUM_SAMPLES:-1000}"

IAS_STATS_KEY="rank${IAS_RANK}_eps${IAS_EPS_TAG}_calibskip${IAS_CALIB_SKIP_LINES}_calibn${IAS_CALIB_NUM_SAMPLES}"

export IAS_STATS_KEY="${IAS_STATS_KEY}"
export IASW_STATS_KEY="${IAS_STATS_KEY}"
export IAS_ALIGNMENT_MODE="${IAS_ALIGNMENT_MODE}"
export IASW_ALIGNMENT_MODE="${IAS_ALIGNMENT_MODE}"

SECRET_KEY="${SECRET_KEY:-DUAL_MACRO_2026}"
DEVICE="${DEVICE:-cuda:0}"

# 是否在 stats 缺失时自动构建 stats
# 默认不自动构建，避免误启动很长的 calibration。
AUTO_BUILD_STATS="${AUTO_BUILD_STATS:-0}"

# 是否强制重新生成水印文本
# 长度消融建议默认 1，避免旧文件污染结果。
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# 长度列表
TARGET_LEN_LIST_STR="${TARGET_LEN_LIST_STR:-500}"
read -r -a TARGET_LEN_LIST <<< "${TARGET_LEN_LIST_STR}"

# 样本构建 batch size
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-2}"

# =========================================================
# 日志目录
# =========================================================
if [ ${IS_TEST} -eq 0 ]; then
    LOG_DIR="${PROJECT_DIR}/results/logs/ablations/token_length"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/run_fixed_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "📝 日志输出至: ${LOG_FILE}"
fi

# =========================================================
# 方法和攻击列表
# =========================================================
ALL_METHODS=(
    "dual:kgw:iasw"
)
ATTACKS_STR="${ATTACKS_STR:-  pegasus dipper back_translation copy_paste del_ins drop swap typo sentence_reorder}"
read -r -a ATTACKS <<< "${ATTACKS_STR}"

# =========================================================
# 函数：必要时构建 IAS-W stats
# =========================================================
build_ias_stats_if_needed() {
    local TARGET_LEN="$1"
    local IAS_STATS_FILE="$2"

    if [ -f "${IAS_STATS_FILE}" ]; then
        echo "✅ IAS-W stats 已存在: ${IAS_STATS_FILE}"
        return 0
    fi

    if [ "${AUTO_BUILD_STATS}" != "1" ]; then
        echo "❌ 缺少当前长度对应的 IAS-W stats:"
        echo "   ${IAS_STATS_FILE}"
        echo ""
        echo "请先生成该长度的 stats，或者使用："
        echo "   AUTO_BUILD_STATS=1 TARGET_LEN_LIST_STR=\"${TARGET_LEN}\" bash scripts/ablations/3_ablation_token_len_fixed.sh"
        exit 1
    fi

    echo "⚠️ 当前长度 stats 不存在，开始自动构建: len=${TARGET_LEN}"

    CALIB_DIR="${PROJECT_DIR}/data/calibration_set/${DS_TYPE}/${MODEL_NAME}"
    mkdir -p "${CALIB_DIR}"

    CALIB_RAW_FILE="${CALIB_DIR}/temp_raw_offset_skip${IAS_CALIB_SKIP_LINES}.jsonl"
    CALIB_BASELINE_FILE="${CALIB_DIR}/ias_baseline_len${TARGET_LEN}_skip${IAS_CALIB_SKIP_LINES}_n${IAS_CALIB_NUM_SAMPLES}.jsonl"

    if [ ! -f "${CALIB_RAW_FILE}" ]; then
        echo "➡️ [Calibration raw] 构建 skip${IAS_CALIB_SKIP_LINES} 原始切分: ${CALIB_RAW_FILE}"
        tail -n +$((IAS_CALIB_SKIP_LINES + 1)) "${RAW_INPUT_FILE}" > "${CALIB_RAW_FILE}"
    else
        echo "⚡ Calibration raw 已存在: ${CALIB_RAW_FILE}"
    fi

    NEED_CALIB_BASELINE=0
    if [ ! -f "${CALIB_BASELINE_FILE}" ]; then
        NEED_CALIB_BASELINE=1
    else
        CALIB_BASELINE_N=$(wc -l < "${CALIB_BASELINE_FILE}" || echo 0)
        if [ "${CALIB_BASELINE_N}" -lt "${IAS_CALIB_NUM_SAMPLES}" ]; then
            echo "⚠️ Calibration baseline 数量不足: ${CALIB_BASELINE_N}/${IAS_CALIB_NUM_SAMPLES}，准备重建"
            NEED_CALIB_BASELINE=1
        fi
    fi

    if [ "${NEED_CALIB_BASELINE}" -eq 1 ]; then
        echo "➡️ [Calibration baseline] 构建 IAS-W 标定三元组: ${CALIB_BASELINE_FILE}"
        rm -f "${CALIB_BASELINE_FILE}"

        python src/utils/build_baseline_dataset.py \
            --input_file "${CALIB_RAW_FILE}" \
            --output_file "${CALIB_BASELINE_FILE}" \
            --dataset_type "${DS_TYPE}" \
            --model_path "${LLM_PATH}" \
            --target_len "${TARGET_LEN}" \
            --num_samples "${IAS_CALIB_NUM_SAMPLES}" \
            --batch_size "${BASELINE_BATCH_SIZE}"
    else
        echo "⚡ IAS-W calibration baseline 已存在: ${CALIB_BASELINE_FILE}"
    fi

    CALIB_BASELINE_N=$(wc -l < "${CALIB_BASELINE_FILE}" || echo 0)
    if [ "${CALIB_BASELINE_N}" -lt "${IAS_CALIB_NUM_SAMPLES}" ]; then
        echo "❌ Calibration baseline 数量不足: ${CALIB_BASELINE_N}/${IAS_CALIB_NUM_SAMPLES}"
        exit 1
    fi

    mkdir -p "$(dirname "${IAS_STATS_FILE}")"

    echo "➡️ [IAS-W Stats] 构建 stats: ${IAS_STATS_FILE}"

    python src/calibration/calibrate_empirical_iasw.py \
        --baseline_file "${CALIB_BASELINE_FILE}" \
        --stats_file "${IAS_STATS_FILE}" \
        --dataset_type "${DS_TYPE}" \
        --model_path "${LLM_PATH}" \
        --embedder_path "${EMBEDDER_PATH}" \
        --seq_len "${TARGET_LEN}" \
        --rank "${IAS_RANK}" \
        --eps "${IAS_EPS}" \
        --num_samples "${IAS_CALIB_NUM_SAMPLES}" \
        --batch_size 64 \
        --stats_key "${IAS_STATS_KEY}" \
        --secret_key "${SECRET_KEY}" \
        --device "${DEVICE}"

    if [ ! -f "${IAS_STATS_FILE}" ]; then
        echo "❌ stats 构建失败: ${IAS_STATS_FILE}"
        exit 1
    fi
}

# =========================================================
# 总配置打印
# =========================================================
echo "==================================================="
echo "Token Length Ablation Config"
echo "PROJECT_DIR        = ${PROJECT_DIR}"
echo "DS_TYPE            = ${DS_TYPE}"
echo "LLM_PATH           = ${LLM_PATH}"
echo "MODEL_NAME         = ${MODEL_NAME}"
echo "EMBEDDER_PATH      = ${EMBEDDER_PATH}"
echo "NUM_SAMPLES        = ${NUM_SAMPLES}"
echo "TARGET_LEN_LIST    = ${TARGET_LEN_LIST_STR}"
echo "IAS_STATS_KEY      = ${IAS_STATS_KEY}"
echo "IAS_ALIGNMENT_MODE = ${IAS_ALIGNMENT_MODE}"
echo "ATTACKS            = ${ATTACKS_STR}"
echo "FORCE_REGENERATE   = ${FORCE_REGENERATE}"
echo "AUTO_BUILD_STATS   = ${AUTO_BUILD_STATS}"
echo "==================================================="

# =========================================================
# 循环 Token 长度
# =========================================================
for TARGET_LEN in "${TARGET_LEN_LIST[@]}"; do
    echo -e "\n==================================================="
    echo "📏 当前 Token 长度: ${TARGET_LEN}"
    echo "==================================================="

    # ---------------------------------------------------------
    # 关键修正：每个长度都重新 export
    # ---------------------------------------------------------
    export TARGET_LEN="${TARGET_LEN}"
    export IAS_TARGET_LEN="${TARGET_LEN}"

    IAS_STATS_DIR="${PROJECT_DIR}/cache/iasw_stats/${DS_TYPE}/${MODEL_NAME}"
    IAS_STATS_FILE="${IAS_STATS_DIR}/ias_stats_${DS_TYPE}_${MODEL_NAME}_len${TARGET_LEN}_${IAS_STATS_KEY}.pt"

    export IAS_STATS_FILE="${IAS_STATS_FILE}"
    export IASW_STATS_FILE="${IAS_STATS_FILE}"

    build_ias_stats_if_needed "${TARGET_LEN}" "${IAS_STATS_FILE}"

    echo "🔧 当前 len${TARGET_LEN} 使用 IAS-W stats:"
    echo "   IAS_STATS_FILE=${IAS_STATS_FILE}"

    PARAM_DIR="${PROJECT_DIR}/results/ablations/token_length/len${TARGET_LEN}"
    RECORDS_DIR="${PARAM_DIR}/records"
    PLOTS_DIR="${PARAM_DIR}/plots"
    SUMMARY_DIR="${PARAM_DIR}/summaries"
    mkdir -p "${RECORDS_DIR}" "${PLOTS_DIR}" "${SUMMARY_DIR}"

    # ---------------------------------------------------------
    # Baseline 文件
    # ---------------------------------------------------------
    BASELINE_FILE="${PROJECT_DIR}/data/main_exp/baseline_triplets/${DS_TYPE}/${MODEL_NAME}/baseline_len${TARGET_LEN}.jsonl"

    if [ ! -f "${BASELINE_FILE}" ]; then
        echo "➡️ [Baseline] 构建评测三元组: ${BASELINE_FILE}"
        mkdir -p "$(dirname "${BASELINE_FILE}")"

        python src/utils/build_baseline_dataset.py \
            --input_file "${RAW_INPUT_FILE}" \
            --output_file "${BASELINE_FILE}" \
            --dataset_type "${DS_TYPE}" \
            --model_path "${LLM_PATH}" \
            --target_len "${TARGET_LEN}" \
            --num_samples "${NUM_SAMPLES}" \
            --batch_size "${BASELINE_BATCH_SIZE}"
    else
        echo "⚡ Baseline 已存在: ${BASELINE_FILE}"
    fi

    BASELINE_N=$(wc -l < "${BASELINE_FILE}" || echo 0)
    if [ "${BASELINE_N}" -lt "${NUM_SAMPLES}" ]; then
        echo "❌ Baseline 数量不足: ${BASELINE_N}/${NUM_SAMPLES}"
        exit 1
    fi

    # =========================================================
    # 循环方法
    # =========================================================
    for METHOD_ENTRY in "${ALL_METHODS[@]}"; do
        TYPE=$(echo "${METHOD_ENTRY}" | cut -d':' -f1)

        if [ "${TYPE}" == "single" ]; then
            METHOD_NAME=$(echo "${METHOD_ENTRY}" | cut -d':' -f2)
            FILE_PREFIX="${METHOD_NAME}"
            EXTRA_ARGS=""
        else
            MICRO=$(echo "${METHOD_ENTRY}" | cut -d':' -f2)
            MACRO=$(echo "${METHOD_ENTRY}" | cut -d':' -f3)
            METHOD_NAME="universal_dual"
            FILE_PREFIX="dual_${MICRO}_${MACRO}"
            EXTRA_ARGS="--micro_method ${MICRO} --macro_method ${MACRO}"
        fi

        TARGET_JSONL="${RECORDS_DIR}/${FILE_PREFIX}.jsonl"
        TARGET_DQ_JSON="${RECORDS_DIR}/detailed_quality_metrics_${FILE_PREFIX}.json"

        echo -e "\n🟢 方法: ${FILE_PREFIX} | token长度: ${TARGET_LEN}"
        echo "TARGET_JSONL=${TARGET_JSONL}"

        # -----------------------------------------------------
        # len200 可以复用主实验结果；如果主实验结果不存在，则正常执行
        # -----------------------------------------------------
        if [ "${TARGET_LEN}" -eq 200 ] && [ "${FORCE_REGENERATE}" != "1" ]; then
            MAIN_RECORD="${PROJECT_DIR}/results/main_exp/${DS_TYPE}/${MODEL_NAME}/records/${FILE_PREFIX}.jsonl"
            MAIN_DQ="${PROJECT_DIR}/results/main_exp/${DS_TYPE}/${MODEL_NAME}/records/detailed_quality_metrics_${FILE_PREFIX}.json"

            if [ -f "${MAIN_RECORD}" ]; then
                echo "⚡ len200 复用主实验结果: ${MAIN_RECORD}"
                ln -sf "${MAIN_RECORD}" "${TARGET_JSONL}"

                if [ -f "${MAIN_DQ}" ]; then
                    ln -sf "${MAIN_DQ}" "${TARGET_DQ_JSON}"
                fi

                if [ -f "${PROJECT_DIR}/results/main_exp/${DS_TYPE}/${MODEL_NAME}/plots/${FILE_PREFIX}_z_dist.pdf" ]; then
                    ln -sf "${PROJECT_DIR}/results/main_exp/${DS_TYPE}/${MODEL_NAME}/plots/${FILE_PREFIX}_z_dist.pdf" \
                        "${PLOTS_DIR}/${FILE_PREFIX}_z_dist.pdf"
                fi

                if [ -f "${PROJECT_DIR}/results/main_exp/${DS_TYPE}/${MODEL_NAME}/plots/${FILE_PREFIX}_roc_combined.pdf" ]; then
                    ln -sf "${PROJECT_DIR}/results/main_exp/${DS_TYPE}/${MODEL_NAME}/plots/${FILE_PREFIX}_roc_combined.pdf" \
                        "${PLOTS_DIR}/${FILE_PREFIX}_roc_combined.pdf"
                fi

                continue
            else
                echo "⚠️ len200 主实验结果不存在，改为正常执行。"
            fi
        fi

        # -----------------------------------------------------
        # Step 1: Generate
        # -----------------------------------------------------
        if [ "${FORCE_REGENERATE}" == "1" ]; then
            echo "🧹 FORCE_REGENERATE=1，删除旧结果: ${TARGET_JSONL}"
            rm -f "${TARGET_JSONL}"
        fi

        if [ ! -f "${TARGET_JSONL}" ]; then
            echo "➡️ [生成阶段] 生成 len${TARGET_LEN} 水印文本..."

            python src/watermarks/runner.py \
                --task generate \
                --baseline_file "${BASELINE_FILE}" \
                --target_file "${TARGET_JSONL}" \
                --dataset_type "${DS_TYPE}" \
                --method "${METHOD_NAME}" ${EXTRA_ARGS} \
                --model_path "${LLM_PATH}" \
                --embedder_path "${EMBEDDER_PATH}" \
                --max_total_tokens "${TARGET_LEN}" \
                --max_sent_tokens 40 \
                --num_candidates "${IAS_CAND}" \
                --num_samples "${NUM_SAMPLES}" \
                --secret_key "${SECRET_KEY}"
        else
            echo "⚡ TARGET_JSONL 已存在，跳过生成: ${TARGET_JSONL}"
        fi

        if [ ! -f "${TARGET_JSONL}" ]; then
            echo "❌ 生成失败，找不到 TARGET_JSONL: ${TARGET_JSONL}"
            exit 1
        fi

        # -----------------------------------------------------
        # Step 2: Baseline detection
        # -----------------------------------------------------
        echo "➡️ [检测基线] 使用 len${TARGET_LEN} stats 计算基础 Z-Score..."

        python src/watermarks/runner.py \
            --task detect \
            --target_file "${TARGET_JSONL}" \
            --dataset_type "${DS_TYPE}" \
            --method "${METHOD_NAME}" ${EXTRA_ARGS} \
            --model_path "${LLM_PATH}" \
            --embedder_path "${EMBEDDER_PATH}" \
            --mode baseline_eval \
            --secret_key "${SECRET_KEY}"

        # -----------------------------------------------------
        # Step 3: Attacks + detection
        # -----------------------------------------------------
        for ATTACK in "${ATTACKS[@]}"; do
            echo "🔥 执行并检测攻击: ${ATTACK}"

            python src/attacks/run_attack.py \
                --target_file "${TARGET_JSONL}" \
                --attack_name "${ATTACK}" \
                --dipper_path "./models/dipper-paraphraser-xxl" \
                --paraphrase_path "./models/tuner007_pegasus_paraphrase" \
                --fwd_trans_path "./models/opus-mt-en-fr" \
                --bwd_trans_path "./models/opus-mt-fr-en" \

            python src/watermarks/runner.py \
                --task detect \
                --target_file "${TARGET_JSONL}" \
                --dataset_type "${DS_TYPE}" \
                --method "${METHOD_NAME}" ${EXTRA_ARGS} \
                --model_path "${LLM_PATH}" \
                --embedder_path "${EMBEDDER_PATH}" \
                --mode attack_eval \
                --attack_name "${ATTACK}" \
                --secret_key "${SECRET_KEY}"
        done

        # -----------------------------------------------------
        # Step 4: Quality
        # -----------------------------------------------------
        echo "➡️ [质量评估] 计算 PPL / semantic similarity..."

        python src/evaluation/evaluate_quality.py \
            --input_file "${TARGET_JSONL}" \
            --model_path "${LLM_PATH}" \
            --embedder_path "${EMBEDDER_PATH}"

        # -----------------------------------------------------
        # Step 5: Plots
        # -----------------------------------------------------
        echo "➡️ [绘图] 生成 Z-score 分布图..."

        python src/evaluation/plot_z_distribution.py \
            --input_file "${TARGET_JSONL}" \
            --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_z_dist.pdf" \
            --method "${FILE_PREFIX}"

        echo "➡️ [绘图] 生成 ROC 曲线..."

        python src/evaluation/plot_roc.py \
            --input_file "${TARGET_JSONL}" \
            --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_roc_combined.pdf" \
            --method "${FILE_PREFIX}"

        # -----------------------------------------------------
        # Step 5.1: IAS-W ablation plots
        # 双通道 IAS-W 也有 ablation_z_scores，尝试生成；失败不影响主流程。
        # -----------------------------------------------------
        echo "➡️ [IAS-W 消融绘图] 尝试生成 4 模式 Z/ROC 图..."

        python src/evaluation/plot_ablation_z_dist.py \
            --input_file "${TARGET_JSONL}" \
            --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_ablation_z_dist.pdf" \
            --method "${FILE_PREFIX}" || true

        python src/evaluation/plot_ablation_roc.py \
            --input_file "${TARGET_JSONL}" \
            --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_ablation_roc.pdf" \
            --method "${FILE_PREFIX}" || true

    done

    # ---------------------------------------------------------
    # Step 6: Summary
    # ---------------------------------------------------------
    echo -e "\n==================================================="
    echo "📊 正在为 ${DS_TYPE^^} len${TARGET_LEN} 生成综合学术评估报表..."

    python src/evaluation/auto_summarize.py \
        --exp_data_dir "${RECORDS_DIR}" \
        --plot_dir "${PLOTS_DIR}" \
        --exp_res_dir "${SUMMARY_DIR}"

    echo "==================================================="
done

echo "🎉 Token 长度消融实验完成，结果保存在: ${PROJECT_DIR}/results/ablations/token_length"
