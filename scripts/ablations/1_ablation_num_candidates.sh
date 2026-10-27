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
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export NLTK_DATA=./models/nltk_data
PROJECT_DIR="${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# =========================================================
# IAS-W default runtime config
# 默认主实验使用 rank64
# =========================================================
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


# =========================================================
# TEST 模式 / 样本数量
# =========================================================
IS_TEST=0
NUM_SAMPLES=500
NUM_CANDIDATES_LIST=(
    8
    16
    32
    64
)

if [ "$1" == "--test" ]; then
    IS_TEST=1
    NUM_SAMPLES=5
    echo "🟢 启用 TEST 模式: NUM_SAMPLES=${NUM_SAMPLES}"
fi

# =========================================================
# 模型与数据路径
# =========================================================
LLM_PATH="./models/llama3.1-8b-instruct"
MODEL_NAME=$(basename "${LLM_PATH}")
EMBEDDER_PATH="./models/bge-small-en-v1.5"

DS_TYPE="c4"
RAW_INPUT_FILE="${PROJECT_DIR}/data/raw_${DS_TYPE}.jsonl"
if [ ! -f "${RAW_INPUT_FILE}" ]; then
    echo "❌ 找不到原始数据文件: ${RAW_INPUT_FILE}！"
    exit 1
fi

# =========================================================
# 日志目录
# =========================================================
if [ ${IS_TEST} -eq 0 ]; then
    LOG_DIR="${PROJECT_DIR}/results/logs/ablations/num_candidates"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "📝 日志输出至: ${LOG_FILE}"
fi

# =========================================================
# 方法和攻击列表
# =========================================================
ALL_METHODS=(
    "single:iasw"
    "dual:kgw:iasw"
)
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

# =========================================================
# 循环候选数
# =========================================================
for NUM_CANDIDATES in "${NUM_CANDIDATES_LIST[@]}"; do
    PARAM_DIR="${PROJECT_DIR}/results/ablations/num_candidates/cand${NUM_CANDIDATES}"
    RECORDS_DIR="${PARAM_DIR}/records"
    PLOTS_DIR="${PARAM_DIR}/plots"
    SUMMARY_DIR="${PARAM_DIR}/summaries"
    mkdir -p "${RECORDS_DIR}" "${PLOTS_DIR}" "${SUMMARY_DIR}"

    # =========================================================
    # baseline 文件（固定200 token）
    # =========================================================
    BASELINE_FILE="${PROJECT_DIR}/data/main_exp/baseline_triplets/${DS_TYPE}/${MODEL_NAME}/baseline_len200.jsonl"
    if [ ! -f "${BASELINE_FILE}" ]; then
        mkdir -p "$(dirname ${BASELINE_FILE})"
        python src/utils/build_baseline_dataset.py \
            --input_file "${RAW_INPUT_FILE}" \
            --output_file "${BASELINE_FILE}" \
            --dataset_type "${DS_TYPE}" \
            --model_path "${LLM_PATH}" \
            --target_len 200 \
            --num_samples ${NUM_SAMPLES} \
            --batch_size 16
    fi

    # =========================================================
    # 循环方法
    # =========================================================
    for METHOD_ENTRY in "${ALL_METHODS[@]}"; do
        TYPE=$(echo ${METHOD_ENTRY} | cut -d':' -f1)
        if [ "$TYPE" == "single" ]; then
            METHOD_NAME=$(echo ${METHOD_ENTRY} | cut -d':' -f2)
            FILE_PREFIX="${METHOD_NAME}"
            EXTRA_ARGS=""
        else
            MICRO=$(echo ${METHOD_ENTRY} | cut -d':' -f2)
            MACRO=$(echo ${METHOD_ENTRY} | cut -d':' -f3)
            METHOD_NAME="universal_dual"
            FILE_PREFIX="dual_${MICRO}_${MACRO}"
            EXTRA_ARGS="--micro_method ${MICRO} --macro_method ${MACRO}"
        fi

        TARGET_JSONL="${RECORDS_DIR}/${FILE_PREFIX}.jsonl"
        TARGET_DQ_JSON="${RECORDS_DIR}/detailed_quality_metrics_${FILE_PREFIX}.json"

        # ===============================
        # cand32 已存在 -> 软连接复用
        # ===============================
        if [ ${NUM_CANDIDATES} -eq 32 ]; then
            ln -sf "${BASELINE_FILE}" "${TARGET_JSONL}" 2>/dev/null || true
            ln -sf "${PROJECT_DIR}/results/main_exp/c4/${MODEL_NAME}/records/${FILE_PREFIX}.jsonl" "${TARGET_JSONL}" 2>/dev/null || true
            ln -sf "${PROJECT_DIR}/results/main_exp/c4/${MODEL_NAME}/records/detailed_quality_metrics_${FILE_PREFIX}.json" "${TARGET_DQ_JSON}" 2>/dev/null || true

            # plots 文件
            ln -sf "${PROJECT_DIR}/results/main_exp/c4/${MODEL_NAME}/plots/${FILE_PREFIX}_z_dist.pdf" "${PLOTS_DIR}/${FILE_PREFIX}_z_dist.pdf" 2>/dev/null || true
            ln -sf "${PROJECT_DIR}/results/main_exp/c4/${MODEL_NAME}/plots/${FILE_PREFIX}_roc_combined.pdf" "${PLOTS_DIR}/${FILE_PREFIX}_roc_combined.pdf" 2>/dev/null || true
            ln -sf "${PROJECT_DIR}/results/main_exp/c4/${MODEL_NAME}/plots/${FILE_PREFIX}_roc_combined_metrics.json" "${PLOTS_DIR}/${FILE_PREFIX}_roc_combined_metrics.json" 2>/dev/null || true

            if [[ "${TYPE}" == "single" && "${FILE_PREFIX}" == "ias" ]]; then
                ln -sf "${PROJECT_DIR}/results/main_exp/c4/${MODEL_NAME}/plots/${FILE_PREFIX}_ablation_z_dist.pdf" "${PLOTS_DIR}/${FILE_PREFIX}_ablation_z_dist.pdf" 2>/dev/null || true
                ln -sf "${PROJECT_DIR}/results/main_exp/c4/${MODEL_NAME}/plots/${FILE_PREFIX}_ablation_roc.pdf" "${PLOTS_DIR}/${FILE_PREFIX}_ablation_roc.pdf" 2>/dev/null || true
                ln -sf "${PROJECT_DIR}/results/main_exp/c4/${MODEL_NAME}/plots/${FILE_PREFIX}_ablation_roc_ablation_metrics.json" "${PLOTS_DIR}/${FILE_PREFIX}_ablation_roc_ablation_metrics.json" 2>/dev/null || true
            fi

            echo "⚡ cand32 已存在，软连接完成，跳过执行"
            continue
        fi

        # ===============================
        # 其余候选数完整执行生成/攻击/检测/绘图
        # ===============================
        echo -e "\n🟢 [生成] 方法: ${FILE_PREFIX} | 候选数: ${NUM_CANDIDATES}"
        python src/watermarks/runner.py \
            --task generate \
            --baseline_file "${BASELINE_FILE}" \
            --target_file "${TARGET_JSONL}" \
            --dataset_type "${DS_TYPE}" \
            --method "${METHOD_NAME}" ${EXTRA_ARGS} \
            --model_path "${LLM_PATH}" \
            --embedder_path "${EMBEDDER_PATH}" \
            --max_total_tokens 200 \
            --max_sent_tokens 40 \
            --num_candidates ${NUM_CANDIDATES} \
            --num_samples ${NUM_SAMPLES}

                # ---------------------------------------------------------
        # [步骤 2] 基础检测
        # ---------------------------------------------------------
        echo -e "➡️ [检测基线] 计算基础 Z-Score..."
        python src/watermarks/runner.py \
            --task detect \
            --target_file "${TARGET_JSONL}" \
            --dataset_type "${DS_TYPE}" \
            --method "${METHOD_NAME}" ${EXTRA_ARGS} \
            --model_path "${LLM_PATH}" \
            --embedder_path "${EMBEDDER_PATH}" \
            --mode baseline_eval



        # ===============================
        # 攻击与检测
        # ===============================
        for ATTACK in "${ATTACKS[@]}"; do
            echo -e "🔥 执行攻击: ${ATTACK}"
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
                --attack_name "${ATTACK}"
        done

        # ===============================
        # 文本质量评估
        # ===============================
        python src/evaluation/evaluate_quality.py \
            --input_file "${TARGET_JSONL}" \
            --model_path "${LLM_PATH}" \
            --embedder_path "${EMBEDDER_PATH}"

        # ===============================
        # 绘图
        # ===============================
        python src/evaluation/plot_z_distribution.py \
            --input_file "${TARGET_JSONL}" \
            --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_z_dist.pdf" \
            --method "${FILE_PREFIX}"

        python src/evaluation/plot_roc.py \
            --input_file "${TARGET_JSONL}" \
            --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_roc_combined.pdf" \
            --method "${FILE_PREFIX}"

        # ===============================
        # IAS 消融绘图
        # ===============================
        if [[ "${TYPE}" == "single" && "${FILE_PREFIX}" == "ias" ]]; then
            python src/evaluation/plot_ablation_z_dist.py \
                --input_file "${TARGET_JSONL}" \
                --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_ablation_z_dist.pdf" \
                --method "${FILE_PREFIX}"

            python src/evaluation/plot_ablation_roc.py \
                --input_file "${TARGET_JSONL}" \
                --output_pdf "${PLOTS_DIR}/${FILE_PREFIX}_ablation_roc.pdf" \
                --method "${FILE_PREFIX}"
        fi

    done


    echo -e "\n==================================================="
    echo "📊 正在为 ${DS_TYPE^^} 数据集生成评估报表..."
    python src/evaluation/auto_summarize.py \
        --exp_data_dir "${RECORDS_DIR}" \
        --plot_dir "${PLOTS_DIR}" \
        --exp_res_dir "${SUMMARY_DIR}"
    echo "==================================================="

done

echo "🎉 候选数消融实验完成，结果保存在 ${PROJECT_DIR}/results/ablations/num_candidates"
