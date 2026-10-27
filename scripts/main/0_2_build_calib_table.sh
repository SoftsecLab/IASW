#!/bin/bash
set -euo pipefail

# === PROJECT ROOT BOOTSTRAP ===
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
cd "${PROJECT_DIR}"
export PROJECT_DIR PROJECT_ROOT
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
# === END PROJECT ROOT BOOTSTRAP ===

# =================================================================
# Environment
# =================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NLTK_DATA="${NLTK_DATA:-./models/nltk_data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${PROJECT_DIR}/results/logs/calibration"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/calib_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# =================================================================
# User configuration
# =================================================================
DATA_DIR="${PROJECT_DIR}/data"

DATASETS=(
    "c4"
    # "wiki"
    # "dolly"
)

# Keep only the models you want to process in this run.
LLM_PATHS=(
    "./models/llama3.1-8b-instruct"
    # "./models/Llama-3.2-3B-Instruct"
    # "./models/qwen2.5-7b-instruct"
    # "./models/Mistral-7B-Instruct"
    # "AbeHou/opt-1.3b-semstamp"
)

TOKEN_LENS=(
    # 200
    50
    100
    300
    500
)

NULL_NUM_SAMPLES=5000
IASW_NUM_SAMPLES=1000
SKIP_LINES=3000

IASW_RANK=64
IASW_EPS=0.0001
IASW_EPS_TAG="0p0001"
IASW_MAX_STATS_SENTENCES=20000
IASW_BASELINE_BATCH_SIZE=2
IASW_BATCH_SIZE=64
EMBEDDER_PATH="./models/bge-small-en-v1.5"

UPDATE_IASW_MARK=1
IASW_MARK_FILE="${PROJECT_DIR}/src/watermarks/ias_w_mark.py"

RUN_IASW=1
RUN_NULL_METHODS=0

# -----------------------------------------------------------------
# IMPORTANT: IAS-W build mode
# -----------------------------------------------------------------
# recalibrate (recommended for the current DPLocal update):
#   - REQUIRE existing baseline
#   - REQUIRE existing stats.pt
#   - REUSE mean/components/eigvals exactly
#   - ONLY rebuild scoring_profiles + calibration_profiles
#
# full:
#   - may build baseline
#   - rebuilds whitening + calibration from scratch
IASW_MODE="${IASW_MODE:-recalibrate}"

# Independent force switches. They intentionally do NOT share one global flag.
FORCE_TEMP_RAW_RERUN="${FORCE_TEMP_RAW_RERUN:-0}"
FORCE_BASELINE_RERUN="${FORCE_BASELINE_RERUN:-0}"
FORCE_IASW_STATS_RERUN="${FORCE_IASW_STATS_RERUN:-1}"
FORCE_NULL_RERUN="${FORCE_NULL_RERUN:-0}"

# In recalibrate mode, create <stats>.pre_recalibrate.bak automatically.
BACKUP_EXISTING_STATS="${BACKUP_EXISTING_STATS:-1}"

NULL_METHODS=(
    "simmark_cosine"
    "simmark_euclidean"
    "semstamp"
    "ksemstamp"
)

if [[ "${IASW_MODE}" != "full" && "${IASW_MODE}" != "recalibrate" ]]; then
    echo "ERROR: IASW_MODE must be 'full' or 'recalibrate', got: ${IASW_MODE}" >&2
    exit 1
fi

# =================================================================
# Verification
# =================================================================
verify_iasw_stats() {
    local stats_file="$1"
    local expected_mode="$2"
    python - "${stats_file}" "${expected_mode}" <<'PYVERIFY'
import sys
import torch

path = sys.argv[1]
mode = sys.argv[2]
try:
    stats = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    stats = torch.load(path, map_location="cpu")

for k in ("mean", "components", "eigvals", "calibration"):
    if k not in stats:
        raise RuntimeError(f"missing required stats key: {k}")

profiles = stats.get("calibration_profiles", {})
scoring_profiles = stats.get("scoring_profiles", {})
for name in ("dual", "standalone"):
    if name not in profiles:
        raise RuntimeError(f"missing calibration profile: {name}")
    pack = profiles[name]
    if "scoring" not in pack or "calibration" not in pack:
        raise RuntimeError(f"invalid calibration profile: {name}")
    if name not in scoring_profiles:
        raise RuntimeError(f"missing scoring profile: {name}")
    if pack["scoring"] != scoring_profiles[name]:
        raise RuntimeError(
            f"scoring/calibration mismatch for {name}: "
            f"calibration={pack['scoring']} runtime={scoring_profiles[name]}"
        )
    item = pack["calibration"].get("dp_local", {})
    if int(item.get("n", 0)) <= 0:
        raise RuntimeError(f"invalid dp_local calibration for {name}: {item}")
    print(f"[VERIFY] {name}: scoring={pack['scoring']} dp_local={item}")

if stats.get("calibration") != profiles["dual"]["calibration"]:
    raise RuntimeError("stats['calibration'] must equal dual calibration")

report_mode = (stats.get("build_config", {}) or {}).get("recalibration", {}).get("mode", "")
if mode == "recalibrate" and report_mode != "recalibrate":
    print("[VERIFY][WARN] recalibration metadata not found in build_config; numerical checks still passed.")

print(f"[VERIFY] IAS-W stats OK: {path}")
PYVERIFY
}

# Create temp raw only when it is actually needed.
ensure_temp_raw() {
    local raw_file="$1"
    local temp_file="$2"
    local ds_type="$3"

    if [[ ! -f "${raw_file}" ]]; then
        echo "❌ 找不到原始数据文件: ${raw_file}" >&2
        return 1
    fi

    if [[ "${FORCE_TEMP_RAW_RERUN}" -eq 1 || ! -f "${temp_file}" ]]; then
        echo "✂️  正在跳过前 ${SKIP_LINES} 行，提取 [${ds_type^^}] 基础人类数据源..."
        tail -n +$((SKIP_LINES + 1)) "${raw_file}" > "${temp_file}"
        echo "✅ 已生成: ${temp_file}"
    else
        echo "⚡ 基础数据源已存在，复用: ${temp_file}"
    fi
}

# =================================================================
# Run
# =================================================================
echo "==================================================="
echo "🎯 IAS-W calibration pipeline"
echo "模式    : ${IASW_MODE}"
echo "日志    : ${LOG_FILE}"
echo "项目目录: ${PROJECT_DIR}"
echo "==================================================="

for LLM_PATH in "${LLM_PATHS[@]}"; do
    MODEL_NAME=$(basename "${LLM_PATH}")

    echo -e "\n#####################################################################"
    echo "🤖 模型: ${MODEL_NAME}"
    echo "📍 路径: ${LLM_PATH}"
    echo "#####################################################################"

    for DS_TYPE in "${DATASETS[@]}"; do
        echo -e "\n====================================================================="
        echo "📦 数据集: ${DS_TYPE^^} | 模型: ${MODEL_NAME}"
        echo "====================================================================="

        RAW_LOCAL_FILE="${DATA_DIR}/raw_${DS_TYPE}.jsonl"
        MODEL_CALIB_DIR="${DATA_DIR}/calibration_set/${DS_TYPE}/${MODEL_NAME}"
        mkdir -p "${MODEL_CALIB_DIR}"
        TEMP_RAW_FILE="${MODEL_CALIB_DIR}/temp_raw_offset.jsonl"

        for LEN in "${TOKEN_LENS[@]}"; do
            echo -e "\n  📏 Token length: ${LEN}"

            if [[ "${RUN_IASW}" -eq 1 ]]; then
                IASW_BASELINE_FILE="${MODEL_CALIB_DIR}/ias_baseline_len${LEN}_skip${SKIP_LINES}_n${IASW_NUM_SAMPLES}.jsonl"
                IASW_STATS_DIR="${PROJECT_DIR}/cache/iasw_stats/${DS_TYPE}/${MODEL_NAME}"
                mkdir -p "${IASW_STATS_DIR}"
                IASW_STATS_KEY="rank${IASW_RANK}_eps${IASW_EPS_TAG}_calibskip${SKIP_LINES}_calibn${IASW_NUM_SAMPLES}"
                IASW_STATS_FILE="${IASW_STATS_DIR}/ias_stats_${DS_TYPE}_${MODEL_NAME}_len${LEN}_${IASW_STATS_KEY}.pt"

                if [[ "${IASW_MODE}" == "recalibrate" ]]; then
                    echo "  🔁 [IAS-W] recalibrate-only: 固定 baseline + 固定 whitening，仅更新经验标定"

                    if [[ ! -f "${IASW_BASELINE_FILE}" ]]; then
                        echo "  ❌ 缺少原 calibration baseline，recalibrate 模式禁止自动重建:"
                        echo "     ${IASW_BASELINE_FILE}"
                        exit 1
                    fi
                    if [[ ! -f "${IASW_STATS_FILE}" ]]; then
                        echo "  ❌ 缺少原 stats.pt，recalibrate 模式禁止自动重建 whitening:"
                        echo "     ${IASW_STATS_FILE}"
                        exit 1
                    fi

                    BASELINE_N=$(wc -l < "${IASW_BASELINE_FILE}" || echo 0)
                    echo "  📊 baseline valid rows = ${BASELINE_N}"
                    if [[ "${BASELINE_N}" -lt 100 ]]; then
                        echo "  ❌ baseline 有效样本过少。"
                        exit 1
                    fi

                    if [[ "${FORCE_IASW_STATS_RERUN}" -eq 1 ]]; then
                        UPDATE_ARGS=()
                        if [[ "${UPDATE_IASW_MARK}" -eq 1 ]]; then
                            UPDATE_ARGS=(--update_iasw_mark --iasw_mark_file "${IASW_MARK_FILE}")
                        fi
                        BACKUP_ARGS=()
                        if [[ "${BACKUP_EXISTING_STATS}" -eq 1 ]]; then
                            BACKUP_ARGS=(--backup_existing)
                        fi

                        python src/calibration/calibrate_empirical_iasw.py \
                            --mode recalibrate \
                            --baseline_file "${IASW_BASELINE_FILE}" \
                            --stats_file "${IASW_STATS_FILE}" \
                            --dataset_type "${DS_TYPE}" \
                            --model_path "${LLM_PATH}" \
                            --embedder_path "${EMBEDDER_PATH}" \
                            --seq_len "${LEN}" \
                            --rank "${IASW_RANK}" \
                            --eps "${IASW_EPS}" \
                            --num_samples "${IASW_NUM_SAMPLES}" \
                            --max_stats_sentences "${IASW_MAX_STATS_SENTENCES}" \
                            --batch_size "${IASW_BATCH_SIZE}" \
                            --stats_key "${IASW_STATS_KEY}" \
                            "${BACKUP_ARGS[@]}" \
                            "${UPDATE_ARGS[@]}"

                        verify_iasw_stats "${IASW_STATS_FILE}" "recalibrate"
                        echo "  ✅ [IAS-W] recalibration 完成；whitening 未重建。"
                    else
                        echo "  ⚡ FORCE_IASW_STATS_RERUN=0，跳过 recalibration。"
                    fi

                else
                    echo "  🧱 [IAS-W] full build: baseline → whitening → calibration"

                    NEED_BUILD_BASELINE=0
                    if [[ "${FORCE_BASELINE_RERUN}" -eq 1 || ! -f "${IASW_BASELINE_FILE}" ]]; then
                        NEED_BUILD_BASELINE=1
                    else
                        BASELINE_N=$(wc -l < "${IASW_BASELINE_FILE}" || echo 0)
                        if [[ "${BASELINE_N}" -lt "${IASW_NUM_SAMPLES}" ]]; then
                            NEED_BUILD_BASELINE=1
                        fi
                    fi

                    if [[ "${NEED_BUILD_BASELINE}" -eq 1 ]]; then
                        ensure_temp_raw "${RAW_LOCAL_FILE}" "${TEMP_RAW_FILE}" "${DS_TYPE}"
                        echo "  🧱 构建 calibration baseline: ${IASW_BASELINE_FILE}"
                        rm -f "${IASW_BASELINE_FILE}"
                        python src/utils/build_baseline_dataset.py \
                            --input_file "${TEMP_RAW_FILE}" \
                            --output_file "${IASW_BASELINE_FILE}" \
                            --dataset_type "${DS_TYPE}" \
                            --model_path "${LLM_PATH}" \
                            --target_len "${LEN}" \
                            --num_samples "${IASW_NUM_SAMPLES}" \
                            --batch_size "${IASW_BASELINE_BATCH_SIZE}"
                    else
                        echo "  ⚡ calibration baseline 已存在，复用: ${IASW_BASELINE_FILE}"
                    fi

                    BASELINE_N=$(wc -l < "${IASW_BASELINE_FILE}" || echo 0)
                    if [[ "${BASELINE_N}" -lt 100 ]]; then
                        echo "  ❌ baseline 有效样本过少。"
                        exit 1
                    fi

                    if [[ "${FORCE_IASW_STATS_RERUN}" -eq 1 || ! -f "${IASW_STATS_FILE}" ]]; then
                        UPDATE_ARGS=()
                        if [[ "${UPDATE_IASW_MARK}" -eq 1 ]]; then
                            UPDATE_ARGS=(--update_iasw_mark --iasw_mark_file "${IASW_MARK_FILE}")
                        fi

                        python src/calibration/calibrate_empirical_iasw.py \
                            --mode full \
                            --baseline_file "${IASW_BASELINE_FILE}" \
                            --stats_file "${IASW_STATS_FILE}" \
                            --dataset_type "${DS_TYPE}" \
                            --model_path "${LLM_PATH}" \
                            --embedder_path "${EMBEDDER_PATH}" \
                            --seq_len "${LEN}" \
                            --rank "${IASW_RANK}" \
                            --eps "${IASW_EPS}" \
                            --num_samples "${IASW_NUM_SAMPLES}" \
                            --max_stats_sentences "${IASW_MAX_STATS_SENTENCES}" \
                            --batch_size "${IASW_BATCH_SIZE}" \
                            --stats_key "${IASW_STATS_KEY}" \
                            "${UPDATE_ARGS[@]}"

                        verify_iasw_stats "${IASW_STATS_FILE}" "full"
                        echo "  ✅ [IAS-W] full stats 构建完成。"
                    else
                        echo "  ⚡ stats 已存在，跳过: ${IASW_STATS_FILE}"
                    fi
                fi
            fi

            if [[ "${RUN_NULL_METHODS}" -eq 1 ]]; then
                ensure_temp_raw "${RAW_LOCAL_FILE}" "${TEMP_RAW_FILE}" "${DS_TYPE}"
                for METHOD in "${NULL_METHODS[@]}"; do
                    OUTPUT_FILE="${MODEL_CALIB_DIR}/calib_params_${METHOD}_len${LEN}.json"
                    if [[ "${FORCE_NULL_RERUN}" -eq 0 && -f "${OUTPUT_FILE}" ]]; then
                        echo "  ⚡ [Null] 已存在，跳过: ${OUTPUT_FILE}"
                        continue
                    fi
                    python src/calibration/calibrate_empirical_null.py \
                        --input_file "${TEMP_RAW_FILE}" \
                        --output_file "${OUTPUT_FILE}" \
                        --dataset_type "${DS_TYPE}" \
                        --seq_len "${LEN}" \
                        --sample_size "${NULL_NUM_SAMPLES}" \
                        --method "${METHOD}" \
                        --tokenizer_path "${LLM_PATH}"
                done
            fi
        done
    done
done

if [[ "${RUN_NULL_METHODS}" -eq 1 ]]; then
    python data/calibration_set/build_calibration_tables.py
fi

echo -e "\n🎉 完成。日志: ${LOG_FILE}"
