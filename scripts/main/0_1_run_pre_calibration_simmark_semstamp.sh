#!/bin/bash
set -e

# === PROJECT ROOT BOOTSTRAP (refactor v4) ===
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
# === END PROJECT ROOT BOOTSTRAP ===


# =================================================================
# ⚙️ 0. 环境配置
# =================================================================
export CUDA_VISIBLE_DEVICES=01
export HF_ENDPOINT=https://hf-mirror.com
export NLTK_DATA=./models/nltk_data
PROJECT_DIR="${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# 🌟 全局日志记录系统
LOG_DIR="${PROJECT_DIR}/results/logs/calibration/"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/calib_run_simmark_ksemstamp_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1


DATASETS=(
    # "c4" 
    "wiki" 
    "dolly"
)
SKIP_LINES=3000 
SEMSTAMP_SBERT="AbeHou/SemStamp-c4-sbert"
PCA_COMPONENTS=16

echo "======================================================"
echo "🎯 启动一体化校准 [PCA 训练 + Gamma 标定]"
echo "======================================================"

for DS_TYPE in "${DATASETS[@]}"; do
    echo -e "\n📦 处理数据集: [ ${DS_TYPE^^} ]"

    RAW_FILE="data/raw_${DS_TYPE}.jsonl"
    if [ ! -f "${RAW_FILE}" ]; then continue; fi

    # 1. 提取物理隔离的 5000 条校准数据
    CALIB_TMP_FILE="data/tmp_calib_${DS_TYPE}.jsonl"
    tail -n +$((SKIP_LINES + 1)) "${RAW_FILE}" | head -n 5000 > "${CALIB_TMP_FILE}"


    # ---------------------------------------------------------
    # 🧪 [Step 2/3] 训练数据集专属 PCA 模型
    # ---------------------------------------------------------
    echo "--- [Step 2/3] 正在训练 ${DS_TYPE^^} 专属 PCA 模型 ---"
    
    # 将临时 jsonl 转换为 Dataset 格式供 PCA 脚本读取
    PCA_TRAIN_DIR="data/temp_pca_${DS_TYPE}"
    python src/utils/prepare_semstamp_data.py --input_file "${CALIB_TMP_FILE}" --output_dir "${PCA_TRAIN_DIR}"

    python src/baselines/simmark_core/train_PCA.py \
        --num_components ${PCA_COMPONENTS} \
        --train_texts_path "${PCA_TRAIN_DIR}" \
        --embedder "hkunlp/instructor-large" \
        --batch_size 128

    # 移动并按数据集命名 (对应你代码里的根据数据集选择逻辑)
    mv "new_pca_model_${PCA_COMPONENTS}.pkl" "src/baselines/simmark_core/${DS_TYPE}_pca_model_${PCA_COMPONENTS}.pkl"
    rm -rf "${PCA_TRAIN_DIR}"

    # ---------------------------------------------------------
    # 🧪 [Step 3/3] SimMark Gamma 校准 (使用新练的 PCA)
    # ---------------------------------------------------------
    echo "--- [Step 3/3] SimMark Gamma 校准 ---"

    case "${DS_TYPE}" in
        "wiki")  COS_A=0.70; COS_B=0.78; EUC_A=0.24; EUC_B=0.32 ;;
        "dolly") COS_A=0.71; COS_B=0.79; EUC_A=0.23; EUC_B=0.31 ;;
        "c4")    COS_A=0.68; COS_B=0.76; EUC_A=0.26; EUC_B=0.34 ;;
    esac

    python src/baselines/simmark_core/calibrate_simmark.py \
        --mode cosine --input_file "${CALIB_TMP_FILE}" --a ${COS_A} --b ${COS_B}

    # 🌟 关键：calibrate 脚本内部也需要根据 DS_TYPE 加载对应的 PCA 模型
    python src/baselines/simmark_core/calibrate_simmark.py \
        --mode euclidean --use_pca --input_file "${CALIB_TMP_FILE}" \
        --dataset_type "${DS_TYPE}" \
        --a ${EUC_A} --b ${EUC_B}

    rm "${CALIB_TMP_FILE}"
done