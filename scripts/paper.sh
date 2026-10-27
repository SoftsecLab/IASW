#!/usr/bin/env bash
set -euo pipefail


# =====================================================================
# 0. Project root
# =====================================================================

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

cd "${_PROJECT_ROOT}"

export PYTHONPATH="${_PROJECT_ROOT}:${PYTHONPATH:-}"


# =====================================================================
# 1. IAS-W alignment source refresh
# =====================================================================
#
# 每次调用 scripts/paper.sh 时都会先从正式 IAS-W 主实验 JSONL
# 重新提取 alignment-scoring ablation。
#
# 输入：
#   results/main_exp/c4/llama3.1-8b-instruct/records/iasw.jsonl
#
# 输出：
#   results/analysis/iasw/alignment_ablation_human/
#
# build_paper.py 的 --only special / --only all 会进一步读取该结果，
# 生成：
#   paper/csv/alignment_ablation_source.csv
#   paper/latex/table06_alignment_ablation.tex
#
# 注意：
#   这一 extraction 是 paper.sh wrapper 的固定前置步骤，因此即使调用
#   --only robustness / quality / efficiency 等模式，也仍会执行。
# =====================================================================

python analysis/iasw_alignment/extract_iasw_alignment_ablation.py \
    --input_file results/main_exp/c4/llama3.1-8b-instruct/records/iasw.jsonl \
    --output_dir results/analysis/iasw/alignment_ablation_human \
    --negative_sources human


# =====================================================================
# 2. Canonical paper table / CSV builder
# =====================================================================
#
# "$@" 会原样传递给 build_paper.py。
#
# 支持：
#   --only all
#   --only robustness
#   --only quality
#   --only efficiency
#   --only ablation
#   --only fusion
#   --only special
#
# 不提供 --only 时默认：
#   --only all
# =====================================================================

python analysis/paper/build_paper.py "$@"


# =====================================================================
# 3. Parameter-ablation figures
# =====================================================================
#
# 输入：
#   paper/csv/candidate_ablation_source.csv
#   paper/csv/token_length_ablation_source.csv
#
# 输出：
#   paper/figures/candidate_ablation_tpr1.{pdf,eps,tiff}
#   paper/figures/token_length_ablation_tpr1.{pdf,eps,tiff}
#
# 注意：
#   该命令目前也是 paper.sh wrapper 的固定后处理步骤，因此无论
#   build_paper.py 使用哪一种 --only 模式，都会尝试根据当前已有
#   parameter-ablation CSV 重新绘制这两张图。
#
#   --only ablation 时，上述两个 CSV 会先由 build_paper.py 重建；
#   其他 --only 模式下，则使用 paper/csv/ 中已有的 CSV。
# =====================================================================


python analysis/paper/plot_attack_profile_tpr5.py

python analysis/paper/plot_cross_backbone_tpr5.py

python analysis/paper/plot_text_quality_tpr5.py

python analysis/paper/plot_efficiency_tradeoff_tpr5.py

python analysis/paper/plot_token_length_ablation_tpr5.py

python analysis/paper/plot_candidate_ablation_tpr5.py


