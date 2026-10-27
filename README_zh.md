# IAS-W: 基于语义锚定的 Token 级文本水印（Semantic Anchoring Enhanced Token-Level Watermarking）

论文《Semantic Anchoring Enhanced Token-Level Watermarking for Large Language
Models》的官方代码仓库。

IAS-W 通过**语义锚定（semantic anchoring）**增强 token 级文本水印的鲁棒性：
在漂白（whitened）语义空间中为每个句位分配秘密方向（anchor），在生成阶段
以锚为导向从 token 级水印的候选句中选择输出句；检测阶段通过对齐感知的
局部动态规划恢复锚序列，并与人称校准后的 token 级统计量做 Fisher 式聚合。
底层 token 级水印规则与检测器保持不变。


## 论文要点

- 主方法：**KGW+IAS-W**（与 SWEET、DiPmark 的组合见兼容性实验）
- 数据集：RealNews（C4 news 子集）、DOLLY、Wiki（WikiText-103）
- 生成模型：OPT-SemStamp（`AbeHou/opt-1.3b-semstamp`）、Llama-3.2-3B、
  Mistral-7B、Qwen2.5-7B、Llama-3.1-8B
- 攻击（9 种）：Typo / Drop / Swap（表面扰动）、Deletion-Insertion /
  Copy-Paste / Sentence Reordering（结构编辑）、PEGASUS / DIPPER（语义重写）、
  Back Translation（翻译）
- 指标：AUROC、TPR@1%FPR、TPR@5%FPR；PPL / CleanSim（质量）；生成时间（效率）
- 主要结论：RealNews 五个模型上 KGW+IAS-W 平均 TPR@5%FPR = **90.4%**，
  在全部五个模型上该指标排名第一。

## 目录结构

```
IASW/
├── src/            # 核心实现
│   ├── watermarks/ # IAS-W 本体（algorithms/macro|micro、single、dual 组合、runtime）
│   ├── attacks/    # 9 种攻击实现
│   ├── baselines/  # KGW/STA/SWEET/DiPmark/SemStamp/k-SemStamp/SimMark/DualMark/SynGuard 等接口适配
│   ├── evaluation/ # ROC / 质量 / 汇总
│   └── calibration/# 经验校准
├── analysis/       # 论文表图构建（paper/）+ 机制/安全/攻击质量分析（kgw_generation、security_validation、quality、iasw_alignment）
├── scripts/        # 实验与复现脚本（reproduce/ 两级复现）
├── configs/        # 模型映射（model_map.example.json，改名为 model_map.json 并指向本地权重）
├── data/           # 数据下载脚本（不包含第三方原始数据）
├── paper/          # paper/csv（表格源数据）、paper/latex、paper/figures
└── tools/          # 复现辅助（检测/基线构建工具）
```

> 说明：analysis/、scripts/、tools/ 只保留论文直接使用的模块与复现主链。

## 复现

### Level 1 — 复现论文数字（无需 GPU）

下载数据包 **IASW-data.zip**（Zenodo: https://doi.org/10.5281/zenodo.23601338 ，
记录页 https://zenodo.org/records/23601338 ），解压后与本仓库同级（布局见
`data/README.md` 与 `DATA.md`）。`paper/csv/` 已包含论文表格的最终数据源；
若想从发布的记录重算，指向数据包根目录运行：

```bash
python analysis/paper/build_paper.py --data-root ../data
```

`--data-root` 会自动读取 `data/records/main/`（扁平布局：
`<dataset>/<model>/<method>.jsonl`）并重建 `paper/csv/`、`paper/latex/`。
不带该参数时，脚本按 Level 2 的本地 `results/` 目录布局读取。

### Level 2 — 全量复现（需要 GPU）

```bash
# 1. 下载原始数据（第三方数据不随仓库分发）
python data/download_datasets.py

# 2. 下载第三方基线权重
python data/download_baseline_assets.py

# 3. 构造 baseline triplets / calibration
bash scripts/main/0_1_run_pre_calibration_simmark_semstamp.sh
bash scripts/main/0_2_build_calib_table.sh

# 4. 主实验
bash scripts/main/1_run_main_pipeline.sh

# 5. 消融与机制
bash scripts/reproduce/40_ablations.sh

# 6. 论文表图重建
bash scripts/reproduce/50_analysis.sh
```

或直接：

```bash
bash scripts/reproduce/all.sh --paper-only   # 只做 Level 1
bash scripts/reproduce/all.sh --full         # Level 1 + Level 2
```

### 环境

论文实验环境：Python 3.9.25、PyTorch 2.1.0、Hugging Face Transformers 4.38、
NVIDIA RTX 4090 48G。见 `requirements.txt` / `environment.yml`。

## 数据与再分发

- **第三方原始数据**（C4 / DOLLY / Wiki）不随仓库分发，通过
  `data/download_datasets.py` 从原始提供方获取；详细重建规则见 `DATA.md`。
- **论文生成数据**（wm_text / attacked_text / detection scores）在独立数据包中
  发布（sanitized，不含第三方 human_text），见 `data/README.md`；
  Zenodo: https://doi.org/10.5281/zenodo.23601338
- **第三方基线权重**不随仓库分发，通过 `data/download_baseline_assets.py`
  按官方 URL + SHA256 获取。

## 实验密钥

论文实验使用固定密钥 `DUAL_MACRO_2026`，随代码发布**仅用于复现论文**。
实际部署中必须替换为私有密钥：

```bash
python -m src ... --secret_key YOUR_PRIVATE_KEY
```

## 许可证

主仓库采用 MIT 许可证（见 `LICENSE`）。`src/baselines/` 下第三方实现保留
各自上游许可证声明（如 THU-BPM MarkLLM 为 Apache-2.0），详见 `NOTICE`。
