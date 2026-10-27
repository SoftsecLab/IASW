# IAS-W: Semantic Anchoring Enhanced Token-Level Watermarking

Official code repository for the paper *Semantic Anchoring Enhanced Token-Level
Watermarking for Large Language Models*.

IAS-W enhances the robustness of token-level text watermarking through
**semantic anchoring**: it assigns a secret direction (anchor) to each sentence
slot in the whitened semantic space, and during generation selects the output
sentence from token-level watermark candidates under the guidance of the anchor.
At detection time, it recovers the anchor sequence via alignment-aware local
dynamic programming and aggregates it with person-calibrated token-level
statistics through a Fisher-style combination. The underlying token-level
watermark rule and detector remain unchanged.

## Paper Highlights

- Primary method: **KGW+IAS-W** (combinations with SWEET and DiPmark are covered
  in the compatibility experiments)
- Datasets: RealNews (C4 news subset), DOLLY, Wiki (WikiText-103)
- Generation models: OPT-SemStamp (`AbeHou/opt-1.3b-semstamp`), Llama-3.2-3B,
  Mistral-7B, Qwen2.5-7B, Llama-3.1-8B
- Attacks (9): Typo / Drop / Swap (surface perturbations),
  Deletion-Insertion / Copy-Paste / Sentence Reordering (structural edits),
  PEGASUS / DIPPER (semantic rewriting), Back Translation
- Metrics: AUROC, TPR@1%FPR, TPR@5%FPR; PPL / CleanSim (quality); generation
  time (efficiency)
- Main result: average TPR@5%FPR of KGW+IAS-W across the five RealNews models is
  **90.4%**, ranked first on all five models for this metric.

## Repository Layout

```
IASW/
├── src/            # Core implementation
│   ├── watermarks/ # IAS-W itself (algorithms/macro|micro, single, dual combinations, runtime)
│   ├── attacks/    # 9 attack implementations
│   ├── baselines/  # Adapters for KGW/STA/SWEET/DiPmark/SemStamp/k-SemStamp/SimMark/DualMark/SynGuard etc.
│   ├── evaluation/ # ROC / quality / summarization
│   └── calibration/# Empirical calibration
├── analysis/       # Paper tables/figures (paper/) + mechanism/security/attack-quality analysis
│                   # (kgw_generation, security_validation, quality, iasw_alignment)
├── scripts/        # Experiment and reproduction scripts (reproduce/ two-level reproduction)
├── configs/        # Model mapping (model_map.example.json; rename to model_map.json and point to local weights)
├── data/           # Data download scripts (third-party raw data NOT included)
├── paper/          # paper/csv (table sources), paper/latex, paper/figures
└── tools/          # Reproduction helpers (detection / baseline construction tools)
```

> Note: `analysis/`, `scripts/`, `tools/` keep only the modules used directly by
> the paper and the main reproduction chain.

## Reproduction

### Level 1 — Reproduce paper numbers (no GPU required)

Download the data package **IASW-data.zip** (Zenodo:
https://doi.org/10.5281/zenodo.23601338 , record page
https://zenodo.org/records/23601338 ), unzip it next to this repository (see
`data/README.md` and `DATA.md` for the layout). `paper/csv/` already contains
the final data sources for the paper tables; to recompute them from the released
records, point to the data package root and run:

```bash
python analysis/paper/build_paper.py --data-root ../data
```

`--data-root` automatically reads `data/records/main/` (flat layout:
`<dataset>/<model>/<method>.jsonl`) and rebuilds `paper/csv/` and
`paper/latex/`. Without this argument, the script reads from the local
`results/` layout used in Level 2.

### Level 2 — Full reproduction (GPU required)

```bash
# 1. Download raw data (third-party data is not shipped with the repository)
python data/download_datasets.py

# 2. Download third-party baseline weights
python data/download_baseline_assets.py

# 3. Build baseline triplets / calibration
bash scripts/main/0_1_run_pre_calibration_simmark_semstamp.sh
bash scripts/main/0_2_build_calib_table.sh

# 4. Main experiments
bash scripts/main/1_run_main_pipeline.sh

# 5. Ablations and mechanism analyses
bash scripts/reproduce/40_ablations.sh

# 6. Rebuild paper tables/figures
bash scripts/reproduce/50_analysis.sh
```

Or directly:

```bash
bash scripts/reproduce/all.sh --paper-only   # Level 1 only
bash scripts/reproduce/all.sh --full         # Level 1 + Level 2
```

### Environment

Paper experiments: Python 3.9.25, PyTorch 2.1.0, Hugging Face Transformers 4.38,
NVIDIA RTX 4090 48G. See `requirements.txt` / `environment.yml`.

## Data and Redistribution

- **Third-party raw data** (C4 / DOLLY / Wiki) is not shipped with the
  repository; obtain it from the original providers via
  `data/download_datasets.py`. Detailed reconstruction rules are in `DATA.md`.
- **Paper-generated data** (wm_text / attacked_text / detection scores) is
  released in a separate data package (sanitized, without third-party
  human_text), see `data/README.md`;
  Zenodo: https://doi.org/10.5281/zenodo.23601338
- **Third-party baseline weights** are not shipped with the repository; fetch
  them via `data/download_baseline_assets.py` using the official URLs + SHA256.

## Experiment Key

The paper experiments use the fixed key `DUAL_MACRO_2026`, released with the
code **for paper reproduction only**. For any real deployment, replace it with a
private key:

```bash
python -m src ... --secret_key YOUR_PRIVATE_KEY
```

## License

The main repository is released under the MIT License (see `LICENSE`). Third-party
implementations under `src/baselines/` retain their own upstream license notices
(e.g., THU-BPM MarkLLM under Apache-2.0); see `NOTICE`.

---

Chinese version: [README_zh.md](README_zh.md)
