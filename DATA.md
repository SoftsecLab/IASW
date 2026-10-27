# DATA.md — Data Sources, Reconstruction Rules, and Release Boundaries

This document explains the sources, redistribution boundaries, and
reconstruction methods for each data layer used in the paper *Semantic Anchoring
Enhanced Token-Level Watermarking for Large Language Models*.

## 1. Three-Layer Data Split

| Layer | Content | Shipped with the repository | How to obtain |
| ----- | ------- | ---------------------------- | ------------- |
| Third-party raw data | C4 / DOLLY / Wiki raw text | No | `data/download_datasets.py` |
| Paper-generated / detection data | wm_text, attacked_text, detection scores | Yes (separate data package, sanitized) | Data package `records/`, `scores/` |
| Model and baseline weights | LLM weights, third-party baseline checkpoints | No | Hugging Face / `data/download_baseline_assets.py` |

## 2. Dataset Sources and Reconstruction

### RealNews (the C4 news-style subset used in the paper)

- Upstream: `allenai/c4` (Common Crawl cleaned version)
- Construction: news-style samples are filtered from C4 (the news filtering
  rules are in `data/download_datasets.py` and
  `src/utils/build_baseline_dataset.py`)
- Reconstruction: given `sample_id`, selection seed, `skip`, `num_samples`,
  `target_len`, and the preprocessing version, the experiment set identical to
  the paper can be reconstructed. **Raw texts are not in the release package.**

### DOLLY

- Upstream: `databricks/dolly-15k` (Hugging Face)
- Reconstruction: same sampling rules.

### Wiki

- Upstream: WikiText-103 (Salesforce)
- Reconstruction: same sampling rules.

## 3. Baseline Triplets and Calibration

- `data/main_exp/baseline_triplets/` (prompt / human_text / no_wm_text) is not
  in the release package; it is reconstructed by `download_datasets.py` +
  `build_baseline_dataset.py` following the sampling rules.
- Calibration is determined by fixed rules, so the full calibration set does not
  need to be released:

```
calibration:
  skip: 3000          # IAS_CALIB_SKIP_LINES
  n: 1000             # IAS_CALIB_NUM_SAMPLES
  source_fields: [human_text, no_wm_text]
```

- The IAS-W calibration statistics used by the official experiments
  (`stats/iasw/*.pt`) are released in the data package as a convenience
  artifact; to rebuild them yourself, run
  `src/calibration/calibrate_empirical_iasw.py`.

## 4. Paper-Generated / Detection Data (Data Package)

The `data/records/` in the data package is sanitized:

Data package file: **IASW-data.zip** (Zenodo:
https://doi.org/10.5281/zenodo.23601338 , record page
https://zenodo.org/records/23601338 )

- **Removed**: `prompt`, `human_text` (third-party raw text), local absolute
  paths, internal log fields
- **Kept**: `sample_id`, `dataset`, `model`, `method`, `no_wm_text`, `wm_text`,
  `detection_*`, `attacks` (attacked_text / detection_results)
- `sample_id` is aligned with the source index of the baseline, so upstream text
  can be recovered together with the reconstruction rules in Section 2.

## 5. Model List

| Model | Source |
| ------ | ------ |
| OPT-SemStamp | `AbeHou/opt-1.3b-semstamp` (the fine-tuned checkpoint used in the original SemStamp setup) |
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| Meta-Llama-3.1-8B-Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Sentence embedder | `BAAI/bge-small-en-v1.5` |

## 6. Third-Party Baseline Weights

The following files are not in the repository; obtain them from upstream via
`data/download_baseline_assets.py`:

- `synguard_core/watermark/sir/model/transform_model_cbert.pth`
- `dualmark_core/model/opt/init_0.25_1.75_default.pth`
- SemStamp clusters (`*-cluster_8_centers.pt`) and SimMark PCA
  (`*_pca_model_16.pkl`): if built for the paper they are released in the data
  package; if from the official repositories, fetch them via official URL +
  SHA256.

## 7. Reproduction Checklist

- [ ] `python data/download_datasets.py`
- [ ] `python data/download_baseline_assets.py`
- [ ] Build the calibration table (`scripts/main/0_2_build_calib_table.sh`)
- [ ] Main experiments (`scripts/main/1_run_main_pipeline.sh`)
- [ ] Ablations (`scripts/reproduce/40_ablations.sh`)
- [ ] Paper tables/figures (`scripts/reproduce/50_analysis.sh` →
  `analysis/paper/build_paper.py`)
