# IASW Tools

`tools/` contains repository utilities that are useful across experiments but
are not part of the reusable watermark runtime.

Scientific post-processing belongs in `analysis/`; experiment launchers belong
in `scripts/`; runtime code belongs in `src/`.

## Active layout

- `baselines/`
  - `build_synguard_mappings.py`: builds the word-to-sentence mapping table
    required by the SynGuard baseline from a local watermark checkpoint.

- `detection/`
  - `detect_kgw_jsonl_cuda.py`: standalone CUDA-aware KGW detector used by
    mechanism diagnostics where device-consistent green-list behavior matters.

## Output policy

`tools/` should not own an output tree. In particular, audits belong under:

```text
results/audits/
```

Do not create `tools/results/`.

## Policy

- Paper tables/figures and scientific post-processing -> `analysis/`.
- Experiment launchers -> `scripts/`.
- Reusable algorithms/runtime -> `src/`.
- One-off migration/patch/install scripts -> archive after the canonical
  runtime/configuration has been installed.
