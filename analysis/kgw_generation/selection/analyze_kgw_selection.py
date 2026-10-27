#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated IASW analysis entry.

Generated from the current analysis scripts without changing their scientific
calculations. Each subcommand executes the original implementation in an
isolated function scope so duplicate helper names do not collide.
"""
from __future__ import annotations

import sys

def _run_paired_bootstrap(argv):
    """Run preserved implementation from analyze_kgw_selection.py[paired-bootstrap]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_selection.py[paired-bootstrap]"] + list(argv)

        """
        Paired / hierarchical bootstrap for the corrected CUDA four-stage KGW experiment.

        Existing conditions:
            KGW
            Random-K1
            Random-K32
            IASW-K32

        Controlled contrasts:
            sentence = Random-K1  - KGW
            pool     = Random-K32 - Random-K1
            select   = IASW-K32   - Random-K32
            total    = IASW-K32   - KGW

        The script reads the per-sample detector CSV files created by
        detect_kgw_jsonl_cuda.py.

        It does NOT regenerate or redetect any text.

        Outputs
        -------
        1. stage_sample_long.csv
        2. stage_paired_effects_per_seed.csv
        3. stage_paired_effects_hierarchical.csv
        4. stage_condition_summary.csv
        5. missing_detector_files.csv

        Hierarchical bootstrap:
            resample seeds with replacement;
            within each sampled seed, resample paired prompt/sample rows with replacement.

        This respects:
            - pairing across methods for the same prompt/sample
            - repeated use of the same prompt set across multiple generation seeds
        """


        import argparse
        from pathlib import Path
        from typing import Dict, List, Tuple

        import numpy as np
        import pandas as pd


        MODELS = [
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Llama-3.2-3B-Instruct",
        ]

        SEEDS = [
            "20260821",
            "20260822",
            "20260823",
            "20260824",
            "20260825",
        ]


        def parse_args():
            p = argparse.ArgumentParser()

            p.add_argument(
                "--old_root",
                default="results/diagnostics/generation/kgw_randomk1_seed_sweep/c4",
                help="Contains KGW and Random-K1 corrected detector CSVs.",
            )

            p.add_argument(
                "--new_root",
                default="results/diagnostics/generation/kgw_selection/c4",
                help="Contains Random-K32 and IASW-K32 corrected detector CSVs.",
            )

            p.add_argument(
                "--output_dir",
                default="results/diagnostics/generation/kgw_selection/c4/statistics",
            )

            p.add_argument(
                "--bootstrap",
                type=int,
                default=10000,
            )

            p.add_argument(
                "--bootstrap_seed",
                type=int,
                default=20260827,
            )

            return p.parse_args()


        def expected_paths(old_root: Path, new_root: Path, model: str, seed: str):
            return {
                "KGW":
                    old_root / model / f"seed_{seed}" / "records"
                    / "kgw_control.kgw_detection.csv",

                "Random-K1":
                    old_root / model / f"seed_{seed}" / "records"
                    / "dual_kgw_random_k1.kgw_detection.csv",

                "Random-K32":
                    new_root / model / f"seed_{seed}" / "records"
                    / "random_k32.kgw_detection.csv",

                "IASW-K32":
                    new_root / model / f"seed_{seed}" / "records"
                    / "iasw_k32.kgw_detection.csv",
            }


        def read_one(path: Path, model: str, seed: str, condition: str):
            df = pd.read_csv(path)

            required = [
                "sample_index",
                "z_no_repeat_filter",
                "green_fraction_no_repeat_filter",
                "num_tokens_no_repeat_filter",
                "z_repeat_filter",
                "green_fraction_repeat_filter",
                "num_tokens_repeat_filter",
            ]

            missing = [
                c for c in required
                if c not in df.columns
            ]

            if missing:
                raise RuntimeError(
                    f"{path}: missing columns {missing}"
                )

            out = df[required].copy()

            out["model"] = model
            out["seed"] = str(seed)
            out["condition"] = condition

            # Keep terminology aligned with the paper.
            out = out.rename(columns={
                "z_no_repeat_filter": "z_all",
                "green_fraction_no_repeat_filter": "green_all",
                "num_tokens_no_repeat_filter": "T_all",
                "z_repeat_filter": "z_repeat",
                "green_fraction_repeat_filter": "green_repeat",
                "num_tokens_repeat_filter": "T_eff",
            })

            # Repeated-bigram contribution can be derived directly.
            out["repeat_bigram_count"] = (
                out["T_all"] - out["T_eff"]
            )

            out["repeat_bigram_rate"] = np.where(
                out["T_all"] > 0,
                out["repeat_bigram_count"] / out["T_all"],
                np.nan,
            )

            return out


        def collect(old_root: Path, new_root: Path):
            frames = []
            missing = []

            for model in MODELS:
                for seed in SEEDS:
                    paths = expected_paths(
                        old_root,
                        new_root,
                        model,
                        seed,
                    )

                    for condition, path in paths.items():
                        if not path.is_file():
                            missing.append({
                                "model": model,
                                "seed": seed,
                                "condition": condition,
                                "expected_file": str(path),
                            })
                            continue

                        frames.append(
                            read_one(
                                path,
                                model,
                                seed,
                                condition,
                            )
                        )

            if not frames:
                raise RuntimeError(
                    "No corrected detector CSV files were found."
                )

            return (
                pd.concat(
                    frames,
                    ignore_index=True,
                ),
                pd.DataFrame(missing),
            )


        CONTRASTS = {
            "sentence_RandomK1_minus_KGW":
                ("Random-K1", "KGW"),

            "pool_RandomK32_minus_RandomK1":
                ("Random-K32", "Random-K1"),

            "selection_IASW_minus_RandomK32":
                ("IASW-K32", "Random-K32"),

            "total_IASW_minus_KGW":
                ("IASW-K32", "KGW"),
        }


        METRICS = [
            "z_repeat",
            "T_eff",
            "green_repeat",
            "z_all",
            "green_all",
            "repeat_bigram_count",
            "repeat_bigram_rate",
        ]


        def paired_frame(
            df: pd.DataFrame,
            model: str,
            seed: str,
            positive: str,
            negative: str,
        ):
            sub = df[
                (df["model"] == model)
                & (df["seed"].astype(str) == str(seed))
            ]

            a = sub[
                sub["condition"] == positive
            ].set_index("sample_index")

            b = sub[
                sub["condition"] == negative
            ].set_index("sample_index")

            ids = a.index.intersection(b.index)

            if len(ids) == 0:
                return None

            result = pd.DataFrame(
                index=ids
            )

            for metric in METRICS:
                result[metric] = (
                    pd.to_numeric(
                        a.loc[ids, metric],
                        errors="coerce",
                    )
                    -
                    pd.to_numeric(
                        b.loc[ids, metric],
                        errors="coerce",
                    )
                )

            result = result.reset_index()
            result["model"] = model
            result["seed"] = str(seed)

            return result


        def bootstrap_mean(
            values: np.ndarray,
            rng,
            n_boot,
        ):
            values = values[
                np.isfinite(values)
            ]

            if len(values) == 0:
                return {
                    "n": 0,
                    "mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_boot": np.nan,
                }

            mean = float(
                np.mean(values)
            )

            idx = rng.integers(
                0,
                len(values),
                size=(n_boot, len(values)),
            )

            boot = values[idx].mean(axis=1)

            ci_low, ci_high = np.percentile(
                boot,
                [2.5, 97.5],
            )

            # Simple two-sided bootstrap sign probability.
            p_boot = min(
                1.0,
                2.0 * min(
                    np.mean(boot <= 0),
                    np.mean(boot >= 0),
                ),
            )

            return {
                "n": int(len(values)),
                "mean": mean,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_boot": float(p_boot),
            }


        def per_seed_effects(
            df: pd.DataFrame,
            rng,
            n_boot,
        ):
            rows = []

            for model in MODELS:
                for seed in SEEDS:
                    for contrast_name, (
                        positive,
                        negative,
                    ) in CONTRASTS.items():

                        paired = paired_frame(
                            df,
                            model,
                            seed,
                            positive,
                            negative,
                        )

                        if paired is None:
                            continue

                        for metric in METRICS:
                            result = bootstrap_mean(
                                paired[metric]
                                .to_numpy(dtype=float),
                                rng,
                                n_boot,
                            )

                            rows.append({
                                "model": model,
                                "seed": seed,
                                "contrast": contrast_name,
                                "positive_condition": positive,
                                "negative_condition": negative,
                                "metric": metric,
                                **result,
                            })

            return pd.DataFrame(rows)


        def hierarchical_bootstrap(
            df: pd.DataFrame,
            rng,
            n_boot,
        ):
            rows = []

            for model in MODELS:
                for contrast_name, (
                    positive,
                    negative,
                ) in CONTRASTS.items():

                    paired_by_seed = {}

                    for seed in SEEDS:
                        paired = paired_frame(
                            df,
                            model,
                            seed,
                            positive,
                            negative,
                        )

                        if paired is not None:
                            paired_by_seed[
                                str(seed)
                            ] = paired

                    available_seeds = list(
                        paired_by_seed.keys()
                    )

                    if not available_seeds:
                        continue

                    for metric in METRICS:

                        # Point estimate:
                        # mean of seed means so no seed gets extra weight
                        # because of a few missing samples.
                        seed_means = []

                        for seed in available_seeds:
                            vals = (
                                paired_by_seed[seed][metric]
                                .to_numpy(dtype=float)
                            )
                            vals = vals[
                                np.isfinite(vals)
                            ]

                            if len(vals):
                                seed_means.append(
                                    np.mean(vals)
                                )

                        if not seed_means:
                            continue

                        point = float(
                            np.mean(seed_means)
                        )

                        boot_values = []

                        n_seeds = len(
                            available_seeds
                        )

                        for _ in range(n_boot):

                            sampled_seed_indices = (
                                rng.integers(
                                    0,
                                    n_seeds,
                                    size=n_seeds,
                                )
                            )

                            sampled_seed_means = []

                            for seed_idx in (
                                sampled_seed_indices
                            ):
                                seed = available_seeds[
                                    seed_idx
                                ]

                                vals = (
                                    paired_by_seed[seed][
                                        metric
                                    ]
                                    .to_numpy(dtype=float)
                                )

                                vals = vals[
                                    np.isfinite(vals)
                                ]

                                if len(vals) == 0:
                                    continue

                                sample_idx = (
                                    rng.integers(
                                        0,
                                        len(vals),
                                        size=len(vals),
                                    )
                                )

                                sampled_seed_means.append(
                                    vals[sample_idx].mean()
                                )

                            if sampled_seed_means:
                                boot_values.append(
                                    np.mean(
                                        sampled_seed_means
                                    )
                                )

                        boot_values = np.asarray(
                            boot_values,
                            dtype=float,
                        )

                        ci_low, ci_high = np.percentile(
                            boot_values,
                            [2.5, 97.5],
                        )

                        p_boot = min(
                            1.0,
                            2.0 * min(
                                np.mean(
                                    boot_values <= 0
                                ),
                                np.mean(
                                    boot_values >= 0
                                ),
                            ),
                        )

                        total_pairs = sum(
                            len(
                                paired_by_seed[s]
                            )
                            for s in available_seeds
                        )

                        rows.append({
                            "model": model,
                            "contrast": contrast_name,
                            "positive_condition": positive,
                            "negative_condition": negative,
                            "metric": metric,
                            "n_seeds": n_seeds,
                            "total_paired_rows": total_pairs,
                            "mean": point,
                            "ci_low": float(ci_low),
                            "ci_high": float(ci_high),
                            "p_boot": float(p_boot),
                        })

            return pd.DataFrame(rows)


        def condition_summary(df: pd.DataFrame):

            return (
                df
                .groupby(
                    [
                        "model",
                        "condition",
                    ],
                    as_index=False,
                )
                .agg(
                    n_rows=(
                        "sample_index",
                        "count",
                    ),
                    z_repeat_mean=(
                        "z_repeat",
                        "mean",
                    ),
                    T_eff_mean=(
                        "T_eff",
                        "mean",
                    ),
                    green_repeat_mean=(
                        "green_repeat",
                        "mean",
                    ),
                    z_all_mean=(
                        "z_all",
                        "mean",
                    ),
                    green_all_mean=(
                        "green_all",
                        "mean",
                    ),
                    repeat_bigram_count_mean=(
                        "repeat_bigram_count",
                        "mean",
                    ),
                    repeat_bigram_rate_mean=(
                        "repeat_bigram_rate",
                        "mean",
                    ),
                )
            )


        def main():

            args = parse_args()

            old_root = Path(
                args.old_root
            )

            new_root = Path(
                args.new_root
            )

            out = Path(
                args.output_dir
            )

            out.mkdir(
                parents=True,
                exist_ok=True,
            )

            rng = np.random.default_rng(
                args.bootstrap_seed
            )

            df, missing = collect(
                old_root,
                new_root,
            )

            long_file = (
                out
                / "stage_sample_long.csv"
            )

            df.to_csv(
                long_file,
                index=False,
                encoding="utf-8-sig",
            )

            per_seed = per_seed_effects(
                df,
                rng,
                args.bootstrap,
            )

            per_seed_file = (
                out
                / "stage_paired_effects_per_seed.csv"
            )

            per_seed.to_csv(
                per_seed_file,
                index=False,
                encoding="utf-8-sig",
            )

            hierarchical = (
                hierarchical_bootstrap(
                    df,
                    rng,
                    args.bootstrap,
                )
            )

            hierarchical_file = (
                out
                / "stage_paired_effects_hierarchical.csv"
            )

            hierarchical.to_csv(
                hierarchical_file,
                index=False,
                encoding="utf-8-sig",
            )

            condition = condition_summary(
                df
            )

            condition_file = (
                out
                / "stage_condition_summary.csv"
            )

            condition.to_csv(
                condition_file,
                index=False,
                encoding="utf-8-sig",
            )

            missing_file = (
                out
                / "missing_detector_files.csv"
            )

            if len(missing):
                missing.to_csv(
                    missing_file,
                    index=False,
                    encoding="utf-8-sig",
                )
            else:
                pd.DataFrame(
                    columns=[
                        "model",
                        "seed",
                        "condition",
                        "expected_file",
                    ]
                ).to_csv(
                    missing_file,
                    index=False,
                    encoding="utf-8-sig",
                )

            print()
            print("Generated:")
            print(" ", long_file)
            print(" ", per_seed_file)
            print(" ", hierarchical_file)
            print(" ", condition_file)
            print(" ", missing_file)

            print()
            print(
                "Main repeat-aware z results:"
            )

            show = hierarchical[
                hierarchical["metric"]
                == "z_repeat"
            ][
                [
                    "model",
                    "contrast",
                    "mean",
                    "ci_low",
                    "ci_high",
                    "p_boot",
                    "n_seeds",
                    "total_paired_rows",
                ]
            ]

            print(
                show.to_string(
                    index=False
                )
            )

            if len(missing):
                print()
                print(
                    "WARNING: missing detector CSVs:"
                )
                print(
                    missing.to_string(
                        index=False
                    )
                )
        main()
    finally:
        sys.argv = old_argv

def _run_candidate_audits(argv):
    """Run preserved implementation from analyze_kgw_selection.py[candidate-audits]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_selection.py[candidate-audits]"] + list(argv)

        """
        Robust IASW-K32 candidate-audit summarizer.

        Fix:
        - safely handles JSON null / Python None values in:
            selected_local_green_fraction
            selected_kgw_delta_z
            candidate kgw_delta_z
            candidate local_green_fraction
            candidate ias_similarity
        - converts missing/non-finite values to NaN
        - never performs None - float arithmetic
        - keeps all other usable audit statistics

        Expected input:
        results/diagnostics/generation/kgw_selection/c4/<MODEL>/seed_20260823/
            candidate_audit/iasw_k32_candidate_audit.jsonl

        Outputs:
            candidate_audit_step_level.csv
            candidate_audit_sample_level.csv
            candidate_audit_model_summary.csv
            missing_candidate_audits.csv
        """


        import argparse
        import json
        import math
        from pathlib import Path

        import numpy as np
        import pandas as pd


        MODELS = [
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-3B-Instruct",
            "Llama-3.2-3B-Instruct",
        ]


        def parse_args():
            p = argparse.ArgumentParser()

            p.add_argument(
                "--root",
                default="results/diagnostics/generation/kgw_selection/c4",
            )

            p.add_argument(
                "--seed",
                default="20260823",
            )

            p.add_argument(
                "--output_dir",
                default="results/diagnostics/generation/kgw_selection/c4/candidate_audit_summary",
            )

            p.add_argument(
                "--bootstrap",
                type=int,
                default=10000,
            )

            p.add_argument(
                "--bootstrap_seed",
                type=int,
                default=20260827,
            )

            return p.parse_args()


        def safe_float(x):
            """
            Convert x to finite float; otherwise return NaN.
            Handles None, strings, numpy scalars, etc.
            """
            if x is None:
                return np.nan

            try:
                y = float(x)
            except (TypeError, ValueError):
                return np.nan

            if not math.isfinite(y):
                return np.nan

            return y


        def safe_nanmean(values):
            arr = np.asarray(
                [safe_float(x) for x in values],
                dtype=float,
            )

            finite = arr[
                np.isfinite(arr)
            ]

            if len(finite) == 0:
                return np.nan

            return float(
                np.mean(finite)
            )


        def safe_sub(a, b):
            a = safe_float(a)
            b = safe_float(b)

            if (
                not np.isfinite(a)
                or not np.isfinite(b)
            ):
                return np.nan

            return float(a - b)


        def pearson(x, y):
            x = np.asarray(
                [safe_float(v) for v in x],
                dtype=float,
            )

            y = np.asarray(
                [safe_float(v) for v in y],
                dtype=float,
            )

            ok = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            x = x[ok]
            y = y[ok]

            if len(x) < 3:
                return np.nan

            if (
                np.std(x) == 0
                or np.std(y) == 0
            ):
                return np.nan

            return float(
                np.corrcoef(
                    x,
                    y,
                )[0, 1]
            )


        def spearman(x, y):
            s = pd.DataFrame({
                "x": [
                    safe_float(v)
                    for v in x
                ],
                "y": [
                    safe_float(v)
                    for v in y
                ],
            }).dropna()

            if len(s) < 3:
                return np.nan

            rx = s["x"].rank(
                method="average"
            )

            ry = s["y"].rank(
                method="average"
            )

            return pearson(
                rx.to_numpy(),
                ry.to_numpy(),
            )


        def audit_path(
            root: Path,
            model: str,
            seed: str,
        ):
            return (
                root
                / model
                / f"seed_{seed}"
                / "candidate_audit"
                / "iasw_k32_candidate_audit.jsonl"
            )


        def read_audit(
            path: Path,
            model: str,
            seed: str,
        ):
            rows = []

            with path.open(
                "r",
                encoding="utf-8",
            ) as f:

                for line_no, line in enumerate(
                    f,
                    start=1,
                ):
                    if not line.strip():
                        continue

                    rec = json.loads(line)

                    candidates = rec.get(
                        "candidates",
                        [],
                    )

                    if not candidates:
                        continue

                    ias = [
                        safe_float(
                            c.get(
                                "ias_similarity"
                            )
                        )
                        for c in candidates
                    ]

                    kgw = [
                        safe_float(
                            c.get(
                                "kgw_delta_z"
                            )
                        )
                        for c in candidates
                    ]

                    gf = [
                        safe_float(
                            c.get(
                                "local_green_fraction"
                            )
                        )
                        for c in candidates
                    ]

                    selected_idx = rec.get(
                        "selected_idx"
                    )

                    selected_kgw = safe_float(
                        rec.get(
                            "selected_kgw_delta_z"
                        )
                    )

                    selected_gf = safe_float(
                        rec.get(
                            "selected_local_green_fraction"
                        )
                    )

                    # Fallback: if top-level selected fields are missing,
                    # recover them from the selected candidate record.
                    selected_candidate = None

                    for c in candidates:
                        if (
                            c.get(
                                "candidate_idx"
                            )
                            == selected_idx
                        ):
                            selected_candidate = c
                            break

                    if selected_candidate is not None:

                        if not np.isfinite(
                            selected_kgw
                        ):
                            selected_kgw = safe_float(
                                selected_candidate.get(
                                    "kgw_delta_z"
                                )
                            )

                        if not np.isfinite(
                            selected_gf
                        ):
                            selected_gf = safe_float(
                                selected_candidate.get(
                                    "local_green_fraction"
                                )
                            )

                    pool_kgw_mean = safe_nanmean(
                        kgw
                    )

                    pool_gf_mean = safe_nanmean(
                        gf
                    )

                    rows.append({
                        "model":
                            model,

                        "seed":
                            str(seed),

                        "source_file":
                            str(path),

                        "line_no":
                            line_no,

                        "sample_idx":
                            rec.get(
                                "sample_idx"
                            ),

                        "sentence_step":
                            rec.get(
                                "sentence_step"
                            ),

                        "num_candidates":
                            len(candidates),

                        "selected_idx":
                            selected_idx,

                        "selected_kgw_percentile":
                            safe_float(
                                rec.get(
                                    "selected_kgw_percentile"
                                )
                            ),

                        "selected_green_percentile":
                            safe_float(
                                rec.get(
                                    "selected_green_percentile"
                                )
                            ),

                        "selected_ias_similarity":
                            safe_float(
                                rec.get(
                                    "selected_ias_similarity"
                                )
                            ),

                        "selected_kgw_delta_z":
                            selected_kgw,

                        "selected_local_green_fraction":
                            selected_gf,

                        "pool_kgw_delta_z_mean":
                            pool_kgw_mean,

                        "pool_local_green_fraction_mean":
                            pool_gf_mean,

                        "selected_minus_pool_kgw":
                            safe_sub(
                                selected_kgw,
                                pool_kgw_mean,
                            ),

                        "selected_minus_pool_green_fraction":
                            safe_sub(
                                selected_gf,
                                pool_gf_mean,
                            ),

                        "pearson_ias_vs_kgw":
                            pearson(
                                ias,
                                kgw,
                            ),

                        "spearman_ias_vs_kgw":
                            spearman(
                                ias,
                                kgw,
                            ),

                        "pearson_ias_vs_green":
                            pearson(
                                ias,
                                gf,
                            ),

                        "spearman_ias_vs_green":
                            spearman(
                                ias,
                                gf,
                            ),

                        # Diagnostics for how much audit information is present.
                        "n_valid_ias":
                            int(
                                np.sum(
                                    np.isfinite(
                                        np.asarray(
                                            ias,
                                            dtype=float,
                                        )
                                    )
                                )
                            ),

                        "n_valid_kgw":
                            int(
                                np.sum(
                                    np.isfinite(
                                        np.asarray(
                                            kgw,
                                            dtype=float,
                                        )
                                    )
                                )
                            ),

                        "n_valid_green":
                            int(
                                np.sum(
                                    np.isfinite(
                                        np.asarray(
                                            gf,
                                            dtype=float,
                                        )
                                    )
                                )
                            ),
                    })

            return pd.DataFrame(
                rows
            )


        def bootstrap_samples(
            sample_df: pd.DataFrame,
            column: str,
            rng,
            n_boot,
        ):
            vals = pd.to_numeric(
                sample_df[column],
                errors="coerce",
            ).dropna().to_numpy(
                dtype=float
            )

            vals = vals[
                np.isfinite(vals)
            ]

            if len(vals) == 0:
                return (
                    np.nan,
                    np.nan,
                    np.nan,
                    0,
                )

            point = float(
                np.mean(vals)
            )

            idx = rng.integers(
                0,
                len(vals),
                size=(
                    n_boot,
                    len(vals),
                ),
            )

            boot = vals[idx].mean(
                axis=1
            )

            lo, hi = np.percentile(
                boot,
                [2.5, 97.5],
            )

            return (
                point,
                float(lo),
                float(hi),
                int(len(vals)),
            )


        def main():
            args = parse_args()

            root = Path(
                args.root
            )

            out = Path(
                args.output_dir
            )

            out.mkdir(
                parents=True,
                exist_ok=True,
            )

            rng = np.random.default_rng(
                args.bootstrap_seed
            )

            frames = []
            missing = []

            for model in MODELS:

                path = audit_path(
                    root,
                    model,
                    args.seed,
                )

                if not path.is_file():
                    missing.append({
                        "model":
                            model,

                        "seed":
                            str(args.seed),

                        "expected_file":
                            str(path),
                    })
                    continue

                x = read_audit(
                    path,
                    model,
                    args.seed,
                )

                if len(x):
                    frames.append(x)

            if not frames:
                raise RuntimeError(
                    "No IASW-K32 candidate audit files found."
                )

            step = pd.concat(
                frames,
                ignore_index=True,
            )

            step_file = (
                out
                / "candidate_audit_step_level.csv"
            )

            step.to_csv(
                step_file,
                index=False,
                encoding="utf-8-sig",
            )

            # ----------------------------------------------------------
            # Aggregate sentence steps within each generated sample first
            # so the bootstrap unit is generated sample, not sentence step.
            # ----------------------------------------------------------

            agg_cols = {
                "sentence_step": "count",

                "selected_kgw_percentile":
                    "mean",

                "selected_green_percentile":
                    "mean",

                "selected_minus_pool_kgw":
                    "mean",

                "selected_minus_pool_green_fraction":
                    "mean",

                "pearson_ias_vs_kgw":
                    "mean",

                "spearman_ias_vs_kgw":
                    "mean",

                "pearson_ias_vs_green":
                    "mean",

                "spearman_ias_vs_green":
                    "mean",

                "n_valid_ias":
                    "sum",

                "n_valid_kgw":
                    "sum",

                "n_valid_green":
                    "sum",
            }

            sample = (
                step
                .groupby(
                    [
                        "model",
                        "seed",
                        "sample_idx",
                    ],
                    as_index=False,
                )
                .agg(
                    agg_cols
                )
                .rename(
                    columns={
                        "sentence_step":
                            "n_sentence_steps",
                    }
                )
            )

            sample_file = (
                out
                / "candidate_audit_sample_level.csv"
            )

            sample.to_csv(
                sample_file,
                index=False,
                encoding="utf-8-sig",
            )

            columns = [
                "selected_kgw_percentile",
                "selected_green_percentile",
                "selected_minus_pool_kgw",
                "selected_minus_pool_green_fraction",
                "pearson_ias_vs_kgw",
                "spearman_ias_vs_kgw",
                "pearson_ias_vs_green",
                "spearman_ias_vs_green",
            ]

            summary_rows = []

            for model in MODELS:

                sub = sample[
                    sample["model"]
                    == model
                ]

                if len(sub) == 0:
                    continue

                for column in columns:

                    mean, lo, hi, n = (
                        bootstrap_samples(
                            sub,
                            column,
                            rng,
                            args.bootstrap,
                        )
                    )

                    summary_rows.append({
                        "model":
                            model,

                        "metric":
                            column,

                        "n_samples":
                            n,

                        "mean":
                            mean,

                        "ci_low":
                            lo,

                        "ci_high":
                            hi,
                    })

            summary = pd.DataFrame(
                summary_rows
            )

            summary_file = (
                out
                / "candidate_audit_model_summary.csv"
            )

            summary.to_csv(
                summary_file,
                index=False,
                encoding="utf-8-sig",
            )

            # ----------------------------------------------------------
            # Additional completeness diagnostics
            # ----------------------------------------------------------

            completeness = (
                step
                .groupby(
                    "model",
                    as_index=False,
                )
                .agg(
                    n_steps=(
                        "sentence_step",
                        "count",
                    ),

                    n_samples=(
                        "sample_idx",
                        "nunique",
                    ),

                    selected_kgw_available=(
                        "selected_kgw_delta_z",
                        lambda x: int(
                            pd.to_numeric(
                                x,
                                errors="coerce",
                            )
                            .notna()
                            .sum()
                        ),
                    ),

                    selected_green_available=(
                        "selected_local_green_fraction",
                        lambda x: int(
                            pd.to_numeric(
                                x,
                                errors="coerce",
                            )
                            .notna()
                            .sum()
                        ),
                    ),
                )
            )

            completeness_file = (
                out
                / "candidate_audit_completeness.csv"
            )

            completeness.to_csv(
                completeness_file,
                index=False,
                encoding="utf-8-sig",
            )

            missing_file = (
                out
                / "missing_candidate_audits.csv"
            )

            if missing:
                pd.DataFrame(
                    missing
                ).to_csv(
                    missing_file,
                    index=False,
                    encoding="utf-8-sig",
                )
            else:
                pd.DataFrame(
                    columns=[
                        "model",
                        "seed",
                        "expected_file",
                    ]
                ).to_csv(
                    missing_file,
                    index=False,
                    encoding="utf-8-sig",
                )

            print()
            print("Generated:")
            print(" ", step_file)
            print(" ", sample_file)
            print(" ", summary_file)
            print(" ", completeness_file)
            print(" ", missing_file)

            print()
            print(
                "Main candidate-level mechanism results:"
            )

            show_metrics = [
                "selected_kgw_percentile",
                "selected_minus_pool_kgw",
                "pearson_ias_vs_kgw",
                "spearman_ias_vs_kgw",
            ]

            show = summary[
                summary["metric"].isin(
                    show_metrics
                )
            ]

            print(
                show.to_string(
                    index=False
                )
            )

            print()
            print(
                "Audit completeness:"
            )

            print(
                completeness.to_string(
                    index=False
                )
            )

            if missing:
                print()
                print(
                    "Missing candidate audits:"
                )

                print(
                    pd.DataFrame(
                        missing
                    ).to_string(
                        index=False
                    )
                )
        main()
    finally:
        sys.argv = old_argv

def _run_selection_ablation(argv):
    """Run preserved implementation from analyze_kgw_selection.py[selection-ablation]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_selection.py[selection-ablation]"] + list(argv)

        """
        Merge:
          1) Existing KGW + Random-K1 five-seed sweep
          2) New Random-K32 + IASW-K32 selection ablation

        and compute:

          Δ_sentence = Random-K1  - KGW
          Δ_pool     = Random-K32 - Random-K1
          Δ_select   = IASW-K32   - Random-K32
          Δ_total    = IASW-K32   - KGW

        for both:
          - z without repeated-bigram filtering
          - z with repeated-bigram filtering
          - green fraction
          - effective/scored token count

        Outputs:
          selection_ablation_per_seed.csv
          selection_ablation_backbone_summary.csv
          selection_ablation_long.csv
          missing_cells.csv

        This script only summarizes mean values from each 100-sample detector summary.
        If sample-level CSVs are available later, paired bootstrap can be added.
        """


        import argparse
        import json
        import math
        from pathlib import Path
        from typing import Dict, Any, List

        import numpy as np
        import pandas as pd


        def parse_args():
            p = argparse.ArgumentParser()

            p.add_argument(
                "--seed_root",
                default="results/diagnostics/generation/kgw_randomk1_seed_sweep/c4",
                help="Existing KGW + Random-K1 five-seed results.",
            )

            p.add_argument(
                "--selection_root",
                default="results/diagnostics/generation/kgw_selection/c4",
                help="New Random-K32 + IASW-K32 results.",
            )

            p.add_argument(
                "--output_dir",
                default="results/diagnostics/generation/kgw_selection/c4/summary",
            )

            p.add_argument("--candidate_k", type=int, default=32)

            return p.parse_args()


        def load_json(path: Path) -> Dict[str, Any]:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)


        def section_mean(d: Dict[str, Any], key: str):
            x = d.get(key)
            if not isinstance(x, dict):
                return np.nan
            try:
                y = float(x.get("mean"))
            except Exception:
                return np.nan
            return y if np.isfinite(y) else np.nan


        def extract(path: Path) -> Dict[str, Any]:
            d = load_json(path)

            return {
                "n": d.get("num_detected_records", np.nan),

                "z_raw": section_mean(
                    d,
                    "z_no_repeat_filter",
                ),

                "z_repeat": section_mean(
                    d,
                    "z_repeat_filter",
                ),

                "green_raw": section_mean(
                    d,
                    "green_fraction_no_repeat_filter",
                ),

                "green_repeat": section_mean(
                    d,
                    "green_fraction_repeat_filter",
                ),

                "tokens_raw": section_mean(
                    d,
                    "num_tokens_no_repeat_filter",
                ),

                "tokens_repeat": section_mean(
                    d,
                    "num_tokens_repeat_filter",
                ),
            }


        def infer_model_seed(path: Path):
            seed = None
            model = None

            parts = list(path.parts)

            for i, part in enumerate(parts):
                if part.startswith("seed_"):
                    seed = part.replace("seed_", "", 1)
                    if i >= 1:
                        model = parts[i - 1]
                    break

            return model, seed


        def collect_existing(root: Path) -> List[Dict[str, Any]]:
            rows = []

            names = {
                "kgw_control.kgw_detection_summary.json": "KGW",
                "dual_kgw_random_k1.kgw_detection_summary.json": "Random-K1",
            }

            for file in root.rglob("*.kgw_detection_summary.json"):
                method = names.get(file.name)
                if method is None:
                    continue

                model, seed = infer_model_seed(file)
                if model is None or seed is None:
                    continue

                row = {
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "source_root": "seed_sweep",
                    "file": str(file),
                }
                row.update(extract(file))
                rows.append(row)

            return rows


        def collect_selection(root: Path, k: int) -> List[Dict[str, Any]]:
            rows = []

            names = {
                f"random_k{k}.kgw_detection_summary.json": f"Random-K{k}",
                f"iasw_k{k}.kgw_detection_summary.json": f"IASW-K{k}",
            }

            for file in root.rglob("*.kgw_detection_summary.json"):
                method = names.get(file.name)
                if method is None:
                    continue

                model, seed = infer_model_seed(file)
                if model is None or seed is None:
                    continue

                row = {
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "source_root": "selection_ablation",
                    "file": str(file),
                }
                row.update(extract(file))
                rows.append(row)

            return rows


        def wide_table(df: pd.DataFrame, k: int):
            metrics = [
                "z_raw",
                "z_repeat",
                "green_raw",
                "green_repeat",
                "tokens_raw",
                "tokens_repeat",
            ]

            pieces = []

            for metric in metrics:
                x = df.pivot_table(
                    index=["model", "seed"],
                    columns="method",
                    values=metric,
                    aggfunc="first",
                ).reset_index()

                x.columns.name = None

                rename = {
                    "KGW": f"{metric}__KGW",
                    "Random-K1": f"{metric}__RandomK1",
                    f"Random-K{k}": f"{metric}__RandomK{k}",
                    f"IASW-K{k}": f"{metric}__IASWK{k}",
                }
                x = x.rename(columns=rename)

                pieces.append(x)

            out = pieces[0]

            for x in pieces[1:]:
                out = out.merge(
                    x,
                    on=["model", "seed"],
                    how="outer",
                )

            # Main controlled contrasts.
            for metric in metrics:
                kgw = f"{metric}__KGW"
                k1 = f"{metric}__RandomK1"
                rk = f"{metric}__RandomK{k}"
                iw = f"{metric}__IASWK{k}"

                if kgw in out and k1 in out:
                    out[f"{metric}__delta_sentence_K1_minus_KGW"] = out[k1] - out[kgw]

                if k1 in out and rk in out:
                    out[f"{metric}__delta_pool_K{k}_minus_K1"] = out[rk] - out[k1]

                if rk in out and iw in out:
                    out[f"{metric}__delta_selection_IASW_minus_RandomK{k}"] = out[iw] - out[rk]

                if kgw in out and iw in out:
                    out[f"{metric}__delta_total_IASW_minus_KGW"] = out[iw] - out[kgw]

            return out.sort_values(["model", "seed"])


        def backbone_summary(wide: pd.DataFrame, k: int):
            delta_suffixes = [
                "delta_sentence_K1_minus_KGW",
                f"delta_pool_K{k}_minus_K1",
                f"delta_selection_IASW_minus_RandomK{k}",
                "delta_total_IASW_minus_KGW",
            ]

            metrics = [
                "z_raw",
                "z_repeat",
                "green_raw",
                "green_repeat",
                "tokens_raw",
                "tokens_repeat",
            ]

            rows = []

            for model, sub in wide.groupby("model"):
                row = {"model": model}

                for metric in metrics:
                    for suffix in delta_suffixes:
                        col = f"{metric}__{suffix}"

                        if col not in sub.columns:
                            continue

                        vals = pd.to_numeric(
                            sub[col],
                            errors="coerce",
                        ).dropna()

                        key = f"{metric}__{suffix}"

                        row[f"{key}__n_seeds"] = len(vals)
                        row[f"{key}__mean"] = vals.mean() if len(vals) else np.nan
                        row[f"{key}__std"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
                        row[f"{key}__min"] = vals.min() if len(vals) else np.nan
                        row[f"{key}__max"] = vals.max() if len(vals) else np.nan

                rows.append(row)

            return pd.DataFrame(rows)


        def missing_cells(df: pd.DataFrame, k: int):
            expected_methods = [
                "KGW",
                "Random-K1",
                f"Random-K{k}",
                f"IASW-K{k}",
            ]

            rows = []

            combos = (
                df[["model", "seed"]]
                .drop_duplicates()
                .sort_values(["model", "seed"])
            )

            for _, rec in combos.iterrows():
                model = rec["model"]
                seed = rec["seed"]

                present = set(
                    df[
                        (df["model"] == model)
                        & (df["seed"] == seed)
                    ]["method"]
                )

                for method in expected_methods:
                    if method not in present:
                        rows.append({
                            "model": model,
                            "seed": seed,
                            "missing_method": method,
                        })

            return pd.DataFrame(rows)


        def main():
            args = parse_args()

            seed_root = Path(args.seed_root)
            selection_root = Path(args.selection_root)
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            rows = []
            rows.extend(collect_existing(seed_root))
            rows.extend(collect_selection(selection_root, args.candidate_k))

            if not rows:
                raise RuntimeError("No detector summary files found.")

            df = pd.DataFrame(rows).sort_values(
                ["model", "seed", "method"]
            )

            long_file = out / "selection_ablation_long.csv"
            df.to_csv(
                long_file,
                index=False,
                encoding="utf-8-sig",
            )

            wide = wide_table(
                df,
                args.candidate_k,
            )

            per_seed_file = out / "selection_ablation_per_seed.csv"
            wide.to_csv(
                per_seed_file,
                index=False,
                encoding="utf-8-sig",
            )

            bb = backbone_summary(
                wide,
                args.candidate_k,
            )

            backbone_file = out / "selection_ablation_backbone_summary.csv"
            bb.to_csv(
                backbone_file,
                index=False,
                encoding="utf-8-sig",
            )

            miss = missing_cells(
                df,
                args.candidate_k,
            )

            missing_file = out / "missing_cells.csv"
            if len(miss):
                miss.to_csv(
                    missing_file,
                    index=False,
                    encoding="utf-8-sig",
                )
            else:
                pd.DataFrame(
                    columns=[
                        "model",
                        "seed",
                        "missing_method",
                    ]
                ).to_csv(
                    missing_file,
                    index=False,
                    encoding="utf-8-sig",
                )

            print("\nGenerated:")
            print(" ", long_file)
            print(" ", per_seed_file)
            print(" ", backbone_file)
            print(" ", missing_file)

            print("\nMost important columns:")
            important = [
                "model",
                "seed",
                "z_repeat__KGW",
                "z_repeat__RandomK1",
                f"z_repeat__RandomK{args.candidate_k}",
                f"z_repeat__IASWK{args.candidate_k}",
                "z_repeat__delta_sentence_K1_minus_KGW",
                f"z_repeat__delta_pool_K{args.candidate_k}_minus_K1",
                f"z_repeat__delta_selection_IASW_minus_RandomK{args.candidate_k}",
                "z_repeat__delta_total_IASW_minus_KGW",
            ]

            important = [
                x for x in important
                if x in wide.columns
            ]

            print(
                wide[important].to_string(index=False)
            )

            if len(miss):
                print("\nWARNING: missing experiment cells:")
                print(miss.to_string(index=False))
            else:
                print("\nAll expected cells are present.")
        main()
    finally:
        sys.argv = old_argv

COMMANDS = {
    "paired-bootstrap": _run_paired_bootstrap,
    "candidate-audits": _run_candidate_audits,
    "selection-ablation": _run_selection_ablation,
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage:\n"
            "  analyze_kgw_selection.py paired-bootstrap <old args>\n"
            "  analyze_kgw_selection.py candidate-audits <old args>\n"
            "  analyze_kgw_selection.py selection-ablation <old args>\n"
            "\nEach subcommand preserves the corresponding old CLI."
        )
        return
    command = sys.argv[1]
    fn = COMMANDS.get(command)
    if fn is None:
        raise SystemExit(f"Unknown subcommand: {command}. Choose from: {', '.join(COMMANDS)}")
    fn(sys.argv[2:])

if __name__ == "__main__":
    main()
