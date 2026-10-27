# -*- coding: utf-8 -*-
"""
Analyze IAS-W candidate-pool audit.

Unit of causally clean comparison:
    all K candidates generated under the SAME current prefix.

Primary questions
-----------------
1. Is the IAS-W-selected candidate above random expectation in KGW evidence?
2. Is IAS similarity positively associated with KGW incremental z within pools?
3. Can the KGW gain be explained only by longer candidates / more scored tokens?
4. Does the effect change with sentence step?

Bootstrap is clustered at sample level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_corr(x, y, kind="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan
    if np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    try:
        if kind == "pearson":
            return float(pearsonr(x[m], y[m]).statistic)
        return float(spearmanr(x[m], y[m]).statistic)
    except Exception:
        return np.nan


def num(x):
    """Convert None/non-finite values to np.nan."""
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def nanmean_safe(x):
    """np.nanmean without warnings when the whole array is missing."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def safe_diff(a, b):
    a = num(a)
    b = num(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(a - b)


def flatten(records):
    pool_rows = []
    candidate_rows = []

    for rec in records:
        candidates = rec.get("candidates", []) or []
        if not candidates:
            continue

        selected_idx = int(rec.get("selected_idx", -1))
        if not (0 <= selected_idx < len(candidates)):
            continue

        sel = candidates[selected_idx]

        def arr(key):
            return np.asarray(
                [num(c.get(key)) for c in candidates],
                dtype=float,
            )

        ias = arr("ias_similarity")
        dz = arr("kgw_delta_z")
        gf = arr("local_green_fraction")
        clen = arr("candidate_tokens")
        dn = arr("delta_tokens_scored")
        dg = arr("delta_green_tokens")

        mean_dz = nanmean_safe(dz)
        mean_gf = nanmean_safe(gf)
        mean_clen = nanmean_safe(clen)
        mean_dn = nanmean_safe(dn)
        mean_dg = nanmean_safe(dg)

        sel_dz = num(sel.get("kgw_delta_z"))
        sel_gf = num(sel.get("local_green_fraction"))
        sel_clen = num(sel.get("candidate_tokens"))
        sel_dn = num(sel.get("delta_tokens_scored"))
        sel_dg = num(sel.get("delta_green_tokens"))

        pool_rows.append({
            "sample_idx": int(rec.get("sample_idx", -1)),
            "sentence_step": int(rec.get("sentence_step", -1)),
            "anchor_index": int(rec.get("anchor_index", -1)),
            "n_candidates": len(candidates),

            "selected_kgw_delta_z": sel_dz,
            "pool_mean_kgw_delta_z": mean_dz,
            "selected_minus_pool_kgw_delta_z":
                safe_diff(sel_dz, mean_dz),

            "selected_green_fraction": sel_gf,
            "pool_mean_green_fraction": mean_gf,
            "selected_minus_pool_green_fraction":
                safe_diff(sel_gf, mean_gf),

            "selected_candidate_tokens": sel_clen,
            "pool_mean_candidate_tokens": mean_clen,
            "selected_minus_pool_candidate_tokens":
                safe_diff(sel_clen, mean_clen),

            "selected_tokens_scored": sel_dn,
            "pool_mean_tokens_scored": mean_dn,
            "selected_minus_pool_tokens_scored":
                safe_diff(sel_dn, mean_dn),

            "selected_green_tokens": sel_dg,
            "pool_mean_green_tokens": mean_dg,
            "selected_minus_pool_green_tokens":
                safe_diff(sel_dg, mean_dg),

            "selected_kgw_percentile": num(
                rec.get("selected_kgw_percentile")
            ),
            "selected_green_percentile": num(
                rec.get("selected_green_percentile")
            ),

            "spearman_ias_vs_kgw_delta_z":
                safe_corr(ias, dz, "spearman"),
            "pearson_ias_vs_kgw_delta_z":
                safe_corr(ias, dz, "pearson"),
            "spearman_ias_vs_green_fraction":
                safe_corr(ias, gf, "spearman"),
            "spearman_ias_vs_candidate_tokens":
                safe_corr(ias, clen, "spearman"),
            "spearman_ias_vs_tokens_scored":
                safe_corr(ias, dn, "spearman"),

            "prefix_generated_tokens": num(
                rec.get("prefix_generated_tokens")
            ),
            "prefix_kgw_z": num(rec.get("prefix_kgw_z")),
        })

        for c in candidates:
            candidate_rows.append({
                "sample_idx": int(rec.get("sample_idx", -1)),
                "sentence_step": int(rec.get("sentence_step", -1)),
                "candidate_idx": int(c.get("candidate_idx", -1)),
                "selected": int(c.get("candidate_idx", -1)) == selected_idx,
                "ias_similarity": num(c.get("ias_similarity")),
                "kgw_delta_z": num(c.get("kgw_delta_z")),
                "local_green_fraction": num(c.get("local_green_fraction")),
                "candidate_tokens": num(c.get("candidate_tokens")),
                "delta_tokens_scored": num(c.get("delta_tokens_scored")),
                "delta_green_tokens": num(c.get("delta_green_tokens")),
            })

    return pd.DataFrame(pool_rows), pd.DataFrame(candidate_rows)


def cluster_bootstrap_sample_means(pool_df, metrics, n_boot, seed):
    """
    Resample sample IDs with replacement, preserving all sentence-step pools
    from each selected sample.
    """
    samples = np.asarray(sorted(pool_df["sample_idx"].unique()))
    rng = np.random.default_rng(seed)

    observed = {
        m: float(pool_df[m].mean())
        for m in metrics
    }

    boot_values = {m: [] for m in metrics}

    grouped = {
        sid: pool_df[pool_df["sample_idx"] == sid]
        for sid in samples
    }

    for _ in range(n_boot):
        chosen = rng.choice(samples, size=len(samples), replace=True)
        pieces = [grouped[sid] for sid in chosen]
        b = pd.concat(pieces, ignore_index=True)

        for m in metrics:
            boot_values[m].append(float(b[m].mean()))

    rows = []
    for m in metrics:
        vals = np.asarray(boot_values[m], dtype=float)
        rows.append({
            "metric": m,
            "observed_mean": observed[m],
            "ci_low": float(np.nanquantile(vals, 0.025)),
            "ci_high": float(np.nanquantile(vals, 0.975)),
        })

    return pd.DataFrame(rows)


def pool_centered_regression(candidate_df, y_col):
    """
    Fixed-effect-like regression using within-pool centered variables:

      y_centered
        ~ IAS_similarity_centered
        + candidate_tokens_centered
        + delta_tokens_scored_centered

    This asks whether IAS similarity predicts KGW evidence beyond simple
    candidate length / number of scored tokens within the SAME pool.
    """
    df = candidate_df.copy()
    pool_keys = ["sample_idx", "sentence_step"]

    needed = [
        y_col,
        "ias_similarity",
        "candidate_tokens",
        "delta_tokens_scored",
    ]
    df = df.dropna(subset=needed).copy()

    if len(df) < 20:
        return {
            "n": len(df),
            "coef_ias": np.nan,
            "coef_candidate_tokens": np.nan,
            "coef_tokens_scored": np.nan,
            "r2": np.nan,
        }

    for c in needed:
        df[c + "_c"] = (
            df[c]
            - df.groupby(pool_keys)[c].transform("mean")
        )

    X = df[
        [
            "ias_similarity_c",
            "candidate_tokens_c",
            "delta_tokens_scored_c",
        ]
    ].to_numpy(dtype=float)
    y = df[y_col + "_c"].to_numpy(dtype=float)

    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[m]
    y = y[m]

    if len(y) < 20:
        return {
            "n": len(y),
            "coef_ias": np.nan,
            "coef_candidate_tokens": np.nan,
            "coef_tokens_scored": np.nan,
            "r2": np.nan,
        }

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum(y ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    return {
        "n": len(y),
        "coef_ias": float(beta[0]),
        "coef_candidate_tokens": float(beta[1]),
        "coef_tokens_scored": float(beta[2]),
        "r2": r2,
    }


def compare_reference(audit_generated_path, reference_path):
    if not audit_generated_path or not reference_path:
        return None

    def load(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                sid = str(x.get("idx", x.get("id", i)))
                rows.append((sid, str(x.get("wm_text", ""))))
        return dict(rows)

    a = load(audit_generated_path)
    b = load(reference_path)

    common = sorted(set(a) & set(b))
    if not common:
        return {
            "common_n": 0,
            "exact_match_n": 0,
            "exact_match_rate": np.nan,
        }

    eq = [a[k] == b[k] for k in common]
    return {
        "common_n": len(common),
        "exact_match_n": int(sum(eq)),
        "exact_match_rate": float(np.mean(eq)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--audit_generated_file", default="")
    ap.add_argument("--reference_file", default="")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.audit_file)
    pool_df, candidate_df = flatten(records)

    if pool_df.empty:
        raise RuntimeError("No candidate pools were loaded.")

    pool_df.to_csv(
        out / "candidate_pool_level.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_df.to_csv(
        out / "candidate_level.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Missingness is expected for a small number of very short/unscorable
    # candidates. Report it explicitly rather than crashing or silently
    # converting it into zero evidence.
    missing_rows = []
    for col in [
        "ias_similarity",
        "kgw_delta_z",
        "local_green_fraction",
        "candidate_tokens",
        "delta_tokens_scored",
        "delta_green_tokens",
    ]:
        n_total = len(candidate_df)
        n_valid = int(candidate_df[col].notna().sum())
        missing_rows.append({
            "field": col,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_missing": n_total - n_valid,
            "missing_rate": (
                float((n_total - n_valid) / n_total)
                if n_total else np.nan
            ),
        })

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(
        out / "candidate_missingness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = [
        "selected_kgw_percentile",
        "selected_green_percentile",
        "selected_minus_pool_kgw_delta_z",
        "selected_minus_pool_green_fraction",
        "selected_minus_pool_candidate_tokens",
        "selected_minus_pool_tokens_scored",
        "spearman_ias_vs_kgw_delta_z",
        "spearman_ias_vs_green_fraction",
        "spearman_ias_vs_candidate_tokens",
        "spearman_ias_vs_tokens_scored",
    ]

    boot = cluster_bootstrap_sample_means(
        pool_df,
        metrics=metrics,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    boot.to_csv(
        out / "candidate_cluster_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_step = (
        pool_df.groupby("sentence_step", as_index=False)
        .agg(
            n_pools=("sample_idx", "count"),
            kgw_percentile=("selected_kgw_percentile", "mean"),
            green_percentile=("selected_green_percentile", "mean"),
            selected_minus_pool_kgw_delta_z=(
                "selected_minus_pool_kgw_delta_z", "mean"
            ),
            selected_minus_pool_green_fraction=(
                "selected_minus_pool_green_fraction", "mean"
            ),
            selected_minus_pool_candidate_tokens=(
                "selected_minus_pool_candidate_tokens", "mean"
            ),
            selected_minus_pool_tokens_scored=(
                "selected_minus_pool_tokens_scored", "mean"
            ),
            spearman_ias_vs_kgw_delta_z=(
                "spearman_ias_vs_kgw_delta_z", "mean"
            ),
        )
    )
    by_step.to_csv(
        out / "candidate_by_sentence_step.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reg_z = pool_centered_regression(candidate_df, "kgw_delta_z")
    reg_green = pool_centered_regression(
        candidate_df,
        "local_green_fraction",
    )
    reg_df = pd.DataFrame([
        {"outcome": "kgw_delta_z", **reg_z},
        {"outcome": "local_green_fraction", **reg_green},
    ])
    reg_df.to_csv(
        out / "candidate_pool_centered_regression.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reference = compare_reference(
        args.audit_generated_file,
        args.reference_file,
    )
    if reference is not None:
        pd.DataFrame([reference]).to_csv(
            out / "audit_reproduction_check.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Human-readable diagnostic.
    summary = {
        r["metric"]: r
        for _, r in boot.iterrows()
    }

    lines = [
        "# IAS-W candidate-level mechanism audit",
        "",
        f"Samples: {pool_df['sample_idx'].nunique()}",
        f"Candidate pools: {len(pool_df)}",
        f"Candidate observations: {len(candidate_df)}",
        "",
        "## Missingness",
        "",
    ]

    for _, r in missing_df.iterrows():
        lines.append(
            f"- {r['field']}: missing {int(r['n_missing'])}/"
            f"{int(r['n_total'])} ({100*r['missing_rate']:.2f}%)"
        )

    lines.append("")

    for key in [
        "selected_kgw_percentile",
        "selected_minus_pool_kgw_delta_z",
        "selected_minus_pool_green_fraction",
        "selected_minus_pool_candidate_tokens",
        "selected_minus_pool_tokens_scored",
        "spearman_ias_vs_kgw_delta_z",
    ]:
        r = summary[key]
        lines.append(
            f"- {key}: {r['observed_mean']:.4f} "
            f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        )

    lines.extend([
        "",
        "## Pool-centered regression",
        "",
        f"- KGW Δz outcome: IAS coefficient={reg_z['coef_ias']:.4f}, "
        f"length coefficient={reg_z['coef_candidate_tokens']:.4f}, "
        f"scored-token coefficient={reg_z['coef_tokens_scored']:.4f}, "
        f"R2={reg_z['r2']:.4f}",
    ])

    if reference is not None:
        lines.extend([
            "",
            "## Reproduction check",
            "",
            f"- Exact wm_text match rate: {reference['exact_match_rate']:.4f} "
            f"({reference['exact_match_n']}/{reference['common_n']})",
        ])

    (out / "candidate_diagnostic_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(boot.to_string(index=False))
    print("\nRegression:")
    print(reg_df.to_string(index=False))
    if reference is not None:
        print("\nReproduction:", reference)

    print("\n[DONE]")
    print(out / "candidate_diagnostic_report.md")


if __name__ == "__main__":
    main()