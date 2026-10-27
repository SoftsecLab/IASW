# -*- coding: utf-8 -*-
"""
Analyze candidate-level audit JSONL produced by
iasw_generation_ablation_runner.py.

Outputs
-------
candidate_pool_overall_summary.csv
candidate_pool_by_step.csv
candidate_pool_selected_vs_pool.csv
candidate_pool_correlations.csv
candidate_pool_diagnostics.md
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
    if kind == "pearson":
        return float(pearsonr(x[m], y[m]).statistic)
    return float(spearmanr(x[m], y[m]).statistic)


def flatten(records):
    pool_rows = []
    cand_rows = []

    for r in records:
        candidates = r.get("candidates", []) or []
        if not candidates:
            continue

        selected_idx = int(r.get("selected_idx", -1))

        dz = np.asarray(
            [c.get("kgw_delta_z", np.nan) for c in candidates],
            dtype=float,
        )
        gf = np.asarray(
            [c.get("local_green_fraction", np.nan) for c in candidates],
            dtype=float,
        )
        lens = np.asarray(
            [c.get("candidate_tokens", np.nan) for c in candidates],
            dtype=float,
        )
        ias = np.asarray(
            [c.get("ias_similarity", np.nan) for c in candidates],
            dtype=float,
        )

        sel = candidates[selected_idx] if 0 <= selected_idx < len(candidates) else {}

        pool_rows.append({
            "sample_idx": r.get("sample_idx"),
            "sentence_step": r.get("sentence_step"),
            "anchor_index": r.get("anchor_index"),
            "selection_policy": r.get("selection_policy"),
            "n_candidates": len(candidates),

            "selected_kgw_delta_z": sel.get("kgw_delta_z", np.nan),
            "pool_mean_kgw_delta_z": np.nanmean(dz),
            "selected_minus_pool_kgw_delta_z":
                sel.get("kgw_delta_z", np.nan) - np.nanmean(dz),

            "selected_green_fraction": sel.get("local_green_fraction", np.nan),
            "pool_mean_green_fraction": np.nanmean(gf),
            "selected_minus_pool_green_fraction":
                sel.get("local_green_fraction", np.nan) - np.nanmean(gf),

            "selected_tokens": sel.get("candidate_tokens", np.nan),
            "pool_mean_tokens": np.nanmean(lens),
            "selected_minus_pool_tokens":
                sel.get("candidate_tokens", np.nan) - np.nanmean(lens),

            "selected_ias_similarity": sel.get("ias_similarity", np.nan),
            "pool_mean_ias_similarity": np.nanmean(ias),
            "selected_minus_pool_ias_similarity":
                sel.get("ias_similarity", np.nan) - np.nanmean(ias),

            "selected_kgw_rank": sel.get("kgw_delta_z_rank", np.nan),
            "selected_kgw_percentile": r.get("selected_kgw_percentile", np.nan),
            "selected_green_percentile": r.get("selected_green_percentile", np.nan),

            "pool_spearman_ias_vs_kgw_delta_z":
                safe_corr(ias, dz, "spearman"),
            "pool_pearson_ias_vs_kgw_delta_z":
                safe_corr(ias, dz, "pearson"),
            "pool_spearman_ias_vs_green_fraction":
                safe_corr(ias, gf, "spearman"),

            "prefix_generated_tokens": r.get("prefix_generated_tokens", np.nan),
            "prefix_kgw_z": r.get("prefix_kgw_z", np.nan),
        })

        for c in candidates:
            cand_rows.append({
                "sample_idx": r.get("sample_idx"),
                "sentence_step": r.get("sentence_step"),
                "selection_policy": r.get("selection_policy"),
                "candidate_idx": c.get("candidate_idx"),
                "selected": c.get("candidate_idx") == selected_idx,
                "ias_similarity": c.get("ias_similarity", np.nan),
                "kgw_delta_z": c.get("kgw_delta_z", np.nan),
                "local_green_fraction": c.get("local_green_fraction", np.nan),
                "candidate_tokens": c.get("candidate_tokens", np.nan),
                "delta_tokens_scored": c.get("delta_tokens_scored", np.nan),
                "delta_green_tokens": c.get("delta_green_tokens", np.nan),
                "ias_rank": c.get("ias_rank", np.nan),
                "kgw_delta_z_rank": c.get("kgw_delta_z_rank", np.nan),
            })

    return pd.DataFrame(pool_rows), pd.DataFrame(cand_rows)


def summarize(pool_df):
    cols = [
        "selected_kgw_delta_z",
        "pool_mean_kgw_delta_z",
        "selected_minus_pool_kgw_delta_z",
        "selected_green_fraction",
        "pool_mean_green_fraction",
        "selected_minus_pool_green_fraction",
        "selected_tokens",
        "pool_mean_tokens",
        "selected_minus_pool_tokens",
        "selected_ias_similarity",
        "pool_mean_ias_similarity",
        "selected_minus_pool_ias_similarity",
        "selected_kgw_percentile",
        "selected_green_percentile",
        "pool_spearman_ias_vs_kgw_delta_z",
        "pool_pearson_ias_vs_kgw_delta_z",
        "pool_spearman_ias_vs_green_fraction",
    ]

    overall = {}
    overall["n_pools"] = len(pool_df)
    overall["n_samples"] = pool_df["sample_idx"].nunique()

    for c in cols:
        overall[f"{c}_mean"] = pool_df[c].mean()
        overall[f"{c}_median"] = pool_df[c].median()

    return pd.DataFrame([overall])


def by_step(pool_df):
    return (
        pool_df.groupby("sentence_step", as_index=False)
        .agg(
            n_pools=("sample_idx", "count"),
            selected_kgw_delta_z=("selected_kgw_delta_z", "mean"),
            pool_mean_kgw_delta_z=("pool_mean_kgw_delta_z", "mean"),
            selected_minus_pool_kgw_delta_z=("selected_minus_pool_kgw_delta_z", "mean"),
            selected_green_fraction=("selected_green_fraction", "mean"),
            pool_mean_green_fraction=("pool_mean_green_fraction", "mean"),
            selected_minus_pool_green_fraction=("selected_minus_pool_green_fraction", "mean"),
            selected_tokens=("selected_tokens", "mean"),
            pool_mean_tokens=("pool_mean_tokens", "mean"),
            selected_minus_pool_tokens=("selected_minus_pool_tokens", "mean"),
            selected_kgw_percentile=("selected_kgw_percentile", "mean"),
            pool_spearman_ias_vs_kgw_delta_z=("pool_spearman_ias_vs_kgw_delta_z", "mean"),
        )
    )


def selected_vs_pool(pool_df):
    rows = []
    metrics = [
        ("KGW incremental z", "selected_kgw_delta_z", "pool_mean_kgw_delta_z"),
        ("Local green fraction", "selected_green_fraction", "pool_mean_green_fraction"),
        ("Candidate tokens", "selected_tokens", "pool_mean_tokens"),
        ("IAS similarity", "selected_ias_similarity", "pool_mean_ias_similarity"),
    ]
    for name, sc, pc in metrics:
        rows.append({
            "metric": name,
            "selected_mean": pool_df[sc].mean(),
            "pool_mean": pool_df[pc].mean(),
            "difference": (pool_df[sc] - pool_df[pc]).mean(),
            "median_difference": (pool_df[sc] - pool_df[pc]).median(),
            "fraction_selected_gt_pool_mean":
                float(np.mean((pool_df[sc] - pool_df[pc]) > 0)),
        })
    return pd.DataFrame(rows)


def correlations(pool_df, cand_df):
    rows = [
        {
            "scope": "within_pool_average",
            "x": "IAS similarity",
            "y": "KGW incremental z",
            "pearson": pool_df["pool_pearson_ias_vs_kgw_delta_z"].mean(),
            "spearman": pool_df["pool_spearman_ias_vs_kgw_delta_z"].mean(),
        },
        {
            "scope": "all_candidates_pooled",
            "x": "IAS similarity",
            "y": "KGW incremental z",
            "pearson": safe_corr(cand_df["ias_similarity"], cand_df["kgw_delta_z"], "pearson"),
            "spearman": safe_corr(cand_df["ias_similarity"], cand_df["kgw_delta_z"], "spearman"),
        },
        {
            "scope": "all_candidates_pooled",
            "x": "IAS similarity",
            "y": "local green fraction",
            "pearson": safe_corr(cand_df["ias_similarity"], cand_df["local_green_fraction"], "pearson"),
            "spearman": safe_corr(cand_df["ias_similarity"], cand_df["local_green_fraction"], "spearman"),
        },
        {
            "scope": "all_candidates_pooled",
            "x": "IAS similarity",
            "y": "candidate tokens",
            "pearson": safe_corr(cand_df["ias_similarity"], cand_df["candidate_tokens"], "pearson"),
            "spearman": safe_corr(cand_df["ias_similarity"], cand_df["candidate_tokens"], "spearman"),
        },
    ]
    return pd.DataFrame(rows)


def make_report(overall, svp, corrs, out_path):
    o = overall.iloc[0]

    pct = o.get("selected_kgw_percentile_mean", np.nan)
    dz = o.get("selected_minus_pool_kgw_delta_z_mean", np.nan)
    green = o.get("selected_minus_pool_green_fraction_mean", np.nan)
    length = o.get("selected_minus_pool_tokens_mean", np.nan)
    rho = o.get("pool_spearman_ias_vs_kgw_delta_z_mean", np.nan)

    lines = []
    lines.append("# IAS-W candidate-selection diagnostic")
    lines.append("")
    lines.append(f"- Candidate pools: {int(o['n_pools'])}")
    lines.append(f"- Samples: {int(o['n_samples'])}")
    lines.append(f"- Mean KGW percentile of IAS-W-selected candidate: {pct:.4f}")
    lines.append(f"- Mean selected-minus-pool KGW incremental z: {dz:.4f}")
    lines.append(f"- Mean selected-minus-pool local green fraction: {green:.4f}")
    lines.append(f"- Mean selected-minus-pool candidate length: {length:.4f} tokens")
    lines.append(f"- Mean within-pool Spearman(IAS, KGW incremental z): {rho:.4f}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if np.isfinite(pct):
        if pct > 0.60:
            lines.append(
                "- The IAS-W-selected candidate is systematically above the random "
                "expectation (0.5) in KGW incremental evidence."
            )
        elif pct < 0.40:
            lines.append(
                "- IAS-W selection tends to prefer candidates with weaker KGW "
                "incremental evidence."
            )
        else:
            lines.append(
                "- IAS-W-selected candidates are close to random expectation in KGW "
                "incremental evidence."
            )

    if np.isfinite(length) and abs(length) >= 2:
        lines.append(
            "- Candidate length differs materially from the pool mean; length is a "
            "potential confound and should be controlled when explaining KGW gain."
        )
    else:
        lines.append(
            "- Candidate length is close to the pool mean, so a pure length explanation "
            "is less likely."
        )

    if np.isfinite(green) and green > 0:
        lines.append(
            "- Selected candidates show positive local green-token enrichment relative "
            "to the pool mean."
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.input)
    pool_df, cand_df = flatten(records)

    if pool_df.empty:
        raise RuntimeError("No candidate pools found in audit JSONL.")

    overall = summarize(pool_df)
    step = by_step(pool_df)
    svp = selected_vs_pool(pool_df)
    corrs = correlations(pool_df, cand_df)

    pool_df.to_csv(out / "candidate_pool_level.csv", index=False, encoding="utf-8-sig")
    cand_df.to_csv(out / "candidate_level.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(out / "candidate_pool_overall_summary.csv", index=False, encoding="utf-8-sig")
    step.to_csv(out / "candidate_pool_by_step.csv", index=False, encoding="utf-8-sig")
    svp.to_csv(out / "candidate_pool_selected_vs_pool.csv", index=False, encoding="utf-8-sig")
    corrs.to_csv(out / "candidate_pool_correlations.csv", index=False, encoding="utf-8-sig")

    make_report(
        overall,
        svp,
        corrs,
        out / "candidate_pool_diagnostics.md",
    )

    print("[DONE]")
    print(out / "candidate_pool_overall_summary.csv")
    print(out / "candidate_pool_by_step.csv")
    print(out / "candidate_pool_correlations.csv")
    print(out / "candidate_pool_diagnostics.md")


if __name__ == "__main__":
    main()
