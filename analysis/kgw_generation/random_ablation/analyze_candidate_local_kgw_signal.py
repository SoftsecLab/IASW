# -*- coding: utf-8 -*-
"""
Re-analyze the EXISTING IAS-W candidate audit using the additive KGW sufficient
statistics, without regenerating anything.

Why this is needed
------------------
The previous candidate audit used:

    z(prefix + candidate) - z(prefix)

as "KGW incremental z". KGW z is length-normalized and therefore non-additive,
so that quantity can obscure local candidate evidence.

For a candidate with:
    dT = newly scored KGW positions
    dG = newly green positions
    gamma = KGW green-list fraction

define:

    local_excess_count = dG - gamma * dT
    local_green_rate   = dG / dT
    local_kgw_z        = (dG - gamma*dT) /
                         sqrt(dT*gamma*(1-gamma))

This script asks whether the IAS-W-selected candidate is above random
expectation according to these local quantities.

No model / generation is used.
"""

from __future__ import annotations

import argparse
import json
import math
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


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


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


def percentile_of_selected(values, selected_idx):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not (0 <= selected_idx < len(arr)):
        return np.nan
    if not np.isfinite(arr[selected_idx]):
        return np.nan

    vals = arr[finite]
    if len(vals) <= 1:
        return np.nan

    selected = arr[selected_idx]

    # Midrank percentile in [0,1].
    below = np.sum(vals < selected)
    equal = np.sum(vals == selected)
    rank_fraction = (below + 0.5 * equal) / len(vals)
    return float(rank_fraction)


def cluster_bootstrap(pool_df, metrics, n_boot, seed):
    ids = np.asarray(sorted(pool_df["sample_idx"].unique()))
    groups = {
        sid: pool_df[pool_df["sample_idx"] == sid]
        for sid in ids
    }

    rng = np.random.default_rng(seed)
    out = []

    for metric in metrics:
        obs = float(pool_df[metric].mean())
        boots = []

        for _ in range(n_boot):
            chosen = rng.choice(ids, size=len(ids), replace=True)
            b = pd.concat([groups[sid] for sid in chosen], ignore_index=True)
            boots.append(float(b[metric].mean()))

        boots = np.asarray(boots, dtype=float)

        out.append({
            "metric": metric,
            "observed_mean": obs,
            "ci_low": float(np.nanquantile(boots, 0.025)),
            "ci_high": float(np.nanquantile(boots, 0.975)),
        })

    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.audit_file)

    pool_rows = []
    cand_rows = []

    for r in records:
        candidates = r.get("candidates", []) or []
        selected_idx = int(r.get("selected_idx", -1))
        if not candidates or not (0 <= selected_idx < len(candidates)):
            continue

        values = []

        for c in candidates:
            dt = num(c.get("delta_tokens_scored"))
            dg = num(c.get("delta_green_tokens"))
            ias = num(c.get("ias_similarity"))
            clen = num(c.get("candidate_tokens"))

            local_rate = np.nan
            local_excess_count = np.nan
            local_z = np.nan

            if np.isfinite(dt) and np.isfinite(dg) and dt > 0:
                local_rate = dg / dt
                local_excess_count = dg - args.gamma * dt
                den = math.sqrt(
                    dt * args.gamma * (1.0 - args.gamma)
                )
                if den > 0:
                    local_z = local_excess_count / den

            values.append({
                "ias_similarity": ias,
                "candidate_tokens": clen,
                "delta_tokens_scored": dt,
                "delta_green_tokens": dg,
                "local_green_rate": local_rate,
                "local_green_excess": (
                    local_rate - args.gamma
                    if np.isfinite(local_rate)
                    else np.nan
                ),
                "local_excess_count": local_excess_count,
                "local_kgw_z": local_z,
            })

        df = pd.DataFrame(values)
        sel = df.iloc[selected_idx]

        pool_rows.append({
            "sample_idx": int(r.get("sample_idx", -1)),
            "sentence_step": int(r.get("sentence_step", -1)),
            "n_candidates": len(df),

            "selected_local_kgw_z": sel["local_kgw_z"],
            "pool_mean_local_kgw_z": df["local_kgw_z"].mean(),
            "selected_minus_pool_local_kgw_z":
                sel["local_kgw_z"] - df["local_kgw_z"].mean(),

            "selected_local_green_rate": sel["local_green_rate"],
            "pool_mean_local_green_rate": df["local_green_rate"].mean(),
            "selected_minus_pool_local_green_rate":
                sel["local_green_rate"] - df["local_green_rate"].mean(),

            "selected_local_excess_count": sel["local_excess_count"],
            "pool_mean_local_excess_count": df["local_excess_count"].mean(),
            "selected_minus_pool_local_excess_count":
                sel["local_excess_count"]
                - df["local_excess_count"].mean(),

            "selected_local_kgw_percentile":
                percentile_of_selected(
                    df["local_kgw_z"].to_numpy(),
                    selected_idx,
                ),
            "selected_local_green_percentile":
                percentile_of_selected(
                    df["local_green_rate"].to_numpy(),
                    selected_idx,
                ),
            "selected_excess_count_percentile":
                percentile_of_selected(
                    df["local_excess_count"].to_numpy(),
                    selected_idx,
                ),

            "spearman_ias_vs_local_kgw_z":
                safe_corr(
                    df["ias_similarity"],
                    df["local_kgw_z"],
                    "spearman",
                ),
            "pearson_ias_vs_local_kgw_z":
                safe_corr(
                    df["ias_similarity"],
                    df["local_kgw_z"],
                    "pearson",
                ),
            "spearman_ias_vs_local_green_rate":
                safe_corr(
                    df["ias_similarity"],
                    df["local_green_rate"],
                    "spearman",
                ),
        })

        for j, row in df.iterrows():
            cand_rows.append({
                "sample_idx": int(r.get("sample_idx", -1)),
                "sentence_step": int(r.get("sentence_step", -1)),
                "candidate_idx": int(j),
                "selected": int(j) == selected_idx,
                **row.to_dict(),
            })

    pool_df = pd.DataFrame(pool_rows)
    cand_df = pd.DataFrame(cand_rows)

    if pool_df.empty:
        raise RuntimeError("No candidate pools found.")

    metrics = [
        "selected_local_kgw_percentile",
        "selected_local_green_percentile",
        "selected_excess_count_percentile",
        "selected_minus_pool_local_kgw_z",
        "selected_minus_pool_local_green_rate",
        "selected_minus_pool_local_excess_count",
        "spearman_ias_vs_local_kgw_z",
        "pearson_ias_vs_local_kgw_z",
        "spearman_ias_vs_local_green_rate",
    ]

    boot = cluster_bootstrap(
        pool_df,
        metrics,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    by_step = (
        pool_df.groupby("sentence_step", as_index=False)
        .agg(
            n_pools=("sample_idx", "count"),
            local_kgw_percentile=(
                "selected_local_kgw_percentile", "mean"
            ),
            local_green_percentile=(
                "selected_local_green_percentile", "mean"
            ),
            selected_minus_pool_local_kgw_z=(
                "selected_minus_pool_local_kgw_z", "mean"
            ),
            selected_minus_pool_local_green_rate=(
                "selected_minus_pool_local_green_rate", "mean"
            ),
            selected_minus_pool_local_excess_count=(
                "selected_minus_pool_local_excess_count", "mean"
            ),
            spearman_ias_vs_local_kgw_z=(
                "spearman_ias_vs_local_kgw_z", "mean"
            ),
        )
    )

    missing = pd.DataFrame([
        {
            "field": c,
            "n_total": len(cand_df),
            "n_valid": int(cand_df[c].notna().sum()),
            "missing_rate": float(cand_df[c].isna().mean()),
        }
        for c in [
            "local_kgw_z",
            "local_green_rate",
            "local_excess_count",
            "delta_tokens_scored",
            "delta_green_tokens",
        ]
    ])

    pool_df.to_csv(
        out_dir / "local_signal_pool_level.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cand_df.to_csv(
        out_dir / "local_signal_candidate_level.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boot.to_csv(
        out_dir / "local_signal_cluster_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_step.to_csv(
        out_dir / "local_signal_by_step.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing.to_csv(
        out_dir / "local_signal_missingness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Local KGW signal bootstrap ===")
    print(boot.to_string(index=False))
    print("\n[DONE]")
    print(out_dir / "local_signal_cluster_bootstrap.csv")
    print(out_dir / "local_signal_by_step.csv")


if __name__ == "__main__":
    main()
