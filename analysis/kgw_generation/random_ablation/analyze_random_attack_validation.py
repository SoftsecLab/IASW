# -*- coding: utf-8 -*-
"""
Analyze Random-K32 vs IASW-K32 after attacks on the existing 100-sample texts.

The Random side is averaged across all available random selection seeds.
No generation or attack is performed by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            x["_line_idx"] = i
            rows.append(x)
    return rows


def sid(row):
    for k in ["idx", "id", "sample_id", "record_idx", "index"]:
        if k in row:
            return str(row[k])
    return str(row["_line_idx"])


def get_z(det, key):
    if not isinstance(det, dict):
        return np.nan
    try:
        v = float(det.get(key, np.nan))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def attack_frame(path, label):
    rows = load_jsonl(path)
    out = []

    for r in rows:
        human = r.get("detection_human", {})
        attacks = r.get("attacks", {})

        for attack, item in attacks.items():
            if not isinstance(item, dict):
                continue
            det = item.get("detection_results", {})
            if not isinstance(det, dict):
                continue

            out.append({
                "sample_id": sid(r),
                "label": label,
                "attack": attack,
                "micro": get_z(det, "z_score_micro"),
                "macro": get_z(det, "z_score_macro_calibrated"),
                "fisher": get_z(det, "z_score_fisher"),
                "human_micro": get_z(human, "z_score_micro"),
                "human_macro": get_z(human, "z_score_macro_calibrated"),
                "human_fisher": get_z(human, "z_score_fisher"),
            })

    return pd.DataFrame(out)


def threshold(x, fpr=0.01):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    return float(np.quantile(a, 1.0 - fpr))


def tpr(pos, neg, fpr=0.01):
    t = threshold(neg, fpr)
    p = np.asarray(pos, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0 or not np.isfinite(t):
        return np.nan
    return float(np.mean(p >= t))


def boot_delta(delta, n_boot, seed):
    x = np.asarray(delta, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = x[idx].mean(axis=1)
    return (
        float(x.mean()),
        float(np.quantile(b, 0.025)),
        float(np.quantile(b, 0.975)),
        len(x),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    records = Path(args.records_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    iasw_files = sorted(records.glob("dual_kgw_iasw_k*_control.jsonl"))
    random_files = [
        p for p in sorted(records.glob("dual_kgw_random_k*_seed*.jsonl"))
        if "random_k1" not in p.stem
    ]

    if not iasw_files or not random_files:
        raise RuntimeError("IASW or Random-K32 files were not found.")

    iasw = attack_frame(iasw_files[0], "IASW")
    random_frames = [
        attack_frame(p, f"R{j}")
        for j, p in enumerate(random_files)
    ]

    attacks = sorted(
        set(iasw["attack"].unique())
        & set.intersection(
            *[set(f["attack"].unique()) for f in random_frames]
        )
    )

    rows = []
    sample_rows = []

    for attack in attacks:
        ii = iasw[iasw["attack"] == attack].set_index("sample_id")

        # Aligned Random mean by sample.
        rand_join = None
        for j, rf in enumerate(random_frames):
            rr = rf[rf["attack"] == attack].set_index("sample_id")[
                ["micro", "macro", "fisher"]
            ].add_suffix(f"_r{j}")
            rand_join = rr if rand_join is None else rand_join.join(
                rr, how="inner"
            )

        common = ii.index.intersection(rand_join.index)
        ii = ii.loc[common]
        rand_join = rand_join.loc[common]

        for metric in ["micro", "macro", "fisher"]:
            rcols = [
                c for c in rand_join.columns
                if c.startswith(metric + "_r")
            ]
            random_mean = rand_join[rcols].mean(axis=1)
            delta = ii[metric] - random_mean

            mean_delta, ci_low, ci_high, n = boot_delta(
                delta,
                args.n_boot,
                args.seed + len(rows),
            )

            # Descriptive TPR@1%; N=100 is small, so score-delta CI is primary.
            iasw_tpr = tpr(
                ii[metric],
                ii["human_" + metric],
                0.01,
            )

            random_tprs = []
            for rf in random_frames:
                rr = rf[rf["attack"] == attack]
                random_tprs.append(
                    tpr(
                        rr[metric],
                        rr["human_" + metric],
                        0.01,
                    )
                )

            rows.append({
                "attack": attack,
                "metric": metric,
                "n_aligned": n,
                "iasw_mean_z": float(ii[metric].mean()),
                "random_mean_z": float(random_mean.mean()),
                "iasw_minus_random_mean_z": mean_delta,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "iasw_tpr1_pct": 100 * iasw_tpr,
                "random_tpr1_pct_mean": 100 * np.nanmean(random_tprs),
                "tpr1_delta_pp": 100 * (
                    iasw_tpr - np.nanmean(random_tprs)
                ),
            })

            for sample_id in common:
                sample_rows.append({
                    "sample_id": sample_id,
                    "attack": attack,
                    "metric": metric,
                    "iasw_z": ii.loc[sample_id, metric],
                    "random_mean_z": random_mean.loc[sample_id],
                    "delta": delta.loc[sample_id],
                })

    result = pd.DataFrame(rows)
    sample_df = pd.DataFrame(sample_rows)

    result.to_csv(
        out / "attack_random_vs_iasw_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sample_df.to_csv(
        out / "attack_random_vs_iasw_sample_level.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Average attack-level gain by score.
    avg = (
        result.groupby("metric", as_index=False)
        .agg(
            mean_z_gain=(
                "iasw_minus_random_mean_z", "mean"
            ),
            mean_tpr1_gain_pp=(
                "tpr1_delta_pp", "mean"
            ),
        )
    )
    avg.to_csv(
        out / "attack_random_vs_iasw_average.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(result.to_string(index=False))
    print("\nAverage:")
    print(avg.to_string(index=False))

    print("\n[DONE]")
    print(out / "attack_random_vs_iasw_summary.csv")


if __name__ == "__main__":
    main()
