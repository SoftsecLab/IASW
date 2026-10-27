# -*- coding: utf-8 -*-
"""
Decompose the EXISTING full-text KGW score difference between IASW-K32 and
Random-K32 into detector components.

No generation. No attack. No model loading.

Reads saved detection_wm from:
  - dual_kgw_iasw_k*_control.jsonl
  - dual_kgw_random_k*_seed*.jsonl

Primary quantities:
  z_micro
  num_tokens_scored
  num_green_tokens
  green_fraction
  excess_count = G - gamma*T
  sqrtT_excess = sqrt(T) * (green_fraction - gamma)

All comparisons are paired by sample ID. The Random side is the per-sample
mean over all available Random-K32 selection seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
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


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def first_num(d, keys):
    if not isinstance(d, dict):
        return np.nan
    for k in keys:
        if k in d:
            v = num(d.get(k))
            if np.isfinite(v):
                return v
    return np.nan


def extract_one(row, gamma):
    det = row.get("detection_wm", {})
    details = det.get("details", {}) if isinstance(det, dict) else {}
    micro = (
        details.get("micro_details", {})
        if isinstance(details, dict)
        else {}
    )
    if not isinstance(micro, dict):
        micro = {}

    z = first_num(det, ["z_score_micro"])
    if not np.isfinite(z):
        z = first_num(micro, ["z_score", "z"])

    T = first_num(
        micro,
        [
            "num_tokens_scored",
            "T",
            "num_scored",
            "scored_tokens",
        ],
    )
    G = first_num(
        micro,
        [
            "num_green_tokens",
            "green_token_count",
            "green_tokens",
        ],
    )
    frac = first_num(
        micro,
        [
            "green_fraction",
            "green_token_fraction",
            "fraction",
        ],
    )

    if not np.isfinite(frac):
        if np.isfinite(T) and np.isfinite(G) and T > 0:
            frac = G / T

    excess_count = (
        G - gamma*T
        if np.isfinite(G) and np.isfinite(T)
        else np.nan
    )

    sqrtT_excess = (
        np.sqrt(T) * (frac - gamma)
        if np.isfinite(T)
        and T > 0
        and np.isfinite(frac)
        else np.nan
    )

    reconstructed_z = (
        excess_count / np.sqrt(T*gamma*(1.0-gamma))
        if np.isfinite(excess_count)
        and np.isfinite(T)
        and T > 0
        and 0 < gamma < 1
        else np.nan
    )

    return {
        "z_micro": z,
        "num_tokens_scored": T,
        "num_green_tokens": G,
        "green_fraction": frac,
        "green_excess": (
            frac - gamma
            if np.isfinite(frac)
            else np.nan
        ),
        "excess_count": excess_count,
        "sqrtT_excess": sqrtT_excess,
        "reconstructed_z": reconstructed_z,
        "words": len(str(row.get("wm_text", "") or "").split()),
    }


def frame(path, label, gamma):
    out = []
    for row in load_jsonl(path):
        out.append({
            "sample_id": sid(row),
            "label": label,
            **extract_one(row, gamma),
        })
    return pd.DataFrame(out)


def bootstrap_delta(x, n_boot, seed):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {
            "n": 0,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "positive_fraction": np.nan,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    b = a[idx].mean(axis=1)

    return {
        "n": len(a),
        "mean_delta": float(a.mean()),
        "median_delta": float(np.median(a)),
        "ci_low": float(np.quantile(b, 0.025)),
        "ci_high": float(np.quantile(b, 0.975)),
        "positive_fraction": float(np.mean(a > 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    records = Path(args.records_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    iasw_files = sorted(
        records.glob("dual_kgw_iasw_k*_control.jsonl")
    )
    random_files = [
        p for p in sorted(
            records.glob("dual_kgw_random_k*_seed*.jsonl")
        )
        if "random_k1" not in p.stem
    ]

    if not iasw_files:
        raise FileNotFoundError("IASW-K control file not found.")
    if not random_files:
        raise FileNotFoundError("Random-K multi-seed files not found.")

    iasw = frame(iasw_files[0], "IASW", args.gamma)
    random_frames = [
        frame(p, f"Random{j}", args.gamma)
        for j, p in enumerate(random_files)
    ]

    metrics = [
        "z_micro",
        "num_tokens_scored",
        "num_green_tokens",
        "green_fraction",
        "green_excess",
        "excess_count",
        "sqrtT_excess",
        "reconstructed_z",
        "words",
    ]

    wide = iasw.set_index("sample_id")[
        metrics
    ].add_prefix("iasw_")

    for j, rf in enumerate(random_frames):
        rr = rf.set_index("sample_id")[metrics]
        for metric in metrics:
            wide[f"random{j}_{metric}"] = rr[metric]

    for metric in metrics:
        cols = [
            f"random{j}_{metric}"
            for j in range(len(random_frames))
        ]
        wide[f"random_mean_{metric}"] = wide[cols].mean(axis=1)
        wide[f"delta_{metric}"] = (
            wide[f"iasw_{metric}"]
            - wide[f"random_mean_{metric}"]
        )

    wide = wide.reset_index()

    rows = []
    for metric in metrics:
        stat = bootstrap_delta(
            wide[f"delta_{metric}"],
            args.n_boot,
            args.seed + len(rows),
        )
        rows.append({
            "metric": metric,
            "iasw_mean": float(wide[f"iasw_{metric}"].mean()),
            "random_mean": float(
                wide[f"random_mean_{metric}"].mean()
            ),
            **stat,
        })

    summary = pd.DataFrame(rows)

    # Detector self-consistency.
    consistency = pd.DataFrame([
        {
            "group": "IASW",
            "mean_abs_z_minus_reconstructed": float(
                np.nanmean(
                    np.abs(
                        wide["iasw_z_micro"]
                        - wide["iasw_reconstructed_z"]
                    )
                )
            ),
        },
        {
            "group": "Random-mean",
            "mean_abs_z_minus_reconstructed": float(
                np.nanmean(
                    np.abs(
                        wide["random_mean_z_micro"]
                        - wide["random_mean_reconstructed_z"]
                    )
                )
            ),
        },
    ])

    wide.to_csv(
        out / "full_kgw_component_sample_level.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        out / "full_kgw_component_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    consistency.to_csv(
        out / "full_kgw_detector_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Full-text KGW component decomposition ===")
    print(summary.to_string(index=False))
    print("\n=== Detector consistency ===")
    print(consistency.to_string(index=False))

    print("\n[DONE]")
    print(out / "full_kgw_component_bootstrap.csv")
    print(out / "full_kgw_detector_consistency.csv")


if __name__ == "__main__":
    main()
