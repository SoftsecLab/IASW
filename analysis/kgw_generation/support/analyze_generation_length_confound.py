# -*- coding: utf-8 -*-
"""
Post-hoc generation-length diagnostic for the EXISTING Random-K32 / IASW-K32
records.

No generation, no detector recomputation, no CausalLM.

The KGW detector uses simple_1 and, in the current implementation,
num_tokens_scored is effectively tokenized_length - 1. Therefore a systematic
difference in num_tokens_scored is also a systematic output-token-length
difference.

This script quantifies:
- saved KGW scored-token count;
- tokenizer token count;
- word count;
- token/word ratio;
- fraction of texts that end substantially below the nominal 200-token target.

It is intended to diagnose the current orchestrator's early-EOS stopping
behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


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


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def extract_saved_T(row):
    det = row.get("detection_wm", {})
    details = det.get("details", {}) if isinstance(det, dict) else {}
    micro = details.get("micro_details", {}) if isinstance(details, dict) else {}
    if not isinstance(micro, dict):
        return np.nan

    for k in ["num_tokens_scored", "T", "num_scored"]:
        if k in micro:
            v = num(micro.get(k))
            if np.isfinite(v):
                return v
    return np.nan


def frame(path, label, tokenizer, target_len):
    out = []

    for row in load_jsonl(path):
        text = str(row.get("wm_text", "") or "")
        ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        token_count = len(ids)
        words = len(text.split())
        T = extract_saved_T(row)

        out.append({
            "sample_id": sid(row),
            "label": label,
            "token_count": token_count,
            "saved_num_tokens_scored": T,
            "token_minus_T": (
                token_count - T
                if np.isfinite(T)
                else np.nan
            ),
            "words": words,
            "tokens_per_word": (
                token_count / words if words > 0 else np.nan
            ),
            "target_len": target_len,
            "token_deficit_from_target": target_len - token_count,
            "reached_90pct_target": token_count >= 0.90 * target_len,
            "reached_95pct_target": token_count >= 0.95 * target_len,
            "reached_target": token_count >= target_len,
        })

    return pd.DataFrame(out)


def bootstrap_delta(x, n_boot, seed):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(
            n=0,
            mean_delta=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            positive_fraction=np.nan,
        )

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boot = a[idx].mean(axis=1)

    return dict(
        n=len(a),
        mean_delta=float(a.mean()),
        median_delta=float(np.median(a)),
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        positive_fraction=float(np.mean(a > 0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_len", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=10000)
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
        raise RuntimeError("IASW/Random files not found.")

    local_only = Path(args.model_path).exists()
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=local_only,
        use_fast=True,
    )

    iasw = frame(
        iasw_files[0],
        "IASW",
        tok,
        args.target_len,
    )
    rands = [
        frame(p, f"Random{j}", tok, args.target_len)
        for j, p in enumerate(random_files)
    ]

    all_df = pd.concat([iasw] + rands, ignore_index=True)
    all_df.to_csv(
        out / "generation_length_sample_level.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_summary = (
        all_df.groupby("label", as_index=False)
        .agg(
            n=("sample_id", "count"),
            token_count_mean=("token_count", "mean"),
            token_count_median=("token_count", "median"),
            scored_tokens_mean=("saved_num_tokens_scored", "mean"),
            token_minus_T_mean=("token_minus_T", "mean"),
            words_mean=("words", "mean"),
            tokens_per_word_mean=("tokens_per_word", "mean"),
            target_deficit_mean=("token_deficit_from_target", "mean"),
            reached_90pct_rate=("reached_90pct_target", "mean"),
            reached_95pct_rate=("reached_95pct_target", "mean"),
            reached_target_rate=("reached_target", "mean"),
        )
    )
    group_summary.to_csv(
        out / "generation_length_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Paired IASW minus mean(Random) by sample.
    base = iasw.set_index("sample_id")[
        ["token_count", "saved_num_tokens_scored", "words", "tokens_per_word"]
    ].add_prefix("iasw_")

    for j, rf in enumerate(rands):
        rr = rf.set_index("sample_id")[
            ["token_count", "saved_num_tokens_scored", "words", "tokens_per_word"]
        ]
        for metric in rr.columns:
            base[f"r{j}_{metric}"] = rr[metric]

    metrics = [
        "token_count",
        "saved_num_tokens_scored",
        "words",
        "tokens_per_word",
    ]

    rows = []
    for metric in metrics:
        rcols = [f"r{j}_{metric}" for j in range(len(rands))]
        base[f"random_mean_{metric}"] = base[rcols].mean(axis=1)
        delta = (
            base[f"iasw_{metric}"]
            - base[f"random_mean_{metric}"]
        )
        stat = bootstrap_delta(
            delta,
            args.n_boot,
            args.seed + len(rows),
        )
        rows.append({
            "metric": metric,
            "iasw_mean": float(base[f"iasw_{metric}"].mean()),
            "random_mean": float(base[f"random_mean_{metric}"].mean()),
            **stat,
        })

    paired = pd.DataFrame(rows)
    paired.to_csv(
        out / "generation_length_paired_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Group summary ===")
    print(group_summary.to_string(index=False))
    print("\n=== IASW minus mean(Random) ===")
    print(paired.to_string(index=False))

    print("\n[DONE]")
    print(out / "generation_length_paired_bootstrap.csv")


if __name__ == "__main__":
    main()
