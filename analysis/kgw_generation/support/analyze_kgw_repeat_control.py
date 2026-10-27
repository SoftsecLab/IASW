# -*- coding: utf-8 -*-
"""
Robust repeated-bigram control for the current KGW implementation.

Key compatibility fix:
    current project uses `ignore_repeated_bigrams`
    (not `ignore_repeated_ngrams`).

The script automatically determines which setting reproduces the already
saved Micro z scores, then reports the alternate setting if supported.

If one mode is intentionally unsupported by the local KGW source
(e.g. raises NotImplementedError), it is reported as unavailable rather than
silently treated as zero.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.baselines.kgw_core.watermark_processor import WatermarkDetector


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if not line.strip():
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


def saved_micro_z(row):
    d = row.get("detection_wm", {})
    return num(d.get("z_score_micro")) if isinstance(d, dict) else np.nan


def detect_stats(detector, text):
    try:
        res = detector.detect(text)
        return {
            "ok": True,
            "error": "",
            "z": num(res.get("z_score")),
            "T": num(res.get("num_tokens_scored")),
            "G": num(res.get("num_green_tokens")),
            "green_fraction": num(res.get("green_fraction")),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "z": np.nan,
            "T": np.nan,
            "G": np.nan,
            "green_fraction": np.nan,
        }


def bigram_stats(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    total = max(len(ids) - 1, 0)
    if total == 0:
        return {
            "token_count": len(ids),
            "total_bigrams": total,
            "unique_bigrams": 0,
            "repeated_bigram_occurrences": 0,
            "unique_bigram_ratio": np.nan,
        }

    bigrams = list(zip(ids[:-1], ids[1:]))
    unique = len(Counter(bigrams))
    return {
        "token_count": len(ids),
        "total_bigrams": total,
        "unique_bigrams": unique,
        "repeated_bigram_occurrences": total - unique,
        "unique_bigram_ratio": unique / total,
    }


def file_frame(path, label, tokenizer, det_true, det_false):
    rows = []
    for r in load_jsonl(path):
        text = str(r.get("wm_text", "") or "")
        a = detect_stats(det_true, text)
        b = detect_stats(det_false, text)
        rows.append({
            "sample_id": sid(r),
            "label": label,
            "saved_z": saved_micro_z(r),
            **bigram_stats(tokenizer, text),

            "true_ok": a["ok"],
            "true_error": a["error"],
            "z_true": a["z"],
            "T_true": a["T"],
            "G_true": a["G"],
            "green_fraction_true": a["green_fraction"],

            "false_ok": b["ok"],
            "false_error": b["error"],
            "z_false": b["z"],
            "T_false": b["T"],
            "G_false": b["G"],
            "green_fraction_false": b["green_fraction"],
        })
    return pd.DataFrame(rows)


def mae(saved, observed):
    a = np.asarray(saved, dtype=float)
    b = np.asarray(observed, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.inf
    return float(np.mean(np.abs(a[m] - b[m])))


def bootstrap_delta(values, n_boot, seed):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "n": 0, "mean_delta": np.nan, "median_delta": np.nan,
            "ci_low": np.nan, "ci_high": np.nan,
            "positive_fraction": np.nan,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = x[idx].mean(axis=1)
    return {
        "n": len(x),
        "mean_delta": float(x.mean()),
        "median_delta": float(np.median(x)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "positive_fraction": float(np.mean(x > 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--selfcheck_tolerance", type=float, default=1e-6)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    records = Path(args.records_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    iasw_files = sorted(records.glob("dual_kgw_iasw_k*_control.jsonl"))
    random_files = [
        p for p in sorted(records.glob("dual_kgw_random_k*_seed*.jsonl"))
        if "random_k1" not in p.stem
    ]
    if not iasw_files or not random_files:
        raise RuntimeError("IASW-K32 or Random-K32 files not found.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=Path(args.model_path).exists(),
        use_fast=True,
    )

    common_kwargs = dict(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme="simple_1",
        device=args.device,
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
    )

    # Correct API name in the current project.
    det_true = WatermarkDetector(
        **common_kwargs,
        ignore_repeated_bigrams=True,
    )
    det_false = WatermarkDetector(
        **common_kwargs,
        ignore_repeated_bigrams=False,
    )

    iasw = file_frame(
        iasw_files[0], "IASW", tokenizer, det_true, det_false
    )
    rands = [
        file_frame(p, f"Random{j}", tokenizer, det_true, det_false)
        for j, p in enumerate(random_files)
    ]
    all_df = pd.concat([iasw] + rands, ignore_index=True)

    all_df.to_csv(
        out / "repeat_bigram_control_sample_level.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mae_true = mae(all_df["saved_z"], all_df["z_true"])
    mae_false = mae(all_df["saved_z"], all_df["z_false"])

    true_valid = int(np.isfinite(all_df["z_true"]).sum())
    false_valid = int(np.isfinite(all_df["z_false"]).sum())

    if mae_true < mae_false:
        standard_mode = "ignore_repeated_bigrams=True"
        standard_suffix = "true"
        standard_mae = mae_true
    else:
        standard_mode = "ignore_repeated_bigrams=False"
        standard_suffix = "false"
        standard_mae = mae_false

    selfcheck = pd.DataFrame([{
        "standard_mode_detected": standard_mode,
        "mae_true": mae_true,
        "mae_false": mae_false,
        "valid_true": true_valid,
        "valid_false": false_valid,
        "standard_mae": standard_mae,
    }])
    selfcheck.to_csv(
        out / "repeat_bigram_control_selfcheck.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Detector-mode self-check ===")
    print(selfcheck.to_string(index=False))

    if standard_mae > args.selfcheck_tolerance:
        raise RuntimeError(
            "Neither repeated-bigram mode reproduces saved Micro z. "
            "Do not interpret this control until detector wiring is checked."
        )

    metrics = [
        "token_count",
        "total_bigrams",
        "unique_bigrams",
        "repeated_bigram_occurrences",
        "unique_bigram_ratio",
        "z_true", "T_true", "G_true", "green_fraction_true",
        "z_false", "T_false", "G_false", "green_fraction_false",
    ]

    wide = iasw.set_index("sample_id")[metrics].add_prefix("iasw_")
    for j, rf in enumerate(rands):
        rr = rf.set_index("sample_id")[metrics]
        for m in metrics:
            wide[f"r{j}_{m}"] = rr[m]

    rows = []
    for m in metrics:
        rcols = [f"r{j}_{m}" for j in range(len(rands))]
        wide[f"random_mean_{m}"] = wide[rcols].mean(axis=1)
        delta = wide[f"iasw_{m}"] - wide[f"random_mean_{m}"]
        rows.append({
            "metric": m,
            "iasw_mean": float(wide[f"iasw_{m}"].mean()),
            "random_mean": float(wide[f"random_mean_{m}"].mean()),
            **bootstrap_delta(delta, args.n_boot, args.seed + len(rows)),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        out / "repeat_bigram_control_paired_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    focus = summary[summary["metric"].isin([
        "unique_bigram_ratio",
        "repeated_bigram_occurrences",
        "z_true", "T_true", "green_fraction_true",
        "z_false", "T_false", "green_fraction_false",
    ])]

    print("\n=== Repeat-bigram control focus ===")
    print(focus.to_string(index=False))

    if true_valid == 0:
        print("\n[NOTE] ignore_repeated_bigrams=True is unsupported by this local KGW implementation.")
    if false_valid == 0:
        print("\n[NOTE] ignore_repeated_bigrams=False is unsupported by this local KGW implementation.")

    print("\n[DONE]")
    print(out / "repeat_bigram_control_paired_bootstrap.csv")


if __name__ == "__main__":
    main()