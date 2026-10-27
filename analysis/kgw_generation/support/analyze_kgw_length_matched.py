# -*- coding: utf-8 -*-
"""
Length-matched KGW detection on EXISTING IASW-K32 / Random-K32 outputs.

This is the key control for the observed full-text micro-z gain.

No generation and no CausalLM are used.

Important:
- KGW green-list generation is device-dependent because the detector's
  torch.Generator is created on the configured device.
- Use the SAME detector device as the main experiment (normally cuda:0).
- The script first verifies that direct full-text detection reproduces the saved
  z_score_micro. It aborts if not.

Two controls are reported:

1. paired-min length:
   For each prompt, truncate IASW and every Random run to the SAME token length:
       L_i = min(lengths across IASW + all Random runs)
   Then compare micro z.

2. fixed lengths:
   Compare at common first N tokens, e.g. 100,120,140.
   A prompt is included only when every compared output has at least N tokens.

Tokenized prefixes are passed directly to WatermarkDetector; text is never
decoded/re-encoded after truncation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.watermarks.algorithms.micro.kgw import KGWMicroAlgorithm


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


def saved_z(row):
    d = row.get("detection_wm", {})
    return num(d.get("z_score_micro")) if isinstance(d, dict) else np.nan


def tokenize_file(path, tokenizer):
    out = {}
    saved = {}

    for row in load_jsonl(path):
        k = sid(row)
        ids = tokenizer(
            str(row.get("wm_text", "") or ""),
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        if (
            tokenizer.bos_token_id is not None
            and len(ids) > 0
            and ids[0].item() == tokenizer.bos_token_id
        ):
            ids = ids[1:]
        out[k] = ids
        saved[k] = saved_z(row)

    return out, saved


def detect_ids(detector, ids, device):
    if len(ids) <= 1:
        return np.nan

    x = ids.to(device)
    try:
        res = detector.detect(tokenized_text=x)
        return num(res.get("z_score"))
    except Exception:
        return np.nan


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
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--fixed_lengths", default="100,120,140")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--selfcheck_tolerance", type=float, default=1e-6)
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

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA detector device requested but CUDA is unavailable. "
            "Do not switch to CPU because KGW green lists are device-dependent."
        )

    local_only = Path(args.model_path).exists()
    tok = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=local_only,
        use_fast=True,
    )

    dummy = SimpleNamespace(gamma=args.gamma, delta=args.delta)
    algo = KGWMicroAlgorithm(
        dummy,
        tok,
        model=None,
        device=args.device,
    )
    detector = algo.detector

    iasw_ids, iasw_saved = tokenize_file(iasw_files[0], tok)
    random_packs = [tokenize_file(p, tok) for p in random_files]
    random_ids = [x[0] for x in random_packs]
    random_saved = [x[1] for x in random_packs]

    common = set(iasw_ids)
    for d in random_ids:
        common &= set(d)
    common = sorted(common)

    # ------------------------------------------------------------
    # Mandatory full-text detector self-check.
    # ------------------------------------------------------------
    checks = []
    for k in common:
        direct = detect_ids(detector, iasw_ids[k], args.device)
        checks.append({
            "sample_id": k,
            "group": "IASW",
            "saved_z": iasw_saved[k],
            "direct_z": direct,
            "abs_diff": abs(direct - iasw_saved[k]),
        })

        for j in range(len(random_ids)):
            direct = detect_ids(
                detector,
                random_ids[j][k],
                args.device,
            )
            checks.append({
                "sample_id": k,
                "group": f"Random{j}",
                "saved_z": random_saved[j][k],
                "direct_z": direct,
                "abs_diff": abs(
                    direct - random_saved[j][k]
                ),
            })

    check_df = pd.DataFrame(checks)
    check_df.to_csv(
        out / "length_matched_detector_selfcheck.csv",
        index=False,
        encoding="utf-8-sig",
    )

    max_diff = float(check_df["abs_diff"].max())
    mean_diff = float(check_df["abs_diff"].mean())
    print(
        f"[SELFCHECK] mean abs diff={mean_diff:.10f}, "
        f"max abs diff={max_diff:.10f}"
    )

    if max_diff > args.selfcheck_tolerance:
        raise RuntimeError(
            "Detector self-check failed. Use the same detector device "
            "as the main experiment before interpreting results."
        )

    # ------------------------------------------------------------
    # Paired-min common length.
    # ------------------------------------------------------------
    sample_rows = []

    for k in common:
        lengths = [len(iasw_ids[k])] + [
            len(d[k]) for d in random_ids
        ]
        L = min(lengths)

        zi = detect_ids(
            detector,
            iasw_ids[k][:L],
            args.device,
        )
        zr = [
            detect_ids(
                detector,
                d[k][:L],
                args.device,
            )
            for d in random_ids
        ]

        sample_rows.append({
            "sample_id": k,
            "control": "paired_min",
            "common_token_length": L,
            "iasw_z": zi,
            "random_mean_z": float(np.nanmean(zr)),
            "delta_z": zi - float(np.nanmean(zr)),
        })

    # ------------------------------------------------------------
    # Fixed common lengths.
    # ------------------------------------------------------------
    fixed_lengths = [
        int(x.strip())
        for x in args.fixed_lengths.split(",")
        if x.strip()
    ]

    for L in fixed_lengths:
        for k in common:
            if len(iasw_ids[k]) < L:
                continue
            if any(len(d[k]) < L for d in random_ids):
                continue

            zi = detect_ids(
                detector,
                iasw_ids[k][:L],
                args.device,
            )
            zr = [
                detect_ids(
                    detector,
                    d[k][:L],
                    args.device,
                )
                for d in random_ids
            ]

            sample_rows.append({
                "sample_id": k,
                "control": f"fixed_{L}",
                "common_token_length": L,
                "iasw_z": zi,
                "random_mean_z": float(np.nanmean(zr)),
                "delta_z": zi - float(np.nanmean(zr)),
            })

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(
        out / "length_matched_sample_level.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows = []
    for control, g in sample_df.groupby("control", sort=False):
        stat = bootstrap_delta(
            g["delta_z"],
            args.n_boot,
            args.seed + len(rows),
        )
        rows.append({
            "control": control,
            "mean_common_token_length": float(
                g["common_token_length"].mean()
            ),
            "iasw_mean_z": float(g["iasw_z"].mean()),
            "random_mean_z": float(g["random_mean_z"].mean()),
            **stat,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        out / "length_matched_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Length-matched KGW comparison ===")
    print(summary.to_string(index=False))

    print("\n[DONE]")
    print(out / "length_matched_bootstrap_summary.csv")


if __name__ == "__main__":
    main()
