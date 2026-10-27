# -*- coding: utf-8 -*-
"""Summarize the unattacked Random-vs-IAS-W generation ablation.

Every JSONL should already have baseline detection refreshed with:
    --method universal_dual --micro_method kgw --macro_method ias

Outputs:
    random_ablation_file_summary.csv
    random_ablation_group_summary.csv
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
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
                if isinstance(x, dict):
                    rows.append(x)
            except Exception:
                pass
    return rows


def z_from_det(det, key):
    if not isinstance(det, dict):
        return np.nan
    try:
        v = float(det.get(key, np.nan))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def threshold_at_fpr(neg, fpr=0.01):
    x = np.asarray(neg, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, 1.0 - fpr))


def tpr_at_fpr(pos, neg, fpr=0.01):
    p = np.asarray(pos, dtype=float)
    p = p[np.isfinite(p)]
    thr = threshold_at_fpr(neg, fpr)
    if len(p) == 0 or not np.isfinite(thr):
        return np.nan
    return float(np.mean(p >= thr))


def infer_group(name: str):
    s = name.lower()
    if "kgw_control" in s:
        return "KGW-continuous"
    if "random_k1" in s:
        return "Random-K1"
    if "random_k32" in s:
        return "Random-K32"
    if "iasw_k32" in s:
        return "IASW-K32"
    return name


def summarize_file(path: Path):
    rows = load_jsonl(path)

    score_keys = {
        "micro": "z_score_micro",
        "macro": "z_score_macro_calibrated",
        "fisher": "z_score_fisher",
    }
    neg = {k: [] for k in score_keys}
    pos = {k: [] for k in score_keys}
    words, times = [], []

    for r in rows:
        dh = r.get("detection_human", {})
        dw = r.get("detection_wm", {})
        for short, key in score_keys.items():
            neg[short].append(z_from_det(dh, key))
            pos[short].append(z_from_det(dw, key))

        text = r.get("wm_text", "")
        if isinstance(text, str) and text.strip():
            words.append(len(text.split()))
        try:
            t = float(r.get("generation_time", np.nan))
            if np.isfinite(t):
                times.append(t)
        except Exception:
            pass

    def finite_mean(values):
        a = np.asarray(values, dtype=float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else np.nan

    meta = rows[0].get("meta_info", {}) if rows else {}
    out = {
        "file": str(path),
        "name": path.stem,
        "group": infer_group(path.stem),
        "n": len(rows),
        "selection_policy": meta.get("candidate_selection_policy", ""),
        "selection_seed": meta.get("candidate_selection_seed", np.nan),
        "generation_seed": meta.get("generation_seed", np.nan),
        "dual_temperature": meta.get("dual_temperature", np.nan),
        "dual_top_k": meta.get("dual_top_k", np.nan),
    }
    for short in score_keys:
        out[f"{short}_tpr1_pct"] = 100 * tpr_at_fpr(pos[short], neg[short], 0.01)
        out[f"wm_{short}_z_mean"] = finite_mean(pos[short])
        out[f"human_{short}_z_mean"] = finite_mean(neg[short])

    out["wm_words_mean"] = float(np.mean(words)) if words else np.nan
    out["generation_time_median_s"] = float(np.median(times)) if times else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    records_dir = Path(args.records_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(records_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files in {records_dir}")

    file_df = pd.DataFrame([summarize_file(p) for p in files])
    file_df.to_csv(out_dir / "random_ablation_file_summary.csv", index=False, encoding="utf-8-sig")

    numeric = [
        "micro_tpr1_pct", "macro_tpr1_pct", "fisher_tpr1_pct",
        "wm_micro_z_mean", "wm_macro_z_mean", "wm_fisher_z_mean",
        "wm_words_mean", "generation_time_median_s",
    ]
    group_rows = []
    for group, g in file_df.groupby("group", sort=False):
        row = {"group": group, "n_runs": len(g)}
        for c in numeric:
            row[c + "_mean"] = g[c].mean()
            row[c + "_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        group_rows.append(row)

    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(out_dir / "random_ablation_group_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== File-level ===")
    print(file_df[[
        "name", "group", "micro_tpr1_pct", "macro_tpr1_pct",
        "fisher_tpr1_pct", "wm_micro_z_mean", "wm_words_mean"
    ]].to_string(index=False))
    print("\n=== Group-level ===")
    print(group_df.to_string(index=False))
    print("\n[DONE]")
    print(out_dir / "random_ablation_file_summary.csv")
    print(out_dir / "random_ablation_group_summary.csv")


if __name__ == "__main__":
    main()
