# -*- coding: utf-8 -*-
"""
Analyze whether IAS-W truly rescues KGW-hard samples using EXISTING
dual_kgw_iasw.jsonl records only.

No generation, no attack, and no model loading are performed.

Core questions
--------------
1. When the KGW branch fails at 1% FPR, can IAS-W rescue those samples?
2. Does Fisher fusion rescue KGW failures, or does it instead lose samples
   that KGW already detects?
3. Is Qwen different from the other backbones in these conditional patterns?
4. Does a fixed macro weight alpha=0.25 improve the rescue/harm balance?

Expected input
--------------
results/main_exp/c4/<backbone>/records/dual_kgw_iasw.jsonl

Output
------
results/diagnostics/backbone/deeper/
    complementarity_attack_summary.csv
    complementarity_hard_subset.csv
    complementarity_sample_categories.csv
    complementarity_model_summary.csv
    figures/*.pdf
    figures/*.eps
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr, spearmanr


ATTACKS = [
    "typo", "drop", "swap", "del_ins", "pegasus", "dipper",
    "back_translation", "copy_paste",
]

MODELS = [
    "opt-1.3b-semstamp",
    "Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct",
    "qwen2.5-7b-instruct",
    "llama3.1-8b-instruct",
]

MODEL_DISPLAY = {
    "opt-1.3b-semstamp": "OPT-1.3B",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Mistral-7B-Instruct": "Mistral-7B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B",
    "llama3.1-8b-instruct": "Llama-3.1-8B",
}


def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
                if isinstance(x, dict):
                    x["_line_idx"] = line_idx
                    rows.append(x)
            except Exception:
                pass
    return rows


def get_nested(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def recursive_find(obj, key):
    vals = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                vals.append(v)
            vals.extend(recursive_find(v, key))
    elif isinstance(obj, list):
        for x in obj:
            vals.extend(recursive_find(x, key))
    return vals


def first_numeric(values):
    for x in values:
        v = safe_float(x)
        if np.isfinite(v):
            return v
    return np.nan


def extract_dual(det):
    if not isinstance(det, dict):
        return None

    z_micro = first_numeric([
        det.get("z_score_micro"),
        get_nested(det, ["details", "micro_details", "z_score"]),
        get_nested(det, ["details", "micro_details", "z_score_micro"]),
    ])

    z_macro = first_numeric([
        det.get("z_score_macro_calibrated"),
        det.get("z_score_macro"),
        get_nested(det, ["details", "macro_details", "z_score_macro_calibrated"]),
        get_nested(det, ["details", "macro_details", "z_score_macro"]),
        get_nested(det, ["details", "macro_details", "z_score"]),
    ])

    if not np.isfinite(z_macro):
        for x in recursive_find(det, "ablation_z_scores"):
            if isinstance(x, dict):
                v = safe_float(x.get("dp_local"))
                if np.isfinite(v):
                    z_macro = v
                    break

    if not (np.isfinite(z_micro) and np.isfinite(z_macro)):
        return None

    z_fisher = first_numeric([
        det.get("z_score_fisher"),
        det.get("z_score"),
    ])

    z_max = first_numeric([det.get("z_score_max")])
    if not np.isfinite(z_max):
        z_max = max(z_micro, z_macro)

    return {
        "z_micro": z_micro,
        "z_macro": z_macro,
        "z_fisher": z_fisher,
        "z_max": z_max,
    }


def attack_det(row, attack):
    a = row.get("attacks", {})
    if not isinstance(a, dict):
        return {}
    item = a.get(attack, {})
    if not isinstance(item, dict):
        return {}
    d = item.get("detection_results", {})
    return d if isinstance(d, dict) else {}


def record_id(row, fallback):
    for k in ["idx", "id", "sample_id", "record_idx", "index"]:
        if k in row:
            return row[k]
    return fallback


def build_long(rows, model):
    out = []

    for i, row in enumerate(rows):
        rid = record_id(row, i)

        base = [
            ("human", 0, "negative_human", row.get("detection_human", {})),
            ("no_wm", 0, "negative_no_wm", row.get("detection_no_wm", {})),
            ("wm", 1, "unattacked", row.get("detection_wm", {})),
        ]

        for source, label, condition, det in base:
            z = extract_dual(det)
            if z is not None:
                out.append({
                    "model": model,
                    "model_display": MODEL_DISPLAY.get(model, model),
                    "row_id": rid,
                    "label": label,
                    "source": source,
                    "condition": condition,
                    **z,
                })

        for attack in ATTACKS:
            z = extract_dual(attack_det(row, attack))
            if z is not None:
                out.append({
                    "model": model,
                    "model_display": MODEL_DISPLAY.get(model, model),
                    "row_id": rid,
                    "label": 1,
                    "source": attack,
                    "condition": attack,
                    **z,
                })

    return pd.DataFrame(out)


def clip_p(p):
    return np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16)


def weighted_fusion(z_micro, z_macro, alpha):
    """
    Weighted geometric aggregation of one-sided p-values.

    alpha=0   -> micro only
    alpha=1   -> macro only
    alpha=0.5 -> equal weighted log-p aggregation

    This is used as a diagnostic score only. It does not overwrite the
    original paper's stored Fisher score.
    """
    p_mi = clip_p(norm.sf(np.asarray(z_micro, dtype=float)))
    p_ma = clip_p(norm.sf(np.asarray(z_macro, dtype=float)))

    lp = (1.0 - alpha) * np.log(p_mi) + alpha * np.log(p_ma)
    return norm.isf(clip_p(np.exp(lp)))


def threshold_at_fpr(scores, fpr=0.01):
    x = np.asarray(scores, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, 1.0 - fpr))


def rate(mask):
    mask = np.asarray(mask, dtype=bool)
    return float(mask.mean()) if len(mask) else np.nan


def corr(x, y, kind="spearman"):
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


def summarize_condition(model_df, condition, alpha):
    neg = model_df[model_df["condition"] == "negative_human"].copy()
    pos = model_df[model_df["condition"] == condition].copy()

    if neg.empty or pos.empty:
        return None, None

    # Stored Fisher may occasionally be missing. Keep only valid cases per score.
    neg["z_alpha"] = weighted_fusion(neg["z_micro"], neg["z_macro"], alpha)
    pos["z_alpha"] = weighted_fusion(pos["z_micro"], pos["z_macro"], alpha)

    thresholds = {
        "micro": threshold_at_fpr(neg["z_micro"]),
        "macro": threshold_at_fpr(neg["z_macro"]),
        "fisher": threshold_at_fpr(neg["z_fisher"]),
        "alpha": threshold_at_fpr(neg["z_alpha"]),
    }

    pos["micro_pass"] = pos["z_micro"] >= thresholds["micro"]
    pos["macro_pass"] = pos["z_macro"] >= thresholds["macro"]
    pos["fisher_pass"] = pos["z_fisher"] >= thresholds["fisher"]
    pos["alpha_pass"] = pos["z_alpha"] >= thresholds["alpha"]

    micro_fail = ~pos["micro_pass"]
    micro_pass = pos["micro_pass"]

    # Rescue: micro failed, alternative detects it.
    macro_rescue = micro_fail & pos["macro_pass"]
    fisher_rescue = micro_fail & pos["fisher_pass"]
    alpha_rescue = micro_fail & pos["alpha_pass"]

    # Harm/loss: micro passed, fusion loses it.
    fisher_loss = micro_pass & ~pos["fisher_pass"]
    alpha_loss = micro_pass & ~pos["alpha_pass"]

    summary = {
        "model": model_df["model"].iloc[0],
        "model_display": model_df["model_display"].iloc[0],
        "condition": condition,
        "n_pos": len(pos),
        "micro_thr_1pct": thresholds["micro"],
        "macro_thr_1pct": thresholds["macro"],
        "fisher_thr_1pct": thresholds["fisher"],
        "alpha_thr_1pct": thresholds["alpha"],
        "micro_tpr1_pct": 100 * rate(pos["micro_pass"]),
        "macro_tpr1_pct": 100 * rate(pos["macro_pass"]),
        "fisher_tpr1_pct": 100 * rate(pos["fisher_pass"]),
        "alpha_tpr1_pct": 100 * rate(pos["alpha_pass"]),
        "micro_fail_n": int(micro_fail.sum()),
        "micro_pass_n": int(micro_pass.sum()),
        "macro_rescue_n": int(macro_rescue.sum()),
        "fisher_rescue_n": int(fisher_rescue.sum()),
        "alpha_rescue_n": int(alpha_rescue.sum()),
        "macro_rescue_rate_among_micro_fail_pct":
            100 * rate(pos.loc[micro_fail, "macro_pass"]),
        "fisher_rescue_rate_among_micro_fail_pct":
            100 * rate(pos.loc[micro_fail, "fisher_pass"]),
        "alpha_rescue_rate_among_micro_fail_pct":
            100 * rate(pos.loc[micro_fail, "alpha_pass"]),
        "fisher_loss_n": int(fisher_loss.sum()),
        "alpha_loss_n": int(alpha_loss.sum()),
        "fisher_loss_rate_among_micro_pass_pct":
            100 * rate(~pos.loc[micro_pass, "fisher_pass"]),
        "alpha_loss_rate_among_micro_pass_pct":
            100 * rate(~pos.loc[micro_pass, "alpha_pass"]),
        "fisher_net_vs_micro_pp":
            100 * (rate(pos["fisher_pass"]) - rate(pos["micro_pass"])),
        "alpha_net_vs_micro_pp":
            100 * (rate(pos["alpha_pass"]) - rate(pos["micro_pass"])),
        "pearson_micro_macro_pos": corr(pos["z_micro"], pos["z_macro"], "pearson"),
        "spearman_micro_macro_pos": corr(pos["z_micro"], pos["z_macro"], "spearman"),
    }

    # Sample-level categories
    categories = pos[[
        "model", "model_display", "row_id", "condition",
        "z_micro", "z_macro", "z_fisher", "z_alpha",
        "micro_pass", "macro_pass", "fisher_pass", "alpha_pass",
    ]].copy()

    def cat(row, fusion_col):
        if not row["micro_pass"] and row[fusion_col]:
            return "RESCUED"
        if row["micro_pass"] and not row[fusion_col]:
            return "LOST"
        if row["micro_pass"] and row[fusion_col]:
            return "BOTH_PASS"
        return "BOTH_FAIL"

    categories["fisher_category"] = categories.apply(
        lambda r: cat(r, "fisher_pass"), axis=1
    )
    categories["alpha_category"] = categories.apply(
        lambda r: cat(r, "alpha_pass"), axis=1
    )

    return summary, categories


def hard_subset_table(model_df, condition, quantiles=(0.10, 0.20, 0.30)):
    pos = model_df[model_df["condition"] == condition].copy()
    neg = model_df[model_df["condition"] == "negative_human"].copy()
    if pos.empty or neg.empty:
        return []

    micro_thr = threshold_at_fpr(neg["z_micro"])
    macro_thr = threshold_at_fpr(neg["z_macro"])

    rows = []
    for q in quantiles:
        cutoff = float(pos["z_micro"].quantile(q))
        hard = pos[pos["z_micro"] <= cutoff].copy()

        rows.append({
            "model": model_df["model"].iloc[0],
            "model_display": model_df["model_display"].iloc[0],
            "condition": condition,
            "bottom_fraction": q,
            "n_hard": len(hard),
            "micro_z_cutoff": cutoff,
            "micro_z_mean": hard["z_micro"].mean(),
            "macro_z_mean": hard["z_macro"].mean(),
            "macro_z_median": hard["z_macro"].median(),
            "macro_detect_rate_in_hard_pct":
                100 * rate(hard["z_macro"] >= macro_thr),
            "micro_detect_rate_in_hard_pct":
                100 * rate(hard["z_micro"] >= micro_thr),
            "spearman_micro_macro_in_hard":
                corr(hard["z_micro"], hard["z_macro"], "spearman"),
        })
    return rows


def save_vector(fig, base):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".eps"), bbox_inches="tight", facecolor="white")


def make_figures(summary_df, hard_df, categories_df, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Net gain/loss per model across attacks.
    if not summary_df.empty:
        tmp = (
            summary_df[summary_df["condition"].isin(ATTACKS)]
            .groupby(["model", "model_display"], as_index=False)[
                ["fisher_net_vs_micro_pp", "alpha_net_vs_micro_pp"]
            ].mean()
        )
        x = np.arange(len(tmp))
        w = 0.36
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.bar(x - w/2, tmp["fisher_net_vs_micro_pp"], width=w,
               label="Stored Fisher - Micro")
        ax.bar(x + w/2, tmp["alpha_net_vs_micro_pp"], width=w,
               label="alpha=0.25 - Micro")
        ax.axhline(0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(tmp["model_display"], rotation=18)
        ax.set_ylabel("Mean TPR change (percentage points)")
        ax.set_title("Fusion gain/loss relative to KGW branch")
        ax.legend(frameon=False)
        ax.grid(axis="y", linewidth=0.5)
        fig.subplots_adjust(bottom=0.22)
        save_vector(fig, fig_dir / "fusion_net_gain_by_model")
        plt.close(fig)

    # 2. Rescue vs loss counts for Qwen.
    q = summary_df[
        (summary_df["model"] == "qwen2.5-7b-instruct")
        & (summary_df["condition"].isin(ATTACKS))
    ].copy()
    if not q.empty:
        x = np.arange(len(q))
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        ax.bar(x - 0.18, q["fisher_rescue_n"], width=0.36,
               label="Fisher rescues")
        ax.bar(x + 0.18, -q["fisher_loss_n"], width=0.36,
               label="Fisher losses")
        ax.axhline(0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(q["condition"], rotation=28, ha="right")
        ax.set_ylabel("Samples (+ rescue / - loss)")
        ax.set_title("Qwen: conditional rescue vs harm")
        ax.legend(frameon=False)
        ax.grid(axis="y", linewidth=0.5)
        fig.subplots_adjust(bottom=0.28)
        save_vector(fig, fig_dir / "qwen_fisher_rescue_vs_loss")
        plt.close(fig)

    # 3. Macro detect rate in bottom-20% micro samples.
    if not hard_df.empty:
        h20 = hard_df[
            (hard_df["bottom_fraction"] == 0.20)
            & (hard_df["condition"].isin(ATTACKS))
        ]
        piv = h20.pivot_table(
            index="model_display",
            columns="condition",
            values="macro_detect_rate_in_hard_pct",
            aggfunc="mean",
        )
        if not piv.empty:
            fig, ax = plt.subplots(figsize=(10.5, 4.8))
            im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=100)
            ax.set_yticks(np.arange(len(piv.index)))
            ax.set_yticklabels(piv.index)
            ax.set_xticks(np.arange(len(piv.columns)))
            ax.set_xticklabels(piv.columns, rotation=30, ha="right")
            ax.set_title("IAS-W detection within bottom-20% KGW-score samples")
            fig.colorbar(im, ax=ax, label="Macro detection rate (%)")
            fig.subplots_adjust(bottom=0.28, left=0.16, right=0.92)
            fig.savefig(fig_dir / "macro_rescue_hard20.pdf",
                        bbox_inches="tight", facecolor="white")
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dataset", default="c4")
    ap.add_argument("--alpha", type=float, default=0.25,
                    help="Fixed macro weight for diagnostic weighted fusion.")
    ap.add_argument("--out", default="results/diagnostics/backbone/deeper")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    base = root / "results" / "main_exp" / args.dataset
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    long_parts = []
    for model in MODELS:
        p = base / model / "records" / "dual_kgw_iasw.jsonl"
        if not p.exists():
            print("[MISS]", p)
            continue
        rows = load_jsonl(p)
        d = build_long(rows, model)
        if not d.empty:
            long_parts.append(d)
            print(f"[LOAD] {MODEL_DISPLAY[model]:14s}: records={len(rows)}, scores={len(d)}")

    if not long_parts:
        raise RuntimeError("No dual_kgw_iasw.jsonl data could be loaded.")

    data = pd.concat(long_parts, ignore_index=True)

    summary_rows = []
    category_parts = []
    hard_rows = []

    for model in MODELS:
        mdf = data[data["model"] == model]
        if mdf.empty:
            continue

        for condition in ["unattacked"] + ATTACKS:
            summary, categories = summarize_condition(mdf, condition, args.alpha)
            if summary is not None:
                summary_rows.append(summary)
                category_parts.append(categories)
            hard_rows.extend(hard_subset_table(mdf, condition))

    summary_df = pd.DataFrame(summary_rows)
    categories_df = pd.concat(category_parts, ignore_index=True) if category_parts else pd.DataFrame()
    hard_df = pd.DataFrame(hard_rows)

    summary_df.to_csv(
        out_dir / "complementarity_attack_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    hard_df.to_csv(
        out_dir / "complementarity_hard_subset.csv",
        index=False, encoding="utf-8-sig"
    )
    categories_df.to_csv(
        out_dir / "complementarity_sample_categories.csv",
        index=False, encoding="utf-8-sig"
    )

    model_summary = (
        summary_df[summary_df["condition"].isin(ATTACKS)]
        .groupby(["model", "model_display"], as_index=False)
        .agg(
            mean_micro_tpr1_pct=("micro_tpr1_pct", "mean"),
            mean_macro_tpr1_pct=("macro_tpr1_pct", "mean"),
            mean_fisher_tpr1_pct=("fisher_tpr1_pct", "mean"),
            mean_alpha_tpr1_pct=("alpha_tpr1_pct", "mean"),
            mean_fisher_net_vs_micro_pp=("fisher_net_vs_micro_pp", "mean"),
            mean_alpha_net_vs_micro_pp=("alpha_net_vs_micro_pp", "mean"),
            mean_macro_rescue_rate_among_micro_fail_pct=(
                "macro_rescue_rate_among_micro_fail_pct", "mean"
            ),
            mean_fisher_rescue_rate_among_micro_fail_pct=(
                "fisher_rescue_rate_among_micro_fail_pct", "mean"
            ),
            mean_alpha_rescue_rate_among_micro_fail_pct=(
                "alpha_rescue_rate_among_micro_fail_pct", "mean"
            ),
            mean_fisher_loss_rate_among_micro_pass_pct=(
                "fisher_loss_rate_among_micro_pass_pct", "mean"
            ),
            mean_alpha_loss_rate_among_micro_pass_pct=(
                "alpha_loss_rate_among_micro_pass_pct", "mean"
            ),
        )
    )
    model_summary.to_csv(
        out_dir / "complementarity_model_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    make_figures(summary_df, hard_df, categories_df, out_dir)

    print("\n[DONE]")
    print("  ", out_dir / "complementarity_attack_summary.csv")
    print("  ", out_dir / "complementarity_hard_subset.csv")
    print("  ", out_dir / "complementarity_model_summary.csv")
    print("  ", out_dir / "figures")


if __name__ == "__main__":
    main()
