#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

try:
    from .build_paper import (
        PAPER_CSV, PAPER_FIGURES, display_method, set_data_root,
        safe_float, collect_main_robustness, collect_efficiency,
    )
except ImportError:
    from build_paper import (
        PAPER_CSV, PAPER_FIGURES, display_method, set_data_root,
        safe_float, collect_main_robustness, collect_efficiency,
    )

PAPER_CSV.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

PRIMARY_DATASET = "c4"
PRIMARY_MODEL = "llama3.1-8b-instruct"
TPR5_COL = "Avg_Attacked_TPR@5%FPR"

SHORT_LABELS = {
    "kgw": "KGW",
    "sta": "STA",
    "sweet": "SWEET",
    "dipmark": "DiPmark",
    "semstamp": "SemStamp",
    "ksemstamp": "k-SemStamp",
    "simmark_cosine": "Cosine-SimMark",
    "simmark_euclidean": "Euclidean-SimMark",
    "dualmark": "DualMark",
    "synguard": "SynGuard",
    "iasw": "IAS-W",
    "dual_sweet_iasw": "SWEET+IAS-W",
    "dual_dipmark_iasw": "DiPmark+IAS-W",
    "dual_kgw_iasw": "KGW+IAS-W",
}

METHODS = [
    "kgw", "sta", "sweet", "dipmark",
    "semstamp", "ksemstamp",
    "simmark_cosine", "simmark_euclidean",
    "dualmark", "synguard", "iasw",
    "dual_sweet_iasw", "dual_dipmark_iasw",
    "dual_kgw_iasw",
]

OURS = "dual_kgw_iasw"

# 统一颜色：普通方法不再随 matplotlib color cycle 变成多种颜色
BASE_COLOR = "#4C78A8"
OURS_COLOR = "#2A9D55"

FIGSIZE = (6.35, 4.65)

# 手工避让。格式：
# method: (dx, dy, horizontalalignment, verticalalignment)

LABEL_POS = {
    # 左上密集区域
    "dualmark":            ( 8,  0, "left",  "center"),
    "synguard":            ( 8,  0, "left",  "center"),
    "sweet":               ( 8,  1, "left",  "center"),
    "kgw":                 ( 8,  -2, "left",  "center"),
    "sta":                 ( 8,  0, "left","center"),

    # 左下
    "dipmark":             ( 8, 0, "left",  "center"),
    "dual_dipmark_iasw":   ( 8, 0, "left",  "center"),

    # 顶部
    "dual_kgw_iasw":       ( 8, 0, "left",  "center"),
    "dual_sweet_iasw":     ( 8, 0, "left",  "center"),

    # 中右区域
    "simmark_cosine":      ( 8, 0, "left",  "center"),
    "simmark_euclidean":   ( 8, 0, "left",  "center"),
    "ksemstamp":           ( 8, 0, "left",  "center"),
    "iasw":                ( 8, 0, "left",  "center"),

    # 最右：放到点左侧
    "semstamp":            (-8, 0, "right", "center"),
}

def method_label(key):
    return SHORT_LABELS.get(key, display_method(key))


def configure_style(args):
    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.labelsize": args.label_size,
        "axes.titlesize": args.label_size,
        "xtick.labelsize": args.tick_size,
        "ytick.labelsize": args.tick_size,
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans"],
        "mathtext.fontset": "stixsans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
    })


def style_axis(ax):
    ax.tick_params(
        axis="both", which="major",
        direction="in", length=4.0, width=0.8, pad=4
    )
    ax.tick_params(
        axis="both", which="minor",
        direction="in", length=2.5, width=0.7
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


def save_figure(fig, stem, args):
    pdf = PAPER_FIGURES / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[DONE] {pdf}")

    if not args.no_eps:
        eps = PAPER_FIGURES / f"{stem}.eps"
        fig.savefig(eps, format="eps", bbox_inches="tight")
        print(f"[DONE] {eps}")

    if not args.no_tiff:
        tiff = PAPER_FIGURES / f"{stem}.tiff"
        fig.savefig(
            tiff, format="tiff", dpi=args.tiff_dpi,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        print(f"[DONE] {tiff}")


def ensure_source_csvs():
    main_csv = PAPER_CSV / "main_robustness_source.csv"
    eff_csv = PAPER_CSV / "efficiency_source.csv"

    if not main_csv.exists():
        collect_main_robustness()
    if not eff_csv.exists():
        collect_efficiency()


def load_main_robustness_df():
    ensure_source_csvs()
    return pd.read_csv(
        PAPER_CSV / "main_robustness_source.csv",
        encoding="utf-8-sig"
    )


def load_efficiency_df():
    ensure_source_csvs()
    return pd.read_csv(
        PAPER_CSV / "efficiency_source.csv",
        encoding="utf-8-sig"
    )


def add_common_args(p):
    p.add_argument("--font-size", type=float, default=10.8)
    p.add_argument("--label-size", type=float, default=12.0)
    p.add_argument("--tick-size", type=float, default=10.6)
    p.add_argument("--legend-size", type=float, default=9.6)
    p.add_argument("--line-width", type=float, default=1.5)
    p.add_argument("--ours-line-width", type=float, default=2.8)
    p.add_argument("--marker-size", type=float, default=4.8)
    p.add_argument("--ours-marker-size", type=float, default=8.5)
    p.add_argument("--tiff-dpi", type=int, default=600)
    p.add_argument("--show-title", action="store_true")
    p.add_argument("--no-eps", action="store_true")
    p.add_argument("--no-tiff", action="store_true")
    return p


def main():
    global PAPER_CSV
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument("--data-root", default=None,
                   help="Released data/ directory (flat records layout).")
    args = p.parse_args()
    if args.data_root:
        import sys
        set_data_root(args.data_root)
        _bp = (sys.modules.get("build_paper")
               or sys.modules.get(f"{__package__}.build_paper"))
        PAPER_CSV = _bp.PAPER_CSV
    configure_style(args)

    rob = load_main_robustness_df()
    eff = load_efficiency_df()

    # Normalize dataset/model strings to avoid case/name-format mismatches
    rob = rob.copy()
    rob["_dataset_norm"] = rob["dataset"].astype(str).str.strip().str.lower()
    rob["_model_norm"] = rob["model"].astype(str).str.strip().str.lower()

    dataset_key = PRIMARY_DATASET.strip().lower()
    model_key = PRIMARY_MODEL.strip().lower()

    sub = rob[
        (rob["_dataset_norm"] == dataset_key) &
        (rob["_model_norm"] == model_key)
    ][["method_key", TPR5_COL]]

    if sub.empty:
        print("[ERROR] No robustness rows matched.")
        print("Requested dataset key:", PRIMARY_DATASET)
        print("Requested model  :", PRIMARY_MODEL)
        print("Available datasets:", sorted(rob["dataset"].dropna().astype(str).unique().tolist()))
        print("Available models  :", sorted(rob["model"].dropna().astype(str).unique().tolist()))
        raise SystemExit(1)

    merged = pd.merge(
        sub,
        eff[["method_key", "mean", "median", "success_rate"]],
        on="method_key",
        how="inner",
    )

    merged = merged[merged.method_key.isin(METHODS)].copy()

    if merged.empty:
        print("[ERROR] Robustness rows were found, but none matched efficiency rows/method keys.")
        print("Robustness methods:", sorted(sub["method_key"].dropna().astype(str).unique().tolist()))
        print("Efficiency methods:", sorted(eff["method_key"].dropna().astype(str).unique().tolist()))
        raise SystemExit(1)

    # 让普通方法先画，ours 最后画，避免星形被遮挡
    merged["_is_ours"] = merged["method_key"].eq(OURS)
    merged = merged.sort_values("_is_ours")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for _, r in merged.iterrows():
        m = r.method_key
        x = safe_float(r["mean"])
        y = safe_float(r[TPR5_COL])

        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        y_pct = y * 100.0
        ours = (m == OURS)

        ax.scatter(
            x,
            y_pct,
            s=135 if ours else 34,
            marker="*" if ours else "o",
            c=OURS_COLOR if ours else BASE_COLOR,
            edgecolors="white",
            linewidths=0.65 if not ours else 0.8,
            zorder=10 if ours else 4,
        )

        dx, dy, ha, va = LABEL_POS.get(
            m, (6, 5, "left", "bottom")
        )

        ann = ax.annotate(
            method_label(m),
            xy=(x, y_pct),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=args.legend_size,
            color="black",
            zorder=12,
            annotation_clip=False,
        )

        # 给文字加很薄的白色描边，避免虚线网格穿过文字。
        ann.set_path_effects([
            pe.withStroke(linewidth=2.2, foreground="white")
        ])

    ax.set_xlabel("Mean Generation Time (s)")
    ax.set_ylabel("Average Attacked TPR@5%FPR (%)")

    if args.show_title:
        ax.set_title(
            "Efficiency–Robustness Trade-off — TPR@5%FPR "
            "(RealNews, Llama-3.1-8B-Instruct)"
        )

    # 更淡的网格，避免和文本竞争视觉注意力
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.65,
        alpha=0.22,
        zorder=0,
    )

    style_axis(ax)

    # 给右侧 SemStamp 和顶部 KGW+IAS-W 留一点空间
    finite_x = pd.to_numeric(merged["mean"], errors="coerce").to_numpy(dtype=float)
    finite_y = pd.to_numeric(merged[TPR5_COL], errors="coerce").to_numpy(dtype=float) * 100.0

    valid = np.isfinite(finite_x) & np.isfinite(finite_y)
    if not np.any(valid):
        print("[ERROR] Matching rows exist, but all x/y values are non-finite.")
        print(merged[["method_key", "mean", TPR5_COL]].to_string(index=False))
        raise SystemExit(1)

    xmin = np.min(finite_x[valid])
    xmax = np.max(finite_x[valid])
    ymin = np.min(finite_y[valid])
    ymax = np.max(finite_y[valid])

    xpad = max(4.0, 0.035 * (xmax - xmin))
    ypad_bottom = 1.2
    ypad_top = 1.6

    ax.set_xlim(max(0.0, xmin - xpad), xmax + xpad)
    ax.set_ylim(ymin - ypad_bottom, ymax + ypad_top)

    # 图中每个点已经直接标注方法名，因此删除底部 legend。
    # 这样既避免冗余，也不存在 legend 颜色与实际点颜色不一致的问题。

    fig.subplots_adjust(
        left=0.15,
        right=0.985,
        top=0.95 if not args.show_title else 0.89,
        bottom=0.14,
    )

    save_figure(fig, "figure6_efficiency_tradeoff_tpr5", args)
    plt.close(fig)


if __name__ == "__main__":
    main()
