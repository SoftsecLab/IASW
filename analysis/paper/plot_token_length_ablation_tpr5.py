#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .build_paper import (
        MAIN_ROOT, PAPER_CSV, PAPER_FIGURES, display_method, set_data_root,
        discover_record_files, load_jsonl, safe_float,
        quality_values_from_record, collect_main_robustness, collect_efficiency,
    )
except ImportError:
    from build_paper import (
        MAIN_ROOT, PAPER_CSV, PAPER_FIGURES, display_method, set_data_root,
        discover_record_files, load_jsonl, safe_float,
        quality_values_from_record, collect_main_robustness, collect_efficiency,
    )

PAPER_CSV.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
PRIMARY_DATASET = "c4"
PRIMARY_MODEL = "llama3.1-8b-instruct"
TPR5_COL = "Avg_Attacked_TPR@5%FPR"

SHORT_LABELS = {
    "kgw": "KGW", "sta": "STA", "sweet": "SWEET", "dipmark": "DiPmark",
    "semstamp": "SemStamp", "ksemstamp": "k-SemStamp",
    "simmark_cosine": "Cosine-SimMark", "simmark_euclidean": "Euclidean-SimMark",
    "dualmark": "DualMark", "synguard": "SynGuard", "iasw": "IAS-W",
    "dual_sweet_iasw": "SWEET+IAS-W", "dual_dipmark_iasw": "DiPmark+IAS-W",
    "dual_kgw_iasw": "KGW+IAS-W",
}

def method_label(key):
    return SHORT_LABELS.get(key, display_method(key))

def configure_style(args):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": args.font_size, "axes.labelsize": args.label_size,
        "axes.titlesize": args.label_size, "xtick.labelsize": args.tick_size,
        "ytick.labelsize": args.tick_size, "legend.fontsize": args.legend_size,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": False, "ytick.right": False,
    })

def style_axis(ax):
    ax.tick_params(axis="both", which="major", direction="in", length=4.0, width=0.8, pad=4)
    ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.7)

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
        fig.savefig(tiff, format="tiff", dpi=args.tiff_dpi, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
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
    return pd.read_csv(PAPER_CSV / "main_robustness_source.csv", encoding="utf-8-sig")

def load_efficiency_df():
    ensure_source_csvs()
    return pd.read_csv(PAPER_CSV / "efficiency_source.csv", encoding="utf-8-sig")

def add_common_args(p):
    p.add_argument("--font-size", type=float, default=11.2)
    p.add_argument("--label-size", type=float, default=12.2)
    p.add_argument("--tick-size", type=float, default=10.8)
    p.add_argument("--legend-size", type=float, default=10.2)
    p.add_argument("--line-width", type=float, default=1.5)
    p.add_argument("--ours-line-width", type=float, default=2.8)
    p.add_argument("--marker-size", type=float, default=4.8)
    p.add_argument("--ours-marker-size", type=float, default=8.5)
    p.add_argument("--tiff-dpi", type=int, default=600)
    p.add_argument("--show-title", action="store_true")
    p.add_argument("--no-eps", action="store_true")
    p.add_argument("--no-tiff", action="store_true")
    return p

FIGSIZE=(5.6,4.2)
LEGEND_NCOL=2
LEGEND_Y=-0.20
BOTTOM=0.30

def load_ablation_csv(name):
    path=PAPER_CSV/name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run build_paper.py --only ablation first.")
    return pd.read_csv(path,encoding="utf-8-sig")

LEGEND_LABELS = {
    "KGW Only": "KGW Only",
    "Token-Level Statistic (KGW+IAS-W)": "Token-Level Statistic",
    "IAS-W Alignment Statistic (KGW+IAS-W)": "IAS-W Alignment Statistic",
    "KGW+IAS-W (Fisher Fusion)": "Fisher Fusion",
}

def plot_discrete(df, method_order, xlabel, stem, args):
    values=sorted(pd.to_numeric(df["value"],errors="coerce").dropna().unique())
    x=np.arange(len(values),dtype=float)
    fig,ax=plt.subplots(figsize=FIGSIZE)
    for method in method_order:
        g=df[df["Method"]==method].copy()
        if g.empty: continue
        g["value"]=pd.to_numeric(g["value"],errors="coerce")
        g[TPR5_COL]=pd.to_numeric(g[TPR5_COL],errors="coerce")
        mapping={float(r["value"]):float(r[TPR5_COL])*100 for _,r in g.iterrows() if np.isfinite(r["value"]) and np.isfinite(r[TPR5_COL])}
        y=[mapping.get(float(v),np.nan) for v in values]
        ax.plot(x,y,marker="o",linewidth=args.line_width,markersize=args.marker_size,label=LEGEND_LABELS.get(method, method))
    ax.set_xticks(x); ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in values])
    ax.set_xlabel(xlabel,labelpad=6); ax.set_ylabel("Average TPR@5%FPR (%)",labelpad=6)
    ax.grid(True,linestyle="--",alpha=.35); style_axis(ax)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,LEGEND_Y),ncol=LEGEND_NCOL,frameon=False,handlelength=2.4,columnspacing=1.4)
    fig.subplots_adjust(left=.15,right=.98,top=.96,bottom=BOTTOM)
    save_figure(fig,stem,args); plt.close(fig)

METHOD_ORDER=["KGW Only","Token-Level Statistic (KGW+IAS-W)","IAS-W Alignment Statistic (KGW+IAS-W)","KGW+IAS-W (Fisher Fusion)"]
def main():
    global MAIN_ROOT, PAPER_CSV
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument("--data-root", default=None,
                   help="Released data/ directory (flat records layout).")
    args = p.parse_args()
    if args.data_root:
        import sys
        set_data_root(args.data_root)
        _bp = (sys.modules.get("build_paper")
               or sys.modules.get(f"{__package__}.build_paper"))
        MAIN_ROOT = _bp.MAIN_ROOT
        PAPER_CSV = _bp.PAPER_CSV
    configure_style(args)
    plot_discrete(load_ablation_csv("token_length_ablation_source.csv"), METHOD_ORDER, "Target Length", "figure9_token_length_ablation_tpr5", args)
if __name__ == "__main__": main()
