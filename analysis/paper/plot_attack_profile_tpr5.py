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
        MAIN_ROOT, PAPER_CSV, PAPER_FIGURES, display_method,
        discover_record_files, load_jsonl, safe_float, set_data_root,
        quality_values_from_record, collect_main_robustness, collect_efficiency,
    )
except ImportError:
    from build_paper import (
        MAIN_ROOT, PAPER_CSV, PAPER_FIGURES, display_method,
        discover_record_files, load_jsonl, safe_float, set_data_root,
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
    "simmark_cosine": "Cos.-SimMark", "simmark_euclidean": "Euc.-SimMark",
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
    p.add_argument("--font-size", type=float, default=11.5)
    p.add_argument("--label-size", type=float, default=12.4)
    p.add_argument("--tick-size", type=float, default=11.0)
    p.add_argument("--legend-size", type=float, default=10.2)
    p.add_argument("--line-width", type=float, default=1.15)
    p.add_argument("--kgw-line-width", type=float, default=1.55)
    p.add_argument("--ours-line-width", type=float, default=2.45)
    p.add_argument("--marker-size", type=float, default=4.2)
    p.add_argument("--ours-marker-size", type=float, default=7.8)
    p.add_argument("--tiff-dpi", type=int, default=600)
    p.add_argument("--show-title", action="store_true")
    p.add_argument("--no-eps", action="store_true")
    p.add_argument("--no-tiff", action="store_true")
    return p

ATTACKS = ["typo","drop","swap","del_ins","pegasus","dipper","back_translation","copy_paste","sentence_reorder"]
ATTACK_DISPLAY = {"typo":"Typo","drop":"Drop","swap":"Swap","del_ins":"Del-Ins","pegasus":"PEGASUS","dipper":"DIPPER","back_translation":"BackTrans","copy_paste":"Copy-Paste","sentence_reorder":"Sent-Reorder"}
METHODS = ["kgw","sta","sweet","dipmark","semstamp","ksemstamp","simmark_cosine","simmark_euclidean","dualmark","synguard","iasw","dual_kgw_iasw"]
OURS = "dual_kgw_iasw"
KGW = "kgw"
_COLOR_POOL = plt.get_cmap("tab20").colors
METHOD_COLORS = {m: _COLOR_POOL[i] for i, m in enumerate(METHODS)}

# ---- Layout knobs: edit here for Fig. 3 ----
FIGSIZE = (6.3, 4.6)
LEGEND_NCOL = 4
LEGEND_Y = -0.305
BOTTOM = 0.40
XTICK_ROTATION = 28


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
    df = load_main_robustness_df()
    sub = df[(df["dataset"] == PRIMARY_DATASET) & (df["model"] == PRIMARY_MODEL)].copy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(ATTACKS))
    for m in METHODS:
        row = sub[sub["method_key"] == m]
        if row.empty: continue
        row = row.iloc[0]
        ys = []
        for a in ATTACKS:
            v = safe_float(row.get(f"{a}_TPR@5%FPR", np.nan))
            ys.append(v * 100 if np.isfinite(v) else np.nan)
        ours = m == OURS
        is_kgw = m == KGW
        linewidth = args.ours_line_width if ours else (args.kgw_line_width if is_kgw else args.line_width)
        alpha = 1.0 if (ours or is_kgw) else 0.76
        ax.plot(
            x, ys,
            marker="*" if ours else "o",
            color=METHOD_COLORS[m],
            linewidth=linewidth,
            markersize=args.ours_marker_size if ours else args.marker_size,
            alpha=alpha,
            label=method_label(m),
            zorder=12 if ours else (7 if is_kgw else 3),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACKS], rotation=XTICK_ROTATION, ha="right", rotation_mode="anchor")
    ax.set_ylabel("TPR@5%FPR (%)")
    ax.set_xlabel("Attack Types", labelpad=2)
    if args.show_title:
        ax.set_title("Attack-wise Robustness — TPR@5%FPR (RealNews, Llama-3.1-8B-Instruct)")
    ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.22); style_axis(ax)
    ax.legend(frameon=False, ncol=LEGEND_NCOL, loc="upper center",
              bbox_to_anchor=(0.5, LEGEND_Y), columnspacing=0.85,
              handlelength=1.8, handletextpad=0.45, borderaxespad=0.0)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95 if not args.show_title else 0.89, bottom=BOTTOM)
    save_figure(fig, "figure3_attack_profile_tpr5", args); plt.close(fig)

if __name__ == "__main__":
    main()
