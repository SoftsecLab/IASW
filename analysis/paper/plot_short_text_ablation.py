#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot sentence-count ablation on RealNews/C4 using TPR@5%FPR.

Expected input:
  results/ablations/sentence_count/c4_llama3.1-8b-instruct_sentences/
      sentence_count_metrics.csv

Outputs PDF, EPS and TIFF. Style follows the paper plotting utilities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    # Also support running from a repository whose root contains results/.
    for candidate in [start, *start.parents]:
        if (candidate / "results").is_dir() and (candidate / "analysis").is_dir():
            return candidate
    return Path.cwd()


def configure_style(args):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial"],
        "mathtext.fontset": "stixsans",
        "font.size": args.font_size,
        "axes.labelsize": args.label_size,
        "axes.titlesize": args.label_size,
        "xtick.labelsize": args.tick_size,
        "ytick.labelsize": args.tick_size,
        "legend.fontsize": args.legend_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
    })


def style_axis(ax):
    ax.tick_params(axis="both", which="major", direction="in",
                   length=4.0, width=0.8, pad=4)
    ax.tick_params(axis="both", which="minor", direction="in",
                   length=2.5, width=0.7)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TPR@5%FPR against the number of retained sentences."
    )
    parser.add_argument(
        "--input-csv", type=Path, default=None,
        help="sentence_count_metrics.csv; defaults to the standard C4 result path."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory; defaults to <project>/paper/figures."
    )
    parser.add_argument("--stem", default="figure7_short_text_ablation_tpr5")
    parser.add_argument("--font-size", type=float, default=11.2)
    parser.add_argument("--label-size", type=float, default=12.2)
    parser.add_argument("--tick-size", type=float, default=10.8)
    parser.add_argument("--legend-size", type=float, default=10.2)
    parser.add_argument("--line-width", type=float, default=1.7)
    parser.add_argument("--ours-line-width", type=float, default=2.8)
    parser.add_argument("--marker-size", type=float, default=4.8)
    parser.add_argument("--ours-marker-size", type=float, default=7.0)
    parser.add_argument("--tiff-dpi", type=int, default=600)
    parser.add_argument("--show-title", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = find_project_root(Path.cwd())
    if args.input_csv is None:
        input_csv = (
            project_root / "results" / "ablations" / "sentence_count"
            / "c4_llama3.1-8b-instruct_sentences"
            / "sentence_count_metrics.csv"
        )
    else:
        input_csv = args.input_csv.expanduser().resolve()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "paper" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_csv.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Pass --input-csv with the path to sentence_count_metrics.csv."
        )

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    required = {"sentence_count", "channel", "tpr_at_5pct_fpr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower().isin(["c4", "realnews"])].copy()
    df["sentence_count"] = pd.to_numeric(df["sentence_count"], errors="coerce")
    df["tpr_at_5pct_fpr"] = pd.to_numeric(df["tpr_at_5pct_fpr"], errors="coerce")
    df["channel"] = df["channel"].astype(str).str.lower()
    df = df.dropna(subset=["sentence_count", "tpr_at_5pct_fpr"])
    df = df[df["channel"].isin(["iasw", "fisher"])]

    # If duplicate rows exist, fail loudly rather than silently mixing runs/cohorts.
    dup = df.duplicated(["sentence_count", "channel"], keep=False)
    if dup.any():
        raise ValueError(
            "Multiple rows found for a sentence_count/channel pair. "
            "Filter the CSV to one experiment batch/cohort before plotting."
        )
    if df.empty:
        raise ValueError("No C4/RealNews rows for KGW, IAS-W and Fisher were found.")

    labels = {
        "kgw": "KGW",
        "iasw": "IAS-W",
        "fisher": "KGW+IAS-W (Fisher Fusion)",
    }
    order = ["kgw", "iasw", "fisher"]
    configure_style(args)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for channel in order:
        sub = df[df["channel"] == channel].sort_values("sentence_count")
        if sub.empty:
            continue
        is_ours = channel == "fisher"
        ax.plot(
            sub["sentence_count"].to_numpy(),
            sub["tpr_at_5pct_fpr"].to_numpy() * 100.0,
            marker="o",
            linewidth=args.ours_line_width if is_ours else args.line_width,
            markersize=args.ours_marker_size if is_ours else args.marker_size,
            label=labels[channel],
        )

    ax.set_xlabel("Number of Retained Sentences", labelpad=6)
    ax.set_ylabel("TPR@5%FPR (%)", labelpad=6)
    ax.set_xticks(sorted(df["sentence_count"].unique()))
    ax.set_ylim(0, 103)
    ax.grid(True, linestyle="--", alpha=0.35)
    style_axis(ax)
    if args.show_title:
        ax.set_title("Short-Text Detection on RealNews")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=2, frameon=False, handlelength=2.4, columnspacing=1.2
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.30)

    pdf = output_dir / f"{args.stem}.pdf"
    eps = output_dir / f"{args.stem}.eps"
    tiff = output_dir / f"{args.stem}.tiff"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    fig.savefig(tiff, format="tiff", dpi=args.tiff_dpi, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[DONE] {pdf}")
    print(f"[DONE] {eps}")
    print(f"[DONE] {tiff}")


if __name__ == "__main__":
    main()
