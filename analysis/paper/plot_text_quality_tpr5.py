#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Fig. 5 plotter for the IAS-W manuscript.

This script is intentionally separated from plot_manuscript_figures.py.
It only draws the two quality boxplots used in Fig. 5:
    left:  Perplexity (PPL)
    right: Semantic Loss = 1 - CleanSim (or 1 - RefSim)

Default data:
    datasets: c4, wiki, dolly
    display:  RealNews, Wiki, DOLLY
    model:    llama3.1-8b-instruct
"""

from __future__ import annotations

import argparse
from typing import Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .build_paper import (
        MAIN_ROOT,
        PAPER_FIGURES,
        discover_record_files,
        load_jsonl,
        safe_float,
        set_data_root,
        records_dir_of,
        quality_values_from_record,
    )
except ImportError:
    from build_paper import (
        MAIN_ROOT,
        PAPER_FIGURES,
        discover_record_files,
        load_jsonl,
        safe_float,
        set_data_root,
        records_dir_of,
        quality_values_from_record,
    )

PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "llama3.1-8b-instruct"
DEFAULT_DATASETS = ["c4", "wiki", "dolly"]

METHODS = [
    "kgw",
    "sta",
    "sweet",
    "dipmark",
    "semstamp",
    "ksemstamp",
    "simmark_cosine",
    "simmark_euclidean",
    "dualmark",
    "synguard",
    "iasw",
    "dual_kgw_iasw",
]

# Figure-only abbreviations for the two longest labels.
METHOD_DISPLAY = {
    "kgw": "KGW",
    "sta": "STA",
    "sweet": "SWEET",
    "dipmark": "DiPmark",
    "semstamp": "SemStamp",
    "ksemstamp": "k-SemStamp",
    "simmark_cosine": "Cos.-SimMark",
    "simmark_euclidean": "Euc.-SimMark",
    "dualmark": "DualMark",
    "synguard": "SynGuard",
    "iasw": "IAS-W",
    "dual_kgw_iasw": "KGW+IAS-W",
}

DATASET_DISPLAY = {
    "c4": "RealNews",
    "wiki": "Wiki",
    "dolly": "DOLLY",
}


def parse_args():
    p = argparse.ArgumentParser("Plot manuscript Fig. 5 quality boxplots")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--data-root",
        default=None,
        help="Released data/ directory (flat records layout).",
    )
    p.add_argument(
        "--datasets",
        default="c4,wiki,dolly",
        help="Comma-separated dataset keys pooled in Fig. 5",
    )
    p.add_argument(
        "--semantic-source",
        default="clean",
        choices=["clean", "human"],
        help="clean -> 1-CleanSim; human -> 1-RefSim",
    )
    p.add_argument("--font-size", type=float, default=11.5)
    p.add_argument("--label-size", type=float, default=12.0)
    p.add_argument("--tick-size", type=float, default=9.8)
    p.add_argument("--title-size", type=float, default=11.2)
    p.add_argument("--tiff-dpi", type=int, default=600)
    p.add_argument("--show-title", action="store_true")
    p.add_argument("--no-eps", action="store_true")
    p.add_argument("--no-tiff", action="store_true")
    return p.parse_args()


def configure_style(args):
    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.labelsize": args.label_size,
        "axes.titlesize": args.title_size,
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
        axis="both",
        which="major",
        direction="in",
        length=4.0,
        width=0.8,
        pad=4,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


def save_figure(fig, stem: str, args):
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
            tiff,
            format="tiff",
            dpi=args.tiff_dpi,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        print(f"[DONE] {tiff}")


def filter_outliers_iqr(values: Iterable[float], k: float = 3.0) -> np.ndarray:
    """
    Remove only extreme plotting outliers so a few abnormal records do not
    crush the displayed scale. This does not modify the underlying data files.
    """
    arr = np.asarray([safe_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size < 8:
        return arr

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1

    if not np.isfinite(iqr) or iqr <= 1e-12:
        lo, hi = np.percentile(arr, [1, 99])
    else:
        lo = q1 - k * iqr
        hi = q3 + k * iqr

    kept = arr[(arr >= lo) & (arr <= hi)]

    # Avoid pathological over-filtering.
    if kept.size >= max(5, int(0.60 * arr.size)):
        return kept

    return arr


def collect_quality_long(
    model: str,
    datasets: List[str],
    methods: List[str],
    semantic_source: str = "clean",
) -> pd.DataFrame:
    rows = []

    for dataset in datasets:
        record_dir = records_dir_of(MAIN_ROOT / dataset / model)
        if not record_dir.exists():
            print(f"[WARN] record dir missing: {record_dir}")
            continue

        record_map = discover_record_files(record_dir)

        for method_key in methods:
            path = record_map.get(method_key)
            if path is None:
                print(f"[WARN] quality missing: {dataset}/{model}/{method_key}")
                continue

            for obj in load_jsonl(path):
                vals = quality_values_from_record(obj)
                if vals is None:
                    continue

                ppl = safe_float(vals.get("PPL"))
                coherence = safe_float(vals.get("Coherence"))
                sim_clean = safe_float(vals.get("SimClean"))

                sim_used = coherence if semantic_source == "human" else sim_clean
                semantic_loss = (
                    1.0 - sim_used if np.isfinite(sim_used) else np.nan
                )

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "method_key": method_key,
                    "PPL": ppl,
                    "SemanticLoss": semantic_loss,
                })

    return pd.DataFrame(rows)


def build_boxplot_data(
    df: pd.DataFrame,
    metric: str,
    methods: List[str],
):
    data = []
    labels = []

    for method in methods:
        vals = df.loc[df["method_key"] == method, metric].dropna().values
        vals = filter_outliers_iqr(vals)
        if len(vals) == 0:
            continue

        data.append(vals)
        labels.append(METHOD_DISPLAY[method])

    return data, labels


def nudge_dualmark_tick(ax, dx: float = 0.18):
    """
    Move the 'DualMark' x tick label slightly to the right to avoid
    overlapping with the label on its left after rotation.
    """
    for label in ax.get_xticklabels():
        if label.get_text() == "DualMark":
            label.set_x(label.get_position()[0] + dx)


def main():
    global MAIN_ROOT
    args = parse_args()
    if getattr(args, "data_root", None):
        import sys
        set_data_root(args.data_root)
        _bp = (sys.modules.get("build_paper")
               or sys.modules.get(f"{__package__}.build_paper"))
        MAIN_ROOT = _bp.MAIN_ROOT
    configure_style(args)

    datasets = [
        x.strip()
        for x in args.datasets.split(",")
        if x.strip()
    ]

    df = collect_quality_long(
        model=args.model,
        datasets=datasets,
        methods=METHODS,
        semantic_source=args.semantic_source,
    )

    if df.empty:
        raise RuntimeError("No quality data found for Fig. 5.")

    ppl_data, ppl_labels = build_boxplot_data(df, "PPL", METHODS)
    sem_data, sem_labels = build_boxplot_data(df, "SemanticLoss", METHODS)

    # Two-panel wide layout. The large global title is intentionally omitted
    # because the paper caption already states datasets and model.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.95))

    # Keep mild outliers visible but reduce their visual weight.
    flierprops = dict(
        marker="o",
        markersize=1.7,
        markerfacecolor="none",
        markeredgewidth=0.30,
    )
    medianprops = dict(linewidth=1.25)
    boxprops = dict(linewidth=0.85)
    whiskerprops = dict(linewidth=0.8)
    capprops = dict(linewidth=0.8)

    axes[0].boxplot(
        ppl_data,
        tick_labels=ppl_labels,
        patch_artist=False,
        showfliers=True,
        widths=0.55,
        flierprops=flierprops,
        medianprops=medianprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )
    axes[0].set_title("Perplexity")
    axes[0].set_ylabel("PPL")
    axes[0].grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.22)
    axes[0].tick_params(axis="x", rotation=32)
    for lab in axes[0].get_xticklabels():
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    style_axis(axes[0])

    sem_ylabel = (
        "1 - CleanSim"
        if args.semantic_source == "clean"
        else "1 - RefSim"
    )

    axes[1].boxplot(
        sem_data,
        tick_labels=sem_labels,
        patch_artist=False,
        showfliers=True,
        widths=0.55,
        flierprops=flierprops,
        medianprops=medianprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )
    axes[1].set_title("Semantic Loss")
    axes[1].set_ylabel(sem_ylabel)
    axes[1].grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.22)
    axes[1].tick_params(axis="x", rotation=32)
    for lab in axes[1].get_xticklabels():
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    style_axis(axes[1])

    if args.show_title:
        dataset_note = ", ".join(
            DATASET_DISPLAY.get(x, x) for x in datasets
        )
        fig.suptitle(
            f"Text Quality Comparison "
            f"({dataset_note}, Llama-3.1-8B-Instruct)",
            y=0.995,
        )

    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        top=0.92 if not args.show_title else 0.87,
        bottom=0.285,
        wspace=0.22,
    )

    save_figure(fig, "figure5_quality_boxplots", args)
    plt.close(fig)


if __name__ == "__main__":
    main()
