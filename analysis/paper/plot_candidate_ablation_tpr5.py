#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ablations"
    / "num_candidates"
)

PAPER_FIGURES = (
    PROJECT_ROOT
    / "paper"
    / "figures"
)

PAPER_FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Candidate settings
# ============================================================

CANDIDATE_VALUES = [8, 16, 32, 64]

EXCLUDED_ATTACKS = set()


# ============================================================
# Curves
# ============================================================

METHOD_ORDER = [
    "IAS-W Only",
    "Token-Level Statistic (KGW+IAS-W)",
    "IAS-W Alignment Statistic (KGW+IAS-W)",
    "KGW+IAS-W (Fisher Fusion)",
]

LEGEND_LABELS = {
    "IAS-W Only": "IAS-W Only",
    "Token-Level Statistic (KGW+IAS-W)": "Token-Level Statistic",
    "IAS-W Alignment Statistic (KGW+IAS-W)": "IAS-W Alignment Statistic",
    "KGW+IAS-W (Fisher Fusion)": "Fisher Fusion",
}


# ============================================================
# Plot layout
# ============================================================

FIGSIZE = (5.6, 4.2)

LEGEND_NCOL = 2
LEGEND_Y = -0.20
BOTTOM = 0.31


# ============================================================
# Arguments
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot candidate-pool ablation directly from "
            "original ROC metric JSON files."
        )
    )

    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=CANDIDATE_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_FIGURES,
    )

    parser.add_argument(
        "--font-size",
        type=float,
        default=11.2,
    )

    parser.add_argument(
        "--label-size",
        type=float,
        default=12.2,
    )

    parser.add_argument(
        "--tick-size",
        type=float,
        default=10.8,
    )

    parser.add_argument(
        "--legend-size",
        type=float,
        default=10.2,
    )

    parser.add_argument(
        "--line-width",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--ours-line-width",
        type=float,
        default=2.6,
    )

    parser.add_argument(
        "--marker-size",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--ours-marker-size",
        type=float,
        default=7.5,
    )

    parser.add_argument(
        "--tiff-dpi",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--show-title",
        action="store_true",
    )

    parser.add_argument(
        "--no-eps",
        action="store_true",
    )

    parser.add_argument(
        "--no-tiff",
        action="store_true",
    )

    return parser


# ============================================================
# Style
# ============================================================

def configure_style(args):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans"],
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
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=4.0,
        width=0.8,
        pad=4,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=2.5,
        width=0.7,
    )


# ============================================================
# JSON helpers
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected dict JSON in {path}"
        )

    return obj


def normalize_name(name: str) -> str:
    """
    Normalize condition / attack names for matching.
    """

    s = str(name).strip().lower()

    s = s.replace("-", "_")
    s = s.replace(" ", "_")

    while "__" in s:
        s = s.replace("__", "_")

    return s


def is_before_attack(name: str) -> bool:
    n = normalize_name(name)

    return (
        n in {
            "before",
            "before_attack",
            "clean",
            "unattacked",
            "no_attack",
        }
        or "before_attack" in n
    )


def is_excluded_attack(name: str) -> bool:
    n = normalize_name(name)

    return any(
        attack in n
        for attack in EXCLUDED_ATTACKS
    )


def extract_tpr5(metrics: Any) -> Optional[float]:
    """
    Extract TPR@5%FPR from a metric dictionary.

    Returns value in [0, 1] when successful.
    """

    if not isinstance(metrics, dict):
        return None

    possible_keys = [
        "TPR@5%FPR",
        "TPR@5% FPR",
        "TPR@5%fpr",
        "tpr@5%fpr",
        "TPR_5FPR",
        "tpr_5fpr",
    ]

    for key in possible_keys:
        if key not in metrics:
            continue

        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            continue

        if not np.isfinite(value):
            continue

        # Handle percentage-formatted JSON defensively.
        if value > 1.0:
            value /= 100.0

        return value

    return None


# ============================================================
# Metric block extraction
# ============================================================

def find_block(
    obj: Dict[str, Any],
    names: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Find a top-level metric block such as:
        Micro
        Macro
        Fisher

    Matching is case-insensitive.
    """

    normalized = {
        normalize_name(k): k
        for k in obj.keys()
    }

    for name in names:
        n = normalize_name(name)

        if n in normalized:
            value = obj[normalized[n]]

            if isinstance(value, dict):
                return value

    return None


def looks_like_condition_block(
    obj: Dict[str, Any],
) -> bool:
    """
    Determine whether obj itself looks like:

        {
            "Before Attack": {...},
            "After Pegasus": {...},
            ...
        }
    """

    if not isinstance(obj, dict):
        return False

    found_metric = 0

    for _, value in obj.items():
        if extract_tpr5(value) is not None:
            found_metric += 1

    return found_metric >= 2


def get_iasw_block(
    obj: Dict[str, Any],
) -> Dict[str, Any]:
    """
    IAS-W-only metrics may either be directly stored as
    condition -> metrics, or under Macro / IAS-W.
    """

    if looks_like_condition_block(obj):
        return obj

    block = find_block(
        obj,
        [
            "Macro",
            "IAS-W",
            "IASW",
            "iasw",
        ],
    )

    if block is not None:
        return block

    raise KeyError(
        "Could not locate IAS-W metric block. "
        f"Top-level keys: {list(obj.keys())}"
    )


# ============================================================
# Attack averaging
# ============================================================

def average_attacked_tpr5(
    block: Dict[str, Any],
    debug_name: str = "",
) -> float:
    """
    Average TPR@5%FPR over attacked settings.

    Before Attack is excluded.

    Output is percentage, e.g. 91.4.
    """

    values = []
    used_conditions = []

    for condition, metrics in block.items():

        if is_before_attack(condition):
            continue

        if is_excluded_attack(condition):
            continue

        value = extract_tpr5(metrics)

        if value is None:
            continue

        values.append(value)
        used_conditions.append(condition)

    if not values:
        raise ValueError(
            f"No attacked TPR@5%FPR values found "
            f"for {debug_name}"
        )

    mean_value = float(
        np.mean(values)
    ) * 100.0

    print(
        f"    {debug_name:<44} "
        f"Nattack={len(values):2d} "
        f"AvgTPR5={mean_value:7.3f}%"
    )

    if len(values) != 9:
        print(
            f"got {len(values)}."
        )

        print(
            "      Used:",
            ", ".join(used_conditions),
        )

    return mean_value


# ============================================================
# Read candidate experiment
# ============================================================

def load_candidate_result(
    candidate_root: Path,
    k: int,
) -> Dict[str, float]:

    plot_dir = (
        candidate_root
        / f"cand{k}"
        / "plots"
    )

    # --------------------------------------------------------
    # IAS-W only
    # --------------------------------------------------------

    iasw_path = (
        plot_dir
        / "iasw_roc_combined_metrics.json"
    )

    iasw_obj = load_json(
        iasw_path
    )

    iasw_block = get_iasw_block(
        iasw_obj
    )

    # --------------------------------------------------------
    # KGW + IAS-W
    # --------------------------------------------------------

    dual_path = (
        plot_dir
        / "dual_kgw_iasw_roc_combined_metrics.json"
    )

    dual_obj = load_json(
        dual_path
    )

    micro_block = find_block(
        dual_obj,
        [
            "Micro",
            "micro",
        ],
    )

    macro_block = find_block(
        dual_obj,
        [
            "Macro",
            "macro",
        ],
    )

    fisher_block = find_block(
        dual_obj,
        [
            "Fisher Fusion",
            "Fisher",
            "fisher_fusion",
            "fisher",
        ],
    )

    if micro_block is None:
        raise KeyError(
            f"Micro block not found in:\n{dual_path}\n"
            f"Top-level keys: {list(dual_obj.keys())}"
        )

    if macro_block is None:
        raise KeyError(
            f"Macro block not found in:\n{dual_path}\n"
            f"Top-level keys: {list(dual_obj.keys())}"
        )

    if fisher_block is None:
        raise KeyError(
            f"Fisher block not found in:\n{dual_path}\n"
            f"Top-level keys: {list(dual_obj.keys())}"
        )

    # --------------------------------------------------------
    # Compute nine-attack average
    # --------------------------------------------------------

    print(
        f"\nCandidate K = {k}"
    )

    result = {
        "IAS-W Only":
            average_attacked_tpr5(
                iasw_block,
                "IAS-W Only",
            ),

        "Token-Level Statistic (KGW+IAS-W)":
            average_attacked_tpr5(
                micro_block,
                "Token-Level Statistic (KGW+IAS-W)",
            ),

        "IAS-W Alignment Statistic (KGW+IAS-W)":
            average_attacked_tpr5(
                macro_block,
                "IAS-W Alignment Statistic (KGW+IAS-W)",
            ),

        "KGW+IAS-W (Fisher Fusion)":
            average_attacked_tpr5(
                fisher_block,
                "KGW+IAS-W (Fisher Fusion)",
            ),
    }

    return result


# ============================================================
# Collect all candidate values
# ============================================================

def collect_candidate_curves(
    candidate_root: Path,
):

    curves = {
        method: []
        for method in METHOD_ORDER
    }

    print(
        "\n"
        + "=" * 78
    )

    print(
        "CANDIDATE POOL ABLATION"
    )

    print(
        "=" * 78
    )

    print(
        f"Root: {candidate_root}"
    )

    print(
        "Metric: average TPR@5%FPR over nine attacked settings"
    )

    for k in CANDIDATE_VALUES:

        result = load_candidate_result(
            candidate_root,
            k,
        )

        for method in METHOD_ORDER:
            curves[method].append(
                result[method]
            )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FINAL VALUES"
    )

    print(
        "=" * 78
    )

    header = (
        f"{'Method':<45}"
        + "".join(
            f"K={k:>8}"
            for k in CANDIDATE_VALUES
        )
    )

    print(header)

    print("-" * len(header))

    for method in METHOD_ORDER:

        row = f"{method:<45}"

        for value in curves[method]:
            row += f"{value:10.3f}"

        print(row)

    print(
        "=" * 78
        + "\n"
    )

    return curves


# ============================================================
# Save figure
# ============================================================

def save_figure(
    fig,
    output_dir: Path,
    stem: str,
    args,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = (
        output_dir
        / f"{stem}.pdf"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    print(
        f"[DONE] {pdf}"
    )

    if not args.no_eps:

        eps = (
            output_dir
            / f"{stem}.eps"
        )

        fig.savefig(
            eps,
            format="eps",
            bbox_inches="tight",
        )

        print(
            f"[DONE] {eps}"
        )

    if not args.no_tiff:

        tiff = (
            output_dir
            / f"{stem}.tiff"
        )

        fig.savefig(
            tiff,
            format="tiff",
            dpi=args.tiff_dpi,
            bbox_inches="tight",
            pil_kwargs={
                "compression": "tiff_lzw"
            },
        )

        print(
            f"[DONE] {tiff}"
        )


# ============================================================
# Plot
# ============================================================

def plot_candidate_ablation(
    curves,
    args,
):

    x = np.arange(
        len(CANDIDATE_VALUES),
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # --------------------------------------------------------
    # Draw four curves
    # --------------------------------------------------------

    for method in METHOD_ORDER:

        y = np.asarray(
            curves[method],
            dtype=float,
        )

        is_final = (
            method
            == "KGW+IAS-W (Fisher Fusion)"
        )

        ax.plot(
            x,
            y,

            # Only final fusion uses diamond.
            marker=(
                "D"
                if is_final
                else "o"
            ),

            linewidth=(
                args.ours_line_width
                if is_final
                else args.line_width
            ),

            markersize=(
                args.ours_marker_size
                if is_final
                else args.marker_size
            ),

            label=LEGEND_LABELS.get(method, method),

            zorder=(
                5
                if is_final
                else 3
            ),
        )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            str(k)
            for k in CANDIDATE_VALUES
        ]
    )

    ax.set_xlabel(
        "Number of Candidates",
        labelpad=6,
    )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    ax.set_ylabel(
        "Average TPR@5%FPR (%)",
        labelpad=6,
    )

    # Do not force a narrow y-range.
    # Let all four statistics determine the visible range.

    # --------------------------------------------------------
    # Grid / ticks
    # --------------------------------------------------------

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    style_axis(ax)

    # --------------------------------------------------------
    # Optional title
    # --------------------------------------------------------

    if args.show_title:

        ax.set_title(
            "Candidate Pool Size Ablation"
        )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            LEGEND_Y,
        ),
        ncol=LEGEND_NCOL,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.35,
        handletextpad=0.6,
    )

    # --------------------------------------------------------
    # Margins
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        top=0.96,
        bottom=BOTTOM,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_figure(
        fig=fig,
        output_dir=args.output_dir,
        stem="figure8_candidate_ablation_tpr5",
        args=args,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    parser = build_parser()

    args = parser.parse_args()

    configure_style(
        args
    )

    curves = collect_candidate_curves(
        candidate_root=args.candidate_root,
    )

    plot_candidate_ablation(
        curves=curves,
        args=args,
    )


if __name__ == "__main__":
    main()