#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input_csv",
        required=True,
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    return p.parse_args()


def main():

    args = parse_args()

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        args.input_csv
    )

    # --------------------------------------------
    # Aggregate
    # --------------------------------------------

    summary = (
        df.groupby(
            ["source", "N_requested"],
            as_index=False,
        )
        .agg(
            mean_cosine=(
                "cosine_to_true_anchor",
                "mean",
            ),
            std_cosine=(
                "cosine_to_true_anchor",
                "std",
            ),
            median_cosine=(
                "cosine_to_true_anchor",
                "median",
            ),
            n_positions=(
                "cosine_to_true_anchor",
                "count",
            ),
        )
    )

    summary.to_csv(
        out_dir
        / "anchor_recovery_summary_eval.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------
    # Main figure
    # --------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6.4, 4.4)
    )

    labels = {
        "wm": "IAS-W",
        "no_wm": "No watermark",
        "human": "Human",
    }

    markers = {
        "wm": "o",
        "no_wm": "s",
        "human": "^",
    }

    for source in [
        "wm",
        "no_wm",
        "human",
    ]:

        sub = summary[
            summary["source"] == source
        ].sort_values(
            "N_requested"
        )

        if sub.empty:
            continue

        ax.plot(
            sub["N_requested"],
            sub["mean_cosine"],
            marker=markers[source],
            linewidth=1.8,
            markersize=5,
            label=labels[source],
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Number of observed documents (N)"
    )

    ax.set_ylabel(
        "Cosine similarity to true anchor"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "anchor_recovery_curve.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        out_dir
        / "anchor_recovery_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------
    # Per-position figure for largest N
    # --------------------------------------------

    max_n = int(
        df["N_requested"].max()
    )

    sub = df[
        df["N_requested"] == max_n
    ].copy()

    fig, ax = plt.subplots(
        figsize=(7.0, 4.4)
    )

    for source in [
        "wm",
        "no_wm",
        "human",
    ]:

        ss = sub[
            sub["source"] == source
        ].sort_values(
            "position_1based"
        )

        if ss.empty:
            continue

        ax.plot(
            ss["position_1based"],
            ss["cosine_to_true_anchor"],
            marker=markers[source],
            linewidth=1.6,
            markersize=4,
            label=labels[source],
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Sentence position"
    )

    ax.set_ylabel(
        "Cosine similarity to true anchor"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / f"anchor_recovery_by_position_N{max_n}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        out_dir
        / f"anchor_recovery_by_position_N{max_n}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 80)
    print("ANCHOR RECOVERY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Largest N:",
        max_n,
    )


if __name__ == "__main__":
    main()