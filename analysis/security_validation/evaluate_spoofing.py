#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def get_detection(
    row,
    field,
):
    d = row.get(
        field,
        {}
    )

    return (
        d
        if isinstance(d, dict)
        else {}
    )


def score(
    d,
    name,
):

    x = d.get(
        name,
        None,
    )

    try:
        x = float(x)

        if np.isfinite(x):
            return x

    except Exception:
        pass

    return np.nan


def empirical_threshold(
    values,
    fpr=0.01,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        raise RuntimeError(
            "No valid negative scores."
        )

    # Higher = more watermarked.
    return float(
        np.quantile(
            x,
            1.0 - fpr,
            method="higher",
        )
    )


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--files",
        nargs="+",
        required=True,
    )

    p.add_argument(
        "--labels",
        nargs="+",
        required=True,
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    p.add_argument(
        "--fpr",
        type=float,
        default=0.01,
    )

    return p.parse_args()


def main():

    args = parse_args()

    if (
        len(args.files)
        != len(args.labels)
    ):
        raise ValueError(
            "--files and --labels "
            "must have same length."
        )

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Use the first file's human + no_wm as null calibration.
    # --------------------------------------------------------

    calibration_rows = (
        load_jsonl(
            args.files[0]
        )
    )

    null_micro = []
    null_macro = []
    null_fisher = []

    for row in calibration_rows:

        for field in [
            "detection_human",
            "detection_no_wm",
        ]:

            d = get_detection(
                row,
                field,
            )

            null_micro.append(
                score(
                    d,
                    "z_score_micro",
                )
            )

            null_macro.append(
                score(
                    d,
                    "z_score_macro_calibrated",
                )
            )

            null_fisher.append(
                score(
                    d,
                    "z_score_fisher",
                )
            )

    tau_micro = empirical_threshold(
        null_micro,
        args.fpr,
    )

    tau_macro = empirical_threshold(
        null_macro,
        args.fpr,
    )

    tau_fisher = empirical_threshold(
        null_fisher,
        args.fpr,
    )

    print()
    print("=" * 80)
    print("EMPIRICAL THRESHOLDS")
    print("=" * 80)

    print(
        f"FPR target = "
        f"{args.fpr * 100:.3f}%"
    )

    print(
        f"KGW    tau = {tau_micro:.6f}"
    )

    print(
        f"IAS-W  tau = {tau_macro:.6f}"
    )

    print(
        f"Fisher tau = {tau_fisher:.6f}"
    )

    # --------------------------------------------------------
    # Evaluate spoofing
    # --------------------------------------------------------

    summary_rows = []
    sample_rows = []

    for path, label in zip(
        args.files,
        args.labels,
    ):

        rows = load_jsonl(
            path
        )

        micro = []
        macro = []
        fisher = []

        for idx, row in enumerate(
            rows
        ):

            d = get_detection(
                row,
                "detection_wm",
            )

            z_micro = score(
                d,
                "z_score_micro",
            )

            z_macro = score(
                d,
                "z_score_macro_calibrated",
            )

            z_fisher = score(
                d,
                "z_score_fisher",
            )

            if not np.isfinite(
                z_fisher
            ):
                continue

            micro.append(
                z_micro
            )

            macro.append(
                z_macro
            )

            fisher.append(
                z_fisher
            )

            sample_rows.append({
                "group": label,
                "index": idx,
                "z_kgw": z_micro,
                "z_iasw": z_macro,
                "z_fisher": z_fisher,
                "pass_kgw": (
                    z_micro
                    >= tau_micro
                ),
                "pass_iasw": (
                    z_macro
                    >= tau_macro
                ),
                "pass_fisher": (
                    z_fisher
                    >= tau_fisher
                ),
            })

        micro = np.asarray(
            micro,
            dtype=float,
        )

        macro = np.asarray(
            macro,
            dtype=float,
        )

        fisher = np.asarray(
            fisher,
            dtype=float,
        )

        summary_rows.append({
            "group": label,
            "n": len(fisher),

            "mean_z_kgw": (
                np.nanmean(micro)
            ),

            "mean_z_iasw": (
                np.nanmean(macro)
            ),

            "mean_z_fisher": (
                np.nanmean(fisher)
            ),

            "kgw_accept_rate": (
                np.nanmean(
                    micro >= tau_micro
                )
            ),

            "iasw_accept_rate": (
                np.nanmean(
                    macro >= tau_macro
                )
            ),

            # This is the main security metric.
            "fisher_spoof_success_rate": (
                np.nanmean(
                    fisher >= tau_fisher
                )
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    samples = pd.DataFrame(
        sample_rows
    )

    summary.to_csv(
        out_dir
        / "spoofing_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    samples.to_csv(
        out_dir
        / "spoofing_per_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("SPOOFING RESULTS")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "[+] Saved:",
        out_dir
        / "spoofing_summary.csv",
    )


if __name__ == "__main__":
    main()