# analysis/security_validation/evaluate_kgw_gate_calibration.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare KGW-gated Fisher detection under three null calibration schemes:

1. human_only
2. no_wm_only
3. human_plus_no_wm

Final decision:
    accept iff
        z_KGW >= tau_K
        AND
        z_Fisher >= tau_F

For each:
    calibration scheme × KGW gate

tau_F is recalibrated empirically to target final FPR.

This script performs NO generation and NO detection.
It only reads existing JSONL results.

Recommended use:
    gate = 1.5 and 2.0

Outputs:
    calibration_summary.csv
    calibration_robustness.csv
    calibration_security.csv
    calibration_fpr_breakdown.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

EXCLUDED_ATTACKS = set()


# ============================================================
# IO
# ============================================================

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def safe_float(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
    except Exception:
        pass

    return np.nan


# ============================================================
# Detection extraction
# ============================================================

def get_detection(row, field):
    d = row.get(field, {})
    return d if isinstance(d, dict) else {}


def extract_scores(d):
    """
    Extract:
        z_KGW
        z_IASW
        z_Fisher
    """

    if not isinstance(d, dict):
        return np.nan, np.nan, np.nan

    z_kgw = safe_float(
        d.get("z_score_micro", np.nan)
    )

    z_iasw = safe_float(
        d.get(
            "z_score_macro_calibrated",
            d.get("z_score_macro", np.nan),
        )
    )

    z_fisher = safe_float(
        d.get(
            "z_score_fisher",
            d.get("z_score", np.nan),
        )
    )

    return z_kgw, z_iasw, z_fisher


def append_detection(out, d):
    z_kgw, z_iasw, z_fisher = extract_scores(d)

    if (
        np.isfinite(z_kgw)
        and np.isfinite(z_fisher)
    ):
        out.append({
            "z_kgw": z_kgw,
            "z_iasw": z_iasw,
            "z_fisher": z_fisher,
        })


# ============================================================
# Null distributions
# ============================================================

def collect_null_distributions(rows):

    human = []
    no_wm = []

    for row in rows:

        append_detection(
            human,
            get_detection(
                row,
                "detection_human",
            ),
        )

        append_detection(
            no_wm,
            get_detection(
                row,
                "detection_no_wm",
            ),
        )

    human_df = pd.DataFrame(human)
    no_wm_df = pd.DataFrame(no_wm)

    combined_df = pd.concat(
        [human_df, no_wm_df],
        ignore_index=True,
    )

    return {
        "human_only": human_df,
        "no_wm_only": no_wm_df,
        "human_plus_no_wm": combined_df,
    }


# ============================================================
# Positive distributions
# ============================================================

def collect_before_scores(rows):

    out = []

    for row in rows:
        append_detection(
            out,
            get_detection(
                row,
                "detection_wm",
            ),
        )

    return pd.DataFrame(out)


def collect_attack_scores(rows):

    attack_names = set()

    for row in rows:

        attacks = row.get("attacks", {})

        if isinstance(attacks, dict):
            attack_names.update(attacks.keys())

    # Remove attacks no longer used in the paper.
    attack_names = {
        x
        for x in attack_names
        if x not in EXCLUDED_ATTACKS
    }

    result = {}

    for attack_name in sorted(attack_names):

        out = []

        for row in rows:

            attacks = row.get("attacks", {})

            if not isinstance(attacks, dict):
                continue

            rec = attacks.get(
                attack_name,
                {},
            )

            if not isinstance(rec, dict):
                continue

            d = rec.get(
                "detection_results",
                {},
            )

            append_detection(
                out,
                d,
            )

        if out:
            result[attack_name] = pd.DataFrame(out)

    return result


# ============================================================
# Spoofing distributions
# ============================================================

def collect_spoof_scores(path):

    rows = load_jsonl(path)

    out = []

    for row in rows:

        append_detection(
            out,
            get_detection(
                row,
                "detection_wm",
            ),
        )

    return pd.DataFrame(out)


# ============================================================
# Decision functions
# ============================================================

def final_accept_mask(
    df,
    kgw_gate,
    fisher_threshold,
):

    if len(df) == 0:
        return np.array([], dtype=bool)

    return (
        (df["z_kgw"].to_numpy() >= kgw_gate)
        &
        (df["z_fisher"].to_numpy() >= fisher_threshold)
    )


def acceptance_rate(
    df,
    kgw_gate,
    fisher_threshold,
):

    mask = final_accept_mask(
        df,
        kgw_gate,
        fisher_threshold,
    )

    if len(mask) == 0:
        return np.nan

    return float(
        np.mean(mask)
    )


def kgw_gate_pass_rate(
    df,
    kgw_gate,
):

    if len(df) == 0:
        return np.nan

    return float(
        np.mean(
            df["z_kgw"].to_numpy()
            >= kgw_gate
        )
    )


def fisher_pass_rate(
    df,
    fisher_threshold,
):

    if len(df) == 0:
        return np.nan

    return float(
        np.mean(
            df["z_fisher"].to_numpy()
            >= fisher_threshold
        )
    )


# ============================================================
# Empirical threshold calibration
# ============================================================

def calibrate_fisher_threshold(
    null_df,
    kgw_gate,
    target_fpr,
):
    """
    Calibrate Fisher threshold for:

        z_KGW >= kgw_gate
        AND
        z_Fisher >= tau_F

    so empirical final FPR does not exceed target_fpr
    on the selected calibration null distribution.

    Example:
        n = 500
        target FPR = 1%
        -> at most 5 accepted null samples.
    """

    if len(null_df) == 0:
        raise RuntimeError(
            "Empty null calibration distribution."
        )

    z_kgw = null_df["z_kgw"].to_numpy()
    z_fisher = null_df["z_fisher"].to_numpy()

    gate_mask = (
        z_kgw >= kgw_gate
    )

    gated_fisher = (
        z_fisher[gate_mask]
    )

    n_total = len(null_df)

    max_fp = int(
        np.floor(
            target_fpr * n_total
        )
    )

    # Gate alone already satisfies target FPR.
    if len(gated_fisher) <= max_fp:
        return -np.inf

    scores = np.sort(
        gated_fisher
    )[::-1]

    # Extremely small target FPR.
    if max_fp <= 0:

        return float(
            np.nextafter(
                scores[0],
                np.inf,
            )
        )

    if max_fp >= len(scores):
        return -np.inf

    # scores[max_fp] would become the (max_fp + 1)-th
    # accepted sample if threshold were <= this value.
    boundary = scores[max_fp]

    threshold = float(
        np.nextafter(
            boundary,
            np.inf,
        )
    )

    return threshold


# ============================================================
# Helpers
# ============================================================

def mean_valid(values):

    values = [
        x
        for x in values
        if np.isfinite(x)
    ]

    if not values:
        return np.nan

    return float(
        np.mean(values)
    )


def min_valid(values):

    values = [
        x
        for x in values
        if np.isfinite(x)
    ]

    if not values:
        return np.nan

    return float(
        np.min(values)
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--main_file",
        required=True,
    )

    p.add_argument(
        "--random_file",
        required=True,
    )

    p.add_argument(
        "--estimated_file",
        required=True,
    )

    p.add_argument(
        "--true_file",
        required=True,
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    p.add_argument(
        "--target_fpr",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--kgw_gates",
        nargs="+",
        type=float,
        default=[
            1.5,
            2.0,
        ],
    )

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print("KGW GATE — NULL CALIBRATION COMPARISON")
    print("=" * 110)

    print(
        f"Main file      : {args.main_file}"
    )

    print(
        f"Target FPR     : {args.target_fpr:.4f}"
    )

    print(
        f"KGW gates      : {args.kgw_gates}"
    )

    print(
        f"Excluded attack: {sorted(EXCLUDED_ATTACKS)}"
    )

    # --------------------------------------------------------
    # Load main experiment
    # --------------------------------------------------------

    print("\n[*] Loading main experiment...")

    main_rows = load_jsonl(
        args.main_file
    )

    null_dfs = collect_null_distributions(
        main_rows
    )

    before_df = collect_before_scores(
        main_rows
    )

    attack_dfs = collect_attack_scores(
        main_rows
    )

    human_df = null_dfs[
        "human_only"
    ]

    no_wm_df = null_dfs[
        "no_wm_only"
    ]

    print(
        f"    human null     = {len(human_df)}"
    )

    print(
        f"    no_wm null     = {len(no_wm_df)}"
    )

    print(
        f"    combined null  = "
        f"{len(null_dfs['human_plus_no_wm'])}"
    )

    print(
        f"    before samples = {len(before_df)}"
    )

    print(
        f"    attacks        = "
        f"{list(attack_dfs.keys())}"
    )

    # --------------------------------------------------------
    # Load spoofing
    # --------------------------------------------------------

    print("\n[*] Loading spoofing experiments...")

    spoof_dfs = {
        "Random-K32":
            collect_spoof_scores(
                args.random_file
            ),

        "Estimated-N200-K32":
            collect_spoof_scores(
                args.estimated_file
            ),

        "True-Anchor-K32":
            collect_spoof_scores(
                args.true_file
            ),
    }

    for name, df in spoof_dfs.items():

        print(
            f"    {name:<24} "
            f"n={len(df)}"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    summary_rows = []
    robustness_rows = []
    security_rows = []
    fpr_rows = []

    calibration_order = [
        "human_only",
        "no_wm_only",
        "human_plus_no_wm",
    ]

    for calibration_name in calibration_order:

        calibration_df = (
            null_dfs[
                calibration_name
            ]
        )

        for kgw_gate in args.kgw_gates:

            fisher_threshold = (
                calibrate_fisher_threshold(
                    null_df=calibration_df,
                    kgw_gate=kgw_gate,
                    target_fpr=args.target_fpr,
                )
            )

            # =================================================
            # FPR breakdown
            # =================================================

            calibration_fpr = acceptance_rate(
                calibration_df,
                kgw_gate,
                fisher_threshold,
            )

            human_fpr = acceptance_rate(
                human_df,
                kgw_gate,
                fisher_threshold,
            )

            no_wm_fpr = acceptance_rate(
                no_wm_df,
                kgw_gate,
                fisher_threshold,
            )

            fpr_rows.append({
                "calibration":
                    calibration_name,

                "kgw_gate":
                    kgw_gate,

                "fisher_threshold":
                    fisher_threshold,

                "calibration_fpr":
                    calibration_fpr,

                "human_fpr":
                    human_fpr,

                "no_wm_fpr":
                    no_wm_fpr,
            })

            # =================================================
            # Before attack
            # =================================================

            before_tpr = acceptance_rate(
                before_df,
                kgw_gate,
                fisher_threshold,
            )

            robustness_rows.append({
                "calibration":
                    calibration_name,

                "kgw_gate":
                    kgw_gate,

                "fisher_threshold":
                    fisher_threshold,

                "condition":
                    "Before Attack",

                "n":
                    len(before_df),

                "accept_rate":
                    before_tpr,
            })

            # =================================================
            # Attacks
            # =================================================

            attack_rates = {}

            for (
                attack_name,
                attack_df,
            ) in attack_dfs.items():

                rate = acceptance_rate(
                    attack_df,
                    kgw_gate,
                    fisher_threshold,
                )

                attack_rates[
                    attack_name
                ] = rate

                robustness_rows.append({
                    "calibration":
                        calibration_name,

                    "kgw_gate":
                        kgw_gate,

                    "fisher_threshold":
                        fisher_threshold,

                    "condition":
                        attack_name,

                    "n":
                        len(attack_df),

                    "accept_rate":
                        rate,
                })

            mean_attack_tpr = mean_valid(
                list(
                    attack_rates.values()
                )
            )

            min_attack_tpr = min_valid(
                list(
                    attack_rates.values()
                )
            )

            # =================================================
            # Spoofing
            # =================================================

            spoof_rates = {}

            for (
                group_name,
                spoof_df,
            ) in spoof_dfs.items():

                final_rate = acceptance_rate(
                    spoof_df,
                    kgw_gate,
                    fisher_threshold,
                )

                kgw_pass = kgw_gate_pass_rate(
                    spoof_df,
                    kgw_gate,
                )

                fisher_pass = fisher_pass_rate(
                    spoof_df,
                    fisher_threshold,
                )

                spoof_rates[
                    group_name
                ] = final_rate

                security_rows.append({
                    "calibration":
                        calibration_name,

                    "kgw_gate":
                        kgw_gate,

                    "fisher_threshold":
                        fisher_threshold,

                    "group":
                        group_name,

                    "n":
                        len(spoof_df),

                    "kgw_gate_pass_rate":
                        kgw_pass,

                    "fisher_pass_rate":
                        fisher_pass,

                    "final_spoof_success_rate":
                        final_rate,
                })

            # =================================================
            # Summary
            # =================================================

            summary_rows.append({

                "calibration":
                    calibration_name,

                "kgw_gate":
                    kgw_gate,

                "fisher_threshold":
                    fisher_threshold,

                "calibration_fpr":
                    calibration_fpr,

                "human_fpr":
                    human_fpr,

                "no_wm_fpr":
                    no_wm_fpr,

                "before_tpr":
                    before_tpr,

                "mean_attack_tpr":
                    mean_attack_tpr,

                "min_attack_tpr":
                    min_attack_tpr,

                "random_spoof":
                    spoof_rates.get(
                        "Random-K32",
                        np.nan,
                    ),

                "estimated_spoof":
                    spoof_rates.get(
                        "Estimated-N200-K32",
                        np.nan,
                    ),

                "true_anchor_spoof":
                    spoof_rates.get(
                        "True-Anchor-K32",
                        np.nan,
                    ),
            })

    # ========================================================
    # DataFrames
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    robustness_df = pd.DataFrame(
        robustness_rows
    )

    security_df = pd.DataFrame(
        security_rows
    )

    fpr_df = pd.DataFrame(
        fpr_rows
    )

    # ========================================================
    # Save
    # ========================================================

    summary_path = (
        output_dir
        / "calibration_summary.csv"
    )

    robustness_path = (
        output_dir
        / "calibration_robustness.csv"
    )

    security_path = (
        output_dir
        / "calibration_security.csv"
    )

    fpr_path = (
        output_dir
        / "calibration_fpr_breakdown.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    robustness_df.to_csv(
        robustness_path,
        index=False,
        encoding="utf-8-sig",
    )

    security_df.to_csv(
        security_path,
        index=False,
        encoding="utf-8-sig",
    )

    fpr_df.to_csv(
        fpr_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Terminal display
    # ========================================================

    print()
    print("=" * 110)
    print("CALIBRATION SUMMARY")
    print("=" * 110)

    display = summary_df.copy()

    percentage_columns = [
        "calibration_fpr",
        "human_fpr",
        "no_wm_fpr",
        "before_tpr",
        "mean_attack_tpr",
        "min_attack_tpr",
        "random_spoof",
        "estimated_spoof",
        "true_anchor_spoof",
    ]

    for col in percentage_columns:

        display[col] = (
            display[col] * 100.0
        )

    print(
        display.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    # --------------------------------------------------------
    # FPR
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("FPR BREAKDOWN")
    print("=" * 110)

    fpr_display = fpr_df.copy()

    for col in [
        "calibration_fpr",
        "human_fpr",
        "no_wm_fpr",
    ]:

        fpr_display[col] = (
            fpr_display[col]
            * 100.0
        )

    print(
        fpr_display.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    # --------------------------------------------------------
    # Security
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("SPOOF SECURITY")
    print("=" * 110)

    security_pivot = (
        security_df.pivot_table(
            index=[
                "calibration",
                "group",
            ],
            columns="kgw_gate",
            values="final_spoof_success_rate",
        )
        * 100.0
    )

    print(
        security_pivot.to_string(
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    # --------------------------------------------------------
    # Robustness
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("PER-ATTACK ROBUSTNESS")
    print("=" * 110)

    for calibration_name in calibration_order:

        print()
        print(
            f"[{calibration_name}]"
        )

        subset = robustness_df[
            robustness_df[
                "calibration"
            ]
            == calibration_name
        ]

        pivot = (
            subset.pivot(
                index="condition",
                columns="kgw_gate",
                values="accept_rate",
            )
            * 100.0
        )

        print(
            pivot.to_string(
                float_format=lambda x: (
                    f"{x:.2f}"
                ),
            )
        )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)

    print(
        f"[+] {summary_path}"
    )

    print(
        f"[+] {fpr_path}"
    )

    print(
        f"[+] {security_path}"
    )

    print(
        f"[+] {robustness_path}"
    )


if __name__ == "__main__":
    main()

# python analysis/security_validation/evaluate_kgw_gate_calibration.py \
#     --main_file results/main_exp/c4/llama3.1-8b-instruct/records/dual_kgw_iasw.jsonl \
#     --random_file results/security_validation/spoofing/c4/llama3.1-8b-instruct/random_k32.jsonl \
#     --estimated_file results/security_validation/spoofing/c4/llama3.1-8b-instruct/estimated_n200_k32.jsonl \
#     --true_file results/security_validation/spoofing/c4/llama3.1-8b-instruct/true_anchor_k32.jsonl \
#     --output_dir results/security_validation/kgw_gate/c4/llama3.1-8b-instruct/calibration_compare \
#     --target_fpr 0.01 \
#     --kgw_gates 1.5 2.0