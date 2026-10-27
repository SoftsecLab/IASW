
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a KGW security gate for KGW + IAS-W Fisher detection.

Decision rule
-------------
accept iff

    z_KGW >= tau_K
    AND
    z_Fisher >= tau_F(tau_K)

For every KGW gate tau_K, tau_F is recalibrated from clean negative
examples so that the FINAL gated detector operates at the target FPR.

Inputs
------
1. Main KGW+IAS-W JSONL:
   contains human/no_wm/wm and attacked texts with detection results.

2. Spoofing JSONLs:
   Random-K32
   Estimated-anchor-K32
   True-anchor-K32

No generation or detection is performed here.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


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
    Return:
        z_kgw
        z_iasw
        z_fisher
    """

    if not isinstance(d, dict):
        return np.nan, np.nan, np.nan

    z_kgw = safe_float(
        d.get(
            "z_score_micro",
            np.nan,
        )
    )

    z_iasw = safe_float(
        d.get(
            "z_score_macro_calibrated",
            d.get(
                "z_score_macro",
                np.nan,
            ),
        )
    )

    z_fisher = safe_float(
        d.get(
            "z_score_fisher",
            d.get(
                "z_score",
                np.nan,
            ),
        )
    )

    return z_kgw, z_iasw, z_fisher


# ============================================================
# Main experiment parsing
# ============================================================

def collect_null_scores(rows):
    """
    Use human + no_wm as the null distribution.

    This follows the current evaluation convention, but importantly
    the FINAL gated threshold is recalibrated for every KGW gate.
    """

    out = []

    for row in rows:

        for field in [
            "detection_human",
            "detection_no_wm",
        ]:

            d = get_detection(row, field)

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

    return pd.DataFrame(out)


def collect_before_scores(rows):
    out = []

    for row in rows:

        d = get_detection(
            row,
            "detection_wm",
        )

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

    return pd.DataFrame(out)


def collect_attack_scores(rows):
    """
    Return:
        dict:
            attack_name -> DataFrame
    """

    attack_names = set()

    for row in rows:
        attacks = row.get("attacks", {})

        if isinstance(attacks, dict):
            attack_names.update(attacks.keys())

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

        if out:
            result[attack_name] = pd.DataFrame(out)

    return result


def collect_spoof_scores(path):
    rows = load_jsonl(path)

    out = []

    for row in rows:

        d = get_detection(
            row,
            "detection_wm",
        )

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

    return pd.DataFrame(out)


# ============================================================
# Gate calibration
# ============================================================

def calibrate_fisher_threshold(
    null_df,
    kgw_gate,
    target_fpr,
):
    """
    Decision:
        z_kgw >= kgw_gate
        AND
        z_fisher >= fisher_threshold

    We choose the Fisher threshold empirically so that the final
    gated detector does not exceed target_fpr on the supplied null set.

    If the KGW gate itself already passes <= target_fpr of null samples,
    Fisher does not need to impose an additional restriction. In that
    case fisher_threshold = -inf.
    """

    z_kgw = null_df["z_kgw"].to_numpy()
    z_fisher = null_df["z_fisher"].to_numpy()

    gate_mask = z_kgw >= kgw_gate

    n = len(null_df)

    if n == 0:
        raise RuntimeError("Empty null distribution.")

    n_gate = int(np.sum(gate_mask))

    # Maximum number of accepted nulls allowed.
    max_fp = int(np.floor(target_fpr * n))

    # If gate alone is already sufficiently strict.
    if n_gate <= max_fp:
        return -np.inf

    gated_fisher = z_fisher[gate_mask]

    # Need at most max_fp samples to survive both conditions.
    #
    # Sort descending. Threshold just above the (max_fp)-th allowed
    # boundary if necessary.
    sorted_scores = np.sort(gated_fisher)[::-1]

    if max_fp <= 0:
        return float(
            np.nextafter(
                sorted_scores[0],
                np.inf,
            )
        )

    if max_fp >= len(sorted_scores):
        return -np.inf

    # Accept values >= threshold.
    #
    # Put threshold immediately above the first value that would make
    # us exceed max_fp.
    boundary = sorted_scores[max_fp]

    threshold = float(
        np.nextafter(
            boundary,
            np.inf,
        )
    )

    return threshold


def acceptance_rate(
    df,
    kgw_gate,
    fisher_threshold,
):

    if len(df) == 0:
        return np.nan

    mask = (
        (df["z_kgw"].to_numpy() >= kgw_gate)
        &
        (df["z_fisher"].to_numpy() >= fisher_threshold)
    )

    return float(
        np.mean(mask)
    )


def fisher_only_rate(
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


def kgw_gate_rate(
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


# ============================================================
# Main
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--main_file",
        required=True,
        help="Main KGW+IAS-W JSONL with baseline and attack detections.",
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
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.279165,
        ],
    )

    return p.parse_args()


def main():

    args = parse_args()

    out_dir = Path(args.output_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("KGW SECURITY GATE SWEEP")
    print("=" * 100)

    print(f"Main file      : {args.main_file}")
    print(f"Target FPR     : {args.target_fpr:.4f}")
    print(f"KGW gates      : {args.kgw_gates}")

    # --------------------------------------------------------
    # Main data
    # --------------------------------------------------------

    print("\n[*] Loading main experiment...")

    main_rows = load_jsonl(
        args.main_file
    )

    null_df = collect_null_scores(
        main_rows
    )

    before_df = collect_before_scores(
        main_rows
    )

    attack_dfs = collect_attack_scores(
        main_rows
    )

    print(
        f"    null samples   = {len(null_df)}"
    )

    print(
        f"    before samples = {len(before_df)}"
    )

    print(
        f"    attacks        = {list(attack_dfs.keys())}"
    )

    # --------------------------------------------------------
    # Spoof data
    # --------------------------------------------------------

    print("\n[*] Loading spoofing experiments...")

    spoof_dfs = {
        "Random-K32": collect_spoof_scores(
            args.random_file
        ),
        "Estimated-N200-K32": collect_spoof_scores(
            args.estimated_file
        ),
        "True-Anchor-K32": collect_spoof_scores(
            args.true_file
        ),
    }

    for name, df in spoof_dfs.items():
        print(
            f"    {name:<22} n={len(df)}"
        )

    # --------------------------------------------------------
    # Sweep
    # --------------------------------------------------------

    robustness_rows = []
    security_rows = []
    summary_rows = []

    for kgw_gate in args.kgw_gates:

        fisher_tau = calibrate_fisher_threshold(
            null_df=null_df,
            kgw_gate=kgw_gate,
            target_fpr=args.target_fpr,
        )

        actual_fpr = acceptance_rate(
            null_df,
            kgw_gate,
            fisher_tau,
        )

        # ----------------------------------------------------
        # Normal robustness
        # ----------------------------------------------------

        before_tpr = acceptance_rate(
            before_df,
            kgw_gate,
            fisher_tau,
        )

        robustness_rows.append({
            "kgw_gate": kgw_gate,
            "fisher_threshold": fisher_tau,
            "condition": "Before Attack",
            "n": len(before_df),
            "accept_rate": before_tpr,
        })

        attack_rates = {}

        for attack_name, df in attack_dfs.items():

            rate = acceptance_rate(
                df,
                kgw_gate,
                fisher_tau,
            )

            attack_rates[attack_name] = rate

            robustness_rows.append({
                "kgw_gate": kgw_gate,
                "fisher_threshold": fisher_tau,
                "condition": attack_name,
                "n": len(df),
                "accept_rate": rate,
            })

        # ----------------------------------------------------
        # Spoof security
        # ----------------------------------------------------

        spoof_rates = {}

        for name, df in spoof_dfs.items():

            final_rate = acceptance_rate(
                df,
                kgw_gate,
                fisher_tau,
            )

            gate_only = kgw_gate_rate(
                df,
                kgw_gate,
            )

            fisher_only = fisher_only_rate(
                df,
                fisher_tau,
            )

            spoof_rates[name] = final_rate

            security_rows.append({
                "kgw_gate": kgw_gate,
                "fisher_threshold": fisher_tau,
                "group": name,
                "n": len(df),
                "kgw_gate_pass_rate": gate_only,
                "fisher_pass_rate": fisher_only,
                "final_spoof_success_rate": final_rate,
            })

        # ----------------------------------------------------
        # Compact summary
        # ----------------------------------------------------

        attacked_values = [
            x
            for x in attack_rates.values()
            if np.isfinite(x)
        ]

        mean_attack_tpr = (
            float(np.mean(attacked_values))
            if attacked_values
            else np.nan
        )

        min_attack_tpr = (
            float(np.min(attacked_values))
            if attacked_values
            else np.nan
        )

        summary_rows.append({
            "kgw_gate": kgw_gate,
            "fisher_threshold": fisher_tau,
            "actual_null_fpr": actual_fpr,

            "before_tpr": before_tpr,
            "mean_attack_tpr": mean_attack_tpr,
            "min_attack_tpr": min_attack_tpr,

            "random_spoof": spoof_rates.get(
                "Random-K32",
                np.nan,
            ),

            "estimated_spoof": spoof_rates.get(
                "Estimated-N200-K32",
                np.nan,
            ),

            "true_anchor_spoof": spoof_rates.get(
                "True-Anchor-K32",
                np.nan,
            ),
        })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    robustness_df = pd.DataFrame(
        robustness_rows
    )

    security_df = pd.DataFrame(
        security_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    robustness_df.to_csv(
        out_dir / "kgw_gate_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    security_df.to_csv(
        out_dir / "kgw_gate_security.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        out_dir / "kgw_gate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    display_df = summary_df.copy()

    rate_cols = [
        "actual_null_fpr",
        "before_tpr",
        "mean_attack_tpr",
        "min_attack_tpr",
        "random_spoof",
        "estimated_spoof",
        "true_anchor_spoof",
    ]

    for col in rate_cols:
        display_df[col] = (
            display_df[col] * 100.0
        )

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("=" * 100)
    print("PER-ATTACK ROBUSTNESS")
    print("=" * 100)

    pivot = robustness_df.pivot(
        index="condition",
        columns="kgw_gate",
        values="accept_rate",
    ) * 100.0

    print(
        pivot.to_string(
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("=" * 100)
    print("SPOOF SECURITY")
    print("=" * 100)

    pivot_sec = security_df.pivot(
        index="group",
        columns="kgw_gate",
        values="final_spoof_success_rate",
    ) * 100.0

    print(
        pivot_sec.to_string(
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("[+] Saved:")
    print(
        "   ",
        out_dir / "kgw_gate_summary.csv",
    )
    print(
        "   ",
        out_dir / "kgw_gate_robustness.csv",
    )
    print(
        "   ",
        out_dir / "kgw_gate_security.csv",
    )


if __name__ == "__main__":
    main()