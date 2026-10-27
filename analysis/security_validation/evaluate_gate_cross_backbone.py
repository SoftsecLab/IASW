# analysis/security_validation/evaluate_gate_cross_backbone.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-backbone validation of KGW-gated Fisher detection.

Compare:

    Original:
        z_Fisher >= tau_original

    KGW-gated:
        z_KGW >= tau_K
        AND
        z_Fisher >= tau_gated

For EACH backbone independently:

    1. Use its own no_wm samples as the null distribution.
    2. Calibrate the original Fisher-only threshold to target FPR.
    3. Calibrate the gated Fisher threshold to the SAME target FPR.
    4. Evaluate Before Attack + current 8 paper attacks.
    5. Report Gated - Original TPR differences.

No generation.
No detection.
No model loading.

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

EXCLUDED_ATTACKS = set()

PAPER_ATTACK_ORDER = [
    "pegasus",
    "dipper",
    "back_translation",
    "copy_paste",
    "del_ins",
    "drop",
    "swap",
    "typo",
]

DEFAULT_MODELS = [
    "opt-1.3b-semstamp",
    "Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct",
    "qwen2.5-7b-instruct",
    "llama3.1-8b-instruct",
]


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

    if isinstance(d, dict):
        return d

    return {}


def extract_scores(d):
    """
    Extract:
        z_KGW
        z_Fisher
    """

    if not isinstance(d, dict):
        return np.nan, np.nan

    z_kgw = safe_float(
        d.get(
            "z_score_micro",
            np.nan,
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

    return z_kgw, z_fisher


def append_detection(out, d):
    z_kgw, z_fisher = extract_scores(d)

    if (
        np.isfinite(z_kgw)
        and np.isfinite(z_fisher)
    ):
        out.append({
            "z_kgw": z_kgw,
            "z_fisher": z_fisher,
        })


# ============================================================
# Dataset extraction
# ============================================================

def collect_no_wm(rows):
    out = []

    for row in rows:
        append_detection(
            out,
            get_detection(
                row,
                "detection_no_wm",
            ),
        )

    return pd.DataFrame(out)


def collect_human(rows):
    out = []

    for row in rows:
        append_detection(
            out,
            get_detection(
                row,
                "detection_human",
            ),
        )

    return pd.DataFrame(out)


def collect_before(rows):
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


def collect_attacks(rows):
    """
    Return:
        attack_name -> DataFrame
    """

    available_attacks = set()

    for row in rows:

        attacks = row.get(
            "attacks",
            {},
        )

        if isinstance(attacks, dict):
            available_attacks.update(
                attacks.keys()
            )

    available_attacks -= EXCLUDED_ATTACKS

    result = {}

    # Preserve paper attack order first.
    ordered_attacks = [
        attack
        for attack in PAPER_ATTACK_ORDER
        if attack in available_attacks
    ]

    # Add any unexpected remaining attacks at the end.
    extras = sorted(
        available_attacks
        - set(ordered_attacks)
    )

    ordered_attacks += extras

    for attack_name in ordered_attacks:

        out = []

        for row in rows:

            attacks = row.get(
                "attacks",
                {},
            )

            if not isinstance(attacks, dict):
                continue

            attack_record = attacks.get(
                attack_name,
                {},
            )

            if not isinstance(
                attack_record,
                dict,
            ):
                continue

            detection = attack_record.get(
                "detection_results",
                {},
            )

            append_detection(
                out,
                detection,
            )

        if out:
            result[
                attack_name
            ] = pd.DataFrame(out)

    return result


# ============================================================
# Empirical calibration
# ============================================================

def calibrate_original_fisher(
    null_df,
    target_fpr,
):
    """
    Fisher-only detector:

        accept iff z_Fisher >= tau

    Empirically choose tau so that FPR <= target_fpr.
    """

    if len(null_df) == 0:
        raise RuntimeError(
            "Empty no_wm null distribution."
        )

    scores = (
        null_df["z_fisher"]
        .to_numpy()
    )

    scores = np.sort(
        scores
    )[::-1]

    n = len(scores)

    max_fp = int(
        np.floor(
            target_fpr * n
        )
    )

    if max_fp <= 0:

        return float(
            np.nextafter(
                scores[0],
                np.inf,
            )
        )

    if max_fp >= n:
        return -np.inf

    boundary = scores[max_fp]

    return float(
        np.nextafter(
            boundary,
            np.inf,
        )
    )


def calibrate_gated_fisher(
    null_df,
    kgw_gate,
    target_fpr,
):
    """
    Gated detector:

        accept iff
            z_KGW >= kgw_gate
            AND
            z_Fisher >= tau

    Calibrate tau so final FPR <= target_fpr.
    """

    if len(null_df) == 0:
        raise RuntimeError(
            "Empty no_wm null distribution."
        )

    z_kgw = (
        null_df["z_kgw"]
        .to_numpy()
    )

    z_fisher = (
        null_df["z_fisher"]
        .to_numpy()
    )

    gate_mask = (
        z_kgw >= kgw_gate
    )

    gated_scores = (
        z_fisher[gate_mask]
    )

    n_total = len(null_df)

    max_fp = int(
        np.floor(
            target_fpr * n_total
        )
    )

    # KGW gate alone already satisfies target FPR.
    if len(gated_scores) <= max_fp:
        return -np.inf

    scores = np.sort(
        gated_scores
    )[::-1]

    if max_fp <= 0:

        return float(
            np.nextafter(
                scores[0],
                np.inf,
            )
        )

    if max_fp >= len(scores):
        return -np.inf

    boundary = scores[max_fp]

    return float(
        np.nextafter(
            boundary,
            np.inf,
        )
    )


# ============================================================
# Acceptance
# ============================================================

def original_accept_rate(
    df,
    fisher_threshold,
):

    if len(df) == 0:
        return np.nan

    mask = (
        df["z_fisher"].to_numpy()
        >= fisher_threshold
    )

    return float(
        np.mean(mask)
    )


def gated_accept_rate(
    df,
    kgw_gate,
    fisher_threshold,
):

    if len(df) == 0:
        return np.nan

    mask = (
        (
            df["z_kgw"].to_numpy()
            >= kgw_gate
        )
        &
        (
            df["z_fisher"].to_numpy()
            >= fisher_threshold
        )
    )

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


# ============================================================
# File discovery
# ============================================================

def find_main_file(
    results_root,
    dataset,
    model,
    filename,
):
    """
    First try exact expected path.

    Then try a few simple fallbacks.
    """

    model_dir = (
        Path(results_root)
        / dataset
        / model
        / "records"
    )

    exact = (
        model_dir
        / filename
    )

    if exact.exists():
        return exact

    candidates = [
        model_dir / "dual_kgw_iasw.jsonl",
        model_dir / "kgw_iasw.jsonl",
        model_dir / "universal_dual.jsonl",
    ]

    for path in candidates:
        if path.exists():
            return path

    if model_dir.exists():

        matches = sorted(
            model_dir.glob(
                "*kgw*iasw*.jsonl"
            )
        )

        if len(matches) == 1:
            return matches[0]

    return None


# ============================================================
# Utility
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


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_root",
        default="results/main_exp",
    )

    parser.add_argument(
        "--dataset",
        default="c4",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )

    parser.add_argument(
        "--filename",
        default="dual_kgw_iasw.jsonl",
    )

    parser.add_argument(
        "--kgw_gate",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--target_fpr",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "results/security_validation/"
            "kgw_gate/cross_backbone"
        ),
    )

    return parser.parse_args()


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
    print("CROSS-BACKBONE KGW GATE VALIDATION")
    print("=" * 110)

    print(
        f"Dataset        : {args.dataset}"
    )

    print(
        f"Target FPR     : "
        f"{args.target_fpr:.3%}"
    )

    print(
        f"KGW gate       : "
        f"{args.kgw_gate:.3f}"
    )

    print(
        f"Null calibration: no_wm only"
    )

    print(
        f"Excluded       : "
        f"{sorted(EXCLUDED_ATTACKS)}"
    )

    print()

    detail_rows = []
    model_summary_rows = []
    threshold_rows = []

    valid_models = []

    # ========================================================
    # Each backbone
    # ========================================================

    for model in args.models:

        print("-" * 110)
        print(f"MODEL: {model}")
        print("-" * 110)

        main_file = find_main_file(
            results_root=args.results_root,
            dataset=args.dataset,
            model=model,
            filename=args.filename,
        )

        if main_file is None:

            print(
                f"[WARN] Main JSONL not found "
                f"for {model}. Skipping."
            )

            print()

            continue

        print(
            f"[*] File: {main_file}"
        )

        rows = load_jsonl(
            main_file
        )

        no_wm_df = collect_no_wm(
            rows
        )

        human_df = collect_human(
            rows
        )

        before_df = collect_before(
            rows
        )

        attack_dfs = collect_attacks(
            rows
        )

        print(
            f"    no_wm  = {len(no_wm_df)}"
        )

        print(
            f"    human  = {len(human_df)}"
        )

        print(
            f"    before = {len(before_df)}"
        )

        print(
            f"    attacks= "
            f"{list(attack_dfs.keys())}"
        )

        if len(no_wm_df) == 0:

            print(
                "[WARN] No valid no_wm "
                "detections. Skipping."
            )

            print()

            continue

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        tau_original = (
            calibrate_original_fisher(
                no_wm_df,
                args.target_fpr,
            )
        )

        tau_gated = (
            calibrate_gated_fisher(
                no_wm_df,
                args.kgw_gate,
                args.target_fpr,
            )
        )

        original_no_wm_fpr = (
            original_accept_rate(
                no_wm_df,
                tau_original,
            )
        )

        gated_no_wm_fpr = (
            gated_accept_rate(
                no_wm_df,
                args.kgw_gate,
                tau_gated,
            )
        )

        original_human_fpr = (
            original_accept_rate(
                human_df,
                tau_original,
            )
            if len(human_df)
            else np.nan
        )

        gated_human_fpr = (
            gated_accept_rate(
                human_df,
                args.kgw_gate,
                tau_gated,
            )
            if len(human_df)
            else np.nan
        )

        null_gate_pass = (
            kgw_gate_pass_rate(
                no_wm_df,
                args.kgw_gate,
            )
        )

        threshold_rows.append({
            "model": model,
            "n_no_wm": len(no_wm_df),
            "n_human": len(human_df),

            "kgw_gate":
                args.kgw_gate,

            "original_fisher_threshold":
                tau_original,

            "gated_fisher_threshold":
                tau_gated,

            "no_wm_kgw_gate_pass_rate":
                null_gate_pass,

            "original_no_wm_fpr":
                original_no_wm_fpr,

            "gated_no_wm_fpr":
                gated_no_wm_fpr,

            "original_human_fpr":
                original_human_fpr,

            "gated_human_fpr":
                gated_human_fpr,
        })

        print(
            f"    Original Fisher tau = "
            f"{tau_original:.4f}"
        )

        print(
            f"    Gated Fisher tau    = "
            f"{tau_gated:.4f}"
        )

        print(
            f"    no_wm FPR           = "
            f"{original_no_wm_fpr * 100:.2f}% "
            f"-> "
            f"{gated_no_wm_fpr * 100:.2f}%"
        )

        print(
            f"    human FPR           = "
            f"{original_human_fpr * 100:.2f}% "
            f"-> "
            f"{gated_human_fpr * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Before attack
        # ----------------------------------------------------

        original_before = (
            original_accept_rate(
                before_df,
                tau_original,
            )
        )

        gated_before = (
            gated_accept_rate(
                before_df,
                args.kgw_gate,
                tau_gated,
            )
        )

        detail_rows.append({
            "model": model,
            "condition": "Before Attack",
            "n": len(before_df),

            "original_tpr":
                original_before,

            "gated_tpr":
                gated_before,

            "delta_tpr":
                gated_before
                - original_before,

            "kgw_gate_pass_rate":
                kgw_gate_pass_rate(
                    before_df,
                    args.kgw_gate,
                ),
        })

        # ----------------------------------------------------
        # Attacks
        # ----------------------------------------------------

        original_attack_rates = []
        gated_attack_rates = []

        for attack in PAPER_ATTACK_ORDER:

            if attack not in attack_dfs:
                continue

            df = attack_dfs[
                attack
            ]

            original_tpr = (
                original_accept_rate(
                    df,
                    tau_original,
                )
            )

            gated_tpr = (
                gated_accept_rate(
                    df,
                    args.kgw_gate,
                    tau_gated,
                )
            )

            gate_pass = (
                kgw_gate_pass_rate(
                    df,
                    args.kgw_gate,
                )
            )

            original_attack_rates.append(
                original_tpr
            )

            gated_attack_rates.append(
                gated_tpr
            )

            detail_rows.append({
                "model": model,
                "condition": attack,
                "n": len(df),

                "original_tpr":
                    original_tpr,

                "gated_tpr":
                    gated_tpr,

                "delta_tpr":
                    gated_tpr
                    - original_tpr,

                "kgw_gate_pass_rate":
                    gate_pass,
            })

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        mean_original = mean_valid(
            original_attack_rates
        )

        mean_gated = mean_valid(
            gated_attack_rates
        )

        model_summary_rows.append({
            "model": model,

            "before_original":
                original_before,

            "before_gated":
                gated_before,

            "before_delta":
                gated_before
                - original_before,

            "mean_attack_original":
                mean_original,

            "mean_attack_gated":
                mean_gated,

            "mean_attack_delta":
                mean_gated
                - mean_original,

            "original_no_wm_fpr":
                original_no_wm_fpr,

            "gated_no_wm_fpr":
                gated_no_wm_fpr,

            "original_human_fpr":
                original_human_fpr,

            "gated_human_fpr":
                gated_human_fpr,

            "original_fisher_threshold":
                tau_original,

            "gated_fisher_threshold":
                tau_gated,
        })

        valid_models.append(
            model
        )

        print(
            f"    Before TPR          = "
            f"{original_before * 100:.2f}% "
            f"-> "
            f"{gated_before * 100:.2f}%"
        )

        print(
            f"    Mean attack TPR     = "
            f"{mean_original * 100:.2f}% "
            f"-> "
            f"{mean_gated * 100:.2f}% "
            f"("
            f"{(mean_gated - mean_original) * 100:+.2f} pp"
            f")"
        )

        print()

    # ========================================================
    # Check
    # ========================================================

    if not valid_models:

        raise RuntimeError(
            "No valid model result files were found."
        )

    # ========================================================
    # DataFrames
    # ========================================================

    detail_df = pd.DataFrame(
        detail_rows
    )

    summary_df = pd.DataFrame(
        model_summary_rows
    )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    # ========================================================
    # Save CSV
    # ========================================================

    detail_path = (
        output_dir
        / "cross_backbone_gate_detail.csv"
    )

    summary_path = (
        output_dir
        / "cross_backbone_gate_summary.csv"
    )

    threshold_path = (
        output_dir
        / "cross_backbone_gate_thresholds.csv"
    )

    detail_df.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Terminal: model summary
    # ========================================================

    print()
    print("=" * 110)
    print("MODEL SUMMARY")
    print("=" * 110)

    display_summary = (
        summary_df.copy()
    )

    percentage_cols = [
        "before_original",
        "before_gated",
        "before_delta",
        "mean_attack_original",
        "mean_attack_gated",
        "mean_attack_delta",
        "original_no_wm_fpr",
        "gated_no_wm_fpr",
        "original_human_fpr",
        "gated_human_fpr",
    ]

    for col in percentage_cols:
        display_summary[col] *= 100.0

    print(
        display_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    # ========================================================
    # Terminal: Original TPR
    # ========================================================

    print()
    print("=" * 110)
    print("ORIGINAL FISHER TPR (%)")
    print("=" * 110)

    original_pivot = (
        detail_df.pivot(
            index="condition",
            columns="model",
            values="original_tpr",
        )
        * 100.0
    )

    ordered_conditions = (
        ["Before Attack"]
        + [
            x
            for x in PAPER_ATTACK_ORDER
            if x in original_pivot.index
        ]
    )

    original_pivot = (
        original_pivot.reindex(
            ordered_conditions
        )
    )

    print(
        original_pivot.to_string(
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    # ========================================================
    # Terminal: Gated TPR
    # ========================================================

    print()
    print("=" * 110)
    print(
        f"KGW-GATED FISHER TPR (%) "
        f"[gate={args.kgw_gate}]"
    )
    print("=" * 110)

    gated_pivot = (
        detail_df.pivot(
            index="condition",
            columns="model",
            values="gated_tpr",
        )
        * 100.0
    )

    gated_pivot = (
        gated_pivot.reindex(
            ordered_conditions
        )
    )

    print(
        gated_pivot.to_string(
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    # ========================================================
    # Terminal: Delta
    # ========================================================

    print()
    print("=" * 110)
    print("GATED - ORIGINAL (percentage points)")
    print("=" * 110)

    delta_pivot = (
        detail_df.pivot(
            index="condition",
            columns="model",
            values="delta_tpr",
        )
        * 100.0
    )

    delta_pivot = (
        delta_pivot.reindex(
            ordered_conditions
        )
    )

    print(
        delta_pivot.to_string(
            float_format=lambda x: (
                f"{x:+.2f}"
            ),
        )
    )

    # ========================================================
    # Terminal: KGW gate pass
    # ========================================================

    print()
    print("=" * 110)
    print(
        f"KGW GATE PASS RATE (%) "
        f"[z_KGW >= {args.kgw_gate}]"
    )
    print("=" * 110)

    gate_pivot = (
        detail_df.pivot(
            index="condition",
            columns="model",
            values="kgw_gate_pass_rate",
        )
        * 100.0
    )

    gate_pivot = (
        gate_pivot.reindex(
            ordered_conditions
        )
    )

    print(
        gate_pivot.to_string(
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    # ========================================================
    # Saved
    # ========================================================

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)

    print(
        f"[+] {summary_path}"
    )

    print(
        f"[+] {detail_path}"
    )

    print(
        f"[+] {threshold_path}"
    )


if __name__ == "__main__":
    main()


# python analysis/security_validation/evaluate_gate_cross_backbone.py \
#     --results_root results/main_exp \
#     --dataset c4 \
#     --kgw_gate 2.0 \
#     --target_fpr 0.01 \
#     --output_dir results/security_validation/kgw_gate/cross_backbone/gate2p0