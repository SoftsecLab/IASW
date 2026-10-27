# analysis/security_validation/sweep_kgw_gate_cross_backbone.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-backbone KGW gate sweep.

For each backbone and each KGW gate:

    accept iff
        z_KGW >= gate
        AND
        z_Fisher >= tau_F(gate, backbone)

tau_F is independently calibrated on that backbone's no_wm samples
to maintain the target empirical FPR.

Current paper attacks:
    pegasus
    dipper
    back_translation
    copy_paste
    del_ins
    drop
    swap
    typo


Optional:
    If Llama3.1 spoof files are supplied, also evaluate:
        Random-K32
        Estimated-N200-K32
        True-Anchor-K32

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

DEFAULT_GATES = [
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
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
    if not isinstance(d, dict):
        return np.nan, np.nan

    z_kgw = safe_float(
        d.get("z_score_micro", np.nan)
    )

    z_fisher = safe_float(
        d.get(
            "z_score_fisher",
            d.get("z_score", np.nan),
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
# Main experiment extraction
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
    available = set()

    for row in rows:
        attacks = row.get("attacks", {})

        if isinstance(attacks, dict):
            available.update(
                attacks.keys()
            )

    available -= EXCLUDED_ATTACKS

    result = {}

    for attack in PAPER_ATTACK_ORDER:

        if attack not in available:
            continue

        out = []

        for row in rows:

            attacks = row.get(
                "attacks",
                {},
            )

            if not isinstance(attacks, dict):
                continue

            attack_record = attacks.get(
                attack,
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
            result[attack] = pd.DataFrame(out)

    return result


# ============================================================
# Spoof extraction
# ============================================================

def collect_spoof(path):
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
# Calibration
# ============================================================

def calibrate_gated_fisher(
    null_df,
    kgw_gate,
    target_fpr,
):
    """
    Calibrate tau_F for:

        z_KGW >= kgw_gate
        AND
        z_Fisher >= tau_F

    using the complete no_wm null distribution.
    """

    if len(null_df) == 0:
        raise RuntimeError(
            "Empty no_wm distribution."
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

    # KGW gate alone is already restrictive enough.
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
# Decision
# ============================================================

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
):
    model_dir = (
        Path(results_root)
        / dataset
        / model
        / "records"
    )

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
        "--results_root",
        default="results/main_exp",
    )

    p.add_argument(
        "--dataset",
        default="c4",
    )

    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )

    p.add_argument(
        "--kgw_gates",
        nargs="+",
        type=float,
        default=DEFAULT_GATES,
    )

    p.add_argument(
        "--target_fpr",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--output_dir",
        default=(
            "results/security_validation/"
            "kgw_gate/cross_backbone/sweep"
        ),
    )

    # Optional spoofing files.
    p.add_argument(
        "--random_file",
        default=None,
    )

    p.add_argument(
        "--estimated_file",
        default=None,
    )

    p.add_argument(
        "--true_file",
        default=None,
    )

    p.add_argument(
        "--spoof_model",
        default="llama3.1-8b-instruct",
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
    print("CROSS-BACKBONE KGW GATE SWEEP")
    print("=" * 110)

    print(
        f"Dataset        : {args.dataset}"
    )

    print(
        f"Target FPR     : "
        f"{args.target_fpr:.3%}"
    )

    print(
        f"KGW gates      : "
        f"{args.kgw_gates}"
    )

    print(
        "Null calibration: no_wm only"
    )

    print(
        f"Excluded       : "
        f"{sorted(EXCLUDED_ATTACKS)}"
    )

    # ========================================================
    # Optional spoofing
    # ========================================================

    spoof_dfs = {}

    if args.random_file:
        spoof_dfs["Random-K32"] = (
            collect_spoof(
                args.random_file
            )
        )

    if args.estimated_file:
        spoof_dfs[
            "Estimated-N200-K32"
        ] = collect_spoof(
            args.estimated_file
        )

    if args.true_file:
        spoof_dfs[
            "True-Anchor-K32"
        ] = collect_spoof(
            args.true_file
        )

    if spoof_dfs:

        print()
        print("[*] Spoofing data:")

        for name, df in spoof_dfs.items():
            print(
                f"    {name:<24} "
                f"n={len(df)}"
            )

    # ========================================================
    # Results
    # ========================================================

    detail_rows = []
    model_gate_rows = []
    spoof_rows = []

    valid_models = []

    # ========================================================
    # Models
    # ========================================================

    for model in args.models:

        print()
        print("-" * 110)
        print(f"MODEL: {model}")
        print("-" * 110)

        main_file = find_main_file(
            args.results_root,
            args.dataset,
            model,
        )

        if main_file is None:

            print(
                "[WARN] Result file not found. "
                "Skipping."
            )

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
                "[WARN] Empty no_wm. Skipping."
            )
            continue

        valid_models.append(
            model
        )

        # ====================================================
        # Gate sweep
        # ====================================================

        for gate in args.kgw_gates:

            tau = calibrate_gated_fisher(
                null_df=no_wm_df,
                kgw_gate=gate,
                target_fpr=args.target_fpr,
            )

            no_wm_fpr = (
                gated_accept_rate(
                    no_wm_df,
                    gate,
                    tau,
                )
            )

            human_fpr = (
                gated_accept_rate(
                    human_df,
                    gate,
                    tau,
                )
                if len(human_df)
                else np.nan
            )

            before_tpr = (
                gated_accept_rate(
                    before_df,
                    gate,
                    tau,
                )
            )

            before_gate_pass = (
                kgw_gate_pass_rate(
                    before_df,
                    gate,
                )
            )

            detail_rows.append({
                "model": model,
                "kgw_gate": gate,
                "condition": "Before Attack",
                "n": len(before_df),
                "fisher_threshold": tau,
                "tpr": before_tpr,
                "kgw_gate_pass_rate":
                    before_gate_pass,
            })

            attack_rates = []

            for attack in PAPER_ATTACK_ORDER:

                if attack not in attack_dfs:
                    continue

                df = attack_dfs[
                    attack
                ]

                tpr = gated_accept_rate(
                    df,
                    gate,
                    tau,
                )

                gate_pass = (
                    kgw_gate_pass_rate(
                        df,
                        gate,
                    )
                )

                attack_rates.append(
                    tpr
                )

                detail_rows.append({
                    "model": model,
                    "kgw_gate": gate,
                    "condition": attack,
                    "n": len(df),
                    "fisher_threshold": tau,
                    "tpr": tpr,
                    "kgw_gate_pass_rate":
                        gate_pass,
                })

            mean_attack = mean_valid(
                attack_rates
            )

            min_attack = min_valid(
                attack_rates
            )

            dipper_tpr = np.nan

            if "dipper" in attack_dfs:
                dipper_tpr = (
                    gated_accept_rate(
                        attack_dfs["dipper"],
                        gate,
                        tau,
                    )
                )

            pegasus_tpr = np.nan

            if "pegasus" in attack_dfs:
                pegasus_tpr = (
                    gated_accept_rate(
                        attack_dfs["pegasus"],
                        gate,
                        tau,
                    )
                )

            copy_paste_tpr = np.nan

            if "copy_paste" in attack_dfs:
                copy_paste_tpr = (
                    gated_accept_rate(
                        attack_dfs["copy_paste"],
                        gate,
                        tau,
                    )
                )

            model_gate_rows.append({
                "model": model,
                "kgw_gate": gate,

                "fisher_threshold":
                    tau,

                "no_wm_fpr":
                    no_wm_fpr,

                "human_fpr":
                    human_fpr,

                "before_tpr":
                    before_tpr,

                "mean_attack_tpr":
                    mean_attack,

                "min_attack_tpr":
                    min_attack,

                "dipper_tpr":
                    dipper_tpr,

                "pegasus_tpr":
                    pegasus_tpr,

                "copy_paste_tpr":
                    copy_paste_tpr,
            })

            # =================================================
            # Spoofing only for designated model
            # =================================================

            if (
                model == args.spoof_model
                and spoof_dfs
            ):

                for (
                    group_name,
                    spoof_df,
                ) in spoof_dfs.items():

                    spoof_rate = (
                        gated_accept_rate(
                            spoof_df,
                            gate,
                            tau,
                        )
                    )

                    spoof_gate_pass = (
                        kgw_gate_pass_rate(
                            spoof_df,
                            gate,
                        )
                    )

                    spoof_rows.append({
                        "model":
                            model,

                        "kgw_gate":
                            gate,

                        "fisher_threshold":
                            tau,

                        "group":
                            group_name,

                        "n":
                            len(spoof_df),

                        "spoof_success_rate":
                            spoof_rate,

                        "kgw_gate_pass_rate":
                            spoof_gate_pass,
                    })

            print(
                f"    gate={gate:<4.2f} "
                f"tau={tau:>7.3f} | "
                f"FPR={no_wm_fpr * 100:>5.2f}% | "
                f"Before={before_tpr * 100:>6.2f}% | "
                f"Mean={mean_attack * 100:>6.2f}% | "
                f"DIPPER={dipper_tpr * 100:>6.2f}%"
            )

    if not valid_models:
        raise RuntimeError(
            "No valid model files found."
        )

    # ========================================================
    # DataFrames
    # ========================================================

    detail_df = pd.DataFrame(
        detail_rows
    )

    model_gate_df = pd.DataFrame(
        model_gate_rows
    )

    spoof_df = pd.DataFrame(
        spoof_rows
    )

    # ========================================================
    # Cross-model macro summary
    # ========================================================

    macro_rows = []

    for gate in args.kgw_gates:

        subset = model_gate_df[
            model_gate_df[
                "kgw_gate"
            ] == gate
        ]

        if len(subset) == 0:
            continue

        macro_rows.append({
            "kgw_gate":
                gate,

            "n_models":
                len(subset),

            "mean_before_tpr":
                mean_valid(
                    subset[
                        "before_tpr"
                    ].tolist()
                ),

            "mean_attack_tpr":
                mean_valid(
                    subset[
                        "mean_attack_tpr"
                    ].tolist()
                ),

            "mean_dipper_tpr":
                mean_valid(
                    subset[
                        "dipper_tpr"
                    ].tolist()
                ),

            "mean_pegasus_tpr":
                mean_valid(
                    subset[
                        "pegasus_tpr"
                    ].tolist()
                ),

            "mean_copy_paste_tpr":
                mean_valid(
                    subset[
                        "copy_paste_tpr"
                    ].tolist()
                ),

            "mean_human_fpr":
                mean_valid(
                    subset[
                        "human_fpr"
                    ].tolist()
                ),

            "mean_no_wm_fpr":
                mean_valid(
                    subset[
                        "no_wm_fpr"
                    ].tolist()
                ),
        })

    macro_df = pd.DataFrame(
        macro_rows
    )

    # ========================================================
    # Merge Llama3.1 security into macro summary
    # ========================================================

    if len(spoof_df) > 0:

        security_pivot = (
            spoof_df.pivot(
                index="kgw_gate",
                columns="group",
                values="spoof_success_rate",
            )
            .reset_index()
        )

        rename_map = {
            "Random-K32":
                "random_spoof",

            "Estimated-N200-K32":
                "estimated_spoof",

            "True-Anchor-K32":
                "true_anchor_spoof",
        }

        security_pivot = (
            security_pivot.rename(
                columns=rename_map
            )
        )

        macro_df = macro_df.merge(
            security_pivot,
            on="kgw_gate",
            how="left",
        )

    # ========================================================
    # Save
    # ========================================================

    detail_path = (
        output_dir
        / "gate_sweep_detail.csv"
    )

    model_gate_path = (
        output_dir
        / "gate_sweep_by_model.csv"
    )

    macro_path = (
        output_dir
        / "gate_sweep_macro_summary.csv"
    )

    spoof_path = (
        output_dir
        / "gate_sweep_spoof_security.csv"
    )

    detail_df.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    model_gate_df.to_csv(
        model_gate_path,
        index=False,
        encoding="utf-8-sig",
    )

    macro_df.to_csv(
        macro_path,
        index=False,
        encoding="utf-8-sig",
    )

    if len(spoof_df) > 0:
        spoof_df.to_csv(
            spoof_path,
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # Display: Macro
    # ========================================================

    print()
    print("=" * 110)
    print("FIVE-BACKBONE MACRO SUMMARY")
    print("=" * 110)

    macro_display = (
        macro_df.copy()
    )

    percent_cols = [
        "mean_before_tpr",
        "mean_attack_tpr",
        "mean_dipper_tpr",
        "mean_pegasus_tpr",
        "mean_copy_paste_tpr",
        "mean_human_fpr",
        "mean_no_wm_fpr",
        "random_spoof",
        "estimated_spoof",
        "true_anchor_spoof",
    ]

    for col in percent_cols:

        if col in macro_display.columns:
            macro_display[col] *= 100.0

    print(
        macro_display.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    # ========================================================
    # Display: mean attack by model
    # ========================================================

    print()
    print("=" * 110)
    print("MEAN 8-ATTACK TPR (%)")
    print("=" * 110)

    pivot = (
        model_gate_df.pivot(
            index="model",
            columns="kgw_gate",
            values="mean_attack_tpr",
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

    # ========================================================
    # Display: DIPPER
    # ========================================================

    print()
    print("=" * 110)
    print("DIPPER TPR (%)")
    print("=" * 110)

    pivot = (
        model_gate_df.pivot(
            index="model",
            columns="kgw_gate",
            values="dipper_tpr",
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

    # ========================================================
    # Display: Pegasus
    # ========================================================

    print()
    print("=" * 110)
    print("PEGASUS TPR (%)")
    print("=" * 110)

    pivot = (
        model_gate_df.pivot(
            index="model",
            columns="kgw_gate",
            values="pegasus_tpr",
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

    # ========================================================
    # Display: Copy-Paste
    # ========================================================

    print()
    print("=" * 110)
    print("COPY-PASTE TPR (%)")
    print("=" * 110)

    pivot = (
        model_gate_df.pivot(
            index="model",
            columns="kgw_gate",
            values="copy_paste_tpr",
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

    # ========================================================
    # Display spoof
    # ========================================================

    if len(spoof_df) > 0:

        print()
        print("=" * 110)
        print(
            f"SPOOF SECURITY (%) "
            f"[{args.spoof_model}]"
        )
        print("=" * 110)

        pivot = (
            spoof_df.pivot(
                index="group",
                columns="kgw_gate",
                values="spoof_success_rate",
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

    # ========================================================
    # Saved
    # ========================================================

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)

    print(
        f"[+] {macro_path}"
    )

    print(
        f"[+] {model_gate_path}"
    )

    print(
        f"[+] {detail_path}"
    )

    if len(spoof_df) > 0:
        print(
            f"[+] {spoof_path}"
        )


if __name__ == "__main__":
    main()

# python analysis/security_validation/sweep_kgw_gate_cross_backbone.py \
#     --results_root results/main_exp \
#     --dataset c4 \
#     --models \
#         opt-1.3b-semstamp \
#         Llama-3.2-3B-Instruct \
#         Mistral-7B-Instruct \
#         qwen2.5-7b-instruct \
#         llama3.1-8b-instruct \
#     --kgw_gates 1.0 1.25 1.5 1.75 2.0 \
#     --target_fpr 0.01 \
#     --random_file results/security_validation/spoofing/c4/llama3.1-8b-instruct/random_k32.jsonl \
#     --estimated_file results/security_validation/spoofing/c4/llama3.1-8b-instruct/estimated_n200_k32.jsonl \
#     --true_file results/security_validation/spoofing/c4/llama3.1-8b-instruct/true_anchor_k32.jsonl \
#     --spoof_model llama3.1-8b-instruct \
#     --output_dir results/security_validation/kgw_gate/cross_backbone/sweep
