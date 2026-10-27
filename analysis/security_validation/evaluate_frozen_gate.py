#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the frozen KGW security gate.

Final decision:
    accept = (z_KGW >= tau_KGW) AND (z_Fisher >= tau_Fisher)

Thresholds are loaded from an independent security-calibration result.

This script:
1. Evaluates Before Attack and all attacks in the main dual_kgw_iasw JSONL.
3. Evaluates existing spoofing JSONL files.
4. Does NOT recalibrate thresholds.
5. Does NOT rerun detection.
6. Does NOT modify source JSONL files.
"""

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--threshold_file",
        required=True,
        help="Independent Sec-Cal gate_threshold.json",
    )

    p.add_argument(
        "--main_file",
        required=True,
        help="Main dual_kgw_iasw.jsonl",
    )

    p.add_argument(
        "--random_file",
        default=None,
        help="Detected Random-K32 spoof JSONL",
    )

    p.add_argument(
        "--estimated_file",
        default=None,
        help="Detected Estimated-anchor-K32 spoof JSONL",
    )

    p.add_argument(
        "--oracle_file",
        default=None,
        help="Detected True-anchor-K32 spoof JSONL",
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    p.add_argument(
        "--exclude_attacks",
        default="",
        help="Comma-separated attacks to exclude",
    )

    return p.parse_args()


# ============================================================
# Basic helpers
# ============================================================

def safe_float(x: Any) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def load_thresholds(path: str) -> Tuple[float, float, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    kgw_gate = safe_float(obj.get("kgw_gate"))
    fisher_threshold = safe_float(obj.get("fisher_threshold"))

    if kgw_gate is None:
        raise RuntimeError(
            f"Missing valid kgw_gate in {path}"
        )

    if fisher_threshold is None:
        raise RuntimeError(
            f"Missing valid fisher_threshold in {path}"
        )

    return kgw_gate, fisher_threshold, obj


# ============================================================
# Detection parser
# ============================================================

def parse_detection(
    res: Any,
) -> Optional[Dict[str, float]]:

    if not isinstance(res, dict):
        return None

    if "error" in res:
        return None

    z_kgw = safe_float(
        res.get("z_score_micro")
    )

    z_iasw = safe_float(
        res.get(
            "z_score_macro_calibrated",
            res.get("z_score_macro"),
        )
    )

    z_fisher = safe_float(
        res.get("z_score_fisher")
    )

    if z_kgw is None or z_fisher is None:
        return None

    return {
        "z_kgw": z_kgw,
        "z_iasw": (
            z_iasw
            if z_iasw is not None
            else float("nan")
        ),
        "z_fisher": z_fisher,
    }


# ============================================================
# Core evaluation
# ============================================================

def evaluate_scores(
    scores: List[Dict[str, float]],
    kgw_gate: float,
    fisher_threshold: float,
) -> Dict[str, Any]:

    if not scores:
        return {
            "n": 0,
            "mean_z_kgw": None,
            "mean_z_iasw": None,
            "mean_z_fisher": None,
            "kgw_gate_pass_rate": None,
            "fisher_only_accept_rate": None,
            "final_accept_rate": None,
        }

    z_kgw = np.asarray(
        [x["z_kgw"] for x in scores],
        dtype=float,
    )

    z_iasw = np.asarray(
        [x["z_iasw"] for x in scores],
        dtype=float,
    )

    z_fisher = np.asarray(
        [x["z_fisher"] for x in scores],
        dtype=float,
    )

    kgw_pass = z_kgw >= kgw_gate

    fisher_pass = (
        z_fisher >= fisher_threshold
    )

    final_accept = (
        kgw_pass & fisher_pass
    )

    if np.all(np.isnan(z_iasw)):
        mean_z_iasw = None
    else:
        mean_z_iasw = float(
            np.nanmean(z_iasw)
        )

    return {
        "n": int(len(scores)),

        "mean_z_kgw": float(
            np.mean(z_kgw)
        ),

        "mean_z_iasw": mean_z_iasw,

        "mean_z_fisher": float(
            np.mean(z_fisher)
        ),

        "kgw_gate_pass_n": int(
            np.sum(kgw_pass)
        ),

        "kgw_gate_pass_rate": float(
            np.mean(kgw_pass)
        ),

        "fisher_only_accept_n": int(
            np.sum(fisher_pass)
        ),

        "fisher_only_accept_rate": float(
            np.mean(fisher_pass)
        ),

        "final_accept_n": int(
            np.sum(final_accept)
        ),

        "final_accept_rate": float(
            np.mean(final_accept)
        ),
    }


# ============================================================
# Main experiment
# ============================================================

def evaluate_main(
    path: str,
    kgw_gate: float,
    fisher_threshold: float,
    excluded_attacks: set,
) -> Dict[str, Any]:

    rows = load_jsonl(path)

    before_scores = []

    attack_scores: Dict[
        str,
        List[Dict[str, float]]
    ] = {}

    for row in rows:

        # --------------------------------------------
        # Before Attack
        # --------------------------------------------

        parsed = parse_detection(
            row.get("detection_wm", {})
        )

        if parsed is not None:
            before_scores.append(parsed)

        # --------------------------------------------
        # Attacks
        # --------------------------------------------

        attacks = row.get("attacks", {})

        if not isinstance(attacks, dict):
            continue

        for attack_name, attack_obj in attacks.items():

            if attack_name in excluded_attacks:
                continue

            if not isinstance(attack_obj, dict):
                continue

            res = attack_obj.get(
                "detection_results",
                {},
            )

            parsed = parse_detection(res)

            if parsed is None:
                continue

            attack_scores.setdefault(
                attack_name,
                [],
            ).append(parsed)

    result = {
        "Before Attack": evaluate_scores(
            before_scores,
            kgw_gate,
            fisher_threshold,
        )
    }

    for attack_name, scores in attack_scores.items():

        result[
            f"After {attack_name}"
        ] = evaluate_scores(
            scores,
            kgw_gate,
            fisher_threshold,
        )

    return result


# ============================================================
# Spoof experiment
# ============================================================

def find_spoof_detection(
    row: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """
    Existing spoof files were detected through runner baseline_eval.

    Preferred field:
        detection_wm

    A few experimental files may store a direct detection result,
    so small fallbacks are included.
    """

    candidates = [
        row.get("detection_wm"),
        row.get("detection_results"),
        row.get("detection"),
    ]

    for res in candidates:
        parsed = parse_detection(res)

        if parsed is not None:
            return parsed

    # Last fallback:
    # detection fields may have been written directly
    # into the top-level row.
    parsed = parse_detection(row)

    if parsed is not None:
        return parsed

    return None


def evaluate_spoof(
    path: str,
    kgw_gate: float,
    fisher_threshold: float,
) -> Dict[str, Any]:

    rows = load_jsonl(path)

    scores = []

    for row in rows:

        parsed = find_spoof_detection(row)

        if parsed is not None:
            scores.append(parsed)

    result = evaluate_scores(
        scores,
        kgw_gate,
        fisher_threshold,
    )

    result["source_file"] = path
    result["total_rows_in_file"] = len(rows)
    result["valid_detection_rows"] = len(scores)

    return result


# ============================================================
# CSV
# ============================================================

def save_csv(
    path: str,
    main_results: Dict[str, Any],
    spoof_results: Dict[str, Any],
):

    fields = [
        "group",
        "condition",
        "n",
        "mean_z_kgw",
        "mean_z_iasw",
        "mean_z_fisher",
        "kgw_gate_pass_n",
        "kgw_gate_pass_rate",
        "fisher_only_accept_n",
        "fisher_only_accept_rate",
        "final_accept_n",
        "final_accept_rate",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for condition, res in main_results.items():

            row = {
                "group": "main",
                "condition": condition,
            }

            for key in fields[2:]:
                row[key] = res.get(key)

            writer.writerow(row)

        for condition, res in spoof_results.items():

            row = {
                "group": "spoof",
                "condition": condition,
            }

            for key in fields[2:]:
                row[key] = res.get(key)

            writer.writerow(row)


# ============================================================
# Pretty output
# ============================================================

def pct(x):
    if x is None:
        return "NA"

    return f"{100.0 * x:.2f}%"


def print_result(
    name: str,
    res: Dict[str, Any],
):

    n = res.get("n", 0)

    if n == 0:
        print(
            f"{name:28s} "
            f"N=0"
        )
        return

    print(
        f"{name:28s} "
        f"N={n:4d} | "
        f"KGWpass={pct(res['kgw_gate_pass_rate']):>7s} | "
        f"FisherOnly={pct(res['fisher_only_accept_rate']):>7s} | "
        f"Final={pct(res['final_accept_rate']):>7s} | "
        f"zK={res['mean_z_kgw']:+.3f} | "
        f"zF={res['mean_z_fisher']:+.3f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    kgw_gate, fisher_threshold, threshold_meta = (
        load_thresholds(
            args.threshold_file
        )
    )

    excluded_attacks = {
        x.strip()
        for x in args.exclude_attacks.split(",")
        if x.strip()
    }

    print()
    print("=" * 100)
    print("FROZEN SECURITY GATE EVALUATION")
    print("=" * 100)

    print(
        f"KGW gate         = "
        f"{kgw_gate:.6f}"
    )

    print(
        f"Fisher threshold = "
        f"{fisher_threshold:.6f}"
    )

    print(
        f"Calibration FPR  = "
        f"{100 * threshold_meta.get('empirical_calibration_fpr', float('nan')):.3f}%"
    )

    print(
        f"Excluded attacks = "
        f"{sorted(excluded_attacks)}"
    )

    print("=" * 100)

    # --------------------------------------------------------
    # Main
    # --------------------------------------------------------

    main_results = evaluate_main(
        args.main_file,
        kgw_gate,
        fisher_threshold,
        excluded_attacks,
    )

    print()
    print("-" * 100)
    print("MAIN ROBUSTNESS")
    print("-" * 100)

    for name, res in main_results.items():
        print_result(name, res)

    # --------------------------------------------------------
    # Spoof
    # --------------------------------------------------------

    spoof_results = {}

    spoof_inputs = [
        (
            "Random-K32",
            args.random_file,
        ),
        (
            "Estimated-N200-K32",
            args.estimated_file,
        ),
        (
            "True-Anchor-K32",
            args.oracle_file,
        ),
    ]

    for name, path in spoof_inputs:

        if path is None:
            continue

        if not os.path.exists(path):
            print(
                f"[WARN] Spoof file not found: "
                f"{path}"
            )
            continue

        spoof_results[name] = evaluate_spoof(
            path,
            kgw_gate,
            fisher_threshold,
        )

    print()
    print("-" * 100)
    print("SECURITY / SPOOFING")
    print("-" * 100)

    for name, res in spoof_results.items():
        print_result(name, res)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    report = {
        "protocol": "frozen_security_gate_evaluation",

        "threshold_source": args.threshold_file,

        "kgw_gate": kgw_gate,

        "fisher_threshold": fisher_threshold,

        "calibration": threshold_meta,

        "excluded_attacks": sorted(
            excluded_attacks
        ),

        "main": main_results,

        "spoof": spoof_results,
    }

    json_path = os.path.join(
        args.output_dir,
        "frozen_gate_results.json",
    )

    csv_path = os.path.join(
        args.output_dir,
        "frozen_gate_results.csv",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    save_csv(
        csv_path,
        main_results,
        spoof_results,
    )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()


# cd ./IASW

# python analysis/security_validation/evaluate_frozen_gate.py \
#     --threshold_file results/security_validation/sec_calibration/c4/llama3.1-8b-instruct/gate_threshold.json \
#     --main_file results/main_exp/c4/llama3.1-8b-instruct/records/dual_kgw_iasw.jsonl \
#     --random_file results/security_validation/spoofing/c4/llama3.1-8b-instruct/random_k32.jsonl \
#     --estimated_file results/security_validation/spoofing/c4/llama3.1-8b-instruct/estimated_n200_k32.jsonl \
#     --oracle_file results/security_validation/spoofing/c4/llama3.1-8b-instruct/true_anchor_k32.jsonl \
#     --output_dir results/security_validation/frozen_gate/c4/llama3.1-8b-instruct