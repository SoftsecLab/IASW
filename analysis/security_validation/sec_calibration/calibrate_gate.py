#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibrate the final Fisher threshold for the KGW security gate.

Final decision:

    accept =
        (z_KGW >= tau_K)
        AND
        (z_Fisher >= tau_F)

tau_K is fixed beforehand.
tau_F is calibrated on an independent no-watermark calibration set.
"""

import argparse
import json
import math
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        required=True,
        help="Detected Sec-Cal JSONL."
    )

    parser.add_argument(
        "--output_file",
        required=True
    )

    parser.add_argument(
        "--kgw_gate",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--target_fpr",
        type=float,
        default=0.01
    )

    return parser.parse_args()


def parse_detection_result(res):
    if not isinstance(res, dict):
        return None

    z_kgw = res.get("z_score_micro")
    z_fisher = res.get("z_score_fisher")

    if z_kgw is None or z_fisher is None:
        return None

    try:
        z_kgw = float(z_kgw)
        z_fisher = float(z_fisher)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(z_kgw):
        return None

    if not math.isfinite(z_fisher):
        return None

    return z_kgw, z_fisher


def load_scores(path):
    scores = []

    bad = 0
    total = 0

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            total += 1

            row = json.loads(line)

            #
            # build_sec_cal.py deliberately places the null text
            # into wm_text.
            #
            # Therefore use detection_wm here.
            #
            res = row.get("detection_wm", {})

            parsed = parse_detection_result(res)

            if parsed is None:
                bad += 1
                continue

            scores.append(parsed)

    print(f"Total rows : {total}")
    print(f"Valid rows : {len(scores)}")
    print(f"Bad rows   : {bad}")

    if not scores:
        raise RuntimeError(
            "No valid detection results found."
        )

    return np.asarray(scores, dtype=float)


def calibrate_threshold(
    z_kgw,
    z_fisher,
    kgw_gate,
    target_fpr,
):

    n = len(z_kgw)

    gate_mask = z_kgw >= kgw_gate

    gate_pass_n = int(gate_mask.sum())
    gate_pass_rate = float(gate_mask.mean())

    print()
    print(f"KGW gate            : {kgw_gate:.6f}")
    print(f"KGW gate pass       : {gate_pass_n}/{n}")
    print(
        f"KGW gate pass rate  : "
        f"{gate_pass_rate * 100:.4f}%"
    )

    if gate_pass_n == 0:
        raise RuntimeError(
            "No calibration samples pass the KGW gate."
        )

    #
    # Important:
    #
    # FPR denominator is ALL null samples.
    #
    # FPR(tau) =
    #
    #   count(
    #       z_KGW >= gate
    #       AND
    #       z_Fisher >= tau
    #   )
    #   --------------------------------
    #              N_null
    #
    #

    candidate_thresholds = np.unique(
        z_fisher[gate_mask]
    )

    candidate_thresholds = np.sort(
        candidate_thresholds
    )

    best_threshold = None
    achieved_fpr = None
    accepted_n = None

    #
    # We want the LOWEST Fisher threshold whose
    # empirical final FPR is <= target.
    #
    # This gives the highest sensitivity subject
    # to the FPR constraint.
    #
    for threshold in candidate_thresholds:

        accepted = (
            gate_mask
            &
            (z_fisher >= threshold)
        )

        current_n = int(accepted.sum())
        current_fpr = current_n / n

        if current_fpr <= target_fpr:

            best_threshold = float(threshold)
            achieved_fpr = float(current_fpr)
            accepted_n = current_n

            break

    #
    # In the unlikely event that even the largest
    # observed score exceeds the desired constraint,
    # move the threshold just above the maximum.
    #
    if best_threshold is None:

        best_threshold = float(
            np.max(z_fisher[gate_mask])
            + 1e-12
        )

        achieved_fpr = 0.0
        accepted_n = 0

    return {
        "fisher_threshold": best_threshold,
        "empirical_fpr": achieved_fpr,
        "accepted_n": accepted_n,
        "gate_pass_n": gate_pass_n,
        "gate_pass_rate": gate_pass_rate,
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("Security Gate Calibration")
    print("=" * 80)

    scores = load_scores(
        args.input_file
    )

    z_kgw = scores[:, 0]
    z_fisher = scores[:, 1]

    result = calibrate_threshold(
        z_kgw=z_kgw,
        z_fisher=z_fisher,
        kgw_gate=args.kgw_gate,
        target_fpr=args.target_fpr,
    )

    report = {
        "protocol": (
            "independent_security_calibration"
        ),

        "negative_source": "no_wm",

        "n_calibration": int(len(scores)),

        "kgw_gate": float(
            args.kgw_gate
        ),

        "target_fpr": float(
            args.target_fpr
        ),

        "fisher_threshold": result[
            "fisher_threshold"
        ],

        "empirical_calibration_fpr": result[
            "empirical_fpr"
        ],

        "accepted_n": result[
            "accepted_n"
        ],

        "kgw_gate_pass_n": result[
            "gate_pass_n"
        ],

        "kgw_gate_pass_rate": result[
            "gate_pass_rate"
        ],

        "z_kgw_summary": {
            "mean": float(
                np.mean(z_kgw)
            ),
            "std": float(
                np.std(z_kgw)
            ),
            "max": float(
                np.max(z_kgw)
            ),
        },

        "z_fisher_summary": {
            "mean": float(
                np.mean(z_fisher)
            ),
            "std": float(
                np.std(z_fisher)
            ),
            "max": float(
                np.max(z_fisher)
            ),
        },
    }

    output_dir = os.path.dirname(
        args.output_file
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    with open(
        args.output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("CALIBRATION RESULT")
    print("=" * 80)

    print(
        f"N calibration       : "
        f"{report['n_calibration']}"
    )

    print(
        f"KGW gate            : "
        f"{report['kgw_gate']:.6f}"
    )

    print(
        f"Fisher threshold    : "
        f"{report['fisher_threshold']:.6f}"
    )

    print(
        f"Target FPR          : "
        f"{report['target_fpr'] * 100:.3f}%"
    )

    print(
        f"Empirical FPR       : "
        f"{report['empirical_calibration_fpr'] * 100:.3f}%"
    )

    print(
        f"Accepted null       : "
        f"{report['accepted_n']}"
        f"/{report['n_calibration']}"
    )

    print(
        f"KGW gate pass       : "
        f"{report['kgw_gate_pass_n']}"
        f"/{report['n_calibration']}"
    )

    print()
    print(
        f"Saved to: "
        f"{args.output_file}"
    )


if __name__ == "__main__":
    main()
