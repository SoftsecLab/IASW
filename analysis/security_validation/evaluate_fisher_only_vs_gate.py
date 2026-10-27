#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fair comparison:
1. Fisher-only detector independently calibrated to 1% FPR
   on the SAME 5000-sample Sec-Cal no_wm set.
2. Compare with the already frozen KGW-gated detector.

No generation / attack / model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEC_CAL_DIR = (
    PROJECT_ROOT
    / "results/security_validation/sec_calibration"
    / "c4/llama3.1-8b-instruct"
)

SPOOF_DIR = (
    PROJECT_ROOT
    / "results/security_validation/spoofing"
    / "c4/llama3.1-8b-instruct"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/security_validation/fisher_only_comparison"
    / "c4/llama3.1-8b-instruct"
)

SPOOF_FILES = {
    "Random-K32":
        SPOOF_DIR / "random_k32.jsonl",

    "Estimated-N200-K32":
        SPOOF_DIR / "estimated_n200_k32.jsonl",

    "True-Anchor-K32":
        SPOOF_DIR / "true_anchor_k32.jsonl",
}


# ============================================================
# Frozen security detector
# IMPORTANT: do not retune these values
# ============================================================

TARGET_FPR = 0.01

KGW_GATE = 2.0

GATED_FISHER_THRESHOLD = 2.236751


# ============================================================
# Score names
# ============================================================

FISHER_KEYS = [
    "z_score_fisher",
    "z_fisher",
    "fisher_z",
    "fisher_score",
]

KGW_KEYS = [
    "z_score_micro",
    "z_score_kgw",
    "z_kgw",
    "kgw_z",
]


# ============================================================
# IO
# ============================================================

def load_records(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    if path.suffix == ".jsonl":
        rows = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line_no, line in enumerate(f, 1):

                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)

                except json.JSONDecodeError as e:

                    raise ValueError(
                        f"Bad JSON at "
                        f"{path}:{line_no}"
                    ) from e

                if isinstance(obj, dict):
                    rows.append(obj)

        return rows

    if path.suffix == ".json":

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return [
                x for x in obj
                if isinstance(x, dict)
            ]

        if isinstance(obj, dict):

            for key in [
                "records",
                "samples",
                "data",
                "results",
            ]:

                if isinstance(
                    obj.get(key),
                    list,
                ):

                    return [
                        x
                        for x in obj[key]
                        if isinstance(x, dict)
                    ]

            return [obj]

    raise ValueError(
        f"Unsupported file: {path}"
    )


# ============================================================
# Score extraction
# ============================================================

def recursive_find_number(
    obj,
    candidate_keys,
):

    if isinstance(obj, dict):

        # First search exact keys at current level
        for key in candidate_keys:

            if key not in obj:
                continue

            try:
                value = float(
                    obj[key]
                )

                if np.isfinite(value):
                    return value

            except (
                TypeError,
                ValueError,
            ):
                pass

        # Then search nested structures
        for value in obj.values():

            result = recursive_find_number(
                value,
                candidate_keys,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):

        for value in obj:

            result = recursive_find_number(
                value,
                candidate_keys,
            )

            if result is not None:
                return result

    return None


def prefer_no_wm_block(row):
    """
    If the record contains a dedicated detection_no_wm block,
    use it instead of accidentally reading another detection block.
    """

    if not isinstance(row, dict):
        return row

    preferred = [
        "detection_no_wm",
        "no_wm_detection",
        "no_wm",
    ]

    for key in preferred:

        block = row.get(key)

        if isinstance(block, dict):
            return block

    return row


def extract_scores(
    rows,
    use_no_wm_block=False,
):

    fisher = []
    kgw = []

    skipped = 0

    for row in rows:

        source = (
            prefer_no_wm_block(row)
            if use_no_wm_block
            else row
        )

        z_fisher = recursive_find_number(
            source,
            FISHER_KEYS,
        )

        z_kgw = recursive_find_number(
            source,
            KGW_KEYS,
        )

        if (
            z_fisher is None
            or z_kgw is None
        ):
            skipped += 1
            continue

        fisher.append(z_fisher)
        kgw.append(z_kgw)

    fisher = np.asarray(
        fisher,
        dtype=float,
    )

    kgw = np.asarray(
        kgw,
        dtype=float,
    )

    return fisher, kgw, skipped


# ============================================================
# Find Sec-Cal score file
# ============================================================

def auto_find_sec_cal_file():

    candidates = []

    for pattern in [
        "**/*.jsonl",
        "**/*.json",
    ]:

        for path in SEC_CAL_DIR.glob(
            pattern
        ):

            # gate_threshold.json only contains
            # threshold metadata, not sample scores
            if path.name == "gate_threshold.json":
                continue

            try:
                rows = load_records(path)

                fisher, kgw, _ = extract_scores(
                    rows,
                    use_no_wm_block=True,
                )

            except Exception:
                continue

            n = len(fisher)

            if n < 1000:
                continue

            name = str(path).lower()

            bonus = 0

            if (
                "no_wm" in name
                or "nowm" in name
            ):
                bonus += 100

            if (
                "sec" in name
                or "cal" in name
            ):
                bonus += 20

            # Strongly prefer ~5000 samples
            score = (
                bonus
                - abs(n - 5000) / 100
            )

            candidates.append(
                (
                    score,
                    n,
                    path,
                )
            )

    if not candidates:

        raise FileNotFoundError(
            "\nCould not automatically find "
            "the Sec-Cal score file under:\n"
            f"{SEC_CAL_DIR}\n\n"
            "Please run with:\n"
            "python "
            "analysis/security_validation/"
            "evaluate_fisher_only_vs_gate.py "
            "--sec-cal /path/to/sec_cal.jsonl"
        )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    print(
        "\nSec-Cal candidates:"
    )

    for score, n, path in candidates[:5]:

        print(
            f"  N={n:5d}  "
            f"{path}"
        )

    selected = candidates[0][2]

    print(
        "\nSelected Sec-Cal:"
    )

    print(
        f"  {selected}"
    )

    return selected


# ============================================================
# Fisher-only calibration
# ============================================================

def calibrate_fisher_only(
    scores,
    target_fpr=0.01,
):

    if len(scores) == 0:
        raise ValueError(
            "No Fisher calibration scores."
        )

    # Empirical upper-tail 99th percentile.
    # For N=5000 and continuous scores this normally
    # produces 50 accepted samples.
    try:

        threshold = np.quantile(
            scores,
            1.0 - target_fpr,
            method="higher",
        )

    except TypeError:

        # Compatibility with older NumPy
        threshold = np.quantile(
            scores,
            1.0 - target_fpr,
            interpolation="higher",
        )

    threshold = float(
        threshold
    )

    accepted = (
        scores >= threshold
    )

    empirical_fpr = float(
        np.mean(accepted)
    )

    return (
        threshold,
        empirical_fpr,
        int(np.sum(accepted)),
    )


# ============================================================
# Spoof evaluation
# ============================================================
def evaluate_spoof(
    name,
    path,
    fisher_only_threshold,
):

    rows = load_records(path)

    fisher = []
    kgw = []

    skipped = 0

    for i, row in enumerate(rows):

        # ====================================================
        # IMPORTANT:
        # Spoof samples must use detection_wm explicitly.
        # Do NOT recursively search the whole JSON record.
        # ====================================================

        detection = row.get(
            "detection_wm"
        )

        if not isinstance(
            detection,
            dict,
        ):
            skipped += 1
            continue

        # KGW / token-level statistic
        z_kgw = detection.get(
            "z_score_micro"
        )

        # Fisher fusion statistic
        z_fisher = detection.get(
            "z_score_fisher"
        )

        if (
            z_kgw is None
            or z_fisher is None
        ):
            skipped += 1
            continue

        try:

            z_kgw = float(
                z_kgw
            )

            z_fisher = float(
                z_fisher
            )

        except (
            TypeError,
            ValueError,
        ):

            skipped += 1
            continue

        if not (
            np.isfinite(z_kgw)
            and np.isfinite(z_fisher)
        ):

            skipped += 1
            continue

        kgw.append(
            z_kgw
        )

        fisher.append(
            z_fisher
        )

    fisher = np.asarray(
        fisher,
        dtype=float,
    )

    kgw = np.asarray(
        kgw,
        dtype=float,
    )

    if len(fisher) == 0:

        raise RuntimeError(
            f"No valid detection_wm scores found in:\n"
            f"{path}"
        )

    # ========================================================
    # 1. Independently calibrated Fisher-only detector
    # ========================================================

    fisher_only = (
        fisher
        >= fisher_only_threshold
    )

    # ========================================================
    # 2. Frozen KGW gate
    # ========================================================

    kgw_pass = (
        kgw
        >= KGW_GATE
    )

    # ========================================================
    # 3. Frozen final security detector
    # ========================================================

    gated_final = (
        kgw_pass
        & (
            fisher
            >= GATED_FISHER_THRESHOLD
        )
    )

    return {

        "condition":
            name,

        "n":
            int(
                len(fisher)
            ),

        "skipped":
            int(
                skipped
            ),

        "mean_z_kgw":
            float(
                np.mean(kgw)
            ),

        "mean_z_fisher":
            float(
                np.mean(fisher)
            ),

        "fisher_only_count":
            int(
                np.sum(
                    fisher_only
                )
            ),

        "fisher_only_rate":
            float(
                np.mean(
                    fisher_only
                )
            ),

        "kgw_gate_count":
            int(
                np.sum(
                    kgw_pass
                )
            ),

        "kgw_gate_rate":
            float(
                np.mean(
                    kgw_pass
                )
            ),

        "gated_final_count":
            int(
                np.sum(
                    gated_final
                )
            ),

        "gated_final_rate":
            float(
                np.mean(
                    gated_final
                )
            ),

        "file":
            str(
                path
            ),
    }

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sec-cal",
        type=Path,
        default=None,
        help=(
            "5000-sample independent "
            "no_wm Sec-Cal score file."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Locate Sec-Cal
    # --------------------------------------------------------

    sec_cal_path = (
        args.sec_cal
        if args.sec_cal is not None
        else auto_find_sec_cal_file()
    )

    sec_rows = load_records(
        sec_cal_path
    )

    sec_fisher, sec_kgw, skipped = (
        extract_scores(
            sec_rows,
            use_no_wm_block=True,
        )
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "SEC-CAL CHECK"
    )

    print(
        "=" * 88
    )

    print(
        f"Sec-Cal file : {sec_cal_path}"
    )

    print(
        f"Valid N      : {len(sec_fisher)}"
    )

    print(
        f"Skipped      : {skipped}"
    )

    if len(sec_fisher) < 4000:

        print(
            "\n[WARN] Expected approximately "
            "5000 Sec-Cal samples."
        )

    # --------------------------------------------------------
    # Fisher-only threshold
    # --------------------------------------------------------

    (
        fisher_only_threshold,
        fisher_only_fpr,
        fisher_only_count,

    ) = calibrate_fisher_only(
        sec_fisher,
        TARGET_FPR,
    )

    # --------------------------------------------------------
    # Verify frozen gated detector
    # --------------------------------------------------------

    sec_gate = (
        sec_kgw >= KGW_GATE
    )

    sec_gated_final = (
        sec_gate
        & (
            sec_fisher
            >= GATED_FISHER_THRESHOLD
        )
    )

    print(
        "\n[Fisher-only @ 1% FPR]"
    )

    print(
        f"Threshold     : "
        f"{fisher_only_threshold:.6f}"
    )

    print(
        f"Empirical FPR : "
        f"{100*fisher_only_fpr:.2f}% "
        f"({fisher_only_count}/"
        f"{len(sec_fisher)})"
    )

    print(
        "\n[Frozen KGW-gated detector]"
    )

    print(
        f"KGW gate      : "
        f"{KGW_GATE:.6f}"
    )

    print(
        f"Fisher tau    : "
        f"{GATED_FISHER_THRESHOLD:.6f}"
    )

    print(
        f"Gate pass     : "
        f"{100*np.mean(sec_gate):.2f}% "
        f"({np.sum(sec_gate)}/"
        f"{len(sec_gate)})"
    )

    print(
        f"Final FPR     : "
        f"{100*np.mean(sec_gated_final):.2f}% "
        f"({np.sum(sec_gated_final)}/"
        f"{len(sec_gated_final)})"
    )

    # --------------------------------------------------------
    # Spoof evaluation
    # --------------------------------------------------------

    results = []

    for name, path in SPOOF_FILES.items():

        result = evaluate_spoof(
            name,
            path,
            fisher_only_threshold,
        )

        results.append(result)

    print(
        "\n"
        + "=" * 88
    )

    print(
        "SPOOFING COMPARISON"
    )

    print(
        "=" * 88
    )

    print(
        f"{'Condition':<24}"
        f"{'N':>6}"
        f"{'zKGW':>10}"
        f"{'zFisher':>11}"
        f"{'Fisher-only':>15}"
        f"{'KGW-pass':>12}"
        f"{'Gated':>12}"
    )

    print(
        "-" * 88
    )

    for r in results:

        print(
            f"{r['condition']:<24}"
            f"{r['n']:>6d}"
            f"{r['mean_z_kgw']:>10.3f}"
            f"{r['mean_z_fisher']:>11.3f}"
            f"{100*r['fisher_only_rate']:>14.2f}%"
            f"{100*r['kgw_gate_rate']:>11.2f}%"
            f"{100*r['gated_final_rate']:>11.2f}%"
        )

    print(
        "-" * 88
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_DIR
        / "fisher_only_vs_gated.json"
    )

    csv_path = (
        OUTPUT_DIR
        / "fisher_only_vs_gated.csv"
    )

    output = {
        "sec_calibration": {
            "file":
                str(sec_cal_path),

            "n":
                int(len(sec_fisher)),

            "target_fpr":
                TARGET_FPR,

            "fisher_only_threshold":
                fisher_only_threshold,

            "fisher_only_fpr":
                fisher_only_fpr,

            "kgw_gate":
                KGW_GATE,

            "gated_fisher_threshold":
                GATED_FISHER_THRESHOLD,

            "gated_sec_cal_fpr":
                float(
                    np.mean(
                        sec_gated_final
                    )
                ),
        },

        "spoofing":
            results,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = [
        "condition",
        "n",
        "skipped",
        "mean_z_kgw",
        "mean_z_fisher",
        "fisher_only_count",
        "fisher_only_rate",
        "kgw_gate_count",
        "kgw_gate_rate",
        "gated_final_count",
        "gated_final_rate",
        "file",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        "\nSaved:"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {csv_path}"
    )


if __name__ == "__main__":
    main()