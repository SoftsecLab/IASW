
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an independent security-calibration set.

Pipeline:
    raw_c4.jsonl
        -> skip first N lines
        -> build_baseline_dataset.py
        -> take no_wm_text
        -> construct runner-compatible security calibration JSONL

This script does NOT modify IAS-W whitening/calibration statistics.
"""

import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_file", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--dataset_type", default="c4")
    parser.add_argument("--model_path", required=True)

    parser.add_argument("--target_len", type=int, default=200)
    parser.add_argument("--skip_lines", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild files even if they already exist."
    )

    return parser.parse_args()


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_temp_raw(raw_file, temp_file, skip_lines):
    print(f"[1/3] Building temporary raw file")
    print(f"      source     : {raw_file}")
    print(f"      skip lines : {skip_lines}")

    n_written = 0

    with open(raw_file, "r", encoding="utf-8") as fin, \
         open(temp_file, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if idx < skip_lines:
                continue

            fout.write(line)
            n_written += 1

    print(f"      remaining rows: {n_written}")

    if n_written == 0:
        raise RuntimeError(
            f"No rows remain after skip_lines={skip_lines}"
        )


def build_baseline(args, temp_file, baseline_file):
    print("\n[2/3] Building independent Sec-Cal baseline")

    cmd = [
        sys.executable,
        "src/utils/build_baseline_dataset.py",

        "--input_file", temp_file,
        "--output_file", baseline_file,

        "--dataset_type", args.dataset_type,
        "--model_path", args.model_path,

        "--target_len", str(args.target_len),
        "--num_samples", str(args.num_samples),
        "--batch_size", str(args.batch_size),
    ]

    print("      command:")
    print("      " + " ".join(cmd))

    subprocess.run(cmd, check=True)

    n = count_lines(baseline_file)

    print(f"      baseline rows: {n}")

    if n < args.num_samples:
        print(
            f"[WARN] Requested {args.num_samples} samples "
            f"but baseline contains {n} rows."
        )


def convert_to_detection_file(baseline_file, detection_file):
    print("\n[3/3] Building runner-compatible detection file")

    total = 0
    skipped = 0

    with open(baseline_file, "r", encoding="utf-8") as fin, \
         open(detection_file, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            text = row.get("no_wm_text", "")

            if not isinstance(text, str) or not text.strip():
                skipped += 1
                continue

            text = text.strip()

            #
            # Runner baseline_eval normally detects:
            #
            #   human_text
            #   no_wm_text
            #   wm_text
            #
            # For Sec-Cal we intentionally put the SAME null text
            # into all three fields.
            #
            # Later threshold calibration reads ONLY detection_wm.
            #
            out = {
                "idx": idx,

                "human_text": text,
                "no_wm_text": text,
                "wm_text": text,

                "security_calibration": True,
                "security_calibration_source": "no_wm_text",
            }

            fout.write(
                json.dumps(out, ensure_ascii=False) + "\n"
            )

            total += 1

    print(f"      valid rows  : {total}")
    print(f"      skipped rows: {skipped}")

    if total == 0:
        raise RuntimeError(
            "No valid no_wm_text found in baseline."
        )


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    temp_file = os.path.join(
        args.output_dir,
        f"temp_raw_skip{args.skip_lines}.jsonl"
    )

    baseline_file = os.path.join(
        args.output_dir,
        (
            f"sec_baseline_len{args.target_len}"
            f"_skip{args.skip_lines}"
            f"_n{args.num_samples}.jsonl"
        )
    )

    detection_file = os.path.join(
        args.output_dir,
        (
            f"sec_cal_len{args.target_len}"
            f"_skip{args.skip_lines}"
            f"_n{args.num_samples}.jsonl"
        )
    )

    print("=" * 80)
    print("Independent Security Calibration Set")
    print("=" * 80)
    print(f"dataset       : {args.dataset_type}")
    print(f"model         : {args.model_path}")
    print(f"target length : {args.target_len}")
    print(f"skip lines    : {args.skip_lines}")
    print(f"num samples   : {args.num_samples}")
    print("=" * 80)

    # ---------------------------------------------------------
    # Step 1: independent raw slice
    # ---------------------------------------------------------

    if args.force or not os.path.exists(temp_file):
        build_temp_raw(
            args.raw_file,
            temp_file,
            args.skip_lines
        )
    else:
        print(f"[1/3] Reusing: {temp_file}")

    # ---------------------------------------------------------
    # Step 2: ordinary no-watermark model generations
    # ---------------------------------------------------------

    if args.force or not os.path.exists(baseline_file):
        build_baseline(
            args,
            temp_file,
            baseline_file
        )
    else:
        print(f"[2/3] Reusing: {baseline_file}")

    # ---------------------------------------------------------
    # Step 3: runner-compatible file
    # ---------------------------------------------------------

    if args.force or not os.path.exists(detection_file):
        convert_to_detection_file(
            baseline_file,
            detection_file
        )
    else:
        print(f"[3/3] Reusing: {detection_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"Baseline:")
    print(f"  {baseline_file}")

    print(f"\nDetection input:")
    print(f"  {detection_file}")


if __name__ == "__main__":
    main()