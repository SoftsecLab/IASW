#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correct standalone KGW detector.

CRITICAL:
The project's KGW greenlist RNG is device-dependent:
    WatermarkLogitsProcessor -> torch.Generator(device=input_ids.device)
    WatermarkDetector        -> torch.Generator(device=self.device)

The watermark was generated on CUDA, so detection MUST also use CUDA.
Using CPU produces a different randperm for the same seed and therefore
measures essentially random greenlists (~0.5 green fraction).

This detector:
  - bypasses runner/plugins/SemStamp
  - uses the project KGW WatermarkDetector directly
  - defaults to cuda:0
  - reports both:
      raw / count-all-bigrams
      repeated-bigram-aware (formal current project detector)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


_ENV_PROJECT_DIR = os.environ.get("PROJECT_DIR")
if _ENV_PROJECT_DIR:
    PROJECT_DIR = Path(_ENV_PROJECT_DIR).resolve()
else:
    _THIS = Path(__file__).resolve()
    PROJECT_DIR = next(
        (p for p in _THIS.parents if (p / "src").is_dir()),
        None,
    )
    if PROJECT_DIR is None:
        raise RuntimeError(
            "Cannot locate project root containing src/. "
            "Set PROJECT_DIR explicitly."
        )

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.baselines.kgw_core.watermark_processor import WatermarkDetector


TEXT_FIELDS = [
    "wm_text",
    "watermarked_text",
    "generated_text",
    "output_text",
    "completion",
    "text",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--model_path", required=True)

    p.add_argument("--output_csv", default="")
    p.add_argument("--summary_json", default="")

    p.add_argument("--text_field", default="")

    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--z_threshold", type=float, default=4.0)

    # IMPORTANT: must match the device family used during KGW generation.
    p.add_argument("--device", default="cuda:0")

    p.add_argument(
        "--sanity_green_threshold",
        type=float,
        default=0.55,
        help="Warn when average raw green fraction is below this value.",
    )

    return p.parse_args()


def read_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def choose_text_field(rows, explicit):

    if explicit:
        if any(explicit in r for r in rows):
            return explicit

        raise KeyError(
            f"Requested text field {explicit!r} not found."
        )

    for field in TEXT_FIELDS:

        vals = [
            r.get(field)
            for r in rows[:10]
        ]

        if any(
            isinstance(x, str) and x.strip()
            for x in vals
        ):
            return field

    keys = sorted(
        set().union(
            *(r.keys() for r in rows[:10])
        )
    )

    raise KeyError(
        "Could not identify generated-text field. "
        f"Available keys: {keys}"
    )


def make_detector(
    tokenizer,
    gamma,
    z_threshold,
    device,
    ignore_repeats,
):

    vocab = list(
        tokenizer.get_vocab().values()
    )

    return WatermarkDetector(
        vocab=vocab,
        gamma=gamma,
        seeding_scheme="simple_1",
        device=device,
        tokenizer=tokenizer,
        z_threshold=z_threshold,
        normalizers=[],
        ignore_repeated_bigrams=ignore_repeats,
        select_green_tokens=True,
    )


def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def summarize(vals):

    vals = [
        float(x)
        for x in vals
        if x is not None
        and math.isfinite(float(x))
    ]

    if not vals:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
        }

    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "std": (
            float(np.std(vals, ddof=1))
            if len(vals) > 1
            else 0.0
        ),
        "median": float(np.median(vals)),
    }


def detect_one(detector, text):

    result = detector.detect(
        text=text,
        return_prediction=False,
        return_scores=True,
    )

    return {} if result is None else dict(result)


def main():

    args = parse_args()

    if not args.device.startswith("cuda"):
        raise RuntimeError(
            "This experiment's KGW text was generated on CUDA. "
            "Do not detect it with CPU: CPU and CUDA torch.Generator "
            "produce different randperm greenlists. "
            "Use --device cuda:0."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but CUDA detection is required "
            "to reproduce the greenlists used during generation."
        )

    inp = Path(args.input)

    rows = read_jsonl(inp)

    if not rows:
        raise RuntimeError(
            f"No records in {inp}"
        )

    field = choose_text_field(
        rows,
        args.text_field,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    # Diagnostic: count all bigrams/tokens.
    detector_raw = make_detector(
        tokenizer,
        args.gamma,
        args.z_threshold,
        args.device,
        ignore_repeats=False,
    )

    # Formal current-project variant: repeated bigrams count once.
    detector_repeat = make_detector(
        tokenizer,
        args.gamma,
        args.z_threshold,
        args.device,
        ignore_repeats=True,
    )

    outputs = []

    for i, row in enumerate(rows):

        text = row.get(field, "")

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            continue

        raw = detect_one(
            detector_raw,
            text,
        )

        rep = detect_one(
            detector_repeat,
            text,
        )

        outputs.append({
            "sample_index": i,

            "source_id": row.get(
                "idx",
                row.get("id", i),
            ),

            "text_field": field,

            "z_no_repeat_filter":
                fnum(raw.get("z_score")),

            "green_fraction_no_repeat_filter":
                fnum(raw.get("green_fraction")),

            "num_tokens_no_repeat_filter":
                fnum(raw.get("num_tokens_scored")),

            "num_green_no_repeat_filter":
                fnum(raw.get("num_green_tokens")),

            "z_repeat_filter":
                fnum(rep.get("z_score")),

            "green_fraction_repeat_filter":
                fnum(rep.get("green_fraction")),

            "num_tokens_repeat_filter":
                fnum(rep.get("num_tokens_scored")),

            "num_green_repeat_filter":
                fnum(rep.get("num_green_tokens")),
        })

    if not outputs:
        raise RuntimeError(
            "No non-empty generated text found."
        )

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else inp.with_suffix(
            ".kgw_detection.csv"
        )
    )

    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else inp.with_suffix(
            ".kgw_detection_summary.json"
        )
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                outputs[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(outputs)

    summary = {
        "input": str(inp),
        "model_path": args.model_path,
        "detector_device": args.device,
        "text_field": field,

        "num_input_records":
            len(rows),

        "num_detected_records":
            len(outputs),

        "z_no_repeat_filter":
            summarize([
                r["z_no_repeat_filter"]
                for r in outputs
            ]),

        "z_repeat_filter":
            summarize([
                r["z_repeat_filter"]
                for r in outputs
            ]),

        "green_fraction_no_repeat_filter":
            summarize([
                r["green_fraction_no_repeat_filter"]
                for r in outputs
            ]),

        "green_fraction_repeat_filter":
            summarize([
                r["green_fraction_repeat_filter"]
                for r in outputs
            ]),

        "num_tokens_no_repeat_filter":
            summarize([
                r["num_tokens_no_repeat_filter"]
                for r in outputs
            ]),

        "num_tokens_repeat_filter":
            summarize([
                r["num_tokens_repeat_filter"]
                for r in outputs
            ]),
    }

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    gf_raw = summary[
        "green_fraction_no_repeat_filter"
    ]["mean"]

    gf_rep = summary[
        "green_fraction_repeat_filter"
    ]["mean"]

    z_raw = summary[
        "z_no_repeat_filter"
    ]["mean"]

    z_rep = summary[
        "z_repeat_filter"
    ]["mean"]

    print()
    print(
        "============================================================"
    )
    print(
        "Correct CUDA KGW detection"
    )
    print(
        "============================================================"
    )
    print("input        =", inp)
    print("model        =", args.model_path)
    print("device       =", args.device)
    print("records      =", len(outputs))
    print("mean GF raw  =", gf_raw)
    print("mean z raw   =", z_raw)
    print("mean GF rep  =", gf_rep)
    print("mean z rep   =", z_rep)
    print("csv          =", output_csv)
    print("summary      =", summary_json)

    if (
        gf_raw is not None
        and gf_raw
        < args.sanity_green_threshold
    ):
        print()
        print(
            "[WARNING] Mean raw green fraction is unexpectedly low."
        )
        print(
            "For KGW-watermarked text with gamma=0.5/delta=2, "
            "a value near 0.5 usually indicates a detector/generation "
            "configuration mismatch."
        )


if __name__ == "__main__":
    main()
