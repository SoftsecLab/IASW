#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated IASW analysis entry.

Generated from the current analysis scripts without changing their scientific
calculations. Each subcommand executes the original implementation in an
isolated function scope so duplicate helper names do not collide.
"""
from __future__ import annotations

import sys

# Canonical layout:
#   audit_kgw_sentence_boundary.py
#   analyze_kgw_sentence_boundary.py

def _run_summary(argv):
    """Run preserved implementation from analyze_kgw_sentence_boundary.py[summary]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_sentence_boundary.py[summary]"] + list(argv)
        """
        Summarize KGW sentence-boundary audit results across models.

        Input layout:
          ROOT/<model>/kgw_matched/
              loop_records.jsonl
              token_records.jsonl

        Outputs:
          retained_vs_tail_summary.csv
          boundary_profile.csv
          sentence_window_summary.csv
          cutoff_quality.csv

        Statistical unit for confidence intervals is SAMPLE, not token/loop, to avoid
        pseudo-replication.
        """


        import argparse
        import csv
        import json
        import math
        from collections import defaultdict
        from pathlib import Path
        from typing import Dict, List, Any, Tuple

        import numpy as np


        METRICS = [
            "actual_is_green",
            "entropy_raw",
            "entropy_effective_pre",
            "top1_prob_effective_pre",
            "green_mass_raw",
            "green_mass_effective_pre",
            "green_mass_effective_wm",
            "green_mass_gain_effective",
            "top1_is_green_effective_pre",
            "top1_is_green_effective_wm",
        ]


        def args_parse():
            p = argparse.ArgumentParser()
            p.add_argument("--root", required=True)
            p.add_argument("--output_dir", required=True)
            p.add_argument("--bootstrap", type=int, default=10000)
            p.add_argument("--seed", type=int, default=20260825)
            p.add_argument("--min_tail_tokens", type=int, default=3)
            p.add_argument("--mid_exclude", type=int, default=5)
            return p.parse_args()


        def read_jsonl(path: Path):
            out = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.append(json.loads(line))
            return out


        def finite(vals):
            out = []
            for x in vals:
                try:
                    y = float(x)
                except Exception:
                    continue
                if math.isfinite(y):
                    out.append(y)
            return out


        def mean(vals):
            a = finite(vals)
            return float(np.mean(a)) if a else float("nan")


        def bootstrap_mean(vals, n_boot, seed):
            a = np.asarray(finite(vals), dtype=float)
            if a.size == 0:
                return float("nan"), float("nan"), float("nan")
            rng = np.random.default_rng(seed)
            b = np.empty(n_boot, dtype=float)
            n = len(a)
            for i in range(n_boot):
                b[i] = np.mean(a[rng.integers(0, n, size=n)])
            return (
                float(np.mean(a)),
                float(np.quantile(b, 0.025)),
                float(np.quantile(b, 0.975)),
            )


        def write_csv(path: Path, rows: List[Dict[str, Any]]):
            if not rows:
                return
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)


        def sample_metric(tokens, key):
            return mean([r.get(key) for r in tokens])


        def main():
            args = args_parse()
            root = Path(args.root)
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            token_files = sorted(root.glob("*/kgw_matched/token_records.jsonl"))
            loop_files = sorted(root.glob("*/kgw_matched/loop_records.jsonl"))

            if not token_files:
                raise SystemExit(f"No token_records.jsonl found under {root}/*/kgw_matched/")

            # ------------------------------------------------------------------
            # A. Retained vs discarded tail, sample-level paired bootstrap
            # ------------------------------------------------------------------
            retained_tail_rows = []

            for fidx, path in enumerate(token_files):
                rows = read_jsonl(path)
                model = rows[0]["model"]

                sent = [r for r in rows if r["mode"] == "sentence_ids"]

                by_sample = defaultdict(list)
                for r in sent:
                    by_sample[int(r["sample_index"])].append(r)

                for midx, metric in enumerate(METRICS):
                    sample_diffs = []
                    sample_ret = []
                    sample_tail = []

                    for sid, rr in by_sample.items():
                        # Only tails from loops with a meaningful discarded tail.
                        # Find eligible loop IDs using actual tail token counts.
                        loop_groups = defaultdict(list)
                        for x in rr:
                            loop_groups[int(x["loop_index"])].append(x)

                        eligible_loops = []
                        for li, xs in loop_groups.items():
                            tail_n = sum(1 for x in xs if x["segment"] == "tail")
                            if tail_n >= args.min_tail_tokens:
                                eligible_loops.append(li)

                        ret_tokens = [
                            x for x in rr
                            if x["segment"] == "retained"
                            and int(x["loop_index"]) in eligible_loops
                        ]
                        tail_tokens = [
                            x for x in rr
                            if x["segment"] == "tail"
                            and int(x["loop_index"]) in eligible_loops
                        ]

                        a = sample_metric(ret_tokens, metric)
                        b = sample_metric(tail_tokens, metric)
                        if math.isfinite(a) and math.isfinite(b):
                            sample_ret.append(a)
                            sample_tail.append(b)
                            sample_diffs.append(a - b)

                    d, lo, hi = bootstrap_mean(
                        sample_diffs,
                        args.bootstrap,
                        args.seed + fidx * 1000 + midx,
                    )
                    retained_tail_rows.append({
                        "model": model,
                        "metric": metric,
                        "n_samples": len(sample_diffs),
                        "retained_mean_across_samples": mean(sample_ret),
                        "tail_mean_across_samples": mean(sample_tail),
                        "retained_minus_tail": d,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    })

            write_csv(out_dir / "retained_vs_tail_summary.csv", retained_tail_rows)

            # ------------------------------------------------------------------
            # B. Boundary profile for actual retained sentence tokens
            #    Negative = before sentence end boundary.
            #    Positive = after previous boundary / sentence start.
            # ------------------------------------------------------------------
            boundary_rows = []

            for path in token_files:
                rows = read_jsonl(path)
                model = rows[0]["model"]
                sent = [
                    r for r in rows
                    if r["mode"] == "sentence_ids" and r["segment"] == "retained"
                ]

                # End side (-5 ... -1)
                for side_key, side_name in [
                    ("boundary_rel_end", "end"),
                    ("boundary_rel_start", "start"),
                ]:
                    rel_values = sorted({
                        int(r[side_key])
                        for r in sent
                        if r.get(side_key) is not None
                    })

                    for rel in rel_values:
                        sub = [r for r in sent if r.get(side_key) == rel]
                        sids = len(set(int(r["sample_index"]) for r in sub))
                        for metric in METRICS:
                            boundary_rows.append({
                                "model": model,
                                "side": side_name,
                                "relative_position": rel,
                                "metric": metric,
                                "n_tokens": len(finite([r.get(metric) for r in sub])),
                                "n_samples": sids,
                                "mean": mean([r.get(metric) for r in sub]),
                            })

            write_csv(out_dir / "boundary_profile.csv", boundary_rows)

            # ------------------------------------------------------------------
            # C. Sentence windows: START(1..5), END(last 5), MID(excluding both)
            #    Compute per-sample means, then bootstrap contrasts.
            # ------------------------------------------------------------------
            window_rows = []

            for fidx, path in enumerate(token_files):
                rows = read_jsonl(path)
                model = rows[0]["model"]
                sent = [
                    r for r in rows
                    if r["mode"] == "sentence_ids" and r["segment"] == "retained"
                ]
                by_sample = defaultdict(list)
                for r in sent:
                    by_sample[int(r["sample_index"])].append(r)

                for midx, metric in enumerate(METRICS):
                    samples = []
                    for sid, rr in by_sample.items():
                        start = [
                            r for r in rr
                            if r.get("sent_pos_from_start") is not None
                            and int(r["sent_pos_from_start"]) <= args.mid_exclude
                        ]
                        end = [
                            r for r in rr
                            if r.get("sent_pos_to_end") is not None
                            and int(r["sent_pos_to_end"]) <= args.mid_exclude
                        ]
                        mid = [
                            r for r in rr
                            if r.get("sent_pos_from_start") is not None
                            and r.get("sent_pos_to_end") is not None
                            and int(r["sent_pos_from_start"]) > args.mid_exclude
                            and int(r["sent_pos_to_end"]) > args.mid_exclude
                        ]

                        s = sample_metric(start, metric)
                        e = sample_metric(end, metric)
                        m = sample_metric(mid, metric)
                        samples.append((s, e, m))

                    for contrast_name, idx_a, idx_b in [
                        ("start_minus_mid", 0, 2),
                        ("end_minus_mid", 1, 2),
                        ("start_minus_end", 0, 1),
                    ]:
                        vals = []
                        for tup in samples:
                            a, b = tup[idx_a], tup[idx_b]
                            if math.isfinite(a) and math.isfinite(b):
                                vals.append(a - b)

                        d, lo, hi = bootstrap_mean(
                            vals,
                            args.bootstrap,
                            args.seed + 100000 + fidx * 1000 + midx * 10 + idx_a,
                        )
                        window_rows.append({
                            "model": model,
                            "metric": metric,
                            "contrast": contrast_name,
                            "n_samples": len(vals),
                            "mean_difference": d,
                            "ci95_low": lo,
                            "ci95_high": hi,
                        })

            write_csv(out_dir / "sentence_window_summary.csv", window_rows)

            # ------------------------------------------------------------------
            # D. Cutoff mapping quality
            # ------------------------------------------------------------------
            cutoff_rows = []
            for path in loop_files:
                rows = read_jsonl(path)
                if not rows:
                    continue
                model = rows[0]["model"]
                sent = [r for r in rows if r["mode"] == "sentence_ids"]
                cutoff_rows.append({
                    "model": model,
                    "n_sentence_loops": len(sent),
                    "cutoff_found_rate": mean([
                        r.get("cutoff_found_from_actual_ids") for r in sent
                    ]),
                    "mean_retained_tokens": mean([r.get("retained_n") for r in sent]),
                    "mean_tail_tokens": mean([r.get("tail_n") for r in sent]),
                    "tail_nonempty_rate": mean([
                        int(int(r.get("tail_n", 0)) > 0) for r in sent
                    ]),
                    "tail_ge_min_rate": mean([
                        int(int(r.get("tail_n", 0)) >= args.min_tail_tokens) for r in sent
                    ]),
                })

            write_csv(out_dir / "cutoff_quality.csv", cutoff_rows)

            print("[DONE]")
            for name in [
                "retained_vs_tail_summary.csv",
                "boundary_profile.csv",
                "sentence_window_summary.csv",
                "cutoff_quality.csv",
            ]:
                print(" ", out_dir / name)
        main()
    finally:
        sys.argv = old_argv

def _run_loop(argv):
    """Run preserved implementation from analyze_kgw_sentence_boundary.py[loop]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_sentence_boundary.py[loop]"] + list(argv)

        """
        Offline loop-by-loop KGW trajectory analysis.

        This script uses the EXISTING outputs from:
            audit_kgw_sentence_boundary.py

        Expected layout:
            ROOT/
              Qwen2.5-3B-Instruct/kgw_matched/token_records.jsonl
              Qwen2.5-7B-Instruct/kgw_matched/token_records.jsonl
              Llama-3.2-3B-Instruct/kgw_matched/token_records.jsonl

        No model loading and no GPU are required.

        Main questions
        --------------
        1. At which outer loop does sentence-wise generation begin to diverge from
           fixed-chunk generation?
        2. Does the divergence accumulate with loop number?
        3. Is the divergence caused by:
              - actual sampled green fraction,
              - pre-watermark green probability mass,
              - KGW steering gain,
              - entropy / top-1 concentration,
              - or effective watermarked green mass?

        Outputs
        -------
        loop_profile.csv
            Absolute per-loop profile for each model and mode.

        loop_contrasts.csv
            Paired per-loop contrast:
                sentence_ids - chunk_ids
            Bootstrap CI uses SAMPLE as the statistical unit.

        cumulative_contrasts.csv
            Paired cumulative contrast through loop j.

        first_token_contrasts.csv
            Same comparison using only the first generated token of each loop.
            Helps distinguish an immediate restart effect from a later within-loop drift.

        early_late_contrasts.csv
            Compare first N retained/raw tokens vs later tokens inside each loop.

        model_difference_in_differences.csv
            Difference-in-differences between Qwen2.5-7B and the control models for
            sentence_ids - chunk_ids.

        trajectory_onset.csv
            Descriptive first loop at which the 95% CI of sentence-chunk excludes zero.

        Notes
        -----
        - Statistical unit = sample.
        - Token-level values are first aggregated within each sample and loop.
        - Cumulative metrics are token-weighted within each sample before bootstrapping.
        - Loop 1 should be especially informative: both modes start from the same prompt.
        """


        import argparse
        import csv
        import json
        import math
        from collections import defaultdict
        from pathlib import Path
        from typing import Any, Dict, Iterable, List, Tuple

        import numpy as np


        DEFAULT_METRICS = [
            "actual_is_green",
            "entropy_raw",
            "entropy_effective_pre",
            "top1_prob_effective_pre",
            "green_mass_raw",
            "green_mass_effective_pre",
            "green_mass_effective_wm",
            "green_mass_gain_effective",
            "top1_is_green_effective_pre",
            "top1_is_green_effective_wm",
        ]


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--root", required=True)
            p.add_argument("--output_dir", required=True)
            p.add_argument("--bootstrap", type=int, default=10000)
            p.add_argument("--seed", type=int, default=20260825)
            p.add_argument("--min_samples", type=int, default=10)
            p.add_argument("--early_tokens", type=int, default=5)
            p.add_argument(
                "--metrics",
                nargs="+",
                default=DEFAULT_METRICS,
            )
            return p.parse_args()


        def read_jsonl(path: Path) -> List[Dict[str, Any]]:
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
            return rows


        def finite_float(x):
            try:
                y = float(x)
            except Exception:
                return None
            return y if math.isfinite(y) else None


        def mean(vals: Iterable[Any]) -> float:
            a = [finite_float(x) for x in vals]
            a = [x for x in a if x is not None]
            return float(np.mean(a)) if a else float("nan")


        def weighted_mean(rows: List[Dict[str, Any]], metric: str) -> float:
            vals = []
            for r in rows:
                x = finite_float(r.get(metric))
                if x is not None:
                    vals.append(x)
            return float(np.mean(vals)) if vals else float("nan")


        def bootstrap_mean(vals: List[float], n_boot: int, seed: int):
            a = np.asarray([x for x in vals if math.isfinite(float(x))], dtype=float)
            if a.size == 0:
                return float("nan"), float("nan"), float("nan")
            rng = np.random.default_rng(seed)
            n = len(a)
            boot = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                boot[i] = np.mean(a[rng.integers(0, n, size=n)])
            return (
                float(np.mean(a)),
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            )


        def write_csv(path: Path, rows: List[Dict[str, Any]]):
            if not rows:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)


        def group_sample_loop(
            rows: List[Dict[str, Any]],
            metric: str,
            mode: str,
            token_selector=None,
        ):
            """
            Return:
              {(sample_index, loop_index): mean metric within selected tokens}
            """
            groups = defaultdict(list)
            for r in rows:
                if r.get("mode") != mode:
                    continue
                if token_selector is not None and not token_selector(r):
                    continue
                x = finite_float(r.get(metric))
                if x is None:
                    continue
                key = (int(r["sample_index"]), int(r["loop_index"]))
                groups[key].append(x)

            return {
                k: float(np.mean(v))
                for k, v in groups.items()
                if v
            }


        def group_sample_loop_token_rows(
            rows: List[Dict[str, Any]],
            mode: str,
        ):
            out = defaultdict(list)
            for r in rows:
                if r.get("mode") != mode:
                    continue
                out[(int(r["sample_index"]), int(r["loop_index"]))].append(r)

            for k in out:
                out[k].sort(key=lambda x: int(x.get("raw_pos", 0)))
            return out


        def paired_loop_contrast(
            rows,
            metric,
            n_boot,
            seed,
            min_samples,
            token_selector=None,
        ):
            chunk = group_sample_loop(
                rows, metric, "chunk_ids", token_selector=token_selector
            )
            sent = group_sample_loop(
                rows, metric, "sentence_ids", token_selector=token_selector
            )

            loops = sorted(
                set(loop for _, loop in chunk)
                | set(loop for _, loop in sent)
            )

            out = []
            for loop in loops:
                sample_ids = sorted(
                    set(s for s, l in chunk if l == loop)
                    & set(s for s, l in sent if l == loop)
                )

                diffs, cvals, svals = [], [], []
                for sid in sample_ids:
                    a = chunk.get((sid, loop))
                    b = sent.get((sid, loop))
                    if a is None or b is None:
                        continue
                    if not (math.isfinite(a) and math.isfinite(b)):
                        continue
                    cvals.append(a)
                    svals.append(b)
                    diffs.append(b - a)

                if len(diffs) < min_samples:
                    continue

                d, lo, hi = bootstrap_mean(
                    diffs, n_boot, seed + loop * 103
                )
                out.append({
                    "loop_index": loop,
                    "metric": metric,
                    "n_pairs": len(diffs),
                    "chunk_mean": float(np.mean(cvals)),
                    "sentence_mean": float(np.mean(svals)),
                    "sentence_minus_chunk": d,
                    "ci95_low": lo,
                    "ci95_high": hi,
                })
            return out


        def absolute_loop_profile(rows, metrics, n_boot, seed, min_samples):
            out = []
            for midx, metric in enumerate(metrics):
                for mode in ("chunk_ids", "sentence_ids"):
                    values = group_sample_loop(rows, metric, mode)
                    loops = sorted(set(loop for _, loop in values))

                    for loop in loops:
                        vals = [
                            v for (sid, li), v in values.items()
                            if li == loop and math.isfinite(v)
                        ]
                        if len(vals) < min_samples:
                            continue
                        m, lo, hi = bootstrap_mean(
                            vals,
                            n_boot,
                            seed + midx * 10000 + loop * 97 + (0 if mode == "chunk_ids" else 1),
                        )
                        out.append({
                            "metric": metric,
                            "mode": mode,
                            "loop_index": loop,
                            "n_samples": len(vals),
                            "mean": m,
                            "ci95_low": lo,
                            "ci95_high": hi,
                        })
            return out


        def cumulative_sample_values(rows, metric, mode):
            """
            For each sample and cumulative loop j, pool all metric-bearing tokens from
            loops <= j and compute a token-weighted cumulative mean.
            """
            by_sample_loop = group_sample_loop_token_rows(rows, mode)
            sample_ids = sorted(set(s for s, _ in by_sample_loop))
            max_loop = max((l for _, l in by_sample_loop), default=0)

            out = {}
            for sid in sample_ids:
                pooled = []
                for loop in range(1, max_loop + 1):
                    pooled.extend(by_sample_loop.get((sid, loop), []))
                    vals = [
                        finite_float(r.get(metric))
                        for r in pooled
                    ]
                    vals = [x for x in vals if x is not None]
                    if vals:
                        out[(sid, loop)] = float(np.mean(vals))
            return out


        def cumulative_contrasts(rows, metrics, n_boot, seed, min_samples):
            out = []
            for midx, metric in enumerate(metrics):
                c = cumulative_sample_values(rows, metric, "chunk_ids")
                s = cumulative_sample_values(rows, metric, "sentence_ids")
                loops = sorted(
                    set(l for _, l in c) | set(l for _, l in s)
                )

                for loop in loops:
                    sids = sorted(
                        set(i for i, l in c if l == loop)
                        & set(i for i, l in s if l == loop)
                    )
                    diffs, cv, sv = [], [], []
                    for sid in sids:
                        a = c.get((sid, loop))
                        b = s.get((sid, loop))
                        if a is None or b is None:
                            continue
                        if math.isfinite(a) and math.isfinite(b):
                            cv.append(a)
                            sv.append(b)
                            diffs.append(b - a)

                    if len(diffs) < min_samples:
                        continue

                    d, lo, hi = bootstrap_mean(
                        diffs,
                        n_boot,
                        seed + 500000 + midx * 10000 + loop * 101,
                    )
                    out.append({
                        "metric": metric,
                        "through_loop": loop,
                        "n_pairs": len(diffs),
                        "chunk_cumulative_mean": float(np.mean(cv)),
                        "sentence_cumulative_mean": float(np.mean(sv)),
                        "sentence_minus_chunk_cumulative": d,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    })
            return out


        def early_late_contrasts(rows, metrics, early_tokens, n_boot, seed, min_samples):
            """
            Within each loop:
              early = raw_pos <= early_tokens
              late  = raw_pos > early_tokens

            Then compare sentence_ids - chunk_ids separately for early and late regions.
            """
            out = []
            selectors = {
                f"first_{early_tokens}": lambda r: int(r.get("raw_pos", 0)) <= early_tokens,
                f"after_{early_tokens}": lambda r: int(r.get("raw_pos", 0)) > early_tokens,
            }

            for midx, metric in enumerate(metrics):
                for sidx, (region, selector) in enumerate(selectors.items()):
                    rows2 = paired_loop_contrast(
                        rows,
                        metric,
                        n_boot,
                        seed + 700000 + midx * 10000 + sidx * 1000,
                        min_samples,
                        token_selector=selector,
                    )
                    for r in rows2:
                        r["region"] = region
                        out.append(r)
            return out


        def first_token_contrasts(rows, metrics, n_boot, seed, min_samples):
            selector = lambda r: int(r.get("raw_pos", 0)) == 1
            out = []
            for midx, metric in enumerate(metrics):
                rr = paired_loop_contrast(
                    rows,
                    metric,
                    n_boot,
                    seed + 900000 + midx * 10000,
                    min_samples,
                    token_selector=selector,
                )
                out.extend(rr)
            return out


        def onset_from_loop_contrasts(loop_rows):
            """
            Descriptive onset:
            first loop where CI excludes zero.
            """
            by_metric = defaultdict(list)
            for r in loop_rows:
                by_metric[r["metric"]].append(r)

            out = []
            for metric, rr in sorted(by_metric.items()):
                rr = sorted(rr, key=lambda x: int(x["loop_index"]))
                first_neg = None
                first_pos = None
                for r in rr:
                    if float(r["ci95_high"]) < 0 and first_neg is None:
                        first_neg = int(r["loop_index"])
                    if float(r["ci95_low"]) > 0 and first_pos is None:
                        first_pos = int(r["loop_index"])

                out.append({
                    "metric": metric,
                    "first_significant_negative_loop": first_neg,
                    "first_significant_positive_loop": first_pos,
                })
            return out


        def model_did(
            all_loop_rows_by_model,
            metrics,
            n_boot,
            seed,
            min_samples,
            token_rows_by_model,
        ):
            """
            Direct sample-level difference-in-differences:
               (sentence - chunk)_modelA - (sentence - chunk)_modelB

            Samples use the same shared prompts and indices.
            """
            comparisons = [
                ("Qwen2.5-7B-Instruct", "Qwen2.5-3B-Instruct"),
                ("Qwen2.5-7B-Instruct", "Llama-3.2-3B-Instruct"),
                ("Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"),
            ]
            out = []

            for comp_idx, (a_model, b_model) in enumerate(comparisons):
                if a_model not in token_rows_by_model or b_model not in token_rows_by_model:
                    continue

                a_rows = token_rows_by_model[a_model]
                b_rows = token_rows_by_model[b_model]

                for midx, metric in enumerate(metrics):
                    a_c = group_sample_loop(a_rows, metric, "chunk_ids")
                    a_s = group_sample_loop(a_rows, metric, "sentence_ids")
                    b_c = group_sample_loop(b_rows, metric, "chunk_ids")
                    b_s = group_sample_loop(b_rows, metric, "sentence_ids")

                    loops = sorted(
                        set(l for _, l in a_c)
                        | set(l for _, l in a_s)
                        | set(l for _, l in b_c)
                        | set(l for _, l in b_s)
                    )

                    for loop in loops:
                        sids = sorted(
                            set(i for i, l in a_c if l == loop)
                            & set(i for i, l in a_s if l == loop)
                            & set(i for i, l in b_c if l == loop)
                            & set(i for i, l in b_s if l == loop)
                        )

                        vals = []
                        for sid in sids:
                            da = a_s[(sid, loop)] - a_c[(sid, loop)]
                            db = b_s[(sid, loop)] - b_c[(sid, loop)]
                            if all(math.isfinite(x) for x in (da, db)):
                                vals.append(da - db)

                        if len(vals) < min_samples:
                            continue

                        d, lo, hi = bootstrap_mean(
                            vals,
                            n_boot,
                            seed + 1200000 + comp_idx * 100000 + midx * 1000 + loop,
                        )
                        out.append({
                            "model_A": a_model,
                            "model_B": b_model,
                            "metric": metric,
                            "loop_index": loop,
                            "n_pairs": len(vals),
                            "DID_A_minus_B": d,
                            "ci95_low": lo,
                            "ci95_high": hi,
                        })
            return out


        def main():
            args = parse_args()
            root = Path(args.root)
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(root.glob("*/kgw_matched/token_records.jsonl"))
            if not files:
                raise SystemExit(
                    f"No token_records.jsonl found under {root}/*/kgw_matched/"
                )

            token_rows_by_model = {}
            loop_profile_all = []
            loop_contrast_all = []
            cumulative_all = []
            first_all = []
            early_late_all = []
            onset_all = []

            for fidx, path in enumerate(files):
                rows = read_jsonl(path)
                if not rows:
                    continue
                model = rows[0]["model"]
                token_rows_by_model[model] = rows
                print(f"[*] {model}: {len(rows)} token records")

                profile = absolute_loop_profile(
                    rows,
                    args.metrics,
                    args.bootstrap,
                    args.seed + fidx * 100000,
                    args.min_samples,
                )
                for r in profile:
                    r["model"] = model
                loop_profile_all.extend(profile)

                model_loop_rows = []
                for midx, metric in enumerate(args.metrics):
                    rr = paired_loop_contrast(
                        rows,
                        metric,
                        args.bootstrap,
                        args.seed + fidx * 100000 + midx * 5000,
                        args.min_samples,
                    )
                    for r in rr:
                        r["model"] = model
                    model_loop_rows.extend(rr)
                loop_contrast_all.extend(model_loop_rows)

                cum = cumulative_contrasts(
                    rows,
                    args.metrics,
                    args.bootstrap,
                    args.seed + fidx * 100000,
                    args.min_samples,
                )
                for r in cum:
                    r["model"] = model
                cumulative_all.extend(cum)

                first = first_token_contrasts(
                    rows,
                    args.metrics,
                    args.bootstrap,
                    args.seed + fidx * 100000,
                    args.min_samples,
                )
                for r in first:
                    r["model"] = model
                first_all.extend(first)

                el = early_late_contrasts(
                    rows,
                    args.metrics,
                    args.early_tokens,
                    args.bootstrap,
                    args.seed + fidx * 100000,
                    args.min_samples,
                )
                for r in el:
                    r["model"] = model
                early_late_all.extend(el)

                onset = onset_from_loop_contrasts(model_loop_rows)
                for r in onset:
                    r["model"] = model
                onset_all.extend(onset)

            did = model_did(
                loop_contrast_all,
                args.metrics,
                args.bootstrap,
                args.seed,
                args.min_samples,
                token_rows_by_model,
            )

            outputs = {
                "loop_profile.csv": loop_profile_all,
                "loop_contrasts.csv": loop_contrast_all,
                "cumulative_contrasts.csv": cumulative_all,
                "first_token_contrasts.csv": first_all,
                "early_late_contrasts.csv": early_late_all,
                "model_difference_in_differences.csv": did,
                "trajectory_onset.csv": onset_all,
            }

            for name, rows in outputs.items():
                write_csv(out_dir / name, rows)
                print(f"[WRITE] {out_dir / name} ({len(rows)} rows)")

            print("\nKey interpretation:")
            print("1) loop_contrasts.csv")
            print("   sentence_minus_chunk < 0 => sentence conditioning hurts that metric at loop j")
            print("2) cumulative_contrasts.csv")
            print("   shows whether the gap accumulates as generation proceeds")
            print("3) first_token_contrasts.csv")
            print("   if near 0 but whole-loop gap is negative, the effect is not an immediate restart effect")
            print("4) early_late_contrasts.csv")
            print("   distinguishes first-N-token vs later-within-loop drift")
            print("5) model_difference_in_differences.csv")
            print("   tests whether Qwen2.5-7B reacts differently from Qwen3B/Llama under shared prompts")
        main()
    finally:
        sys.argv = old_argv

def _run_retained(argv):
    """Run preserved implementation from analyze_kgw_sentence_boundary.py[retained]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_sentence_boundary.py[retained]"] + list(argv)

        """
        Retained-only global-token trajectory analysis for KGW.

        Uses EXISTING token_records.jsonl produced by:
            audit_kgw_sentence_boundary.py

        No GPU / model loading required.

        Key correction relative to the previous loop analysis
        -----------------------------------------------------
        For sentence_ids, ONLY tokens with:
            segment == "retained"
        are used.

        For chunk_ids, all generated tokens are retained in the audit, so the same
        retained filter is applied consistently.

        After filtering, tokens are re-indexed by their FINAL kept-token position:
            global_pos = 1, 2, ..., T

        This directly analyzes what the detector actually sees.

        Expected layout
        ---------------
        ROOT/
          Qwen2.5-3B-Instruct/kgw_matched/token_records.jsonl
          Qwen2.5-7B-Instruct/kgw_matched/token_records.jsonl
          Llama-3.2-3B-Instruct/kgw_matched/token_records.jsonl

        Outputs
        -------
        final_sequence_summary.csv
            Full-sequence retained-only metrics, including green fraction and KGW z-proxy.

        global_bin_contrasts.csv
            sentence_ids - chunk_ids by final-token bins (default width 20).

        cumulative_global_contrasts.csv
            sentence_ids - chunk_ids through final token positions 20,40,...,160.

        global_position_profile.csv
            Absolute metric profiles by model/mode/bin.

        sentence_stage_profile.csv
            sentence_ids only:
              global stage x sentence position (start/mid/end).

        sentence_stage_contrasts.csv
            sentence_ids only:
              start-mid, end-mid, start-end within each global stage.

        sample_level_final_contrasts.csv
            Per-sample full-sequence sentence-chunk differences for downstream analysis.

        model_did_global_bins.csv
            Difference-in-differences for Qwen7B vs controls by final-token bin.

        position_onset.csv
            Descriptive first final-token bin/cumulative threshold whose CI excludes zero.

        Important
        ---------
        Bootstrap statistical unit = prompt/sample, never token.
        """


        import argparse
        import csv
        import json
        import math
        from collections import defaultdict
        from pathlib import Path
        from typing import Any, Dict, Iterable, List, Tuple

        import numpy as np


        DEFAULT_METRICS = [
            "actual_is_green",
            "entropy_raw",
            "entropy_effective_pre",
            "top1_prob_effective_pre",
            "green_mass_raw",
            "green_mass_effective_pre",
            "green_mass_effective_wm",
            "green_mass_gain_effective",
            "top1_is_green_effective_pre",
            "top1_is_green_effective_wm",
        ]


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--root", required=True)
            p.add_argument("--output_dir", required=True)
            p.add_argument("--bootstrap", type=int, default=10000)
            p.add_argument("--seed", type=int, default=20260825)
            p.add_argument("--min_samples", type=int, default=10)

            p.add_argument("--bin_width", type=int, default=20)
            p.add_argument("--max_tokens", type=int, default=160)
            p.add_argument("--gamma", type=float, default=0.5)

            p.add_argument("--sentence_edge_tokens", type=int, default=5)
            p.add_argument("--stage1_end", type=int, default=60)
            p.add_argument("--stage2_end", type=int, default=120)

            p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
            return p.parse_args()


        def read_jsonl(path: Path) -> List[Dict[str, Any]]:
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows


        def ff(x):
            try:
                y = float(x)
            except Exception:
                return None
            return y if math.isfinite(y) else None


        def mean(vals: Iterable[Any]) -> float:
            a = [ff(x) for x in vals]
            a = [x for x in a if x is not None]
            return float(np.mean(a)) if a else float("nan")


        def bootstrap_mean(vals: List[float], n_boot: int, seed: int):
            a = np.asarray([float(x) for x in vals if math.isfinite(float(x))], dtype=float)
            if a.size == 0:
                return float("nan"), float("nan"), float("nan")

            rng = np.random.default_rng(seed)
            n = len(a)
            boot = np.empty(n_boot, dtype=float)

            for i in range(n_boot):
                boot[i] = np.mean(a[rng.integers(0, n, size=n)])

            return (
                float(np.mean(a)),
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            )


        def write_csv(path: Path, rows: List[Dict[str, Any]]):
            if not rows:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)


        def retained_rows_with_global_pos(
            rows: List[Dict[str, Any]],
            max_tokens: int,
        ) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
            """
            Return {(sample_index, mode): [retained token rows in final order]}.

            sentence_ids:
                only segment == retained
            chunk_ids:
                audit marks all kept tokens as retained; same filter is used.

            Global final position is assigned after filtering.
            """
            groups = defaultdict(list)

            for r in rows:
                mode = r.get("mode")
                if mode not in ("chunk_ids", "sentence_ids"):
                    continue

                if r.get("segment") != "retained":
                    continue

                sid = int(r["sample_index"])
                groups[(sid, mode)].append(dict(r))

            for key, rr in groups.items():
                rr.sort(key=lambda x: (int(x["loop_index"]), int(x["raw_pos"])))
                if max_tokens > 0:
                    rr[:] = rr[:max_tokens]

                for i, r in enumerate(rr, start=1):
                    r["global_pos"] = i

            return groups


        def sample_metric(rr: List[Dict[str, Any]], metric: str) -> float:
            return mean([r.get(metric) for r in rr])


        def green_z_proxy(rr: List[Dict[str, Any]], gamma: float) -> float:
            vals = [ff(r.get("actual_is_green")) for r in rr]
            vals = [x for x in vals if x is not None]
            T = len(vals)
            if T <= 0:
                return float("nan")

            G = float(np.sum(vals))
            denom = math.sqrt(gamma * (1.0 - gamma) * T)
            if denom <= 0:
                return float("nan")
            return (G - gamma * T) / denom


        def get_common_sample_ids(groups):
            chunk = {sid for sid, mode in groups if mode == "chunk_ids"}
            sent = {sid for sid, mode in groups if mode == "sentence_ids"}
            return sorted(chunk & sent)


        def full_sequence_summary(
            model: str,
            groups,
            metrics,
            gamma,
            n_boot,
            seed,
            min_samples,
        ):
            rows_out = []
            samples_out = []

            sids = get_common_sample_ids(groups)

            # Per-sample records
            for sid in sids:
                c = groups[(sid, "chunk_ids")]
                s = groups[(sid, "sentence_ids")]

                rec = {
                    "model": model,
                    "sample_index": sid,
                    "chunk_tokens": len(c),
                    "sentence_tokens": len(s),
                    "chunk_green_fraction": sample_metric(c, "actual_is_green"),
                    "sentence_green_fraction": sample_metric(s, "actual_is_green"),
                    "green_fraction_delta": (
                        sample_metric(s, "actual_is_green")
                        - sample_metric(c, "actual_is_green")
                    ),
                    "chunk_z_proxy": green_z_proxy(c, gamma),
                    "sentence_z_proxy": green_z_proxy(s, gamma),
                    "z_proxy_delta": green_z_proxy(s, gamma) - green_z_proxy(c, gamma),
                }

                for metric in metrics:
                    cv = sample_metric(c, metric)
                    sv = sample_metric(s, metric)
                    rec[f"chunk__{metric}"] = cv
                    rec[f"sentence__{metric}"] = sv
                    rec[f"delta__{metric}"] = sv - cv

                samples_out.append(rec)

            # Bootstrap full-sequence contrasts
            special = [
                ("green_fraction", "green_fraction_delta"),
                ("z_proxy", "z_proxy_delta"),
            ]

            for idx, (metric, key) in enumerate(special):
                vals = [r[key] for r in samples_out if math.isfinite(float(r[key]))]
                if len(vals) < min_samples:
                    continue

                d, lo, hi = bootstrap_mean(vals, n_boot, seed + idx * 101)
                rows_out.append({
                    "model": model,
                    "metric": metric,
                    "n_pairs": len(vals),
                    "sentence_minus_chunk": d,
                    "ci95_low": lo,
                    "ci95_high": hi,
                })

            for midx, metric in enumerate(metrics):
                vals = [
                    r[f"delta__{metric}"]
                    for r in samples_out
                    if math.isfinite(float(r[f"delta__{metric}"]))
                ]
                if len(vals) < min_samples:
                    continue

                d, lo, hi = bootstrap_mean(
                    vals,
                    n_boot,
                    seed + 10000 + midx * 137,
                )
                rows_out.append({
                    "model": model,
                    "metric": metric,
                    "n_pairs": len(vals),
                    "sentence_minus_chunk": d,
                    "ci95_low": lo,
                    "ci95_high": hi,
                })

            return rows_out, samples_out


        def bin_ranges(max_tokens: int, width: int):
            out = []
            start = 1
            while start <= max_tokens:
                end = min(max_tokens, start + width - 1)
                out.append((start, end))
                start = end + 1
            return out


        def slice_global(rr, start, end):
            return [
                r for r in rr
                if start <= int(r["global_pos"]) <= end
            ]


        def global_bin_analysis(
            model,
            groups,
            metrics,
            max_tokens,
            bin_width,
            n_boot,
            seed,
            min_samples,
        ):
            profile = []
            contrasts = []
            sids = get_common_sample_ids(groups)

            for bidx, (start, end) in enumerate(bin_ranges(max_tokens, bin_width)):
                for midx, metric in enumerate(metrics):
                    cvals, svals, diffs = [], [], []

                    for sid in sids:
                        c = slice_global(groups[(sid, "chunk_ids")], start, end)
                        s = slice_global(groups[(sid, "sentence_ids")], start, end)

                        # Require at least one retained token from BOTH modes.
                        if not c or not s:
                            continue

                        cv = sample_metric(c, metric)
                        sv = sample_metric(s, metric)

                        if math.isfinite(cv) and math.isfinite(sv):
                            cvals.append(cv)
                            svals.append(sv)
                            diffs.append(sv - cv)

                    if len(diffs) < min_samples:
                        continue

                    dm, dlo, dhi = bootstrap_mean(
                        diffs,
                        n_boot,
                        seed + 100000 + bidx * 1000 + midx * 17,
                    )

                    contrasts.append({
                        "model": model,
                        "metric": metric,
                        "bin_start": start,
                        "bin_end": end,
                        "n_pairs": len(diffs),
                        "chunk_mean": float(np.mean(cvals)),
                        "sentence_mean": float(np.mean(svals)),
                        "sentence_minus_chunk": dm,
                        "ci95_low": dlo,
                        "ci95_high": dhi,
                    })

                    for mode, vals in [("chunk_ids", cvals), ("sentence_ids", svals)]:
                        m, lo, hi = bootstrap_mean(
                            vals,
                            n_boot,
                            seed + 200000 + bidx * 1000 + midx * 17 + (0 if mode == "chunk_ids" else 1),
                        )
                        profile.append({
                            "model": model,
                            "metric": metric,
                            "mode": mode,
                            "bin_start": start,
                            "bin_end": end,
                            "n_samples": len(vals),
                            "mean": m,
                            "ci95_low": lo,
                            "ci95_high": hi,
                        })

            return profile, contrasts


        def cumulative_analysis(
            model,
            groups,
            metrics,
            max_tokens,
            bin_width,
            n_boot,
            seed,
            min_samples,
        ):
            out = []
            sids = get_common_sample_ids(groups)

            thresholds = [end for _, end in bin_ranges(max_tokens, bin_width)]

            for tidx, end in enumerate(thresholds):
                for midx, metric in enumerate(metrics):
                    cvals, svals, diffs = [], [], []

                    for sid in sids:
                        c = slice_global(groups[(sid, "chunk_ids")], 1, end)
                        s = slice_global(groups[(sid, "sentence_ids")], 1, end)

                        # To make cumulative positions comparable, require both modes
                        # to actually reach this threshold.
                        if len(c) < end or len(s) < end:
                            continue

                        cv = sample_metric(c, metric)
                        sv = sample_metric(s, metric)

                        if math.isfinite(cv) and math.isfinite(sv):
                            cvals.append(cv)
                            svals.append(sv)
                            diffs.append(sv - cv)

                    if len(diffs) < min_samples:
                        continue

                    d, lo, hi = bootstrap_mean(
                        diffs,
                        n_boot,
                        seed + 300000 + tidx * 1000 + midx * 19,
                    )

                    out.append({
                        "model": model,
                        "metric": metric,
                        "through_global_pos": end,
                        "n_pairs": len(diffs),
                        "chunk_cumulative_mean": float(np.mean(cvals)),
                        "sentence_cumulative_mean": float(np.mean(svals)),
                        "sentence_minus_chunk_cumulative": d,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    })

            return out


        def global_stage(pos: int, stage1_end: int, stage2_end: int):
            if pos <= stage1_end:
                return "early"
            if pos <= stage2_end:
                return "middle"
            return "late"


        def sentence_zone(r, edge_tokens: int):
            """
            sentence_ids retained tokens already carry:
              sent_pos_from_start
              sent_pos_to_end
            """
            a = r.get("sent_pos_from_start")
            b = r.get("sent_pos_to_end")
            if a is None or b is None:
                return None

            a = int(a)
            b = int(b)

            # If a very short sentence belongs to both start and end, put it into
            # "edge_both" instead of silently double-counting.
            is_start = a <= edge_tokens
            is_end = b <= edge_tokens

            if is_start and is_end:
                return "edge_both"
            if is_start:
                return "start"
            if is_end:
                return "end"
            return "mid"


        def sentence_stage_analysis(
            model,
            groups,
            metrics,
            edge_tokens,
            stage1_end,
            stage2_end,
            n_boot,
            seed,
            min_samples,
        ):
            """
            sentence_ids only; retained final tokens only.
            """
            profile = []
            contrasts = []

            sids = sorted({sid for sid, mode in groups if mode == "sentence_ids"})

            # {(stage, zone, metric): [per-sample mean]}
            sample_bucket = defaultdict(list)

            for midx, metric in enumerate(metrics):
                per_key_sample = defaultdict(list)

                for sid in sids:
                    rr = groups[(sid, "sentence_ids")]

                    buckets = defaultdict(list)
                    for r in rr:
                        stage = global_stage(
                            int(r["global_pos"]),
                            stage1_end,
                            stage2_end,
                        )
                        zone = sentence_zone(r, edge_tokens)
                        if zone is None:
                            continue
                        x = ff(r.get(metric))
                        if x is None:
                            continue
                        buckets[(stage, zone)].append(x)

                    for key, vals in buckets.items():
                        if vals:
                            per_key_sample[(key[0], key[1], sid)] = float(np.mean(vals))

                stages = ["early", "middle", "late"]
                zones = ["start", "mid", "end", "edge_both"]

                for sidx, stage in enumerate(stages):
                    for zidx, zone in enumerate(zones):
                        vals = [
                            v for (st, zn, sid), v in per_key_sample.items()
                            if st == stage and zn == zone
                        ]
                        if len(vals) < min_samples:
                            continue

                        m, lo, hi = bootstrap_mean(
                            vals,
                            n_boot,
                            seed + 400000 + midx * 10000 + sidx * 1000 + zidx * 100,
                        )
                        profile.append({
                            "model": model,
                            "metric": metric,
                            "global_stage": stage,
                            "sentence_zone": zone,
                            "n_samples": len(vals),
                            "mean": m,
                            "ci95_low": lo,
                            "ci95_high": hi,
                        })

                    for cidx, (za, zb, cname) in enumerate([
                        ("start", "mid", "start_minus_mid"),
                        ("end", "mid", "end_minus_mid"),
                        ("start", "end", "start_minus_end"),
                    ]):
                        vals = []
                        common_sids = sorted({
                            sid for (st, zn, sid) in per_key_sample
                            if st == stage and zn == za
                        } & {
                            sid for (st, zn, sid) in per_key_sample
                            if st == stage and zn == zb
                        })

                        for sid in common_sids:
                            a = per_key_sample[(stage, za, sid)]
                            b = per_key_sample[(stage, zb, sid)]
                            if math.isfinite(a) and math.isfinite(b):
                                vals.append(a - b)

                        if len(vals) < min_samples:
                            continue

                        d, lo, hi = bootstrap_mean(
                            vals,
                            n_boot,
                            seed + 500000 + midx * 10000 + sidx * 1000 + cidx * 100,
                        )
                        contrasts.append({
                            "model": model,
                            "metric": metric,
                            "global_stage": stage,
                            "contrast": cname,
                            "n_pairs": len(vals),
                            "mean_difference": d,
                            "ci95_low": lo,
                            "ci95_high": hi,
                        })

            return profile, contrasts


        def model_did_bins(
            token_rows_by_model,
            metrics,
            max_tokens,
            bin_width,
            n_boot,
            seed,
            min_samples,
        ):
            comparisons = [
                ("Qwen2.5-7B-Instruct", "Qwen2.5-3B-Instruct"),
                ("Qwen2.5-7B-Instruct", "Llama-3.2-3B-Instruct"),
                ("Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"),
            ]

            groups_by_model = {
                model: retained_rows_with_global_pos(rows, max_tokens)
                for model, rows in token_rows_by_model.items()
            }

            out = []

            for comp_idx, (a_model, b_model) in enumerate(comparisons):
                if a_model not in groups_by_model or b_model not in groups_by_model:
                    continue

                ga = groups_by_model[a_model]
                gb = groups_by_model[b_model]

                sids = sorted(
                    set(get_common_sample_ids(ga))
                    & set(get_common_sample_ids(gb))
                )

                for bidx, (start, end) in enumerate(bin_ranges(max_tokens, bin_width)):
                    for midx, metric in enumerate(metrics):
                        vals = []

                        for sid in sids:
                            try:
                                ac = slice_global(ga[(sid, "chunk_ids")], start, end)
                                ass = slice_global(ga[(sid, "sentence_ids")], start, end)
                                bc = slice_global(gb[(sid, "chunk_ids")], start, end)
                                bs = slice_global(gb[(sid, "sentence_ids")], start, end)
                            except KeyError:
                                continue

                            if not ac or not ass or not bc or not bs:
                                continue

                            da = sample_metric(ass, metric) - sample_metric(ac, metric)
                            db = sample_metric(bs, metric) - sample_metric(bc, metric)

                            if math.isfinite(da) and math.isfinite(db):
                                vals.append(da - db)

                        if len(vals) < min_samples:
                            continue

                        d, lo, hi = bootstrap_mean(
                            vals,
                            n_boot,
                            seed + 600000 + comp_idx * 100000 + bidx * 1000 + midx * 23,
                        )

                        out.append({
                            "model_A": a_model,
                            "model_B": b_model,
                            "metric": metric,
                            "bin_start": start,
                            "bin_end": end,
                            "n_pairs": len(vals),
                            "DID_A_minus_B": d,
                            "ci95_low": lo,
                            "ci95_high": hi,
                        })

            return out


        def descriptive_onset(bin_rows, cumulative_rows):
            out = []

            # Per-bin onset
            by = defaultdict(list)
            for r in bin_rows:
                by[(r["model"], r["metric"])].append(r)

            for (model, metric), rr in sorted(by.items()):
                rr = sorted(rr, key=lambda x: int(x["bin_start"]))

                first_neg = None
                first_pos = None

                for r in rr:
                    if float(r["ci95_high"]) < 0 and first_neg is None:
                        first_neg = f'{r["bin_start"]}-{r["bin_end"]}'
                    if float(r["ci95_low"]) > 0 and first_pos is None:
                        first_pos = f'{r["bin_start"]}-{r["bin_end"]}'

                out.append({
                    "model": model,
                    "metric": metric,
                    "analysis": "bin",
                    "first_significant_negative": first_neg,
                    "first_significant_positive": first_pos,
                })

            # Cumulative onset
            by = defaultdict(list)
            for r in cumulative_rows:
                by[(r["model"], r["metric"])].append(r)

            for (model, metric), rr in sorted(by.items()):
                rr = sorted(rr, key=lambda x: int(x["through_global_pos"]))

                first_neg = None
                first_pos = None

                for r in rr:
                    if float(r["ci95_high"]) < 0 and first_neg is None:
                        first_neg = int(r["through_global_pos"])
                    if float(r["ci95_low"]) > 0 and first_pos is None:
                        first_pos = int(r["through_global_pos"])

                out.append({
                    "model": model,
                    "metric": metric,
                    "analysis": "cumulative",
                    "first_significant_negative": first_neg,
                    "first_significant_positive": first_pos,
                })

            return out


        def main():
            args = parse_args()

            root = Path(args.root)
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(root.glob("*/kgw_matched/token_records.jsonl"))
            if not files:
                raise SystemExit(
                    f"No token_records.jsonl found under {root}/*/kgw_matched/"
                )

            token_rows_by_model = {}

            final_summary_all = []
            sample_final_all = []
            profile_all = []
            bins_all = []
            cumulative_all = []
            stage_profile_all = []
            stage_contrast_all = []

            for fidx, path in enumerate(files):
                rows = read_jsonl(path)
                if not rows:
                    continue

                model = rows[0]["model"]
                token_rows_by_model[model] = rows

                groups = retained_rows_with_global_pos(rows, args.max_tokens)

                print(f"[*] {model}")
                print(f"    raw token records : {len(rows)}")
                print(f"    kept sample/modes : {len(groups)}")

                final_summary, sample_final = full_sequence_summary(
                    model,
                    groups,
                    args.metrics,
                    args.gamma,
                    args.bootstrap,
                    args.seed + fidx * 100000,
                    args.min_samples,
                )
                final_summary_all.extend(final_summary)
                sample_final_all.extend(sample_final)

                profile, bins = global_bin_analysis(
                    model,
                    groups,
                    args.metrics,
                    args.max_tokens,
                    args.bin_width,
                    args.bootstrap,
                    args.seed + fidx * 100000,
                    args.min_samples,
                )
                profile_all.extend(profile)
                bins_all.extend(bins)

                cum = cumulative_analysis(
                    model,
                    groups,
                    args.metrics,
                    args.max_tokens,
                    args.bin_width,
                    args.bootstrap,
                    args.seed + fidx * 100000,
                    args.min_samples,
                )
                cumulative_all.extend(cum)

                sp, sc = sentence_stage_analysis(
                    model,
                    groups,
                    args.metrics,
                    args.sentence_edge_tokens,
                    args.stage1_end,
                    args.stage2_end,
                    args.bootstrap,
                    args.seed + fidx * 100000,
                    args.min_samples,
                )
                stage_profile_all.extend(sp)
                stage_contrast_all.extend(sc)

            did = model_did_bins(
                token_rows_by_model,
                args.metrics,
                args.max_tokens,
                args.bin_width,
                args.bootstrap,
                args.seed,
                args.min_samples,
            )

            onset = descriptive_onset(bins_all, cumulative_all)

            outputs = {
                "final_sequence_summary.csv": final_summary_all,
                "sample_level_final_contrasts.csv": sample_final_all,
                "global_position_profile.csv": profile_all,
                "global_bin_contrasts.csv": bins_all,
                "cumulative_global_contrasts.csv": cumulative_all,
                "sentence_stage_profile.csv": stage_profile_all,
                "sentence_stage_contrasts.csv": stage_contrast_all,
                "model_did_global_bins.csv": did,
                "position_onset.csv": onset,
            }

            for name, rr in outputs.items():
                write_csv(out_dir / name, rr)
                print(f"[WRITE] {out_dir / name} ({len(rr)} rows)")

            print("\nKey checks:")
            print("1) final_sequence_summary.csv")
            print("   Qwen7B z_proxy / green_fraction should reproduce the retained-only C3-C1 direction.")
            print("2) global_bin_contrasts.csv")
            print("   Locates the final-token region where sentence_ids starts losing green enrichment.")
            print("3) cumulative_global_contrasts.csv")
            print("   Tests whether the final retained-text gap accumulates with token position.")
            print("4) model_did_global_bins.csv")
            print("   Tests whether Qwen7B differs from Qwen3B/Llama under shared prompts.")
            print("5) sentence_stage_contrasts.csv")
            print("   Tests sentence start/mid/end behavior separately in early/middle/late final text.")
        main()
    finally:
        sys.argv = old_argv

def _run_plot_loop(argv):
    """Run preserved implementation from analyze_kgw_sentence_boundary.py[plot-loop]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_sentence_boundary.py[plot-loop]"] + list(argv)

        """
        Plot loop-wise sentence-minus-chunk trajectory from loop_contrasts.csv.

        Creates one PNG per metric. No seaborn; one figure per metric.
        """

        import argparse
        import csv
        from collections import defaultdict
        from pathlib import Path

        import matplotlib.pyplot as plt


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--input", required=True)
            p.add_argument("--output_dir", required=True)
            p.add_argument(
                "--metrics",
                nargs="+",
                default=[
                    "actual_is_green",
                    "entropy_raw",
                    "green_mass_effective_pre",
                    "green_mass_effective_wm",
                    "green_mass_gain_effective",
                ],
            )
            return p.parse_args()


        def main():
            args = parse_args()
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            rows = []
            with open(args.input, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            for metric in args.metrics:
                sub = [r for r in rows if r["metric"] == metric]
                if not sub:
                    continue

                by_model = defaultdict(list)
                for r in sub:
                    by_model[r["model"]].append(r)

                fig, ax = plt.subplots(figsize=(7.2, 4.6))
                for model, rr in sorted(by_model.items()):
                    rr = sorted(rr, key=lambda x: int(x["loop_index"]))
                    x = [int(r["loop_index"]) for r in rr]
                    y = [float(r["sentence_minus_chunk"]) for r in rr]
                    lo = [float(r["ci95_low"]) for r in rr]
                    hi = [float(r["ci95_high"]) for r in rr]
                    yerr = [
                        [yy - ll for yy, ll in zip(y, lo)],
                        [hh - yy for yy, hh in zip(y, hi)],
                    ]
                    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=model)

                ax.axhline(0.0, linewidth=1)
                ax.set_xlabel("Outer generation loop")
                ax.set_ylabel("Sentence-ids minus chunk-ids")
                ax.set_title(metric)
                ax.legend()
                fig.tight_layout()
                path = out / f"loop_delta_{metric}.png"
                fig.savefig(path, dpi=200)
                plt.close(fig)
                print("[WRITE]", path)
        main()
    finally:
        sys.argv = old_argv

def _run_plot_retained(argv):
    """Run preserved implementation from analyze_kgw_sentence_boundary.py[plot-retained]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_sentence_boundary.py[plot-retained]"] + list(argv)

        """
        Plot retained-only final-token trajectory.

        Creates separate figures for each metric:
          1) per-bin sentence-minus-chunk
          2) cumulative sentence-minus-chunk

        No seaborn. No explicit colors/styles.
        """

        import argparse
        import csv
        from collections import defaultdict
        from pathlib import Path

        import matplotlib.pyplot as plt


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--bin_csv", required=True)
            p.add_argument("--cumulative_csv", required=True)
            p.add_argument("--output_dir", required=True)
            p.add_argument(
                "--metrics",
                nargs="+",
                default=[
                    "actual_is_green",
                    "entropy_effective_pre",
                    "green_mass_effective_pre",
                    "green_mass_effective_wm",
                    "green_mass_gain_effective",
                ],
            )
            return p.parse_args()


        def read_csv(path):
            with open(path, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f))


        def main():
            args = parse_args()
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            bins = read_csv(args.bin_csv)
            cum = read_csv(args.cumulative_csv)

            for metric in args.metrics:
                # --------------------------------------------------------------
                # Per-bin
                # --------------------------------------------------------------
                sub = [r for r in bins if r["metric"] == metric]
                if sub:
                    by_model = defaultdict(list)
                    for r in sub:
                        by_model[r["model"]].append(r)

                    fig, ax = plt.subplots(figsize=(7.4, 4.8))

                    for model, rr in sorted(by_model.items()):
                        rr = sorted(rr, key=lambda x: int(x["bin_start"]))
                        x = [
                            (int(r["bin_start"]) + int(r["bin_end"])) / 2
                            for r in rr
                        ]
                        y = [float(r["sentence_minus_chunk"]) for r in rr]
                        lo = [float(r["ci95_low"]) for r in rr]
                        hi = [float(r["ci95_high"]) for r in rr]
                        yerr = [
                            [yy - ll for yy, ll in zip(y, lo)],
                            [hh - yy for yy, hh in zip(y, hi)],
                        ]
                        ax.errorbar(
                            x, y, yerr=yerr,
                            marker="o", capsize=3, label=model
                        )

                    ax.axhline(0.0, linewidth=1)
                    ax.set_xlabel("Final kept-token position (bin midpoint)")
                    ax.set_ylabel("Sentence-ids minus chunk-ids")
                    ax.set_title(f"Retained-only bin contrast: {metric}")
                    ax.legend()
                    fig.tight_layout()

                    path = out / f"bin_delta_{metric}.png"
                    fig.savefig(path, dpi=200)
                    plt.close(fig)
                    print("[WRITE]", path)

                # --------------------------------------------------------------
                # Cumulative
                # --------------------------------------------------------------
                sub = [r for r in cum if r["metric"] == metric]
                if sub:
                    by_model = defaultdict(list)
                    for r in sub:
                        by_model[r["model"]].append(r)

                    fig, ax = plt.subplots(figsize=(7.4, 4.8))

                    for model, rr in sorted(by_model.items()):
                        rr = sorted(rr, key=lambda x: int(x["through_global_pos"]))
                        x = [int(r["through_global_pos"]) for r in rr]
                        y = [float(r["sentence_minus_chunk_cumulative"]) for r in rr]
                        lo = [float(r["ci95_low"]) for r in rr]
                        hi = [float(r["ci95_high"]) for r in rr]
                        yerr = [
                            [yy - ll for yy, ll in zip(y, lo)],
                            [hh - yy for yy, hh in zip(y, hi)],
                        ]
                        ax.errorbar(
                            x, y, yerr=yerr,
                            marker="o", capsize=3, label=model
                        )

                    ax.axhline(0.0, linewidth=1)
                    ax.set_xlabel("Final kept-token prefix length")
                    ax.set_ylabel("Cumulative sentence-ids minus chunk-ids")
                    ax.set_title(f"Retained-only cumulative contrast: {metric}")
                    ax.legend()
                    fig.tight_layout()

                    path = out / f"cumulative_delta_{metric}.png"
                    fig.savefig(path, dpi=200)
                    plt.close(fig)
                    print("[WRITE]", path)
        main()
    finally:
        sys.argv = old_argv

COMMANDS = {
    "summary": _run_summary,
    "loop": _run_loop,
    "retained": _run_retained,
    "plot-loop": _run_plot_loop,
    "plot-retained": _run_plot_retained,
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage:\n"
            "  analyze_kgw_sentence_boundary.py summary <old summary args>\n"
            "  analyze_kgw_sentence_boundary.py loop <old loop args>\n"
            "  analyze_kgw_sentence_boundary.py retained <old retained args>\n"
            "  analyze_kgw_sentence_boundary.py plot-loop <old plot args>\n"
            "  analyze_kgw_sentence_boundary.py plot-retained <old plot args>\n"
            "\nEach subcommand preserves the corresponding old CLI."
        )
        return
    command = sys.argv[1]
    fn = COMMANDS.get(command)
    if fn is None:
        raise SystemExit(f"Unknown subcommand: {command}. Choose from: {', '.join(COMMANDS)}")
    fn(sys.argv[2:])

if __name__ == "__main__":
    main()
