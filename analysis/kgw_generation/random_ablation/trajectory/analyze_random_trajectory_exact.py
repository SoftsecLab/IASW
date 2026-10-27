# -*- coding: utf-8 -*-
"""
Exact-span cumulative KGW trajectory analysis.

This fixes an important issue in the previous trajectory script:
it reconstructed text by sentence-tokenizing and joining sentences with spaces.
For a context-dependent token watermark, even small textual/tokenization changes
can invalidate the detector trajectory.

This version NEVER reconstructs the text.

It obtains sentence END CHARACTER OFFSETS from the original wm_text and scores:

    original_text[:end_char]

Therefore each cumulative prefix is an exact substring of the stored text.
At 100% progress, the score is computed on the exact original wm_text.

A mandatory self-check compares:
    direct detector z(original wm_text)
against
    saved detection_wm.z_score_micro

If they do not match, the script reports the mismatch and stops by default.

No CausalLM is loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.watermarks.algorithms.micro.kgw import KGWMicroAlgorithm


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            x["_line_idx"] = i
            rows.append(x)
    return rows


def sid(row):
    for k in ["idx", "id", "sample_id", "record_idx", "index"]:
        if k in row:
            return str(row[k])
    return str(row["_line_idx"])


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def saved_micro_z(row):
    d = row.get("detection_wm", {})
    if not isinstance(d, dict):
        return np.nan
    return num(d.get("z_score_micro"))


def exact_sentence_end_offsets(text):
    """
    Return monotonically increasing character offsets. Every prefix text[:end]
    is taken directly from the original string.

    Prefer NLTK Punkt spans when available. Fall back to regex sentence-end
    spans without modifying characters.
    """
    text = str(text or "")
    if not text:
        return []

    offsets = []

    try:
        import nltk
        tokenizer = nltk.data.load(
            "tokenizers/punkt/english.pickle"
        )
        spans = list(tokenizer.span_tokenize(text))
        offsets = [int(end) for _, end in spans if end > 0]
    except Exception:
        # Sentence-ending punctuation + following closing quote/bracket.
        for m in re.finditer(
            r'[.!?。！？]+(?:["\')\]]+)?(?=\s|$)',
            text,
        ):
            offsets.append(int(m.end()))

    offsets = sorted(set(o for o in offsets if 0 < o <= len(text)))

    # 100% must be exact original text.
    if not offsets or offsets[-1] != len(text):
        offsets.append(len(text))

    return offsets


def detector_stats(algo, text, gamma):
    try:
        res = algo.detector.detect(text)
    except Exception:
        res = {}

    z = num(res.get("z_score"))
    T = num(
        res.get(
            "num_tokens_scored",
            res.get("T", res.get("num_scored")),
        )
    )
    G = num(
        res.get(
            "num_green_tokens",
            res.get("green_token_count", res.get("green_tokens")),
        )
    )

    frac = np.nan
    if np.isfinite(T) and np.isfinite(G) and T > 0:
        frac = G / T

    return {
        "z": z,
        "n_scored": T,
        "n_green": G,
        "green_rate": frac,
        "green_excess": (
            frac - gamma
            if np.isfinite(frac)
            else np.nan
        ),
    }


def score_file(path, label, algo, gamma):
    out = []
    selfcheck = []

    for row in load_jsonl(path):
        sample_id = sid(row)
        text = str(row.get("wm_text", "") or "")
        if not text.strip():
            continue

        # Mandatory exact full-text detector check.
        full = detector_stats(algo, text, gamma)
        saved = saved_micro_z(row)

        selfcheck.append({
            "sample_id": sample_id,
            "label": label,
            "saved_z": saved,
            "direct_exact_z": full["z"],
            "abs_diff": (
                abs(saved - full["z"])
                if np.isfinite(saved)
                and np.isfinite(full["z"])
                else np.nan
            ),
        })

        ends = exact_sentence_end_offsets(text)
        n_steps = len(ends)

        for step, end_char in enumerate(ends, start=1):
            prefix = text[:end_char]
            st = (
                full
                if end_char == len(text)
                else detector_stats(algo, prefix, gamma)
            )

            out.append({
                "sample_id": sample_id,
                "label": label,
                "sentence_step": step,
                "n_sentences_total": n_steps,
                "progress": step / max(n_steps, 1),
                "end_char": end_char,
                "text_chars_total": len(text),
                **st,
            })

    return pd.DataFrame(out), pd.DataFrame(selfcheck)


def bootstrap_delta(x, n_boot, seed):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]

    if len(a) == 0:
        return {
            "n": 0,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "positive_fraction": np.nan,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    b = a[idx].mean(axis=1)

    return {
        "n": len(a),
        "mean_delta": float(a.mean()),
        "median_delta": float(np.median(a)),
        "ci_low": float(np.quantile(b, 0.025)),
        "ci_high": float(np.quantile(b, 0.975)),
        "positive_fraction": float(np.mean(a > 0)),
    }


def progress_frame(df, progress_points):
    rows = []

    for sample_id, g in df.groupby("sample_id"):
        g = g.sort_values("sentence_step")
        n = len(g)
        if n == 0:
            continue

        for q in progress_points:
            target = max(1, int(math.ceil(q*n)))
            r = g[g["sentence_step"] == target]
            if r.empty:
                continue
            rr = r.iloc[0]

            rows.append({
                "sample_id": sample_id,
                "progress_bin": q,
                "sentence_step": target,
                "z": rr["z"],
                "n_scored": rr["n_scored"],
                "green_rate": rr["green_rate"],
                "green_excess": rr["green_excess"],
            })

    return pd.DataFrame(rows)


def build_random_mean(frames, q, metric):
    merged = None

    for j, df in enumerate(frames):
        d = df[df["progress_bin"] == q][
            ["sample_id", metric]
        ].rename(columns={metric: f"{metric}_r{j}"})

        merged = (
            d
            if merged is None
            else merged.merge(d, on="sample_id", how="inner")
        )

    cols = [
        c for c in merged.columns
        if c.startswith(metric + "_r")
    ]
    merged["random_mean"] = merged[cols].mean(axis=1)
    return merged[["sample_id", "random_mean"]]


def summarize_progress(
    iasw,
    random_frames,
    progress_points,
    metrics,
    n_boot,
    seed,
):
    ip = progress_frame(iasw, progress_points)
    rps = [
        progress_frame(r, progress_points)
        for r in random_frames
    ]

    rows = []

    for q in progress_points:
        ii = ip[ip["progress_bin"] == q]

        for metric in metrics:
            rm = build_random_mean(rps, q, metric)

            m = ii[
                ["sample_id", metric]
            ].merge(rm, on="sample_id", how="inner")

            delta = m[metric] - m["random_mean"]

            stat = bootstrap_delta(
                delta,
                n_boot,
                seed + len(rows),
            )

            rows.append({
                "progress": q,
                "metric": metric,
                "iasw_mean": float(m[metric].mean()),
                "random_mean": float(m["random_mean"].mean()),
                **stat,
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument(
        "--selfcheck_tolerance",
        type=float,
        default=1e-6,
    )
    ap.add_argument(
        "--allow_selfcheck_mismatch",
        action="store_true",
    )
    args = ap.parse_args()

    records = Path(args.records_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    iasw_files = sorted(
        records.glob("dual_kgw_iasw_k*_control.jsonl")
    )
    random_files = [
        p for p in sorted(
            records.glob("dual_kgw_random_k*_seed*.jsonl")
        )
        if "random_k1" not in p.stem
    ]

    if not iasw_files or not random_files:
        raise RuntimeError("IASW/Random files not found.")

    local_only = Path(args.model_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="right",
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dummy_args = SimpleNamespace(
        gamma=args.gamma,
        delta=args.delta,
    )
    algo = KGWMicroAlgorithm(
        dummy_args,
        tokenizer,
        model=None,
        device=args.device,
    )

    print("[SCORE] IASW exact spans")
    iasw, check_i = score_file(
        iasw_files[0],
        "IASW",
        algo,
        args.gamma,
    )

    random_frames = []
    checks = [check_i]

    for j, p in enumerate(random_files):
        print("[SCORE] Random exact spans:", p)
        df, ck = score_file(
            p,
            f"Random{j}",
            algo,
            args.gamma,
        )
        random_frames.append(df)
        checks.append(ck)

    check = pd.concat(checks, ignore_index=True)
    check.to_csv(
        out / "trajectory_detector_selfcheck.csv",
        index=False,
        encoding="utf-8-sig",
    )

    valid_diff = check["abs_diff"].dropna()

    max_diff = (
        float(valid_diff.max())
        if len(valid_diff)
        else np.nan
    )
    mean_diff = (
        float(valid_diff.mean())
        if len(valid_diff)
        else np.nan
    )

    print(
        f"[SELFCHECK] mean abs diff={mean_diff:.10f}, "
        f"max abs diff={max_diff:.10f}"
    )

    if (
        np.isfinite(max_diff)
        and max_diff > args.selfcheck_tolerance
        and not args.allow_selfcheck_mismatch
    ):
        raise RuntimeError(
            "Exact full-text KGW detector does not reproduce saved "
            f"detection_wm scores. max_abs_diff={max_diff}. "
            "Do not interpret trajectory results yet."
        )

    progress_points = [0.25, 0.50, 0.75, 1.00]
    metrics = [
        "z",
        "n_scored",
        "green_rate",
        "green_excess",
    ]

    summary = summarize_progress(
        iasw,
        random_frames,
        progress_points,
        metrics,
        args.n_boot,
        args.seed,
    )

    iasw.to_csv(
        out / "trajectory_exact_iasw_steps.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for j, df in enumerate(random_frames):
        df.to_csv(
            out / f"trajectory_exact_random{j}_steps.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary.to_csv(
        out / "trajectory_exact_by_progress.csv",
        index=False,
        encoding="utf-8-sig",
    )

    focus = summary[summary["metric"] == "z"].copy()
    focus.to_csv(
        out / "trajectory_exact_micro_z_focus.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Exact trajectory micro-z ===")
    print(focus.to_string(index=False))

    print("\n[DONE]")
    print(out / "trajectory_detector_selfcheck.csv")
    print(out / "trajectory_exact_micro_z_focus.csv")


if __name__ == "__main__":
    main()
