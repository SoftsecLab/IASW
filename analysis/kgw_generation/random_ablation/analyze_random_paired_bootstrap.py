# -*- coding: utf-8 -*-
"""
Paired sample-level analysis for the existing Random-K1 / Random-K32 /
IASW-K32 control experiment.

No generation and no detection are performed.

Main comparison:
    IASW-K32 micro z
      minus
    mean(Random-K32 micro z over random selection seeds)

Bootstrap resamples prompts/samples, not individual detector scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
                if isinstance(x, dict):
                    x["_line_idx"] = line_idx
                    rows.append(x)
            except Exception:
                pass
    return rows


def record_id(row, fallback):
    for key in ["idx", "id", "sample_id", "record_idx", "index"]:
        if key in row:
            return str(row[key])
    return str(fallback)


def z(det, key):
    if not isinstance(det, dict):
        return np.nan
    try:
        v = float(det.get(key, np.nan))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def file_to_frame(path, label):
    rows = load_jsonl(path)
    out = []
    for i, r in enumerate(rows):
        det = r.get("detection_wm", {})
        out.append({
            "sample_id": record_id(r, i),
            "label": label,
            "micro": z(det, "z_score_micro"),
            "macro": z(det, "z_score_macro_calibrated"),
            "fisher": z(det, "z_score_fisher"),
            "words": len(str(r.get("wm_text", "")).split()),
        })
    return pd.DataFrame(out)


def bootstrap_mean_delta(delta, n_boot, seed):
    x = np.asarray(delta, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "n": 0,
            "mean_delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "positive_fraction": np.nan,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = x[idx].mean(axis=1)

    return {
        "n": len(x),
        "mean_delta": float(x.mean()),
        "median_delta": float(np.median(x)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "positive_fraction": float(np.mean(x > 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    records = Path(args.records_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kgw = records / "kgw_control.jsonl"
    k1 = records / "dual_kgw_random_k1.jsonl"
    iasw_candidates = sorted(records.glob("dual_kgw_iasw_k*_control.jsonl"))
    random_candidates = sorted(records.glob("dual_kgw_random_k*_seed*.jsonl"))

    # Exclude k1 from the random-K multi-seed glob just in case.
    random_candidates = [
        p for p in random_candidates
        if "random_k1" not in p.stem
    ]

    required = [kgw, k1]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    if not iasw_candidates:
        raise FileNotFoundError("IASW-K control file not found.")
    if not random_candidates:
        raise FileNotFoundError("Random-K multi-seed files not found.")

    iasw_path = iasw_candidates[0]

    frames = {
        "KGW": file_to_frame(kgw, "KGW"),
        "Random-K1": file_to_frame(k1, "Random-K1"),
        "IASW-K32": file_to_frame(iasw_path, "IASW-K32"),
    }

    random_frames = []
    for p in random_candidates:
        random_frames.append(file_to_frame(p, p.stem))

    # Build one aligned wide frame.
    base = frames["IASW-K32"].set_index("sample_id")[
        ["micro", "macro", "fisher", "words"]
    ].add_prefix("iasw_")

    for name in ["KGW", "Random-K1"]:
        f = frames[name].set_index("sample_id")[
            ["micro", "macro", "fisher", "words"]
        ]
        prefix = "kgw_" if name == "KGW" else "k1_"
        base = base.join(f.add_prefix(prefix), how="inner")

    random_cols = {"micro": [], "macro": [], "fisher": [], "words": []}

    for j, f in enumerate(random_frames):
        ff = f.set_index("sample_id")[["micro", "macro", "fisher", "words"]]
        for metric in random_cols:
            col = f"rand{j}_{metric}"
            base[col] = ff[metric]
            random_cols[metric].append(col)

    for metric, cols in random_cols.items():
        base[f"random_mean_{metric}"] = base[cols].mean(axis=1)
        base[f"random_std_{metric}"] = base[cols].std(axis=1, ddof=1)

    base = base.reset_index()

    # Planned comparisons.
    comparisons = [
        ("Random-K1 - KGW", "k1", "kgw"),
        ("Random-K32 - Random-K1", "random_mean", "k1"),
        ("IASW-K32 - Random-K32", "iasw", "random_mean"),
        ("IASW-K32 - KGW", "iasw", "kgw"),
    ]

    result_rows = []

    for label, lhs, rhs in comparisons:
        for metric in ["micro", "macro", "fisher", "words"]:
            delta = base[f"{lhs}_{metric}"] - base[f"{rhs}_{metric}"]
            stat = bootstrap_mean_delta(
                delta,
                n_boot=args.n_boot,
                seed=args.seed + len(result_rows),
            )
            result_rows.append({
                "comparison": label,
                "metric": metric,
                **stat,
            })

    results = pd.DataFrame(result_rows)

    base.to_csv(
        out / "paired_sample_level.csv",
        index=False,
        encoding="utf-8-sig",
    )
    results.to_csv(
        out / "paired_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Compact markdown report.
    focus = results[
        (results["comparison"] == "IASW-K32 - Random-K32")
        & (results["metric"].isin(["micro", "macro", "fisher", "words"]))
    ]

    lines = [
        "# Paired Random-vs-IAS-W bootstrap",
        "",
        f"Aligned samples: {len(base)}",
        f"Random runs: {len(random_frames)}",
        "",
        "## IASW-K32 minus mean(Random-K32)",
        "",
    ]

    for _, r in focus.iterrows():
        lines.append(
            f"- {r['metric']}: mean Δ={r['mean_delta']:.4f}, "
            f"95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}], "
            f"P(Δ>0) across samples={r['positive_fraction']:.3f}"
        )

    (out / "paired_bootstrap_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(results.to_string(index=False))
    print("\n[DONE]")
    print(out / "paired_bootstrap_summary.csv")
    print(out / "paired_bootstrap_report.md")


if __name__ == "__main__":
    main()
