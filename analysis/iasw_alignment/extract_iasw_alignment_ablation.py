# -*- coding: utf-8 -*-
"""
Extract and summarize IAS-W alignment-scoring ablation results.

Recommended location:
    analysis/iasw_alignment/extract_iasw_alignment_ablation.py

Recommended input:
    results/main_exp/c4/llama3.1-8b-instruct/records/iasw.jsonl

This script saves BOTH:
1) detailed/raw results for every individual attack;
2) grouped averages used by the paper table.

No information is discarded.

Outputs
-------
1. iasw_alignment_raw_scores.csv
   Per-sample z-scores:
       strict / window / dp_global / dp_local

2. iasw_alignment_metrics_by_attack.csv
   Long-format AUROC / TPR@1%FPR / TPR@5%FPR for Base and every attack.
   The legacy iasw_alignment_auc_by_attack.csv is also retained.

3. iasw_alignment_metrics_wide.csv
   Wide-format detailed three-metric table (plus legacy AUROC-only CSV):
       Strategy, Base, Typo, Drop, Swap, Del-Ins, Copy-Paste,
       PEGASUS, Back Translation, DIPPER

4. iasw_alignment_metrics_grouped.csv
   Paper-style grouped table for all three metrics (plus legacy AUROC-only CSV):
       Strategy, Base, Surface, Structural, Paraphrase, Translation, Avg.

5. iasw_alignment_metrics_grouped.tex
   One LaTeX file containing three separate grouped tables: AUROC, TPR@1%FPR,
   and TPR@5%FPR. The legacy filename is written as an identical alias.

6. iasw_alignment_metadata.json
   Input path, negative pool, attack grouping, and averaging rule.

Grouping
--------
Surface:
    typo, drop, swap

Structural:
    del_ins, copy_paste

Paraphrase:
    pegasus, dipper

Translation:
    back_translation

Avg.:
    arithmetic mean over all 8 retained individual attacks.
    It is NOT the mean of the four group means.

Default negative pool:
    human + no_wm

This matches the historical IAS alignment-analysis script and is useful when
comparing against the old Table 6. Use "--negative_sources human" if you want
human-only negatives instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


MODES = [
    "strict",
    "window",
    "dp_global",
    "dp_local",
]

MODE_LABELS = {
    "strict": "Strict",
    "window": "Window",
    "dp_global": "Global DP",
    "dp_local": "Local DP+SNR",
}

PAPER_ATTACK_ORDER = [
    "typo",
    "drop",
    "swap",
    "del_ins",
    "copy_paste",
    "pegasus",
    "back_translation",
    "dipper",
]

ATTACK_LABELS = {
    "typo": "Typo",
    "drop": "Drop",
    "swap": "Swap",
    "del_ins": "Del-Ins",
    "copy_paste": "Copy-Paste",
    "pegasus": "PEGASUS",
    "back_translation": "Back Translation",
    "dipper": "DIPPER",
}

ATTACK_GROUPS = {
    "Surface": [
        "typo",
        "drop",
        "swap",
    ],
    "Structural": [
        "del_ins",
        "copy_paste",
    ],
    "Paraphrase": [
        "pegasus",
        "dipper",
    ],
    "Translation": [
        "back_translation",
        ],
}

ALL_PAPER_ATTACKS = (
    ATTACK_GROUPS["Surface"]
    + ATTACK_GROUPS["Structural"]
    + ATTACK_GROUPS["Paraphrase"]
    + ATTACK_GROUPS["Translation"]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract detailed and grouped IAS-W alignment-ablation AUCs."
        )
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Existing IAS-W result JSONL after detection and attacks.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis/iasw/alignment_ablation",
    )

    parser.add_argument(
        "--negative_sources",
        nargs="+",
        choices=["human", "no_wm"],
        default=["human", "no_wm"],
        help=(
            "Negative pool used for AUC. Default keeps the historical "
            "IAS alignment-analysis convention."
        ),
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal places in the generated LaTeX table.",
    )

    parser.add_argument(
        "--latex_caption",
        type=str,
        default=(
            "Ablation of IAS-W alignment scoring strategies. "
            "Values are grouped AUC scores."
        ),
    )

    parser.add_argument(
        "--latex_label",
        type=str,
        default="tab:iasw_alignment_ablation",
    )

    return parser.parse_args()


def safe_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except Exception:
        return None

    if not np.isfinite(value):
        return None

    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as exc:
                print(
                    f"[WARN] Skip malformed JSON at line "
                    f"{line_idx}: {exc}"
                )
                continue

            if isinstance(row, dict):
                rows.append(row)

    return rows


def nested_get(obj: Any, keys: List[str]) -> Optional[Any]:
    cur = obj

    for key in keys:
        if not isinstance(cur, dict):
            return None

        if key not in cur:
            return None

        cur = cur[key]

    return cur


def recursive_find_ablation_scores(
    obj: Any,
) -> Optional[Dict[str, Any]]:
    """
    Compatibility fallback for slightly different result layouts.
    """
    if isinstance(obj, dict):
        value = obj.get("ablation_z_scores")

        if isinstance(value, dict):
            return value

        for child in obj.values():
            found = recursive_find_ablation_scores(child)

            if found is not None:
                return found

    elif isinstance(obj, list):
        for child in obj:
            found = recursive_find_ablation_scores(child)

            if found is not None:
                return found

    return None


def extract_alignment_scores(
    detection_result: Any,
) -> Optional[Dict[str, float]]:
    """
    Supported layouts include:

    single IAS-W:
        detection_result
          -> details
             -> ablation_z_scores

    dual KGW+IAS-W:
        detection_result
          -> details
             -> macro_details
                -> ablation_z_scores
    """
    if not isinstance(detection_result, dict):
        return None

    candidates = [
        detection_result.get("ablation_z_scores"),
        nested_get(
            detection_result,
            ["details", "ablation_z_scores"],
        ),
        nested_get(
            detection_result,
            [
                "details",
                "macro_details",
                "ablation_z_scores",
            ],
        ),
        nested_get(
            detection_result,
            [
                "details",
                "macro_details",
                "details",
                "ablation_z_scores",
            ],
        ),
        nested_get(
            detection_result,
            ["macro_details", "ablation_z_scores"],
        ),
    ]

    score_dict = None

    for candidate in candidates:
        if isinstance(candidate, dict):
            score_dict = candidate
            break

    if score_dict is None:
        score_dict = recursive_find_ablation_scores(
            detection_result
        )

    if not isinstance(score_dict, dict):
        return None

    output: Dict[str, float] = {}

    for mode in MODES:
        value = safe_float(
            score_dict.get(mode)
        )

        if value is None:
            return None

        output[mode] = value

    return output


def discover_attacks(
    rows: List[Dict[str, Any]],
) -> List[str]:
    found = set()

    for row in rows:
        attacks = row.get("attacks", {})

        if not isinstance(attacks, dict):
            continue

        for attack_name, attack_obj in attacks.items():
            if (
                isinstance(attack_obj, dict)
                and attack_obj.get("detection_results") is not None
            ):
                found.add(str(attack_name))

    ordered = [
        attack
        for attack in PAPER_ATTACK_ORDER
        if attack in found
    ]

    extras = sorted(
        found.difference(PAPER_ATTACK_ORDER)
    )

    return ordered + extras


def append_raw_row(
    output: List[Dict[str, Any]],
    sample_idx: int,
    label: int,
    source: str,
    condition: str,
    attack: str,
    detection_result: Any,
):
    scores = extract_alignment_scores(
        detection_result
    )

    if scores is None:
        return

    row = {
        "sample_idx": int(sample_idx),
        "label": int(label),
        "source": source,
        "condition": condition,
        "attack": attack,
    }

    row.update(scores)
    output.append(row)


def build_raw_score_table(
    rows: List[Dict[str, Any]],
    negative_sources: List[str],
    attack_names: List[str],
) -> pd.DataFrame:
    output: List[Dict[str, Any]] = []

    for sample_idx, row in enumerate(rows):
        # Negative: human
        if "human" in negative_sources:
            append_raw_row(
                output=output,
                sample_idx=sample_idx,
                label=0,
                source="human",
                condition="negative",
                attack="",
                detection_result=row.get(
                    "detection_human"
                ),
            )

        # Negative: no-watermark machine text
        if "no_wm" in negative_sources:
            append_raw_row(
                output=output,
                sample_idx=sample_idx,
                label=0,
                source="no_wm",
                condition="negative",
                attack="",
                detection_result=row.get(
                    "detection_no_wm"
                ),
            )

        # Positive: before attack
        append_raw_row(
            output=output,
            sample_idx=sample_idx,
            label=1,
            source="wm",
            condition="base",
            attack="",
            detection_result=row.get(
                "detection_wm"
            ),
        )

        attacks = row.get(
            "attacks",
            {},
        )

        if not isinstance(attacks, dict):
            continue

        # Positive: each attacked watermarked text
        for attack_name in attack_names:
            attack_obj = attacks.get(
                attack_name
            )

            if not isinstance(
                attack_obj,
                dict,
            ):
                continue

            append_raw_row(
                output=output,
                sample_idx=sample_idx,
                label=1,
                source="attacked_wm",
                condition=f"after_{attack_name}",
                attack=attack_name,
                detection_result=attack_obj.get(
                    "detection_results"
                ),
            )

    df = pd.DataFrame(output)

    if df.empty:
        raise RuntimeError(
            "No IAS-W ablation_z_scores were extracted."
        )

    return df


def auc_score(
    positives: np.ndarray,
    negatives: np.ndarray,
) -> float:
    positives = np.asarray(positives, dtype=float)
    negatives = np.asarray(negatives, dtype=float)
    positives = positives[np.isfinite(positives)]
    negatives = negatives[np.isfinite(negatives)]

    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    y_true = np.concatenate([
        np.zeros(len(negatives), dtype=int),
        np.ones(len(positives), dtype=int),
    ])
    y_score = np.concatenate([negatives, positives])

    if roc_auc_score is not None:
        return float(roc_auc_score(y_true, y_score))

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    n_neg = len(negatives)
    n_pos = len(positives)
    positive_rank_sum = float(np.sum(ranks[n_neg:]))
    u = positive_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def detection_metrics(
    positives: np.ndarray,
    negatives: np.ndarray,
) -> Dict[str, float]:
    """Compute AUROC and TPR at 1% / 5% FPR from the same score pools."""
    positives = np.asarray(positives, dtype=float)
    negatives = np.asarray(negatives, dtype=float)
    positives = positives[np.isfinite(positives)]
    negatives = negatives[np.isfinite(negatives)]

    out = {
        "AUROC": float("nan"),
        "TPR@1%FPR": float("nan"),
        "TPR@5%FPR": float("nan"),
    }
    if len(positives) == 0 or len(negatives) == 0:
        return out

    out["AUROC"] = auc_score(positives, negatives)
    try:
        from sklearn.metrics import roc_curve
        y_true = np.concatenate([
            np.zeros(len(negatives), dtype=int),
            np.ones(len(positives), dtype=int),
        ])
        y_score = np.concatenate([negatives, positives])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        for target, label in [
            (0.01, "TPR@1%FPR"),
            (0.05, "TPR@5%FPR"),
        ]:
            idx = np.where(fpr <= target)[0]
            out[label] = float(tpr[idx[-1]]) if len(idx) else 0.0
    except Exception:
        pass
    return out

def build_auc_by_attack(
    raw_df: pd.DataFrame,
    attack_names: List[str],
) -> pd.DataFrame:
    """
    One row = one strategy × one condition, with AUROC, TPR@1%FPR,
    and TPR@5%FPR computed from the same negative pool.
    """
    negative_df = raw_df[raw_df["label"] == 0]
    rows = []

    for mode in MODES:
        negatives = negative_df[mode].to_numpy(dtype=float)

        def append_condition(condition, attack, display_name, positives):
            metrics = detection_metrics(positives, negatives)
            rows.append({
                "mode": mode,
                "strategy": MODE_LABELS[mode],
                "condition": condition,
                "attack": attack,
                "display_name": display_name,
                **metrics,
                "n_pos": int(np.isfinite(positives).sum()),
                "n_neg": int(np.isfinite(negatives).sum()),
            })

        base_pos = raw_df[raw_df["condition"] == "base"][mode].to_numpy(dtype=float)
        append_condition("base", "", "Base", base_pos)

        for attack_name in attack_names:
            positives = raw_df[
                raw_df["condition"] == f"after_{attack_name}"
            ][mode].to_numpy(dtype=float)
            append_condition(
                f"after_{attack_name}",
                attack_name,
                ATTACK_LABELS.get(attack_name, attack_name),
                positives,
            )

    return pd.DataFrame(rows)

def build_wide_auc_table(
    auc_df: pd.DataFrame,
    attack_names: List[str],
) -> pd.DataFrame:
    """Detailed wide table, one row per strategy × metric."""
    display_columns = [
        "Base",
        *[ATTACK_LABELS.get(attack, attack) for attack in attack_names],
    ]
    rows = []
    for mode in MODES:
        sub = auc_df[auc_df["mode"] == mode]
        for metric in ["AUROC", "TPR@1%FPR", "TPR@5%FPR"]:
            row = {
                "mode": mode,
                "Strategy": MODE_LABELS[mode],
                "metric": metric,
            }
            for display_name in display_columns:
                hit = sub[sub["display_name"] == display_name]
                row[display_name] = (
                    float(hit.iloc[0][metric]) if len(hit) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)

def exact_group_mean(
    attack_auc: Dict[str, float],
    required_attacks: List[str],
) -> float:
    """
    Strict paper-table averaging.

    If any required attack is unavailable, return NaN rather than silently
    averaging a partial group.
    """
    values = []

    for attack in required_attacks:
        value = attack_auc.get(
            attack,
            np.nan,
        )

        if not np.isfinite(value):
            return float("nan")

        values.append(
            float(value)
        )

    return float(
        np.mean(values)
    )


def build_grouped_auc_table(
    auc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Grouped manuscript table for AUROC, TPR@1%FPR, and TPR@5%FPR."""
    rows = []
    for mode in MODES:
        sub = auc_df[auc_df["mode"] == mode]
        for metric in ["AUROC", "TPR@1%FPR", "TPR@5%FPR"]:
            base_hit = sub[sub["condition"] == "base"]
            base_value = (
                float(base_hit.iloc[0][metric]) if len(base_hit) else np.nan
            )
            attack_values = {}
            for _, row in sub.iterrows():
                attack_name = str(row.get("attack", ""))
                if attack_name:
                    attack_values[attack_name] = float(row[metric])

            rows.append({
                "mode": mode,
                "Strategy": MODE_LABELS[mode],
                "metric": metric,
                "Base": base_value,
                "Surface": exact_group_mean(attack_values, ATTACK_GROUPS["Surface"]),
                "Structural": exact_group_mean(attack_values, ATTACK_GROUPS["Structural"]),
                "Paraphrase": exact_group_mean(attack_values, ATTACK_GROUPS["Paraphrase"]),
                "Translation": exact_group_mean(attack_values, ATTACK_GROUPS["Translation"]),
                "Avg.": exact_group_mean(attack_values, ALL_PAPER_ATTACKS),
            })
    return pd.DataFrame(rows)

def latex_escape(text: str) -> str:
    return (
        text
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def save_grouped_latex(
    grouped_df: pd.DataFrame,
    path: Path,
    precision: int,
    caption: str,
    label: str,
):
    """Write three separate table environments into one .tex file."""
    columns = [
        "Base", "Surface", "Structural", "Paraphrase", "Translation", "Avg."
    ]
    blocks = []
    specs = [
        ("AUROC", "AUROC", "auroc"),
        ("TPR@1%FPR", "TPR@1\\%FPR", "tpr1"),
        ("TPR@5%FPR", "TPR@5\\%FPR", "tpr5"),
    ]

    for metric, metric_caption, suffix in specs:
        metric_df = grouped_df[grouped_df["metric"] == metric].copy()
        maxima = {}
        for col in columns:
            values = pd.to_numeric(metric_df[col], errors="coerce")
            maxima[col] = float(values.max()) if values.notna().any() else np.nan

        base_caption = latex_escape(caption)
        if "AUC" in base_caption or "AUROC" in base_caption:
            table_caption = (
                "Ablation of IAS-W alignment scoring strategies. "
                f"Values are grouped {metric_caption} scores."
            )
        else:
            table_caption = f"{base_caption} Metric: {metric_caption}."

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{table_caption}}}",
            rf"\label{{{latex_escape(label)}_{suffix}}}",
            r"\small",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Strategy & Base & Surface & Structural & Paraphrase & Translation & Avg. \\",
            r"\midrule",
        ]

        for _, row in metric_df.iterrows():
            cells = [latex_escape(str(row["Strategy"]))]
            for col in columns:
                value = safe_float(row[col])
                if value is None:
                    cells.append("--")
                    continue
                text = f"{value:.{precision}f}"
                maximum = maxima[col]
                if (
                    np.isfinite(maximum)
                    and np.isclose(value, maximum, rtol=0.0, atol=10 ** (-(precision + 2)))
                ):
                    text = r"\textbf{" + text + "}"
                cells.append(text)
            lines.append(" & ".join(cells) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        blocks.append("\n".join(lines))

    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

def main():
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"No valid JSONL rows found: {input_path}")

    attack_names = discover_attacks(rows)
    if not attack_names:
        raise RuntimeError("No attacks with detection_results were found.")

    print("=" * 90)
    print("IAS-W Alignment Ablation Extraction")
    print("=" * 90)
    print(f"input_file       = {input_path}")
    print(f"output_dir       = {output_dir}")
    print(f"negative_sources = {args.negative_sources}")
    print(f"attacks          = {attack_names}")
    print("=" * 90)

    raw_df = build_raw_score_table(
        rows=rows,
        negative_sources=args.negative_sources,
        attack_names=attack_names,
    )
    raw_path = output_dir / "iasw_alignment_raw_scores.csv"
    raw_df.to_csv(raw_path, index=False)

    metrics_df = build_auc_by_attack(raw_df=raw_df, attack_names=attack_names)
    metrics_long_path = output_dir / "iasw_alignment_metrics_by_attack.csv"
    metrics_df.to_csv(metrics_long_path, index=False)

    # Backward-compatible AUROC-only long file.
    auc_long_df = metrics_df.copy()
    auc_long_df["auc"] = auc_long_df["AUROC"]
    auc_long_path = output_dir / "iasw_alignment_auc_by_attack.csv"
    auc_long_df.to_csv(auc_long_path, index=False)

    wide_df = build_wide_auc_table(auc_df=metrics_df, attack_names=attack_names)
    metrics_wide_path = output_dir / "iasw_alignment_metrics_wide.csv"
    wide_df.to_csv(metrics_wide_path, index=False)
    auc_wide_df = wide_df[wide_df["metric"] == "AUROC"].drop(columns=["metric"])
    auc_wide_path = output_dir / "iasw_alignment_auc_wide.csv"
    auc_wide_df.to_csv(auc_wide_path, index=False)

    grouped_df = build_grouped_auc_table(metrics_df)
    grouped_metrics_path = output_dir / "iasw_alignment_metrics_grouped.csv"
    grouped_df.to_csv(grouped_metrics_path, index=False)
    auc_grouped_df = grouped_df[grouped_df["metric"] == "AUROC"].drop(columns=["metric"])
    grouped_auc_path = output_dir / "iasw_alignment_auc_grouped.csv"
    auc_grouped_df.to_csv(grouped_auc_path, index=False)

    latex_path = output_dir / "iasw_alignment_metrics_grouped.tex"
    save_grouped_latex(
        grouped_df=grouped_df,
        path=latex_path,
        precision=args.precision,
        caption=args.latex_caption,
        label=args.latex_label,
    )
    # Keep the historical filename as an identical multi-table alias.
    legacy_latex_path = output_dir / "iasw_alignment_auc_grouped.tex"
    legacy_latex_path.write_text(latex_path.read_text(encoding="utf-8"), encoding="utf-8")

    metadata = {
        "input_file": str(input_path),
        "negative_sources": list(args.negative_sources),
        "modes": MODES,
        "mode_labels": MODE_LABELS,
        "metrics": ["AUROC", "TPR@1%FPR", "TPR@5%FPR"],
        "discovered_attacks": attack_names,
        "paper_attack_groups": ATTACK_GROUPS,
        "paper_avg_attacks": ALL_PAPER_ATTACKS,
        "paper_avg_definition": (
            "Arithmetic mean over all individual attacks for each metric; "
            "not the mean of the four grouped means."
        ),
        "outputs": {
            "raw_scores": str(raw_path),
            "metrics_by_attack": str(metrics_long_path),
            "metrics_wide": str(metrics_wide_path),
            "metrics_grouped": str(grouped_metrics_path),
            "latex_grouped": str(latex_path),
            "legacy_auc_by_attack": str(auc_long_path),
            "legacy_auc_wide": str(auc_wide_path),
            "legacy_auc_grouped": str(grouped_auc_path),
            "legacy_latex": str(legacy_latex_path),
        },
    }
    metadata_path = output_dir / "iasw_alignment_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 90)
    print("Grouped manuscript metrics")
    print("=" * 90)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        print(grouped_df.drop(columns=["mode"]).to_string(index=False))

    missing_paper_attacks = [
        attack for attack in PAPER_ATTACK_ORDER if attack not in attack_names
    ]
    if missing_paper_attacks:
        print("\n[WARN] Missing attacks required by the paper grouping:")
        for attack in missing_paper_attacks:
            print(f"  - {attack}")
        print("Corresponding grouped values are NaN; no partial average is used.")

    print("\n[Saved]")
    print(f"  raw sample scores : {raw_path}")
    print(f"  metrics by attack : {metrics_long_path}")
    print(f"  metrics wide      : {metrics_wide_path}")
    print(f"  grouped metrics   : {grouped_metrics_path}")
    print(f"  grouped LaTeX     : {latex_path}")
    print(f"  metadata          : {metadata_path}")


if __name__ == "__main__":
    main()
