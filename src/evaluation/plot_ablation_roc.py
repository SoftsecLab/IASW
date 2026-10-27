# src/evaluation/plot_ablation_roc.py
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 4-Mode Ablation ROC Curves for IAS-Mark (Multi-Attack)"
    )
    parser.add_argument("--input_file", type=str, required=True, help="包含所有检测结果的集成 JSONL 文件")
    parser.add_argument("--output_pdf", type=str, default="results/ablation_roc_comparison.pdf")
    parser.add_argument("--method", type=str, default="ias")
    return parser.parse_args()


def extract_ablation_scores_from_result(res: dict):
    """
    从单个 detection result 中提取 4 种消融模式分数
    """
    if not res or "error" in res:
        return None

    ablation_dict = res.get("ablation_z_scores", {})
    if not ablation_dict and "details" in res:
        ablation_dict = res["details"].get("ablation_z_scores", {})

    if not ablation_dict:
        return None

    return {
        "strict": ablation_dict.get("strict", None),
        "window": ablation_dict.get("window", None),
        "dp_global": ablation_dict.get("dp_global", None),
        "dp_local": ablation_dict.get("dp_local", None),
    }


def extract_all_ablation_scores(jsonl_file: str):
    """
    从同一个集成 JSONL 文件中提取：
    - clean 分数（detection_human 优先，否则 detection_no_wm）
    - wm 分数（detection_wm）
    - 所有攻击的 attacked 分数（attacks[*].detection_results）
    """
    clean_scores = {"strict": [], "window": [], "dp_global": [], "dp_local": []}
    wm_scores = {"strict": [], "window": [], "dp_global": [], "dp_local": []}
    atk_scores_dict = {}

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            res_clean = row.get("detection_human", {})
            if not res_clean:
                res_clean = row.get("detection_no_wm", {})

            res_wm = row.get("detection_wm", {})

            parsed_clean = extract_ablation_scores_from_result(res_clean)
            parsed_wm = extract_ablation_scores_from_result(res_wm)

            if parsed_clean:
                for mode, val in parsed_clean.items():
                    if val is not None:
                        clean_scores[mode].append(val)

            if parsed_wm:
                for mode, val in parsed_wm.items():
                    if val is not None:
                        wm_scores[mode].append(val)

            attacks_dict = row.get("attacks", {})
            for atk_name, atk_data in attacks_dict.items():
                res_atk = atk_data.get("detection_results", {})
                parsed_atk = extract_ablation_scores_from_result(res_atk)

                if parsed_atk:
                    if atk_name not in atk_scores_dict:
                        atk_scores_dict[atk_name] = {
                            "strict": [],
                            "window": [],
                            "dp_global": [],
                            "dp_local": [],
                        }

                    for mode, val in parsed_atk.items():
                        if val is not None:
                            atk_scores_dict[atk_name][mode].append(val)

    clean_scores = {k: np.array(v) for k, v in clean_scores.items()}
    wm_scores = {k: np.array(v) for k, v in wm_scores.items()}
    atk_scores_dict = {
        atk: {mode: np.array(vals) for mode, vals in mode_dict.items()}
        for atk, mode_dict in atk_scores_dict.items()
    }

    return clean_scores, wm_scores, atk_scores_dict


def get_tpr_at_fpr(fpr, tpr, stage_name, mode_name, report_dict):
    targets = [0.001, 0.01, 0.05]
    target_names = ["0.1%", "1%", "5%"]

    if mode_name not in report_dict:
        report_dict[mode_name] = {}
    report_dict[mode_name][stage_name] = {}

    results = []
    for target, name in zip(targets, target_names):
        valid_indices = np.where(fpr <= target)[0]
        best_tpr = tpr[valid_indices[-1]] if len(valid_indices) > 0 else 0.0
        results.append(f"TPR@{name}FPR={best_tpr*100:.1f}%")
        report_dict[mode_name][stage_name][f"TPR@{name}FPR"] = float(best_tpr)

    print(f"    [{mode_name:10s}] {stage_name:20s} -> " + " | ".join(results))


def plot_single_mode_roc_multi_attack(
    clean_scores,
    wm_scores,
    atk_scores_dict,
    mode_name,
    ax,
    report_dict,
):
    """
    单个子图 = 单个消融模式
    图中画：
    - Base
    - 所有攻击
    """
    if len(clean_scores) == 0 or len(wm_scores) == 0:
        ax.set_title(f"{mode_name} (No Data)")
        ax.grid(True, linestyle="--", alpha=0.6)
        return

    # Base
    y_true_base = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(wm_scores))])
    y_scores_base = np.concatenate([clean_scores, wm_scores])
    fpr_base, tpr_base, _ = roc_curve(y_true_base, y_scores_base)
    auc_base = auc(fpr_base, tpr_base)

    get_tpr_at_fpr(fpr_base, tpr_base, "Before Attack", mode_name, report_dict)
    report_dict[mode_name]["Before Attack"]["AUC"] = float(auc_base)

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=2)
    ax.plot(
        fpr_base,
        tpr_base,
        color="black",
        lw=2.5,
        linestyle="-",
        label=f"Base (AUC = {auc_base:.3f})"
    )

    # 所有攻击
    cmap = plt.get_cmap("tab10")
    for idx, (atk_name, atk_mode_scores) in enumerate(atk_scores_dict.items()):
        attacked_scores = atk_mode_scores.get(mode_name, np.array([]))
        if len(attacked_scores) == 0:
            continue

        y_true_atk = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(attacked_scores))])
        y_scores_atk = np.concatenate([clean_scores, attacked_scores])
        fpr_atk, tpr_atk, _ = roc_curve(y_true_atk, y_scores_atk)
        auc_atk = auc(fpr_atk, tpr_atk)

        stage_name = f"After {atk_name}"
        get_tpr_at_fpr(fpr_atk, tpr_atk, stage_name, mode_name, report_dict)
        report_dict[mode_name][stage_name]["AUC"] = float(auc_atk)

        ax.plot(
            fpr_atk,
            tpr_atk,
            color=cmap(idx % 10),
            lw=2,
            linestyle="--",
            alpha=0.9,
            label=f"{atk_name} (AUC = {auc_atk:.3f})"
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12, fontweight="bold")

    title_map = {
        "strict": "1. Strict Alignment",
        "window": "2. Sliding Window",
        "dp_global": "3. DP Global Alignment",
        "dp_local": "4. DP Local Alignment",
    }
    ax.set_title(title_map.get(mode_name, mode_name), fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)

    print(f"\n[*] 正在读取 [{args.method.upper()}] 的数据，准备绘制 4 模式消融 ROC 曲线（多攻击）...")
    print(f"[*] 数据来源文件: {args.input_file}")

    clean_data, wm_data, atk_data = extract_all_ablation_scores(args.input_file)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    metrics_report = {
        "method": args.method,
        "evaluated_attacks": list(atk_data.keys()),
    }

    if len(atk_data) == 0:
        print("[!] 提示：文件中未检测到任何攻击数据。图中只会绘制 Base 曲线。")
    else:
        print(f"[+] 检测到以下 {len(atk_data)} 种攻击数据: {', '.join(atk_data.keys())}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    modes = [
        ("strict", axes[0, 0]),
        ("window", axes[0, 1]),
        ("dp_global", axes[1, 0]),
        ("dp_local", axes[1, 1]),
    ]

    for mode, ax in modes:
        print(f"\n--- Processing Mode: {mode.upper()} ---")
        plot_single_mode_roc_multi_attack(
            clean_scores=clean_data[mode],
            wm_scores=wm_data[mode],
            atk_scores_dict=atk_data,
            mode_name=mode,
            ax=ax,
            report_dict=metrics_report,
        )

    plt.tight_layout()
    plt.savefig(args.output_pdf, format="pdf", dpi=300, bbox_inches="tight")

    json_output_path = args.output_pdf.replace(".pdf", "_ablation_metrics.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4, ensure_ascii=False)

    print(f"\n[+] 分析完毕！")
    print(f"    - 4 模式多攻击对比图已保存至: {args.output_pdf}")
    print(f"    - 核心指标已保存至: {json_output_path}\n")


if __name__ == "__main__":
    main()