# src/evaluation/plot_ablation_z_dist.py
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot 4-Mode Ablation Z-Score Distribution (KDE)")
    parser.add_argument("--input_file", type=str, required=True, help="包含所有检测结果的集成 JSONL 文件")
    parser.add_argument("--output_pdf", type=str, default="results/ablation_z_dist.pdf")
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
    从同一个集成 JSONL 文件中同时提取：
    - clean 分数（优先 detection_human，否则 detection_no_wm）
    - wm 分数（detection_wm）
    """
    clean_scores = {"strict": [], "window": [], "dp_global": [], "dp_local": []}
    wm_scores = {"strict": [], "window": [], "dp_global": [], "dp_local": []}

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

    return (
        {k: np.array(v) for k, v in clean_scores.items()},
        {k: np.array(v) for k, v in wm_scores.items()},
    )


def plot_single_mode_kde(clean_scores, wm_scores, ax, title):
    if len(clean_scores) == 0 or len(wm_scores) == 0:
        ax.set_title(f"{title} (No Data)")
        ax.grid(True, linestyle="--", alpha=0.5)
        return

    sns.kdeplot(clean_scores, fill=True, color="skyblue", label="Clean (No Watermark)", ax=ax, linewidth=2)
    sns.kdeplot(wm_scores, fill=True, color="forestgreen", label="Watermarked (IAS)", ax=ax, linewidth=2)

    all_scores = np.concatenate([clean_scores, wm_scores])
    x_min = max(-5, np.percentile(all_scores, 1) - 2)
    x_max = min(25, np.percentile(all_scores, 99) + 3)

    x = np.linspace(x_min, x_max, 1000)
    theoretical_y = (1 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)
    ax.plot(x, theoretical_y, color="black", linestyle=":", label="Theoretical $\mathcal{N}(0,1)$", linewidth=2.5)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(4.0, color="red", linestyle="-.", linewidth=1.5, label="Threshold (Z=4.0)")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Z-Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim([x_min, x_max])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)

    print(f"[*] 正在读取 [{args.method.upper()}] 的数据用于绘制 4 模式消融 Z-score 分布图...")
    print(f"[*] 数据来源文件: {args.input_file}")

    clean_data, wm_data = extract_all_ablation_scores(args.input_file)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    title_map = {
        "strict": "1. Strict Alignment",
        "window": "2. Sliding Window",
        "dp_global": "3. DP Global Alignment",
        "dp_local": "4. DP Local Alignment"
    }

    modes = ["strict", "window", "dp_global", "dp_local"]
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for mode, ax in zip(modes, ax_list):
        plot_single_mode_kde(clean_data[mode], wm_data[mode], ax, title_map[mode])

    plt.tight_layout()
    plt.savefig(args.output_pdf, format="pdf", dpi=300, bbox_inches="tight")
    print(f"[+] 4 模式 Z-score 分布图绘制完毕！已保存至: {args.output_pdf}")


if __name__ == "__main__":
    main()