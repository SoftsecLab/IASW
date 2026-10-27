# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def extract_values(file_path, key="ppl_wm"):
    """从 detailed_quality_metrics 文件中提取 values"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if key not in data:
        return []

    obj = data[key]

    if isinstance(obj, dict):
        for k in ["values", "scores", "raw", "raw_values"]:
            if k in obj and isinstance(obj[k], list):
                return [float(x) for x in obj[k] if x is not None]
    elif isinstance(obj, list):
        return [float(x) for x in obj if x is not None]

    return []


def plot_distribution(method_values_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")

    # =========================
    # 1. KDE + Histogram
    # =========================
    plt.figure(figsize=(10, 7))

    for method, values in method_values_dict.items():
        if len(values) == 0:
            continue

        sns.kdeplot(values, label=method, fill=True, alpha=0.3)

    plt.xlabel("Perplexity (PPL)")
    plt.ylabel("Density")
    plt.title("PPL Distribution (KDE)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "ppl_kde.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] KDE 图已保存: {out_path}")

    # =========================
    # 2. Boxplot
    # =========================
    plt.figure(figsize=(10, 6))

    data = []
    labels = []

    for method, values in method_values_dict.items():
        if len(values) == 0:
            continue
        data.append(values)
        labels.append(method)

    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xticks(rotation=30)
    plt.ylabel("PPL")
    plt.title("PPL Boxplot")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "ppl_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Boxplot 已保存: {out_path}")

    # =========================
    # 3. CDF 曲线
    # =========================
    plt.figure(figsize=(10, 7))

    for method, values in method_values_dict.items():
        if len(values) == 0:
            continue

        sorted_vals = np.sort(values)
        cdf = np.arange(len(sorted_vals)) / float(len(sorted_vals))

        plt.plot(sorted_vals, cdf, label=method)

    plt.xlabel("PPL")
    plt.ylabel("CDF")
    plt.title("PPL Cumulative Distribution")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "ppl_cdf.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] CDF 图已保存: {out_path}")


def main(exp_data_dir, output_dir):
    method_values = {}

    for file in os.listdir(exp_data_dir):
        if not file.startswith("detailed_quality_metrics"):
            continue

        method = file.replace("detailed_quality_metrics_", "").replace(".json", "")
        path = os.path.join(exp_data_dir, file)

        values = extract_values(path, key="ppl_wm")

        if len(values) == 0:
            print(f"[!] {method} 没有 values，跳过")
            continue

        method_values[method] = values
        print(f"[+] {method}: {len(values)} samples")

    if not method_values:
        print("[!] 没有可用数据")
        return

    plot_distribution(method_values, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_data_dir", type=str, required=True,
                        help="detailed_quality_metrics 所在目录")
    parser.add_argument("--output_dir", type=str, default="plots_quality",
                        help="输出目录")

    args = parser.parse_args()

    main(args.exp_data_dir, args.output_dir)