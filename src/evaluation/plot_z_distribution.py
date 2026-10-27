# src/evaluation/plot_z_distribution.py
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Z-Score Distribution (KDE)")
    # 🌟 核心修改 1：统一使用单文件输入
    parser.add_argument("--input_file", type=str, required=True, help="包含所有检测结果的集成 JSONL 文件")
    parser.add_argument("--output_pdf", type=str, required=True, help="输出的高清 PDF 路径")
    parser.add_argument("--method", type=str, default="unknown_method")
    return parser.parse_args()

def parse_single_result(res: dict):
    """提取单个结果的各个维度 Z-Score"""
    if not res or "error" in res:
        return None
        
    z_micro = res.get("z_score_micro", 0.0)
    if z_micro == 0.0 and "micro" in res:
        z_micro = res.get("micro", {}).get("z_score", 0.0)
        
    z_macro_raw = res.get("z_score_macro", 0.0)
    if z_macro_raw == 0.0 and "macro" in res:
        z_macro_raw = res.get("macro", {}).get("z_score", 0.0)
        
    # 分布图展示的是最终用于决策的标定后宏观分数
    z_macro_calib = res.get("z_score_macro_calibrated", z_macro_raw)
    
    z_max = res.get("z_score_max", None)
    z_fisher = res.get("z_score_fisher", None)
    
    return z_micro, z_macro_calib, z_max, z_fisher

def extract_all_scores(jsonl_file: str):
    """🌟 核心修改 2：从同一文件中同时提取 Clean 和 WM 的得分"""
    clean_scores = {"micro": [], "macro": [], "max": [], "fisher": []}
    wm_scores = {"micro": [], "macro": [], "max": [], "fisher": []}
    
    has_precomputed_fusion = True
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            # 使用 human_text 作为负样本，没有则用 no_wm_text
            res_clean = row.get("detection_human", {})
            if not res_clean: 
                res_clean = row.get("detection_no_wm", {})
                
            res_wm = row.get("detection_wm", {})
            
            def append_to(res, target_dict):
                nonlocal has_precomputed_fusion
                parsed = parse_single_result(res)
                if parsed:
                    z_mic, z_mac, z_mx, z_fish = parsed
                    target_dict["micro"].append(z_mic)
                    target_dict["macro"].append(z_mac)
                    if z_mx is not None: target_dict["max"].append(z_mx)
                    else: has_precomputed_fusion = False
                    if z_fish is not None: target_dict["fisher"].append(z_fish)
                    else: has_precomputed_fusion = False

            append_to(res_clean, clean_scores)
            append_to(res_wm, wm_scores)

    def to_tuple(d):
        return (
            np.array(d["micro"]), np.array(d["macro"]),
            np.array(d["max"]) if has_precomputed_fusion else None,
            np.array(d["fisher"]) if has_precomputed_fusion else None
        )
        
    return to_tuple(clean_scores), to_tuple(wm_scores)

def plot_kde(clean_scores, wm_scores, ax, title, color_clean, color_wm):
    """在指定的子图上绘制 KDE 核密度估计曲线"""
    if len(clean_scores) == 0 or len(wm_scores) == 0:
        ax.set_title(f"{title} (No Data)")
        return
        
    sns.kdeplot(clean_scores, fill=True, color=color_clean, label="Clean (No Watermark)", ax=ax, linewidth=2)
    sns.kdeplot(wm_scores, fill=True, color=color_wm, label="Watermarked", ax=ax, linewidth=2)
    
    # 绘制理论标准正态分布 N(0,1)
    all_scores = np.concatenate([clean_scores, wm_scores])
    x_min = max(-5, np.percentile(all_scores, 1) - 2) 
    x_max = min(25, np.percentile(all_scores, 99) + 3)
    
    x = np.linspace(x_min, x_max, 1000)
    theoretical_y = (1/math.sqrt(2*math.pi)) * np.exp(-0.5 * x**2)
    ax.plot(x, theoretical_y, color='black', linestyle=':', label="Theoretical $\mathcal{N}(0,1)$", linewidth=2.5)
    
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(4.0, color='red', linestyle='-.', linewidth=1.5, label="Threshold (Z=4.0)")
    
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel("Z-Score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_xlim([x_min, x_max])
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

def calc_max_fusion(micro_scores, macro_scores):
    return np.maximum(micro_scores, macro_scores)

def calc_fisher_fusion(micro_scores, macro_scores):
    z_micro = np.clip(micro_scores, a_min=None, a_max=37.0)
    z_macro = np.clip(macro_scores, a_min=None, a_max=37.0)
    p_micro = np.clip(stats.norm.sf(z_micro), 1e-300, 1.0)
    p_macro = np.clip(stats.norm.sf(z_macro), 1e-300, 1.0)
    chi2_scores = -2 * np.log(p_micro) - 2 * np.log(p_macro)
    p_fisher = np.clip(stats.chi2.sf(chi2_scores, df=4), 1e-300, 1.0 - 1e-16)
    return stats.norm.isf(p_fisher)

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)
    
    print(f"[*] 正在读取 [{args.method.upper()}] 的数据用于绘制 Z-score 分布图...")
    print(f"[*] 数据来源文件: {args.input_file}")
    
    clean_scores, wm_scores = extract_all_scores(args.input_file)
    clean_micro, clean_macro, clean_max, clean_fisher = clean_scores
    wm_micro, wm_macro, wm_max, wm_fisher = wm_scores
    
    valid_micro = np.var(clean_micro) > 1e-4 or np.var(wm_micro) > 1e-4
    valid_macro = np.var(clean_macro) > 1e-4 or np.var(wm_macro) > 1e-4

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    if "dual" in args.method.lower() and valid_micro and valid_macro:
        print(f"[*] 检测到 Dual-Channel 方法，启动双通道融合机制 (Max & Fisher) ...")
        if clean_max is None or wm_max is None:
            clean_max = calc_max_fusion(clean_micro, clean_macro)
            wm_max = calc_max_fusion(wm_micro, wm_macro)
            clean_fisher = calc_fisher_fusion(clean_micro, clean_macro)
            wm_fisher = calc_fisher_fusion(wm_micro, wm_macro)
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plot_kde(clean_micro, wm_micro, axes[0, 0], f"[{args.method.upper()}] Micro Channel", 'skyblue', 'navy')
        plot_kde(clean_macro, wm_macro, axes[0, 1], f"[{args.method.upper()}] Macro Channel (Calibrated)", 'lightgreen', 'forestgreen')
        plot_kde(clean_max, wm_max, axes[1, 0], f"[{args.method.upper()}] Max Fusion", 'plum', 'purple')
        plot_kde(clean_fisher, wm_fisher, axes[1, 1], f"[{args.method.upper()}] Fisher Fusion", 'paleturquoise', 'teal')

    elif valid_micro and valid_macro:
        print(f"[*] 检测到普通双通道数据，正在绘制 1x2 双子图...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        plot_kde(clean_micro, wm_micro, axes[0], f"[{args.method.upper()}] Micro Channel", 'skyblue', 'navy')
        plot_kde(clean_macro, wm_macro, axes[1], f"[{args.method.upper()}] Macro Channel (Calibrated)", 'lightgreen', 'forestgreen')
        
    elif valid_macro:
        print(f"[*] 检测到单通道(宏观)数据，正在绘制 1x1 单图...")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_kde(clean_macro, wm_macro, ax, f"[{args.method.upper()}] Macro Channel", 'lightgreen', 'forestgreen')
        
    elif valid_micro:
        print(f"[*] 检测到单通道(微观)数据，正在绘制 1x1 单图...")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_kde(clean_micro, wm_micro, ax, f"[{args.method.upper()}] Micro Channel", 'skyblue', 'crimson')
        
    else:
        print("[!] 错误：未检测到任何有效的分布数据！")
        return

    plt.tight_layout()
    plt.savefig(args.output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    print(f"[+] Z-score 分布图绘制完毕！已保存至: {args.output_pdf}")

if __name__ == "__main__":
    main()