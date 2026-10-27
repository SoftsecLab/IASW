# src/evaluation/plot_roc.py
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc

def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROC Curves for Watermark Evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="包含所有检测结果的集成 JSONL 文件")
    parser.add_argument("--output_pdf", type=str, default="results/roc_combined.pdf", help="输出的高清 PDF 路径")
    parser.add_argument("--method", type=str, default="unknown_method")
    parser.add_argument(
        "--negative_source",
        type=str,
        choices=["auto", "human", "no_wm"],
        default="auto",
        help=(
            "ROC 负样本来源：auto=保持原逻辑，优先 detection_human，缺失时回退 "
            "detection_no_wm；human=仅使用 detection_human；"
            "no_wm=仅使用 detection_no_wm"
        ),
    )
    return parser.parse_args()

def parse_single_result(res: dict):
    if not res or "error" in res:
        return None

    has_explicit_micro = (
        "z_score_micro" in res
        or "micro" in res
    )

    has_explicit_macro = (
        "z_score_macro" in res
        or "z_score_macro_calibrated" in res
        or "macro" in res
    )

    # Standalone detector:
    # use its top-level final score as the single ROC channel.
    if (
        not has_explicit_micro
        and not has_explicit_macro
        and "z_score" in res
    ):
        score = float(res["z_score"])
        return score, 0.0, 0.0, None, None

    # Existing micro / macro / dual output.
    z_micro = res.get("z_score_micro", 0.0)

    if z_micro == 0.0 and "micro" in res:
        z_micro = (
            res.get("micro", {})
            .get("z_score", 0.0)
        )

    z_macro_raw = res.get(
        "z_score_macro",
        0.0,
    )

    if z_macro_raw == 0.0 and "macro" in res:
        z_macro_raw = (
            res.get("macro", {})
            .get("z_score", 0.0)
        )

    z_macro_calib = res.get(
        "z_score_macro_calibrated",
        z_macro_raw,
    )

    z_max = res.get(
        "z_score_max",
        None,
    )

    z_fisher = res.get(
        "z_score_fisher",
        None,
    )

    return (
        float(z_micro),
        float(z_macro_raw),
        float(z_macro_calib),
        z_max,
        z_fisher,
    )

def extract_all_scores(jsonl_file: str, negative_source: str = "auto"):
    """
    动态扫描文件内存在的所有攻击数据。

    negative_source:
    - auto: 保持原脚本逻辑，优先 detection_human，字段缺失或为空时回退 detection_no_wm
    - human: 仅使用 detection_human
    - no_wm: 仅使用 detection_no_wm
    """
    clean_scores = {"micro": [], "macro_raw": [], "macro_calib": [], "max": [], "fisher": []}
    wm_scores = {"micro": [], "macro_raw": [], "macro_calib": [], "max": [], "fisher": []}
    
    atk_scores_dict = {} # 动态装载攻击得分: {attack_name: {"micro": [], ...}}
    has_precomputed_fusion = True
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            if negative_source == "human":
                res_clean = row.get("detection_human", {})
            elif negative_source == "no_wm":
                res_clean = row.get("detection_no_wm", {})
            else:  # auto：严格保持原脚本行为
                res_clean = row.get("detection_human", {})
                if not res_clean:
                    res_clean = row.get("detection_no_wm", {})

            res_wm = row.get("detection_wm", {})
            
            def append_to(res, target_dict):
                nonlocal has_precomputed_fusion
                parsed = parse_single_result(res)
                if parsed:
                    z_mic, z_mac_raw, z_mac_calib, z_mx, z_fish = parsed
                    target_dict["micro"].append(z_mic)
                    target_dict["macro_raw"].append(z_mac_raw)
                    target_dict["macro_calib"].append(z_mac_calib)
                    if z_mx is not None: target_dict["max"].append(z_mx)
                    else: has_precomputed_fusion = False
                    if z_fish is not None: target_dict["fisher"].append(z_fish)
                    else: has_precomputed_fusion = False

            append_to(res_clean, clean_scores)
            append_to(res_wm, wm_scores)
            
            # 🌟 动态发现本行存在的所有攻击并提取检测结果
            attacks_dict = row.get("attacks", {})
            for atk_name, atk_data in attacks_dict.items():
                res_atk = atk_data.get("detection_results", {})
                if res_atk:
                    if atk_name not in atk_scores_dict:
                        atk_scores_dict[atk_name] = {"micro": [], "macro_raw": [], "macro_calib": [], "max": [], "fisher": []}
                    append_to(res_atk, atk_scores_dict[atk_name])

    def to_tuple(d):
        return (
            np.array(d["micro"]), np.array(d["macro_raw"]), np.array(d["macro_calib"]),
            np.array(d["max"]) if has_precomputed_fusion else None,
            np.array(d["fisher"]) if has_precomputed_fusion else None
        )
        
    atk_tuples = {atk: to_tuple(d) for atk, d in atk_scores_dict.items()}
    return to_tuple(clean_scores), to_tuple(wm_scores), atk_tuples

def get_tpr_at_fpr(fpr, tpr, stage_name, channel_name, report_dict):
    targets = [0.001, 0.01, 0.05]
    target_names = ["0.1%", "1%", "5%"]
    
    if channel_name not in report_dict: report_dict[channel_name] = {}
    report_dict[channel_name][stage_name] = {}
    
    results = []
    for target, name in zip(targets, target_names):
        valid_indices = np.where(fpr <= target)[0]
        best_tpr = tpr[valid_indices[-1]] if len(valid_indices) > 0 else 0.0
        results.append(f"TPR@{name}={best_tpr*100:.1f}%")
        report_dict[channel_name][stage_name][f"TPR@{name}FPR"] = float(best_tpr)
    print(f"    [{channel_name}] {stage_name:25s} -> " + " | ".join(results))

def calc_max_fusion(micro_scores, macro_scores): return np.maximum(micro_scores, macro_scores)

def calc_fisher_fusion(micro_scores, macro_scores):
    z_micro = np.clip(micro_scores, a_min=None, a_max=37.0)
    z_macro = np.clip(macro_scores, a_min=None, a_max=37.0)
    p_micro = np.clip(stats.norm.sf(z_micro), 1e-300, 1.0)
    p_macro = np.clip(stats.norm.sf(z_macro), 1e-300, 1.0)
    chi2_scores = -2 * np.log(p_micro) - 2 * np.log(p_macro)
    p_fisher = np.clip(stats.chi2.sf(chi2_scores, df=4), 1e-300, 1.0 - 1e-16)
    return stats.norm.isf(p_fisher)

def plot_roc_curves(clean_scores, wm_scores, atk_scores_dict, channel_name, ax, title, report_dict):
    if len(clean_scores) == 0 or len(wm_scores) == 0: return
    
    y_true_base = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(wm_scores))])
    fpr_base, tpr_base, _ = roc_curve(y_true_base, np.concatenate([clean_scores, wm_scores]))
    auc_base = auc(fpr_base, tpr_base)
    
    get_tpr_at_fpr(fpr_base, tpr_base, "Before Attack", channel_name, report_dict)
    report_dict[channel_name]["Before Attack"]["AUC"] = float(auc_base)
    
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=2) 
    ax.plot(fpr_base, tpr_base, color='black', lw=2.5, linestyle='-', label=f'Base (AUC = {auc_base:.3f})')
            
    cmap = plt.get_cmap("tab10")
    for idx, (atk_name, atk_scores) in enumerate(atk_scores_dict.items()):
        if len(atk_scores) == 0: continue
              
        clean_arr = np.asarray(clean_scores, dtype=float)
        atk_arr = np.asarray(atk_scores, dtype=float)

        clean_valid = np.isfinite(clean_arr)
        atk_valid = np.isfinite(atk_arr)

        bad_clean = np.where(~clean_valid)[0]
        bad_atk = np.where(~atk_valid)[0]

        if len(bad_clean) > 0 or len(bad_atk) > 0:
            print(
                f"    [NaN AUDIT] {channel_name} / {atk_name}: "
                f"clean_invalid={len(bad_clean)}/{len(clean_arr)}, "
                f"attack_invalid={len(bad_atk)}/{len(atk_arr)}"
            )
            print("        bad clean idx:", bad_clean[:20].tolist())
            print("        bad attack idx:", bad_atk[:20].tolist())

        # 只保留有限值
        clean_use = clean_arr[clean_valid]
        atk_use = atk_arr[atk_valid]

        # 防止整个攻击都不可评分
        if len(clean_use) == 0 or len(atk_use) == 0:
            print(
                f"    [WARN] {channel_name} / {atk_name}: "
                f"没有足够的有效 score，跳过该攻击。"
            )
            continue

        # 注意：过滤 score 后，label 也必须重新构造
        y_true_atk = np.concatenate([
            np.zeros(len(clean_use), dtype=int),
            np.ones(len(atk_use), dtype=int),
        ])

        scores_atk = np.concatenate([clean_use, atk_use])

        fpr_atk, tpr_atk, _ = roc_curve(y_true_atk, scores_atk)
        auc_atk = auc(fpr_atk, tpr_atk)
        stage_name = f"After {atk_name.capitalize()}"
        get_tpr_at_fpr(fpr_atk, tpr_atk, stage_name, channel_name, report_dict)
        report_dict[channel_name][stage_name]["AUC"] = float(auc_atk)
        
        ax.plot(fpr_atk, tpr_atk, color=cmap(idx % 10), lw=2, linestyle='--', alpha=0.85,
                label=f'{atk_name.capitalize()} (AUC = {auc_atk:.3f})')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_macro_dual_roc_curves(clean_raw, wm_raw, clean_calib, wm_calib, atk_calib_dict, channel_name, ax, title, report_dict):
    if len(clean_raw) == 0 or len(wm_raw) == 0: return
    
    y_true_base = np.concatenate([np.zeros(len(clean_raw)), np.ones(len(wm_raw))])
    
    fpr_raw_base, tpr_raw_base, _ = roc_curve(y_true_base, np.concatenate([clean_raw, wm_raw]))
    auc_raw_base = auc(fpr_raw_base, tpr_raw_base)
    
    fpr_calib_base, tpr_calib_base, _ = roc_curve(y_true_base, np.concatenate([clean_calib, wm_calib]))
    auc_calib_base = auc(fpr_calib_base, tpr_calib_base)

    get_tpr_at_fpr(fpr_calib_base, tpr_calib_base, "Before Attack", channel_name, report_dict)
    report_dict[channel_name]["Before Attack"]["AUC"] = float(auc_calib_base)
    
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=2) 
    ax.plot(fpr_raw_base, tpr_raw_base, color='gray', lw=2, linestyle='-.', alpha=0.6, label=f'RAW Base (AUC = {auc_raw_base:.3f})')
    ax.plot(fpr_calib_base, tpr_calib_base, color='black', lw=2.5, linestyle='-', label=f'CALIB Base (AUC = {auc_calib_base:.3f})')
            
    cmap = plt.get_cmap("tab10")
    for idx, (atk_name, atk_calib) in enumerate(atk_calib_dict.items()):
        if len(atk_calib) == 0: continue
        
        y_true_atk = np.concatenate([np.zeros(len(clean_raw)), np.ones(len(atk_calib))])
        fpr_calib_atk, tpr_calib_atk, _ = roc_curve(y_true_atk, np.concatenate([clean_calib, atk_calib]))
        auc_calib_atk = auc(fpr_calib_atk, tpr_calib_atk)
        
        stage_name = f"After {atk_name.capitalize()}"
        get_tpr_at_fpr(fpr_calib_atk, tpr_calib_atk, stage_name, channel_name, report_dict)
        report_dict[channel_name][stage_name]["AUC"] = float(auc_calib_atk)
        
        ax.plot(fpr_calib_atk, tpr_calib_atk, color=cmap(idx % 10), lw=2, linestyle='--', alpha=0.85,
                label=f'{atk_name.capitalize()} (AUC = {auc_calib_atk:.3f})')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)
    
    print(f"\n[*] 正在读取 [{args.method.upper()}] 的数据绘制 ROC 全家桶对比图...")
    print(f"[*] 负样本来源: {args.negative_source}")
    
    clean_scores, wm_scores, atk_tuples = extract_all_scores(
        args.input_file,
        negative_source=args.negative_source,
    )
    clean_micro, clean_raw_macro, clean_calib_macro, clean_max, clean_fisher = clean_scores
    wm_micro, wm_raw_macro, wm_calib_macro, wm_max, wm_fisher = wm_scores
    
    # 动态拆解所有攻击的分数
    atk_micro, atk_raw_macro, atk_calib_macro, atk_max, atk_fisher = {}, {}, {}, {}, {}
    for atk, scores in atk_tuples.items():
        atk_micro[atk], atk_raw_macro[atk], atk_calib_macro[atk], atk_max[atk], atk_fisher[atk] = scores
    
    # valid_micro = np.var(clean_micro) > 1e-4 or np.var(wm_micro) > 1e-4
    # valid_macro = np.var(clean_raw_macro) > 1e-4 or np.var(wm_raw_macro) > 1e-4

    def is_valid_channel(clean_arr, wm_arr):
        if len(clean_arr) == 0 or len(wm_arr) == 0:
            return False
        all_scores = np.concatenate([clean_arr, wm_arr])
        return not np.allclose(all_scores, 0.0)

    valid_micro = is_valid_channel(clean_micro, wm_micro)
    valid_macro = is_valid_channel(clean_raw_macro, wm_raw_macro)
        
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    metrics_report = {"method": args.method, "evaluated_attacks": list(atk_tuples.keys())}
    
    if len(atk_tuples) == 0:
        print("[!] 提示: 文件中未检测到任何攻击数据。ROC曲线将仅绘制攻击前的 Base 表现。")
    else:
        print(f"[+] 检测到以下 {len(atk_tuples)} 种攻击数据: {', '.join(atk_tuples.keys())}")
        
    if "dual" in args.method.lower() and valid_micro and valid_macro:
        print("[+] 检测到 Dual-Channel 方法，正在绘制 4 子图大盘...")
        
        # 智能回退计算
        if clean_max is None or wm_max is None:
            clean_max = calc_max_fusion(clean_micro, clean_calib_macro)
            wm_max = calc_max_fusion(wm_micro, wm_calib_macro)
            clean_fisher = calc_fisher_fusion(clean_micro, clean_calib_macro)
            wm_fisher = calc_fisher_fusion(wm_micro, wm_calib_macro)
            
            for atk in atk_tuples.keys():
                if len(atk_micro[atk]) > 0:
                    atk_max[atk] = calc_max_fusion(atk_micro[atk], atk_calib_macro[atk])
                    atk_fisher[atk] = calc_fisher_fusion(atk_micro[atk], atk_calib_macro[atk])
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        print("\n--- Micro Channel (微观通道) ---")
        plot_roc_curves(clean_micro, wm_micro, atk_micro, "Micro", axes[0, 0], f"[DUAL] Micro Channel", metrics_report)
        print("\n--- Macro Channel (宏观通道) ---")
        plot_macro_dual_roc_curves(clean_raw_macro, wm_raw_macro, clean_calib_macro, wm_calib_macro, atk_calib_macro, "Macro", axes[0, 1], f"[DUAL] Macro Channel (Calibrated)", metrics_report)
        print("\n--- Max Fusion (最大值融合) ---")
        plot_roc_curves(clean_max, wm_max, atk_max, "Max Fusion", axes[1, 0], f"[DUAL] Max Fusion", metrics_report)
        print("\n--- Fisher's Method Fusion (费雪融合) ---")
        plot_roc_curves(clean_fisher, wm_fisher, atk_fisher, "Fisher Fusion", axes[1, 1], f"[DUAL] Fisher Fusion", metrics_report)

    elif valid_micro and valid_macro:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        print("\n--- Micro Channel ---")
        plot_roc_curves(clean_micro, wm_micro, atk_micro, "Micro", axes[0], f"[{args.method.upper()}] Micro Channel", metrics_report)
        print("\n--- Macro Channel ---")
        plot_macro_dual_roc_curves(clean_raw_macro, wm_raw_macro, clean_calib_macro, wm_calib_macro, atk_calib_macro, "Macro", axes[1], f"[{args.method.upper()}] Macro Channel", metrics_report)
        
    elif valid_macro:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        print(f"\n--- Macro Only ---")
        plot_macro_dual_roc_curves(clean_raw_macro, wm_raw_macro, clean_calib_macro, wm_calib_macro, atk_calib_macro, "Macro", ax, f"[{args.method.upper()}] ROC Curve", metrics_report)
        
    elif valid_micro:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        print(f"\n--- Micro Only ---")
        plot_roc_curves(clean_micro, wm_micro, atk_micro, "Micro", ax, f"[{args.method.upper()}] ROC Curve", metrics_report)
        
    else:
        print("[!] 错误：未检测到任何有效的 ROC 绘图数据！")
        return

    plt.tight_layout()
    plt.savefig(args.output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    
    json_output_path = args.output_pdf.replace(".pdf", "_metrics.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=4, ensure_ascii=False)
        
    print(f"\n[+] 分析完毕！高质量聚合图表已保存至: {args.output_pdf}")

if __name__ == "__main__":
    main()