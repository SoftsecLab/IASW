# src/evaluation/auto_summarize.py
# -*- coding: utf-8 -*-

import os
import glob
import json
import pandas as pd
import numpy as np


def safe_round(x, ndigits=4):
    try:
        if x is None:
            return "-"
        if isinstance(x, (int, float, np.integer, np.floating)):
            if np.isnan(x) or np.isinf(x):
                return "-"
            return round(float(x), ndigits)
    except Exception:
        pass
    return "-"


def safe_get_nested_mean(data, key):
    """安全地从 detailed_quality_metrics 的嵌套字典中提取 mean 值"""
    if not data or key not in data:
        return "-"
    if isinstance(data[key], dict) and "mean" in data[key]:
        return safe_round(data[key]["mean"])
    if isinstance(data[key], (int, float)):
        return safe_round(data[key])
    return "-"


def extract_numeric_values(data, key):
    """
    从 detailed_quality_metrics 中提取样本级 values。
    如果文件里没有 values，只返回空列表。
    """
    if not data or key not in data:
        return []

    obj = data[key]

    if isinstance(obj, dict):
        for value_key in [
            "values", "scores", "raw", "raw_values",
            "all", "items", "per_sample", "sample_values"
        ]:
            if value_key in obj and isinstance(obj[value_key], list):
                values = obj[value_key]
                break
        else:
            values = []
    elif isinstance(obj, list):
        values = obj
    else:
        values = []

    clean_values = []
    for v in values:
        try:
            fv = float(v)
            if not np.isnan(fv) and not np.isinf(fv):
                clean_values.append(fv)
        except Exception:
            continue

    return clean_values


def summarize_metric_from_quality(data, key, prefix, outlier_mode=None):
    """
    兼容两种质量文件：
    1. 有 values：重新计算完整分布；
    2. 只有 mean/std/min/max/median 等统计字段：直接读取已有统计。
    """
    result = {
        f"{prefix}_N": 0,
        f"{prefix}_Mean": "-",
        f"{prefix}_Median": "-",
        f"{prefix}_Std": "-",
        f"{prefix}_Min": "-",
        f"{prefix}_P90": "-",
        f"{prefix}_P95": "-",
        f"{prefix}_Max": "-",
    }

    if outlier_mode:
        result[f"{prefix}_Outlier_Threshold"] = "-"
        result[f"{prefix}_Outlier_Count"] = "-"
        result[f"{prefix}_Outlier_Ratio"] = "-"

    values = extract_numeric_values(data, key)
    if values:
        return summarize_distribution(values, prefix, outlier_mode=outlier_mode)

    if not data or key not in data:
        return result

    obj = data[key]

    if isinstance(obj, dict):
        n = obj.get("n", obj.get("count", obj.get("num_samples", 0)))
        result[f"{prefix}_N"] = n if isinstance(n, int) else 0

        field_map = {
            "Mean": ["mean", "avg"],
            "Median": ["median", "p50", "50%"],
            "Std": ["std", "std_dev"],
            "Min": ["min"],
            "P90": ["p90", "90%"],
            "P95": ["p95", "95%"],
            "Max": ["max"],
        }

        for out_name, candidates in field_map.items():
            for c in candidates:
                if c in obj:
                    result[f"{prefix}_{out_name}"] = safe_round(obj[c])
                    break

        q1 = obj.get("q1", obj.get("25%", obj.get("p25", None)))
        q3 = obj.get("q3", obj.get("75%", obj.get("p75", None)))

        if outlier_mode and q1 is not None and q3 is not None:
            try:
                q1 = float(q1)
                q3 = float(q3)
                iqr = q3 - q1
                if outlier_mode == "iqr_high":
                    threshold = q3 + 1.5 * iqr
                else:
                    threshold = q1 - 1.5 * iqr
                result[f"{prefix}_Outlier_Threshold"] = safe_round(threshold)
            except Exception:
                pass

        return result

    if isinstance(obj, (int, float)):
        result[f"{prefix}_N"] = 1
        result[f"{prefix}_Mean"] = safe_round(obj)
        result[f"{prefix}_Median"] = safe_round(obj)
        result[f"{prefix}_Min"] = safe_round(obj)
        result[f"{prefix}_Max"] = safe_round(obj)

    return result


def summarize_distribution(values, prefix, outlier_mode=None):
    """
    对样本级质量指标做轻量分布统计。
    outlier_mode:
      - None: 不统计异常值
      - "iqr_high": 仅统计高端异常值，适合 PPL、重复率
      - "iqr_low": 仅统计低端异常值，适合相似度/连贯性过低
    """
    result = {
        f"{prefix}_N": 0,
        f"{prefix}_Mean": "-",
        f"{prefix}_Median": "-",
        f"{prefix}_Std": "-",
        f"{prefix}_Min": "-",
        f"{prefix}_P90": "-",
        f"{prefix}_P95": "-",
        f"{prefix}_Max": "-",
    }

    if not values:
        if outlier_mode:
            result[f"{prefix}_Outlier_Threshold"] = "-"
            result[f"{prefix}_Outlier_Count"] = "-"
            result[f"{prefix}_Outlier_Ratio"] = "-"
        return result

    arr = np.array(values, dtype=float)
    result.update({
        f"{prefix}_N": int(len(arr)),
        f"{prefix}_Mean": safe_round(np.mean(arr)),
        f"{prefix}_Median": safe_round(np.median(arr)),
        f"{prefix}_Std": safe_round(np.std(arr)),
        f"{prefix}_Min": safe_round(np.min(arr)),
        f"{prefix}_P90": safe_round(np.percentile(arr, 90)),
        f"{prefix}_P95": safe_round(np.percentile(arr, 95)),
        f"{prefix}_Max": safe_round(np.max(arr)),
    })

    if outlier_mode:
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1

        if outlier_mode == "iqr_high":
            threshold = q3 + 1.5 * iqr
            count = int(np.sum(arr > threshold))
        elif outlier_mode == "iqr_low":
            threshold = q1 - 1.5 * iqr
            count = int(np.sum(arr < threshold))
        else:
            threshold = None
            count = 0

        result[f"{prefix}_Outlier_Threshold"] = safe_round(threshold)
        result[f"{prefix}_Outlier_Count"] = count
        result[f"{prefix}_Outlier_Ratio"] = safe_round(count / len(arr))

    return result


def summarize_quality_diagnostics(method, quality):
    record = {"Method": method}

    record.update(summarize_metric_from_quality(
        quality, "ppl_wm", "PPL", outlier_mode="iqr_high"
    ))

    rep_diag = summarize_metric_from_quality(
        quality, "rep15_wm", "Rep15", outlier_mode="iqr_high"
    )
    record.update(rep_diag)

    ent_diag = summarize_metric_from_quality(
        quality, "ent3_wm", "Entropy", outlier_mode=None
    )
    record.update(ent_diag)

    coh_diag = summarize_metric_from_quality(
        quality, "sim_wm_human", "Coherence", outlier_mode="iqr_low"
    )
    record.update(coh_diag)

    sim_diag = summarize_metric_from_quality(
        quality, "sim_wm_clean", "SimClean", outlier_mode="iqr_low"
    )
    record.update(sim_diag)

    return record


def is_ablation_metrics_file(path: str) -> bool:
    """
    过滤 IAS 消融绘图生成的 metrics 文件，例如：
      - ias_ablation_roc_ablation_metrics.json
      - xxx_ablation_metrics.json
    """
    name = os.path.basename(path).lower()
    return name.endswith("_ablation_metrics.json")

def collect_metrics_json_files(exp_data_dir, exp_res_dir, plot_dir=""):
    """
    如果明确提供 plot_dir，则只从 plot_dir 读取 metrics，
    防止同时混入原实验 plots 中的另一套负样本结果。
    """
    if plot_dir:
        candidate_dirs = [plot_dir]
    else:
        candidate_dirs = [
            exp_res_dir,
            exp_data_dir,
        ]

        sibling_plots = os.path.join(
            os.path.dirname(exp_data_dir.rstrip("/")),
            "plots"
        )
        candidate_dirs.append(sibling_plots)

    metrics_files = []
    seen = set()

    for directory in candidate_dirs:
        if not directory or not os.path.isdir(directory):
            continue

        for path in glob.glob(
            os.path.join(directory, "*_metrics.json")
        ):
            if is_ablation_metrics_file(path):
                continue

            if path not in seen:
                metrics_files.append(path)
                seen.add(path)

    return metrics_files


#     return metrics_files


def summarize_time_cost(exp_data_dir, exp_res_dir):
    """独立的外挂函数：安全统计生成时间分布，绝不影响原有逻辑"""
    time_records = []

    all_jsonl_files = glob.glob(os.path.join(exp_data_dir, "*.jsonl"))
    gen_files = [
        f for f in all_jsonl_files
        if not os.path.basename(f).startswith(("baseline", "attacked", "detect", "temp"))
    ]

    if not gen_files:
        return

    for file_path in gen_files:
        basename = os.path.basename(file_path)
        method_name = basename.replace(".jsonl", "")

        times = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if "generation_time" in row and row["generation_time"] is not None:
                        times.append(float(row["generation_time"]))
                    elif "time_cost_seconds" in row and row["time_cost_seconds"] is not None:
                        times.append(float(row["time_cost_seconds"]))
                except Exception:
                    continue

        if not times:
            time_records.append({
                "Method": method_name,
                "Valid_Samples": 0,
                "Mean_Time(s)": "-", "Std_Dev": "-",
                "Min": "-", "25%_Q1": "-", "Median": "-", "75%_Q3": "-", "Max": "-"
            })
        else:
            arr = np.array(times)
            time_records.append({
                "Method": method_name,
                "Valid_Samples": len(arr),
                "Mean_Time(s)": round(np.mean(arr), 4),
                "Std_Dev": round(np.std(arr), 4),
                "Min": round(np.min(arr), 4),
                "25%_Q1": round(np.percentile(arr, 25), 4),
                "Median": round(np.median(arr), 4),
                "75%_Q3": round(np.percentile(arr, 75), 4),
                "Max": round(np.max(arr), 4)
            })

    if time_records:
        df_time = pd.DataFrame(time_records)
        df_time = df_time.sort_values(by=["Method"])
        out_time_path = os.path.join(exp_res_dir, "final_time_cost_summary.csv")
        df_time.to_csv(out_time_path, index=False, encoding="utf-8-sig")
        print(f"⏱️  3. 生成时间测评表 (Time Cost) [{len(df_time)} 行]: {out_time_path}")


def is_composite_dual_method(method_name: str) -> bool:
    """
    Only project composite methods named ``dual_...`` are dual-channel rows.

    This deliberately excludes standalone baselines such as ``dualmark``
    even though its name begins with the letters ``dual``.
    """
    name = str(method_name or "").strip().lower()
    base = name.split(" (", 1)[0]
    return base.startswith("dual_")


def normalize_stage_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = s.replace("-", "_")
    s = " ".join(s.split())
    return s


def get_stage_metrics(channel_dict: dict, stage_name: str):
    if not isinstance(channel_dict, dict):
        return None

    if stage_name in channel_dict:
        return channel_dict[stage_name]

    target_norm = normalize_stage_name(stage_name)
    for real_stage_name, stage_metrics in channel_dict.items():
        if normalize_stage_name(real_stage_name) == target_norm:
            return stage_metrics

    return None


def load_quality_file(exp_data_dir, method):
    q_path_new = os.path.join(exp_data_dir, f"detailed_quality_metrics_{method}.json")
    q_path_old = os.path.join(exp_data_dir, f"detailed_quality_metrics_gen_{method}.json")

    quality = {}
    if os.path.exists(q_path_new):
        with open(q_path_new, "r", encoding="utf-8") as f:
            quality = json.load(f)
    elif os.path.exists(q_path_old):
        with open(q_path_old, "r", encoding="utf-8") as f:
            quality = json.load(f)

    return quality


def main(exp_data_dir, exp_res_dir, plot_dir=""):
    robustness_records = {}
    quality_records = {}
    quality_diag_records = {}
    all_attacks = set()

    target_metrics = ["AUC", "TPR@0.1%FPR", "TPR@1%FPR", "TPR@5%FPR"]

    metrics_files = collect_metrics_json_files(
        exp_data_dir=exp_data_dir,
        exp_res_dir=exp_res_dir,
        plot_dir=plot_dir
    )

    if not metrics_files:
        search_hint = []
        if plot_dir:
            search_hint.append(plot_dir)
        search_hint.append(exp_res_dir)
        search_hint.append(exp_data_dir)
        sibling_plots = os.path.join(os.path.dirname(exp_data_dir.rstrip("/")), "plots")
        search_hint.append(sibling_plots)

        print(f"⚠️ 在以下位置均未找到任何非 ablation 的 *_metrics.json 文件：")
        for p in search_hint:
            print(f"   - {p}")
        return

    print(f"🔎 共读取到 {len(metrics_files)} 个有效 metrics 文件（已自动忽略 ablation metrics）")

    for file_path in metrics_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                mdata = json.load(f)
            except Exception:
                continue

        method = mdata.get("method", "unknown")
        attack = mdata.get("attack_name", "unknown")

        evaluated_attacks = mdata.get("evaluated_attacks", [])
        if isinstance(evaluated_attacks, list):
            for atk in evaluated_attacks:
                all_attacks.add(atk)

        if attack != "unknown":
            all_attacks.add(attack)

        # ---------------------------------------------------
        # A. 文本质量指标
        # ---------------------------------------------------
        if method not in quality_records:
            quality = load_quality_file(exp_data_dir, method)

            quality_records[method] = {
                "Method": method,
                "PPL": safe_get_nested_mean(quality, "ppl_wm"),
                "MAUVE": safe_get_nested_mean(quality, "mauve_score"),
                "Coherence (WM vs Human)": safe_get_nested_mean(quality, "sim_wm_human"),
                "Entropy": safe_get_nested_mean(quality, "ent3_wm"),
                "Repetition_Rate_15": safe_get_nested_mean(quality, "rep15_wm"),
                "Sim_WM_vs_Clean": safe_get_nested_mean(quality, "sim_wm_clean")
            }

            quality_diag_records[method] = summarize_quality_diagnostics(method, quality)

        # ---------------------------------------------------
        # B. 鲁棒性指标
        # ---------------------------------------------------
        def get_metric(channel, stage, metric):
            try:
                if metric not in target_metrics:
                    return "-"
                if channel not in mdata:
                    return "-"

                channel_dict = mdata[channel]
                stage_metrics = get_stage_metrics(channel_dict, stage)

                if not isinstance(stage_metrics, dict):
                    return "-"
                if metric not in stage_metrics:
                    return "-"

                return round(stage_metrics[metric], 4)
            except Exception:
                return "-"

        if is_composite_dual_method(method):
            channels_to_extract = [
                ("Micro", f"{method} (Word-level)"),
                ("Macro", f"{method} (Semantic-level)"),
                ("Max Fusion", f"{method} (Max Fusion)"),
                ("Fisher Fusion", f"{method} (Fisher Fusion)")
            ]
        else:
            active_ch = "Micro"
            if "Micro" not in mdata or ("Micro" in mdata and "Before Attack" not in mdata["Micro"]):
                if "Macro" in mdata and "Before Attack" in mdata["Macro"]:
                    active_ch = "Macro"
            channels_to_extract = [(active_ch, method)]

        for ch_key, row_name in channels_to_extract:
            if row_name not in robustness_records:
                robustness_records[row_name] = {"Method": row_name}
                for m in target_metrics:
                    robustness_records[row_name][f"Unattacked_{m}"] = get_metric(ch_key, "Before Attack", m)

            if isinstance(evaluated_attacks, list) and evaluated_attacks:
                for atk in evaluated_attacks:
                    stage_name = f"After {atk}"
                    for m in target_metrics:
                        atk_metric_key = f"{atk}_{m}"
                        robustness_records[row_name][atk_metric_key] = get_metric(ch_key, stage_name, m)

            elif attack != "unknown":
                for m in target_metrics:
                    atk_metric_key = f"{attack}_{m}"
                    robustness_records[row_name][atk_metric_key] = get_metric(ch_key, "After Attack", m)

    print(f"🔍 [汇总分析] 自动检测到以下 {len(all_attacks)} 种攻击模式及参数变体：")
    for atk in sorted(list(all_attacks)):
        print(f"   - {atk}")

    def sort_key(method_name):
        if is_composite_dual_method(method_name):
            base_name = method_name.split()[0]
            if "Word-level" in method_name:
                return (1, base_name, 0)
            if "Semantic-level" in method_name:
                return (1, base_name, 1)
            if "Max Fusion" in method_name:
                return (1, base_name, 2)
            if "Fisher Fusion" in method_name:
                return (1, base_name, 3)
            return (1, base_name, 4)
        return (0, method_name, 0)

    # 1. 鲁棒性宽表
    if robustness_records:
        df_rob = pd.DataFrame(list(robustness_records.values()))
        df_rob["sort_key"] = df_rob["Method"].apply(sort_key)
        df_rob = df_rob.sort_values(by=["sort_key", "Method"]).drop("sort_key", axis=1)

        attack_cols = []
        for atk in sorted(list(all_attacks)):
            for m in target_metrics:
                attack_cols.append(f"{atk}_{m}")

        for col in attack_cols:
            if col not in df_rob.columns:
                df_rob[col] = "-"

        df_rob = df_rob.fillna("-")

        unattacked_cols = [f"Unattacked_{m}" for m in target_metrics]
        cols_rob = ["Method"] + unattacked_cols + attack_cols
        df_rob = df_rob[[c for c in cols_rob if c in df_rob.columns]]

        out_rob_path = os.path.join(exp_res_dir, "final_robustness_summary.csv")
        df_rob.to_csv(out_rob_path, index=False, encoding="utf-8-sig")
        print(f"🛡️  1. 鲁棒性测评表 (Robustness) [{len(df_rob)} 行]: {out_rob_path}")

    # 2. 文本质量主表
    if quality_records:
        df_qual = pd.DataFrame(list(quality_records.values()))
        df_qual = df_qual.sort_values(by=["Method"])
        df_qual = df_qual.fillna("-")

        out_qual_path = os.path.join(exp_res_dir, "final_quality_summary.csv")
        df_qual.to_csv(out_qual_path, index=False, encoding="utf-8-sig")
        print(f"📝  2. 文本质量主表 (Quality Mean) [{len(df_qual)} 行]: {out_qual_path}")

    # 2.1 文本质量诊断表
    if quality_diag_records:
        df_qdiag = pd.DataFrame(list(quality_diag_records.values()))
        df_qdiag = df_qdiag.sort_values(by=["Method"])
        df_qdiag = df_qdiag.fillna("-")

        out_qdiag_path = os.path.join(exp_res_dir, "final_quality_diagnostics.csv")
        df_qdiag.to_csv(out_qdiag_path, index=False, encoding="utf-8-sig")
        print(f"🔬  2.1 文本质量诊断表 (Quality Diagnostics) [{len(df_qdiag)} 行]: {out_qdiag_path}")

    # 3. 生成时间统计
    summarize_time_cost(exp_data_dir, exp_res_dir)

    print("\n✅ 自动化汇总完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_data_dir", type=str, required=True, help="实验中间数据目录 (JSONL 所在处)")
    parser.add_argument("--plot_dir", type=str, default="", help="绘图输出目录 (PDF 与 *_metrics.json 所在处)")
    parser.add_argument("--exp_res_dir", type=str, required=True, help="结果输出目录 (CSV 汇总表输出目录)")
    args = parser.parse_args()

    os.makedirs(args.exp_res_dir, exist_ok=True)
    main(args.exp_data_dir, args.exp_res_dir, args.plot_dir)