#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 human 与 no_wm 两种负样本的全部鲁棒性汇总结果合并到一个 CSV。

默认读取：
  results/main_exp/<dataset>/<model>/summaries/final_robustness_summary.csv
  results/main_exp_no_wm/<dataset>/<model>/summaries/final_robustness_summary.csv

默认输出：
  results/summaries/final_robustness_human_vs_no_wm.csv

每个指标单元格格式：
  human/no_wm
例如：
  0.9987/0.9979
"""

import argparse
from pathlib import Path

import pandas as pd


DATASET_ORDER = {
    "c4": 0,
    "wiki": 1,
    "dolly": 2,
}

MODEL_ORDER = {
    "llama3.1-8b-instruct": 0,
    "opt-1.3b-semstamp": 1,
    "llama-3.2-3b-instruct": 2,
    "qwen2.5-7b-instruct": 3,
    "mistral-7b-instruct": 4,
}

KEY_COLUMNS = ["Dataset", "Model", "Method"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="合并 human 和 no_wm 两种负样本的鲁棒性汇总结果"
    )
    parser.add_argument(
        "--human_root",
        type=Path,
        default=Path("results/main_exp"),
        help="human 负样本结果根目录",
    )
    parser.add_argument(
        "--no_wm_root",
        type=Path,
        default=Path("results/main_exp_no_wm"),
        help="no_wm 负样本结果根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/summaries/final_robustness_human_vs_no_wm.csv"),
        help="合并后的总 CSV 输出路径",
    )
    return parser.parse_args()


def normalize_cell(value):
    """将空值统一显示为 '-'，其余内容保持不变。"""
    if pd.isna(value):
        return "-"

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return "-"

    return text


def collect_summaries(root: Path, source_name: str):
    """
    自动扫描：
      <root>/<dataset>/<model>/summaries/final_robustness_summary.csv
    """
    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")

    files = sorted(root.glob("*/*/summaries/final_robustness_summary.csv"))
    if not files:
        raise FileNotFoundError(
            f"在 {root} 中没有找到 final_robustness_summary.csv"
        )

    frames = []
    global_order = 0

    for file_path in files:
        relative = file_path.relative_to(root)
        dataset = relative.parts[0]
        model = relative.parts[1]

        df = pd.read_csv(
            file_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )

        if "Method" not in df.columns:
            print(f"⚠️ 跳过缺少 Method 列的文件：{file_path}")
            continue

        df.insert(0, "Model", model)
        df.insert(0, "Dataset", dataset)

        # 记录原始行顺序，合并后尽量保持 auto_summarize 的方法排序。
        df[f"__order_{source_name}"] = range(
            global_order,
            global_order + len(df),
        )
        global_order += len(df)

        frames.append(df)

    if not frames:
        raise RuntimeError(f"{source_name} 目录中没有可用的汇总表")

    result = pd.concat(frames, ignore_index=True, sort=False)

    # 防止同一个数据集、模型和方法被重复读取。
    duplicate_mask = result.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicates = result.loc[duplicate_mask, KEY_COLUMNS]
        print(f"⚠️ [{source_name}] 检测到重复记录，将保留最后一条：")
        print(duplicates.to_string(index=False))
        result = result.drop_duplicates(KEY_COLUMNS, keep="last")

    return result, len(files)


def get_metric_columns(human_df, no_wm_df):
    """按 human 表列顺序为主，追加 no_wm 中额外出现的列。"""
    ignored = set(KEY_COLUMNS) | {"__order_human", "__order_no_wm"}

    metric_columns = [
        col for col in human_df.columns
        if col not in ignored
    ]

    for col in no_wm_df.columns:
        if col not in ignored and col not in metric_columns:
            metric_columns.append(col)

    return metric_columns


def add_missing_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = "-"
    return df


def combine_results(human_df, no_wm_df):
    metric_columns = get_metric_columns(human_df, no_wm_df)

    human_df = add_missing_columns(human_df.copy(), metric_columns)
    no_wm_df = add_missing_columns(no_wm_df.copy(), metric_columns)

    human_columns = KEY_COLUMNS + ["__order_human"] + metric_columns
    no_wm_columns = KEY_COLUMNS + ["__order_no_wm"] + metric_columns

    human_df = human_df[human_columns].rename(
        columns={col: f"{col}__human" for col in metric_columns}
    )
    no_wm_df = no_wm_df[no_wm_columns].rename(
        columns={col: f"{col}__no_wm" for col in metric_columns}
    )

    merged = human_df.merge(
        no_wm_df,
        on=KEY_COLUMNS,
        how="outer",
    )

    output = merged[KEY_COLUMNS].copy()

    for metric in metric_columns:
        human_col = f"{metric}__human"
        no_wm_col = f"{metric}__no_wm"

        output[metric] = [
            f"{normalize_cell(human_value)}/{normalize_cell(no_wm_value)}"
            for human_value, no_wm_value in zip(
                merged[human_col],
                merged[no_wm_col],
            )
        ]

    # 自定义数据集和模型顺序。
    output["__dataset_rank"] = output["Dataset"].str.lower().map(
        DATASET_ORDER
    ).fillna(999)

    output["__model_rank"] = output["Model"].str.lower().map(
        MODEL_ORDER
    ).fillna(999)

    # 优先使用 human 表中的原始方法顺序；缺失时使用 no_wm 顺序。
    merged_order = merged["__order_human"].combine_first(
        merged["__order_no_wm"]
    )
    output["__method_rank"] = merged_order.fillna(10**9)

    output = output.sort_values(
        by=[
            "__dataset_rank",
            "Dataset",
            "__model_rank",
            "Model",
            "__method_rank",
            "Method",
        ],
        kind="stable",
    )

    output = output.drop(
        columns=["__dataset_rank", "__model_rank", "__method_rank"]
    ).reset_index(drop=True)

    human_keys = set(map(tuple, human_df[KEY_COLUMNS].to_numpy()))
    no_wm_keys = set(map(tuple, no_wm_df[KEY_COLUMNS].to_numpy()))

    stats = {
        "total_rows": len(output),
        "paired_rows": len(human_keys & no_wm_keys),
        "human_only_rows": len(human_keys - no_wm_keys),
        "no_wm_only_rows": len(no_wm_keys - human_keys),
        "metric_columns": len(metric_columns),
    }

    return output, stats


def main():
    args = parse_args()

    print("===================================================")
    print("📊 合并 human 与 no_wm 两种负样本结果")
    print("===================================================")
    print(f"Human 根目录 : {args.human_root}")
    print(f"No-WM 根目录 : {args.no_wm_root}")
    print(f"输出文件     : {args.output}")
    print("单元格顺序   : human/no_wm")
    print("===================================================")

    human_df, human_file_count = collect_summaries(
        args.human_root,
        "human",
    )
    no_wm_df, no_wm_file_count = collect_summaries(
        args.no_wm_root,
        "no_wm",
    )

    combined_df, stats = combine_results(human_df, no_wm_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("===================================================")
    print("✅ 合并完成")
    print("===================================================")
    print(f"Human 汇总文件数     : {human_file_count}")
    print(f"No-WM 汇总文件数     : {no_wm_file_count}")
    print(f"总行数               : {stats['total_rows']}")
    print(f"两套结果均存在       : {stats['paired_rows']}")
    print(f"仅 Human 存在        : {stats['human_only_rows']}")
    print(f"仅 No-WM 存在        : {stats['no_wm_only_rows']}")
    print(f"指标列数             : {stats['metric_columns']}")
    print(f"输出文件             : {args.output}")
    print("===================================================")


if __name__ == "__main__":
    main()
