# src/calibration/calibrate_empirical_null.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# 允许从项目根目录直接运行
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.watermarks.algorithms.macro.semstamp import SemStampMacroAlgorithm
from src.watermarks.algorithms.macro.ksemstamp import KSemStampMacroAlgorithm
from src.watermarks.algorithms.macro.simmark_cosine import SimMarkCosineMacroAlgorithm
from src.watermarks.algorithms.macro.simmark_euclidean import SimMarkEuclideanMacroAlgorithm


MACRO_ALGORITHM_REGISTRY = {
    "semstamp": SemStampMacroAlgorithm,
    "ksemstamp": KSemStampMacroAlgorithm,
    "simmark_cosine": SimMarkCosineMacroAlgorithm,
    "simmark_euclidean": SimMarkEuclideanMacroAlgorithm,
}


def build_macro_algorithm(name, args, tokenizer, embedder=None, device="cuda"):
    name = name.lower().strip()
    if name not in MACRO_ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown macro algorithm: {name}")
    return MACRO_ALGORITHM_REGISTRY[name](args, tokenizer, embedder, device=device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="通用宏观水印经验零分布标定（非 IAS）"
    )
    parser.add_argument("--input_file", type=str, required=True, help="包含真实人类文本的原始数据集路径(JSONL)")
    parser.add_argument("--output_file", type=str, required=True, help="标定参数输出 JSON 路径")
    parser.add_argument("--dataset_type", type=str, required=True, help="数据集类型，如 c4/wiki/dolly")
    parser.add_argument("--seq_len", type=int, required=True, help="目标截断 token 长度")
    parser.add_argument("--sample_size", type=int, default=5000, help="用于标定的样本数量")
    parser.add_argument("--method", type=str, required=True, help="宏观方法名：semstamp/ksemstamp/simmark_cosine/simmark_euclidean")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer 路径")
    parser.add_argument("--embedder_path", type=str, default="", help="可选的 embedder 路径")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text_column", type=str, default="", help="若指定，则优先从该字段读取文本")
    return parser.parse_args()


def infer_model_key(model_path: str) -> str:
    model_path = (model_path or "").lower()
    if "llama3.2-3b" in model_path or "llama-3.2-3b" in model_path:
        return "llama3.2-3b"
    if "llama3.1-8b" in model_path or "llama-3.1-8b" in model_path or "llama3" in model_path:
        return "llama3.1-8b"
    return os.path.basename(model_path.rstrip("/")) or "unknown_model"


def pick_text(data: dict, preferred_key: str = "") -> str:
    candidate_keys = []
    if preferred_key:
        candidate_keys.append(preferred_key)
    candidate_keys.extend(["text", "content", "human_text", "human_reference"])

    for key in candidate_keys:
        value = data.get(key, "")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def main():
    args = parse_args()
    method = args.method.lower().strip()

    if method == "ias":
        raise ValueError(
            "IAS 不应走 calibrate_empirical_null.py；请改用 calibrate_empirical_ias.py。"
        )

    print(f"[*] 加载 Tokenizer: {args.tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except Exception as e:
        raise RuntimeError(f"加载 Tokenizer 失败: {e}")

    macro_args = SimpleNamespace(
        dataset_type=args.dataset_type,
        model_path=args.tokenizer_path,
        embedder_path=args.embedder_path,
        device=args.device,
    )

    print(f"[*] 初始化宏观检测器: {method} (dataset={args.dataset_type}, device={args.device})")
    detector = build_macro_algorithm(
        name=method,
        args=macro_args,
        tokenizer=tokenizer,
        embedder=None,
        device=args.device,
    )

    print(f"[*] 从 {args.input_file} 提取人类文本，并截断至 {args.seq_len} tokens ...")
    texts = []
    short_count = 0

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            human_text = pick_text(row, args.text_column)
            if not human_text:
                continue

            tokens = tokenizer.encode(human_text, add_special_tokens=False)
            if len(tokens) < max(10, args.seq_len // 2):
                short_count += 1
                continue

            truncated_text = tokenizer.decode(tokens[:args.seq_len], skip_special_tokens=True)
            texts.append(truncated_text)

            if len(texts) >= args.sample_size:
                break

    if len(texts) < 100:
        raise RuntimeError(
            f"提取到的有效文本过少: {len(texts)} 条，short/invalid 跳过数={short_count}"
        )

    print(f"[*] 开始计算 {len(texts)} 条样本的检测分数...")
    raw_scores = []
    error_count = 0

    for text in tqdm(texts, desc=f"Scoring {method} (len={args.seq_len})"):
        try:
            result = detector.detect(text)

            # 对宏观方法优先读取 z_score_macro
            score = result.get("z_score_macro", result.get("z_score", None))
            if score is not None:
                raw_scores.append(float(score))

        except Exception as e:
            error_count += 1
            if error_count == 1:
                print(f"\n[!] 首次检测异常: {e}")
                print("--- traceback ---")
                traceback.print_exc()
                print("-----------------\n")
            continue

    if len(raw_scores) < 100:
        raise RuntimeError(
            f"成功得到的有效分数过少: {len(raw_scores)} 条；总报错数={error_count}"
        )

    mu = float(np.mean(raw_scores))
    sigma = float(np.std(raw_scores))
    model_key = infer_model_key(args.tokenizer_path)

    print("\n" + "=" * 60)
    print(f"✅ {method} 标定完成")
    print(f"   dataset     : {args.dataset_type}")
    print(f"   model_key   : {model_key}")
    print(f"   seq_len     : {args.seq_len}")
    print(f"   sample_size : {len(raw_scores)}")
    print(f"   mu          : {mu:.6f}")
    print(f"   sigma       : {sigma:.6f}")
    print("=" * 60 + "\n")

    calib_params = {
        "method": method,
        "dataset_type": args.dataset_type.lower(),
        "model_key": model_key,
        "dataset_len": args.seq_len,
        "sample_size": len(raw_scores),
        "mu": mu,
        "sigma": sigma,
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(calib_params, f, indent=4, ensure_ascii=False)

    print(f"[*] 参数已保存至: {args.output_file}")


if __name__ == "__main__":
    main()