# src/calibration/calibrate_empirical_ias.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import traceback

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.watermarks.ias_mark import IASMark


def parse_args():
    parser = argparse.ArgumentParser(
        description="IAS 四种对齐模式经验零分布标定"
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="人类真实语料 JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSON 路径")
    parser.add_argument("--dataset_type", type=str, required=True, help="c4/wiki/dolly")
    parser.add_argument("--num_samples", type=int, default=2000, help="样本数量")
    parser.add_argument("--seq_len", type=int, default=200, help="截断 token 长度")
    parser.add_argument("--model_path", type=str, required=True, help="Tokenizer/LLM 路径")
    parser.add_argument("--embedder_path", type=str, required=True, help="BGE Embedder 路径")
    parser.add_argument("--text_column", type=str, default="text", help="JSONL 文本字段")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def infer_model_key(model_path: str) -> str:
    model_path = (model_path or "").lower()
    if "llama3.2-3b" in model_path or "llama-3.2-3b" in model_path:
        return "llama3.2-3b"
    if "llama3.1-8b" in model_path or "llama-3.1-8b" in model_path or "llama3" in model_path:
        return "llama3.1-8b"
    return os.path.basename(model_path.rstrip("/")) or "unknown_model"


def main():
    args = parse_args()

    print(f"[*] 加载 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"[*] 初始化 IASMark: dataset={args.dataset_type}, device={args.device}")
    engine = IASMark(
        embedder_path_or_model=args.embedder_path,
        tokenizer=tokenizer,
        model_name_path=args.model_path,
        dataset_type=args.dataset_type,
        device=args.device,
    )

    mode_scores = {
        "strict": [],
        "window": [],
        "dp_global": [],
        "dp_local": [],
    }

    valid_count = 0
    error_count = 0

    print(f"[*] 开始扫描数据集并标定 IAS 四模式分布，目标长度={args.seq_len} tokens")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=args.num_samples, desc="Calibrating IAS")
        for line in f:
            if not line.strip():
                continue
            if valid_count >= args.num_samples:
                break

            try:
                row = json.loads(line)
                raw_text = row.get(args.text_column, "")
                if not isinstance(raw_text, str) or not raw_text.strip():
                    continue

                tokens = tokenizer.encode(raw_text, add_special_tokens=False)
                if len(tokens) < max(50, args.seq_len // 2):
                    continue

                cropped_text = tokenizer.decode(tokens[:args.seq_len], skip_special_tokens=True)
                res = engine.detect(cropped_text)

                ablation_raw = res.get("ablation_raw_scores", {})
                if not ablation_raw:
                    continue

                ok = False
                for mode in mode_scores:
                    if mode in ablation_raw:
                        mode_scores[mode].append(float(ablation_raw[mode]))
                        ok = True

                if ok:
                    valid_count += 1
                    pbar.update(1)

            except Exception as e:
                error_count += 1
                if error_count == 1:
                    print(f"\n[!] 首次异常: {e}")
                    print("--- traceback ---")
                    traceback.print_exc()
                    print("-----------------\n")
                continue

        pbar.close()

    if valid_count < 100:
        raise RuntimeError(
            f"IAS 标定失败：有效样本过少，仅 {valid_count} 条，异常数={error_count}"
        )

    model_key = infer_model_key(args.model_path)

    modes_out = {}
    for mode, scores in mode_scores.items():
        if not scores:
            continue
        arr = np.array(scores, dtype=np.float64)
        modes_out[mode] = {
            "mu": float(np.mean(arr)),
            "sigma": float(np.std(arr)),
            "count": int(len(arr)),
        }

    output = {
        "method": "ias",
        "dataset_type": args.dataset_type.lower(),
        "model_key": model_key,
        "seq_len": int(args.seq_len),
        "sample_size": int(valid_count),
        "modes": modes_out,
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ IAS 四模式标定完成")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("=" * 70)
    print(f"[*] 已保存到: {args.output_file}")


if __name__ == "__main__":
    main()