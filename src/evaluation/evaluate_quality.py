# src/evaluation/evaluate_quality.py
# -*- coding: utf-8 -*-

import os
# 强制指定 NLTK 离线数据包路径
os.environ["NLTK_DATA"] = "./models/nltk_data"

import json
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import mauve
except ImportError:
    mauve = None
    print("Warning: mauve-text 未安装，将跳过 MAUVE 计算。如需计算请运行: pip install mauve-text")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Comprehensive Text Quality with Detailed Statistics")
    parser.add_argument("--input_file", type=str, required=True, help="生成的集成数据 JSONL 文件")
    # 🌟 删除了 --clean_gen_file，因为数据现在都在 input_file 里面了
    parser.add_argument("--model_path", type=str, required=True, help="用于计算 PPL 的大模型路径")
    parser.add_argument("--embedder_path", type=str, default="./models/all-MiniLM-L6-v2", help="计算语义保真度的轻量级模型")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

@torch.no_grad()
def calculate_ppl(text: str, model, tokenizer, device) -> float:
    if not text or not text.strip(): return float('nan')
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    if inputs["input_ids"].shape[1] < 2: return float('nan')
    outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

def get_ngrams(words: list, n: int):
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def calculate_repetition_and_diversity(text: str, n: int) -> tuple:
    """返回 (重复率, 多样性)"""
    words = text.lower().split()
    if len(words) < n: return float('nan'), float('nan')
    ngrams = get_ngrams(words, n)
    if len(ngrams) == 0: return float('nan'), float('nan')
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    rep_rate = 1.0 - (unique_ngrams / float(total_ngrams))
    diversity = unique_ngrams / float(total_ngrams)
    return rep_rate, diversity

def calculate_entropy(text: str, n: int = 3) -> float:
    words = text.lower().split()
    if len(words) < n: return float('nan')
    ngrams = get_ngrams(words, n)
    if len(ngrams) == 0: return float('nan')
    from collections import Counter
    counts = Counter(ngrams)
    entropy = 0.0
    for count in counts.values():
        p = count / len(ngrams)
        entropy -= p * math.log(p)
    return entropy

def calculate_semantic_sim(text1: str, text2: str, embedder) -> float:
    if not text1 or not text2 or not text1.strip() or not text2.strip(): 
        return float('nan')
    emb1 = embedder.encode(text1, convert_to_numpy=True)
    emb2 = embedder.encode(text2, convert_to_numpy=True)
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]
    return float(sim)

def compute_detailed_stats(data_list, bins=5):
    """计算一维数组的详细统计指标和区间分布"""
    data = np.array(data_list)
    data = data[~np.isnan(data)] # 过滤掉 NaN 值
    
    if len(data) == 0:
        return {"status": "empty_or_all_nan"}
        
    stats = {
        "count": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "var": float(np.var(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "p25": float(np.percentile(data, 25)),
        "median": float(np.percentile(data, 50)),
        "p75": float(np.percentile(data, 75))
    }
    
    counts, bin_edges = np.histogram(data, bins=bins)
    distribution = {}
    for i in range(len(counts)):
        range_str = f"[{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f})"
        if i == len(counts) - 1:
            range_str = f"[{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}]"
        distribution[range_str] = int(counts[i])
        
    stats["distribution_bins"] = distribution
    return stats

def main():
    args = parse_args()
    
    print(f"[*] Loading LLM for PPL Evaluation: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model_path = args.model_path
    model_load_kwargs = {
        "device_map": args.device,
        "torch_dtype": torch.bfloat16,
    }

    model_name = str(model_path).rstrip("/")

    # 兼容旧版 semstamp 基座模型
    if model_name.endswith("opt-1.3b-semstamp") or "AbeHou/opt-1.3b-semstamp" in model_name:
        model_load_kwargs["use_safetensors"] = False
        print(f"[*] 检测到旧版评估模型 {model_path}，自动设置 use_safetensors=False")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_load_kwargs
    ).eval()
    
    print(f"[*] Loading Embedder for Semantic Fidelity: {args.embedder_path}")
    embedder = SentenceTransformer(args.embedder_path, device=args.device)
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    metrics_keys = ["ppl", "len", "rep2", "rep3", "rep4", "rep15", "ent3"]
    # 前缀保持 clean_gen 不变，以防破坏 downstream auto_summarize.py 的取值逻辑
    prefixes = ["human", "wm", "clean_gen"]
    
    raw_metrics = {f"{k}_{p}": [] for k in metrics_keys for p in prefixes}
    
    raw_metrics.update({
        "sim_wm_human": [],   
        "sim_clean_human": [],
        "sim_wm_clean": []    
    })
    
    mauve_human_texts = []
    mauve_wm_texts = []
    
    print("[*] Evaluating Comprehensive Text Quality and Collecting Raw Data...")
    for i, row in enumerate(tqdm(data)):
        # 🌟 修复 A: 统一字段名，兼容旧的 human_reference
        human_text = row.get("human_text", row.get("human_reference", ""))
        wm_text = row.get("wm_text", "")
        # 🌟 修复 B: 直接从当前行取 no_wm_text
        clean_gen = row.get("no_wm_text", "")
            
        if human_text and wm_text:
            mauve_human_texts.append(human_text)
            mauve_wm_texts.append(wm_text)

        for txt, prefix in [(human_text, "human"), (wm_text, "wm"), (clean_gen, "clean_gen")]:
            if not txt: continue
            
            raw_metrics[f"len_{prefix}"].append(len(txt.split()))
            raw_metrics[f"ppl_{prefix}"].append(calculate_ppl(txt, model, tokenizer, args.device))
            
            rep2, _ = calculate_repetition_and_diversity(txt, n=2)
            rep3, _ = calculate_repetition_and_diversity(txt, n=3)
            rep4, _ = calculate_repetition_and_diversity(txt, n=4)
            rep15, _ = calculate_repetition_and_diversity(txt, n=15)
            ent3 = calculate_entropy(txt, n=3)
            
            raw_metrics[f"rep2_{prefix}"].append(rep2)
            raw_metrics[f"rep3_{prefix}"].append(rep3)
            raw_metrics[f"rep4_{prefix}"].append(rep4)
            raw_metrics[f"rep15_{prefix}"].append(rep15)
            raw_metrics[f"ent3_{prefix}"].append(ent3)
            
        sim_w_h = calculate_semantic_sim(wm_text, human_text, embedder)
        raw_metrics["sim_wm_human"].append(sim_w_h)
        
        if clean_gen:
            sim_c_h = calculate_semantic_sim(clean_gen, human_text, embedder)
            sim_w_c = calculate_semantic_sim(wm_text, clean_gen, embedder)
            raw_metrics["sim_clean_human"].append(sim_c_h)
            raw_metrics["sim_wm_clean"].append(sim_w_c)

    print("\n[*] Computing Detailed Statistics and Distributions...")
    detailed_results = {}
    for metric_name, values in raw_metrics.items():
        if len(values) > 0:
            detailed_results[metric_name] = compute_detailed_stats(values, bins=6)

    mauve_score = float('nan')
    if mauve is not None and len(mauve_human_texts) > 0 and len(mauve_wm_texts) > 0:
        print("\n[*] Computing MAUVE Score (Macro-level distribution alignment)...")
        try:
            device_id = -1
            if "cuda" in args.device:
                try:
                    device_id = int(args.device.split(":")[1])
                except (IndexError, ValueError):
                    device_id = 0
            
            min_len = min(len(mauve_human_texts), len(mauve_wm_texts))
            out = mauve.compute_mauve(
                p_text=mauve_human_texts[:min_len],
                q_text=mauve_wm_texts[:min_len],
                device_id=device_id,
                max_text_length=256,
                featurize_model_name="gpt2-large"
            )
            mauve_score = float(out.mauve)
        except Exception as e:
            print(f"[-] Error calculating MAUVE: {e}")
            
    detailed_results["mauve_score"] = mauve_score

    print("\n=========================================================================")
    print("📊 文本质量高级统计摘要报告 (完整分布已存入 JSON)")
    print("=========================================================================")
    
    def print_stat(name, key):
        if key in detailed_results and "mean" in detailed_results[key]:
            stats = detailed_results[key]
            print(f"[{name}] Mean: {stats['mean']:.3f} | Std: {stats['std']:.3f} | Min: {stats['min']:.3f} | Max: {stats['max']:.3f}")
    
    print("--- 语言困惑度 (PPL) ---")
    print_stat("Human", "ppl_human")
    print_stat("Clean", "ppl_clean_gen")
    print_stat("WM   ", "ppl_wm")
    
    print("\n--- 宏观重复崩坏率 (Rep-15) ---")
    print_stat("Human", "rep15_human")
    print_stat("Clean", "rep15_clean_gen")
    print_stat("WM   ", "rep15_wm")

    print("\n--- 三角语义保真度 (Cosine Similarity) ---")
    print_stat("WM vs Human   ", "sim_wm_human")
    print_stat("Clean vs Human", "sim_clean_human")
    print_stat("WM vs Clean   ", "sim_wm_clean")
    
    print("\n--- 整体分布质量 (MAUVE) ---")
    if not math.isnan(mauve_score):
        print(f"[MAUVE] Score: {mauve_score:.4f} (越接近 1 越好)")
    else:
        print("[MAUVE] Score: N/A")
    print("=========================================================================\n")

    output_dir = os.path.dirname(args.input_file)
    base_name = os.path.basename(args.input_file).replace(".jsonl", "")
    out_json_path = os.path.join(output_dir, f"detailed_quality_metrics_{base_name}.json")
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)
    
    print(f"[✔] 完整统计数据（含方差、最值、区间分布等及 MAUVE 分数）已保存至: {out_json_path}")

if __name__ == "__main__":
    main()