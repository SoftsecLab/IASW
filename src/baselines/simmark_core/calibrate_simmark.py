# src/baselines/simmark_core/calibrate_simmark.py
# -*- coding: utf-8 -*-
import random
import os
import json
import pickle
import numpy as np
import torch
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser(description='SimMark Gamma Calibration Tool')
    parser.add_argument('--input_file', type=str, required=True, help='输入 JSONL 路径')
    parser.add_argument('--dataset_type', type=str, default='c4', help='数据集标识 (用于加载专属 PCA)')
    parser.add_argument('--embedder_path', type=str, default='hkunlp/instructor-large')
    parser.add_argument('--mode', type=str, choices=['cosine', 'euclidean'], default='cosine')
    parser.add_argument('--a', type=float, required=True, help='区间下界')
    parser.add_argument('--b', type=float, required=True, help='区间上界')
    parser.add_argument('--use_pca', action='store_true', help='欧氏距离模式下是否使用 PCA')
    return parser.parse_args()

def run_calibration(args):
    print(f"[*] 模式: {args.mode.upper()} | 数据集: {args.dataset_type.upper()} | 目标区间: [{args.a}, {args.b}]")
    
    # 1. 加载数据
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    all_texts = [row.get("text", row.get("prompt", "")) for row in data]
    
    # 根据模式选择 Instruction
    if args.mode == 'euclidean' or args.use_pca:
        instruction = "Represent the sentence for PCA:"
    else:
        instruction = "Represent the sentence for cosine similarity:"
        
    print(f"[*] 加载 Embedder 模型: {args.embedder_path}...")
    embedder = SentenceTransformer(args.embedder_path)

    # 2. 处理 PCA (仅欧氏距离)
    pca_transformer = None
    if args.mode == 'euclidean' and args.use_pca:
        # 🌟 动态匹配数据集专属 PCA 模型名
        pca_filename = f"{args.dataset_type.lower()}_pca_model_16.pkl"
        pca_path = os.path.join(os.path.dirname(__file__), pca_filename)
        
        if os.path.exists(pca_path):
            print(f"[3] 正在加载专属 PCA 模型: {pca_path}")
            with open(pca_path, 'rb') as f:
                pca_transformer = pickle.load(f)
        else:
            # 这里的逻辑仅作为脚本内部兜底，建议通过 0_1 脚本在外部统一训练
            print(f"[!] 错误: 找不到 PCA 模型 {pca_path}。请先运行 Step 1 训练模型。")
            return

    # 3. 计算距离分布
    print("[4] 计算相邻句对距离并测算 Gamma...")
    distances = []
    
    # 设置 Embedder 编码参数
    encode_kwargs = {
        "prompt": instruction,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False
    }

    for text in tqdm(all_texts, desc="计算距离"):
        sents = sent_tokenize(text)
        if len(sents) < 2: continue
        
        # 编码
        sent_embs = embedder.encode(sents, **encode_kwargs)
        
        # 🌟 核心：如果是欧氏距离且开启了 PCA，执行转换
        if args.mode == 'euclidean' and pca_transformer:
            sent_embs = pca_transformer.transform(sent_embs)
        
        # 计算相邻句对距离
        for i in range(1, len(sents)):
            emb1 = sent_embs[i-1].reshape(1, -1)
            emb2 = sent_embs[i].reshape(1, -1)
            
            if args.mode == 'euclidean':
                dist = euclidean_distances(emb1, emb2)[0, 0]
            else:
                dist = cosine_similarity(emb1, emb2)[0, 0]
            distances.append(dist)
            
    distances = np.array(distances)
    
    if len(distances) == 0:
        print("❌ 错误：有效句对数量为 0，请检查数据长度。")
        return

    # 4. 统计与 Gamma 计算
    print("\n[+] 距离分布统计:")
    print(f"有效相邻句对数量: {len(distances)}")
    print(f"均值: {np.mean(distances):.4f}, 中位数: {np.median(distances):.4f}")
    print(f"最大/最小值: {np.max(distances):.4f} / {np.min(distances):.4f}")
    
    # 计算 Gamma
    counts, bins = np.histogram(distances, bins=1000)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mask = (bin_centers >= args.a) & (bin_centers <= args.b)
    gamma = np.sum(counts[mask]) / len(distances)
    
    print("\n" + "="*50)
    print(f"📊 校准结果 (数据集: {args.dataset_type.upper()} | 模式: {args.mode.upper()})")
    print(f"目标区间 [a, b]: [{args.a}, {args.b}]")
    print(f"👉 算出的 Golden Gamma 值: {gamma:.4f}")
    print("="*50)

if __name__ == '__main__':
    args = parse_args()
    run_calibration(args)