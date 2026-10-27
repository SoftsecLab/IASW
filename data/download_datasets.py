# -*- coding: utf-8 -*-
# 运行方式：python download_datasets.py
# 依赖环境：pip install datasets tqdm

import os
import json
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# ⚙️ 配置区
# ==========================================
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目标拉取数量（注意：Dolly 总共只有 15k 条，会自动拉取全部）
TARGET_SAMPLES = 50000 
MIN_WORDS = 50  # 最低词数过滤，丢弃极短的废话和标题

# ==========================================
# 🛠️ 提取策略区 (针对不同数据集结构)
# ==========================================
def extract_c4(sample):
    return sample.get("text", "").strip()

def extract_wiki(sample):
    text = sample.get("text", "").strip()
    # 过滤掉维基百科的纯标题行 (例如: " = History = ")
    if text.startswith("=") and text.endswith("="):
        return ""
    return text

def extract_dolly(sample):
    """
    将 Dolly 的结构化指令转化为连贯的对话文本
    """
    instruction = sample.get("instruction", "").strip()
    context = sample.get("context", "").strip()
    response = sample.get("response", "").strip()
    
    # 智能拼接
    text = f"Instruction: {instruction}\n"
    if context:
        text += f"Context: {context}\n"
    text += f"Response: {response}"
    return text

# 数据集注册表：(数据集名, 子集/配置名, split, 提取函数, 输出文件名)
DATASETS_CONFIG = [
    ("allenai/c4", "en", "train", extract_c4, "raw_c4.jsonl"),
    ("wikitext", "wikitext-103-v1", "train", extract_wiki, "raw_wiki.jsonl"),
    ("databricks/databricks-dolly-15k", None, "train", extract_dolly, "raw_dolly.jsonl")
]

# ==========================================
# 🚀 核心下载引擎
# ==========================================
def main():
    print(f"[*] 初始化下载管线，数据将保存至: {OUTPUT_DIR}\n")
    
    for ds_path, config_name, split, extract_func, out_filename in DATASETS_CONFIG:
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        print(f"---------------------------------------------------")
        print(f"📦 正在拉取: {ds_path} ({config_name if config_name else 'default'})")
        
        try:
            # 开启流式下载，拒绝硬盘爆炸
            if config_name:
                dataset = load_dataset(ds_path, config_name, split=split, streaming=True)
            else:
                dataset = load_dataset(ds_path, split=split, streaming=True)
                
            saved_count = 0
            
            with open(out_path, 'w', encoding='utf-8') as f:
                # Dolly 数据量较小，单独处理进度条上限
                pbar_total = 15000 if "dolly" in ds_path else TARGET_SAMPLES
                pbar = tqdm(total=pbar_total, desc=f"Saving {out_filename}")
                
                for sample in dataset:
                    if saved_count >= TARGET_SAMPLES:
                        break
                        
                    # 1. 提取文本
                    raw_text = extract_func(sample)
                    
                    # 2. 质量与长度过滤 (极其重要！)
                    # 如果文本太短，或者被提取函数判定为废弃(返回"")，则跳过
                    if not raw_text or len(raw_text.split()) < MIN_WORDS:
                        continue
                        
                    # 3. 统一构建标准化字典并保存
                    clean_data = {"text": raw_text}
                    f.write(json.dumps(clean_data, ensure_ascii=False) + "\n")
                    
                    saved_count += 1
                    pbar.update(1)
                    
                pbar.close()
                print(f"✅ 成功保存 {saved_count} 条有效长文本至 -> {out_path}")
                
        except Exception as e:
            print(f"❌ 拉取 {ds_path} 失败，错误信息: {e}")
            print("👉 提示: 请检查网络，或者配置 HF_ENDPOINT=https://hf-mirror.com")

if __name__ == "__main__":
    main()