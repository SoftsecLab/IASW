# -*- coding: utf-8 -*-
import os
import re
import json
import torch
import time  # 🌟 引入 time 模块
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="构建 [Prompt, Human_Text, No_WM_Text] 三元组基准数据集")
    parser.add_argument("--input_file", type=str, required=True, help="第一步下载的清洗后原始数据 (JSONL)")
    parser.add_argument("--output_file", type=str, required=True, help="输出的三元组数据 (JSONL)")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["c4", "wiki", "dolly"], help="数据集类型")
    parser.add_argument("--model_path", type=str, required=True, help="LLM 路径")
    
    parser.add_argument("--prompt_len", type=int, default=30, help="C4/Wiki 预估的 Prompt 截取长度")
    parser.add_argument("--target_len", type=int, default=200, help="目标续写/截取的 Token 长度")
    parser.add_argument("--num_samples", type=int, default=500, help="目标最终合格的生成样本数")
    
    parser.add_argument("--batch_size", type=int, default=16, help="LLM 批量生成大小 (根据显存调整)")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度 (保证基线与水印生成的分布一致)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

def extract_prompt_and_human_text(raw_text: str, dataset_type: str, tokenizer, args):
    """
    智能边界切割：返回 (prompt, human_text) 元组
    如果该数据不符合条件（太短等），返回 (None, None)
    """
    if dataset_type == "dolly":
        # Dolly 是明确的结构化文本，以 "Response: " 作为界限
        parts = raw_text.split("\nResponse: ")
        if len(parts) != 2:
            return None, None
        
        prompt = parts[0].strip() + "\nResponse: "
        human_text = parts[1].strip()
        
        # 将 Dolly 的 human_text 截断到接近 target_len 的位置
        h_tokens = tokenizer.encode(human_text, add_special_tokens=False)
        if len(h_tokens) < args.target_len * 0.75: # 动态下限
            return None, None
        human_text = tokenizer.decode(h_tokens[:args.target_len], skip_special_tokens=True)
        # 寻找最后一个句号
        match = re.search(r'(?<=[.!?。！？\n])[^.!?。！？\n]*$', human_text)
        if match and match.start() > 0:
            human_text = human_text[:match.start()].strip()
            
        return prompt, human_text

    else: # c4 或 wiki
        tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        # 总长度必须能够支撑 prompt + target_len
        if len(tokens) < args.prompt_len + (args.target_len * 0.75):
            return None, None
            
        # 1. 找 Prompt 边界 (向后找)
        base_prompt = tokenizer.decode(tokens[:args.prompt_len], skip_special_tokens=True)
        match_prompt = re.search(r'[.!?。！？\n]', raw_text[len(base_prompt):])
        
        if match_prompt:
            prompt_end_idx = len(base_prompt) + match_prompt.end()
            prompt = raw_text[:prompt_end_idx].strip()
        else:
            prompt = base_prompt.strip()
            
        # 2. 找 Human Text 边界 (向回找)
        remaining_text = raw_text[len(prompt):].strip()
        r_tokens = tokenizer.encode(remaining_text, add_special_tokens=False)
        human_base = tokenizer.decode(r_tokens[:args.target_len], skip_special_tokens=True)
        
        match_human = re.search(r'(?<=[.!?。！？\n])[^.!?。！？\n]*$', human_base)
        if match_human and match_human.start() > 0:
            human_text = human_base[:match_human.start()].strip()
        else:
            human_text = human_base.strip()
            
        return prompt, human_text

# 🌟 新增：生成器函数，按需逐条读取并解析合法的前缀
def get_valid_pair_generator(file_path, dataset_type, tokenizer, args):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            raw_text = json.loads(line).get("text", "")
            prompt, human_text = extract_prompt_and_human_text(raw_text, dataset_type, tokenizer, args)
            if prompt and human_text:
                yield (prompt, human_text)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"\n[*] 正在加载 Tokenizer 和 LLM: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left") # 关键：左填充用于批量生成
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        

    model_path = args.model_path
    model_load_kwargs = {
        "device_map": "cuda:0",
        "torch_dtype": torch.float16,
    }

    # 兼容旧版 semstamp 基座模型：AbeHou/opt-1.3b-semstamp
    # 这个模型仓库里不是 safetensors 分片，需显式关闭
    model_name = str(model_path).rstrip("/")

    if model_name.endswith("opt-1.3b-semstamp") or "AbeHou/opt-1.3b-semstamp" in model_name:
        model_load_kwargs["use_safetensors"] = False
        print(f"[*] 检测到旧版模型 {model_path}，自动设置 use_safetensors=False")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_load_kwargs
    ).eval()
    

    print(f"[*] 准备构建 {args.num_samples} 条合法三元组数据 (Target Len: {args.target_len})...")
    
    # 🌟 核心修改：使用流式读取，不够就继续补
    pair_gen = get_valid_pair_generator(args.input_file, args.dataset_type, tokenizer, args)
    results = []
    min_required_tokens = int(args.target_len * 0.75) 
    
    # 使用 tqdm 追踪最终合法数据的收集进度
    pbar = tqdm(total=args.num_samples, desc="Generating Valid Triplets")
    
    while len(results) < args.num_samples:
        # 1. 动态收集一个 Batch 的前缀
        batch_pairs = []
        try:
            while len(batch_pairs) < args.batch_size:
                batch_pairs.append(next(pair_gen))
        except StopIteration:
            pass # 文件已经读到底了
            
        if not batch_pairs:
            print("\n[!] 警告：原始文件数据已彻底耗尽，但仍未达到目标样本数！")
            break

        # 2. 执行批量生成
        prompts = [p[0] for p in batch_pairs]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_lens = [len(ids) for ids in inputs.input_ids]
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.target_len + 30, # 多给点余量用于找句号
                min_new_tokens=min_required_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        batch_cost_time = time.time() - start_time
        avg_generation_time = batch_cost_time / len(batch_pairs)
            
        # 3. 逐条解码并执行严格过滤
        for j, out_ids in enumerate(outputs):
            gen_ids = out_ids[prompt_lens[j]:]
            gen_text_raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            gen_text_clean = gen_text_raw
            match_gen = re.search(r'(?<=[.!?。！？\n])[^.!?。！？\n]*$', gen_text_raw)
            if match_gen and match_gen.start() > 0:
                gen_text_clean = gen_text_raw[:match_gen.start()].strip()
            
            # 过滤逻辑：只要生成的足够长，就视为合格数据加入结果
            if len(tokenizer.encode(gen_text_clean, add_special_tokens=False)) >= min_required_tokens:
                results.append({
                    "prompt": batch_pairs[j][0],
                    "human_text": batch_pairs[j][1],
                    "no_wm_text": gen_text_clean,
                    "generation_time": avg_generation_time 
                })
                pbar.update(1)
                
                # 一旦达到目标数量，立刻停止
                if len(results) >= args.num_samples:
                    break

    pbar.close()

    # 安全截断，确保绝对不会超过指定的数量
    results = results[:args.num_samples]

    # 保存文件
    print(f"[*] 生成流结束，成功构建完美三元组数量: {len(results)}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"✅ 基准数据集已保存至: {args.output_file}")

if __name__ == "__main__":
    main()