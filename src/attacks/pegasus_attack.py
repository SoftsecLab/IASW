# src/attacks/pegasus_attack.py
# -*- coding: utf-8 -*-

import re
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# 忽略 HuggingFace 的长度提示警告
warnings.filterwarnings("ignore")

class PegasusAttacker:
    def __init__(self, model_path: str, embedder_path: str, device: str = "cuda"):
        self.device = device
        
        print(f"[*] 加载 Pegasus 释义模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
        
        # 动态获取当前模型的安全最大长度限制 (修复 CUDA 越界问题的核心)
        max_pos = getattr(self.model.config, "max_position_embeddings", 512)
        max_len = getattr(self.model.config, "max_length", 512)
        self.safe_max_len = min(max_pos, max_len, 256) 
        print(f"[*] Pegasus 释义模型最大安全序列长度设定为: {self.safe_max_len}")
        
        print(f"[*] 加载语义裁判模型: {embedder_path}")
        self.embedder = SentenceTransformer(embedder_path, device=device)

    def split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?。！？\n])', text) if s.strip()]

    @torch.no_grad()
    def attack_sentence(self, sentence: str, num_beams: int, max_drop: float) -> str:
        # 使用动态 safe_max_len 进行截断
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=self.safe_max_len).to(self.device)
        
        # 【防御性拦截】：如果句子太长，接近了模型的上下文极限，
        # 强行生成会导致 CUDA 崩溃。此时我们直接放弃修改，原样返回该句。
        if inputs["input_ids"].shape[1] >= self.safe_max_len - 5:
            return sentence
            
        try:
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=self.safe_max_len,
                do_sample=False,
                # 🌟 [防退化补丁] 高压 Beam Search 下极其容易陷入死循环，必须加上严厉惩罚
                repetition_penalty=1.5,       
                no_repeat_ngram_size=3        
            )
        except Exception as e:
            # 捕获异常，清理显存，确保流水线不中断
            del inputs
            torch.cuda.empty_cache()
            return sentence
            
        candidates = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        
        # 🌟 [防OOM补丁 1] 宽束搜索生成完毕后，立刻释放极其庞大的 decoder 显存
        del inputs
        del outputs
        torch.cuda.empty_cache()
        
        # 语义过滤
        orig_emb = self.embedder.encode(sentence, convert_to_tensor=True)
        cand_embs = self.embedder.encode(candidates, convert_to_tensor=True)
        similarities = util.cos_sim(orig_emb, cand_embs)[0].tolist()
        
        best_candidate = sentence
        # 阈值计算：必须保留 1 - max_drop 的语义 (默认保留 90%)
        threshold = similarities[0] * (1.0 - max_drop)
        
        # 倒序遍历寻找改动最大（相似度最低）且符合阈值的句子
        for cand, sim in zip(reversed(candidates), reversed(similarities)):
            if sim >= threshold and cand.strip().lower() != sentence.strip().lower():
                best_candidate = cand
                break
                
        # 🌟 [防OOM补丁 2] 裁判打分完毕后，立刻释放 embedding 张量
        del orig_emb
        del cand_embs
        torch.cuda.empty_cache()
                
        return best_candidate

    def attack_text(self, text: str, num_beams: int = 25, max_drop: float = 0.1) -> str:
        """对外暴露的核心调用接口"""
        if not text: return ""
        sentences = self.split_sentences(text)
        attacked_sentences = []
        for sent in sentences:
            attacked_sentences.append(self.attack_sentence(sent, num_beams, max_drop))
        return " ".join(attacked_sentences)