# src/attacks/dipper_attack.py
# -*- coding: utf-8 -*-

import re
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")

class DipperAttacker:
    def __init__(self, model_path: str = "kalpeshk2011/dipper-paraphraser-xxl", device: str = "cuda"):
        self.device = device
        
        # 懒加载阶段触发
        print(f"[*] 加载 DIPPER 释义模型 (T5-XXL): {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # DIPPER 基于 T5 架构，必须采用 bf16 加载以防止 OOM
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        ).to(device).eval()

    def split_sentences(self, text: str) -> list[str]:
        """按标点符号切分句子，DIPPER 在句子/段落级别工作得最好"""
        return [s.strip() for s in re.split(r'(?<=[.!?。！？\n])', text) if s.strip()]

    @torch.no_grad()
    def attack_sentence(self, sentence: str, lex_div: int, order_div: int) -> str:
        if not sentence: return ""
        
        # DIPPER 的专属 Prompt 格式
        prompt = f"lexical = {lex_div}, order = {order_div} {sentence}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        try:
            outputs = self.model.generate(
                **inputs,
                do_sample=True,          # DIPPER 推荐使用采样来增加多样性
                top_p=0.75,
                top_k=None,
                max_length=512,
            )
            attacked_sent = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 🌟 [防 OOM 补丁] 超大模型单次生成后必须立刻清理局部缓存
            del inputs
            del outputs
            torch.cuda.empty_cache()
            
            return attacked_sent
            
        except Exception as e:
            # 防御性回退：如果生成失败，清理显存并返回原句
            del inputs
            torch.cuda.empty_cache()
            return sentence

    def attack_text(self, text: str, lex_div: int = 60, order_div: int = 60) -> str:
        """
        对外暴露的核心调用接口
        lex_div: 词汇多样性 (0-100)
        order_div: 语序多样性 (0-100)
        """
        if not text: return ""
        sentences = self.split_sentences(text)
        attacked_sentences = [self.attack_sentence(sent, lex_div, order_div) for sent in sentences]
        return " ".join(attacked_sentences)