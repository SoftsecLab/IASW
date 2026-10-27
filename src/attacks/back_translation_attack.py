# src/attacks/back_translation_attack.py
# -*- coding: utf-8 -*-

import re
import torch
import warnings
from transformers import MarianMTModel, MarianTokenizer

warnings.filterwarnings("ignore")

class BackTranslationAttacker:
    def __init__(self, fwd_model_path: str, bwd_model_path: str, device: str = "cuda"):
        self.device = device
        
        # 懒加载阶段触发，仅在确定执行回译攻击时挂载模型
        self.fwd_tokenizer = MarianTokenizer.from_pretrained(fwd_model_path)
        self.fwd_model = MarianMTModel.from_pretrained(fwd_model_path).to(device).eval()
        
        self.bwd_tokenizer = MarianTokenizer.from_pretrained(bwd_model_path)
        self.bwd_model = MarianMTModel.from_pretrained(bwd_model_path).to(device).eval()

    def split_sentences(self, text: str) -> list[str]:
        """机器翻译对单句效果最好，按标点切分"""
        return [s.strip() for s in re.split(r'(?<=[.!?。！？\n])', text) if s.strip()]

    @torch.no_grad()
    def _translate(self, text: str, model, tokenizer) -> str:
        if not text: return ""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        try:
            translated = model.generate(
                **inputs, 
                max_length=512,
                # 🌟 [防退化补丁] 防止原始翻译模型在不熟悉的语境下重复输出
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        except Exception as e:
            # 防御性回退：如果生成失败则原样返回
            return text 

    def attack_sentence(self, sentence: str) -> str:
        if not sentence: return ""
        # 1. 正向翻译 (如 英 -> 法)
        pivot_text = self._translate(sentence, self.fwd_model, self.fwd_tokenizer)
        # 2. 反向翻译 (如 法 -> 英)
        back_translated = self._translate(pivot_text, self.bwd_model, self.bwd_tokenizer)
        return back_translated

    def attack_text(self, text: str) -> str:
        """对外暴露的核心调用接口"""
        if not text: return ""
        sentences = self.split_sentences(text)
        attacked_sentences = [self.attack_sentence(sent) for sent in sentences]
        return " ".join(attacked_sentences)