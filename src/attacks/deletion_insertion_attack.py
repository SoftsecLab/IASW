# src/attacks/deletion_insertion_attack.py
# -*- coding: utf-8 -*-

import re
import random

class DeletionInsertionAttacker:
    def __init__(self, seed: int = 42):
        self.seed = seed
        # 🌟 架构升级：使用独立的随机数生成器实例，避免污染全局 random 状态
        self.rng = random.Random(self.seed)

    def split_sentences(self, text: str) -> list[str]:
        """按标点符号切分句子"""
        return [s.strip() for s in re.split(r'(?<=[.!?。！？\n])', text) if s.strip()]

    def attack_text(self, wm_text: str, human_text: str, drop_prob: float, insert_prob: float) -> str:
        """
        核心调用接口
        drop_prob: 删除句子概率
        insert_prob: 插入人类句子概率
        """
        if not wm_text: return ""
        
        wm_sents = self.split_sentences(wm_text)
        human_sents = self.split_sentences(human_text) if human_text else []
        
        # 建立噪音池
        noise_pool = human_sents if human_sents else ["Furthermore, we should consider this point."]
        
        attacked_sentences = []
        
        for sent in wm_sents:
            # 1. Deletion Attack (删除)
            if self.rng.random() < drop_prob:
                continue # 丢弃该句
                
            attacked_sentences.append(sent)
            
            # 2. Insertion Attack (插入噪音)
            if self.rng.random() < insert_prob:
                attacked_sentences.append(self.rng.choice(noise_pool))
                
        # 判断语言环境决定拼接符 (简单以是否包含中文字符来判断)
        separator = "" if any('\u4e00' <= char <= '\u9fff' for char in wm_text) else " "
        return separator.join(attacked_sentences)