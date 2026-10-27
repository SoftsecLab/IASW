# src/attacks/perturbation_attack.py
# -*- coding: utf-8 -*-

import random

class PerturbationAttacker:
    def __init__(self, seed: int = 42):
        self.seed = seed
        # 🌟 架构升级：使用独立的随机数生成器实例，避免污染全局 random 状态
        self.rng = random.Random(self.seed)

    def attack_drop(self, text: str, ratio: float) -> str:
        """
        随机删词攻击 (Random Deletion)
        ratio: 删词比例 (0~1)
        """
        if not text: return ""
        words = text.split()
        if not words: return text
        
        # 按照 (1 - ratio) 的概率保留单词
        kept_words = [w for w in words if self.rng.random() > ratio]
        
        # 防御性回退：如果删得一个词都不剩了，至少保留第一个词
        if not kept_words:
            kept_words = words[:1]
            
        return " ".join(kept_words)

    def attack_swap(self, text: str, ratio: float) -> str:
        """
        随机交换攻击 (Random Swap)
        ratio: 换位比例 (0~1)
        """
        if not text: return ""
        words = text.split()
        n = len(words)
        if n < 2: return text
        
        # 每进行一次 swap 操作，会影响 2 个单词，所以交换次数是总词数 * ratio / 2
        num_swaps = int(n * ratio / 2) 
        num_swaps = max(1, num_swaps) if ratio > 0 else 0
        
        for _ in range(num_swaps):
            idx1, idx2 = self.rng.sample(range(n), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return " ".join(words)