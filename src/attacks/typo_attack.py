# src/attacks/typo_attack.py
# -*- coding: utf-8 -*-

import string
import random

class TypoAttacker:
    def __init__(self, seed: int = 42):
        self.seed = seed
        # 🌟 架构升级：使用独立的随机数生成器实例，绝对不污染全局
        self.rng = random.Random(self.seed)

    def attack_text(self, text: str, noise_ratio: float) -> str:
        """
        模拟人类键盘敲击错误，对文本进行四种随机破坏：
        1. 邻近字母乱序 (Swap)
        2. 随机字母替换 (Substitute)
        3. 字母遗漏 (Delete)
        4. 误触多打 (Insert)
        """
        if not text:
            return ""
            
        chars = list(text)
        num_noises = max(1, int(len(chars) * noise_ratio))
        
        # 找出所有字母的索引（尽量不破坏标点符号和空格，保证分句逻辑依然正常）
        alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]
        if not alpha_indices:
            return text
            
        # 随机抽取需要破坏的位置
        target_indices = self.rng.sample(alpha_indices, min(num_noises, len(alpha_indices)))
        
        for idx in target_indices:
            operation = self.rng.choice(['swap', 'substitute', 'delete', 'insert'])
            
            if operation == 'swap' and idx < len(chars) - 1 and chars[idx+1].isalpha():
                # 相邻字母打反了 (e.g., "hello" -> "helol")
                chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
                
            elif operation == 'substitute':
                # 敲错键盘 (用随机字母替换)
                chars[idx] = self.rng.choice(string.ascii_lowercase)
                
            elif operation == 'delete':
                # 漏打了一个字母
                chars[idx] = ''
                
            elif operation == 'insert':
                # 多按了一个字母
                chars[idx] = chars[idx] + self.rng.choice(string.ascii_lowercase)

        return "".join(chars)