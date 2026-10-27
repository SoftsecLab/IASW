# src/attacks/sentence_reorder_attack.py
# -*- coding: utf-8 -*-
"""句子重排序攻击 (Sentence Reorder / Shuffle Attack).

将水印文本按句子切分后，按比例打乱句子顺序，检验水印是否依赖
句子级位置/顺序约束。纯规则攻击，无需加载模型。

两种模式:
  - shuffle         : 按比例抽取句子子集做随机置换，其余原位保留；
  - adjacent_swap   : 按概率交换相邻句子对，扰动更温和、可读性更好。
"""

import re
import random


def is_valid_sentence_fragment(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    words = re.findall(r"[A-Za-z]+", s)
    if len(words) < 2:
        return False
    if max(len(w) for w in words) < 2:
        return False
    return True


class SentenceReorderAttacker:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(self.seed)

    def split_sentences(self, text: str) -> list[str]:
        parts = [
            s.strip()
            for s in re.split(r'(?<=[.!?。！？\n])', text)
            if is_valid_sentence_fragment(s)
        ]
        return [s for s in parts if any(ch.isalpha() for ch in s)]

    def _shuffle_subset(self, sents: list[str], ratio: float) -> list[str]:
        """抽取约 n*ratio 个位置做随机置换，其余原位保留。

        保证输出与原序不同：若 shuffle 后恰好还原（短文本上概率不低），
        对选中子集做一次循环移位，确保至少发生一次位置交换。
        """
        n = len(sents)
        k = max(1, int(n * ratio))
        k = min(k, n)
        if k < 2 and n >= 2:
            k = 2  # 至少交换一对，避免"抽了但没动"
        idx = list(range(n))
        chosen = self.rng.sample(idx, k)
        chosen_vals = [sents[i] for i in chosen]
        self.rng.shuffle(chosen_vals)
        out = list(sents)
        for pos, val in zip(chosen, chosen_vals):
            out[pos] = val
        # 防御：若还原成原序，循环移位一次，保证攻击确实发生
        if out == sents and k >= 2:
            chosen_vals = chosen_vals[1:] + chosen_vals[:1]
            for pos, val in zip(chosen, chosen_vals):
                out[pos] = val
        return out

    def _adjacent_swap(self, sents: list[str], ratio: float) -> list[str]:
        """按概率对每对相邻句子 (i, i+1) 做一次交换。"""
        out = list(sents)
        n = len(out)
        i = 0
        while i < n - 1:
            if self.rng.random() < ratio:
                out[i], out[i + 1] = out[i + 1], out[i]
                i += 2  # 不级联交换，避免同一句被反复挪动
            else:
                i += 1
        return out

    def attack_text(self, wm_text: str, human_text: str = "",
                    ratio: float = 0.5, mode: str = "shuffle") -> str:
        if not wm_text:
            return ""
        sents = self.split_sentences(wm_text)
        if len(sents) < 2:
            return wm_text.strip()

        if mode == "adjacent_swap":
            out = self._adjacent_swap(sents, ratio)
        else:
            out = self._shuffle_subset(sents, ratio)

        separator = "" if any('一' <= ch <= '鿿' for ch in wm_text) else " "
        return separator.join(out)
