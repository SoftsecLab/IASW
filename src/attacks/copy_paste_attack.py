# src/attacks/copy_paste_attack.py
# -*- coding: utf-8 -*-

import re
import random

def is_valid_sentence_fragment(s):
    s = s.strip()
    if not s:
        return False

    # 提取真正的英文单词
    words = re.findall(r"[A-Za-z]+", s)

    # 至少两个词，并且至少有一个不是单字母缩写
    if len(words) < 2:
        return False

    if max(len(w) for w in words) < 2:
        return False

    return True

class CopyPasteAttacker:
    def __init__(self):
        # 纯规则攻击，无需加载任何模型
        pass


    def split_sentences(self, text: str) -> list[str]:
        """
        按标点符号切分句子，并过滤纯标点/纯格式标记片段。

        Copy-Paste 攻击的单位应当是有效文本句，而不是 "."、".."
        或类似 "^{16}" 的格式碎片。否则 prepend 模式可能只截取到
        无语义标点，产生异常短的 attacked_text。
        """
        parts = [
            s.strip()
            for s in re.split(r'(?<=[.!?。！？\n])', text)
            # if s.strip()
            if is_valid_sentence_fragment(s)
        ]

        # 至少包含一个 Unicode 字母。
        # 可保留 García 等带重音字符，同时过滤纯标点和纯数字标记。
        valid_parts = [
            s for s in parts
            if any(ch.isalpha() for ch in s)
        ]

        return valid_parts

    def attack_text(self, wm_text: str, human_text: str, mix_ratio: float, mix_mode: str) -> str:
        """
        核心调用接口
        mix_ratio: 保留的水印文本比例
        mix_mode: prepend(前置), append(后置), random(随机), interleave(交替)
        """
        if not wm_text: return ""
        
        wm_sents = self.split_sentences(wm_text)
        human_sents = self.split_sentences(human_text) if human_text else []
        
        # 防御性补丁：即使没有足够的 human_text，也用占位符凑合，防止程序崩溃
        if not human_sents:
            human_sents = ["This is a dummy human sentence."] * len(wm_sents)

        # 计算要截取的句子数量
        keep_wm_count = max(1, int(len(wm_sents) * mix_ratio))
        # 动态计算需要多少人类句子来填补剩余的比例
        if mix_ratio >= 1.0:
            keep_human_count = 0
        else:
            keep_human_count = max(1, int(keep_wm_count * ((1 - mix_ratio) / mix_ratio)))
        
        # 截取相应的片段
        wm_part = wm_sents[:keep_wm_count]
        # 为了防止 human 文本不够长，通过列表乘法循环利用
        human_part = (human_sents * (keep_human_count // len(human_sents) + 1))[:keep_human_count]

        # 核心拼接逻辑
        if mix_mode == "prepend":
            # 人类文本在前，水印在后（极易导致水印索引向后偏移，制造严重全局失步）
            mixed_sents = human_part + wm_part
            
        elif mix_mode == "append":
            # 水印在前，人类文本在后（只会稀释浓度，不会导致开头水印失步）
            mixed_sents = wm_part + human_part
            
        elif mix_mode == "interleave":
            # 交替插入（最极端的碎片化失步测试）
            mixed_sents = [val for pair in zip(human_part, wm_part) for val in pair]
            # 补齐尾巴
            if len(human_part) > len(wm_part): 
                mixed_sents.extend(human_part[len(wm_part):])
            if len(wm_part) > len(human_part): 
                mixed_sents.extend(wm_part[len(human_part):])
                
        elif mix_mode == "random":
            # 随机混合
            mixed_sents = wm_part + human_part
            random.seed(42) # 固定随机种子以保证实验可复现
            random.shuffle(mixed_sents)
            
        else:
            mixed_sents = wm_part

        return " ".join(mixed_sents)