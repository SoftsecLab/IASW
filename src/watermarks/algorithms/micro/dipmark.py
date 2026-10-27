# src/watermarks/algorithms/micro/dipmark.py
# -*- coding: utf-8 -*-

import gc
import os
import torch
from math import sqrt
from transformers import LogitsProcessorList

from src.watermarks.algorithms.base import BaseMicroAlgorithm
from src.watermarks.runtime.profiling import ProfilingLogitsProcessor

from src.baselines.dipmark_core.transformers import WatermarkLogitsProcessor
from src.baselines.dipmark_core.dipmark import Dip_Reweight
from src.baselines.dipmark_core.contextcode import PrevN_ContextCodeExtractor


def compute_z_score(token_quantiles, T, gamma):
    observed_count = sum(1 for x in token_quantiles if x > gamma)
    numer = observed_count - gamma * T
    denom = sqrt(T * gamma * (1 - gamma))
    z_score = numer / denom if denom > 0 else 0.0
    return z_score


class DiPMarkMicroAlgorithm(BaseMicroAlgorithm):
    """
    DiPMark 特殊处理：
    - 不长期持有 processor
    - 每次 generate / detect / dual 调用时局部创建
    - 避免跨样本显存累计
    """
    def __init__(self, args, tokenizer, model, device="cuda"):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        self.alpha = 0.4
        self.gamma = 0.5
        self.prevN = 1
        self.hash_key = b"15485863"

        self.profile_enabled = os.environ.get("DUAL_PROFILE_PROCESSOR", "0") == "1"

        # 只保存统计，不保存 processor 实体
        self._profile_name = "dipmark_processor"
        self._profile_call_count = 0
        self._profile_total_time = 0.0

    def _make_raw_processor(self):
        return WatermarkLogitsProcessor(
            private_key=self.hash_key,
            reweight=Dip_Reweight(self.alpha),
            context_code_extractor=PrevN_ContextCodeExtractor(self.prevN)
        )

    def _make_processor(self):
        raw_processor = self._make_raw_processor()
        return ProfilingLogitsProcessor(
            raw_processor,
            name=self._profile_name,
            enabled=self.profile_enabled
        )

    # dual 用：每次返回一个新的 processor，避免跨样本累计
    def build_processor(self):
        return self._make_processor()

    def reset_profile(self):
        self._profile_call_count = 0
        self._profile_total_time = 0.0

    def get_profile(self) -> dict:
        avg = self._profile_total_time / self._profile_call_count if self._profile_call_count > 0 else 0.0
        return {
            "name": self._profile_name,
            "call_count": self._profile_call_count,
            "total_time": self._profile_total_time,
            "avg_time": avg,
        }

    # 给 dual 在每个 sample 结束后调用，把当次 processor 的 profile 累加进总统计
    def absorb_profile(self, processor):
        if hasattr(processor, "call_count"):
            self._profile_call_count += processor.call_count
        if hasattr(processor, "total_time"):
            self._profile_total_time += processor.total_time

    # single 用
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        local_processor = self._make_processor()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=getattr(self.args, "max_total_tokens", 200),
                do_sample=True,
                temperature=getattr(self.args, "temperature", 1.0),
                top_p=getattr(self.args, "top_p", 1.0),
                top_k=getattr(self.args, "top_k", 0),
                logits_processor=LogitsProcessorList([local_processor]),
                pad_token_id=self.tokenizer.eos_token_id
            )

        self.absorb_profile(local_processor)

        generated_text = self.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)

        del local_processor
        del inputs
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_text

    def detect(self, text: str) -> dict:
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(self.device)

        if input_ids.shape[1] < 10:
            return {
                "z_score": 0.0,
                "z_score_micro": 0.0,
                "z_score_macro": 0.0,
                "details": {"reason": "too short"}
            }

        local_processor = self._make_raw_processor()

        vocab_size = len(self.tokenizer.vocab)
        token_quantiles = []
        input_ids_for_quantile = torch.tensor([[]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            try:
                for i in range(input_ids.shape[1]):
                    current_token = input_ids[:, i]
                    current_token_quantile = local_processor.get_green_token_quantile(
                        input_ids_for_quantile, vocab_size, current_token
                    )
                    token_quantiles.append(current_token_quantile[0].item())
                    input_ids_for_quantile = torch.cat(
                        (input_ids_for_quantile, current_token.unsqueeze(0)),
                        dim=-1
                    )

                z_score = compute_z_score(token_quantiles, input_ids.shape[1], self.gamma)
            except Exception:
                z_score = 0.0

        del local_processor
        del input_ids
        del input_ids_for_quantile
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        z = float(z_score)

        return {
            "z_score": z,
            "z_score_micro": z,
            "z_score_macro": 0.0,
            "details": {
                "method": "dipmark",
                "alpha": self.alpha,
                "prevN": self.prevN
            }
        }