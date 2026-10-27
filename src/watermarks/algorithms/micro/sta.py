# src/watermarks/algorithms/micro/sta.py
# -*- coding: utf-8 -*-

import gc
import os
import time
import torch
from types import SimpleNamespace

from src.watermarks.algorithms.base import BaseMicroAlgorithm
from src.baselines.sta_core.sample_watermark_processor import (
    SampleWatermark,
    SampleWatermarkDetector,
)


class STAMicroAlgorithm(BaseMicroAlgorithm):
    """
    STA 统一适配器：
    - single: generate / detect
    - dual:   generate_candidates / detect

    注意：
    STA 不是标准 logits_processor 路线，
    因此通过 custom candidate generation 干净接入 dual。
    """
    uses_custom_candidate_generation = True

    def __init__(self, args, tokenizer, model, device="cuda"):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        self.n_sample_per_token = 1
        self.n_sample_per_token_above_entropy_threshold = 4
        self.entropy_threshold = 1.35
        self.use_entropy_threshold = True
        self.gamma = 0.5
        self.sampling_temp = 1.0

        self.hash_key1 = 15485863
        self.hash_key2 = 17624813

        self.vocab_list = list(tokenizer.get_vocab().values())

        self.watermark = SampleWatermark(
            vocab=self.vocab_list,
            n_sample_per_token=self.n_sample_per_token,
            n_sample_per_token_above_entropy_threshold=self.n_sample_per_token_above_entropy_threshold,
            hash_key1=self.hash_key1,
            hash_key2=self.hash_key2
        )

        self.detector = SampleWatermarkDetector(
            vocab=self.vocab_list,
            n_sample_per_token=self.n_sample_per_token,
            z_threshold=2.0,
            gamma=self.gamma,
            hash_key1=self.hash_key1,
            hash_key2=self.hash_key2
        )

        self.profile_enabled = os.environ.get("DUAL_PROFILE_PROCESSOR", "0") == "1"
        self.call_count = 0
        self.total_time = 0.0

    # ------------------------------------------------------------------
    # profiling
    # ------------------------------------------------------------------
    def reset_profile(self):
        self.call_count = 0
        self.total_time = 0.0

    def get_profile(self) -> dict:
        avg = self.total_time / self.call_count if self.call_count > 0 else 0.0
        return {
            "name": "sta_custom_generator",
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time": avg,
        }

    # ------------------------------------------------------------------
    # 兼容旧接口：STA 不走 logits_processor
    # ------------------------------------------------------------------
    def build_processor(self):
        raise NotImplementedError(
            "STA 不走标准 logits_processor，请走 generate_candidates()。"
        )

    # ------------------------------------------------------------------
    # single 生成
    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        sta_args = SimpleNamespace(
            max_new_tokens=getattr(self.args, "max_total_tokens", 200),
            sampling_temp=self.sampling_temp,
            use_entropy_threshold=self.use_entropy_threshold,
            entropy_threshold=self.entropy_threshold,
            use_sampling=True,
            gamma=self.gamma,
            top_k=getattr(self.args, "top_k", 50),
            top_p=getattr(self.args, "top_p", 0.9)
        )

        with torch.no_grad():
            output_ids, _ = self.watermark.generate_with_watermark(
                input_ids,
                sta_args,
                self.model,
                self.device,
                self.tokenizer
            )

        generated_ids = output_ids[0, prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        del inputs
        del input_ids
        del output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_text

    # ------------------------------------------------------------------
    # dual 候选生成
    # ------------------------------------------------------------------
    def _generate_one_candidate(self, prefix_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        prefix_ids: shape [1, seq_len]
        返回 shape [1, seq_len + generated_len]
        """
        sta_args = SimpleNamespace(
            max_new_tokens=max_new_tokens,
            sampling_temp=self.sampling_temp,
            use_entropy_threshold=self.use_entropy_threshold,
            entropy_threshold=self.entropy_threshold,
            use_sampling=True,
            gamma=self.gamma,
            top_k=getattr(self.args, "top_k", 50),
            top_p=getattr(self.args, "top_p", 0.9)
        )

        with torch.no_grad():
            output_ids, _ = self.watermark.generate_with_watermark(
                prefix_ids.clone(),
                sta_args,
                self.model,
                self.device,
                self.tokenizer
            )
        return output_ids

    def generate_candidates(self, current_ids, max_new_tokens: int, num_candidates: int):
        """
        返回与 HF generate(num_return_sequences=N) 对齐的 tensor：
        shape [num_candidates, total_seq_len]
        """
        t0 = time.perf_counter()

        outputs = []
        for _ in range(num_candidates):
            one_out = self._generate_one_candidate(
                prefix_ids=current_ids,
                max_new_tokens=max_new_tokens
            )
            outputs.append(one_out[0])

        outputs = torch.stack(outputs, dim=0)

        dt = time.perf_counter() - t0
        if self.profile_enabled:
            self.call_count += 1
            self.total_time += dt

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return outputs

    # ------------------------------------------------------------------
    # detect
    # ------------------------------------------------------------------
    def detect(self, text: str) -> dict:
        if not text.strip():
            return {
                "z_score": 0.0,
                "z_score_micro": 0.0,
                "z_score_macro": 0.0,
                "details": {"reason": "too short"}
            }

        try:
            dummy_args = SimpleNamespace(prompt_max_length=8192)
            with torch.no_grad():
                z_score, p_value = self.detector.detect(
                    text,
                    dummy_args,
                    self.device,
                    self.tokenizer
                )
        except Exception:
            z_score = 0.0
            p_value = 1.0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        z = float(z_score)

        return {
            "z_score": z,
            "z_score_micro": z,
            "z_score_macro": 0.0,
            "details": {
                "p_value": float(p_value),
                "method": "sta"
            }
        }