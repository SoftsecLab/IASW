# src/watermarks/algorithms/micro/sweet.py
# -*- coding: utf-8 -*-

import os
import gc
import torch

from src.baselines.sweet_core.sweet import SweetLogitsProcessor, SweetDetector

from src.watermarks.algorithms.base import BaseMicroAlgorithm
from src.watermarks.runtime.profiling import ProfilingLogitsProcessor


class SweetMicroAlgorithm(BaseMicroAlgorithm):
    def __init__(self, args, tokenizer, model, device="cuda"):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        self.gamma = getattr(args, "gamma", 0.5)
        self.delta = getattr(args, "delta", 2.0)
        self.entropy_threshold = getattr(args, "entropy_threshold", 0.9)
        self.max_detect_tokens = getattr(args, "max_sweet_detect_tokens", 1024)

        self.vocab_list = list(tokenizer.get_vocab().values())

        raw_processor = SweetLogitsProcessor(
            vocab=self.vocab_list,
            gamma=self.gamma,
            delta=self.delta,
            entropy_threshold=self.entropy_threshold
        )

        profile_enabled = os.environ.get("DUAL_PROFILE_PROCESSOR", "0") == "1"
        self.processor = ProfilingLogitsProcessor(
            raw_processor,
            name="sweet_processor",
            enabled=profile_enabled
        )

        self.detector = SweetDetector(
            vocab=self.vocab_list,
            gamma=self.gamma,
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            entropy_threshold=self.entropy_threshold
        )

    def build_processor(self):
        return self.processor

    def reset_profile(self):
        self.processor.reset_profile()

    def get_profile(self) -> dict:
        return self.processor.get_profile()
    def detect(self, text: str) -> dict:
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]

        original_num_tokens = len(input_ids)

        # ==================================================
        # 防止超长攻击文本导致 SWEET 检测阶段 OOM
        # 默认截断到 1024 tokens
        # ==================================================
        if original_num_tokens > self.max_detect_tokens:
            input_ids = input_ids[:self.max_detect_tokens]

        truncated_num_tokens = len(input_ids)

        if truncated_num_tokens < 10:
            return {
                "z_score": 0.0,
                "z_score_micro": 0.0,
                "z_score_macro": 0.0,
                "details": {
                    "reason": "too short",
                    "original_num_tokens": original_num_tokens,
                    "truncated_num_tokens": truncated_num_tokens,
                    "max_detect_tokens": self.max_detect_tokens,
                    "truncated": original_num_tokens > self.max_detect_tokens,
                }
            }

        self.model.eval()

        output = None
        probs = None
        ent = None

        try:
            with torch.no_grad():
                output = self.model(
                    torch.unsqueeze(input_ids.to(self.device), 0),
                    return_dict=True
                )

                probs = torch.softmax(output.logits, dim=-1)

                ent = -torch.where(
                    probs > 0,
                    probs * probs.log(),
                    probs.new([0.0])
                ).sum(dim=-1)

                entropy_list = ent[0].cpu().tolist()

            entropy = [0.0] + entropy_list[:-1]
            tokenized_prefix = torch.tensor([], dtype=torch.long)

            res = self.detector.detect(
                tokenized_text=input_ids,
                tokenized_prefix=tokenized_prefix,
                entropy=entropy
            )

            z_score = res.get("z_score", 0.0)
            if z_score == -100.0:
                z_score = 0.0

            z = float(z_score)

            return {
                "z_score": z,
                "z_score_micro": z,
                "z_score_macro": 0.0,
                "details": {
                    "watermarking_fraction": res.get("watermarking_fraction", 0),
                    "num_tokens_scored": res.get("num_tokens_scored", 0),
                    "num_green_tokens": res.get("num_green_tokens", 0),
                    "method": "sweet",
                    "original_num_tokens": original_num_tokens,
                    "truncated_num_tokens": truncated_num_tokens,
                    "max_detect_tokens": self.max_detect_tokens,
                    "truncated": original_num_tokens > self.max_detect_tokens,
                }
            }

        finally:
            del output
            del probs
            del ent
            del input_ids

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
