# src/watermarks/algorithms/micro/kgw.py
# -*- coding: utf-8 -*-

import os
import gc
import torch

from src.baselines.kgw_core.watermark_processor import (
    WatermarkLogitsProcessor,
    WatermarkDetector,
)

from src.watermarks.algorithms.base import BaseMicroAlgorithm
from src.watermarks.runtime.profiling import ProfilingLogitsProcessor


class KGWMicroAlgorithm(BaseMicroAlgorithm):
    def __init__(self, args, tokenizer, model, device="cuda"):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        self.gamma = getattr(args, "gamma", 0.5)
        self.delta = getattr(args, "delta", 2.0)

        self.vocab_list = list(tokenizer.get_vocab().values())

        raw_processor = WatermarkLogitsProcessor(
            vocab=self.vocab_list,
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme="simple_1"
        )

        profile_enabled = os.environ.get("DUAL_PROFILE_PROCESSOR", "0") == "1"
        self.processor = ProfilingLogitsProcessor(
            raw_processor,
            name="kgw_processor",
            enabled=profile_enabled
        )

        self.detector = WatermarkDetector(
            vocab=self.vocab_list,
            gamma=self.gamma,
            seeding_scheme="simple_1",
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            normalizers=[]
        )

    def build_processor(self):
        return self.processor

    def reset_profile(self):
        self.processor.reset_profile()

    def get_profile(self) -> dict:
        return self.processor.get_profile()

    def detect(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "z_score": 0.0,
                "z_score_micro": 0.0,
                "z_score_macro": 0.0,
                "details": {"reason": "empty text"}
            }

        try:
            with torch.no_grad():
                res = self.detector.detect(text)
                z = float(res.get("z_score", 0.0))
        except Exception as e:
            z = 0.0
            res = {"error": str(e)}

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "z_score": z,
            "z_score_micro": z,
            "z_score_macro": 0.0,
            "details": res
        }