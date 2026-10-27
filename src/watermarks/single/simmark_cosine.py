# src/watermarks/single/simmark_cosine.py
# -*- coding: utf-8 -*-

from src.watermarks.plugins import BaseWatermark, register_watermark
from src.watermarks.algorithms.macro.simmark_cosine import SimMarkCosineMacroAlgorithm


@register_watermark("simmark_cosine")
class SimMarkCosinePlugin(BaseWatermark):
    def __init__(self, args, tokenizer, model, embedder=None):
        super().__init__(args, tokenizer, model, embedder)
        self.algo = SimMarkCosineMacroAlgorithm(
            args=args,
            tokenizer=tokenizer,
            embedder=embedder,
            device=self.device,
            model=model
        )

    def generate(self, prompt: str) -> str:
        return self.algo.generate(prompt)

    def detect(self, text: str) -> dict:
        return self.algo.detect(text)