# src/watermarks/single/semstamp.py
# -*- coding: utf-8 -*-

from src.watermarks.plugins import BaseWatermark, register_watermark
from src.watermarks.algorithms.macro.semstamp import SemStampMacroAlgorithm


@register_watermark("semstamp")
class SemStampPlugin(BaseWatermark):
    def __init__(self, args, tokenizer, model, embedder=None):
        super().__init__(args, tokenizer, model, embedder)
        self.algo = SemStampMacroAlgorithm(
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