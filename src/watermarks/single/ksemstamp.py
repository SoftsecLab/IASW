# src/watermarks/single/ksemstamp.py
# -*- coding: utf-8 -*-

from src.watermarks.plugins import BaseWatermark, register_watermark
from src.watermarks.algorithms.macro.ksemstamp import KSemStampMacroAlgorithm


@register_watermark("ksemstamp")
class KSemStampPlugin(BaseWatermark):
    def __init__(self, args, tokenizer, model, embedder=None):
        super().__init__(args, tokenizer, model, embedder)
        self.algo = KSemStampMacroAlgorithm(
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