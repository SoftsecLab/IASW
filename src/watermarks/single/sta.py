# src/watermarks/single/sta.py
# -*- coding: utf-8 -*-

from src.watermarks.plugins import BaseWatermark, register_watermark
from src.watermarks.algorithms.micro.sta import STAMicroAlgorithm


@register_watermark("sta")
class STAPlugin(BaseWatermark):
    def __init__(self, args, tokenizer, model, embedder=None):
        super().__init__(args, tokenizer, model, embedder)
        self.algo = STAMicroAlgorithm(args, tokenizer, model, device=self.device)

    def generate(self, prompt: str) -> str:
        return self.algo.generate(prompt)

    def detect(self, text: str) -> dict:
        return self.algo.detect(text)