# src/watermarks/single/sweet.py
# -*- coding: utf-8 -*-

import gc
import torch
from transformers import LogitsProcessorList

from src.watermarks.plugins import BaseWatermark, register_watermark
from src.watermarks.algorithms.micro.sweet import SweetMicroAlgorithm


@register_watermark("sweet")
class SweetPlugin(BaseWatermark):
    def __init__(self, args, tokenizer, model, embedder=None):
        super().__init__(args, tokenizer, model, embedder)
        self.algo = SweetMicroAlgorithm(args, tokenizer, model, device=self.device)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=getattr(self.args, "max_total_tokens", 200),
                do_sample=True,
                temperature=0.7,
                top_k=getattr(self.args, "top_k", 50),
                logits_processor=LogitsProcessorList([self.algo.build_processor()]),
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)

        del inputs
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_text

    def detect(self, text: str) -> dict:
        return self.algo.detect(text)