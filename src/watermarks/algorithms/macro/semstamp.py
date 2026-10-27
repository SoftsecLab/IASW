# src/watermarks/algorithms/macro/semstamp.py
# -*- coding: utf-8 -*-

import re
import torch
import nltk
from transformers import GenerationConfig

from src.watermarks.algorithms.base import BaseMacroAlgorithm

from src.baselines.semstamp_core.sbert_lsh_model import SBERTLSHModel
from src.baselines.semstamp_core.sampling_lsh_utils import (
    lsh_reject_completion,
    get_mask_from_seed,
    reject_close_generation,
)
from src.baselines.semstamp_core.detection_utils import detect_lsh


class SemStampMacroAlgorithm(BaseMacroAlgorithm):
    """
    SemStamp 统一适配器：
    - single 模式：generate / detect
    - dual 模式：init_context / select_best_candidate / detect
    """
    needs_dual_calibration = True

    def __init__(self, args, tokenizer, embedder=None, device="cuda", model=None):
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.model = model

        self.gamma = 0.25
        self.margin = 0.02
        self.lsh_dim = 3

        official_embedder_path = "AbeHou/SemStamp-c4-sbert"
        print(f"[*] [SemStamp] 强制加载官方微调 LSH 模型: {official_embedder_path}")

        self.lsh_model = SBERTLSHModel(
            lsh_model_path=official_embedder_path,
            device=self.device,
            batch_size=1,
            lsh_dim=self.lsh_dim
        )

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    # ------------------------------------------------------------------
    # single 生成逻辑
    # ------------------------------------------------------------------
    def _complete_partial_sentence(self, prompt: str) -> tuple[str, str]:
        if re.search(r"[.!?。！？\n]\s*$", prompt):
            return prompt, ""

        if self.model is None:
            raise ValueError("SemStamp generate() requires a language model, but model=None was provided.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=40,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True
            )

        generated_str = self.tokenizer.decode(
            outputs.sequences[0, prompt_len:],
            skip_special_tokens=True
        )

        match = re.search(r"[.!?。！？\n]", generated_str)
        if match:
            completion = generated_str[:match.end()]
        else:
            completion = generated_str + "."

        del outputs
        return prompt + completion, completion

    def generate(self, prompt: str) -> str:
        if self.model is None:
            raise ValueError("SemStamp generate() requires a language model, but model=None was provided.")

        full_prompt, completion = self._complete_partial_sentence(prompt)

        max_total_tokens = getattr(self.args, "max_total_tokens", 200)
        completion_tokens = len(self.tokenizer.encode(completion, add_special_tokens=False))
        remaining_tokens = max(10, max_total_tokens - completion_tokens)

        gen_config = GenerationConfig(
            max_new_tokens=remaining_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_text = lsh_reject_completion(
            prompt=full_prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            gen_config=gen_config,
            lsh_model=self.lsh_model,
            lsh_dim=self.lsh_dim,
            lmbd=self.gamma,
            margin=self.margin,
            verbose=False,
            device=self.device
        )

        watermarked_text = output_text[len(full_prompt):].strip()
        return completion + (" " if completion and not completion.endswith("\n") else "") + watermarked_text

    # ------------------------------------------------------------------
    # dual 宏观逻辑
    # ------------------------------------------------------------------
    def init_context(self, prompt: str):
        prompt = prompt.strip()
        if prompt and prompt[-1] not in ["!", ".", "?", "！", "。", "？"]:
            prompt += "."

        prompt_sentences = nltk.sent_tokenize(prompt)
        last_prompt_sent = prompt_sentences[-1] if prompt_sentences else prompt

        prev_seed = self.lsh_model.get_hash([last_prompt_sent])[0]
        return prev_seed

    def select_best_candidate(self, candidates: list, prev_seed: int) -> tuple:
        mask = get_mask_from_seed(self.lsh_dim, self.gamma, prev_seed)

        accepted_texts, _ = reject_close_generation(
            self.lsh_model,
            candidates,
            margin=self.margin,
            cutoff=None
        )

        if len(accepted_texts) > 0:
            for text in accepted_texts:
                seed = self.lsh_model.get_hash([text])[0]
                if seed in mask:
                    return text, seed

        best_sentence = candidates[0]
        best_seed = self.lsh_model.get_hash([best_sentence])[0]
        return best_sentence, best_seed

    # ------------------------------------------------------------------
    # 检测逻辑（single / dual 共用）
    # ------------------------------------------------------------------
    def detect(self, text: str) -> dict:
        sentences = nltk.sent_tokenize(text.strip())

        if len(sentences) <= 1:
            return {
                "z_score": 0.0,
                "z_score_micro": 0.0,
                "z_score_macro": 0.0,
                "details": {"reason": "too short"}
            }

        zscore = detect_lsh(
            sents=sentences,
            lsh_model=self.lsh_model,
            lmbd=self.gamma,
            lsh_dim=self.lsh_dim
        )

        z = float(zscore)

        return {
            "z_score": z,
            "z_score_micro": 0.0,
            "z_score_macro": z,
            "details": {"method": "semstamp"}
        }