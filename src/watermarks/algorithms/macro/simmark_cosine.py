# src/watermarks/algorithms/macro/simmark_cosine.py
# -*- coding: utf-8 -*-

import re
import torch
from transformers import GenerationConfig
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from src.watermarks.algorithms.base import BaseMacroAlgorithm

from src.baselines.simmark_core.sampling_utils import (
    DummyPCA,
    cosine_reject_completion,
)
from src.baselines.simmark_core.detection import detect, calculate_embeddings
from src.baselines.simmark_core import sampling_utils


class SimMarkCosineMacroAlgorithm(BaseMacroAlgorithm):
    """
    SimMark Cosine 统一适配器：
    - single: generate / detect
    - dual:   init_context / select_best_candidate / detect
    """
    needs_dual_calibration = True

    def __init__(self, args, tokenizer, embedder=None, device="cuda", model=None):
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.model = model

        from sentence_transformers import SentenceTransformer
        print("[*] [SimMark Cosine] 强制加载官方专属 hkunlp/instructor-large 模型...")
        self.embedder = SentenceTransformer("hkunlp/instructor-large").to(self.device)

        ds_params = {
            "c4":    {"a": 0.68, "b": 0.76, "gamma": 0.1773},
            "wiki":  {"a": 0.70, "b": 0.78, "gamma": 0.1956},
            "dolly": {"a": 0.71, "b": 0.79, "gamma": 0.1941},
        }

        current_ds = getattr(args, "dataset_type", "c4").lower()
        p = ds_params.get(current_ds, ds_params["c4"])

        self.mode = "cosine"
        self.a = p["a"]
        self.b = p["b"]
        self.whole_interval = (self.a, self.b)
        self.K = getattr(args, "simmark_K", 250)
        self.gamma = p["gamma"]

        self.pca_model = DummyPCA()
        self.instruction = "Represent the sentence for cosine similarity:"

    # ------------------------------------------------------------------
    # single 生成逻辑
    # ------------------------------------------------------------------
    def _complete_partial_sentence(self, prompt: str) -> tuple[str, str]:
        if re.search(r"[.!?。！？\n]\s*$", prompt):
            return prompt, ""

        if self.model is None:
            raise ValueError("SimMarkCosine generate() requires a language model, but model=None was provided.")

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return prompt + completion, completion

    def generate(self, prompt: str) -> str:
        if self.model is None:
            raise ValueError("SimMarkCosine generate() requires a language model, but model=None was provided.")

        sampling_utils.MAX_TRIALS = getattr(self.args, "max_trials", 100)

        full_prompt, completion = self._complete_partial_sentence(prompt)
        prompt_sentences = sent_tokenize(full_prompt)
        last_sent = prompt_sentences[-1] if prompt_sentences else full_prompt

        max_total_tokens = getattr(self.args, "max_total_tokens", 200)
        completion_tokens = len(self.tokenizer.encode(completion, add_special_tokens=False))
        remaining_tokens = max(10, max_total_tokens - completion_tokens)

        gen_config = GenerationConfig(
            max_new_tokens=remaining_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

        response = cosine_reject_completion(
            prompt=last_sent,
            model=self.model,
            tokenizer=self.tokenizer,
            gen_config=gen_config,
            embedder=self.embedder,
            whole_interval=self.whole_interval,
            pca_model=self.pca_model,
            instruction=self.instruction,
            verbose=False,
            device=self.device
        )

        watermarked_text = response[len(last_sent):].strip()
        return completion + (" " if completion and not completion.endswith("\n") else "") + watermarked_text

    # ------------------------------------------------------------------
    # dual 宏观逻辑
    # ------------------------------------------------------------------
    def init_context(self, prompt: str):
        prompt = prompt.strip()
        if prompt and prompt[-1] not in ["!", ".", "?", "！", "。", "？"]:
            prompt += "."
        prompt_sentences = sent_tokenize(prompt)
        return prompt_sentences[-1] if prompt_sentences else prompt

    def select_best_candidate(self, candidates: list, context: str) -> tuple:
        prev_sent = context

        prev_embed = self.embedder.encode(
            prev_sent,
            prompt=self.instruction,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).reshape(1, -1)

        cands_embed = self.embedder.encode(
            candidates,
            prompt=self.instruction,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        prev_embed_pca = self.pca_model.transform(prev_embed.cpu().detach().numpy())
        cands_embed_pca = self.pca_model.transform(cands_embed.cpu().detach().numpy())

        sims = cosine_similarity(prev_embed_pca, cands_embed_pca)[0]

        best_candidate = candidates[0]
        for i, sim in enumerate(sims):
            if self.a <= sim <= self.b:
                best_candidate = candidates[i]
                break

        return best_candidate, best_candidate

    # ------------------------------------------------------------------
    # 检测逻辑（single / dual 共用）
    # ------------------------------------------------------------------
    def detect(self, text: str) -> dict:
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return {
                "z_score": 0.0,
                "z_score_micro": 0.0,
                "z_score_macro": 0.0,
                "details": {"reason": "too short"}
            }

        embeddings_list = calculate_embeddings(
            [text],
            self.embedder,
            instruction=self.instruction,
            fix_space=True
        )

        zscore = detect(
            sents=sentences,
            mode=self.mode,
            embeddings=embeddings_list[0],
            pca_model=self.pca_model,
            gamma=self.gamma,
            skip_count=1,
            alt_sent=False,
            fix_space=True,
            whole_interval=self.whole_interval,
            verbose=False,
            K=self.K
        )

        z = float(zscore)

        return {
            "z_score": z,
            "z_score_micro": 0.0,
            "z_score_macro": z,
            "details": {"method": "simmark_cosine"}
        }