# src/watermarks/algorithms/macro/ksemstamp.py
# -*- coding: utf-8 -*-

import os
import re
import torch
import nltk
from transformers import GenerationConfig
from sentence_transformers import SentenceTransformer

from src.watermarks.algorithms.base import BaseMacroAlgorithm

from src.baselines.semstamp_core.sampling_kmeans_utils import (
    kmeans_reject_completion,
    get_cluster_id,
    get_cluster_mask,
    kmeans_reject_overlap,
)
from src.baselines.semstamp_core.detection_utils import detect_kmeans


class KSemStampMacroAlgorithm(BaseMacroAlgorithm):
    """
    k-SemStamp 统一适配器：
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
        self.k_dim = 8
        self.margin = 0.02

        current_ds = getattr(args, "dataset_type", "c4").lower()
        self.cc_path = f"src/baselines/semstamp_core/{current_ds}-cluster_8_centers.pt"

        if not os.path.exists(self.cc_path):
            raise FileNotFoundError(
                f"[k-SemStamp] 找不到聚类中心文件，请检查路径: {self.cc_path}"
            )

        print(f"[*] [k-SemStamp] 加载聚类中心: {self.cc_path}")
        self.cluster_centers = torch.load(self.cc_path, map_location="cpu")

        official_embedder_path = "AbeHou/SemStamp-c4-sbert"
        print(f"[*] [k-SemStamp] 强制加载官方专属特征提取器: {official_embedder_path}")
        self.embedder = SentenceTransformer(official_embedder_path).to(self.device)

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
            raise ValueError("KSemStamp generate() requires a language model, but model=None was provided.")

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
            raise ValueError("KSemStamp generate() requires a language model, but model=None was provided.")

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

        output_text = kmeans_reject_completion(
            prompt=full_prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            gen_config=gen_config,
            embedder=self.embedder,
            k_dim=self.k_dim,
            lmbd=self.gamma,
            cluster_centers=self.cluster_centers,
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

        curr_cluster_id = get_cluster_id(last_prompt_sent, self.cluster_centers, self.embedder)
        return curr_cluster_id

    def select_best_candidate(self, candidates: list, prev_cluster_id: int) -> tuple:
        mask = get_cluster_mask(prev_cluster_id, self.k_dim, self.gamma)

        best_sentence = candidates[0]
        best_cluster_id = get_cluster_id(best_sentence, self.cluster_centers, self.embedder)

        for cand in candidates:
            accepted_text, cluster_id = kmeans_reject_overlap(
                cand,
                self.embedder,
                self.cluster_centers,
                self.margin
            )
            if accepted_text is not None and cluster_id in mask:
                best_sentence = accepted_text
                best_cluster_id = cluster_id
                break

        return best_sentence, best_cluster_id

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

        zscore = detect_kmeans(
            sents=sentences,
            embedder=self.embedder,
            lmbd=self.gamma,
            k_dim=self.k_dim,
            cluster_centers=self.cluster_centers
        )

        z = float(zscore)

        return {
            "z_score": z,
            "z_score_micro": 0.0,
            "z_score_macro": z,
            "details": {"method": "ksemstamp"}
        }