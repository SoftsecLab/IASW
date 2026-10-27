# src/watermarks/algorithms/macro/simmark_euclidean.py
# -*- coding: utf-8 -*-

import os
import re
import pickle
import torch
from transformers import GenerationConfig
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import euclidean_distances

from src.watermarks.algorithms.base import BaseMacroAlgorithm

from src.baselines.simmark_core.sampling_utils import (
    DummyPCA,
    euclidean_reject_completion,
)
from src.baselines.simmark_core.detection import detect, calculate_embeddings
from src.baselines.simmark_core import sampling_utils


class SimMarkEuclideanMacroAlgorithm(BaseMacroAlgorithm):
    """
    SimMark Euclidean 统一适配器：
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
        print("[*] [SimMark Euclidean] 强制加载官方专属 hkunlp/instructor-large 模型...")
        self.embedder = SentenceTransformer("hkunlp/instructor-large").to(self.device)

        ds_params = {
            "c4":    {"a": 0.26, "b": 0.34, "gamma": 0.2484},
            "wiki":  {"a": 0.24, "b": 0.32, "gamma": 0.2418},
            "dolly": {"a": 0.23, "b": 0.31, "gamma": 0.2463},
        }

        current_ds = getattr(args, "dataset_type", "c4").lower()
        p = ds_params.get(current_ds, ds_params["c4"])

        self.mode = "euclidean"
        self.a = p["a"]
        self.b = p["b"]
        self.whole_interval = (self.a, self.b)
        self.K = getattr(args, "simmark_K", 250)
        self.gamma = p["gamma"]

        self.use_pca = True

        if self.use_pca:
            self.instruction = "Represent the sentence for PCA:"
            self.pca_path = f"src/baselines/simmark_core/{current_ds}_pca_model_16.pkl"

            if not os.path.exists(self.pca_path):
                print(f"⚠️ 找不到专属 PCA 模型 {self.pca_path}，回退到通用模型")
                self.pca_path = "src/baselines/simmark_core/pca_model_16.pkl"

            with open(self.pca_path, "rb") as f:
                self.pca_model = pickle.load(f)
        else:
            self.pca_model = DummyPCA()
            self.instruction = "Represent the sentence for euclidean distance:"

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
            raise ValueError("SimMarkEuclidean generate() requires a language model, but model=None was provided.")

        sampling_utils.MAX_TRIALS = getattr(self.args, "max_trials", 100)

        # 先补全半句 prompt
        full_prompt, completion = self._complete_partial_sentence(prompt)

        # 取最后一个完整句，作为 SimMark 的实际语义水印上下文
        prompt_sentences = sent_tokenize(full_prompt)
        last_sent = prompt_sentences[-1] if prompt_sentences else full_prompt

        # 把“补句消耗”的 token 从总预算里扣掉
        max_total_tokens = getattr(self.args, "max_total_tokens", 200)
        completion_tokens = len(
            self.tokenizer.encode(completion, add_special_tokens=False)
        )
        remaining_tokens = max(10, max_total_tokens - completion_tokens)

        gen_config = GenerationConfig(
            max_new_tokens=remaining_tokens,
            do_sample=True,
            temperature=getattr(self.args, "temperature", 0.7),
            top_k=getattr(self.args, "top_k", 50),
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

        response = euclidean_reject_completion(
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

        dists = euclidean_distances(prev_embed_pca, cands_embed_pca)[0]

        best_candidate = candidates[0]
        for i, dist in enumerate(dists):
            if self.a <= dist <= self.b:
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
            "details": {"method": "simmark_euclidean"}
        }