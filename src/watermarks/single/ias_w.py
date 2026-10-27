# src/watermarks/single/ias_w.py
# -*- coding: utf-8 -*-

"""
Single-channel IAS-W watermark plugin.

This plugin registers method name "iasw".

Usage example:
    export IASW_STATS_FILE=/path/to/ias_stats_xxx_rank64_xxx.pt
    export IASW_STATS_KEY=rank64_eps0p0001_calibskip3000_calibn1000
    export IASW_ALIGNMENT_MODE=dp_local
    export IASW_TARGET_LEN=200

    python src/watermarks/runner.py \
        --task generate \
        --method iasw \
        --baseline_file ... \
        --target_file ... \
        --dataset_type c4 \
        --model_path ... \
        --embedder_path ... \
        --max_total_tokens 200 \
        --max_sent_tokens 40 \
        --num_candidates 32
"""

import gc
import os
import re
import nltk
import torch

from src.watermarks.plugins import BaseWatermark, register_watermark
from src.watermarks.algorithms.macro.ias_w import IASWMacroAlgorithm


@register_watermark("iasw")
class IASWPlugin(BaseWatermark):
    def __init__(self, args, tokenizer, model, embedder=None):
        super().__init__(args, tokenizer, model, embedder)

        self.num_candidates = getattr(args, "num_candidates", 32)
        self.max_sent_tokens = getattr(args, "max_sent_tokens", 40)
        self.max_total_tokens = getattr(args, "max_total_tokens", 200)

        # IAS-W runtime config is mainly controlled by:
        #   IASW_STATS_FILE (legacy alias: IAS_STATS_FILE)
        #   IASW_STATS_KEY  (legacy alias: IAS_STATS_KEY)
        #   IASW_ALIGNMENT_MODE (legacy alias: IAS_ALIGNMENT_MODE)
        #
        # IASWMacroAlgorithm internally builds IASWMark.
        self.algo = IASWMacroAlgorithm(
            args=args,
            tokenizer=tokenizer,
            embedder=embedder,
            device=self.device,
        )

        self._print_runtime_config()

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def _print_runtime_config(self):
        """Print IAS-W runtime information for debugging."""
        stats_file = (
            os.environ.get("IASW_STATS_FILE")
            or os.environ.get("IAS_STATS_FILE")
            or getattr(self.args, "iasw_stats_file", "")
            or getattr(self.args, "ias_stats_file", "")
        )

        stats_key = (
            os.environ.get("IASW_STATS_KEY")
            or os.environ.get("IAS_STATS_KEY")
            or getattr(self.args, "iasw_stats_key", "")
            or getattr(self.args, "ias_stats_key", "")
        )

        alignment_mode = (
            os.environ.get("IASW_ALIGNMENT_MODE")
            or os.environ.get("IAS_ALIGNMENT_MODE")
            or getattr(self.args, "ias_alignment_mode", "dp_local")
        )

        print("[*] Single IAS-W plugin initialized")
        print(f"    num_candidates  = {self.num_candidates}")
        print(f"    max_sent_tokens = {self.max_sent_tokens}")
        print(f"    max_total_tokens= {self.max_total_tokens}")
        print(f"    stats_key       = {stats_key if stats_key else '<registry/default>'}")
        print(f"    stats_file      = {stats_file if stats_file else '<registry/default>'}")
        print(f"    alignment_mode  = {alignment_mode}")

    def _ensure_sentence_ended_prompt(self, prompt: str) -> str:
        """
        Stage A:
        If the prompt is an incomplete sentence, generate a short continuation
        to complete the current sentence boundary.
        """
        if re.search(r"[.!?。！？\n]\s*$", prompt):
            return prompt

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
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
                return_dict_in_generate=True,
            )

        prompt_len = input_ids.shape[1]
        generated_str = self.tokenizer.decode(
            outputs.sequences[0, prompt_len:],
            skip_special_tokens=True,
        )

        match = re.search(r"[.!?。！？\n]", generated_str)
        if match:
            prompt = prompt + generated_str[: match.end()]
        else:
            prompt = prompt + generated_str + "."

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return prompt

    def generate(self, prompt: str) -> str:
        """
        Single-channel IAS-W generation.

        For each sentence:
            1. sample multiple candidate sentences from the LLM;
            2. encode and whiten candidate sentence embeddings in IASWMark;
            3. select the candidate best aligned with the secret anchor;
            4. append the selected sentence to the generated text.
        """
        # Stage A: complete the residual prompt sentence if needed.
        prompt = self._ensure_sentence_ended_prompt(prompt)

        # Stage B: sentence-level IAS-W candidate selection.
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        current_seq = input_ids

        generated_text = ""
        sentence_idx = 0

        while current_seq.shape[1] - input_ids.shape[1] < self.max_total_tokens:
            input_len = current_seq.shape[1]
            attention_mask = torch.ones_like(current_seq)

            with torch.no_grad():
                outputs = self.model.generate(
                    current_seq,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_sent_tokens,
                    num_return_sequences=self.num_candidates,
                    do_sample=True,
                    temperature=0.7,
                    top_k=getattr(self.args, "top_k", 50),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            candidates = []

            for out in outputs:
                new_tokens = out[input_len:]
                new_text = self.tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                if len(new_text)>5000:
                    print(
                        "[BAD DECODE]",
                        len(new_text),
                        new_text[:300]
                    )

                sents = nltk.sent_tokenize(new_text.strip())
                first_sent = sents[0] if sents else new_text
                first_sent = first_sent.strip()

                if first_sent:
                    candidates.append(first_sent)

            has_eos = self.tokenizer.eos_token_id in outputs[0][input_len:]

            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Remove duplicate candidates while preserving order.
            seen = set()
            dedup_candidates = []
            for cand in candidates:
                if cand not in seen:
                    seen.add(cand)
                    dedup_candidates.append(cand)

            candidates = dedup_candidates

            if not candidates:
                break
            for i,c in enumerate(candidates):
                if len(c) > 5000:
                    print(
                        "[BAD CANDIDATE]",
                        {
                            "candidate_idx": i,
                            "char_len": len(c),
                            "token_len": len(
                                self.tokenizer.encode(c)
                            ),
                            "sentence_idx": sentence_idx,
                            "preview": c[:300]
                        },
                        flush=True
                    )

            best_sentence, sentence_idx = self.algo.select_best_candidate(
                candidates,
                sentence_idx,
            )
            if len(best_sentence) > 5000:
                print(
                    "[BAD SELECTED SENTENCE]",
                    {
                        "char_len": len(best_sentence),
                        "token_len": len(
                            self.tokenizer.encode(best_sentence)
                        ),
                        "sentence_idx": sentence_idx,
                        "preview": best_sentence[:300]
                    },
                    flush=True
                )

            generated_text += " " + best_sentence

            if len(generated_text)>5000:
                print(
                    "[BAD GENERATED TEXT]",
                    {
                        "char_len":len(generated_text),
                        "sentence_idx":sentence_idx,
                        "tail":generated_text[-500:]
                    },
                    flush=True
                )

            current_seq = self.tokenizer(
                prompt + generated_text,
                return_tensors="pt",
            ).input_ids.to(self.device)

            if has_eos:
                break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_text.strip()

    def detect(self, text: str) -> dict:
        """
        Single-channel IAS-W detection.

        IASWMacroAlgorithm returns macro z-score and ablation scores.
        """
        return self.algo.detect(text)