# src/watermarks/single/dualmark.py
# -*- coding: utf-8 -*-

import gc
import os
import sys
from pathlib import Path

import torch
from transformers import LogitsProcessorList

from src.watermarks.plugins import BaseWatermark, register_watermark


@register_watermark("dualmark")
class DualMarkPlugin(BaseWatermark):
    """
    Standalone DualMark baseline.

    Published representative:
        Twins = KGW + TSW

    Generation:
        Official TwinsLogitsProcessor.

    Detection:
        Official KGW z-score + official TSW z-score.

    For the IASW evaluation framework, top-level `z_score`
    is max(z_kgw, z_tsw). With the official Twins config,
    both thresholds are 4.0, so this is the continuous
    counterpart of the published OR decision rule.
    """

    def __init__(self, args, tokenizer, model, embedder=None):
        super().__init__(args, tokenizer, model, embedder)

        # Released Twins uses a fixed OPT-1.3B semantic/token
        # reference for non-OPT target backbones.
        self.reference_opt_path = Path(
            getattr(
                args,
                "dualmark_reference_opt_path",
                "",
            )
            or "./models/opt-1.3b"
        ).expanduser().resolve()

        if not (
            self.reference_opt_path / "config.json"
        ).is_file():
            raise FileNotFoundError(
                "DualMark reference OPT is missing: "
                f"{self.reference_opt_path}"
            )

        # The vendored Twins core reads this when constructing
        # TwinsConfig/TwinsUtils.
        os.environ[
            "DUALMARK_REFERENCE_OPT_PATH"
        ] = str(self.reference_opt_path)

        project_root = Path(__file__).resolve().parents[3]

        self.core_root = (
            project_root
            / "src"
            / "baselines"
            / "dualmark_core"
        )

        self.config_path = (
            self.core_root
            / "config"
            / "Twins.json"
        )

        if not self.core_root.is_dir():
            raise FileNotFoundError(
                f"DualMark core not found: {self.core_root}"
            )

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"DualMark config not found: {self.config_path}"
            )

        # -----------------------------------------------------
        # Official vendored DualMark uses top-level imports:
        #     utils.*
        #     watermark.*
        #
        # Import lazily so it does not interfere with other
        # standalone baselines.
        # -----------------------------------------------------
        core_str = str(self.core_root)

        if core_str not in sys.path:
            sys.path.insert(0, core_str)

        from utils.transformers_config import TransformersConfig
        from watermark.twins.twins import (
            TwinsConfig,
            TwinsUtils,
            TwinsLogitsProcessor,
        )

        # TSW needs the base model embedding matrix.
        if self.model is None:
            raise RuntimeError(
                "DualMark/Twins requires the base CausalLM "
                "during initialization because TSW uses the "
                "generation-model input embedding matrix."
            )

        # MarkLLM-style transformer config.
        #
        # These kwargs are stored in gen_kwargs by
        # TransformersConfig. We still explicitly call
        # model.generate() below so that IASW uses the same
        # prompt-suffix convention as the other baselines.
        self.ext_transformers_config = TransformersConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            vocab_size=len(self.tokenizer),
            device=self.device,
            max_new_tokens=getattr(
                self.args,
                "max_total_tokens",
                200,
            ),
            do_sample=True,
            temperature=0.7,
            top_k=getattr(
                self.args,
                "top_k",
                50,
            ),
        )

        self.config = TwinsConfig(
            str(self.config_path),
            self.ext_transformers_config,
        )

        self.utils = TwinsUtils(self.config)

        self.processor = TwinsLogitsProcessor(
            self.config,
            self.utils,
        )

        self.threshold_kgw = float(
            self.config.z_threshold_kgw
        )
        self.threshold_tsw = float(
            self.config.z_threshold_ts
        )

        print("[DualMark] Loaded official Twins = KGW + TSW")
        print(f"[DualMark] core   = {self.core_root}")
        print(f"[DualMark] config = {self.config_path}")
        print(f"[DualMark] ckpt   = {self.config.ckpt_path}")
        print(
            "[DualMark] thresholds: "
            f"KGW={self.threshold_kgw}, "
            f"TSW={self.threshold_tsw}"
        )

    # =========================================================
    # Mutable TSW state
    # =========================================================

    def _reset_tsw_state(self):
        """
        Official TwinsUtils accumulates gamma/delta values
        across score/generation calls. Every independent text
        must therefore start from an empty state.
        """
        self.utils.gamma_list = torch.empty(
            0,
            dtype=torch.float,
            device=self.device,
        )

        self.utils.delta_list = torch.empty(
            0,
            dtype=torch.float,
            device=self.device,
        )

    # =========================================================
    # Generation
    # =========================================================

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        self._reset_tsw_state()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        input_ids = inputs["input_ids"]

        attention_mask = inputs.get(
            "attention_mask",
            torch.ones_like(input_ids),
        )

        prompt_len = int(input_ids.shape[1])

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=getattr(
                self.args,
                "max_total_tokens",
                200,
            ),
            do_sample=True,
            temperature=0.7,
            top_k=getattr(
                self.args,
                "top_k",
                50,
            ),
            logits_processor=LogitsProcessorList(
                [self.processor]
            ),
            pad_token_id=pad_token_id,
        )

        # Important:
        # IASW records only the generated continuation,
        # not prompt + continuation.
        generated_text = self.tokenizer.decode(
            outputs[0, prompt_len:],
            skip_special_tokens=True,
        )

        del inputs
        del outputs

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_text.strip()

    # =========================================================
    # Detection
    # =========================================================

    @torch.no_grad()
    def detect(self, text: str) -> dict:
        text = str(text or "").strip()

        if not text:
            return {
                "z_score": 0.0,
                "z_score_kgw": 0.0,
                "z_score_tsw": 0.0,
                "details": {
                    "method": "DualMark-Twins",
                    "pairing": "KGW+TSW",
                    "error": "empty_text",
                },
            }

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].to(self.device)

        num_tokens = int(encoded.numel())

        min_required = max(
            int(self.config.prefix_length_kgw),
            int(self.config.prefix_length_ts),
        ) + 1

        if num_tokens < min_required:
            return {
                "z_score": 0.0,
                "z_score_kgw": 0.0,
                "z_score_tsw": 0.0,
                "details": {
                    "method": "DualMark-Twins",
                    "pairing": "KGW+TSW",
                    "error": "text_too_short",
                    "num_tokens": num_tokens,
                },
            }

        # KGW stream
        self._reset_tsw_state()

        z_kgw, kgw_mask = (
            self.utils.score_sequence_kgw(encoded)
        )

        # TSW stream.
        #
        # Reset again because the official TSW scorer uses
        # mutable gamma_list / delta_list.
        # TSW stream
        self._reset_tsw_state()

        ts_error = None

        try:
            z_tsw, tsw_mask = (
                self.utils.score_sequence_ts(encoded)
            )
            z_tsw = float(z_tsw)
            ts_valid = True

        except (ValueError, ZeroDivisionError) as e:
            z_tsw = float("nan")
            tsw_mask = []
            ts_valid = False
            ts_error = str(e)

        z_kgw = float(z_kgw)


        # Official Twins config uses identical z thresholds
        # (4.0 and 4.0). Hence:
        #
        #   KGW > 4 OR TSW > 4
        #
        # is equivalent to:
        #
        #   max(KGW, TSW) > 4
        #
        # This continuous score allows the IASW framework
        # to compute AUC and empirical TPR@FPR.

        if ts_valid:
            z_final = max(
                z_kgw,
                z_tsw,
            )

            official_or_prediction = bool(
                z_kgw > self.threshold_kgw
                or z_tsw > self.threshold_tsw
            )

        else:
            # TSW 当前样本不可评分。
            # 顶层保持有限值，以免整批检测崩溃；
            # 同时明确记录该样本 TSW invalid。
            z_final = z_kgw

            official_or_prediction = bool(
                z_kgw > self.threshold_kgw
            )


        kgw_num_scored = sum(
            1
            for x in kgw_mask
            if x in (0, 1)
        )

        tsw_num_scored = sum(
            1
            for x in tsw_mask
            if x in (0, 1)
        )

        # Do not save the complete masks: this would make the
        # 500-sample + attack JSONLs unnecessarily large.
        return {
            "z_score": float(z_final),
            "z_score_kgw": z_kgw,
            "z_score_tsw": z_tsw,
            "details": {
                "method": "DualMark-Twins",
                "pairing": "KGW+TSW",
                "roc_score_rule": "max_component_z",
                "official_threshold_kgw": self.threshold_kgw,
                "official_threshold_tsw": self.threshold_tsw,
                "official_or_prediction": official_or_prediction,
                "num_tokens": num_tokens,
                "kgw_num_scored": int(kgw_num_scored),
                "tsw_num_scored": int(tsw_num_scored),
            },
        }
