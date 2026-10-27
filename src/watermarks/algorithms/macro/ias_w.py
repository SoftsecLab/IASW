# src/watermarks/algorithms/macro/ias_w.py
# -*- coding: utf-8 -*-
"""
Macro wrapper for IAS-W.

The external macro method name is "iasw".
Internally, IASWMark has been replaced by IAS-W:
    Whitened Independent Anchor Semantic Watermark.

Important:
    IAS-W performs its own internal calibration using IASW_STATS_FILE
    or IASW_STATS_REGISTRY. Therefore dual-level calibration must stay disabled.
"""

import os
import torch
from sentence_transformers import SentenceTransformer

from src.watermarks.algorithms.base import BaseMacroAlgorithm
from src.watermarks.ias_w_mark import IASWMark


class IASWMacroAlgorithm(BaseMacroAlgorithm):
    # IAS-W already returns calibrated macro z-score.
    # UniversalDualPlugin should not calibrate it again.
    needs_dual_calibration = False

    def __init__(self, args, tokenizer, embedder=None, device="cuda"):
        self.args = args
        self.tokenizer = tokenizer
        self.device = device

        if embedder is None:
            default_path = getattr(args, "embedder_path", "./models/bge-small-en-v1.5")
            if not default_path:
                default_path = "./models/bge-small-en-v1.5"

            print(f"[IAS-W] Loading embedder inside IASWMacroAlgorithm: {default_path}")
            embedder = SentenceTransformer(default_path).to(self.device)

        self.embedder = embedder

        stats_file = (
            getattr(args, "iasw_stats_file", "")
            or os.environ.get("IASW_STATS_FILE", "")
            or getattr(args, "ias_stats_file", "")
            or os.environ.get("IAS_STATS_FILE", "")
            or ""
        )

        stats_key = (
            getattr(args, "iasw_stats_key", "")
            or os.environ.get("IASW_STATS_KEY", "")
            or getattr(args, "ias_stats_key", "")
            or os.environ.get("IAS_STATS_KEY", "")
            or ""
        )

        target_len = (
            getattr(args, "target_len", None)
            or getattr(args, "max_total_tokens", None)
            or getattr(args, "iasw_target_len", None)
            or os.environ.get("IASW_TARGET_LEN", "")
            or os.environ.get("IAS_TARGET_LEN", "")
            or os.environ.get("TARGET_LEN", "")
            or 200
        )

        self.ias_w_mark = IASWMark(
            embedder_path_or_model=self.embedder,
            tokenizer=self.tokenizer,
            model_name_path=getattr(self.args, "model_path", "llama3.1-8b"),
            dataset_type=getattr(self.args, "dataset_type", "c4"),
            secret_key=getattr(self.args, "secret_key", "DUAL_MACRO_2026"),
            device=self.device,
            stats_file=stats_file if stats_file else None,
            stats_key=stats_key if stats_key else None,
            target_len=target_len,
        
            scoring_profile=(
                "dual"
                if str(getattr(self.args, "method", "") or "").lower().strip()
                in {"universal_dual", "dual"}
                else "standalone"
            ),
        )

    def init_context(self, prompt: str):
        # IAS / IAS-W context is the current sentence index.
        return 0

    def select_best_candidate(self, candidates: list, sentence_index: int):
        best_sentence, _ = self.ias_w_mark.select_best_candidate(candidates, sentence_index)
        return best_sentence, sentence_index + 1

    def detect(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "z_score": 0.0,
                "z_score_micro": 0.0,
                "z_score_macro": 0.0,
                "details": {},
            }

        try:
            res = self.ias_w_mark.detect(text)
            z = float(res.get("z_score", 0.0))
        except Exception as e:
            z = 0.0
            res = {"error": str(e)}

        return {
            "z_score": z,
            "z_score_micro": 0.0,
            "z_score_macro": z,
            "details": res,
        }