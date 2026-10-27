# src/watermarks/single/synguard.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, LogitsProcessorList

from src.watermarks.plugins import BaseWatermark, register_watermark


def _is_under(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    return str(path).startswith(str(root) + "/")


@register_watermark("synguard")
class SynGuardPlugin(BaseWatermark):
    """
    Project wrapper for the released SynGuard implementation:

        SIR + SynthID

    Important:
    - generation uses the released SIR and SynthID logits processors;
    - output is completion-only, matching the IASW project convention;
    - detection preserves SynGuard's released weighted combined score;
    - no Fisher/max fusion is introduced here.
    """

    def __init__(self, args, tokenizer, model, embedder=None):
        super().__init__(args, tokenizer, model, embedder)

        self.core = (
            Path(__file__).resolve().parents[2]
            / "baselines"
            / "synguard_core"
        ).resolve()

        if not self.core.is_dir():
            raise FileNotFoundError(
                f"SynGuard core not found: {self.core}"
            )

        # ------------------------------------------------------------------
        # Released SynGuard uses top-level packages such as:
        #   watermark.*
        #   utils.*
        #
        # Since plugins are lazily loaded and each experiment runs in a
        # separate process, put only this vendored core at sys.path[0].
        # ------------------------------------------------------------------
        core_str = str(self.core)
        if core_str not in sys.path:
            sys.path.insert(0, core_str)

        import utils.transformers_config as tf_mod
        import watermark.sir_synthid.sir_synthid as sg_mod

        # Fail loudly rather than silently mixing another vendored MarkLLM.
        if not _is_under(tf_mod.__file__, self.core):
            raise RuntimeError(
                "SynGuard namespace collision: "
                "utils.transformers_config came from "
                f"{tf_mod.__file__}, expected under {self.core}. "
                "Run external baselines in separate Python processes."
            )

        if not _is_under(sg_mod.__file__, self.core):
            raise RuntimeError(
                "SynGuard namespace collision: "
                "watermark.sir_synthid came from "
                f"{sg_mod.__file__}, expected under {self.core}. "
                "Run external baselines in separate Python processes."
            )

        TransformersConfig = tf_mod.TransformersConfig
        SIR_SynthID = sg_mod.SIR_SynthID

        # SynGuard/SIR mapping must match the model's logits
        # vocabulary dimension, NOT len(tokenizer).
        #
        # Example for OPT-1.3B:
        #   len(tokenizer)      = 50265
        #   model vocab/logits  = 50272
        #   official SIR map    = 50272
        self.tokenizer_vocab_size = len(self.tokenizer)

        if self.model is not None:
            self.vocab_size = int(
                self.model.config.vocab_size
            )

            output_embeddings = (
                self.model.get_output_embeddings()
            )

            if output_embeddings is not None:
                output_vocab_size = int(
                    output_embeddings.weight.shape[0]
                )

                if output_vocab_size != self.vocab_size:
                    raise RuntimeError(
                        "Model vocab mismatch: "
                        f"config.vocab_size={self.vocab_size}, "
                        f"output_embeddings={output_vocab_size}"
                    )
        else:
            # Detection should not need to load the CausalLM.
            # Reading AutoConfig is lightweight and gives the exact
            # vocabulary dimension used by the model logits.
            model_cfg = AutoConfig.from_pretrained(
                self.args.model_path,
                local_files_only=True,
            )

            self.vocab_size = int(
                model_cfg.vocab_size
            )

        if self.tokenizer_vocab_size > self.vocab_size:
            raise RuntimeError(
                "Tokenizer vocabulary is larger than model vocabulary: "
                f"tokenizer={self.tokenizer_vocab_size}, "
                f"model={self.vocab_size}"
            )

        # Build a patched runtime config without changing the copied
        # official config.
        self.runtime_config = self._build_runtime_config()

        # NOTE:
        # The released TransformersConfig stores kwargs in gen_kwargs but
        # does not promote top_k/temperature to attributes.
        #
        # SynthIDConfig itself defaults to temperature=0.7 and top_k=-1.
        # We preserve that released behavior here. Project top-k is applied
        # by HF sampling outside the watermark processor.
        tf_cfg = TransformersConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            device=self.device,
            max_new_tokens=int(
                getattr(self.args, "max_total_tokens", 200)
            ),
            do_sample=True,
        )

        self.engine = SIR_SynthID(
            str(self.runtime_config),
            tf_cfg,
        )

        model_delta = float(self.engine.config.model_delta)

        sir_threshold = (
            float(self.engine.config.sir_config.z_threshold) + 1.0
        ) / 2.0

        synthid_threshold = float(
            self.engine.config.synthid_config.threshold
        )

        self.official_threshold = (
            (1.0 - model_delta) * sir_threshold
            + model_delta * synthid_threshold
        )

        print("[SynGuard] Loaded official SIR + SynthID")
        print(f"[SynGuard] core      = {self.core}")
        print(f"[SynGuard] config    = {self.runtime_config}")
        print(
            f"[SynGuard] tokenizer_vocab = "
            f"{self.tokenizer_vocab_size}"
        )
        print(
            f"[SynGuard] model_vocab     = "
            f"{self.vocab_size}"
        )
        print(
            "[SynGuard] mapping   = "
            f"{self.engine.config.sir_config.mapping_name}"
        )
        print(
            "[SynGuard] embedding = "
            f"{self.engine.config.sir_config.embedding_model_path}"
        )
        print(
            f"[SynGuard] model_delta={model_delta} | "
            f"official_threshold={self.official_threshold:.6f}"
        )

    # ==================================================================
    # Runtime config
    # ==================================================================

    def _build_runtime_config(self):
        official_cfg = (
            self.core
            / "config"
            / "SIR_SynthID.json"
        )

        config_override = str(
            getattr(self.args, "synguard_config", "") or ""
        ).strip()

        if config_override:
            official_cfg = (
                Path(config_override).expanduser().resolve()
            )

        if not official_cfg.is_file():
            raise FileNotFoundError(
                f"SynGuard config not found: {official_cfg}"
            )

        cfg = json.loads(
            official_cfg.read_text(encoding="utf-8")
        )

        # --------------------------------------------------------------
        # SIR transform model
        # --------------------------------------------------------------
        transform_path = (
            self.core
            / "watermark"
            / "sir"
            / "model"
            / "transform_model_cbert.pth"
        ).resolve()

        # --------------------------------------------------------------
        # Compositional BERT
        # --------------------------------------------------------------
        embedding_override = str(
            getattr(
                self.args,
                "synguard_embedding_path",
                "",
            )
            or ""
        ).strip()

        embedding_path = Path(
            embedding_override
            or "./models/compositional-bert-large-uncased"
        ).expanduser().resolve()

        # --------------------------------------------------------------
        # Exact tokenizer-specific SIR mapping
        # --------------------------------------------------------------
        mapping_path = (
            self.core
            / "watermark"
            / "sir"
            / "mapping"
            / f"300_mapping_{self.vocab_size}.json"
        ).resolve()

        if not transform_path.is_file():
            raise FileNotFoundError(
                "SynGuard transform model not found: "
                f"{transform_path}"
            )

        if not embedding_path.is_dir():
            raise FileNotFoundError(
                "SynGuard compositional-BERT not found: "
                f"{embedding_path}"
            )

        if not mapping_path.is_file():
            mapping_dir = (
                self.core
                / "watermark"
                / "sir"
                / "mapping"
            )
            available = sorted(
                p.name
                for p in mapping_dir.glob(
                    "300_mapping_*.json"
                )
            )

            raise FileNotFoundError(
                "No exact SynGuard SIR mapping for "
                f"tokenizer vocab_size={self.vocab_size}.\n"
                f"Expected: {mapping_path}\n"
                f"Available: {available}"
            )

        # Validate before SIR is allowed to initialize.
        mapping = json.loads(
            mapping_path.read_text(encoding="utf-8")
        )

        if not isinstance(mapping, list):
            raise RuntimeError(
                "SynGuard mapping must be a JSON list, "
                f"got {type(mapping).__name__}"
            )

        if len(mapping) != self.vocab_size:
            raise RuntimeError(
                "SynGuard mapping length mismatch: "
                f"mapping={len(mapping)}, "
                f"tokenizer={self.vocab_size}"
            )

        scale_dimension = int(cfg["scale_dimension"])

        for i, value in enumerate(mapping):
            if (
                not isinstance(value, int)
                or value < 0
                or value >= scale_dimension
            ):
                raise RuntimeError(
                    "Invalid SynGuard mapping value: "
                    f"mapping[{i}]={value}, "
                    f"scale_dimension={scale_dimension}"
                )

        cfg["transform_model_name"] = str(transform_path)
        cfg["embedding_model_path"] = str(embedding_path)
        cfg["mapping_name"] = str(mapping_path)

        if cfg.get("model_delta", None) is None:
            raise RuntimeError(
                "SynGuard config missing model_delta"
            )

        runtime_dir = self.core / "runtime_config"
        runtime_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        runtime_path = (
            runtime_dir
            / f"SIR_SynthID_vocab{self.vocab_size}.json"
        )

        rendered = (
            json.dumps(
                cfg,
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )

        if (
            not runtime_path.exists()
            or runtime_path.read_text(
                encoding="utf-8"
            )
            != rendered
        ):
            runtime_path.write_text(
                rendered,
                encoding="utf-8",
            )

        return runtime_path

    # ==================================================================
    # Generation
    # ==================================================================

    def _reset_generation_state(self):
        """
        Released SynthIDLogitsProcessor stores generation state.
        Each independent sample must start from state=None.
        """
        processor = self.engine.logits_processor_synthid

        if hasattr(processor, "state"):
            processor.state = None

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        if self.model is None:
            raise RuntimeError(
                "SynGuard generation requires the base CausalLM."
            )

        self._reset_generation_state()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        attention_mask = inputs.get(
            "attention_mask",
            torch.ones_like(input_ids),
        )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        # Important:
        # SynthIDLogitsProcessor already applies its configured
        # temperature (released default = 0.7), so do NOT also pass
        # temperature=0.7 to model.generate, otherwise temperature
        # would effectively be applied twice.
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(
                getattr(
                    self.args,
                    "max_total_tokens",
                    200,
                )
            ),
            do_sample=True,
            top_k=int(
                getattr(
                    self.args,
                    "top_k",
                    50,
                )
            ),
            logits_processor=LogitsProcessorList([
                self.engine.logits_processor_sir,
                self.engine.logits_processor_synthid,
            ]),
            pad_token_id=pad_id,
        )

        generated_text = self.tokenizer.decode(
            outputs[0, prompt_len:],
            skip_special_tokens=True,
        )

        # Do not allow current sample's SynthID history to leak
        # into the next independent sample.
        self._reset_generation_state()

        del outputs
        del inputs

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (generated_text or "").strip()

    # ==================================================================
    # Detection
    # ==================================================================

    @torch.no_grad()
    def detect(self, text: str) -> dict:
        result = self.engine.detect_watermark(
            text,
            return_dict=True,
        )

        score = float(result["score"])
        sir_score = float(result["sir_score"])
        synthid_score = float(
            result["synthid_score"]
        )
        model_delta = float(result["model_delta"])

        return {
            # IASW runner/evaluation expects z_score as the
            # primary scalar. For SynGuard this is an alias
            # of its official combined score, not a Gaussian z.
            "z_score": score,
            "score": score,

            "sir_score": sir_score,
            "synthid_score": synthid_score,

            "details": {
                "method": "SynGuard",
                "implementation":
                    "official_released_sir_synthid",
                "score_type":
                    "official_combined_score",

                "model_delta": model_delta,

                "official_threshold":
                    float(self.official_threshold),

                "official_is_watermarked":
                    bool(result["is_watermarked"]),

                "note": (
                    "Top-level z_score is an interface alias "
                    "for the official SynGuard combined score; "
                    "it is not a Gaussian z-statistic."
                ),
            },
        }
