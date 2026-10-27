# src/watermarks/runtime/loaders.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


@dataclass
class RuntimeBundle:
    tokenizer: object
    model: object
    embedder: object
    device: str


def _resolve_dtype(task: str, device: str):
    if (
        not torch.cuda.is_available()
        or not str(device).startswith("cuda")
    ):
        return torch.float32

    if task == "generate":
        return torch.float16

    if task == "detect":
        return torch.bfloat16

    return torch.float16


def load_runtime(
    args,
    task: str,
    load_model: bool = True,
    load_embedder: bool = True,
) -> RuntimeBundle:

    device = getattr(
        args,
        "device",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # ============================================================
    # Tokenizer
    # ============================================================

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side=(
            "left"
            if task == "generate"
            else "right"
        ),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================================
    # 1. CausalLM
    # ============================================================

    model = None

    if load_model:
        dtype = _resolve_dtype(
            task,
            device,
        )

        model_path = args.model_path

        model_load_kwargs = {
            "device_map": device,
            "torch_dtype": dtype,
        }

        model_name = str(
            model_path
        ).rstrip("/")

        if (
            model_name.endswith("opt-1.3b-semstamp")
            or
            "AbeHou/opt-1.3b-semstamp"
            in model_name
        ):
            model_load_kwargs["use_safetensors"] = False
            model_load_kwargs["torch_dtype"] = torch.bfloat16

            print(
                f"[*] 检测到旧版模型 {model_path}，"
                f"自动设置 use_safetensors=False"
            )

        print(
            f"[*] Loading CausalLM: {model_path}"
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_path,
                **model_load_kwargs,
            )
            .eval()
        )

        model.config.pad_token_id = (
            tokenizer.eos_token_id
        )

    else:
        print(
            "[*] Skip loading CausalLM: "
            "current detection method does not require LLM."
        )

    # ============================================================
    # 2. Runtime-level embedder
    # ============================================================

    embedder = None

    if load_embedder:
        embedder_path = getattr(
            args,
            "embedder_path",
            "",
        )

        if (
            embedder_path
            and os.path.exists(embedder_path)
        ):
            # IMPORTANT:
            # Import only when the selected method really needs it.
            try:
                from sentence_transformers import SentenceTransformer
            except (ImportError, ModuleNotFoundError) as exc:
                raise ImportError(
                    "The selected watermark method requires the "
                    "runtime SentenceTransformer embedder, but "
                    "'sentence-transformers' is not available."
                ) from exc

            print(
                f"[*] Loading Runtime Embedder: "
                f"{embedder_path}"
            )

            embedder = (
                SentenceTransformer(
                    embedder_path
                )
                .to(device)
            )

        elif embedder_path:
            print(
                "[!] Runtime embedder path does not exist, "
                f"skip loading here: {embedder_path}"
            )

        else:
            print(
                "[*] No embedder_path provided, "
                "skip runtime embedder."
            )

    else:
        print(
            "[*] Skip loading runtime embedder: "
            "current method does not require it."
        )

    return RuntimeBundle(
        tokenizer=tokenizer,
        model=model,
        embedder=embedder,
        device=device,
    )