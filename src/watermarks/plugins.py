# src/watermarks/plugins.py
# -*- coding: utf-8 -*-

from importlib import import_module

import torch


WATERMARK_REGISTRY = {}


# ============================================================
# Lazy plugin specifications
# ============================================================

WATERMARK_SPECS = {
    "kgw": (
        "src.watermarks.single.kgw",
        "KGWPlugin",
    ),
    "sweet": (
        "src.watermarks.single.sweet",
        "SweetPlugin",
    ),
    "dipmark": (
        "src.watermarks.single.dipmark",
        "DiPMarkPlugin",
    ),
    "sta": (
        "src.watermarks.single.sta",
        "STAPlugin",
    ),

    "ias": (
        "src.watermarks.single.ias",
        "IASPlugin",
    ),
    "iasw": (
        "src.watermarks.single.ias_w",
        "IASWPlugin",
    ),
    # compatibility alias
    "ias_w": (
        "src.watermarks.single.ias_w",
        "IASWPlugin",
    ),

    "semstamp": (
        "src.watermarks.single.semstamp",
        "SemStampPlugin",
    ),
    "ksemstamp": (
        "src.watermarks.single.ksemstamp",
        "KSemStampPlugin",
    ),
    "simmark_cosine": (
        "src.watermarks.single.simmark_cosine",
        "SimMarkCosinePlugin",
    ),
    "simmark_euclidean": (
        "src.watermarks.single.simmark_euclidean",
        "SimMarkEuclideanPlugin",
    ),
    "dualmark": (
        "src.watermarks.single.dualmark",
        "DualMarkPlugin",
    ),

    "synguard": (
        "src.watermarks.single.synguard",
        "SynGuardPlugin",
    ),

    "universal_dual": (
        "src.watermarks.dual.orchestrator",
        "UniversalDualPlugin",
    ),
    "dual": (
        "src.watermarks.dual.orchestrator",
        "UniversalDualPlugin",
    ),
}


class BaseWatermark:
    def __init__(self, args, tokenizer, model, embedder=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.embedder = embedder
        self.device = getattr(
            args,
            "device",
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def detect(self, text: str) -> dict:
        raise NotImplementedError


def register_watermark(name: str):
    """
    Keep decorator registration for compatibility.

    Importing one concrete plugin registers only that plugin.
    """
    name = str(name).lower().strip()

    def decorator(cls):
        WATERMARK_REGISTRY[name] = cls
        return cls

    return decorator


def available_watermarks():
    """
    Return all configured methods without importing their implementations.
    """
    return sorted(WATERMARK_SPECS.keys())


def get_watermark_class(name: str):
    """
    Lazily import only the requested watermark plugin.
    """
    name = str(name).lower().strip()

    # Already loaded.
    if name in WATERMARK_REGISTRY:
        return WATERMARK_REGISTRY[name]

    if name not in WATERMARK_SPECS:
        raise ValueError(
            f"Unknown watermark method: {name}. "
            f"Available methods: {available_watermarks()}"
        )

    module_name, class_name = WATERMARK_SPECS[name]

    try:
        module = import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", None)

        detail = (
            f" Missing module/dependency: {missing}."
            if missing
            else ""
        )

        raise ImportError(
            f"Failed to load watermark method '{name}' from "
            f"'{module_name}'.{detail}"
        ) from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Plugin class '{class_name}' was not found "
            f"in module '{module_name}'."
        ) from exc

    # Cache aliases too.
    WATERMARK_REGISTRY[name] = cls

    return cls