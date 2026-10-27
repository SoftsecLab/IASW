# src/watermarks/registry.py
# -*- coding: utf-8 -*-

from importlib import import_module


# ============================================================
# Lazy algorithm specifications
# ============================================================

MICRO_ALGORITHM_SPECS = {
    "kgw": (
        "src.watermarks.algorithms.micro.kgw",
        "KGWMicroAlgorithm",
    ),
    "sweet": (
        "src.watermarks.algorithms.micro.sweet",
        "SweetMicroAlgorithm",
    ),
    "dipmark": (
        "src.watermarks.algorithms.micro.dipmark",
        "DiPMarkMicroAlgorithm",
    ),
    "sta": (
        "src.watermarks.algorithms.micro.sta",
        "STAMicroAlgorithm",
    ),
}


# third field:
#   True  -> constructor also receives model=model
#   False -> constructor does not receive model
MACRO_ALGORITHM_SPECS = {
    "ias": (
        "src.watermarks.algorithms.macro.ias",
        "IASMacroAlgorithm",
        False,
    ),
    "iasw": (
        "src.watermarks.algorithms.macro.ias_w",
        "IASWMacroAlgorithm",
        False,
    ),
    "iaswa": (
        "src.watermarks.algorithms.macro.ias_wa",
        "IASWAMacroAlgorithm",
        False,
    ),

    "semstamp": (
        "src.watermarks.algorithms.macro.semstamp",
        "SemStampMacroAlgorithm",
        True,
    ),
    "ksemstamp": (
        "src.watermarks.algorithms.macro.ksemstamp",
        "KSemStampMacroAlgorithm",
        True,
    ),
    "simmark_cosine": (
        "src.watermarks.algorithms.macro.simmark_cosine",
        "SimMarkCosineMacroAlgorithm",
        True,
    ),
    "simmark_euclidean": (
        "src.watermarks.algorithms.macro.simmark_euclidean",
        "SimMarkEuclideanMacroAlgorithm",
        True,
    ),
}


MICRO_ALIASES = {}

MACRO_ALIASES = {
    "ias_w": "iasw",
    "ias_wa": "iaswa",
}


def normalize_micro_name(name: str) -> str:
    name = str(name).lower().strip()
    return MICRO_ALIASES.get(name, name)


def normalize_macro_name(name: str) -> str:
    name = str(name).lower().strip()
    return MACRO_ALIASES.get(name, name)


def _load_class(module_name: str, class_name: str, kind: str, name: str):
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
            f"Failed to load {kind} algorithm '{name}' "
            f"from '{module_name}'.{detail}"
        ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"{kind.capitalize()} class '{class_name}' "
            f"was not found in '{module_name}'."
        ) from exc


def build_micro_algorithm(
    name,
    args,
    tokenizer,
    model,
    device="cuda",
):
    name = normalize_micro_name(name)

    if name not in MICRO_ALGORITHM_SPECS:
        raise ValueError(
            f"Unknown micro algorithm: {name}. "
            f"Available: {sorted(MICRO_ALGORITHM_SPECS)}"
        )

    module_name, class_name = MICRO_ALGORITHM_SPECS[name]

    cls = _load_class(
        module_name,
        class_name,
        "micro",
        name,
    )

    return cls(
        args,
        tokenizer,
        model,
        device=device,
    )


def build_macro_algorithm(
    name,
    args,
    tokenizer,
    embedder=None,
    model=None,
    device="cuda",
):
    name = normalize_macro_name(name)

    if name not in MACRO_ALGORITHM_SPECS:
        raise ValueError(
            f"Unknown macro algorithm: {name}. "
            f"Available: {sorted(MACRO_ALGORITHM_SPECS)}"
        )

    (
        module_name,
        class_name,
        needs_model,
    ) = MACRO_ALGORITHM_SPECS[name]

    cls = _load_class(
        module_name,
        class_name,
        "macro",
        name,
    )

    kwargs = {
        "device": device,
    }

    if needs_model:
        kwargs["model"] = model

    return cls(
        args,
        tokenizer,
        embedder,
        **kwargs,
    )


def build_dual_watermarker(
    args,
    tokenizer,
    model,
    embedder=None,
):
    from src.watermarks.dual.orchestrator import UniversalDualPlugin

    return UniversalDualPlugin(
        args,
        tokenizer,
        model,
        embedder,
    )


def get_dual_registry():
    from src.watermarks.dual.orchestrator import UniversalDualPlugin

    return {
        "dual": UniversalDualPlugin,
        "universal_dual": UniversalDualPlugin,
    }