# src/watermarks/ias_w_mark.py
# -*- coding: utf-8 -*-

import os
import re
import json
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# ================= IASW_STATS_REGISTRY_START =================
IASW_STATS_REGISTRY = {   'c4': {   'llama3.1-8b': {   50: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/llama3.1-8b-instruct/ias_stats_c4_llama3.1-8b-instruct_len50_rank64_eps0p0001_calibskip3000_calibn1000.pt'},
                                 100: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/llama3.1-8b-instruct/ias_stats_c4_llama3.1-8b-instruct_len100_rank64_eps0p0001_calibskip3000_calibn1000.pt'},
                                 200: {   'rank128_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/llama3.1-8b-instruct/ias_stats_c4_llama3.1-8b-instruct_len200_rank128_eps0p0001_calibskip3000_calibn1000.pt',
                                          'rank32_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/llama3.1-8b-instruct/ias_stats_c4_llama3.1-8b-instruct_len200_rank32_eps0p0001_calibskip3000_calibn1000.pt',
                                          'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/llama3.1-8b-instruct/ias_stats_c4_llama3.1-8b-instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'},
                                 300: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/llama3.1-8b-instruct/ias_stats_c4_llama3.1-8b-instruct_len300_rank64_eps0p0001_calibskip3000_calibn1000.pt'},
                                 500: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/llama3.1-8b-instruct/ias_stats_c4_llama3.1-8b-instruct_len500_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
              'llama3.2-3b': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/Llama-3.2-3B-Instruct/ias_stats_c4_Llama-3.2-3B-Instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
              'mistral-7b-instruct': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/Mistral-7B-Instruct/ias_stats_c4_Mistral-7B-Instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
              'opt-1.3b-semstamp': {   50: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/opt-1.3b-semstamp/ias_stats_c4_opt-1.3b-semstamp_len50_rank64_eps0p0001_calibskip3000_calibn1000.pt'},
                                       100: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/opt-1.3b-semstamp/ias_stats_c4_opt-1.3b-semstamp_len100_rank64_eps0p0001_calibskip3000_calibn1000.pt'},
                                       200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/opt-1.3b-semstamp/ias_stats_c4_opt-1.3b-semstamp_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
              'qwen2.5-7b-instruct': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/c4/qwen2.5-7b-instruct/ias_stats_c4_qwen2.5-7b-instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}}},
    'dolly': {   'llama3.1-8b': {   200: {   'rank128_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/dolly/llama3.1-8b-instruct/ias_stats_dolly_llama3.1-8b-instruct_len200_rank128_eps0p0001_calibskip3000_calibn1000.pt',
                                             'rank32_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/dolly/llama3.1-8b-instruct/ias_stats_dolly_llama3.1-8b-instruct_len200_rank32_eps0p0001_calibskip3000_calibn1000.pt',
                                             'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/dolly/llama3.1-8b-instruct/ias_stats_dolly_llama3.1-8b-instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
                 'llama3.2-3b': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/dolly/Llama-3.2-3B-Instruct/ias_stats_dolly_Llama-3.2-3B-Instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
                 'mistral-7b-instruct': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/dolly/Mistral-7B-Instruct/ias_stats_dolly_Mistral-7B-Instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
                 'qwen2.5-7b-instruct': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/dolly/qwen2.5-7b-instruct/ias_stats_dolly_qwen2.5-7b-instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}}},
    'wiki': {   'llama3.1-8b': {   200: {   'rank128_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/wiki/llama3.1-8b-instruct/ias_stats_wiki_llama3.1-8b-instruct_len200_rank128_eps0p0001_calibskip3000_calibn1000.pt',
                                            'rank32_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/wiki/llama3.1-8b-instruct/ias_stats_wiki_llama3.1-8b-instruct_len200_rank32_eps0p0001_calibskip3000_calibn1000.pt',
                                            'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/wiki/llama3.1-8b-instruct/ias_stats_wiki_llama3.1-8b-instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
                'llama3.2-3b': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/wiki/Llama-3.2-3B-Instruct/ias_stats_wiki_Llama-3.2-3B-Instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
                'mistral-7b-instruct': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/wiki/Mistral-7B-Instruct/ias_stats_wiki_Mistral-7B-Instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
                'opt-1.3b-semstamp': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/wiki/opt-1.3b-semstamp/ias_stats_wiki_opt-1.3b-semstamp_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}},
                'qwen2.5-7b-instruct': {   200: {   'rank64_eps0p0001_calibskip3000_calibn1000': 'cache/iasw_stats/wiki/qwen2.5-7b-instruct/ias_stats_wiki_qwen2.5-7b-instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt'}}}}
# ================= IASW_STATS_REGISTRY_END =================


BASE_SCORE_MODES = [
    "strict",
    "window",
    "dp_global",
    "dp_local",
]

# ============================================================
# IAS-W alignment-scoring defaults
#
# Cross-backbone sweep on current main-exp data selected a
# stronger Local-DP gap while keeping the historical Window,
# Global-DP behavior, and activation-area normalization.
#
# IMPORTANT: Global DP and Local DP intentionally use separate
# gap parameters. Do not collapse them into one shared value.
# ============================================================
ALIGNMENT_WINDOW_SIZE = 2

# Detection scoring is intentionally profile-specific.
#
# standalone:
#   selected by the 7-unit DIPPER-first constrained sweep.
#
# dual:
#   retains the historical setting because it is the only configuration
#   that satisfies all dual-Fisher constraints without paper-rank loss.
IASW_SCORING_PROFILES = {
    "standalone": {
        "global_gap_penalty": -0.05,
        "local_gap_penalty": -0.125,
        "local_normalization": "sqrt_area",
    },
    "dual": {
        "global_gap_penalty": -0.05,
        "local_gap_penalty": -0.05,
        "local_normalization": "area",
    },
}

# Dataset-level overrides used when building NEW calibration statistics.
# Runtime detection should preferentially use scoring_profiles saved in stats.pt.
# Old .pt files without scoring_profiles therefore keep the historical defaults.
IASW_DATASET_SCORING_OVERRIDES = {
    "c4": {
        "dual": {
            "global_gap_penalty": -0.05,
            "local_gap_penalty": -0.10,
            "local_normalization": "sqrt_area",
        },
    },
    "wiki": {
        "dual": {
            "global_gap_penalty": -0.05,
            "local_gap_penalty": -0.125,
            "local_normalization": "sqrt_area",
        },
    },
    "dolly": {
        "dual": {
            "global_gap_penalty": -0.05,
            "local_gap_penalty": -0.085,
            "local_normalization": "area",
        },
    },
}


def get_iasw_scoring_profiles_for_dataset(
    dataset_type: str,
) -> Dict[str, Dict[str, Any]]:
    """Return complete IAS-W scoring profiles for one dataset."""
    profiles = {
        name: dict(config)
        for name, config in IASW_SCORING_PROFILES.items()
    }

    dataset_key = (dataset_type or "").strip().lower()
    overrides = IASW_DATASET_SCORING_OVERRIDES.get(dataset_key, {})

    for profile_name, override in overrides.items():
        merged = dict(profiles.get(profile_name, {}))
        merged.update(dict(override))
        profiles[profile_name] = merged

    return profiles


DEFAULT_SCORING_PROFILE = "dual"


# ============================================================
# Utilities
# ============================================================

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?。！？\n])", text)
        if s and s.strip()
    ]
    return sentences


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def infer_model_key(model_path: str) -> str:
    """
    Keep this consistent with calibrate_empirical_ias.py.
    """
    p = (model_path or "").lower()

    if "qwen2.5" in p or "qwen-2.5" in p:
        return "qwen2.5-7b-instruct"

    if "llama3.2" in p or "llama-3.2" in p:
        return "llama3.2-3b"

    if "mistral" in p:
        return "mistral-7b-instruct"

    if "opt-1.3b" in p or "semstamp" in p:
        return "opt-1.3b-semstamp"

    if "llama3.1" in p or "llama-3.1" in p or "llama3" in p or "llama-3" in p:
        return "llama3.1-8b"

    base = os.path.basename(model_path.rstrip("/"))
    return base.lower() if base else "unknown_model"


def _project_root() -> str:
    """
    Current file:
        <project_root>/src/watermarks/ias_w_mark.py
    """
    this_file = os.path.abspath(__file__)
    watermarks_dir = os.path.dirname(this_file)
    src_dir = os.path.dirname(watermarks_dir)
    return os.path.dirname(src_dir)


def _resolve_existing_path(path: str) -> str:
    """
    Resolve registry path.

    Registry normally stores project-relative path, e.g.
        cache/iasw_stats/xxx.pt
    """
    if not path:
        return ""

    path = path.replace("\\", "/")

    if os.path.isabs(path) and os.path.exists(path):
        return path

    if os.path.exists(path):
        return path

    root_path = os.path.join(_project_root(), path)
    if os.path.exists(root_path):
        return root_path

    return path


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ============================================================
# Alignment scoring
# ============================================================

def compute_all_alignment_scores(
    sim_matrix: np.ndarray,
    window_size: int = ALIGNMENT_WINDOW_SIZE,
    global_gap_penalty: float = -0.05,
    local_gap_penalty: float = -0.05,
    local_normalization: str = "area",
) -> Dict[str, float]:
    N, M = sim_matrix.shape

    if N == 0 or M == 0:
        return {
            "strict": 0.0,
            "window": 0.0,
            "dp_global": 0.0,
            "dp_local": 0.0,
        }

    baseline_sim = 0.0
    shifted = sim_matrix - baseline_sim

    strict_vals = []
    for i in range(min(N, M)):
        strict_vals.append(shifted[i][i])
    strict_score = float(np.mean(strict_vals)) if strict_vals else 0.0

    window_vals = []
    for i in range(N):
        left = max(0, i - window_size)
        right = min(M, i + window_size + 1)
        if left < right:
            window_vals.append(np.max(shifted[i, left:right]))
    window_score = float(np.mean(window_vals)) if window_vals else 0.0

    dp_g = np.zeros((N + 1, M + 1), dtype=np.float32)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            match = dp_g[i - 1][j - 1] + shifted[i - 1][j - 1]
            delete = dp_g[i - 1][j] + 0.0
            insert = dp_g[i][j - 1] + global_gap_penalty
            dp_g[i][j] = max(match, delete, insert)

    dp_global_score = float(np.max(dp_g[N, :]) / max(N, 1))

    dp_l = np.zeros((N + 1, M + 1), dtype=np.float32)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            match = dp_l[i - 1][j - 1] + shifted[i - 1][j - 1]
            delete = dp_l[i - 1][j] + local_gap_penalty
            insert = dp_l[i][j - 1] + local_gap_penalty
            dp_l[i][j] = max(0.0, match, delete, insert)

    max_score = float(np.max(dp_l))

    # Positive activation area of the Local-DP matrix.
    # This is not a traceback path length.
    activation_area = float(np.sum(dp_l > 0) / max(N, 1))

    if local_normalization in {"area", "activation_area"}:
        denom = max(activation_area, 1.0)
    elif local_normalization == "sqrt_area":
        denom = math.sqrt(max(activation_area, 1.0))
    elif local_normalization == "none":
        denom = 1.0
    elif local_normalization == "n":
        denom = float(max(N, 1))
    else:
        raise ValueError(
            f"Unsupported IAS-W local normalization: {local_normalization}"
        )

    dp_local_score = float(max_score / denom)

    return {
        "strict": strict_score,
        "window": window_score,
        "dp_global": dp_global_score,
        "dp_local": dp_local_score,
    }


# ============================================================
# IAS-W Mark
# ============================================================

class IASWMark:
    """
    IAS-W core engine.

    Class name remains IASWMark to keep the rest of the project unchanged.
    """

    def __init__(
        self,
        embedder_path_or_model,
        tokenizer=None,
        model_name_path: str = "llama3.1-8b",
        dataset_type: str = "c4",
        secret_key: str = "DUAL_MACRO_2026",
        device: str = "cuda",
        stats_file: Optional[str] = None,
        stats_key: Optional[str] = None,
        target_len: Optional[int] = None,
        scoring_profile: Optional[str] = None,
    ):
        self.device = device
        self.secret_key = secret_key
        self.tokenizer = tokenizer
        self.dataset_type = (dataset_type or "c4").lower()
        self.model_name_path = (model_name_path or "llama3.1-8b").lower()
        self.model_key = infer_model_key(self.model_name_path)

        self.target_len = _safe_int(
            target_len
            or os.environ.get("IASW_TARGET_LEN")
            or os.environ.get("IAS_TARGET_LEN")
            or os.environ.get("TARGET_LEN")
            or 200,
            200,
        )

        self.stats_key = (
            stats_key
            or os.environ.get("IASW_STATS_KEY", "")
            or os.environ.get("IAS_STATS_KEY", "")
            or ""
        ).strip()

        self.alignment_mode = (
            os.environ.get("IASW_ALIGNMENT_MODE", "")
            or os.environ.get("IAS_ALIGNMENT_MODE", "")
            or "dp_local"
        ).strip()

        if self.alignment_mode not in BASE_SCORE_MODES:
            raise ValueError(
                f"Unsupported IAS alignment mode: {self.alignment_mode}. "
                f"Supported modes: {BASE_SCORE_MODES}"
            )

        self.scoring_profile = (
            scoring_profile
            or os.environ.get("IASW_SCORING_PROFILE", "")
            or DEFAULT_SCORING_PROFILE
        ).strip().lower()

        if self.scoring_profile not in IASW_SCORING_PROFILES:
            raise ValueError(
                f"Unsupported IAS-W scoring profile: {self.scoring_profile}. "
                f"Available: {list(IASW_SCORING_PROFILES.keys())}"
            )

        self.scoring_config = dict(
            IASW_SCORING_PROFILES[self.scoring_profile]
        )

        if isinstance(embedder_path_or_model, str):
            self.embedder = SentenceTransformer(embedder_path_or_model).to(self.device)
        else:
            self.embedder = embedder_path_or_model

        self.original_dim = int(self.embedder.get_sentence_embedding_dimension())

        self.stats_file = self._resolve_stats_file(stats_file)
        self.stats = self._load_stats(self.stats_file)

        stats_profiles = self.stats.get(
            "scoring_profiles",
            {}
        )

        # Runtime scoring is stats-authoritative.
        #
        # IMPORTANT:
        # - New stats.pt files store the exact scoring profile used to build
        #   their empirical calibration; runtime must use that saved profile.
        # - Legacy stats.pt files without scoring_profiles keep the historical
        #   source defaults above. Do NOT apply dataset overrides here, because
        #   that would create a new-score / old-calibration mismatch.
        if (
            isinstance(stats_profiles, dict)
            and self.scoring_profile in stats_profiles
        ):
            self.scoring_config.update(
                stats_profiles[self.scoring_profile]
            )

        self.mean = self.stats["mean"].float().to(self.device)
        self.components = self.stats["components"].float().to(self.device)
        self.eigvals = self.stats["eigvals"].float().to(self.device)
        self.eps = float(self.stats.get("eps", 1e-4))

        self.vector_dim = int(self.components.shape[1])
        self.calibration = self.stats.get("calibration", {})
        self.calibration_profiles = self.stats.get("calibration_profiles", {})

        self._validate_scoring_calibration_consistency()

        if int(self.components.shape[0]) != self.original_dim:
            raise ValueError(
                "IAS-W stats dimension mismatch: "
                f"embedder_dim={self.original_dim}, "
                f"stats_original_dim={self.components.shape[0]}, "
                f"stats_file={self.stats_file}"
            )

        print(
            f"[IAS-W] Loaded stats: {self.stats_file} | "
            f"dataset={self.dataset_type}, model_key={self.model_key}, "
            f"target_len={self.target_len}, whiten_dim={self.vector_dim}, "
            f"alignment={self.alignment_mode}, profile={self.scoring_profile}, "
            f"local_gap={self.scoring_config['local_gap_penalty']}, "
            f"local_norm={self.scoring_config['local_normalization']}"
        )

    def _validate_scoring_calibration_consistency(self) -> None:
        """Fail fast when a stats file mixes one scoring rule with another calibration.

        Legacy stats without profile metadata are allowed and use the historical
        source defaults. Profile-aware stats must be internally self-consistent.
        """
        profile_pack = self.calibration_profiles.get(self.scoring_profile, {})
        if not isinstance(profile_pack, dict):
            return

        saved_scoring = profile_pack.get("scoring", None)
        if not isinstance(saved_scoring, dict):
            return

        def canonical(cfg: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "global_gap_penalty": float(cfg.get("global_gap_penalty", -0.05)),
                "local_gap_penalty": float(cfg.get("local_gap_penalty", -0.05)),
                "local_normalization": str(cfg.get("local_normalization", "area")),
            }

        active = canonical(self.scoring_config)
        calibrated = canonical(saved_scoring)
        if active != calibrated:
            raise RuntimeError(
                "IAS-W scoring/calibration mismatch detected. "
                f"profile={self.scoring_profile}, active={active}, "
                f"calibrated={calibrated}, stats_file={self.stats_file}. "
                "Recalibrate this stats file with the same scoring profile "
                "instead of mixing new scoring with old mu/sigma."
            )

    # ------------------------------------------------------------
    # Stats loading
    # ------------------------------------------------------------

    def _resolve_stats_file(self, stats_file: Optional[str]) -> str:
        if stats_file:
            resolved = _resolve_existing_path(stats_file)
            if os.path.exists(resolved):
                return resolved
            raise FileNotFoundError(f"IAS-W stats file not found: {stats_file}")

        env_stats = (
            os.environ.get("IASW_STATS_FILE", "")
            or os.environ.get("IAS_STATS_FILE", "")
        ).strip()

        if env_stats:
            resolved = _resolve_existing_path(env_stats)
            if os.path.exists(resolved):
                return resolved
            raise FileNotFoundError(f"IAS-W stats file from environment not found: {env_stats}")

        registry_stats = self._resolve_stats_from_registry()
        if registry_stats:
            resolved = _resolve_existing_path(registry_stats)
            if os.path.exists(resolved):
                return resolved
            raise FileNotFoundError(
                "IAS-W stats file from IASW_STATS_REGISTRY not found:\n"
                f"  registry path = {registry_stats}\n"
                f"  resolved path = {resolved}"
            )

        raise ValueError(
            "IAS-W requires stats file, but none was found.\n"
            "Please do one of the following:\n"
            "  1) export IASW_STATS_FILE=/path/to/ias_stats.pt\n"
            "  2) run src/calibration/calibrate_empirical_iasw.py to build/update IAS-W stats\n"
            "  3) pass stats_file into IASWMark\n"
        )

    def _resolve_stats_from_registry(self) -> str:
        registry = IASW_STATS_REGISTRY

        ds_table = registry.get(self.dataset_type, None)
        if ds_table is None:
            ds_table = registry.get("c4", None)

        if not ds_table:
            return ""

        model_table = ds_table.get(self.model_key, None)
        if model_table is None:
            model_table = ds_table.get("llama3.1-8b", None)

        if not model_table:
            return ""

        # seq_len exact match first, otherwise use closest.
        len_keys = []
        for k in model_table.keys():
            try:
                len_keys.append(int(k))
            except Exception:
                pass

        if not len_keys:
            return ""

        if self.target_len in len_keys:
            selected_len = self.target_len
        else:
            selected_len = min(len_keys, key=lambda x: abs(x - self.target_len))
            print(
                f"[IAS-W][WARN] No exact stats for len={self.target_len}; "
                f"use closest len={selected_len}."
            )

        len_table = model_table.get(selected_len, None)
        if len_table is None:
            len_table = model_table.get(str(selected_len), None)

        if not isinstance(len_table, dict) or not len_table:
            return ""

        if self.stats_key:
            if self.stats_key not in len_table:
                raise KeyError(
                    f"IAS_STATS_KEY={self.stats_key} not found in registry for "
                    f"dataset={self.dataset_type}, model={self.model_key}, len={selected_len}. "
                    f"Available keys: {list(len_table.keys())}"
                )
            return len_table[self.stats_key]

        if len(len_table) == 1:
            only_key = next(iter(len_table.keys()))
            return len_table[only_key]

        available = list(len_table.keys())
        raise ValueError(
            "Multiple IAS stats candidates found in IASW_STATS_REGISTRY.\n"
            f"dataset={self.dataset_type}, model={self.model_key}, len={selected_len}\n"
            f"available keys={available}\n"
            "Please set IASW_STATS_KEY or IASW_STATS_FILE. Legacy IAS_STATS_KEY / IAS_STATS_FILE aliases are still accepted."
        )

    @staticmethod
    def _load_stats(stats_file: str) -> Dict[str, Any]:
        try:
            stats = torch.load(stats_file, map_location="cpu", weights_only=False)
        except TypeError:
            stats = torch.load(stats_file, map_location="cpu")

        required = ["mean", "components", "eigvals", "calibration"]
        missing = [k for k in required if k not in stats]

        if missing:
            raise KeyError(
                f"IAS-W stats file missing keys: {missing}. "
                f"stats_file={stats_file}"
            )
            

        return stats

    # ------------------------------------------------------------
    # Encoding and whitening
    # ------------------------------------------------------------

    @torch.no_grad()
    def _encode_raw(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.original_dim), device=self.device)

        emb = self.embedder.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )

        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def _whiten(self, emb: torch.Tensor) -> torch.Tensor:
        x = emb - self.mean
        z = torch.matmul(x, self.components)
        z = z / torch.sqrt(self.eigvals + self.eps)
        z = F.normalize(z, p=2, dim=1)
        return z

    @torch.no_grad()
    def encode_whitened(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.vector_dim), device=self.device)

        raw = self._encode_raw(texts)
        return self._whiten(raw)

    # ------------------------------------------------------------
    # Target vector and candidate selection
    # ------------------------------------------------------------

    def get_target_semantic_vector(self, sentence_index: int) -> torch.Tensor:
        payload = f"{self.secret_key}_{sentence_index}"
        seed = int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:8],
            byteorder="big",
        )

        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)

        random_vector = torch.randn(
            1,
            self.vector_dim,
            generator=rng,
            device=self.device,
        )

        return F.normalize(random_vector, p=2, dim=1)

    @torch.no_grad()
    def select_best_candidate(self, candidates: List[str], sentence_index: int) -> Tuple[str, float]:
        if not candidates:
            return "", 0.0

        target_vector = self.get_target_semantic_vector(sentence_index)
        embeddings = self.encode_whitened(candidates)
        similarities = torch.mm(embeddings, target_vector.T).squeeze(1)

        best_idx = torch.argmax(similarities).item()
        return candidates[best_idx], float(similarities[best_idx].item())

    # ------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------

    def _get_calibration_params(self, mode: str) -> Tuple[float, float]:
        item = None

        profile_pack = self.calibration_profiles.get(
            self.scoring_profile, {}
        )

        if isinstance(profile_pack, dict):
            profile_cal = profile_pack.get("calibration", {})
            if isinstance(profile_cal, dict):
                item = profile_cal.get(mode, None)

        # For strict/window/global (or legacy stats), fall back to the
        # historical calibration dictionary.
        if item is None:
            item = self.calibration.get(mode, None)

        # For profile-specific Local-DP detection, do not silently use the
        # wrong calibration if the profile has not been installed.
        if (
            mode == "dp_local"
            and self.scoring_profile == "standalone"
            and (
                not isinstance(profile_pack, dict)
                or not isinstance(profile_pack.get("calibration", {}), dict)
                or "dp_local" not in profile_pack.get("calibration", {})
            )
        ):
            raise RuntimeError(
                "IAS-W standalone profile requires profile-specific "
                "dp_local calibration. Run "
                "the scoring-profile install step (analysis/iasw_alignment) "
                "after the alignment-fusion sweep."
            )

        if item is None:
            return 0.0, 1.0

        if isinstance(item, dict):
            mu = safe_float(item.get("mu", 0.0))
            sigma = safe_float(item.get("sigma", 1.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            mu = safe_float(item[0])
            sigma = safe_float(item[1], 1.0)
        else:
            return 0.0, 1.0

        if sigma <= 1e-8:
            sigma = 1.0

        return mu, sigma

    # ------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------

    @torch.no_grad()
    def _compute_raw_scores(self, text: str) -> Dict[str, Any]:
        sentences = split_sentences(text)

        if not sentences:
            raw_scores = {
                "strict": 0.0,
                "window": 0.0,
                "dp_global": 0.0,
                "dp_local": 0.0,
            }

            return {
                "raw_scores": raw_scores,
                "num_sentences": 0,
                "sim_matrix_shape": [0, 0],
            }

        embeddings = self.encode_whitened(sentences)
        n_sent = len(sentences)
        n_targets = n_sent + 10

        targets = []
        for i in range(n_targets):
            targets.append(self.get_target_semantic_vector(i).squeeze(0))

        target_mat = torch.stack(targets, dim=0).to(self.device)
        sim_matrix = torch.matmul(embeddings, target_mat.T).detach().cpu().numpy()

        raw_scores = compute_all_alignment_scores(
            sim_matrix,
            window_size=ALIGNMENT_WINDOW_SIZE,
            global_gap_penalty=float(
                self.scoring_config["global_gap_penalty"]
            ),
            local_gap_penalty=float(
                self.scoring_config["local_gap_penalty"]
            ),
            local_normalization=str(
                self.scoring_config["local_normalization"]
            ),
        )

        return {
            "raw_scores": raw_scores,
            "num_sentences": n_sent,
            "sim_matrix_shape": [int(sim_matrix.shape[0]), int(sim_matrix.shape[1])],
        }

    @torch.no_grad()
    def detect(self, text: str) -> Dict[str, Any]:
        pack = self._compute_raw_scores(text)
        raw_scores = pack["raw_scores"]

        z_scores = {}
        for mode, raw_score in raw_scores.items():
            mu, sigma = self._get_calibration_params(mode)
            z_scores[mode] = float((float(raw_score) - mu) / sigma)

        main_raw = float(raw_scores.get(self.alignment_mode, 0.0))
        main_z = float(z_scores.get(self.alignment_mode, 0.0))

        return {
            "z_score": main_z,
            "mean_sim": main_raw,
            "num_sentences": int(pack.get("num_sentences", 0)),
            "ablation_raw_scores": raw_scores,
            "ablation_z_scores": z_scores,
            "ias_details": {
                "method": "ias_w",
                "method_full_name": "Whitened Independent Anchor Semantic Watermark",
                "stats_file": self.stats_file,
                "stats_key": self.stats.get("stats_key", self.stats_key),
                "dataset_type": self.dataset_type,
                "model_key": self.model_key,
                "target_len": self.target_len,
                "alignment_mode": self.alignment_mode,
                "original_dim": self.original_dim,
                "whiten_dim": self.vector_dim,
                "sim_matrix_shape": pack.get("sim_matrix_shape", [0, 0]),
                "scoring_profile": self.scoring_profile,
                "alignment_scoring": {
                    "window_size": ALIGNMENT_WINDOW_SIZE,
                    "global_gap_penalty": float(
                        self.scoring_config["global_gap_penalty"]
                    ),
                    "local_gap_penalty": float(
                        self.scoring_config["local_gap_penalty"]
                    ),
                    "local_normalization": str(
                        self.scoring_config["local_normalization"]
                    ),
                },
            },
            # Keep this alias for older IAS-W analysis scripts.
            "iasw_details": {
                "stats_file": self.stats_file,
                "alignment_mode": self.alignment_mode,
                "whiten_dim": self.vector_dim,
                "sim_matrix_shape": pack.get("sim_matrix_shape", [0, 0]),
                "scoring_profile": self.scoring_profile,
                "alignment_scoring": {
                    "window_size": ALIGNMENT_WINDOW_SIZE,
                    "global_gap_penalty": float(
                        self.scoring_config["global_gap_penalty"]
                    ),
                    "local_gap_penalty": float(
                        self.scoring_config["local_gap_penalty"]
                    ),
                    "local_normalization": str(
                        self.scoring_config["local_normalization"]
                    ),
                },
            },
        }