# src/calibration/calibrate_empirical_iasw.py
# -*- coding: utf-8 -*-
"""
Build IAS-W whitening statistics and profile-specific empirical calibration.

This script replaces the old IAS empirical calibration script.

Old IAS calibration:
    raw alignment score -> manually copied CALIBRATION_TABLE in ias_w_mark.py

New IAS-W calibration:
    1. collect natural sentence embeddings from calibration baseline
    2. compute whitening statistics:
        mean, components, eigvals, eps, rank
    3. compute empirical calibration for official IAS-W scoring profiles:
        dual       : Local-DP gap=-0.05, normalization=area
        standalone : Local-DP gap=-0.125, normalization=sqrt_area
    4. save everything into a .pt stats file
    5. save a readable .json report
    6. optionally update IASW_STATS_REGISTRY in src/watermarks/ias_w_mark.py

Recommended input:
    A calibration baseline JSONL containing human_text and no_wm_text.

Example:
    python src/calibration/calibrate_empirical_iasw.py \
        --baseline_file data/calibration/c4/baseline_len200_skip3000_n1000.jsonl \
        --stats_file cache/iasw_stats/ias_stats_c4_llama3.1-8b-instruct_len200_rank64_eps0p0001_calibskip3000_calibn1000.pt \
        --dataset_type c4 \
        --model_path ./models/llama3.1-8b-instruct \
        --embedder_path ./models/bge-small-en-v1.5 \
        --seq_len 200 \
        --rank 64 \
        --eps 0.0001 \
        --num_samples 1000 \
        --max_stats_sentences 20000 \
        --update_iasw_mark \
        --iasw_mark_file src/watermarks/ias_w_mark.py

The generated .pt file is the real runtime dependency for IAS-W.
The registry only stores the path to this .pt file.
"""

import os
import re
import ast
import sys
import json
import pprint
import hashlib
import argparse
import traceback
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# Ensure project root is importable when executed from arbitrary cwd.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.watermarks.ias_w_mark import (
    ALIGNMENT_WINDOW_SIZE,
    IASW_SCORING_PROFILES,
    compute_all_alignment_scores,
    get_iasw_scoring_profiles_for_dataset,
)


# ============================================================
# Constants
# ============================================================

REGISTRY_START = "# ================= IASW_STATS_REGISTRY_START ================="
REGISTRY_END = "# ================= IASW_STATS_REGISTRY_END ================="

BASE_ALIGNMENT_MODES = [
    "strict",
    "window",
    "dp_global",
    "dp_local",
]

ALL_SCORE_MODES = [
    "strict",
    "window",
    "dp_global",
    "dp_local",
]


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build IAS-W whitening statistics and empirical calibration."
    )

    # New recommended input.
    parser.add_argument(
        "--baseline_file",
        type=str,
        default="",
        help=(
            "Calibration baseline JSONL. Recommended. "
            "Expected fields: human_text and no_wm_text."
        ),
    )

    # Backward-compatible raw dataset input.
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help=(
            "Optional raw human dataset JSONL for backward compatibility. "
            "If baseline_file is provided, dataset_path is ignored."
        ),
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Text field used when --dataset_path is provided.",
    )

    parser.add_argument(
        "--text_fields",
        type=str,
        default="human_text,no_wm_text",
        help=(
            "Comma-separated text fields used from baseline_file. "
            "Default: human_text,no_wm_text"
        ),
    )

    parser.add_argument(
        "--stats_file",
        type=str,
        default="",
        help="Output .pt stats file. If omitted, a path under cache/iasw_stats will be generated.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help=(
            "Optional readable JSON report path. "
            "If omitted, report will be saved as stats_file + '.json'. "
            "For backward compatibility, if this ends with .pt and stats_file is empty, "
            "it will be treated as stats_file."
        ),
    )

    parser.add_argument("--dataset_type", type=str, required=True, help="c4/wiki/dolly/etc.")
    parser.add_argument("--model_path", type=str, required=True, help="Tokenizer/LLM path.")
    parser.add_argument("--embedder_path", type=str, required=True, help="Sentence embedder path.")

    parser.add_argument("--num_samples", type=int, default=1000, help="Number of baseline rows to use.")
    parser.add_argument("--seq_len", type=int, default=200, help="Target token length used for registry key.")
    parser.add_argument("--rank", type=int, default=64, help="Whitening rank.")
    parser.add_argument("--eps", type=float, default=1e-4, help="Whitening epsilon.")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--max_stats_sentences", type=int, default=20000, help="Max sentences for whitening.")
    parser.add_argument(
        "--max_calibration_texts",
        type=int,
        default=-1,
        help="Max negative texts for calibration. -1 means all collected texts.",
    )

    parser.add_argument(
        "--crop_baseline_to_seq_len",
        action="store_true",
        help=(
            "If set, crop baseline text fields to seq_len using tokenizer. "
            "Usually not needed if baseline_file was already generated with target_len."
        ),
    )

    parser.add_argument("--secret_key", type=str, default="DUAL_MACRO_2026")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--stats_key",
        type=str,
        default="",
        help=(
            "Optional registry key. If omitted, generated as "
            "rank{rank}_eps{eps}_calibn{num_samples}."
        ),
    )

    parser.add_argument(
        "--update_iasw_mark",
        action="store_true",
        help="Update IASW_STATS_REGISTRY in ias_w_mark.py with the generated stats file path.",
    )

    parser.add_argument(
        "--iasw_mark_file",
        type=str,
        default="src/watermarks/ias_w_mark.py",
        help="Path to ias_w_mark.py for --update_iasw_mark.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=("full", "recalibrate"),
        default="full",
        help=(
            "full: rebuild whitening + empirical calibration; "
            "recalibrate: load whitening from existing stats_file and only rebuild "
            "profile-specific empirical calibration."
        ),
    )

    parser.add_argument(
        "--backup_existing",
        action="store_true",
        help=(
            "In recalibrate mode, save a byte-for-byte backup as "
            "<stats_file>.pre_recalibrate.bak before overwriting the stats file."
        ),
    )

    return parser.parse_args()


# ============================================================
# Basic utilities
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] JSON parse failed at line {line_idx}: {e}")

    return rows


def save_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def parse_text_fields(text_fields: str) -> List[str]:
    fields = []
    for item in (text_fields or "").split(","):
        item = item.strip()
        if item:
            fields.append(item)

    if not fields:
        fields = ["human_text", "no_wm_text"]

    return fields


def infer_model_key(model_path: str) -> str:
    """
    Infer a stable model key for registry lookup.
    Keep this consistent with future IASMark.
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


def eps_to_tag(eps: float) -> str:
    """
    0.0001 -> 0p0001
    1e-05  -> 1em05 if scientific notation appears
    """
    s = f"{eps:g}"
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    s = s.replace("+", "")
    return s


def make_stats_key(args) -> str:
    if args.stats_key:
        return args.stats_key.strip()

    eps_tag = eps_to_tag(args.eps)
    return f"rank{int(args.rank)}_eps{eps_tag}_calibn{int(args.num_samples)}"


def make_default_stats_file(args, model_key: str, stats_key: str) -> str:
    """
    Use the active IAS-W cache directory while preserving the historical
    ``ias_stats_*.pt`` filename convention for compatibility.
    """
    dataset = args.dataset_type.lower()
    model_tag = model_key.replace("/", "_")
    filename = f"ias_stats_{dataset}_{model_tag}_len{int(args.seq_len)}_{stats_key}.pt"
    return os.path.join("cache", "iasw_stats", dataset, model_tag, filename)


def resolve_output_paths(args, model_key: str, stats_key: str) -> Tuple[str, str]:
    stats_file = args.stats_file.strip()

    # Backward convenience: if --output_file ends with .pt and --stats_file is empty,
    # use output_file as stats_file.
    if not stats_file and args.output_file and args.output_file.strip().endswith(".pt"):
        stats_file = args.output_file.strip()

    if not stats_file:
        stats_file = make_default_stats_file(args, model_key, stats_key)

    if args.output_file and args.output_file.strip() and not args.output_file.strip().endswith(".pt"):
        report_file = args.output_file.strip()
    else:
        report_file = stats_file + ".json"

    return stats_file, report_file


def crop_text_to_seq_len(text: str, tokenizer, seq_len: int) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return ""

    tokens = tokens[:seq_len]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ============================================================
# Data loading
# ============================================================

def load_rows_from_raw_dataset(args) -> List[Dict[str, Any]]:
    if not args.dataset_path:
        raise ValueError("Either --baseline_file or --dataset_path must be provided.")

    print(f"[*] Loading tokenizer for raw dataset cropping: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    rows = []
    valid_count = 0
    error_count = 0

    print(f"[*] Loading raw dataset: {args.dataset_path}")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=args.num_samples, desc="Loading raw texts")
        for line in f:
            if args.num_samples > 0 and valid_count >= args.num_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                raw_text = obj.get(args.text_column, "")

                if not isinstance(raw_text, str) or not raw_text.strip():
                    continue

                tokens = tokenizer.encode(raw_text, add_special_tokens=False)
                if len(tokens) < max(50, args.seq_len // 2):
                    continue

                cropped = tokenizer.decode(tokens[:args.seq_len], skip_special_tokens=True)

                if not cropped.strip():
                    continue

                rows.append({"human_text": cropped})
                valid_count += 1
                pbar.update(1)

            except Exception as e:
                error_count += 1
                if error_count == 1:
                    print(f"\n[WARN] First raw dataset error: {e}")
                    traceback.print_exc()
                continue

        pbar.close()

    print(f"[*] Raw dataset valid rows: {len(rows)} | errors: {error_count}")
    return rows


def load_calibration_rows(args) -> Tuple[List[Dict[str, Any]], List[str], str]:
    """
    Return:
        rows, text_fields, input_kind
    """
    if args.baseline_file:
        print(f"[*] Loading calibration baseline: {args.baseline_file}")
        rows = load_jsonl(args.baseline_file)

        if args.num_samples > 0:
            rows = rows[:args.num_samples]

        text_fields = parse_text_fields(args.text_fields)

        if args.crop_baseline_to_seq_len:
            print(f"[*] Cropping baseline fields to seq_len={args.seq_len}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)

            for row in tqdm(rows, desc="Cropping baseline"):
                for field in text_fields:
                    text = row.get(field, "")
                    if isinstance(text, str) and text.strip():
                        row[field] = crop_text_to_seq_len(text, tokenizer, args.seq_len)

        return rows, text_fields, "baseline_file"

    rows = load_rows_from_raw_dataset(args)
    return rows, ["human_text"], "dataset_path"


def collect_sentences_from_rows(
    rows: List[Dict[str, Any]],
    fields: List[str],
    max_sentences: int,
) -> List[str]:
    sentences = []

    for row in rows:
        for field in fields:
            text = row.get(field, "")
            if not isinstance(text, str) or not text.strip():
                continue

            for sent in split_sentences(text):
                if sent:
                    sentences.append(sent)

                if max_sentences > 0 and len(sentences) >= max_sentences:
                    return sentences

    return sentences


def collect_texts_from_rows(
    rows: List[Dict[str, Any]],
    fields: List[str],
) -> List[str]:
    texts = []

    for row in rows:
        for field in fields:
            text = row.get(field, "")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    return texts


# ============================================================
# Alignment scoring
# ============================================================

# Alignment scoring is imported from src.watermarks.ias_w_mark so
# calibration and runtime use the exact same implementation and profile definitions.


# ============================================================
# Whitening and IAS-W raw score computation
# ============================================================

@torch.no_grad()
def encode_texts(
    embedder,
    texts: List[str],
    device: str,
    batch_size: int,
    show_progress_bar: bool = True,
) -> torch.Tensor:
    if not texts:
        dim = embedder.get_sentence_embedding_dimension()
        return torch.empty((0, dim), device=device)

    emb = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=show_progress_bar,
    )

    emb = F.normalize(emb, p=2, dim=1)
    return emb


def compute_whitening_stats(
    embeddings: torch.Tensor,
    rank: int,
    eps: float,
) -> Dict[str, Any]:
    x = embeddings.detach().float().cpu()

    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"Need at least 2 embeddings, got shape={tuple(x.shape)}")

    n, d = x.shape

    mean = x.mean(dim=0)
    xc = x - mean

    cov = torch.matmul(xc.T, xc) / max(n - 1, 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)

    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if rank <= 0 or rank > d:
        rank = d

    eigvals = eigvals[:rank].clamp_min(float(eps)).contiguous()
    components = eigvecs[:, :rank].contiguous()

    return {
        "mean": mean.contiguous(),
        "components": components,
        "eigvals": eigvals,
        "eps": float(eps),
        "rank": int(rank),
        "original_dim": int(d),
        "num_stats_sentences": int(n),
    }


@torch.no_grad()
def whiten_embeddings(
    raw_embeddings: torch.Tensor,
    stats: Dict[str, Any],
    device: str,
) -> torch.Tensor:
    mean = stats["mean"].float().to(device)
    components = stats["components"].float().to(device)
    eigvals = stats["eigvals"].float().to(device)
    eps = float(stats.get("eps", 1e-4))

    x = raw_embeddings - mean
    z = torch.matmul(x, components)
    z = z / torch.sqrt(eigvals + eps)
    z = F.normalize(z, p=2, dim=1)

    return z


def get_target_vector(
    sentence_index: int,
    vector_dim: int,
    secret_key: str,
    device: str,
) -> torch.Tensor:
    payload = f"{secret_key}_{sentence_index}"
    seed = int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8],
        byteorder="big",
    )

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    v = torch.randn(
        1,
        vector_dim,
        generator=rng,
        device=device,
    )

    return F.normalize(v, p=2, dim=1)


@torch.no_grad()
def compute_iasw_similarity_matrix_for_text(
    text: str,
    embedder,
    stats: Dict[str, Any],
    device: str,
    secret_key: str,
) -> Optional[np.ndarray]:
    """Build the whitened sentence-to-anchor similarity matrix once per text."""
    sentences = split_sentences(text)
    if not sentences:
        return None

    raw_embeddings = encode_texts(
        embedder=embedder,
        texts=sentences,
        device=device,
        batch_size=64,
        show_progress_bar=False,
    )
    observed = whiten_embeddings(raw_embeddings, stats=stats, device=device)

    n_sent = len(sentences)
    n_targets = n_sent + 10
    vector_dim = int(stats["components"].shape[1])

    targets = [
        get_target_vector(
            sentence_index=i,
            vector_dim=vector_dim,
            secret_key=secret_key,
            device=device,
        ).squeeze(0)
        for i in range(n_targets)
    ]

    target_mat = torch.stack(targets, dim=0).to(device)
    return torch.matmul(observed, target_mat.T).detach().cpu().numpy()


def score_similarity_matrix_for_profile(
    sim_matrix: Optional[np.ndarray],
    scoring_profile: str,
    scoring_profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    if sim_matrix is None:
        return {mode: 0.0 for mode in BASE_ALIGNMENT_MODES}

    if scoring_profile not in scoring_profiles:
        raise ValueError(
            f"Unknown IAS-W scoring profile: {scoring_profile}. "
            f"Available: {list(scoring_profiles)}"
        )

    cfg = scoring_profiles[scoring_profile]
    return compute_all_alignment_scores(
        sim_matrix,
        window_size=ALIGNMENT_WINDOW_SIZE,
        global_gap_penalty=float(cfg["global_gap_penalty"]),
        local_gap_penalty=float(cfg["local_gap_penalty"]),
        local_normalization=str(cfg["local_normalization"]),
    )


def summarize_empirical_scores(
    raw_by_mode: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    calibration = {}
    for mode, values in raw_by_mode.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            mu, sigma = 0.0, 1.0
        else:
            mu = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=0))
            if sigma < 1e-8:
                sigma = 1.0

        calibration[mode] = {
            "mu": mu,
            "sigma": sigma,
            "n": int(len(arr)),
        }
    return calibration


def build_empirical_calibration_profiles(
    texts: List[str],
    embedder,
    stats: Dict[str, Any],
    device: str,
    secret_key: str,
    scoring_profiles: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, List[float]]]]:
    """Build empirical calibration for every official IAS-W scoring profile."""
    raw_profiles = {
        profile_name: {mode: [] for mode in BASE_ALIGNMENT_MODES}
        for profile_name in scoring_profiles
    }

    error_count = 0
    for text in tqdm(texts, desc="Calibrating IAS-W profiles"):
        try:
            sim_matrix = compute_iasw_similarity_matrix_for_text(
                text=text,
                embedder=embedder,
                stats=stats,
                device=device,
                secret_key=secret_key,
            )

            for profile_name in scoring_profiles:
                raw_scores = score_similarity_matrix_for_profile(
                    sim_matrix,
                    scoring_profile=profile_name,
                    scoring_profiles=scoring_profiles,
                )
                for mode in BASE_ALIGNMENT_MODES:
                    raw_profiles[profile_name][mode].append(
                        float(raw_scores.get(mode, 0.0))
                    )
        except Exception as e:
            error_count += 1
            if error_count == 1:
                print(f"\n[WARN] First calibration error: {e}")
                traceback.print_exc()

    if error_count > 0:
        print(f"[WARN] Calibration errors skipped: {error_count}")

    calibration_profiles = {}
    for profile_name, raw_by_mode in raw_profiles.items():
        calibration_profiles[profile_name] = {
            "scoring": dict(scoring_profiles[profile_name]),
            "calibration": summarize_empirical_scores(raw_by_mode),
            "source": "src/calibration/calibrate_empirical_iasw.py",
        }

    return calibration_profiles, raw_profiles


# ============================================================
# IASMark registry updater
# ============================================================

def _extract_registry_from_content(content: str) -> Dict[str, Any]:
    start_idx = content.find(REGISTRY_START)
    end_idx = content.find(REGISTRY_END)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return {}

    block = content[start_idx:end_idx]

    m = re.search(r"IASW_STATS_REGISTRY\s*=\s*(\{.*\})", block, flags=re.S)
    if not m:
        return {}

    try:
        obj = ast.literal_eval(m.group(1))
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        print(f"[WARN] Failed to parse existing IASW_STATS_REGISTRY: {e}")

    return {}


def _make_registry_block(registry: Dict[str, Any]) -> str:
    registry_text = pprint.pformat(
        registry,
        indent=4,
        width=120,
        sort_dicts=True,
    )

    return (
        f"{REGISTRY_START}\n"
        f"IASW_STATS_REGISTRY = {registry_text}\n"
        f"{REGISTRY_END}\n"
    )


def _project_root_from_iasw_mark_file(iasw_mark_file: str) -> str:
    # Expected: <project_root>/src/watermarks/ias_w_mark.py
    abs_file = os.path.abspath(iasw_mark_file)
    watermarks_dir = os.path.dirname(abs_file)
    src_dir = os.path.dirname(watermarks_dir)
    project_root = os.path.dirname(src_dir)
    return project_root


def _normalize_registry_path(stats_file: str, iasw_mark_file: str) -> str:
    """
    Prefer a project-relative path when possible.
    """
    stats_abs = os.path.abspath(stats_file)
    project_root = _project_root_from_iasw_mark_file(iasw_mark_file)

    try:
        rel = os.path.relpath(stats_abs, project_root)
        if not rel.startswith(".."):
            return rel.replace("\\", "/")
    except Exception:
        pass

    return stats_file.replace("\\", "/")


def update_iasw_mark_registry(
    iasw_mark_file: str,
    dataset_type: str,
    model_key: str,
    seq_len: int,
    stats_key: str,
    stats_file: str,
):
    if not os.path.exists(iasw_mark_file):
        raise FileNotFoundError(f"ias_w_mark.py not found: {iasw_mark_file}")

    with open(iasw_mark_file, "r", encoding="utf-8") as f:
        content = f.read()

    registry = _extract_registry_from_content(content)

    dataset_type = dataset_type.lower()
    model_key = model_key.lower()
    seq_len_key = int(seq_len)

    registry_path = _normalize_registry_path(stats_file, iasw_mark_file)

    registry.setdefault(dataset_type, {})
    registry[dataset_type].setdefault(model_key, {})
    registry[dataset_type][model_key].setdefault(seq_len_key, {})
    registry[dataset_type][model_key][seq_len_key][stats_key] = registry_path

    new_block = _make_registry_block(registry)

    start_idx = content.find(REGISTRY_START)
    end_idx = content.find(REGISTRY_END)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_idx = end_idx + len(REGISTRY_END)

        new_content = (
            content[:start_idx]
            + new_block.rstrip()
            + content[end_idx:]
        )
    else:
        # Insert before compute_all_alignment_scores if possible.
        insert_pos = content.find("def compute_all_alignment_scores")
        if insert_pos == -1:
            insert_pos = content.find("class IASMark")

        if insert_pos == -1:
            # Fallback: prepend after encoding line/module docstring area.
            new_content = new_block + "\n\n" + content
        else:
            new_content = (
                content[:insert_pos]
                + new_block
                + "\n"
                + content[insert_pos:]
            )

    with open(iasw_mark_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("[+] IASW_STATS_REGISTRY updated:")
    print(f"    file      = {iasw_mark_file}")
    print(f"    dataset   = {dataset_type}")
    print(f"    model_key = {model_key}")
    print(f"    seq_len   = {seq_len_key}")
    print(f"    stats_key = {stats_key}")
    print(f"    path      = {registry_path}")


# ============================================================
# Existing-stats / reproducibility helpers
# ============================================================

def load_torch_stats(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def whitening_fingerprint(stats: Dict[str, Any]) -> str:
    """Stable fingerprint of generation/detection shared whitening state."""
    h = hashlib.sha256()
    for key in ("mean", "components", "eigvals"):
        if key not in stats:
            raise KeyError(f"Existing IAS-W stats missing whitening key: {key}")
        tensor = stats[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    h.update(f"eps={float(stats.get('eps', 1e-4)):.17g}".encode("utf-8"))
    h.update(f"rank={int(stats.get('rank', stats['components'].shape[1]))}".encode("utf-8"))
    return h.hexdigest()


def validate_existing_stats_for_recalibration(
    stats: Dict[str, Any],
    args,
    dataset_type: str,
    model_key: str,
    stats_file: str,
    embedder_dim: int,
) -> None:
    required = ("mean", "components", "eigvals")
    missing = [k for k in required if k not in stats]
    if missing:
        raise KeyError(
            f"Cannot recalibrate: existing stats missing whitening keys {missing}: {stats_file}"
        )

    components = stats["components"]
    if int(components.shape[0]) != int(embedder_dim):
        raise ValueError(
            "Cannot recalibrate: embedder dimension does not match existing whitening. "
            f"embedder_dim={embedder_dim}, stats_dim={components.shape[0]}, stats_file={stats_file}"
        )

    existing_rank = int(components.shape[1])
    if int(args.rank) != existing_rank:
        raise ValueError(
            "Cannot recalibrate with a different whitening rank. "
            f"requested_rank={args.rank}, existing_rank={existing_rank}. "
            "Use --mode full only if you intentionally want a new whitening transform."
        )

    existing_eps = float(stats.get("eps", args.eps))
    if not np.isclose(existing_eps, float(args.eps), rtol=0.0, atol=1e-12):
        raise ValueError(
            "Cannot recalibrate with a different whitening eps. "
            f"requested_eps={args.eps}, existing_eps={existing_eps}."
        )

    stored_dataset = str(stats.get("dataset_type", "")).lower().strip()
    if stored_dataset and stored_dataset != dataset_type:
        raise ValueError(
            f"Dataset mismatch: stats={stored_dataset}, requested={dataset_type}, file={stats_file}"
        )

    stored_model = str(stats.get("model_key", "")).lower().strip()
    if stored_model and stored_model != model_key.lower():
        raise ValueError(
            f"Model mismatch: stats={stored_model}, requested={model_key}, file={stats_file}"
        )


# ============================================================
# Main build function
# ============================================================

def build_iasw_stats(args):
    dataset_type = args.dataset_type.lower()
    model_key = infer_model_key(args.model_path)
    stats_key = make_stats_key(args)
    stats_file, report_file = resolve_output_paths(args, model_key, stats_key)
    build_mode = str(args.mode).strip().lower()

    print("=" * 100)
    print("Build IAS-W Stats")
    print("=" * 100)
    print(f"mode                  = {build_mode}")
    print(f"dataset_type          = {dataset_type}")
    print(f"model_path            = {args.model_path}")
    print(f"model_key             = {model_key}")
    print(f"embedder_path         = {args.embedder_path}")
    print(f"seq_len               = {args.seq_len}")
    print(f"rank                  = {args.rank}")
    print(f"eps                   = {args.eps}")
    print(f"num_samples           = {args.num_samples}")
    print(f"max_stats_sentences   = {args.max_stats_sentences}")
    print(f"max_calibration_texts = {args.max_calibration_texts}")
    print(f"stats_key             = {stats_key}")
    print(f"stats_file            = {stats_file}")
    print(f"report_file           = {report_file}")
    print("=" * 100)

    if build_mode == "recalibrate" and not args.baseline_file:
        raise ValueError("--mode recalibrate requires --baseline_file so the null sample set is explicit.")
    if build_mode == "recalibrate" and not os.path.exists(stats_file):
        raise FileNotFoundError(
            f"--mode recalibrate requires an existing stats file: {stats_file}"
        )

    rows, text_fields, input_kind = load_calibration_rows(args)
    if not rows:
        raise RuntimeError("No calibration rows loaded.")

    print(f"[*] Loaded rows: {len(rows)} | input_kind={input_kind}")
    print(f"[*] Text fields: {text_fields}")

    print(f"[*] Loading embedder: {args.embedder_path}")
    embedder = SentenceTransformer(args.embedder_path).to(args.device)
    embedder_dim = int(embedder.get_sentence_embedding_dimension())

    baseline_hash = ""
    if args.baseline_file and os.path.exists(args.baseline_file):
        baseline_hash = file_sha256(args.baseline_file)
        print(f"[*] Calibration baseline SHA256: {baseline_hash}")

    if build_mode == "full":
        print("[*] Collecting sentences for whitening...")
        stats_sentences = collect_sentences_from_rows(
            rows=rows,
            fields=text_fields,
            max_sentences=args.max_stats_sentences,
        )

        if len(stats_sentences) < 50:
            print(
                f"[WARN] Only {len(stats_sentences)} sentences collected. "
                f"Formal experiments should use more calibration data."
            )
        if len(stats_sentences) < 2:
            raise RuntimeError(f"Too few sentences for whitening: {len(stats_sentences)}")

        print(f"[*] Collected whitening sentences: {len(stats_sentences)}")
        print("[*] Encoding whitening sentences...")
        embeddings = encode_texts(
            embedder=embedder,
            texts=stats_sentences,
            device=args.device,
            batch_size=args.batch_size,
            show_progress_bar=True,
        )

        print("[*] Computing whitening statistics...")
        stats = compute_whitening_stats(
            embeddings=embeddings,
            rank=args.rank,
            eps=args.eps,
        )
        whitening_fp_before = whitening_fingerprint(stats)
        print(
            "[*] Whitening stats: "
            f"original_dim={stats['original_dim']}, "
            f"rank={stats['rank']}, "
            f"num_stats_sentences={stats['num_stats_sentences']}"
        )
        print(f"[*] Whitening fingerprint: {whitening_fp_before}")
    else:
        print("[*] Recalibration-only mode: loading EXISTING whitening from stats file.")
        stats = load_torch_stats(stats_file)
        validate_existing_stats_for_recalibration(
            stats=stats,
            args=args,
            dataset_type=dataset_type,
            model_key=model_key,
            stats_file=stats_file,
            embedder_dim=embedder_dim,
        )
        whitening_fp_before = whitening_fingerprint(stats)
        print(f"[*] Existing whitening fingerprint: {whitening_fp_before}")
        print("[*] mean/components/eigvals will NOT be recomputed.")

        if args.backup_existing:
            backup_file = stats_file + ".pre_recalibrate.bak"
            shutil.copy2(stats_file, backup_file)
            print(f"[*] Existing stats backup saved: {backup_file}")

    print("[*] Collecting calibration texts...")
    cal_texts = collect_texts_from_rows(rows=rows, fields=text_fields)
    if args.max_calibration_texts > 0:
        cal_texts = cal_texts[:args.max_calibration_texts]

    if len(cal_texts) < 10:
        print(
            f"[WARN] Only {len(cal_texts)} calibration texts collected. "
            f"Stats may be unstable."
        )
    if len(cal_texts) == 0:
        raise RuntimeError("No calibration texts collected.")
    print(f"[*] Calibration texts: {len(cal_texts)}")

    active_scoring_profiles = get_iasw_scoring_profiles_for_dataset(dataset_type)
    print("[*] Active IAS-W scoring profiles:")
    print(json.dumps(active_scoring_profiles, ensure_ascii=False, indent=2))

    print("[*] Building profile-specific empirical calibration...")
    calibration_profiles, raw_profiles = build_empirical_calibration_profiles(
        texts=cal_texts,
        embedder=embedder,
        stats=stats,
        device=args.device,
        secret_key=args.secret_key,
        scoring_profiles=active_scoring_profiles,
    )

    required_profiles = {"dual", "standalone"}
    missing_profiles = required_profiles - set(calibration_profiles)
    if missing_profiles:
        raise RuntimeError(f"IAS-W calibration missing profiles: {sorted(missing_profiles)}")

    bad_items = []
    for profile_name in sorted(required_profiles):
        profile_cal = calibration_profiles[profile_name].get("calibration", {})
        for mode in BASE_ALIGNMENT_MODES:
            item = profile_cal.get(mode, {})
            if int(item.get("n", 0)) <= 0:
                bad_items.append(f"{profile_name}:{mode}")
    if bad_items:
        raise RuntimeError(f"IAS-W calibration failed for: {bad_items}")

    # Install the exact profile definition used for this calibration.
    stats["calibration_profiles"] = calibration_profiles
    stats["scoring_profiles"] = {
        name: dict(config) for name, config in active_scoring_profiles.items()
    }
    stats["scoring_profile_scope"] = "dataset"
    stats["scoring_profile_dataset"] = dataset_type
    stats["default_scoring_profile"] = "dual"

    # Backward-compatible alias used by historical KGW+IAS-W code.
    stats["calibration"] = dict(calibration_profiles["dual"]["calibration"])
    stats["stats_key"] = stats_key
    stats["method"] = "ias_w"
    stats["method_full_name"] = "Whitened Independent Anchor Semantic Watermark"
    stats["dataset_type"] = dataset_type
    stats["model_key"] = model_key
    stats["model_path"] = args.model_path
    stats["embedder_path"] = args.embedder_path
    stats["secret_key_note"] = "Targets are generated from secret_key at runtime."

    if build_mode == "full":
        stats["build_config"] = {
            "input_kind": input_kind,
            "baseline_file": args.baseline_file,
            "baseline_sha256": baseline_hash,
            "dataset_path": args.dataset_path,
            "text_fields": text_fields,
            "text_column": args.text_column,
            "dataset_type": dataset_type,
            "model_path": args.model_path,
            "model_key": model_key,
            "embedder_path": args.embedder_path,
            "seq_len": int(args.seq_len),
            "rank": int(args.rank),
            "eps": float(args.eps),
            "num_samples": int(args.num_samples),
            "max_stats_sentences": int(args.max_stats_sentences),
            "max_calibration_texts": int(args.max_calibration_texts),
            "crop_baseline_to_seq_len": bool(args.crop_baseline_to_seq_len),
            "secret_key": args.secret_key,
            "scoring_profile_scope": "dataset",
            "scoring_profile_dataset": dataset_type,
            "scoring_profiles": {
                name: dict(config) for name, config in active_scoring_profiles.items()
            },
            "whitening_fingerprint": whitening_fp_before,
        }
    else:
        build_config = dict(stats.get("build_config", {}) or {})
        build_config["recalibration"] = {
            "mode": "recalibrate",
            "baseline_file": args.baseline_file,
            "baseline_sha256": baseline_hash,
            "dataset_type": dataset_type,
            "model_path": args.model_path,
            "model_key": model_key,
            "embedder_path": args.embedder_path,
            "seq_len": int(args.seq_len),
            "rank": int(args.rank),
            "eps": float(args.eps),
            "num_samples": int(args.num_samples),
            "max_calibration_texts": int(args.max_calibration_texts),
            "secret_key": args.secret_key,
            "scoring_profile_scope": "dataset",
            "scoring_profile_dataset": dataset_type,
            "scoring_profiles": {
                name: dict(config) for name, config in active_scoring_profiles.items()
            },
            "whitening_fingerprint": whitening_fp_before,
        }
        stats["build_config"] = build_config

    whitening_fp_after = whitening_fingerprint(stats)
    if build_mode == "recalibrate" and whitening_fp_after != whitening_fp_before:
        raise RuntimeError(
            "BUG: whitening changed during recalibration-only mode. "
            f"before={whitening_fp_before}, after={whitening_fp_after}"
        )

    os.makedirs(os.path.dirname(stats_file) or ".", exist_ok=True)
    torch.save(stats, stats_file)

    report = {
        "method": "ias_w",
        "method_full_name": "Whitened Independent Anchor Semantic Watermark",
        "build_mode": build_mode,
        "dataset_type": dataset_type,
        "model_key": model_key,
        "seq_len": int(args.seq_len),
        "rank": int(stats.get("rank", stats["components"].shape[1])),
        "eps": float(stats.get("eps", args.eps)),
        "stats_key": stats_key,
        "stats_file": stats_file,
        "original_dim": int(stats.get("original_dim", stats["components"].shape[0])),
        "whiten_dim": int(stats["components"].shape[1]),
        "num_stats_sentences": int(stats.get("num_stats_sentences", 0)),
        "num_calibration_texts": int(len(cal_texts)),
        "baseline_sha256": baseline_hash,
        "whitening_fingerprint": whitening_fp_after,
        "calibration": stats["calibration"],
        "calibration_profiles": calibration_profiles,
        "scoring_profiles": stats["scoring_profiles"],
        "scoring_profile_scope": stats["scoring_profile_scope"],
        "scoring_profile_dataset": stats["scoring_profile_dataset"],
        "default_scoring_profile": "dual",
        "build_config": stats["build_config"],
    }
    save_json(report_file, report)

    print("\n" + "=" * 100)
    print("✅ IAS-W statistics/calibration completed")
    print("=" * 100)
    print(f"[+] mode                  = {build_mode}")
    print(f"[+] Stats file saved      = {stats_file}")
    print(f"[+] Report file saved     = {report_file}")
    print(f"[+] Whitening fingerprint = {whitening_fp_after}")
    if build_mode == "recalibrate":
        print("[+] Whitening preserved   = YES")
    print("[+] Profile-specific dp_local calibration:")
    for profile_name in ("dual", "standalone"):
        pack = calibration_profiles[profile_name]
        print(f"    {profile_name}: scoring={pack['scoring']}")
        print("      dp_local=" + json.dumps(pack["calibration"]["dp_local"], ensure_ascii=False))
    print("=" * 100)

    if args.update_iasw_mark:
        update_iasw_mark_registry(
            iasw_mark_file=args.iasw_mark_file,
            dataset_type=dataset_type,
            model_key=model_key,
            seq_len=int(args.seq_len),
            stats_key=stats_key,
            stats_file=stats_file,
        )

    return stats_file, report_file

def main():
    args = parse_args()

    if not args.baseline_file and not args.dataset_path:
        raise ValueError("Either --baseline_file or --dataset_path must be provided.")

    build_iasw_stats(args)


if __name__ == "__main__":
    main()