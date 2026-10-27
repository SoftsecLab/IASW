# src/calibration/calibrate_empirical_ias_wa.py
# -*- coding: utf-8 -*-
"""
Build IAS-WA whitening statistics, empirical anchor bank, and calibration.

IAS-WA = Whitened empirical Anchor-bank IAS.

Compared with IAS-W:
    IAS-W  target_i = random vector in whitened space
    IAS-WA target_i = key-selected vector from an empirical anchor bank

Recommended command:
    python src/calibration/calibrate_empirical_ias_wa.py \
        --baseline_file data/calibration_set/c4/llama3.1-8b-instruct/ias_baseline_len200_skip3000_n1000.jsonl \
        --stats_file cache/iaswa_stats/c4/llama3.1-8b-instruct/iaswa_stats_c4_llama3.1-8b-instruct_len200_rank64_eps0p0001_bank20000_calibskip3000_calibn1000.pt \
        --dataset_type c4 \
        --model_path ./models/llama3.1-8b-instruct \
        --embedder_path ./models/bge-small-en-v1.5 \
        --seq_len 200 \
        --rank 64 \
        --eps 0.0001 \
        --num_samples 1000 \
        --max_stats_sentences 20000 \
        --anchor_bank_size 20000
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


BASE_ALIGNMENT_MODES = ["strict", "window", "dp_global", "dp_local"]


# ============================================================
# Utilities
# ============================================================


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?。！？\n])", text)
        if s and s.strip()
    ]


def infer_model_key(model_path: str) -> str:
    p = (model_path or "").lower()
    if "llama3.2-3b" in p or "llama-3.2-3b" in p:
        return "llama3.2-3b"
    if "llama3.1-8b" in p or "llama-3.1-8b" in p or "llama3" in p:
        return "llama3.1-8b"
    if "qwen2.5" in p or "qwen" in p:
        return "qwen2.5-7b-instruct"
    if "mistral" in p:
        return "mistral-7b-instruct"
    if "opt-1.3b" in p or "semstamp" in p:
        return "opt-1.3b-semstamp"
    return os.path.basename(model_path.rstrip("/")) or "unknown_model"


def compute_all_alignment_scores(
    sim_matrix: np.ndarray,
    window_size: int = 2,
    gap_penalty: float = -0.05,
) -> Dict[str, float]:
    if sim_matrix is None:
        return {m: 0.0 for m in BASE_ALIGNMENT_MODES}
    N, M = sim_matrix.shape
    if N == 0 or M == 0:
        return {m: 0.0 for m in BASE_ALIGNMENT_MODES}

    shifted = sim_matrix.astype(np.float32)

    strict_score = float(np.mean([shifted[i, i] for i in range(min(N, M))]))

    window_vals = []
    for i in range(N):
        left = max(0, i - window_size)
        right = min(M, i + window_size + 1)
        window_vals.append(float(np.max(shifted[i, left:right])))
    window_score = float(np.mean(window_vals)) if window_vals else 0.0

    dp_g = np.zeros((N + 1, M + 1), dtype=np.float32)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            match = dp_g[i - 1, j - 1] + shifted[i - 1, j - 1]
            delete = dp_g[i - 1, j] + 0.0
            insert = dp_g[i, j - 1] + gap_penalty
            dp_g[i, j] = max(match, delete, insert)
    dp_global_score = float(np.max(dp_g[N, :]) / max(N, 1))

    dp_l = np.zeros((N + 1, M + 1), dtype=np.float32)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            match = dp_l[i - 1, j - 1] + shifted[i - 1, j - 1]
            delete = dp_l[i - 1, j] + gap_penalty
            insert = dp_l[i, j - 1] + gap_penalty
            dp_l[i, j] = max(0.0, match, delete, insert)

    max_score = float(np.max(dp_l))
    path_length = float(np.sum(dp_l > 0) / max(N, 1))
    dp_local_score = float(max_score / max(path_length, 1.0))

    return {
        "strict": strict_score,
        "window": window_score,
        "dp_global": dp_global_score,
        "dp_local": dp_local_score,
    }


def collect_sentences_from_baseline(rows: List[Dict[str, Any]], fields: List[str], max_sentences: int) -> List[str]:
    out = []
    for row in rows:
        for field in fields:
            for s in split_sentences(row.get(field, "")):
                if s:
                    out.append(s)
                    if max_sentences > 0 and len(out) >= max_sentences:
                        return out
    return out


def collect_texts_from_baseline(rows: List[Dict[str, Any]], fields: List[str]) -> List[str]:
    out = []
    for row in rows:
        for field in fields:
            text = row.get(field, "")
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return out


@torch.no_grad()
def encode_sentences(
    model: SentenceTransformer,
    sentences: List[str],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    embs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
        batch = sentences[i : i + batch_size]
        e = model.encode(
            batch,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
        )
        e = F.normalize(e, p=2, dim=1)
        embs.append(e.detach().cpu())
    if not embs:
        return torch.empty((0, int(model.get_sentence_embedding_dimension())), dtype=torch.float32)
    return torch.cat(embs, dim=0).float()


def build_whitening(raw_embeddings: torch.Tensor, rank: int, eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_embeddings.ndim != 2 or raw_embeddings.shape[0] < 2:
        raise ValueError("Need at least two sentence embeddings for whitening.")
    n, d = raw_embeddings.shape
    r = min(int(rank), int(d), int(n - 1))
    if r <= 0:
        raise ValueError(f"Invalid whitening rank: rank={rank}, n={n}, d={d}")

    mean = raw_embeddings.mean(dim=0)
    x = raw_embeddings - mean
    cov = torch.matmul(x.T, x) / max(n - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order][:r].clamp_min(1e-12).contiguous()
    components = eigvecs[:, order][:, :r].contiguous()
    return mean.contiguous(), components, eigvals


def whiten_embeddings(raw_embeddings: torch.Tensor, mean: torch.Tensor, components: torch.Tensor, eigvals: torch.Tensor, eps: float) -> torch.Tensor:
    z = torch.matmul(raw_embeddings - mean, components)
    z = z / torch.sqrt(eigvals + eps)
    z = F.normalize(z, p=2, dim=1)
    return z.contiguous()


def get_anchor_index(secret_key: str, sentence_index: int, bank_size: int, vector_dim: int) -> int:
    payload = (
        f"{secret_key}|IASWA_ANCHOR|"
        f"sent={sentence_index}|bank={bank_size}|dim={vector_dim}"
    )
    h = hashlib.sha256(payload.encode("utf-8")).digest()
    val = int.from_bytes(h[:8], byteorder="big", signed=False)
    return int(val % bank_size)


def target_matrix_from_bank(anchor_bank: torch.Tensor, num_targets: int, secret_key: str) -> torch.Tensor:
    bank_size = int(anchor_bank.shape[0])
    vector_dim = int(anchor_bank.shape[1])
    indices = [get_anchor_index(secret_key, j, bank_size, vector_dim) for j in range(num_targets)]
    idx = torch.tensor(indices, dtype=torch.long)
    return anchor_bank.index_select(0, idx)


def compute_iaswa_raw_scores_for_text(
    text: str,
    model: SentenceTransformer,
    mean: torch.Tensor,
    components: torch.Tensor,
    eigvals: torch.Tensor,
    eps: float,
    anchor_bank: torch.Tensor,
    secret_key: str,
    batch_size: int,
    device: str,
) -> Dict[str, float]:
    sentences = split_sentences(text)
    if not sentences:
        return {m: 0.0 for m in BASE_ALIGNMENT_MODES}

    raw = encode_sentences(model, sentences, batch_size=batch_size, device=device)
    z = whiten_embeddings(raw, mean, components, eigvals, eps)

    N = len(sentences)
    M = N + 10
    targets = target_matrix_from_bank(anchor_bank, M, secret_key)
    sim_matrix = torch.matmul(z, targets.T).numpy()
    return compute_all_alignment_scores(sim_matrix, window_size=2)


def build_calibration(raw_scores: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    bad = []
    for mode in BASE_ALIGNMENT_MODES:
        vals = np.asarray(raw_scores.get(mode, []), dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size <= 0:
            bad.append(mode)
            out[mode] = {"mu": 0.0, "sigma": 1.0, "n": 0}
            continue
        mu = float(np.mean(vals))
        sigma = float(np.std(vals, ddof=1)) if vals.size > 1 else 1.0
        if not np.isfinite(sigma) or sigma <= 1e-8:
            sigma = 1.0
        out[mode] = {"mu": mu, "sigma": sigma, "n": int(vals.size)}
    if bad:
        raise RuntimeError(f"IAS-WA calibration got no valid samples for modes: {bad}")
    return out


# ============================================================
# Args
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description="Build IAS-WA empirical anchor-bank stats and calibration.")

    p.add_argument("--baseline_file", type=str, required=True, help="Calibration baseline JSONL with human_text/no_wm_text.")
    p.add_argument("--stats_file", type=str, required=True, help="Output .pt stats file.")
    p.add_argument("--output_file", type=str, default="", help="Optional JSON report path. Default: stats_file + '.json'.")
    p.add_argument("--dataset_type", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--embedder_path", type=str, required=True)

    p.add_argument("--text_fields", type=str, default="human_text,no_wm_text")
    p.add_argument("--num_samples", type=int, default=1000, help="Number of baseline rows to use.")
    p.add_argument("--seq_len", type=int, default=200)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_stats_sentences", type=int, default=20000)
    p.add_argument("--anchor_bank_size", type=int, default=20000, help="0 or negative means use all whitened stats sentences.")
    p.add_argument("--anchor_bank_seed", type=int, default=2026)
    p.add_argument("--max_calibration_texts", type=int, default=-1)
    p.add_argument("--stats_key", type=str, default="")
    p.add_argument("--secret_key", type=str, default="DUAL_MACRO_2026")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    fields = [x.strip() for x in args.text_fields.split(",") if x.strip()]
    if not fields:
        raise ValueError("text_fields cannot be empty.")

    rows_all = load_jsonl(args.baseline_file)
    if args.num_samples > 0 and len(rows_all) < args.num_samples:
        raise RuntimeError(
            f"Baseline file has only {len(rows_all)} rows, expected at least {args.num_samples}: {args.baseline_file}"
        )
    rows = rows_all[: args.num_samples] if args.num_samples > 0 else rows_all

    print(f"[*] Loaded baseline rows: {len(rows)} from {args.baseline_file}")
    print(f"[*] Text fields: {fields}")

    stats_sentences = collect_sentences_from_baseline(rows, fields, args.max_stats_sentences)
    if len(stats_sentences) < 2:
        raise RuntimeError(f"Too few stats sentences: {len(stats_sentences)}")
    print(f"[*] Stats sentences: {len(stats_sentences)}")

    print(f"[*] Loading embedder: {args.embedder_path}")
    embedder = SentenceTransformer(args.embedder_path).to(args.device)
    original_dim = int(embedder.get_sentence_embedding_dimension())
    print(f"[*] Embedder dim: {original_dim}")

    raw_embs = encode_sentences(embedder, stats_sentences, batch_size=args.batch_size, device=args.device)
    mean, components, eigvals = build_whitening(raw_embs, rank=args.rank, eps=args.eps)
    whiten_dim = int(components.shape[1])
    print(f"[*] Whitening rank: {whiten_dim}")

    whitened_all = whiten_embeddings(raw_embs, mean, components, eigvals, args.eps)

    if args.anchor_bank_size and args.anchor_bank_size > 0 and args.anchor_bank_size < whitened_all.shape[0]:
        rng = np.random.default_rng(int(args.anchor_bank_seed))
        idx = rng.choice(whitened_all.shape[0], size=int(args.anchor_bank_size), replace=False)
        idx = np.sort(idx)
        anchor_bank = whitened_all[torch.tensor(idx, dtype=torch.long)].contiguous()
    else:
        anchor_bank = whitened_all.contiguous()

    anchor_bank = F.normalize(anchor_bank, p=2, dim=1).cpu()
    print(f"[*] Anchor bank size: {anchor_bank.shape[0]}")

    cal_texts = collect_texts_from_baseline(rows, fields)
    expected_cal_texts = len(rows) * len(fields)
    if len(cal_texts) < expected_cal_texts:
        raise RuntimeError(
            f"Collected calibration texts {len(cal_texts)} < expected {expected_cal_texts}. "
            f"Check fields={fields} in {args.baseline_file}."
        )
    if args.max_calibration_texts and args.max_calibration_texts > 0:
        cal_texts = cal_texts[: args.max_calibration_texts]
    print(f"[*] Calibration texts: {len(cal_texts)}")

    raw_scores = {m: [] for m in BASE_ALIGNMENT_MODES}
    for text in tqdm(cal_texts, desc="IAS-WA calibration raw scores"):
        raw = compute_iaswa_raw_scores_for_text(
            text=text,
            model=embedder,
            mean=mean,
            components=components,
            eigvals=eigvals,
            eps=args.eps,
            anchor_bank=anchor_bank,
            secret_key=args.secret_key,
            batch_size=args.batch_size,
            device=args.device,
        )
        for mode in BASE_ALIGNMENT_MODES:
            raw_scores[mode].append(float(raw.get(mode, 0.0)))

    calibration = build_calibration(raw_scores)
    bad_modes = [mode for mode, item in calibration.items() if int(item.get("n", 0)) != len(cal_texts)]
    if bad_modes:
        raise RuntimeError(f"Calibration n mismatch for modes: {bad_modes}")

    stats_key = args.stats_key.strip()
    if not stats_key:
        eps_tag = str(args.eps).replace(".", "p")
        stats_key = f"rank{whiten_dim}_eps{eps_tag}_bank{anchor_bank.shape[0]}_calibn{len(rows)}"

    model_key = infer_model_key(args.model_path)

    stats = {
        "method": "ias_wa",
        "method_full_name": "Whitened empirical Anchor-bank Independent Anchor Semantic Watermark",
        "dataset_type": args.dataset_type.lower(),
        "model_key": model_key,
        "seq_len": int(args.seq_len),
        "rank": int(whiten_dim),
        "eps": float(args.eps),
        "stats_key": stats_key,
        "original_dim": int(original_dim),
        "whiten_dim": int(whiten_dim),
        "mean": mean.cpu(),
        "components": components.cpu(),
        "eigvals": eigvals.cpu(),
        "anchor_bank": anchor_bank.cpu(),
        "anchor_bank_size": int(anchor_bank.shape[0]),
        "anchor_bank_seed": int(args.anchor_bank_seed),
        "num_stats_sentences": int(len(stats_sentences)),
        "num_calibration_texts": int(len(cal_texts)),
        "calibration": calibration,
        "alignment_mode": "dp_local",
        "build_config": {
            "baseline_file": args.baseline_file,
            "text_fields": fields,
            "num_samples": int(args.num_samples),
            "max_stats_sentences": int(args.max_stats_sentences),
            "anchor_bank_size_arg": int(args.anchor_bank_size),
            "secret_key": args.secret_key,
        },
    }

    os.makedirs(os.path.dirname(args.stats_file) or ".", exist_ok=True)
    torch.save(stats, args.stats_file)
    print(f"[OK] Saved IAS-WA stats: {args.stats_file}")

    report = {
        "method": stats["method"],
        "method_full_name": stats["method_full_name"],
        "dataset_type": stats["dataset_type"],
        "model_key": stats["model_key"],
        "seq_len": stats["seq_len"],
        "rank": stats["rank"],
        "eps": stats["eps"],
        "stats_key": stats["stats_key"],
        "stats_file": args.stats_file,
        "original_dim": stats["original_dim"],
        "whiten_dim": stats["whiten_dim"],
        "anchor_bank_size": stats["anchor_bank_size"],
        "anchor_bank_seed": stats["anchor_bank_seed"],
        "num_stats_sentences": stats["num_stats_sentences"],
        "num_calibration_texts": stats["num_calibration_texts"],
        "calibration": calibration,
        "build_config": stats["build_config"],
    }
    out_json = args.output_file.strip() or (args.stats_file + ".json")
    save_json(out_json, report)
    print(f"[OK] Saved IAS-WA report: {out_json}")

    print("\nCalibration summary:")
    for mode, item in calibration.items():
        print(f"  {mode:9s} mu={item['mu']:.6f} sigma={item['sigma']:.6f} n={item['n']}")


if __name__ == "__main__":
    main()
