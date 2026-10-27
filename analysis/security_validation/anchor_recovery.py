#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-document anchor recovery attack for IAS-W.

Purpose
-------
Test whether the fixed sentence-indexed IAS-W anchors can be estimated
from many watermarked documents generated under the same secret key.

Attack:
    v_hat_i(N) = normalize(mean_d u_{d,i})

where u_{d,i} is the whitened + normalized embedding of sentence i
from watermarked document d.

Controls:
    1. watermarked text
    2. no-watermark text
    3. human text

Outputs:
    anchor_recovery_per_position.csv
    anchor_recovery_summary.csv
    estimated_anchors_N*.pt
    config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# Sentence splitting
# ============================================================

_SENT_SPLIT = re.compile(
    r"(?<=[.!?。！？])\s+|(?<=[.!?。！？])"
)


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()

    if not text:
        return []

    sents = _SENT_SPLIT.split(text)

    return [
        s.strip()
        for s in sents
        if s and s.strip()
    ]


# ============================================================
# IO
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                print(
                    f"[WARN] invalid JSON at line {line_idx}"
                )
                continue

            if isinstance(obj, dict):
                obj["_line_idx"] = line_idx
                rows.append(obj)

    return rows


def first_existing_text(
    row: Dict[str, Any],
    candidates: List[str],
) -> str:

    for key in candidates:
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_text(
    row: Dict[str, Any],
    source: str,
) -> str:

    if source == "wm":
        return first_existing_text(
            row,
            [
                "watermarked_text",
                "wm_text",
                "generated_text",
                "text_wm",
            ],
        )

    if source == "no_wm":
        return first_existing_text(
            row,
            [
                "no_wm_text",
                "unwatermarked_text",
                "clean_model_text",
            ],
        )

    if source == "human":
        return first_existing_text(
            row,
            [
                "human_text",
                "reference_text",
            ],
        )

    raise ValueError(
        f"Unknown source: {source}"
    )


# ============================================================
# Whitening stats
# ============================================================

def recursive_find_tensor(
    obj: Any,
    candidate_keys: List[str],
) -> Optional[torch.Tensor]:

    if isinstance(obj, dict):

        for key in candidate_keys:
            if key in obj:
                value = obj[key]

                if torch.is_tensor(value):
                    return value

                if isinstance(
                    value,
                    (list, tuple, np.ndarray),
                ):
                    try:
                        return torch.as_tensor(value)
                    except Exception:
                        pass

        for value in obj.values():
            found = recursive_find_tensor(
                value,
                candidate_keys,
            )

            if found is not None:
                return found

    return None


def load_whitening(
    stats_file: str,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:

    print(
        "[*] Loading whitening stats:",
        stats_file,
    )

    obj = torch.load(
        stats_file,
        map_location="cpu",
    )

    mu = recursive_find_tensor(
        obj,
        [
            "mu",
            "mean",
            "embedding_mean",
            "whitening_mean",
        ],
    )

    W = recursive_find_tensor(
        obj,
        [
            "W",
            "whitening_matrix",
            "projection",
            "proj",
            "components",
        ],
    )

    if mu is None:
        raise RuntimeError(
            "Cannot find whitening mean in stats file."
        )

    if W is None:
        raise RuntimeError(
            "Cannot find whitening/projection matrix "
            "in stats file."
        )

    mu = mu.float().squeeze().to(device)
    W = W.float().to(device)

    print(
        f"[*] mu shape = {tuple(mu.shape)}"
    )
    print(
        f"[*] W shape  = {tuple(W.shape)}"
    )

    return mu, W


def apply_whitening(
    embeddings: torch.Tensor,
    mu: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """
    Try the two common project conventions:

        z = (e - mu) @ W
    or
        z = (e - mu) @ W.T

    The correct orientation is inferred from dimensions.
    """

    x = embeddings.float() - mu.unsqueeze(0)

    if (
        W.ndim != 2
        or x.ndim != 2
    ):
        raise RuntimeError(
            f"Unexpected shapes: "
            f"x={tuple(x.shape)}, "
            f"W={tuple(W.shape)}"
        )

    if x.shape[1] == W.shape[0]:
        z = x @ W

    elif x.shape[1] == W.shape[1]:
        z = x @ W.T

    else:
        raise RuntimeError(
            "Whitening dimension mismatch: "
            f"embedding dim={x.shape[1]}, "
            f"W={tuple(W.shape)}"
        )

    z = F.normalize(
        z,
        p=2,
        dim=1,
    )

    return z


# ============================================================
# True IAS-W anchors
# ============================================================

@torch.no_grad()
def get_true_anchor_matrix(
    secret_key: str,
    num_targets: int,
    dim: int,
    device: str,
) -> torch.Tensor:
    """
    Same anchor construction style already used in the project:

        payload = f"{secret_key}_{i}"
        SHA256 -> first 8 bytes -> seed
        randn -> normalize
    """

    targets = []

    for i in range(num_targets):

        payload = f"{secret_key}_{i}"

        seed = int.from_bytes(
            hashlib.sha256(
                payload.encode("utf-8")
            ).digest()[:8],
            byteorder="big",
        )

        try:
            rng = torch.Generator(
                device=device
            )
        except Exception:
            rng = torch.Generator()

        rng.manual_seed(seed)

        v = torch.randn(
            1,
            dim,
            generator=rng,
            device=device,
        )

        v = F.normalize(
            v,
            p=2,
            dim=1,
        )

        targets.append(
            v.squeeze(0)
        )

    return torch.stack(
        targets,
        dim=0,
    )


# ============================================================
# Encoding
# ============================================================

@torch.no_grad()
def encode_whitened_sentences(
    embedder: SentenceTransformer,
    sentences: List[str],
    mu: torch.Tensor,
    W: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:

    if not sentences:
        return torch.empty(
            (0, W.shape[-1]),
            device=device,
        )

    emb = embedder.encode(
        sentences,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        device=device,
    )

    emb = emb.float().to(device)

    return apply_whitening(
        emb,
        mu,
        W,
    )


# ============================================================
# Build document representations
# ============================================================

@torch.no_grad()
def encode_documents(
    rows: List[Dict[str, Any]],
    source: str,
    embedder: SentenceTransformer,
    mu: torch.Tensor,
    W: torch.Tensor,
    device: str,
    batch_size: int,
    max_positions: int,
) -> List[torch.Tensor]:

    documents = []

    for row in tqdm(
        rows,
        desc=f"Encoding {source}",
    ):

        text = get_text(
            row,
            source,
        )

        if not text:
            documents.append(
                torch.empty(
                    (0, W.shape[-1]),
                    device="cpu",
                )
            )
            continue

        sents = split_sentences(text)

        if max_positions > 0:
            sents = sents[:max_positions]

        if not sents:
            documents.append(
                torch.empty(
                    (0, W.shape[-1]),
                    device="cpu",
                )
            )
            continue

        z = encode_whitened_sentences(
            embedder,
            sents,
            mu,
            W,
            device,
            batch_size,
        )

        documents.append(
            z.detach().cpu()
        )

    return documents


# ============================================================
# Anchor estimation
# ============================================================

def estimate_anchors(
    docs: List[torch.Tensor],
    n_docs: int,
    max_positions: int,
) -> Tuple[torch.Tensor, np.ndarray]:

    selected = docs[:n_docs]

    dim = None

    for d in selected:
        if d.ndim == 2 and d.shape[0] > 0:
            dim = int(d.shape[1])
            break

    if dim is None:
        raise RuntimeError(
            "No valid document embeddings."
        )

    estimates = []
    counts = []

    for pos in range(max_positions):

        vectors = []

        for doc in selected:

            if (
                doc.ndim == 2
                and doc.shape[0] > pos
            ):
                vectors.append(
                    doc[pos]
                )

        counts.append(
            len(vectors)
        )

        if not vectors:
            estimates.append(
                torch.zeros(dim)
            )
            continue

        mat = torch.stack(
            vectors,
            dim=0,
        )

        mean_vec = mat.mean(
            dim=0
        )

        norm = torch.linalg.vector_norm(
            mean_vec
        )

        if float(norm) <= 1e-12:
            estimates.append(
                torch.zeros_like(mean_vec)
            )
        else:
            estimates.append(
                mean_vec / norm
            )

    return (
        torch.stack(estimates),
        np.asarray(counts),
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_estimate(
    estimated: torch.Tensor,
    true_anchors: torch.Tensor,
    counts: np.ndarray,
    source: str,
    N: int,
) -> List[Dict[str, Any]]:

    rows = []

    M = min(
        estimated.shape[0],
        true_anchors.shape[0],
    )

    for pos in range(M):

        e = estimated[pos]
        v = true_anchors[pos].cpu()

        if (
            counts[pos] <= 0
            or float(torch.linalg.vector_norm(e))
            <= 1e-12
        ):
            cosine = np.nan
        else:
            cosine = float(
                torch.dot(e, v)
            )

        rows.append({
            "source": source,
            "N_requested": N,
            "position": pos,
            "position_1based": pos + 1,
            "n_documents_at_position": int(
                counts[pos]
            ),
            "cosine_to_true_anchor": cosine,
        })

    return rows


# ============================================================
# CLI
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--input_file",
        required=True,
    )

    p.add_argument(
        "--stats_file",
        required=True,
    )

    p.add_argument(
        "--embedder_path",
        required=True,
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    p.add_argument(
        "--secret_key",
        default="DUAL_MACRO_2026",
    )

    p.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--max_positions",
        type=int,
        default=12,
    )

    p.add_argument(
        "--n_values",
        nargs="+",
        type=int,
        default=[
            10,
            25,
            50,
            100,
            200,
            400,
        ],
    )

    p.add_argument(
        "--sources",
        nargs="+",
        default=[
            "wm",
            "no_wm",
            "human",
        ],
        choices=[
            "wm",
            "no_wm",
            "human",
        ],
    )

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_jsonl(
        args.input_file
    )

    print(
        f"[*] Loaded {len(rows)} records"
    )

    if not rows:
        raise RuntimeError(
            "No records loaded."
        )

    print(
        "[*] Loading embedder:",
        args.embedder_path,
    )

    embedder = SentenceTransformer(
        args.embedder_path,
        device=args.device,
    )

    mu, W = load_whitening(
        args.stats_file,
        args.device,
    )

    # Determine whitened dimension.
    if W.shape[0] == mu.numel():
        whitened_dim = int(
            W.shape[1]
        )
    elif W.shape[1] == mu.numel():
        whitened_dim = int(
            W.shape[0]
        )
    else:
        raise RuntimeError(
            "Cannot determine whitened dimension."
        )

    print(
        f"[*] whitened dim = {whitened_dim}"
    )

    true_anchors = get_true_anchor_matrix(
        secret_key=args.secret_key,
        num_targets=args.max_positions,
        dim=whitened_dim,
        device=args.device,
    )

    true_anchors_cpu = (
        true_anchors
        .detach()
        .cpu()
    )

    torch.save(
        true_anchors_cpu,
        out_dir / "true_anchors.pt",
    )

    all_results = []

    encoded_by_source = {}

    for source in args.sources:

        docs = encode_documents(
            rows=rows,
            source=source,
            embedder=embedder,
            mu=mu,
            W=W,
            device=args.device,
            batch_size=args.batch_size,
            max_positions=args.max_positions,
        )

        encoded_by_source[source] = docs

        valid_docs = sum(
            1
            for d in docs
            if d.shape[0] > 0
        )

        print(
            f"[*] {source}: "
            f"{valid_docs}/{len(docs)} "
            f"valid documents"
        )

        for N in args.n_values:

            N_eff = min(
                N,
                len(docs),
            )

            estimated, counts = (
                estimate_anchors(
                    docs,
                    n_docs=N_eff,
                    max_positions=args.max_positions,
                )
            )

            torch.save(
                {
                    "source": source,
                    "N_requested": N,
                    "N_effective": N_eff,
                    "anchors": estimated,
                    "counts": torch.as_tensor(
                        counts
                    ),
                },
                out_dir
                / (
                    f"estimated_anchors_"
                    f"{source}_N{N}.pt"
                ),
            )

            result_rows = evaluate_estimate(
                estimated=estimated,
                true_anchors=true_anchors_cpu,
                counts=counts,
                source=source,
                N=N,
            )

            all_results.extend(
                result_rows
            )

    df = pd.DataFrame(
        all_results
    )

    per_pos_path = (
        out_dir
        / "anchor_recovery_per_position.csv"
    )

    df.to_csv(
        per_pos_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        df.groupby(
            ["source", "N_requested"],
            as_index=False,
        )
        .agg(
            mean_cosine=(
                "cosine_to_true_anchor",
                "mean",
            ),
            median_cosine=(
                "cosine_to_true_anchor",
                "median",
            ),
            max_cosine=(
                "cosine_to_true_anchor",
                "max",
            ),
            min_cosine=(
                "cosine_to_true_anchor",
                "min",
            ),
            mean_documents_per_position=(
                "n_documents_at_position",
                "mean",
            ),
        )
    )

    summary_path = (
        out_dir
        / "anchor_recovery_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        **vars(args),
        "num_input_rows": len(rows),
        "whitened_dim": whitened_dim,
    }

    with open(
        out_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("ANCHOR RECOVERY SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("[Saved]")
    print(" ", per_pos_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()