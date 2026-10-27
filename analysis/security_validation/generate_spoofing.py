#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KGW-free multi-document IAS-W spoofing experiment.

Modes
-----
random:
    Generate K ordinary candidates and select one randomly.

estimated:
    Generate K ordinary candidates and select the candidate most aligned
    with an anchor estimated from previously observed watermarked texts.

true:
    Oracle upper bound. Generate K ordinary candidates and select using
    the real IAS-W anchor.

IMPORTANT
---------
Candidate generation NEVER uses the KGW logits processor.

Therefore any KGW evidence in generated spoofing texts is incidental,
not deliberately embedded.

The official IASWMark is reused for:
    - sentence embedding
    - whitening
    - true anchor generation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.watermarks.ias_w_mark import IASWMark


# ============================================================
# IO
# ============================================================

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def save_jsonl(path, rows):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = str(path) + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    os.replace(
        tmp,
        path,
    )


# ============================================================
# Sentence extraction
# ============================================================

def first_sentence(text: str) -> str:
    """
    Keep the generation-side sentence extraction simple and close to
    the current dual orchestrator.

    Prefer NLTK because the formal orchestrator also uses
    nltk.sent_tokenize().
    """

    text = (text or "").strip()

    if not text:
        return ""

    try:
        import nltk

        try:
            nltk.data.find(
                "tokenizers/punkt"
            )
        except LookupError:
            nltk.download(
                "punkt",
                quiet=True,
            )

        sents = nltk.sent_tokenize(
            text
        )

        if sents:
            return sents[0].strip()

    except Exception:
        pass

    m = re.search(
        r".*?[.!?。！？](?:\s|$)",
        text,
        flags=re.S,
    )

    if m:
        return m.group(0).strip()

    return text.strip()


# ============================================================
# Estimated anchors
# ============================================================

def load_estimated_anchors(
    path: str,
    device: str,
):

    obj = torch.load(
        path,
        map_location="cpu",
    )

    if isinstance(obj, dict):
        anchors = obj.get(
            "anchors"
        )

        counts = obj.get(
            "counts",
            None,
        )
    else:
        anchors = obj
        counts = None

    if anchors is None:
        raise RuntimeError(
            f"No anchors found in {path}"
        )

    anchors = (
        torch.as_tensor(anchors)
        .float()
        .to(device)
    )

    anchors = F.normalize(
        anchors,
        p=2,
        dim=1,
    )

    print(
        f"[*] Loaded estimated anchors: "
        f"{tuple(anchors.shape)}"
    )

    if counts is not None:
        print(
            "[*] Anchor support counts:",
            torch.as_tensor(counts).tolist(),
        )

    return anchors


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_candidates(
    model,
    tokenizer,
    current_text: str,
    num_candidates: int,
    max_sent_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
):

    inputs = tokenizer(
        current_text,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    input_len = (
        inputs["input_ids"].shape[1]
    )

    kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get(
            "attention_mask",
            None,
        ),
        "max_new_tokens": max_sent_tokens,
        "num_return_sequences": num_candidates,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if top_k >= 0:
        kwargs["top_k"] = top_k

    if 0.0 < top_p <= 1.0:
        kwargs["top_p"] = top_p

    # --------------------------------------------------------
    # Deliberately NO logits_processor here.
    #
    # This is the core of the security experiment:
    # candidates contain no intentionally embedded KGW.
    # --------------------------------------------------------

    outputs = model.generate(
        **kwargs
    )

    candidates = []

    for i in range(
        outputs.shape[0]
    ):

        new_ids = outputs[
            i,
            input_len:
        ]

        text = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        )

        sent = first_sentence(
            text
        )

        if sent:
            candidates.append(
                sent
            )

    return candidates


# ============================================================
# Selection
# ============================================================

@torch.no_grad()
def select_candidate(
    candidates: List[str],
    sentence_index: int,
    mode: str,
    iasw: IASWMark,
    estimated_anchors,
    rng: random.Random,
):

    if not candidates:
        return "", {
            "selected_idx": -1,
            "selected_similarity": None,
        }

    # --------------------------------------------------------
    # Random control
    # --------------------------------------------------------

    if mode == "random":

        idx = rng.randrange(
            len(candidates)
        )

        return candidates[idx], {
            "selected_idx": idx,
            "selected_similarity": None,
        }

    # --------------------------------------------------------
    # Target vector
    # --------------------------------------------------------

    if mode == "true":

        target = (
            iasw
            .get_target_semantic_vector(
                sentence_index
            )
        )

    elif mode == "estimated":

        if (
            estimated_anchors is None
            or sentence_index
            >= estimated_anchors.shape[0]
        ):
            # No recovered anchor available.
            idx = rng.randrange(
                len(candidates)
            )

            return candidates[idx], {
                "selected_idx": idx,
                "selected_similarity": None,
                "fallback": "random_no_anchor",
            }

        target = (
            estimated_anchors[
                sentence_index
            ]
            .unsqueeze(0)
        )

    else:
        raise ValueError(
            mode
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # use official IASWMark whitening.
    # --------------------------------------------------------

    embeddings = (
        iasw.encode_whitened(
            candidates
        )
    )

    similarities = torch.mm(
        embeddings,
        target.T,
    ).squeeze(1)

    idx = int(
        torch.argmax(
            similarities
        ).item()
    )

    return candidates[idx], {
        "selected_idx": idx,
        "selected_similarity": float(
            similarities[idx].item()
        ),
        "mean_candidate_similarity": float(
            similarities.mean().item()
        ),
        "max_candidate_similarity": float(
            similarities.max().item()
        ),
    }


# ============================================================
# One sample
# ============================================================

@torch.no_grad()
def spoof_generate_one(
    prompt,
    model,
    tokenizer,
    iasw,
    mode,
    estimated_anchors,
    num_candidates,
    max_total_tokens,
    max_sent_tokens,
    temperature,
    top_k,
    top_p,
    seed,
):

    rng = random.Random(
        seed
    )

    current_text = prompt
    generated_sentences = []

    sentence_index = 0

    trajectory = []

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    while True:

        generated_text = " ".join(
            generated_sentences
        ).strip()

        generated_ids = tokenizer.encode(
            generated_text,
            add_special_tokens=False,
        )

        if (
            len(generated_ids)
            >= max_total_tokens
        ):
            break

        candidates = generate_candidates(
            model=model,
            tokenizer=tokenizer,
            current_text=current_text,
            num_candidates=num_candidates,
            max_sent_tokens=max_sent_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if not candidates:
            break

        selected, info = select_candidate(
            candidates=candidates,
            sentence_index=sentence_index,
            mode=mode,
            iasw=iasw,
            estimated_anchors=estimated_anchors,
            rng=rng,
        )

        if not selected:
            break

        generated_sentences.append(
            selected
        )

        trajectory.append({
            "sentence_index": sentence_index,
            "num_candidates": len(candidates),
            **info,
        })

        current_text = (
            prompt
            + " "
            + " ".join(
                generated_sentences
            )
        ).strip()

        sentence_index += 1

        # Safety guard against pathological generations.
        if sentence_index >= 64:
            break

    final_text = " ".join(
        generated_sentences
    ).strip()

    # Match requested token budget more closely.
    ids = tokenizer.encode(
        final_text,
        add_special_tokens=False,
    )

    if len(ids) > max_total_tokens:
        ids = ids[:max_total_tokens]

        final_text = tokenizer.decode(
            ids,
            skip_special_tokens=True,
        )

    return (
        final_text,
        trajectory,
    )


# ============================================================
# Args
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--baseline_file",
        required=True,
    )

    p.add_argument(
        "--output_file",
        required=True,
    )

    p.add_argument(
        "--model_path",
        required=True,
    )

    p.add_argument(
        "--embedder_path",
        required=True,
    )

    p.add_argument(
        "--dataset_type",
        default="c4",
    )

    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "random",
            "estimated",
            "true",
        ],
    )

    p.add_argument(
        "--estimated_anchor_file",
        default="",
    )

    p.add_argument(
        "--stats_file",
        default="",
    )

    p.add_argument(
        "--stats_key",
        default="",
    )

    p.add_argument(
        "--secret_key",
        default="DUAL_MACRO_2026",
    )

    p.add_argument(
        "--target_len",
        type=int,
        default=200,
    )

    p.add_argument(
        "--num_samples",
        type=int,
        default=300,
    )

    p.add_argument(
        "--start_index",
        type=int,
        default=0,
    )

    p.add_argument(
        "--num_candidates",
        type=int,
        default=32,
    )

    p.add_argument(
        "--max_total_tokens",
        type=int,
        default=200,
    )

    p.add_argument(
        "--max_sent_tokens",
        type=int,
        default=40,
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--top_k",
        type=int,
        default=-1,
    )

    p.add_argument(
        "--top_p",
        type=float,
        default=-1.0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260917,
    )

    p.add_argument(
        "--device",
        default="cuda:0",
    )

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    print("=" * 80)
    print("IAS-W SPOOFING SECURITY EXPERIMENT")
    print("=" * 80)

    print(
        f"mode            = {args.mode}"
    )
    print(
        f"num_candidates  = {args.num_candidates}"
    )
    print(
        f"num_samples     = {args.num_samples}"
    )
    print(
        f"start_index     = {args.start_index}"
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed % (2**32 - 1)
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "[*] Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    print(
        "[*] Loading model..."
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.model_path,
            torch_dtype=(
                torch.float16
                if "cuda" in args.device
                else torch.float32
            ),
            trust_remote_code=True,
        )
        .to(args.device)
    )

    model.eval()

    # --------------------------------------------------------
    # Official IAS-W
    # --------------------------------------------------------

    print(
        "[*] Loading official IASWMark..."
    )

    iasw = IASWMark(
        embedder_path_or_model=(
            args.embedder_path
        ),
        tokenizer=tokenizer,
        model_name_path=args.model_path,
        dataset_type=args.dataset_type,
        secret_key=args.secret_key,
        device=args.device,
        stats_file=(
            args.stats_file
            if args.stats_file
            else None
        ),
        stats_key=(
            args.stats_key
            if args.stats_key
            else None
        ),
        target_len=args.target_len,
        scoring_profile="dual",
    )

    # --------------------------------------------------------
    # Estimated anchors
    # --------------------------------------------------------

    estimated_anchors = None

    if args.mode == "estimated":

        if not args.estimated_anchor_file:
            raise ValueError(
                "--estimated_anchor_file "
                "required for estimated mode."
            )

        estimated_anchors = (
            load_estimated_anchors(
                args.estimated_anchor_file,
                args.device,
            )
        )

        if (
            estimated_anchors.shape[1]
            != iasw.vector_dim
        ):
            raise RuntimeError(
                "Estimated anchor dimension "
                f"{estimated_anchors.shape[1]} "
                "does not match IAS-W "
                f"{iasw.vector_dim}."
            )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    all_rows = load_jsonl(
        args.baseline_file
    )

    start = args.start_index

    end = min(
        start + args.num_samples,
        len(all_rows),
    )

    rows = all_rows[
        start:end
    ]

    print(
        f"[*] Evaluation rows: "
        f"{start}:{end}"
    )

    output_rows = []

    for local_idx, row in enumerate(
        tqdm(
            rows,
            desc=f"Spoof-{args.mode}",
        )
    ):

        global_idx = (
            start + local_idx
        )

        prompt = row.get(
            "prompt",
            "",
        )

        out = dict(
            row
        )

        if not prompt:
            out["wm_text"] = ""
            out["error"] = "empty_prompt"

            output_rows.append(
                out
            )
            continue

        sample_seed = (
            args.seed
            + global_idx
        )

        # Reseed model sampling as well.
        random.seed(
            sample_seed
        )

        np.random.seed(
            sample_seed
            % (2**32 - 1)
        )

        torch.manual_seed(
            sample_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                sample_seed
            )

        t0 = time.time()

        try:

            spoof_text, trajectory = (
                spoof_generate_one(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    iasw=iasw,
                    mode=args.mode,
                    estimated_anchors=(
                        estimated_anchors
                    ),
                    num_candidates=(
                        args.num_candidates
                    ),
                    max_total_tokens=(
                        args.max_total_tokens
                    ),
                    max_sent_tokens=(
                        args.max_sent_tokens
                    ),
                    temperature=(
                        args.temperature
                    ),
                    top_k=args.top_k,
                    top_p=args.top_p,
                    seed=sample_seed,
                )
            )

            out["wm_text"] = (
                spoof_text
            )

            out["generation_time"] = (
                time.time() - t0
            )

            out["security_meta"] = {
                "experiment": (
                    "iasw_anchor_spoofing"
                ),
                "mode": args.mode,
                "kgw_generation": False,
                "num_candidates": (
                    args.num_candidates
                ),
                "sample_index": (
                    global_idx
                ),
                "seed": sample_seed,
                "trajectory": trajectory,
            }

        except Exception as e:

            out["wm_text"] = ""
            out["error"] = str(e)

            print(
                f"\n[ERROR] sample "
                f"{global_idx}: {e}"
            )

        output_rows.append(
            out
        )

        # Incremental save.
        if (
            len(output_rows) % 10 == 0
        ):
            save_jsonl(
                args.output_file,
                output_rows,
            )

    save_jsonl(
        args.output_file,
        output_rows,
    )

    print()
    print(
        "[+] Saved:",
        args.output_file,
    )


if __name__ == "__main__":
    main()