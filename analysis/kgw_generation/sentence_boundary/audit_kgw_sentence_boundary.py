#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KGW sentence-boundary / retained-vs-discarded-tail audit.

Goal
----
Diagnose why Qwen2.5-7B loses KGW green-token enrichment under sentence-wise
generation while Qwen2.5-3B and Llama-3.2-3B do not.

This script DOES NOT modify the project orchestrator. It directly runs KGW with
the project's current WatermarkLogitsProcessor and records token-level evidence.

Two modes
---------
chunk_ids:
    Repeated max_sent_tokens generation calls; keep every generated token ID.
    This is the C1 control from the previous generation-regime audit.

sentence_ids:
    Generate <= max_sent_tokens, decode the raw continuation, keep only the
    first sentence, but preserve the ORIGINAL generated token IDs corresponding
    to that sentence. No decode->re-encode reconstruction.
    This is the C3 control and isolates first-sentence cutoff / continuation.

For every sentence_ids loop, the script records:
    raw continuation = retained first sentence + discarded tail

For every generated token, it records:
    - actual KGW green/red
    - raw model entropy BEFORE KGW
    - raw model top-1 probability
    - raw green-list probability mass
    - effective sampling entropy BEFORE KGW after temperature/top-k/top-p
    - effective green mass BEFORE KGW
    - effective green mass AFTER KGW
    - KGW green-mass gain
    - whether base/watermarked top-1 token is green
    - sentence-start and sentence-end relative positions

Important
---------
The "raw" entropy is taken from the logits received by the KGW logits processor,
i.e. before KGW bias. The "effective" metrics additionally apply the requested
temperature/top-k/top-p, matching the sampling restriction approximately.

Run this file from the project or use the companion shell script.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

# ---------------------------------------------------------------------------
# Robust project import
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_env_project = os.environ.get("PROJECT_DIR", "").strip()
PROJECT_DIR = Path(_env_project).resolve() if _env_project else _THIS.parent.parent

if not (PROJECT_DIR / "src").is_dir():
    for p in _THIS.parents:
        if (p / "src").is_dir():
            PROJECT_DIR = p
            break

if not (PROJECT_DIR / "src").is_dir():
    raise RuntimeError(
        f"Cannot locate project root containing src/. Script={_THIS}. "
        "Set PROJECT_DIR=<PROJECT_ROOT>."
    )

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.baselines.kgw_core.watermark_processor import WatermarkLogitsProcessor

try:
    import nltk
    nltk_dir = os.environ.get("NLTK_DATA", "/home/share/models/nltk_data")
    if os.path.isdir(nltk_dir) and nltk_dir not in nltk.data.path:
        nltk.data.path.append(nltk_dir)
except Exception:
    nltk = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_file", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_name", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=40)
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")

    p.add_argument("--max_total_tokens", type=int, default=160)
    p.add_argument("--max_sent_tokens", type=int, default=40)
    p.add_argument("--max_loops", type=int, default=64)

    # Match the controlled Random ablation by default.
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0)

    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20260823)

    p.add_argument("--boundary_left", type=int, default=5)
    p.add_argument("--boundary_right", type=int, default=10)
    p.add_argument("--modes", nargs="+", default=["chunk_ids", "sentence_ids"])

    # Save human-readable raw/retained/tail strings in loop JSONL.
    p.add_argument("--save_text", action="store_true")
    return p.parse_args()


def dtype_from_name(x: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(x, "auto")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: str, n: int):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if n > 0 and len(rows) >= n:
                break
    return rows


def prompt_from_row(row: Dict[str, Any], field: str) -> str:
    if field in row and str(row[field]).strip():
        return str(row[field]).strip()
    if str(row.get("prompt", "")).strip():
        return str(row["prompt"]).strip()
    raise KeyError(
        f"Prompt field '{field}' is missing/empty. Available keys={sorted(row.keys())}"
    )


def first_sentence(text: str) -> str:
    x = (text or "").strip()
    if not x:
        return ""
    if nltk is not None:
        try:
            sents = nltk.sent_tokenize(x)
            if sents:
                return sents[0].strip()
        except Exception:
            pass

    # Fallback close to the project's behavior.
    m = re.search(r".*?[.!?。！？](?:\s|$)", x, flags=re.S)
    if m:
        return m.group(0).strip()
    return x


def find_first_sentence_token_cut(
    tokenizer,
    raw_new_ids: torch.Tensor,
    sent: str,
) -> Tuple[int, bool]:
    """
    Return the shortest prefix length of the ACTUAL generated token IDs whose
    decoded text reaches the retained first sentence.

    This avoids using tokenizer.encode(sent) as the retained sequence.
    """
    ids = raw_new_ids.view(-1)
    target = (sent or "").strip()
    if not target:
        return 0, True

    for k in range(1, ids.numel() + 1):
        dec = tokenizer.decode(ids[:k], skip_special_tokens=True).strip()
        if dec == target or dec.startswith(target):
            return k, True

    # Rare fallback. It is explicitly logged so it cannot silently contaminate
    # the analysis.
    enc = tokenizer.encode(target, add_special_tokens=False)
    return min(len(enc), int(ids.numel())), False


def entropy_from_probs(p: torch.Tensor) -> float:
    nz = p > 0
    if not bool(nz.any()):
        return float("nan")
    return float((-(p[nz] * torch.log(p[nz]))).sum().item())


def apply_sampling_restrictions(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """
    Approximate HF sampling warpers on a single 1-D logit vector.

    KGW's logits processor acts first; temperature/top-k/top-p are applied
    afterwards by generate(). This function is used only for audit metrics,
    not for actual generation.
    """
    x = logits.float().clone()

    if temperature and temperature > 0 and temperature != 1.0:
        x = x / float(temperature)

    if top_k is not None and int(top_k) > 0 and int(top_k) < x.numel():
        k = int(top_k)
        kth = torch.topk(x, k).values[-1]
        x[x < kth] = -float("inf")

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum > float(top_p)
        # Keep the first token above the threshold, as HF does.
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        x[sorted_idx[remove]] = -float("inf")

    return x


def probs_and_stats(logits: torch.Tensor, green_ids: torch.Tensor) -> Dict[str, Any]:
    probs = torch.softmax(logits.float(), dim=-1)
    top_prob, top_id = torch.max(probs, dim=-1)
    green_mass = probs[green_ids].sum()

    # Fast membership for only one token.
    top_is_green = bool((green_ids == top_id).any().item())

    return {
        "entropy": entropy_from_probs(probs),
        "top1_prob": float(top_prob.item()),
        "top1_id": int(top_id.item()),
        "top1_is_green": int(top_is_green),
        "green_mass": float(green_mass.item()),
    }


# ---------------------------------------------------------------------------
# Audited KGW logits processor
# ---------------------------------------------------------------------------
class AuditedKGWProcessor(LogitsProcessor):
    """
    Wrap the project's WatermarkLogitsProcessor.

    Batch size is intentionally 1 in this audit. That makes token-step alignment
    exact and avoids mixing candidate-level behavior into this experiment.
    """

    def __init__(
        self,
        vocab_ids: List[int],
        gamma: float,
        delta: float,
        device: str,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        self.raw = WatermarkLogitsProcessor(
            vocab=vocab_ids,
            gamma=gamma,
            delta=delta,
            seeding_scheme="simple_1",
        )
        # Ensure randperm generator lives on the same device as input IDs.
        try:
            self.raw.rng = torch.Generator(device=device)
        except Exception:
            self.raw.rng = torch.Generator()

        self.delta = float(delta)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)

        self.records: List[Dict[str, Any]] = []
        self._prev_green_ids: Optional[torch.Tensor] = None

    def _finalize_previous_actual_token(self, input_ids: torch.Tensor):
        if self._prev_green_ids is None or not self.records:
            return
        if self.records[-1].get("actual_is_green") is not None:
            return

        # The final input token is the token sampled at the previous generation
        # step.
        sampled = int(input_ids[0, -1].item())
        self.records[-1]["actual_token_id"] = sampled
        self.records[-1]["actual_is_green"] = int(
            bool((self._prev_green_ids == sampled).any().item())
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        assert input_ids.shape[0] == 1, "Audit requires batch size 1."

        # At step t, input_ids[-1] contains the token sampled at step t-1.
        self._finalize_previous_actual_token(input_ids)

        raw_scores = scores[0].float()
        green_ids = self.raw._get_greenlist_ids(input_ids[0])

        raw_stats = probs_and_stats(raw_scores, green_ids)

        pre_effective_logits = apply_sampling_restrictions(
            raw_scores,
            self.temperature,
            self.top_k,
            self.top_p,
        )
        pre_eff_stats = probs_and_stats(pre_effective_logits, green_ids)

        wm_scores = raw_scores.clone()
        wm_scores[green_ids] += self.delta

        wm_effective_logits = apply_sampling_restrictions(
            wm_scores,
            self.temperature,
            self.top_k,
            self.top_p,
        )
        wm_eff_stats = probs_and_stats(wm_effective_logits, green_ids)

        rec = {
            "step": len(self.records),
            "actual_token_id": None,
            "actual_is_green": None,

            "entropy_raw": raw_stats["entropy"],
            "top1_prob_raw": raw_stats["top1_prob"],
            "top1_id_raw": raw_stats["top1_id"],
            "top1_is_green_raw": raw_stats["top1_is_green"],
            "green_mass_raw": raw_stats["green_mass"],

            "entropy_effective_pre": pre_eff_stats["entropy"],
            "top1_prob_effective_pre": pre_eff_stats["top1_prob"],
            "top1_id_effective_pre": pre_eff_stats["top1_id"],
            "top1_is_green_effective_pre": pre_eff_stats["top1_is_green"],
            "green_mass_effective_pre": pre_eff_stats["green_mass"],

            "entropy_effective_wm": wm_eff_stats["entropy"],
            "top1_prob_effective_wm": wm_eff_stats["top1_prob"],
            "top1_id_effective_wm": wm_eff_stats["top1_id"],
            "top1_is_green_effective_wm": wm_eff_stats["top1_is_green"],
            "green_mass_effective_wm": wm_eff_stats["green_mass"],

            "green_mass_gain_effective": (
                wm_eff_stats["green_mass"] - pre_eff_stats["green_mass"]
            ),
        }
        self.records.append(rec)

        self._prev_green_ids = green_ids

        # Apply exactly the project's KGW bias.
        out = scores.clone()
        out[0, green_ids] += self.delta
        return out

    def finalize_last(self, generated_new_ids: torch.Tensor):
        if not self.records or self._prev_green_ids is None:
            return
        if self.records[-1].get("actual_is_green") is not None:
            return
        ids = generated_new_ids.view(-1)
        if ids.numel() == 0:
            return

        sampled = int(ids[-1].item())
        self.records[-1]["actual_token_id"] = sampled
        self.records[-1]["actual_is_green"] = int(
            bool((self._prev_green_ids == sampled).any().item())
        )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_raw_loop(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    n_new: int,
    args,
    vocab_ids: List[int],
):
    audit_proc = AuditedKGWProcessor(
        vocab_ids=vocab_ids,
        gamma=args.gamma,
        delta=args.delta,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=int(n_new),
        do_sample=True,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        logits_processor=LogitsProcessorList([audit_proc]),
        pad_token_id=tokenizer.pad_token_id,
    )
    if int(args.top_k) >= 0:
        kwargs["top_k"] = int(args.top_k)

    out = model.generate(**kwargs)
    seq = out.sequences if hasattr(out, "sequences") else out
    raw_new = seq[0, input_ids.shape[1]:].detach()

    audit_proc.finalize_last(raw_new)

    # In unusual early-stop cases, align conservatively.
    n = min(len(audit_proc.records), int(raw_new.numel()))
    records = audit_proc.records[:n]
    raw_new = raw_new[:n]

    for j, rec in enumerate(records):
        token_id = int(raw_new[j].item())
        rec["actual_token_id"] = token_id
        rec["token_text"] = tokenizer.decode([token_id], skip_special_tokens=False)

    return raw_new, records


def segment_stats(records: List[Dict[str, Any]], lo: int, hi: int) -> Dict[str, Any]:
    sub = records[lo:hi]
    out: Dict[str, Any] = {"n": len(sub)}
    if not sub:
        for k in [
            "green_fraction",
            "entropy_raw",
            "entropy_effective_pre",
            "top1_prob_effective_pre",
            "green_mass_raw",
            "green_mass_effective_pre",
            "green_mass_effective_wm",
            "green_mass_gain_effective",
            "top1_green_rate_pre",
            "top1_green_rate_wm",
        ]:
            out[k] = float("nan")
        return out

    def mean_key(key):
        vals = [
            float(r[key]) for r in sub
            if r.get(key) is not None and math.isfinite(float(r[key]))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    out.update({
        "green_fraction": mean_key("actual_is_green"),
        "entropy_raw": mean_key("entropy_raw"),
        "entropy_effective_pre": mean_key("entropy_effective_pre"),
        "top1_prob_effective_pre": mean_key("top1_prob_effective_pre"),
        "green_mass_raw": mean_key("green_mass_raw"),
        "green_mass_effective_pre": mean_key("green_mass_effective_pre"),
        "green_mass_effective_wm": mean_key("green_mass_effective_wm"),
        "green_mass_gain_effective": mean_key("green_mass_gain_effective"),
        "top1_green_rate_pre": mean_key("top1_is_green_effective_pre"),
        "top1_green_rate_wm": mean_key("top1_is_green_effective_wm"),
    })
    return out


def run_mode(
    mode: str,
    model,
    tokenizer,
    prompt: str,
    sample_index: int,
    args,
    vocab_ids: List[int],
):
    set_seed(args.seed + sample_index)

    current_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"].to(args.device)

    prompt_len = int(current_ids.shape[1])
    loops = []
    tokens_out = []
    actual_sentence_number = 0

    for loop_idx in range(1, args.max_loops + 1):
        generated_so_far = int(current_ids.shape[1]) - prompt_len
        remaining = args.max_total_tokens - generated_so_far
        if remaining <= 0:
            break

        n_new = min(args.max_sent_tokens, remaining)
        raw_new, recs = generate_raw_loop(
            model, tokenizer, current_ids, n_new, args, vocab_ids
        )
        if raw_new.numel() == 0:
            break

        raw_text = tokenizer.decode(raw_new, skip_special_tokens=True)

        if mode == "chunk_ids":
            cut = int(raw_new.numel())
            first_sent = raw_text.strip()
            cutoff_found = True
        elif mode == "sentence_ids":
            first_sent = first_sentence(raw_text)
            cut, cutoff_found = find_first_sentence_token_cut(
                tokenizer, raw_new, first_sent
            )
            cut = max(1, min(cut, int(raw_new.numel())))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        retained = raw_new[:cut]
        discarded = raw_new[cut:]

        retained_stats = segment_stats(recs, 0, cut)
        tail_stats = segment_stats(recs, cut, len(recs))

        loop_row = {
            "sample_index": sample_index,
            "mode": mode,
            "loop_index": loop_idx,
            "raw_n": int(raw_new.numel()),
            "retained_n": int(retained.numel()),
            "tail_n": int(discarded.numel()),
            "cutoff_found_from_actual_ids": int(cutoff_found),

            "retained_green_fraction": retained_stats["green_fraction"],
            "tail_green_fraction": tail_stats["green_fraction"],
            "retained_entropy_raw": retained_stats["entropy_raw"],
            "tail_entropy_raw": tail_stats["entropy_raw"],
            "retained_entropy_effective_pre": retained_stats["entropy_effective_pre"],
            "tail_entropy_effective_pre": tail_stats["entropy_effective_pre"],
            "retained_top1_prob_effective_pre": retained_stats["top1_prob_effective_pre"],
            "tail_top1_prob_effective_pre": tail_stats["top1_prob_effective_pre"],
            "retained_green_mass_raw": retained_stats["green_mass_raw"],
            "tail_green_mass_raw": tail_stats["green_mass_raw"],
            "retained_green_mass_effective_pre": retained_stats["green_mass_effective_pre"],
            "tail_green_mass_effective_pre": tail_stats["green_mass_effective_pre"],
            "retained_green_mass_effective_wm": retained_stats["green_mass_effective_wm"],
            "tail_green_mass_effective_wm": tail_stats["green_mass_effective_wm"],
            "retained_green_mass_gain_effective": retained_stats["green_mass_gain_effective"],
            "tail_green_mass_gain_effective": tail_stats["green_mass_gain_effective"],
            "retained_top1_green_rate_pre": retained_stats["top1_green_rate_pre"],
            "tail_top1_green_rate_pre": tail_stats["top1_green_rate_pre"],
            "retained_top1_green_rate_wm": retained_stats["top1_green_rate_wm"],
            "tail_top1_green_rate_wm": tail_stats["top1_green_rate_wm"],
        }

        if args.save_text:
            loop_row.update({
                "raw_text": raw_text,
                "retained_text": tokenizer.decode(retained, skip_special_tokens=True),
                "discarded_tail_text": tokenizer.decode(discarded, skip_special_tokens=True),
                "first_sentence_nltk": first_sent,
            })

        loops.append(loop_row)

        if mode == "sentence_ids":
            actual_sentence_number += 1

        # Token rows, including retained and discarded tail.
        for j, rec in enumerate(recs):
            is_retained = j < cut
            token_row = {
                "sample_index": sample_index,
                "mode": mode,
                "loop_index": loop_idx,
                "raw_pos": j + 1,
                "segment": "retained" if is_retained else "tail",
                "retained_sentence_index": (
                    actual_sentence_number if mode == "sentence_ids" and is_retained else None
                ),
                **rec,
            }

            if mode == "sentence_ids" and is_retained:
                # Distance to sentence start/end for actual retained text.
                token_row["sent_pos_from_start"] = j + 1
                token_row["sent_pos_to_end"] = cut - j

                # Separate relative-position columns to avoid double-count ambiguity
                # when a short sentence is inside both windows.
                token_row["boundary_rel_start"] = (
                    j + 1 if (j + 1) <= args.boundary_right else None
                )
                to_end = cut - j
                token_row["boundary_rel_end"] = (
                    -to_end if to_end <= args.boundary_left else None
                )
            else:
                token_row["sent_pos_from_start"] = None
                token_row["sent_pos_to_end"] = None
                token_row["boundary_rel_start"] = None
                token_row["boundary_rel_end"] = None

            tokens_out.append(token_row)

        # Advance using original token IDs only.
        current_ids = torch.cat(
            [current_ids, retained.view(1, -1).to(current_ids.device)],
            dim=1,
        )

        eos = tokenizer.eos_token_id
        if eos is not None and int(eos) in [int(x) for x in retained.tolist()]:
            break

    return loops, tokens_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or Path(args.model_path.rstrip("/")).name
    rows = load_rows(args.baseline_file, args.num_samples)
    if not rows:
        raise RuntimeError("No baseline rows loaded.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype_from_name(args.dtype),
    }

    print(f"[*] Loading: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **load_kwargs,
    ).to(args.device)
    model.eval()

    vocab_ids = list(tokenizer.get_vocab().values())

    config = {
        "model": model_name,
        "model_path": args.model_path,
        "baseline_file": args.baseline_file,
        "num_samples": len(rows),
        "args": vars(args),
        "tokenizer_len": len(tokenizer),
        "vocab_len": len(vocab_ids),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    loop_path = out_dir / "loop_records.jsonl"
    token_path = out_dir / "token_records.jsonl"

    with loop_path.open("w", encoding="utf-8") as lf, \
         token_path.open("w", encoding="utf-8") as tf:

        for i, row in enumerate(tqdm(rows, desc=f"{model_name} boundary audit")):
            prompt = prompt_from_row(row, args.prompt_field)

            for mode in args.modes:
                loops, toks = run_mode(
                    mode=mode,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    sample_index=i,
                    args=args,
                    vocab_ids=vocab_ids,
                )

                for r in loops:
                    r["model"] = model_name
                    r["sample_id"] = row.get("idx", row.get("id", i))
                    lf.write(json.dumps(r, ensure_ascii=False) + "\n")

                for r in toks:
                    r["model"] = model_name
                    r["sample_id"] = row.get("idx", row.get("id", i))
                    tf.write(json.dumps(r, ensure_ascii=False) + "\n")

                lf.flush()
                tf.flush()

    print("\n[DONE]")
    print("  ", loop_path)
    print("  ", token_path)
    print("  ", out_dir / "config.json")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
