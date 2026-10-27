# -*- coding: utf-8 -*-
"""
Dual-mode attack quality analysis.

功能：
1. existing_quality:
   复用已有 JSONL 中的 quality 字段，用于快速趋势分析。

2. unified_evaluator:
   统一使用一个 evaluator LM 重新计算 PPL，
   统一使用一个 embedder 重新计算语义相似度，
   用于正式论文分析。

核心修正：
1. 增加 valid_n / missing_metric_n / missing_metric_rate；
2. 对 NaN 样本不再默认视为 non-destructive；
3. destructive_rate、semantic_bad_rate 等 rate 只在有效样本分母上计算；
4. 将“语义/长度破坏”和“自动质量指标风险”拆分，避免 typo/swap 被 PPL 单独误判为人类视角破坏性攻击；
5. 默认不保存样本级大 CSV，避免输出文件过大。

输出：
- attack_quality_summary_existing.csv
- attack_quality_summary_unified.csv
- attack_quality_summary_all_modes.csv
- attack_quality_summary_all_modes_by_model_method.csv
- attack_quality_bad_cases_all_modes.jsonl
- attack_quality_report.md

可选输出：
- attack_quality_samples_existing.csv
- attack_quality_samples_unified.csv
- attack_quality_samples_all_modes.csv
"""

import os
import re
import json
import math
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Basic utils
# ============================================================

def calc_nan():
    return float("nan")


def is_valid_number(x):
    try:
        x = float(x)
        return not math.isnan(x) and not math.isinf(x)
    except Exception:
        return False


def safe_float(x):
    try:
        if x is None:
            return calc_nan()
        return float(x)
    except Exception:
        return calc_nan()


def safe_delta(a, b):
    a = safe_float(a)
    b = safe_float(b)
    if not is_valid_number(a) or not is_valid_number(b):
        return calc_nan()
    return float(a - b)


def safe_ratio(a, b):
    a = safe_float(a)
    b = safe_float(b)
    if not is_valid_number(a) or not is_valid_number(b) or b == 0:
        return calc_nan()
    return float(a / b)


def short_text(text, max_len=300):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ..."


def valid_numeric_series(s):
    """
    返回一个布尔 Series：
    True  表示该位置是有效数值；
    False 表示 NaN / inf / -inf / 非数值。
    """
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.notna()


def bool_with_na(condition, valid_mask):
    """
    将 condition 转成可空布尔值：
    - valid_mask=True  时保留 True/False；
    - valid_mask=False 时置为 pd.NA。
    """
    out = pd.Series(pd.NA, index=condition.index, dtype="boolean")
    out.loc[valid_mask] = condition.loc[valid_mask].astype(bool)
    return out


# ============================================================
# Text quality metrics
# ============================================================

def calculate_ppl(text, model, tokenizer, device, max_length=1024):
    """
    计算单条文本的 PPL。

    注意：
    - 使用 torch.inference_mode()，避免构建计算图，降低显存占用；
    - 只作为攻击后文本质量的自动指标，不等价于人类可读性。
    """
    if model is None or tokenizer is None:
        return calc_nan()

    if not isinstance(text, str) or not text.strip():
        return calc_nan()

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)

        if inputs["input_ids"].shape[1] < 2:
            return calc_nan()

        with torch.inference_mode():
            outputs = model(
                input_ids=inputs["input_ids"],
                labels=inputs["input_ids"]
            )

        loss = outputs.loss.detach().float().cpu().item()

        if not is_valid_number(loss):
            return calc_nan()

        try:
            return float(math.exp(loss))
        except OverflowError:
            return float("inf")
    except RuntimeError as e:
        # 不中断整批分析，尤其是遇到极端长文本或显存波动时。
        if "out of memory" in str(e).lower() and device.startswith("cuda"):
            torch.cuda.empty_cache()
        return calc_nan()


def calc_rep(text, n):
    if not isinstance(text, str) or not text.strip():
        return calc_nan()

    words = text.lower().split()
    if len(words) < n:
        return calc_nan()

    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return calc_nan()

    return float(1.0 - len(set(ngrams)) / len(ngrams))


def calc_entropy(text, n=3):
    if not isinstance(text, str) or not text.strip():
        return calc_nan()

    words = text.lower().split()
    if len(words) < n:
        return calc_nan()

    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return calc_nan()

    counts = Counter(ngrams)

    ent = 0.0
    for c in counts.values():
        p = c / len(ngrams)
        ent -= p * math.log(p)

    return float(ent)


def calc_distinct(text, n):
    if not isinstance(text, str) or not text.strip():
        return calc_nan()

    words = text.lower().split()
    if len(words) < n:
        return calc_nan()

    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return calc_nan()

    return float(len(set(ngrams)) / len(ngrams))


def compute_text_quality(text, model, tokenizer, device, max_length=1024):
    words = text.split() if isinstance(text, str) else []

    return {
        "len": len(words),
        "char_len": len(text) if isinstance(text, str) else 0,
        "ppl": calculate_ppl(
            text=text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length
        ),
        "rep2": calc_rep(text, 2),
        "rep3": calc_rep(text, 3),
        "rep4": calc_rep(text, 4),
        "rep15": calc_rep(text, 15),
        "ent3": calc_entropy(text, 3),
        "distinct1": calc_distinct(text, 1),
        "distinct2": calc_distinct(text, 2),
    }


def calc_sim(a, b, embedder):
    if embedder is None:
        return calc_nan()

    if not isinstance(a, str) or not isinstance(b, str):
        return calc_nan()

    if not a.strip() or not b.strip():
        return calc_nan()

    v1 = embedder.encode(
        a,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    v2 = embedder.encode(
        b,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return float(cosine_similarity([v1], [v2])[0][0])


# ============================================================
# Load evaluator models
# ============================================================

def resolve_torch_dtype(dtype_name, device):
    """
    根据参数和设备选择 evaluator LM 的精度。
    - 4090/48G 上 bf16/fp16 都可以；
    - A4500/20G 建议 fp16，且不建议跑 8B evaluator；
    - auto 会优先使用 bf16，如果当前 CUDA 设备不支持 bf16，则回退 fp16。
    """
    dtype_name = str(dtype_name).lower()

    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16

    if device.startswith("cuda"):
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    return torch.float32


def load_evaluator_lm(model_path, device, dtype_name="auto"):
    print(f"\n[*] 加载统一 PPL evaluator:")
    print(f"    {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = resolve_torch_dtype(dtype_name, device)
    print(f"    dtype = {torch_dtype}")

    if device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    return model, tokenizer


def load_embedder(embedder_path, device):
    print(f"\n[*] 加载统一 semantic embedder:")
    print(f"    {embedder_path}")
    return SentenceTransformer(embedder_path, device=device)


# ============================================================
# JSONL file discovery and parsing
# ============================================================

def find_jsonl_files(root_dir):
    jsonl_files = []

    for root, _, files in os.walk(root_dir):
        if os.path.basename(root) != "records":
            continue

        for fname in files:
            if not fname.endswith(".jsonl"):
                continue

            lower = fname.lower()

            if lower.startswith(("baseline", "detect", "temp")):
                continue

            jsonl_files.append(os.path.join(root, fname))

    return sorted(jsonl_files)


def infer_model_method_from_path(path):
    """
    兼容常见目录结构：
    results/main_exp/{model_name}/{watermark_method}/records/xxx.jsonl

    注意：
    本脚本的主分析不按 model/method 聚合，
    这里只保留 metadata，方便后续排查。
    """
    parts = path.split(os.sep)

    if "records" not in parts:
        return "", ""

    ridx = len(parts) - 1 - parts[::-1].index("records")

    model_name = ""
    wm_method = ""

    if ridx - 2 >= 0:
        model_name = parts[ridx - 2]
    if ridx - 1 >= 0:
        wm_method = parts[ridx - 1]

    return model_name, wm_method


def get_attack_text(atk):
    if not isinstance(atk, dict):
        return ""

    candidate_keys = [
        "attacked_text",
        "attack_text",
        "paraphrased_text",
        "output_text",
        "text",
    ]

    for key in candidate_keys:
        val = atk.get(key, "")
        if isinstance(val, str) and val.strip():
            return val

    return ""


def get_attack_name(atk_key, atk):
    if isinstance(atk, dict):
        for key in ["attack_name", "attack", "method", "attack_method"]:
            val = atk.get(key, None)
            if val:
                return str(val)

    return str(atk_key)


# ============================================================
# Existing quality extraction
# ============================================================

def get_existing_role_quality(row, role):
    """
    role:
    - human
    - clean
    - wm
    """
    q = row.get("quality", {})

    if not isinstance(q, dict):
        return None

    val = q.get(role, None)

    if isinstance(val, dict):
        return val

    return None


def get_existing_attack_quality(atk):
    """
    原始 evaluate_all_quality.py 中 attack quality 结构大致为：

    atk["quality"] = {
        "metrics": {...},
        "semantic": {...},
        "delta_vs_wm": {...}
    }
    """
    if not isinstance(atk, dict):
        return None, None, None

    q = atk.get("quality", None)

    if not isinstance(q, dict):
        return None, None, None

    metrics = q.get("metrics", None)
    semantic = q.get("semantic", None)
    delta_vs_wm = q.get("delta_vs_wm", None)

    if not isinstance(metrics, dict):
        metrics = None
    if not isinstance(semantic, dict):
        semantic = None
    if not isinstance(delta_vs_wm, dict):
        delta_vs_wm = None

    return metrics, semantic, delta_vs_wm


def normalize_existing_quality(q):
    """
    将已有 quality 字段统一成当前脚本使用的字段。
    若已有字段缺失，则置为 NaN。
    """
    if not isinstance(q, dict):
        q = {}

    return {
        "len": safe_float(q.get("len")),
        "char_len": safe_float(q.get("char_len")),
        "ppl": safe_float(q.get("ppl")),
        "rep2": safe_float(q.get("rep2")),
        "rep3": safe_float(q.get("rep3")),
        "rep4": safe_float(q.get("rep4")),
        "rep15": safe_float(q.get("rep15")),
        "ent3": safe_float(q.get("ent3")),
        "distinct1": safe_float(q.get("distinct1")),
        "distinct2": safe_float(q.get("distinct2")),
    }


def normalize_existing_semantic(s):
    if not isinstance(s, dict):
        s = {}

    return {
        "sim_attacked_vs_wm": safe_float(s.get("sim_attacked_vs_wm")),
        "sim_attacked_vs_human": safe_float(s.get("sim_attacked_vs_human")),
        "sim_attacked_vs_clean": safe_float(s.get("sim_attacked_vs_clean")),
    }


# ============================================================
# Unified or existing calculation
# ============================================================

def get_quality_by_mode(
    text,
    existing_q,
    eval_mode,
    model,
    tokenizer,
    device,
    max_length
):
    if eval_mode == "existing_quality":
        return normalize_existing_quality(existing_q)

    if eval_mode == "unified_evaluator":
        return compute_text_quality(
            text=text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length
        )

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def get_attack_semantic_by_mode(
    attacked,
    wm,
    human,
    clean,
    existing_semantic,
    eval_mode,
    embedder
):
    if eval_mode == "existing_quality":
        return normalize_existing_semantic(existing_semantic)

    if eval_mode == "unified_evaluator":
        return {
            "sim_attacked_vs_wm": calc_sim(attacked, wm, embedder),
            "sim_attacked_vs_human": calc_sim(attacked, human, embedder),
            "sim_attacked_vs_clean": calc_sim(attacked, clean, embedder),
        }

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


# ============================================================
# Destructive attack judgment
# ============================================================

def weighted_available_average(components, weights):
    """
    对多个 0-1 score 做加权平均，自动忽略缺失项。
    这样可以避免某个自动指标缺失时把 acceptable_score 整体变成 NaN。
    """
    cols = list(weights.keys())
    w = pd.Series(weights, dtype="float64")

    comp = components[cols].apply(pd.to_numeric, errors="coerce")
    comp = comp.replace([np.inf, -np.inf], np.nan)

    valid = comp.notna().astype("float64")
    denom = valid.mul(w, axis=1).sum(axis=1)
    numer = comp.fillna(0.0).mul(w, axis=1).sum(axis=1)

    out = numer / denom.replace(0, np.nan)
    return out.clip(0, 1)


def add_destructive_flags(df, args):
    """
    增强版攻击质量判定。

    关键变化：
    1. 不再把 destructive_rate 直接解释为“人类视角破坏性攻击率”；
    2. 新增 semantic_destructive：语义/长度层面的破坏，更接近人类可接受性；
    3. 新增 auto_quality_risk：PPL/重复/熵等自动指标报警，更接近模型指标风险；
    4. destructive 保留为 strict diagnostic flag，即两类风险的并集，用于排查，不建议作为主文核心指标；
    5. acceptable_score 调整为更偏语义和长度，降低 PPL 对 typo/swap 的主导影响。
    """
    df = df.copy()

    # ------------------------------------------------------------
    # 1. Valid masks
    # ------------------------------------------------------------
    df["semantic_valid"] = valid_numeric_series(df["sim_attacked_vs_wm"])
    df["length_valid"] = valid_numeric_series(df["len_ratio"])

    # fluency 有两个判断来源：ppl_ratio 和 attacked_ppl
    df["ppl_ratio_valid"] = valid_numeric_series(df["ppl_ratio"])
    df["attacked_ppl_valid"] = valid_numeric_series(df["attacked_ppl"])
    df["fluency_valid"] = df["ppl_ratio_valid"] | df["attacked_ppl_valid"]

    df["rep3_valid"] = valid_numeric_series(df["attacked_rep3"])
    df["rep4_valid"] = valid_numeric_series(df["attacked_rep4"])
    df["rep15_valid"] = valid_numeric_series(df["attacked_rep15"])
    df["repeat_valid"] = df["rep3_valid"] | df["rep4_valid"] | df["rep15_valid"]

    df["entropy_valid"] = valid_numeric_series(df["ent3_delta"])

    # 传统 core_valid 保留：表示语义、长度、流畅性三类主指标均有效。
    df["core_valid"] = (
        df["semantic_valid"] &
        df["length_valid"] &
        df["fluency_valid"]
    )

    valid_cols = [
        "semantic_valid",
        "length_valid",
        "fluency_valid",
        "repeat_valid",
        "entropy_valid",
    ]
    df["valid_metric_count"] = df[valid_cols].sum(axis=1)
    df["missing_metric_count"] = len(valid_cols) - df["valid_metric_count"]
    df["missing_any_core_metric"] = ~df["core_valid"]

    # ------------------------------------------------------------
    # 2. Bad flags with pd.NA
    # ------------------------------------------------------------
    semantic_cond = df["sim_attacked_vs_wm"] < args.semantic_threshold
    df["semantic_bad"] = bool_with_na(semantic_cond, df["semantic_valid"])

    length_cond = (
        (df["len_ratio"] < args.min_len_ratio) |
        (df["len_ratio"] > args.max_len_ratio)
    )
    df["length_bad"] = bool_with_na(length_cond, df["length_valid"])

    fluency_cond = (
        ((df["ppl_ratio"] > args.ppl_ratio_threshold) & df["ppl_ratio_valid"]) |
        ((df["attacked_ppl"] > args.ppl_abs_threshold) & df["attacked_ppl_valid"])
    )
    df["fluency_bad"] = bool_with_na(fluency_cond, df["fluency_valid"])

    repeat_cond = (
        ((df["attacked_rep3"] > args.rep3_threshold) & df["rep3_valid"]) |
        ((df["attacked_rep4"] > args.rep4_threshold) & df["rep4_valid"]) |
        ((df["attacked_rep15"] > args.rep15_threshold) & df["rep15_valid"])
    )
    df["repeat_bad"] = bool_with_na(repeat_cond, df["repeat_valid"])

    entropy_cond = df["ent3_delta"] < -args.ent3_drop_threshold
    df["entropy_bad"] = bool_with_na(entropy_cond, df["entropy_valid"])

    # ------------------------------------------------------------
    # 3. Split the original destructive judgment into two dimensions
    # ------------------------------------------------------------
    semantic_bad_known = (df["semantic_bad"] == True) | (df["length_bad"] == True)
    semantic_valid_all = df["semantic_valid"] & df["length_valid"]

    df["valid_for_semantic_destructive"] = semantic_valid_all | semantic_bad_known
    df["semantic_destructive"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[semantic_bad_known, "semantic_destructive"] = True
    df.loc[
        (~semantic_bad_known) & df["valid_for_semantic_destructive"],
        "semantic_destructive"
    ] = False

    auto_risk_known = (
        (df["fluency_bad"] == True) |
        (df["repeat_bad"] == True) |
        (df["entropy_bad"] == True)
    )
    auto_valid_any = df["fluency_valid"] | df["repeat_valid"] | df["entropy_valid"]

    df["valid_for_auto_quality_risk"] = auto_valid_any | auto_risk_known
    df["auto_quality_risk"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[auto_risk_known, "auto_quality_risk"] = True
    df.loc[
        (~auto_risk_known) & df["valid_for_auto_quality_risk"],
        "auto_quality_risk"
    ] = False

    # strict destructive 保留原来的“任一坏项即报警”逻辑，但仅作为诊断项。
    strict_bad_known = semantic_bad_known | auto_risk_known
    df["valid_for_destructive"] = (
        df["valid_for_semantic_destructive"] |
        df["valid_for_auto_quality_risk"] |
        strict_bad_known
    )

    df["destructive"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[strict_bad_known, "destructive"] = True
    df.loc[
        (~strict_bad_known) & df["valid_for_destructive"],
        "destructive"
    ] = False

    # 为避免语义误解，额外复制一个更明确的别名。
    df["strict_destructive"] = df["destructive"]

    # ------------------------------------------------------------
    # 4. acceptable_score
    # ------------------------------------------------------------
    sim_score = (
        (df["sim_attacked_vs_wm"] - args.semantic_bad_floor) /
        (args.semantic_good_ceiling - args.semantic_bad_floor)
    )
    df["semantic_score"] = sim_score.clip(0, 1)

    ppl_ratio = df["ppl_ratio"].replace(0, np.nan)
    ppl_log = np.log(ppl_ratio)

    ppl_bad_log = math.log(args.ppl_ratio_threshold)
    if ppl_bad_log <= 0:
        ppl_bad_log = math.log(2.0)

    df["fluency_score"] = (1.0 - ppl_log / ppl_bad_log).clip(0, 1)

    len_ratio = df["len_ratio"].replace(0, np.nan)
    len_log_dev = np.abs(np.log(len_ratio))

    len_bad_log = max(
        abs(math.log(args.min_len_ratio)),
        abs(math.log(args.max_len_ratio))
    )
    if len_bad_log <= 0:
        len_bad_log = abs(math.log(0.7))

    df["length_score"] = (1.0 - len_log_dev / len_bad_log).clip(0, 1)

    df["repeat_score"] = (
        1.0 - df["attacked_rep3"] / args.rep3_threshold
    ).clip(0, 1)

    ent_drop = -df["ent3_delta"]
    df["entropy_score"] = (
        1.0 - ent_drop / args.ent3_drop_threshold
    ).clip(0, 1)

    # 更偏人类可接受性：语义与长度权重更高，PPL 权重更低。
    # 避免 typo/swap 因 PPL 异常被 acceptable_score 过度惩罚。
    components = pd.DataFrame({
        "semantic_score": df["semantic_score"],
        "length_score": df["length_score"],
        "repeat_score": df["repeat_score"],
        "fluency_score": df["fluency_score"],
        "entropy_score": df["entropy_score"],
    })

    score_weights = {
        "semantic_score": args.acceptable_semantic_weight,
        "length_score": args.acceptable_length_weight,
        "repeat_score": args.acceptable_repeat_weight,
        "fluency_score": args.acceptable_fluency_weight,
        "entropy_score": args.acceptable_entropy_weight,
    }

    df["acceptable_score"] = weighted_available_average(components, score_weights)

    # 如果语义和长度都无法判断，且没有任何已知坏项，则 acceptable_score 不参与均值。
    # 这样避免缺失核心质量信息的样本被误纳入平均。
    valid_for_acceptability = df["valid_for_semantic_destructive"] | strict_bad_known
    df.loc[~valid_for_acceptability, "acceptable_score"] = np.nan

    return df


# ============================================================
# Collect sample-level records
# ============================================================

def build_record(
    eval_mode,
    attack_name,
    atk_key,
    model_name,
    wm_method,
    file_path,
    line_id,
    human,
    clean,
    wm,
    attacked,
    q_human,
    q_clean,
    q_wm,
    q_atk,
    semantic,
    save_text_examples=False,
):
    rec = {
        "eval_mode": eval_mode,
        "attack_name": attack_name,
        "attack_key": atk_key,

        "model_name": model_name,
        "wm_method": wm_method,

        "file": file_path,
        "line_id": line_id,

        "human_len": safe_float(q_human.get("len")),
        "clean_len": safe_float(q_clean.get("len")),
        "wm_len": safe_float(q_wm.get("len")),
        "attacked_len": safe_float(q_atk.get("len")),

        "human_ppl": safe_float(q_human.get("ppl")),
        "clean_ppl": safe_float(q_clean.get("ppl")),
        "wm_ppl": safe_float(q_wm.get("ppl")),
        "attacked_ppl": safe_float(q_atk.get("ppl")),

        "wm_rep2": safe_float(q_wm.get("rep2")),
        "wm_rep3": safe_float(q_wm.get("rep3")),
        "wm_rep4": safe_float(q_wm.get("rep4")),
        "wm_rep15": safe_float(q_wm.get("rep15")),
        "wm_ent3": safe_float(q_wm.get("ent3")),
        "wm_distinct1": safe_float(q_wm.get("distinct1")),
        "wm_distinct2": safe_float(q_wm.get("distinct2")),

        "attacked_rep2": safe_float(q_atk.get("rep2")),
        "attacked_rep3": safe_float(q_atk.get("rep3")),
        "attacked_rep4": safe_float(q_atk.get("rep4")),
        "attacked_rep15": safe_float(q_atk.get("rep15")),
        "attacked_ent3": safe_float(q_atk.get("ent3")),
        "attacked_distinct1": safe_float(q_atk.get("distinct1")),
        "attacked_distinct2": safe_float(q_atk.get("distinct2")),

        "sim_attacked_vs_wm": safe_float(semantic.get("sim_attacked_vs_wm")),
        "sim_attacked_vs_human": safe_float(semantic.get("sim_attacked_vs_human")),
        "sim_attacked_vs_clean": safe_float(semantic.get("sim_attacked_vs_clean")),
    }

    if save_text_examples:
        rec.update({
            "wm_text_short": short_text(wm),
            "attacked_text_short": short_text(attacked),
            "human_text_short": short_text(human),
            "clean_text_short": short_text(clean),
        })

    rec["ppl_delta"] = safe_delta(rec["attacked_ppl"], rec["wm_ppl"])
    rec["ppl_ratio"] = safe_ratio(rec["attacked_ppl"], rec["wm_ppl"])

    rec["len_delta"] = safe_delta(rec["attacked_len"], rec["wm_len"])
    rec["len_ratio"] = safe_ratio(rec["attacked_len"], rec["wm_len"])

    rec["rep2_delta"] = safe_delta(rec["attacked_rep2"], rec["wm_rep2"])
    rec["rep3_delta"] = safe_delta(rec["attacked_rep3"], rec["wm_rep3"])
    rec["rep4_delta"] = safe_delta(rec["attacked_rep4"], rec["wm_rep4"])
    rec["rep15_delta"] = safe_delta(rec["attacked_rep15"], rec["wm_rep15"])

    rec["ent3_delta"] = safe_delta(rec["attacked_ent3"], rec["wm_ent3"])
    rec["distinct1_delta"] = safe_delta(rec["attacked_distinct1"], rec["wm_distinct1"])
    rec["distinct2_delta"] = safe_delta(rec["attacked_distinct2"], rec["wm_distinct2"])

    return rec


def collect_records_for_mode(
    jsonl_files,
    eval_mode,
    args,
    model=None,
    tokenizer=None,
    embedder=None,
):
    all_rows = []
    missing_existing_quality = 0
    total_attack_items = 0

    print(f"\n{'=' * 80}")
    print(f"[*] 开始收集模式: {eval_mode}")
    print(f"{'=' * 80}")

    for path in tqdm(jsonl_files, desc=f"Files [{eval_mode}]"):
        model_name, wm_method = infer_model_method_from_path(path)

        with open(path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]

        for line_id, line in enumerate(
            tqdm(lines, desc=os.path.basename(path), leave=False)
        ):
            try:
                row = json.loads(line)
            except Exception:
                continue

            human = row.get("human_text", row.get("human_reference", ""))
            clean = row.get("no_wm_text", "")
            wm = row.get("wm_text", "")

            attacks = row.get("attacks", {})
            if not isinstance(attacks, dict):
                continue

            existing_q_human = get_existing_role_quality(row, "human")
            existing_q_clean = get_existing_role_quality(row, "clean")
            existing_q_wm = get_existing_role_quality(row, "wm")

            q_human = get_quality_by_mode(
                text=human,
                existing_q=existing_q_human,
                eval_mode=eval_mode,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                max_length=args.max_length
            )

            q_clean = get_quality_by_mode(
                text=clean,
                existing_q=existing_q_clean,
                eval_mode=eval_mode,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                max_length=args.max_length
            )

            q_wm = get_quality_by_mode(
                text=wm,
                existing_q=existing_q_wm,
                eval_mode=eval_mode,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                max_length=args.max_length
            )

            for atk_key, atk in attacks.items():
                if not isinstance(atk, dict):
                    continue

                attacked = get_attack_text(atk)
                if not attacked:
                    continue

                total_attack_items += 1

                attack_name = get_attack_name(atk_key, atk)

                existing_metrics, existing_semantic, _ = get_existing_attack_quality(atk)

                if eval_mode == "existing_quality":
                    if (
                        existing_q_wm is None or
                        existing_metrics is None or
                        existing_semantic is None
                    ):
                        missing_existing_quality += 1

                q_atk = get_quality_by_mode(
                    text=attacked,
                    existing_q=existing_metrics,
                    eval_mode=eval_mode,
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    max_length=args.max_length
                )

                semantic = get_attack_semantic_by_mode(
                    attacked=attacked,
                    wm=wm,
                    human=human,
                    clean=clean,
                    existing_semantic=existing_semantic,
                    eval_mode=eval_mode,
                    embedder=embedder,
                )

                rec = build_record(
                    eval_mode=eval_mode,
                    attack_name=attack_name,
                    atk_key=atk_key,
                    model_name=model_name,
                    wm_method=wm_method,
                    file_path=path,
                    line_id=line_id,
                    human=human,
                    clean=clean,
                    wm=wm,
                    attacked=attacked,
                    q_human=q_human,
                    q_clean=q_clean,
                    q_wm=q_wm,
                    q_atk=q_atk,
                    semantic=semantic,
                    save_text_examples=args.save_text_examples,
                )

                all_rows.append(rec)

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df = add_destructive_flags(df, args)

    if eval_mode == "existing_quality":
        print(f"\n[*] existing_quality 模式统计:")
        print(f"    attack items = {total_attack_items}")
        print(f"    missing existing quality items = {missing_existing_quality}")
        if missing_existing_quality > 0:
            print("    [提醒] 有些样本没有已有 quality 字段，对应指标会是 NaN。")
            print("           新版 summary 会用 valid_n / missing_metric_rate 显示缺失比例。")
            print("           destructive_rate 不再把缺失样本默认算作非破坏性。")

    return df


# ============================================================
# Aggregation
# ============================================================

def boolean_true_count(s):
    return int((s == True).sum())


def boolean_valid_count(s):
    return int(s.notna().sum())


def safe_rate(true_n, valid_n):
    if valid_n == 0:
        return calc_nan()
    return float(true_n / valid_n)


def aggregate_summary(df, group_cols):
    """
    修正版 aggregation：
    - n: 总样本数；
    - valid_n: destructive 判断有效样本数；
    - missing_metric_n: destructive 判断无效样本数；
    - missing_metric_rate: 无效样本比例；
    - destructive_rate: destructive_true_n / valid_n；
    - 其他 bad_rate: 各自 valid_n 分母上的比例。
    """
    if df.empty:
        return pd.DataFrame()

    metrics = [
        "attacked_ppl",
        "wm_ppl",
        "ppl_delta",
        "ppl_ratio",

        "attacked_len",
        "wm_len",
        "len_delta",
        "len_ratio",

        "attacked_rep2",
        "attacked_rep3",
        "attacked_rep4",
        "attacked_rep15",

        "rep2_delta",
        "rep3_delta",
        "rep4_delta",
        "rep15_delta",

        "attacked_ent3",
        "ent3_delta",

        "attacked_distinct1",
        "attacked_distinct2",
        "distinct1_delta",
        "distinct2_delta",

        "sim_attacked_vs_wm",
        "sim_attacked_vs_human",
        "sim_attacked_vs_clean",

        "acceptable_score",

        "semantic_score",
        "length_score",
        "fluency_score",
        "repeat_score",
        "entropy_score",
    ]

    flag_valid_map = {
        "semantic_bad": "semantic_valid",
        "length_bad": "length_valid",
        "fluency_bad": "fluency_valid",
        "repeat_bad": "repeat_valid",
        "entropy_bad": "entropy_valid",

        "semantic_destructive": "valid_for_semantic_destructive",
        "auto_quality_risk": "valid_for_auto_quality_risk",
        "destructive": "valid_for_destructive",
        "strict_destructive": "valid_for_destructive",
    }

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: key for col, key in zip(group_cols, keys)}

        n = int(len(g))
        valid_n = int(g["valid_for_destructive"].sum()) if "valid_for_destructive" in g else 0
        missing_metric_n = n - valid_n

        row["n"] = n
        row["valid_n"] = valid_n
        row["missing_metric_n"] = missing_metric_n
        row["missing_metric_rate"] = safe_rate(missing_metric_n, n)

        # 额外输出各类有效样本数，便于诊断是哪类指标缺失
        for valid_col in [
            "semantic_valid",
            "length_valid",
            "fluency_valid",
            "repeat_valid",
            "entropy_valid",
            "core_valid",
            "valid_for_semantic_destructive",
            "valid_for_auto_quality_risk",
            "valid_for_destructive",
        ]:
            if valid_col in g.columns:
                row[f"{valid_col}_n"] = int(g[valid_col].sum())

        if "valid_metric_count" in g.columns:
            row["valid_metric_count_mean"] = float(g["valid_metric_count"].mean())
        if "missing_metric_count" in g.columns:
            row["missing_metric_count_mean"] = float(g["missing_metric_count"].mean())

        # 数值指标
        for m in metrics:
            if m not in g.columns:
                continue

            numeric = pd.to_numeric(g[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
            row[f"{m}_valid_n"] = int(numeric.notna().sum())
            row[f"{m}_mean"] = float(numeric.mean()) if numeric.notna().any() else calc_nan()
            row[f"{m}_median"] = float(numeric.median()) if numeric.notna().any() else calc_nan()
            row[f"{m}_std"] = float(numeric.std()) if numeric.notna().sum() > 1 else calc_nan()

        # flag rate
        for flag, valid_col in flag_valid_map.items():
            if flag not in g.columns:
                continue

            if valid_col in g.columns:
                denom = int(g[valid_col].sum())
            else:
                denom = boolean_valid_count(g[flag])

            true_n = boolean_true_count(g[flag])
            row[f"{flag}_true_n"] = true_n
            row[f"{flag}_valid_n"] = denom
            row[f"{flag}_rate"] = safe_rate(true_n, denom)

        rows.append(row)

    summary = pd.DataFrame(rows)

    sort_cols = []
    ascending = []

    if "eval_mode" in summary.columns:
        sort_cols.append("eval_mode")
        ascending.append(True)

    if "semantic_destructive_rate" in summary.columns:
        sort_cols.append("semantic_destructive_rate")
        ascending.append(True)
    elif "destructive_rate" in summary.columns:
        sort_cols.append("destructive_rate")
        ascending.append(True)

    if "acceptable_score_mean" in summary.columns:
        sort_cols.append("acceptable_score_mean")
        ascending.append(False)

    if sort_cols:
        summary = summary.sort_values(by=sort_cols, ascending=ascending)

    return summary


# ============================================================
# Output
# ============================================================

def save_outputs(df_existing, df_unified, args):
    os.makedirs(args.out_dir, exist_ok=True)

    all_dfs = []

    if df_existing is not None and not df_existing.empty:
        all_dfs.append(df_existing)

        existing_summary_path = os.path.join(
            args.out_dir,
            "attack_quality_summary_existing.csv"
        )

        aggregate_summary(
            df_existing,
            ["eval_mode", "attack_name"]
        ).to_csv(existing_summary_path, index=False)

        print(f"[*] 已保存 existing 汇总表: {existing_summary_path}")

        if args.save_samples:
            existing_samples_path = os.path.join(
                args.out_dir,
                "attack_quality_samples_existing.csv"
            )
            df_existing.to_csv(existing_samples_path, index=False)
            print(f"[*] 已保存 existing 样本表: {existing_samples_path}")
        else:
            print("[*] 跳过保存 existing 样本级大表。如需保存，请添加 --save_samples")

    if df_unified is not None and not df_unified.empty:
        all_dfs.append(df_unified)

        unified_summary_path = os.path.join(
            args.out_dir,
            "attack_quality_summary_unified.csv"
        )

        aggregate_summary(
            df_unified,
            ["eval_mode", "attack_name"]
        ).to_csv(unified_summary_path, index=False)

        print(f"[*] 已保存 unified 汇总表: {unified_summary_path}")

        if args.save_samples:
            unified_samples_path = os.path.join(
                args.out_dir,
                "attack_quality_samples_unified.csv"
            )
            df_unified.to_csv(unified_samples_path, index=False)
            print(f"[*] 已保存 unified 样本表: {unified_samples_path}")
        else:
            print("[*] 跳过保存 unified 样本级大表。如需保存，请添加 --save_samples")

    if not all_dfs:
        print("[!] 没有可保存的结果。")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    all_summary_path = os.path.join(
        args.out_dir,
        "attack_quality_summary_all_modes.csv"
    )

    summary_all = aggregate_summary(
        df_all,
        ["eval_mode", "attack_name"]
    )

    summary_all.to_csv(all_summary_path, index=False)

    print(f"[*] 已保存合并汇总表: {all_summary_path}")

    all_summary_by_model_method_path = os.path.join(
        args.out_dir,
        "attack_quality_summary_all_modes_by_model_method.csv"
    )

    summary_model_method = aggregate_summary(
        df_all,
        ["eval_mode", "attack_name", "model_name", "wm_method"]
    )

    summary_model_method.to_csv(all_summary_by_model_method_path, index=False)

    print(f"[*] 已保存辅助分组汇总表: {all_summary_by_model_method_path}")

    if args.save_samples:
        all_samples_path = os.path.join(
            args.out_dir,
            "attack_quality_samples_all_modes.csv"
        )
        df_all.to_csv(all_samples_path, index=False)
        print(f"[*] 已保存合并样本表: {all_samples_path}")
    else:
        print("[*] 跳过保存 all_modes 样本级大表。如需保存，请添加 --save_samples")

    if not args.no_bad_cases:
        bad_cases_path = os.path.join(
            args.out_dir,
            "attack_quality_bad_cases_all_modes.jsonl"
        )
        write_bad_cases(df_all, bad_cases_path, args.max_bad_cases)
        print(f"[*] 已保存坏案例: {bad_cases_path}")
    else:
        print("[*] 跳过保存 bad cases。")

    report_path = os.path.join(args.out_dir, "attack_quality_report.md")
    write_report(summary_all, report_path)

    print(f"[*] 已保存报告: {report_path}")

    print("\n" + "=" * 80)
    print("[*] 汇总预览：")
    print("=" * 80)

    preview_cols = [
        "eval_mode",
        "attack_name",
        "n",
        "valid_n",
        "missing_metric_rate",
        "semantic_destructive_rate",
        "auto_quality_risk_rate",
        "destructive_rate",
        "acceptable_score_mean",
        "sim_attacked_vs_wm_mean",
        "ppl_ratio_mean",
        "len_ratio_mean",
        "attacked_rep3_mean",
        "attacked_rep15_mean",
        "ent3_delta_mean",
    ]
    preview_cols = [c for c in preview_cols if c in summary_all.columns]

    print(summary_all[preview_cols].to_string(index=False))

def write_bad_cases(df, out_path, max_cases=200):
    if df.empty:
        return

    cols = [
        "eval_mode",
        "attack_name",
        "model_name",
        "wm_method",
        "file",
        "line_id",

        "valid_for_semantic_destructive",
        "valid_for_auto_quality_risk",
        "valid_for_destructive",
        "core_valid",
        "valid_metric_count",
        "missing_metric_count",

        "sim_attacked_vs_wm",
        "sim_attacked_vs_human",
        "sim_attacked_vs_clean",

        "attacked_ppl",
        "wm_ppl",
        "ppl_delta",
        "ppl_ratio",

        "attacked_len",
        "wm_len",
        "len_delta",
        "len_ratio",

        "attacked_rep3",
        "attacked_rep4",
        "attacked_rep15",
        "ent3_delta",

        "acceptable_score",

        "semantic_bad",
        "length_bad",
        "fluency_bad",
        "repeat_bad",
        "entropy_bad",

        "semantic_destructive",
        "auto_quality_risk",
        "destructive",
        "strict_destructive",

        "wm_text_short",
        "attacked_text_short",
        "human_text_short",
        "clean_text_short",
    ]

    cols = [c for c in cols if c in df.columns]

    tmp = df.copy()

    # ------------------------------------------------------------
    # 修复点：
    # pandas 的 nullable boolean 中可能有 pd.NA，
    # 不能直接 astype(int)，必须先 fillna(False)。
    # bad cases 优先展示语义/长度层面真正破坏的样本，
    # 其次展示自动质量指标报警样本。
    # ------------------------------------------------------------
    if "semantic_destructive" in tmp.columns:
        tmp["_semantic_destructive_sort"] = (
            tmp["semantic_destructive"]
            .eq(True)
            .fillna(False)
            .astype("int8")
        )
    else:
        tmp["_semantic_destructive_sort"] = 0

    if "auto_quality_risk" in tmp.columns:
        tmp["_auto_quality_risk_sort"] = (
            tmp["auto_quality_risk"]
            .eq(True)
            .fillna(False)
            .astype("int8")
        )
    else:
        tmp["_auto_quality_risk_sort"] = 0

    if "destructive" in tmp.columns:
        tmp["_destructive_sort"] = (
            tmp["destructive"]
            .eq(True)
            .fillna(False)
            .astype("int8")
        )
    else:
        tmp["_destructive_sort"] = 0

    if "valid_for_destructive" in tmp.columns:
        tmp["_valid_sort"] = (
            tmp["valid_for_destructive"]
            .eq(True)
            .fillna(False)
            .astype("int8")
        )
    else:
        tmp["_valid_sort"] = 0

    if "acceptable_score" in tmp.columns:
        tmp["_score_sort"] = (
            pd.to_numeric(tmp["acceptable_score"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(999.0)
        )
    else:
        tmp["_score_sort"] = 999.0

    if "sim_attacked_vs_wm" in tmp.columns:
        tmp["_sim_sort"] = (
            pd.to_numeric(tmp["sim_attacked_vs_wm"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(999.0)
        )
    else:
        tmp["_sim_sort"] = 999.0

    # 优先输出：
    # 1. destructive=True 的样本；
    # 2. 有效样本；
    # 3. acceptable_score 低的样本；
    # 4. 语义相似度低的样本。
    bad = tmp.sort_values(
        by=[
            "_semantic_destructive_sort",
            "_auto_quality_risk_sort",
            "_destructive_sort",
            "_valid_sort",
            "_score_sort",
            "_sim_sort",
        ],
        ascending=[False, False, False, False, True, True]
    ).head(max_cases)

    def json_safe_value(v):
        """
        将 pandas / numpy 标量安全转成 JSON 可写格式。
        """
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass

        if isinstance(v, (np.bool_, bool)):
            return bool(v)

        if isinstance(v, (np.integer, int)):
            return int(v)

        if isinstance(v, (np.floating, float)):
            return float(v) if is_valid_number(v) else None

        return v

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in bad.iterrows():
            obj = {}

            for c in cols:
                obj[c] = json_safe_value(row[c])

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_report(summary_all, out_path):
    lines = []

    lines.append("# Attack Quality Analysis Report\n")

    lines.append("本报告用于辅助分析攻击后文本质量，区分语义/长度层面的破坏与自动质量指标风险。")
    lines.append("分析不以生成模型或水印方法为主，而是将样本按攻击方法聚合。")
    lines.append("\n")

    lines.append("## Evaluation Modes\n")
    lines.append("- `existing_quality`：复用已有 JSONL 中的 quality 字段，适合快速趋势分析。")
    lines.append("- `unified_evaluator`：统一使用一个固定 evaluator model 重新计算 PPL，并统一使用同一个 embedder 计算语义相似度，适合正式论文分析。")
    lines.append("\n")

    lines.append("## Validity Correction\n")
    lines.append("- `n`：该攻击方法的总样本数。")
    lines.append("- `valid_n`：可用于 destructive 判断的有效样本数。")
    lines.append("- `missing_metric_rate`：因为核心指标缺失而不能可靠判断 destructive 的样本比例。")
    lines.append("- `semantic_destructive_rate`：语义相似度或长度比例异常的比例，更接近人类视角下的内容破坏。")
    lines.append("- `auto_quality_risk_rate`：PPL、重复或熵等自动指标报警的比例，更接近模型质量指标风险。")
    lines.append("- `destructive_rate`：严格诊断项，是上述两类风险的并集，只在 `valid_n` 分母上计算，不建议直接解释为人类视角破坏率。")
    lines.append("\n")

    lines.append("## Key Interpretation\n")
    lines.append("- `sim_attacked_vs_wm_mean` 越低，说明攻击后文本与原水印文本语义偏离越大。")
    lines.append("- `ppl_ratio_mean` 越高，说明攻击后文本相对原水印文本的模型困惑度上升越明显；但它不等价于人类可读性，尤其对 typo/swap 等表层扰动较敏感。")
    lines.append("- `len_ratio_mean` 明显偏离 1，说明攻击可能造成截断、压缩或过度扩写。")
    lines.append("- `attacked_rep3_mean`、`attacked_rep15_mean` 越高，说明重复退化越严重。")
    lines.append("- `semantic_destructive_rate` 越高，说明攻击越可能改变语义或造成明显长度异常。")
    lines.append("- `auto_quality_risk_rate` 越高，说明攻击越可能触发自动质量指标报警。")
    lines.append("- `destructive_rate` 是严格诊断项，不建议单独用于判断 typo/swap 等表层扰动是否为人类视角破坏性攻击。")
    lines.append("- `acceptable_score_mean` 越高，说明该攻击更接近可接受改写。")
    lines.append("\n")

    if summary_all.empty:
        lines.append("未找到有效攻击样本。\n")
    else:
        show_cols = [
            "eval_mode",
            "attack_name",
            "n",
            "valid_n",
            "missing_metric_rate",
            "semantic_destructive_rate",
            "auto_quality_risk_rate",
            "destructive_rate",
            "acceptable_score_mean",
            "sim_attacked_vs_wm_mean",
            "ppl_ratio_mean",
            "len_ratio_mean",
            "attacked_rep3_mean",
            "attacked_rep15_mean",
            "ent3_delta_mean",
        ]
        show_cols = [c for c in show_cols if c in summary_all.columns]

        lines.append("## Overall Summary\n")
        lines.append(summary_all[show_cols].to_markdown(index=False))
        lines.append("\n")

        for mode in summary_all["eval_mode"].dropna().unique():
            sub = summary_all[summary_all["eval_mode"] == mode].copy()

            if sub.empty:
                continue

            lines.append(f"## Mode: {mode}\n")

            worst_sort_cols = [c for c in ["semantic_destructive_rate", "auto_quality_risk_rate", "destructive_rate", "acceptable_score_mean"] if c in sub.columns]
            worst_ascending = []
            for c in worst_sort_cols:
                if c == "acceptable_score_mean":
                    worst_ascending.append(True)
                else:
                    worst_ascending.append(False)

            best_sort_cols = [c for c in ["semantic_destructive_rate", "auto_quality_risk_rate", "destructive_rate", "acceptable_score_mean"] if c in sub.columns]
            best_ascending = []
            for c in best_sort_cols:
                if c == "acceptable_score_mean":
                    best_ascending.append(False)
                else:
                    best_ascending.append(True)

            worst = sub.sort_values(
                by=worst_sort_cols,
                ascending=worst_ascending
            ).head(5)

            best = sub.sort_values(
                by=best_sort_cols,
                ascending=best_ascending
            ).head(5)

            lines.append("### Highest Semantic Destructiveness / Quality Risk\n")
            lines.append(worst[show_cols].to_markdown(index=False))
            lines.append("\n")

            lines.append("### Most Acceptable Attacks\n")
            lines.append(best[show_cols].to_markdown(index=False))
            lines.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default="results/main_exp",
        help="实验结果根目录。"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/diagnostics/quality/attack_quality",
        help="输出目录。"
    )

    parser.add_argument(
        "--run_existing",
        action="store_true",
        help="运行 existing_quality 趋势分析。"
    )
    parser.add_argument(
        "--run_unified",
        action="store_true",
        help="运行 unified_evaluator 正式分析。"
    )

    parser.add_argument(
        "--evaluator_model_path",
        type=str,
        default="./models/llama3.1-8b-instruct",
        help="统一用于 PPL 评估的 evaluator LM。"
    )
    parser.add_argument(
        "--embedder_path",
        type=str,
        default="./models/bge-small-en-v1.5",
        help="统一用于语义相似度评估的 embedding model。"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--evaluator_dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="统一 PPL evaluator 的推理精度。4090 可用 auto/bf16；A4500 建议 fp16 且使用更小 evaluator。"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="PPL 计算时 tokenizer 截断长度。"
    )

    # ------------------------------------------------------------
    # Destructive attack thresholds
    # ------------------------------------------------------------
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=0.80,
        help="sim_attacked_vs_wm 低于该值则判为 semantic_bad。"
    )
    parser.add_argument(
        "--min_len_ratio",
        type=float,
        default=0.70,
        help="len_ratio 低于该值则判为 length_bad。"
    )
    parser.add_argument(
        "--max_len_ratio",
        type=float,
        default=1.30,
        help="len_ratio 高于该值则判为 length_bad。"
    )
    parser.add_argument(
        "--ppl_ratio_threshold",
        type=float,
        default=2.00,
        help="ppl_ratio 高于该值则判为 fluency_bad。"
    )
    parser.add_argument(
        "--ppl_abs_threshold",
        type=float,
        default=150.0,
        help="attacked_ppl 高于该值则判为 fluency_bad。"
    )
    parser.add_argument(
        "--rep3_threshold",
        type=float,
        default=0.20,
        help="attacked_rep3 高于该值则判为 repeat_bad。"
    )
    parser.add_argument(
        "--rep4_threshold",
        type=float,
        default=0.15,
        help="attacked_rep4 高于该值则判为 repeat_bad。"
    )
    parser.add_argument(
        "--rep15_threshold",
        type=float,
        default=0.05,
        help="attacked_rep15 高于该值则判为 repeat_bad。"
    )
    parser.add_argument(
        "--ent3_drop_threshold",
        type=float,
        default=0.80,
        help="ent3_delta 低于 -该值则判为 entropy_bad。"
    )

    # acceptable_score normalization
    parser.add_argument(
        "--semantic_bad_floor",
        type=float,
        default=0.75
    )
    parser.add_argument(
        "--semantic_good_ceiling",
        type=float,
        default=0.92
    )

    # acceptable_score weights.
    # 默认更偏人类可接受性：语义和长度主导，降低 PPL 对 typo/swap 的主导影响。
    parser.add_argument("--acceptable_semantic_weight", type=float, default=0.55)
    parser.add_argument("--acceptable_length_weight", type=float, default=0.25)
    parser.add_argument("--acceptable_repeat_weight", type=float, default=0.10)
    parser.add_argument("--acceptable_fluency_weight", type=float, default=0.05)
    parser.add_argument("--acceptable_entropy_weight", type=float, default=0.05)

    parser.add_argument(
        "--max_bad_cases",
        type=int,
        default=200
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="保存样本级明细 CSV。默认不保存，因为文件会非常大。"
    )
    parser.add_argument(
        "--save_text_examples",
        action="store_true",
        help="在样本级结果和 bad cases 中保存文本片段。默认不保存，避免文件过大。"
    )
    parser.add_argument(
        "--no_bad_cases",
        action="store_true",
        help="不保存 bad cases jsonl。"
    )

    args = parser.parse_args()

    # 如果两个都没指定，默认两个都跑
    if not args.run_existing and not args.run_unified:
        args.run_existing = True
        args.run_unified = True

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[*] root_dir = {args.root_dir}")
    print(f"[*] out_dir  = {args.out_dir}")
    print(f"[*] device   = {args.device}")
    print(f"[*] run_existing = {args.run_existing}")
    print(f"[*] run_unified  = {args.run_unified}")

    jsonl_files = find_jsonl_files(args.root_dir)

    print(f"\n[*] 找到 JSONL 文件数: {len(jsonl_files)}")

    if not jsonl_files:
        print("[!] 未找到 JSONL 文件，请检查 root_dir。")
        return

    df_existing = pd.DataFrame()
    df_unified = pd.DataFrame()

    # 1. existing_quality 趋势版
    if args.run_existing:
        df_existing = collect_records_for_mode(
            jsonl_files=jsonl_files,
            eval_mode="existing_quality",
            args=args,
            model=None,
            tokenizer=None,
            embedder=None,
        )

    # 2. unified_evaluator 正式版
    if args.run_unified:
        model, tokenizer = load_evaluator_lm(
            model_path=args.evaluator_model_path,
            device=args.device,
            dtype_name=args.evaluator_dtype
        )

        embedder = load_embedder(
            embedder_path=args.embedder_path,
            device=args.device
        )

        df_unified = collect_records_for_mode(
            jsonl_files=jsonl_files,
            eval_mode="unified_evaluator",
            args=args,
            model=model,
            tokenizer=tokenizer,
            embedder=embedder,
        )

        del model
        del tokenizer
        del embedder

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    save_outputs(df_existing, df_unified, args)

    print("\n✅ 攻击后文本质量分析完成")


if __name__ == "__main__":
    main()


"""
========================
运行指令示例
========================

1. 默认同时运行 existing_quality 和 unified_evaluator：
   注意：会加载统一 evaluator model，耗时较长，但结果最完整。

python analysis/quality/analyze_attack_quality.py \
  --root_dir results/main_exp \
  --out_dir results/diagnostics/quality/attack_quality \
  --evaluator_model_path ./models/llama3.1-8b-instruct \
  --embedder_path ./models/bge-small-en-v1.5 \
  --device cuda:0 \
  --evaluator_dtype auto


2. 只运行 existing_quality 趋势分析：
   不重新加载大模型，速度快。
   适合先看趋势，以及检查 valid_n / missing_metric_rate。

python analysis/quality/analyze_attack_quality.py \
  --root_dir results/main_exp \
  --out_dir results/diagnostics/quality/attack_quality \
  --run_existing


3. 只运行 unified_evaluator 正式分析：
   统一重算 PPL 和语义相似度。
   适合最终论文表格。

python analysis/quality/analyze_attack_quality.py \
  --root_dir results/main_exp \
  --out_dir results/diagnostics/quality/attack_quality \
  --run_unified \
  --evaluator_model_path ./models/llama3.1-8b-instruct \
  --embedder_path ./models/bge-small-en-v1.5 \
  --device cuda:0 \
  --evaluator_dtype auto


4. 保存样本级明细表：
   默认不建议开，因为 CSV 会很大。
   只有需要逐样本排查时再开。

python analysis/quality/analyze_attack_quality.py \
  --root_dir results/main_exp \
  --out_dir results/diagnostics/quality/attack_quality \
  --run_existing \
  --save_samples


5. 保存文本片段：
   需要和 --save_samples 一起使用才有意义。
   文件会更大。

python analysis/quality/analyze_attack_quality.py \
  --root_dir results/main_exp \
  --out_dir results/diagnostics/quality/attack_quality \
  --run_existing \
  --save_samples \
  --save_text_examples
"""