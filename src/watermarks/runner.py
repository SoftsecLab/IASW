# src/watermarks/runner.py
# -*- coding: utf-8 -*-

import os
import time
import signal
import argparse
import random

import numpy as np

import torch
from tqdm import tqdm

from src.watermarks.plugins import (
    get_watermark_class,
    available_watermarks,
)
from src.watermarks.runtime.loaders import load_runtime
from src.watermarks.runtime.io import (
    load_jsonl,
    save_jsonl_atomic,
)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Generation timed out!")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Watermark Runner")

    parser.add_argument("--task", type=str, required=True, choices=["generate", "detect"])

    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--embedder_path", type=str, default="")

    parser.add_argument("--baseline_file", type=str, default="")
    parser.add_argument("--target_file", type=str, required=True)

    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)

    parser.add_argument("--mode", type=str, default="", choices=["", "baseline_eval", "attack_eval"])
    parser.add_argument("--attack_name", type=str, default="")

    parser.add_argument("--max_total_tokens", type=int, default=200)
    parser.add_argument("--max_sent_tokens", type=int, default=40)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=50)

    # Controlled generation-ablation options. Defaults preserve existing runs.
    parser.add_argument(
        "--candidate_selection_policy",
        type=str,
        default="macro",
        choices=["macro", "random", "first"],
    )
    parser.add_argument("--candidate_selection_seed", type=int, default=20260823)
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=-1,
        help=(
            "If >=0, reseed Python/NumPy/PyTorch per sample using "
            "generation_seed + sample_index."
        ),
    )
    parser.add_argument("--dual_temperature", type=float, default=1.0)
    parser.add_argument(
        "--dual_top_k", type=int, default=-1,
        help="Negative means: do not explicitly pass top_k in dual generation.",
    )
    parser.add_argument(
        "--dual_top_p", type=float, default=-1.0,
        help="Negative means: do not explicitly pass top_p in dual generation.",
    )
    parser.add_argument(
        "--dual_stagea_candidates", type=int, default=-1,
        help="Negative means use --num_candidates, preserving legacy behavior.",
    )

    # Optional candidate-level IAS-W/KGW mechanism audit.
    # Empty candidate_audit_file means completely disabled.
    parser.add_argument("--candidate_audit_file", type=str, default="")
    parser.add_argument(
        "--candidate_audit_max_samples",
        type=int,
        default=0,
        help="0 means audit all generated samples.",
    )
    parser.add_argument(
        "--candidate_audit_save_text",
        action="store_true",
        help="Save full candidate text instead of a short preview.",
    )

    parser.add_argument("--secret_key", type=str, default="DUAL_MACRO_2026")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--micro_method", type=str, default="kgw")
    parser.add_argument("--macro_method", type=str, default="ias")
    parser.add_argument("--timeout", type=int, default=300)

    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--entropy_threshold", type=float, default=0.9)

    # DualMark / released Twins non-OPT reference model
    parser.add_argument(
        "--dualmark_reference_opt_path",
        type=str,
        default="./models/opt-1.3b",
    )

    # SynGuard baseline
    parser.add_argument(
        "--synguard_config",
        type=str,
        default="",
    )
    parser.add_argument(
        "--synguard_embedding_path",
        type=str,
        default="./models/compositional-bert-large-uncased",
    )

    return parser.parse_args()


def _build_engine(args, runtime):
    method = str(args.method).lower().strip()

    try:
        plugin_cls = get_watermark_class(method)
    except ValueError as exc:
        raise ValueError(
            f"未知算法: {method}。"
            f"可用算法: {available_watermarks()}"
        ) from exc

    return plugin_cls(
        args,
        runtime.tokenizer,
        runtime.model,
        runtime.embedder,
    )

# ============================================================
# Runtime Loading Policy
# ============================================================

DUAL_METHOD_NAMES = {
    "universal_dual",
    "dual",
}


# Detection: methods that really need the base CausalLM
# ------------------------------------------------------------

# Sweet detection computes entropy using the base LLM.
# DualMark/Twins requires the base model because TSW uses
# the generation model's input embedding matrix.
MICRO_METHODS_NEED_MODEL_IN_DETECT = {
    "sweet",
    "dualmark",
}


# ------------------------------------------------------------
# Runtime-level SentenceTransformer requirements
# ------------------------------------------------------------

# Single methods that require the embedder loaded by runtime/loaders.py.
#
# SemStamp / k-SemStamp / SimMark are intentionally NOT listed here:
# they load their own official semantic models inside their implementations.
SINGLE_METHODS_NEED_RUNTIME_EMBEDDER = {
    "ias",
    "iasw",
    "ias_w",   # compatibility alias
}


# Macro methods in universal_dual that require the runtime embedder.
MACRO_METHODS_NEED_RUNTIME_EMBEDDER = {
    "ias",
    "iasw",
    "ias_w",   # compatibility alias


}


def _lower_attr(
    args,
    name: str,
    default: str = "",
) -> str:
    return str(
        getattr(args, name, default)
        or default
    ).lower().strip()


def _is_dual_method(args) -> bool:
    return (
        _lower_attr(args, "method")
        in DUAL_METHOD_NAMES
    )


def needs_model_for_detection(args) -> bool:
    """
    Whether detection requires the base AutoModelForCausalLM.

    Rules
    -----
    single sweet:
        True

    universal_dual + micro=sweet:
        True

    everything else:
        False
    """

    method = _lower_attr(
        args,
        "method",
    )

    if _is_dual_method(args):
        micro_method = _lower_attr(
            args,
            "micro_method",
            "kgw",
        )

        return (
            micro_method
            in MICRO_METHODS_NEED_MODEL_IN_DETECT
        )

    return (
        method
        in MICRO_METHODS_NEED_MODEL_IN_DETECT
    )


def needs_runtime_embedder(args) -> bool:
    """
    Whether load_runtime() should load the runtime-level
    SentenceTransformer.

    This policy is shared by generation and detection.
    """

    method = _lower_attr(
        args,
        "method",
    )

    if _is_dual_method(args):
        macro_method = _lower_attr(
            args,
            "macro_method",
            "ias",
        )

        return (
            macro_method
            in MACRO_METHODS_NEED_RUNTIME_EMBEDDER
        )

    return (
        method
        in SINGLE_METHODS_NEED_RUNTIME_EMBEDDER
    )


def needs_embedder_for_detection(args) -> bool:
    return needs_runtime_embedder(args)


def needs_embedder_for_generation(args) -> bool:
    return needs_runtime_embedder(args)

def run_generation(args):
    if not args.baseline_file:
        raise ValueError("--task generate 时必须提供 --baseline_file")

    os.makedirs(os.path.dirname(args.target_file), exist_ok=True)

    print(f"[*] 加载基座模型: {args.model_path}")

    load_embedder = needs_embedder_for_generation(args)

    print(
        f"[*] Runtime policy for Generation "
        f"(Method: {args.method})..."
    )

    if _is_dual_method(args):
        print(
            f"    - dual config: "
            f"micro={args.micro_method}, "
            f"macro={args.macro_method}"
        )

    print("    - load_model   : True")
    print(f"    - load_embedder: {load_embedder}")

    runtime = load_runtime(
        args,
        task="generate",
        load_model=True,
        load_embedder=load_embedder,
    )
    print(f"[*] 成功挂载水印引擎: {args.method.upper()} | 数据集: {args.dataset_type.upper()}")
    watermark_engine = _build_engine(args, runtime)

    print(f"[*] 从基线文件初始化模板: {args.baseline_file}")
    full_baseline = load_jsonl(args.baseline_file)
    data = full_baseline[:args.num_samples]

    audit_file = str(getattr(args, "candidate_audit_file", "") or "").strip()
    if audit_file:
        os.makedirs(os.path.dirname(audit_file) or ".", exist_ok=True)
        if os.path.exists(audit_file):
            print(f"[*] 删除旧 candidate audit 文件并重新记录: {audit_file}")
            os.remove(audit_file)

    if os.path.exists(args.target_file):
        print(f"[*] 检测到已有结果文件，但本次将忽略旧文件并全量重跑: {args.target_file}")

    signal.signal(signal.SIGALRM, timeout_handler)
    need_save = False

    for index, row in enumerate(tqdm(data)):
        prompt = row.get("prompt", "")
        if not prompt:
            continue

        row.pop("wm_text", None)
        row.pop("generation_time", None)
        row.pop("error", None)

        start_time = time.time()
        try:
            signal.alarm(args.timeout)

            # Optional per-sample reseeding for paired ablations.
            # Default -1 leaves the original behavior untouched.
            if int(getattr(args, "generation_seed", -1)) >= 0:
                sample_seed = int(args.generation_seed) + int(index)
                random.seed(sample_seed)
                np.random.seed(sample_seed % (2**32 - 1))
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)

            with torch.no_grad():
                row["wm_text"] = watermark_engine.generate(prompt)

            row["generation_time"] = time.time() - start_time
            need_save = True

        except TimeoutException:
            print(f"\n[!] 🚨 超时跳过: 第 {index} 条数据")
            row["wm_text"] = ""
            row["error"] = "timeout"
            need_save = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[!] ❌ 生成失败: 第 {index} 条数据 | 原因: {str(e)}")
            row["wm_text"] = ""
            row["error"] = str(e)
            need_save = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        finally:
            signal.alarm(0)

        row["meta_info"] = {
            "method": args.method,
            "dataset_type": args.dataset_type,
            "model": os.path.basename(args.model_path),
            "candidate_selection_policy": getattr(
                args, "candidate_selection_policy", "macro"
            ),
            "candidate_selection_seed": int(
                getattr(args, "candidate_selection_seed", 20260823)
            ),
            "generation_seed": int(getattr(args, "generation_seed", -1)),
            "dual_temperature": float(getattr(args, "dual_temperature", 1.0)),
            "dual_top_k": int(getattr(args, "dual_top_k", -1)),
            "dual_top_p": float(getattr(args, "dual_top_p", -1.0)),
            "dual_stagea_candidates": int(
                getattr(args, "dual_stagea_candidates", -1)
            ),
            "candidate_audit_file": str(
                getattr(args, "candidate_audit_file", "") or ""
            ),
            "candidate_audit_max_samples": int(
                getattr(args, "candidate_audit_max_samples", 0)
            ),
        }

    if need_save:
        save_jsonl_atomic(args.target_file, data)
        print(f"[+] 任务更新成功，已保存至: {args.target_file}")
    else:
        print("⚡ 本次没有可保存内容。")

@torch.no_grad() # 确保加上这个装饰器
def run_detection(args):
    if not os.path.exists(args.target_file):
        print(f"❌ 找不到目标文件: {args.target_file}")
        return

    if args.mode not in {"baseline_eval", "attack_eval"}:
        raise ValueError("--task detect 时必须提供 --mode baseline_eval 或 attack_eval")

    load_model = needs_model_for_detection(args)
    load_embedder = needs_embedder_for_detection(args)

    print(f"[*] Loading Runtime for Detection (Method: {args.method})...")
    if _is_dual_method(args):
        print(
            f"    - dual config: micro={args.micro_method}, macro={args.macro_method}"
        )
    print(f"    - load_model   : {load_model}")
    print(f"    - load_embedder: {load_embedder}")

    runtime = load_runtime(
        args,
        task="detect",
        load_model=load_model,
        load_embedder=load_embedder,
    )
    watermark_engine = _build_engine(args, runtime)

    print(f"[*] Reading {args.target_file}...")
    data = load_jsonl(args.target_file)

    try:
        # for row in tqdm(data, desc=f"Refreshing Scores ({args.mode})"):
        for idx, row in enumerate(tqdm(data, desc=f"Refreshing Scores ({args.mode})")):
            if args.mode == "baseline_eval":
                row["detection_human"] = watermark_engine.detect(row.get("human_text", ""))
                row["detection_no_wm"] = watermark_engine.detect(row.get("no_wm_text", ""))

                wm_text = row.get("wm_text", "")
                if wm_text and wm_text.strip():
                    row["detection_wm"] = watermark_engine.detect(wm_text)
                else:
                    row["detection_wm"] = {"error": "empty_or_failed_generation"}

            elif args.mode == "attack_eval":
                attack_name = args.attack_name
                if not attack_name:
                    raise ValueError("--task detect --mode attack_eval 时必须提供 --attack_name")

                if "attacks" not in row or attack_name not in row["attacks"]:
                    continue

                attack_record = row["attacks"][attack_name]
                attacked_text = attack_record.get("attacked_text", "")


                if attacked_text and attacked_text.strip():
                    attack_record["detection_results"] = watermark_engine.detect(attacked_text)
                else:
                    attack_record["detection_results"] = {"error": "empty_attacked_text"}

    except KeyboardInterrupt:
        print("\n[!] 🛑 检测被用户中断！正在放弃本次更新以保护原文件...")
        return

    print("[*] 检测完成，正在执行原子写入...")
    save_jsonl_atomic(args.target_file, data)
    print(f"[+] 成功更新得分并保存至: {args.target_file}")


def main():
    args = parse_args()

    if args.task == "generate":
        run_generation(args)
        return

    if args.task == "detect":
        run_detection(args)
        return

    raise ValueError(f"未知 task: {args.task}")


if __name__ == "__main__":
    main()