# src/watermarks/dual/orchestrator.py
# -*- coding: utf-8 -*-

import os
import re
import gc
import time
import hashlib
import json
import torch
import numpy as np

from src.watermarks.plugins import BaseWatermark, register_watermark
from src.watermarks.runtime.profiling import safe_mean
from src.watermarks.dual.calibrator import infer_model_key, get_calibration_params

from src.watermarks.registry import (
    build_micro_algorithm,
    build_macro_algorithm,
    normalize_micro_name,
    normalize_macro_name,
)

@register_watermark("universal_dual")
class UniversalDualPlugin(BaseWatermark):
    def __init__(self, args, tokenizer, model, embedder=None):
        super().__init__(args, tokenizer, model, embedder)

        self.micro_name = normalize_micro_name(
            getattr(
                args,
                "micro_method",
                "kgw",
            )
        )

        self.macro_name = normalize_macro_name(
            getattr(
                args,
                "macro_method",
                "ias",
            )
        )
        self.dataset_type = getattr(args, "dataset_type", "c4").lower()
        self.model_key = infer_model_key(getattr(args, "model_path", ""))

        # Candidate-selection ablation. Default "macro" preserves the original behavior.
        self.candidate_selection_policy = str(
            getattr(args, "candidate_selection_policy", "macro") or "macro"
        ).lower().strip()
        self.candidate_selection_seed = int(
            getattr(args, "candidate_selection_seed", 20260823)
        )

        if self.candidate_selection_policy not in {"macro", "random", "first"}:
            raise ValueError(
                "candidate_selection_policy must be one of: macro, random, first"
            )

        # Random/first are ablation controls for the CURRENT KGW + IAS-W path.
        # Current routing: macro_method=ias -> original IAS; macro_method=iasw -> current IAS-W.
        if self.candidate_selection_policy != "macro" and self.macro_name != "iasw":
            raise ValueError(
                "candidate_selection_policy=random/first is intended for "
                "macro_method=iasw (current IAS-W) only."
            )

        # Dual sampling controls. Negative top-k/top-p means do not explicitly
        # pass that argument, preserving legacy orchestrator behavior.
        self.dual_temperature = float(getattr(args, "dual_temperature", 1.0))
        self.dual_top_k = int(getattr(args, "dual_top_k", -1))
        self.dual_top_p = float(getattr(args, "dual_top_p", -1.0))
        self.dual_stagea_candidates = int(
            getattr(args, "dual_stagea_candidates", -1)
        )

        # ---------------------------------------------------------
        # Optional candidate-level mechanism audit.
        #
        # Disabled by default, so normal main experiments are unchanged.
        # The audit is intended for:
        #   micro_method=kgw, macro_method=iasw, policy=macro
        #
        # It records all candidates generated under the SAME current prefix
        # and measures both IAS-W semantic similarity and KGW evidence.
        # ---------------------------------------------------------
        self.candidate_audit_file = str(
            getattr(args, "candidate_audit_file", "") or ""
        ).strip()
        self.candidate_audit_max_samples = int(
            getattr(args, "candidate_audit_max_samples", 0)
        )
        self.candidate_audit_save_text = bool(
            getattr(args, "candidate_audit_save_text", False)
        )
        self._candidate_audit_sample_counter = 0

        self.profile_enabled = os.environ.get("DUAL_PROFILE", "0") == "1"
        self.profile_verbose = os.environ.get("DUAL_PROFILE_VERBOSE", "0") == "1"

        self.micro = self._build_micro_algorithm()
        self.macro = self._build_macro_algorithm()



    def _build_micro_algorithm(self):
        return build_micro_algorithm(
            name=self.micro_name,
            args=self.args,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
        )

    def _build_macro_algorithm(self):
        return build_macro_algorithm(
            name=self.macro_name,
            args=self.args,
            tokenizer=self.tokenizer,
            embedder=self.embedder,
            model=self.model,
            device=self.device,
        )

    def _dual_sampling_kwargs(self) -> dict:
        kwargs = {
            "do_sample": True,
            "temperature": float(self.dual_temperature),
        }
        if self.dual_top_k >= 0:
            kwargs["top_k"] = int(self.dual_top_k)
        if 0.0 < self.dual_top_p <= 1.0:
            kwargs["top_p"] = float(self.dual_top_p)
        return kwargs

    def _advance_ias_context_without_macro_selection(self, context):
        # Current IAS-W uses an integer sentence/anchor index.
        if isinstance(context, (int, np.integer)):
            return int(context) + 1
        raise TypeError(
            "Random candidate-selection control expects the current IAS-W "
            "integer sentence-index context."
        )

    def _select_candidate_for_ablation(
        self, candidates, context, original_prompt: str, loop_idx: int
    ):
        """Return selected_sentence, new_context, selected_idx.

        macro  : original IAS-W selection.
        random : uniform reproducible candidate-index selection.
        first  : sanity control.
        """
        if not candidates:
            raise ValueError("Candidate list is empty.")

        if self.candidate_selection_policy == "macro":
            best_sentence, new_context = self.macro.select_best_candidate(
                candidates, context
            )
            selected_idx = candidates.index(best_sentence)
            return best_sentence, new_context, selected_idx

        if self.candidate_selection_policy == "random":
            payload = (
                f"{self.candidate_selection_seed}|{original_prompt}|"
                f"{loop_idx}|{len(candidates)}"
            )
            digest = hashlib.sha256(payload.encode("utf-8")).digest()
            selected_idx = int.from_bytes(
                digest[:8], byteorder="big", signed=False
            ) % len(candidates)
        elif self.candidate_selection_policy == "first":
            selected_idx = 0
        else:
            raise ValueError(self.candidate_selection_policy)

        selected_sentence = candidates[selected_idx]
        new_context = self._advance_ias_context_without_macro_selection(context)
        return selected_sentence, new_context, selected_idx

    @staticmethod
    def _audit_safe_float(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    @staticmethod
    def _audit_rank_desc(values):
        """
        Descending rank: 1 = best. NaN/None -> None.
        Ties receive the minimum rank.
        """
        arr = np.asarray(
            [np.nan if v is None else v for v in values],
            dtype=float,
        )
        out = [None] * len(arr)
        finite_idx = np.where(np.isfinite(arr))[0]

        if len(finite_idx) == 0:
            return out

        vals = arr[finite_idx]
        order = np.argsort(-vals)
        sorted_vals = vals[order]

        local_ranks = {}
        current_rank = 1
        for pos, local_idx in enumerate(order):
            if pos == 0:
                current_rank = 1
            elif not np.isclose(
                sorted_vals[pos],
                sorted_vals[pos - 1],
                rtol=1e-12,
                atol=1e-12,
            ):
                current_rank = pos + 1
            local_ranks[int(local_idx)] = int(current_rank)

        for local_idx, original_idx in enumerate(finite_idx):
            out[int(original_idx)] = local_ranks.get(int(local_idx))

        return out

    @staticmethod
    def _audit_percentile(rank, n):
        """
        Convert descending rank to [0,1]:
          best -> 1
          worst -> 0
        Uniform random expectation is approximately 0.5.
        """
        if rank is None or n <= 1:
            return None
        return float((n - int(rank)) / (n - 1))

    def _audit_append_jsonl(self, obj):
        if not self.candidate_audit_file:
            return

        path = os.path.abspath(self.candidate_audit_file)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _audit_iasw_scores(self, candidates, context):
        """
        Return IAS-W cosine scores for all candidates under one shared prefix.

        Current main-project IAS-W path:
            macro_method=iasw
            IASWMacroAlgorithm.ias_w_mark -> IASWMark
        """
        if not candidates:
            return np.asarray([], dtype=float)

        mark = getattr(self.macro, "ias_w_mark", None)
        if mark is None:
            # Tolerate an alternate internal attribute name if the wrapper
            # is renamed later.
            mark = getattr(self.macro, "iasw_mark", None)

        if mark is None:
            raise AttributeError(
                "Current IAS-W mark object was not found on macro algorithm."
            )

        with torch.no_grad():
            target = mark.get_target_semantic_vector(int(context))
            embeddings = mark.encode_whitened(candidates)
            similarities = torch.mm(
                embeddings,
                target.T,
            ).squeeze(1)

        return similarities.detach().float().cpu().numpy()

    def _audit_kgw_stats(self, text):
        """
        Read KGW detector statistics directly, avoiding the micro wrapper's
        gc()/empty_cache() overhead for every candidate.
        """
        if not text or not text.strip():
            return {
                "z": 0.0,
                "num_tokens_scored": 0.0,
                "num_green_tokens": 0.0,
                "green_fraction": None,
            }

        detector = getattr(self.micro, "detector", None)
        if detector is None:
            raise AttributeError("KGW detector not found on micro algorithm.")

        try:
            with torch.no_grad():
                res = detector.detect(text)
        except Exception:
            res = {}

        z = self._audit_safe_float(res.get("z_score"))
        n_scored = self._audit_safe_float(
            res.get(
                "num_tokens_scored",
                res.get("T", res.get("num_scored")),
            )
        )
        n_green = self._audit_safe_float(
            res.get(
                "num_green_tokens",
                res.get("green_token_count", res.get("green_tokens")),
            )
        )
        green_fraction = self._audit_safe_float(
            res.get(
                "green_fraction",
                res.get("green_token_fraction", res.get("fraction")),
            )
        )

        if (
            green_fraction is None
            and n_scored is not None
            and n_green is not None
            and n_scored > 0
        ):
            green_fraction = float(n_green / n_scored)

        return {
            "z": z,
            "num_tokens_scored": n_scored,
            "num_green_tokens": n_green,
            "green_fraction": green_fraction,
        }

    def _candidate_audit_enabled_for_sample(self, sample_idx):
        if not self.candidate_audit_file:
            return False

        if self.micro_name != "kgw" or self.macro_name != "iasw":
            return False

        if self.candidate_selection_policy != "macro":
            return False

        limit = int(self.candidate_audit_max_samples)
        return limit <= 0 or sample_idx < limit

    def _audit_candidate_pool(
        self,
        *,
        sample_idx,
        loop_idx,
        context,
        generated_so_far,
        candidates,
        selected_idx,
        ias_scores,
    ):
        """
        Candidate-level analysis under a shared current prefix.

        This function is deliberately called AFTER candidate generation but
        BEFORE the selected sentence is appended to the trajectory.
        """
        if not candidates:
            return

        prefix_stats = self._audit_kgw_stats(generated_so_far)
        rows = []

        for cand_idx, cand in enumerate(candidates):
            combined = (
                generated_so_far
                + (" " if generated_so_far and cand else "")
                + cand
            )

            stats_after = self._audit_kgw_stats(combined)

            def delta(key):
                a = stats_after.get(key)
                b = prefix_stats.get(key)
                if a is None or b is None:
                    return None
                return float(a - b)

            dz = delta("z")
            dn = delta("num_tokens_scored")
            dg = delta("num_green_tokens")

            local_green_fraction = None
            if (
                dn is not None
                and dg is not None
                and dn > 0
            ):
                local_green_fraction = float(dg / dn)

            rows.append({
                "candidate_idx": int(cand_idx),
                "ias_similarity": self._audit_safe_float(
                    ias_scores[cand_idx]
                ),
                "candidate_tokens": int(self._count_tokens(cand)),
                "kgw_z_after_candidate": stats_after.get("z"),
                "kgw_delta_z": dz,
                "delta_tokens_scored": dn,
                "delta_green_tokens": dg,
                "local_green_fraction": local_green_fraction,
                "candidate_text": (
                    cand
                    if self.candidate_audit_save_text
                    else cand[:240]
                ),
            })

        ias_ranks = self._audit_rank_desc(
            [r["ias_similarity"] for r in rows]
        )
        dz_ranks = self._audit_rank_desc(
            [r["kgw_delta_z"] for r in rows]
        )
        green_ranks = self._audit_rank_desc(
            [r["local_green_fraction"] for r in rows]
        )

        for i, row in enumerate(rows):
            row["ias_rank"] = ias_ranks[i]
            row["kgw_delta_z_rank"] = dz_ranks[i]
            row["local_green_fraction_rank"] = green_ranks[i]

        selected = rows[selected_idx]

        obj = {
            "sample_idx": int(sample_idx),
            "sentence_step": int(loop_idx),
            "anchor_index": int(context),
            "selection_policy": self.candidate_selection_policy,
            "num_candidates": int(len(candidates)),
            "selected_idx": int(selected_idx),

            "prefix_generated_tokens": int(
                self._count_tokens(generated_so_far)
            ),
            "prefix_kgw_z": prefix_stats.get("z"),
            "prefix_tokens_scored": prefix_stats.get("num_tokens_scored"),
            "prefix_green_tokens": prefix_stats.get("num_green_tokens"),

            "selected_ias_similarity": selected.get("ias_similarity"),
            "selected_kgw_delta_z": selected.get("kgw_delta_z"),
            "selected_local_green_fraction": selected.get(
                "local_green_fraction"
            ),
            "selected_candidate_tokens": selected.get("candidate_tokens"),
            "selected_delta_tokens_scored": selected.get(
                "delta_tokens_scored"
            ),
            "selected_delta_green_tokens": selected.get(
                "delta_green_tokens"
            ),

            "selected_kgw_rank": selected.get("kgw_delta_z_rank"),
            "selected_kgw_percentile": self._audit_percentile(
                selected.get("kgw_delta_z_rank"),
                len(rows),
            ),
            "selected_green_rank": selected.get(
                "local_green_fraction_rank"
            ),
            "selected_green_percentile": self._audit_percentile(
                selected.get("local_green_fraction_rank"),
                len(rows),
            ),

            "candidates": rows,
        }

        self._audit_append_jsonl(obj)

    def _count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompt: str) -> str:
        prof = {
            "stageA_time": 0.0,
            "outer_loops": 0,
            "generate_times": [],
            "avg_candidate_first_sent_tokens_per_loop": [],
            "best_sentence_tokens_per_loop": [],
            "selected_raw_new_tokens_per_loop": [],
        }
        try:
            import nltk
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Dual generation requires NLTK for sentence tokenization."
            ) from exc

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        from transformers import LogitsProcessorList

        audit_sample_idx = int(self._candidate_audit_sample_counter)
        self._candidate_audit_sample_counter += 1

        original_prompt_for_selection = prompt
        original_prompt_preview = prompt[:120].replace("\n", "\\n")

        # ---------------------------------------------------------
        # Stage A: 如果 prompt 不是完整句，先用微观水印补全到句号
        # 修复点:
        # 1) 使用 input_ids 而不是未定义的 current_ids
        # 2) 兼容 Tensor / outputs.sequences 两种 generate 返回格式
        # ---------------------------------------------------------
        if not re.search(r'[.!?。！？\n]\s*$', prompt):
            t0 = time.perf_counter()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

            local_processor = self.micro.build_processor()

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=getattr(self.args, "max_sent_tokens", 40),
                    num_return_sequences=(
                        self.dual_stagea_candidates
                        if self.dual_stagea_candidates > 0
                        else getattr(self.args, "num_candidates", 32)
                    ),
                    **self._dual_sampling_kwargs(),
                    logits_processor=LogitsProcessorList([local_processor]),
                    pad_token_id=self.tokenizer.eos_token_id
                )

            if self.micro_name == "dipmark" and hasattr(self.micro, "absorb_profile"):
                self.micro.absorb_profile(local_processor)

            prompt_len = input_ids.shape[1]
            generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs

            generated_str = self.tokenizer.decode(
                generated_ids[0, prompt_len:],
                skip_special_tokens=True
            )

            match = re.search(r'[.!?。！？\n]', generated_str)
            if match:
                prompt += generated_str[:match.end()]
            else:
                prompt += (generated_str + ".")

            del outputs
            del generated_ids
            del local_processor
            prof["stageA_time"] = time.perf_counter() - t0

        current_text = prompt
        current_ids = self.tokenizer.encode(current_text, return_tensors="pt").to(self.device)
        prompt_total_len = current_ids.shape[1]

        context = self.macro.init_context(current_text)
        max_total_tokens = getattr(self.args, "max_total_tokens", 200)

        # Only the generated continuation is scored by the normal detector,
        # so candidate-level KGW increments are measured on this continuation
        # prefix rather than on the original prompt.
        generated_so_far = ""

        if self.profile_enabled:
            self.micro.reset_profile()

        while current_ids.shape[1] - prompt_total_len < max_total_tokens:
            prof["outer_loops"] += 1
            loop_idx = prof["outer_loops"]

            input_len = current_ids.shape[1]
            attention_mask = torch.ones_like(current_ids)

            t0 = time.perf_counter()
            with torch.no_grad():
                if getattr(self.micro, "uses_custom_candidate_generation", False):
                    outputs = self.micro.generate_candidates(
                        current_ids=current_ids,
                        max_new_tokens=getattr(self.args, "max_sent_tokens", 40),
                        num_candidates=getattr(self.args, "num_candidates", 32),
                    )
                else:
                    outputs = self.model.generate(
                        input_ids=current_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=getattr(self.args, "max_sent_tokens", 40),
                        num_return_sequences=getattr(self.args, "num_candidates", 32),
                        **self._dual_sampling_kwargs(),
                        logits_processor=LogitsProcessorList([self.micro.build_processor()]),
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            gen_elapsed = time.perf_counter() - t0
            prof["generate_times"].append(gen_elapsed)

            candidates = []
            valid_outputs = []
            cand_first_sent_lens = []

            # 这里假设 outputs 为 Tensor；若后续某微观方法改成 return_dict_in_generate，
            # 也可以类似 Stage A 再做一次兼容包装
            for i in range(outputs.shape[0]):
                new_tokens = outputs[i][input_len:]
                cand_str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                sents = nltk.sent_tokenize(cand_str.strip())
                first_sent = sents[0] if sents else cand_str
                first_sent = first_sent.strip()

                candidates.append(first_sent)
                valid_outputs.append(outputs[i])
                cand_first_sent_lens.append(self._count_tokens(first_sent))

            avg_cand_tok = safe_mean(cand_first_sent_lens)
            prof["avg_candidate_first_sent_tokens_per_loop"].append(avg_cand_tok)

            audit_this_sample = self._candidate_audit_enabled_for_sample(
                audit_sample_idx
            )
            audit_context_before_selection = context
            audit_ias_scores = None

            if audit_this_sample:
                audit_ias_scores = self._audit_iasw_scores(
                    candidates,
                    audit_context_before_selection,
                )

            best_sentence, context, best_idx = self._select_candidate_for_ablation(
                candidates=candidates,
                context=context,
                original_prompt=original_prompt_for_selection,
                loop_idx=loop_idx,
            )

            if audit_this_sample:
                self._audit_candidate_pool(
                    sample_idx=audit_sample_idx,
                    loop_idx=loop_idx,
                    context=audit_context_before_selection,
                    generated_so_far=generated_so_far,
                    candidates=candidates,
                    selected_idx=best_idx,
                    ias_scores=audit_ias_scores,
                )
            selected_full_seq = valid_outputs[best_idx]
            new_gen_tokens = selected_full_seq[input_len:]

            has_eos = self.tokenizer.eos_token_id in new_gen_tokens
            best_sentence_tok = self._count_tokens(best_sentence)
            selected_raw_tok = int(new_gen_tokens.shape[0])

            prof["best_sentence_tokens_per_loop"].append(best_sentence_tok)
            prof["selected_raw_new_tokens_per_loop"].append(selected_raw_tok)

            if self.profile_enabled and self.profile_verbose:
                print(
                    f"[DUAL_PROFILE][loop={loop_idx}] "
                    f"micro={self.micro_name} macro={self.macro_name} "
                    f"selection={self.candidate_selection_policy} "
                    f"gen_time={gen_elapsed:.4f}s "
                    f"avg_cand_first_sent_tok={avg_cand_tok:.2f} "
                    f"best_sent_tok={best_sentence_tok} "
                    f"selected_raw_new_tok={selected_raw_tok} "
                    f"has_eos={has_eos}"
                )

            current_text += (" " if not current_text.endswith((" ", "\n")) else "") + best_sentence
            generated_so_far += (
                " "
                if generated_so_far
                and not generated_so_far.endswith((" ", "\n"))
                else ""
            ) + best_sentence

            current_ids = self.tokenizer.encode(current_text, return_tensors="pt").to(self.device)

            del outputs
            del valid_outputs

            if has_eos:
                break

        generated_ids = current_ids[0, prompt_total_len:]
        final_generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        final_output_tokens = int(generated_ids.shape[0])

        if self.profile_enabled:
            total_generate_time = sum(prof["generate_times"])
            avg_generate_time = safe_mean(prof["generate_times"])
            avg_cand_first_sent_tok = safe_mean(prof["avg_candidate_first_sent_tokens_per_loop"])
            avg_best_sentence_tok = safe_mean(prof["best_sentence_tokens_per_loop"])
            avg_selected_raw_tok = safe_mean(prof["selected_raw_new_tokens_per_loop"])

            processor_profile = self.micro.get_profile()
            proc_name = processor_profile.get("name", "unknown")
            proc_calls = processor_profile.get("call_count", 0)
            proc_total = processor_profile.get("total_time", 0.0)
            proc_avg = processor_profile.get("avg_time", 0.0)

            print(
                f"[DUAL_PROFILE][summary] "
                f"micro={self.micro_name} macro={self.macro_name} "
                f"selection={self.candidate_selection_policy} "
                f"outer_loops={prof['outer_loops']} "
                f"stageA_time={prof['stageA_time']:.4f}s "
                f"total_generate_time={total_generate_time:.4f}s "
                f"avg_generate_time_per_loop={avg_generate_time:.4f}s "
                f"avg_candidate_first_sent_tokens={avg_cand_first_sent_tok:.2f} "
                f"avg_best_sentence_tokens={avg_best_sentence_tok:.2f} "
                f"avg_selected_raw_new_tokens={avg_selected_raw_tok:.2f} "
                f"final_output_tokens={final_output_tokens} "
                f"processor={proc_name} "
                f"processor_calls={proc_calls} "
                f"processor_total_time={proc_total:.4f}s "
                f"processor_avg_time={proc_avg:.6f}s "
                f"prompt_preview='{original_prompt_preview}'"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return final_generated_text

    def detect(self, text: str) -> dict:

        try:
            import scipy.stats as stats
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Dual Fisher-fusion detection requires scipy."
            ) from exc
        micro_res = self.micro.detect(text)
        macro_res = self.macro.detect(text)

        z_micro = float(micro_res.get("z_score_micro", micro_res.get("z_score", 0.0)))
        raw_macro = float(macro_res.get("z_score_macro", macro_res.get("z_score", 0.0)))

        if self.macro.needs_dual_calibration:
            mu, sigma = get_calibration_params(
                tokenizer=self.tokenizer,
                text=text,
                dataset_type=self.dataset_type,
                model_key=self.model_key,
                macro_name=self.macro_name
            )
            calibrated_macro = (raw_macro - mu) / sigma if sigma != 0 else raw_macro
        else:
            calibrated_macro = raw_macro

        z_max = max(z_micro, calibrated_macro)

        p_micro = max(stats.norm.sf(z_micro), 1e-300)
        p_macro = max(stats.norm.sf(calibrated_macro), 1e-300)

        chi2_stat = -2 * np.log(p_micro) - 2 * np.log(p_macro)
        p_fisher = stats.chi2.sf(chi2_stat, df=4)
        z_fisher = float(stats.norm.isf(min(max(p_fisher, 1e-300), 1.0 - 1e-16)))

        return {
            "z_score": z_fisher,
            "z_score_micro": z_micro,
            "z_score_macro": raw_macro,
            "z_score_macro_calibrated": calibrated_macro,
            "z_score_max": z_max,
            "z_score_fisher": z_fisher,
            "details": {
                "micro_details": micro_res.get("details", {}),
                "macro_details": macro_res.get("details", {})
            }
        }