
# -*- coding: utf-8 -*-
import os
import json
import time
import argparse
import shutil
from tqdm import tqdm

# Concrete attack implementations are imported lazily in AttackRouter.


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Text Attack Router (统一文本攻击大管家)")

    # 基础 I/O 配置
    parser.add_argument("--target_file", type=str, required=True, help="集成数据的 JSONL 文件路径")
    parser.add_argument("--attack_name", type=str, required=True, help="要执行的攻击名称 (如 dipper, pegasus, copy_paste等)")
    parser.add_argument("--device", type=str, default="cuda:0")

    # 模型路径配置
    parser.add_argument("--dipper_path", type=str, default="/home/share/models/dipper-paraphraser-xxl")
    parser.add_argument("--paraphrase_path", type=str, default="/home/share/models/tuner007_pegasus_paraphrase")
    parser.add_argument("--embedder_path", type=str, default="/home/share/models/all-MiniLM-L6-v2")
    parser.add_argument("--fwd_trans_path", type=str, default="/home/share/models/opus-mt-en-fr")
    parser.add_argument("--bwd_trans_path", type=str, default="/home/share/models/opus-mt-fr-en")

    # 攻击超参数
    parser.add_argument("--dipper_lex_div", type=int, default=60)
    parser.add_argument("--dipper_order_div", type=int, default=60)
    parser.add_argument("--pegasus_num_beams", type=int, default=25)
    parser.add_argument("--pegasus_max_drop", type=float, default=0.1)
    parser.add_argument("--cp_mix_ratio", type=float, default=0.5)
    parser.add_argument("--cp_mix_mode", type=str, choices=["append", "prepend", "random", "interleave"], default="prepend")
    parser.add_argument("--del_drop_prob", type=float, default=0.2)
    parser.add_argument("--del_insert_prob", type=float, default=0.2)
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    parser.add_argument("--swap_ratio", type=float, default=0.1)
    parser.add_argument("--typo_ratio", type=float, default=0.05)
    parser.add_argument("--reorder_ratio", type=float, default=0.5)
    parser.add_argument("--reorder_mode", type=str, choices=["shuffle", "adjacent_swap"], default="shuffle")
    parser.add_argument("--seed", type=int, default=42)

    # 只增加两个简单的异常输出保护阈值
    parser.add_argument("--max_attack_chars", type=int, default=50000)
    parser.add_argument("--max_attack_expansion", type=float, default=20.0)

    return parser.parse_args()


def is_abnormal_attack(attacked_text, wm_text, max_chars=50000, max_expansion=20.0):
    """只拦截明显爆炸的攻击输出。"""
    if not isinstance(attacked_text, str) or not attacked_text.strip():
        return True, "empty_or_invalid"

    if len(attacked_text) > max_chars:
        return True, f"too_long:{len(attacked_text)}"

    base_len = max(len(wm_text), 1)
    expansion = len(attacked_text) / base_len

    if expansion > max_expansion:
        return True, f"expansion_too_large:{expansion:.2f}x"

    return False, ""


class AttackRouter:
    def __init__(self, args):
        self.args = args
        self.attack_name = args.attack_name
        self.attacker_instance = None
        self.models_loaded = False

    def _lazy_load_model(self):
        """懒加载策略：只在需要时加载对应模型"""
        if self.models_loaded:
            return

        print(f"[*] 正在挂载攻击核心: [{self.attack_name.upper()}] ...")

        if self.attack_name == "dipper":
            from src.attacks.dipper_attack import DipperAttacker
            self.attacker_instance = DipperAttacker(self.args.dipper_path, self.args.device)

        elif self.attack_name == "pegasus":
            from src.attacks.pegasus_attack import PegasusAttacker
            self.attacker_instance = PegasusAttacker(
                self.args.paraphrase_path,
                self.args.embedder_path,
                self.args.device
            )

        elif self.attack_name == "back_translation":
            from src.attacks.back_translation_attack import BackTranslationAttacker
            self.attacker_instance = BackTranslationAttacker(
                self.args.fwd_trans_path,
                self.args.bwd_trans_path,
                self.args.device
            )

        elif self.attack_name == "copy_paste":
            from src.attacks.copy_paste_attack import CopyPasteAttacker
            self.attacker_instance = CopyPasteAttacker()

        elif self.attack_name == "del_ins":
            from src.attacks.deletion_insertion_attack import DeletionInsertionAttacker
            self.attacker_instance = DeletionInsertionAttacker(self.args.seed)

        elif self.attack_name in ["drop", "swap"]:
            from src.attacks.perturbation_attack import PerturbationAttacker
            self.attacker_instance = PerturbationAttacker(self.args.seed)

        elif self.attack_name == "typo":
            from src.attacks.typo_attack import TypoAttacker
            self.attacker_instance = TypoAttacker(self.args.seed)

        elif self.attack_name == "sentence_reorder":
            from src.attacks.sentence_reorder_attack import SentenceReorderAttacker
            self.attacker_instance = SentenceReorderAttacker(self.args.seed)

        else:
            raise ValueError(f"❌ 未知或未注册的攻击方法: {self.attack_name}")

        self.models_loaded = True

    def apply_attack(self, wm_text: str, human_text: str) -> str:
        self._lazy_load_model()

        if not wm_text:
            return ""

        if self.attack_name == "dipper":
            return self.attacker_instance.attack_text(
                wm_text,
                self.args.dipper_lex_div,
                self.args.dipper_order_div
            )

        elif self.attack_name == "pegasus":
            return self.attacker_instance.attack_text(
                wm_text,
                self.args.pegasus_num_beams,
                self.args.pegasus_max_drop
            )

        elif self.attack_name == "back_translation":
            return self.attacker_instance.attack_text(wm_text)

        elif self.attack_name == "copy_paste":
            return self.attacker_instance.attack_text(
                wm_text,
                human_text,
                self.args.cp_mix_ratio,
                self.args.cp_mix_mode
            )

        elif self.attack_name == "del_ins":
            return self.attacker_instance.attack_text(
                wm_text,
                human_text,
                self.args.del_drop_prob,
                self.args.del_insert_prob
            )

        elif self.attack_name == "drop":
            return self.attacker_instance.attack_drop(
                wm_text,
                self.args.drop_ratio
            )

        elif self.attack_name == "swap":
            return self.attacker_instance.attack_swap(
                wm_text,
                self.args.swap_ratio
            )

        elif self.attack_name == "typo":
            return self.attacker_instance.attack_text(
                wm_text,
                self.args.typo_ratio
            )

        elif self.attack_name == "sentence_reorder":
            return self.attacker_instance.attack_text(
                wm_text,
                human_text,
                self.args.reorder_ratio,
                self.args.reorder_mode
            )


def main():
    args = parse_args()

    if not os.path.exists(args.target_file):
        print(f"❌ 找不到目标文件: {args.target_file}")
        return

    router = AttackRouter(args)

    save_key = args.attack_name

    if args.attack_name == "dipper":
        if args.dipper_lex_div != 60 or args.dipper_order_div != 60:
            save_key = f"dipper_{args.dipper_lex_div}_{args.dipper_order_div}"

    if args.attack_name == "sentence_reorder":
        if args.reorder_ratio != 0.5 or args.reorder_mode != "shuffle":
            save_key = f"sentence_reorder_{args.reorder_mode}_{args.reorder_ratio}"

    with open(args.target_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_records = []
    need_save = False

    print("\n===================================================")
    print(f"🗡️ 启动统一文本攻击协议: [{args.attack_name.upper()}] (存储字段: {save_key})")
    print("===================================================")

    for line in tqdm(lines, desc=f"Attacking ({save_key})", leave=False):
        if not line.strip():
            continue

        record = json.loads(line.strip())

        if "attacks" not in record:
            record["attacks"] = {}

        wm_text = record.get("wm_text", "")
        human_text = record.get("human_text", record.get("human_reference", ""))

        if not wm_text:
            updated_records.append(record)
            continue

        # 已有结果：正常则跳过；明显异常则移除 attacked_text，避免后续 detection 卡住
        old = record["attacks"].get(save_key)
        if isinstance(old, dict) and "attacked_text" in old:
            bad, reason = is_abnormal_attack(
                old["attacked_text"],
                wm_text,
                args.max_attack_chars,
                args.max_attack_expansion
            )

            if not bad:
                updated_records.append(record)
                continue

            print(
                f"\n[WARN] 删除异常历史攻击输出: "
                f"{save_key}, reason={reason}"
            )

            record["attacks"][save_key] = {
                "error": reason,
                "invalid_attack_output": True
            }
            need_save = True
            updated_records.append(record)
            continue

        # 已经标记过异常，不重复攻击
        if isinstance(old, dict) and old.get("invalid_attack_output"):
            updated_records.append(record)
            continue

        start_time = time.time()

        try:
            attacked_text = router.apply_attack(wm_text, human_text)
            cost_time = time.time() - start_time

            # 新增：攻击结果返回后做简单长度/膨胀检查
            bad, reason = is_abnormal_attack(
                attacked_text,
                wm_text,
                args.max_attack_chars,
                args.max_attack_expansion
            )

            if bad:
                print(
                    f"\n[WARN] 异常攻击输出已跳过: "
                    f"{save_key}, reason={reason}"
                )
                record["attacks"][save_key] = {
                    "error": reason,
                    "attack_time": round(cost_time, 4),
                    "invalid_attack_output": True
                }
            else:
                record["attacks"][save_key] = {
                    "attacked_text": attacked_text,
                    "attack_time": round(cost_time, 4)
                }

            need_save = True

        except Exception as e:
            print(f"\n[!] 严重警告：处理单条数据异常，跳过此条。错误详情: {e}")
            record["attacks"][save_key] = {
                "error": str(e)
            }
            need_save = True

        updated_records.append(record)

    if need_save:
        temp_file = args.target_file + ".tmp"

        with open(temp_file, "w", encoding="utf-8") as f:
            for rec in updated_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        shutil.move(temp_file, args.target_file)
        print(f"✅ [{args.attack_name}] 攻击处理完成并保存 (存储为 {save_key})。")

    else:
        print(f"⚡ 所有数据均已包含有效 [{save_key}] 攻击结果，无需重复执行。")


if __name__ == "__main__":
    main()
