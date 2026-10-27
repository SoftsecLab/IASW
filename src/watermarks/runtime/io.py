# src/watermarks/runtime/io.py
# -*- coding: utf-8 -*-

import json
import os
import shutil


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl_atomic(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_file = path + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.move(temp_file, path)


def row_key(row: dict):
    idx = row.get("idx", None)
    return idx if idx is not None else row.get("prompt", "")


def build_completed_generation_cache(existing_rows):
    """
    仅缓存那些已经成功生成 wm_text 的条目
    """
    cache = {}
    valid_count = 0
    for item in existing_rows:
        if item.get("wm_text") and item["wm_text"].strip() and "error" not in item:
            cache[row_key(item)] = item
            valid_count += 1
    return cache, valid_count


def merge_generation_progress(data_rows, completed_cache):
    merged = []
    for row in data_rows:
        key = row_key(row)
        if key in completed_cache:
            merged.append(completed_cache[key])
        else:
            merged.append(row)
    return merged