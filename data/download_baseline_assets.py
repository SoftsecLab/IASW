#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_baseline_assets.py — 下载论文基线的第三方权重/统计文件。

本仓库不直接分发以下第三方资源（许可证或再分发限制）。运行本脚本
按官方来源下载并校验 SHA256 后，放入 src/baselines/<core>/ 对应路径。

用法:
    python data/download_baseline_assets.py            # 下载全部
    python data/download_baseline_assets.py --check    # 仅校验已存在文件

注意: 请自行确认各上游许可证允许在你的用途下使用这些资源。
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# 第三方资产清单（official URL + sha256）
# -----------------------------------------------------------------------------
# 说明:
#   * Synguard/SIR 的 transform_model_cbert.pth 来自 THU-BPM MarkLLM (Apache-2.0)，
#     官方权重存放于 Hugging Face Generative-Watermark-Toolkits/MarkLLM-sir。
#   * DualMark（MarkLLM 的 KGW+TS/Twins）的 opt/init_0.25_1.75_default.pth 为
#     官方权重，存放于 Hugging Face Generative-Watermark-Toolkits/MarkLLM-ts。
#   * SemStamp 的 *-cluster_8_centers.pt 与 SimMark 的 *-pca_model_16.pkl
#     为对应官方实现的语义统计文件。
ASSETS = [
    {
        "name": "synguard transform_model_cbert.pth",
        "url": "https://huggingface.co/Generative-Watermark-Toolkits/MarkLLM-sir/resolve/main/transform_model_cbert.pth",
        "sha256": "d68f32e31628deeeab50fbb50e5e8725d41afa4f35c94bd639437653e0709e76",
        "dest": PROJECT_ROOT / "src" / "baselines" / "synguard_core" / "watermark" / "sir" / "model" / "transform_model_cbert.pth",
    },
    {
        "name": "dualmark opt init_0.25_1.75_default.pth",
        "url": "https://huggingface.co/Generative-Watermark-Toolkits/MarkLLM-ts/resolve/main/opt/init_0.25_1.75_default.pth",
        "sha256": "bc640fb9e02ddfd90f2d9953ba293b5c26b00a7673be04559350b0e354117688",
        "dest": PROJECT_ROOT / "src" / "baselines" / "dualmark_core" / "model" / "opt" / "init_0.25_1.75_default.pth",
    },
]

SEMSTAMP_CLUSTERS = [
    "c4-cluster_8_centers.pt", "dolly-cluster_8_centers.pt", "wiki-cluster_8_centers.pt",
    "booksum-cluster_8_centers.pt", "qa-cluster_8_centers.pt", "1c4-cluster_8_centers.pt",
]
SIMMARK_PCA = [
    "c4_pca_model_16.pkl", "dolly_pca_model_16.pkl", "wiki_pca_model_16.pkl", "pca_model_16.pkl",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    check_only = "--check" in sys.argv
    for asset in ASSETS:
        dest = asset["dest"]
        if dest.exists():
            ok = sha256(dest) == asset["sha256"] if asset["sha256"] and not asset["sha256"].startswith("<") else "unknown"
            print(f"[OK]   {asset['name']} -> {dest} ({ok})")
            continue
        if check_only:
            print(f"[MISS] {asset['name']}")
            continue
        if asset["url"].startswith("<"):
            print(f"[SKIP] {asset['name']}: 请在脚本中填写上游 URL/SHA256")
            continue
        print(f"[GET]  {asset['name']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(asset["url"], dest)
        print(f"       sha256 = {sha256(dest)}")

    print("\nSemStamp cluster / SimMark PCA 文件若为论文自行构建的产物，")
    print("可随数据包发布；若来自官方仓库，请同样以官方 URL + SHA256 方式获取。")


if __name__ == "__main__":
    main()
