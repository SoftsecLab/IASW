#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path


def build_mapping(
    vocab_size: int,
    scale_dimension: int,
    seed: int,
):
    rng = random.Random(seed)

    return [
        rng.randint(0, scale_dimension - 1)
        for _ in range(vocab_size)
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "src/baselines/synguard_core/"
            "watermark/sir/mapping"
        ),
    )

    parser.add_argument(
        "--vocab_sizes",
        type=int,
        nargs="+",
        default=[
            128256,
            152064,
        ],
    )

    parser.add_argument(
        "--scale_dimension",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("SynGuard / SIR mapping builder")
    print("=" * 80)
    print("output_dir      =", out_dir)
    print("scale_dimension =", args.scale_dimension)
    print("seed            =", args.seed)
    print("vocab_sizes     =", args.vocab_sizes)
    print()

    for vocab_size in args.vocab_sizes:
        path = (
            out_dir
            / (
                f"{args.scale_dimension}_"
                f"mapping_{vocab_size}.json"
            )
        )

        if path.exists() and not args.overwrite:
            print(
                f"[SKIP] {path.name} already exists"
            )
            continue

        mapping = build_mapping(
            vocab_size=vocab_size,
            scale_dimension=args.scale_dimension,
            seed=args.seed,
        )

        assert len(mapping) == vocab_size
        assert min(mapping) >= 0
        assert max(mapping) < args.scale_dimension

        path.write_text(
            json.dumps(mapping),
            encoding="utf-8",
        )

        print(
            f"[OK] {path.name}: "
            f"len={len(mapping)}, "
            f"range=[{min(mapping)}, "
            f"{max(mapping)}]"
        )


if __name__ == "__main__":
    main()