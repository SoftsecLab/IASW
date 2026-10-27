#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated IASW analysis entry.

Generated from the current analysis scripts without changing their scientific
calculations. Each subcommand executes the original implementation in an
isolated function scope so duplicate helper names do not collide.
"""
from __future__ import annotations

import sys

def _run_analyze(argv):
    """Run preserved implementation from analyze_kgw_seed_sweep.py[analyze]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_seed_sweep.py[analyze]"] + list(argv)

        import argparse
        import json
        from pathlib import Path
        import pandas as pd



        def parse_args():

            parser = argparse.ArgumentParser()

            parser.add_argument(
                "--root",
                required=True
            )

            parser.add_argument(
                "--output_dir",
                default="results/analysis/kgw_generation/seed_sweep/legacy_outputs"
            )

            return parser.parse_args()



        def load_json(path):

            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)



        def extract(path):

            data = load_json(path)


            def mean_value(key):

                try:
                    return data[key]["mean"]
                except:
                    return None


            return {

                "z_raw":
                    mean_value(
                        "z_no_repeat_filter"
                    ),

                "z_repeat":
                    mean_value(
                        "z_repeat_filter"
                    ),

                "green_raw":
                    mean_value(
                        "green_fraction_no_repeat_filter"
                    ),

                "green_repeat":
                    mean_value(
                        "green_fraction_repeat_filter"
                    ),

                "tokens_raw":
                    mean_value(
                        "num_tokens_no_repeat_filter"
                    ),

                "tokens_repeat":
                    mean_value(
                        "num_tokens_repeat_filter"
                    ),

                "samples":
                    data.get(
                        "num_detected_records",
                        None
                    )
            }



        def get_info(path):

            """
            path:

            root/
              model/
                seed_xxx/
                  records/
                      xxx_summary.json
            """

            seed = None

            for p in path.parts:

                if p.startswith("seed_"):
                    seed = p.replace(
                        "seed_",
                        ""
                    )


            model = path.parent.parent.parent.name


            if "kgw_control" in path.name:

                method="KGW"

            elif "dual_kgw_random_k1" in path.name:

                method="Random-K1"

            else:

                method="Unknown"



            return model, seed, method



        def collect(root):

            rows=[]


            for file in root.rglob("*summary.json"):

                if (
                    "kgw_control" not in file.name
                    and
                    "dual_kgw_random_k1" not in file.name
                ):
                    continue


                model,seed,method=get_info(file)


                row={

                    "model":model,

                    "seed":seed,

                    "method":method,

                    "file":str(file)

                }


                row.update(
                    extract(file)
                )


                rows.append(row)



            return pd.DataFrame(rows)



        def make_delta(df):


            kgw=df[
                df.method=="KGW"
            ].copy()


            rand=df[
                df.method=="Random-K1"
            ].copy()



            keys=[
                "model",
                "seed"
            ]


            merge=pd.merge(
                kgw,
                rand,
                on=keys,
                suffixes=(
                    "_kgw",
                    "_random"
                )
            )


            merge["delta_z_raw"]=(
                merge.z_raw_random
                -
                merge.z_raw_kgw
            )


            merge["delta_z_repeat"]=(
                merge.z_repeat_random
                -
                merge.z_repeat_kgw
            )


            merge["delta_green_repeat"]=(
                merge.green_repeat_random
                -
                merge.green_repeat_kgw
            )


            merge["delta_tokens_repeat"]=(
                merge.tokens_repeat_random
                -
                merge.tokens_repeat_kgw
            )


            keep=[

                "model",
                "seed",

                "z_raw_kgw",
                "z_raw_random",
                "delta_z_raw",

                "z_repeat_kgw",
                "z_repeat_random",
                "delta_z_repeat",

                "green_repeat_kgw",
                "green_repeat_random",
                "delta_green_repeat",

                "tokens_repeat_kgw",
                "tokens_repeat_random",
                "delta_tokens_repeat"

            ]


            return merge[keep]



        def make_backbone_summary(delta):


            summary=(

                delta
                .groupby("model")
                .agg(

                    seeds=("seed","count"),

                    mean_delta_z_repeat=
                    (
                        "delta_z_repeat",
                        "mean"
                    ),

                    std_delta_z_repeat=
                    (
                        "delta_z_repeat",
                        "std"
                    ),

                    min_delta_z_repeat=
                    (
                        "delta_z_repeat",
                        "min"
                    ),

                    max_delta_z_repeat=
                    (
                        "delta_z_repeat",
                        "max"
                    ),


                    mean_delta_green=
                    (
                        "delta_green_repeat",
                        "mean"
                    ),


                    mean_delta_tokens=
                    (
                        "delta_tokens_repeat",
                        "mean"
                    )

                )

                .reset_index()

            )


            return summary.sort_values(
                "mean_delta_z_repeat"
            )



        def main():


            args=parse_args()


            root=Path(args.root)

            out=Path(
                args.output_dir
            )

            out.mkdir(
                exist_ok=True
            )


            print(
                "Scanning:",
                root
            )


            df=collect(root)


            if len(df)==0:

                raise RuntimeError(
                    "No summary json found"
                )


            df=df.sort_values(
                [
                    "model",
                    "seed",
                    "method"
                ]
            )


            raw_file=out/"results/analysis/kgw_generation/seed_sweep/kgw_seed_summary.csv"

            df.to_csv(
                raw_file,
                index=False,
                encoding="utf-8-sig"
            )


            delta=make_delta(df)


            delta_file=out/"kgw_seed_delta.csv"

            delta.to_csv(
                delta_file,
                index=False,
                encoding="utf-8-sig"
            )


            backbone=make_backbone_summary(
                delta
            )


            backbone_file=(
                out/
                "kgw_backbone_summary.csv"
            )


            backbone.to_csv(
                backbone_file,
                index=False,
                encoding="utf-8-sig"
            )



            print("\nGenerated:")

            print(
                raw_file
            )

            print(
                delta_file
            )

            print(
                backbone_file
            )


            print("\nBackbone summary:")
            print(
                backbone.to_string(
                    index=False
                )
            )



        if __name__=="__main__":

            main()
        main()
    finally:
        sys.argv = old_argv

def _run_summarize(argv):
    """Run preserved implementation from analyze_kgw_seed_sweep.py[summarize]."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["analyze_kgw_seed_sweep.py[summarize]"] + list(argv)

        """
        Summarize KGW vs Random-K1 seed sweep results.

        Input:
        results/diagnostics/generation/kgw_randomk1_seed_sweep/c4/

        Expected structure:

        root/
         ├── Model-A/
         │    ├── seed_xxx/
         │    │     └── records/
         │    │          ├── kgw_control.kgw_detection_summary.json
         │    │          └── dual_kgw_random_k1.kgw_detection_summary.json

        Output:
        one flat CSV table.

        """

        import argparse
        import json
        from pathlib import Path
        import pandas as pd


        def parse_args():
            parser = argparse.ArgumentParser()

            parser.add_argument(
                "--root",
                required=True,
                help="experiment root directory"
            )

            parser.add_argument(
                "--output",
                default="results/analysis/kgw_generation/seed_sweep/kgw_seed_summary.csv"
            )

            return parser.parse_args()


        def load_json(path):

            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)



        def extract_summary(path):

            data = load_json(path)

            def get(section, key):

                try:
                    return data[section]["mean"]
                except Exception:
                    return None


            return {

                # raw KGW
                "z_raw":
                    get(
                        "z_no_repeat_filter",
                        "mean"
                    ),

                "z_repeat":
                    get(
                        "z_repeat_filter",
                        "mean"
                    ),

                "green_fraction_raw":
                    get(
                        "green_fraction_no_repeat_filter",
                        "mean"
                    ),

                "green_fraction_repeat":
                    get(
                        "green_fraction_repeat_filter",
                        "mean"
                    ),

                "tokens_raw":
                    get(
                        "num_tokens_no_repeat_filter",
                        "mean"
                    ),

                "tokens_repeat":
                    get(
                        "num_tokens_repeat_filter",
                        "mean"
                    ),

                "num_samples":
                    data.get(
                        "num_detected_records",
                        None
                    )
            }



        def infer_method(filename):

            if "kgw_control" in filename:
                return "KGW"

            if "dual_kgw_random_k1" in filename:
                return "Random-K1"

            return "Unknown"



        def main():

            args = parse_args()

            root = Path(args.root)

            records = []


            for json_file in root.rglob("*summary.json"):

                name = json_file.name

                if (
                    "kgw_control" not in name
                    and
                    "dual_kgw_random_k1" not in name
                ):
                    continue


                # 
                # path:
                #
                # root/model/seed_xxx/records/file.json
                #

                parts = json_file.parts


                model = None
                seed = None


                for p in parts:

                    if p.startswith("seed_"):
                        seed = p.replace(
                            "seed_",
                            ""
                        )

                # 找 records 上一级
                try:
                    records_dir = json_file.parent
                    model = records_dir.parent.parent.name

                except:
                    model="unknown"


                row = {

                    "model":
                        model,

                    "seed":
                        seed,

                    "method":
                        infer_method(name),

                    "file":
                        str(json_file)

                }


                row.update(
                    extract_summary(json_file)
                )


                records.append(row)



            df = pd.DataFrame(records)


            if len(df)==0:

                raise RuntimeError(
                    "No summary json found"
                )


            # 排序

            df = df.sort_values(
                [
                    "model",
                    "seed",
                    "method"
                ]
            )


            # 保存

            df.to_csv(
                args.output,
                index=False,
                encoding="utf-8-sig"
            )


            print(
                "\nSaved:",
                args.output
            )

            print(
                df.to_string(index=False)
            )


        if __name__=="__main__":

            main()
        main()
    finally:
        sys.argv = old_argv

COMMANDS = {"analyze": _run_analyze, "summarize": _run_summarize}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage:\n"
            "  analyze_kgw_seed_sweep.py analyze <old analyze args>\n"
            "  analyze_kgw_seed_sweep.py summarize <old summarize args>\n"
            "\nEach subcommand preserves the corresponding old CLI."
        )
        return
    command = sys.argv[1]
    fn = COMMANDS.get(command)
    if fn is None:
        raise SystemExit(f"Unknown subcommand: {command}. Choose from: {', '.join(COMMANDS)}")
    fn(sys.argv[2:])

if __name__ == "__main__":
    main()
