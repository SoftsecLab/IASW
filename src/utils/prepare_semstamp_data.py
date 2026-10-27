import os
import json
import argparse
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"❌ 找不到文件: {args.input_file}")
        return

    extracted_data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 自动适配多种字段名
            text = data.get("text", data.get("response", data.get("completion", data.get("content", ""))))
            if text and text.strip():
                extracted_data.append({"text": text.strip()})

    ds = Dataset.from_list(extracted_data)
    ds.save_to_disk(args.output_dir)
    print(f"✅ 已转换 {len(ds)} 条数据至 HF 格式: {args.output_dir}")

if __name__ == "__main__":
    main()