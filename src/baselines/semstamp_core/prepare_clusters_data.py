import os
import json
from datasets import Dataset

def extract_and_convert(input_file, output_dir, skip_rows=10000, take_rows=20000, is_qa=False):
    """
    从 JSONL 中提取数据并转化为 HF Dataset 格式
    """
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
        return

    extracted_data = []
    
    print(f"[*] 正在处理: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # 1. 跳过前 skip_rows 条数据
            if i < skip_rows:
                continue
            
            # 2. 达到抽取数量限制则停止
            if i >= skip_rows + take_rows:
                break
                
            data = json.loads(line)
            
            # 3. 提取文本内容 (请根据你 jsonl 的实际 key 进行修改)
            if is_qa:
                # 💡 QA 数据集：尽量只提取答案部分进行聚类，这样生成的语义中心更准确
                # 假设你的答案字段叫 "response"、"answer" 或 "completion"
                text = data.get("response", data.get("answer", data.get("completion", "")))
                
                # 如果没有单独的答案字段，退化为取全部文本
                if not text:
                    text = data.get("text", "")
            else:
                # Wiki 通用文本提取
                text = data.get("text", data.get("content", ""))
                
            # 清洗空文本
            if text and text.strip():
                # 注意：SemStamp 底层的 K-Means 脚本默认读取 "text" 列
                extracted_data.append({"text": text.strip()})

    # 4. 转化为 HuggingFace Dataset 并保存到磁盘
    ds = Dataset.from_list(extracted_data)
    os.makedirs(output_dir, exist_ok=True)
    ds.save_to_disk(output_dir)
    print(f"✅ 成功提取 {len(ds)} 条数据并保存至: {output_dir}\n")


if __name__ == "__main__":
    # 目标输出根目录
    TARGET_DIR = "src/baselines/semstamp_core"
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 1. 处理 Wiki 数据集
    extract_and_convert(
        input_file="data/raw_wiki_50000.jsonl",
        output_dir=os.path.join(TARGET_DIR, "wiki-train"),
        skip_rows=10000,
        take_rows=20000,  # 2万条用来跑 K-Means 已经绰绰有余了
        is_qa=False
    )

    # 2. 处理 QA 数据集
    extract_and_convert(
        input_file="data/raw_qa_domain_50000.jsonl",
        output_dir=os.path.join(TARGET_DIR, "qa-train"),
        skip_rows=10000,
        take_rows=20000,
        is_qa=True
    )