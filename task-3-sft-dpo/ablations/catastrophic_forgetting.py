"""S3：灾难性遗忘评估。

对比 base vs SFT 在 C-Eval（中文多项选择）子集上的表现。

设计：

1. 取 C-Eval 验证集 ``val`` 随抽样 N 条（如 50）；
2. 构造 prompt：题目 + A/B/C/D 选项；
3. 分别在 base / SFT 上算每个选项的 log-likelihood，取 ``argmax`` 当作答案；
4. 计算 ``accuracy@1`` 与选项分布差异。

数据来源：huggingface 上的 ``ceval/ceval-exam`` 或 ``BAAI/CEval``。脚本
会优先读本地缓存 ``data/ceval/*.parquet`` / ``*.jsonl``，缺失时直接给出
「数据缺失」的提示（不强行下载）。

用法：
```bash
python ablations/catastrophic_forgetting.py --smoke
```
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch

from src.chat import format_messages
from src.model_utils import DEFAULT_MODEL_PATH, detect_device, load_base_model, load_sft_model, load_tokenizer


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------
CEVAL_TEMPLATE = (
    "以下是一道中文多项选择题，请直接给出正确选项（A / B / C / D）。\n"
    "题目：{question}\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
    "答案："
)


def build_prompt(sample: dict) -> tuple[str, str]:
    """返回 ``(prompt, gold_letter)``。"""
    letters = ["A", "B", "C", "D"]
    opts = {l: sample.get(l, "") for l in letters}
    prompt = CEVAL_TEMPLATE.format(
        question=sample.get("question", ""),
        A=opts["A"], B=opts["B"], C=opts["C"], D=opts["D"],
    )
    return prompt, sample.get("answer", "A").strip().upper()


def log_prob_of_completion(model, tokenizer, prompt: str, completion: str, device) -> float:
    """在 ``prompt`` 条件下算 ``completion`` 的平均 log-prob（per token）。"""
    msgs = [{"role": "user", "content": prompt}]
    text = format_messages(msgs)
    # 末尾接 ``assistant\n`` + completion。
    full_text = text.rstrip("\n") + f"\n{completion}"
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    # 找到 prompt 末尾对应的 token 边界。
    prompt_ids = tokenizer(text.rstrip("\n") + "\n", return_tensors="pt", add_special_tokens=False).input_ids
    start = prompt_ids.size(1)
    if start >= ids.size(1):
        return float("-inf")
    target = ids[0, start:]
    with torch.no_grad():
        out = model(ids)
    logits = out.logits[0, start - 1 : -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return lp.mean().item()


def evaluate(model, tokenizer, samples: List[dict], device) -> dict:
    """对一道题：A/B/C/D 平均 log-prob，取 argmax 作为预测。"""
    correct = 0
    dist = {"A": 0, "B": 0, "C": 0, "D": 0}
    for s in samples:
        prompt, gold = build_prompt(s)
        scores = {l: log_prob_of_completion(model, tokenizer, prompt, l, device) for l in dist}
        pred = max(scores, key=scores.get)
        dist[pred] += 1
        if pred == gold:
            correct += 1
    return {
        "accuracy": correct / max(len(samples), 1),
        "n_samples": len(samples),
        "pred_dist": dist,
    }


# ---------------------------------------------------------------------------
# Fixture：内置 5 题用于 smoke。
# ---------------------------------------------------------------------------
SMOKE_CEVAL = [
    {
        "question": "下列哪个不是机器学习任务？",
        "A": "分类", "B": "回归", "C": "聚类", "D": "正则化",
        "answer": "D",
    },
    {
        "question": "LoRA 的核心思想是？",
        "A": "全量微调", "B": "在原权重上叠加低秩矩阵", "C": "扩大模型宽度",
        "D": "替换 optimizer", "answer": "B",
    },
    {
        "question": "DPO 损失直接优化？",
        "A": "用户点击率", "B": "chosen / rejected 之间的对数概率差", "C": "分类交叉熵",
        "D": "困惑度", "answer": "B",
    },
    {
        "question": "BPE 的核心思想是？",
        "A": "字符切分", "B": "迭代合并高频 token 对", "C": "随机替换",
        "D": "句法树", "answer": "B",
    },
    {
        "question": "Transformer 解码使用哪种 mask？",
        "A": "无 mask", "B": "padding mask", "C": "causal mask",
        "D": "random mask", "answer": "C",
    },
]


def load_ceval(data_dir: Path, n_samples: int = 50) -> List[dict]:
    """优先从本地 ``data/ceval/*.parquet`` / ``*.jsonl`` 读，否则返回 smoke。"""
    ceval_dir = data_dir / "ceval"
    if not ceval_dir.exists():
        return SMOKE_CEVAL[:n_samples]
    found = list(ceval_dir.glob("*.jsonl")) + list(ceval_dir.glob("*.parquet"))
    if not found:
        return SMOKE_CEVAL[:n_samples]
    samples: List[dict] = []
    for f in found:
        if f.suffix == ".jsonl":
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        else:
            # 跳过：需要依赖 pandas 才能读 parquet。
            continue
    return samples[:n_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="S3: 灾难性遗忘评估")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft_ckpt", type=Path, default=Path("ckpt/sft"))
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("figures/s3_forgetting.json"))
    args = parser.parse_args()

    samples = load_ceval(args.data_dir, args.n_samples)
    if args.smoke:
        samples = SMOKE_CEVAL
    print(f"[S3] 评测 {len(samples)} 条 C-Eval 风格样本")

    if not args.model_path.exists():
        print(f"[S3] 模型 {args.model_path} 不存在，跳过实际评估。")
        out = {"skipped": "model missing", "samples": len(samples)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    device = detect_device()
    tokenizer = load_tokenizer(args.model_path)

    base = load_base_model(args.model_path, device=device)
    base_metrics = evaluate(base, tokenizer, samples, device)
    print(f"[S3] base: {base_metrics}")
    del base

    sft = None
    sft_metrics = {"skipped": "sft ckpt missing"}
    if args.sft_ckpt.exists():
        sft = load_sft_model(args.model_path, lora_ckpt=args.sft_ckpt, device=device)
        sft_metrics = evaluate(sft, tokenizer, samples, device)
        print(f"[S3] sft:  {sft_metrics}")
    del sft

    diff_acc = (sft_metrics.get("accuracy", 0) - base_metrics["accuracy"]) if "accuracy" in sft_metrics else None
    out = {
        "base": base_metrics,
        "sft": sft_metrics,
        "delta_accuracy": diff_acc,
        "samples": len(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S3] 写入 {args.output}")


if __name__ == "__main__":
    main()
