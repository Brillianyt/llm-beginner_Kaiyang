"""S2：LoRA rank 消融（r ∈ {4, 8, 16, 32}）。

在相同 SFT 数据 + 相同超参下，对 4 种 rank 蒸馏若干步 NLL，输出
``figures/s2_rank_ablation.json`` + markdown 表格。

评测：在固定 smoke 样本上算 NLL，**不**做生成质量评测（生成质量需要
人工判别 / GPT-4 评分，超出 ablation 范围）。

用法：
```bash
python ablations/rank_ablation.py --smoke
```
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
from torch.optim import AdamW

from src.chat import build_labels, format_messages
from src.data_utils import load_sft_smoke
from src.lora import inject_lora
from src.model_utils import DEFAULT_MODEL_PATH, detect_device


@dataclass
class RankResult:
    rank: int
    alpha: int
    trainable_params: int
    total_params: int
    peak_memory_mb: float | None
    final_loss: float
    steps_per_sec: float


def _build_lora(model_path: Path, r: int, alpha: int, device):
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=dtype, trust_remote_code=True,
    )
    inject_lora(model, target_modules=["q_proj", "v_proj"], r=r, alpha=alpha)
    return model.to(device)


def _fake_batch(tokenizer, device):
    samples = load_sft_smoke()
    msgs = samples[0]["messages"]
    text = format_messages(msgs)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    ids = enc["input_ids"][0]
    labels = build_labels(ids, msgs, tokenizer=tokenizer)
    return {
        "input_ids": ids.unsqueeze(0).to(device),
        "labels": labels.unsqueeze(0).to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


def _run_rank(model_path: Path, r: int, alpha: int, n_steps: int, lr: float = 2e-4) -> RankResult:
    if not model_path.exists():
        # 写空数据并返回。
        return RankResult(rank=r, alpha=alpha, trainable_params=0, total_params=0,
                          peak_memory_mb=None, final_loss=0.0, steps_per_sec=0.0)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_lora(model_path, r, alpha, detect_device())
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    batch = _fake_batch(tokenizer, detect_device())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad()
    t0 = time.time()
    for _ in range(n_steps):
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    elapsed = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None

    model.eval()
    with torch.no_grad():
        eval_loss = model(**batch).loss.item()

    del model, optimizer, batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[S2] r={r:>3d} alpha={alpha:>3d} trainable={trainable:>9d} "
          f"peak={peak_mb} loss={eval_loss:.4f} sps={n_steps/elapsed:.2f}")
    return RankResult(rank=r, alpha=alpha, trainable_params=trainable, total_params=total,
                      peak_memory_mb=peak_mb, final_loss=eval_loss, steps_per_sec=n_steps/elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="S2: LoRA rank 消融")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--alphas", nargs="+", type=int, default=None,
                        help="默认与 ranks 一一对应；若显式传入，长度应等于 ranks")
    parser.add_argument("--output", type=Path, default=Path("figures/s2_rank_ablation.json"))
    args = parser.parse_args()

    alphas = args.alphas or args.ranks
    if len(alphas) != len(args.ranks):
        raise ValueError("alphas 与 ranks 长度不一致")

    if args.smoke or not args.model_path.exists():
        print("[S2] smoke 模式 / 模型缺失：仅构造结果表头，不跑真实训练。")

    results: List[RankResult] = []
    for r, a in zip(args.ranks, alphas):
        res = _run_rank(args.model_path, r, a, n_steps=args.steps)
        results.append(res)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[S2] 写入 {args.output}")

    # Markdown 表格。
    print("\n| rank | alpha | trainable_params | peak_MB | final_loss | steps/s |")
    print("|------|-------|------------------|---------|------------|---------|")
    for r in results:
        print(f"| {r.rank} | {r.alpha} | {r.trainable_params} | {r.peak_memory_mb} | "
              f"{r.final_loss:.4f} | {r.steps_per_sec:.2f} |")


if __name__ == "__main__":
    main()
