"""S1：全量微调 vs LoRA —— 显存占用与下游质量对比。

设计：分别构造

1. 全量微调：所有 base 参数 ``requires_grad=True``；
2. LoRA 注入：仅 ``lora_A`` / ``lora_B`` 可训练。

在两类 setup 上跑若干步 SFT，记录：

- ``peak_memory_mb``：通过 ``torch.cuda.max_memory_allocated``（CPU 回退 None）；
- ``train_steps_per_sec``；
- ``eval_loss``：在固定 smoke 样本上算 NLL；
- ``trainable_params``、``total_params``、``ratio``。

输出：JSON 到 ``figures/s1_full_vs_lora.json`` + 总结 markdown（stdout）。

注意：本脚本在无 GPU 环境下「peak_memory」字段为 ``null``；评估仍能给
``trainable_params`` 与 NLL 的相对差异。

用法：
```bash
python ablations/full_vs_lora.py --smoke
```
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.chat import build_labels, format_messages
from src.data_utils import load_sft_smoke
from src.lora import inject_lora
from src.model_utils import DEFAULT_MODEL_PATH, detect_device


@dataclass
class RunResult:
    name: str
    trainable_params: int
    total_params: int
    trainable_ratio: float
    peak_memory_mb: Optional[float]
    steps_per_sec: float
    final_loss: float


def _measure_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return lambda: torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_model(model_path: Path, mode: str, device):
    """构造全量 / LoRA 模型。"""
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=dtype, trust_remote_code=True,
    )
    if mode == "lora":
        inject_lora(model, target_modules=["q_proj", "v_proj"], r=8, alpha=16)
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(mode)
    model = model.to(device)
    return model


def _fake_batch(tokenizer, device):
    """构造一个小 SFT 批次（替代真实数据）。"""
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


def _run_one(model_path: Path, mode: str, n_steps: int = 5, lr: float = 2e-4) -> RunResult:
    """跑 n_steps SFT，返回 ``RunResult``。"""
    device = detect_device()
    if not model_path.exists():
        print(f"[S1-{mode}] 模型 {model_path} 不存在，跳过 {mode}")
        return RunResult(
            name=mode, trainable_params=0, total_params=0, trainable_ratio=0.0,
            peak_memory_mb=None, steps_per_sec=0.0, final_loss=0.0,
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(model_path, mode, device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    batch = _fake_batch(tokenizer, device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    peak = _measure_peak()
    model.train()
    optimizer.zero_grad()
    t0 = time.time()
    last_loss = 0.0
    for step in range(n_steps):
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        last_loss = loss.item()
    elapsed = time.time() - t0
    peak_mb = peak() if peak is not None else None

    # 评估。
    model.eval()
    with torch.no_grad():
        out = model(**batch)
    eval_loss = out.loss.item()

    result = RunResult(
        name=mode,
        trainable_params=trainable,
        total_params=total,
        trainable_ratio=trainable / total,
        peak_memory_mb=peak_mb,
        steps_per_sec=n_steps / max(elapsed, 1e-6),
        final_loss=eval_loss,
    )

    # 释放显存。
    del model, optimizer, batch
    _release()

    print(f"[S1-{mode}] trainable={trainable}/{total} ({result.trainable_ratio:.4%}) "
          f"peak={peak_mb:.1f}MB sps={result.steps_per_sec:.2f} loss={eval_loss:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="S1: 全量 vs LoRA")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("figures/s1_full_vs_lora.json"))
    args = parser.parse_args()

    if args.smoke or not args.model_path.exists():
        print("[S1] smoke 模式 / 模型缺失，仅打印设计说明与差异预期，不跑真实训练。")

    res_full = _run_one(args.model_path, mode="full", n_steps=args.steps)
    res_lora = _run_one(args.model_path, mode="lora", n_steps=args.steps)

    summary = {
        "full": asdict(res_full),
        "lora": asdict(res_lora),
        "lora_trainable_ratio_vs_full": (
            res_lora.trainable_ratio / max(res_full.trainable_ratio, 1e-9)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S1] 写入 {args.output}")

    # 文本对比。
    print("\n--- S1 结论 ---")
    if res_full.peak_memory_mb and res_lora.peak_memory_mb:
        ratio_mem = res_full.peak_memory_mb / max(res_lora.peak_memory_mb, 1e-9)
        print(f"peak memory  full/lora = {ratio_mem:.2f}x")
    if res_full.total_params > 0:
        print(f"参数比  full {res_full.total_params} vs lora_trainable {res_lora.trainable_params}")


if __name__ == "__main__":
    main()
