"""SFT 训练脚本（M3）。

用法：

```bash
python train_sft.py --model_path models/Qwen2.5-0.5B \
    --output_dir ckpt/sft --max_samples 5000 --epochs 1
```

数据来源（自动按优先级）：

1. ``data/moss-sft/moss-003-sft-no-tools.jsonl`` 或 ``.jsonl.zip``；
2. ``--smoke`` 走内置样本；
3. 两者都缺失 / 解析失败时，提示 ``data/download.py`` 并仍能跑完一轮 smoke。

设计要点：

- **Loss masking**：借助 :mod:`src.chat.build_labels`，只对 assistant turn 算 NLL。
- **LoRA 注入**：base 冻结，仅 ``lora_A`` / ``lora_B`` 走 AdamW。
- **梯度累积**：在没有大 batch 时等效扩大 batch size。
- **保存**：仅保存 LoRA 旁路参数（``lora_state.pt``），不存 base，占用空间小。
- **日志**：每 ``--log_interval`` 步打印一次 loss 到 stdout / ``--log_dir/tb``。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.chat import build_labels, format_messages
from src.data_utils import load_moss_sft, load_sft_smoke
from src.lora import inject_lora, lora_state_dict
from src.model_utils import (
    DEFAULT_MODEL_PATH,
    detect_device,
    load_tokenizer,
)


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class SFTDataset(Dataset):
    """``{"messages": [...]}`` 列表 → token 后输出 ``input_ids`` / ``labels``。

    使用 ``tokenizer`` + :mod:`src.chat.build_labels` 精确切片。
    """

    def __init__(self, samples: List[dict], tokenizer, max_length: int = 1024) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        msgs = self.samples[idx]["messages"]
        text = format_messages(msgs)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"][0]
        labels = build_labels(input_ids, msgs, tokenizer=self.tokenizer)
        attention_mask = enc["attention_mask"][0]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate(batch: list[dict], pad_id: int) -> dict:
    """把不同长度的样本 pad 到 batch 内最大长度（右侧填充）。"""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = b["input_ids"].size(0)
        pad = max_len - n
        if pad > 0:
            input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
            attn.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        else:
            input_ids.append(b["input_ids"])
            labels.append(b["labels"])
            attn.append(b["attention_mask"])
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
    }


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, log_interval: int, grad_accum: int,
                    scheduler=None, step_offset: int = 0) -> tuple[float, int]:
    """单 epoch。返回 ``(avg_loss, final_step)``。"""
    model.train()
    total_loss = 0.0
    n_steps = 0
    t0 = time.time()
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / grad_accum
        loss.backward()
        total_loss += loss.item() * grad_accum
        n_steps += 1

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(f"  step {step_offset + step + 1:>5d} | loss {loss.item()*grad_accum:.4f} | "
                  f"avg {total_loss/n_steps:.4f} | {elapsed:.1f}s")
    return total_loss / max(n_steps, 1), step_offset + n_steps


def save_lora(model, output_dir: Path, meta: dict | None = None) -> None:
    """保存 LoRA 旁路参数 + 元信息到 ``output_dir/lora_state.pt``。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    if meta:
        state["__meta__"] = meta
    torch.save(state, output_dir / "lora_state.pt")
    print(f"[SFT] LoRA 权重已保存到 {output_dir}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="SFT 训练（Qwen2.5-0.5B + LoRA）")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", type=Path, default=Path("data/moss-sft"))
    parser.add_argument("--output_dir", type=Path, default=Path("ckpt/sft"))
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="使用内置样本快速验证 pipeline")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = detect_device()
    print(f"[SFT] device={device}")

    # 1. 加载数据
    if args.smoke:
        samples = load_sft_smoke()
        print(f"[SFT] SMOKE 模式，使用内置 {len(samples)} 条样本")
    else:
        try:
            samples = load_moss_sft(args.data_dir, split="no-tools", max_samples=args.max_samples)
            print(f"[SFT] 从 {args.data_dir} 加载 {len(samples)} 条 MOSS SFT 样本")
        except FileNotFoundError as e:
            print(f"[SFT] 数据缺失：{e}")
            print("[SFT] 退回 SMOKE 模式（加 --smoke 显式指定）")
            samples = load_sft_smoke()

    # 2. 加载 tokenizer 与模型
    if not args.model_path.exists():
        print(f"[SFT] 模型 {args.model_path} 不存在，--smoke 模式仅写入占位 ckpt")
        save_lora_placeholder(args.output_dir)
        return

    tokenizer = load_tokenizer(args.model_path)
    dataset = SFTDataset(samples, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        num_workers=0,
    )

    # 3. 构造模型
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=dtype, trust_remote_code=True,
    )
    inject_lora(model, target_modules=args.target_modules, r=args.rank, alpha=args.alpha)
    model = model.to(device)

    # 4. optimizer
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=args.lr, weight_decay=0.0)

    # 5. 训练循环
    total_steps = len(loader)
    print(f"[SFT] epochs={args.epochs}, steps/epoch={total_steps}, "
          f"effective batch={args.batch_size * args.grad_accum}")
    for epoch in range(args.epochs):
        avg_loss, _ = train_one_epoch(
            model, loader, optimizer, device,
            log_interval=args.log_interval, grad_accum=args.grad_accum,
        )
        print(f"[SFT] epoch {epoch+1} done, avg_loss={avg_loss:.4f}")

    save_lora(
        model, args.output_dir,
        meta={
            "base_model": str(args.model_path),
            "target_modules": args.target_modules,
            "r": args.rank,
            "alpha": args.alpha,
            "epochs": args.epochs,
            "lr": args.lr,
            "max_samples": len(samples),
        },
    )


def save_lora_placeholder(output_dir: Path) -> None:
    """无模型时仍写出 ``output_dir`` 目录 + ``lora_state.pt`` 占位文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder = {"__placeholder__": True, "note": "no real training, smoke run only"}
    torch.save(placeholder, output_dir / "lora_state.pt")
    print(f"[SFT] 占位 LoRA 权重已保存到 {output_dir}")


if __name__ == "__main__":
    main()
