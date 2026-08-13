"""DPO 训练脚本（M4）。

DPO 损失（论文 DPO: Your Language Model is Secretly a Reward Model）：

    L_DPO = -E_(x, yw, yl) [ log σ( β * ( Δ_θ ) ) ]
    Δ_θ = log π_θ(yw | x) - log π_θ(yl | x)
        - log π_ref(yw | x) + log π_ref(yl | x)

其中 ``yw`` / ``yl`` 是 chosen / rejected。

实现要点：

1. **reference model**：与 policy 同架构、加载 SFT 权重；冻结，仅 forward；
2. **chosen / rejected 各 forward 一次**：一个 batch 内 4 次 forward
   （policy×2 + ref×2）；不要 batch 合并避免 ``-100`` mask 影响 log_softmax；
3. **DPO log-prob**：按所有非 ``-100`` token 取平均（与 TRL 实现一致），
   也可以按 sum（这里给出平均 + 注释解释）。

用法：

```bash
python train_dpo.py --model_path models/Qwen2.5-0.5B \
    --sft_ckpt ckpt/sft --output_dir ckpt/dpo \
    --max_samples 2000 --epochs 1
```
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.chat import format_messages
from src.data_utils import load_dpo, load_dpo_smoke
from src.lora import inject_lora, lora_state_dict, load_lora_state_dict
from src.model_utils import (
    DEFAULT_MODEL_PATH,
    detect_device,
    load_reference_model,
    load_tokenizer,
)


# ---------------------------------------------------------------------------
# DPO Dataset
# ---------------------------------------------------------------------------
class DPODataset(Dataset):
    """``[{"prompt": ..., "chosen": ..., "rejected": ...}, ...]``。

    这里 ``prompt`` 为单字符串或消息列表；``chosen`` / ``rejected`` 单字符串。
    每条样本产出 chosen 与 rejected 的完整 prompt-completion 输入。
    """

    def __init__(self, samples: List[dict], tokenizer, max_length: int = 1024) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def _build_prompt(self, prompt) -> List[dict]:
        if isinstance(prompt, list):
            return list(prompt)
        return [{"role": "user", "content": prompt}]

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        prompt_msgs = self._build_prompt(s["prompt"])
        chosen = s["chosen"]
        rejected = s["rejected"]

        # 拼接完整对话（chosen / rejected 各自生成一个完整序列）。
        chosen_msgs = prompt_msgs + [{"role": "assistant", "content": chosen}]
        rejected_msgs = prompt_msgs + [{"role": "assistant", "content": rejected}]
        chosen_text = format_messages(chosen_msgs)
        rejected_text = format_messages(rejected_msgs)

        # 同时拿到 prompt 单独编码，用长度切片得到 completion 段。
        prompt_text = format_messages(prompt_msgs)
        if not prompt_text.endswith("assistant\n"):
            # 确保 prompt 末尾接 assistant 头。
            prompt_text = prompt_text.rstrip("\n") + f"\n{format_messages([{'role': 'assistant', 'content': ''}])}"

        # prompt 段取 prompt_text，剩余为 completion 段。
        pid = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        cid = self.tokenizer(chosen_text, add_special_tokens=False).input_ids
        rid = self.tokenizer(rejected_text, add_special_tokens=False).input_ids

        # 截断与对齐。
        cid = cid[: self.max_length]
        rid = rid[: self.max_length]
        prompt_len = min(len(pid), len(cid))

        c_labels = [-100] * prompt_len + cid[prompt_len:]
        r_labels = [-100] * prompt_len + rid[prompt_len:]

        return {
            "chosen_input_ids": torch.tensor(cid, dtype=torch.long),
            "chosen_labels": torch.tensor(c_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rid, dtype=torch.long),
            "rejected_labels": torch.tensor(r_labels, dtype=torch.long),
        }


def dpo_collate(batch: list[dict], pad_id: int) -> dict:
    """DPO 专用 collate：chosen / rejected 分别 padding。"""
    def _pad(seqs, label_value):
        max_len = max(s.size(0) for s in seqs)
        padded, labels = [], []
        for s, l in zip(seqs, label_value):
            pad = max_len - s.size(0)
            if pad > 0:
                padded.append(torch.cat([s, torch.full((pad,), pad_id, dtype=torch.long)]))
                labels.append(torch.cat([l, torch.full((pad,), -100, dtype=torch.long)]))
            else:
                padded.append(s)
                labels.append(l)
        return torch.stack(padded), torch.stack(labels)

    c_ids, c_lbl = _pad([b["chosen_input_ids"] for b in batch],
                        [b["chosen_labels"] for b in batch])
    r_ids, r_lbl = _pad([b["rejected_input_ids"] for b in batch],
                        [b["rejected_labels"] for b in batch])
    return {
        "chosen_input_ids": c_ids, "chosen_labels": c_lbl,
        "rejected_input_ids": r_ids, "rejected_labels": r_lbl,
    }


# ---------------------------------------------------------------------------
# DPO 损失
# ---------------------------------------------------------------------------
def _gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """对每个样本取非 ``-100`` 位置 token 平均 log-prob。

    log-prob 在词表维度做 log_softmax，再按 label 索引采样；最后对所有
    active 位置取平均（这是 TRL / 官方实现的常用做法，与 sum 相比只差一个
    固定常数 ``1/T``，对训练动态影响微弱）。
    """
    # logits: (B, T, V)；labels: (B, T)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # 把 ``-100`` 的位置替换成 0（gather 后会再 mask 掉）。
    safe_labels = labels.clamp(min=0)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    mask = (labels != -100).float()
    summed = (gathered * mask).sum(dim=-1)
    counts = mask.sum(dim=-1).clamp(min=1.0)
    return summed / counts


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DPO 损失 + reward margin。

    L = -E[ log σ( β * ( Δθ ) ) ]
    Δθ = (log π(yw|x) - log π(yl|x)) - (log π_ref(yw|x) - log π_ref(yl|x))

    Returns:
        (loss, margin)。margin 可用于监控训练是否拉开 chosen / rejected。
    """
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    margin = (pi_logratios - ref_logratios).mean().detach()
    return loss, margin


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def train_one_epoch(
    policy, ref, loader, optimizer, device,
    log_interval: int, grad_accum: int, beta: float,
    log_path: Path | None = None,
    step_offset: int = 0,
) -> tuple[float, float, int]:
    """单 epoch；返回 ``(avg_loss, avg_margin, final_step)``。"""
    policy.train()
    ref.eval()
    total_loss, total_margin, n_steps = 0.0, 0.0, 0
    log_buf: list[str] = []
    t0 = time.time()
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        cid = batch["chosen_input_ids"].to(device)
        clb = batch["chosen_labels"].to(device)
        rid = batch["rejected_input_ids"].to(device)
        rlb = batch["rejected_labels"].to(device)

        # policy forward：chosen + rejected
        out_c = policy(input_ids=cid, labels=None)
        out_r = policy(input_ids=rid, labels=None)
        pi_c = _gather_logprobs(out_c.logits, clb)
        pi_r = _gather_logprobs(out_r.logits, rlb)

        # ref forward：chosen + rejected（无梯度）
        with torch.no_grad():
            ref_c = ref(input_ids=cid, labels=None).logits
            ref_r = ref(input_ids=rid, labels=None).logits
            ref_c_lp = _gather_logprobs(ref_c, clb)
            ref_r_lp = _gather_logprobs(ref_r, rlb)

        loss, margin = dpo_loss(pi_c, pi_r, ref_c_lp, ref_r_lp, beta=beta)
        loss = loss / grad_accum
        loss.backward()
        total_loss += loss.item() * grad_accum
        total_margin += margin.item()
        n_steps += 1

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            mean_loss = total_loss / n_steps
            mean_margin = total_margin / n_steps
            line = (f"  step {step_offset + step + 1:>5d} | loss {loss.item()*grad_accum:.4f} "
                    f"| margin {mean_margin:.4f} | {elapsed:.1f}s")
            print(line)
            log_buf.append(line)

    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(log_buf) + "\n")
    return total_loss / max(n_steps, 1), total_margin / max(n_steps, 1), step_offset + n_steps


def save_lora(model, output_dir: Path, meta: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    if meta:
        state["__meta__"] = meta
    torch.save(state, output_dir / "lora_state.pt")
    print(f"[DPO] LoRA 权重已保存到 {output_dir}")


def save_lora_placeholder(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder = {"__placeholder__": True, "note": "no real training, smoke run only"}
    torch.save(placeholder, output_dir / "lora_state.pt")
    print(f"[DPO] 占位 LoRA 权重已保存到 {output_dir}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="DPO 训练（Qwen2.5-0.5B + SDT-LoRA 之上）")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft_ckpt", type=Path, default=Path("ckpt/sft"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/dpo"))
    parser.add_argument("--output_dir", type=Path, default=Path("ckpt/dpo"))
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--log_path", type=Path, default=Path("figures/dpo_train.log"))
    parser.add_argument("--margin_curve", type=Path, default=Path("figures/dpo_margin.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = detect_device()
    print(f"[DPO] device={device}")

    # 1. DPO 数据
    if args.smoke:
        samples = load_dpo_smoke()
        print(f"[DPO] SMOKE 模式，使用内置 {len(samples)} 条样本")
    else:
        try:
            samples = load_dpo(args.data_dir, max_samples=args.max_samples)
            print(f"[DPO] 从 {args.data_dir} 加载 {len(samples)} 条偏好样本")
        except FileNotFoundError as e:
            print(f"[DPO] 数据缺失：{e}")
            samples = load_dpo_smoke()

    # 2. tokenizer / dataset / loader
    if not args.model_path.exists():
        print(f"[DPO] 模型 {args.model_path} 不存在，--smoke 模式仅写入占位 ckpt")
        save_lora_placeholder(args.output_dir)
        # 即使没模型，也写一个空 margin 曲线。
        args.margin_curve.parent.mkdir(parents=True, exist_ok=True)
        args.margin_curve.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding="utf-8")
        return

    tokenizer = load_tokenizer(args.model_path)
    dataset = DPODataset(samples, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: dpo_collate(b, tokenizer.pad_token_id),
    )

    # 3. 构造 policy / ref
    from transformers import AutoModelForCausalLM
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    policy = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=dtype, trust_remote_code=True,
    )
    inject_lora(policy, target_modules=args.target_modules, r=args.rank, alpha=args.alpha)
    if args.sft_ckpt.exists():
        ckpt = args.sft_ckpt / "lora_state.pt" if args.sft_ckpt.is_dir() else args.sft_ckpt
        state = torch.load(ckpt, map_location="cpu")
        load_lora_state_dict(policy, state)
    policy = policy.to(device)

    ref = load_reference_model(
        args.model_path, sft_ckpt=args.sft_ckpt,
        target_modules=args.target_modules, r=args.rank, alpha=args.alpha,
        device=device,
    )

    # 4. optimizer
    lora_params = [p for n, p in policy.named_parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=args.lr, weight_decay=0.0)

    # 5. 训练
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")
    margin_history: list[dict] = []
    for epoch in range(args.epochs):
        avg_loss, avg_margin, _ = train_one_epoch(
            policy, ref, loader, optimizer, device,
            log_interval=args.log_interval, grad_accum=args.grad_accum,
            beta=args.beta, log_path=args.log_path,
        )
        print(f"[DPO] epoch {epoch+1} done, avg_loss={avg_loss:.4f}, avg_margin={avg_margin:.4f}")
        margin_history.append({"epoch": epoch + 1, "avg_loss": avg_loss, "avg_margin": avg_margin})

    args.margin_curve.parent.mkdir(parents=True, exist_ok=True)
    args.margin_curve.write_text(json.dumps(margin_history, ensure_ascii=False, indent=2), encoding="utf-8")
    save_lora(policy, args.output_dir, meta={
        "base_model": str(args.model_path),
        "target_modules": args.target_modules,
        "r": args.rank, "alpha": args.alpha,
        "epochs": args.epochs, "lr": args.lr, "beta": args.beta,
        "max_samples": len(samples), "init_from": str(args.sft_ckpt),
    })


if __name__ == "__main__":
    main()
