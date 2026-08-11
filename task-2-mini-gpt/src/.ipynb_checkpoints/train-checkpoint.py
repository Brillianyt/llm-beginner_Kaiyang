"""M4: 训练入口,跑 `python src/train.py` 即可。

设计要点:
- AdamW,weight_decay 不作用于 bias / LayerNorm
- cosine LR schedule + linear warmup
- gradient clipping at 1.0
- 按 dev perplexity 选 best checkpoint
- 仅 FP32(不引入 AMP,见 plan 风险表)

用法:
    cd task-2-mini-gpt
    python src/train.py [--device cuda|cpu|auto] [--max-iters N] ...
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

# 让脚本能 import src.*
HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent
sys.path.insert(0, str(TASK_ROOT))

from src.config import MiniGPTConfig, TrainConfig
from src.model import MiniGPT
from src.tokenizer import BPETokenizer


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机采样 batch:(B, T) 输入 + (B, T) 目标(右移一位)。"""
    n = len(data) - block_size - 1
    if n <= 0:
        raise ValueError(f"数据长度 {len(data)} 不足以切 block_size={block_size}")
    starts = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts])
    return x.to(device), y.to(device)


def get_lr(
    it: int,
    warmup_iters: int,
    max_iters: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup → cosine decay。"""
    if it < warmup_iters:
        return peak_lr * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + (peak_lr - min_lr) * coeff


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    dev_data: torch.Tensor,
    cfg: TrainConfig,
) -> dict[str, float]:
    """train 采 eval_iters 个 batch 算平均 cross-entropy loss(随机采样,用于监控训练)。"""
    model.eval()
    losses: list[float] = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(train_data, cfg.model_config.block_size, cfg.batch_size, cfg.device)
        logits = model(x)
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            ).item()
        )
    model.train()
    return {"train": sum(losses) / len(losses)}


@torch.no_grad()
def estimate_dev_ppl_eval_method(
    model: MiniGPT,
    dev_text: str,
    tok: BPETokenizer,
    cfg: TrainConfig,
) -> float:
    """与 eval/run.py:50 test_perplexity_on_dev 完全一致的算法。

    算法:dev.txt → tokenize → 取前 4096 token → 按 block_size 非重叠窗口 →
    sum NLL → 除以总 token 数 → exp。

    为什么跟 estimate_loss 分开:训练监控用 20 batch 随机采样(快、有偏),
    best.pt 保存必须用与 eval harness 一致的口径(否则会出现"训练日志说过了,
    eval 说没过"的 ~5pp 系统偏差)。
    """
    model.eval()
    dev_ids = tok.encode(dev_text)[:4096]
    block = cfg.model_config.block_size
    nll, n_tok = 0.0, 0
    for i in range(0, max(1, len(dev_ids) - 1), block):
        window = dev_ids[i : i + block + 1]
        if len(window) < 2:
            break
        chunk = torch.tensor([window], dtype=torch.long, device=cfg.device)
        logits = model(chunk)
        nll += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        n_tok += chunk.size(1) - 1
    model.train()
    if n_tok == 0:
        return float("inf")
    return math.exp(nll / n_tok)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="从已有 best.pt 加载 weights(不加载 optimizer state)")
    return ap


def resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main() -> None:
    args = build_argparser().parse_args()

    # 1. 路径与设备
    data_dir = TASK_ROOT / "data"
    ckpt_dir = TASK_ROOT / "ckpt"
    tokenizer_path = ckpt_dir / "tokenizer.json"
    device = resolve_device(args.device)
    seed = args.seed if args.seed is not None else 42
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # 2. 加载 tokenizer
    tok = BPETokenizer.from_pretrained(str(tokenizer_path))
    print(f"[setup] tokenizer vocab_size={tok.vocab_size}")

    # 3. 模型配置(命令行覆盖默认值)
    model_cfg = MiniGPTConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model if args.d_model is not None else 256,
        n_heads=args.n_heads if args.n_heads is not None else 4,
        n_layers=args.n_layers if args.n_layers is not None else 4,
        d_ff=args.d_ff if args.d_ff is not None else 1024,
        block_size=args.block_size if args.block_size is not None else 256,
        dropout=args.dropout if args.dropout is not None else 0.1,
        weight_tying=True,
    )

    cfg = TrainConfig(
        data_dir=data_dir,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        model_config=model_cfg,
        batch_size=args.batch_size if args.batch_size is not None else 32,
        max_iters=args.max_iters if args.max_iters is not None else 3000,
        device=device,
        seed=seed,
        learning_rate=args.lr if args.lr is not None else 3e-4,
    )
    # data 编码(预 tokenize 到 long tensor)
    train_text = (data_dir / "train.txt").read_text(encoding="utf-8")
    dev_text = (data_dir / "dev.txt").read_text(encoding="utf-8")
    train_ids = tok.encode(train_text)
    dev_ids = tok.encode(dev_text)
    train_data = torch.tensor(train_ids, dtype=torch.long)
    dev_data = torch.tensor(dev_ids, dtype=torch.long)
    print(
        f"[setup] device={device}, train_tokens={len(train_data)}, "
        f"dev_tokens={len(dev_data)}"
    )
    print(f"[setup] model_config={model_cfg}")

    # 4. 构造模型与优化器
    model = MiniGPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] model params={n_params:,}")

    # 4.1 resume(只加载 weights,不加载 optimizer state)
    if args.resume_from is not None and args.resume_from.exists():
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        prev_iter = ckpt.get("extra", {}).get("iter", "?")
        prev_dev = ckpt.get("extra", {}).get("dev_loss", "?")
        print(f"[resume] loaded weights from {args.resume_from.name} (prev iter={prev_iter}, prev dev={prev_dev})")
        print(f"[resume] note: optimizer state NOT resumed (fresh AdamW); LR schedule starts from iter 0")

    decay_params, nodecay_params = [], []
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )

    # 5. 训练循环
    best_dev_ppl = float("inf")
    t_start = time.time()
    for it in range(cfg.max_iters):
        lr = get_lr(it, cfg.warmup_iters, cfg.max_iters,
                    cfg.learning_rate, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, cfg.model_config.block_size, cfg.batch_size, cfg.device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if it % cfg.log_interval == 0 or it == cfg.max_iters - 1:
            print(
                f"iter {it:5d} | lr {lr:.2e} | train_loss {loss.item():.4f} | "
                f"elapsed {time.time()-t_start:.1f}s"
            )

        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            # 用与 eval harness 一致的指标保存 best.pt
            train_metrics = estimate_loss(model, train_data, dev_data, cfg)
            dev_ppl = estimate_dev_ppl_eval_method(model, dev_text, tok, cfg)
            print(
                f"  [eval] train_loss={train_metrics['train']:.4f} "
                f"dev_ppl(eval-method)={dev_ppl:.2f}"
            )
            if dev_ppl < best_dev_ppl:
                best_dev_ppl = dev_ppl
                model.save_checkpoint(
                    str(ckpt_dir / "best.pt"),
                    tokenizer_path=str(tokenizer_path),
                    extra={"iter": it, "dev_ppl": dev_ppl},
                )
                print(f"  [ckpt] saved best.pt (dev_ppl={dev_ppl:.2f})")

    print(
        f"\n[done] best dev_ppl(eval-method)={best_dev_ppl:.2f}"
    )


if __name__ == "__main__":
    main()