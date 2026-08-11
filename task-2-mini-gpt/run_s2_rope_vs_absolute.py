"""S2 redo: RoPE vs Absolute PE 长序列外推,**先确认两者都训到饱和再对比**。

旧版问题:rope 2000 iters ppl 188,absolute ppl 653。差距 3.5×,对比被"训练质量"污染。

新版:
  Phase 1:rope 和 absolute 各训 4000 iters(更长)
  Phase 2:验证两者 loss 都进入饱和(最后 1000 iters loss 下降 < 5%)
  Phase 3:外推对比(eval @ block=64/128/256)
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.config import MiniGPTConfig
from src.model import MiniGPT
from src.tokenizer import BPETokenizer

CKPT_DIR = HERE / "ckpt"
TOKENIZER_PATH = CKPT_DIR / "tokenizer.json"
DATA_DIR = HERE / "data"


def get_lr(it, warmup, max_iters, peak_lr, min_lr):
    if it < warmup:
        return peak_lr * (it + 1) / (warmup + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * decay_ratio))


def estimate_dev_ppl(model, dev_text, tok, block):
    model.eval()
    dev_ids = tok.encode(dev_text)[:4096]
    nll, n_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, max(1, len(dev_ids) - 1), block):
            window = dev_ids[i : i + block + 1]
            if len(window) < 2:
                break
            chunk = torch.tensor([window], dtype=torch.long, device=next(model.parameters()).device)
            logits = model(chunk)
            nll += F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1), reduction="sum",
            ).item()
            n_tok += chunk.size(1) - 1
    model.train()
    return math.exp(nll / n_tok) if n_tok > 0 else float("inf")


def train_one(pos_encoding, train_data, dev_text, tok, cfg_base, max_iters, device, save_path):
    cfg = MiniGPTConfig(
        vocab_size=tok.vocab_size, d_model=cfg_base["d_model"],
        n_heads=cfg_base["n_heads"], n_layers=cfg_base["n_layers"],
        d_ff=cfg_base["d_ff"], block_size=cfg_base["block_size"],
        dropout=0.0, weight_tying=True, pos_encoding=pos_encoding,
    )
    print(f"\n[S2/{pos_encoding}] === train {max_iters} iters ===", flush=True)
    model = MiniGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[S2/{pos_encoding}] params={n_params:,}", flush=True)

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.1},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=3e-4, betas=(0.9, 0.95),
    )
    block_size = cfg.block_size
    best_ppl = float("inf")
    loss_log = []
    t0 = time.time()
    for it in range(max_iters):
        lr = get_lr(it, 100, max_iters, 3e-4, 3e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        starts = torch.randint(0, len(train_data) - block_size - 1, (16,))
        x = torch.stack([train_data[s : s + block_size] for s in starts]).to(device)
        y = torch.stack([train_data[s + 1 : s + 1 + block_size] for s in starts]).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_log.append((it, loss.item()))
        if it % 200 == 0 or it == max_iters - 1:
            ppl = estimate_dev_ppl(model, dev_text, tok, block_size)
            print(f"  iter {it:4d} | loss {loss.item():.4f} | dev_ppl {ppl:.2f}", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                model.save_checkpoint(str(save_path), tokenizer_path=str(TOKENIZER_PATH),
                                      extra={"iter": it, "dev_ppl": ppl, "pos_encoding": pos_encoding})
    print(f"[S2/{pos_encoding}] best_ppl={best_ppl:.2f} | {time.time()-t0:.1f}s", flush=True)
    return model, best_ppl, loss_log


def check_saturation(loss_log, window=1000):
    """最后 window 个 iter 的 loss 下降幅度,<5% 视为饱和。"""
    if len(loss_log) < 2 * window:
        return False, float("inf")
    early = [l for _, l in loss_log[-2*window:-window]]
    late = [l for _, l in loss_log[-window:]]
    early_avg = sum(early) / len(early)
    late_avg = sum(late) / len(late)
    decline = (early_avg - late_avg) / early_avg * 100
    return decline < 5.0, decline


def eval_extrapolation(model, pos_encoding, dev_text, tok, test_blocks, max_block):
    """Eval 模型在不同 seq length 下的 ppl。需要 block_size >= max_block。"""
    cfg = MiniGPTConfig(
        vocab_size=tok.vocab_size, d_model=model.config.d_model,
        n_heads=model.config.n_heads, n_layers=model.config.n_layers,
        d_ff=model.config.d_ff, block_size=max_block + 1,
        dropout=0.0, weight_tying=True, pos_encoding=pos_encoding,
    )
    eval_model = MiniGPT(cfg).to(next(model.parameters()).device)
    sd = model.state_dict()
    esd = eval_model.state_dict()
    for k in esd:
        if k in sd and sd[k].shape == esd[k].shape:
            esd[k].copy_(sd[k])
    eval_model.load_state_dict(esd, strict=False)
    eval_model.eval()
    rows = []
    for tb in test_blocks:
        ppl = estimate_dev_ppl(eval_model, dev_text, tok, tb)
        rows.append({"block_size": tb, "ppl": ppl})
        print(f"    [{pos_encoding} @ block={tb}] ppl={ppl:.2f}", flush=True)
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    tok = BPETokenizer.from_pretrained(str(TOKENIZER_PATH))
    train_text = (DATA_DIR / "train.txt").read_text(encoding="utf-8")
    dev_text = (DATA_DIR / "dev.txt").read_text(encoding="utf-8")
    train_ids = tok.encode(train_text)
    train_data = torch.tensor(train_ids, dtype=torch.long)

    # 小模型(block=64 让 extrapolation 有空间)
    cfg_base = {"d_model": 256, "n_heads": 4, "n_layers": 4, "d_ff": 1024, "block_size": 64}
    max_iters = 4000

    # === Phase 1: train both ===
    rope_ckpt = CKPT_DIR / "s2_rope.pt"
    abs_ckpt = CKPT_DIR / "s2_abs.pt"
    rope_model, rope_ppl, rope_log = train_one("rope", train_data, dev_text, tok, cfg_base, max_iters, device, rope_ckpt)
    abs_model, abs_ppl, abs_log = train_one("absolute", train_data, dev_text, tok, cfg_base, max_iters, device, abs_ckpt)

    # === Phase 2: 检查饱和 ===
    rope_sat, rope_decl = check_saturation(rope_log)
    abs_sat, abs_decl = check_saturation(abs_log)
    print(f"\n[S2] saturation check: rope decline={rope_decl:.1f}% ({'saturated' if rope_sat else 'NOT saturated'}), "
          f"absolute decline={abs_decl:.1f}% ({'saturated' if abs_sat else 'NOT saturated'})", flush=True)

    # === Phase 3: 外推对比 ===
    print(f"\n[S2] === extrapolation eval ===", flush=True)
    test_blocks = [64, 128, 256]
    rope_rows = eval_extrapolation(rope_model, "rope", dev_text, tok, test_blocks, max(test_blocks))
    abs_rows = eval_extrapolation(abs_model, "absolute", dev_text, tok, test_blocks, max(test_blocks))

    # 写报告
    out_path = HERE / "figures" / "s2_pe_extrapolation.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S2: 绝对 PE vs RoPE 长序列外推(饱和后对比)",
        "",
        "## 训练饱和度",
        "",
        "| 模型 | 训练 iters | best dev_ppl(block=64) | 最后 1000 iters loss 下降 | 饱和? |",
        "|---|---|---|---|---|",
        f"| rope | {max_iters} | {rope_ppl:.2f} | {rope_decl:.1f}% | {'✓' if rope_sat else '✗'} |",
        f"| absolute | {max_iters} | {abs_ppl:.2f} | {abs_decl:.1f}% | {'✓' if abs_sat else '✗'} |",
        "",
        "## 外推对比",
        "",
        "| 模型 | block=64(训练长度) | block=128(2×外推) | block=256(4×外推) |",
        "|---|---|---|---|",
        f"| rope | {rope_rows[0]['ppl']:.2f} | {rope_rows[1]['ppl']:.2f} | {rope_rows[2]['ppl']:.2f} |",
        f"| absolute | {abs_rows[0]['ppl']:.2f} | {abs_rows[1]['ppl']:.2f} | {abs_rows[2]['ppl']:.2f} |",
        "",
        "## 退化率(以 block=64 ppl 为基准)",
        "",
        "| 模型 | block=64 | block=128 退化 | block=256 退化 |",
        "|---|---|---|---|",
    ]
    for label, rows in [("rope", rope_rows), ("absolute", abs_rows)]:
        base = rows[0]["ppl"]
        d128 = (rows[1]["ppl"] - base) / base * 100
        d256 = (rows[2]["ppl"] - base) / base * 100
        lines.append(f"| {label} | {base:.2f} | +{d128:.1f}% | +{d256:.1f}% |")

    if not (rope_sat and abs_sat):
        lines += [
            "",
            f"**注意**:两者未完全饱和(下降 >5%),对比可能仍偏训练质量。重跑或更长训练。",
        ]
    lines += [
        "",
        "## 解读",
        "",
        "- RoPE 的旋转角数学上对任意 pos 成立,且相对位置差代数一致",
        "- Sinusoidal 绝对 PE 也能扩展到任意 pos,但训练时没见过的位置模式不保证泛化",
        "- **前提**:两者都训到饱和(base ppl 接近),对比才有意义",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[S2] wrote {out_path}", flush=True)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "rope": {"best_ppl": rope_ppl, "saturation_decline_pct": rope_decl, "saturated": rope_sat, "extrapolation": rope_rows},
        "absolute": {"best_ppl": abs_ppl, "saturation_decline_pct": abs_decl, "saturated": abs_sat, "extrapolation": abs_rows},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()