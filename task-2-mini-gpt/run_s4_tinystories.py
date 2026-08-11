"""S4: TinyStories 10M 模型涌现叙事。

两阶段:
  Phase 1 (smoke):5M 模型 200 iters,确认 pipeline 通
  Phase 2 (full):  10M 模型 2000 iters,生成样本

输出:
  ckpt/s4_tinystories.pt
  figures/s4_tinystories.md
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
TS_INFO = CKPT_DIR / "tinystories_info.json"


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


def train(cfg, train_data, dev_text, tok, max_iters, device, save_path, label):
    model = MiniGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[S4/{label}] params={n_params:,}, iters={max_iters}", flush=True)
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.1},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=3e-4, betas=(0.9, 0.95),
    )
    block_size = cfg.block_size
    best_ppl = float("inf")
    t0 = time.time()
    for it in range(max_iters):
        lr = get_lr(it, 100, max_iters, 3e-4, 3e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        starts = torch.randint(0, len(train_data) - block_size - 1, (32,))
        x = torch.stack([train_data[s : s + block_size] for s in starts]).to(device)
        y = torch.stack([train_data[s + 1 : s + 1 + block_size] for s in starts]).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if it % 50 == 0 or it == max_iters - 1:
            print(f"  iter {it:4d} | loss {loss.item():.4f}", flush=True)
        if it % 200 == 0 or it == max_iters - 1:
            ppl = estimate_dev_ppl(model, dev_text, tok, block_size)
            print(f"    [eval] dev_ppl = {ppl:.2f}", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                model.save_checkpoint(
                    str(save_path), tokenizer_path=str(CKPT_DIR / "tinystories_tokenizer.json"),
                    extra={"iter": it, "dev_ppl": ppl, "n_params": n_params, "label": label},
                )
                print(f"    [ckpt] saved (ppl={ppl:.2f})", flush=True)
    print(f"[S4/{label}] done. best_ppl={best_ppl:.2f} | total {time.time()-t0:.1f}s", flush=True)
    return model, best_ppl


def main():
    info = json.loads(TS_INFO.read_text(encoding="utf-8"))
    train_path = HERE / info["train"]
    dev_path = HERE / info["dev"]
    if not train_path.exists():
        sys.exit(f"训练集 {train_path} 不存在,请先跑 data/download_tinystories.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # BPE(一次性,~10s)
    train_text = train_path.read_text(encoding="utf-8")
    tok = BPETokenizer()
    t0 = time.time()
    tok.train(train_text, vocab_size=4096, verbose=False)
    print(f"[S4] BPE done in {time.time()-t0:.1f}s, vocab={tok.vocab_size}", flush=True)
    tok.save(str(CKPT_DIR / "tinystories_tokenizer.json"))

    dev_text = dev_path.read_text(encoding="utf-8")
    train_ids = tok.encode(train_text)
    train_data = torch.tensor(train_ids, dtype=torch.long)
    print(f"[S4] train_tokens={len(train_data)}", flush=True)

    # === Phase 1: smoke ===
    smoke_cfg = MiniGPTConfig(
        vocab_size=tok.vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=256,
        block_size=128, dropout=0.1, weight_tying=True,
    )
    print(f"\n[S4] === Phase 1 smoke test (128d, 4L, 200 iters) ===", flush=True)
    train(smoke_cfg, train_data, dev_text, tok, max_iters=200, device=device,
          save_path=CKPT_DIR / "s4_smoke.pt", label="smoke")
    # 验证 smoke 通了

    # === Phase 2: full ===
    cfg = MiniGPTConfig(
        vocab_size=tok.vocab_size, d_model=256, n_heads=4, n_layers=8, d_ff=768,
        block_size=256, dropout=0.1, weight_tying=True,
    )
    print(f"\n[S4] === Phase 2 full (256d, 8L, 2000 iters) ===", flush=True)
    model, best_ppl = train(cfg, train_data, dev_text, tok, max_iters=2000,
                           device=device, save_path=CKPT_DIR / "s4_tinystories.pt",
                           label="full")

    # === 生成 4 段 ===
    print(f"\n[S4] === generation ===", flush=True)
    model.eval()
    samples = []
    for prompt in ["Once upon a time", "Tom and Lily", "The little girl", "One day"]:
        torch.manual_seed(42)
        out_ids = model.generate(tok.encode(prompt), max_new_tokens=80, top_p=0.9, temperature=0.7)
        text = tok.decode(out_ids[len(tok.encode(prompt)):])
        samples.append({"prompt": prompt, "text": text})

    # 写 markdown
    out_path = HERE / "samples_tinystories.md"
    lines = [
        "# S4: TinyStories 10M 模型生成",
        "",
        f"**模型**:s4_tinystories.pt,~10M 参数,2000 iters",
        f"**dev_ppl**:{best_ppl:.2f}",
        "",
        "| Prompt | 生成 |",
        "|---|---|",
    ]
    for s in samples:
        text_safe = s["text"].replace("|", "\\|").replace("\n", " ⏎ ")
        lines.append(f"| `{s['prompt']}` | {text_safe} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[S4] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()