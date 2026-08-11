"""Batch generation benchmark:对比 batch=1 串行 vs batch=N 并行。

MiniGPT.generate 支持 list[list[int]] 输入,KV cache 在 batch dim 自然独立。
预期:batch=N 比 N×batch=1 快(消除 N 次 Python 循环开销 + 更好的 GPU 利用)。

用法:python bench_batch_gen.py [--out figures/batch_bench.md]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.model import load_for_eval


def bench(model, prompts, n_runs=3, max_new=64, label="?"):
    times = []
    for _ in range(n_runs):
        torch.manual_seed(42)
        t0 = time.perf_counter()
        _ = model.generate(prompts, max_new_tokens=max_new, temperature=0.0)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    print(f"  [{label:12s}] n={len(prompts):2d}  median={med*1000:7.1f}ms", flush=True)
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=HERE / "ckpt" / "best.pt")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "batch_bench.md")
    args = ap.parse_args()

    model, tok = load_for_eval(str(args.ckpt))
    model.eval()
    base_prompts = [
        "中新社北京",
        "据新华社报道",
        "近年来,随着科技发展",
        "近日,某省召开",
    ]

    print(f"[batch bench] ckpt={args.ckpt.name}, block_size={model.block_size}", flush=True)
    rows = []
    for n in [1, 2, 4, 8]:
        # batch=1 串行
        t_serial = 0.0
        if n > 1:
            prompts1 = [tok.encode(p) for p in base_prompts[:n]]
            t_serial = sum(bench(model, [p], n_runs=2, label=f"serial-1x{n}") for p in prompts1)
        # batch=n
        prompts_n = [tok.encode(p) for p in base_prompts[:n]]
        t_batch = bench(model, prompts_n, n_runs=2, label=f"batch-{n}")
        speedup = t_serial / t_batch if t_serial > 0 else float("inf")
        rows.append({"n": n, "t_serial_ms": t_serial * 1000 if n > 1 else None, "t_batch_ms": t_batch * 1000, "speedup": speedup})
        if n > 1:
            print(f"    speedup at n={n}: {speedup:.2f}x", flush=True)

    # 写报告
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Batch Generation 速度对比",
        "",
        f"模型:`{args.ckpt.name}`(block_size={model.block_size})",
        "对比:N× 串行 batch=1 vs 一次 batch=N(后者用 KV cache 在 batch 维独立)",
        "",
        "| n | 串行 (ms) | batch (ms) | speedup |",
        "|---|---|---|---|",
    ]
    for r in rows:
        ts = f"{r['t_serial_ms']:.1f}" if r["t_serial_ms"] is not None else "—"
        lines.append(f"| {r['n']} | {ts} | {r['t_batch_ms']:.1f} | {r['speedup']:.2f}× |")
    lines += [
        "",
        "## 实现要点",
        "",
        "- `MiniGPT.generate(prompts, ...)` 接受 list[int] 或 list[list[int]]",
        "- 不同长度 prompt 自动 right-pad 到 batch 内最长(用 PAD_ID=0)",
        "- KV cache 的 batch 维天然独立,无需修改 attention",
        "- 返回值:单条 → list[int];批量 → list[list[int]]",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[batch bench] wrote {args.out}", flush=True)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()