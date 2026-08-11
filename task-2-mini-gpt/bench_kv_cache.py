"""S3: KV cache 开/关 推理速度对比。

对比 use_kv_cache=True / False 下,生成 N 个 token 的耗时与吞吐。
预期:KV cache 把每步 O(T) 重算降到 O(1),加速比随 prompt+T 线性增长。

用法:python bench_kv_cache.py [--ckpt ckpt/best.pt] [--out figures/s3_kv_cache_bench.md]
"""
from __future__ import annotations

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


def bench_generate(model, prompt_ids, max_new_tokens, use_cache, n_runs=3):
    """同一 prompt + max_new_tokens 跑 n_runs 次,返回耗时列表(秒)。"""
    times = []
    for _ in range(n_runs):
        torch.manual_seed(42)
        t0 = time.perf_counter()
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # greedy 让结果确定,避免采样抖动
            use_kv_cache=use_cache,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return times, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=HERE / "ckpt" / "best.pt")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "s3_kv_cache_bench.md")
    ap.add_argument("--n-runs", type=int, default=3)
    args = ap.parse_args()

    model, tok = load_for_eval(str(args.ckpt))
    model.eval()
    block_size = model.block_size

    print(f"[S3 bench] ckpt={args.ckpt.name}, block_size={block_size}, "
          f"vocab={tok.vocab_size}")

    # 测试矩阵:不同 prompt 长度 × 不同生成长度
    # prompt 从短到长,生成也从短到长
    prompt_text = "深度学习是机器学习的一个分支,通过多层神经网络自动学习数据的层次化表示。"
    base_ids = tok.encode(prompt_text)
    print(f"[S3 bench] base prompt tokens = {len(base_ids)}")

    rows = []
    for prompt_mult in [1, 2, 4]:
        prompt_ids = base_ids[: len(base_ids) * prompt_mult // 2]  # 截短/保留
        # 强制长度:用 repeat 直到达到目标
        target_prompt_len = block_size * prompt_mult // 2
        while len(prompt_ids) < target_prompt_len:
            prompt_ids = prompt_ids + base_ids
        prompt_ids = prompt_ids[:target_prompt_len]
        for max_new in [32, 64, 128]:
            t_cache, _ = bench_generate(model, prompt_ids, max_new, use_cache=True, n_runs=args.n_runs)
            t_nocache, _ = bench_generate(model, prompt_ids, max_new, use_cache=False, n_runs=args.n_runs)

            cache_med = statistics.median(t_cache)
            nocache_med = statistics.median(t_nocache)
            speedup = nocache_med / cache_med if cache_med > 0 else float("inf")
            tokens_per_sec_cache = max_new / cache_med
            tokens_per_sec_nocache = max_new / nocache_med

            rows.append({
                "prompt_len": len(prompt_ids),
                "max_new": max_new,
                "cache_ms": cache_med * 1000,
                "nocache_ms": nocache_med * 1000,
                "speedup": speedup,
                "tok_per_s_cache": tokens_per_sec_cache,
                "tok_per_s_nocache": tokens_per_sec_nocache,
            })
            print(
                f"  prompt={len(prompt_ids):4d} new={max_new:3d} | "
                f"cache={cache_med*1000:7.1f}ms ({tokens_per_sec_cache:5.1f} tok/s)  "
                f"no-cache={nocache_med*1000:7.1f}ms ({tokens_per_sec_nocache:5.1f} tok/s)  "
                f"speedup={speedup:.2f}x"
            )

    # 写 markdown 报告
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S3: KV cache 推理速度对比",
        "",
        f"模型:`{args.ckpt.name}`(block_size={block_size}, vocab={tok.vocab_size})  ",
        f"设备:CPU  \n生成方式:greedy(temperature=0),每组取 {args.n_runs} 次中位数  ",
        "",
        "## 结果",
        "",
        "| prompt_len | max_new | cache (ms) | no-cache (ms) | speedup | cache tok/s | no-cache tok/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['prompt_len']} | {r['max_new']} | "
            f"{r['cache_ms']:.1f} | {r['nocache_ms']:.1f} | "
            f"**{r['speedup']:.2f}x** | {r['tok_per_s_cache']:.1f} | {r['tok_per_s_nocache']:.1f} |"
        )
    lines += [
        "",
        "## 解读",
        "",
        f"- KV cache 加速比随 **prompt + max_new** 总长度线性增长(理论上 O(T²) → O(T))",
        f"- 朴素路径每步重算整个 (prompt + 已生成),复杂度 O(T²);cache 路径每步 O(1)",
        f"- 实测在 prompt={rows[-1]['prompt_len']}, max_new={rows[-1]['max_new']} 时,加速比 ≈ {rows[-1]['speedup']:.1f}x",
        "",
        "## 实现要点",
        "",
        "- `MiniGPT.forward(ids, kv_cache=None, return_cache=False)`:接受可选 cache",
        "- `MiniGPT.generate(..., use_kv_cache=True)`:新增参数,True 走增量、False 走朴素",
        "- cache 拼接在 K/V 的 seq 维(dim=-2)",
        "- 增量解码时新 token 的 position_offset 自动从 cache 长度推断",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    # 也存 JSON 方便后续分析
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[S3 bench] wrote {args.out} and {json_path}")


if __name__ == "__main__":
    main()