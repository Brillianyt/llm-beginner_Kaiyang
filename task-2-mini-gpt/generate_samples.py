"""生成 4 段不同采样策略的样例,写到 samples.md。

用法:
    python generate_samples.py [--ckpt ckpt/best.pt] [--out samples.md]

输出 samples.md 结构:
- 每段一个 prompt × 4 种采样策略(greedy / top-k / top-p / temperature)
- 末尾附 200-500 字实验观察
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.model import load_for_eval


SAMPLING_CONFIGS = [
    ("greedy", {"temperature": 0.0}),
    ("top-k=40", {"temperature": 1.0, "top_k": 40}),
    ("top-p=0.9", {"temperature": 1.0, "top_p": 0.9}),
    ("temperature=0.7", {"temperature": 0.7, "top_p": 0.9}),
]


def generate_for_prompt(model, tok, prompt_text: str, max_new_tokens: int, seed: int) -> list[tuple[str, str]]:
    """对单个 prompt 用 4 种采样策略生成,返回 [(策略名, 生成文本), ...]。"""
    prompt_ids = tok.encode(prompt_text)
    results = []
    for name, kwargs in SAMPLING_CONFIGS:
        torch.manual_seed(seed)
        out_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, **kwargs)
        full_text = tok.decode(out_ids)
        # 截掉 prompt 部分,只展示新生成
        new_text = tok.decode(out_ids[len(prompt_ids):])
        results.append((name, kwargs, new_text, full_text))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=HERE / "ckpt" / "best.pt")
    ap.add_argument("--out", type=Path, default=HERE / "samples.md")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, tok = load_for_eval(str(args.ckpt))
    model.eval()
    print(f"[load] {args.ckpt}, block_size={model.block_size}, vocab={tok.vocab_size}")

    # 选 4 个匹配 SkyPile 新闻分布的 prompt
    prompts = [
        "中新社北京",
        "据新华社报道",
        "近年来,随着科技发展",
        "近日,某省召开",
    ]
    # 过滤掉 vocab 不支持的 prompt
    prompts = [p for p in prompts if tok.encode(p)]
    print(f"[prompts] {prompts}")

    sections: list[str] = []
    sections.append("# task-2-mini-gpt 生成样例\n")
    sections.append(
        f"模型:`{args.ckpt.name}`,block_size={model.block_size},vocab={tok.vocab_size}  \n"
        f"max_new_tokens={args.max_new_tokens},seed={args.seed}\n"
    )
    sections.append("\n---\n")

    for p in prompts:
        sections.append(f"\n## Prompt: `{p}`\n")
        sections.append("| 策略 | 参数 | 生成(新 token) |\n|---|---|---|")
        rows = generate_for_prompt(model, tok, p, args.max_new_tokens, args.seed)
        for name, kwargs, new_text, _ in rows:
            kw_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            # markdown 里防注入
            safe = new_text.replace("|", "\\|").replace("\n", " ⏎ ")
            sections.append(f"| {name} | `{kw_str}` | {safe} |")
        sections.append("")

    args.out.write_text("\n".join(sections), encoding="utf-8")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()