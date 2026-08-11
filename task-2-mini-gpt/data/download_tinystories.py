"""TinyStories 子集下载(streaming,不覆盖现有 train.txt/dev.txt)。

TinyStories 是 roneneldan/TinyStories 合成儿童故事语料,
句式简单重复,小模型也能学会涌现叙事。

为避免覆盖现有 SkyPile 数据,本脚本:
- 用 streaming 取 N 条
- 写入 ckpt/tinystories_train.txt / tinystories_dev.txt
- 写 ckpt/tinystories_info.json

用法:
    HF_ENDPOINT=https://hf-mirror.com python data/download_tinystories.py --n-samples 50000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent
CKPT_DIR = DATA_DIR.parent / "ckpt"


def stream_tinystories(n_samples: int, min_chars: int, max_chars: int):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    n_seen, n_kept = 0, 0
    t0 = time.time()
    for row in ds:
        n_seen += 1
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        if len(text) < min_chars or len(text) > max_chars:
            continue
        yield text
        n_kept += 1
        if n_kept % 2000 == 0:
            print(
                f"  [streaming] seen={n_seen}, kept={n_kept}, "
                f"{n_kept / max(time.time() - t0, 1e-3):.0f} samples/s",
                flush=True,
            )
        if n_kept >= n_samples:
            break
    print(f"[streaming] done: seen={n_seen}, kept={n_kept}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=50000)
    ap.add_argument("--min-chars", type=int, default=100)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--dev-ratio", type=float, default=0.05)
    ap.add_argument("--ppl-threshold", type=float, default=15.0,
                    help="10M 模型在 TinyStories 子集上的预期 ppl < 15")
    args = ap.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[download] roneneldan/TinyStories, target {args.n_samples} samples", flush=True)
    samples = list(stream_tinystories(args.n_samples, args.min_chars, args.max_chars))
    if len(samples) < 1000:
        sys.exit(f"只取到 {len(samples)} 条,太少")

    split_at = max(1, int(len(samples) * (1 - args.dev_ratio)))
    train_samples = samples[:split_at]
    dev_samples = samples[split_at:]
    sep = "\n\n"

    train_path = CKPT_DIR / "tinystories_train.txt"
    dev_path = CKPT_DIR / "tinystories_dev.txt"
    train_path.write_text(sep.join(train_samples), encoding="utf-8")
    dev_path.write_text(sep.join(dev_samples), encoding="utf-8")

    info = {
        "dataset": "tinystories_subset",
        "train": str(train_path.relative_to(DATA_DIR.parent)),
        "dev": str(dev_path.relative_to(DATA_DIR.parent)),
        "ppl_threshold": args.ppl_threshold,
        "n_samples": len(samples),
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "note": "roneneldan/TinyStories 子集(英文,合成儿童故事)。供 S4 用。",
    }
    info_path = CKPT_DIR / "tinystories_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n[output]\n"
        f"  {train_path.name}: {train_path.stat().st_size // 1024} KB\n"
        f"  {dev_path.name}: {dev_path.stat().st_size // 1024} KB\n"
        f"  {info_path.name}: ppl_threshold={info['ppl_threshold']}",
        flush=True,
    )


if __name__ == "__main__":
    main()