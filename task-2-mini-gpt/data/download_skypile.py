"""Streaming 下载 SkyPile-150B 子集,生成 train.txt / dev.txt / dataset_info.json。

为什么独立成脚本而不是 download.py 的 --dataset 分支:
- 现有 download.py 的 get_skypile() 只打印提示(README 建议手动跑 streaming)
- streaming 参数(样本数 / 长度过滤 / 输出目录)需要 CLI 化,不应该塞进 download.py 的统一分发
- 把"取子集"这件事单独抽出来,未来换数据源只改这一个文件

用法:
    # 默认取 5 万条,长度 [100, 2000] 字符
    python data/download_skypile.py

    # 自定义规模
    python data/download_skypile.py --n-samples 20000 --max-chars 1500

    # 加速(国内)
    HF_ENDPOINT=https://hf-mirror.com python data/download_skypile.py

输出:
    data/train.txt        (90% 样本,段落以 \n\n 分隔)
    data/dev.txt          (5% 样本)
    data/dataset_info.json  (含 ppl_threshold 与子集元信息)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent


def stream_skypile(n_samples: int, min_chars: int, max_chars: int):
    """Generator:从 SkyPile-150B streaming 取过滤后的样本。"""
    from datasets import load_dataset

    ds = load_dataset(
        "Skywork/SkyPile-150B",
        split="train",
        streaming=True,
    )

    n_seen = 0
    n_kept = 0
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
            elapsed = time.time() - t0
            print(
                f"  [streaming] seen={n_seen}, kept={n_kept}, "
                f"{n_kept / max(elapsed, 1e-3):.0f} samples/s"
            )
        if n_kept >= n_samples:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--n-samples", type=int, default=50000,
        help="保留多少条样本(默认 5 万,中文 500 字符/条 ≈ 25MB)",
    )
    ap.add_argument("--min-chars", type=int, default=100, help="最短字符数")
    ap.add_argument("--max-chars", type=int, default=2000, help="最长字符数")
    ap.add_argument("--dev-ratio", type=float, default=0.05, help="dev 占比")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument(
        "--ppl-threshold", type=float, default=50.0,
        help="dev ppl 通过阈值;~25MB 中文 + 5M 模型应远低于此",
    )
    args = ap.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com")
        print("[提示] 也可: export HF_ENDPOINT=https://hf-mirror.com\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 流式取样
    print(f"[streaming] Skywork/SkyPile-150B, 目标 {args.n_samples} 条, "
          f"长度 [{args.min_chars}, {args.max_chars}]\n")
    t0 = time.time()
    samples = list(stream_skypile(args.n_samples, args.min_chars, args.max_chars))
    print(f"\n[streaming] 完成:实际取到 {len(samples)} 条,耗时 {time.time()-t0:.1f}s")

    if len(samples) < 100:
        sys.exit(f"[错误] 只取到 {len(samples)} 条样本,太少。请检查网络或调小 --min-chars")

    # 2. 切分 train / dev
    split_at = max(1, int(len(samples) * (1 - args.dev_ratio)))
    train_samples = samples[:split_at]
    dev_samples = samples[split_at:]

    # 段落以 \n\n 分隔,符合预分词正则对段落边界的处理
    sep = "\n\n"
    train_path = args.out_dir / "train.txt"
    dev_path = args.out_dir / "dev.txt"
    train_path.write_text(sep.join(train_samples), encoding="utf-8")
    dev_path.write_text(sep.join(dev_samples), encoding="utf-8")

    # 3. dataset_info.json(eval/run.py 会读这个)
    info = {
        "dataset": "skypile_subset",
        "train": "train.txt",
        "dev": "dev.txt",
        "ppl_threshold": args.ppl_threshold,
        "n_samples": len(samples),
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "note": (
            f"SkyPile-150B 子集,通过 streaming 取 {len(samples)} 条长度 "
            f"[{args.min_chars}, {args.max_chars}] 字符的样本。"
        ),
    }
    info_path = args.out_dir / "dataset_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[output]")
    print(f"  {train_path.name}: {train_path.stat().st_size // 1024} KB ({len(train_samples)} samples)")
    print(f"  {dev_path.name}: {dev_path.stat().st_size // 1024} KB ({len(dev_samples)} samples)")
    print(f"  {info_path.name}: ppl_threshold={info['ppl_threshold']}")


if __name__ == "__main__":
    main()