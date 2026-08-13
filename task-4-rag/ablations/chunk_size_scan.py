"""S1: chunk_size 扫描 —— 128 / 256 / 512 / 1024 字符对 Recall 的影响。

用法
----
    python ablations/chunk_size_scan.py
    python ablations/chunk_size_scan.py --sizes 128 256 512 --top-k 10

每个 chunk_size 都要重建索引（chunk 边界不同），所以会跑 3-4 次 embedding。
缺模型 / 缺 PDF 时直接跳过并打印原因。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.indexer import build_index
from src.retriever import Retriever
from src.utils import GOLD_QA_PATH, normalize_text, PDF_PATH


def gold_qa():
    items = []
    with GOLD_QA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def recall_at_k(retriever: Retriever, gold: list[dict], k: int) -> dict:
    hit = 0
    rr = 0.0
    for item in gold:
        anchors = [normalize_text(a) for a in item["gold_anchors"]]
        results = retriever.retrieve(item["question"], k=k)
        rank = None
        for i, r in enumerate(results, 1):
            txt = normalize_text(r.get("text", ""))
            if any(a and a in txt for a in anchors):
                rank = i
                break
        if rank:
            hit += 1
            rr += 1 / rank
    n = len(gold)
    return {
        "recall_at_10": round(hit / n, 3),
        "mrr": round(rr / n, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk-mode", choices=["char", "paragraph"], default="paragraph")
    args = ap.parse_args()

    if not PDF_PATH.exists() or PDF_PATH.stat().st_size == 0:
        print(f"[chunk_size_scan] {PDF_PATH} 不存在，无法跑本消融")
        sys.exit(2)
    gold = gold_qa()
    print(f"[chunk_size_scan] gold QA: {len(gold)} 条；扫描 sizes={args.sizes}")

    results = []
    for size in args.sizes:
        overlap = min(args.overlap, size // 4)
        t0 = time.perf_counter()
        print(f"\n--- chunk_size={size}, overlap={overlap} ---")
        try:
            build_index(pdf_path=PDF_PATH, chunk_size=size, overlap=overlap,
                        chunk_mode=args.chunk_mode, rebuild=True)
        except Exception as exc:
            print(f"  索引失败：{exc}")
            continue
        try:
            retriever = Retriever()
        except Exception as exc:
            print(f"  retriever 加载失败：{exc}")
            continue
        m = recall_at_k(retriever, gold, k=args.top_k)
        m["chunk_size"] = size
        m["overlap"] = overlap
        m["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(m)
        print(f"  -> Recall@{args.top_k}={m['recall_at_10']}  MRR={m['mrr']}  "
              f"({m['seconds']}s)")

    out = ROOT / "ablations" / "chunk_size_scan.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[chunk_size_scan] 全部结果已写入 {out}")
    # 简单汇总
    if results:
        best = max(results, key=lambda x: x["recall_at_10"])
        print(f" 最佳 chunk_size = {best['chunk_size']} "
              f"(Recall@{args.top_k}={best['recall_at_10']})")
        print(" 预期趋势：chunk_size 过小（128）召回边界切散 anchor；")
        print(" 过大的 chunk（1024）单个 chunk 含太多噪声，召回 Top-10 易漏；")
        print(" 256-512 通常是 NNDL 这种长文本的最佳区间。")


if __name__ == "__main__":
    main()
