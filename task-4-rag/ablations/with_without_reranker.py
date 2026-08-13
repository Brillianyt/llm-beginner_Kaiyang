"""S2: 加 / 不加 reranker 的 Recall + 生成忠实性对比。

设计要点
--------
1. 两侧使用同一索引（chunk_size 固定）；唯一区别是 reranker 是否介入。
2. 召回阶段 k 给 20，rerank 取 top 4 ~ 10，对比 gold anchor 命中数。
3. faithfulness 简化用「答案是否引用 prompt 给的 source 标签」做粗略估计。
   完整 faithfulness 应跑 RAGAS（见 S4），本消融只给出方向性观察。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retriever import Retriever
from src.reranker import BGEReranker, rerank
from src.generator import QwenGenerator, merge_and_truncate_contexts
from src.utils import GOLD_QA_PATH, normalize_text


def load_gold():
    out = []
    with GOLD_QA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def recall_curve(results: list[dict], gold: list[dict]) -> dict:
    """对给定 top-k 列表统计 Recall@k / MRR。"""
    hit = {1: 0, 3: 0, 5: 0, 10: 0}
    rr = 0.0
    for item, top in zip(gold, results):
        anchors = [normalize_text(a) for a in item["gold_anchors"]]
        rank = None
        for i, r in enumerate(top, 1):
            txt = normalize_text(r.get("text", ""))
            if any(a and a in txt for a in anchors):
                rank = i
                break
        if rank:
            for k in hit:
                if rank <= k:
                    hit[k] += 1
            rr += 1 / rank
    n = len(gold)
    return {
        "recall_at_1": round(hit[1] / n, 3),
        "recall_at_3": round(hit[3] / n, 3),
        "recall_at_5": round(hit[5] / n, 3),
        "recall_at_10": round(hit[10] / n, 3),
        "mrr": round(rr / n, 3),
    }


def faithfulness_proxy(answer: str, sources: list[dict]) -> float:
    """粗略 faithful 估计：answer 中是否引用了任一 source 标签。

    真实 faithfulness 应跑 RAGAS，本脚本只做方向性比较。
    """
    if not answer or not sources:
        return 0.0
    labels = {s.get("source", "") for s in sources}
    a = normalize_text(answer)
    matched = sum(1 for l in labels if l and normalize_text(l) in a)
    return round(matched / max(len(labels), 1), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rerank-cand", type=int, default=20,
                    help="rerank 前召回候选数")
    ap.add_argument("--with-faith", action="store_true",
                    help="额外生成短答案，估算 faithfulness")
    args = ap.parse_args()

    gold = load_gold()
    print(f"[with_without_reranker] gold QA: {len(gold)} 条")

    retriever = Retriever()
    rr = None
    try:
        rr = BGEReranker()
    except Exception as exc:
        print(f"[with_without_reranker] reranker 加载失败: {exc}")
        rr = None

    # 1) 不加 reranker
    no_rr_lists = [retriever.retrieve(item["question"], k=args.top_k)
                   for item in gold]
    no_rr_metrics = recall_curve(no_rr_lists, gold)
    print(f"\n[no rerank]  {no_rr_metrics}")

    # 2) 加 reranker
    if rr is not None:
        with_rr_lists = []
        for item in gold:
            cand = retriever.retrieve(item["question"], k=max(args.top_k, args.rerank_cand))
            top = rerank(item["question"], cand, top_k=args.rerank_cand,
                         final_k=args.top_k, reranker=rr)
            with_rr_lists.append(top)
        with_rr_metrics = recall_curve(with_rr_lists, gold)
        print(f"[with rerank] {with_rr_metrics}")
    else:
        with_rr_metrics = None
        print("[with rerank] skipped (reranker unavailable)")

    # 3) faithfulness 粗略估计（可选）
    faithfulness = {}
    if args.with_faith:
        try:
            gen = QwenGenerator()
        except Exception as exc:
            print(f"  generator 加载失败: {exc}")
            gen = None
        if gen is not None:
            for label, srcs in (("no_rerank", no_rr_lists),
                                ("with_rerank", with_rr_lists or [])):
                if not srcs:
                    continue
                scores = []
                for item, top in zip(gold[:10], srcs[:10]):
                    ctx = merge_and_truncate_contexts(top)
                    res = gen.generate(item["question"], ctx)
                    scores.append(faithfulness_proxy(res.answer, ctx))
                faithfulness[label] = round(sum(scores) / len(scores), 3)
            print(f"\n[faithfulness proxy] {faithfulness}")

    out = {
        "no_rerank": no_rr_metrics,
        "with_rerank": with_rr_metrics,
        "faithfulness_proxy": faithfulness,
    }
    out_path = ROOT / "ablations" / "with_without_reranker.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
