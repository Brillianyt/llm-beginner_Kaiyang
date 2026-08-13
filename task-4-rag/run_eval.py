"""在 30 条 NNDL gold QA 上跑检索评测。

支持
----
- ``--no-rerank``：跳过 reranker，对比纯召回
- ``--top-k``：召回量（默认 10，与 eval/run.py 对齐）
- ``--output``：结果写到 JSON（默认 ``eval/result.json``）

Examples
--------
    python run_eval.py
    python run_eval.py --no-rerank
    python run_eval.py --top-k 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.retriever import Retriever
from src.reranker import BGEReranker, rerank
from src.utils import DEFAULT_TOP_K, GOLD_QA_PATH, normalize_text


def evaluate(retriever: Retriever, gold: list[dict],
             top_k: int = DEFAULT_TOP_K,
             reranker: BGEReranker | None = None,
             use_rerank: bool = True,
             verbose: bool = False) -> dict:
    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    details = []
    for item in gold:
        q = item["question"]
        anchors = [normalize_text(a) for a in item.get("gold_anchors", [])]
        if use_rerank and reranker is not None:
            # 召回更多候选再 rerank 取 top_k
            cand = retriever.retrieve(q, k=max(top_k, 20))
            results = rerank(q, cand, top_k=20, final_k=top_k, reranker=reranker)
        else:
            results = retriever.retrieve(q, k=top_k)

        matched_rank = None
        matched_anchor = None
        for rank, r in enumerate(results, 1):
            text = normalize_text(r.get("text", ""))
            for a in anchors:
                if a and a in text:
                    matched_rank = rank
                    matched_anchor = a
                    break
            if matched_rank:
                break
        if matched_rank:
            for k in hit_counts:
                if matched_rank <= k:
                    hit_counts[k] += 1
            reciprocal_ranks.append(1 / matched_rank)
        else:
            reciprocal_ranks.append(0)
        rec = {
            "id": item.get("id"),
            "hit": matched_rank is not None,
            "rank": matched_rank,
            "matched_anchor": matched_anchor,
            "source_file": item.get("source_file"),
        }
        details.append(rec)
        if verbose:
            rec2 = {
                **rec,
                "top_sources": [r.get("source") for r in results[:3]],
            }
            print("  hit={hit} rank={rank} anchor={matched_anchor}".format(**rec))

    n = len(gold)
    metrics = {
        f"recall_at_{k}": round(hit_counts[k] / n, 3) for k in sorted(hit_counts)
    }
    metrics["mrr"] = round(sum(reciprocal_ranks) / n, 3)
    return {"n": n, "use_rerank": use_rerank, "top_k": top_k, **metrics,
            "details": details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true", help="跳过 reranker")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--output", type=str, default="eval/result.json")
    args = ap.parse_args()

    if not GOLD_QA_PATH.exists():
        print(f"[run_eval] 缺少 {GOLD_QA_PATH}；请先跑 data/download.py")
        sys.exit(1)

    gold = []
    with GOLD_QA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold.append(json.loads(line))
    print(f"[run_eval] 加载 {len(gold)} 条 gold QA")

    try:
        retriever = Retriever()
    except Exception as exc:
        print(f"[run_eval] 加载索引失败: {exc}")
        print("  请先跑 python build_index.py 建好索引")
        sys.exit(1)

    reranker = None
    if not args.no_rerank:
        try:
            reranker = BGEReranker()
        except Exception as exc:
            print(f"[run_eval] reranker 加载失败: {exc}；按无 reranker 跑")

    for use_rerank in (False, True) if (not args.no_rerank and reranker) else (
            (True,) if args.no_rerank else (False,)):
        res = evaluate(retriever, gold, top_k=args.top_k,
                       reranker=reranker, use_rerank=use_rerank)
        label = "with_rerank" if use_rerank else "no_rerank"
        print(f"\n[{label}] n={res['n']} top_k={res['top_k']}")
        for k in (1, 3, 5, 10):
            print(f"  Recall@{k:>2} = {res[f'recall_at_{k}']:.3f}")
        print(f"  MRR        = {res['mrr']:.3f}")

    # 主结果：默认带 reranker（若有），否则 no rerank
    final = evaluate(retriever, gold, top_k=args.top_k,
                     reranker=reranker, use_rerank=not args.no_rerank and reranker is not None)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
