"""S3: Query rewriting / HyDE 增强。

两种 query 增强策略对比：

1. **Query Rewriting**（轻量、不依赖生成）
   - 拼一个扩写模板，让查询句子更接近文档语气。
   - 例如："什么是 X？" → "本文介绍 X 的概念和原理。X ..."

2. **HyDE**（Hypothetical Document Embeddings）
   - 让 LLM 生成一段假设性短答案文本，用它去检索。
   - 检索阶段直接拿到「答案风格」的向量，理论上更接近真实文档。

本脚本只测「Query Rewriting」一行串式模板；HyDE 是可选高阶实验，需要
本地 LLM 推理，缺模型时自动跳过。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retriever import Retriever
from src.utils import BGE_QUERY_PREFIX, GOLD_QA_PATH, normalize_text


def rewrite_lite(query: str) -> str:
    """轻量改写：把问句扩展为陈述性短句，串联同义关键词。

    思路：BGE 对陈述句的检索效果往往比问句更好（训练语料偏陈述）。
    加一个固定的扩写模板即可拿到稳定的 Recall 提升。
    """
    rules = [
        (r"是什么", "的定义与基本概念"),
        (r"为什么", "的原因与原理"),
        (r"有哪些", "的主要类别与特点"),
        (r"如何", "的实现方法与步骤"),
        (r"怎么", "的做法与注意事项"),
        (r"区别", "之间的差异与联系"),
    ]
    expansion = ""
    for pat, exp in rules:
        if re.search(pat, query):
            expansion = exp
            break
    if not expansion:
        expansion = "的核心概念与原理"
    return f"本文介绍{query}{expansion}。"


def recall_at_k(retriever: Retriever, queries: list[tuple[str, list[str]]],
                k: int = 10) -> dict:
    hit = 0
    rr = 0.0
    for q, anchors in queries:
        results = retriever.retrieve(q, k=k)
        rank = None
        for i, r in enumerate(results, 1):
            txt = normalize_text(r.get("text", ""))
            if any(a and a in txt for a in anchors):
                rank = i
                break
        if rank:
            hit += 1
            rr += 1 / rank
    n = len(queries)
    return {"recall_at_10": round(hit / n, 3), "mrr": round(rr / n, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--hyde", action="store_true", help="额外跑 HyDE（需要 LLM）")
    args = ap.parse_args()

    items = []
    with GOLD_QA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    queries = [(item["question"], [normalize_text(a) for a in item["gold_anchors"]])
               for item in items]
    print(f"[query_rewriting] gold QA: {len(queries)} 条")

    retriever = Retriever()

    # 1) 原始 query
    raw = [(q, a) for q, a in queries]
    m_raw = recall_at_k(retriever, raw, k=args.top_k)
    print(f"\n[raw query]   {m_raw}")

    # 2) Query rewriting
    rewritten = [(rewrite_lite(q), a) for q, a in queries]
    m_rw = recall_at_k(retriever, rewritten, k=args.top_k)
    print(f"[rewritten]   {m_rw}")

    delta = round(m_rw["recall_at_10"] - m_raw["recall_at_10"], 3)
    print(f"\nΔ Recall@10 = {delta:+.3f}")

    # 3) HyDE（可选）
    m_hyde = None
    if args.hyde:
        try:
            from src.generator import QwenGenerator
        except Exception as exc:
            print(f"[query_rewriting] 导入 generator 失败: {exc}")
        else:
            try:
                gen = QwenGenerator()
            except Exception as exc:
                print(f"[query_rewriting] generator 加载失败: {exc}")
                gen = None
            if gen is not None:
                hyde = []
                for q, a in queries:
                    fake_ctx = [{
                        "text": "本段用于提示 LLM 生成假设性短答案。",
                        "source": "hyde_prompt",
                    }]
                    res = gen.generate(q, fake_ctx)
                    # 用生成的答案 + 原始 query 一起编码
                    new_q = q + " " + res.answer[:200]
                    hyde.append((new_q, a))
                m_hyde = recall_at_k(retriever, hyde, k=args.top_k)
                print(f"[HyDE]         {m_hyde}")

    out = {"raw": m_raw, "rewritten": m_rw, "hyde": m_hyde,
           "delta_recall_at_10": delta}
    out_path = ROOT / "ablations" / "query_rewriting.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
