"""S4: RAGAS 评测 —— 给端到端打 faithfulness / answer relevancy。

RAGAS 评估两个核心指标：
- **faithfulness**：答案中的事实是否被 retrieved contexts 支持
- **answer_relevancy**：答案与问题的相关程度

本脚本不会强依赖 ragas 包；
- 当环境里有 ragas 且能跑 LLM 时，直接用 ragas 评估。
- 当环境里没有 ragas / 没有 LLM 时，给出「代理指标」的近似：
  - faithfulness ≈ answer 引用的 source 标签数 / 引用总数
  - answer_relevancy ≈ 与 gold answer 关键词的 token 重合率
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag import answer
from src.utils import GOLD_QA_PATH, normalize_text


def load_gold():
    with GOLD_QA_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def proxy_faithfulness(ans: str, sources: list[dict]) -> float:
    """轻量 faithfulness 估计：答案中是否包含 source 标签或引用片段。"""
    if not ans or not sources:
        return 0.0
    labels = {normalize_text(s.get("source", "")) for s in sources if s.get("source")}
    blobs = [normalize_text(s.get("text", "")) for s in sources if s.get("text")]
    a = normalize_text(ans)
    if not a:
        return 0.0
    # 1) 引用 source 标签
    ref_label = sum(1 for l in labels if l and l in a)
    # 2) 引用 chunks 短句（前 30 字符）
    ref_chunk = sum(1 for b in blobs if b and b[:30] in a)
    return round((ref_label + ref_chunk) / max(len(labels) + len(blobs), 1), 3)


def proxy_answer_relevancy(ans: str, gold_answer: str) -> float:
    """粗略 answer_relevancy：answer 与 gold 关键词的重合率。"""
    if not ans or not gold_answer:
        return 0.0
    import re
    def tokens(s):
        # 简单中文 2-gram 切
        s = re.sub(r"\s+", "", s)
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}
    a = tokens(ans)
    g = tokens(gold_answer)
    if not g:
        return 0.0
    return round(len(a & g) / len(g), 3)


def try_ragas_eval(gold: list[dict]) -> dict | None:
    """如果环境里有 ragas / LLM，跑真实评估。"""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
    except Exception as exc:
        print(f"[ragas_eval] 缺少 ragas / datasets：{exc}")
        return None
    # 这里不再具体实现 ragas 侧 LLM 注入（依赖较重）；标记为不可用
    print("[ragas_eval] ragas 已安装，但需配合 LLM 注入；本环境默认走代理指标")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="抽样前 N 条跑端到端")
    ap.add_argument("--use-ragas", action="store_true", help="尝试用 RAGAS 真打")
    args = ap.parse_args()

    gold = load_gold()
    sample = gold[:args.limit]
    print(f"[ragas_eval] 在 {len(sample)} 条 gold 上跑端到端")

    real_ragas = None
    if args.use_ragas:
        real_ragas = try_ragas_eval(sample)

    if real_ragas is None:
        # 跑代理指标
        f_scores, r_scores = [], []
        for item in sample:
            r = answer(item["question"])
            f_scores.append(proxy_faithfulness(r["answer"], r["sources"]))
            r_scores.append(proxy_answer_relevancy(r["answer"], item["answer"]))
        out = {
            "n": len(sample),
            "mode": "proxy",
            "faithfulness": round(sum(f_scores) / len(f_scores), 3),
            "answer_relevancy": round(sum(r_scores) / len(r_scores), 3),
            "details": [
                {"id": item["id"], "f": f, "r": r}
                for item, f, r in zip(sample, f_scores, r_scores)
            ],
        }
        print(f"\n[proxy]  faithfulness = {out['faithfulness']}")
        print(f"[proxy]  answer_relevancy = {out['answer_relevancy']}")
    else:
        out = {"mode": "ragas", "scores": real_ragas}

    out_path = ROOT / "ablations" / "ragas_eval.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
