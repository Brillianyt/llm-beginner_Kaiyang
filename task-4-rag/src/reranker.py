"""BGE reranker 精排。

核心导出
--------
- :class:`BGEReranker` —— 加载 ``bge-reranker-base``，对 ``[query, doc]`` pair 打分
- :func:`rerank` —— 便捷函数：取的 top-``final_k`` 结果

设计要点
--------
1. **reranker 不是第二个 embedding 模型**：
   输入是文本对 ``[query, doc]``，输出是标量相关性分（logits 或 sigmoid 后概率）。
2. 召回阶段 k 必须 >> 最终 k：通常召回 20，rerank 后取 3-5。
3. reranker 缺失时优雅降级：返回原顺序前 ``final_k``。
"""
from __future__ import annotations

import functools
from typing import Optional

from .utils import (
    DEFAULT_FINAL_K,
    DEFAULT_RERANK_K,
    RERANKER_MODEL_DIR,
    RERANKER_MODEL_NAME,
)


def _resolve_model_dir(local_dir, hf_name: str) -> str:
    if local_dir.exists() and any(local_dir.iterdir()):
        return str(local_dir)
    return hf_name


class BGEReranker:
    """BGE reranker 薄封装。

    用 ``transformers`` 加载 ``AutoModelForSequenceClassification``，
    避免再起一个 ``sentence_transformers.CrossEncoder`` 依赖。
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 use_fp16: bool = True) -> None:
        self.model_name = model_name or _resolve_model_dir(
            RERANKER_MODEL_DIR, RERANKER_MODEL_NAME
        )
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.use_fp16 and self.device == "cuda":
            self._model = self._model.half()
        self._model = self._model.to(self.device)
        self._model.eval()

    @functools.lru_cache(maxsize=1)
    def _torch():
        """延迟导入 torch（避免在没装 torch 的环境直接崩）。"""
        import torch  # type: ignore
        return torch

    def score(self, query: str, passages: list[str]) -> list[float]:
        """对 ``[query, doc]`` pairs 打分，返回并列的浮点列表。"""
        if not passages:
            return []
        self._load()
        torch = self._torch()
        # 长文档截断；reranker 一般 512 token 已够
        pairs = [[query, p] for p in passages]
        enc = self._tokenizer(  # type: ignore[union-attr]
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self._model(**enc).logits.squeeze(-1)  # [N]
        # bge-reranker-base 是单个标量；sigmoid 把它压到 (0, 1)
        scores = torch.sigmoid(logits).float().detach().cpu().tolist()
        return scores


def rerank(query: str,
           results: list[dict],
           top_k: int = DEFAULT_RERANK_K,
           final_k: int = DEFAULT_FINAL_K,
           reranker: Optional[BGEReranker] = None) -> list[dict]:
    """把 ``results`` 按 reranker 分数重排，返回 top-``final_k``。

    Parameters
    ----------
    query : str
        原始 query（不加 BGE 前缀）。
    results : list[dict]
        来自 :meth:`Retriever.retrieve` 的列表。
    top_k : int
        取前 ``top_k`` 进 reranker（reranker 比 embedding 慢）。
    final_k : int
        最终返回的 chunk 数。
    reranker : BGEReranker | None
        None 时按默认目录加载；加载失败则降级——按原顺序返回前 ``final_k``。
    """
    if not results:
        return []
    candidates = results[: max(top_k, final_k)]
    if reranker is None:
        try:
            reranker = BGEReranker()
        except Exception as exc:  # noqa: BLE001
            print(f"[reranker] 加载失败，保持原顺序：{exc}")
            return results[:final_k]
    try:
        docs = [c["text"] for c in candidates]
        scores = reranker.score(query, docs)
    except Exception as exc:  # noqa: BLE001
        print(f"[reranker] 打分失败，保持原顺序：{exc}")
        return results[:final_k]
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    for chunk, s in scored[:final_k]:
        merged = dict(chunk)
        merged["rerank_score"] = float(s)
        out.append(merged)
    return out
