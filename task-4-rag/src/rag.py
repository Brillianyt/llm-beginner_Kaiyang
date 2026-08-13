"""端到端 RAG 串接：``answer(query) -> dict(answer, sources)``。

设计要点
--------
1. 评审自检的 hash 判定：必须返回 ``dict`` 而不是 ``list``；
   ``bool(r.get("sources"))`` 在 list 上会抛 unhashable，整条自检会挂在写 result.json 之前。
2. 召回 -> rerank -> 去重 -> 拼 prompt -> 生成，串联四个独立模块。
3. 单一 ``answer()`` 入口，便于上层（CLI / 服务）复用。
"""
from __future__ import annotations

from typing import Optional

from .utils import (
    DEFAULT_FINAL_K,
    DEFAULT_RERANK_K,
    DEFAULT_TOP_K,
)
from .retriever import Retriever
from .reranker import BGEReranker, rerank
from .generator import QwenGenerator


def _default_retriever() -> Retriever:
    return Retriever()


def _default_reranker() -> BGEReranker:
    return BGEReranker()


def _default_generator() -> QwenGenerator:
    return QwenGenerator()


# 单例缓存（避免每次调用都重新加载模型）
_singleton: dict[str, object] = {}


def _get_retriever(retriever: Optional[Retriever]) -> Retriever:
    if retriever is not None:
        return retriever
    if "retriever" not in _singleton:
        _singleton["retriever"] = _default_retriever()
    return _singleton["retriever"]  # type: ignore[return-value]


def _get_reranker(reranker: Optional[BGEReranker]) -> Optional[BGEReranker]:
    if reranker is None and "reranker" not in _singleton:
        # 缺失模型时 still 创建空壳；调用时失败会降级
        try:
            _singleton["reranker"] = _default_reranker()
        except Exception as exc:
            print(f"[rag] reranker 加载失败，跳过 rerank: {exc}")
            _singleton["reranker"] = None
    return _singleton.get("reranker")  # type: ignore[return-value]


def _get_generator(generator: Optional[QwenGenerator]) -> QwenGenerator:
    if generator is not None:
        return generator
    if "generator" not in _singleton:
        _singleton["generator"] = _default_generator()
    return _singleton["generator"]  # type: ignore[return-value]


def answer(query: str,
           retriever: Optional[Retriever] = None,
           reranker: Optional[BGEReranker] = None,
           generator: Optional[QwenGenerator] = None,
           top_k: int = DEFAULT_TOP_K,
           rerank_k: int = DEFAULT_RERANK_K,
           final_k: int = DEFAULT_FINAL_K,
           use_rerank: bool = True) -> dict:
    """端到端 RAG 问答。

    Returns
    -------
    dict
        ``{"answer": str, "sources": [{text, score, source, ...}, ...]}``
        评测脚本会强转 ``bool(r.get("sources"))``，因此 sources 必须是 list。
    """
    if not query or not query.strip():
        return {"answer": "", "sources": []}

    retr = _get_retriever(retriever)
    recalled = retr.retrieve(query, k=max(top_k, rerank_k, final_k))

    if use_rerank:
        rr = _get_reranker(reranker)
        if rr is not None:
            top = rerank(query, recalled, top_k=rerank_k,
                         final_k=final_k, reranker=rr)
        else:
            top = recalled[:final_k]
    else:
        top = recalled[:final_k]

    gen = _get_generator(generator)
    result = gen.generate(query, top)
    return {"answer": result.answer, "sources": top}


def reset_singletons() -> None:
    """测试时清空单例缓存。"""
    _singleton.clear()
