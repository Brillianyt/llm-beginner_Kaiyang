"""稠密向量召回器。

核心导出
--------
- :class:`Retriever` —— 单例式封装「embedding + FAISS 索引 + BGE 前缀」，
  对外只暴露 :meth:`retrieve(query, k)`。

设计要点
--------
1. 构造时立即加载索引，避免评测时再延迟；
2. ``retrieve`` 严格按入参 ``k`` 召回；
3. 返参 ``[{"text", "score", "source"}]``，source 透传
   :class:`src.chunker` 写入的 ``kb.pdf#p{N}`` 标签，方便用户定位。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .indexer import BGEEmbedder, load_index
from .utils import DEFAULT_TOP_K, EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_NAME


def _resolve_model_dir(local_dir, hf_name: str) -> str:
    if local_dir.exists() and any(local_dir.iterdir()):
        return str(local_dir)
    return hf_name


class Retriever:
    """BGE 召回封装。

    Parameters
    ----------
    embedder : BGEEmbedder | None
        复用现成 embedder，None 时内部按 ``EMBEDDING_MODEL_DIR`` 加载。
    index, chunks : 来自 :func:`src.indexer.load_index`
        显式注入用于消融 / 多索引场景。
    """

    def __init__(self,
                 embedder: Optional[BGEEmbedder] = None,
                 index=None,
                 chunks: Optional[list[dict]] = None,
                 auto_load: bool = True) -> None:
        if index is None or chunks is None:
            if not auto_load:
                raise ValueError("未提供 index / chunks，且 auto_load=False")
            index, chunks = load_index()
        self.index = index
        self.chunks = chunks
        self.embedder = embedder or BGEEmbedder(
            model_name=_resolve_model_dir(EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_NAME)
        )

    def embed_query(self, query: str) -> np.ndarray:
        """对单条 query 编码：自动加 BGE 查询前缀。"""
        emb = self.embedder.encode([query], is_query=True)
        # encode 内部会做 L2 normalize，归一化后内积 ≡ cosine 相似度
        return emb

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
        """返回 top-k 召回。

        Returns
        -------
        list[dict]
            ``{"text": str, "score": float, "source": str}``，
            ``score`` 是 cosine 相似度（范围约 [-1, 1]，越接近 1 越相关）。
        """
        if not query or not query.strip():
            return []
        emb = self.embed_query(query)  # shape [1, D]
        # FAISS 要求 2D contiguous
        scores, idxs = self.index.search(np.ascontiguousarray(emb), k)
        out: list[dict] = []
        for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0 or i >= len(self.chunks):
                continue
            chunk = self.chunks[i]
            out.append({
                "text": chunk["text"],
                "score": float(s),
                "source": chunk.get("source", "unknown"),
                "chunk_id": chunk.get("chunk_id", i),
            })
        return out

    def size(self) -> int:
        """索引里的 chunk 数。"""
        return len(self.chunks)
