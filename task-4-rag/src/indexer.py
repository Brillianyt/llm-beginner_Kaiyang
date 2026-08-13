"""BGE embedding + FAISS 索引构建。

核心导出
--------
- :class:`BGEEmbedder` —— 加载 BGE 中文 embedding 模型，统一接口
- :func:`build_index` —— 从 PDF 抽取 → 切分 → embedding → FAISS，序列化到磁盘
- :func:`load_index` —— 读取已序列化的索引与 chunk 元数据

设计要点
--------
1. **BGE 查询前缀**：query 必加 ``"为这个句子生成表示以用于检索相关文章："``，
   document 不加；漏加或加错会让 Recall 掉 5-10 个点。
2. **L2 归一化**：FAISS ``IndexFlatIP``（内积）在向量已归一化时等价于 cosine；
   归一化是做对了的内积检索，忘了归一化检索分数全乱。
3. **可序列化**：embedding 与 chunk 元数据分别落到
   ``models/index/faiss.index`` 和 ``models/index/chunks.json``，
   避免每次启动都重算 embedding（CPU 上重建索引需要几分钟）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .utils import (
    BGE_QUERY_PREFIX,
    CHUNKS_FILE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    EMBEDDING_MODEL_DIR,
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    INDEX_FILE,
    PDF_PATH,
    batched,
    timer,
)
from .chunker import chunk_documents
from .pdf_loader import extract_pdf_pages


# ----------------------------- Embedder -----------------------------------


class BGEEmbedder:
    """BGE 中文 embedding 模型薄封装。

    优先加载本地 ``models/bge-small-zh-v1.5``，如果不存在就 fallback 到
    Hugging Face 仓库名（需要在可联网环境，或预先 ``export HF_ENDPOINT``）。
    """

    def __init__(self, model_name: str | None = None,
                 device: str | None = None,
                 normalize: bool = True) -> None:
        self.model_name = model_name or _resolve_model_dir(EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_NAME)
        self.normalize = normalize
        self._model = None
        self.device = device  # 留 None 让 sentence-transformers 自己挑

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers；请 pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        if self.normalize:
            # 默认 BGE 推荐 cosine，库默认就已经 normalize_embeddings=True
            self._model.encode(["测试"], normalize_embeddings=True)

    def encode(self, texts: list[str], is_query: bool = False,
               batch_size: int = 64) -> np.ndarray:
        """把一段文本列表编码成 ``[N, D]`` float32 矩阵。

        ``is_query=True`` 时自动加 BGE 查询前缀。
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._load()
        if is_query:
            prefixed = [BGE_QUERY_PREFIX + t for t in texts]
        else:
            prefixed = list(texts)
        emb = self._model.encode(  # type: ignore[union-attr]
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return emb.astype(np.float32)


def _resolve_model_dir(local_dir: Path, hf_name: str) -> str:
    """优先本地目录，缺失则回退到 HF name（让 sentence-transformers 自取）。"""
    if local_dir.exists() and any(local_dir.iterdir()):
        return str(local_dir)
    return hf_name


# ----------------------------- Index build --------------------------------


def _build_faiss(emb: np.ndarray):
    """返回 ``IndexFlatIP``（内积）；向量已 L2 normalize 时等价于 cosine。"""
    import faiss  # type: ignore
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(emb))
    return index


def build_index(pdf_path: Path | str = PDF_PATH,
                chunk_size: int = DEFAULT_CHUNK_SIZE,
                overlap: int = DEFAULT_OVERLAP,
                chunk_mode: str = "paragraph",
                batch_size: int = 64,
                rebuild: bool = False) -> dict:
    """从 PDF 构建索引并序列化到 disk。

    Parameters
    ----------
    pdf_path : Path | str
        输入 PDF 路径，缺失时直接 ``ValueError``。
    chunk_size, overlap : int
        切分参数。
    chunk_mode : {"char", "paragraph"}
        段落归并或纯字符切，前者更适合 PDF 文本。
    rebuild : bool
        强制重建；默认为 ``False``（已存在时直接复用）。

    Returns
    -------
    dict
        ``{"index_path": str, "chunks_path": str, "num_chunks": int}``
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not rebuild and INDEX_FILE.exists() and CHUNKS_FILE.exists():
        n = len(json.loads(CHUNKS_FILE.read_text(encoding="utf-8")))
        print(f"[indexer] 复用现有索引: {INDEX_FILE} ({n} chunks)")
        return {"index_path": str(INDEX_FILE), "chunks_path": str(CHUNKS_FILE),
                "num_chunks": n, "cached": True}

    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"PDF 不存在或为空: {pdf_path}；请先跑 data/download.py --skip-models 拉 PDF"
        )

    with timer("PDF 抽取"):
        pages = extract_pdf_pages(pdf_path)
    if not pages:
        raise RuntimeError("PDF 抽取返回空；请检查 pypdf 是否安装或 PDF 是否损坏")
    docs = [(p["source"], p["text"]) for p in pages]
    print(f"[indexer] 抽取 {len(docs)} 页")

    with timer("切分"):
        chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap,
                                 mode=chunk_mode)
    if not chunks:
        raise RuntimeError("切分得到 0 个 chunk，请检查 chunk_size / PDF 文本")
    print(f"[indexer] 生成 {len(chunks)} 个 chunk "
          f"(chunk_size={chunk_size}, overlap={overlap}, mode={chunk_mode})")

    with timer("embedding"):
        embedder = BGEEmbedder()
        all_emb: list[np.ndarray] = []
        for batch in batched([c["text"] for c in chunks], batch_size):
            emb = embedder.encode(batch, is_query=False)
            all_emb.append(emb)
        emb = np.concatenate(all_emb, axis=0).astype(np.float32)
    print(f"[indexer] embedding shape = {emb.shape}")

    with timer("FAISS 索引"):
        index = _build_faiss(emb)

    # 序列化
    import faiss  # type: ignore
    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[indexer] 索引写入 {INDEX_FILE}")
    print(f"[indexer] chunk 元数据写入 {CHUNKS_FILE}")
    return {"index_path": str(INDEX_FILE), "chunks_path": str(CHUNKS_FILE),
            "num_chunks": len(chunks), "cached": False}


def load_index():
    """读取索引与 chunk 列表；缺失时抛 ``FileNotFoundError``。"""
    import faiss  # type: ignore
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"未找到索引: {INDEX_FILE} / {CHUNKS_FILE}；请先 build_index()"
        )
    index = faiss.read_index(str(INDEX_FILE))
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return index, chunks
