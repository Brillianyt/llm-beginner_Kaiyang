"""RAG 流水线通用工具。

- 路径常量（PDF / 索引 / 模型 / 缓存）
- 文本规范化（评测时去掉所有空白与 gold_anchors 对齐）
- 简易日志 / 计时器
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
INDEX_DIR = MODELS_DIR / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
PDF_PATH = DATA_DIR / "kb.pdf"
GOLD_QA_PATH = DATA_DIR / "gold_qa.jsonl"

# 模型本地路径（可由 download.py 预先下载）
EMBEDDING_MODEL_DIR = MODELS_DIR / "bge-small-zh-v1.5"
RERANKER_MODEL_DIR = MODELS_DIR / "bge-reranker-base"
# Hugging Face repo 名称（本地不存在时下载）
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# BGE 中文检索模型推荐前缀（query 必加，document 不加）
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 检索 / 生成通用配置
DEFAULT_CHUNK_SIZE = 256
DEFAULT_OVERLAP = 32
DEFAULT_TOP_K = 10        # eval 召回量
DEFAULT_RERANK_K = 20     # 召回候选数（rerank 之前）
DEFAULT_FINAL_K = 4       # 拼进 prompt 的最终片段数
DEFAULT_MAX_CONTEXT_CHARS = 2400  # 上下文总字符上限


def normalize_text(text: str) -> str:
    """去掉所有空白，用于评测 gold anchor 命中与 chunk 文本对齐。

    评测口径明确要求「去掉所有空白后的字符串包含任一 anchor 即命中」，
    因此工具层就用同一套规范化逻辑，避免在评测脚本里再写一遍。
    """
    return re.sub(r"\s+", "", str(text))


def read_jsonl(path: Path) -> list[dict]:
    """逐行读取 JSON Lines；跳过空行；UTF-8 编码。"""
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(__import__("json").loads(line))
    return items


def chunk_sources() -> list[dict]:
    """返回 [{chunk_id, text, source}] 列表。

    chunk_id 用于跨进程 / 跨运行对齐；source 存 PDF 内的页码或文件名。
    """
    raise NotImplementedError("由 indexer 调用；这里只占位避免出现循环 import")


def batched(items: Iterable, n: int) -> Iterator[list]:
    """把可迭代对象切成固定大小的批次（最后一批可能更短）。"""
    batch: list = []
    for it in items:
        batch.append(it)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


@contextmanager
def timer(label: str = ""):
    """上下文管理器式计时器：with timer("build index") as t: ... 打印耗时。"""
    t0 = time.perf_counter()
    yield locals()
    elapse = time.perf_counter() - t0
    print(f"[timer] {label}: {elapse:.2f}s")
