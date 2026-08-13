"""文档切分模块：按字符切 + 段落边界归并。

核心导出
--------
- :func:`chunk_text` —— 按字符数切片的固定大小切分器，带 overlap
- :func:`chunk_paragraphs` —— 先按段落自然边界切，再做长度归并
- :func:`chunk_text_with_boundaries` —— 同时返回每个 chunk 的起止偏移，便于溯源

设计要点
--------
1. ``chunk_size`` / ``overlap`` **以字符计**（不是词元数）。
   自检 ``chunking_sanity`` 按字符平均长度核验，
   必须落在 ``(chunk_size * 0.5, chunk_size * 1.2)`` 才算通过。
2. 重叠区 ``overlap`` 至少为 1，否则会陷入死循环；下限 ``max(0, chunk_size - 1)``。
3. 末尾 chunk 长度可能不足 ``chunk_size``，但**不为空**——自检对平均长度
   的容忍度本就允许末尾短 chunk，不强行丢弃。
4. 段落边界归并 (``chunk_paragraphs``) 是 PDF 抽取的最佳实践：
   直接在字符级切会把句子从中间分开，召回时 anchor 经常被切散。
"""
from __future__ import annotations

import re
from typing import Iterable

# 段落分隔：连续两个以上换行 / 全角空格 / 段首空白
_PARA_SPLIT_RE = re.compile(r"\n\s*\n|　　")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按字符切片的固定大小切分器。

    Parameters
    ----------
    text : str
        输入文本（建议先做空白归一化）。
    chunk_size : int
        每个 chunk 的目标字符数（包含 overlap 部分）。
    overlap : int
        相邻 chunk 之间的重叠字符数，必须 < ``chunk_size``。

    Returns
    -------
    list[str]
        切分后的 chunk 列表。每段至少 1 个字符。

    Notes
    -----
    - ``chunk_size <= 0`` 抛出 ``ValueError``。
    - ``overlap >= chunk_size`` 把 overlap 夹到 ``chunk_size - 1``，
      避免 ``step = 0`` 的死循环。
    - 不做段落 / 句子对齐；如需语义切分请用 :func:`chunk_paragraphs`。
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，得到 {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap 不能为负，得到 {overlap}")
    # 保护：避免 step <= 0 时死循环
    step = max(1, chunk_size - overlap)
    if step > chunk_size:
        step = chunk_size

    cleaned = text or ""
    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + chunk_size, n)
        piece = cleaned[start:end]
        if piece:  # 至少保证非空
            chunks.append(piece)
        if end >= n:
            break
        start += step
    return chunks


def chunk_text_with_boundaries(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """与 :func:`chunk_text` 等价，但每个 chunk 附带起止偏移。

    偏移对应于原始 ``text``（未做空白归一化），用于 PDF 抽取后的
    高亮引用或位置回溯。
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，得到 {chunk_size}")
    step = max(1, chunk_size - overlap)
    chunks: list[dict] = []
    start = 0
    n = len(text or "")
    while start < n:
        end = min(start + chunk_size, n)
        piece = (text or "")[start:end]
        if piece:
            chunks.append({"text": piece, "start": start, "end": end})
        if end >= n:
            break
        start += step
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """粗暴的段落切分：连续两个换行 / 全角空格视为段落边界。

    对 PDF 抽取后的文本已经足够：LaTeX 段落、章节标题、列表项之间
    都会留下空行；表格行内通常没有空行，会被并入同一段。
    """
    if not text:
        return []
    parts = _PARA_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_paragraphs(text: str, chunk_size: int, overlap: int) -> list[str]:
    """段落边界归并：先把段落作为最小单元，再贪心合并到接近 ``chunk_size``。

    必要时会落回 :func:`chunk_text` 在单个段落内再次切分；末尾不足
    ``chunk_size`` 的部分仍返回，保证自检对 chunk 数的下限。

    Parameters
    ----------
    text : str
        原始文本。
    chunk_size : int
        目标字符数。
    overlap : int
        跨段重叠的字符数（实现上只在落回字符切时使用；段落级合并按整段吞）。
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，得到 {chunk_size}")
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append("".join(buf))
        buf = []
        buf_len = 0

    for p in paragraphs:
        if len(p) >= chunk_size:
            # 当前段落比目标还长：先 flush 累计，再对段落内做字符切
            flush()
            for sub in chunk_text(p, chunk_size=chunk_size, overlap=overlap):
                chunks.append(sub)
            continue
        # 贪心：还能塞就塞，塞不下就 flush
        if buf_len + len(p) + 1 > chunk_size and buf:
            flush()
        buf.append(p)
        buf_len += len(p) + (1 if buf_len else 0)
    flush()
    return chunks


def chunk_documents(docs: Iterable[tuple[str, str]],
                    chunk_size: int, overlap: int,
                    mode: str = "char") -> list[dict]:
    """批量切分多个文档，返回 ``[{text, source, chunk_id}]``。

    Parameters
    ----------
    docs : Iterable[tuple[str, str]]
        (source_label, text) 列表；source_label 会写入每条 chunk，
        检索返回时方便用户定位来源。
    chunk_size, overlap : int
        透传给具体切分函数。
    mode : {"char", "paragraph"}
        ``char`` 使用 :func:`chunk_text`；``paragraph`` 使用
        :func:`chunk_paragraphs`。
    """
    if mode not in {"char", "paragraph"}:
        raise ValueError(f"不支持的切分模式: {mode}")
    out: list[dict] = []
    cid = 0
    for source, text in docs:
        if mode == "char":
            pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        else:
            pieces = chunk_paragraphs(text, chunk_size=chunk_size, overlap=overlap)
        for p in pieces:
            out.append({"text": p, "source": source, "chunk_id": cid})
            cid += 1
    return out
