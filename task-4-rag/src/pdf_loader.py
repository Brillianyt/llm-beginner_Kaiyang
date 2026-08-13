"""PDF 文本抽取模块。

封装 :mod:`pypdf` 提供按页抽取 + 整体拼接。
- 缺失 PDF / pypdf 时返回 ``None`` 并打印降级提示。
- 段落之间插入空行，配合 :func:`src.chunker.chunk_paragraphs` 使用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .utils import PDF_PATH


def extract_pdf_pages(pdf_path: Path | str = PDF_PATH) -> Optional[list[dict]]:
    """按页抽取 PDF 文本。

    Returns
    -------
    list[dict] | None
        每个元素 ``{"source": "kb.pdf#p{idx}", "text": str}``；
        抽取失败 / 文件缺失时返回 ``None``。
    """
    path = Path(pdf_path)
    if not path.exists() or path.stat().st_size == 0:
        print(f"[pdf_loader] {path} 不存在或为空，跳过 PDF 抽取")
        return None
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        print("[pdf_loader] 未安装 pypdf，跳过 PDF 抽取；请 pip install pypdf")
        return None
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"[pdf_loader] 打开 PDF 失败 {path}: {exc}")
        return None
    pages: list[dict] = []
    for idx, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            print(f"[pdf_loader] 第 {idx} 页抽取失败: {exc}")
            text = ""
        # 段落之间补一个空行，便于下游 chunk_paragraphs 切分
        text = text.replace("\r\n", "\n").strip()
        pages.append({"source": f"kb.pdf#p{idx}", "text": text})
    return pages


def extract_pdf_text(pdf_path: Path | str = PDF_PATH) -> Optional[str]:
    """把所有页拼成单个字符串；段落间补空行。"""
    pages = extract_pdf_pages(pdf_path)
    if not pages:
        return None
    return "\n\n".join(p["text"] for p in pages if p["text"])
