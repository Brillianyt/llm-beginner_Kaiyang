"""一键构建索引：PDF -> chunk -> embedding -> FAISS。

Examples
--------
    python build_index.py                    # 复用已有索引
    python build_index.py --rebuild          # 强制重建
    python build_index.py --chunk-size 512   # 自定义 chunk_size
    python build_index.py --chunk-mode char  # 纯字符切（不用段落归并）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.indexer import build_index
from src.utils import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, PDF_PATH


def main():
    ap = argparse.ArgumentParser(description="RAG 一键索引构建")
    ap.add_argument("--rebuild", action="store_true", help="强制重建索引")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                    help=f"chunk 字符数（默认 {DEFAULT_CHUNK_SIZE}）")
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                    help=f"chunk overlap 字符数（默认 {DEFAULT_OVERLAP}）")
    ap.add_argument("--chunk-mode", choices=["char", "paragraph"],
                    default="paragraph", help="切分模式：char / paragraph")
    ap.add_argument("--pdf", type=str, default=str(PDF_PATH), help="PDF 路径")
    args = ap.parse_args()

    info = build_index(
        pdf_path=args.pdf,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        chunk_mode=args.chunk_mode,
        rebuild=args.rebuild,
    )
    print("[build_index] 完成:", info)


if __name__ == "__main__":
    main()
