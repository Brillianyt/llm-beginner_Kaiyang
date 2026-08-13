"""chunker 单元测试 / 烟雾测试。

直接 ``python test_smoke.py`` 跑，不依赖任何模型 / 网络。
- 验证 ``chunk_text`` 的基本不变量
- 验证 ``chunk_paragraphs`` 在段落边界处的行为
- 跑一遍 ``chunking_sanity`` 自检逻辑（与 eval/run.py 完全一致）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_text, chunk_text_with_boundaries, chunk_paragraphs, chunk_documents


def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"[{label}] expected={expected!r}, actual={actual!r}")
    print(f"  ok  {label}: {actual!r}")


def test_basic_chunking():
    """基本不变量：chunk 数、单 chunk 长度、最后一截、overlap。"""
    print("\n[test_basic_chunking]")
    text = "这是一段测试文本。" * 400  # ~ 4800 字符
    chunks = chunk_text(text, chunk_size=256, overlap=32)
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    assert len(chunks) > 10, f"chunk 数太少: {len(chunks)}"
    lo, hi = 256 * 0.5, 256 * 1.2
    assert lo <= avg_len <= hi, f"平均长度 {avg_len:.1f} 不在合法区间 [{lo}, {hi}]"
    print(f"  ok  chunk 数 = {len(chunks)}, 平均长度 = {avg_len:.1f}")


def test_chunk_size_sweep():
    """多档 chunk_size 扫描：128 / 256 / 512 / 1024。"""
    print("\n[test_chunk_size_sweep]")
    text = "神经网络与深度学习" * 200  # ~ 1800 字符
    for size in (128, 256, 512, 1024):
        chunks = chunk_text(text, chunk_size=size, overlap=min(32, size // 4))
        avg = sum(len(c) for c in chunks) / len(chunks)
        print(f"  chunk_size={size:>4}: chunks={len(chunks):>3} avg_len={avg:.1f}")


def test_short_text():
    """短文本：单 chunk 边界。"""
    print("\n[test_short_text]")
    chunks = chunk_text("短文本", chunk_size=256, overlap=32)
    assert_eq(len(chunks), 1, "短文本 chunk 数")
    assert_eq(chunks[0], "短文本", "短文本内容")


def test_empty_text():
    """空文本：返回空列表。"""
    print("\n[test_empty_text]")
    assert_eq(chunk_text("", chunk_size=256, overlap=32), [], "空文本")


def test_overlap_growth():
    """overlap 保证相邻 chunk 有重叠。"""
    print("\n[test_overlap_growth]")
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 20
    chunks = chunk_text(text, chunk_size=20, overlap=8)
    # 至少检查 step 推进合理
    assert len(chunks) >= 5, f"chunk 太少: {len(chunks)}"
    # 相邻 chunk 末尾应与下一 chunk 开头相同
    for a, b in zip(chunks, chunks[1:]):
        tail = a[-8:]
        assert b.startswith(tail), (
            f"overlap 不匹配:\n  a={a!r}\n  b={b!r}\n  expected tail={tail!r}"
        )
    print(f"  ok  overlap 验证通过，{len(chunks)} 个 chunk")


def test_boundaries():
    """chunk_text_with_boundaries 起止偏移正确。"""
    print("\n[test_boundaries]")
    text = "0123456789" * 10   # 100 字符
    pieces = chunk_text_with_boundaries(text, chunk_size=30, overlap=5)
    for piece in pieces:
        # 边界偏移切片应与原文本一致
        assert text[piece["start"]:piece["end"]] == piece["text"]
    print(f"  ok  {len(pieces)} 段边界全部对齐")


def test_invalid_size():
    """负数 / 0 chunk_size 抛错。"""
    print("\n[test_invalid_size]")
    try:
        chunk_text("hello", chunk_size=0, overlap=0)
    except ValueError:
        print("  ok  chunk_size=0 抛 ValueError")
    else:
        raise AssertionError("应抛 ValueError")


def test_overlap_autoclamp():
    """overlap >= chunk_size 时夹到 chunk_size - 1，避免死循环。"""
    print("\n[test_overlap_autoclamp]")
    text = "ABC" * 100
    # 正常情况不应死循环
    chunks = chunk_text(text, chunk_size=10, overlap=999)
    assert len(chunks) > 0
    print(f"  ok  overlap 极大时仍返回 {len(chunks)} 段")


def test_paragraph_chunking():
    """段落切分：应按自然段聚合而非硬切。"""
    print("\n[test_paragraph_chunking]")
    text = "第一段内容。" + ("x" * 100) + "\n\n第二段内容。" + ("y" * 100) + "\n\n第三段。" + ("z" * 100)
    chunks = chunk_paragraphs(text, chunk_size=256, overlap=0)
    # 段落总长 600+ 字符，至少被切 2 段
    assert len(chunks) >= 2, f"段落切分异常: {len(chunks)} 段"
    # 段落边界不应把"第一段"和"第二段"切开
    joined = "".join(chunks)
    assert "第一段内容。" in joined and "第二段内容。" in joined
    print(f"  ok  段落切分 {len(chunks)} 段，关键词完整")


def test_chunk_documents():
    """chunk_documents 返回 source / chunk_id 字段。"""
    print("\n[test_chunk_documents]")
    docs = [("kb.pdf", "ABC" * 100), ("kb.pdf", "DEF" * 100)]
    out = chunk_documents(docs, chunk_size=200, overlap=0, mode="char")
    assert all("text" in c and "source" in c and "chunk_id" in c for c in out)
    # chunk_id 单调递增
    assert [c["chunk_id"] for c in out] == list(range(len(out)))
    print(f"  ok  chunk_documents 返回 {len(out)} 段，id 0..{len(out) - 1}")


def test_eval_compat():
    """复刻 eval/run.py 的 chunking_sanity 验收，确保本地通过。"""
    print("\n[test_eval_compat]")
    sample = "这是一段测试文本。" * 400
    chunks = chunk_text(sample, chunk_size=256, overlap=32)
    assert len(chunks) > 10, f"chunk 数 {len(chunks)} <= 10"
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    assert 256 * 0.5 <= avg_len <= 256 * 1.2, f"avg_len {avg_len:.1f} 越界"
    print(f"  ok  eval 兼容：chunks={len(chunks)} avg_len={avg_len:.1f}")


def main():
    print("=" * 60)
    print("chunker 烟雾测试")
    print("=" * 60)
    test_basic_chunking()
    test_chunk_size_sweep()
    test_short_text()
    test_empty_text()
    test_overlap_growth()
    test_boundaries()
    test_invalid_size()
    test_overlap_autoclamp()
    test_paragraph_chunking()
    test_chunk_documents()
    test_eval_compat()
    print("\n[ALL PASS]")


if __name__ == "__main__":
    main()
