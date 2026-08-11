"""M1 里程碑验证:BPE round-trip 全面检查。

README 任务二 M1:手写简化版 BPE tokenizer,自检 `tokenizer_roundtrip` 通过
(encode→decode 能还原中文)。

用法:python verify_bpe.py [--tokenizer ckpt/tokenizer.json]
输出:
  - 控制台:逐样本 OK/FAIL + 汇总
  - verify_bpe_report.md:作为 M1 通过的 artifact
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import regex as _re

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.tokenizer import BPETokenizer


# 覆盖各语种 + 边界条件 + 特殊场景
TEST_CASES = [
    # 纯中文
    ("纯中文", "床前明月光"),
    ("纯中文长句", "山西警方扫黑除恶行动集中收网打掉涉黑涉恶犯罪组织和团伙"),
    ("中文+数字", "2024年生产总值为1360000亿元"),
    # 英文
    ("英文短句", "Hello, world!"),
    ("英文长句", "Once upon a time in a small village by the sea"),
    ("英文+数字", "Python 3.11 was released in October 2022"),
    # 中英混合
    ("中英混合", "深度学习需要mathematics基础"),
    ("混合带符号", "iPhone 15 Pro Max 价格 ¥9999"),
    # 标点 / 空白
    ("中文标点", "他说:\"你好,世界!\"。"),
    ("全角空格", "床前\u3000明月光"),
    ("换行+段落", "第一段内容。\n\n第二段开始。\n\n第三段。"),
    # 字节边界
    ("BMP外", "𠮷"),  # 4 字节 UTF-8
    ("Emoji", "🌙🌟⭐"),  # 4 字节 ×3
    ("BOM", "\ufeff起"),
    # 特殊 token 字面
    ("特殊token字面", "<unk><bos><eos><pad>"),
    # 罕见 / OOV
    ("极罕见字", "龘"),  # 4 字节 UTF-8 + 极低频
    # 边界
    ("空字符串", ""),
    ("单空格", " "),
    ("单字符", "a"),
    # 重复 / 退化
    ("重复字", "啊啊啊啊啊啊啊啊啊啊啊啊"),
    ("重复词", "啦啦啦啦啦啦啦啦啦啦啦啦"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("ckpt/tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("verify_bpe_report.md"))
    args = ap.parse_args()

    tok = BPETokenizer.from_pretrained(str(args.tokenizer))
    print(f"[M1 验证] tokenizer loaded: vocab_size={tok.vocab_size}, merges={len(tok.merges)}")

    rows = []
    fails = []
    for name, text in TEST_CASES:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        ok = decoded == text
        rows.append((name, text, ids, decoded, ok))
        if not ok:
            fails.append((name, text, decoded))

    n_pass = sum(1 for r in rows if r[4])
    n_total = len(rows)
    overall_pass = (n_pass == n_total)

    # 写报告
    lines = [
        "# M1 里程碑验证:BPE round-trip",
        "",
        f"**Tokenizer**: `{args.tokenizer.name}` (vocab_size={tok.vocab_size}, merges={len(tok.merges)})  ",
        f"**结果**: {'✓ 通过' if overall_pass else '✗ 失败'} ({n_pass}/{n_total} 样本)  ",
        f"**README M1 要求**:手写简化版 BPE tokenizer,自检 `tokenizer_roundtrip` 通过(encode→decode 能还原中文)",
        "",
        "## 详细测试",
        "",
        "| # | 类别 | 输入 | tokens | 解码 | 通过 |",
        "|---|---|---|---|---|---|",
    ]
    for i, (name, text, ids, decoded, ok) in enumerate(rows, 1):
        text_safe = text.replace("|", "\\|").replace("\n", "\\n")
        decoded_safe = decoded.replace("|", "\\|").replace("\n", "\\n")
        ids_preview = str(ids[:8]) + ("..." if len(ids) > 8 else "")
        mark = "✓" if ok else "✗"
        lines.append(f"| {i} | {name} | `{text_safe!r}` | {len(ids)} ({ids_preview}) | `{decoded_safe!r}` | {mark} |")

    lines.append("")
    if fails:
        lines.append("## 失败样本")
        for name, text, decoded in fails:
            lines.append(f"- **{name}**: input={text!r}, output={decoded!r}")

    args.out.write_text("\n".join(lines), encoding="utf-8")

    # 控制台汇总
    print(f"\n[M1 验证] {n_pass}/{n_total} 通过")
    if overall_pass:
        print(f"[M1 ✓] round-trip 全部通过,artifact 写入 {args.out}")
    else:
        print(f"[M1 ✗] 有 {len(fails)} 个失败:")
        for name, text, decoded in fails:
            print(f"  - {name}: in={text!r}, out={decoded!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()