"""file_search 工具：本地文件名 / 内容检索。

实现要点（来自 README §M1 + SYNTHESIS §9）：
- 必须 `Path.resolve()` 后 `is_relative_to` 校验，**禁止 `..` 越界**
- pattern 支持文件名通配（fnmatch）和内容正则
- 既返回匹配的文件名列表，也返回内容片段（满足 README "返回内容片段" 要求）
- 默认工作根 = task-5-tool-agent 父目录（即仓库根），允许的子目录限定
"""
from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .base import Tool

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": (
            "在指定目录下搜索文件名（glob）或内容（正则）。"
            "返回匹配的文件路径列表 + 第一个匹配的内容片段（上下文 60 字）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "文件名 glob（如 '*.md'）或内容正则（如 'TODO'）。"
                    ),
                },
                "dir": {
                    "type": "string",
                    "description": (
                        "搜索目录（相对路径，相对仓库根）。"
                        "不允许越界到工作区外。"
                    ),
                },
            },
            "required": ["pattern", "dir"],
        },
    },
}


def _safe_root(allowed_root: Path) -> Path:
    """解析并校验最终路径在 allowed_root 之内。"""
    return allowed_root.resolve()


def _resolve_target(target: str, root: Path) -> Path:
    """把相对路径解析为绝对路径，并校验越界。

    - 绝对路径：必须 is_relative_to(root)，否则越界
    - 相对路径：相对 root 解析，必须落在 root 内
    """
    if not target:
        target = "."
    p = Path(target)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (root / target).resolve()
    # 越界保护
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise PermissionError(
            f"路径越界：{resolved} 不在允许根 {root} 内"
        ) from e
    return resolved


def _looks_like_filename(pattern: str) -> bool:
    """启发式：是否更像文件名（而不是内容正则）。

    文件名特征（满足任一即视为文件名）：
    - 包含 glob 字符（* ? [）
    - 以常见文件后缀结尾（.md/.txt/.py/.json/.csv/...）
    - 显式路径分隔符

    默认走内容搜索（更宽松）。
    """
    # 显式 glob
    if any(c in pattern for c in "*?["):
        return True
    # 文件后缀
    file_suffixes = (
        ".md", ".txt", ".py", ".json", ".csv", ".yaml", ".yml",
        ".xml", ".html", ".pdf", ".log", ".sh", ".java", ".cpp",
        ".js", ".ts", ".go", ".rs", ".c", ".h", ".sql",
    )
    if pattern.lower().endswith(file_suffixes):
        return True
    # 含路径分隔符 / 点号超过 1 个 → 更像文件名
    if "/" in pattern or pattern.count(".") > 1:
        return True
    # 默认视为内容搜索
    return False


def run(args: dict[str, Any], allowed_root: Path | None = None) -> str:
    """文件检索入口。

    参数：
        args:
            pattern: 文件名 glob 或内容正则
            dir: 搜索目录
        allowed_root: 允许的根目录（默认仓库根 = task-5-tool-agent 父级）
    """
    if "pattern" not in args or "dir" not in args:
        raise KeyError("pattern 或 dir 缺失")

    pattern = str(args["pattern"])
    target_dir = str(args["dir"])

    if allowed_root is None:
        allowed_root = Path(__file__).resolve().parents[3]  # task-5-tool-agent/src/tools/file_search.py
    allowed_root = _safe_root(allowed_root)

    search_dir = _resolve_target(target_dir, allowed_root)
    if not search_dir.exists():
        raise FileNotFoundError(f"目录不存在：{search_dir}")
    if not search_dir.is_dir():
        raise NotADirectoryError(f"不是目录：{search_dir}")

    is_filename_search = _looks_like_filename(pattern)
    matches: list[tuple[Path, str | None]] = []

    if is_filename_search:
        # 文件名匹配（glob / 精确名）
        for f in search_dir.rglob("*"):
            if not f.is_file():
                continue
            if fnmatch(f.name, pattern):
                snippet = _first_paragraph(f)
                matches.append((f.relative_to(search_dir), snippet))
    else:
        # 内容搜索
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"正则编译失败：{e}") from e
        for f in search_dir.rglob("*"):
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            m = regex.search(text)
            if m:
                snippet = _extract_snippet(text, m.start())
                matches.append((f.relative_to(search_dir), snippet))

    if not matches:
        return f"在 {target_dir} 下未找到匹配 '{pattern}' 的文件。"

    lines = [f"在 {target_dir} 下找到 {len(matches)} 个匹配 '{pattern}' 的文件："]
    for rel, snippet in matches[:20]:
        if snippet:
            lines.append(f"- {rel}  [内容预览] {snippet}")
        else:
            lines.append(f"- {rel}")
    if len(matches) > 20:
        lines.append(f"...（共 {len(matches)} 个，截断显示前 20 个）")
    return "\n".join(lines)


def _extract_snippet(text: str, pos: int, width: int = 60) -> str:
    """提取匹配位置前后的文本片段，去掉换行。"""
    start = max(0, pos - width // 2)
    end = min(len(text), pos + width // 2)
    snippet = text[start:end].replace("\n", " ").replace("\r", " ")
    return snippet.strip()


def _first_paragraph(path: Path, max_chars: int = 200) -> str | None:
    """读文件第一段（首个空行之前），用于文件名匹配时回显内容。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    # 第一个空行之前
    parts = text.split("\n\n", 1)
    para = parts[0] if parts else text
    return para[:max_chars].strip()


class FileSearch(Tool):
    """file_search 工具类（Tool 基类包装）。"""
    name = TOOL_SCHEMA["function"]["name"]
    description = TOOL_SCHEMA["function"]["description"]
    parameters = TOOL_SCHEMA["function"]["parameters"]

    def run(self, args: dict[str, Any]) -> str:
        return run(args)


__all__ = ["TOOL_SCHEMA", "run", "FileSearch"]