"""Tools 包：导出 4 个工具模块 + 默认 ToolRegistry。

`eval/run.py` 走 `from src.tools import calculator, python_sandbox, file_search, wiki`，
然后按名取 `mod.run({...})`。所以每个子模块必须直接导出 `run` 和 `TOOL_SCHEMA`。

这里同时提供一个 `default_registry()` 把 4 个工具一次性注册好给 ReActAgent。
"""
from __future__ import annotations

# 显式子模块导入（eval/run.py 按名取 run）
from . import calculator, python_sandbox, file_search, wiki
from .base import Tool, ToolRegistry
from .calculator import Calculator
from .file_search import FileSearch
from .python_sandbox import PythonSandbox
from .wiki import Wiki

ALL_TOOLS: list[Tool] = [Calculator(), PythonSandbox(), FileSearch(), Wiki()]


def default_registry() -> ToolRegistry:
    """返回一个已注册 4 个默认工具的 registry。"""
    reg = ToolRegistry()
    for t in ALL_TOOLS:
        reg.register(t)
    return reg


__all__ = [
    "calculator", "python_sandbox", "file_search", "wiki",
    "Tool", "ToolRegistry",
    "Calculator", "PythonSandbox", "FileSearch", "Wiki",
    "ALL_TOOLS", "default_registry",
]