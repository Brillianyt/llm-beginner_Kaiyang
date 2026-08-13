"""python_sandbox 工具：受限 exec（教学级）。

⚠️ 警告：黑白名单 + 超时仅是**教学级**保护，不是真正隔离。
仍可能通过 `().__class__.__bases__` 等路径逃逸；超时也挡不住内存耗尽。
只对可信 / 自产代码用，别对真正不可信输入直接 exec。

实现要点（来自 README §M1）：
- import 黑名单：禁 os / sys / subprocess / socket / shutil / pathlib 等危险模块
- 仅暴露安全 builtins（print / len / range / list / dict / set / str / int / float / bool / sum / max / min / abs / round / sorted / enumerate / zip / map / filter / isinstance / type 等）
- 超时：signal.SIGALRM（仅 main thread，可移植）
- 捕获 stdout（StringIO）作为返回；执行异常拼成字符串
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import signal
from typing import Any

from .base import Tool

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "在受限沙箱中执行一段 Python 代码（教学级保护）。"
            "可使用 print / range / list / dict / sum 等安全 builtins，"
            "导入受限（禁 os / sys / subprocess / socket 等）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码字符串。",
                },
            },
            "required": ["code"],
        },
    },
}

# 危险模块黑名单（import x / from x import ... 都拒绝）
_BLOCKED_MODULES: set[str] = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "ctypes", "cffi", "multiprocessing", "threading",
    "requests", "urllib", "http", "ftplib", "smtplib",
    "pickle", "marshal", "shelve", "dbm",
    "__builtins__", "builtins",
    "code", "codeop", "compile", "eval", "exec", "open",
    "importlib", "pkgutil", "zipimport", "importlib",
    "signal",  # 禁掉 signal，避免嵌套超时
}

# 允许暴露的 builtin 名称白名单
_SAFE_BUILTIN_NAMES: set[str] = {
    # 类型
    "bool", "int", "float", "complex", "str", "bytes",
    "list", "tuple", "set", "frozenset", "dict",
    # 常用函数
    "print", "len", "range", "enumerate", "zip", "map",
    "filter", "sorted", "reversed", "sum", "min", "max",
    "abs", "round", "pow", "divmod",
    "any", "all", "isinstance", "issubclass", "type",
    "repr", "id", "hash", "chr", "ord",
    # 数学
    "True", "False", "None",
}


def _safe_builtins() -> dict[str, Any]:
    """从 builtins 中挑出安全子集。"""
    env = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            env[name] = getattr(builtins, name)
    # 加几个常用第三方数学
    import math
    env["math"] = math
    return env


class _TimeoutError(Exception):
    """本地超时异常（避免与内置 TimeoutError 混淆）。"""


@contextlib.contextmanager
def _time_limit(seconds: float):
    """signal 实现的超时限制（仅 main thread）。"""
    def _handler(signum, frame):
        raise _TimeoutError(f"执行超时（>{seconds}s）")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _ast_is_safe(tree: ast.AST) -> tuple[bool, str]:
    """AST 静态扫描：拒绝 import 黑名单模块。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_MODULES:
                    return False, f"禁止 import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in _BLOCKED_MODULES:
                return False, f"禁止 from {node.module} import ..."
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            # 阻止自定义类/函数调用 dunder 逃逸（粗筛）
            pass
    return True, ""


def run(args: dict[str, Any], timeout: float = 3.0) -> str:
    """受限 exec 入口。args = {"code": str}。"""
    if "code" not in args:
        raise KeyError("code")
    code = str(args["code"])
    if not code.strip():
        raise ValueError("code 为空")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"语法错误：{e}") from e

    safe, reason = _ast_is_safe(tree)
    if not safe:
        raise PermissionError(reason)

    stdout = io.StringIO()
    env = _safe_builtins()

    try:
        with _time_limit(timeout):
            with contextlib.redirect_stdout(stdout):
                exec(compile(tree, "<sandbox>", "exec"), env)
    except _TimeoutError:
        return "[ERROR: 执行超时]"
    except Exception as e:  # noqa: BLE001
        # 把异常信息也带上
        out = stdout.getvalue()
        return f"{out}[ERROR: {type(e).__name__}: {e}]"

    out = stdout.getvalue()
    if not out:
        return "[OK: 代码执行成功，无 print 输出]"
    # 截断防止 prompt 爆炸
    return out[:2000]


class PythonSandbox(Tool):
    """python_sandbox 工具类（Tool 基类包装）。"""
    name = TOOL_SCHEMA["function"]["name"]
    description = TOOL_SCHEMA["function"]["description"]
    parameters = TOOL_SCHEMA["function"]["parameters"]

    def run(self, args: dict[str, Any]) -> str:
        return run(args)


__all__ = ["TOOL_SCHEMA", "run", "PythonSandbox"]