"""Calculator 工具：四则运算 + 常见数学函数。

实现要点：
- **不直接 eval**：用 ast.parse 解析表达式，构造 AST 白名单遍历器
- 仅允许：常量、二元运算、函数调用（白名单）、一元运算（+ -）
- 禁所有 `__xxx__` 名称、属性访问、import、赋值
- 函数白名单：abs / round / min / max / sum / pow + math 模块下子集
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from .base import Tool

# ------------------------------ schema ------------------------------------
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "执行四则运算和常见数学函数（+ - * / % ** abs round min max sum pow "
            "sqrt sin cos tan log log10 log2 exp floor ceil）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "算术表达式，例如 '(123+456)*789' 或 'sqrt(2026)'。"
                    ),
                },
            },
            "required": ["expression"],
        },
    },
}

# 函数白名单（来自 math 模块，仅暴露安全子集）
_ALLOWED_FUNCS: dict[str, Any] = {
    # 基础
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow,
    # 数学
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log10": math.log10,
    "log2": math.log2, "exp": math.exp, "floor": math.floor,
    "ceil": math.ceil, "fabs": math.fabs,
    # 双目
    "hypot": math.hypot, "atan2": math.atan2,
}

# 二元运算映射
_BINOPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 一元运算映射
_UNARYOPS: dict[type, Any] = {
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}


class _SafeEvaluator:
    """AST 安全求值器。

    接受 ast.Expression 节点树，按白名单逐节点求值。
    拒绝任何 _UnaryOp/BinOp/Call 之外的 AST 节点（Name/Constant 不含 dunder）。
    """

    def evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持常量类型 {type(node.value).__name__}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](
                self.evaluate(node.left), self.evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](self.evaluate(node.operand))
        if isinstance(node, ast.Call):
            # 仅允许 func = Name，且 name 在白名单
            if not isinstance(node.func, ast.Name):
                raise ValueError("只允许直接函数调用（不支持属性/方法）")
            if node.func.id not in _ALLOWED_FUNCS:
                raise ValueError(f"函数 {node.func.id!r} 不在白名单")
            # 拒绝关键字参数
            if node.keywords:
                raise ValueError("不支持关键字参数")
            args = [self.evaluate(a) for a in node.args]
            return _ALLOWED_FUNCS[node.func.id](*args)
        if isinstance(node, ast.Name):
            # 仅允许常量（pi / e）
            consts = {"pi": math.pi, "e": math.e}
            if node.id in consts:
                return consts[node.id]
            raise ValueError(f"未定义名称 {node.id!r}")
        raise ValueError(f"不支持 AST 节点 {type(node).__name__}")


def run(args: dict[str, Any]) -> str:
    """计算器入口。args = {"expression": str}。"""
    if "expression" not in args:
        raise KeyError("expression")
    expr = str(args["expression"]).strip()
    if not expr:
        raise ValueError("expression 为空")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"语法错误：{e}") from e
    evaluator = _SafeEvaluator()
    result = evaluator.evaluate(tree)
    # 数值格式化：保留 10 位精度，去掉无意义的 .0
    if isinstance(result, float):
        if result.is_integer():
            return str(int(result))
        # 用 round 控制尾零 + 不强制科学计数
        return f"{result:.10f}".rstrip("0").rstrip(".") or "0"
    return str(result)


# 直接导出 schema 兼容接口（tool 命名空间统一）
class Calculator(Tool):
    """calculator 工具类（Tool 基类包装）。"""
    name = TOOL_SCHEMA["function"]["name"]
    description = TOOL_SCHEMA["function"]["description"]
    parameters = TOOL_SCHEMA["function"]["parameters"]

    def run(self, args: dict[str, Any]) -> str:
        return run(args)


__all__ = ["TOOL_SCHEMA", "run", "Calculator"]