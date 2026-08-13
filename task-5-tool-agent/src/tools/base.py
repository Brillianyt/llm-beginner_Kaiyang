"""Tool 基类 + ToolRegistry（策略模式 + 错误注入钩子）。

设计要点（来自 reference/patterns/strategy-tools.md）：
- Tool 基类提供 `name / description / parameters / run` 四件套，子类实现 `run`
- 注册表用 dict 查表（O(1)），统一抛异常转 Observation 字符串
- `_error_inject` 字典支持 S4 错误注入（一次性或参数化 error_rate）
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """工具基类。所有具体工具继承本类，实现 run(args) -> str。

    - `name`        ：工具名（小写英文，路由 key）
    - `description` ：一句话功能说明，写进 system prompt
    - `parameters`  ：OpenAI function calling JSON schema（type=object）
    - `run(args)`   ：执行入口。返回字符串（不是 JSON），错误请 raise。
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    @abstractmethod
    def run(self, args: dict[str, Any]) -> str:
        """根据 args 调用工具，返回字符串结果。"""

    def to_openai_schema(self) -> dict:
        """导出 OpenAI function calling 格式的 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """策略注册表 + 错误统一处理 + 错误注入钩子（S4）。

    关键 API：
    - register(tool)         ：注册工具
    - schema_list()          ：导出 OpenAI 风格的 schema 列表（喂给 PromptBuilder）
    - call(name, args)       ：查表 + try/except 兜底，返回字符串 Observation
    - names()                ：当前已注册的所有工具名
    - inject_error(...)      ：S4 钩子，下次调用指定工具时返回 [ERROR: ...]
    - set_error_rate(...)    ：S4 钩子，按概率返回 [ERROR: ...]
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # 一次性错误注入（弹夹式）
        self._error_inject: dict[str, str] = {}
        # 概率错误注入（错误率模式）
        self._error_rate: dict[str, float] = {}
        self._error_msg: dict[str, str] = {}

    # ------------------------------------------------------------------ 注册
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schema_list(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    # ------------------------------------------------------------------ 调用
    def call(self, name: str, args: dict[str, Any]) -> str:
        """统一入口。任何异常 → '[ERROR: ...]' 字符串，**不抛**。"""
        # 一次性错误注入（弹夹式，pop 一次）
        if name in self._error_inject:
            msg = self._error_inject.pop(name)
            return f"[ERROR: {msg}]"
        # 概率错误注入
        if name in self._error_rate:
            rate = self._error_rate[name]
            if rate > 0 and random.random() < rate:
                msg = self._error_msg.get(name, "[Injected Error]")
                return f"[ERROR: {msg}]"
        # 未知工具
        if name not in self._tools:
            return (
                f"[ERROR: 未知工具 '{name}'，"
                f"可用：{list(self._tools.keys())}]"
            )
        # 正常调用 + 兜底异常
        try:
            return str(self._tools[name].run(args))
        except KeyError as e:
            return f"[ERROR: {name} 缺少参数 {e}]"
        except ZeroDivisionError:
            return f"[ERROR: {name} 除零错误]"
        except TimeoutError:
            return f"[ERROR: {name} 执行超时]"
        except Exception as e:  # noqa: BLE001
            return f"[ERROR: {name} 抛 {type(e).__name__}: {e}]"

    # ------------------------------------------------------------------ S4 钩子
    def inject_error(self, name: str, msg: str = "[Injected Error]") -> None:
        """一次性错误注入：下一次 call(name, ...) 返回 [ERROR: msg]。"""
        self._error_inject[name] = msg

    def set_error_rate(self, name: str, rate: float,
                       msg: str = "[Injected Error]") -> None:
        """概率错误注入：call 时按 rate 概率返回 [ERROR: msg]。"""
        if rate < 0:
            rate = 0.0
        if rate > 1:
            rate = 1.0
        self._error_rate[name] = rate
        self._error_msg[name] = msg

    def clear_errors(self) -> None:
        """清空所有错误注入（一次性 + 概率）。"""
        self._error_inject.clear()
        self._error_rate.clear()
        self._error_msg.clear()