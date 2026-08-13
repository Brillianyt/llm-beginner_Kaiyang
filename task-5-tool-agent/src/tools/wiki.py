"""wiki 工具：维基百科 API 查询。

实现要点（来自 README §M1 + SYNTHESIS §9）：
- 用 wikipedia-api（Python 封装），**必须**填 user-agent，否则 403
- 支持中英文：wikipedia.set_lang('zh' / 'en')
- 返回 summary（截断到 500 字）+ 关键 metadata
- 网络不可用 → raise（registry 会兜成 [ERROR: ...]）
"""
from __future__ import annotations

from typing import Any

from .base import Tool

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "查询维基百科条目（支持中英文，按 query 自动判断）。"
            "返回条目的摘要文字（截断到 500 字），便于提取人名/年份/事实。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "要查询的条目名（如 'Alan Turing' 或 '图灵机'）。"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

_MAX_SUMMARY = 500
_USER_AGENT = "task-5-tool-agent/1.0 (educational; contact: dev@example.com)"


def _is_chinese(text: str) -> bool:
    return any('一' <= ch <= '鿿' for ch in text)


def _fetch(query: str) -> tuple[str, str]:
    """拉取维基百科条目 summary。

    返回 (summary, url)。网络异常由 registry 统一兜底。
    """
    # wikipedia-api 0.15+：Wikipedia(user_agent, language)
    import wikipediaapi  # type: ignore
    lang = "zh" if _is_chinese(query) else "en"
    wiki = wikipediaapi.Wikipedia(user_agent=_USER_AGENT, language=lang)
    page = wiki.page(query)
    if not page.exists():
        # 英文兜底：中文 query 不存在时，翻译重试
        if lang == "zh":
            wiki_en = wikipediaapi.Wikipedia(user_agent=_USER_AGENT,
                                             language="en")
            page = wiki_en.page(query)
            if not page.exists():
                return "", ""
        else:
            return "", ""
    summary = page.summary[:_MAX_SUMMARY]
    return summary, page.fullurl


def run(args: dict[str, Any]) -> str:
    """维基百科查询入口。args = {"query": str}。"""
    if "query" not in args:
        raise KeyError("query")
    query = str(args["query"]).strip()
    if not query:
        raise ValueError("query 为空")
    summary, url = _fetch(query)
    if not summary:
        return f"[NOT FOUND] 维基百科未找到条目：{query!r}"
    return f"{summary}\n[来源] {url}"


class Wiki(Tool):
    """wiki 工具类（Tool 基类包装）。"""
    name = TOOL_SCHEMA["function"]["name"]
    description = TOOL_SCHEMA["function"]["description"]
    parameters = TOOL_SCHEMA["function"]["parameters"]

    def run(self, args: dict[str, Any]) -> str:
        return run(args)


__all__ = ["TOOL_SCHEMA", "run", "Wiki"]