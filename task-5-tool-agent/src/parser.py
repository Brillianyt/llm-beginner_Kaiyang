"""ActionParser：从模型自然语言输出解析 Thought / Action / Action Input。

设计要点（来自 SYNTHESIS §2.4）：
- 三个正则（THOUGHT/ACTION/INPUT），DOTALL 模式支持多行
- Action 必须存在；Action Input 可以缺失（仅当 Action 是 Final Answer 时视为合法）
- "Final Answer" 的 Action Input 是字符串（答案文本）
- 其他工具的 Action Input 必须能 json.loads；解析失败 → 兜底 `{"_raw": raw}` 让工具自行处理
- 解析失败 → {"retry": True}，让 agent 主循环追加"请严格按格式"提示
"""
from __future__ import annotations

import json
import re
from typing import Any

# Thought 1: ... / Thought: ... 都支持
THOUGHT_RE = re.compile(
    r"Thought\s*\d*\s*:\s*(.+?)(?=\n\s*Action|\Z)", re.DOTALL
)
ACTION_RE = re.compile(
    r"Action\s*\d*\s*:\s*(.+?)(?=\n\s*Action Input|\Z)", re.DOTALL
)
INPUT_RE = re.compile(
    r"Action Input\s*\d*\s*:\s*(.+?)(?=\n\s*Observation|\Z)", re.DOTALL
)


class ActionParser:
    """解析 LLM 输出 → {thought, action, action_input, retry?}。"""

    def parse(self, text: str) -> dict[str, Any]:
        """主入口。返回 dict 至少含 thought/action/action_input。

        失败时返回 {"retry": True, "reason": "..."}。
        """
        if not text or not text.strip():
            return {"retry": True, "reason": "empty response"}

        thought_m = THOUGHT_RE.search(text)
        action_m = ACTION_RE.search(text)
        input_m = INPUT_RE.search(text)

        if not thought_m:
            return {"retry": True, "reason": "no Thought"}

        thought = thought_m.group(1).strip()

        if not action_m:
            # 没有 Action 但有 Thought，提示模型补齐
            return {
                "retry": True,
                "thought": thought,
                "reason": "no Action",
            }

        action = action_m.group(1).strip()
        # 去掉可能的尾部换行 / 引号
        action = action.strip().strip("`").strip()

        raw_input = input_m.group(1).strip() if input_m else ""

        # Action Input 解析：Final Answer 直接是字符串，其它必须是 JSON
        if action == "Final Answer":
            action_input = raw_input
            # 去掉外层引号（模型常把答案包在引号里）
            if (action_input.startswith('"') and action_input.endswith('"')) \
                    or (action_input.startswith("'")
                        and action_input.endswith("'")):
                action_input = action_input[1:-1]
        else:
            try:
                action_input = json.loads(raw_input) if raw_input else {}
            except json.JSONDecodeError:
                # 兜底：把 raw 当字符串包成 {"_raw": raw}
                action_input = {"_raw": raw_input}

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    # ----------------------------------------------------------- 自检工具
    @staticmethod
    def is_retry(parsed: dict) -> bool:
        return bool(parsed.get("retry"))


__all__ = ["ActionParser"]