"""ActionParser：从模型自然语言输出解析 Thought / Action / Action Input。

设计要点（来自 SYNTHESIS §2.4）：
- 三个正则（THOUGHT/ACTION/INPUT），DOTALL 模式支持多行
- Action 必须存在；Action Input 可以缺失（仅当 Action 是 Final Answer 时视为合法）
- "Final Answer" 的 Action Input 是字符串（答案文本）
- 其他工具的 Action Input 必须能 json.loads；解析失败 → 兜底 `{"_raw": raw}` 让工具自行处理
- 解析失败 → {"retry": True}，让 agent 主循环追加"请严格按格式"提示
- 多 Action 解析（并行调用）：支持一次响应里出现多组 Action/Action Input
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
# Action Input 必须后跟下一个 Action/Action Input 或者文末
INPUT_RE = re.compile(
    r"Action Input\s*\d*\s*:\s*(.+?)(?=\n\s*Action(?: Input)?\s*\d*\s*:|\Z)",
    re.DOTALL,
)


def _parse_action_input(action: str, raw_input: str) -> Any:
    """解析 Action Input。Final Answer 是字符串，其它必须是 JSON。"""
    if action == "Final Answer":
        action_input = raw_input
        # 去掉外层引号（模型常把答案包在引号里）
        if (action_input.startswith('"') and action_input.endswith('"')) \
                or (action_input.startswith("'")
                    and action_input.endswith("'")):
            action_input = action_input[1:-1]
        return action_input
    try:
        return json.loads(raw_input) if raw_input else {}
    except json.JSONDecodeError:
        return {"_raw": raw_input}


class ActionParser:
    """解析 LLM 输出 → {thought, action, action_input, retry?}。"""

    def parse(self, text: str) -> dict[str, Any]:
        """主入口。返回 dict 至少含 thought/action/action_input。

        多 Action 情况：返回 {"thought": ..., "actions": [{action, action_input}, ...]}
        单 Action 情况：保持原 {thought, action, action_input, actions}，两种 key 都填。

        失败时返回 {"retry": True, "reason": "..."}。
        """
        if not text or not text.strip():
            return {"retry": True, "reason": "empty response"}

        thought_m = THOUGHT_RE.search(text)
        action_matches = list(ACTION_RE.finditer(text))
        input_matches = list(INPUT_RE.finditer(text))

        if not thought_m:
            return {"retry": True, "reason": "no Thought"}

        thought = thought_m.group(1).strip()

        if not action_matches:
            return {
                "retry": True,
                "thought": thought,
                "reason": "no Action",
            }

        # 解析所有 Action / Action Input 对
        # Action Input 必须出现在对应 Action 之后
        actions = []
        j = 0
        for am in action_matches:
            action = am.group(1).strip().strip("`").strip()
            # 找出现在 am 之后的第一个 Action Input
            raw_input = ""
            while j < len(input_matches) and input_matches[j].start() < am.end():
                j += 1
            if j < len(input_matches):
                raw_input = input_matches[j].group(1).strip()
                j += 1
            action_input = _parse_action_input(action, raw_input)
            actions.append({"action": action, "action_input": action_input})

        # 兼容老接口（单 Action 也填 action/action_input 顶层）
        result = {
            "thought": thought,
            "actions": actions,
        }
        if actions:
            result["action"] = actions[0]["action"]
            result["action_input"] = actions[0]["action_input"]
        return result

    # ----------------------------------------------------------- 自检工具
    @staticmethod
    def is_retry(parsed: dict) -> bool:
        return bool(parsed.get("retry"))


__all__ = ["ActionParser"]