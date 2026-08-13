"""PromptBuilder：组装 system prompt + few-shot + 当前 task 的 messages。

设计要点（来自 SYNTHESIS §2.3 + §5）：
- 三层：角色设定 + 动态工具列表 + few-shot（覆盖单工具/多工具/查询）
- few-shot 作为 user/assistant 对话历史加入（比 system 字符串更符合 chat 训练分布）
- `append_observation` 把历史 Thought/Action/Observation 拼成 assistant/user 对
- S3 参数化：few_shot_count ∈ {0, 1, 3}；include_error_hint ∈ {True, False}
"""
from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------- system
SYSTEM_PROMPT_TMPL = """你是工具调用助手，可以调用以下工具：

{tool_descriptions}

严格按以下格式输出，每一轮只输出一组（Thought / Action / Action Input），然后等待 Observation：

Thought: <你的推理，下一步该做什么>
Action: <工具名，必须是上述之一或 "Final Answer">
Action Input: <合法 JSON 对象（对应工具的 parameters），或最终答案字符串>

工作流程：
1. 输出 Thought 推理
2. 输出 Action 选择工具
3. 输出 Action Input（JSON 格式）
4. 等待 Observation
5. 重复直到输出 Action: Final Answer

注意：
- 只输出 Thought / Action / Action Input 三行，不要解释其它内容
- Action Input 必须是合法 JSON（双引号、无尾逗号）；Final Answer 的 Action Input 是纯字符串
- 如果工具失败，Observation 会以 [ERROR: ...] 开头，请换工具或修正参数
- 最终答案用 Action: Final Answer + Action Input: <答案字符串>
{error_hint}"""


_ERROR_HINT = """- 遇到 [ERROR: ...] 时，分析原因并换工具 / 修正参数，不要放弃"""

# --------------------------------------------------------------------- few-shot
FEW_SHOTS: list[dict[str, str]] = [
    {
        "task": "计算 (123+456)*789 是几位数",
        "trajectory": (
            "Thought: 先算出乘积。\n"
            "Action: calculator\n"
            "Action Input: {\"expression\": \"(123+456)*789\"}\n"
            "Observation: 456831\n"
            "Thought: 456831 有 6 位。\n"
            "Action: Final Answer\n"
            "Action Input: 6 位"
        ),
    },
    {
        "task": "查 Geoffrey Hinton 出生年份，并算到 2026 年多少岁",
        "trajectory": (
            "Thought: 先查 Hinton 的维基百科。\n"
            "Action: wiki\n"
            "Action Input: {\"query\": \"Geoffrey Hinton\"}\n"
            "Observation: 杰弗里·埃弗里斯特·辛顿（Geoffrey Everest Hinton，1947 年 12 月 6 日—... [来源] https://...\n"
            "Thought: 出生 1947，2026 - 1947 = 79。\n"
            "Action: calculator\n"
            "Action Input: {\"expression\": \"2026-1947\"}\n"
            "Observation: 79\n"
            "Thought: 出生 1947 年，到 2026 年 79 岁（若已过生日）。\n"
            "Action: Final Answer\n"
            "Action Input: 1947 年出生，到 2026 年 79 岁"
        ),
    },
    {
        "task": "在 data/agent-fixtures 下找所有 .md 文件",
        "trajectory": (
            "Thought: 用 file_search 找 .md 文件。\n"
            "Action: file_search\n"
            "Action Input: {\"pattern\": \"*.md\", \"dir\": \"data/agent-fixtures\"}\n"
            "Observation: 在 data/agent-fixtures 下找到 2 个匹配 '*.md' 的文件：\\n- README.md\\n- todo_note.md\n"
            "Thought: 找到两个文件。\n"
            "Action: Final Answer\n"
            "Action Input: 2 个 .md 文件：README.md、todo_note.md"
        ),
    },
]


def _format_tools(schemas: list[dict]) -> str:
    """把 OpenAI schema 列表格式化成 system prompt 里的人类可读段落。"""
    lines: list[str] = []
    for s in schemas:
        f = s.get("function", {})
        params = f.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))
        lines.append(f"- {f.get('name', '?')}: {f.get('description', '')}")
        for pname, pinfo in props.items():
            req_tag = "（必填）" if pname in required else "（可选）"
            ptype = pinfo.get("type", "?")
            desc = pinfo.get("description", "")
            lines.append(f"    - {pname} ({ptype}){req_tag}: {desc}")
    return "\n".join(lines)


class PromptBuilder:
    """组装 chat messages。"""

    def __init__(
        self,
        tool_schemas: list[dict[str, Any]],
        few_shot_count: int = 3,
        include_error_hint: bool = True,
    ) -> None:
        self.tool_descriptions = _format_tools(tool_schemas)
        self.few_shot_count = max(0, min(few_shot_count, len(FEW_SHOTS)))
        self.include_error_hint = include_error_hint

    def _system_prompt(self) -> str:
        hint = _ERROR_HINT if self.include_error_hint else ""
        return SYSTEM_PROMPT_TMPL.format(
            tool_descriptions=self.tool_descriptions,
            error_hint=hint,
        )

    def initial_messages(self, task: str) -> list[dict[str, str]]:
        """拼 system + few-shot + 当前 task，返回 messages 列表。"""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        for ex in FEW_SHOTS[: self.few_shot_count]:
            messages.append({"role": "user", "content": ex["task"]})
            messages.append({"role": "assistant", "content": ex["trajectory"]})
        messages.append({"role": "user", "content": task})
        return messages

    def append_observation(
        self,
        messages: list[dict[str, str]],
        thought: str,
        action: str,
        action_input: Any,
        observation: str,
    ) -> list[dict[str, str]]:
        """把一步 Thought/Action/Action Input + Observation 追加到 messages。"""
        import json
        # Final Answer 不再追加 Observation（终止语义）
        if action == "Final Answer":
            return messages
        # Action Input 是 dict 就 dump 成 JSON 字符串；其它保持原样
        if isinstance(action_input, dict):
            ai_text = json.dumps(action_input, ensure_ascii=False)
        else:
            ai_text = str(action_input)
        asst_msg = (
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {ai_text}"
        )
        messages.append({"role": "assistant", "content": asst_msg})
        # 截断 Observation 防止 prompt 爆炸
        obs = observation if len(observation) <= 1500 else (
            observation[:1500] + "...(已截断)"
        )
        messages.append({"role": "user", "content": f"Observation: {obs}"})
        return messages

    def retry_message(self, reason: str = "") -> str:
        """解析失败时追加的提示。"""
        return (
            "请严格按以下格式输出（每轮一组 Thought / Action / Action Input）：\n"
            "Thought: <推理>\n"
            "Action: <工具名>\n"
            "Action Input: <JSON 或最终答案字符串>\n"
            f"（之前解析失败：{reason or '格式不符'}）"
        )


__all__ = ["PromptBuilder", "FEW_SHOTS"]