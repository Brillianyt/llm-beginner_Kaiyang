"""ReActAgent 主类：手写 ReAct 循环（半状态机）。

设计要点（来自 SYNTHESIS §1-3 + patterns/state-machine-react.md）：
- 状态机骨架：INIT → THOUGHT → ACTION → OBSERVE → ... → FINAL → TERMINATE
- LLM 决策 action，agent 负责 state 转移 + 工具路由 + 错误兜底
- max_steps=10（README 默认 7 偏紧，多步任务要 7-10 步）
- 终止条件：Final Answer / 步数耗尽 / 连续 3 步 Thought 重复
- 所有异常 → Observation 字符串，绝不冒泡出主循环
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .llm_client import LLMClient, LLMConfig, LLMError
from .parser import ActionParser
from .prompt import PromptBuilder
from .tools import default_registry
from .tools.base import ToolRegistry
from .trace import AgentTrace, make_step, trace_to_text

log = logging.getLogger(__name__)

# 状态常量
STATE_INIT = "INIT"
STATE_THOUGHT = "THOUGHT"
STATE_ACTION = "ACTION"
STATE_OBSERVE = "OBSERVE"
STATE_RETRY = "RETRY"
STATE_FINAL = "FINAL"
STATE_TERMINATE = "TERMINATE"


class ReActAgent:
    """手写 ReAct agent（状态机驱动）。

    参数：
        llm_client   : LLMClient 实例（None 则自动建）
        tools        : ToolRegistry（None 则用 4 个默认工具）
        max_steps    : 最大步数（默认 10）
        model        : 模型名（None 则走 env 或默认）
        few_shot_count: S3 用，few-shot 数量（0/1/3）
        include_error_hint: S3 用，是否在 system 加错误恢复提示
    """

    DEFAULT_MAX_STEPS = 10

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        model: str | None = None,
        few_shot_count: int = 3,
        include_error_hint: bool = True,
    ) -> None:
        if tools is None:
            tools = default_registry()
        self.tools = tools
        self.max_steps = max(2, int(max_steps))
        if llm_client is None:
            cfg = LLMConfig()
            if model:
                cfg.model = model
            llm_client = LLMClient(cfg)
        self.llm = llm_client
        # 兼容假 LLM（无 .config）：仅在有 config 时取 model
        self.model = model or getattr(
            getattr(self.llm, "config", None), "model", "stub-model"
        )
        self.prompt_builder = PromptBuilder(
            tool_schemas=self.tools.schema_list(),
            few_shot_count=few_shot_count,
            include_error_hint=include_error_hint,
        )
        self.parser = ActionParser()

    # ============================================================ 主入口
    def run(self, task: str) -> AgentTrace:
        """执行一次 ReAct 循环，返回 AgentTrace dict。

        AgentTrace = {"steps": [...], "final_answer": str, "success": bool}
        """
        messages = self.prompt_builder.initial_messages(task)
        steps: list[dict[str, Any]] = []
        state = STATE_INIT
        final_answer: str = ""
        success = False

        for step_idx in range(self.max_steps):
            # ----- INIT / THOUGHT
            state = STATE_THOUGHT
            try:
                response = self.llm.chat(
                    messages, model=self.model
                )
            except LLMError as e:
                # LLM 不可达 → 终止（避免空跑）
                log.warning("LLM 不可达：%s", e)
                steps.append(make_step(
                    step_idx, state,
                    thought=f"[LLMError] {e}",
                ))
                final_answer = self._best_effort(steps)
                return self._finalize(steps, final_answer, success=False)

            parsed = self.parser.parse(response)

            # ----- RETRY：解析失败
            if self.parser.is_retry(parsed):
                state = STATE_RETRY
                steps.append(make_step(
                    step_idx, state,
                    thought=parsed.get("thought", ""),
                ))
                messages.append({
                    "role": "user",
                    "content": self.prompt_builder.retry_message(
                        parsed.get("reason", "")
                    ),
                })
                continue

            # ----- 终止条件：Final Answer
            if parsed["action"] == "Final Answer":
                state = STATE_FINAL
                final_answer = str(parsed["action_input"])
                steps.append(make_step(
                    step_idx, state,
                    thought=parsed["thought"],
                    action="Final Answer",
                    action_input=final_answer,
                ))
                success = True
                break

            # ----- ACTION：路由 + 调用
            state = STATE_ACTION
            action = parsed["action"]
            action_input = parsed["action_input"]
            observation = self.tools.call(action, action_input)
            step = make_step(
                step_idx, state,
                thought=parsed["thought"],
                action=action,
                action_input=action_input,
                observation=observation,
            )
            steps.append(step)

            # ----- OBSERVE：拼回 prompt
            state = STATE_OBSERVE
            messages = self.prompt_builder.append_observation(
                messages,
                thought=parsed["thought"],
                action=action,
                action_input=action_input,
                observation=observation,
            )

            # ----- 卡死检测
            if self._is_stuck(steps):
                final_answer = self._best_effort(steps)
                return self._finalize(steps, final_answer, success=False)

        # ----- 步数耗尽
        if not final_answer:
            final_answer = self._best_effort(steps)
        return self._finalize(steps, final_answer, success=success)

    # ============================================================ 终止辅助
    def _is_stuck(self, steps: list[dict[str, Any]]) -> bool:
        """最近 3 步 Thought 字符串完全相同 → 卡死。"""
        if len(steps) < 3:
            return False
        recent = [s.get("thought", "").strip() for s in steps[-3:]]
        return len({t for t in recent if t}) <= 1

    def _best_effort(self, steps: list[dict[str, Any]]) -> str:
        """步数耗尽时从最后几条 Observation 提取答案。"""
        for s in reversed(steps):
            obs = s.get("observation", "")
            if obs and not s.get("is_error"):
                return obs[:300]
        return ""

    def _finalize(
        self,
        steps: list[dict[str, Any]],
        final_answer: str,
        success: bool,
    ) -> AgentTrace:
        return {
            "steps": steps,
            "final_answer": final_answer,
            "success": success,
        }

    # ============================================================ S4 钩子
    def inject_error(self, tool_name: str, msg: str = "[Injected Error]") -> None:
        """S4 钩子：在指定工具的下一次调用返回 [ERROR: msg]。"""
        self.tools.inject_error(tool_name, msg)

    def set_error_rate(self, tool_name: str, rate: float,
                       msg: str = "[Injected Error]") -> None:
        """S4 钩子：按概率注入错误。"""
        self.tools.set_error_rate(tool_name, rate, msg)

    def clear_errors(self) -> None:
        """清空所有错误注入。"""
        self.tools.clear_errors()

    # ============================================================ 调试
    def __repr__(self) -> str:
        return (
            f"ReActAgent(model={self.model!r}, "
            f"max_steps={self.max_steps}, "
            f"tools={self.tools.names()})"
        )


# --------------------------------------------------------------------- smoke
def _smoke_self_test() -> dict:
    """无 LLM 时跑最小自检（验证状态机骨架正确）。

    用一个返回固定字符串的假 LLMClient 跑两轮 ReAct。
    """
    from .llm_client import LLMConfig

    class _FakeLLM:
        """按队列返回固定文本的假客户端。"""

        def __init__(self, replies: list[str]) -> None:
            self.replies = list(replies)
            self.calls: list[list[dict]] = []

        def chat(self, messages, model=None, **kw):
            self.calls.append(list(messages))
            if self.replies:
                return self.replies.pop(0)
            return "Thought: 没有更多回复\nAction: Final Answer\nAction Input: 兜底答案"

    fake = _FakeLLM([
        "Thought: 算乘积\nAction: calculator\n"
        "Action Input: {\"expression\": \"(123+456)*789\"}",
        "Thought: 6 位\nAction: Final Answer\nAction Input: 6 位",
    ])
    agent = ReActAgent(llm_client=fake, max_steps=5)
    trace = agent.run("计算 (123+456)*789 是几位数")
    return trace


if __name__ == "__main__":
    import json
    trace = _smoke_self_test()
    print(trace_to_text(trace))
    print("---")
    print(json.dumps(trace, ensure_ascii=False, indent=2))


__all__ = ["ReActAgent", "trace_to_text"]