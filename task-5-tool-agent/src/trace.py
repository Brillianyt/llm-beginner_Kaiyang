"""AgentTrace 数据结构 + 格式化工具。

AgentTrace = dict 含 steps / final_answer / success
每步 step 含：
- step_idx        : int
- state           : str  (THOUGHT/ACTION/FINAL/...)
- thought         : str
- action          : str  (工具名 / "Final Answer")
- action_input    : Any  (dict 或 str)
- observation     : str  (工具返回；Final Answer 步无 observation)
- is_error        : bool (observation 以 [ERROR: 开头)
"""
from __future__ import annotations

import json
from typing import Any

AgentTrace = dict[str, Any]
Step = dict[str, Any]


def make_step(
    step_idx: int,
    state: str,
    thought: str = "",
    action: str = "",
    action_input: Any = None,
    observation: str = "",
) -> Step:
    """构造一个 step dict。"""
    return {
        "step_idx": step_idx,
        "state": state,
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "observation": observation,
        "is_error": isinstance(observation, str)
                     and observation.startswith("[ERROR:"),
    }


def trace_to_text(trace: AgentTrace) -> str:
    """把 trace 格式化成可读字符串（写 result.json / 调试用）。"""
    lines: list[str] = []
    for s in trace.get("steps", []):
        idx = s.get("step_idx", 0) + 1
        state = s.get("state", "?")
        lines.append(f"[{state}] step {idx}")
        if s.get("thought"):
            lines.append(f"  Thought: {s['thought']}")
        if s.get("action"):
            ai = s.get("action_input", "")
            if not isinstance(ai, str):
                ai = json.dumps(ai, ensure_ascii=False)
            lines.append(f"  Action: {s['action']}")
            lines.append(f"  Action Input: {ai}")
        obs = s.get("observation", "")
        if obs:
            lines.append(f"  Observation: {obs}")
    if trace.get("final_answer") is not None:
        lines.append(f"Final Answer: {trace['final_answer']}")
    lines.append(f"Success: {trace.get('success', False)}")
    return "\n".join(lines)


__all__ = ["AgentTrace", "Step", "make_step", "trace_to_text"]