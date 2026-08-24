"""Subagent base class.

Per blueprint Part III §3.2:

* Independent message history (the parent never sees the child's log),
* Forced summary truncation (≤ 2 KB by default),
* Tool allowlist enforced per-subagent,
* Step budget independent of the parent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.llm_client import LLMClient, LLMError
from src.mcp_server import call_tool

log = logging.getLogger("subagents")

MAX_SUMMARY_CHARS = 2048  # 2 KB hard cap (blueprint §3.2 point 2)


@dataclass
class SubagentResult:
    name: str
    task: str
    summary: str
    steps: int
    error: Optional[str] = None


class BaseSubagent:
    """Shared scaffolding — subclasses set ``name`` / ``system_prompt`` /
    ``allowed_tools`` / ``max_steps``.

    A subagent is *read-only by default* (per blueprint §3.1) — only
    enable Write/Edit for subagents that need it (e.g. a future
    test-fixer subagent).
    """

    name: str = "base"
    system_prompt: str = "You are a subagent."
    allowed_tools: List[str] = []
    max_steps: int = 5
    readonly: bool = True

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        # Build the subagent's view of available tools.
        from src.tools import ALL_TOOLS  # local import avoids cycle
        self._tool_schemas: List[Dict[str, Any]] = [
            t.input_schema for t in ALL_TOOLS if t.name in self.allowed_tools
        ]
        self._tool_names = set(self.allowed_tools)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run(self, task: str, repo_root: str) -> SubagentResult:
        """Drive the subagent loop until it stops or hits ``max_steps``."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        steps = 0
        error: Optional[str] = None
        summary = ""
        try:
            while steps < self.max_steps:
                steps += 1
                resp = self.llm.chat(messages, tools=self._tool_schemas or None)
                msg = resp.message
                if not msg.tool_calls:
                    summary = msg.content or ""
                    break

                # Echo assistant turn before tool results.
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                })
                for tc in msg.tool_calls:
                    fn = tc["function"]
                    name = fn["name"]
                    args = fn["arguments"] if isinstance(fn["arguments"], dict) else {}
                    if name not in self._tool_names:
                        observation = f"[ERROR] tool '{name}' not in allowlist"
                    else:
                        result = call_tool(name, args, repo_root)
                        observation = (
                            f"[ERROR] {result.content}" if result.is_error else result.content
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": observation[:6000],
                    })
            else:
                summary = "(subagent hit max_steps without emitting text)"
                log.warning("[%s] max_steps=%d reached", self.name, self.max_steps)
        except LLMError as e:
            error = f"llm_error: {e}"
            log.warning("[%s] llm error: %s", self.name, e)

        return SubagentResult(
            name=self.name,
            task=task,
            summary=_truncate(summary or "(no summary)", MAX_SUMMARY_CHARS),
            steps=steps,
            error=error,
        )


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n...[truncated 2KB]..."