"""Subagent base class — implements the three isolation guarantees.

Borrowed from Claude Code ``src/coordinator/workerAgent.ts``:

* Each worker has a **dedicated system prompt** that ends with
  "Report back with concise summary — the coordinator will synthesize
  your results".
* The orchestrator passes the worker only the **tools it should be able
  to call** (``allowed_tools``). Internal orchestration tools are
  excluded so a worker cannot recursively spawn more workers.
* Workers expose only ``summary`` text — never their trace or message
  log — back to the coordinator.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.llm_client import LLMClient, LLMError
from src.mcp_server import call_tool
from src.tools import ALL_TOOLS

log = logging.getLogger("subagents")


@dataclass
class SubagentResult:
    name: str
    task: str
    summary: str
    steps: int
    error: Optional[str] = None


class BaseSubagent:
    """Shared scaffolding for all subagents.

    Subclasses set ``name``, ``system_prompt`` and ``allowed_tools``.
    The main agent only ever calls :meth:`run`.
    """

    name: str = "base"
    system_prompt: str = "You are a subagent."
    allowed_tools: List[str] = []
    max_steps: int = 5

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        # Filter the global tool registry to this subagent's allowlist.
        self._tool_schemas = [
            t.input_schema
            for t in ALL_TOOLS
            if t.name in self.allowed_tools
        ]
        self._tool_names = set(self.allowed_tools)

    # -- public ------------------------------------------------------------

    def run(self, task: str, repo_root: str) -> SubagentResult:
        """Drive the subagent loop until it stops or hits ``max_steps``."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        steps = 0
        error: Optional[str] = None
        summary = ""
        while steps < self.max_steps:
            steps += 1
            try:
                resp = self.llm.chat(messages, tools=self._tool_schemas)
            except LLMError as e:
                error = f"llm_error: {e}"
                log.warning("[%s] llm error: %s", self.name, e)
                break

            msg = resp.message
            if not msg.tool_calls:
                summary = msg.content or ""
                break

            # Append the assistant message (OpenAI requires echoing
            # tool_calls before the tool messages).
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                }
            )
            for tc in msg.tool_calls:
                fn = tc["function"]
                if fn["name"] not in self._tool_names:
                    observation = f"[ERROR] tool '{fn['name']}' is not in this subagent's allowlist"
                else:
                    result = call_tool(fn["name"], fn["arguments"], repo_root)
                    observation = (
                        f"[ERROR] {result.content}" if result.is_error else result.content
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": observation[:6000],
                    }
                )
        else:
            # Loop exited via the ``while`` condition (max steps).
            summary = "(subagent hit max_steps without emitting text)"
            log.warning("[%s] max_steps=%d reached", self.name, self.max_steps)

        return SubagentResult(
            name=self.name,
            task=task,
            summary=summary or "(no summary)",
            steps=steps,
            error=error,
        )
