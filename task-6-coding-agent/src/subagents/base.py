"""Subagent base class.

Per blueprint Part III §3.2:

* Independent message history (the parent never sees the child's log),
* Forced summary truncation (≤ 2 KB by default),
* Tool allowlist enforced per-subagent,
* Step budget independent of the parent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm_client import LLMClient, LLMError, to_wire_tool_calls
from src.mcp_server import call_tool

log = logging.getLogger("subagents")

MAX_SUMMARY_CHARS = 2048  # 2 KB hard cap (blueprint §3.2 point 2)


@dataclass
class TranscriptEntry:
    """One step in a subagent's internal trace.

    Captures enough state to **replay** the subagent's run without
    re-running the LLM: which tool was called, with which arguments,
    and what observation came back.
    """
    kind: str  # "tool_call" | "observation" | "summary"
    name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    error: bool = False
    ts: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "arguments": self.arguments,
            "observation": self.observation,
            "error": self.error,
            "ts": self.ts,
        }


@dataclass
class SubagentResult:
    name: str
    task: str
    summary: str
    steps: int
    error: Optional[str] = None
    # Full internal trace — used by ``TraceReplay`` to dry-run the
    # subagent without the LLM. Empty list if the run errored before
    # any tool was called.
    transcript: List[TranscriptEntry] = field(default_factory=list)


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
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in ALL_TOOLS if t.name in self.allowed_tools
        ]
        self._tool_names = set(self.allowed_tools)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run(self, task: str, repo_root: str) -> SubagentResult:
        """Drive the subagent loop until it stops or hits ``max_steps``."""
        from datetime import datetime
        def _now() -> str:
            return datetime.utcnow().isoformat() + "Z"

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        steps = 0
        error: Optional[str] = None
        summary = ""
        transcript: List[TranscriptEntry] = []
        try:
            while steps < self.max_steps:
                steps += 1
                resp = self.llm.chat(messages, tools=self._tool_schemas or None)
                msg = resp.message
                if not msg.tool_calls:
                    summary = msg.content or ""
                    transcript.append(TranscriptEntry(
                        kind="summary", observation=summary, ts=_now(),
                    ))
                    break

                # Echo assistant turn before tool results.
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": to_wire_tool_calls(msg.tool_calls),
                })
                for tc in msg.tool_calls:
                    fn = tc["function"]
                    name = fn["name"]
                    args = fn["arguments"] if isinstance(fn["arguments"], dict) else {}
                    transcript.append(TranscriptEntry(
                        kind="tool_call", name=name, arguments=args, ts=_now(),
                    ))
                    if name not in self._tool_names:
                        observation = f"[ERROR] tool '{name}' not in allowlist"
                        is_err = True
                    else:
                        result = call_tool(name, args, repo_root)
                        observation = (
                            f"[ERROR] {result.content}" if result.is_error else result.content
                        )
                        is_err = result.is_error
                    transcript.append(TranscriptEntry(
                        kind="observation", name=name,
                        observation=observation, error=is_err, ts=_now(),
                    ))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": observation[:6000],
                    })
            else:
                summary = "(subagent hit max_steps without emitting text)"
                transcript.append(TranscriptEntry(
                    kind="summary", observation=summary, ts=_now(),
                ))
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
            transcript=transcript,
        )


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n...[truncated 2KB]..."