"""Subagent implementations.

Three independent attributes per the design doc:

1. **Independent message list** — each subagent seeds its own messages
   (``[system, user_task]``) and never sees the parent's history.
2. **Independent step budget** — ``max_steps`` defaults to 5, far below
   the parent's 30.
3. **Tool allowlist** — ``allowed_tools`` is enforced by ``BaseSubagent``.

The main agent receives only the subagent's final plain-text summary —
never the subagent's trace or messages.
"""
from .base import BaseSubagent, SubagentResult
from .code_search import CodeSearchSubagent
from .test_runner import TestRunnerSubagent

__all__ = [
    "BaseSubagent",
    "SubagentResult",
    "CodeSearchSubagent",
    "TestRunnerSubagent",
]
