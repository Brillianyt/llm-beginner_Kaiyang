"""search_executor — read-only code search subagent.

Per blueprint Part III §3.3A. Tools allowed: ``read_file``. Never
modifies files, never runs Bash.
"""
from __future__ import annotations

from typing import List

from .base import BaseSubagent


class SearchExecutorSubagent(BaseSubagent):
    name = "search_executor"
    max_steps = 5
    allowed_tools: List[str] = ["read_file"]
    readonly = True

    system_prompt = (
        "You are the **search_executor** subagent.\n\n"
        "Your job: locate symbols, file locations, and short snippets. You "
        "may only call `read_file`. You may NEVER modify files or run tests.\n\n"
        "When you have enough information, stop calling tools and produce a "
        "concise plain-text summary (no markdown headings, no JSON). Keep "
        "it under 300 words. The coordinator will synthesise your findings "
        "with its own; do not return long traces."
    )