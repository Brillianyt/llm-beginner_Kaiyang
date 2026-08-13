"""Code-search subagent — read-only exploration of the repository."""
from __future__ import annotations

from typing import List

from .base import BaseSubagent


class CodeSearchSubagent(BaseSubagent):
    name = "code_search"
    max_steps = 5
    allowed_tools: List[str] = ["read_file", "list_files"]
    system_prompt = (
        "You are the **code_search** subagent.\n\n"
        "Your job: locate symbols, file locations, and short snippets in the "
        "repository. You may only call `read_file` and `list_files`. You may "
        "**never** modify files or run tests.\n\n"
        "When you have enough information, stop calling tools and produce a "
        "**concise plain-text summary** (no markdown headings, no JSON) — e.g.\n"
        "  calculator.add is defined in calculator.py:2; current body is\n"
        "      def add(a, b):\n"
        "          return a - b\n"
        "  The expected behaviour per the tests is a + b.\n\n"
        "Keep the summary under 300 words. The coordinator will synthesize "
        "your findings with its own; do not return long traces."
    )
