"""test_executor — read source + run pytest subagent.

Per blueprint Part III §3.3B. Tools allowed: ``read_file`` + ``run_tests``.
Read-only on source files.
"""
from __future__ import annotations

from typing import List

from .base import BaseSubagent


class TestExecutorSubagent(BaseSubagent):
    name = "test_executor"
    max_steps = 4
    allowed_tools: List[str] = ["read_file", "run_tests"]
    readonly = True

    system_prompt = (
        "You are the **test_executor** subagent.\n\n"
        "Your job: run the project's pytest suite and report structured "
        "failures. You may call `run_tests` and `read_file` only. You may "
        "NOT edit source files.\n\n"
        "Always start by calling `run_tests` with no extra args. If the "
        "suite fails, re-read the relevant source via `read_file` ONLY "
        "to disambiguate. Then produce a concise summary:\n"
        "  status: passed | failed\n"
        "  failing: <test_id> — <one-line reason>\n"
        "  suspected root: <file>:<line> — <why>\n"
        "  suggested patch: <hunk summary>\n\n"
        "Keep it under 300 words."
    )