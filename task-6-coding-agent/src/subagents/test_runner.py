"""Test-runner subagent — runs the test suite and summarises failures."""
from __future__ import annotations

from typing import List

from .base import BaseSubagent


class TestRunnerSubagent(BaseSubagent):
    name = "test_runner"
    max_steps = 3
    allowed_tools: List[str] = ["run_tests", "read_file"]
    system_prompt = (
        "You are the **test_runner** subagent.\n\n"
        "Your job: run the project's pytest suite and report the outcome. You "
        "may only call `run_tests` (and optionally `read_file` to inspect "
        "the failure context). You may **never** edit files.\n\n"
        "Always start by calling `run_tests` with no extra args. If the suite "
        "fails, re-read the relevant file with `read_file` only when needed "
        "to disambiguate. Then return a concise summary like:\n"
        "  status: failed\n"
        "  failing: test_add_negative_number (calculator.py:5 returns a - b)\n"
        "  short reason: 'add(-2, 3)' returned -5, expected 1.\n\n"
        "Keep it under 300 words. The coordinator will decide what to do next."
    )
