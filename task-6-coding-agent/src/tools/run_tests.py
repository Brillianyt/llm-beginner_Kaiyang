"""run_tests tool — invoke ``python -m pytest`` safely in the repo."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import BaseTool, run_subprocess


class RunTestsTool(BaseTool):
    name: ClassVar[str] = "run_tests"
    description: ClassVar[str] = (
        "Run the project's pytest suite in the repository root. Uses "
        "`python -m pytest -q` by default. Returns the last ~8_000 chars "
        "of combined stdout+stderr plus the exit code. Never executes a "
        "shell — command is always a list."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra pytest flags, e.g. ['-x', '-k', 'add']",
            },
            "timeout": {
                "type": "integer",
                "description": "Hard cap (seconds). Default 60; raise for SWE-bench.",
                "minimum": 5,
                "maximum": 3600,
                "default": 60,
            },
        },
    }
    is_read_only: ClassVar[bool] = False  # may touch pytest cache / venv

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        extra = args.get("extra_args") or []
        if not all(isinstance(s, str) for s in extra):
            return "[ERROR] extra_args must be list of strings"
        timeout = int(args.get("timeout", 60))
        # Command list — never shell=True.
        cmd = [sys.executable, "-m", "pytest", "-q", *extra]
        result = run_subprocess(cmd, cwd=repo_root, timeout=timeout)
        blob = (result.stdout or "") + (result.stderr or "")
        tail = blob[-8000:]
        ok = result.returncode == 0
        summary = (
            f"exit_code={result.returncode} "
            f"({'passed' if ok else 'FAILED'})\n"
            f"--- pytest output (tail) ---\n{tail}"
        )
        return summary
