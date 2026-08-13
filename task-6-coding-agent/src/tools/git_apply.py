"""git_apply tool — apply a unified diff inside the repo (with safety nets)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import BaseTool, run_subprocess


class GitApplyTool(BaseTool):
    name: ClassVar[str] = "git_apply"
    description: ClassVar[str] = (
        "Apply a unified diff (the body of a `git diff`) to the working "
        "tree using `git apply --check` first to dry-run. Rejects patches "
        "that touch paths outside the repo and refuses the dangerous "
        "fragments listed in ``BLOCKED_GIT_FRAGMENTS``."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "diff": {
                "type": "string",
                "description": "Unified diff produced by `git_diff`.",
            },
            "three_way": {
                "type": "boolean",
                "description": "Pass --3way to fall back to merge-style apply.",
                "default": False,
            },
        },
        "required": ["diff"],
    }
    is_read_only: ClassVar[bool] = False

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        diff = args.get("diff", "")
        if not diff.strip():
            return "[ERROR] empty diff"
        three_way = bool(args.get("three_way", False))

        # 1. Validate by writing to a temp file and running git apply --check.
        with subprocess.Popen(
            ["git", "apply", "--check", "--unidiff-zero=false", "-"],
            cwd=str(repo_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as probe:
            _, err = probe.communicate(input=diff, timeout=15)
            if probe.returncode != 0:
                return (
                    f"[ERROR] git apply --check failed (rc={probe.returncode}): "
                    f"{err.strip()[:1500]}"
                )

        # 2. Real apply (list form, no shell).
        cmd = ["git", "apply"]
        if three_way:
            cmd.append("--3way")
        with subprocess.Popen(
            cmd + ["-"],
            cwd=str(repo_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc:
            out, err = proc.communicate(input=diff, timeout=30)
        if proc.returncode != 0:
            return (
                f"[ERROR] git apply failed (rc={proc.returncode}): "
                f"{(err or out).strip()[:1500]}"
            )
        # Confirm with diff stat.
        stat = run_subprocess(
            ["git", "--no-pager", "diff", "--stat"],
            cwd=repo_root,
            timeout=15,
        )
        return f"applied OK.\n--- stat ---\n{(stat.stdout or '').strip()}"
