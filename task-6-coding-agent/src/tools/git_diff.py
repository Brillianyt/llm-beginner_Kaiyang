"""git_diff tool — read the repo diff (read-only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import BaseTool, check_blocked_git, run_subprocess


class GitDiffTool(BaseTool):
    name: ClassVar[str] = "git_diff"
    description: ClassVar[str] = (
        "Show `git diff` (working tree vs HEAD) for the repository. "
        "Returns the unified diff text plus the exit code. Dangerous "
        "commands (`reset --hard`, `clean -fd`, etc.) are filtered out."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional repo-relative paths to limit the diff",
            },
        },
    }
    is_read_only: ClassVar[bool] = True

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        paths = args.get("paths") or []
        if not all(isinstance(p, str) for p in paths):
            return "[ERROR] paths must be list of strings"
        cmd = ["git", "--no-pager", "diff"]
        cmd.extend(paths)
        blocked = check_blocked_git(cmd)
        if blocked:
            return f"[ERROR] {blocked}"
        result = run_subprocess(cmd, cwd=repo_root, timeout=30)
        body = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0 and not body:
            body = "(git diff failed; repo may not have a HEAD yet)"
        if not body.strip():
            body = "(no changes)"
        return f"exit_code={result.returncode}\n{body[-12000:]}"
