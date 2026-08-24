"""grep — ripgrep-backed content search (read-only).

Claude Code's Grep tool wrapper. Searches repo-relative or absolute
paths, honours .gitignore, caps output.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import BaseTool, run_subprocess, safe_resolve


class GrepTool(BaseTool):
    name: ClassVar[str] = "grep"
    description: ClassVar[str] = (
        "A powerful content search tool built on ripgrep. Returns matching "
        "lines (or file paths) for a regex pattern. Use it to find where "
        "a symbol is defined or referenced — e.g. `grep(pattern='def L031', "
        "output_mode='content')`. Path can be repo-relative or absolute "
        "inside the repo."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern (ripgrep syntax).",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default: repo root).",
                "default": ".",
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": (
                    "content: matching lines; files_with_matches: only file "
                    "paths (default); count: per-file match counts."
                ),
                "default": "files_with_matches",
            },
            "context": {
                "type": "integer",
                "minimum": 0,
                "maximum": 20,
                "description": "Lines of context before/after each match.",
                "default": 0,
            },
        },
        "required": ["pattern"],
    }
    is_read_only: ClassVar[bool] = True
    max_result_chars: ClassVar[int] = 30_000

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        pattern = args.get("pattern", "")
        if not pattern:
            return "[ERROR] pattern is required"
        path_arg = args.get("path") or "."
        output_mode = args.get("output_mode") or "files_with_matches"
        context = int(args.get("context") or 0)

        # Normalise path (relative or absolute-inside-repo).
        if path_arg in (".", ""):
            search_path = repo_root
        else:
            p = Path(path_arg)
            if p.is_absolute():
                try:
                    rel = p.resolve().relative_to(repo_root)
                except ValueError:
                    return f"[ERROR] path escapes repo root: {path_arg}"
                search_path = repo_root / rel
            else:
                try:
                    search_path = safe_resolve(path_arg, repo_root)
                except PermissionError as e:
                    return f"[ERROR] {e}"
        if not search_path.exists():
            return f"[ERROR] path not found: {path_arg}"

        if shutil.which("rg") is None:
            return "[ERROR] ripgrep (`rg`) not installed — install it or use list_files + read_file"

        cmd = ["rg", "--no-heading", "-n"]
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        if context:
            cmd.extend(["-C", str(context)])
        cmd.extend(["--", pattern, str(search_path)])

        cp = run_subprocess(cmd, cwd=repo_root, timeout=30)
        out = (cp.stdout or "").strip()
        if not out:
            return "(no matches)"
        return out[: self.max_result_chars] + ("\n...[truncated]..." if len(out) > self.max_result_chars else "")
