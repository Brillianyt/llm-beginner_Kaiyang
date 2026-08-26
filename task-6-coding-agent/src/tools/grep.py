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
        "A powerful search tool built on ripgrep.\n"
        "\n"
        "Usage:\n"
        "  - ALWAYS use `grep` for search tasks. NEVER invoke `grep`/`rg` as "
        "a `run_bash` command — `grep` is optimised for sandbox-correct access.\n"
        "  - Supports full ripgrep regex (e.g., \"log.*Error\", "
        "\"function\\\\s+\\\\w+\"). Literal braces need escaping: "
        "`interface\\\\{\\\\}` finds `interface{}` in Go code.\n"
        "  - Filter files with `glob` (e.g., \"*.py\", \"*.{ts,tsx}\") or "
        "`type` (e.g., \"py\", \"rust\"); narrow `path` to a directory.\n"
        "  - Output modes: \"content\" shows matching lines (supports "
        "`context=N` and `-n` line numbers), \"files_with_matches\" shows only "
        "file paths (default), \"count\" shows per-file match counts.\n"
        "  - When the issue text doesn't name a file or you're unsure which "
        "file to edit, use `grep(pattern='<keyword>', "
        "output_mode='files_with_matches')` first to locate the buggy file; "
        "then `read_file` it.\n"
        "  - Use `output_mode='content'` with `context=N` to see surrounding "
        "lines.\n"
        "  - Pass `-i=true` for case-insensitive search (useful for "
        "case-sensitivity bugs — try searching the lowercase form first).\n"
        "  - Use `head_limit=N` to cap large result sets; the tool reports "
        "`applied_limit` when truncated so you can paginate with "
        "`offset`/`head_limit`.\n"
        "  - Multiline patterns need `multiline=true` (e.g., "
        "`struct \\\\{\\\\[\\\\s\\\\S]*?field`)."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regular expression pattern to search for in file contents (ripgrep syntax).",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in (rg PATH). Defaults to repo root.",
                "default": ".",
            },
            "glob": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g., \"*.py\", \"*.{ts,tsx}\") — maps to rg --glob.",
            },
            "type": {
                "type": "string",
                "description": "File type to search (rg --type). Common types: py, js, rust, go, java.",
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": (
                    "Output mode. \"content\" shows matching lines (supports "
                    "context=N, -n line numbers). \"files_with_matches\" "
                    "shows file paths only (default, best for locating). "
                    "\"count\" shows per-file match counts."
                ),
                "default": "files_with_matches",
            },
            "context": {
                "type": "integer",
                "minimum": 0,
                "maximum": 20,
                "description": "Lines of context before/after each match (rg -C). Requires output_mode=content.",
                "default": 0,
            },
            "-i": {
                "type": "boolean",
                "description": "Case-insensitive search (rg -i).",
                "default": False,
            },
            "multiline": {
                "type": "boolean",
                "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default false.",
                "default": False,
            },
            "head_limit": {
                "type": "integer",
                "minimum": 0,
                "description": "Limit output to first N lines/entries (default 250). Pass 0 for unlimited (use sparingly — large result sets waste context).",
                "default": 250,
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Skip first N lines/entries before applying head_limit, equivalent to \"| tail -n +N | head -N\". Default 0.",
                "default": 0,
            },
        },
        "required": ["pattern"],
    }
    is_read_only: ClassVar[bool] = True
    max_result_chars: ClassVar[int] = 20_000

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        pattern = args.get("pattern", "")
        if not pattern:
            return "[ERROR] pattern is required"
        path_arg = args.get("path") or "."
        type_filter = args.get("type") or ""
        glob_filter = args.get("glob") or ""
        output_mode = args.get("output_mode") or "files_with_matches"
        context = int(args.get("context") or 0)
        case_insensitive = bool(args.get("-i"))
        multiline = bool(args.get("multiline"))
        head_limit = args.get("head_limit")
        if head_limit is None:
            head_limit = 250
        else:
            head_limit = int(head_limit)
        offset = int(args.get("offset") or 0)

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

        cmd = ["rg", "--no-heading"]
        if output_mode == "content":
            cmd.append("-n")
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        if case_insensitive:
            cmd.append("-i")
        if multiline:
            cmd.extend(["-U", "--multiline-dotall"])
        if context and output_mode == "content":
            cmd.extend(["-C", str(context)])
        if glob_filter:
            cmd.extend(["--glob", glob_filter])
        if type_filter:
            cmd.extend(["--type", type_filter])

        # head_limit / offset: rg doesn't have a clean --offset for
        # files_with_matches mode, so we apply offset/head_limit post-hoc
        # to the captured stdout. For content mode this trims matching lines;
        # for files_with_matches it trims file paths.
        cap_lines: int | None = None
        if head_limit and head_limit > 0:
            cap_lines = head_limit + offset

        cmd.extend(["--", pattern, str(search_path)])

        cp = run_subprocess(cmd, cwd=repo_root, timeout=30)
        out = (cp.stdout or "")
        if not out.strip():
            return "(no matches)"
        # Apply offset + head_limit on the captured output.
        all_lines = out.splitlines()
        total = len(all_lines)
        if offset:
            all_lines = all_lines[offset:]
        truncated_at = None
        if head_limit and head_limit > 0 and len(all_lines) > head_limit:
            all_lines = all_lines[:head_limit]
            truncated_at = head_limit
        out = "\n".join(all_lines)
        if truncated_at is not None or (head_limit == 0 and cap_lines is None):
            # We applied a limit and there were more.
            applied_note = ""
            if offset:
                applied_note += f"offset={offset}, "
            applied_note += f"limit={head_limit}" if head_limit else "limit=unbounded"
            footer = f"\n...[truncated at {applied_note}; {total} total entries; pass offset/head_limit to paginate]"
            return (out + footer)[: self.max_result_chars]
        if len(out) > self.max_result_chars:
            return out[: self.max_result_chars] + "\n...[truncated by char limit]..."
        return out
