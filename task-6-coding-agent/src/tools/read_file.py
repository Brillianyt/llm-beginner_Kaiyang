"""read_file tool — bounded file reading with strict path sandboxing."""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import BaseTool, safe_resolve


class ReadFileTool(BaseTool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read a UTF-8 text file from the repository. Returns at most "
        "`max_chars` characters (default 50_000). Path must be relative "
        "and stay inside the repository root — absolute paths and "
        "traversal ('..') are rejected."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repo-relative path, e.g. 'calculator.py'",
            },
            "max_chars": {
                "type": "integer",
                "description": "Soft cap on returned characters (default 50000)",
                "minimum": 100,
                "maximum": 200000,
                "default": 50000,
            },
        },
        "required": ["path"],
    }
    is_read_only: ClassVar[bool] = True

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        path_str = args["path"]
        max_chars = int(args.get("max_chars", 50000))
        target = safe_resolve(path_str, repo_root)
        if not target.exists():
            return f"[ERROR] file not found: {path_str}"
        if not target.is_file():
            return f"[ERROR] not a regular file: {path_str}"
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...[truncated at {max_chars} chars]..."
        # Echo a small prefix with the path for human/LLM clarity.
        return f"=== {path_str} ({len(text)} chars) ===\n{text}"
