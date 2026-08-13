"""write_file tool — guarded file writer with path + content checks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import BaseTool, safe_resolve


class WriteFileTool(BaseTool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create or overwrite a UTF-8 text file inside the repository. "
        "Parent directories are created if missing. Path must be relative "
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
            "content": {
                "type": "string",
                "description": "Full new file contents (UTF-8).",
            },
        },
        "required": ["path", "content"],
    }
    is_read_only: ClassVar[bool] = False

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        path_str = args["path"]
        content = args["content"]
        target = safe_resolve(path_str, repo_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return f"wrote {len(content)} bytes to {path_str}"
