"""read_file — bounded file reading with cat -n line numbers.

Per file-system-spec §1:

* path must be absolute, inside the repo,
* default reads up to 2000 lines from the start,
* ``offset`` / ``limit`` allow paging long files,
* returns ``cat -n`` style content with line numbers,
* max file size 256 KB (before read) — caller pages via offset/limit.

The return shape is documented in blueprint Part I §1.1:

    {
      "file_path": "<abs>",
      "content": "<cat -n body>",
      "num_lines": N,
      "start_line": 0,
      "total_lines": M,
      "encoding": "utf-8",
      "truncated": bool
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import BaseTool, safe_resolve


class ReadFileTool(BaseTool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Reads a UTF-8 text file from the local filesystem. "
        "The file_path must be an absolute path. By default reads up to "
        "2000 lines from the start; use `offset`/`limit` for paging long "
        "files. Result is returned in `cat -n` format (line numbers starting "
        "at 1). Files larger than 256 KB are rejected — page with "
        "offset/limit."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file (must be inside the repo root).",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Line number to start reading from (0-based).",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2000,
                "description": "Maximum number of lines to read (default 2000).",
            },
        },
        "required": ["file_path"],
    }
    is_read_only: ClassVar[bool] = True
    max_result_chars: ClassVar[int] = 200_000  # cat -n body can be larger

    SIZE_LIMIT_BYTES = 256 * 1024
    DEFAULT_LIMIT = 2000

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        path_str = args["file_path"]
        offset = int(args.get("offset", 0) or 0)
        limit = int(args.get("limit", self.DEFAULT_LIMIT) or self.DEFAULT_LIMIT)

        # safe_resolve requires a relative path; we re-implement the check
        # here to keep the absolute-path contract from file-system-spec §1.2.
        p = Path(path_str)
        if not p.is_absolute():
            raise PermissionError(f"file_path must be absolute, got: {path_str}")
        try:
            p.relative_to(repo_root)
        except ValueError as e:
            raise PermissionError(f"file_path escapes repo root: {path_str}") from e
        target = p.resolve(strict=False)

        if not target.exists():
            return f"[ERROR] file not found: {path_str}"
        if not target.is_file():
            return f"[ERROR] not a regular file: {path_str}"
        size = target.stat().st_size
        truncated = size > self.SIZE_LIMIT_BYTES
        with target.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        total_lines = len(all_lines)
        body = all_lines[offset : offset + limit]
        # cat -n format: 1-based line number + TAB + line content
        numbered = [f"{i + 1 + (offset or 0)}\t{line.rstrip()}" for i, line in enumerate(body)]
        content = "\n".join(numbered)
        pieces = {
            "file_path": str(target),
            "content": content,
            "num_lines": len(body),
            "start_line": offset or 0,
            "total_lines": total_lines,
            "encoding": "utf-8",
            "truncated": truncated,
        }
        return _render(pieces)


def _render(d: Dict[str, Any]) -> str:
    parts = [f"=== {d['file_path']} ===",
             f"lines {d['start_line']}..{d['start_line'] + d['num_lines']} "
             f"of {d['total_lines']}  ({d['encoding']})"]
    if d["truncated"]:
        parts.append("[file exceeds 256 KB; use offset/limit to page]")
    parts.append("")
    parts.append(d["content"])
    return "\n".join(parts)