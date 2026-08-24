"""write_file — atomic file write with read-first guard.

Per file-system-spec §3:

* ``file_path`` must be absolute, not a directory,
* must have been read via ``read_file`` in the same session,
* atomic write via ``tmp + os.replace``,
* returns the unified diff for the change.

Blueprint Part I §1.2 return:

    {
      "file_path": "<abs>",
      "type": "create" | "update",
      "bytes_written": N,
      "diff": "<unified diff string>",
      "git_diff": {...} | None
    }
"""
from __future__ import annotations

import difflib
import os
import uuid
from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import (
    BaseTool,
    has_been_read,
    has_been_read_for,
    mark_read,
    mark_read_for,
    safe_resolve,
)


class WriteFileTool(BaseTool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Writes a UTF-8 file. The file_path must be absolute and inside the "
        "repo. Overwrites the file (creates if missing). Returns a unified "
        "diff of the change. **Read the file first** via `read_file` in the "
        "same session — the tool will reject writes to unread files."
    )
    input_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "pattern": "^/",
                "description": (
                    "Absolute path to the file (must start with '/', "
                    "and stay inside the repo root)."
                ),
            },
            "content": {
                "type": "string",
                "description": "Full new file contents (UTF-8).",
            },
        },
        "required": ["file_path", "content"],
    }
    is_read_only: ClassVar[bool] = False

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        path_str = args["file_path"]
        content = args.get("content", "")

        # Validate path is absolute and inside the repo.
        p = Path(path_str)
        if not p.is_absolute():
            raise PermissionError(f"file_path must be absolute, got: {path_str}")
        try:
            p.relative_to(repo_root)
        except ValueError as e:
            raise PermissionError(f"file_path escapes repo root: {path_str}") from e
        target = p.resolve(strict=False)

        # Reject directory paths.
        if target.exists() and target.is_dir():
            return f"[ERROR] path is a directory, not a file: {path_str}"

        # Read-first guard — consult the tool's own per-instance
        # registry first (if any), then fall back to the module-level
        # shared registry. Two parallel agents get isolated state.
        if target.exists() and not has_been_read_for(str(target), self._read_paths):
            return (
                f"[ERROR] file '{path_str}' has not been read yet. "
                f"Call `read_file` first to load its current contents."
            )

        # Capture old contents for the diff.
        old_text = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        is_create = not target.exists()

        # Atomic write.
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(content, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
        # Mark as read so subsequent writes succeed without re-reading.
        mark_read_for(str(target), self._read_paths)

        diff_text = _make_diff(
            target=str(target),
            old=old_text,
            new=content,
        )
        return _render({
            "file_path": str(target),
            "type": "create" if is_create else "update",
            "bytes_written": len(content.encode("utf-8")),
            "diff": diff_text,
            "git_diff": None,
        })


def _make_diff(*, target: str, old: str, new: str) -> str:
    if old == new:
        return "(no changes)"
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{target}",
        tofile=f"b/{target}",
    )
    return "".join(diff)


def _render(d: Dict[str, Any]) -> str:
    head = (
        f"{d['type']}  {d['file_path']}\n"
        f"bytes_written={d['bytes_written']}\n"
    )
    return head + "---\n" + d["diff"]