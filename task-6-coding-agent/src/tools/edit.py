"""edit — precise string replacement.

Per blueprint Part I §1.2 (file-system-spec §2):

* ``file_path`` must be absolute, inside the repo.
* ``old_string`` must be **unique** in the file (else fail), unless
  ``replace_all=true``.
* Requires the file to have been read via ``read_file`` in this session
  (shared ``READ_REGISTRY`` from ``base.py``).
* On success, returns the unified diff plus a ``match_count``.
* Auto-creates parent directories? **No.** Edit is for existing files.
"""
from __future__ import annotations

import difflib
import os
import uuid
from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import (
    BaseTool,
    has_been_read_for,
    mark_read_for,
    safe_resolve,
)


class EditTool(BaseTool):
    name: ClassVar[str] = "edit"
    description: ClassVar[str] = (
        "Performs an exact string replacement in a file. The file_path "
        "must be absolute. `old_string` must match **uniquely** unless "
        "`replace_all` is true. Requires that `read_file` was called on "
        "this file earlier in the conversation. Use this for surgical "
        "edits; use `write_file` only when rewriting the whole file."
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
            "old_string": {
                "type": "string",
                "minLength": 1,
                "description": "Exact substring to replace. Must match at least once.",
            },
            "new_string": {
                "type": "string",
                "description": (
                    "Replacement text (must be different from old_string). "
                    "May be empty to delete the matched substring."
                ),
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence of old_string. Default false.",
                "default": False,
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }
    is_read_only: ClassVar[bool] = False

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:
        path_str = args["file_path"]
        old = args["old_string"]
        new = args["new_string"]
        replace_all = bool(args.get("replace_all", False))

        if not isinstance(old, str) or not isinstance(new, str):
            return "[ERROR] old_string and new_string must be strings"
        if not old:
            return "[ERROR] old_string is empty"
        if old == new:
            return "[ERROR] old_string and new_string are identical"

        p = Path(path_str)
        if not p.is_absolute():
            raise PermissionError(f"file_path must be absolute, got: {path_str}")
        try:
            p.relative_to(repo_root)
        except ValueError as e:
            raise PermissionError(f"file_path escapes repo root: {path_str}") from e
        target = p.resolve(strict=False)

        if not target.exists():
            return f"[ERROR] file does not exist: {path_str}"
        if target.is_dir():
            return f"[ERROR] path is a directory, not a file: {path_str}"
        if not has_been_read_for(str(target), self._read_paths):
            return (
                f"[ERROR] file '{path_str}' has not been read yet. "
                f"Call `read_file` first to load its current contents."
            )

        text = target.read_text(encoding="utf-8", errors="replace")
        match_count = text.count(old)

        if match_count == 0:
            return (
                f"[ERROR] old_string not found in {path_str}. "
                f"Re-read the file and copy the exact text."
            )
        if match_count > 1 and not replace_all:
            return (
                f"[ERROR] old_string matched {match_count} times in "
                f"{path_str} — must be unique. Pass replace_all=true to "
                f"replace every occurrence, or supply a longer old_string."
            )

        if replace_all:
            new_text = text.replace(old, new)
        else:
            new_text = text.replace(old, new, 1)

        # Atomic write (tmp + os.replace, same pattern as WriteFileTool).
        tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(new_text, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
        mark_read_for(str(target), self._read_paths)

        diff_text = _make_diff(target=str(target), old=text, new=new_text)
        return _render({
            "file_path": str(target),
            "match_count": text.count(old) if replace_all else 1,
            "bytes_written": len(new_text.encode("utf-8")) - len(text.encode("utf-8")),
            "diff": diff_text,
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
        f"edit applied to {d['file_path']}  "
        f"(match_count={d['match_count']}, bytes_delta={d['bytes_written']:+d})\n"
    )
    return head + "---\n" + d["diff"]