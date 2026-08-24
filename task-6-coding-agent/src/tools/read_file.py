"""read_file — bounded file reading with cat -n line numbers.

Per file-system-spec §1:

* path must be absolute, inside the repo,
* default reads up to ``DEFAULT_LIMIT`` (400) lines from the start,
* ``offset`` / ``limit`` allow paging long files,
* returns ``cat -n`` style content with line numbers,
* max file size 256 KB (before read) — caller pages via offset/limit,
* the rendered output self-limits to ``max_result_chars`` so the
  ``lines X..Y of N`` header is **honest** about what was actually
  returned.  When the requested limit would push the body past
  ``max_result_chars``, the body is shrunk to fit, the header is
  rewritten to reflect the *actual* line range, and an explicit
  ``[output truncated at N chars; call read_file again with offset=K
  to continue]`` marker is added so the model knows to page.

The return shape is documented in blueprint Part I §1.1:

    {
      "file_path": "<abs>",
      "content": "<cat -n body>",
      "num_lines": N,           # actual lines returned (≤ requested limit)
      "start_line": 0,
      "total_lines": M,
      "encoding": "utf-8",
      "truncated": bool         # file size > 256 KB
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import BaseTool, mark_read_for, safe_resolve

# Reserve this many chars for the rendered header so the final string
# (header + body) fits inside ``self.max_result_chars`` and the
# downstream ``BaseTool.__call__`` char cap is a no-op for honest
# read_file output.  Generous on purpose: long absolute paths blow the
# size of the ``=== ... ===`` line.  If you lower this, also lower
# ``max_result_chars`` proportionally.
_HEADER_RESERVE_CHARS = 400


class ReadFileTool(BaseTool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read a UTF-8 file. file_path absolute. Header is 1-based and "
        "shows the *actual* line range returned. If truncated, the "
        "response ends with `call read_file again with offset=K` — call "
        "again with offset=K to continue. Pass `include_line_numbers=true` "
        "for cat -n output. Default clean text (safer to feed back to "
        "write_file). Files > 256 KB are rejected."
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
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Line number to start reading from (0-based).",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Maximum number of lines to read (default 400).",
            },
            "include_line_numbers": {
                "type": "boolean",
                "description": (
                    "When true, prefix every line with its 1-based number (cat -n). "
                    "Default is false — emit clean text, which is safer to feed "
                    "back into write_file."
                ),
                "default": False,
            },
        },
        "required": ["file_path"],
    }
    is_read_only: ClassVar[bool] = True
    max_result_chars: ClassVar[int] = 8_000  # hard cap on tool output (Qwen 7B + 8K ctx)

    SIZE_LIMIT_BYTES = 256 * 1024
    DEFAULT_LIMIT = 400  # lines per call (Qwen 7B + 8K ctx needs small reads)

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
            # Give the model a constructive next step instead of a bare
            # error — a bare "file not found" makes models (especially
            # 7B-class) loop guessing filenames forever.
            return (
                f"[ERROR] file not found: {path_str}\n"
                f"[HINT] if you are guessing a path, use `list_files` "
                f"(or `list_files` with a subdirectory) to discover the "
                f"actual file layout under {repo_root} instead of "
                f"guessing filenames."
            )
        if not target.is_file():
            return f"[ERROR] not a regular file: {path_str}"
        # Record this read so a later write_file/edit is allowed.
        # ``BaseTool`` instances use ``self._read_paths`` (per-instance,
        # parallel-agent-safe).  The module-level ``READ_REGISTRY`` is a
        # separate test-only state and is NOT consulted by this code path.
        mark_read_for(str(target), self._read_paths)
        size = target.stat().st_size
        truncated_size = size > self.SIZE_LIMIT_BYTES
        with target.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        total_lines = len(all_lines)
        # Self-manage the char budget so the rendered ``lines X..Y of N``
        # header is honest about what was actually returned.  We *do not*
        # rely on ``BaseTool.__call__``'s safety-net char cap for the
        # body — that cap silently chops the last few lines and the
        # header still claims ``lines 0..400 of 642``, leaving the model
        # thinking it has data it does not (astropy-14365 bug 2026-08-24:
        # line 309 ``if v == "NO":`` was hidden behind the silent chop).
        body_budget = self.max_result_chars - _HEADER_RESERVE_CHARS
        if body_budget < 256:
            # Defensive: header reserve left no room for content.  Force
            # at least one line of body so the model sees something.
            body_budget = 256
        max_requested = min(limit, max(0, total_lines - offset))
        actual_lines = max_requested
        if actual_lines > 0:
            # Walk from the tail backwards so we keep the first lines of
            # the requested window — those are usually what the model
            # actually needs to anchor on.
            tail_chars = sum(len(line) for line in all_lines[offset:offset + actual_lines])
            while actual_lines > 0 and tail_chars > body_budget:
                actual_lines -= 1
                tail_chars -= len(all_lines[offset + actual_lines])
        body = all_lines[offset:offset + actual_lines]
        # Default: clean text (no line-number prefix). Qwen2.5-Coder
        # tends to echo `cat -n` output back into `write_file` content,
        # which corrupts the file. Pass `include_line_numbers=true` to
        # opt into the spec-described cat -n format.
        if bool(args.get("include_line_numbers", False)):
            numbered = [f"{i + 1 + (offset or 0)}\t{line.rstrip()}" for i, line in enumerate(body)]
            content = "\n".join(numbered)
        else:
            content = "".join(body)
        truncated_chars = actual_lines < max_requested
        next_offset = (offset or 0) + actual_lines
        pieces = {
            "file_path": str(target),
            "content": content,
            "num_lines": len(body),
            # Display 1-based line numbers in the header so they match
            # ``grep -n`` and ``include_line_numbers=true`` output.  The
            # ``offset`` parameter remains 0-based for tool-API
            # stability.
            "start_line_disp": (offset or 0) + 1,
            "end_line_disp": (offset or 0) + actual_lines,
            "total_lines": total_lines,
            "encoding": "utf-8",
            "truncated_size": truncated_size,
            "truncated_chars": truncated_chars,
            "next_offset": next_offset,
            "max_result_chars": self.max_result_chars,
        }
        return _render(pieces)


def _render(d: Dict[str, Any]) -> str:
    parts = [
        f"=== {d['file_path']} ===",
        f"lines {d['start_line_disp']}..{d['end_line_disp']} "
        f"of {d['total_lines']}  ({d['encoding']})",
    ]
    if d["truncated_size"]:
        parts.append("[file exceeds 256 KB; use offset/limit to page]")
    if d["truncated_chars"]:
        parts.append(
            f"[output truncated at {d['max_result_chars']} chars; "
            f"call read_file again with offset={d['next_offset']} "
            f"to continue]"
        )
    parts.append("")
    parts.append(d["content"])
    return "\n".join(parts)