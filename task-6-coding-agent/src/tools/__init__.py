"""Atomic tool implementations for the MCP server.

Per blueprint Part I, the registered tools are:
  * read_file    — bounded file read (cat -n format, offset/limit paging)
  * write_file   — atomic full overwrite with read-first guard
  * run_tests    — structured pytest runner
  * git_diff     — per-file unified diffs
  * git_apply    — apply / dry-run a unified diff
  * run_bash     — sandboxed subprocess runner (shlex + path containment
                   + deny-list for destructive commands)

All tools are stateless, raise through ``BaseTool.__call__`` (never
through MCP), and use :func:`safe_resolve` for path safety.
"""
from .base import (
    BaseTool,
    ToolResult,
    safe_resolve,
    run_subprocess,
    check_blocked_git,
    BLOCKED_GIT_FRAGMENTS,
)
from .read_file import ReadFileTool
from .write_file import WriteFileTool
from .run_tests import RunTestsTool
from .git_diff import GitDiffTool
from .git_apply import GitApplyTool
from .edit import EditTool
from .list_files import ListFilesTool
from .grep import GrepTool
from .run_bash import RunBashTool


def make_tool_set() -> list[BaseTool]:
    """Return a fresh, isolated set of tool instances.

    Use this when you need per-agent tool state (e.g. isolated
    read-before-write registries). The module-level ``ALL_TOOLS`` is
    kept for the eval harness and the in-process call path; multi-agent
    harnesses should call this factory instead of sharing ``ALL_TOOLS``.
    """
    return [
        ReadFileTool(),
        WriteFileTool(),
        EditTool(),
        ListFilesTool(),
        GrepTool(),
        RunTestsTool(),
        GitDiffTool(),
        GitApplyTool(),
        RunBashTool(),
    ]


ALL_TOOLS = make_tool_set()

__all__ = [
    "BaseTool",
    "ToolResult",
    "safe_resolve",
    "run_subprocess",
    "check_blocked_git",
    "BLOCKED_GIT_FRAGMENTS",
    "ReadFileTool",
    "WriteFileTool",
    "EditTool",
    "ListFilesTool",
    "GrepTool",
    "RunTestsTool",
    "GitDiffTool",
    "GitApplyTool",
    "RunBashTool",
    "ALL_TOOLS",
    "make_tool_set",
]