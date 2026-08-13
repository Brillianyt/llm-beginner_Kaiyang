"""Atomic tool implementations for the MCP server.

Each tool:
* exposes ``name`` / ``description`` / ``input_schema``
* enforces a hardened sandbox (path resolution + subprocess list form)
* returns a plain string observation (never raises through MCP)

The tools are **stateless** — the agent / MCP client owns the message log.
"""
from .base import BaseTool, ToolResult, safe_resolve, run_subprocess, BLOCKED_GIT_FRAGMENTS
from .read_file import ReadFileTool
from .write_file import WriteFileTool
from .run_tests import RunTestsTool
from .git_diff import GitDiffTool
from .git_apply import GitApplyTool
from .list_files import ListFilesTool

# Order is preserved in ``list_tools()`` for stable schema ids.
# We store *instances* because the MCP server re-uses the same objects
# across calls. The classes themselves remain importable for tests.
def _make_instances():
    return [
        ReadFileTool(),
        WriteFileTool(),
        RunTestsTool(),
        GitDiffTool(),
        GitApplyTool(),
        ListFilesTool(),
    ]

ALL_TOOLS: list = _make_instances()

__all__ = [
    "BaseTool",
    "ToolResult",
    "safe_resolve",
    "run_subprocess",
    "BLOCKED_GIT_FRAGMENTS",
    "ReadFileTool",
    "WriteFileTool",
    "RunTestsTool",
    "GitDiffTool",
    "GitApplyTool",
    "ListFilesTool",
    "ALL_TOOLS",
]
