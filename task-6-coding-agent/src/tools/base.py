"""Tool base class + safety helpers shared by all tools.

Design notes — borrowed from Claude Code ``src/Tool.ts``:

* ``buildTool({...})`` factory pattern: every tool declares its metadata
  (name, description, schema) declaratively in one place.
* ``is_read_only`` mirrors ``isReadOnly()`` so the orchestrator can decide
  whether tools may run in parallel.
* ``call(args, repo_root) -> str`` always returns a string — never raises,
  so an MCP tool crash cannot take the whole server down.
* Path safety: ``safe_resolve`` is the single choke-point for file ops.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

log = logging.getLogger("coding_agent.tools")


@dataclass
class ToolResult:
    """Uniform return shape — keep simple so MCP clients can serialise it."""

    content: str
    is_error: bool = False


class BaseTool:
    """Common scaffolding for MCP tools.

    Subclasses override the class-level metadata and implement ``call``.
    ``__call__`` wraps exceptions into :class:`ToolResult` so that the MCP
    server can keep running even when a tool misbehaves.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_schema: ClassVar[Dict[str, Any]] = {}
    is_read_only: ClassVar[bool] = True
    max_result_chars: ClassVar[int] = 50_000

    # -- schema helpers ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """MCP-friendly representation (used by ``mcp_server.list_tools``)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    # -- invocation -------------------------------------------------------
    def __call__(self, args: Dict[str, Any], repo_root: Path) -> ToolResult:
        try:
            content = self.call(args, repo_root)
            if len(content) > self.max_result_chars:
                # Prevent one tool from blowing the model's context window.
                content = content[: self.max_result_chars] + "\n...[truncated]..."
            return ToolResult(content=content)
        except PermissionError as e:
            # Sandboxing refusals are surfaced cleanly, not as crashes.
            log.warning("tool=%s denied: %s", self.name, e)
            return ToolResult(content=f"[ERROR] permission denied: {e}", is_error=True)
        except subprocess.TimeoutExpired as e:
            log.warning("tool=%s timeout", self.name)
            return ToolResult(content=f"[ERROR] command timeout after {e.timeout}s", is_error=True)
        except Exception as e:  # noqa: BLE001 - we genuinely catch everything
            log.exception("tool=%s crashed: %s", self.name, e)
            return ToolResult(content=f"[ERROR] tool crashed: {e}", is_error=True)

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def safe_resolve(path: str | os.PathLike, repo_root: Path) -> Path:
    """Resolve ``path`` against ``repo_root`` and reject escapes.

    Rules (per ``SYNTHESIS.md §2.1``):
      * If ``path`` is absolute → reject.
      * Resolve relative to ``repo_root`` (follows symlinks).
      * Final path **must** be ``is_relative_to(repo_root)``.

    A failing check raises :class:`PermissionError`, which the MCP layer
    converts to a structured error response.
    """
    p = Path(path)
    if p.is_absolute():
        raise PermissionError(f"absolute path rejected: {path}")
    candidate = (repo_root / p).resolve(strict=False)
    # Common normalisation step — strict=False handles non-existent files
    try:
        candidate.relative_to(repo_root)
    except ValueError as e:
        raise PermissionError(f"path escapes repo root: {path}") from e
    return candidate


# Git subcommands / flags that could destroy uncommitted work.
# Checked at the *tool* layer so even an MCP client speaking raw stdio
# cannot bypass them.
BLOCKED_GIT_FRAGMENTS = (
    "reset --hard",
    "reset --merge",
    "clean -fd",
    "clean -ffdx",
    "checkout --",
    "checkout .",
    "branch -D ",
    "branch --delete --force",
    "push --force",
    "push -f",
)


def run_subprocess(
    args: list[str],
    cwd: Path,
    *,
    timeout: int = 60,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run ``args`` (list form) inside ``cwd`` with a hard timeout.

    * Never ``shell=True`` (per ``SYNTHESIS.md §6.1``).
    * Captures stdout/stderr.
    * Replaces the parent env with a minimal copy + ``extra_env`` to
      avoid leaking secrets into the subprocess.
    """
    if not args:
        raise ValueError("empty command")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        args,
        cwd=str(cwd),
        shell=False,           # critical — prevents shell injection
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def check_blocked_git(args: list[str]) -> Optional[str]:
    """Return a reason string if ``args`` matches a blocked git pattern."""
    blob = " ".join(args)
    for fragment in BLOCKED_GIT_FRAGMENTS:
        if fragment in blob:
            return f"git command contains blocked fragment: '{fragment}'"
    return None


# Useful for logging from inside MCP server; the *server* layer also
# calls ``sys.stderr`` directly for handshake messages.
def log_tool(name: str, args: Dict[str, Any]) -> None:
    log.info("tool=%s args=%s", name, json.dumps(args, ensure_ascii=False, default=str)[:200])
