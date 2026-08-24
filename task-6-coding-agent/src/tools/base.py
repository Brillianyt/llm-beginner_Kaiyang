"""Tool base class + safety helpers.

Per blueprint Part I (file-system-spec §0):

* Tools declare ``name`` / ``description`` / ``input_schema`` declaratively.
* ``call(args, repo_root) -> dict`` returns a *structured* result — never
  raises through MCP.
* Path safety is centralised in :func:`safe_resolve`.

Per blueprint Part I (file-system-spec §6) the Bash / git tools use
``subprocess.run(..., args=[...], shell=False, cwd=cwd)`` to prevent
shell injection.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

log = logging.getLogger("coding_agent.tools")


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class BaseTool:
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_schema: ClassVar[Dict[str, Any]] = {}
    is_read_only: ClassVar[bool] = True
    max_result_chars: ClassVar[int] = 50_000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def __call__(self, args: Dict[str, Any], repo_root: Path) -> ToolResult:
        try:
            content = self.call(args, repo_root)
            if len(content) > self.max_result_chars:
                content = content[: self.max_result_chars] + "\n...[truncated]..."
            return ToolResult(content=content)
        except PermissionError as e:
            log.warning("tool=%s denied: %s", self.name, e)
            return ToolResult(content=f"[ERROR] permission denied: {e}", is_error=True)
        except subprocess.TimeoutExpired as e:
            log.warning("tool=%s timeout", self.name)
            return ToolResult(content=f"[ERROR] command timeout after {e.timeout}s", is_error=True)
        except Exception as e:  # noqa: BLE001
            log.exception("tool=%s crashed: %s", self.name, e)
            return ToolResult(content=f"[ERROR] tool crashed: {e}", is_error=True)

    def call(self, args: Dict[str, Any], repo_root: Path) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Path safety (single choke-point)
# ---------------------------------------------------------------------------

def safe_resolve(path: str | os.PathLike, repo_root: Path) -> Path:
    """Resolve ``path`` against ``repo_root`` and reject escapes.

    Rules (file-system-spec §1):
      * absolute paths are rejected,
      * ``..`` traversal is rejected via ``is_relative_to(repo_root)``.

    Raises :class:`PermissionError` on failure — the MCP layer converts it
    to a structured error response.
    """
    p = Path(path)
    if p.is_absolute():
        raise PermissionError(f"absolute path rejected: {path}")
    candidate = (repo_root / p).resolve(strict=False)
    try:
        candidate.relative_to(repo_root)
    except ValueError as e:
        raise PermissionError(f"path escapes repo root: {path}") from e
    return candidate


# ---------------------------------------------------------------------------
# Subprocess safety
# ---------------------------------------------------------------------------

# Git subcommands / flags that could destroy uncommitted work.
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
    args: list,
    cwd: Path,
    *,
    timeout: int = 60,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run ``args`` (list form) inside ``cwd`` with a hard timeout.

    * Never ``shell=True``.
    * Captures stdout/stderr.
    * Strips the parent env to PATH + HOME + ``extra_env``.
    """
    if not args:
        raise ValueError("empty command")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        args,
        cwd=str(cwd),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def check_blocked_git(args: list) -> Optional[str]:
    """Return a reason string if ``args`` matches a blocked git pattern."""
    blob = " ".join(args)
    for fragment in BLOCKED_GIT_FRAGMENTS:
        if fragment in blob:
            return f"git command contains blocked fragment: '{fragment}'"
    return None