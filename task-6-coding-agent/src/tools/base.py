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

    def __init__(self) -> None:
        # Per-instance read-before-write registry. Multiple CodingAgent
        # instances (e.g. ablations S2 runs 3 in a row) get isolated
        # state — instance 2 can't see instance 1's reads. See P4.
        self._read_paths: set = set()

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


# ---------------------------------------------------------------------------
# Read-before-write registry.
# ---------------------------------------------------------------------------
#
# We keep a module-level fallback for callers that haven't been wired up
# to a per-instance registry (eval harness, ablations, tests). When
# ``write_file`` / ``edit`` is invoked through a ``BaseTool`` instance,
# they consult ``self._read_paths`` so two parallel CodingAgent
# instances don't bleed reads into each other.

# Module-level fallback (single-process, single-agent).
READ_REGISTRY: set = set()


def mark_read(abs_path: str) -> None:
    READ_REGISTRY.add(abs_path)


def has_been_read(abs_path: str) -> bool:
    return abs_path in READ_REGISTRY


def clear_read_registry() -> None:
    READ_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Per-instance wiring
# ---------------------------------------------------------------------------

# When a tool is constructed inside a ``CodingAgent`` instance, the agent
# patches its ``_read_paths`` attribute so the read-before-write guard
# is per-agent. This module-level holder just records the latest agent
# scope (used by subagents and the eval harness via ``clear_read_registry``).
_CURRENT_AGENT_READS: set | None = None


def set_agent_reads(paths: set | None) -> None:
    """Bind the module-level helpers to a specific CodingAgent's
    per-instance read registry. Pass ``None`` to revert to the shared
    module-level registry."""
    global _CURRENT_AGENT_READS
    _CURRENT_AGENT_READS = paths


def mark_read_for(abs_path: str, reads: set | None) -> None:
    if reads is not None:
        reads.add(abs_path)
    else:
        mark_read(abs_path)


def has_been_read_for(abs_path: str, reads: set | None) -> bool:
    if reads is not None:
        return abs_path in reads
    return has_been_read(abs_path)