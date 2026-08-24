"""Hook system — borrowed from Claude Code's PreToolUse / PostToolUse.

We expose a tiny registry of callbacks that can ``allow`` / ``deny`` tool
invocations and post-process observations. Hooks must never raise.

The only built-in default is the **no-test-write** hook: the eval task
explicitly forbids editing ``test_*.py``, so we deny such writes early.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, List

log = logging.getLogger("hooks")


class HookDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"


PreHook = Callable[[str, Dict[str, Any]], HookDecision]
PostHook = Callable[[str, Dict[str, Any], str], None]


class HookSystem:
    def __init__(self) -> None:
        self._pre: List[PreHook] = []
        self._post: List[PostHook] = []

    def register_pre(self, fn: PreHook) -> None:
        self._pre.append(fn)

    def register_post(self, fn: PostHook) -> None:
        self._post.append(fn)

    def fire_pre(self, tool_name: str, args: Dict[str, Any]) -> tuple:
        for hook in self._pre:
            try:
                decision = hook(tool_name, args)
            except Exception as e:  # noqa: BLE001
                log.warning("pre-hook raised: %s", e)
                continue
            if decision == HookDecision.DENY:
                return HookDecision.DENY, args
        return HookDecision.ALLOW, args

    def fire_post(self, tool_name: str, args: Dict[str, Any], observation: str) -> str:
        for hook in self._post:
            try:
                hook(tool_name, args, observation)
            except Exception as e:  # noqa: BLE001
                log.warning("post-hook raised: %s", e)
        return observation


def default_pre_hooks() -> List[PreHook]:
    """Built-in safety hooks applied to every CodingAgent by default."""

    def _block_test_writes(tool: str, args: Dict[str, Any]) -> HookDecision:
        if tool == "write_file":
            path = (args.get("file_path") or args.get("path") or "").lower()
            if "test_" in path or "/tests/" in path or path.endswith("_test.py"):
                return HookDecision.DENY
        return HookDecision.ALLOW

    return [_block_test_writes]


def default_post_hooks(log_path: str | Path | None = None) -> List[PostHook]:
    """Built-in audit hooks applied to every CodingAgent by default.

    * ``audit_logger`` — appends every tool call (name, args, truncated
      observation) to a JSONL audit log so operators can replay what
      the agent did after the fact. Defaults to
      ``<cwd>/.coding-agent-audit.jsonl`` unless ``log_path`` is given.

    Audit logs are *append-only*; the file is opened once at construction
    so a long-running agent doesn't keep reopening. If the log directory
    doesn't exist, this is a no-op (audit logging is best-effort).
    """
    import json
    from datetime import datetime
    from pathlib import Path

    p: Path | None = None
    fh = None
    if log_path is not None:
        p = Path(log_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = p.open("a", encoding="utf-8")
        except OSError as e:
            log.warning("could not open audit log %s: %s", p, e)
            p = None
            fh = None

    def audit_logger(tool: str, args: Dict[str, Any], obs: str) -> None:
        if fh is None:
            return
        try:
            # Truncate observation to keep the audit log bounded.
            entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "tool": tool,
                "args": {k: v for k, v in args.items() if k != "content"},
                "content_len": len(args.get("content", "") or ""),
                "observation_excerpt": obs[:400],
                "observation_len": len(obs),
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
        except OSError as e:
            log.warning("audit write failed: %s", e)

    return [audit_logger]