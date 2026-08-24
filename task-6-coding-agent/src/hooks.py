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