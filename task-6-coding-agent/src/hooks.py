"""Hook system — borrowed from Claude Code's ``PreToolUse`` / ``PostToolUse``.

Only the MVP is implemented: an in-process registry of callbacks that can
``allow`` / ``deny`` (and theoretically ``modify``) tool invocations.

Why bother with hooks at all if we already have tool-level safety?

* **Single point of audit** — every tool call (including ones we forgot
  to harden) goes through one path, so logs are uniform.
* **Per-tenant customisation** — the eval harness could register a hook
  that bans writing to ``test_*.py``, or a hook that auto-formats after
  ``write_file``.
* **Cheap to test** — each hook is a plain function, easy to unit-test.

``register_pre_tool_use`` / ``register_post_tool_use`` are the only public
verbs. Hooks must never raise.
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

    # -- registration ------------------------------------------------------

    def register_pre(self, fn: PreHook) -> None:
        self._pre.append(fn)

    def register_post(self, fn: PostHook) -> None:
        self._post.append(fn)

    # -- firing ------------------------------------------------------------

    def fire_pre(self, tool_name: str, args: Dict[str, Any]) -> tuple[HookDecision, Dict[str, Any]]:
        """Run all PreToolUse hooks in registration order.

        Returns the final decision + (possibly modified) args. The first
        ``DENY`` short-circuits the chain.
        """
        for hook in self._pre:
            try:
                decision = hook(tool_name, args)
            except Exception as e:  # noqa: BLE001
                log.warning("pre-hook %s raised: %s", getattr(hook, "__name__", hook), e)
                continue
            if decision == HookDecision.DENY:
                return HookDecision.DENY, args
            if decision == HookDecision.MODIFY:
                # Hooks in MVP can mutate ``args`` in place.
                continue
        return HookDecision.ALLOW, args

    def fire_post(self, tool_name: str, args: Dict[str, Any], observation: str) -> str:
        for hook in self._post:
            try:
                hook(tool_name, args, observation)
            except Exception as e:  # noqa: BLE001
                log.warning("post-hook %s raised: %s", getattr(hook, "__name__", hook), e)
        return observation


# ---------------------------------------------------------------------------
# Built-in defaults — applied to every CodingAgent unless the user
# overrides them.
# ---------------------------------------------------------------------------

def default_pre_hooks() -> List[PreHook]:
    """Return a fresh list of safety pre-hooks."""

    def _block_test_writes(tool: str, args: Dict[str, Any]) -> HookDecision:
        if tool in ("write_file",):
            path = (args.get("path") or "").lower()
            # Eval tasks explicitly forbid editing tests/. Surface a clear
            # error so the agent retries with a non-test path.
            if "test_" in path or "/tests/" in path or path.endswith("_test.py"):
                return HookDecision.DENY
        return HookDecision.ALLOW

    return [_block_test_writes]
