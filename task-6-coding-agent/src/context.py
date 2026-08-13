"""Context management helpers — compaction, token estimation.

The full Claude Code implementation (``src/services/compact/compact.ts``)
spawns a sub-LLM call to *summarise* the history. We deliberately do not:
the Qwen2.5-Coder-7B context is 32K and our toy-repo fix rarely needs
more than ~4K tokens. A simple sliding-window compaction is enough.

Trigger: when the cumulative character count exceeds
``COMPACT_THRESHOLD_CHARS`` we collapse the middle of the history into
a single ``[compacted]`` system message and keep the latest few tool
turns verbatim.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

log = logging.getLogger("context")

# Qwen2.5-Coder-7B has a 32K-token context. We use ~25K characters as a
# rough proxy (≈ 6 chars / token). Tweakable via env var if needed.
COMPACT_THRESHOLD_CHARS = 25_000
KEEP_HEAD = 3           # initial system + first 2 user/assistant
KEEP_TAIL = 5           # last few tool results


def estimate_chars(messages: Iterable[Dict[str, Any]]) -> int:
    """Cheap character total — used as a token proxy."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        # tool_calls arguments also cost tokens
        for tc in m.get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments")
            if isinstance(args, str):
                total += len(args)
            elif isinstance(args, dict):
                total += len(str(args))
    return total


def maybe_compact(
    messages: List[Dict[str, Any]],
    *,
    threshold: int = COMPACT_THRESHOLD_CHARS,
    keep_head: int = KEEP_HEAD,
    keep_tail: int = KEEP_TAIL,
) -> tuple[List[Dict[str, Any]], bool]:
    """Return (possibly compacted messages, did_compact)."""
    if not messages:
        return messages, False
    if estimate_chars(messages) < threshold:
        return messages, False

    head = messages[: max(1, keep_head)]
    tail = messages[-keep_tail:] if len(messages) > keep_head + keep_tail else []
    if not tail:
        return messages, False
    compacted = (
        head
        + [
            {
                "role": "system",
                "content": (
                    "[compacted] history truncated to fit context window — "
                    "see trace log for full record."
                ),
            }
        ]
        + tail
    )
    log.info(
        "compaction: %d → %d messages (chars %d → %d)",
        len(messages),
        len(compacted),
        estimate_chars(messages),
        estimate_chars(compacted),
    )
    return compacted, True
