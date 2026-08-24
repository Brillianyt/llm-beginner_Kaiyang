"""Context management — simple sliding-window compaction.

When the cumulative character count of ``messages`` exceeds
``COMPACT_THRESHOLD_CHARS`` we collapse the middle of the history into
a single ``[compacted]`` system note and keep the latest few tool turns
verbatim. The full record is always preserved in the ``Trace`` for replay.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

log = logging.getLogger("context")

COMPACT_THRESHOLD_CHARS = 25_000
KEEP_HEAD = 3            # initial system + first user/assistant
KEEP_TAIL = 5            # latest few turns


def estimate_chars(messages: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
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
) -> Tuple[List[Dict[str, Any]], bool]:
    """Return (possibly compacted messages, did_compact)."""
    if not messages:
        return messages, False
    if estimate_chars(messages) < threshold:
        return messages, False
    head = messages[: max(1, keep_head)]
    tail = messages[-keep_tail:] if len(messages) > keep_head + keep_tail else []
    if not tail:
        return messages, False
    compacted = head + [
        {
            "role": "system",
            "content": (
                "[compacted] earlier history collapsed to fit the context window; "
                "see the trace log for the full record."
            ),
        }
    ] + tail
    log.info(
        "compaction: %d → %d msgs (chars %d → %d)",
        len(messages),
        len(compacted),
        estimate_chars(messages),
        estimate_chars(compacted),
    )
    return compacted, True