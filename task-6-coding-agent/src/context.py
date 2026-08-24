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
    """Return (possibly compacted messages, did_compact).

    If we slice the middle out, the system message is still at index 0
    but its position relative to the new content has changed. We
    re-apply ``cache_control={"type": "ephemeral"}`` so the OpenAI/SGLang
    prompt cache can re-key on the new prefix. Without this re-mark the
    cache would be invalidated by every compaction.
    """
    if not messages:
        return messages, False
    if estimate_chars(messages) < threshold:
        # Even on the no-compaction path, defensively re-apply the cache
        # marker in case the caller mutated the system message.
        _ensure_cache_marker(messages)
        return messages, False
    head = messages[: max(1, keep_head)]
    tail = messages[-keep_tail:] if len(messages) > keep_head + keep_tail else []
    if not tail:
        _ensure_cache_marker(messages)
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
    _ensure_cache_marker(compacted)
    log.info(
        "compaction: %d → %d msgs (chars %d → %d)",
        len(messages),
        len(compacted),
        estimate_chars(messages),
        estimate_chars(compacted),
    )
    return compacted, True


def _ensure_cache_marker(messages: List[Dict[str, Any]]) -> None:
    """Re-apply the prompt-cache marker on the first system message.

    Mutates ``messages`` in place. Backends that don't honour
    ``cache_control`` (Ollama, llama.cpp) ignore this field; we don't
    gate on the backend to keep the call site trivial.
    """
    if not messages:
        return
    head = messages[0]
    if head.get("role") != "system":
        return
    if head.get("cache_control") == {"type": "ephemeral"}:
        return
    messages[0] = {**head, "cache_control": {"type": "ephemeral"}}