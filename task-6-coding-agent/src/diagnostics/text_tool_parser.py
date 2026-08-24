"""Offline / diagnostics-only tool-call text parser.

Moved out of ``src/agent.py`` under the A-2 hard-prohibit architecture
(2026-08-24).  The agent's main loop **must not** import this module — the
vLLM ``qwen_coder_json`` parser plugin (``src/vllm_plugin/qwen_coder_tool_parser.py``)
is the sole source of OpenAI ``message.tool_calls``.

This module exists for three reasons that don't compromise the architectural
invariant:

1. **Offline replay / trace analysis** — open a recorded trace JSON and
   ask "did the model emit a tool call in text even though the server
   parser missed it?"  That's a debugging tool, not a runtime path.
2. **Standalone unit testing** — running ``python3
   src/diagnostics/text_tool_parser.py`` directly exercises the regex on
   canned samples, no MCP / vLLM / agent fixture required.
3. **Migration cleanup** — historical artefact from the SGLang era when
   the agent relied on this regex parser.  Kept as-is for diagnosis, not
   for production use.

If you find yourself reaching for ``from src.diagnostics.text_tool_parser
import ...`` inside ``src/agent.py`` — **stop**.  Fix the vLLM parser
plugin instead.  That's the whole point of moving it out of agent.py:

> The chat-completions ``tool_calls`` field is the canonical source of
> truth.  Anything that touches ``message.content`` to recover tool calls
> is silently broken by design — it will mask upstream parser bugs.

Static enforcement lives in
``test_smoke.py::test_agent_never_introspects_text_for_tool_calls``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# Hand-rolled regex extractor (legacy, line-by-line JSON fallback after the
# regex pass).  Tolerates single-level nested braces inside ``arguments``;
# deeper nesting breaks — see ``test_extract_handles_one_level_only``.
#
# This is the OLD, broken pattern. The new system uses
# ``src.vllm_plugin.qwen_coder_tool_parser.parse_text`` server-side.
_JSON_TOOL_RE = re.compile(
    r"\{\s*\"name\"\s*:\s*\"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\"\s*,\s*"
    r"\"arguments\"\s*:\s*(?P<args>\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}",
    re.DOTALL,
)


def parse_text_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract ``{"name": ..., "arguments": ...}`` blocks from text.

    Returns OpenAI-shaped tool_calls dicts, or ``None`` if no parseable
    tool call is found.  Handles both fenced (`` ```json ... ``` ``)
    and raw JSON output.

    NOTE: offline / diagnostics only.  Do NOT call from
    ``src/agent.py``.  See module docstring for the invariant.
    """
    if not text:
        return None
    out: List[Dict[str, Any]] = []
    seen_signatures: set = set()

    def _emit(name: str, args: dict) -> None:
        sig = (name, json.dumps(args, sort_keys=True, default=str))
        if sig in seen_signatures:
            return
        seen_signatures.add(sig)
        out.append({"name": name, "arguments": args})

    # Regex pass — first come, first served.
    for m in _JSON_TOOL_RE.finditer(text):
        try:
            args = json.loads(m.group("args"))
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        _emit(m.group("name"), args)
    # Line-by-line bare-JSON fallback.  Dedup against the regex pass so
    # a JSON line that both paths captured isn't emitted twice.
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        if "\"name\"" not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
            continue
        if not isinstance(obj["arguments"], dict):
            continue
        _emit(obj["name"], obj["arguments"])
    return out or None


# Backward-compatibility alias.  Old name ``_parse_text_tool_calls``
# (private-with-leading-underscore).  Preserves existing imports from
# offline scripts / debug notebooks / academic pull-request leftovers.
_parse_text_tool_calls = parse_text_tool_calls


_CANONICAL_SAMPLES = [
    # hermes flat-JSON
    ('{"name": "list_files", "arguments": {"path": "."}}',
     [{"name": "list_files", "arguments": {"path": "."}}]),
    # bare JSON in ```json fence
    ('```json\n{"name": "run_tests", "arguments": {"cmd": "pytest"}}\n```',
     [{"name": "run_tests", "arguments": {"cmd": "pytest"}}]),
    # No tool call -> None (don't false-positive on prose)
    ("好的，让我先看看 calculator.py 的内容再决定怎么改。", None),
]


if __name__ == "__main__":  # pragma: no cover
    import sys

    print("=" * 64)
    print("Offline text-tool-call parser (diagnostic only)")
    print("DO NOT use this from src/agent.py — see module docstring.")
    print("=" * 64)
    failures = 0
    for i, (sample, expected) in enumerate(_CANONICAL_SAMPLES, 1):
        got = parse_text_tool_calls(sample)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"\n[{i}] {status}\n  in  : {sample!r}")
        print(f"  want: {expected}\n  got : {got}")
        if not ok:
            failures += 1
    print()
    print("=" * 64)
    print(f"  {len(_CANONICAL_SAMPLES) - failures}/{len(_CANONICAL_SAMPLES)} passed")
    print("=" * 64)
    sys.exit(0 if failures == 0 else 1)
