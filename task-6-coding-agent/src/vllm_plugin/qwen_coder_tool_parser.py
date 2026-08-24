"""Minimal vLLM tool-call parser for Qwen2.5-Coder-7B-Instruct.

A vLLM ``ToolParser`` that takes ``message.content`` from the model and
returns OpenAI-shaped ``tool_calls``. **One regex, one path.**

Why minimal
-----------
Earlier versions of this file had a 5-shape cascade (bare JSON /
``<tool_call>`` / ``<response>`` / ``<function_call>`` /
``<function=name>...``) plus an inner ``_try_load_json`` that fell back to
XML-tag-split extraction.  Each of those was a "first shape that works
wins" fallback — exactly the lazy pattern we are deliberately removing
from the codebase.

This rewrite acknowledges that the JSON payload inside any wrapper shape
is structurally identical: ``{"name": ..., "arguments": {...}}``. The
XML wrappers (``<tool_call>`` / ``<response>`` / code fences) are
**transparent** — we strip fences, then look directly for the JSON.

What's not handled
------------------
The XML-tag-split form

    ``<tool_call><name>X</name><arguments>{...}</arguments></tool_call>``

is **not** supported.  This form puts ``name`` and ``arguments`` in
separate XML tags rather than a single JSON object, so the regex cannot
find a canonical ``{"name": ..., "arguments": ...}`` pair.  We tested
this on the gate-real model outputs (8 prompts, see
``/tmp/gate/gate_result.json``) and confirmed:

* 7 / 8 turns have a JSON-shaped payload that the minimal regex extracts.
* 1 / 8 turns (prompt [6]) uses XML-tag-split and is *intentionally* not
  parsed.  The agent sees ``tool_calls=[]`` and treats it as text.

This is documented as a deliberate trade-off: **12% miss rate** on the
edge case in exchange for a parser whose correctness can be audited in
one screen.  Users hitting this form can update the chat template or
system prompt to instruct the model to emit a single JSON object
(``output exactly one JSON object`` already in
``coder_chat_template.jinja`` line 13).

Usage (vLLM 0.23+)
------------------
Start the server with::

    python -m vllm.entrypoints.openai.api_server \\
        --model models/Qwen2.5-Coder-7B-Instruct \\
        --enable-auto-tool-choice \\
        --tool-call-parser qwen_coder_json \\
        --tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py \\
        ...

Loaded via ``ToolParserManager.import_tool_parser``.  Self-registers as
``qwen_coder_json`` at module import.

Standalone smoke test
---------------------
::

    python src/vllm_plugin/qwen_coder_tool_parser.py
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, List, Optional, Sequence

try:
    import regex as re  # type: ignore
except ImportError:  # pragma: no cover - regex ships with vLLM
    import re  # type: ignore

# ---------------------------------------------------------------------------
# vLLM imports — guarded so this module is also importable standalone.
# ---------------------------------------------------------------------------
try:
    from vllm.entrypoints.chat_utils import make_tool_call_id
    from vllm.entrypoints.openai.engine.protocol import (
        ExtractedToolCallInformation,
        FunctionCall,
        ToolCall,
    )
    from vllm.logger import init_logger
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import (
        Tool,
        ToolParser,
        ToolParserManager,
    )
    _VLLM_AVAILABLE = True
except Exception:  # pragma: no cover
    _VLLM_AVAILABLE = False
    ToolParser = object  # type: ignore[misc,assignment]
    ToolParserManager = None  # type: ignore[assignment]
    TokenizerLike = Any  # type: ignore[misc,assignment]
    Tool = Any  # type: ignore[misc,assignment]
    ExtractedToolCallInformation = Any  # type: ignore[misc,assignment]
    FunctionCall = Any  # type: ignore[misc,assignment]
    ToolCall = Any  # type: ignore[misc,assignment]
    make_tool_call_id = lambda: f"call_{id(object())}"  # type: ignore[assignment]

    def init_logger(name):  # type: ignore[no-redef]
        import logging
        return logging.getLogger(name)


logger = init_logger("qwen_coder_tool_parser")


# ---------------------------------------------------------------------------
# One regex, one path.
# ---------------------------------------------------------------------------

# Markdown code fence opener/closer — stripped before regex matching.  This
# is **not** a fallback; it is a single deterministic pre-processing step.
_RE_FENCE = re.compile(r"```(?:json|xml)?\s*\n?(.*?)```", re.DOTALL)

# Canonical tool-call payload: ``{"name": "X", "arguments": {...}}``.
# The pattern tolerates nested objects up to two levels deep; deeper
# nesting breaks it (rare in real Coder outputs we tested).
_RE_TOOL_CALL = re.compile(
    r"\{\s*\"name\"\s*:\s*\"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\"\s*,"
    r"\s*\"arguments\"\s*:\s*"
    r"(?P<args>\{(?:[^{}]|\{[^{}]*\})*\})"
    r"\s*\}",
    re.DOTALL,
)


def _strip_fences(text: str) -> str:
    """Deterministic pre-processing: remove `````json\n...\n``` `` /
    `````xml\n...\n``` `` wrappers, if present.  No fallback behaviour
    beyond the obvious single regex substitution.
    """
    return _RE_FENCE.sub(lambda m: m.group(1), text)


def _validate_call(name: str, arguments: Any) -> bool:
    """Strict check: ``name`` is a non-empty string, ``arguments`` is a
    dict.  Anything else is dropped (not rescued via field-name
    guessing).
    """
    return (
        isinstance(name, str)
        and bool(name)
        and isinstance(arguments, dict)
    )


# ---------------------------------------------------------------------------
# Public API (works without vLLM)
# ---------------------------------------------------------------------------

def parse_text(model_output: str) -> List[dict]:
    """Return ``[{"name": ..., "arguments": ...}, ...]``.

    Single path: strip fences, run one regex, validate strictly.  No
    shape cascade, no JSON-then-XML fallback, no field-name guessing.
    """
    if not model_output:
        return []
    haystack = _strip_fences(model_output)
    out: List[dict] = []
    for m in _RE_TOOL_CALL.finditer(haystack):
        try:
            args = json.loads(m.group("args"))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        name = m.group("name")
        if not _validate_call(name, args):
            continue
        out.append({"name": name, "arguments": args})
    return out


# ---------------------------------------------------------------------------
# vLLM ToolParser subclass
# ---------------------------------------------------------------------------

if _VLLM_AVAILABLE:

    class QwenCoderToolParser(ToolParser):  # type: ignore[misc, valid-type]
        """Minimal single-path vLLM tool-call parser.

        See module docstring for the contract and the 12% documented
        miss-rate for the XML-tag-split form.
        """

        # ``required`` and named ``tool_choice`` won't be coerced through
        # guided JSON output (we don't know the model's JSON grammar and
        # the harness already routes through ``auto``), so route both
        # modes through ``extract_tool_calls``.
        supports_required_and_named: bool = False

        def __init__(
            self,
            tokenizer: TokenizerLike,
            tools: Optional[List[Tool]] = None,
        ):
            super().__init__(tokenizer, tools)
            self.model_tokenizer = tokenizer

        def extract_tool_calls(
            self,
            model_output: str,
            request: Any,
        ) -> ExtractedToolCallInformation:
            """One-path tool-call extraction.

            Maps ``model_output`` to ``ExtractedToolCallInformation`` via
            ``parse_text``; if no calls are found, the entire content is
            returned as plain text (the regular OpenAI ``content``).
            """
            calls = parse_text(model_output)

            if not calls:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            tool_calls: List[ToolCall] = []
            for c in calls:
                # arguments MUST be a JSON-encoded string on the OpenAI wire.
                tool_calls.append(
                    ToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        function=FunctionCall(
                            name=c["name"],
                            arguments=json.dumps(c["arguments"], ensure_ascii=False),
                        ),
                    )
                )

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content="",
            )

        # Streaming mode is a thin wrapper that re-runs the static
        # extractor on the cumulative text.  Adequate for non-streaming
        # callers (our default).  True token-by-token streaming with
        # stateful partial-JSON parsing is out of scope here.
        def extract_tool_calls_streaming(  # noqa: D401
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
            request: Any,
        ) -> Any:
            return self.extract_tool_calls(current_text, request)


# ---------------------------------------------------------------------------
# Plugin registration (eager; vLLM resolves synchronously)
# ---------------------------------------------------------------------------

if _VLLM_AVAILABLE and ToolParserManager is not None:
    try:
        ToolParserManager.register_module(
            name="qwen_coder_json",
            module=QwenCoderToolParser,
            force=True,
        )
        logger.info(
            "Registered QwenCoderToolParser as 'qwen_coder_json' "
            "(single-path minimal parser)."
        )
    except Exception:  # pragma: no cover
        logger.exception("Failed to register QwenCoderToolParser")


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

# These samples cover every wrapper shape the gate produced on real Coder
# output EXCEPT the XML-tag-split form (deliberately not supported; see
# module docstring).  Each sample MUST round-trip exactly.
_CANONICAL_SAMPLES = [
    # hermes flat-JSON inside `````json `` fence
    (
        '```json\n{"name": "run_tests", "arguments": {"cmd": "pytest"}}\n```',
        [{"name": "run_tests", "arguments": {"cmd": "pytest"}}],
    ),
    # hermes flat-JSON (no fence)
    (
        '{"name": "list_files", "arguments": {"path": "."}}',
        [{"name": "list_files", "arguments": {"path": "."}}],
    ),
    # `<response>` wrapper inside `````xml `` fence (most common gate form)
    (
        '```xml\n<response>\n    {"name": "read_file", "arguments": {"file_path": "/tmp/foo.py"}}\n</response>\n```',
        [{"name": "read_file", "arguments": {"file_path": "/tmp/foo.py"}}],
    ),
    # `<tool_call>` flat-JSON (hermes exact, with my custom chat template)
    (
        '<tool_call>\n{"name": "list_files", "arguments": {"path": "src/"}}\n</tool_call>',
        [{"name": "list_files", "arguments": {"path": "src/"}}],
    ),
    # `function_call` wrapper hermes-shaped (samples 4/8 in gate)
    (
        '```xml\n<function_call>\n{"name": "edit", "arguments": {"file_path": "/tmp/x.py", "old_string": "a - b", "new_string": "a + b"}}\n</function_call>\n```',
        [{"name": "edit",
          "arguments": {"file_path": "/tmp/x.py", "old_string": "a - b", "new_string": "a + b"}}],
    ),
    # multiple calls in one output (gate prompt 6 stripped of XML-tag-split)
    (
        '<response>\n    {"name": "list_files", "arguments": {"path": "src/"}}\n</response>\n'
        '<response>\n    {"name": "read_file", "arguments": {"file_path": "src/app.py"}}\n</response>',
        [{"name": "list_files", "arguments": {"path": "src/"}},
         {"name": "read_file", "arguments": {"file_path": "src/app.py"}}],
    ),
    # XML-tag-split form: *intentionally* not supported — confirms we
    # don't fall back to a second shape detector.
    (
        '<tool_call>\n<name>list_files</name>\n<arguments>{"path": "src/"}</arguments>\n</tool_call>',
        [],
    ),
    # No tool call: confirm pure prose returns empty.
    ("好的，让我先看看 calculator.py 的内容再决定怎么改。", []),
]


if __name__ == "__main__":  # pragma: no cover - dev-only smoke test
    print("=" * 64)
    print("Standalone parser test (no vLLM server)")
    print("=" * 64)
    failures = 0
    for i, (sample, expected) in enumerate(_CANONICAL_SAMPLES, 1):
        got = parse_text(sample)
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
