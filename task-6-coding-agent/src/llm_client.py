"""OpenAI-compatible LLM client.

Thin wrapper over the official ``openai`` SDK so the agent talks to any
local endpoint (Ollama / vLLM / sglang / llama.cpp) without caring about
transport. All HTTP failures are surfaced as :class:`LLMError` so the
agent loop can degrade gracefully instead of crashing.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("llm_client")


class LLMError(RuntimeError):
    """Wraps any HTTP / schema failure so the agent can recover gracefully."""


@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ChatCompletion:
    message: ChatMessage
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)


class LLMClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:30000/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.timeout = float(os.environ.get("OPENAI_TIMEOUT", timeout))
        self.temperature = float(os.environ.get("OPENAI_TEMPERATURE", temperature))
        self.max_tokens = int(os.environ.get("OPENAI_MAX_TOKENS", max_tokens))
        self._client = self._make_client()

    def _make_client(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise LLMError(f"openai package not installed: {e}") from e
        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    @property
    def endpoint_summary(self) -> str:
        return f"{self.model} @ {self.base_url}"

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletion:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = [_mcp_to_openai_tool(t) for t in tools]
            payload["tool_choice"] = tool_choice or "auto"
        log.debug("chat → %s msgs=%d tools=%d", self.model, len(messages), len(tools or []))
        _dump_request(self.model, payload)
        try:
            resp = self._client.chat.completions.create(**payload)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"chat completion failed: {e}") from e
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = []
            for tc in msg.tool_calls:
                fn = tc.function
                args = fn.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": fn.name, "arguments": args if isinstance(args, dict) else {}},
                })
        return ChatCompletion(
            message=ChatMessage(role="assistant", content=msg.content, tool_calls=tool_calls),
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            },
        )

    def ping(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("ping failed: %s", e)
            return False


def make_offline_client() -> "LLMClient":
    """Return a client whose ``chat`` always raises ``LLMError``.

    Lets smoke tests run without a model server.
    """
    client = LLMClient()

    def _boom(*args, **kwargs):
        raise LLMError("offline client: no model available")

    client.chat = _boom  # type: ignore[assignment]
    return client


def _mcp_to_openai_tool(t: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an MCP-shaped tool dict to OpenAI Chat Completions shape.

    MCP uses ``{name, description, inputSchema}``; OpenAI expects
    ``{type: "function", function: {name, description, parameters}}``.
    If ``t`` already has ``type=function``, it's passed through (we just
    rename ``inputSchema`` → ``parameters`` defensively).
    """
    if t.get("type") == "function" and "function" in t:
        fn = t["function"]
        params = fn.get("parameters") or fn.get("inputSchema") or {}
        return {"type": "function", "function": {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": params,
        }}
    return {"type": "function", "function": {
        "name": t.get("name", ""),
        "description": t.get("description", ""),
        "parameters": t.get("inputSchema") or t.get("parameters") or {},
    }}


def _dump_request(model: str, payload: Dict[str, Any]) -> None:
    """Print a concise but complete view of what the agent is sending to vLLM.

    Controlled by env var ``LLM_DEBUG``:
    - ``0``         : silent (production default behaviour)
    - ``1``         : summary only (one block per request)
    - ``2`` (or unset): full dump — every message, every tool_call argument,
                          the response, the usage.  This is the value you
                          want when debugging why the model misbehaves.

    Output goes to stderr so it doesn't pollute stdout when the agent
    is being driven from a script.  Lines are prefixed with
    ``[LLM_DEBUG]`` for easy grep.
    """
    import sys
    import os
    import time as _time
    level = os.environ.get("LLM_DEBUG", "2").strip()
    if level == "0":
        return
    try:
        lvl = int(level)
    except ValueError:
        lvl = 2

    msgs = payload.get("messages", [])
    n_msgs = len(msgs)
    last = msgs[-1] if msgs else {}
    sys.stderr.write("\n" + "=" * 78 + "\n")
    sys.stderr.write(
        f"[LLM_DEBUG] {model}  msgs={n_msgs}  tools={len(payload.get('tools') or [])}  "
        f"temp={payload.get('temperature')}  max_tokens={payload.get('max_tokens')}  "
        f"tool_choice={payload.get('tool_choice')}  "
        f"t={_time.strftime('%H:%M:%S')}\n"
    )

    if lvl == 1:
        # Summary only — show the last message + the system message length.
        sys0 = msgs[0] if msgs else {}
        sys0_content = sys0.get("content", "") or ""
        sys.stderr.write(
            f"[LLM_DEBUG]   system: role={sys0.get('role')} len={len(sys0_content)}\n"
        )
        sys.stderr.write(
            f"[LLM_DEBUG]   last  : role={last.get('role')} "
            f"content[:80]={(last.get('content') or '')[:80]!r} "
            f"tool_calls={len(last.get('tool_calls') or [])}\n"
        )
        sys.stderr.write("=" * 78 + "\n")
        return

    # Full dump (level 2).
    for j, m in enumerate(msgs):
        role = m.get("role", "?")
        content = m.get("content")
        tool_calls = m.get("tool_calls") or []
        tool_call_id = m.get("tool_call_id")
        sys.stderr.write(f"[LLM_DEBUG] --- msg[{j}] role={role} ---\n")
        if tool_call_id:
            sys.stderr.write(f"[LLM_DEBUG]   tool_call_id={tool_call_id}\n")
        if content is not None:
            if isinstance(content, str):
                sys.stderr.write(f"[LLM_DEBUG]   content({len(content)} chars):\n")
                for ln in content.splitlines():
                    sys.stderr.write(f"[LLM_DEBUG]     | {ln}\n")
            else:
                sys.stderr.write(f"[LLM_DEBUG]   content (non-str): {content!r}\n")
        if tool_calls:
            for k, tc in enumerate(tool_calls):
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments", "")
                sys.stderr.write(
                    f"[LLM_DEBUG]   tool_call[{k}] name={fn.get('name')} "
                    f"id={tc.get('id', '?')}\n"
                )
                try:
                    args_str = args_raw if isinstance(args_raw, str) else json.dumps(args_raw)
                except Exception:
                    args_str = str(args_raw)
                sys.stderr.write(f"[LLM_DEBUG]     arguments:\n")
                for ln in str(args_str).splitlines():
                    sys.stderr.write(f"[LLM_DEBUG]       | {ln}\n")
    sys.stderr.write("=" * 78 + "\n")
    sys.stderr.flush()


def to_wire_tool_calls(tool_calls):
    """Serialize internal tool_calls to OpenAI wire format.

    Internally ``function.arguments`` is a dict (parsed by
    :class:`LLMClient`, synthesised by :meth:`_parse_text_tool_calls`), but
    the Chat Completions protocol — and vLLM's validator in particular —
    requires it to be a JSON *string* on the wire.

    Returns ``None`` when there's nothing to send so the message field
    can be omitted entirely.
    """
    if not tool_calls:
        return None
    import json
    out = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        elif args is None:
            args = "{}"
        out.append({
            "id": tc.get("id"),
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": args},
        })
    return out