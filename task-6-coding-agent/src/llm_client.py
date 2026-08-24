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