"""OpenAI-compatible LLM client.

Why this module exists:

* ``CodingAgent`` should not know whether it's talking to Ollama, vLLM,
  SGLANG or llama.cpp — only that the endpoint speaks the OpenAI
  Chat Completions schema.
* The base URL is configurable via ``OPENAI_BASE_URL``; the model via
  ``QWEN_MODEL`` (defaults to a Qwen2.5-Coder-7B-Instruct variant).
* All HTTP errors are surfaced as ``LLMError`` so the agent loop can
  convert them into ``[ERROR] ...`` tool observations and recover.

**Switching to SGLANG**:

.. code-block:: bash

   python -m sglang.launch_server \\
       --model-path Qwen/Qwen2.5-Coder-7B-Instruct \\
       --port 30000

Then run the agent with:

.. code-block:: bash

   export OPENAI_BASE_URL=http://localhost:30000/v1
   export QWEN_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
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

    def to_openai(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class ChatCompletion:
    message: ChatMessage
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Thin wrapper over the OpenAI SDK that supports local endpoints.

    If the ``openai`` package is unavailable, the client raises
    :class:`LLMError` so the agent can degrade gracefully (the smoke
    test path runs without an actual model).
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:11434/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.environ.get("QWEN_MODEL", "qwen2.5-coder:7b-instruct")
        self.timeout = float(os.environ.get("OPENAI_TIMEOUT", timeout))
        self.temperature = float(os.environ.get("OPENAI_TEMPERATURE", temperature))
        self.max_tokens = int(os.environ.get("OPENAI_MAX_TOKENS", max_tokens))
        self._client = self._make_client()

    # ------------------------------------------------------------------
    # Factory / introspection
    # ------------------------------------------------------------------
    def _make_client(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise LLMError(f"openai package not installed: {e}") from e
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @property
    def endpoint_summary(self) -> str:
        return f"{self.model} @ {self.base_url}"

    # ------------------------------------------------------------------
    # Core chat call
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletion:
        """Issue one Chat Completion call.

        ``messages`` are plain OpenAI dicts. ``tools`` are passed straight
        through (MCP-tool schema is OpenAI-compatible).
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            payload["tool_choice"] = tool_choice or "auto"
        log.debug("chat → %s messages=%d tools=%d", self.model, len(messages), len(tools or []))
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
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": fn.name,
                            "arguments": args if isinstance(args, dict) else {},
                        },
                    }
                )
        return ChatCompletion(
            message=ChatMessage(
                role="assistant",
                content=msg.content,
                tool_calls=tool_calls,
            ),
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            },
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )

    # ------------------------------------------------------------------
    # Lightweight helpers
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """Cheap health check — ``True`` if the endpoint responds."""
        try:
            self._client.models.list()
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("ping failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# Stand-alone helper for offline / mock runs
# ---------------------------------------------------------------------------

def make_offline_client() -> "LLMClient":
    """Return a client that raises ``LLMError`` on every chat call.

    Used by smoke tests so they don't require a running model server.
    """
    # We still construct the OpenAI client (cheap), but we wrap the chat
    # method to always error.
    client = LLMClient()
    real_chat = client.chat

    def _boom(*args, **kwargs):
        raise LLMError("offline client: no model available")

    client.chat = _boom  # type: ignore[assignment]
    client._real_chat = real_chat  # type: ignore[attr-defined]
    return client
