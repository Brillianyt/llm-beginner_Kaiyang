"""通用 LLM 客户端：OpenAI 兼容协议（Ollama / SGLang / OpenAI / vLLM）。

设计要点（来自 SYNTHESIS §6）：
- 通过环境变量 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL 切换后端
- 默认走 Ollama：http://localhost:11434/v1
- SGLang 切：`export OPENAI_BASE_URL=http://localhost:30000/v1`
- 超时 60s（Ollama 冷启动 10-30s）+ 可选预热一次
- 失败抛自定义 LLMError，由 agent 主循环 catch 成 Observation 字符串
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# 超时（秒）。Ollama 冷启动可达 30s，预留 60s
_DEFAULT_TIMEOUT = 60.0


class LLMError(Exception):
    """LLM 客户端异常（agent 主循环会 catch 转 Observation）。"""


@dataclass
class LLMConfig:
    """LLM 配置。"""
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:11434/v1"
        )
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "ollama")
    )
    model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENAI_MODEL", "qwen2.5:7b-instruct"
        )
    )
    timeout: float = _DEFAULT_TIMEOUT
    temperature: float = 0.0
    max_tokens: int = 1024


class LLMClient:
    """OpenAI 兼容客户端。封装所有 HTTP 调用，agent 只用 chat(messages)。"""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client: Any | None = None
        self._warmed_up: bool = False

    def _ensure_client(self) -> Any:
        """懒加载 openai 客户端。"""
        if self._client is None:
            try:
                # openai>=1.x
                from openai import OpenAI  # type: ignore
                self._client = OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                )
            except Exception as e:  # noqa: BLE001
                raise LLMError(f"openai 客户端初始化失败：{e}") from e
        return self._client

    def chat(self, messages: list[dict[str, str]],
             model: str | None = None,
             temperature: float | None = None,
             max_tokens: int | None = None) -> str:
        """OpenAI 风格 chat 调用。返回 assistant 文本。

        messages: [{"role": "system/user/assistant", "content": str}]
        """
        # 预热（只跑一次）
        if not self._warmed_up:
            self._warmup()
            self._warmed_up = True

        client = self._ensure_client()
        mdl = model or self.config.model
        temp = self.config.temperature if temperature is None else temperature
        mtok = self.config.max_tokens if max_tokens is None else max_tokens
        try:
            resp = client.chat.completions.create(
                model=mdl,
                messages=messages,
                temperature=temp,
                max_tokens=mtok,
                timeout=self.config.timeout,
            )
        except Exception as e:  # noqa: BLE001
            raise LLMError(
                f"LLM 调用失败（base_url={self.config.base_url}）：{e}"
            ) from e

        # 兼容 stream / 非 stream
        choice = resp.choices[0]
        text = getattr(choice.message, "content", "") or ""
        return text

    def _warmup(self) -> None:
        """预热一次模型加载（容忍失败）。"""
        try:
            client = self._ensure_client()
            client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.0,
                max_tokens=8,
                timeout=min(self.config.timeout, 30.0),
            )
        except Exception:  # noqa: BLE001
            # 预热失败不影响后续正式调用（agent 主循环会处理）
            pass

    def switch_model(self, model: str) -> None:
        """切换模型（S2 消融用）。不重建 client（基地址不变）。"""
        self.config.model = model

    def switch_backend(self, base_url: str, api_key: str | None = None) -> None:
        """切换 base_url（SGLang / vLLM 切换用）。"""
        self.config.base_url = base_url
        if api_key is not None:
            self.config.api_key = api_key
        self._client = None  # 重新构造


__all__ = ["LLMClient", "LLMConfig", "LLMError"]