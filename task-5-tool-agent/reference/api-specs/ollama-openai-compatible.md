# Ollama OpenAI 兼容 API

## 来源
- 链接：https://github.com/ollama/ollama/blob/main/docs/openai.md
- 组织：Ollama
- 发布时间：持续维护
- 核心定位：让 Ollama 本地模型对外暴露 OpenAI 风格的 HTTP API，便于无缝对接 OpenAI 客户端 SDK

## 关键要点

### 1. 启动 Ollama 服务

```bash
ollama pull qwen2.5:7b-instruct
ollama serve  # 默认监听 http://localhost:11434
```

服务起来后，提供两个 API：
- 原生 API：`http://localhost:11434/api/...`（POST）
- OpenAI 兼容 API：`http://localhost:11434/v1/...`（POST + GET）

### 2. OpenAI 兼容的 endpoint

| 路径 | 对应 OpenAI |
|---|---|
| `POST /v1/chat/completions` | `client.chat.completions.create()` |
| `POST /v1/completions` | `client.completions.create()` |
| `GET /v1/models` | `client.models.list()` |

### 3. 客户端配置

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # 任意非空字符串，Ollama 不校验
)
```

也可以用环境变量：

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
```

环境变量设置后，直接 `from openai import OpenAI; client = OpenAI()` 即可。

### 4. Chat Completions 调用

```python
response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
    ],
    temperature=0,
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

### 5. Tool Calling 支持

Ollama 在 0.3+ 版本开始支持 `tools` 字段（OpenAI 风格）。具体支持哪些模型取决于模型本身——Qwen2.5-Instruct 系列官方支持。

调用示例：

```python
response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": "北京天气"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
# response.choices[0].message.tool_calls 是 list
```

### 6. 与本任务的特殊关系

**本任务的手写 ReAct 走 prompt 风格，不依赖 Ollama 的 `tools` 字段**——这样兼容性最好，老版 Ollama / llama.cpp / vLLM 都能用。即使工具 schema 按 OpenAI 格式写，调用时也不传 `tools` 参数，而是在 system prompt 里手动把 schema 文本塞进去。

### 7. 替代部署方案（README 已列）

| 方案 | 显存 | 速度 | 配置 |
|---|---|---|---|
| Ollama 原生 | 8GB+ | 中 | 简单 |
| vLLM (AWQ 量化) | 6GB+ | 快 | 中等 |
| llama.cpp (GGUF q4_k_m) | 5GB+ | 慢 | 简单 |

我们的客户端代码只用 `openai` SDK，三个方案都可以无缝切换——这就是 README 强调"OpenAI 兼容 API"的价值。

### 8. 常见坑

- **未启动 `ollama serve` 就调 API**：连接拒绝。`ollama serve` 在新版会自动后台启动，但有时需要手动。
- **`api_key` 误填为真的 OpenAI key**：会真的去连 openai.com，浪费 token。**填 `"ollama"` 或任意非空字符串**。
- **`base_url` 末尾漏 `/v1`**：会报 404。**必须 `http://localhost:11434/v1`**。
- **模型名带冒号**：`qwen2.5:7b-instruct` 是 Ollama 命名，不是 OpenAI 的 `gpt-4` 风格。
- **流式 vs 非流式**：本任务不需要流式（agent 循环要拿到完整响应再解析），用默认非流式即可。
- **超时设置**：默认 SDK 超时很长，但本地推理也要 1-30 秒。建议 `client = OpenAI(timeout=60)`。

## 与我们任务的关联

- **M2 主循环**：用 `openai` SDK 调 Ollama，prompt 拼到 messages 里发出去，解析响应内容。
- **S2 不同模型尺寸**：换 `model` 字段即可——`qwen2.5:1.5b-instruct`、`qwen2.5:7b-instruct`、`qwen2.5:14b-instruct`。
- **环境变量**：README 已经规定 `OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama`，按这个配置客户端就行。

## 代码片段

```python
# src/llm_client.py（建议封装一层）
import os
from openai import OpenAI

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            timeout=60,
        )
    return _client

def chat(messages, model=None, temperature=0):
    client = get_client()
    model = model or os.getenv("OPENAI_MODEL", "qwen2.5:7b-instruct")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
```

## 不确定 / 需验证的点
- Ollama 当前版本是否原生支持 Qwen2.5 的 `tools` 字段：需要实际跑一次 API 看响应里有没有 `tool_calls`。
- 不同 Ollama 版本的 `parallel_tool_calls` 支持：本任务用不到，但 S3 消融时可能要测。
- llama.cpp server (`llama-server`) 是否提供 `/v1/chat/completions`：较新版本支持，需查版本。
