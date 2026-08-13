# OpenAI Function Calling / Tool Use 接口规范

## 来源
- 链接：https://platform.openai.com/docs/guides/function-calling
- 组织：OpenAI
- 发布时间：2023 年 6 月首次发布 `functions` 参数，2024 年升级为 `tools` 参数
- 核心定位：让 LLM 在响应中生成结构化 JSON，调用客户端预先注册的"工具"

> 注：本任务用 Ollama 提供 OpenAI 兼容 API，所以这里的规范就是我们的接口规范。

## 关键要点

### 1. Tool Schema 格式

OpenAI 风格的 tool 定义：

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的当前天气",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名，例如 '北京'"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "温度单位"
        }
      },
      "required": ["city"],
      "additionalProperties": false
    }
  }
}
```

要点：
- `type: "function"` 是顶层标识。
- `name`：工具名，必须匹配 `[a-zA-Z0-9_-]+`，最长 64 字符。
- `description`：给模型读的描述，**写得越清楚，模型调用越准**。
- `parameters`：JSON Schema 子集，至少包含 `type / properties / required`。

### 2. Chat Completions API 调用方式

Python 客户端（OpenAI SDK 或 Ollama 兼容层）：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama
    api_key="ollama",  # 任意字符串
)

response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[
        {"role": "system", "content": "你是一个助手..."},
        {"role": "user", "content": "北京今天多少度？"},
    ],
    tools=[TOOL_SCHEMA],  # 工具列表
    tool_choice="auto",   # auto / none / 指定某个 tool
)

# 模型返回 tool_calls
message = response.choices[0].message
if message.tool_calls:
    for tc in message.tool_calls:
        tool_name = tc.function.name
        tool_args = json.loads(tc.function.arguments)
```

注意：Ollama 的 OpenAI 兼容层**不一定支持 `tools` 字段**——Qwen2.5 模型在 Ollama 上需要检查版本；老版本可能要用 prompt 注入的方式手写 ReAct。

### 3. 关键参数

| 参数 | 取值 | 用途 |
|---|---|---|
| `tool_choice` | `"auto"` / `"none"` / `{"type": "function", "function": {"name": "xxx"}}` | 控制模型是否强制调工具 |
| `parallel_tool_calls` | bool | 是否允许一次返回多个 tool_calls（默认 True） |
| `temperature` | 0-2 | 工具调用建议设 0 或很低的值 |
| `max_tokens` | int | 长 trace 时适当调大 |

### 4. Function Calling vs ReAct 的差异

| 维度 | Function Calling | ReAct（prompt 风格） |
|---|---|---|
| 模型依赖 | 需要模型原生支持 tools 字段 | 任何 instruct 模型都能用 |
| 格式 | 结构化 JSON | 自然语言文本 |
| Thought 可见 | 不可见（除非用 `reasoning_effort`） | 可见 |
| 错误恢复 | 模型自己决定 | 可以显式喂 Observation |
| 实现难度 | 客户端简单（不用解析） | 需要自己写解析器 |

**对本任务的启示**：README 强调"手写 ReAct 循环"，所以我们走 prompt 风格，不用 `tools` 字段——但工具 schema 仍然按 OpenAI 格式写，方便 S1 Qwen-Agent 对照。

### 5. 常见坑

- **`tool_choice="auto"` 时模型可能不调工具**：尤其在简单问题上。S3 prompt 消融要测"显式 vs 隐式调用"。
- **`tools` 字段太长**：4 个工具 + 完整 description 大约 1-2k token，要考虑 prompt 预算。
- **JSON 解析失败**：模型有时返回的 `arguments` 不是合法 JSON（多一个逗号、嵌套错乱），需要兜底解析（正则或宽松 JSON）。
- **`required` 字段漏写**：模型可能不传必填参数，工具内部必须 catch KeyError。

## 与我们任务的关联

- **M1 工具 schema**：严格按上面的格式写 `TOOL_SCHEMA`。
- **M2 主循环**：模型不传 `tools` 字段，自己从 prompt 里解析 `Action / Action Input`，但解析出的 action 名要和 schema 里的 `name` 一致。
- **S1 Qwen-Agent 对照**：Qwen-Agent 用 OpenAI 格式的 `tools` 字段直接调模型，比手写 ReAct 简单；这是 baseline。
- **API key**：`api_key="ollama"` 是 Ollama 兼容层的通行做法，README 已明确说明。

## 代码片段（本任务的工具 schema 示例）

```python
# src/tools/calculator.py
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行四则运算和常见数学函数。支持的运算符: +, -, *, /, **, %; "
                       "支持的函数: sqrt, sin, cos, tan, log, log2, log10, exp, abs, round; "
                       "支持的常量: pi, e。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "合法算术表达式，例如 '(123+456)*789' 或 'sqrt(2026)'",
                },
            },
            "required": ["expression"],
        },
    },
}

def run(args: dict) -> str:
    expr = args["expression"]
    # ... 安全 eval ...
    return str(result)
```

## 不确定 / 需验证的点
- Ollama 当前版本是否对 Qwen2.5 支持 `tools` 字段：需 `curl http://localhost:11434/v1/chat/completions` 测试，或查 Ollama 文档。
- 不同 Ollama 版本的 `tool_choice` 支持情况：老版本可能忽略该参数。
