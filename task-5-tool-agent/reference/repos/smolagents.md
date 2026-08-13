# smolagents 框架学习笔记

## 来源
- 链接：https://github.com/huggingface/smolagents
- 作者/组织：HuggingFace
- 发布时间：2024 年底发布，2025 年持续更新
- 核心定位：极简（~1000 行核心代码）的 agent 框架，主推 CodeAgent（让模型直接写 Python 代码调用工具，而非 JSON tool call）

## 关键要点

### 1. 极简哲学
> "Simplicity: the logic for agents fits in ~1,000 lines of code (see agents.py). We kept abstractions to their minimal shape above raw code!"

这与本任务"手写约 200 行 ReAct 循环"的精神高度一致——smolagents 给了"agent 本质就是循环 + 工具注册"的参考实现。

### 2. 两种 agent 类型
- **CodeAgent**：模型生成 Python 代码片段（`thought → code → observation`），沙箱里执行。支持多步推理 + 多工具复合调用。
- **ToolCallingAgent**：模型生成结构化 JSON tool call（类似 OpenAI function calling），逐个工具调用。

本任务本质是 ToolCallingAgent 思路（OpenAI function calling 风格），但 CodeAgent 的"代码即 action"也是值得借鉴的简洁设计。

### 3. Tool 定义方式
smolagents 提供两种工具定义：
- 装饰器风格：
  ```python
  from smolagents import tool
  
  @tool
  def get_weather(city: str) -> str:
      """获取城市的当前天气。"""
      return f"{city} 25 度，晴"
  ```
- 子类风格：继承 `Tool` 基类，实现 `forward(self, *args, **kwargs)`。

装饰器风格的好处是签名即 schema（自动从 Python 类型注解生成），这对本任务"参数 schema 必须严谨"很有启发。

### 4. 安全性：E2B / Docker 沙箱
smolagents 推荐把 Python 代码放在 E2B 沙箱或 Docker 里跑，**而不是像我们 README 里说的"教学级黑白名单"**。这印证了 README 关于 python_sandbox 安全性的提醒——黑名单不可靠。

### 5. 模型无关
通过 LiteLLM 统一接入 OpenAI / Anthropic / 本地 transformers / Ollama。本任务用 Ollama 兼容 OpenAI API，所以可以无缝对接 smolagents。

## 与我们任务的关联

- **M1 工具定义**：参考 smolagents 的 `@tool` 装饰器 / `Tool` 子类模式。我们 README 要求 `TOOL_SCHEMA + run(args)`，可以在内部用 `Tool` 基类封装，外部导出 schema 和 run 接口。
- **M2 主循环**：smolagents 的 `agents.py` 是公开的（~1000 行），值得直接读源码学习"prompt 拼装 + 输出解析 + 步数控制"的实现细节。本笔记不复制源码，但建议在动手前 `git clone` 下来读 `agents.py`。
- **S1 加分项**：smolagents 也可以当对照——但本任务限定不让直接用框架的 agent 封装，所以最多用来"读源码学思路"，不能直接 import。
- **S3 prompt 消融**：smolagents 暴露了 `max_steps`、`planning_step` 等参数，是天然的消融维度。

## 代码片段（装饰器风格的"签名即 schema"思路）

```python
from typing import Literal

# 受 smolagents 启发的工具定义
class Tool:
    name: str
    description: str
    parameters: dict  # OpenAI function calling schema
    
    def run(self, args: dict) -> str:
        raise NotImplementedError


class Calculator(Tool):
    name = "calculator"
    description = "执行四则运算和数学函数（+ - * /、sqrt、sin、cos、log、abs 等）"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "合法算术表达式，例如 '(123+456)*789'"
            },
        },
        "required": ["expression"],
    }
    
    def run(self, args: dict) -> str:
        return str(_safe_eval(args["expression"]))
```

## 我们应该怎么借鉴

1. **工具用类封装**：`Tool` 基类 + `name / description / parameters / run` 四件套，子类实现具体逻辑。注册用 dict：`tools = {"calculator": Calculator(), ...}`。
2. **parameters schema 是"接口契约"**：从 Python 函数签名自动生成也好、手写也好，必须完整。少一个字段就少一层保险。
3. **主循环极简化**：smolagents 核心就 `while steps < max: response = llm(messages); parse; tool.run(); append` 这五步——我们 200 行完全可以做到。
4. **错误处理用 try/except 围 tool.run**：把异常信息拼成 "Observation: <error message>" 追加到 messages。smolagents 内部就是这么做的。
5. **不要直接抄 smolagents 的 CodeAgent**：它的"模型写 Python 代码"思路在 7B 模型上不可靠（代码错误率高），我们用 JSON tool call 更稳。

## 不需要借鉴的
- CodeAgent 的"代码即 action"路线：7B 模型写 Python 代码容易出语法错误，对 M4 命中率反而不利。
- MCP 集成（`ToolCollection.from_mcp`）：任务规模用不上。
- 多 agent 编排：本任务单 agent。
