# Qwen-Agent 框架学习笔记

## 来源
- 链接：https://github.com/QwenLM/Qwen-Agent
- 作者/组织：阿里通义千问团队（QwenLM）
- 发布时间：2023 年中持续维护
- 核心定位：基于 Qwen 系列模型的 agent 框架，内置 ReAct、Function Calling、Tool Use、Code Interpreter 等多种 agent 模式

> 说明：以下内容综合自公开 README、官方文档摘要与社区博客。我没有直接读仓库源码，所以具体 API 细节（如类名、字段名）以官方文档为准；下面给的是基于行业惯例的合理推断，请以你实际 `pip install qwen-agent` 后查到的 API 为准。

## 关键要点

### 1. 多种 agent 模板
Qwen-Agent 内置了至少两类 agent：
- **ReAct 风格 agent**：自己手写循环、解析 Thought/Action，工具以字典形式注册。
- **Function Calling 风格 agent**：依赖 Qwen 模型原生 tool calling 能力（Qwen2.5 已支持 OpenAI 兼容的 `tools` 字段）。

S1 加分项要做"Qwen-Agent 对照"，最直接的做法是选其中一种（推荐 Function Calling 风格，因为 prompt 短、成功率高），用同样 4 个工具注册进去，对比命中率。

### 2. 工具注册方式（推断）
社区资料里 Qwen-Agent 的工具通常是一个 Python 类，导出：
- `name`：字符串
- `description`：给模型看的描述
- `parameters`：JSON Schema 风格的 dict（OpenAI function calling 格式）
- `call(args)`：实际执行函数

注册一般通过一个 dict/registry：

```python
from qwen_agent.tools import Tool

class Calculator(Tool):
    name = "calculator"
    description = "执行四则运算和数学函数"
    parameters = [{"name": "expression", "type": "string", "required": True}]
    
    def call(self, params, **kwargs):
        return str(eval_safe(params["expression"]))

tool_map = {"calculator": Calculator()}
```

> 这部分 API 是基于行业惯例推断的，请以实际版本为准。

### 3. ReAct 提示词风格
Qwen-Agent 的 ReAct 模板沿用论文格式，但工具列表写在 system message 头部：

```
You are a helpful assistant. You have access to the following tools:

calculator: ...
python_sandbox: ...
file_search: ...
wiki: ...

Use the following format strictly:
Thought: ...
Action: <tool_name>
Action Input: <JSON>
Observation: <tool output>
... (repeat)
Final Answer: <answer string>
```

注意"严格"二字——7B 模型对格式极其敏感，prompt 里要明确写"严格"和"不要解释 Action 之外的内容"。

### 4. 模型选择
Qwen-Agent 默认推荐 `qwen2.5-7b-instruct` 或更大的 `qwen2.5-14b-instruct` / `qwen2.5-72b-instruct`。Function Calling 能力随模型增大显著提升——这就是 README 里 S2 加分项的逻辑基础（不同尺寸模型的成功率对比）。

## 与我们任务的关联

- **S1 加分项**：直接用 Qwen-Agent 的 Agent 类 + 同样的 4 个工具，对比手写 ReAct 的命中率。如果手写版反而高，说明我们的 prompt/解析逻辑更精细；如果 Qwen-Agent 高，说明我们 prompt 工程还需加强。
- **M2 设计参考**：Qwen-Agent 的 prompt 模板可以直接借鉴（CC-BY 风格的开源协议），改写一下适配我们的 4 个工具即可。
- **M1 工具 schema**：Qwen-Agent 的 `Tool` 类和我们 README 要求的 `TOOL_SCHEMA` + `run(args)` 几乎一一对应，写一个适配层就能同时给我们自己的 agent 和 Qwen-Agent 用。
- **S2 模型尺寸**：换 base URL 即可——`qwen2.5:1.5b-instruct`、`qwen2.5:7b-instruct`、`qwen2.5:14b-instruct` 都是 Ollama 现成支持。

## 代码片段（基于行业惯例的合理推测）

```python
# 用 Qwen-Agent 写一个对照版
from qwen_agent.agents import ReActAgent  # 类名以官方为准
from qwen_agent.tools import Tool

class CalculatorTool(Tool):
    name = "calculator"
    description = "执行四则运算"
    parameters = [{"name": "expression", "type": "string", "required": True}]
    def call(self, params):
        return str(eval_safe(params["expression"]))

# 注册 4 个工具
llm_cfg = {
    "model": "qwen2.5:7b-instruct",
    "model_server": "http://localhost:11434/v1",
    "api_key": "ollama",
}
agent = ReActAgent(
    llm=llm_cfg,
    tool_list=[CalculatorTool(), PythonSandboxTool(), FileSearchTool(), WikiTool()],
)
trace = agent.run("计算 (123+456)*789")
```

## 我们应该怎么借鉴

1. **工具 schema 字段要全**：name / description / parameters（含 type、description、required），Qwen-Agent 的工具就是这种结构，可以直接照搬。
2. **description 写得像产品文档**：用户能看懂的句子，不要写"实现 calculator"。
3. **system prompt 顶部固定"工具列表 + 格式说明"，中间留 few-shot**：这和 Qwen-Agent 的结构一致，是 prompt 工程的成熟套路。
4. **用 Qwen-Agent 当 baseline**：我们的手写版如果显著低于 Qwen-Agent，要回头看 prompt 和错误恢复是否到位；如果高于，说明手写 prompt 投入是值得的。
5. **不要被框架的 agent 类限制**：Qwen-Agent 的 ReAct 是给通用模型用的，对 7B 模型来说模板可能过于冗长；手写时可以精简 few-shot，只保留最相关的 1-2 个示例。

## 不确定 / 需验证的点
- Qwen-Agent 当前版本（>=0.0.10）的具体 import path：`from qwen_agent.agents import ReActAgent` 还是 `from qwen_agent import Agent`？需要 `pip show qwen-agent` 看版本。
- Qwen-Agent 的 `Tool.call` 签名：`(params, **kwargs)` 还是 `(params, user_id)` 等，需要查文档或 `help()`。
- Qwen-Agent 的 AgentTrace 结构：与我们 README 要求的 `{"steps": [...], "final_answer": ..., "success": bool}` 是否兼容，需要跑通后适配。
