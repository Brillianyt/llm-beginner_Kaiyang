# smolagents - CodeAgent 学习笔记

## 来源
- 仓库：https://github.com/huggingface/smolagents
- 维护者：HuggingFace（Aymeric Roucher 团队）
- 核心定位：**极简（"smol"）Python agent 框架**，主打 ReAct + CodeAct 两种范式

## 关键要点
1. **两种 Agent 类**：
   - `CodeAgent`：LLM 每步生成 Python 代码，本地 `exec` 执行；适合复杂多步操作。
   - `ToolCallingAgent`：LLM 用 JSON 工具调用；兼容性更好（OpenAI / Anthropic API 原生）。
2. **极简 API**：
   ```python
   from smolagents import CodeAgent, HfApiModel, tool

   @tool
   def my_tool(arg: str) -> str:
       """docstring 会变成 description."""
       return "..."

   agent = CodeAgent(tools=[my_tool], model=HfApiModel())
   result = agent.run("...")
   ```
3. **核心循环（伪代码）**：
   ```python
   while step < max_steps:
       prompt = build_prompt(system, history, tools)
       action = model.generate(prompt)         # Python 代码 或 JSON 调用
       observation = execute(action)            # sandbox exec 或 tool call
       history.append({"step": step, "action": action, "observation": observation})
       if action.finished:
           return action.final_answer
   ```
4. **内置能力**：
   - `planning_interval=N`：每隔 N 步让 LLM 重新规划（防止长任务走偏）
   - `max_steps`：硬上限
   - `step_callbacks`：每步钩子（用于 trace / 日志）
   - `final_answer_checks`：解析最终答案前的后处理
5. **Tool 接口**：`@tool` 装饰器从 docstring 自动生成 OpenAI 风格 schema；参数类型注解转 JSON Schema。
6. **模型抽象**：`HfApiModel`（HF Inference）/ `LiteLLMModel`（OpenAI / Anthropic / Ollama）/ `TransformersModel`（本地 transformers）。

## 与我们任务的关联
- **M3（agent loop）+ M4（Trace）**：smolagents 的核心循环就是我们要写的 `CodingAgent.run`。其 step_callbacks 机制正好对应我们要在 Trace 里记录的每步内容。
- **本地模型对接**：本任务用本地 Qwen2.5-Coder-7B（OpenAI 兼容端点），对应 `LiteLLMModel(model_id="openai/Qwen/Qwen2.5-Coder-7B-Instruct", api_base="http://localhost:11434/v1")`。我们手写版本直接用 `openai.OpenAI(base_url=..., api_key=...)` 客户端。
- **规划间隔**：smolagents 的 `planning_interval=3` 思路值得借鉴——每 3 步让 LLM 重看一次 plan，避免在错的方向上反复 tool call。

## 代码片段（smolagents 风格 agent loop）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="EMPTY")

class CodeAgent:
    def __init__(self, tools, model="qwen2.5-coder:7b", max_steps=20):
        self.tools = tools
        self.model = model
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        history = [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": task}]
        for step in range(self.max_steps):
            resp = client.chat.completions.create(
                model=self.model,
                messages=history,
                tools=[t.openai_schema for t in self.tools],
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                history.append(msg)
                for call in msg.tool_calls:
                    obs = self.execute_tool(call)
                    history.append({"role": "tool", "tool_call_id": call.id, "content": str(obs)})
            else:
                return msg.content  # 显式回答 = done
        return "max steps reached"
```

## 我们应该怎么借鉴
1. **直接照抄 step 循环结构**：`while not done: build_prompt → model → parse action → execute → record → check done`。
2. **planning_interval 机制**：实现一个 `_maybe_replan()` 函数，每 3 步调用一次「基于现状重写 plan」的 prompt；plan 不变就继续，变就重置后续策略。
3. **Tool schema 复用 OpenAI function calling 格式**：MCP `inputSchema` 和 OpenAI `tools[].function.parameters` 是同一份 JSON Schema，可以共用一份 dict。
4. **Trace 钩子**：smolagents 的 `step_callbacks` 思想——在每步末尾调用一个 `on_step(step_info)`，把 step_info append 到 trace list，最后 `return trace`。
5. **不要 import smolagents**：本任务强调「手写」，对照学习即可，**不要直接 `pip install smolagents` 然后让它跑**——那就不算自己实现了。但可以在 `ablations/` 里写一个对照实验：同一 prompt，分别用自己的 CodingAgent 和 smolagents 的 CodeAgent 跑，对比 token 消耗与成功率（S2 加分项）。

## 主要参考来源
- smolagents 仓库：https://github.com/huggingface/smolagents
- README 与 quick start：在仓库根 README.md
- HuggingFace 博客：smolagents 介绍页（搜索 "smolagents huggingface"）