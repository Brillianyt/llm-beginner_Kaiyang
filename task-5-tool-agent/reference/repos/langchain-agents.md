# LangChain Agents 学习笔记

## 来源
- 链接：https://github.com/langchain-ai/langchain
- 作者/组织：LangChain AI
- 发布时间：持续迭代（v0.1 → v0.2 → v0.3）
- 核心定位：通用 LLM 应用框架，Agent 模块提供 ReAct、Plan-and-Execute、Conversational 等多种 agent 模板

> 说明：本任务"不许用框架的 agent 封装"，所以 LangChain 主要是**反向参考**——看它怎么组织代码、抽哪些接口、踩过什么坑，然后我们自己手写。

## 关键要点

### 1. `create_react_agent` + `AgentExecutor` 双层结构
- `create_react_agent(llm, tools, prompt)` 返回一个"能决策"的 agent（本质上是 callable，接收 messages 返回 messages）。
- `AgentExecutor(agent=..., tools=..., max_iterations=...)` 包一层负责"调工具、解析输出、写回 Observation、控制步数"。

这种"决策 / 执行"分层和我们手写 agent 的 `ReActAgent.run()` 一一对应：决策层是 prompt + LLM，执行层是 while 循环 + tool 调用。

### 2. 0.2+ 迁移到 LangGraph
社区资料指出：从 LangChain 0.2 起，`create_react_agent` 已迁移到 `langgraph.prebuilt.create_react_agent`，原 `langchain.agents` 模块大幅瘦身。这意味着：

- 老教程（v0.1）的代码可能在 v0.3 已不能直接跑。
- 选 LangChain 版本对照时**先 pin 住版本**：`pip install langchain==0.2.x`。

### 3. ReAct Prompt 模板
LangChain Hub 上的 `hwchase17/react` 是社区事实标准的 ReAct 提示模板：

```
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

`agent_scratchpad` 是 LangChain 自动维护的"历史 Thought/Action/Observation"字符串拼装，**这正是我们要手写的核心部分**。

### 4. Tool 抽象
LangChain 的 Tool 抽象：
- `BaseTool` 基类，子类实现 `_run` 和 `_arun`（同步 / 异步）。
- `tool` 装饰器把函数转成 Tool，自动从 docstring 提取 description。
- Tool 的 schema 通过 Pydantic 表达，LangChain 自动转成 OpenAI function calling JSON。

### 5. 关键经验教训（社区共识）
- LangChain 0.2 之前的 `initialize_agent` 已被弃用，新代码不要用。
- `verbose=True` 是调试神器，所有 trace 会打到 stdout，本任务可以用类似机制做 `trace_recorder`。
- `max_iterations` 默认 15，本任务 10 题大多是 1-3 步就够，10 步上限保险。

## 与我们任务的关联

- **M2 主循环骨架**：直接照搬 LangChain 的 `while steps < max: llm → parse → tool.run → append` 结构。
- **M4 命中率**：LangChain 默认的 ReAct prompt 在 7B 模型上命中率不一定高（因为默认是英文 prompt 模板、且 few-shot 偏向问答），我们手写可以针对中文任务集调优。
- **S1 加分项**：可以用 LangChain 做对照——但需要 pin 版本，且要写一个适配器把我们 4 个工具包装成 LangChain Tool。
- **trace 结构**：LangChain `AgentExecutor` 返回的 dict 包含 `intermediate_steps`，可以映射成我们 README 要求的 `AgentTrace["steps"]`。

## 代码片段（AgentExecutor 风格的伪代码）

```python
# 受 LangChain AgentExecutor 启发的伪代码
class ReActAgent:
    def __init__(self, llm, tools, max_steps=10):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
    
    def run(self, task: str) -> dict:
        messages = self._build_prompt(task)
        steps = []
        for i in range(self.max_steps):
            response = self.llm.chat(messages)
            thought, action, action_input = self._parse(response)
            steps.append({"thought": thought, "action": action, "action_input": action_input})
            
            if action == "Final Answer":
                return {"steps": steps, "final_answer": action_input, "success": True}
            
            observation = self._call_tool(action, action_input)
            steps[-1]["observation"] = observation
            messages = self._append(messages, thought, action, action_input, observation)
        
        # 步数耗尽：尝试从最后几步拼出答案
        return {"steps": steps, "final_answer": steps[-1].get("observation", ""), "success": False}
```

这就是 LangChain AgentExecutor 内部的大致流程（去掉错误处理细节后），约 30 行代码。

## 我们应该怎么借鉴

1. **scratchpad 字符串拼接**：把历史的 Thought/Action/Action Input/Observation 按固定格式拼起来，作为下一轮 messages 的一部分。LangChain 的 `agent_scratchpad` 是范本。
2. **`Final Answer` 当作一个特殊的"工具"**：模型选这个 action 就终止。这避免了"检测模型说'完成了'"这种脆弱方式。
3. **`max_iterations` 必须设**：默认建议 10。本任务 10 题里第 5、9 题最多 3-4 步，10 步足够。
4. **错误处理用 `try/except` + Observation 反馈**：和 LangChain `AgentExecutor.handle_parsing_errors` 等价。
5. **trace 至少包含 step_idx / thought / action / action_input / observation**：和 LangChain `intermediate_steps` 兼容，方便后续对照。
6. **不要用 LangChain 的 Tool 抽象**：它和 Pydantic 深度耦合，引入我们项目得不偿失。

## 不需要借鉴的
- LangChain 的 callback / event 机制（太重）。
- LangGraph 的图编排（本任务是线性循环，不需要 DAG）。
- Conversational Memory / Chat History 类功能（任务之间不共享上下文）。
