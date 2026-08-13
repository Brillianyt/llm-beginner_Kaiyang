# Qwen-Agent 学习笔记（多 Agent 架构参考）

## 来源
- 仓库：https://github.com/QwenLM/Qwen-Agent
- 文档：仓库根目录 README.md + docs/
- 维护者：阿里通义千问团队
- 核心定位：**基于 Qwen 模型的 Agent 框架**，原生支持工具调用、多 Agent 编排、code interpreter

## 关键要点
1. **Agent 类型**：
   - `Assistant`：单 agent，工具调用 + RAG
   - `Router`：根据输入路由到不同子 agent
   - `UserSimulator`：模拟用户行为
2. **核心抽象**：
   - `Agent` 类：所有 agent 的基类，含 `_run(messages)` 与 `run(messages, stream=False)` 接口
   - `LLM` 类：统一模型客户端，支持 DashScope / OpenAI 兼容后端（包括 vLLM / Ollama）
   - `Tool` 类：通过 `register_tool(name, func, description, schema)` 注册工具
3. **工具调用格式**：默认 OpenAI function calling JSON（与 OpenAI / vLLM / Ollama 兼容）；对 Qwen 系列模型有专门 prompt 模板优化。
4. **多 Agent 协作**：`GroupChat`（轮流发言）/ `Router`（分类分发）/ `UserSimulator`（对话模拟）。
5. **Code Interpreter**：内置一个 `code_interpreter` 工具，LLM 生成 Python 代码 → sandbox 执行 → 返回 stdout/stderr。这是 CodeAct 思路在 Qwen 框架里的实现。
6. **MCP 支持**（最新版）：已支持把 MCP server 接入作为外部工具源。

## 与我们任务的关联
- **本地模型对接**：Qwen-Agent 对 Qwen 系列模型有「原生优化」的 prompt 模板，可以借鉴其 system prompt 结构（虽然我们手写 agent 不直接 import Qwen-Agent）。
- **多 Agent 架构**：Qwen-Agent 的 `GroupChat` 思路对应我们的 subagent——可以参考其消息分发协议，但不要照搬（我们的实现要更简单）。
- **Tools 接口**：其 `register_tool(name, func, description, schema)` 四元组正是我们要设计的 MCP tool 元数据格式。

## 代码片段（Qwen-Agent 的工具注册与调用模式）

```python
from qwen_agent.agents import Assistant
from qwen_agent.tools import register_tool

@register_tool('read_file')
class ReadFileTool:
    description = 'Read the content of a file.'
    parameters = [{
        'name': 'path',
        'type': 'string',
        'required': True,
        'description': 'File path relative to repo root.',
    }]

    def call(self, params: dict, **kwargs) -> str:
        p = (REPO_ROOT / params['path']).resolve()
        assert p.is_relative_to(REPO_ROOT)
        return p.read_text(encoding='utf-8')

agent = Assistant(
    llm={'model': 'qwen2.5-coder-7b-instruct', 'base_url': 'http://localhost:11434/v1'},
    function_list=['read_file', 'write_file', 'run_tests'],
)
response = agent.run(messages=[{'role': 'user', 'content': '修 calculator.add bug'}])
```

## 我们应该怎么借鉴
1. **不要直接 import Qwen-Agent**：README 把它列为「对照参考」而非依赖。我们的 CodingAgent 必须自己写循环，但可以**抄一份 Qwen-Agent 的 system prompt 结构**（角色定义 + 工具列表 + 工作流步骤）。
2. **工具注册四元组**：name / description / parameters / callable——这跟 MCP 的 tool schema 完全对齐，可以共享一份元数据 dict。
3. **多 Agent 编排思路**：Qwen-Agent 的 `GroupChat` 轮询模式适合「plan → code_search → patch → test_runner」流水，但我们的 subagent 设计更简单——主 agent **显式调用**而非消息轮转。**建议**：先做显式 subagent 调用（更可控），有余力再考虑 GroupChat 风格。
4. **本地端点兼容**：Qwen-Agent 的 `base_url` 参数对应我们 `openai.OpenAI(base_url="http://localhost:11434/v1")`，对接 Ollama / vLLM / llama.cpp 都一样。
5. **code interpreter 不适合我们的 MCP 场景**：Qwen-Agent 的 code_interpreter 允许 LLM 自由 exec Python——强大但危险（沙箱成本高）。我们走 MCP 显式工具更安全，每一步行为可控。
6. **做 ablation 对照**（S2 加分项）：同一任务跑两份——一份用自己的 CodingAgent，一份用 Qwen-Agent 的 Assistant——对比 token / 成功率 / 步数。

## 主要参考来源
- 仓库：https://github.com/QwenLM/Qwen-Agent
- 文档：仓库内 docs/ 目录 + README
- 中文实践：博客园 / CSDN 多个 Qwen-Agent 教程