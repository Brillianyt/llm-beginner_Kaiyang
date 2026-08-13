"""task-5-tool-agent src package.

ReAct 工具调用 agent 的实现：
- `agent.py`：ReActAgent 主循环（状态机）
- `prompt.py`：PromptBuilder + few-shot
- `parser.py`：ActionParser
- `llm_client.py`：OpenAI 兼容客户端（Ollama / SGLang / OpenAI）
- `trace.py`：AgentTrace 数据结构
- `tools/`：4 个工具 + Tool 基类 + 注册表
"""