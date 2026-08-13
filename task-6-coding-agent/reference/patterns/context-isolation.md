# 上下文隔离（Context Isolation）—— Subagent 设计模式

## 来源
- Claude Code 源码：`src/coordinator/coordinatorMode.ts`、`src/coordinator/workerAgent.ts`
- Anthropic 多 agent 文档：https://www.anthropic.com/engineering/building-effective-agents
- Martin Fowler「外部化状态」：https://martinfowler.com/articles/externalizing-state.html

## 关键要点
1. **核心思想**：subagent 拥有**完全独立**的 message 列表、工具子集、步数上限；与主 agent 的 context **不共享**
2. **三个独立属性**（缺一不可）：
   - **独立 message 列表**：subagent 从空 messages 开始（或只收到 task 字符串），不继承主 agent 历史
   - **独立步数上限**：subagent 自己的 `max_steps`，不消耗主 agent 预算
   - **工具子集白名单**：subagent 只能调白名单内的 tool（防止 subagent 越权）
3. **返回摘要而非 trace**：subagent 完成后只把「结论摘要」返回主 agent；主 agent 不接收中间 tool call / observation 列表
4. **为何重要**：
   - 防止主 context 被 subagent 的细节撑爆
   - 防止 subagent 的思考过程污染主对话（噪声干扰）
   - 防止 subagent 调用危险 tool（如 git reset）
5. **Claude Code 实现**：通过 `ToolUseContext.options.tools` 的子集 + `setAppStateForTasks` vs `setAppState` 区分
6. **Worker Agent 设计原则**（来自 Claude Code `workerAgent.ts` system prompt）：
   - Complete the task fully
   - Use tools proactively
   - Be thorough in research
   - Report back with concise summary
   - Investigate and fix errors before reporting failure

## 与我们任务的关联
- **README M3 提到 subagent**：主 agent 只看 subagent 摘要，不吞全部 trace
- **Subagent 类型建议**（本任务规模）：
  - `code_search_subagent`：只暴露 `read_file + grep`（如有），5 步上限
  - `test_runner_subagent`：只暴露 `run_tests + read_file`，3 步上限
- **避免陷阱**：subagent 共享主 agent 的 message 列表会失去隔离意义；subagent 调 `write_file` 可能改坏代码

## 文字版隔离模型

```
            ┌────────────────────────────────────────┐
            │           Main CodingAgent             │
            │                                        │
            │  messages_main = [                     │
            │    {system: "你是主 agent"},            │
            │    {user: issue 文本},                 │
            │    {assistant: "我先派 subagent 搜索"}, │
            │    {tool_result: subagent_summary},    │  ← 只接收摘要字符串
            │  ]                                     │
            │  max_steps = 30                        │
            │  tools = [read, write, run_tests,      │
            │           git_diff, git_apply,         │
            │           delegate_subagent]           │
            └────────────┬───────────────────────────┘
                         │ 派发 subagent
                         ▼
            ┌────────────────────────────────────────┐
            │       Subagent (e.g. code_search)      │
            │                                        │
            │  messages_sub = [                      │   ← 完全独立
            │    {system: "你是搜索 subagent"},       │
            │    {user: "找出 add() 函数定义"},      │
            │    {assistant: ..., tool_call: ...},   │
            │  ]                                     │
            │  max_steps = 5                         │   ← 独立上限
            │  tools = [read_file, grep_file]        │   ← 工具白名单
            └────────────┬───────────────────────────┘
                         │ 返回
                         ▼
            "搜索结果：calculator.add 定义在 calculator.py:2，
             当前实现是 return a - b（疑似 bug）。"
            ↑ 这就是摘要；具体 tool_call 序列不返回
```

## 代码片段（Python Subagent 实现）

```python
class Subagent:
    """基类：subagent 必须独立 messages / 独立步数 / 工具白名单。"""

    def __init__(self, name: str, system_prompt: str, allowed_tools: list[str], max_steps: int = 5):
        self.name = name
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools  # ★ 工具白名单
        self.max_steps = max_steps          # ★ 独立步数上限

    def run(self, task: str, llm, all_tools: dict) -> str:
        # ★ 全新 message 列表，不继承主 agent 历史
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        # ★ 工具子集
        tools = [all_tools[t] for t in self.allowed_tools]
        for step in range(self.max_steps):
            resp = llm.chat(messages, tools=tools)
            if not resp.tool_calls:
                return resp.content           # ★ 返回最终字符串（摘要）
            for call in resp.tool_calls:
                obs = tools[call.name].call(call.args)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})
        return "[subagent reached max steps without conclusion]"


# 具体实现
class CodeSearchSubagent(Subagent):
    def __init__(self):
        super().__init__(
            name="code_search",
            system_prompt="你是代码搜索 subagent。用 read_file / grep 找代码定义，"
                          "返回位置和简短摘要，不修改文件。",
            allowed_tools=["read_file", "grep"],   # ★ 不能 write_file / run_tests
            max_steps=5,
        )

class TestRunnerSubagent(Subagent):
    def __init__(self):
        super().__init__(
            name="test_runner",
            system_prompt="你是测试执行 subagent。跑 pytest 并分析失败原因，"
                          "返回哪些测试失败、为什么。不修改代码。",
            allowed_tools=["read_file", "run_tests"],  # ★ 不能 write_file
            max_steps=3,
        )

# Main Agent 使用
class CodingAgent:
    def __init__(self, ..., subagents: dict[str, Subagent]):
        self.subagents = subagents

    def delegate_subagent(self, name: str, task: str) -> str:
        sub = self.subagents[name]
        summary = sub.run(task, self.llm, self.tool_registry)
        # ★ 只把 summary 注入主 agent 的 messages
        self.messages.append({"role": "tool", "content": f"[{name}] {summary}"})
        return summary
```

## 我们应该怎么借鉴
1. **每个 subagent 都有独立 max_steps**——主 agent 30、code_search 5、test_runner 3，不要共用计数器
2. **每个 subagent 都有 tools 白名单**——绝对不要给 `write_file` 给 search 类 subagent
3. **每个 subagent 都有 system prompt**——明确「返回摘要」「不修改代码」「专注单一任务」
4. **返回字符串而非 dict**——主 agent 拿到的是 plain text，丢进 message 当 observation
5. **不要复用主 agent 的 tool registry**——subagent 应该只看到自己的工具子集；tool registry 是个 dict[tool_name, Tool]，subagent 取出子集
6. **可以加 SubagentStop 钩子**（加分项）——subagent 完成时触发，参考 Claude Code hook 系统
7. **写 ablation 实验**（S2 加分项）：有 subagent vs 无 subagent，对比 token 消耗和成功率

## 主要参考来源
- Claude Code `src/coordinator/workerAgent.ts`：worker agent 定义 + 工具白名单
- Claude Code `src/coordinator/coordinatorMode.ts`：coordinator 模式切换
- Anthropic Building Effective Agents：https://www.anthropic.com/engineering/building-effective-agents