# 状态机模式与 ReAct 循环

## 来源
- 模式分类：经典软件工程设计模式（State Machine / FSM）
- 应用领域：对话系统、agent 循环、游戏 AI、工作流引擎
- 相关参考：LangGraph（LangChain 的图状态机实现）、AWS Step Functions、Microsoft Autogen

## 关键要点

### 1. ReAct 循环天然是状态机
把 ReAct 展开成状态转移：

```
[INIT] --接收 task--> [PLANNING]
                       |
                       v
                   [THOUGHT] --生成 Thought--> [ACTION]
                                                  |
                                                  v
                                          [ACTION_PARSE]
                                          |           |
                                解析成功 ↓           解析失败 ↓
                                        [OBSERVATION]    [RETRY_PROMPT]
                                              |                |
                                              v                v
                                          [APPEND]      返回 [THOUGHT]
                                              |
                                              v
                                       {Final Answer?}
                                       |            |
                                       是 ↓          否 ↓
                                   [TERMINATE]    {步数用尽?}
                                                  |            |
                                                  是 ↓          否 ↓
                                              [TERMINATE]   返回 [THOUGHT]
```

虽然 200 行代码不需要显式定义 enum 状态，但**设计时心里要有这张图**，写出来的代码才会清晰。

### 2. 状态机的优势
- **可调试**：每一步都在固定状态转移上，trace 能精确指出"卡在哪一步"。
- **可扩展**：加新状态（如"等待用户确认"、"调用 LLM Judge 评估"）不影响其他状态。
- **可测试**：每个状态转移可以独立单测（喂入状态 + 输入，断言输出）。

### 3. 实现 ReAct 时状态机的对应代码结构

```python
class ReActAgent:
    def run(self, task):
        state = "INIT"
        messages = self._init_prompt(task)
        steps = []
        
        while state != "TERMINATE":
            if state == "INIT":
                state = "THOUGHT"
            elif state == "THOUGHT":
                response = self.llm.chat(messages)
                parsed = self._parse(response)
                if parsed["action"] == "Final Answer":
                    return self._finalize(parsed["action_input"], steps, success=True)
                steps.append(parsed)
                state = "ACTION" if parsed["action"] else "RETRY"
            elif state == "ACTION":
                try:
                    obs = self.tools[parsed["action"]].run(parsed["action_input"])
                except Exception as e:
                    obs = f"[ERROR: {type(e).__name__}: {e}]"
                steps[-1]["observation"] = obs
                state = "THOUGHT"
            elif state == "RETRY":
                messages.append({"role": "user", "content": "请按格式输出：Thought / Action / Action Input"})
                state = "THOUGHT"
            if len(steps) > self.max_steps:
                return self._finalize(steps[-1].get("observation", ""), steps, success=False)
```

### 4. 状态机的"边"——转移条件
每条边都有明确触发条件：
- `INIT → THOUGHT`：始终转移
- `THOUGHT → ACTION`：解析出有效 action
- `THOUGHT → TERMINATE`：解析出 `Final Answer`
- `THOUGHT → RETRY`：解析失败（无 action / action 名不在工具表里）
- `RETRY → THOUGHT`：始终转移（重新调用 LLM）
- `ACTION → THOUGHT`：始终转移（无论工具是否抛错）

### 5. 状态机 + LLM 的张力
纯 FSM 的所有边都由代码决定，但 ReAct 的"决策"由 LLM 决定（哪个 action）。所以本任务其实是"半状态机"：状态骨架在代码里，状态内的内容由 LLM 生成。这种"FSM 框架 + LLM 内容"的混合模式是现代 agent 框架（LangGraph、AWS Bedrock Agents）的共同选择。

## 与我们任务的关联

- **M2 主循环**：写成隐式状态机（用 `if/elif` 分支或一个 dict-of-handlers），代码可读性会远好于"一个长 while + 嵌套 if"。
- **M3 错误恢复**：`ACTION → OBSERVATION` 这条边的 Observation 可以是工具返回，也可以是错误字符串——状态机天然容纳这种情况。
- **trace 结构**：每个 step 对应一次状态转移，trace 里记录 state 名有助于调试（`{"state": "ACTION", "tool": "calculator", ...}`）。
- **S4 错误注入**：可以在 `ACTION` 状态前插入 `INJECT_ERROR` 状态，模拟工具失败。

## 我们应该怎么借鉴

1. **主循环写成 `while True` + `if state == ...`**：约 20 行能写完，比 100 行嵌套 if 清晰。
2. **每个状态对应一个 handler 函数**：`_handle_thought()`、`_handle_action()`，主循环只负责分发。
3. **trace 里加 `state` 字段**：调试时一眼看出卡在哪。
4. **终止条件集中在函数顶部**：`_should_terminate(parsed, step_count)`，避免散落在循环体里。
5. **不要过度抽象**：本任务 4 个状态就够了，不要学 LangGraph 把所有东西画成 DAG。
6. **错误恢复是状态机的一条边，不是异常**：把 `[ERROR: ...]` 当成合法的 Observation 字符串喂回去，循环继续。

## 不需要借鉴的
- 全 FSM（每条边都靠代码）：本任务 LLM 必须参与决策。
- 可视化状态机图：200 行代码不需要画图工具。

## 参考实现
- LangGraph：用 Python dict 定义状态和边，自动驱动 LLM 调用。
- 本仓库 `task-1-transformer/`（如果存在）通常有 Transformer 的状态机实现可参考——但和本任务无关，列在这里仅为知识完整性。
