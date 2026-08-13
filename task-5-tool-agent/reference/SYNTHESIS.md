# SYNTHESIS · 工具调用 Agent 综合架构设计

> 这是 task-5-tool-agent 的核心设计文档，汇集前面 papers / repos / patterns / api-specs 的精华，
> 给动手写代码的同学一份"读完即可撸 200 行 ReAct 循环 + 4 个工具"的蓝图。

---

## 1. 整体架构

### 1.1 层次图

```
�──────────────────────────────────────────────────────────────────┐
│                          用户 / 自检脚本                          │
│                    (eval/run.py + 10 题任务集)                    │
└────────────────┬─────────────────────────────────────────────────┘
                 │ ReActAgent().run(task)
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                       ReActAgent 主类                            │
│                                                                  │
│  ┌────────────┐   ┌──────────────┐   ┌─────────────────────┐   │
│  │ prompt_    │   │ action_      │   │ trace_recorder      │   │
│  │ builder    │   │ parser       │   │                     │   │
│  └────────────┘   └──────────────�   └─────────────────────┘   │
│           │                │                    │                │
│           ▼                ▼                    ▼                │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              while state != TERMINATE:               │      │
│  │   state ∈ {INIT, THOUGHT, ACTION, OBSERVE,          │      │
│  │             RETRY, FINAL, TERMINATE}                 │      │
│  └──────────────────────────────────────────────────────┘      │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Tool Router / Registry                        │
│         (按 dict[action_name] 查表，try/except 兜底)              │
└────────────────┬─────────────────────────────────────────────────┘
                 │
      ┌──────────┼──────────┬─────────────┐
      ▼          ▼          ▼             ▼
  calculator  python_    file_search   wiki
              sandbox
```

### 1.2 数据流

```
task (str)
   │
   ▼
[1] prompt_builder.build(task, tools) → messages: List[dict]
   │
   ▼
[2] llm_client.chat(messages) → response_text: str
   │
   ▼
[3] action_parser.parse(response_text) → {thought, action, action_input}
   │ (parse 失败 → {thought, action=None, retry=True})
   ▼
[4] decision: action == "Final Answer" ？
   ├── 是 → 终止，return AgentTrace
   └── 否 → registry.call(action, action_input) → observation: str
              │ (工具抛异常 → "[ERROR: ...]")
              ▼
[5] trace.append({thought, action, action_input, observation})
   │
   ▼
[6] prompt_builder.append(messages, thought, action, action_input, observation)
   │
   ▼
[7] 步数 < max_steps ? → 回到 [2]
              │
              └── 否 → 终止，返回 best-effort
```

### 1.3 关键设计原则

1. **单一职责**：prompt_builder / action_parser / tool_registry / trace_recorder 各管一摊，主循环只负责状态转移。
2. **失败即字符串**：所有异常（解析失败 / 工具抛错 / 网络超时）一律转成 Observation 字符串，**不要让任何异常冒泡出主循环**。
3. **确定性终止**：3 种终止条件（Final Answer / 步数耗尽 / 连续无进展），主循环外不暴露。
4. **可观测性**：每个 step 都进 trace，trace 写入 `eval/result.json` 便于自检和人工 review。

---

## 2. 核心组件设计

### 2.1 `ReActAgent` 主类（状态机形式）

文件：`src/agent.py`

```python
class ReActAgent:
    def __init__(self, llm_client=None, tools=None, max_steps=10, model=None):
        self.llm = llm_client or default_client()
        self.tools = ToolRegistry()  # 见 2.2
        for t in (tools or default_tools()):
            self.tools.register(t)
        self.max_steps = max_steps
        self.model = model or "qwen2.5:7b-instruct"
        self.prompt_builder = PromptBuilder(self.tools.schema_list())
        self.parser = ActionParser()
    
    def run(self, task: str) -> dict:
        """主入口。返回 AgentTrace = {steps, final_answer, success}。"""
        messages = self.prompt_builder.initial_messages(task)
        steps = []
        
        for step_idx in range(self.max_steps):
            # === THOUGHT 状态：调 LLM ===
            response = self.llm.chat(messages, model=self.model)
            parsed = self.parser.parse(response)
            
            # === 解析失败 → RETRY ===
            if parsed.get("retry"):
                messages.append({"role": "user", "content":
                    "请严格按格式输出：Thought / Action / Action Input / Observation"})
                continue
            
            step = {"step_idx": step_idx, "thought": parsed["thought"]}
            steps.append(step)
            
            # === 终止条件：Final Answer ===
            if parsed["action"] == "Final Answer":
                step["action"] = "Final Answer"
                step["action_input"] = parsed["action_input"]
                return {
                    "steps": steps,
                    "final_answer": parsed["action_input"],
                    "success": True,
                }
            
            # === ACTION 状态：路由 + 调用 ===
            step["action"] = parsed["action"]
            step["action_input"] = parsed["action_input"]
            observation = self.tools.call(parsed["action"], parsed["action_input"])
            step["observation"] = observation
            
            # === OBSERVE 状态：拼回 prompt ===
            messages = self.prompt_builder.append_observation(
                messages, parsed, observation
            )
            
            # === 连续无进展检测 ===
            if self._is_stuck(steps):
                return {
                    "steps": steps,
                    "final_answer": self._best_effort(steps),
                    "success": False,
                }
        
        # === 步数耗尽 ===
        return {
            "steps": steps,
            "final_answer": self._best_effort(steps),
            "success": False,
        }
    
    def inject_error(self, tool_name: str, error_msg: str = "[Injected]"):
        """S4 钩子：在指定工具的下一次调用抛错。"""
        self.tools.inject_error(tool_name, error_msg)
    
    def _is_stuck(self, steps: list) -> bool:
        """最近 3 步 Thought 重复就视为卡住。"""
        if len(steps) < 3:
            return False
        recent_thoughts = [s["thought"] for s in steps[-3:]]
        return len(set(recent_thoughts)) <= 1
    
    def _best_effort(self, steps: list) -> str:
        """步数耗尽时从最后几步 Observation 提取答案。"""
        for s in reversed(steps):
            if s.get("observation") and "[ERROR" not in s["observation"]:
                return s["observation"][:300]
        return ""
```

**代码量**：约 60-80 行（含 docstring 和错误兜底）。README 要求"约 200 行"，加上 prompt_builder / parser / registry 三个组件总共约 200 行。

### 2.2 `Tool` 基类 + `ToolRegistry`

文件：`src/tools/base.py`

```python
from abc import ABC, abstractmethod

class Tool(ABC):
    name: str
    description: str
    parameters: dict  # OpenAI function calling JSON schema
    
    @abstractmethod
    def run(self, args: dict) -> str:
        ...
    
    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """策略注册表 + 错误统一处理。"""
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._error_inject: dict[str, str] = {}  # S4 用
    
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
    
    def schema_list(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]
    
    def call(self, name: str, args: dict) -> str:
        # S4 错误注入
        if name in self._error_inject:
            msg = self._error_inject.pop(name)
            return f"[ERROR: {msg}]"
        if name not in self._tools:
            return f"[ERROR: 未知工具 '{name}'，可用：{list(self._tools.keys())}]"
        try:
            return self._tools[name].run(args)
        except Exception as e:
            return f"[ERROR: {name} 抛 {type(e).__name__}: {e}]"
    
    def inject_error(self, name: str, msg: str):
        self._error_inject[name] = msg
```

### 2.3 `PromptBuilder`

文件：`src/prompt.py`

**核心职责**：
1. 拼 system prompt：角色 + 工具列表 + few-shot 示例 + 输出格式约束
2. 把历史的 Thought/Action/Action Input/Observation 拼成 OpenAI chat 的 message 序列

```python
SYSTEM_PROMPT = """你是工具调用助手，可以调用以下工具：

{tool_descriptions}

严格按以下格式输出，每一步只输出一行：
Thought: <你的推理，下一步该做什么>
Action: <工具名，必须是上述之一或 "Final Answer">
Action Input: <JSON 对象，对应工具的 parameters>

工作流程：
1. 输出 Thought 推理
2. 输出 Action 选择工具
3. 输出 Action Input（JSON 格式）
4. 等待 Observation
5. 重复直到输出 Action: Final Answer

注意：
- 不要解释 Thought 之外的内容
- Action Input 必须是合法 JSON（双引号、无尾逗号）
- 如果工具失败，会在 Observation 看到 [ERROR: ...]，请换工具或修正参数
- 最终答案用 Action: Final Answer + Action Input: <答案字符串>"""


FEW_SHOT = [
    {
        "task": "计算 (123+456)*789 是几位数",
        "trajectory": """Thought 1: 先算出乘积。
Action 1: calculator
Action Input 1: {"expression": "(123+456)*789"}
Observation 1: 456831
Thought 2: 456831 有 6 位。
Action 2: Final Answer
Action Input 2: 6 位"""
    },
    {
        "task": "查 Geoffrey Hinton 出生年份，并算到 2026 年多少岁",
        "trajectory": """Thought 1: 先查 Hinton 的维基百科。
Action 1: wiki
Action Input 1: {"query": "Geoffrey Hinton"}
Observation 1: 杰弗里·埃弗里斯特·辛顿（Geoffrey Everest Hinton，1947 年 12 月 6 日—...
Thought 2: 出生 1947，2026 - 1947 = 79。
Action 2: calculator
Action Input 2: {"expression": "2026-1947"}
Observation 2: 79
Thought 3: 出生 1947 年，到 2026 年 79 岁（若已过生日）。
Action 3: Final Answer
Action Input 3: 1947 年出生，到 2026 年 79 岁""",
    },
    {
        "task": "在 data/agent-fixtures 下找所有 .md 文件",
        "trajectory": """Thought 1: 用 file_search 找 .md 文件。
Action 1: file_search
Action Input 1: {"pattern": "*.md", "dir": "data/agent-fixtures"}
Observation 1: README.md, todo_note.md
Thought 2: 找到两个文件。
Action 2: Final Answer
Action Input 2: 2 个 .md 文件：README.md、todo_note.md""",
    },
]


class PromptBuilder:
    def __init__(self, tool_schemas: list[dict]):
        self.tool_descriptions = self._format_tools(tool_schemas)
    
    def initial_messages(self, task: str) -> list[dict]:
        sys = SYSTEM_PROMPT.format(tool_descriptions=self.tool_descriptions)
        messages = [{"role": "system", "content": sys}]
        # few-shot 作为 assistant/user 对话加入
        for ex in FEW_SHOT:
            messages.append({"role": "user", "content": ex["task"]})
            messages.append({"role": "assistant", "content": ex["trajectory"]})
        messages.append({"role": "user", "content": task})
        return messages
    
    def append_observation(self, messages, parsed, observation):
        assistant_msg = (
            f"Thought: {parsed['thought']}\n"
            f"Action: {parsed['action']}\n"
            f"Action Input: {parsed['action_input']}"
        )
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        return messages
    
    def _format_tools(self, schemas) -> str:
        lines = []
        for s in schemas:
            f = s["function"]
            params = f["parameters"]
            required = params.get("required", [])
            lines.append(f"- {f['name']}: {f['description']}")
            for pname, pinfo in params["properties"].items():
                req = "（必填）" if pname in required else ""
                lines.append(f"    - {pname}: {pinfo.get('description', '')}{req}")
        return "\n".join(lines)
```

### 2.4 `ActionParser`

文件：`src/parser.py`

**核心职责**：从模型自然语言输出解析 `Thought / Action / Action Input`。

```python
import json
import re

THOUGHT_RE = re.compile(r"Thought\s*\d*\s*:\s*(.+?)(?=\n\s*Action|\Z)", re.DOTALL)
ACTION_RE = re.compile(r"Action\s*\d*\s*:\s*(.+?)(?=\n\s*Action Input|\Z)", re.DOTALL)
INPUT_RE = re.compile(r"Action Input\s*\d*\s*:\s*(.+?)(?=\n\s*Observation|\Z)", re.DOTALL)


class ActionParser:
    def parse(self, text: str) -> dict:
        thought_m = THOUGHT_RE.search(text)
        action_m = ACTION_RE.search(text)
        input_m = INPUT_RE.search(text)
        
        if not thought_m:
            return {"retry": True, "reason": "no Thought"}
        if not action_m:
            return {"retry": True, "thought": thought_m.group(1).strip(),
                    "reason": "no Action"}
        
        thought = thought_m.group(1).strip()
        action = action_m.group(1).strip()
        raw_input = input_m.group(1).strip() if input_m else ""
        
        # Action Input 必须是 JSON（除非是 Final Answer）
        if action != "Final Answer":
            try:
                action_input = json.loads(raw_input)
            except json.JSONDecodeError:
                # 兜底：把 raw 当字符串，传给工具时工具内部按需处理
                action_input = {"_raw": raw_input}
        else:
            action_input = raw_input
        
        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }
```

### 2.5 `trace_recorder`

文件：`src/trace.py`（或内嵌在 agent.py）

trace 已经是 AgentTrace dict，每步追加到 `steps` 列表：

```python
# 写入 eval/result.json 时格式化
def trace_to_text(trace: dict) -> str:
    lines = []
    for s in trace.get("steps", []):
        if "thought" in s:
            lines.append(f"Thought {s.get('step_idx', '?')+1}: {s['thought']}")
        if "action" in s and s["action"] != "Final Answer":
            lines.append(f"Action {s.get('step_idx', '?')+1}: {s['action']}")
            lines.append(f"Action Input {s.get('step_idx', '?')+1}: {s['action_input']}")
            lines.append(f"Observation {s.get('step_idx', '?')+1}: {s['observation']}")
        elif "action" in s and s["action"] == "Final Answer":
            lines.append(f"Final Answer: {s['action_input']}")
    return "\n".join(lines)
```

---

## 3. 状态机设计（ReAct 循环的状态转移）

### 3.1 状态集

```python
STATE_INIT = "INIT"
STATE_THOUGHT = "THOUGHT"
STATE_ACTION = "ACTION"
STATE_OBSERVE = "OBSERVE"
STATE_RETRY = "RETRY"
STATE_FINAL = "FINAL"
STATE_TERMINATE = "TERMINATE"
```

### 3.2 转移条件

| 当前状态 | 事件 | 下一状态 |
|---|---|---|
| INIT | 接收 task | THOUGHT |
| THOUGHT | 解析成功 → action = "Final Answer" | FINAL |
| THOUGHT | 解析成功 → action 是工具名 | ACTION |
| THOUGHT | 解析失败 | RETRY |
| RETRY | 提示重试 | THOUGHT |
| ACTION | 工具调用成功 | OBSERVE |
| ACTION | 工具抛异常（已 catch 成字符串） | OBSERVE |
| OBSERVE | 拼回 prompt | THOUGHT |
| 任意 | 步数 ≥ max_steps | TERMINATE |
| 任意 | _is_stuck() = True | TERMINATE |

### 3.3 终止条件汇总

1. **Final Answer**：模型显式输出 `Action: Final Answer` + `Action Input: <答案>`。
2. **步数耗尽**：连续 `max_steps`（默认 10）次状态转移后仍无 Final Answer。
3. **连续无进展**：最近 3 步 Thought 字符串完全相同，视为卡死。

---

## 4. 错误恢复策略

### 4.1 异常分类与处理

| 异常类型 | 来源 | 处理 |
|---|---|---|
| `json.JSONDecodeError` | 解析 Action Input | 重试，要求按 JSON 格式 |
| `KeyError` | 工具内部 args 缺字段 | Observation = `[ERROR: 缺少参数 ...]` |
| `TimeoutError` | 网络超时 | Observation = `[ERROR: 超时]` |
| 网络异常 | wiki 工具 | Observation = `[ERROR: 网络不可用]` |
| `ZeroDivisionError` | calculator | Observation = `[ERROR: 除零]` |
| 其他 | 工具代码 bug | Observation = `[ERROR: <type>: <msg>]` |

### 4.2 重试策略

- **解析失败**：仅重试 1 次（追加"请严格按格式"提示），再失败则放弃本步。
- **工具失败**：让模型自己决定重试或换工具，**不在主循环强制重试**——这符合 ReAct 的"agent 自我纠错"原则。
- **步数耗尽**：返回 best-effort（最后一条 Observation 的前 300 字），success=False。

### 4.3 错误恢复示例 trace

```
Thought 1: 先算乘积。
Action 1: calculator
Action Input 1: {"expression": "(123+456*789"}  # 漏右括号
Observation 1: [ERROR: calculator 抛 SyntaxError: invalid syntax]
Thought 2: 上一步表达式有语法错，漏了右括号。修正后重试。
Action 2: calculator
Action Input 2: {"expression": "(123+456)*789"}
Observation 2: 456831
Thought 3: 456831 有 6 位。
Action 3: Final Answer
Action Input 3: 6 位
```

**注意**：这里的"自我纠错"完全靠 prompt 引导——我们不写"重试逻辑"，只保证 Observation 包含足够信息。

---

## 5. Prompt 模板设计

### 5.1 System Prompt 结构

```
[角色设定] 你是工具调用助手...
[工具列表] 4 个工具，每个含 description + parameters + 必填字段
[格式约束] Thought / Action / Action Input / Observation
[行为约束] 不要解释 Thought 之外、Action Input 必须是 JSON、错误处理、Final Answer 用法
[few-shot] （可选放 system 末尾或作 user/assistant 对话）
```

### 5.2 Few-shot 选择建议

- **数量**：2-3 个示例，太多会挤掉工具列表。
- **覆盖**：必须包含单工具（calculator）、多工具（wiki→calculator）、简单查询（file_search）三类。
- **格式**：作为 `user/assistant` 对话历史加入 messages，比 system 内的字符串更符合 chat 模型训练分布。

### 5.3 输出格式约束

- **明确"严格"二字**：7B 模型对模糊指令不可靠。
- **指定分隔符**：用 `Thought:` / `Action:` / `Action Input:` / `Observation:` 而非空格分隔。
- **Final Answer 是特殊 Action**：工具列表里把 `"Final Answer"` 也作为"伪工具"加入 prompt，告诉模型它不是真工具但格式一样。

### 5.4 Prompt 长度控制

- system prompt：约 500-800 token
- few-shot：3 个示例约 600 token
- 每步 Observation：限制 500 token（wiki summary 已经截断）
- 总 prompt 上限：4096 token（Qwen2.5-7B-Instruct 推荐上下文 8k，留余量给 response）

---

## 6. 与 SGLang / 其他推理后端的集成

### 6.1 SGLang 简述

SGLang 是另一种高性能 LLM 服务框架，提供 OpenAI 兼容 API。本任务**默认用 Ollama**，但 SGLang 可作为加速替代。

### 6.2 切换方式

只需修改两个环境变量：

```bash
export OPENAI_BASE_URL="http://localhost:30000/v1"  # SGLang 默认端口
export OPENAI_API_KEY="EMPTY"
export OPENAI_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

客户端代码（`openai` SDK）一行不用改。

### 6.3 可能需要的调整

- **Prompt 格式**：SGLang 可能对 system prompt 的处理略有差异，跑不通时把 system 拆成 user/assistant。
- **Token 上限**：SGLang 默认 max_tokens 较小，agent 主循环要显式传 `max_tokens=2048`。
- **Tool 字段**：SGLang 对 OpenAI `tools` 字段的支持取决于版本，本任务走 prompt 风格所以无影响。

---

## 7. 加分项 S1-S4 的实现思路

### 7.1 S1 · Qwen-Agent 对照

```python
# ablations/s1_qwen_agent.py
from qwen_agent.agents import ReActAgent as QwenAgent
from src.tools import Calculator, PythonSandbox, FileSearch, Wiki

qwen_tools = [Calculator(), PythonSandbox(), FileSearch(), Wiki()]
agent = QwenAgent(llm={"model": "qwen2.5:7b-instruct",
                       "model_server": "http://localhost:11434/v1"},
                  tool_list=qwen_tools)

# 跑同样的 10 题，对比成功率
```

**对比报告**：在 REPORT.md 里加一节"自写 vs Qwen-Agent 成功率对照"，分析差异原因（prompt 详细程度、few-shot 选择、错误处理粒度）。

### 7.2 S2 · 不同模型尺寸

只需 `OPENAI_MODEL` 环境变量切换：

```bash
# 1.5B（轻量基线）
OPENAI_MODEL=qwen2.5:1.5b-instruct python eval/run.py

# 7B（标准）
OPENAI_MODEL=qwen2.5:7b-instruct python eval/run.py

# 14B（高质量）
OPENAI_MODEL=qwen2.5:14b-instruct python eval/run.py
```

预期：1.5B 命中率 < 50%（格式遵从差），14B 命中率 > 70%（function calling 原生支持）。

### 7.3 S3 · Prompt 模板消融

参数化 PromptBuilder：

```python
PromptBuilder(
    tool_schemas=...,
    few_shot_count=0,    # 0/1/3 个示例
    include_thought=True, # 是否在格式约束里强调 Thought
    error_handling_hint=True,  # 是否在 system 里写"工具会报错，按 Observation 改"
)
```

跑 4-6 个组合，看哪个参数对命中率影响最大。

### 7.4 S4 · 错误注入（inject_error）

```python
# ablations/s4_error_recovery.py
agent = ReActAgent()
for rate in [0.0, 0.2, 0.5, 0.8]:
    success = 0
    for task in tasks:
        if random.random() < rate:
            agent.inject_error("calculator", "[模拟失败]")
        trace = agent.run(task["task"])
        if answer_matches(trace["final_answer"], task["expected_answer_contains"]):
            success += 1
        agent = ReActAgent()  # 重置注入状态
    print(f"error_rate={rate}, success_rate={success/len(tasks)}")
```

`inject_error` 钩子已经在 `ToolRegistry` 里实现（见 2.2），本实验就是测 `M3` 在不同错误率下的鲁棒性。

---

## 8. 代码组织建议

```
task-5-tool-agent/
├── data/                       # （已存在）任务集 + 夹具
├── eval/                       # （已存在）自检脚本
├── reference/                  # 本目录，研究材料
├── src/                        # 学生实现（**留给实现 agent**）
│   ├── __init__.py
│   ├── agent.py                # ReActAgent 主类 + inject_error 钩子
│   ├── llm_client.py           # OpenAI 客户端封装
│   ├── prompt.py               # PromptBuilder + few-shot
│   ├── parser.py               # ActionParser
│   ├── trace.py                # trace 格式化
│   └── tools/
│       ├── __init__.py         # ALL_TOOLS, default_registry()
│       ├── base.py             # Tool ABC + ToolRegistry
│       ├── calculator.py       # TOOL_SCHEMA + run
│       ├── python_sandbox.py   # 受限 exec
│       ├── file_search.py      # 文件名/内容检索
│       └── wiki.py             # wikipedia-api 封装
├── ablations/                  # S1-S4 消融脚本（实现完主体后补）
│   ├── s1_qwen_agent.py
│   ├── s2_model_sizes.py
│   ├── s3_prompt_ablation.py
│   └── s4_error_recovery.py
├── test_smoke.py               # 4 个工具的快速 smoke test
└── REPORT.md                   # 最终实验报告（M4 命中率 + 观察）
```

---

## 9. README 没明确指出但实现时容易踩的坑

1. **Qwen2.5-7B 在 Ollama 上的 `tools` 字段支持**：早期版本不支持，需要走 prompt 风格（恰好契合任务"手写 ReAct"）。如果走 native tool calling，模型可能在中文任务上反而 prompt 短、不稳定。
2. **`Action Input` 是 JSON 字符串不是 dict**：模型输出的是字符串 `"{\"expression\": \"1+1\"}"`，必须 `json.loads()`，且要兜底解析失败。
3. **`Final Answer` 的 Action Input 不是 JSON**：就是纯字符串答案，不要套 `json.dumps`。
4. **Wiki summary 长度**：默认完整 summary 可能 > 2000 token，必须截断到 500-800。
5. **calculator 的 `eval` 安全**：README 已警告，但实现时容易忘——至少要禁 `__` 前缀、禁 `import`、禁 `open`。
6. **file_search 路径越界**：相对路径 `../` 能逃出工作区，必须 `Path.resolve()` 后校验。
7. **eval 脚本按 `expected_answer_contains` 校验**：`normalize_answer` 会去掉逗号和空白，但不去掉全角数字"１２"。数值题答案要给原始数字而非带格式的字符串。
8. **trace 的 `success` 字段**：eval 脚本**不信任**这个字段，按 `final_answer` 关键词判断。但 success 字段对调试有用——True 表示 agent 显式输出 Final Answer。
9. **max_steps 太低**：任务 5（wiki + calculator）可能需要 3-4 步，加上 1-2 步探索和 1-2 步纠错，**建议 max_steps=10**。
10. **Ollama 冷启动慢**：第一次 `ollama chat` 要加载模型（10-30 秒），可能超时。客户端 timeout 设 60s，且考虑 warm-up 一次。

---

## 10. 推荐的实现顺序

1. **写 `Tool` 基类 + 4 个工具**：先跑通 `eval/run.py` 的 `tools_individual` 测试。
2. **写 `PromptBuilder` + `ActionParser`**：在 REPL 里手动喂 prompt 看 LLM 输出是否符合格式。
3. **写 `ReActAgent` 最小版本**：先支持单步 Thought → Action → Final Answer。
4. **跑 10 题看基线命中率**：大概率 < 60%，但能定位是哪些题失败。
5. **调优**：加 few-shot、修解析正则、调 max_steps、加错误恢复。
6. **S4 错误注入**：复用主循环，跑 `error_recovery`。
7. **S1/S2/S3**：作为对照实验补在 REPORT 里。

---

## 附录：与 DoD 的对应关系速查

| DoD | 对应设计 | 代码位置 |
|---|---|---|
| **M1** 4 工具 + TOOL_SCHEMA + run | Tool 基类 + 4 个子类 | `src/tools/*.py` |
| **M2** ReAct 循环 + 路由 + 终止 | ReActAgent.run | `src/agent.py` |
| **M3** 异常 → Observation | ToolRegistry.call + parser retry | `src/tools/base.py` + `src/parser.py` |
| **M4** 10 题 > 60% 命中率 | 完整系统 + few-shot 调优 | `eval/result.json` |
| **S1** Qwen-Agent 对照 | ablations/s1 | `ablations/s1_qwen_agent.py` |
| **S2** 不同尺寸 | 改 OPENAI_MODEL | `ablations/s2_model_sizes.py` |
| **S3** prompt 消融 | PromptBuilder 参数化 | `ablations/s3_prompt_ablation.py` |
| **S4** inject_error | ToolRegistry.inject_error | `src/tools/base.py` + `ablations/s4` |

---

**祝实现顺利**。这份设计已经覆盖了从架构到接口、从状态机到错误恢复的全链路，按这个蓝图写代码基本不会走偏。如果有不清楚的地方，先回到 `papers/react.md` 看原论文思想，再回到 `patterns/` 看对应模式的具体落地。
