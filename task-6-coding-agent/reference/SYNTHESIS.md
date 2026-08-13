# SYNTHESIS.md —— Mini Coding Agent 架构综合设计

> 本文档是 task-6 的实现蓝图。
> 整合了：本地 Claude Code 源码学习（`reference/claude-code/`）+ 公开论文（CodeAct、SWE-bench）+ 业界成熟实现（smolagents、OpenHands、Qwen-Agent）+ 软件架构模式（分层、六边形、渐进式披露、上下文隔离、钩子）。
>
> **不要照抄 Claude Code 的 TypeScript 语法**，但**要抄它的架构决策**。

---

## 1. 整体架构（三层栈 + agentic loop）

### 1.1 层次关系（文字版）

```
┌──────────────────────────────────────────────────────────────────┐
│                        Main Agent Loop                            │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  CodingAgent.run(repo_path, issue) -> Trace              │    │
│  │  - while not done:                                       │    │
│  │      messages, tools, skills = state                     │    │
│  │      response = llm.chat(messages, tools)                │    │
│  │      if response.tool_calls: execute + record            │    │
│  │      else: extract patch → done                          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  顶层：Subagent 派发（可选）                                       │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  CodeSearchSubagent  (tools: read_file + grep, 5 步)     │    │
│  │  TestRunnerSubagent  (tools: run_tests + read_file, 3 步)│    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  中层：Skill 加载器（可选）                                         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  SkillLoader(skills_dir)                                 │    │
│  │    .list_skills() → [{name, description}]                │    │
│  │    .load(name) → full markdown body                     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  底层：MCP Server + Tools                                          │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  MCPStdioServer(list_tools, register read_file/write_..) │    │
│  │    read_file(path) → str                                │    │
│  │    write_file(path, content) → ok                       │    │
│  │    run_tests(cwd) → pytest_output                       │    │
│  │    git_diff(repo) → unified_diff                        │    │
│  │    git_apply(repo, diff) → ok/err                       │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼ JSON-RPC over stdio / direct call
┌──────────────────────────────────────────────────────────────────┐
│  Local Execution Layer (subprocess + cwd + timeout + path check)  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流（一次 issue 完整解决过程）

```
1. 评测脚本 → CodingAgent.run(repo_path, issue)
2. state = AgentState(
     messages=[system_prompt + issue],
     turn_count=0,
     trace={"steps": [], "patch": "", "tests_passed": False}
   )
3. while turn_count < max_turns and not done:
   3.1 加载 system prompt（注入工具列表 + skill 列表）
   3.2 response = openai_client.chat(messages, tools=tool_schemas)
   3.3 if response.tool_calls:
         for call in response.tool_calls:
           # PreToolUse hook 拦截
           obs = execute_tool(call, repo_path)
           # PostToolUse hook 审计
           trace["steps"].append({thought, tool_call, observation})
           messages.append(tool_result_msg(call.id, obs))
         turn_count += 1
         # 可选：context compaction
       else:
         # LLM 没调工具 → 提取 patch → done
         trace["patch"] = extract_patch_from_response(content)
         break
4. trace["tests_passed"] = run_pytest(repo_path).returncode == 0
5. return trace
```

### 1.3 职责边界（来自 Claude Code `packages/builtin-tools/` + `src/coordinator/`）

| 层 | 职责 | 不应做 |
|---|---|---|
| **Tools** | 原子操作：读、写、跑测试、git diff/apply | 不含工作流；不含状态 |
| **Skills** | 组织化工作流（prompt 模板 + scripts） | 不调工具（被 agent 调） |
| **Subagents** | 独立子任务（独立 messages / 步数 / 工具子集） | 调 write_file 改主项目代码 |
| **Main Agent** | 编排：决定调哪个 tool / skill / subagent | 不直接写实现逻辑 |

---

## 2. 核心组件设计

### 2.1 MCP Server（对应 M1 + Claude Code `packages/mcp-client/`）

**协议选择**：stdio（最简）。HTTP/SSE 等后续扩展。

**暴露的工具集（≥ 5 个）**：
```python
# src/mcp_server.py
@mcp.tool()
def read_file(path: str) -> str: ...        # 1

@mcp.tool()
def write_file(path: str, content: str) -> str: ...  # 2

@mcp.tool()
def str_replace_editor(path: str, old_text: str, new_text: str) -> str: ...  # 3（学习 OpenHands）

@mcp.tool()
def run_tests(cwd: str, timeout: int = 60) -> str: ...  # 4

@mcp.tool()
def git_diff(repo: str) -> str: ...          # 5

@mcp.tool()
def git_apply(repo: str, diff: str) -> str: ...  # 6
```

**安全措施（README 已强调）**：
- **路径 resolve + 边界校验**：每个 file 工具 `(REPO_ROOT / path).resolve().is_relative_to(REPO_ROOT)`
- **subprocess list 形式**：`subprocess.run([sys.executable, "-m", "pytest"], cwd=repo, timeout=60, capture_output=True)`，**不要 `shell=True`**
- **git 危险命令白名单**：禁用 `git reset --hard` / `git clean -fd` / `git checkout -- <file>`
- **不打印到 stdout**：log 走 stderr；MCP server 不能污染 JSON-RPC 流

**`list_tools()` 导出方式**：
```python
# 模块顶层维护静态 dict（自检只用 import）
def list_tools() -> list[dict]:
    return [
        {"name": "read_file", "description": "...", "inputSchema": {...}},
        ...
    ]

# 装饰器注册的版本用于实战 stdio server
@mcp.tool()
def read_file(path: str) -> str: ...
```

### 2.2 Skill 加载器（对应 M2 + Claude Code `src/skills/loadSkillsDir.ts`）

**文件结构**（对齐 Anthropic Skills）：
```
src/skills/
├── code-review/
│   ├── SKILL.md          # 必含
│   └── references/       # 可选
├── pr-description-writer/
│   └── SKILL.md
└── test-runner/
    └── SKILL.md
```

**frontmatter 格式**：
```markdown
---
name: code-review
description: 当需要 review PR diff、检查命名规范、识别潜在 bug 时加载。适用于 review 单文件改动或给出重构建议。
when_to_use: （可选，比 description 更强的触发条件）
allowed_tools: （可选，限定该 skill 可用的工具）
---

# Code Review 工作流

## Step 1：理解改动范围
先用 `read_file` 读取待 review 的文件。
...
```

**渐进式披露（来自 Claude Code 源码）**：
- `list_skills()` 只返回 `[{"name", "description"}]` —— 约 20-50 token/skill
- `load(name)` 才读 SKILL.md 完整正文 —— 可达 5K-20K token
- agent 在 system prompt 里看到 skill 列表 + 触发条件；LLM 决定用哪个

**路由策略**（v1 简单版）：基于 description 关键词匹配；v2 可让 LLM 自评

### 2.3 Subagent（对应 README M3 + Claude Code `src/coordinator/workerAgent.ts`）

**三个独立属性（缺一不可）**：
1. **独立 message 列表**：`messages = [system_prompt_sub, user_task]`，不继承主 agent
2. **独立步数上限**：`max_steps = 5`（主 agent 给 30）
3. **工具白名单**：`allowed_tools = ["read_file", "grep"]`

**主 agent 调用 subagent 后只接收 final_summary 字符串**：
```python
def delegate_subagent(self, name: str, task: str) -> str:
    sub = self.subagents[name]
    summary = sub.run(task, self.llm, self.tool_registry)
    # ★ 只把 summary 字符串塞回主 messages；不暴露 subagent 的 tool_call 历史
    self.messages.append({"role": "tool", "content": f"[{name}] {summary}"})
    return summary
```

**Worker agent system prompt 模式**（来自 Claude Code `workerAgent.ts`）：
```
你是 code_search subagent。你的任务是搜索代码定义并返回位置。
- 使用 read_file / grep 找代码
- 不修改任何文件
- 返回简洁摘要：「calculator.add 定义在 calculator.py:2，当前实现是 return a - b」
- 不要在主 agent 上下文里暴露完整 tool 调用历史
```

### 2.4 Agentic Loop（对应 M3 + Claude Code `src/query.ts` 的 `queryLoop`）

**状态机**（来自 Claude Code `queryLoop`）：
```python
# Claude Code 用 turn_count + Terminal reason；Python 翻译版：
class TerminalReason(Enum):
    COMPLETED = "completed"        # LLM 输出无 tool_call + 显式结束
    TESTS_PASSED = "tests_passed"  # ★ 我们特有的（Claude Code 没有）
    MAX_TURNS = "max_turns"
    ERROR = "error"

@dataclass
class AgentState:
    messages: list[dict]
    turn_count: int = 0
    max_turns: int = 30
    patch: str = ""
    tests_passed: bool = False
    done_reason: Optional[TerminalReason] = None
    trace: list[dict] = field(default_factory=list)

def step(state: AgentState, llm, tools) -> AgentState:
    """对应 Claude Code queryLoop 的一次循环迭代"""
    response = llm.chat(state.messages, tools=tool_schemas)
    if not response.tool_calls:
        # 终止条件 1：LLM 输出纯文本（提交 patch）
        state.patch = extract_patch(response.content)
        state.done_reason = TerminalReason.COMPLETED
        return state
    for call in response.tool_calls:
        # 终止条件 2：subagent 自动跑测试（通过则 done）
        if call.name == "submit_patch":
            state.patch = call.args["diff"]
            state.tests_passed = True
            state.done_reason = TerminalReason.TESTS_PASSED
            return state
        # 执行工具
        obs = tools[call.name].call(call.args, REPO_ROOT)
        state.trace.append({"thought": response.thought, "tool_call": call, "observation": obs})
        state.messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})
    state.turn_count += 1
    if state.turn_count >= state.max_turns:
        state.done_reason = TerminalReason.MAX_TURNS
    return state
```

**Termination 条件优先级**（来自 Claude Code `queryLoop:1397-1814`）：
1. `submit_patch` 工具调用 → `TESTS_PASSED`（我们特有）
2. LLM 输出纯文本 + 无 tool_use → `COMPLETED`
3. `max_turns` 到达 → `MAX_TURNS`
4. 用户中断 / PreToolUse 拦截 → `ERROR`

**错误恢复策略**：
- 工具失败 → 把 error 字符串塞进 message 让 LLM 下一轮重试
- 连续 3 次失败 → 跳过该工具改方案
- LLM 进入死循环（连续 5 步没新增 insight）→ 强制结束

### 2.5 Trace 数据结构（对应 M4 + Claude Code `src/QueryEngine.ts` 的 `SDKMessage`）

```python
# 自检按 trace.get(...) 取值，所以必须是 dict
trace = {
    # 必需字段（DoD 强制）
    "steps": [
        {
            "thought": "需要先看 calculator.py 哪里写错了",
            "tool_call": {"name": "read_file", "arguments": {"path": "calculator.py"}},
            "observation": "def add(a, b):\n    return a - b",
            "duration_ms": 234,
        },
        # ...
    ],
    "patch": "diff --git a/calculator.py ...",
    "tests_passed": True,
    # 推荐字段（分析用）
    "done_reason": "tests_passed",
    "turn_count": 4,
    "duration_ms": 8500,
    "summary": "修复了 calculator.add 的 bug（return a + b）",
    "tool_call_count": 6,
    "subagent_invocations": [],
    "skill_loads": [],
    "compaction_events": 0,
}
```

### 2.6 Context Compaction（对应 M3 长任务 + Claude Code `src/services/compact/compact.ts`）

**触发条件**：message 累计 token 数 > 25000（Qwen2.5-Coder-7B 默认 32K context，留 buffer）

**简化策略**（v1 不用 LLM 摘要）：
```python
def compact_messages(messages, max_tokens=25000):
    total = sum(token_count(m) for m in messages)
    if total < max_tokens * 0.8:
        return messages
    # 保留：system + 前 2 条对话 + 最后 5 条工具结果
    keep = [messages[0]] + messages[1:3] + [{"role": "system", "content": "[compacted]"}] + messages[-5:]
    return keep
```

**Claude Code 的进阶做法**（v2 再学）：用 LLM 生成摘要，保留 skill 内容（5K 上限/skill），还原关键文件（5K 上限/文件）

---

## 3. Claude Code 源码导航（来自 `reference/claude-code/`）

> 我们不抄语法，但抄架构。下面是核心文件清单。

### 必读（第一梯队）

| 文件 | 行数 | 我们对应 | 学什么 |
|---|---|---|---|
| `CLAUDE.md` | 29KB | 总纲 | 项目设计哲学、feature flag 体系 |
| `packages/mcp-client/interfaces.ts` | 75 行 | `src/ports.py` | **依赖倒置**（DI interfaces） |
| `packages/mcp-client/manager.ts` | ~150 | `src/mcp_client.py` | McpManager 类的 API 抽象 |
| `packages/mcp-client/execution.ts` | 120+ | `src/mcp_client.py` | callMcpTool + timeout + progress |
| `packages/builtin-tools/.../FileReadTool.ts` | 1106 | `src/tools/read_file.py` | buildTool({...}) 工厂模式 |
| `src/Tool.ts` | 803 | `src/tool_context.py` | ToolUseContext（DI 容器） |
| `src/query.ts` (queryLoop, 393-2057) | 1664 | `src/agent.py` | **主循环** + Terminal 枚举 |
| `src/QueryEngine.ts` | ~1300 | `src/agent.py` | 高级编排 + 持久化 mutableMessages |
| `src/context.ts` | 189 | `src/prompt_builder.py` | System prompt 构造 |
| `src/services/compact/compact.ts` | ~1700 | `src/agent.py` 的 `_compact` | **LLM-based 摘要压缩** |

### 应该读（第二梯队）

| 文件 | 行数 | 我们对应 | 学什么 |
|---|---|---|---|
| `src/skills/loadSkillsDir.ts` | 1080 | `src/skill_loader.py` | **渐进式披露完整实现** |
| `src/coordinator/coordinatorMode.ts` | ~150 | `src/subagents/` | coordinator 模式切换 |
| `src/coordinator/workerAgent.ts` | 68 | `src/subagents/code_search.py` | **worker agent 定义模板** |
| `src/Tool.ts` (CanUseToolFn) | - | `src/agent.py` 的 hook | hook 系统实现 |
| `src/Task.ts` | 125 | `src/subagents/base.py` | TaskType / TaskStatus 枚举 |

### 推荐读（第三梯队，加分项）

| 文件 | 学什么 |
|---|---|
| `DEV-LOG.md`（52KB） | 演进历史、工程教训（如 mock.module 污染、JSC RSS 暴涨） |
| `AGENTS.md` | 与 CLAUDE.md 高度重叠 |
| `spec/` | 设计规范（按 feature 日期组织） |
| `src/coordinator/coordinatorMode.ts` | subagent 协调的高级模式 |

---

## 4. 借鉴与简化策略

### ✅ 抄什么（架构思想）

1. **依赖注入模式**：`McpClientDependencies` 接口 → Python `Protocol` 类
2. **配置 schema 化**：zod schema → Python `pydantic.BaseModel`
3. **`buildTool({...})` 工厂模式**：所有工具元数据集中声明
4. **状态机驱动 queryLoop**：`while true + return {reason}` 多出口
5. **Terminal reason 枚举**：显式记录「为什么停」
6. **Skill 渐进式披露**：list 阶段只读 frontmatter，load 才读正文
7. **Worker agent 工具白名单**：subagent 必须有独立 `allowed_tools`
8. **PreToolUse / PostToolUse hook**：安全护栏 + 审计
9. **`ToolUseContext` 容器**：所有 tool 共享同一份上下文

### ❌ 不抄什么（避免过度工程化）

1. **完整 Ink UI** —— 我们只要 CLI
2. **17 个 workspace packages** —— 单包足够
3. **7 个 API provider** —— 我们只支持 OpenAI 兼容
4. **Langfuse telemetry** —— 加分项，先不做
5. **19 个 feature flags** —— 我们 0 个
6. **ACP 协议 / Remote Control Server** —— 完全用不到
7. **Agent 状态总线（`setAppStateForTasks`）** —— 直接 return summary 就行
8. **LLM 摘要 compaction** —— v1 用简单截断；v2 再升级
9. **多 worker 调度 / Team management** —— 1-2 个 subagent 足够

---

## 5. 代码组织建议

```
task-6-coding-agent/
├── src/
│   ├── mcp_server.py            # MCP server 入口（独立可运行）
│   ├── mcp_client.py            # MCP client（与 server 通过 stdio 通信）
│   ├── skill_loader.py          # Skill 加载器（约 50-80 行）
│   ├── skills/                  # 具体 Skills
│   │   ├── code-review/SKILL.md
│   │   ├── pr-description-writer/SKILL.md
│   │   └── test-runner/SKILL.md
│   ├── subagents/               # Subagent 实现
│   │   ├── base.py              # Subagent 基类
│   │   ├── code_search.py       # 代码搜索 subagent
│   │   └── test_runner.py       # 测试执行 subagent
│   ├── agent.py                 # CodingAgent 主类（含 agentic loop）
│   ├── tools/                   # MCP server 暴露的工具
│   │   ├── base.py              # Tool 基类 + 工厂
│   │   ├── read_file.py
│   │   ├── write_file.py
│   │   ├── str_replace_editor.py
│   │   ├── run_tests.py
│   │   ├── git_diff.py
│   │   └── git_apply.py
│   ├── hooks.py                 # Hook 系统（PreToolUse / PostToolUse）
│   ├── trace.py                 # Trace 数据结构
│   ├── ports.py                 # 端口协议（Protocol 类）
│   ├── prompt_builder.py        # System prompt 构造
│   └── llm_client.py            # OpenAI 兼容客户端封装
├── ablations/                   # S1-S4 消融脚本
│   ├── s1_quantization.py
│   ├── s2_subagent.py
│   ├── s3_skill.py
│   └── s4_swebench.py
├── test_smoke.py                # 单元测试
├── eval/
│   ├── run.py                   # 已有自检脚本
│   ├── result.json              # 自检结果
│   └── tutor_prompt.md          # AI tutor prompt
├── data/
│   ├── download.py              # 已有数据准备脚本
│   └── toy-repo/                # 生成的目标 repo
├── reference/                   # 本研究材料库
│   ├── CLAUDE-CODE.md
│   ├── SYNTHESIS.md             # 本文件
│   ├── claude-code/             # Claude Code 源码（已 clone）
│   ├── papers/
│   ├── repos/
│   ├── patterns/
│   └── api-specs/
└── REPORT.md                    # 最终实验报告
```

**行数估算**（与 Claude Code 17000 行 TypeScript 对比）：
- `tools/` 每个 20-50 行 × 6 = ~240 行
- `mcp_server.py` + `mcp_client.py` = ~200 行
- `skill_loader.py` = ~80 行
- `agent.py` = ~200 行
- `subagents/` 每个 80 行 × 2 = ~160 行
- `hooks.py` = ~80 行
- `prompt_builder.py` = ~60 行
- 测试 + ablation = ~400 行
- **总计：约 1400 行 Python**（5-6 周工作量）

---

## 6. 关键风险与陷阱

### 6.1 MCP 协议细节陷阱

1. **stdio server 不能 print 到 stdout**：所有 log 必须 `logging.warning(...)` 走 stderr；否则污染 JSON-RPC 流
2. **路径必须双重校验**：`(REPO_ROOT / path).resolve()` 后 `.is_relative_to(REPO_ROOT)`；LLM 输出 `../../../etc/passwd` 直接挡
3. **subprocess 必须 list 形式 + timeout**：`subprocess.run([sys.executable, "-m", "pytest"], cwd=repo, timeout=60, capture_output=True)`；**不要 `shell=True`**
4. **JSON-RPC 错误用 error 对象**：工具抛异常 → SDK 包装为 `{"code": -32603, "message": str(e)}`；不要让 server 进程退出
5. **`list_tools()` 与 SDK 分离**：自检只 `from src.mcp_server import list_tools`；不要让 import 触发 SDK 启动

### 6.2 SWE-bench 评测陷阱

1. **每个 instance 必须 `git checkout base_commit`**：题面是 base_commit 上的代码；在 HEAD 上跑会错位
2. **不能改 `test_patch`**：prompt 要明示「禁止修改 tests/ 目录」
3. **必须跑全量测试**：`python -m pytest -q`（不限单个）；`PASS_TO_PASS` 失败也判失败（防回归）
4. **超时设置**：SWE-bench 一题可能 5-10 分钟；run_tests 工具 timeout 默认 60 太短，SWE-bench 跑时调到 300+
5. **失败 patch 不污染 repo**：如果 `git apply` 失败，agent 应该回滚重试；不能让目录污染状态
6. **评测沙箱**：toy repo 直接跑；SWE-bench 真跑用 Docker 或 venv 隔离，避免污染主机 Python 环境

### 6.3 Agent Loop 陷阱

1. **停机条件必须明确**：「测试通过 / 模型显式完成 / max_turns」三选一；不能「步数到了硬停」
2. **Trace 必须是 dict**（自检按 `trace.get(...)` 取值）；可以是 dict 子类；不能是 dataclass
3. **Subagent 必须真隔离**：subagent 用独立 `messages = []`，返回字符串；不要把 trace 列表塞回主 context
4. **Subagent 工具白名单**：code_search 类 subagent 绝不能有 `write_file`（破坏隔离）
5. **长任务 context 爆**：Qwen2.5-Coder-7B 32K context；做简单截断 compaction 即可
6. **错误恢复而非放弃**：tool 失败时把 error 字符串塞进 message，让 LLM 下一轮重试；连续失败才改方案

### 6.4 模型适配陷阱

1. **Qwen2.5-Coder-7B 用 code-style 工具调用**：在 system prompt 里以「Python 函数签名」格式列出工具，比纯 JSON schema 准确率高
2. **本地端点是 OpenAI 兼容**：`openai.OpenAI(base_url="http://localhost:11434/v1", api_key="EMPTY")`；Ollama / vLLM / llama.cpp 都一样
3. **思维链可关**：本地 7B 模型不需要 thinking mode；如开可能消耗大量 token 而无收益
4. **温度设 0 或 0.1**：agent 任务要稳定输出，不要随机性
5. **max_tokens 设大**：工具调用可能生成上千字（含 file content / patch）；默认 2048 可能不够

### 6.5 README 没明确指出的陷阱（来自 Claude Code 源码阅读）

1. **Mock 模块是进程全局**（Claude Code `tests/mocks/log.ts` 注释）：测试时 mock 文件系统会影响其他测试；避免用 `mock.patch` 直接替换工具函数
2. **状态机不要共享可变状态**（Claude Code `queryLoop` 的 `state = {...}` 模式）：每轮用 dict 整体替换 state 字段；避免在循环内 mutate
3. **Subagent 的 system prompt 要明确「返回摘要」**（来自 `workerAgent.ts`）："Report back with concise summary — the coordinator will synthesize your results"
4. **Tool 路径校验放在 tool 内部**（而非 CodingAgent 主体）：保证 tool 可独立单测；CodingAgent 不必关心沙箱
5. **Skill 加载器不能预加载所有正文**（来自 `loadSkillsDir.ts:78-105`）：estimateSkillFrontmatterTokens 只算 frontmatter 开销；正文按需
6. **Worker agent 排除 internal orchestration tools**（来自 `workerAgent.ts:24-39`）：subagent 不能调「派 subagent」的工具；防止失控

---

## 7. 实施路径建议（5-6 周）

### Week 1：环境 + 数据 + 模型
- `pip install -r requirements.txt`
- `python data/download.py`（生成 toy-repo）
- 启动本地 Qwen2.5-Coder-7B（Ollama 最简单：7B 8GB 显存；Q4_K_M 量化版 4-5GB）
- 验证 `openai.OpenAI(base_url="http://localhost:11434/v1").chat.completions.create(...)` 能拿到响应

### Week 2：MCP server（M1）
- 用 FastMCP 实现 5-6 个 tool
- 路径 resolve + is_relative_to 校验
- subprocess list 形式 + timeout
- 模块顶层导出 `list_tools()` 供自检
- 通过 `mcp_server_lists_tools` 测试

### Week 3：Skill 加载器 + Skills（M2）
- 实现 `SkillLoader`（约 50-80 行）
- 写 2-3 个 SKILL.md（带 frontmatter）
- description 写清楚「何时加载」
- 通过 `skill_loader_metadata` 测试

### Week 4：主 agent loop（M3 + M4）
- 实现 `CodingAgent.run` 主循环
- 接入 MCP client（或直接调 tool 函数）
- 实现 submit_patch 终止信号
- 在 toy-repo 上跑通：`add` 改为 `a + b`
- 通过 `toy_repo_patch` 测试

### Week 5：Subagent + 优化
- 实现 code_search subagent + test_runner subagent
- 实现 context compaction（简单截断版）
- 加 hook 系统（可选）
- 通过 S2 ablation 对照实验

### Week 6：消融实验 + 报告
- S1（Q4 vs FP16）、S2（Subagent）、S3（Skill）对照
- S4（SWE-bench Lite）跑 1 题（可选）
- 写 `REPORT.md`（200-500 字实验观察）
- 提交到 nndl-discussion

---

## 8. 与 SGLANG 的集成（如后续换推理后端）

SGLANG 是高性能推理后端（https://github.com/sgl-project/sglang），提供 OpenAI 兼容 API。

**集成方式**（几乎零改动）：
```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-Coder-7B-Instruct --port 30000
```

```python
# src/llm_client.py
self.client = openai.OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
# 其他代码不变
```

**注意事项**：
- SGLANG 默认支持 OpenAI function calling，与 MCP tool schema 兼容
- 流式响应（`stream=True`）需开启；Claude Code 用 streaming，我们的 v1 不必
- Batch 推理（SGLANG 原生）可加速，但 agent loop 不友好（要等所有请求结束）

---

## 9. 自检与调试建议

### 单元测试（不必复杂）
```python
# test_smoke.py
def test_skill_loader():
    loader = SkillLoader("src/skills")
    skills = loader.list_skills()
    assert len(skills) >= 2
    assert all("name" in s and "description" in s for s in skills)

def test_tool_path_safety():
    tool = ReadFileTool(REPO_ROOT)
    with pytest.raises(PermissionError):
        tool.call({"path": "../../../etc/passwd"})
```

### 调试 trace
```python
# 跑完后 trace 自动写到 debug_trace.json
import json
trace = agent.run(repo_path, issue)
with open("debug_trace.json", "w") as f:
    json.dump(trace, f, indent=2, default=str)
```

### 失败模式对照
| 现象 | 可能原因 |
|---|---|
| `mcp_server_lists_tools` 失败 | `list_tools()` 没在模块顶层导出，或工具数 < 5 |
| `skill_loader_metadata` 失败 | frontmatter 缺 `name` 或 `description` |
| `toy_repo_patch` 失败 | agent 没改对文件；或 pytest 路径不对；或 trace 缺 `patch` |
| 死循环 | 没明确 stop 信号；或 max_turns 太大 |
| Token 爆 | 没做 compaction；或工具返回太多内容 |

---

## 10. 总结：3-5 条核心设计建议

1. **三层栈单向依赖**：Tools（无状态）→ Skills（frontmatter + markdown）→ Subagents（独立 context）；每层只暴露 list + execute 两个动词
2. **Skill 渐进式披露**：list 只读 frontmatter，load 才读正文；description 写「何时加载」而非「是什么」
3. **Subagent 三个独立属性**：独立 messages、独立 max_steps、工具白名单；返回字符串摘要而非 trace
4. **Agentic loop 状态机**：while 循环 + 多个 return reason（completed / tests_passed / max_turns / error）；Trace 必须含 steps / patch / tests_passed
5. **安全护栏优先**：路径 resolve + is_relative_to、subprocess list 形式、timeout、git 危险命令黑名单——这些是必做项，不是可选

---

## 附录：术语对照表

| 中文 | 英文 | 备注 |
|---|---|---|
| 能力三层栈 | Capability Stack | Tools / Skills / Subagents |
| 渐进式披露 | Progressive Disclosure | Anthropic Skills 核心概念 |
| 上下文隔离 | Context Isolation | subagent 独立 message 列表 |
| 端口-适配器 | Ports & Adapters | 六边形架构 |
| 钩子 | Hook | PreToolUse / PostToolUse |
| 终止原因 | Terminal Reason | Claude Code 枚举（completed/max_turns/aborted/...） |
| 工具调用上下文 | ToolUseContext | Claude Code 共享 DI 容器 |
| 工作节点 | Worker Agent | subagent 在 Claude Code 的术语 |
| 协调器模式 | Coordinator Mode | Claude Code 多 agent 编排 |
| 压缩 | Compaction | 长任务 context 自动摘要 |
| 滑动窗口 | Sliding Window | context 截断策略 |

---

> **下一步**：把这个 SYNTHESIS.md 当作实现蓝图，开始 Week 2（MCP server）。M1 通过后再写 M2（Skill loader），最后做 M3（agent loop）。
>
> Claude Code 源码已经 clone 在 `reference/claude-code/`，随时可查 `src/query.ts`（主循环）+ `src/skills/loadSkillsDir.ts`（Skill）+ `src/coordinator/workerAgent.ts`（subagent）作为参考。