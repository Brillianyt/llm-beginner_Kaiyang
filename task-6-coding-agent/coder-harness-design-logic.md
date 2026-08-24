# Coder Harness — 设计逻辑 & 工程逻辑

> 任务来源：你给的课程/项目 spec
> 这份文件是 **设计逻辑 + 工程逻辑** 的提取文档，不写代码实现。
> 配套前一份：`coder-harness-file-system-spec.md`（文件系统工具集的 prompt + schema）

---

## 0. 整体架构（设计逻辑）

```
┌────────────────────────────────────────────────────────────────┐
│                         CodingAgent.run(repo, issue)            │
│                                                                │
│  while not done:                                               │
│      ┌─────────────────┐                                        │
│      │  Model (LLM)   │ ← system prompt (含 Skill 索引)         │
│      └────────┬────────┘                                        │
│               │  tool_use                                      │
│    ┌──────────┼──────────────────────┐                          │
│    │          │                      │                          │
│    ▼          ▼                      ▼                          │
│ ToolCall  → Tool    Skill load  → Skill (progressive)           │
│            │                              │                    │
│            ▼                              ▼                    │
│         observation                expanded instructions        │
│            │                              │                    │
│            └──────────────┬───────────────┘                    │
│                           ▼                                    │
│                  Trace.append(step)                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                      ▲              ▲
                      │ summary only │
                      │              │
              ┌───────┴──────┐  ┌────┴────────┐
              │ Subagent(任务)│  │ Subagent(搜索)│
              │ 独立 context  │  │ 独立 context  │
              └───────────────┘  └──────────────┘
```

**设计要点（why）：**

1. **主 agent 只持有工具调用轨迹，subagent 持有完整对话**——子 agent 是"廉价的搜索引擎/执行器"，主 agent 只吞摘要，避免 trace 爆炸。
2. **Skill 是 lazy 加载的 prompt**——index（小）+ body（大）分两步塞入，把上下文留给真正在用的内容。
3. **MCP 是工具层接口协议**——所有副作用操作都过 MCP，model 看到一致的 `tool_use` 形状，host 替换 transport 很便宜（stdio ↔ HTTP ↔ SSE）。

---

# Part I — MCP 工具集（5 个）

## 1.1 `read_file`

### 设计逻辑（why）

| 决定 | 原因 |
|---|---|
| 路径必须是绝对路径 | agent 在多 cwd 下不歧义；权限规则的字符串匹配可以精确 |
| 自动用 line-number 返回 | 给模型稳定锚点，方便后续 Edit 定位 |
| 大文件分页（offset/limit） | 避免一次 read 撑爆 ctx |
| 支持图片内联渲染 | 多模态模型通用，把"看截图"统一到 read_file |
| 必须确认文件存在 | "读不存在的文件"被显式错误反馈，让模型放弃这条路径 |

### 工程逻辑（how）

```python
@mcp.tool()
def read_file(file_path: str, offset: int | None = None,
              limit: int | None = None) -> dict:
    """
    返回 shape：
    {
      "file_path": "<abs>",
      "content": "<str>",
      "num_lines": N,
      "start_line": 0,
      "total_lines": M,
      "encoding": "utf-8",
      "truncated": bool
    }
    """
```

实现层要做的事：

1. **路径规范化** — `os.path.abspath` + `~` 展开；UNC 路径短路直接拒绝（来自 Claude Code 的安全教训）。
2. **大小前置校验** — 拿到 `stat().st_size`，超过阈值（如 256KB）直接抛错，告知"用 offset/limit 分页"。
3. **逐行读取 + 切片** — `open(..., encoding="utf-8")`，`f.readlines()[offset:offset+limit]`。
4. **cat -n 格式输出** — `f"{i+1}\t{line.rstrip()}"`。
5. **找不到文件** — 抛清晰异常，错误信息包含"绝对路径是什么 + 工作目录是什么"。

---

## 1.2 `write_file`

### 设计逻辑（why）

| 决定 | 原因 |
|---|---|
| 全量覆盖（不增量） | 跟 `Edit` 互补；写新文件或大重构时一步到位 |
| 必须先 `read_file` 过 | 防止 agent 盲目覆盖未知状态（保护现有内容） |
| 拒绝把目录当路径 | `write_file(path="src/")` 误删一整个目录，常见错误 |
| 覆盖前先备份（可选） | `*.bak`，调试时方便回滚 |

### 工程逻辑（how）

```python
@mcp.tool()
def write_file(file_path: str, content: str) -> dict:
    """
    返回 shape：
    {
      "file_path": "<abs>",
      "type": "create" | "update",
      "bytes_written": N,
      "diff": "<unified diff string>",
      "git_diff": {...} | None
    }
    """
```

实现要点：

1. **`read_file` 前置检查**（host 侧记录每个 path 是否被读过；不在 read cache 的路径拒写）。
2. **路径必须是文件** — `Path(file_path).suffix != ""`；存在且是目录就拒绝。
3. **原子写入** — 先写 `tmp = f"{file_path}.{uuid}.tmp"`，再 `os.replace(tmp, file_path)`，避免半写状态。
4. **目录自动建** — 父目录不存在就 `mkdir(parents=True)`，但**只允许一级**，深度嵌套要报错让用户确认。
5. **返回 unified diff** — `difflib.unified_diff(old, new, fromfile, tofile)`，方便 host UI 渲染给用户看。

---

## 1.3 `run_tests`

### 设计逻辑（why）

| 决定 | 原因 |
|---|---|
| 由 harness 决定如何运行，不写死 pytest/unittest | 不同语言/框架的 agent 都能复用 |
| 返回结构化的失败信息 | agent 需要知道"哪条测试失败/为什么失败"，不是 stdout 大段 |
| 单次 timeout + 显式重试 | 区分"测试不稳定"和"测试真实失败" |
| 失败时自动截断 traceback | 防止一个测试失败把 50KB 输出塞进 context |

### 工程逻辑（how）

```python
@mcp.tool()
def run_tests(cmd: str = "python -m pytest -x --tb=short",
              cwd: str | None = None,
              timeout_s: int = 300) -> dict:
    """
    返回 shape：
    {
      "exit_code": int,
      "passed": int,
      "failed": int,
      "errors": int,
      "duration_s": float,
      "stdout_tail": "<last 50 lines>",
      "stderr_tail": "<last 50 lines>",
      "failures": [
        {"file": "...", "line": N, "test": "...", "msg": "..."}
      ],
      "truncated": bool
    }
    """
```

实现层：

1. **subprocess + timeout** — `subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, timeout=timeout_s)`。
2. **pytest 输出解析** — 加 `-v --tb=line --no-header`，用 `pytest-json-report` 或自写正则拿到 `passed/failed` 统计。
3. **失败用例提取** — 解析 `FAILED test_xxx::test_yyy - AssertionError: ...` 形态，转成结构化 `failures[]`。
4. **stdout/stderr 双向截断** — 只保留 tail（N 行）；host 把它整个塞到 tool_result。
5. **异常到结构的映射** — `TimeoutExpired` → `exit_code = -1, errors = -1`；`FileNotFoundError` → `cmd_not_found` 错误信息。

---

## 1.4 `git_diff`

### 设计逻辑（why）

| 决定 | 原因 |
|---|---|
| 既要看工作区未暂存，也支持 `--cached` | agent 既要 review 现状、也要看已 staged |
| 返回 patch + 文件级统计 | 模型靠 patch 理解改动，靠 stats 沟通进度 |
| 文件过滤参数 | 大仓只查自己改的部分，省 token |
| 不修改 repo 状态（不 stage、不 commit） | 与 `git_apply` 严格分工 |

### 工程逻辑（how）

```python
@mcp.tool()
def git_diff(repo_path: str,
             staged: bool = False,
             file_path: str | None = None,
             context_lines: int = 3) -> dict:
    """
    返回 shape：
    {
      "files": [
        {"path": "...", "additions": N, "deletions": M,
         "status": "modified|added|deleted",
         "patch": "<unified diff>"}
      ],
      "total_files": N,
      "truncated": bool
    }
    """
```

实现要点：

1. **subprocess git** — `git -C {repo_path} diff --unified={context_lines} [options]`。
2. **参数映射**：`staged=True` → `--cached`；`file_path` → 追加路径过滤。
3. **逐文件切 patch** — 用正则 `/^diff --git a\/(\S+) b\/(\S+)/m` 切分，存到 `files[]`。
4. **保护 `.env` / credentials** — 任何匹配 `*.env / *credentials* / *secret*` 的文件一律 redact（不让 patch 进 context）。
5. **大输出保护** — 单文件 patch 超 100KB 就截断 + `truncated=True`。

---

## 1.5 `git_apply`

### 设计逻辑（why）

| 决定 | 原因 |
|---|---|
| 接受 unified diff patch 字符串（不是文件路径） | agent 自己生成 patch，闭环；host 不用管中间文件 |
| 默认 dry-run | 校验兼容性，让 agent 自检再 commit |
| conflict 时返回错误，不强行覆盖 | patch 没匹配上就别写 |
| 写完自动暂存（可选） | `apply_and_stage=True` 让 agent 一次两步走 |

### 工程逻辑（how）

```python
@mcp.tool()
def git_apply(repo_path: str,
              patch: str,
              dry_run: bool = True,
              three_way: bool = False) -> dict:
    """
    返回 shape：
    {
      "applied": bool,
      "files_touched": [...],
      "fuzzy": bool,
      "conflicts": [{"path": "...", "hunk": N}],
      "error": str | None
    }
    """
```

实现层：

1. **写临时 patch** — `tempfile.NamedTemporaryFile(suffix=".patch")`，写入 `patch`。
2. **git apply 调用** — `git -C {repo_path} apply --check {patch_file}` 预检；再 `apply [--3way]` 真正应用。
3. **dry_run vs 真应用** — `dry_run=True` 只跑 `--check`，返回"会成功/会失败"；`dry_run=False` 才真写。
4. **错误捕获** — `subprocess.CalledProcessError` 的 stderr 经常含 `error: patch failed: ...` + `hunk @@ -N,M +...`，正则解析回填到 `conflicts[]`。
5. **不写就回退** — 任何错误 patch 文件已 `.unlink()`，repo 状态保持原样。

---

# Part II — SkillLoader（Progressive Disclosure）

## 2.1 整体设计

> "Skills are folders of instructions Claude can discover and load on demand."

```
.claude/skills/                   # repo 级
~/.claude/skills/                 # user 级
src/skills/                       # harness 项目内

每个 skill 一个文件夹，内含 SKILL.md（必备） + 可选 scripts/、resources/
```

**核心思想（why）：**

- **Index（小）和 Body（大）分开**——启动时只塞 SKILL.md 的 `name` + `description`（每个 ≤1024 字符）作为索引；模型点击后才把全文拉进 ctx。
- **不用 model embedding，纯关键词匹配足够**——TF-IDF / 简单 token overlap；返回 top-k。
- **Skill 与 Tool 不同**——Skill 改的是"模型怎么做"，Tool 改的是"模型能做什么"。
- **对齐 Anthropic 官方 Skills 约定**——YAML front-matter + Markdown 正文，生态可复用。

## 2.2 SKILL.md 规范

```markdown
---
name: code-review                       # kebab-case；唯一
description: "Review a code change for quality, correctness, and style. Use when the user asks for review, or before opening a PR."  # ≤1024 字符
---

# Title

## When to use this skill
- Trigger phrases / contexts

## Steps
1. ...
2. ...

## Output format
- What the agent should produce

## Examples
- Minimal worked example
```

front-matter 字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | ✅ | kebab-case，目录名匹配 |
| `description` | ✅ | 单行陈述：做什么 + 何时触发。决定是否被匹配 |
| `version` | ⛏ | 可选；harness 自己用，不进 ctx |
| `allowed-tools` | ⛏ | 可选；声明 skill 启用期内能用哪些 tool |

## 2.3 `SkillLoader` 工程逻辑

```python
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self._index: list[SkillIndex] = []      # 启动时构建，存 name+description
        self._bodies: dict[str, str] = {}       # 懒加载，存全文

    def scan(self) -> None:
        """扫描 skills_dir/*/SKILL.md，解析 front-matter，建索引"""
        for skill_md in self.skills_dir.glob("*/SKILL.md"):
            meta, body = parse_frontmatter(skill_md.read_text())
            self._index.append(SkillIndex(
                name=meta["name"],
                description=meta["description"],
                path=skill_md,
            ))
            self._bodies[meta["name"]] = body

    def search(self, query: str, k: int = 3) -> list[SkillIndex]:
        """对 query vs description 做 token overlap / TF-IDF，返回 top-k"""
        return score_and_rank(self._index, query)[:k]

    def load(self, name: str) -> str:
        """命中后注入：把全文塞给 agent"""
        return self._bodies[name]

    def system_prompt_section(self, char_budget: int = 8000) -> str:
        """生成系统提示词里的 '## Available skills' 段落"""
        lines = ["## Available skills\n",
                 "Use one when its description matches the task. "
                 "Loading a skill expands its full instructions into context.\n"]
        used = 0
        for s in self._index:
            entry = f"- `{s.name}` — {s.description}\n"
            if used + len(entry) > char_budget:
                break
            lines.append(entry)
            used += len(entry)
        return "".join(lines)
```

**关键工程细节（来自 Claude Code 实现）：**

| 设计点 | 取自 |
|---|---|
| Index 阶段占 ctx 预算的 1%（≈8K 字符/200K ctx 窗口） | `SKILL_BUDGET_CONTEXT_PERCENT = 0.01` |
| 单条 description 上限 1536 字符（v2.1.117 调高） | `MAX_LISTING_DESC_CHARS = 1536` |
| `getCharBudget(ctx_window_tokens)` 接受上下文窗口大小动态算预算 | 200K 模型和 8K 模型的 index 大小不同 |
| Search 第二阶段用 `DiscoverSkills` 工具做（不依赖被动匹配） | `DiscoverSkillsTool` |
| Skills 可"被教"——会话内生成新 skill（`/skillify`） | `src/skills/bundled/skillify.ts` |

## 2.4 3 个示例 Skill

### A. `code-review/SKILL.md`

```markdown
---
name: code-review
description: "Perform a structured code review on a diff. Use when the user asks 'review my changes', before opening a PR, or after running tests."
---

# Code Review

## Inputs you receive
- `repo_path`: absolute path to repo
- `diff`: unified diff (from `git_diff`)
- optional `focus`: e.g. correctness | performance | security | style

## Steps
1. Read each hunk and the surrounding context (≥10 lines each side).
2. For each issue, classify:
   - **must-fix**: bug, security, data loss, broken test
   - **should-fix**: clear style/idiom violation with cited reason
   - **nit**: subjective; mention only if asked
3. Produce a review in this exact shape:
   - **Summary** (1-2 sentences)
   - **Findings** (bullets: severity, file:line, why, suggested fix)
   - **Praise** (1-3 bullets, brief)
4. If no issues: say "No issues found" + 1 sentence why.

## Output format
Markdown only, no preamble. Maximum 30 findings per call.
```

### B. `pr-description-writer/SKILL.md`

```markdown
---
name: pr-description-writer
description: "Generate a PR title + body from a diff and recent commits. Use when opening a PR or when the user asks for a changelog / PR description."
---

# PR Description Writer

## Inputs
- `repo_path`
- `base_branch` (default: `main`)
- optional `commit_range`: e.g. `HEAD~3..HEAD`

## Steps
1. `git log {base_branch}..HEAD --oneline` — recent style/format
2. `git diff {base_branch}...HEAD` — full diff
3. Write a PR title (≤70 chars, imperative mood, no period).
4. Write the body in this template:
```

````markdown
## Summary
- 1-3 bullets explaining the *what* and *why*

## Test plan
- [ ] Unit tests added/updated
- [ ] `pytest` passes locally
- [ ] Manual verification step (if UI)

## Risk
- Rollback plan (revert commit / revert PR)
- Affected modules
````

### C. `test-runner/SKILL.md`

```markdown
---
name: test-runner
description: "Diagnose and fix failing tests. Use when `run_tests` returns failures, or the user reports a failing test."
---

# Test Runner & Diagnostic

## Workflow

### 1. Parse the failure
- Identify failing file/line from `failures[]`
- Read the source under test AND the test itself (≥20 lines context)

### 2. Classify the failure
| Class | Symptom | Fix path |
|---|---|---|
| implementation_bug | test correct, source wrong | edit source |
| test_bug | test wrong assertion / setup | fix test (only if user permits) |
| flaky | first run fail, second pass | re-run 1–2 more times before touching code |
| environment | missing dep / wrong cwd | ask user, don't `pip install` blindly |

### 3. Fix loop
```
while not done:
    read source under test (read_file)
    generate patch (in your head)
    apply patch  (write_file or Edit equivalent)
    run tests   (run_tests)
    if still failing:
        look at new failures, iterate
        if stuck after N turns → call user
```

### 4. Confirm
- Run the full test suite one more time after a green run.
- Output: `✅ <N> tests passed, 0 failed`

## Hard rules
- Do NOT modify tests to make them pass unless the user explicitly approves.
- Stop iterating after 8 turns —summarize state and call user for help.
```

---

# Part III — Subagent 模式

## 3.1 设计逻辑（why）

**问题：** 主 agent 直跑 search/test，全 trace 进 context 会爆：
- `Grep` 输出 200 个文件路径 + 内容 = 50K tokens
- 一个 test 失败 traceback 30K tokens × 5 次迭代

**核心解法：** **隔离 context**。

| 角色 | ctx 内容 | 主 agent 看到 |
|---|---|---|
| Search subagent | 全量搜索结果 + 中间推理 | 仅最终摘要（≤2KB） |
| Test-exec subagent | 完整 pytest output + 多次重试 | 仅结构化失败 + 修复建议 |

**Claude Code 的实现原则（直接借鉴）：**

- **Read-only by default**：`exploreAgent` 启动 prompt 第一句就是 `READ-ONLY MODE - NO FILE MODIFICATIONS`。子 agent 不被允许修改文件。
- **明确工具白名单**：子 agent 只能用 `Glob / Grep / Read / Bash(read-only)`，不能 Write/Edit。
- **强制并行**：子 agent 鼓励同时发起多次搜索（`spawn multiple parallel tool calls`）。
- **Final report 直接 message**：子 agent 不写文件，把总结以 assistant message 返回。

## 3.2 接口工程逻辑

```python
# src/subagents/__init__.py
from dataclasses import dataclass

@dataclass
class SubagentTask:
    name: str          # "search-executor"
    goal: str          # 自然语言目标
    context: dict      # 注入的初始数据
    readonly: bool = True

@dataclass
class SubagentResult:
    summary: str       # 强制 ≤2KB 的精炼结论
    artifacts: list    # 可选：子 agent 产出的文件/数据
    turns_used: int
    duration_s: float

async def dispatch(task: SubagentTask) -> SubagentResult:
    """开独立 context 跑子 agent，主 agent 阻塞等待，只拿到 summary。"""
    ...
```

要点：

1. **独立 message history**——子 agent 拿到 `[system: prompt]` + `[user: goal]` + ... 自己闭环，不污染主对话。
2. **Forced truncation**——`summary` 长度硬上限（如 2048 tokens），超过就 LLM 二次精炼。
3. **工具子集**——主 agent 看到的 `tool_registry` ≠ 子 agent 的；按 `name` 注入。
4. **超时 + 重试上限**——子 agent 单次 5 分钟、最多 8 个 turn。

## 3.3 两个示例 subagent

### A. `search-executor`

```
system_prompt:
  You are a code search specialist. READ-ONLY.
  Tools allowed: read_file, glob, grep.
  Never create, edit, or delete files. Never run Bash that modifies state.

input: { "query": "find all places calculator.add is called",
          "repo_path": "..." }

output (≤2KB summary):
  - Found 3 call sites:
    1. src/calculator/__init__.py:12 — public re-export
    2. tests/test_calculator.py:45,67 — 2 unit tests
    3. scripts/demo.py:8 — sample code
  - No other references. Definition: src/calculator/core.py:3
```

### B. `test-executor`

```
system_prompt:
  You are a test execution specialist. READ-ONLY on source.
  Tools allowed: read_file, run_tests, git_diff.
  Never modify source code. You may fix test files ONLY if user permits.

input: { "failed_tests": [...],
          "repo_path": "...",
          "user_permits_edit_tests": false }

output (≤2KB summary):
  - Root cause: calculator.add returns a-b in src/calculator/core.py:8
  - Reproduction: tests/test_add.py::test_two_plus_two expects 4, got 0
  - Suggested patch location: src/calculator/core.py:8 — change `-` to `+`
  - No need to edit tests.
```

---

# Part IV — CodingAgent 主循环

## 4.1 API 与数据结构

```python
# src/agent.py
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class TraceStep:
    kind: Literal["thought", "tool_call", "tool_result", "observation", "summary"]
    payload: dict

@dataclass
class Trace:
    task: str
    steps: list[TraceStep] = field(default_factory=list)

    def append(self, step: TraceStep): ...
    def total_tokens(self) -> int: ...
    def last_n(self, n: int) -> list[TraceStep]: ...

class CodingAgent:
    def __init__(self, model: str, mcp_client, skill_loader: SkillLoader,
                 subagent_registry: dict[str, Callable]):
        self.model = model
        self.mcp = mcp_client
        self.skills = skill_loader
        self.subagents = subagent_registry

    def run(self, repo_path: str, issue: str) -> Trace:
        ...
```

## 4.2 循环工程逻辑

```python
def run(self, repo_path: str, issue: str) -> Trace:
    trace = Trace(task=issue)

    # 1. Build system prompt (deterministic)
    system = compose_system_prompt(
        skill_index_section=self.skills.system_prompt_section(),
        mcp_tool_descriptions=self.mcp.describe_all_tools(),
        rules=["绝对路径", "先 Read 再 Edit/Write", "并行独立调用",
               "Skill 命中再加载", "Subagent 只看摘要"],
    )

    # 2. Initial user message
    history = [{"role": "user",
                "content": f"Issue:\n{issue}\n\nRepo: {repo_path}\n"
                          f"Constraints: do not modify tests."}]

    for turn in range(MAX_TURNS := 50):
        # 3. Model decides next step
        response = self.model.chat(system=system, messages=history)

        # 4. Branch on response type
        if response.has_tool_calls():
            for call in response.tool_calls:
                tool_name = call.name
                args = call.args

                trace.append(TraceStep("tool_call", {"name": tool_name, "args": args}))

                # 4a. Subagent dispatch (meta-tool)
                if tool_name == "dispatch_subagent":
                    result = self.dispatch_subagent(args["subagent"], args["goal"])
                    observation = result.summary  # 主 agent 看不到子 agent 全部 trace
                    trace.append(TraceStep("summary", {"subagent": args["subagent"],
                                                        "summary": result.summary}))

                # 4b. Skill load (meta-tool)
                elif tool_name == "load_skill":
                    skill_body = self.skills.load(args["skill_name"])
                    observation = skill_body   # 全文塞入本轮 history
                    trace.append(TraceStep("thought", {"skill_loaded": args["skill_name"]}))

                # 4c. Normal MCP tool
                else:
                    observation = self.mcp.call(tool_name, **args)
                    trace.append(TraceStep("tool_result", {"name": tool_name,
                                                            "output": truncate(observation)}))

                history.append({"role": "tool", "name": tool_name,
                                "content": str(observation)})

        elif response.has_text():
            # 5. Final answer?
            if self._looks_done(response.text):
                trace.append(TraceStep("summary", {"final": response.text}))
                break
            history.append({"role": "assistant", "content": response.text})
            # 3'. Re-prompt: ask model to use a tool
            history.append({"role": "user",
                            "content": "Provide a tool call or final summary."})

    return trace
```

**关键工程决策（why）：**

1. **System prompt 拼装是 deterministic**——host 控审，让 skill index / 工具描述 / 规则稳定不变，便于 cache。
2. **Meta-tools 走同一个 tool_use 路径**——`dispatch_subagent` 和 `load_skill` 对模型来说跟 `read_file` 长得一模一样，模型学一次接口就能调所有"动作"。
3. **每个 turn 单独 trace step**——方便事后 debug / 评估 / 重放。
4. **MAX_TURNS 硬上限**——50 步不停就要么 break 要么 ping user，agent 不能无限循环。
5. **Final answer 检测** — 不靠 magic string，靠 `_looks_done(text)` 看是否是 `"## Summary"`/`"Done"`/测试全绿 / `</answer>` 之类的信号，避免"看起来在总结但还没真做完"。

## 4.3 防呆 / 重试 / 暂停

```python
# 这些是 soft skills，编码时容易忘

class CodingAgent:
    def _same_tool_3_times(self, trace: Trace) -> bool:
        """3 个连续 turn 都发同一个 tool call → 注入提醒"""
        ...

    def _stuck_loop_breaker(self, response) -> Action:
        """重复相同思路 5 次 → 触发 fork（强制开 subagent 探索）或 pause"""
        ...

    def _handle_tool_error(self, name: str, err: Exception) -> Action:
        """沙箱 / 权限 / timeout 错误 → 注入 'try a different approach' 提醒"""
        ...

    def _final_answer_check(self, response) -> bool:
        """最终答案要求包含：changes 描述 + 验证证据 + 失败处置"""
        ...
```

---

# Part V — Toy-repo 工作流（M0 → M1 → M3/M4）

## 5.1 任务定义

```
data/toy-repo/
├── src/calculator/
│   └── core.py              ← 含 bug：def add(a, b): return a - b
├── tests/
│   └── test_add.py          ← 2 passed, 1 failed (test_two_plus_two)
└── ISSUE.md                 ← "Fix bug in calculator.add; tests cannot be modified"
```

## 5.2 评估（M0）

```
toy_repo_patch  自检：
  1. agent 修改了 src/calculator/core.py，把 `return a - b` 改成 `return a + b`
  2. agent **未**改动 tests/ 任何文件
  3. 跑 python -m pytest → 全绿
  4. 返回最终的 Trace（≤ N 步）
  5. （可选）git diff 干净，无遗留调试 print
```

## 5.3 step-by-step（agent 视角）

```
turn 1  Read ISSUE.md
turn 1  Glob tests/  Read tests/test_add.py
turn 2  Grep "def add"  (定位 src/calculator/core.py)
turn 3  Read src/calculator/core.py
turn 4  write_file (or Edit) src/calculator/core.py, fix the operator
turn 5  run_tests cmd=python -m pytest
turn 6  final answer → "✅ 3 tests passed"
```

## 5.4 抽象的成功模式（喂给 SWE-bench）

把 toy 跑通的 `Trace` 提取成策略：

| 模式 | 来自 toy 的成功 |
|---|---|
| **Read before Write** | 改 `core.py` 前必须先 Read 它（被 Skill 规则强制） |
| **Run tests after every Edit** | 修完立刻验证，不积累错误 |
| **No test modification** | sandbox 路径过滤 + system 规则双管 |
| **最小改动** | Edit > Write；如果 1 行可修就别整个覆写 |

把这 4 条写进 `META_SKILL.md`（一个永远加载的 meta-skill），后续 SWE-bench Lite 任务直接复用。

---

# Part VI — 自检机制（self-check 契约）

每周末都该跑一遍：

| 里程碑 | 自检名 | 检查 |
|---|---|---|
| M1 | `mcp_server_lists_tools` | 调用 `list_tools()`，断言至少包含 5 个 name |
| M1 | `tool_smoke` | 每个 tool 各跑一次 happy path（写文件→读回→diff→apply） |
| M3 | `subagent_dispatch` | mock 父 agent，断言子 agent 返回 `summary ≤ 2KB`、独立 ctx |
| M4 | `toy_repo_patch` | 在 `data/toy-repo` 上完整跑一遍 agent.run()，断言 `pytest` 全绿 |
| M4 | `trace_within_budget` | 断言 Trace 总 token ≤ N |

把所有自检组织到 `tests/` 下，主入口 `pytest tests/` 一键。

---

# 附录 — 关键引用

| 主题 | 源 |
|---|---|
| Skill front-matter / 1% char budget / 1536 desc cap | `packages/builtin-tools/src/tools/SkillTool/prompt.ts` |
| `DiscoverSkills` 用于 mid-conversation 搜索 | `packages/builtin-tools/src/tools/DiscoverSkillsTool/prompt.ts` |
| Explore subagent 的 read-only system prompt | `packages/builtin-tools/src/tools/AgentTool/built-in/exploreAgent.ts` |
| Skillify（自动从 session 提炼 skill） | `src/skills/bundled/skillify.ts` |
| SKILL.md 示例 | `.claude/skills/{interview,teach-me}/SKILL.md` |
| 文件系统工具 schema 细节 | `coder-harness-file-system-spec.md` |
