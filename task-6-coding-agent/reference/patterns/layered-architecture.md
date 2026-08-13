# 分层架构（Layered Architecture）—— 三层栈设计模式

## 来源
- 综合参考：Claude Code 源码 `src/`、`packages/`，Anthropic Skills 仓库，README 的「能力三层栈」表格
- 经典来源：Martin Fowler《Patterns of Enterprise Application Architecture》

## 关键要点
1. **三层栈严格单向依赖**：底层 → 中层 → 顶层，只允许上层调用下层，不允许反向依赖。Claude Code 实际是：Tool（基础能力） → Skill/MCP（扩展封装） → Subagent（独立子任务）
2. **每层职责边界清晰**：
   - **底层 Tools / MCP**：原子、无状态、可跨 agent 复用（`read_file`、`write_file`、`run_tests`）
   - **中层 Skills**：组织化能力包，含 frontmatter + markdown 正文 + 可选 scripts/，按 description 路由、按需加载
   - **顶层 Subagents**：独立 message 列表 + 独立步数上限 + 工具子集白名单，处理可并行的子任务
3. **数据流单向**：用户 → MainAgent → (Skill? / Subagent?) → Tool → Observation → MainAgent → 用户
4. **状态隔离**：每层有自己的状态表示（Tool 无状态；Skill 是静态元数据；Subagent 有自己的 message 栈）
5. **演化路径**：v1 可以只做底层（Tools）；v2 加 Skill（按需加载 prompt）；v3 加 Subagent（隔离并行）

## 与我们任务的关联
- **M1 / M2 / M3 对应三层栈**：Tools（MCP server）、Skills（loader）、Subagents（独立 agent）— README 已明确指定
- **避免循环依赖**：Subagent 不能反向依赖 MainAgent 的状态；Subagent 返回「摘要字符串」，MainAgent 拿到的是新信息而非共享状态
- **演化友好**：先实现单层（M1 → M3）让 toy repo 跑通，再加 Skill（M2）、Subagent（S2）作为加分项

## 文字版架构图

```
┌──────────────────────────────────────────────────────────┐
│                    Main Agent Loop                       │   ← 顶层
│  (CodingAgent.run: issue → plan → tool calls → trace)   │
└─────────────┬─────────────────────────────┬──────────────┘
              │ 调用                         │ 派发
              ▼                             ▼
      ┌──────────────────┐         ┌────────────────────┐
      │   Skill Loader   │         │    Subagents       │   ← 中层
      │  (description    │         │  (独立 message     │
      │   match → load)  │         │   list + 工具子集) │
      └─────────┬────────┘         └─────────┬──────────┘
                │ 触发                       │ 调用
                ▼                            ▼
┌──────────────────────────────────────────────────────────┐
│                    MCP Server / Tools                     │   ← 底层
│  read_file  write_file  run_tests  git_diff  git_apply   │
│  (+ str_replace_editor / search_code / glob)             │
└──────────────────────────────────────────────────────────┘
                │ JSON-RPC over stdio
                ▼
┌──────────────────────────────────────────────────────────┐
│          Local Execution (subprocess + cwd)              │
└──────────────────────────────────────────────────────────┘
```

## 代码片段（Python 三层最小骨架）

```python
# 底层 (src/tools/read_file.py) —— 纯函数 + 安全校验
def read_file(repo_root: Path, path: str) -> str:
    p = (repo_root / path).resolve()
    if not p.is_relative_to(repo_root):
        raise PermissionError("path traversal blocked")
    return p.read_text(encoding="utf-8")

# 中层 (src/skill_loader.py) —— 只在 list 时读 frontmatter
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self._meta = {}
        for md in skills_dir.glob("*/SKILL.md"):
            text = md.read_text(encoding="utf-8")
            meta, _ = parse_frontmatter(text)
            self._meta[meta["name"]] = meta["description"]
    def list_skills(self): return [{"name": n, "description": d} for n, d in self._meta.items()]
    def load(self, name): return (self.skills_dir / name / "SKILL.md").read_text()

# 顶层 (src/agent.py) —— 编排三层
class CodingAgent:
    def run(self, repo_path, issue) -> Trace:
        trace = {"steps": [], "patch": "", "tests_passed": False}
        messages = [system_prompt(self.tools, self.skill_loader.list_skills()), user_msg(issue)]
        for step in range(self.max_steps):
            resp = self.client.chat(messages, tools=self.tools)
            if not resp.tool_calls:
                break  # LLM 显式结束
            for call in resp.tool_calls:
                obs = self.execute_tool(call, repo_path)
                trace["steps"].append({"thought": ..., "tool_call": call, "observation": obs})
                messages.append(tool_result_msg(call.id, obs))
            if self.should_compact(messages):
                messages = self.compact(messages)
        return trace
```

## 我们应该怎么借鉴
1. **v1 只做底层 + 顶层**：先让 `CodingAgent` 直接调 5 个 tool（在 main agent 类内），不做 Skill / Subagent；让 toy repo 跑通
2. **v2 加 Skill**：把工作流指令（"先 read_file → run_tests → edit_file"）从硬编码 prompt 改成按需加载的 SKILL.md
3. **v3 加 Subagent**：把「代码搜索」「测试诊断」抽到独立 agent，主 agent 只看摘要
4. **接口约束**：每层只暴露 `list` + `execute` 两个动词。Skill 不暴露内部 state；Subagent 不暴露 message 列表
5. **依赖方向**：上层 import 下层；下层永远不 import 上层。`src/agent.py` import `src/tools/*.py`，但反过来不行
6. **状态共享方式**：Tool 通过参数/返回值；Skill 通过 context（description 字符串）；Subagent 通过摘要返回值——**永不共享可变状态**

## 主要参考来源
- Claude Code 源码：`src/coordinator/`, `src/skills/`, `packages/builtin-tools/`
- Anthropic Skills 仓库：https://github.com/anthropics/skills
- Martin Fowler《Patterns of Enterprise Application Architecture》
- README 任务背景「能力三层栈」表格