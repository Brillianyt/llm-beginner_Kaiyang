# 任务六：Mini Coding Agent

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 一句话目标

用本地 Qwen2.5-Coder-7B-Instruct 复刻一个极简版 Claude Code：手写 MCP server（暴露 ≥ 5 个工具）、约 50 行的 Skill 加载器、1-2 个 Subagent 和一条 agentic loop，能在 `data/toy-repo` 上自主读懂 issue、改代码、跑测试直到 `python -m pytest` 全绿。

## 任务情境

假装你被派去给团队造一个「本地能跑的 Claude Code」，组长的要求是：

- 模型只能用本地部署的 Qwen2.5-Coder-7B-Instruct，不许调云端 API
- 工具要走 MCP 协议接入，能力要拆成 Tools / Skills / Subagents 三层
- 五六周后汇报：toy-repo 能自动修通，再讲清楚三层栈各自解决什么、agent loop 怎么停机

本任务**完全超出**实践书 v2 的覆盖范围，是全系列最大的一次难度跳变：子系统最多，且没有教材章节兜底，基本靠你查文档 + 读开源实现搭出来。

## 输入 / 输出

| | 内容 |
|---|---|
| **给你** | `data/toy-repo`（`data/download.py` 生成，含一个故意写错的 `calculator.add` + 三个 pytest）/ 可选 SWE-bench Lite 抽样元数据（`--with-swebench`，对应 repo 需自行 clone 到 `data/repos/`）/ 本地 Qwen2.5-Coder-7B-Instruct（Ollama / vLLM / llama.cpp+GGUF，建议 8GB+ 显存或走 Q4_K_M 量化）/ 官方 MCP Python SDK |
| **交付** | 1. `src/mcp_server.py`（可独立启动，暴露 ≥ 5 个工具） 2. `src/skill_loader.py` + `src/skills/` 下 2-3 个 Skill 3. `src/agent.py`（agent loop，能产出 `Trace`） 4. toy-repo 的 patch / trace 5. `eval/result.json`（自检结果） 6. 一段 200–500 字实验观察 |

## Definition of Done

必做 4 项，缺一不算完成：

- [ ] **M1** 手写 MCP server，顶层导出 `list_tools()`，自检 `mcp_server_lists_tools` 通过（枚举到 ≥ 5 个工具）
- [ ] **M2** 写 Skill 加载器 + 2-3 个带 YAML front-matter 的 `SKILL.md`，自检 `skill_loader_metadata` 通过（每个 skill 都有 name + description）
- [ ] **M3** 实现 agent loop（`CodingAgent.run` 返回 `Trace`），自检 `toy_repo_patch` 通过（修好 `calculator.add` 并让 `python -m pytest` 全绿）
- [ ] **M4** trace 完整记录每步 thought / tool_call / observation，`Trace` 含 `steps` / `patch` / `tests_passed`

加分（任选）：

- [ ] **S1** Q4_K_M 量化 vs FP16 的成功率对比
- [ ] **S2** 单 agent vs 加 Subagent 的词元消耗与成功率对比
- [ ] **S3** 纯 prompt vs 加 Skill 的成功率对比
- [ ] **S4** 在 SWE-bench Lite 抽样 3 题上 ≥ 1 题 `tests_passed`（很难，跑通 1 题就合格）

## 实施步骤（建议节奏：5-6 周）

### 第 1 周：环境 + 数据 + 模型部署

```bash
pip install -r requirements.txt

# 生成本地 toy repo + 打印模型部署提示
python data/download.py

# 可选：额外下载 SWE-bench Lite 抽样元数据；对应 repo 需另行 clone 到 data/repos/
python data/download.py --with-swebench
```

跑完应看到 `data/toy-repo/`（含 `calculator.py`、`calculator.py.orig` 基准快照、`test_calculator.py`、`ISSUE.md`）。本地起一个 OpenAI 兼容端点跑 Qwen2.5-Coder-7B-Instruct（脚本会打印 Ollama / vLLM / llama.cpp+GGUF 三种命令）。

**输入**：requirements + 模型权重
**输出**：toy-repo 就位、本地模型端点能响应一次最简 chat 请求

**常见坑**：

- 显存不够直接上 Q4_K_M GGUF（llama.cpp）或 AWQ（vLLM），别硬塞 FP16
- 新机器没配 git 身份时，`download.py` 的初始 commit 会静默失败——这是预期行为，自检靠 `calculator.py.orig` 快照重置、不依赖 git，别去补 git 身份
- 本地端点要确认是 OpenAI 兼容格式（`/v1/chat/completions`），否则 `openai` 客户端连不上

### 第 2 周：MCP server（M1）

**输入**：toy-repo 路径
**输出**：`src/mcp_server.py`，通过 `mcp_server_lists_tools` 自检

暴露工具：`read_file` / `write_file` / `run_tests` / `git_diff` / `git_apply`。模块顶层导出 `list_tools() -> List[dict]`（每个含 `name`）供自检枚举。

> ⚠️ 安全（逐条不可省）：
> - `read_file` / `write_file` 先 `Path.resolve()` 规整路径并校验落在目标 repo 内，拒绝 `..` 越界与绝对路径（否则 LLM 产出的路径能读写仓库外任意文件）
> - 跑 git / pytest 用 `subprocess` 的 list 形式（`args=[...]`、不要 `shell=True`、限定 `cwd`），别把 issue 文本 / 文件名拼进 shell 串（shell 注入）
> - `git_apply` 等只在工作 repo 内操作，避免 `git reset --hard` / `git clean -fd` / `git checkout --` 这类会丢未提交改动或删文件的命令

**常见坑**：

- 工具抛异常时要返回错误响应（结构化 error），别让整个 MCP server crash
- 工具 schema 不全：每个工具要有 name / description / input_schema，否则 client 不知道怎么调
- 自检只 import `list_tools()`，但 server 本身要能 `python src/mcp_server.py` 独立跑起来（stdio 握手），两条路都得通

### 第 3 周：Skill 加载器 + Skills（M2）

**输入**：`src/skills/*/SKILL.md`
**输出**：`src/skill_loader.py`（约 50 行）+ 2-3 个 Skill，通过 `skill_loader_metadata` 自检

实现 `class SkillLoader(skills_dir)`：扫描 `src/skills/*/SKILL.md`，按 description 匹配，命中后再把完整正文塞进 agent context（progressive disclosure）。每个 `SKILL.md` 用 YAML front-matter 写 `name` / `description` + 正文，对齐 [Anthropic Skills](https://github.com/anthropics/skills) 约定。至少 2-3 个，比如：

- `code-review/SKILL.md`：代码审查 workflow
- `pr-description-writer/SKILL.md`：PR 描述生成
- `test-runner/SKILL.md`：测试运行与失败诊断

**常见坑**：

- front-matter 漏了 `name` 或 `description`：`list_skills()` 返回项缺 key 直接挂自检（每项必须含 name + description）
- 一上来就把所有 SKILL.md 正文全读进 context：那就不叫渐进式披露了，应只匹配 description、命中后再读 body
- `description` 写得太泛（「处理代码」），匹配不到任务；要写清楚「何时加载」

### 第 4-5 周：Subagent + 主 agent loop（M3 + M4）

**输入**：toy-repo 的 `ISSUE.md`（修 `calculator.add`，不许改测试文件）
**输出**：`src/agent.py`、`src/subagents/`，通过 `toy_repo_patch` 自检

实现内容：

1. `src/subagents/`：1-2 个独立 context 的子 agent（如「代码搜索」「测试执行」），主 agent 只看摘要、不吞全部 trace
2. `src/agent.py`：`class CodingAgent`，`run(repo_path, issue) -> Trace`，循环 `while not done: model → tool → observation → loop`，支持调用 Skill、派发 Subagent
3. 先在 `data/toy-repo` 上跑通：bug 是 `add(a, b)` 写成了 `return a - b`，agent 要读文件、定位、改回 `a + b`、跑 `python -m pytest` 确认全绿，再挑战 SWE-bench Lite

**常见坑**：

- `Trace` 不是 dict（或 dict 子类）/ 缺 `patch` / `tests_passed`：自检按 `trace.get(...)` 取值，取不到当失败
- 停机条件写成「步数到了硬停」：要有明确的 done 信号（测试通过 / 模型显式声明完成），否则要么早停要么空转
- 自检每次从 `calculator.py.orig` 恢复 buggy 版本再跑——别假设上一轮已经修过、这轮可以空跑
- 长任务 context 爆了：必要时做 context compaction（压缩历史），否则 7B 模型上下文很快撑满
- Subagent 没真隔离（共用同一份 message 列表）就失去意义，应给独立 message 列表 + 独立步数上限 + 工具子集

## 实现约定

| 文件 | 必须导出 |
|---|---|
| `src/mcp_server.py` | 可独立运行的 MCP server（`python src/mcp_server.py` 启动）；并在模块顶层导出 `list_tools() -> List[dict]`（每个含 `name`）供自检枚举工具 |
| `src/skill_loader.py` | `class SkillLoader(skills_dir: str)`（构造时传入 skills 根目录）含 `list_skills() -> List[dict]`（每项含 `name`、`description`）、`load(name) -> str` |
| `src/agent.py` | `class CodingAgent` 含 `run(repo_path: str, issue: str) -> Trace`；`Trace` 是 dict（或 dict 子类），含键 `steps`、`patch`、`tests_passed: bool`，自检按 `trace.get(...)` 取值 |

接口可以改，但改了请同步调整 `eval/run.py`。

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 | 对应 DoD |
|---|---|---|
| `mcp_server_lists_tools` | 启动 MCP server 后能枚举到 ≥ 5 个工具 | M1 |
| `skill_loader_metadata` | SkillLoader.list_skills() 返回的每个 skill 都有 name + description | M2 |
| `toy_repo_patch` | 在 `data/toy-repo` 上修复 `calculator.add`，并让 `python -m pytest` 通过 | M3 |
| `swebench_lite_sample` | 可选进阶：在 SWE-bench Lite 抽样 3 题上 ≥ 1 题 tests_passed（很难，跑通 1 题就合格） | S4 |

结果写入 `eval/result.json`，提交时附上。

## AI Tutor 反馈

把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 整段贴给 Claude / Qwen / DeepSeek，连同你的代码。模型会按统一格式（必检 / 加分 / 优先级）给你针对性 review。

## 实验建议

- Q4_K_M 量化 vs FP16 的成功率
- 单 agent vs 加 Subagent 的词元消耗与成功率
- 纯 prompt vs 加 Skill 的成功率

## 能力三层栈

| 层 | 概念 | 角色 |
|---|---|---|
| 底层 | **Tools / MCP** | 原子工具，通过 MCP server 接入，无状态、可跨 agent 复用 |
| 中层 | **Skills** | 组织化能力包（SKILL.md + scripts + references），按需加载、渐进式披露 |
| 顶层 | **Subagents** | 独立 context 的子 agent，处理可并行或需隔离的子任务 |

## 前置阅读（非必需）

- [SWE-bench](https://www.swebench.com/)
- [CodeAct 论文](https://arxiv.org/abs/2402.01030)
- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
- [smolagents](https://github.com/huggingface/smolagents)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [Anthropic Skills](https://github.com/anthropics/skills)

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 「llm-beginner 实践成果」分类发帖，附：

1. 你的 fork 仓库链接
2. `eval/result.json` 内容（贴文本即可）
3. DoD checklist 勾选状态
4. toy-repo 的 patch / trace（若跑通 SWE-bench Lite 题，请把 trace 和 patch 一起附上）
5. 200-500 字实验观察：你做了哪些消融、看到了什么有意思的现象

## 时间

约 5-6 周。本任务是全系列最大的难度跳变：子系统最多、且完全超出教材覆盖。如果在 M3 卡住，先把 agent loop 缩到最小（只用 read_file / write_file / run_tests 三件套，不接 Skill、不派 Subagent）在 toy-repo 上跑通，再逐层加回 Skill 和 Subagent。
