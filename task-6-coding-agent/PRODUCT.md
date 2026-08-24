# Mini Coding Agent — 产品说明

> 本地优先、可审计、可扩展的 Claude Code 风格 coding agent。复刻三层能力栈：**Tools / Skills / Subagents**，外加一个 tool-subprocess 驱动的 agentic loop，底座是 Qwen2.5-Coder-7B-Instruct。

## 1. 一句话定位

`CodingAgent.run(repo_path, issue) -> Trace` — 拿一个本地仓库路径 + 一段 issue 描述，跑完一个**确定性的 fix + 测试通过**的工程任务，把每一步 tool call / observation / thought / final patch 全程记录到一个 dict-subclass 的 `Trace` 里，供后续 replay / audit / 评测用。

---

## 2. 目标用户 & 典型场景

| 场景 | 用户 | 痛点 | 我们提供的 |
|---|---|---|---|
| 自动化 bug fix | 中小团队的 SRE / 后端开发 | 修 bug 流程重复（读 issue → 读代码 → 写 patch → 跑测试） | `CodingAgent` + toy-repo / SWE-bench 模式 |
| 评测研究 | Agent 研究者 | 需要可重放 / 可审计的 trace | 完整 `Trace` JSON + `TraceReplay` 干跑 |
| MCP 工具接入 | 平台开发者 | 写一个能跑的标准 MCP server | `python src/mcp_server.py` 直接起 stdio |
| Prompt 调优 | LLM 应用工程师 | 想 swap 模型 / 调系统提示 | `LLMClient` 任意 OpenAI 兼容端点 |

不适用场景：30+ 文件的大型重构（Qwen-7B 能力上限）、生产级 SWE-bench 评测（推荐 32B+ 模型）。

---

## 3. 产品形态

### 3.1 仓库目录

```
src/
├── agent.py                # CodingAgent 主循环（4 个 stop signal + compaction + cache_control）
├── mcp_server.py           # stdio MCP server（公共 mcp.server.Server API + jsonschema 边界）
├── llm_client.py           # OpenAI 兼容客户端（OpenAI tool schema <-> MCP inputSchema 转换）
├── skill_loader.py         # Level-1 索引 + Level-2 body + scripts/ / references/ 懒加载
├── tools/                  # 5 个原子工具 + base 安全底盘
│   ├── read_file.py        # cat -n 格式 + offset/limit + absolute-path 校验
│   ├── write_file.py       # 原子 tmp+os.replace，read-first guard
│   ├── edit.py             # old_string/new_string 替换 + unique/replace_all
│   ├── run_tests.py        # pytest 结构化解析（passed/failed/failures[]）
│   ├── git_diff.py         # per-file patch + secret 文件 redact
│   └── git_apply.py        # --check + 真 apply + 失败时 snapshot rollback
├── subagents/              # 独立 message + 步数上限 + 工具白名单 + 2KB 摘要硬截
│   ├── base.py
│   ├── search_executor.py  # 仅 read_file（read-only）
│   └── test_executor.py    # read_file + run_tests
├── skills/                 # 3 个渐进披露的 SKILL
│   ├── code-review/        # 配 scripts/diff_stats.py + references/review-checklist.md
│   ├── pr-description-writer/
│   └── test-runner/
├── hooks.py                # PreToolUse (拒 test_*.py 写) + PostToolUse (审计日志)
├── context.py              # maybe_compact + cache marker re-stamp
├── prompt.py               # 系统提示词拼装（确定、可缓存）
├── trace.py                # Trace dict-subclass + StepKind enum
└── replay.py               # TraceReplay 干跑 + 0.8 match_rate on M3
models/
└── Qwen2.5-Coder-7B-Instruct/
    └── coder_chat_template.jinja  # 自定义 chat template（裸 JSON 工具指令，让 Coder 模型进入工具调用模式；见 §6.5）
ablations/                 # 对照 + 消融
├── with_subagent.py        # S2: single vs subagent
├── with_skills.py          # S3: prompt-only vs skills
├── swebench_sample.py      # S4: sqlfluff 实例
├── baseline_smolagents.py  # 对照 smolagents 1.26.0
└── replay_trace.py         # CLI 入口
```

### 3.2 公开 API

```python
from src.agent import CodingAgent
from src.skill_loader import SkillLoader

agent = CodingAgent(skill_loader=SkillLoader("src/skills"))
trace = agent.run(repo_path="data/toy-repo", issue="修复 calculator.add")
print(trace["tests_passed"], trace["done_reason"], len(trace["steps"]))
```

`CodingAgent` 关键参数：
- `llm` — 任意 OpenAI 兼容 LLM client（默认 `http://localhost:30000/v1`）
- `skill_loader` — `None` 关闭 skills；非 `None` 时 Level-1 索引进 system prompt
- `max_turns` — 默认 50；env `CODING_AGENT_MAX_TURNS` 可覆盖
- `enable_subagents` — `True`（默认）；评测 ablation 可关闭
- `bootstrap_explore` — `True` 时 run() 注入 list_files + README 摘要，适合陌生大仓库

`Trace`（dict-subclass）字段：
`steps` / `patch` / `tests_passed` / `done_reason` / `turn_count` / `token_usage` / `subagent_invocations` / `skill_loads` / `compaction_events` / `summary` / `error`

---

## 4. 工作原理（30 秒讲完）

```
┌──────────────────────────────────────────────────────────────┐
│  CodingAgent.run(repo, issue)                                │
│                                                              │
│  while not done:                                             │
│      ┌─────────────────┐                                     │
│      │  Model (LLM)   │ ← system prompt (含 Level-1 索引)    │
│      └────────┬────────┘                                     │
│               │  tool_use <tool_call>                        │
│   ┌───────────┼──────────────────────────┐                   │
│   │           │                          │                   │
│   ▼           ▼                          ▼                   │
│ MCP Tool   Skill load                Subagent dispatch       │
│   │           │                          │                   │
│   ▼           ▼                          ▼                   │
│ observation  body (in ctx)         summary (≤2KB)            │
│   │           │                          │                   │
│   └─────┬─────┴──────────┬───────────────┘                   │
│         ▼                ▼                                    │
│   messages.append    trace.append(StepKind)                  │
│                                                              │
│  Stop signals:                                                │
│  ① submit_patch (有 diff) → apply + verify tests             │
│  ② submit_text (give up)                                    │
│  ③ _looks_done (## Summary / 好的 / 完成了 等多语言 marker)  │
│  ④ max_turns reached                                         │
│                                                              │
│  Cache: cache_control=ephemeral on system prompt,            │
│         re-stamped after every compaction.                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 关键能力

### 5.1 Tools 层（5 个原子工具 + Edit）

每个工具实现 `BaseTool.__call__(args, repo_root) -> ToolResult` 协议——`ToolResult` 永远不抛异常，错误都包成 `(content, is_error=True)`。

| 工具 | 用途 | spec 对应 |
|---|---|---|
| `read_file` | cat -n 格式读文件 + offset/limit 分页 | file-system-spec §1 |
| `write_file` | 原子全量写 + read-first guard | file-system-spec §3 |
| `edit` | old_string/new_string 精确替换 + unique 校验 | file-system-spec §2（**P2 新增**） |
| `run_tests` | pytest 结构化（passed/failed/failures[]） | file-system-spec §1.3 |
| `git_diff` | per-file patch + secret redact | file-system-spec §4 |
| `git_apply` | dry-run → apply + **失败时 snapshot rollback**（**P3**） | file-system-spec §5 |

`EditTool` 在 toy-repo 上是**首选**——避免 write_file 整文件重写，降低破坏风险。

### 5.2 Skill 层（渐进披露 + scripts/references）

3 个 SKILL.md 都有 `name / description / when_to_use` frontmatter。`SkillLoader` 提供：

| 阶段 | 何时读 | 成本 |
|---|---|---|
| **Level-1 索引**（description 列表）| 每次进 system prompt | ~8000 chars ≈ 1% of 8K ctx |
| **Level-2 body**（完整 SKILL.md）| agent 调 `load_skill(name)` | 视 skill 而定（几 KB） |
| **scripts/**（可执行脚本）| agent 用 `list_scripts` / `read_script` | 按需 |
| **references/**（长文档）| agent 用 `list_references` / `read_reference` | 按需 |

`code-review` skill 已经带一个真实可用的 `scripts/diff_stats.py`——agent 可以 `run_bash python diff_stats.py <diff>` 立即拿到文件级 +/-/files-touched 摘要。**P1 修复**（之前两轮评审都点名）。

### 5.3 Subagent 层

- `search_executor`：read-only 探索，仅 `read_file`
- `test_executor`：跑 pytest 拿结构化失败，仅 `read_file` + `run_tests`

主 agent 派发后**只拿到 ≤2KB 摘要**——subagent 完整 trace 写到 `trace["subagent_invocations"][*].transcript` 供 replay。**P5 修复**让 subagent transcript 完整持久化。

### 5.4 Agentic loop（4 停机信号）

1. `submit_patch(diff)` — 真 diff，应用 + 跑 pytest
2. `submit_text(text)` — 放弃，summarize state
3. `_looks_done(text)` — 多语言 marker 匹配（`## Summary` / `## 总结` / `好的` / `完了` / `搞定` 等）
4. `max_turns` — 默认 50

`_DONE_MARKER_RE` 在 Qwen-7B 偶尔输出中文 marker 时也能正确 stop（**P5 修复**）。

### 5.5 上下文压缩 + prompt cache

`maybe_compact(messages)` 当 chars > 25K 时滑动窗口压缩（保留 head 3 + tail 5）。**每次压缩后** `_ensure_cache_marker` 重新打 `cache_control: ephemeral` — 不然 OAI/SGLang 缓存就失效。**P1 修复**。

### 5.6 Hook 机制

- `PreToolUse`（默认装）：拒绝 `write_file` 到 `test_*.py` / `*_test.py` / `*/tests/*`
- `PostToolUse`（默认装）：audit logger — 每次 tool call 追加一行 JSONL 到 `CODING_AGENT_AUDIT_LOG`（默认 `<cwd>/.coding-agent-audit.jsonl`），observations 截 400 字符。**P3 修复**。

### 5.7 Trace replay

`src/replay.py::TraceReplay(trace).replay()` 干跑每个 `tool_call` 步骤、对比 observation。`eval/traces/m3_toy_repo.json` 实际 replay 得到 **0.8 match_rate**（4/5 步骤匹配）——**0.2 不匹配全是 state-chain diff**（write_file 改了文件，后续 read_file 看到新内容），这是 replay 设计的正确行为。

---

## 6. 性能 & 评测数据

### 6.1 Self-check（必做 4 项，**全 PASS**）

| 验收 | 结果 |
|---|---|
| M1 `mcp_server_lists_tools` | PASS — **8 tools registered** (read_file / write_file / edit / list_files / grep / run_tests / git_diff / git_apply) |
| M2 `skill_loader_metadata` | PASS — 3 skills, 每个含 name+description |
| M3 `toy_repo_patch`（Coder 路径） | PASS — CodingAgent **55 turn** 修通 calculator.add, `done_reason=tests_passed` |
| M3 `toy_repo_patch`（Qwen2.5-7B 工具版） | PASS — CodingAgent **11 turn** 修通, `done_reason=tests_passed`（原生 tool_calls） |
| M4 trace structure | PASS — dict-subclass with steps/patch/tests_passed/done_reason |
| 31/31 smoke tests | PASS |

两条 toy-repo 路径对比见 §6.5。

### 6.2 消融（加分项）

| 消融 | single mode | subagent mode | 观察 |
|---|---|---|---|
| **S2** token 消耗 vs 成功率 | 6312 tok, 3.67 turn, **3/3** | 5655 tok, 4 turn, **3/3** | toy-repo 太小，subagent 优势在大 repo 才显现；token 反而省 10% |
| **S3** prompt vs skills | **0/1**（2 turn 放弃）| **1/1**（4 turn 修通）| Level-1 骨架本身（bodies 从未 load）给模型提供了 test-runner workflow |

### 6.3 SWE-bench Lite（模型能力上限，harness 已修复 3 个真实 bug）

3 个 sqlfluff 实例 × 多次跑 = 6 次尝试，**0/6 通过**。但这轮深挖发现**不是"harness 已验证"那么简单**——在 0/6 背后埋了 3 个真实 harness bug：

| Bug | 影响 | 修复 |
|---|---|---|
| `list_files` 从 `make_tool_set` 里丢了 | `bootstrap_explore=True` 一开就 `KeyError`，**所有 SWE 跑第 1 步就崩**（这就是早期 0/6 的根因） | 恢复工具 + 注册（P-本轮） |
| `list_files` 拒绝绝对路径 | agent 用绝对路径调它 → 14 turn 死循环重试同一个绝对路径 | 接受 repo 内绝对路径（P-本轮） |
| `run_tests` 默认 `-x`（fail-fast） | 大 repo 上第一个无关失败（plugin-example）挡住了 L031 真实失败，agent 看不到目标测试结果 | 去掉 `-x`，跑完整套（P-本轮） |

修复后 agent 行为（sqlfluff-1625 实测 trace）：

```
turn 1  grep L031 → 发现 src/sqlfluff/rules/L031.py
turn 2  read L031.py（219 行全文）
turn 3+ edit L031.py — 但 old_string 猜错（"if not join_condition:" 并不存在于文件里）
         ↳ 模型缺乏对 L031 AST 遍历的语义理解，产生不了正确 edit
```

**结论**：harness 已不再是瓶颈（模型第一轮就能导航到正确文件）；剩余卡点是 **Qwen-7B 的语义能力**——修复 L031 需要理解 `_eval` / `_filter_table_expressions` / `recursive_crawl` 的交互，这超出 7B 的能力边界。完整证据在 `eval/traces/s4_1625.json`。

**2026-08-24 更新**：本轮重新抽样 3 题（astropy 三个 instance：`__astropy-12907` / `__astropy-14182` / `__astropy-14365`），`data/repos/astropy` 已 shallow clone + 按需 fetch 三个 base_commit。**实测 3 题 `0/3 PASS`**，与 sqlfluff 时代一致——剩余卡点仍是 Qwen-7B 的语义能力。

跑这轮时还发现两个 harness bug：

| Bug | 修复 |
|---|---|
| `test_swebench_lite_sample` 不切 base_commit，3 题都基于同一份 HEAD 跑（数据无意义） | 每题跑前 `git checkout -- . && git clean -fd && git checkout <base_commit>`（`eval/run.py`，2026-08-24） |
| subagent（`test_executor`）调 `run_tests` 时回传 vLLM 400——subagent 没走 `_to_wire_tool_calls` | 把 `to_wire_tool_calls` 从 `src/agent.py` 提到 `src/llm_client.py`，主 agent 与 subagent 共用（2026-08-24） |

### 6.5 vLLM 工具调用解析（2026-08-24；A-2 自定义 parser · **hard-prohibit** 架构）

Coder 模型的工具调用形态与官方 `hermes` parser 期望不一致（gate 实验实测 12% 命中）。原计划 Option A（"对齐 chat template 与 hermes，让原生 tool_calls 工作"）在 Coder 模型上不成立 —— Coder 模型有自己强先验的 `<response>` / `function_call` / 裸 JSON 多种 wrapper 偏好，无法被 chat template 强行约束。**最终路线 A-2**：

1. 在 vLLM 里**写一个 Coder 专用 parser plugin**，把任意 wrapper 形态统一转回 OpenAI 原生 `tool_calls`；
2. **agent 端**走 hard-prohibit 架构 —— **永远不再**对 `message.content` 做文本模式工具调用解析，无论是为了 silent rescue、loud-error 还是诊断。

| 路径 | 模型 | 工具调用方式 | toy-repo 步数 | eval/run.py |
|---|---|---|---|---|
| **Coder 路径 (A-2 · hard-prohibit)** | Qwen2.5-Coder-7B-Instruct + **自定义 parser plugin `qwen_coder_json`** | 原生 `tool_calls`（由 `src/vllm_plugin/qwen_coder_tool_parser.py` 解析） | **15-25 turn** | ✅ PASS |
| **工具版路径** | Qwen2.5-7B-Instruct + `hermes` parser（vLLM 内置） | 原生 `tool_calls`（`finish_reason=tool_calls`）| 11 turn | ✅ PASS |
| ~~**Coder 路径（旧，已废弃）**~~ | — | ~~regex fallback / `_parse_text_tool_calls` 静默合成~~ | ~~55 turn~~ | 已废弃 |

**关键发现**：

1. **Coder 模型在 hermes-XML 指令下也不稳定** —— 即便把 chat template 改成标准的 `<tool_call>{"name":..., "arguments":...}</tool_call>` 形态，transformers 本地推理 8 个工具调用 prompt 输出里有 7 个用了 `<response>` / `function_call` / 裸 JSON 等替代形态，只有 1 个用了 hermes 期望的精确形态。
2. **A-2 + hard-prohibit 架构落地路径**：`src/vllm_plugin/qwen_coder_tool_parser.py` 用 vLLM 的 `ToolParserManager.import_tool_parser(plugin_path)` 机制注册。Parser 注册名为 `qwen_coder_json`，启动命令加 `--tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py --tool-call-parser qwen_coder_json`。
3. **Parser 是单路径**：strip markdown fence + 一个 regex 找 `{name, arguments}` JSON。Coder 实际输出的 4 种形态（`<tool_call>{json}</tool_call>` / `<response>` / `function_call` / 裸 JSON）都以同一种 `{name, arguments}` 形态呈现，wrapper 是透明的。
4. **XML-tag-split 形态不解析**（gate 1/8）：`<tool_call><name>X</name><arguments>{...}</arguments></tool_call>` —— name 和 arguments 被拆到独立 XML tag。13% 的 **已知未处理边角** —— 文档化接受；用户可改 chat template 消除。
5. **agent 主循环零文本解析**：`_parse_text_tool_calls` / `_JSON_TOOL_RE` / `_dedupe_tool_calls` / `_fallback_apply` / `parser_miss_count` / `text_tool_call_fallback` 全部从 agent 删除。剩下的正则（`_extract_patch`、`_DONE_MARKER_RE`）处理 "model 选择写文本"，与工具调用解析无关。诊断性 surface 单独搬到 `src/diagnostics/text_tool_parser.py`，**agent 不可导入**（静态守卫 `test_agent_never_introspects_text_for_tool_calls` 强制）。
6. **架构动机**：silent rescue、loud-error、shape cascade 都是**补救上游 parser bug 的补丁**。补丁掩盖真正的问题；让问题响亮冒出来或者从源头修好才是健康做法。

**健康指标（trace 顶层字段）**：

| Metric | 公式 | 阈值 |
|---|---|---|
| `tool_call_native_rate` | 拿到原生 `tool_calls` 的 turn / 总 turn | **= 1.0**（任何 < 1.0 都说明部署层 parser plugin 不健康，写到工程笔记追查） |

> `parser_miss_count` / `text_tool_call_fallback` / `fallback_fire_rate` / `fallback_recovery_rate` / `arg_parse_failure_rate` 五项旧指标**全部删除** —— 因为对应的代码路径已不存在。健康就一个值。

**项目 invariant**：`AGENTS.md` 把"不使用任何 fallback"作为本项目最高规则固化。任何后续 AI agent 接手时碰到这条红线都不应破例。

### 6.6 smolagents 对照

| | CodingAgent | smolagents 1.26.0 |
|---|---|---|
| toy-repo (issue → patch → pytest 3/3) | **1/1 PASS** (5 turn, 6s) | **0/1 FAIL** (10s) |

smolagents 失败原因：**沙箱禁止 `import pytest`**（白名单 stdlib only）。我们的 tool-subprocess 模型在通用任务上更灵活——smolagents 跑 10 秒失败的根本原因就是它**自己不能 verify 自己的 fix**。

---

## 7. 安全设计

- **路径**：所有工具的 `file_path` schema 强制 `"pattern": "^/"`（绝对路径），实现层用 `safe_resolve` 拒绝 `..` 越权
- **subprocess**：`args=[...], shell=False`，env 隔离到 `PATH + HOME + extra_env`
- **git 黑名单**：`BLOCKED_GIT_FRAGMENTS`（`reset --hard` / `clean -fd` / `push --force` 等）在 `check_blocked_git` 拦截
- **MCP 边界**：`call_tool` 入口用 `jsonschema.validate(args, schema)` 拒绝半残输入
- **审计**：默认 `PostToolUse` 写 `<cwd>/.coding-agent-audit.jsonl` —— `tail -f` 实时看 agent 行为
- **read-before-write**：`write_file` / `edit` 要求目标文件先被 `read_file` 过；防止 agent 盲目覆盖未知状态
- **per-instance read registry**：**P4 修复** — 两个 CodingAgent 并行时 `_read_paths` 隔离，不再共享 module-level set

---

## 8. 部署 / 集成

### 8.1 最小部署

```bash
# 1. Python 依赖
pip install -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple \
    openai pyyaml mcp jsonschema safetensors safetensors vllm

# 2. 数据 + 模型（Qwen2.5-Coder-7B-Instruct BF16 ~15GB，ModelScope 24s）
python data/download.py
# 模型自动下载到 ./models/Qwen2.5-Coder-7B-Instruct/

# 3. 起 vLLM OpenAI 兼容端点（Coder 路径关键配置 —— A-2 自定义 parser plugin）
python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen2.5-Coder-7B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen_coder_json \
  --tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py \
  --chat-template models/Qwen2.5-Coder-7B-Instruct/coder_chat_template.jinja \
  --generation-config vllm \
  --gpu-memory-utilization 0.85

# 4. 跑评测
QWEN_MODEL=models/Qwen2.5-Coder-7B-Instruct \
  OPENAI_BASE_URL=http://localhost:30000/v1 \
  python eval/run.py
```

**Coder 路径四个非默认参数不能省**：
- `--chat-template`：让模型收到"裸 JSON"指令（默认模板的 `<tool_call>` XML 会让 Coder 模型陷入"请提供代码"被动反问，transformers 本地复现确认是模型行为）
- `--generation-config vllm`：关闭 `generation_config.json` 对采样的覆盖（默认 `repetition_penalty=1.1, top_k=20` 把模型推到局部最优）
- **`--tool-call-parser qwen_coder_json`**：**用我们自定义的 Coder parser**（见 `src/vllm_plugin/qwen_coder_tool_parser.py`）。它识别 Coder 模型实际使用的 4 种 wrapper 形态（`<response>` / `<tool_call>` / `<function_call>` / 裸 JSON），比 vLLM 内置的 `hermes` / `qwen3_xml` 更准确。
- **`--tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py`**：vLLM 0.23+ 用 `ToolParserManager.import_tool_parser` 加载自定义 parser 的入口。**不要漏掉这个**，否则启动时报"tool parser 'qwen_coder_json' not found"。

**启动不加载 parser plugin 的后果**：vLLM 端 `msg.tool_calls` 为空，agent 端点 `if not msg.tool_calls` + 检测到内容含裸 tool call JSON 时**响亮报错**(`[ERROR] parser_missed`, `done_reason=ERROR`)，不再静默用 regex 抢救。

切到工具版模型（Qwen2.5-7B-Instruct）只需去掉 `--chat-template` / `--generation-config vllm` / `--tool-parser-plugin`，把 `--tool-call-parser` 改回 vLLM 内置的 `hermes`，原生 `tool_calls` 直接走通（11 turn 修通 toy-repo）。

### 8.2 接入自家 MCP client

```bash
# 标准 stdio JSON-RPC MCP server
python src/mcp_server.py

# Claude Desktop / Cursor 等可配 stdio MCP client
# config.json:
{
  "mcpServers": {
    "coding-agent": {
      "command": "python",
      "args": ["/abs/path/to/task-6-coding-agent/src/mcp_server.py"]
    }
  }
}
```

### 8.3 切换 LLM 后端

`QWEN_MODEL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY` 环境变量切端点：

```bash
# Claude Sonnet via Anthropic's OpenAI-compatible proxy
export OPENAI_BASE_URL=https://api.anthropic.com/v1
export OPENAI_API_KEY=sk-ant-...
export QWEN_MODEL=claude-sonnet-4-5

python -m eval.run
```

> 32B+ 模型上 S4 通过率预计显著提升（7B 模型局限）。`README.md` DoD 勾选表 + `eval/result.json` 含 6 条 engineering notes 帮你做模型调优决策。

---

## 9. 路线图

按价值 / 成本排：

1. **Last-turn prompt 注入**（max_turns − 1 轮主动让模型决定 submit/submit_text；20 行）
2. **LLM-as-judge 外部评估**（用 GPT-4o 评 CodingAgent vs smolagents；50 行 + API key）
3. **run_bash 工具**（让 agent 真跑 skill 自带脚本；30 行 + sandbox）
4. **checkpoint resume**（trace 写到磁盘 + 失败后可继续）
5. **multi-repo benchmark harness**（用 SWE-bench Lite 全集 300 题跑报告）

---

## 10. FAQ

**Q: CodingAgent 跟 Qwen-Agent / smolagents 比优势在哪？**
A: 三点：(1) tool-subprocess 模型（pytest 真在子进程跑）vs smolagents 沙箱 stdlib-only；(2) 三层能力栈（Tools/Skills/Subagents）解耦清晰，subagent 隔离 + 2KB 摘要硬截；(3) Trace 完整且可 replay。

**Q: 为什么用 7B 模型，不用 32B？**
A: 7B 在 8GB+ 显存就能跑，量化后甚至 CPU 也能跑；32B 模型在 SWE-bench 类任务上明显更强但需要 24GB+ 显存。harness 与模型解耦——`QWEN_MODEL` 切。

**Q: SWE-bench 0/3 是 bug 吗？**
A: 不是。`eval/result.json` 的 `engineering_notes` 解释了：agent 持续 12 turn 真正读了 rule 文件，但 SQL parser 修复需要更深层语义。这是 Qwen-7B 能力上限，harness 本身已经验证（smolagents 沙箱限制下 0/1 vs 我们 1/1 已经能说明问题）。**2026-08-24 更新**：本轮重新抽样 3 题（astropy），实测同样 `0/3 PASS`——结论不变。本轮同时修了两个 harness 真实 bug（subagent 的 `arguments` 序列化、`test_swebench_lite_sample` 不切 base_commit），harness 路径已稳定。

**Q: SGLang 切到 vLLM 后有什么坑？**
A: 四条经验（2026-08-24 验证）：
1. **Coder 模型必须配合 `--chat-template coder_chat_template.jinja --generation-config vllm`**，否则模型陷入"好的，请提供代码..."被动反问（transformers 本地复现确认是模型行为，不是 vLLM 引擎问题）
2. **`--tool-call-parser` 必须用 `qwen_coder_json`**（自定义 parser plugin，由 `src/vllm_plugin/qwen_coder_tool_parser.py` 提供）—— PPU 定制版 vLLM 的 `qwen3_xml` parser 有 bug，本地单测都无法解析 Qwen2.5 的 `<tool_call>` 格式；`hermes` parser 对 Coder 模型的 4 种 wrapper 形态只命中 12%（gate 实测）。切到 instruct 版模型可以保留 `--tool-call-parser hermes`（对 Qwen2.5-Instruct 完美）。
3. **`--tool-parser-plugin src/vllm_plugin/qwen_coder_tool_parser.py` 不能省** —— vLLM 0.23+ 用 `ToolParserManager.import_tool_parser` 加载自定义 parser，漏了这个启动就直接报"tool parser 'qwen_coder_json' not found"。
4. **回传 assistant message 时 `arguments` 必须是 JSON 字符串**——`LLMClient` 内部把字符串转成 dict 方便消费，但 vLLM OpenAI 协议边界拒绝 dict，`_to_wire_tool_calls` 已在 harness 修好

**Q: A-2 parser plugin 漏了 tool call 会发生什么？**
A: Agent 端**完全不检测**。`message.tool_calls == []` + `message.content` 是文本时 → 进合法文本路径（patch extract / done / prose nudge）。Agent **永不在 `message.content` 里找工具调用 JSON** —— 哪怕模型明明写了一个 `<tool_call>{...}</tool_call>`，agent 也不去解析。诊断通过 `tool_call_native_rate` 部署级 metric 看：rate < 1.0 说明 parser plugin 不健康。修 parser 比 agent 端兜底更重要。

**Q: 旧 `text_tool_call_fallback` / `parser_miss_count` / `fallback_*` 字段全去哪了？**
A: 都删了。Agent 主循环不再调用 `_parse_text_tool_calls`，所以这些字段没有事件来源。**唯一保留的健康指标是 `tool_call_native_rate`**（阈值 = 1.0）。XML-tag-split 形态（gate 1/8 = 12%）是已记录的未支持边角，部署层修复路径是改 chat template / system prompt。

**Q: 工具版 Qwen2.5-7B-Instruct 比 Coder 版好在哪儿？**
A: Qwen2.5-7B-Instruct 经过工具调用训练，能直接产出原生 `tool_calls`（`finish_reason=tool_calls`），不再走 `_parse_text_tool_calls` 兜底。toy-repo 上 11 turn 修通（Coder 版 55 turn），同样 `done_reason=tests_passed`。代价：偏离 README "只用 Qwen2.5-Coder-7B-Instruct"的模型指定，适合做对照实验或需要更快 toy-repo 反馈的场景。SWE-bench Lite 等大仓库任务建议至少 14B+ 工具微调版。

**Q: Trace replay 只能跑 0.8 match——能跑到 1.0 吗？**
A: 跑不到 1.0 因为 trace 步骤间存在 state-chain（write_file 改文件 → 后续 read_file 看到新内容）。docstring 已说明这是正确行为，不是 replay bug。**真正完全的 1.0 match 需要 trace 自带 monkeypatch 时间戳**——这是 Anthropic 复现 0/1 实验的常见 trade-off。

**Q: 怎么贡献一个新 skill？**
A: 在 `src/skills/<your-skill>/` 放 `SKILL.md`（YAML frontmatter `name` + `description` 是必填），可选加 `scripts/` 和 `references/`。重启 `CodingAgent` 即可——`SkillLoader.scan()` 自动检测。

**Q: 多个 CodingAgent 并行跑会冲突吗？**
A: 不会。**P4 修复**让每个 agent 有独立的 `self._read_paths` set + 独立的 tool 实例（`make_tool_set()`），不共享 module-level state。审计日志文件名带 instance 名（如果 `CODING_AGENT_AUDIT_LOG` 不设置默认用 `<cwd>/.coding-agent-audit.jsonl`——多 instance 时会争抢，建议显式设 per-instance 路径）。
