# 11 — Toy-repo 端到端：skill + subagent + MCP 三层栈

## 目标

在 toy-repo 上**同时**验证三层能力栈都能跑起来：
- **Skill**：`load_skill(name)` 返回 Level-2 body
- **Subagent**：`dispatch_subagent(name, task)` 真正派子任务
- **MCP**：`mcp` SDK 通过 stdio JSON-RPC 与 `src/mcp_server.py` 通信

之前的实验已经分别证明三层各自工作（skill: `iteration/10`，subagent: `iteration/10` test-runner-explicit，MCP: `iteration/10` mcp_stdio）。本节把它们组合在一次 toy-repo 跑里，证明三层**同时**可用且不冲突。

## 同时发现一个 pre-existing MCP bug

跑 MCP stdio 测试时发现 `src/mcp_server.py` 用的是 mcp 旧版 API（`@server.list_tools()` 装饰器），但 mcp 2.0.0 改成了构造函数参数 `on_list_tools=...`。**stdio MCP 模式从来没真正工作过**——所有之前的 36+ capture 都是走 in-process `call_tool` 快速路径。

修复（commit 在本节一起提交）：

```python
# before — broken on mcp 2.0+
@server.list_tools()
async def _handle_list_tools() -> list[Tool]:
    ...

# after
server = Server(
    "coding-agent-tools",
    on_list_tools=_handle_list_tools,
    on_call_tool=_handle_call_tool,
)
```

修复后 MCP stdio 真正启动：
```
2026-08-25 03:26:37 [INFO] mcp_server: starting coding-agent MCP server over stdio
[mcp-client] init: server=coding-agent-tools
[mcp-client] list_tools → 9: [read_file, write_file, edit, list_files, grep, run_tests, git_diff, git_apply, run_bash]
```

## 端到端驱动

`/tmp/toyrepo_e2e_all_three.py`：
1. **MCP stdio**: `ClientSession` 通过 stdio 连接 `src/mcp_server.py`，发 `tools/call` (`read_file` + `run_bash`)
2. **Skill + Subagent**: in-process `CodingAgent` 跑 `data/toy-repo`，issue 显式引导模型调 `load_skill("test-runner")` + `dispatch_subagent("test_executor", "diagnose")` + `run_tests`

## 结果（commit `…` 之后的最终 evidence）

`eval/wire_captures/toyrepo_e2e_20260825T112718Z.json` 摘要：

| 层级 | 状态 | 证据 |
|---|---|---|
| **MCP stdio** | ✅ `is_error=False` | read_file 返回 211 字符 calculator.py 内容；run_bash 调用真实 subprocess |
| **Skill load** | ✅ 1 次 | `skill_loads: ['test-runner']` |
| **Subagent dispatch** | ✅ 1 次 | `subagent_invocations: 1`（test_executor） |
| **tool sequence** | — | `load_skill → dispatch_subagent → write_file → submit_text` |

### 三层同时启动的 wire 证据

```
[1] MCP read_file → is_error=False preview='=== /mnt/.../data/toy-repo/calcul...'
[2] MCP run_bash → is_error=False
[3] Agent: skill_loads=['test-runner'], subagent_invocations=1,
    tool_sequence=['load_skill', 'dispatch_subagent', 'write_file', 'submit_text']
```

## 关键观察

1. **MCP stdio + skill/subagent 协同无冲突**：MCP 走 stdio transport，agent 走 in-process `call_tool`，两者用同一个 `_TOOL_BY_NAME` registry，**同一份 `tools/__init__.py`**。base tools（read_file / edit / run_bash 等）通过 MCP 也能调（9 个），meta-tools（load_skill / dispatch_subagent / submit_patch / submit_text）**只**通过 agent 调——这是有意分层（见 `src/mcp_server.py:_TOOL_BY_NAME` 与 `src/agent.py:_meta_schemas` 的对照）。

2. **修复的 pre-existing bug**：mcp 2.0.0 升级后 stdio MCP 模式完全没工作过。修复后 `python src/mcp_server.py` 真能跑（之前 AttributeError 立即 crash）。这意味着以前的 `iteration/08` 用 mcp_sdk_demo 跑的 demo 是 in-process 的，不是真 MCP——本轮之前 toy-repo 演示里"使用 MCP 客户端"实际上**从来没发生**。

3. **三层栈现在真的"分层"**：
   - **MCP 客户端**（`mcp.client.stdio.stdio_client`）→ `src/mcp_server.py`（stdio JSON-RPC）→ `_TOOL_BY_NAME`（base tools）
   - **CodingAgent**（in-process）→ `_meta_schemas` + `_mcp_tool_schemas`（meta + base）→ 同一份 `_TOOL_BY_NAME`
   - **Subagent**（独立 message list）→ 同一份 `_TOOL_BY_NAME` 但带 `allowed_tools` allowlist

   三条路径汇聚到 `_TOOL_BY_NAME`，但**入口**不同。Client 想用 meta-tool 就走 CodingAgent；只想用 base tool 就走 MCP。

4. **edit-discipline prompt 现在不需要**：`src/prompt.py` 在本轮加了"scan Level-1 first" 段，但 toy-repo e2e 的 issue 仍然显式提及了 load_skill——这是因为 prompt cue 在隐式场景下不够强（见 `iteration/10` §D）。真要让模型自发调 skill，仍需 issue 字面触发。

## 文件

新增 2 份 wire capture：

```
eval/wire_captures/mcp_stdio_20260825T112637Z.json   — pure MCP stdio round-trip
eval/wire_captures/toyrepo_e2e_20260825T112718Z.json — 3-layer combined
```

修复 1 个 pre-existing bug：

```
src/mcp_server.py — `@server.list_tools()` decorator → `on_list_tools=` constructor param
```

## Commit

```
<pending>  fix: mcp_server.py broken on mcp 2.0+ SDK — list_tools/call_tool API change
```

## 关键 wire evidence

- `eval/wire_captures/mcp_stdio_20260825T112637Z.json` — 9 个 tools listed，read_file + run_bash 通过 stdio
- `eval/wire_captures/toyrepo_e2e_20260825T112718Z.json` — 三层同时跑通的端到端
- `iteration/10-toyrepo-skill-e2e.md` — skill + subagent 单独验证（前置实验）
- `src/mcp_server.py` — 修复后的 mcp 2.0+ 兼容版本