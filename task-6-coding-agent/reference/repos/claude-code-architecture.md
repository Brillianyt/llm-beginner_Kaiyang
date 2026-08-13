# Claude Code 架构解析

## 来源
- 产品：Claude Code（Anthropic 出品的终端 coding agent）
- 官方文档：https://code.claude.com/docs
- 中文实践：https://www.cnblogs.com/Chary/p/19918229 （3 万字拆解）
- 核心定位：**Claude Code 是本任务要复刻的目标**，理解它的三层架构与 hook/subagent 系统对设计 Mini 版至关重要

## 关键要点
1. **三层架构**（与本任务完全对齐）：
   - **Core Layer**：主对话 context（200K~1M token），质量随 token 增长反而下降
   - **Delegation Layer**：Subagent，独立 context window，**只回摘要**，避免污染主线
   - **Extension Layer**：MCP（外部工具）/ Hooks（生命周期拦截）/ Skills（领域知识）
2. **Hook 系统**（Claude Code 的标志性能力）：
   - 事件类型：`PreToolUse` / `PostToolUse` / `SessionStart` / `SessionEnd` / `Stop` / `SubagentStop` / `Notification`
   - 动作类型：`command`（跑脚本）/ `HTTP`（webhook）/ `prompt`（LLM 判 yes/no）/ `agent`（派 subagent 验证）
   - 配置层级：项目级 `.claude/settings.json` / 个人 `.claude/settings.local.json` / 用户级
   - 典型用途：拦截 `rm -rf` 等危险命令（PreToolUse 验证）、审计所有改动（PostToolUse 记录）
3. **Subagent 隔离**：
   - 最多 10 个并行 subagent
   - 每个有**全新的干净 context window**（不继承主线历史）
   - 返回**仅摘要**给主线（不暴露完整 trace）
   - `SubagentStop` 钩子在 subagent 完成时触发（区别于主线的 `Stop`）
4. **Agentic Loop**（非官方文档推断）：
   - 状态机驱动：`READ → UNDERSTAND → PLAN → ACT → VERIFY → DONE`
   - 停机信号：用户输入 / `Stop` 钩子返回 stop_reason / 工具调用稳定 1 轮无新动作
5. **与本任务对应关系**：
   - 我们没有「200K token」窗口（Qwen2.5-Coder-7B 默认 32K），所以 context compaction 几乎是必需
   - 我们做 1-2 个 subagent 就够（README 明确），但「独立 message 列表 + 独立步数上限 + 工具子集」三个属性必须都有
   - Hook 系统是加分项；先不做也能通过必做 DoD

## 与我们任务的关联
- **M3 / M4**：Claude Code 的核心 loop 范式就是我们要照搬的——只是规模小、步数少、模型小。
- **S2（Subagent 对照实验）**：Claude Code 的 subagent 设计给了我们 baseline 思路——subagent 只回摘要给主线，主线 step 数减少但成功率提升。
- **加分项（Hook 系统）**：可以在 `src/agent.py` 里加 `register_hook(event, callback)` 接口，支持 `PreToolUse` 拦截危险命令；这跟 Claude Code 的 `PreToolUse:bash hook error` 是同一回事。

## 代码片段（Claude Code 风格 hook 拦截伪代码）

```python
class CodingAgent:
    def __init__(self):
        self.hooks = {"PreToolUse": [], "PostToolUse": []}

    def register_hook(self, event: str, callback):
        self.hooks[event].append(callback)

    def execute_tool(self, tool_call):
        # PreToolUse：所有钩子过一遍；任何一个 deny 就中止
        for cb in self.hooks["PreToolUse"]:
            decision = cb(tool_call)
            if decision.get("action") == "deny":
                return {"error": decision.get("reason", "denied by hook")}
        obs = self.tool_registry[tool_call["name"]](**tool_call["arguments"])
        for cb in self.hooks["PostToolUse"]:
            cb(tool_call, obs)
        return obs
```

## 我们应该怎么借鉴
1. **Subagent 必须真的隔离**：把 subagent 实现成 `def run_subagent(task: str) -> str`，内部新建 `messages = []`，调用结束后只 `return summary`；主 agent 拿到的就是 1-2 句话摘要，**不要把 trace 列表塞进主 context**——这就失去了隔离的意义。
2. **步数上限要独立**：主 agent 给 30 步上限，code_search subagent 给 8 步上限（够用即可）。如果共用一个计数器，subagent 会偷走主 agent 的预算。
3. **工具子集**：subagent 只暴露必要工具。code_search subagent 只有 `read_file` + `grep`；test_runner subagent 只有 `run_tests` + `read_file`。不允许 subagent 调 `write_file`（破坏隔离）。
4. **PreToolUse hook 是 MVP 安全保障**：第一版可以只挂一个 hook：拦截 `write_file` 路径里含 `tests/` 的请求——直接对应 README 提到的「不许改测试文件」。
5. **state machine 而非无限循环**：状态显式记录在 `agent.state = "READ" | "UNDERSTAND" | ...`，每步转移一次，便于调试和 trace 复盘。
6. **失败恢复**：tool 返回 error 时，不是直接 break，而是记到 history 让 LLM 下一轮重试或换方案——Claude Code 的「最多重试 N 次」是经验值。
7. **不要照抄 prompt**：Claude Code 的 system prompt 是 Anthropic 内部优化过的；我们要自己写一份针对 Qwen2.5-Coder-7B 的，**用工具函数签名格式**而非 JSON schema（Qwen 在 code-style 工具调用上更稳）。

## 主要参考来源
- 官方文档：https://code.claude.com/docs
- 中文深度拆解：https://www.cnblogs.com/Chary/p/19918229
- Hook 系统详解：https://blog.csdn.net/qq_44810930/article/details/156146071
- 入门到精通：https://blog.csdn.net/SaberJYang/article/details/157465912