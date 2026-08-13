# Claude Code 完整源码 · 本地参考

> 已 clone 到 `reference/claude-code/`，已加入 `.gitignore`，不入公开仓库。

## 来源

- 仓库：`git@github.com:claude-code-best/claude-code.git`（也支持 `https://github.com/claude-code-best/claude-code.git`）
- 定位：复刻 Claude Code 风格的本地 Coding Agent
- 与本任务关系：task-6 的目标就是「极简版 Claude Code」，这份源码是**最直接、最权威**的对照实现

## 仓库结构（高层导航）

```
claude-code/
├── CLAUDE.md              # 项目级 Claude 指令（必读，理解整体约定）
├── AGENTS.md              # agent 配置说明
├── DEV-LOG.md             # 开发日志（看作者的演进思路）
├── README.md              # 项目介绍
├── packages/              # 核心 monorepo 包
│   ├── mcp-client/        # ★ MCP client 实现
│   ├── agent-tools/       # agent 工具集
│   ├── builtin-tools/     # 内置工具
│   ├── workflow-engine/   # 工作流引擎
│   ├── remote-control-server/
│   └── ...
├── src/                   # 主应用源码（TypeScript）
│   ├── QueryEngine.ts     # ★ 推理引擎入口
│   ├── Task.ts            # ★ 任务抽象
│   ├── Tool.ts            # ★ 工具抽象
│   ├── coordinator/       # 协调器
│   ├── context.ts         # 上下文管理
│   ├── cost-tracker.ts    # 成本追踪
│   ├── assistant/         # assistant 角色
│   ├── buddy/             # buddy / 协作角色
│   ├── cli/               # 命令行入口
│   ├── commands/          # 命令系统
│   └── ...
├── spec/                  # 设计规范（feature/ 目录按日期组织）
├── docs/                  # 用户文档
├── teach-me/              # 教学示例
└── tests/                 # 测试
```

## 必读清单（优先级排序）

### 第一梯队（理解架构）

1. `CLAUDE.md`（29 KB）—— 项目灵魂文档，写了 Claude Code 的核心约定
2. `packages/mcp-client/` —— MCP 客户端实现，对应我们 task-6 的 M1
3. `src/QueryEngine.ts`、`src/Task.ts`、`src/Tool.ts` —— agentic loop 的核心抽象

### 第二梯队（实现细节）

4. `src/context.ts`、`src/context/` —— 上下文管理、compaction 策略
5. `src/coordinator/` —— 多 agent 协调器（subagent 模型）
6. `src/commands/` —— 命令系统设计（plugin 模式）
7. `packages/agent-tools/` —— 工具实现参考

### 第三梯队（演化历史）

8. `DEV-LOG.md`（52 KB）—— 作者开发日志，能看到设计决策的演进
9. `spec/` —— 规范化设计文档（按 feature 日期组织）
10. `teach-me/` —— 教学示例

## 对我们 task-6 设计的指导

读完这些文件，应该能回答：

| 我们的问题 | 去看哪里 |
|---|---|
| MCP server 怎么暴露工具 | `packages/mcp-client/` 的 server 实现 |
| Skill / Progressive Disclosure 怎么实现 | `src/context.ts`、`src/commands/` |
| Subagent 怎么隔离 context | `src/coordinator/`、`src/buddy/` |
| Agentic loop 的状态机 | `src/QueryEngine.ts`、`src/Task.ts` |
| 工具异常处理 | `packages/agent-tools/` |
| Context compaction | `src/context.ts` |
| Trace / 步骤记录 | `src/cost-tracker.ts`、`src/Task.ts` |

## 借鉴与简化策略

我们的 task-6 是「极简版」，所以从 Claude Code 提取**最核心的 10-15%**：

- ✅ MCP 客户端 + 服务端握手协议（`packages/mcp-client/`）
- ✅ 工具基类抽象（`src/Tool.ts`）
- ✅ Agent loop 主循环（`src/QueryEngine.ts`）
- ✅ 上下文压缩策略（`src/context.ts`）
- ❌ 完整的 monorepo 构建系统（用简单 Python 包代替）
- ❌ TUI 界面（CLI 即可）
- ❌ 多模型 fallback、cost tracking、telemetry 等生产级功能（先不做）

## 注意事项

- 仓库是 **TypeScript**，我们是 **Python**——只抄架构思想，不抄语法
- 仓库非常大，**不要尝试完整阅读**，按上面的梯队顺序看即可
- 实现 agent 应优先看 `packages/mcp-client/` 和 `src/QueryEngine.ts`
