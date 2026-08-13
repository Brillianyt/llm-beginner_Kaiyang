# Claude Code 源码学习笔记（最权威对照）

> 本笔记基于本地 `/mnt/workspace/llm-beginner_Kaiyang/task-6-coding-agent/reference/claude-code/` 源码阅读整理。
> 源码是 TypeScript（约 4.3MB 反编译/恢复版本），我们是 Python——**只抄架构思想，不抄语法**。

## 来源
- 仓库：`reference/claude-code/`（已 clone 至本地）
- 定位：Anthropic Claude Code CLI 的反编译/恢复实现（decompiled/restored version）
- 与本任务关系：**这是 task-6 的复刻目标**——每个核心模块都对应 README 里我们要做的能力三层栈

## 顶层结构（基于 CLAUDE.md）

```
claude-code/
├── CLAUDE.md              # 项目总纲（29KB）
├── AGENTS.md              # agent 配置说明（与 CLAUDE.md 内容高度重叠）
├── DEV-LOG.md             # 开发日志（52KB，演进历史）
├── packages/              # 17 个 workspace 包
│   ├── mcp-client/        # ★ MCP 客户端实现（对应 M1）
│   ├── builtin-tools/     # 60 个内置工具（对应 M1 工具集）
│   ├── agent-tools/       # agent 工具集
│   ├── workflow-engine/   # 工作流引擎
│   └── ...
├── src/
│   ├── QueryEngine.ts     # ★ 推理引擎入口（对应 M3 主循环）
│   ├── query.ts           # ★ 主 API 查询函数（对应 M3 主循环核心）
│   ├── Task.ts            # ★ 任务抽象（对应 subagent Task）
│   ├── Tool.ts            # ★ 工具抽象（对应 M1 工具接口）
│   ├── context.ts         # ★ 上下文管理（对应 M3 system prompt 构造）
│   ├── context/           # context 子模块（含 notifications、memory 等）
│   ├── coordinator/       # ★ subagent 协调器（对应 Subagent 设计）
│   ├── buddy/             # 协作角色
│   ├── commands/          # ★ 命令系统（plugin 模式参考，对应 Skill 加载）
│   ├── skills/            # Skill 加载器（bundled 技能）
│   ├── services/compact/  # ★ context compaction（对应 M3 长任务压缩）
│   └── ...
├── spec/                  # 设计规范（按 feature 日期组织）
└── docs/                  # 用户文档
```

## 关键模块 1：`packages/mcp-client/`（对应 M1）

### `interfaces.ts` —— 依赖注入接口

通过 TypeScript interface 实现**依赖倒置**（类似六边形架构的 ports）：

```typescript
export interface Logger { debug/info/warn/error(msg, ...args): void }
export interface FeatureGate { isEnabled(flag: string): boolean }
export interface AuthProvider { getTokens/refreshTokens(...): Promise<...> }
export interface HttpConfig { getUserAgent(): string; getSessionId?(): string }
export interface McpClientDependencies {
  logger: Logger                    // 必填
  analytics?: AnalyticsSink         // 全部可选
  featureGate?: FeatureGate
  auth?: AuthProvider
  httpConfig: HttpConfig            // 必填
  ...  // 其他全可选
}
```

**对 Python 的映射**：在 `src/mcp_client.py` 里建一组轻量协议类（`class Logger(Protocol)`）——不强制继承，只约束方法签名。CodingAgent 把自己的 `logging` wrapper 注入即可。

### `types.ts` —— 配置与状态 schema（zod）

```typescript
export const TransportType = z.enum(['stdio', 'sse', 'sse-ide', 'http', 'ws', 'sdk', 'claudeai-proxy'])
export const McpStdioServerConfigSchema = z.object({
  type: z.literal('stdio').optional(),
  command: z.string().min(1, 'Command cannot be empty'),
  args: z.array(z.string()).default([]),
  env: z.record(z.string(), z.string()).optional(),
})
export type ConnectedMCPServer = {
  client: Client
  name: string
  type: 'connected' | 'failed' | 'needs-auth' | 'pending' | 'disabled'
  capabilities: ServerCapabilities
  cleanup: () => Promise<void>
}
```

**关键设计**：用 zod schema 同时做**类型推导 + 运行时校验**。Python 里我们用 `pydantic` 做同样事情——`class McpStdioServerConfig(BaseModel): command: str; args: list[str] = []`。

### `manager.ts` —— McpManager 类（API 抽象）

```typescript
export interface McpManager {
  connect(name: string, config: McpServerConfig): Promise<MCPServerConnection>
  disconnect(name: string): Promise<void>
  disconnectAll(): Promise<void>
  getConnections(): Map<string, MCPServerConnection>
  getTools(serverName: string): CoreTool[]
  getAllTools(): CoreTool[]
  callTool(serverName: string, toolName: string, args: unknown): Promise<unknown>
  on<E extends keyof McpManagerEvents>(event, handler): void   // ★ 事件总线
}
const MCP_TIMEOUT_MS = 30_000
const MCP_REQUEST_TIMEOUT_MS = 60_000
```

**事件驱动模型**：`on('toolsChanged', handler)` / `on('error', handler)` / `on('authRequired', handler)`。我们的 Python 版可以用 `asyncio.Event` 或简单的回调列表实现。

### `execution.ts` —— callMcpTool 实现

```typescript
const DEFAULT_MCP_TOOL_TIMEOUT_MS = 100_000_000  // 约 27.8 小时（注释："effectively infinite"）

export interface CallToolOptions {
  client: ConnectedMCPServer
  tool: string
  args: Record<string, unknown>
  signal: AbortSignal                  // ★ AbortController 支持取消
  onProgress?: (data: {...}) => void   // ★ 进度回调
  timeoutMs?: number
}

export async function callMcpTool(options, deps): Promise<CallToolResult> {
  const progressInterval = setInterval(() => deps.logger.debug(`Tool still running`), 30_000)
  const result = await Promise.race([
    mcpClient.callTool({name, arguments, _meta}, CallToolResultSchema, {signal, timeout, onprogress}),
    createTimeoutPromise(serverName, tool, effectiveTimeout),
  ])
  if ('isError' in result && result.isError) throw new McpToolCallError(serverName, tool, errorDetails)
  return { content, _meta, structuredContent }
}
```

**对 Python 的映射**：
- `AbortSignal` → `asyncio.CancelledError`
- `setInterval` 进度日志 → `asyncio.create_task(periodic_log())`
- `Promise.race` → `asyncio.wait_for(coro, timeout=...)`
- `McpToolCallError` 自定义异常 + JSON-RPC error payload 结构

## 关键模块 2：`src/Tool.ts`（对应 M1 工具抽象）

803 行的大文件，核心是 `ToolUseContext` 类型——这是**所有工具调用时的统一上下文**：

```typescript
export type ToolUseContext = {
  options: {
    commands: Command[]
    debug: boolean
    mainLoopModel: string
    tools: Tools                         // ★ 可用工具列表（递归引用）
    mcpClients: MCPServerConnection[]    // ★ MCP 连接
    mcpResources: Record<string, ServerResource[]>
    maxBudgetUsd?: number
    customSystemPrompt?: string
    appendSystemPrompt?: string
    refreshTools?: () => Tools           // ★ MCP server 中途连接后重新拉取
  }
  abortController: AbortController
  readFileState: FileStateCache
  getAppState(): AppState
  setAppState(f: (prev: AppState) => AppState): void
  setAppStateForTasks?: ...              // ★ subagent 用，永远透传到 root store
  ...
}
```

**核心洞见**：
1. **工具上下文是 DI 容器**——所有 tool 调用都拿到同一份 context，tool 内部用 `context.getAppState()` 取数据
2. **`refreshTools` 回调**——MCP server 是 lazy connect 的，可能 agent 跑到一半才连上；要能在每步重新拉 tool list
3. **`setAppStateForTasks` vs `setAppState`**——主线 setAppState 对 async agent 是 no-op；任务相关的 state 必须用 `setAppStateForTasks` 透传。这正是**「subagent context 隔离」在状态层的体现**

## 关键模块 3：`packages/builtin-tools/src/tools/FileReadTool/FileReadTool.ts`（M1 工具实现参考）

构建工具用 `buildTool({...})` 工厂，所有元数据集中声明：

```typescript
export const FileReadTool = buildTool({
  name: FILE_READ_TOOL_NAME,
  searchHint: 'read files, images, PDFs, notebooks',
  maxResultSizeChars: 100_000,            // ★ 输出上限（防止 LLM context 爆）
  strict: true,                           // ★ schema 严格校验
  async description() { return DESCRIPTION },
  async prompt() { ... },                 // ★ 动态 prompt（可基于 limits 调整）
  get inputSchema() { return inputSchema() },
  get outputSchema() { return outputSchema() },
  userFacingName, getToolUseSummary,
  isReadOnly() { return true },            // ★ 权限系统用
  isConcurrencySafe() { return true },     // ★ 调度系统用（只读可并行）
  getPath({file_path}) { return file_path || getCwd() },
  validateInput({file_path, pages}, context) { ... },
  async checkPermissions(input, context): Promise<PermissionDecision> { ... },
  async call(input, context) { ... },      // ★ 实际执行
})
```

**对我们的启示**——Python 工具类应至少有这些属性：

```python
class CodingTool:
    name: str
    description: str
    input_schema: dict   # JSON Schema
    is_readonly: bool    # 决定可并行调度
    max_result_chars: int = 50_000
    def validate(self, args: dict) -> Optional[str]: ...    # 返回 None / 错误信息
    def call(self, args: dict, repo_root: Path) -> str: ...
```

## 关键模块 4：`src/query.ts` —— 主 agentic loop（核心！对应 M3）

**完整循环结构**：

```typescript
async function* queryLoop(params, consumedCommandUuids, consumedAutonomyCommands) {
  let state: State = { messages, toolUseContext, turnCount: 1, ... }
  
  while (true) {                          // ★ 无限循环，靠 continue + return 退出
    // 1. 迭代开始：拆解 state
    let { messages, toolUseContext, turnCount } = state
    
    // 2. 异步预取（skill discovery / extra tool prefetch）—— 不阻塞主循环
    const pendingSkillPrefetch = skillPrefetch?.startSkillDiscoveryPrefetch(null, messages, toolUseContext)
    
    // 3. 调用 API（流式）
    const stream = await streamApi(messagesForQuery, ...)
    const assistantMessages = []; const toolUseBlocks = []
    for await (const event of stream) {
      // 处理 message_start / content_block_start / message_delta / message_stop
      // 收集 text blocks + tool_use blocks
    }
    
    // 4. 终止条件 1：纯文本输出（无 tool_use）→ 完成
    if (toolUseBlocks.length === 0) return { reason: 'completed' }
    
    // 5. 工具执行：streaming 或批量
    const toolResults = await streamingToolExecutor
      ? await streamingToolExecutor(toolUseBlocks, toolUseContext)
      : await batchExecuteTools(toolUseBlocks, toolUseContext)
    
    // 6. 终止条件 2：用户中断
    if (toolUseContext.abortController.signal.aborted) {
      return { reason: 'aborted_tools' }
    }
    
    // 7. 终止条件 3：Stop hook 阻止继续
    if (shouldPreventContinuation) return { reason: 'hook_stopped' }
    
    // 8. 终止条件 4：超过 maxTurns
    const nextTurnCount = turnCount + 1
    if (maxTurns && nextTurnCount > maxTurns) {
      yield createAttachmentMessage({type: 'max_turns_reached', ...})
      return { reason: 'max_turns', turnCount: nextTurnCount }
    }
    
    // 9. 准备下一轮：state 整体替换
    state = { messages: messagesForQuery.concat(assistantMessages, toolResults), ..., turnCount: nextTurnCount }
  }  // while (true)
}
```

**终止原因枚举（Terminal 类型）**：
- `'completed'` — 正常完成（LLM 输出纯文本回复）
- `'max_turns'` — 步数耗尽
- `'aborted_streaming' / 'aborted_tools'` — 用户中断
- `'hook_stopped'` — Stop hook 拦截
- `'prompt_too_long'` — context 撑爆
- `'model_error'` — API 错误
- `'image_error'` — 图片处理错误

**对我们的启示**（Python 翻译）：
```python
@dataclass
class AgentState:
    messages: list[dict]
    turn_count: int = 1
    patch_diff: str = ""
    tests_passed: bool = False
    done_reason: Optional[str] = None

def run(self, repo_path, issue) -> Trace:
    state = AgentState(messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": issue}])
    while True:
        resp = self.client.chat(messages=state.messages, tools=TOOL_SCHEMAS)
        if not resp.tool_calls:
            return self._build_trace(state, done_reason="completed")
        for call in resp.tool_calls:
            obs = self._execute_tool(call, repo_path)
            state.messages.append({"role": "tool", "content": obs})
        state.turn_count += 1
        if state.turn_count > self.max_turns:
            return self._build_trace(state, done_reason="max_turns")
        if state.tests_passed:                # 显式 done 信号
            return self._build_trace(state, done_reason="tests_passed")
```

## 关键模块 5：`src/QueryEngine.ts`（高级编排层，对应 M4 Trace）

把 `query()` 的异步生成器再包一层，处理 conversation 生命周期：

```typescript
export class QueryEngine {
  private config: QueryEngineConfig
  private mutableMessages: Message[]       // ★ 跨 turn 持久化
  private abortController: AbortController
  private permissionDenials: SDKPermissionDenial[]   // 拒绝审计
  private totalUsage: NonNullableUsage               // 累计 token 消耗
  private readFileState: FileStateCache              // 文件读取快照（用于 diff）
  
  constructor(config: QueryEngineConfig) { ... }
  
  async *submitMessage(prompt, options): AsyncGenerator<SDKMessage, void, unknown> {
    // 1. 每 turn 清空 permissionDenials
    // 2. 拉 system prompt（fetchSystemPromptParts）
    // 3. 跑 query()，收集消息
    // 4. 按消息类型分发（assistant / user / compact_boundary / attachment）
    // 5. 关键：yield 'result' 消息（含 num_turns、duration_ms、total_cost_usd、usage、permission_denials）
    // 6. 处理 max_turns_reached → 直接 return
  }
}
```

**Trace 结构洞察**（来自 SDKMessage）：
```typescript
yield {
  type: 'result',
  subtype: 'success' | 'error_max_turns' | ...,
  duration_ms: number,
  duration_api_ms: number,
  is_error: boolean,
  num_turns: number,
  stop_reason: string | null,
  session_id: string,
  total_cost_usd: number,
  usage: { input_tokens, output_tokens, cache_read_input_tokens, ... },
  modelUsage: Record<string, ModelUsage>,
  permission_denials: SDKPermissionDenial[],
  errors: string[],
  uuid: string,
}
```

**Python 翻译**——我们 `Trace` 字典要至少包含：
```python
{
    "steps": [...],                  # 必需
    "patch": "...",                  # 必需（unified diff）
    "tests_passed": True/False,      # 必需
    "duration_ms": int,              # 推荐
    "turn_count": int,               # 推荐
    "done_reason": "tests_passed"|"max_turns"|"completed",  # 推荐
    "tool_calls": [{"name", "args", "obs", "duration_ms"}, ...],
    "summary": "...",                # 推荐
}
```

## 关键模块 6：`src/services/compact/compact.ts`（对应 M3 长任务压缩）

核心函数 `compactConversation(messages, context, ...)` —— 用 LLM 生成历史摘要替换旧消息：

```typescript
const POST_COMPACT_TOKEN_BUDGET = 50_000                  // ★ 压缩后可用 token 上限
const POST_COMPACT_MAX_FILES_TO_RESTORE = 5              // ★ 最多还原 5 个文件
const POST_COMPACT_MAX_TOKENS_PER_FILE = 5_000           // ★ 每个文件最多 5K token
const POST_COMPACT_MAX_TOKENS_PER_SKILL = 5_000          // ★ 每个 Skill 最多 5K
const POST_COMPACT_SKILLS_TOKEN_BUDGET = 25_000          // ★ Skill 总预算

export async function compactConversation(messages, context, cacheSafeParams, ...): Promise<CompactionResult> {
  const preCompactTokenCount = tokenCountWithEstimation(messages)
  
  // 1. 触发 PreCompact hook
  await executePreCompactHooks({trigger: isAutoCompact ? 'auto' : 'manual'}, ...)
  
  // 2. 用 LLM 生成摘要（forked-agent 路径可复用 prompt cache）
  const compactPrompt = getCompactPrompt(customInstructions)
  for (;;) {
    summaryResponse = await streamCompactSummary({messages, summaryRequest, ...})
    summary = getAssistantMessageText(summaryResponse)
    if (!summary?.startsWith(PROMPT_TOO_LONG_ERROR_MESSAGE)) break
    // 3. 如果摘要请求本身超长，截断最早的几条 API round 重试
    ptlAttempts++
    messagesToSummarize = truncateHeadForPTLRetry(messagesToSummarize, ...)
  }
  
  // 4. 构造 post-compact 消息列表（系统摘要 + 关键文件附件 + Skill 精简版）
  const messages = buildPostCompactMessages(result)
  return { messages, compactMetadata: {preCompactTokenCount, ...} }
}
```

**对我们的简化启示**——Qwen2.5-Coder-7B 用 32K context，没必要做完整 LLM 摘要压缩。我们只需：
```python
def maybe_compact(messages, max_tokens=30000):
    total = sum(token_count(m) for m in messages)
    if total < max_tokens * 0.8:
        return messages
    # 简单策略：保留 system + 前 3 条 user/assistant + 最后 5 条 tool/assistant
    # 把中间内容折叠成一句话："[历史已压缩，详见 trace log]"
    return [messages[0]] + messages[1:4] + [{"role": "system", "content": "[compacted]"}] + messages[-5:]
```

## 关键模块 7：`src/skills/loadSkillsDir.ts`（对应 M2 Skill 加载）

**渐进式披露实现**（约 480 行）：

```typescript
async function loadSkillsFromSkillsDir(basePath, source): Promise<SkillWithPath[]> {
  const entries = await fs.readdir(basePath)
  return Promise.all(entries.map(async entry => {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) return null  // ★ 只接受目录形式
    const skillDirPath = join(basePath, entry.name)
    const skillFilePath = join(skillDirPath, 'SKILL.md')
    const content = await fs.readFile(skillFilePath, 'utf-8')
    
    const { frontmatter, content: markdownContent } = parseFrontmatter(content, skillFilePath)
    const skillName = entry.name
    const parsed = parseSkillFrontmatterFields(frontmatter, markdownContent, skillName)
    
    return {
      skill: createSkillCommand({
        skillName, markdownContent,
        description,            // ★ 来自 frontmatter，用于 list 阶段
        allowedTools: [],        // ★ Skill 可限制可用的 tool 集合
        whenToUse: undefined,    // ★ 比 description 更详细的触发条件
        contentLength: markdownContent.length,  // ★ 用于预算
        // ...
        async getPromptForCommand(args, toolUseContext) {
          // ★ 按需加载：只有在 LLM 真要触发该 skill 时才执行
          let finalContent = baseDir ? `Base directory: ${baseDir}\n\n${markdownContent}` : markdownContent
          finalContent = substituteArguments(finalContent, args, true, argumentNames)
          return [{ type: 'text', text: finalContent }]
        },
      }),
      filePath: skillFilePath,
    }
  }))
}
```

**关键设计**：
1. **目录形式是唯一支持格式**——`skill-name/SKILL.md`，单文件 SKILL.md 不被支持
2. **`allowed-tools` frontmatter**——Skill 可以声明自己能用哪些 tool（白名单），权限沙箱
3. **`contentLength` 预算**——避免某个 skill 把 context 撑爆
4. **deduplication by realpath**——同一文件被多个路径引用（symlink）只加载一次
5. **`whenToUse` vs `description`**——description 是 LLM 看的；whenToUse 是更强的触发条件（如「当用户说 review PR」）

## 关键模块 8：`src/coordinator/`（对应 Subagent 设计）

`coordinatorMode.ts` 决定是否进入 coordinator 模式（CLAUDE_CODE_COORDINATOR_MODE=1）；

`workerAgent.ts` 定义了 **worker agent**：

```typescript
const WORKER_AGENT: BuiltInAgentDefinition = {
  agentType: 'worker',
  whenToUse: 'Worker agent for coordinator mode. Executes research, implementation, and verification tasks autonomously with the full standard tool set.',
  tools: getWorkerTools(),  // ★ 排除 INTERNAL_ORCHESTRATION_TOOLS（TEAM_CREATE / SEND_MESSAGE / SYNTHETIC_OUTPUT）
  source: 'built-in',
  getSystemPrompt: () =>
    `You are a worker agent spawned by a coordinator. Your job is to complete the task described in the prompt thoroughly and report back with a concise summary of what you did and what you found.
    
    Guidelines:
    - Complete the task fully — don't leave it half-done, but don't gold-plate either.
    - Use tools proactively: read files, search code, run commands, edit files.
    - Be thorough in research: check multiple locations, consider different naming conventions.
    - For implementation: make targeted changes, run tests to verify, commit if appropriate.
    - Report back with actionable findings — the coordinator will synthesize your results.
    - If you encounter errors, investigate and attempt to fix them before reporting failure.
    - NEVER create documentation files unless explicitly instructed.`,
}

const INTERNAL_ORCHESTRATION_TOOLS = new Set([
  TEAM_CREATE_TOOL_NAME, TEAM_DELETE_TOOL_NAME,
  SEND_MESSAGE_TOOL_NAME, SYNTHETIC_OUTPUT_TOOL_NAME,
])
```

**对 Subagent 设计的核心启示**：
1. **独立 system prompt**——worker 有自己的角色定位（"be thorough, report back concisely"）
3. **工具子集白名单**——明确排除 "orchestration tools"（避免 worker 自己也开 subagent 失控）
4. **返回摘要不返回 trace**——"Report back with actionable findings — the coordinator will synthesize your results"

## 关键模块 9：`src/context.ts`（对应 M3 System Prompt 构造）

构造对话上下文：

```typescript
export const getSystemContext = memoize(async () => {
  const gitStatus = await getGitStatus()    // 当前分支、状态、最近 5 条 commit
  return {
    ...(gitStatus && { gitStatus }),
    ...(feature('BREAK_CACHE_COMMAND') && injection ? {cacheBreaker: `[CACHE_BREAKER: ${injection}]`} : {}),
  }
})

export const getUserContext = memoize(async () => {
  const claudeMd = await getClaudeMds(getMemoryFiles())    // 扫描 CLAUDE.md 文件
  return {
    ...(claudeMd && { claudeMd }),
    currentDate: `Today's date is ${getLocalISODate()}.`,
  }
})
```

**给我们的启示**——System prompt 应包含：
- 当前日期（让 LLM 知道时间，避免幻觉旧版本）
- 工作目录（cwd）路径
- 可用工具列表（自动从 tool registry 拉）
- 工作流指令（"你的任务是修复 bug，最后用 submit_patch 工具提交"）

## CLAUDE.md / AGENTS.md / DEV-LOG.md 三件套的设计哲学

### CLAUDE.md（项目级 agent 指令）
- 强制 `bun run precheck` 必须零错误（typecheck + lint + test）
- Conventional Commits 提交规范
- Runtime 用 Bun（不是 Node.js）
- 17 个 workspace packages + monorepo 构建
- **19 个 feature flags**——通过 `import { feature } from 'bun:bundle'` 控制
- 7 个 API 兼容层（firstParty / bedrock / vertex / foundry / openai / gemini / grok）
- 关键工作流：`src/cli.tsx` → `src/main.tsx` → `src/QueryEngine.ts` → `src/query.ts`

### DEV-LOG.md（演进历史，可读）
- 52KB 篇幅，含决策由来（"为什么切到 Vite build"、"为什么 mock.module 会污染同进程"）
- 暴露了不少工程教训：
  - **JSC/Bun 全量加载导致 RSS 暴涨**——必须做代码分割
  - **mock.module 是进程全局**（last-write-wins），不可预测测试执行顺序
  - **`as any` 禁止**——用 `as unknown as SpecificType` 双重断言
  - **`feature()` 只能用在 if/三元条件位置**——Bun 编译器限制

### AGENTS.md
- 内容与 CLAUDE.md 大量重叠（双份约束文件）

## 我们应该借鉴什么（具体清单）

### ✅ 抄架构思想
1. **依赖注入接口**（`McpClientDependencies`）—— 在 Python 里用 `Protocol` 类替代
2. **配置用 schema + 类型推导**—— Python 用 `pydantic.BaseModel` 替代 zod
3. **`buildTool({name, schema, isReadOnly, isConcurrencySafe, validateInput, call})` 工厂模式**—— Python 用 dataclass + classmethod
4. **状态机驱动的 query loop**—— while true + 多个 `return {reason}` 出口
5. **Terminal reason 枚举**—— 显式记录「为什么停」
6. **Skill 渐进式披露**—— list 阶段只暴露 name+description，load 阶段才读 markdownContent
7. **Worker agent 工具白名单**—— 用 `allowed_tools` 参数限定 subagent 可调工具
8. **Pre/Post hook 系统**—— 抽象出 `register_hook('PreToolUse', callback)` 接口

### ❌ 不抄（避免过度工程化）
1. **完整 Ink UI** —— 我们只要 CLI
2. **17 个 workspace packages** —— 单包足够
3. **7 个 API provider** —— 我们只支持 OpenAI 兼容
4. **Langfuse telemetry** —— 加分项，先不做
5. **19 个 feature flags** —— 我们 0 个
6. **Async agent 状态总线** —— 直接 return summary 就行
7. **云端 RCP（Remote Control Server）** —— 完全用不到

## Python 实现映射表

| Claude Code 文件 | 我们的 Python 文件 | 行数估算 |
|---|---|---|
| `packages/mcp-client/manager.ts` | `src/mcp_client.py` | ~150 |
| `packages/mcp-client/execution.ts` | `src/mcp_client.py` (call_tool 函数) | ~80 |
| `packages/builtin-tools/.../FileReadTool.ts` | `src/tools/read_file.py` 等 | 每个 ~30 |
| `src/Tool.ts` (ToolUseContext) | `src/tool_context.py` | ~50 |
| `src/query.ts` (queryLoop) | `src/agent.py` (CodingAgent.run) | ~200 |
| `src/QueryEngine.ts` | `src/agent.py` 同上 | 合并 |
| `src/services/compact/compact.ts` | `src/agent.py` 的 `_maybe_compact()` | ~30 |
| `src/context.ts` | `src/prompt_builder.py` | ~80 |
| `src/coordinator/workerAgent.ts` | `src/subagents/code_search.py` 等 | 每个 ~100 |
| `src/skills/loadSkillsDir.ts` | `src/skill_loader.py` | ~80 |
| `src/Tool.ts` + `src/types/message.ts` | `src/trace.py` | ~50 |

合计：约 1000-1200 行 Python，**足够 5-6 周工作量**。

## 主要参考文件路径

- `reference/claude-code/CLAUDE.md` —— 项目总纲（必读）
- `reference/claude-code/AGENTS.md` —— agent 配置（与 CLAUDE.md 重叠）
- `reference/claude-code/DEV-LOG.md` —— 开发日志（演进思路）
- `reference/claude-code/packages/mcp-client/interfaces.ts` —— DI 接口（★最清晰）
- `reference/claude-code/packages/mcp-client/manager.ts` —— McpManager 类
- `reference/claude-code/packages/mcp-client/execution.ts` —— callMcpTool
- `reference/claude-code/packages/builtin-tools/src/tools/FileReadTool/FileReadTool.ts` —— 工具实现样板
- `reference/claude-code/src/Tool.ts` —— ToolUseContext 类型
- `reference/claude-code/src/query.ts` (queryLoop, line 393-2057) —— 主循环
- `reference/claude-code/src/QueryEngine.ts` —— 高级编排
- `reference/claude-code/src/services/compact/compact.ts` —— 压缩策略
- `reference/claude-code/src/context.ts` —— system prompt 构造
- `reference/claude-code/src/skills/loadSkillsDir.ts` —— Skill 加载（★值得细读）
- `reference/claude-code/src/coordinator/coordinatorMode.ts` + `workerAgent.ts` —— subagent 设计