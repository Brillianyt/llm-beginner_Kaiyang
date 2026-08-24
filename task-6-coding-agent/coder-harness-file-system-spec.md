# 文件系统工具集规格 — 取自 Claude Code

> 来源目录：`/mnt/d/.reasearch_on_CCB/claude-code`
> 覆盖工具：`Read` / `Edit` / `Write` / `Glob` / `Grep` / `Bash`（含 git 提交流程）
> 适用范围：构建一个能让 AI 自主打开/修改/检索文件并提交 git 的小型 Coder Harness

---

## 0. 工具接口契约

工具的协议级接口（取自 `packages/agent-tools/src/types.ts`），每个工具都必须实现：

```ts
interface CoreTool<Input, Output, P, Context> {
  // —— 身份 ——
  readonly name: string                 // 工具注册名（模型看到的）
  aliases?: string[]
  searchHint?: string                  // 工具检索/分类用的关键词

  // —— 输入/输出 schema ——
  readonly inputSchema: Input          // Zod schema（也可给 inputJSONSchema 走 MCP）
  readonly inputJSONSchema?: ToolInputJSONSchema
  outputSchema?: z.ZodType<unknown>

  // —— 执行 ——
  call(args, context, canUseTool, parentMessage, onProgress?): Promise<ToolResult<Output>>

  // —— 描述/系统提示词 ——
  description(input, { isNonInteractiveSession, toolPermissionContext, tools }): Promise<string>
  prompt({ getToolPermissionContext, tools, agents, allowedAgentTypes? }): Promise<string>

  // —— 行为属性 ——
  isConcurrencySafe(input): boolean        // 是否可与同工具其他实例并发
  isEnabled(): boolean                     // 是否对当前会话可见
  isReadOnly(input): boolean               // 是否只读（决定是否走沙箱/审批快路径）
  isDestructive?(input): boolean           // 是否有破坏性
  isOpenWorld?(input): boolean             // 是否对外副作用（如 HTTP）
  interruptBehavior?(): 'cancel' | 'block'
  requiresUserInteraction?(): boolean

  // —— MCP/LSP 标记 ——
  isMcp?: boolean
  isLsp?: boolean
  shouldDefer?: boolean                    // 启动时不立即注入，只在用到时才拉
  alwaysLoad?: boolean                     // 永远注入

  // —— 输入校验 + 权限 ——
  validateInput?(input, context): Promise<{result:true}|{result:false;message;errorCode}>
  checkPermissions(input, context): Promise<{behavior:'allow';updatedInput}|{behavior:'deny';message}|{behavior:'passthrough'}>

  // —— 实用工具 ——
  inputsEquivalent?(a, b): boolean
  getPath?(input): string                          // 用于权限规则匹配的目标路径
  toAutoClassifierInput(input): unknown            // 用于自动分类器路由
  backfillObservableInput?(input): void

  // —— 输出处理 ——
  maxResultSizeChars: number                       // 输出最大字符数（超出截断/落盘）
  userFacingName(input): string
  mapToolResultToToolResultBlockParam(content, toolUseID): any
  isResultTruncated?(output): boolean
  getToolUseSummary?(input): string | null
  getActivityDescription?(input): string | null
  isTransparentWrapper?(): boolean
  isSearchOrReadCommand?(input): { isSearch, isRead, isList? }
}
```

**关键设计点（搬过来即可）：**

- **Prompt 与 Schema 分开**：`description()` 给模型看一句简介，`prompt()` 返回完整系统提示词（含行为规约、安全条款、跨工具协作说明）。
- **行为属性驱动控制流**：`isReadOnly` / `isDestructive` / `isOpenWorld` 三个布尔决定沙箱边界、是否需要审批、是否计入历史。
- **权限系统可插拔**：`checkPermissions` 返回 `allow / deny / passthrough`，未来加 per-tool 规则不用动工具实现。
- **`getPath`**：让一个统一的权限匹配器能用文件路径 wildcard 规则来放行/拦截所有"访问某个路径"的工具（Read/Edit/Write/Glob/Grep 都实现）。

---

## 1. `Read` — 读文件

### 1.1 系统提示词

```text
Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Results are returned using cat -n format, with line numbers starting at 1
- This tool allows Claude Code to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as Claude Code is a multimodal LLM.
- This tool can read PDF files (.pdf). For large PDFs (more than 10 pages), you MUST provide the pages parameter to read specific page ranges (e.g., pages: "1-5"). Reading a large PDF without the pages parameter will fail. Maximum 20 pages per request.
- This tool can read Jupyter notebooks (.ipynb files) and returns all cells with their outputs, combining code, text, and visualizations.
- This tool can only read files, not directories. To read a directory, use an ls command via the Bash tool.
- You will regularly be asked to read screenshots. If the user provides a path to a screenshot, ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
```

### 1.2 Input schema

```ts
z.strictObject({
  file_path: z.string().describe('The absolute path to the file to read'),
  offset:    z.number().int().nonnegative().optional()
              .describe('The line number to start reading from. Only provide if the file is too large to read at once'),
  limit:     z.number().int().positive().optional()
              .describe('The number of lines to read. Only provide if the file is too large to read at once.'),
  pages:     z.string().optional()
              .describe('Page range for PDF files (e.g., "1-5", "3", "10-20"). Maximum 20 pages per request.'),
})
```

### 1.3 关键常量 / 限制

- `MAX_LINES_TO_READ = 2000`（单次最多读 2000 行）
- 默认 `maxSizeBytes = 256KB`（总文件大小，先于读取校验，超限直接抛错）
- 默认 `maxTokens = 25000`（实际输出 token，超限抛错）
- `FILE_UNCHANGED_STUB`：文件没变化时返回的占位文本
  ```
  File unchanged since last read. The content from the earlier Read tool_result in this conversation is still current — refer to that instead of re-reading.
  ```
- 支持图片内联渲染（多模态 LLM 友好）
- 支持 PDF（>10 页必须用 `pages`）
- 支持 `.ipynb`（合并 cell + outputs）

---

## 2. `Edit` — 精确字符串替换

### 2.1 系统提示词

```text
Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + tab. Everything after that is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.
- The file_path must be a file path, not a directory path. If the path resolves to an existing directory, the tool will reject it. Use a path that points to an existing file.
```

### 2.2 Input schema

```ts
z.strictObject({
  file_path:   z.string().describe('The absolute path to the file to modify'),
  old_string:  z.string().describe('The text to replace'),
  new_string:  z.string().describe('The text to replace it with (must be different from old_string)'),
  replace_all: z.boolean().default(false).optional()
                  .describe('Replace all occurrences of old_string (default false)'),
})
```

### 2.3 关键约束

- **必须先 Read**：未读过的文件尝试 Edit → 报错。强制模型先建立上下文。
- **`old_string` 必须唯一**：不唯一就 fail，强制给上下文或改用 `replace_all`。
- **line-number prefix 不进 match**：`Read` 输出形如 `   42\tcontent`，匹配 `old_string` 时要把 `   42\t` 这一段剔除。
- `replace_all`：批量重命名（变量、import 等）。
- `FILE_UNEXPECTEDLY_MODIFIED_ERROR`：用户在工具运行期间动了文件，下次必须重新读。
  ```
  File has been unexpectedly modified. Read it again before attempting to write it.
  ```
- 返回 `gitDiff`（如果文件在 repo 内）：含 patch、additions、deletions、status = `modified|added`。

---

## 3. `Write` — 创建/整体覆盖

### 3.1 系统提示词

```text
Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- Prefer the Edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.
- The file_path must be a distinct file path, not a directory path. If the path resolves to an existing directory, the tool will reject it with a clear error message. Use a path that includes a filename with an appropriate extension (e.g., `my-docs/analysis/api/report.md`).
```

### 3.2 Input schema

```ts
z.strictObject({
  file_path: z.string().describe('The absolute path to the file to write (must be absolute, not relative)'),
  content:   z.string().describe('The content to write to the file'),
})
```

### 3.3 关键约束

- **覆盖式写入**：覆盖现有文件 → 与 Edit 互补（差量 vs 全量）。
- **先 Read 再覆盖**：保护模型不盲目覆写未知状态。
- **禁止路径当目录**：传目录路径会被拒，避免误删。
- 返回 `type: 'create' | 'update'` 让 UI 区分新建 vs 覆盖。

---

## 4. `Glob` — 文件名模式匹配

### 4.1 描述

```text
- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
```

### 4.2 Input schema

```ts
z.strictObject({
  pattern: z.string().describe('The glob pattern to match files against'),
  path:    z.string().optional()
             .describe('The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter "undefined" or "null" - simply omit it for the default behavior.'),
})
```

### 4.3 行为属性

- `isReadOnly: true`
- `isConcurrencySafe: true`
- `maxResultSizeChars: 100_000`
- `isSearchOrReadCommand: { isSearch: true, isRead: false }`
- 默认上限 100 文件，结果按修改时间排序。
- 输入校验会做 `expandPath`（`~` / 相对路径展开）+ UNC 路径短路（安全）。
- `toAutoClassifierInput` 只取 `pattern` 字符串，自动路由用。

---

## 5. `Grep` — 内容搜索（ripgrep 包装）

### 5.1 描述

```text
A powerful search tool built on ripgrep

Usage:
- ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.
- Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
- Filter files with glob parameter (e.g., "*.js", "*.{ts,tsx}") or type parameter (e.g., "js", "py", "rust")
- Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
- Use Agent tool for open-ended searches requiring multiple rounds
- Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)
- Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`
```

### 5.2 Input schema

```ts
z.strictObject({
  pattern:      z.string().describe('The regular expression pattern to search for in file contents'),
  path:         z.string().optional().describe('File or directory to search in (rg PATH). Defaults to cwd.'),
  glob:         z.string().optional().describe('Glob pattern to filter files (rg --glob). e.g. "*.js", "*.{ts,tsx}"'),
  output_mode:  z.enum(['content','files_with_matches','count']).optional()
                   .describe('Output mode: "content" shows matching lines, "files_with_matches" shows file paths (default), "count" shows match counts.'),
  '-B':         z.number().optional().describe('Lines of context BEFORE match (rg -B). Requires output_mode: "content".'),
  '-A':         z.number().optional().describe('Lines of context AFTER match (rg -A). Requires output_mode: "content".'),
  '-C':         z.number().optional().describe('Alias for context.'),
  context:      z.number().optional().describe('Lines of context before and after match (rg -C).'),
  '-n':         z.boolean().optional().describe('Show line numbers (rg -n). Defaults to true.'),
  '-i':         z.boolean().optional().describe('Case insensitive (rg -i).'),
  type:         z.string().optional().describe('File type (rg --type): js, py, rust, go, java…'),
  head_limit:   z.number().optional().describe('Limit output to first N lines/entries. Defaults to 250. Pass 0 for unlimited.'),
  offset:       z.number().optional().describe('Skip first N lines/entries before applying head_limit.'),
  multiline:    z.boolean().optional().describe('Enable multiline mode (rg -U --multiline-dotall). Default: false.'),
})
```

### 5.3 默认与排除

- 自动排除 VCS 目录：`.git / .svn / .hg / .bzr / .jj / .sl`
- `DEFAULT_HEAD_LIMIT = 250`（无界内容搜索会塞爆上下文）
- `head_limit: 0` 表示无界（高开销，慎用）
- ripgrep 而非 grep：literal braces 需要 `interface\{\}` 转义

---

## 6. `Bash` — 命令执行 & Git 工作流

### 6.1 系统提示词（精简版骨架）

```text
Executes a given bash command and returns its output.

The working directory persists between commands, but shell state does not. The shell environment is initialized from the user's profile (bash or zsh).

IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool:
- File search: Use Glob (NOT find or ls)
- Content search: Use Grep (NOT grep or rg)
- Read files: Use Read (NOT cat/head/tail)
- Edit files: Use Edit (NOT sed/awk)
- Write files: Use Write (NOT echo >/cat <<EOF)
- Communication: Output text directly (NOT echo/printf)

# Instructions
- If your command will create new directories or files, first run `ls` to verify the parent directory exists.
- Always quote file paths that contain spaces with double quotes.
- Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`.
- Optional timeout (up to <MAX>ms). Default <DEF>ms.
- You can use `run_in_background` to run a command in the background; you'll be notified on completion. Don't append `&`.
- Multiple commands:
  - Independent commands → multiple Bash calls in ONE message in parallel.
  - Dependent commands → chain with `&&`.
  - Use `;` only when you don't care if earlier commands fail.
  - DO NOT use newlines to separate commands.
- For git commands:
  - Prefer creating a new commit rather than amending.
  - Before destructive ops (reset --hard, push --force, checkout --), check for safer alternatives.
  - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign) unless explicitly asked.
- Avoid unnecessary sleep:
  - Do not sleep between commands that can run immediately.
  - Use `run_in_background` for long-running work; don't poll.
  - Don't retry failing commands in a sleep loop.

## Command sandbox (optional)
By default, your command will be run in a sandbox...
[sandbox list of allow/deny paths, network hosts, override policy]

# Git operations — see §7
```

### 6.2 Input schema

```ts
z.strictObject({
  command:                   z.string().describe('The command to execute'),
  timeout:                   z.number().optional().describe('Optional timeout in milliseconds (max <MAX>).'),
  description:               z.string().optional().describe(`Clear, concise description of what this command does in active voice.
                                                            Never use words like "complex" or "risk" — just describe what it does.
                                                            For simple commands (git, npm, standard CLI tools), keep it brief (5-10 words):
                                                              - ls → "List files in current directory"
                                                              - git status → "Show working tree status"
                                                              - npm install → "Install package dependencies"
                                                            For obscured flags / pipes, add context.`),
  run_in_background:         z.boolean().optional().describe('Set to true to run this command in the background. Use Read to read the output later.'),
  dangerouslyDisableSandbox: z.boolean().optional().describe('Set true to dangerously override sandbox mode.'),
})
```

> 内部还有 `_simulatedSedEdit`（隐藏字段，仅 SedEditPermissionRequest 设置），模型 schema 里省略，避免模型借此绕过沙箱。

### 6.3 输出 schema

```ts
z.object({
  stdout:                         z.string(),
  stderr:                         z.string(),
  rawOutputPath:                  z.string().optional(),
  interrupted:                    z.boolean(),
  isImage:                        z.boolean().optional(),
  backgroundTaskId:               z.string().optional(),
  backgroundedByUser:             z.boolean().optional(),
  assistantAutoBackgrounded:      z.boolean().optional(),
  dangerouslyDisableSandbox:      z.boolean().optional(),
  returnCodeInterpretation:       z.string().optional(),
  noOutputExpected:               z.boolean().optional(),
  structuredContent:              z.array(z.any()).optional(),
  persistedOutputPath:            z.string().optional(),  // 输出过大时落盘的路径
  persistedOutputSize:            z.number().optional(),
})
```

### 6.4 关键限制

- 默认超时（`<DEF>` ms / 分钟），最大超时（`<MAX>` ms / 分钟），可显式 `timeout` 覆盖。
- `description` 字段是**必须**的，给 UI 渲染、未读不执行。这是 harnsss 的"安全展示"。
- 沙箱模式默认开启：`Filesystem: {read:{denyOnly, allowWithinDeny}, write:{allowOnly, denyWithinAllow}}` + `Network: {allowedHosts, deniedHosts, allowUnixSockets}`。
- 模型被明令禁止"自绕过沙箱"，除非（a）用户显式要求，或（b）看到明确的"沙箱导致失败"的证据，且调用会向用户弹出权限请求。

---

## 7. Git 工作流（嵌在 Bash 的 prompt 里）

### 7.1 Git Safety Protocol（完整照搬）

```text
Git Safety Protocol:
- NEVER update the git config
- NEVER run destructive git commands (push --force, reset --hard, checkout ., restore ., clean -f, branch -D) unless the user explicitly requests these actions
- NEVER skip hooks (--no-verify, --no-gpg-sign, etc) unless the user explicitly requests it
- NEVER run force push to main/master, warn the user if they request it
- CRITICAL: Always create NEW commits rather than amending, unless the user explicitly requests a git amend. When a pre-commit hook fails, the commit did NOT happen — so --amend would modify the PREVIOUS commit, which may result in destroying work or losing previous changes. Instead, after hook failure, fix the issue, re-stage, and create a NEW commit
- When staging files, prefer adding specific files by name rather than using "git add -A" or "git add .", which can accidentally include sensitive files (.env, credentials) or large binaries
- NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive
```

### 7.2 Commit 标准流程

```text
1. Run the following bash commands in parallel:
   - git status (never use -uall; memory issues on large repos)
   - git diff (both staged and unstaged)
   - git log (recent commit messages, for style)
2. Analyze all staged changes; draft a commit message:
   - Summarize the nature (new feature / enhancement / bug fix / refactor / test / docs)
   - Don't commit secrets (.env, credentials.json); warn if requested
   - Concise (1-2 sentences), focus on the "why", not "what"
   - Match repo's existing message style
3. Run in parallel:
   - Add relevant untracked files by NAME (not `-A` / `.`)
   - Create the commit (HEREDOC for the message)
   - Run git status AFTER the commit, to verify success
4. If commit fails due to pre-commit hook: FIX the issue and create a NEW commit (do NOT amend)
```

HEREDOC 模板：

```bash
git commit -m "$(cat <<'EOF'
   Commit message here.

   Co-Authored-By: ...

   EOF
   )"
```

### 7.3 PR / 创建流程

```text
1. parallel: git status, git diff, track-remote check, git log + git diff <base>...HEAD
2. draft title (≤70 chars) + body (use body for details)
3. parallel: create branch if needed, push -u if needed, gh pr create with HEREDOC body

gh pr create --title "the pr title" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points>

## Test plan
[Bulleted markdown checklist of TODOs for testing the pull request...]

EOF
)"

DO NOT use TodoWrite or Agent tools.
Return the PR URL when done.
```

---

## 8. 跨工具协同 & 行为约定

把这些写成 harness 的 system prompt 一节：

```text
- Prefer the dedicated tools over Bash for file/content operations:
  Read > cat/head/tail
  Edit > sed/awk
  Write > echo >/cat <<EOF
  Glob > find/ls
  Grep > grep/rg
  Output text directly > echo/printf

- Always use absolute paths for file_path.

- Before calling Edit or Write on an existing file, you MUST call Read on that file at least once in the current conversation. The tool will reject the call otherwise.

- Edit requires old_string to be unique. If it isn't, either expand context or set replace_all=true.

- For long files, read with offset + limit instead of loading the whole file at once.

- Run independent commands in parallel (multiple tool calls in one message). Chain with `&&` only when dependent.

- For git commits: ALWAYS follow the Git Safety Protocol and the 4-step commit flow. NEVER amend unless asked. NEVER skip hooks.

- Don't proactively create *.md / README / docs files unless the user asks.

- If a command output is too large, the tool persists it to a file and returns a path — read it back with Read when you need detail.

- For destructive ops (force push, reset --hard, rm -rf), require explicit user approval even if the safety rule allows them.
```

---

## 9. 落地建议（最小可用 Coder Harness）

按这套规格做最小实现，优先级建议：

| # | 工具 | 最低实现成本 | 必加项 |
|---|------|------------|--------|
| 1 | Bash  | 高（有沙箱/超时/后台）| description 字段必须 + stdout/stderr 全保留 + 路径白名单 |
| 2 | Read  | 低 | 走 cat -n + 长度截断 + 大文件分页 |
| 3 | Edit  | 中 | Read 前提校验 + old_string 唯一性 + replace_all |
| 4 | Write | 低 | Read 前提校验 + 拒绝目录路径 |
| 5 | Glob  | 低 | ripgrep `--files` 或 `fast-glob` |
| 6 | Grep  | 中 | ripgrep 子进程 + output_mode + 自动排除 .git |

**其他必要模块：**

- `CoreTool` 接口（§0）
- 权限引擎（`checkPermissions`）
- 沙箱（filesystem allow/deny + network allow/deny）
- Git Safety 协议（强制嵌入 Bash 的 `prompt()`）
- 一个"模型看不到的安全前置层"：阻止破坏性 git 命令、hook 失败时禁止 amend、自动排除 `.env`

---

## 10. 引用源

| 章节 | 文件 |
|------|------|
| §0 工具接口 | `packages/agent-tools/src/types.ts` |
| §1 Read   | `packages/builtin-tools/src/tools/FileReadTool/{prompt.ts, FileReadTool.ts, limits.ts}` |
| §2 Edit   | `packages/builtin-tools/src/tools/FileEditTool/{prompt.ts, types.ts, constants.ts}` |
| §3 Write  | `packages/builtin-tools/src/tools/FileWriteTool/{prompt.ts, FileWriteTool.ts}` |
| §4 Glob   | `packages/builtin-tools/src/tools/GlobTool/{prompt.ts, GlobTool.ts}` |
| §5 Grep   | `packages/builtin-tools/src/tools/GrepTool/{prompt.ts, GrepTool.ts}` |
| §6 Bash   | `packages/builtin-tools/src/tools/BashTool/{prompt.ts, BashTool.tsx}` |
| §7 Git 流程 | 同上 §6 prompt.ts 内 `getCommitAndPRInstructions()` |
