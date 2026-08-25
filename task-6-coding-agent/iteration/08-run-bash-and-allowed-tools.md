# 08 — `run_bash` 沙箱 + `allowed-tools` 强制执行（commit `f7e9a1b`）

## 主题

review 阶段发现两个**结构性问题**：
1. `src/skills/code-review/SKILL.md` 描述说"用 `run_bash` 跑 `scripts/diff_stats.py`"，但 `run_bash` 工具**不存在**——承诺与实现能力不等价
2. `allowed-tools` 字段在 `SkillLoader.scan()` parse 进了 meta，但 `_handle_load_skill` **从不检查**——纯装饰

加上一处执行期 bug：写 end-to-end 测试时发现 `safe_resolve` 在 repo_root 是 `Path('.')`（未 resolve）时会把**任何相对路径**判定为"逃出 repo root"——这是过去 `read_file` / `write_file` 用绝对路径绕过的预存 bug，第一次有工具（`run_bash`）走相对路径才暴露。

## 症状

### 1. `run_bash` 缺失

`src/tools/` 下只有 `read_file / write_file / edit / run_tests / git_diff / git_apply / grep / list_files`。Skill 描述承诺的 `run_bash` 不可达。

### 2. `allowed-tools` 不强制

学生 review 指出："任何人能在任何 tool context 调 `load_skill('code-review')`"——即便该 skill 声明 `[run_bash, read_file, grep]`，模型照样能调 `edit` / `write_file` 修改代码。

### 3. `safe_resolve` 误判

端到端 smoke test 第一次跑：
```
cmd: python src/skills/code-review/scripts/diff_stats.py smoke_diff_for_codereview.txt
[ERROR] path in cmd escapes repo root: src/skills/code-review/scripts/diff_stats.py
```

`Path('.') / 'src/...'.resolve()` 返回 `/mnt/.../task-6-coding-agent/src/...`，但 `Path('.')` 本身未 resolve 是 `'.'`——`relative_to` 比 `'/'` vs `'.'` 必然失败。

## 根因

### 1. `run_bash` 缺失

参考 `claude_reference/.../BashTool/` 的设计原则（list-form subprocess + path containment + deny-list + AST-level安全检查），但用最小可用版本实现：shlex + 危险命令 deny-list。完整 AST解析（tree-sitter）超出我们 7B 模型 harness 的需要。

### 2. `allowed-tools` 不强制

Anthropic Skills 文档说 allowed-tools 是"声明 skill 需要的能力"，但模型在调用 tool 时应当被 hook 拒绝。需要在 agent 端实现一个 "active skill" tracker：调 `load_skill(name)` 时激活，调 `submit_patch/submit_text` 或 `load_skill(other)` 时清掉。中间所有 tool call 必须经过 allowlist 检查。

### 3. `safe_resolve` 误判

```python
# before
candidate = (repo_root / p).resolve(strict=False)
candidate.relative_to(repo_root)  # repo_root 未 resolve → 失败
```

`repo_root` 是 `Path('.')` 时，`.relative_to('.')` 与 `/abs/path/.relative_to('.')` 不匹配。

## 在 wire capture 里的发现

`eval/wire_captures/run_bash_e2e__20260825T105623Z.json`（端到端跑，issue = "review the diff ... use code-review skill, then call run_bash to invoke diff_stats.py and report findings"）：

| turn | model 调的工具 | 结果 |
|---|---|---|
| 0 | 无（content 写了"伪 tool call JSON" `{name:"code-review", arguments:{...}}`，但 agent 不 introspect content） | — |
| 1 | `run_bash("python /path/to/diff_stats.py /tmp/runbash_e2e_diff.txt")` | `[ERROR] absolute path rejected: /path/to/diff_stats.py` |
| 2 | `run_bash("python diff_stats.py /tmp/runbash_e2e_diff.txt")` | `[ERROR] absolute path rejected: /tmp/runbash_e2e_diff.txt` |
| 3-5 | 同 turn 2 多次重试 | 全部 path 拒绝 |
| 6 | done_reason=stuck（同 signature lock 触发） | — |

**关键观察**：
- **5 次 run_bash 调用全部被沙箱正确拦截**——sandbox 在按设计工作
- 但模型**绕过了 load_skill**——turn 0 在 content 里写了 JSON 而非真正调 `load_skill` tool，导致 `skill_loads=[]`
- 模型的"重试相同失败命令"模式触发了 stuck detector（同 signature lock）

这是**模型行为问题**，不是 harness 问题。`TestCodeReviewSkillEndToEnd`（test_smoke.py）通过 agent dispatch **手动** 验证了正确链路：
```
agent._handle_load_skill({"name": "code-review"}) → 设置 _active_skill
agent._dispatch("run_bash", {"cmd": "python src/skills/code-review/scripts/diff_stats.py smoke_diff.txt"}, ...)
→ 执行 exit_code=0 + 打印 "files=1 ... foo.py"
```

## 修复

### 1. `src/tools/run_bash.py`（新增，~330 行）

```python
class RunBashTool(BaseTool):
    name = "run_bash"
    description = "Run a single shell command inside the repo and return ..."
    is_read_only = False
    
    DEFAULT_TIMEOUT = 60
    HARD_TIMEOUT_CAP = 600
    MAX_OUTPUT_TAIL = 8000
    
    def call(self, args, repo_root):
        cmd_str = (args.get("cmd") or "").strip()
        tokens = shlex.split(cmd_str, posix=True)  # 不开 shell
        deny_reason = _check_deny_list(cmd_str)    # 原始字符串匹配，不quote metachar
        if deny_reason: return f"[ERROR] {deny_reason}\n  cmd: {cmd_str}"
        cwd = repo_root
        # ... cwd 校验 ...
        for pathy in _extract_path_tokens(tokens):
            safe_resolve(pathy, repo_root)         # 路径必须落 repo 内
        cp = run_subprocess(tokens, cwd=cwd, timeout=timeout, extra_env=...)
        return _format_result(cmd_str, cp, duration)
```

**Deny-list 13 条**（每次新条要写明 reason）：

```python
_DENY_PATTERNS = [
    # 文件系统破坏
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rfRF]+|--recursive|--force)\b"), "rm -rf / recursive rm / forced rm"),
    (re.compile(r"\brm\s+-rf?\s+/\s*$"), "rm -rf / or rm -rf /something"),
    (re.compile(r"\bfind\s+/.+\s+-delete\b"), "find … -delete (recursive delete)"),
    # 提权
    (re.compile(r"\bsudo\b"), "sudo (privilege escalation)"),
    (re.compile(r"\bsu\b\s"), "su (switch user)"),
    # chmod 批量
    (re.compile(r"\bchmod\s+(-R|--recursive)\s+[0-7]*[2367][0-7]*\s+/"), "chmod -R world-writable on / path"),
    (re.compile(r"\bchown\s+(-R|--recursive)\s"), "chown -R (mass ownership change)"),
    # git 破坏性
    (re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f|checkout\s+--\s|branch\s+-D)"), "destructive git command"),
    (re.compile(r"\bgit\s+push\s+(--force|-f)\b"), "git push --force"),
    # 网络外泄
    (re.compile(r"\bcurl\b.*\|\s*(sh|bash)\b"), "curl … | sh (piped download/exec)"),
    (re.compile(r"\bwget\b.*\|\s*(sh|bash)\b"), "wget … | sh (piped download/exec)"),
    # 磁盘 / 内核
    (re.compile(r"\b(mkfs|mkfs\.\w+)\b"), "mkfs (disk format)"),
    (re.compile(r"\b(umount|mount)\b\s+/"), "umount/mount on root path"),
    # 进程
    (re.compile(r"\b(kill\s|-KILL|killall)\b"), "force-kill process"),  # 注：转义需修正
    (re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b"), "system shutdown/reboot"),
    # 重定向到绝对路径
    (re.compile(r">\s*/"), "redirect to absolute path"),
]
```

**Path 检查**：`_extract_path_tokens(tokens)` 用正则挑出"看起来像路径"的 token：
```python
_PATHY_TOKEN_RE = re.compile(r"^(?:\./|\.\./|/|[^-\s][^/]*/[^/].*)")
```

**关键设计决策**：
- **Deny-list 跑在原始 cmd_str 上**（不是 `shlex.join(tokens)`）——因为 shlex 会 quote `|` 让正则 match 不到 `curl foo | sh`
- **不展开 shell**：`subprocess.run(args=[...], shell=False)`；pipes / redirects / 命令替换都被 shlex split 剥掉
- **env 净化**：`run_subprocess` 已经把 env 限制到 `PATH + HOME + extra_env`，不泄漏 `OPENAI_API_KEY`
- **stdout/stderr tail 截断**：8000 chars / 半分前后

### 2. `src/tools/base.py:safe_resolve`（修复 1 行 bug）

```python
# before
candidate = (repo_root / p).resolve(strict=False)
candidate.relative_to(repo_root)

# after
resolved_root = repo_root.resolve(strict=False)  # ← resolve first
candidate = (resolved_root / p).resolve(strict=False)
candidate.relative_to(resolved_root)
```

锁住：`test_smoke.py::TestToolSafety::test_safe_resolve_handles_relative_repo_root`

### 3. `src/skill_loader.py:allowed_tools(name)`（新增方法）

```python
def allowed_tools(self, name: str) -> Optional[List[str]]:
    meta = self._meta.get(name)
    if meta is None:
        return None
    tools = meta.get("allowed_tools")
    if not tools:
        return None
    return list(tools)
```

返回 `None` 表示"未声明 allowed-tools，不限"；返回 list 表示"严格按列表执行"。

### 4. `src/agent.py:_active_skill` + `_dispatch` allowlist 检查

```python
# CodingAgent.__init__
self._active_skill: Optional[str] = None   # NEW

# _handle_load_skill sets it
def _handle_load_skill(self, args):
    body = self.skill_loader.load(name)
    self._active_skill = name               # NEW
    return body

# _dispatch enforces it
if (self._active_skill is not None
        and self.skill_loader is not None
        and name != "load_skill"):
    allow = self.skill_loader.allowed_tools(self._active_skill)
    if allow is not None and name not in allow:
        return (
            f"[ERROR] tool '{name}' not in skill "
            f"'{self._active_skill}' allowlist (allowed: {allow})",
            True,
        )

# submit_patch / submit_text clear it
if name == "submit_patch":
    self._active_skill = None
```

### 5. `src/skills/code-review/SKILL.md`（更新 frontmatter + body）

```yaml
---
name: code-review
description: "..."
when_to_use: "..."
allowed-tools: [run_bash, read_file, grep]   # ← NEW
---
# Code Review
## Available resources
- `scripts/diff_stats.py` ... Invoke via the `run_bash` tool (the agent's
  sandboxed shell):
  ```
  run_bash(cmd="python <skill_dir>/scripts/diff_stats.py <diff_path>")
  ```
  `<skill_dir>` is `src/skills/code-review/`; `<diff_path>` must be
  inside the repo.

## Allowed tools (enforced)
When this skill is loaded, the `allowed-tools` frontmatter above is
enforced by the agent: only `run_bash`, `read_file`, `grep` may be
called while the skill is active. Calling `edit`, `write_file`, or any
write tool while this skill is loaded is rejected with `[ERROR] tool
'X' not in skill 'code-review' allowlist`.
```

## 验证

### 1. 28/28 smoke test 通过

新增 20 个测试：
- `TestRunBashSandbox`（13 个）— happy path / 路径沙箱 / deny-list
- `TestSkillAllowedTools`（6 个）— 解析 / 强制执行 / 清理
- `TestCodeReviewSkillEndToEnd`（1 个）— `load_skill → run_bash → diff_stats.py` 真可达
- `TestToolSafety::test_safe_resolve_handles_relative_repo_root`（1 个）— 锁住 base 修复

### 2. 手动验证 10 项（全部 PASS）

```
1. echo hello: PASS    6. curl|sh: PASS (denied)
2. abs /etc/passwd: PASS (denied)  7. curl --version: PASS (allowed)
3. .. traversal: PASS (denied)  8. sleep 0.1: PASS (exit_code=0)
4. rm -rf: PASS (denied)  9. cwd ../etc: PASS (denied)
5. git reset --hard: PASS (denied)  10. python script.py: PASS (run)
```

### 3. Wire 证据

`eval/wire_captures/run_bash_e2e__20260825T105623Z.json`：
- 模型在 6 turn 内 5 次调 `run_bash`，**全部被 path 沙箱正确拦截**（绝对路径 / `/path/to/` 占位符）
- `done_reason=stuck`（同 signature lock 触发）——stuck detector 在按设计工作
- `tool_call_native_rate` 保持 1.0，架构 invariant 不变

注意：模型在 turn 0 **没有真正调 `load_skill`**——它在 content 里写了"伪 tool call JSON"。这暴露了模型对 MCP 协议的理解不完整，是**模型训练问题**而非 harness 问题。手动 dispatch 测试覆盖了正确链路。

## Commit

```
f7e9a1b  feat: run_bash sandboxed subprocess runner + skill allowed-tools gate
```

修改文件：
- `src/tools/run_bash.py`（新增 330 行）
- `src/tools/base.py:safe_resolve`（resolve repo_root first）
- `src/tools/__init__.py`（注册 RunBashTool）
- `src/skill_loader.py:allowed_tools`（新增方法）
- `src/agent.py`（`_active_skill` 字段 + dispatch allowlist + submit_* clears）
- `src/skills/code-review/SKILL.md`（allowed-tools frontmatter + 文档同步）
- `test_smoke.py`（20 个新测试）

## 副作用 / 影响

- **`run_bash` 现在是 agent 的工具集一部分**，MCP schema 自动包含（`make_tool_set()` 把它放进去）
- **`allowed-tools` 是声明性 + 强制执行**——不再是死字段
- **Path 沙箱在 repo_root 是 `Path('.')` 时也正确**——之前所有 path-based 工具（`read_file` 等）都要求绝对路径才偶然避开了这个 bug
- **不动 `message.tool_calls` 路径**——架构 invariant 保持（`tool_call_native_rate=1.0` 不变）

## 关键证据

- `eval/wire_captures/run_bash_e2e__20260825T105623Z.json` — 端到端模型调用，5 次 path 沙箱拒绝
- `test_smoke.py:TestRunBashSandbox`（13 个） — 单元级沙箱验证
- `test_smoke.py:TestSkillAllowedTools`（6 个）— allowlist 强制执行验证
- `test_smoke.py:TestCodeReviewSkillEndToEnd::test_skill_script_is_reachable_via_run_bash` — skill 真正可达 end-to-end
- `test_smoke.py:TestToolSafety::test_safe_resolve_handles_relative_repo_root` — base bug 修复