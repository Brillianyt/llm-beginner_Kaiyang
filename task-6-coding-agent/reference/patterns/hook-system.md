# 钩子系统（Hook System）—— PreToolUse / PostToolUse 设计模式

## 来源
- Claude Code 官方文档：https://code.claude.com/docs（Hooks 章节）
- Claude Code 源码：`src/Tool.ts` 中的 `CanUseToolFn`、`src/utils/hooks/` 目录
- 经典模式：Lifecycle Hook（Webpack / Git hooks / VS Code extensions）

## 关键要点
1. **核心思想**：在工具调用的生命周期关键点插入外部回调，让外层能拦截、修改、审计
2. **生命周期事件**（Claude Code 定义）：
   - `PreToolUse` —— 工具执行**前**调用，可 allow / deny / modify args
   - `PostToolUse` —— 工具执行**后**调用，可读取结果并附加 metadata
   - `SessionStart` / `SessionEnd` —— 对话开始 / 结束时
   - `Stop` / `SubagentStop` —— 主线 / subagent 终止时
   - `Notification` —— 系统通知
3. **钩子动作类型**：
   - `command` —— 跑 shell 脚本（最常见）
   - `HTTP` —— 调 webhook
   - `prompt` —— 让 LLM 判 yes/no（轻量验证）
   - `agent` —— 派 subagent 做深度验证
4. **PreToolUse 决策**：
   - `allow` —— 放行
   - `deny` —— 拒绝，工具不执行，返回 `permission_denial` 给 LLM
   - `modify` —— 改写 args 后再执行
5. **PostToolUse 用途**：记录 trace、写 audit log、转换 observation、给 LLM 加额外信息
6. **配置层级**：项目级 `.claude/settings.json`（团队共享）/ 本地 `settings.local.json`（个人 gitignored）/ 用户级

## 与我们任务的关联
- **README 安全章节**：手动列出几条硬编码规则（不许 `..` 越界、不许 `git reset --hard` 等）；用 hook 系统可以**统一抽象**这些规则
- **加分项**：实现 `register_hook('PreToolUse', callback)` API，让评测能挂自定义安全钩子
- **自动审计**：PostToolUse 钩子把每个 tool call 写一行 audit log；SWE-bench 评测时方便反推

## 文字版钩子流

```
                    CodingAgent.run
                         │
                         ▼
                ┌────────────────────┐
                │  PreToolUse 钩子   │
                │  (按顺序串行执行)  │
                └────────┬───────────┘
                         │ 任一返回 deny → 中止
                         ▼
                ┌────────────────────┐
                │   工具实际执行      │
                │  (subprocess.run)   │
                └────────┬───────────┘
                         │
                         ▼
                ┌────────────────────┐
                │  PostToolUse 钩子  │
                │  (可改 observation)│
                └────────┬───────────┘
                         │
                         ▼
                ┌────────────────────┐
                │  返回给 LLM        │
                └────────────────────┘
```

## 代码片段（Python Hook 实现）

```python
from enum import Enum
from typing import Callable, Any

class HookDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"

class HookEvent(Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"

class HookSystem:
    def __init__(self):
        self._hooks: dict[HookEvent, list[Callable]] = {e: [] for e in HookEvent}

    def register(self, event: HookEvent, callback: Callable):
        self._hooks[event].append(callback)

    def fire_pre_tool_use(self, tool_name: str, args: dict) -> tuple[HookDecision, dict]:
        """所有 PreToolUse 钩子按顺序串行执行；任一 deny 就中止。"""
        for hook in self._hooks[HookEvent.PRE_TOOL_USE]:
            try:
                decision = hook(tool_name, args)
                if decision == HookDecision.DENY:
                    return HookDecision.DENY, args
                elif decision == HookDecision.MODIFY:
                    args = ...  # hook 可以修改 args
            except Exception as e:
                log.warning(f"hook {hook.__name__} raised: {e}")
        return HookDecision.ALLOW, args

    def fire_post_tool_use(self, tool_name: str, args: dict, observation: Any) -> Any:
        """所有 PostToolUse 钩子执行；可附加 metadata / 改写 observation。"""
        for hook in self._hooks[HookEvent.POST_TOOL_USE]:
            try:
                hook(tool_name, args, observation)
            except Exception as e:
                log.warning(f"hook {hook.__name__} raised: {e}")
        return observation


# CodingAgent 中使用
class CodingAgent:
    def __init__(self):
        self.hooks = HookSystem()
        # 默认装几条安全钩子
        self.hooks.register(HookEvent.PRE_TOOL_USE, self._block_tests_modification)
        self.hooks.register(HookEvent.PRE_TOOL_USE, self._block_dangerous_git)

    def execute_tool(self, call, repo_path) -> str:
        decision, args = self.hooks.fire_pre_tool_use(call.name, call.args)
        if decision == HookDecision.DENY:
            return f"[ERROR] blocked by PreToolUse hook: {args.get('reason', 'denied')}"
        obs = self.tools[call.name].call(args, repo_path)
        obs = self.hooks.fire_post_tool_use(call.name, args, obs)
        return obs

    @staticmethod
    def _block_tests_modification(tool_name, args) -> HookDecision:
        if tool_name == "write_file" and "test" in args.get("path", "").lower():
            return HookDecision.DENY
        return HookDecision.ALLOW

    @staticmethod
    def _block_dangerous_git(tool_name, args) -> HookDecision:
        if tool_name in ("git_reset", "git_clean") and args.get("hard"):
            return HookDecision.DENY
        return HookDecision.ALLOW
```

## 我们应该怎么借鉴
1. **MVP 先不做**：必做 4 项 DoD 不要求 hook；可作为加分项
2. **如果做：先实现 PreToolUse**：理由——是安全护栏（拦截危险命令）；PostToolUse 主要为了 audit，加分场景少
3. **钩子返回值用 enum**：避免字符串魔法值（"allow"/"deny"/"modify"）
4. **钩子异常不传播**：单个钩子出错不能让整个 agent loop 挂掉；catch 并 log
5. **配置化**（可选）：钩子可以从 `.claude/settings.json` 加载；v1 可以硬编码
6. **审计日志**：PostToolUse 里把每次 call 写到 JSONL（每行一个 JSON 对象）；事后回放
7. **跨进程钩子**（高阶）：通过 stdin/stdout 调外部脚本；v1 不做
8. **Stop 钩子**：当 CodingAgent 准备结束时触发；可用于「自动 commit」「自动发通知」——加分项

## 主要参考来源
- Claude Code 官方文档：https://code.claude.com/docs（Hooks 章节）
- Claude Code 中文实践：https://blog.csdn.net/SaberJYang/article/details/157465912
- Anthropic Claude Code Hooks 详解：https://blog.csdn.net/qq_44810930/article/details/156146071