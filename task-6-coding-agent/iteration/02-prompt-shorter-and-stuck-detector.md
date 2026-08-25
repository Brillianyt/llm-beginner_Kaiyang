# 02 — Prompt 缩短 + 第一版 Stuck 检测器（commit `28754d2`）

## 主题

两件事：(a) 系统提示词从 4731 chars 瘦身到 1462 chars；(b) 引入第一版基于 `(tool_name, args)` 签名的 stuck 检测器。

## 症状 / 根因

### (a) Prompt 太长

最初 prompt 沿用 SGLang 时代的完整 subagent 协议、`explore-before-submit` 启发式、详细 tool descriptions。Qwen2.5-Coder-7B-Instruct 上下文 16K，其中 ~4.7K 是 system prompt，加上每轮 tool response（最大 4000 chars）和历史，几轮后就触发 `max_tokens=4096` 的输出截断。

### (b) 模型无限循环

7B 模型没有 Claude Opus 的"5 步无 insight 自动放弃"启发式。一旦它选定一个错误方向（例如反复调用 `list_files` / `read_file` 而不 `edit`），会一直重复同一组 tool calls。在 50 turn 默认上限下，token 浪费 + trace 噪声巨大。

## 在 wire capture 里的发现

### (a) 的证据

`audit_14365__20260824T161513Z.json` turn 1 request body 完整 dump 显示 `messages[0].content` 长度 ≈ 4731 chars，包含：
- "## Tools" 段 5 个 tool 的完整 schema
- "## Skills (Level-1)" 段 + 示例
- "## Subagents" 段 + 边界说明
- "## Termination" 段 + 多个 if-then 规则
- "## Edit discipline" 段（后续 commit `127005d` 才加的）

但模型首轮 assistant `message.content` 是**空字符串**、只调一次 `read_file`。——4.7K 提示词淹没，模型抓不住重点。

### (b) 的证据

`flow_analysis_14365__20260824T165254Z.json` 显示 9 turn 里有 4 turn 是**完全相同**的 `list_files` + `read_file` 组合，模型没进入 `edit` 阶段就耗光 turn budget。

## 修复

### (a) Prompt 缩短

`src/prompt.py` 重写：
- Tool descriptions 改成 "一行摘要 + 关键参数" 格式
- 删掉 "explore-before-submit" 长篇启发式
- SkillLoader Level-1 索引保留（"Skills (Level-1 — load body via `load_skill`)" + `(none)` 当没有 skill）
- Termination 简化为 3 条规则（fix complete / stuck / 永不 edit 测试文件）

总长度 4731 → 1462 chars。

### (b) 第一版 stuck 检测器

`src/agent.py` 内：

```python
self._recent_signatures: list[str] = []
# inside the per-turn loop, after collecting msg.tool_calls:
sig = "|".join(
    f"{t['function']['name']}:{json.dumps(t['function']['arguments'], sort_keys=True, default=str)}"
    for t in msg.tool_calls
)
self._recent_signatures.append(sig)
if len(self._recent_signatures) >= 3 and len(set(self._recent_signatures[-3:])) == 1:
    done_reason = DoneReason.STUCK
    log.info("stuck-loop detected: same tool signature 3 turns in a row")
    break
```

`src/trace.py` 新增：

```python
class DoneReason(str, Enum):
    ...
    STUCK = "stuck"
```

阈值取 **3** 而非 Claude Code 的 **5**：Qwen2.5-Coder-7B 的循环速度比 Opus 快得多（每 turn ~1-2 秒 vs Opus 的 ~5 秒），3 turn 已足以判断"无 insight"。

## 验证

- 8/8 `test_smoke.py` 通过
- Toy-repo 仍 PASS（3-7 turn，pytest 3/3 green）
- Astropy 14365 仍 FAIL（但 stuck detector 这次没救到——见 commit `99ffaeb` 解释的 (b) 限制）

## 第一版 stuck 检测器的局限

它**只抓"完全相同签名"**。但 astropy-14365 上观察到的真实卡顿模式是：

```
turn 1: edit(old="A", new="B")
turn 2: edit(old="B", new="C")   ← 改了，但
turn 3: edit(old="C", new="B")   ← 又改回去
turn 4: edit(old="B", new="C")   ← 永远在 B/C 之间切换
turn 5: edit(old="C", new="B")
turn 6: edit(old="B", new="C")
```

—— args 每一轮都不同，签名永远不相同。但 `run_tests` 的输出在所有这些轮里都**完全一致**（因为编辑是 cosmetic）。这正是 Bug D 的主题（见 `06-recent-edit-scope-and-summary-lock.md`）。

## Commit

```
28754d2  fix: prompt shorter + stuck-loop detector
```

修改文件：
- `src/prompt.py`（4731 → 1462 chars）
- `src/agent.py`（+stuck 检测）
- `src/trace.py`（+`DoneReason.STUCK`）
- `src/tools/base.py`（清理注释）
- `src/tools/read_file.py`、`src/tools/git_apply.py`（小重构）
- `src/subagents/base.py`（+ `to_wire_tool_calls` import）
- `src/vllm_plugin/__init__.py` + `qwen_coder_tool_parser.py`（新增 vLLM parser plugin，350 行）

## 关键证据

- `eval/wire_captures/audit_14365__20260824T161513Z.json` — turn 1 system prompt 长度（修复前 4731+）
- `eval/wire_captures/flow_analysis_14365__20260824T165254Z.json` — 9 turn 中 4 turn 是同一组 `list_files`+`read_file`
- `eval/wire_captures/stuck_detector_14365__20260825T021926Z.json` — 修复后，test-summary 锁定（Bug D 主题）