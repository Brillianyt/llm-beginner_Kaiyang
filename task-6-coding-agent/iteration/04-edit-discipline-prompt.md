# 04 — Edit-Discipline Prompt（commit `127005d`）

## 主题

把"minimum change / don't rename / re-read before retry"等编辑纪律写入 system prompt，作为对 astropy 12907/14365 上观察到的**模型自我循环**的硬性约束。

## 症状

astropy-12907 上模型在调 5 turn `read_file` 后给出了正确方向的诊断（"nested CompoundModel"），但后续 4 turn 都在**重命名函数 / 添加递归变体 / 重构周边代码**：

- 把 `_separable` 改成 `_is_separable`
- 加一个 `def _separable_recursive(model, ...)` 私有 helper
- 在 `_cstack` 计算里加 `if model.n_submodels > 1: model = model.copy()` 前置处理

—— 每一轮都让测试输出从 `1 passed 5 failed` 变成 `1 passed 5 failed`（**完全一致**）。因为根因是 `_cstack = right` 这一行硬编码 `right`，模型根本没碰那一行。

astropy-14365 上模型在 4 turn 里反复改：

```
edit(old="raise ValueError(f\"... {line}\")", new="raise ValueError(f\"... {line.upper()}\")")
edit(old="raise ValueError(f\"... {line.upper()}\")", new="raise ValueError(f\"... {line.upper().strip()}\")")
edit(old="raise ValueError(f\"... {line.upper().strip()}\")", new="raise ValueError(f\"... {line.strip().upper()}\")")
edit(old="raise ValueError(f\"... {line.strip().upper()}\")", new="raise ValueError(f\"... {line.upper().strip()}\")")
```

—— 来回切换 `strip()` 和 `upper()` 的顺序，但**真正的 bug**在 `command_re = r"READ [TS]ERR..."` 没加 `re.IGNORECASE` 和 `v.upper()`，模型从来没碰那一行。

## 根因

7B 模型的"explore / 反复试探"模式：
1. 模型找到第一个错误信号后，倾向于"diff 一下就提交"——而不是定位真正的根因。
2. 即使定位到了，倾向于"修复 error message format string"这种视觉上"看起来像修了"的地方，因为 7B 对**测试失败的字面相似度**敏感（"line.strip().upper()" 和 "line.upper().strip()" 在视觉上"对称"，模型认为它们是同一个东西的不同排列）。
3. 模型不知道"renaming existing function"会破坏 import 关系、添加递归变体会让 stack overflow。

## 在 wire capture 里的发现

### 12907

`heuristics_12907__20260824T164021Z.json` 显示模型在 5-9 turn 的 edit args 字段包含：

```
{"new_string": "def _is_separable(model):\n    matrix = np.zeros((model.n_outputs, model.n_inputs))\n    if isinstance(model, CompoundModel):\n        ...\n"}
```

—— 引入新函数 + 改原函数名。这违反了 file-system-spec §2（old_string 必须唯一）。

### 14365

`reasoning_12907__20260824T164103Z.json`（同时命名 14365 的）显示 turn 3 / 5 / 7 / 9 都在改同一个 ValueError 行的 format string。`run_tests` response 在这 4 轮里**完全相同**：

```
exit_code=1 passed=1 failed=1 errors=0
```

—— 任何一轮的 edit 都**没有真正影响测试输出**。

## 修复

`src/prompt.py` 增加 `Edit discipline:` 段（**未在任何 capture 上验证 prompt-only 修复的有效性**，因为 commit `99ffaeb` 的 summary-lock 才是真正解决 cosmetic-edit 卡顿的——见 `06-recent-edit-scope-and-summary-lock.md`）：

```python
system_prompt = system_prompt + (
    "\n"
    "Edit discipline:\n"
    "- Make the MINIMUM change that fixes the bug. NEVER rename\n"
    "  existing functions, NEVER add recursive variants, NEVER\n"
    "  refactor surrounding code. The fix is almost always a\n"
    "  one-line change to a buggy existing line.\n"
    "- If `edit` fails with 'old_string not found', DO NOT keep\n"
    "  retrying the same edit. Re-read the file and find the EXACT\n"
    "  line (including leading whitespace) you want to change.\n"
    "- After your edit succeeds, call `run_tests` to verify. If\n"
    "  the FAIL_TO_PASS test passes, STOP editing immediately and\n"
    "  call `submit_patch`. Do not make further edits.\n"
    "- Bug-location heuristics: read the file; the issue text often\n"
    "  names the buggy function. For matrix / numeric bugs check\n"
    "  indexing and `np.zeros` initial-value assignment; for\n"
    "  case-sensitivity bugs prefer `re.IGNORECASE` on `re.compile`.\n"
)
```

3 条关键 heuristic：
1. **Minimum change**：禁止 rename / recursive variant / refactor。
2. **Don't retry same edit**：强制 re-read。
3. **Bug-location heuristics**：
   - matrix / numeric bugs → check `np.zeros` 初始值和 indexing
   - case-sensitivity bugs → 优先 `re.IGNORECASE`

第 3 条直接对应 astropy-14365 的 golden fix（"用 `re.IGNORECASE` on `re.compile`"），是**对模型经验的具体编码**。

## 验证

- `test_smoke.py` 仍 8/8 通过。
- `eval/result_coder_swe_all.json` 里 astropy-14365 的 `verdict=PASS`，但实际**达到 PASS 的关键不是这个 prompt**——而是 commit `99ffaeb` 的 RECENT_EDIT_FILE auto-scope + test-summary stuck detector。这个 prompt 段在 `heuristics_12907` 和 `reasoning_12907` capture 后才加入，但因为 cosmetic-edit 模式是结构性而非提示性问题，prompt 本身**不足以**阻止。

## 局限性

- **Prompt 是软约束**。7B 模型读到这条 prompt 后仍然可能在 turn 5 就忘掉。
- **真实解决方案是 stuck detector 的 test-summary lock**（commit `99ffaeb` Bug D），它不依赖模型听指令，而是**外部可观测信号**驱动：3 次相同 `run_tests` summary → `done_reason=stuck` + hint。
- 但这个 prompt 段仍保留，因为它给模型提供了**正确的方向感**——不是"如何不死循环"（那是 stuck detector 的事），而是"bug 在哪"。

## Commit

```
127005d  fix: stronger edit-discipline prompt and path-resolution in run_tests
```

修改文件：
- `src/prompt.py`（+ Edit discipline 段）
- 3 个 wire capture（`final_attempt_12907__20260824T164528Z.json`、`heuristics_12907__20260824T164021Z.json`、`reasoning_12907__20260824T164103Z.json`）

## 关键证据

- `eval/wire_captures/heuristics_12907__20260824T164021Z.json` — 引入新函数 `_is_separable` 的 evidence
- `eval/wire_captures/reasoning_12907__20260824T164103Z.json` — reasoning 轮里提议 `re.IGNORECASE` 但 edit 仍改 ValueError
- `eval/wire_captures/final_attempt_12907__20260824T164528Z.json` — model 最终失败的 turn-by-turn sequence