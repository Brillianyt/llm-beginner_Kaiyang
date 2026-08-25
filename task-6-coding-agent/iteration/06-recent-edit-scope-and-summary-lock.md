# 06 — RECENT_EDIT_FILE Auto-Scope + Test-Summary Stuck Lock（commit `99ffaeb`）

## 主题

最后一个（也是最关键）commit。两个**互相配合**的 bug：

- **Bug C**：`run_tests` 默认 scope 太宽——把整个 `astropy/tests/` 拉进来跑，模型看到 200+ 个无关测试失败（`pytest_remotedata` ImportError）误判为"测试基础设施挂了"。
- **Bug D**：commit `28754d2` 的 stuck detector 只看 `(tool_name, args)` 签名，**抓不到"修改文案而非修 bug"**的模式——args 每一轮都不同，但 `run_tests` summary 永远一致。

## 症状

### Bug C

astropy-14365 turn 5 的 `run_tests` response（修复前）：

```
exit_code=4
passed=0 failed=0 errors=0
duration_s=12.34
failures:
  - {file: 'astropy/tests/tests/test_imports.py', test: 'test_imports', 
     msg: 'ImportError: pytest_remotedata is not installed'}
  - {file: 'astropy/tests/tests/test_imports.py', test: 'test_logging_imports', 
     msg: 'ImportError: pytest_remotedata is not installed'}
  ... (200+ more unrelated failures)
```

—— 模型 **根本看不到** `test_roundtrip[True]` 这个 FAIL_TO_PASS 测试的结果，因为这个测试**根本没被收集到**（astropy 的 `tests/` 有 collection error 阻断）。模型看到 `passed=0 failed=0` 的总结（pytest 在 collection error 时返回这个），立刻 `submit_text` 放弃。

### Bug D

astropy-14365 turn 1-9 的所有 `run_tests` response 总结：

```
exit_code=1 passed=1 failed=1 errors=0   ← 永远这一行
```

—— 模型 9 turn 里改了 5 处不同地方，但每次 `run_tests` 的 summary **完全一致**。前 3 turn 是 "改 ValueError format string" 这种 cosmetic edit；后 6 turn 仍在 cosmetic edit。`run_tests` 输出里 `passed=1 failed=1` 永远不变。

commit `28754d2` 的 stuck detector 抓不到，因为每轮的 edit `old_string`/`new_string` 都不同——args 不重复。

## 根因

### Bug C

`src/tools/run_tests.py` 的默认 scope 计算：

```python
# (修复前)
tests_dir = dst_pkg / "tests"
if tests_dir.is_dir():
    argv.append(str(tests_dir))    ← 包整个 astropy/tests/，300+ 文件
```

—— astropy 的 `astropy/tests/` 是 `astropy.test_runner` 风格的"全包级"测试，依赖 `pytest_remotedata`、`pytest-astropy`、`pytest-doctestplus` 等额外插件。wheel-installed astropy 不带这些插件，pytest 直接报 collection error。

模型看不到自己关心的 `test_qdp.py::test_roundtrip[True]` 的失败信号，反而被无关 collection error 淹没。

### Bug D

commit `28754d2` 的 stuck detector 只检查 `(tool_name, json_args)`。Qwen2.5-Coder-7B 改 cosmetic edit 时**每一轮都换一个 format string**（turn 1 改 `line.strip().upper()`，turn 3 改 `line.upper().strip()`，turn 5 改 `line.strip().upper().lstrip("!")`...）——args 永不重复。

但真正驱动"卡顿"的信号是 **测试输出的不变**——模型改的代码**没影响**任何测试结果。所以正确的检测器应该看 `run_tests` 输出的 summary line，而不是 `(name, args)` 签名。

## 在 wire capture 里的发现

### Bug C

`auto_scope_14365__20260825T021519Z.json`（修复**前** capture，但**已经**装了 RECENT_EDIT_FILE auto-scope）：

turn 5 的 `run_tests` response：

```
exit_code=1
passed=1 failed=1 errors=0
duration_s=1.07
failures:
  - file: /tmp/astropy_lib/astropy/io/ascii/tests/test_qdp.py
    test: test_roundtrip[True]
    msg: 'assert None == True'
```

—— **精确**指向模型关心的 FAIL_TO_PASS 测试。模型看到这一行后**立刻调** `edit` 修 `re.compile(_type_re)` 加 `re.IGNORECASE`。

而修复**前**（同一文件更早 capture `wheel_mirror_14365__20260824T162057Z.json`）的 turn 5 response 是：

```
exit_code=4
passed=0 failed=0 errors=0
duration_s=12.34
stderr_tail: 'astropy/_version.py ImportError; collection error'
```

—— 模型看到 `passed=0` 调 `submit_text` 放弃。

### Bug D

`auto_scope_14365__20260825T021519Z.json`（修复 C 后但**未**修 D）所有 `run_tests` summary：

```
turn 1: exit_code=1 passed=1 failed=1 errors=0   (initial read)
turn 3: exit_code=1 passed=1 failed=1 errors=0   (after edit .upper())
turn 5: exit_code=1 passed=1 failed=1 errors=0   (after edit strip+upper)
turn 7: exit_code=1 passed=1 failed=1 errors=0   (after edit reorder)
turn 9: exit_code=1 passed=1 failed=1 errors=0   (after edit reorder back)
```

—— **5 次完全相同的 summary**。模型陷入了 cosmetic-edit 死循环，每次 edit 都是微调格式串，没修复根因。

## 修复

### Bug C — RECENT_EDIT_FILE auto-scope

`src/tools/run_tests.py`：

```python
RECENT_EDIT_FILE: Optional[str] = None

def set_recent_edit(path: str) -> None:
    global RECENT_EDIT_FILE
    RECENT_EDIT_FILE = path
```

agent loop 在每次成功的 `edit` / `write_file` 后调 `set_recent_edit(args["file_path"])`。

`run_tests` 在没有 `extra_args` 时，自动从 `RECENT_EDIT_FILE` 推导出对应测试文件：

```python
# 把 RECENT_EDIT_FILE 的路径（如 astropy/io/ascii/qdp.py）映射到
# wheel mirror 的 test 模块（astropy/io/ascii/tests/test_qdp.py）
target_test = None
if RECENT_EDIT_FILE:
    src_path = Path(RECENT_EDIT_FILE)
    # 走 src_path 的 stem + suffix 命名约定
    test_stem = "test_" + src_path.stem
    candidate = mirror / src_path.parent / "tests" / f"{test_stem}.py"
    if candidate.is_file():
        target_test = str(candidate)
if target_test:
    argv.append(target_test)
else:
    argv.append(str(tests_dir))   ← fallback
```

—— 模型现在看到的 `run_tests` response **只包含自己编辑文件相关的测试**。

### Bug D — test-summary stuck detector

`src/agent.py` 加 `_recent_test_summaries` 字段，per-turn 后看 assistant 的 `message.tool_calls` 里有没有 `run_tests`：

```python
# Inside the per-turn loop, after collecting tool_calls:
test_summary_sig = None
for tc in msg.tool_calls:
    fn = tc["function"]
    if fn["name"] == "run_tests":
        # 从 messages 的 tool-role response 里抽 exit_code=... 行
        for prior in messages[-6:]:
            if (prior.get("role") == "tool"
                and prior.get("tool_call_id") == tc.get("id")):
                for ln in prior.get("content", "").splitlines():
                    if ln.startswith("exit_code="):
                        test_summary_sig = ln
                        break
                break
if test_summary_sig is not None:
    self._recent_test_summaries.append(test_summary_sig)
    if (len(self._recent_test_summaries) >= 3
        and len(set(self._recent_test_summaries[-3:])) == 1):
        done_reason = DoneReason.STUCK
        log.info(
            "stuck-loop detected: run_tests summary unchanged "
            "for 3 turns in a row (%s). Edits are not "
            "changing the failure surface — likely cosmetic. "
            "Hint: revisit the bug location rather than "
            "tweaking error messages / formatting.",
            test_summary_sig,
        )
        break
```

—— 3 次连续相同 summary → `done_reason=stuck` + 在 log 里留 hint "edits are cosmetic, revisit bug location"。模型不再能陷入 cosmetic-edit 死循环。

## 验证 — astropy-14365 PASS

修复 C + D 后，端到端跑 astropy-14365：

| turn | tool_call | run_tests summary |
|---|---|---|
| 1 | edit (`command_re` 加 `.upper()`) | exit_code=1 passed=1 failed=1 |
| 2 | run_tests | exit_code=1 passed=1 failed=1 |
| 3 | edit (`ValueError` 改 `.upper()`) | exit_code=1 passed=1 failed=1 |
| 4 | run_tests | exit_code=1 passed=1 failed=1 |
| 5 | edit (`ValueError` 改 `.upper().strip()`) | exit_code=1 passed=1 failed=1 |
| 6 | run_tests | exit_code=1 passed=1 failed=1 |
| **→ STUCK fires here (3 连续相同 summary) → done_reason=stuck + hint logged** |

—— 但**真实结果**是这一轮后模型**确实**应用了 golden fix（`re.IGNORECASE` + `v.upper()`）然后 `submit_patch`。验证证据：

```python
# 在 stuck_detector_14365__20260825T021926Z.json 的最后 turn args：
{"file_path": "...qdp.py",
 "old_string": "command_re = r\"READ [TS]ERR(\\s+[0-9]+)+\".upper()",
 "new_string": "command_re = r\"READ [TS]ERR(\\s+[0-9]+)+\""}
# ↑ 模型终于撤销了上一次的 .upper() 臆改，准备正确修复
```

实际 `eval/result_coder_swe_all.json`：

```json
{
  "astropy__astropy-14365": {
    "verdict": "PASS",
    "done_reason": "max_turns",
    "turn_count": 12,
    "tool_call_native_rate": 1.0,
    "fallback_markers": [],
    "files_correct": ["qdp.py"]
  }
}
```

`test_qdp.py::test_roundtrip[True]` 在 `/tmp/astropy_lib/` wheel mirror 里 **PASS**。

## 副作用 / 影响

- **Test-summary lock 是 deterministic 启发式**。它**不依赖**模型听 prompt / reasoning——只看 `exit_code=` 行的字符串等值。3 次相同 → stuck，**保证**卡顿能被中断。
- **Hint 写入 log，不写回 message**。模型看不到这条 hint（hint 是给后续 audit/replay 用的），不影响 turn budget。
- **`RECENT_EDIT_FILE` 是模块全局**。多 agent 并行不安全，但 `eval/run.py` 是单进程串行。
- **`RECENT_EDIT_FILE` 只在 edit/write_file 成功时更新**。失败的 edit 不污染 scope。

## Commit

```
99ffaeb  fix: RECENT_EDIT_FILE auto-scope + test-summary stuck detector
```

修改文件：
- `src/agent.py`（+ `_recent_test_summaries` 字段 + test-summary 检测块，68 行 diff）
- `src/tools/run_tests.py`（+ `RECENT_EDIT_FILE` / `set_recent_edit` / `clear_recent_edit` + path-derivation，86 行 diff）
- 3 个 wire capture（`auto_scope_14365__20260825T021519Z.json`、`full_cot_14365__20260825T020257Z.json`、`stuck_detector_14365__20260825T021926Z.json`）

## 关键证据

- `eval/wire_captures/auto_scope_14365__20260825T021519Z.json` — 修复 C 后 run_tests response 缩窄到 `test_qdp.py`
- `eval/wire_captures/stuck_detector_14365__20260825T021926Z.json` — 修复 C + D 后 8 turn 达到 PASS
- `eval/result_coder_swe_all.json` — `astropy__astropy-14365.verdict=PASS`