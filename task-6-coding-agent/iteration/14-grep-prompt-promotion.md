# 14 — grep / list_files prompt promotion（hint 实验）

> 2026-08-26（本会话）。iteration/13 收尾时的支线目标：验证
> "在 system prompt 里把 `grep` 推上去" 能否让 astropy-14182 翻成 PASS。

## 1. 改动

`src/prompt.py:_TERSE_TOOL_DESC` 里两个条目扩写：

### `grep`（11 字 → 165 字）
**修前**：
```
ripgrep search for a pattern across the repo.
```

**修后**：
```
ripgrep search for a pattern across the repo. **When the issue text
doesn't name a file or you're unsure which file to edit, use
`grep(pattern='<keyword_or_symbol>', output_mode='files_with_matches')`
first to locate the buggy file**; then `read_file` it. Default
output_mode='files_with_matches' returns only paths (good for locating);
use `output_mode='content'` with `context=N` to see the matching lines.
```

### `list_files`（40 字 → 100 字）
**修前**：
```
List files under a directory (skip .git).
```

**修后**：
```
List files under a directory (skip .git). Use when you need to see what
files exist in a directory without knowing specific filenames.
```

只动 _TERSE_TOOL_DESC 里的描述，不动 prompt 总长度限制（系统提示总长仍 ~2.9K chars）。

## 2. 实验结果（astropy-14182）

`eval/wire_captures/verify_template_14182_grep__20260826T084200Z.json`：

| 指标 | 修前 | 修后 |
|---|---|---|
| tool_calls 总数 | 3（全 run_tests） | 12（6×read_file + 6×edit） |
| done_reason | stuck（3 重复 run_tests） | max_turns |
| tool_call_native_rate | 1.0 | 1.0 |
| fallback_markers | [] | [] |
| 编辑的文件 | （无） | `astropy/table/connect.py` ❌ |
| golden file | `astropy/io/ascii/rst.py` | `astropy/io/ascii/rst.py` |
| verdict | WRONG_FILE | **WRONG_FILE** |

→ 14182 **没翻成 PASS**，但失败模式变了：

| 修前 | 修后 |
|---|---|
| 模型**没探索**，直接跑 tests | 模型**过度探索**，跟着 traceback 进了 `connect.py` |
| 0 个 read_file | 6 个 read_file，全在同一文件 |
| 0 个 grep | 0 个 grep ← **关键** |

## 5. grep 仍未被调用的根因（模型 capability，不是 prompt）

issue 文本的 traceback 长这样：

```
File "/usr/lib/python3/dist-packages/astropy/io/ascii/connect.py", line 26, in io_write
    return write(table, filename, **kwargs)
File "/usr/lib/python3/dist-packages/astropy/io/ascii/ui.py", line 856, in write
```

→ **issue 文本里有文件名**（`connect.py`）。我的 grep 触发条件是"issue text doesn't name a file"，对 14182 **不触发**。模型按 traceback 路径直接调 read_file 进了 connect.py。

但 traceback 是误导——`connect.py:26 io_write` 只是 dispatch layer，真正的 bug 在 `astropy/io/ascii/rst.py`（RST writer 的 `__init__.py` 里 header_rows 处理）。一个 7B 模型不擅长"traceback 不一定指向真正的修复点"这条经验法则。

→ **14182 在不调换模型的情况下基本无解**。

## 6. grep 推广仍然有效（对其他实例）

虽然 14182 没翻，但 grep 推广对其他类型的实例仍然有意义：

- **issue 文本完全没有文件线索**（如 "the foo() function returns wrong value"）：grep 可以定位 `def foo`
- **issue 说了一类文件但没说具体哪个**（如 "the parser in ascii/ format"）：grep `ascii/` 找到候选
- **issue 提到函数/符号名**：grep 直接定位定义点

修前的 11 字描述，模型完全没理由把 grep 当作"先于 read_file 的探索工具"——它只是同等地位的另一个工具。修后明确写了"issue text doesn't name a file → 先 grep 再 read_file"，模型有了触发条件。

## 7. commit 决定

- `src/prompt.py` 的改动保留——**对其他实例有用，对 14182 无害**。
- iteration/14（本文件）记录这次实验，证明 14182 翻成 PASS 需要换模型、不只是 prompt 工程。
- astropy-14182 的最终 verdict 仍是 WRONG_FILE——这次实验把失败模式从"模型卡住"变为"模型走错路"，是诊断性进展不是突破。

## 8. 提交

```
- src/prompt.py: _TERSE_TOOL_DESC grep / list_files 扩写
- eval/wire_captures/verify_template_14182_grep__20260826T084200Z.json
- iteration/14-grep-prompt-promotion.md（本文件）
```