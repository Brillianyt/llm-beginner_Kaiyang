# 01 — `read_file` 头部"自吹自擂"（commit `90095e4`）

## 症状

模型调用 `read_file("/abs/path/long_file.py")` 不带 `limit` 参数时，工具响应里出现：

```
=== 1 lines 0..400 of 642 ===
   1 | ...
   ...
 400 | ...
```

但**实际只返回了 360 行**——剩余 40 行被 `BaseTool.__call__` 的 char cap（默认 8000 chars）**静默**砍掉。模型以为它看完了完整文件，于是基于错误信息做编辑。

## 根因

`src/tools/read_file.py` 内的 `ReadFileTool` 把头部和 body 分别构造，但 `body = all_lines[offset : offset + limit]` 直接拿前 400 行，**没有任何关于最终输出字符总数的预算**。最终 `BaseTool.__call__` 在外层做了一个统一 cap：

```python
def __call__(self, ...):
    raw = tool.run(...)
    if len(raw) > self.max_result_chars:
        return raw[: self.max_result_chars] + "..."
```

—— 但 cap 砍的是字符串尾部的字符（不是行），于是 body 的最后几行被切到一半，而 `ReadFileTool` 自己先前返回的 `num_lines=400` / `total_lines=642` / `start_line=0` 字段没有被回算。模型看到的字符串里：

```
=== 1 lines 0..400 of 642 ===    ← 声称返回 400 行
   1 | ...
   ...
 360 | import os                   ← 第 360 行中断
 os                              ← 第 361 行只剩 3 个字符
                                 ← 后面的行全没了
```

## 在 wire capture 里的发现

`audit_14365__20260824T161513Z.json` 里 model 在 turn 4 调 `read_file` 读 `qdp.py`（600+ 行的实际 QDP 实现），response 字符串最后 5 个字符是 `... [truncated`，**但没有"how to continue"的提示**。观察后续 turn 的 assistant `message.content`，模型说：

> "I have read the file fully. It seems the issue is..."

—— 模型显然没意识到文件被截断，把后续的修复决策建立在不完整的源码视图上。

## 修复

`src/tools/read_file.py` 内 `ReadFileTool.run`：

1. **预算 `max_result_chars`**：先算出"header 长度"，再算"body 还能塞多少 chars"。
2. **当 body 必须被截**时：
   - 重新计算实际能塞的行数 `actual_limit`
   - 把头部改成 `=== lines 0..{actual_limit} of {total} ===`
   - 在 body 末尾追加 **`[output truncated at N chars; call read_file again with offset=K to continue]`**

```python
_HEADER_RESERVE_CHARS = 400

# Inside ReadFileTool.run, after building ``body = all_lines[offset : offset+limit]``:
joined = "".join(body)
header = f"=== 1 lines {offset}..{offset+len(body)-1} of {total_lines} ===\n"
header += "path: ...\n"
budget = self.max_result_chars - len(header) - 200  # 200 reserved for marker
if len(joined) > budget:
    joined = joined[:budget]
    # recompute actual_lines included
    joined += f"\n[output truncated at {budget} chars; call read_file again with offset=K to continue]"
```

## 验证证据

1. **代码层**：`src/tools/read_file.py` 重写后，`ReadFileTool` 不再依赖外层 cap——`max_result_chars` 在自身内被强制执行。
2. **行为层**：wire capture `audit_14365__20260824T161513Z.json` 的 turn 5（修复后）显示 response 末尾出现 `[output truncated at N chars; ...]` 标记，且头部 `lines 0..360 of 642` 与 body 行数对齐。
3. **模型行为**：后续 turn 的 assistant `tool_calls.arguments` 不再基于残缺文件推断（不再给出"file is fully read"的错误假设）。

## Commit

```
90095e4  fix: read_file honest header — no more silent char-cap lie
```

修改文件：
- `src/tools/read_file.py`（98 行）
- `test_smoke.py`（减少 852 行 → 1092 行的 stat 是反向的；实际是 test_smoke 重写）
- `eval/wire_captures/astropy_14365_post_fix__20260824T145554Z.json`（1765 行新增 capture）

## 次级收益

这一 commit 同时把 `read_file` 默认行为从"2000 行"改成"`DEFAULT_LIMIT=400` 行"。理由：

- 2000 行远超 `max_result_chars`（8000 chars ≈ 200 行代码）
- 400 行 + 显式 marker 比 2000 行 + 静默截断更安全
- 模型学会"如果 marker 出现就调 `read_file(offset=K)`"，prompt 不需要教（自然 emerge）